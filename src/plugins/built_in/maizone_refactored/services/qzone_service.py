"""
QQ空间服务模块
封装了所有与QQ空间API的直接交互，是插件的核心业务逻辑层。
"""

import asyncio
import base64
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles
import aiohttp
import bs4
import json5
import orjson

from src.common.logger import get_logger
from src.plugin_system.apis import config_api, cross_context_api, person_api

from .content_service import ContentService
from .cookie_service import CookieService
from .image_service import ImageService
from .reply_tracker_service import ReplyTrackerService

if TYPE_CHECKING:
    from .comment_tracking_service import CommentTrackingService
    from .person_memory_service import PersonMemoryService

logger = get_logger("MaiZone.QZoneService")


def _loads_lenient(text: str) -> Any:
    """容忍 ``_Callback(...)`` / jsonp 包裹以及 ``undefined`` 的宽松 JSON 解析。

    QZone 的部分只读接口（访客、计数、点赞名单）会用回调函数包裹 JSON 返回，
    这里统一剥离包裹再交给 orjson 解析。解析失败向上抛出，由调用方兜底为 None/[]。
    """
    t = text.strip()
    for prefix in ("_Callback(", "callback(", "_preloadCallback("):
        if t.startswith(prefix):
            t = t[len(prefix) :]
            if t.endswith(");"):
                t = t[:-2]
            elif t.endswith(")"):
                t = t[:-1]
            break
    t = t.replace("undefined", "null")
    return orjson.loads(t)


def _fmt_comment_time(node: dict) -> tuple[str, int]:
    """从一个评论节点解析 ``(格式化时间字符串, 原始时间戳)``。

    优先用 ``createTime2``（已是可读的 YYYY-MM-DD HH:MM:SS）作为展示串，
    ``create_time`` 作为原始时间戳（int，用于时间线比较）；拿不到时间戳返回 0。
    """
    ts = 0
    raw = node.get("create_time")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        ts = int(raw)
    elif isinstance(raw, str) and raw.isdigit():
        ts = int(raw)

    fmt = ""
    if node.get("createTime2"):
        fmt = str(node.get("createTime2"))
    elif ts:
        try:
            fmt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        except (ValueError, OSError):
            fmt = ""
    return fmt, ts


def _parse_comment_tree(commentlist: Any) -> list[dict]:
    """把 msglist / msgdetail 的 ``commentlist[]`` 展开为扁平评论列表（含楼中楼 ``list_3``）。

    统一供 ``_list_feeds`` 与 ``_get_feed_detail`` 复用，避免两份解析拷贝。每条评论字段：

    - ``qq_account``：评论者 QQ（原始值，可能是 int）
    - ``nickname`` / ``content``
    - ``comment_tid``：评论 tid（顶层为 feed 内从 1 起的楼层序号；楼中楼为子回复 tid）
    - ``parent_tid``：顶层为 None；楼中楼为其所属顶层评论的 tid
    - ``create_time``：格式化时间字符串（解析失败为 ""）
    - ``create_ts``：原始时间戳（int），拿不到为 0，用于时间线比较
    - ``t2_subtype``：顶层评论类型（0 普通 / 1 回复型），楼中楼固定 None
    - ``is_sub``：是否为楼中楼子回复
    - ``reply_num``：顶层评论的楼中楼数量（reply_num / replyNum 多候选）

    ``_list_feeds`` 下游只读前 6 个字段，其余为接话闭环新增，多出来的键不影响既有逻辑。
    ``list_3`` 的 @/targetuin 等字段未全部实测确认，这里只取确定存在的字段并全程防御性判空。
    """
    comments: list[dict] = []
    if not isinstance(commentlist, list):
        return comments

    for c in commentlist:
        if not isinstance(c, dict):
            continue
        c_fmt, c_ts = _fmt_comment_time(c)

        # 楼中楼数量：reply_num / replyNum 多候选取值
        reply_num = 0
        for k in ("reply_num", "replyNum"):
            v = c.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                reply_num = int(v)
                break
            if isinstance(v, str) and v.isdigit():
                reply_num = int(v)
                break

        # t2_subtype 防御性取值（0 普通 / 1 回复型）
        t2_raw = c.get("t2_subtype")
        if isinstance(t2_raw, (int, float)) and not isinstance(t2_raw, bool):
            t2_subtype: int | None = int(t2_raw)
        elif isinstance(t2_raw, str) and t2_raw.lstrip("-").isdigit():
            t2_subtype = int(t2_raw)
        else:
            t2_subtype = None

        comments.append(
            {
                "qq_account": c.get("uin"),
                "nickname": c.get("name"),
                "content": c.get("content"),
                "comment_tid": c.get("tid"),
                "parent_tid": None,  # 主评论没有父ID
                "create_time": c_fmt,
                "create_ts": c_ts,
                "t2_subtype": t2_subtype,
                "is_sub": False,
                "reply_num": reply_num,
            }
        )

        # 楼中楼子回复 list_3
        sub_list = c.get("list_3")
        if isinstance(sub_list, list):
            for r in sub_list:
                if not isinstance(r, dict):
                    continue
                r_fmt, r_ts = _fmt_comment_time(r)
                comments.append(
                    {
                        "qq_account": r.get("uin"),
                        "nickname": r.get("name"),
                        "content": r.get("content"),
                        "comment_tid": r.get("tid"),
                        "parent_tid": c.get("tid"),  # 父ID是所属主评论的 tid
                        "create_time": r_fmt,
                        "create_ts": r_ts,
                        "t2_subtype": None,
                        "is_sub": True,
                        "reply_num": 0,
                    }
                )
    return comments


class QZoneService:
    """
    QQ空间服务类，负责所有API交互和业务流程编排。
    """

    # --- API Endpoints ---
    ZONE_LIST_URL = "https://user.qzone.qq.com/proxy/domain/ic2.qzone.qq.com/cgi-bin/feeds/feeds3_html_more"
    EMOTION_PUBLISH_URL = "https://user.qzone.qq.com/proxy/domain/taotao.qzone.qq.com/cgi-bin/emotion_cgi_publish_v6"
    # 转发说说（转发锐评功能）：QQ空间转发走 publish 同族 CGI，在 publish 参数上追加 rt_tid/rt_uin
    # 转发标识。⚠️ 未经 live 验证（本机不允许发起真实 QZone 请求），首次开启需真机校验，见 repost_feed。
    EMOTION_FORWARD_URL = "https://user.qzone.qq.com/proxy/domain/taotao.qzone.qq.com/cgi-bin/emotion_cgi_publish_v6"
    DOLIKE_URL = "https://user.qzone.qq.com/proxy/domain/w.qzone.qq.com/cgi-bin/likes/internal_dolike_app"
    COMMENT_URL = "https://user.qzone.qq.com/proxy/domain/taotao.qzone.qq.com/cgi-bin/emotion_cgi_re_feeds"
    LIST_URL = "https://user.qzone.qq.com/proxy/domain/taotao.qq.com/cgi-bin/emotion_cgi_msglist_v6"
    # 单条说说详情（含完整评论楼 + 楼中楼），接话闭环用；与 msglist 同域同签名，跨用户可读
    MSGDETAIL_URL = "https://user.qzone.qq.com/proxy/domain/taotao.qq.com/cgi-bin/emotion_cgi_msgdetail_v6"
    REPLY_URL = "https://user.qzone.qq.com/proxy/domain/taotao.qzone.qq.com/cgi-bin/emotion_cgi_re_feeds"
    # 只读社交接口（社交闭环功能：计数门 / 访客回访 / 回赞）
    # 访客列表：已只读实测可用（走 user.qzone 代理域，返回 _Callback 包裹的 JSON，code=0）
    VISITOR_URL = "https://user.qzone.qq.com/proxy/domain/g.qzone.qq.com/cgi-bin/friendshow/cgi_get_visitor_more"
    # 未读计数：轻量事件门用（实测该 URL 目前返回 404，失败时整体回退全量，绝不影响监控）
    COUNT_URL = "https://mobile.qzone.qq.com/get_count"
    # 点赞名单：拉自己说说的点赞者（失败则跳过回赞，不影响主流程）
    LIKE_LIST_URL = "https://user.qzone.qq.com/proxy/domain/r.qzone.qq.com/cgi-bin/likes/get_like_list_app"

    def __init__(
        self,
        get_config: Callable,
        content_service: ContentService,
        image_service: ImageService,
        cookie_service: CookieService,
        reply_tracker: ReplyTrackerService | None = None,
        person_memory: "PersonMemoryService | None" = None,
        comment_tracking: "CommentTrackingService | None" = None,
    ):
        self.get_config = get_config
        self.content_service = content_service
        self.image_service = image_service
        self.cookie_service = cookie_service
        # 如果没有提供 reply_tracker 实例，则创建一个新的
        self.reply_tracker = reply_tracker if reply_tracker is not None else ReplyTrackerService()
        # 空间人物记忆服务（可选）：负责把互动对象注册进记人系统并积累空间互动记录
        self.person_memory = person_memory
        # 好友说说评论追踪服务（可选）：接话闭环用，追踪 bot 评论过的好友说说
        self.comment_tracking = comment_tracking
        # 用于防止并发回复/评论的内存锁
        self.processing_comments = set()

    # --- Public Methods (High-Level Business Logic) ---
    async def _get_cross_context(self) -> str:
        """获取并构建跨群聊上下文"""
        context = ""
        user_id = self.get_config("cross_context.user_id")

        if user_id:
            logger.info(f"检测到互通组用户ID: {user_id}，准备获取上下文...")
            try:
                context = await cross_context_api.build_cross_context_for_user(
                    user_id=user_id,
                    platform="QQ",  # 硬编码为QQ
                    limit_per_stream=10,
                    stream_limit=3,
                )
                if context:
                    logger.info("成功获取到互通组上下文。")
                else:
                    logger.info("未获取到有效的互通组上下文。")
            except Exception as e:
                logger.error(f"获取互通组上下文时发生异常: {e}")
        return context

    async def send_feed(self, topic: str, stream_id: str | None) -> dict[str, Any]:
        """发送一条说说（支持AI配图）"""
        cross_context = await self._get_cross_context()

        # 检查是否启用AI配图
        ai_image_enabled = self.get_config("ai_image.enable_ai_image", False)
        provider = self.get_config("ai_image.provider", "siliconflow")

        image_path = None

        if ai_image_enabled:
            # 启用AI配图：文本模型生成说说+图片提示词
            story, image_info = await self.content_service.generate_story_with_image_info(topic, context=cross_context)
            if not story:
                return {"success": False, "message": "生成说说内容失败"}

            # 根据provider调用对应的生图服务
            if provider == "novelai":
                try:
                    from .novelai_service import MaiZoneNovelAIService
                    novelai_service = MaiZoneNovelAIService(self.get_config)

                    if novelai_service.is_available():
                        # 解析画幅
                        aspect_ratio = image_info.get("aspect_ratio", "方图")
                        size_map = {
                            "方图": (1024, 1024),
                            "横图": (1216, 832),
                            "竖图": (832, 1216),
                        }
                        width, height = size_map.get(aspect_ratio, (1024, 1024))

                        logger.info("🎨 开始生成NovelAI配图...")
                        success, img_path, msg = await novelai_service.generate_image_from_prompt_data(
                            prompt=image_info.get("prompt", ""),
                            negative_prompt=image_info.get("negative_prompt"),
                            include_character=image_info.get("include_character", False),
                            width=width,
                            height=height
                        )

                        if success and img_path:
                            image_path = img_path
                            logger.info("✅ NovelAI配图生成成功")
                        else:
                            logger.warning(f"⚠️ NovelAI配图生成失败: {msg}")
                    else:
                        logger.warning("NovelAI服务不可用（未配置API Key）")

                except Exception as e:
                    logger.error(f"NovelAI配图生成出错: {e}", exc_info=True)

            elif provider == "siliconflow":
                try:
                    # 调用硅基流动生成图片
                    success, img_path = await self.image_service.generate_image_from_prompt(
                        prompt=image_info.get("prompt", ""),
                        save_dir=None  # 使用默认images目录
                    )
                    if success and img_path:
                        image_path = img_path
                        logger.info("✅ 硅基流动配图生成成功")
                    else:
                        logger.warning("⚠️ 硅基流动配图生成失败")
                except Exception as e:
                    logger.error(f"硅基流动配图生成出错: {e}", exc_info=True)
        else:
            # 不使用AI配图：只生成说说文本
            story = await self.content_service.generate_story(topic, context=cross_context)
            if not story:
                return {"success": False, "message": "生成说说内容失败"}

        qq_account = config_api.get_global_config("bot.qq_account", "")
        api_client = await self._get_api_client(qq_account, stream_id)
        if not api_client:
            return {"success": False, "message": "获取QZone API客户端失败"}

        # 加载图片
        images_bytes = []

        # 使用AI生成的图片
        if image_path and image_path.exists():
            try:
                with open(image_path, "rb") as f:
                    images_bytes.append(f.read())
                logger.info("添加AI配图到说说")
            except Exception as e:
                logger.error(f"读取AI配图失败: {e}")

        try:
            success, _ = await api_client["publish"](story, images_bytes)
            if success:
                return {"success": True, "message": story}
            return {"success": False, "message": "发布说说至QQ空间失败"}
        except Exception as e:
            logger.error(f"发布说说时发生异常: {e}")
            return {"success": False, "message": f"发布说说异常: {e}"}

    async def send_feed_from_activity(self, activity: str) -> dict[str, Any]:
        """根据日程活动发送一条说说"""
        cross_context = await self._get_cross_context()
        story = await self.content_service.generate_story_from_activity(activity, context=cross_context)
        if not story:
            return {"success": False, "message": "根据活动生成说说内容失败"}

        if self.get_config("send.enable_ai_image", False):
            await self.image_service.generate_images_for_story(story)

        qq_account = config_api.get_global_config("bot.qq_account", "")
        api_client = await self._get_api_client(qq_account, stream_id=None)
        if not api_client:
            return {"success": False, "message": "获取QZone API客户端失败"}

        try:
            success, _ = await api_client["publish"](story, [])
            if success:
                return {"success": True, "message": story}
            return {"success": False, "message": "发布说说至QQ空间失败"}
        except Exception as e:
            logger.error(f"根据活动发布说说时发生异常: {e}")
            return {"success": False, "message": f"发布说说异常: {e}"}

    async def read_and_process_feeds(self, target_name: str, stream_id: str | None) -> dict[str, Any]:
        """读取并处理指定好友的说说"""
        # 判断输入是QQ号还是昵称
        target_qq = None

        if target_name.isdigit():
            # 输入是纯数字，当作QQ号处理
            target_qq = int(target_name)
        else:
            # 输入是昵称，查询person_info获取QQ号
            target_person_id = await person_api.get_person_id_by_name(target_name)
            if not target_person_id:
                return {"success": False, "message": f"找不到名为'{target_name}'的好友"}
            person_info = await person_api.get_person_info(target_person_id)
            target_qq = person_info.get("user_id")
            if not target_qq:
                return {"success": False, "message": f"好友'{target_name}'没有关联QQ号"}

        qq_account = config_api.get_global_config("bot.qq_account", "")
        logger.debug(f"准备获取API客户端，qq_account={qq_account}")
        api_client = await self._get_api_client(qq_account, stream_id)
        if not api_client:
            logger.error("API客户端获取失败，返回错误")
            return {"success": False, "message": "获取QZone API客户端失败"}

        logger.debug("API客户端获取成功，准备读取说说")
        num_to_read = self.get_config("read.read_number", 5)

        # 尝试执行，如果Cookie失效则自动重试一次
        for retry_count in range(2):  # 最多尝试2次
            try:
                logger.debug(f"开始调用 list_feeds，target_qq={target_qq}, num={num_to_read}")
                feeds = await api_client["list_feeds"](target_qq, num_to_read)
                logger.debug(f"list_feeds 返回，feeds数量={len(feeds) if feeds else 0}")
                if not feeds:
                    return {"success": True, "message": f"没有从'{target_name}'的空间获取到新说说。"}

                logger.debug(f"准备处理 {len(feeds)} 条说说")

                # --- LLM 选择评论目标（可配置开关；失败则回退到逐条随机） ---
                decision_map: dict[str, bool] | None = None
                if self.get_config("selection.enable_llm_selection", True):
                    try:
                        decision_map = await self._select_comment_targets(feeds, lambda _f: target_name)
                    except Exception as e:
                        logger.warning(f"LLM 选择评论目标异常，回退到逐条随机评论: {e}")
                        decision_map = None

                total_liked = 0
                total_commented = 0
                for feed in feeds:
                    fid = feed.get("tid", "")
                    comment_decision = decision_map.get(fid, False) if decision_map is not None else None
                    result = await self._process_single_feed(
                        feed, api_client, str(target_qq), target_name, comment_decision
                    )
                    if result["liked"]:
                        total_liked += 1
                    if result["commented"]:
                        total_commented += 1
                    await asyncio.sleep(random.uniform(3, 7))

                # 构建详细的反馈信息
                stats_parts = []
                if total_liked > 0:
                    stats_parts.append(f"点赞了{total_liked}条")
                if total_commented > 0:
                    stats_parts.append(f"评论了{total_commented}条")

                if stats_parts:
                    stats_msg = "、".join(stats_parts)
                    message = f"成功查看了'{target_name}'的空间，{stats_msg}。"
                else:
                    message = f"成功查看了'{target_name}'的 {len(feeds)} 条说说，但这次没有进行互动。"

                return {
                    "success": True,
                    "message": message,
                    "stats": {"total": len(feeds), "liked": total_liked, "commented": total_commented},
                }
            except RuntimeError as e:
                # QQ空间API返回的业务错误
                error_msg = str(e)

                # 检查是否是Cookie失效（-3000错误）
                if "错误码: -3000" in error_msg and retry_count == 0:
                    logger.warning("检测到Cookie失效（-3000错误），准备删除缓存并重试...")

                    # 删除Cookie缓存文件
                    cookie_file = self.cookie_service._get_cookie_file_path(qq_account)
                    if cookie_file.exists():
                        try:
                            cookie_file.unlink()
                            logger.info(f"已删除过期的Cookie缓存文件: {cookie_file}")
                        except Exception as delete_error:
                            logger.error(f"删除Cookie文件失败: {delete_error}")

                    # 重新获取API客户端（会自动获取新Cookie）
                    logger.info("正在重新获取Cookie...")
                    api_client = await self._get_api_client(qq_account, stream_id)
                    if not api_client:
                        logger.error("重新获取API客户端失败")
                        return {"success": False, "message": "Cookie已失效，且无法重新获取。请检查Bot和Napcat连接状态。"}

                    logger.info("Cookie已更新，正在重试...")
                    continue  # 继续循环，重试一次

                # 其他业务错误或重试后仍失败
                logger.warning(f"QQ空间API错误: {e}")
                return {"success": False, "message": error_msg}
            except Exception as e:
                # 其他未知异常
                logger.error(f"读取和处理说说时发生异常: {e}")
                return {"success": False, "message": f"处理说说时出现异常: {e}"}
        return {"success": False, "message": "读取和处理说说时发生未知错误，循环意外结束。"}

    async def monitor_feeds(self, stream_id: str | None = None):
        """监控并处理所有好友的动态，包括回复自己说说的评论"""
        logger.info("开始执行好友动态监控...")
        qq_account = config_api.get_global_config("bot.qq_account", "")

        # 尝试执行，如果Cookie失效则自动重试一次
        for retry_count in range(2):  # 最多尝试2次
            api_client = await self._get_api_client(qq_account, stream_id)
            if not api_client:
                logger.error("监控失败：无法获取API客户端")
                return

            try:
                # --- 第一步: 单独处理自己说说的评论 ---
                if self.get_config("monitor.enable_auto_reply", False):
                    try:
                        # 传入新参数，表明正在检查自己的说说
                        own_feeds = await api_client["list_feeds"](qq_account, 5)
                        if own_feeds:
                            logger.info(f"获取到自己 {len(own_feeds)} 条说说，检查评论...")
                            for feed in own_feeds:
                                await self._reply_to_own_feed_comments(feed, api_client)
                                await asyncio.sleep(random.uniform(3, 5))
                    except Exception as e:
                        logger.error(f"处理自己说说评论时发生异常: {e}")

                # --- 第二步: 处理好友的动态 ---
                friend_feeds = await api_client["monitor_list_feeds"](20)
                if not friend_feeds:
                    logger.info("监控完成：未发现好友新说说")
                    return

                logger.info(f"监控任务: 发现 {len(friend_feeds)} 条好友新动态，准备处理...")

                # --- LLM 选择评论目标（可配置开关；失败则回退到逐条随机） ---
                decision_map: dict[str, bool] | None = None
                if self.get_config("selection.enable_llm_selection", True):
                    try:
                        decision_map = await self._select_comment_targets(
                            friend_feeds, lambda f: str(f.get("target_qq", ""))
                        )
                    except Exception as e:
                        logger.warning(f"LLM 选择评论目标异常，回退到逐条随机评论: {e}")
                        decision_map = None

                monitor_stats = {"total": 0, "liked": 0, "commented": 0}
                for feed in friend_feeds:
                    target_qq = feed.get("target_qq")
                    if not target_qq or str(target_qq) == str(qq_account):  # 确保不重复处理自己的
                        continue

                    fid = feed.get("tid", "")
                    comment_decision = decision_map.get(fid, False) if decision_map is not None else None
                    result = await self._process_single_feed(
                        feed, api_client, str(target_qq), str(target_qq), comment_decision
                    )
                    monitor_stats["total"] += 1
                    if result.get("liked"):
                        monitor_stats["liked"] += 1
                    if result.get("commented"):
                        monitor_stats["commented"] += 1
                    await asyncio.sleep(random.uniform(5, 10))

                logger.info(
                    f"监控任务完成: 处理了{monitor_stats['total']}条动态，"
                    f"点赞{monitor_stats['liked']}条，评论{monitor_stats['commented']}条"
                )
                return  # 成功完成，直接返回

            except RuntimeError as e:
                # QQ空间API返回的业务错误
                error_msg = str(e)

                # 检查是否是Cookie失效（-3000错误）
                if "错误码: -3000" in error_msg and retry_count == 0:
                    logger.warning("检测到Cookie失效（-3000错误），准备删除缓存并重试...")

                    # 删除Cookie缓存文件
                    cookie_file = self.cookie_service._get_cookie_file_path(qq_account)
                    if cookie_file.exists():
                        try:
                            cookie_file.unlink()
                            logger.info(f"已删除过期的Cookie缓存文件: {cookie_file}")
                        except Exception as delete_error:
                            logger.error(f"删除Cookie文件失败: {delete_error}")

                    # 重新获取API客户端会在下一次循环中自动进行
                    logger.info("Cookie已删除，正在重试...")
                    continue  # 继续循环，重试一次

                # 其他业务错误或重试后仍失败
                logger.error(f"监控好友动态时发生业务错误: {e}")
                return

            except Exception as e:
                # 其他未知异常
                logger.error(f"监控好友动态时发生异常: {e}")
                return

    # --- Internal Helper Methods ---


    async def _reply_to_own_feed_comments(self, feed: dict, api_client: dict):
        """处理对自己说说的评论并进行回复"""
        qq_account = config_api.get_global_config("bot.qq_account", "")
        comments = feed.get("comments", [])
        content = feed.get("content", "")
        fid = feed.get("tid", "")
        images = feed.get("images", [])  # 获取说说中的图片
        story_time = feed.get("created_time", "")  # 获取说说发送时间

        if not comments or not fid:
            return

        # 1. 将评论分为用户评论和自己的回复
        user_comments = [c for c in comments if str(c.get("qq_account")) != str(qq_account)]

        if not user_comments:
            return

        # 直接检查评论是否已回复，不做验证清理
        comments_to_process = []
        for comment in user_comments:
            comment_tid = comment.get("comment_tid")
            if not comment_tid:
                continue

            comment_key = f"{fid}_{comment_tid}"
            # 检查持久化记录和内存锁
            if not self.reply_tracker.has_replied(fid, comment_tid) and comment_key not in self.processing_comments:
                logger.debug(f"锁定待回复评论: {comment_key}")
                self.processing_comments.add(comment_key)
                comments_to_process.append(comment)

        if not comments_to_process:
            logger.debug(f"说说 {fid} 下的所有评论都已回复过或正在处理中")
            return

        logger.info(f"发现自己说说下的 {len(comments_to_process)} 条新评论，准备回复...")
        for comment in comments_to_process:
            comment_tid = comment.get("comment_tid")
            comment_key = f"{fid}_{comment_tid}"
            nickname = comment.get("nickname", "")
            comment_content = comment.get("content", "")
            commenter_qq = str(comment.get("qq_account", "")) if comment.get("qq_account") else None
            comment_time = comment.get("create_time", "")  # 获取评论时间

            try:
                reply_content = await self.content_service.generate_comment_reply(
                    story_content=content,
                    story_time=story_time,  # 传递说说发送时间
                    comment_content=comment_content,
                    comment_time=comment_time,  # 传递评论时间
                    commenter_name=nickname,
                    commenter_qq=commenter_qq,
                    images=images,  # 传递说说中的图片
                )
                if reply_content:
                    success = await api_client["reply"](fid, qq_account, nickname, reply_content, comment_tid)
                    if success:
                        self.reply_tracker.mark_as_replied(fid, comment_tid)
                        logger.info(f"成功回复'{nickname}'的评论: '{reply_content}'")
                        # 记人：注册评论者并积累一条空间互动记录（失败不影响主流程）
                        await self._remember_interaction(
                            qq=commenter_qq,
                            nickname=nickname,
                            interaction_type="回复了ta在你说说下的评论",
                            their_content=comment_content,
                            bot_reply=reply_content,
                        )
                    else:
                        logger.error(f"回复'{nickname}'的评论失败")
                    await asyncio.sleep(random.uniform(10, 20))
                else:
                    logger.warning(f"生成回复内容失败，跳过回复'{nickname}'的评论")
            except Exception as e:
                logger.error(f"回复'{nickname}'的评论时发生异常: {e}")
            finally:
                # 无论成功与否，都解除锁定
                logger.debug(f"解锁评论: {comment_key}")
                if comment_key in self.processing_comments:
                    self.processing_comments.remove(comment_key)

    async def _validate_and_cleanup_reply_records(self, fid: str, my_replies: list[dict]):
        """验证并清理已删除的回复记录"""
        # 获取当前记录中该说说的所有已回复评论ID
        recorded_replied_comments = self.reply_tracker.get_replied_comments(fid)

        if not recorded_replied_comments:
            return

        # 从API返回的我的回复中提取parent_tid（即被回复的评论ID）
        current_replied_comments = set()
        for reply in my_replies:
            parent_tid = reply.get("parent_tid")
            if parent_tid:
                current_replied_comments.add(parent_tid)

        # 找出记录中有但实际已不存在的回复
        deleted_replies = recorded_replied_comments - current_replied_comments

        if deleted_replies:
            logger.info(f"检测到 {len(deleted_replies)} 个回复已被删除，清理记录...")
            for comment_tid in deleted_replies:
                self.reply_tracker.remove_reply_record(fid, comment_tid)
                logger.debug(f"已清理删除的回复记录: feed_id={fid}, comment_id={comment_tid}")

    async def _select_comment_targets(self, feeds: list[dict], author_of: Callable[[dict], str]) -> dict[str, bool] | None:
        """对一批说说执行 LLM 选择：像人一样挑出真正值得评论的目标。

        :param feeds: 说说列表（元素需含 tid、content、rt_con、images 等字段）。
        :param author_of: 回调，传入单条 feed 返回作者展示名。
        :return:
            - dict[fid -> bool]：评论决策表（True=选中评论，False=看过但不评论）。
              未被选中的候选会同时被标记为「看过无兴趣」，防止下一轮重复进入候选。
            - None：选择失败，调用方应回退到原有逐条随机评论逻辑。
        """
        candidate_limit = int(self.get_config("selection.candidate_limit", 8))
        max_comments = int(self.get_config("selection.max_comments_per_round", 3))

        # 收集未处理过的候选（既未评论过，也未被标记为「看过无兴趣」），复用 reply_tracker 去重
        candidates: list[dict] = []
        for feed in feeds:
            fid = feed.get("tid", "")
            if not fid:
                continue
            if self.reply_tracker.has_replied(fid, "main_comment") or self.reply_tracker.has_replied(
                fid, "seen_no_comment"
            ):
                continue
            candidates.append(feed)
            if len(candidates) >= candidate_limit:
                break

        if not candidates:
            # 没有新候选，返回空决策表（本批 feed 一律不评论）
            return {}

        # 构建给 LLM 的精简候选信息（作者、时间、内容摘要、是否带图）
        payload: list[dict] = []
        for feed in candidates:
            content = feed.get("content", "")
            rt_con = (
                feed.get("rt_con", {}).get("content", "")
                if isinstance(feed.get("rt_con"), dict)
                else feed.get("rt_con", "")
            )
            payload.append(
                {
                    "author": author_of(feed),
                    "time": feed.get("created_time", "") or "",
                    "content": content or rt_con or "",
                    "has_image": bool(feed.get("images")),
                }
            )

        selections = await self.content_service.select_feeds_to_comment(payload, max_select=max_comments)
        if selections is None:
            # 选择失败（模型/解析异常）→ 让调用方回退到逐条随机
            return None

        # 依据序号映射回 feed，构建决策表
        selected_fids: set[str] = set()
        for sel in selections:
            i = sel["index"] - 1
            if 0 <= i < len(candidates):
                fid = candidates[i].get("tid", "")
                if fid:
                    selected_fids.add(fid)

        decision: dict[str, bool] = {}
        for feed in candidates:
            fid = feed.get("tid", "")
            if not fid:
                continue
            if fid in selected_fids:
                decision[fid] = True
            else:
                decision[fid] = False
                # 标记为「看过但没兴趣评论」，防止下一轮再次进入候选被反复评估
                self.reply_tracker.mark_as_replied(fid, "seen_no_comment")

        # 选择阶段日志（便于观察「选择感」）
        if selected_fids:
            briefs = []
            for sel in selections:
                i = sel["index"] - 1
                if 0 <= i < len(payload):
                    reason = sel.get("reason", "") or "（未说明理由）"
                    briefs.append(f"[{sel['index']}]{payload[i]['author']}: {reason}")
            logger.info(f"[空间选择] 候选 {len(candidates)} 条 → 选中 {len(briefs)} 条：" + "；".join(briefs))
        else:
            logger.info(f"[空间选择] 候选 {len(candidates)} 条 → 本轮没有想评论的动态，跳过评论")

        return decision

    async def _process_single_feed(
        self, feed: dict, api_client: dict, target_qq: str, target_name: str, comment_decision: bool | None = None
    ) -> dict:
        """处理单条说说，决定是否评论和点赞

        :param comment_decision: 评论决策。None=沿用原有逐条随机；True=LLM 选中，评论；False=LLM 未选中，跳过评论。

        返回:
            dict: {"liked": bool, "commented": bool}
        """
        content = feed.get("content", "")
        fid = feed.get("tid", "")
        # 正确提取转发内容（rt_con 可能是字典或字符串）
        rt_con = feed.get("rt_con", {}).get("content", "") if isinstance(feed.get("rt_con"), dict) else feed.get("rt_con", "")
        images = feed.get("images", [])

        result = {"liked": False, "commented": False}

        # --- 处理评论 ---
        comment_key = f"{fid}_main_comment"
        # comment_decision: None=沿用原有逐条随机；True=LLM 选中评论；False=LLM 未选中跳过
        if comment_decision is None:
            should_comment = random.random() <= self.get_config("read.comment_possibility", 0.3)
        else:
            should_comment = comment_decision

        if (
            should_comment
            and not self.reply_tracker.has_replied(fid, "main_comment")
            and comment_key not in self.processing_comments
        ):
            logger.debug(f"锁定待评论说说: {comment_key}")
            self.processing_comments.add(comment_key)
            try:
                # 使用空间专用评论方法
                comment_text = await self.content_service.generate_qzone_comment(
                    target_name=target_name,
                    content=content or rt_con or "说说内容",
                    rt_con=rt_con if content else None,
                    images=images,
                    target_qq=target_qq,
                )
                if comment_text:
                    success, new_comment_tid = await api_client["comment"](target_qq, fid, comment_text)
                    if success:
                        self.reply_tracker.mark_as_replied(fid, "main_comment")
                        logger.info(f"成功评论'{target_name}'的说说: '{comment_text}'")
                        result["commented"] = True
                        # 记人：注册对方并积累一条空间互动记录（失败不影响主流程）
                        await self._remember_interaction(
                            qq=target_qq,
                            nickname=target_name,
                            interaction_type="评论了ta的说说",
                            their_content=content or rt_con or "",
                            bot_reply=comment_text,
                        )
                        # 接话闭环：仅在功能开启时把这条已评论的好友说说加入追踪，
                        # 关闭时不写盘（省成本、避免存储无界增长；开启后新评论自然进入追踪）。
                        if self.comment_tracking is not None and self.get_config(
                            "comment_reply.enable_comment_reply", False
                        ):
                            try:
                                self.comment_tracking.add_commented_feed(
                                    feed_id=fid,
                                    host_qq=target_qq,
                                    host_name=target_name,
                                    bot_comment_content=comment_text,
                                    bot_comment_tid=new_comment_tid,
                                )
                            except Exception as e:
                                logger.warning(f"[接话] 追踪已评论说说失败(fid={fid}): {e}")
                    else:
                        logger.error(f"评论'{target_name}'的说说失败")
            except Exception as e:
                logger.error(f"评论'{target_name}'的说说时发生异常: {e}")
            finally:
                logger.debug(f"解锁说说: {comment_key}")
                if comment_key in self.processing_comments:
                    self.processing_comments.remove(comment_key)

        # --- 处理点赞 (逻辑不变) ---
        like_probability = self.get_config("read.like_possibility", 1.0)
        if random.random() <= like_probability:
            logger.info(f"准备点赞说说: target_qq={target_qq}, fid={fid}")
            like_success = await api_client["like"](target_qq, fid)
            if like_success:
                logger.info(f"成功点赞'{target_name}'的说说: fid={fid}")
                result["liked"] = True
            else:
                logger.warning(f"点赞'{target_name}'的说说失败: fid={fid}")
        else:
            logger.debug(f"概率未命中，跳过点赞: probability={like_probability}")

        return result

    async def _remember_interaction(
        self,
        qq: str | int | None,
        nickname: str | None,
        interaction_type: str,
        their_content: str | None,
        bot_reply: str | None,
    ) -> None:
        """把一次成功的空间互动接入记人系统：注册对方 + 追加互动记录。

        整体包裹 try/except，任何失败只告警，绝不影响评论 / 回复主流程。
        """
        if self.person_memory is None or not qq:
            return
        try:
            await self.person_memory.register_person(qq, nickname)
            self.person_memory.add_interaction(
                qq=qq,
                nickname=nickname,
                interaction_type=interaction_type,
                their_content=their_content,
                bot_reply=bot_reply,
            )
        except Exception as e:
            logger.warning(f"记录空间互动记忆失败(qq={qq}): {e}")

    # --- 好友说说「接话闭环」---

    @staticmethod
    def _detect_replies_to_bot(
        comments: list[dict],
        bot_qq: str,
        bot_tid: Any,
        bot_ts: int,
        bot_name: str,
        host_qq: str,
    ) -> list[dict]:
        """双路检测「回复了 bot 评论」的新消息（按时间升序返回，先接早的）。

        路 (a) 楼中楼：``list_3`` 里 ``parent_tid`` 指向 bot 评论 tid 的子回复（is_sub 且 parent 命中）。
        路 (b) 平铺 level-2 评论：``uin != bot``、时间晚于 bot 评论，且（content 含 ``@bot昵称``
               或 ``uin == 楼主``）。因为 bot 自己的 ``_reply`` 实测表现为平铺 level-2（t2_subtype=1、
               content="@昵称 正文"、不进 list_3），故他人回复 bot 也走此形态。

        时间戳缺失（拿不到 create_ts）时对路 (b) 放宽时间过滤，改由上层 seen 去重兜底。
        """
        bot_qq = str(bot_qq or "")
        bot_tid_str = str(bot_tid) if bot_tid is not None else None
        host_qq = str(host_qq or "")
        at_token = f"@{bot_name}" if bot_name else None

        found: list[dict] = []
        seen_local: set[str] = set()

        def _push(node: dict) -> None:
            k = f"{node.get('qq_account')}_{node.get('comment_tid')}"
            if k not in seen_local:
                seen_local.add(k)
                found.append(node)

        for c in comments:
            if not isinstance(c, dict):
                continue
            uin = str(c.get("qq_account", "") or "")
            if not uin or uin == bot_qq:
                continue
            c_ts = c.get("create_ts", 0) or 0

            # 路 (a): 楼中楼子回复，parent 指向 bot 的评论楼
            if c.get("is_sub"):
                if bot_tid_str is not None and str(c.get("parent_tid")) == bot_tid_str:
                    _push(c)
                continue

            # 路 (b): 平铺 level-2 评论
            # 时间要晚于 bot 评论；任一时间戳拿不到时放宽（靠 seen 去重兜底）
            later = (c_ts == 0 or bot_ts == 0) or (c_ts >= bot_ts)
            if not later:
                continue
            content = str(c.get("content", "") or "")
            mentions_bot = bool(at_token and at_token in content)
            is_host = uin == host_qq
            if mentions_bot or is_host:
                _push(c)

        found.sort(key=lambda x: x.get("create_ts", 0) or 0)
        return found

    async def check_comment_replies(self) -> None:
        """好友说说接话闭环：轮询 bot 评论过的好友说说，发现回复 bot 的新消息则结合记忆定向接话。

        默认关闭（``comment_reply.enable_comment_reply=false``）。整体失败隔离，绝不影响监控主流程。
        """
        try:
            if not self.get_config("comment_reply.enable_comment_reply", False):
                return
            if self.comment_tracking is None:
                return

            ttl_hours = float(self.get_config("comment_reply.tracking_ttl_hours", 72))
            ttl_seconds = ttl_hours * 3600
            max_replies_per_round = int(self.get_config("comment_reply.max_replies_per_round", 3))
            per_feed_limit = int(self.get_config("comment_reply.per_feed_reply_limit", 2))

            # 先清理过期，再取活跃追踪 feed
            self.comment_tracking.cleanup(ttl_seconds)
            active = self.comment_tracking.get_active_feeds(ttl_seconds)
            if not active:
                return

            qq_account = str(config_api.get_global_config("bot.qq_account", "") or "")
            api_client = await self._get_api_client(qq_account, None)
            if not api_client:
                logger.debug("[接话] 无法获取API客户端，跳过本轮接话检查")
                return

            logger.info(f"[接话] 开始检查 {len(active)} 条已评论好友说说的回复情况")
            replies_done = 0

            for fid, entry in active:
                if replies_done >= max_replies_per_round:
                    break
                host_qq = str(entry.get("host_qq", "") or "")
                host_name = entry.get("host_name", "") or host_qq
                if not host_qq:
                    continue

                # feed 之间留间隔，降低风控
                await asyncio.sleep(random.uniform(3, 6))

                detail = await api_client["get_feed_detail"](host_qq, fid)
                if not detail:
                    continue
                comments = detail.get("comments", [])
                if not comments:
                    continue

                # 定位 bot 自己的楼：uin==bot 的顶层评论（reply_tracker 的 main_comment 幂等
                # 保证 bot 对每条好友说说只评一次，故 uin==bot 的顶层评论唯一）
                bot_comment = next(
                    (
                        c
                        for c in comments
                        if not c.get("is_sub") and str(c.get("qq_account", "")) == qq_account
                    ),
                    None,
                )
                if bot_comment is None:
                    # 找不到 bot 的评论：多半评论被删，从追踪表剔除
                    logger.info(f"[接话] 说说 {fid} 下未找到 bot 的评论（可能已被删），移除追踪")
                    self.comment_tracking.remove_feed(fid)
                    continue

                bot_tid = bot_comment.get("comment_tid")
                bot_ts = bot_comment.get("create_ts", 0) or 0
                bot_name = bot_comment.get("nickname", "") or ""
                bot_comment_content = entry.get("bot_comment_content", "") or bot_comment.get("content", "") or ""
                # 回填当初评论时没拿到的楼层序号
                if bot_tid:
                    self.comment_tracking.update_bot_comment_tid(fid, bot_tid)

                # --- 双路检测「回复了 bot」的新消息 ---
                new_replies = self._detect_replies_to_bot(
                    comments=comments,
                    bot_qq=qq_account,
                    bot_tid=bot_tid,
                    bot_ts=bot_ts,
                    bot_name=bot_name,
                    host_qq=host_qq,
                )

                for rep in new_replies:
                    if replies_done >= max_replies_per_round:
                        break
                    replier_uin = str(rep.get("qq_account", "") or "")
                    rep_tid = rep.get("comment_tid")
                    key = f"{replier_uin}_{rep_tid}"

                    if self.comment_tracking.has_seen(fid, key):
                        continue
                    # 无论最终是否接话，先标记已见，避免下轮重复评估
                    self.comment_tracking.mark_reply_seen(fid, key)

                    # 防刷楼 1：本 feed 接话次数达上限 → 仍追踪但只 mark seen 不再回复
                    if self.comment_tracking.get_reply_count(fid) >= per_feed_limit:
                        continue
                    # 防刷楼 2：同一 feed 的同一个人只接一次话
                    if self.comment_tracking.has_replied_to(fid, replier_uin):
                        continue

                    replier_name = rep.get("nickname", "") or replier_uin
                    reply_text = rep.get("content", "") or ""

                    # 结合记忆生成接话（客套收尾会返回 None 表示不接）
                    thread_reply = await self.content_service.generate_thread_reply(
                        host_name=host_name,
                        host_qq=host_qq,
                        story_content=detail.get("content", "") or bot_comment_content,
                        bot_comment=bot_comment_content,
                        replier_name=replier_name,
                        replier_qq=replier_uin,
                        reply_content=reply_text,
                    )
                    if not thread_reply:
                        logger.info(f"[接话] 对 {replier_name} 的回复选择不接话（客套/无实质内容/生成失败）")
                        continue

                    # parent_tid 用 bot 自己的评论楼层序号：把接话挂在 bot 起的楼里，content 里 @对方
                    parent_tid = bot_tid or rep_tid
                    ok = await api_client["reply"](fid, host_qq, replier_name, thread_reply, parent_tid)
                    if ok:
                        self.comment_tracking.record_reply(fid, replier_uin)
                        replies_done += 1
                        logger.info(f"[接话] 在 {host_name} 的说说楼里接了 {replier_name} 的话: '{thread_reply}'")
                        # 记人：把这次「在ta的说说下和ta聊了起来」记进空间互动记忆
                        await self._remember_interaction(
                            qq=replier_uin,
                            nickname=replier_name,
                            interaction_type="在ta的说说下和ta聊了起来",
                            their_content=reply_text,
                            bot_reply=thread_reply,
                        )
                        # 接话之间留较长间隔，模拟真人节奏、降低风控
                        await asyncio.sleep(random.uniform(10, 20))
                    else:
                        logger.warning(f"[接话] 回复 {replier_name} 失败(fid={fid})")

            logger.info(f"[接话] 本轮接话检查完成，共接话 {replies_done} 次")
        except Exception as e:
            logger.error(f"[接话] 接话闭环轮次异常（不影响监控主流程）: {e}")

    # --- 社交闭环只读能力（计数门 / 访客回访 / 回赞）---
    # 这些方法为 SocialLoopService 提供数据，全部失败隔离：任何异常都返回 None/[]，
    # 由上层决定「回退全量」或「跳过本轮」，绝不向监控主流程抛异常。

    async def get_unread_counts(self) -> dict | None:
        """获取 QQ 空间未读计数（轻量事件门用）。

        :return: 形如 ``{"feed": int, "comment": int, "visitor": int, "like": int}``；
                 任何失败（网络/解析/接口404）都返回 ``None``，调用方据此回退到全量执行。
        """
        try:
            qq_account = config_api.get_global_config("bot.qq_account", "")
            api_client = await self._get_api_client(qq_account, None)
            if not api_client:
                return None
            return await api_client["get_count"]()
        except Exception as e:
            logger.debug(f"获取未读计数异常（将回退全量）: {e}")
            return None

    async def get_visitors(self) -> list[dict]:
        """获取空间访客列表。

        :return: ``[{"uin": str, "name": str, "time": int}, ...]``；失败返回空列表。
        """
        try:
            qq_account = config_api.get_global_config("bot.qq_account", "")
            api_client = await self._get_api_client(qq_account, None)
            if not api_client:
                return []
            return await api_client["get_visitors"]()
        except Exception as e:
            logger.error(f"获取访客列表异常: {e}")
            return []

    async def get_recent_likers(self, feed_count: int = 3) -> list[dict]:
        """获取自己最近 ``feed_count`` 条说说的点赞者（跨说说去重合并）。

        :return: ``[{"uin": str, "name": str}, ...]``；失败返回空列表。
        """
        try:
            qq_account = config_api.get_global_config("bot.qq_account", "")
            api_client = await self._get_api_client(qq_account, None)
            if not api_client:
                return []
            own_feeds = await api_client["list_feeds"](str(qq_account), max(1, feed_count))
            if not own_feeds:
                return []
            seen: dict[str, str] = {}
            for feed in own_feeds:
                fid = feed.get("tid", "")
                if not fid:
                    continue
                likers = await api_client["get_likers"](fid)
                for lk in likers:
                    u = lk.get("uin")
                    if u and u not in seen:
                        seen[u] = lk.get("name", "") or ""
                # 名单接口之间留出间隔，降低风控风险
                await asyncio.sleep(random.uniform(2, 4))
            return [{"uin": u, "name": n} for u, n in seen.items()]
        except Exception as e:
            logger.error(f"获取最近点赞者异常: {e}")
            return []

    async def like_user_latest_feed(self, target_qq: str | int) -> dict:
        """给指定用户的最新一条说说点赞（只点赞不评论，用于回赞轻接触）。

        :return: ``{"success": bool, "tid": str, "message": str}``。
        """
        try:
            qq_account = config_api.get_global_config("bot.qq_account", "")
            api_client = await self._get_api_client(qq_account, None)
            if not api_client:
                return {"success": False, "tid": "", "message": "获取API客户端失败"}
            # 取几条以规避 _list_feeds 对已评论好友说说的过滤，取其中最新可见的一条
            feeds = await api_client["list_feeds"](str(target_qq), 3)
            if not feeds:
                return {"success": False, "tid": "", "message": "对方没有可见说说"}
            fid = feeds[0].get("tid", "")
            if not fid:
                return {"success": False, "tid": "", "message": "最新说说无有效tid"}
            ok = await api_client["like"](str(target_qq), fid)
            return {"success": bool(ok), "tid": fid, "message": "点赞成功" if ok else "点赞失败"}
        except RuntimeError as e:
            # QQ空间业务错误（含 Cookie 失效），交由上层记录，不抛出
            return {"success": False, "tid": "", "message": str(e)}
        except Exception as e:
            logger.error(f"给用户 {target_qq} 最新说说点赞异常: {e}")
            return {"success": False, "tid": "", "message": f"异常: {e}"}

    async def get_friend_feed_candidates(self, num: int = 20) -> list[dict]:
        """拉取好友动态时间线的原始候选列表（复用 ``monitor_list_feeds``），供转发锐评环节做候选过滤。

        **不是**新增的常态拉取途径：仅当转发环节通过「每日次数 + 尝试冷却」闸后，每冷却周期至多
        调用一次，因此不给监控主循环增加常态请求。任何失败返回 ``[]``，绝不影响主流程。

        返回元素结构同 ``monitor_list_feeds``：``{"target_qq","tid","content","rt_con","images","comments"}``。
        """
        try:
            qq_account = config_api.get_global_config("bot.qq_account", "")
            api_client = await self._get_api_client(qq_account, None)
            if not api_client:
                return []
            return await api_client["monitor_list_feeds"](num)
        except Exception as e:
            logger.error(f"[转发] 获取好友动态候选失败: {e}")
            return []

    async def repost_feed(
        self, target_qq: str | int, feed_id: str, original_content: str, comment: str
    ) -> dict:
        """把好友的（转发类）说说转发到自己空间并配一句短评。

        ⚠️ 转发接口未经 live 验证（详见 ``_repost``）。任何业务/网络失败均只返回失败字典、不抛出。

        :return: ``{"success": bool, "tid": str, "message": str}``。
        """
        try:
            qq_account = config_api.get_global_config("bot.qq_account", "")
            api_client = await self._get_api_client(qq_account, None)
            if not api_client:
                return {"success": False, "tid": "", "message": "获取API客户端失败"}
            ok, new_tid = await api_client["repost"](
                str(target_qq), str(feed_id), original_content or "", comment
            )
            return {
                "success": bool(ok),
                "tid": new_tid,
                "message": "转发成功" if ok else "转发失败（接口未经live验证，请核对返回）",
            }
        except RuntimeError as e:
            # QQ空间业务错误（含 Cookie 失效），交由上层记录，不抛出
            return {"success": False, "tid": "", "message": str(e)}
        except Exception as e:
            logger.error(f"[转发] 转发说说异常: {e}")
            return {"success": False, "tid": "", "message": f"异常: {e}"}

    def _generate_gtk(self, skey: str) -> str:
        hash_val = 5381
        for char in skey:
            hash_val += (hash_val << 5) + ord(char)
        return str(hash_val & 2147483647)

    async def _renew_and_load_cookies(self, qq_account: str, stream_id: str | None) -> dict[str, str] | None:
        cookie_dir = Path(__file__).resolve().parent.parent / "cookies"
        cookie_dir.mkdir(exist_ok=True)
        cookie_file_path = cookie_dir / f"cookies-{qq_account}.json"

        # 优先尝试通过Napcat HTTP服务获取最新的Cookie
        try:
            logger.info("尝试通过Napcat HTTP服务获取Cookie...")
            host = self.get_config("cookie.http_fallback_host", "172.20.130.55")
            port = self.get_config("cookie.http_fallback_port", "9999")
            napcat_token = self.get_config("cookie.napcat_token", "")

            cookie_data = await self._fetch_cookies_http(host, port, napcat_token)
            if cookie_data and "cookies" in cookie_data:
                cookie_str = cookie_data["cookies"]
                parsed_cookies = {
                    k.strip(): v.strip() for k, v in (p.split("=", 1) for p in cookie_str.split("; ") if "=" in p)
                }
                # 成功获取后，异步写入本地文件作为备份
                try:
                    async with aiofiles.open(cookie_file_path, "wb") as f:
                        await f.write(orjson.dumps(parsed_cookies))
                    logger.info(f"通过Napcat服务成功更新Cookie，并已保存至: {cookie_file_path}")
                except Exception as e:
                    logger.warning(f"保存Cookie到文件时出错: {e}")
                return parsed_cookies
            else:
                logger.warning("通过Napcat服务未能获取有效Cookie。")

        except Exception as e:
            logger.warning(f"通过Napcat HTTP服务获取Cookie时发生异常: {e}。将尝试从本地文件加载。")

        # 如果通过服务获取失败，则尝试从本地文件加载
        logger.info("尝试从本地Cookie文件加载...")
        if cookie_file_path.exists():
            try:
                async with aiofiles.open(cookie_file_path, "rb") as f:
                    content = await f.read()
                    cookies = orjson.loads(content)
                    logger.info(f"成功从本地文件加载Cookie: {cookie_file_path}")
                    return cookies
            except Exception as e:
                logger.error(f"从本地文件 {cookie_file_path} 读取或解析Cookie失败: {e}")
        else:
            logger.warning(f"本地Cookie文件不存在: {cookie_file_path}")

        logger.error("所有获取Cookie的方式均失败。")
        return None

    async def _fetch_cookies_http(self, host: str, port: int, napcat_token: str) -> dict | None:
        """通过HTTP服务器获取Cookie"""
        # 从配置中读取主机和端口，如果未提供则使用传入的参数
        final_host = self.get_config("cookie.http_fallback_host", host)
        final_port = self.get_config("cookie.http_fallback_port", port)
        url = f"http://{final_host}:{final_port}/get_cookies"

        max_retries = 5
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                headers = {"Content-Type": "application/json"}
                if napcat_token:
                    headers["Authorization"] = f"Bearer {napcat_token}"

                payload = {"domain": "user.qzone.qq.com"}

                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0)) as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        resp.raise_for_status()

                        if resp.status != 200:
                            error_msg = f"Napcat服务返回错误状态码: {resp.status}"
                            if resp.status == 403:
                                error_msg += " (Token验证失败)"
                            raise RuntimeError(error_msg)

                        data = await resp.json()
                        if data.get("status") != "ok" or "cookies" not in data.get("data", {}):
                            raise RuntimeError(f"获取 cookie 失败: {data}")
                        return data["data"]

            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"无法连接到Napcat服务(尝试 {attempt + 1}/{max_retries}): {url}，错误: {e!s}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                logger.error(f"无法连接到Napcat服务(最终尝试): {url}，错误: {e!s}")
                raise RuntimeError(f"无法连接到Napcat服务: {url}") from e
            except Exception as e:
                logger.error(f"获取cookie异常: {e!s}")
                raise

        raise RuntimeError(f"无法连接到Napcat服务: 超过最大重试次数({max_retries})")

    async def _get_api_client(self, qq_account: str, stream_id: str | None) -> dict | None:
        logger.debug(f"开始获取API客户端，qq_account={qq_account}")
        cookies = await self.cookie_service.get_cookies(qq_account, stream_id)
        if not cookies:
            logger.error(
                "获取API客户端失败：未能获取到Cookie。请检查Napcat连接是否正常，或是否存在有效的本地Cookie文件。"
            )
            return None

        logger.debug(f"Cookie获取成功，keys: {list(cookies.keys())}")

        p_skey = cookies.get("p_skey") or cookies.get("p_skey".upper())
        if not p_skey:
            logger.error(f"获取API客户端失败：Cookie中缺少关键的 'p_skey'。Cookie内容: {cookies}")
            return None

        logger.debug("p_skey获取成功")

        gtk = self._generate_gtk(p_skey)
        uin = cookies.get("uin", "").lstrip("o")
        if not uin:
            logger.error(f"获取API客户端失败：Cookie中缺少关键的 'uin'。Cookie内容: {cookies}")
            return None

        logger.debug(f"uin={uin}, gtk={gtk}, 准备构造API客户端")

        async def _request(method, url, params=None, data=None, headers=None):
            final_headers = {"referer": f"https://user.qzone.qq.com/{uin}", "origin": "https://user.qzone.qq.com"}
            if headers:
                final_headers.update(headers)

            async with aiohttp.ClientSession(cookies=cookies) as session:
                timeout = aiohttp.ClientTimeout(total=20)
                async with session.request(
                    method, url, params=params, data=data, headers=final_headers, timeout=timeout
                ) as response:
                    response.raise_for_status()
                    return await response.text()

        async def _publish(content: str, images: list[bytes]) -> tuple[bool, str]:
            """发布说说"""
            try:
                post_data = {
                    "syn_tweet_verson": "1",
                    "paramstr": "1",
                    "who": "1",
                    "con": content,
                    "feedversion": "1",
                    "ver": "1",
                    "ugc_right": "1",
                    "to_sign": "0",
                    "hostuin": uin,
                    "code_version": "1",
                    "format": "json",
                    "qzreferrer": f"https://user.qzone.qq.com/{uin}",
                }

                # 处理图片上传
                if images:
                    logger.info(f"开始上传 {len(images)} 张图片...")
                    pic_bos = []
                    richvals = []

                    for i, img_bytes in enumerate(images):
                        try:
                            # 上传图片到QQ空间
                            upload_result = await _upload_image(img_bytes, i)
                            if upload_result:
                                pic_bos.append(upload_result["pic_bo"])
                                richvals.append(upload_result["richval"])
                                logger.info(f"图片 {i + 1} 上传成功")
                            else:
                                logger.error(f"图片 {i + 1} 上传失败")
                        except Exception as e:
                            logger.error(f"上传图片 {i + 1} 时发生异常: {e}")

                    if pic_bos and richvals:
                        # 完全按照原版格式设置图片参数
                        post_data["pic_bo"] = ",".join(pic_bos)
                        post_data["richtype"] = "1"
                        post_data["richval"] = "\t".join(richvals)  # 原版使用制表符分隔

                        logger.info(f"准备发布带图说说: {len(pic_bos)} 张图片")
                        logger.info(f"pic_bo参数: {post_data['pic_bo']}")
                        logger.info(f"richval参数长度: {len(post_data['richval'])} 字符")
                    else:
                        logger.warning("所有图片上传失败，将发布纯文本说说")

                res_text = await _request("POST", self.EMOTION_PUBLISH_URL, params={"g_tk": gtk}, data=post_data)
                result = orjson.loads(res_text)
                tid = result.get("tid", "")

                if tid:
                    if images and pic_bos:
                        logger.info(f"成功发布带图说说，tid: {tid}，包含 {len(pic_bos)} 张图片")
                    else:
                        logger.info(f"成功发布文本说说，tid: {tid}")
                else:
                    logger.error(f"发布说说失败，API返回: {result}")

                return bool(tid), tid
            except Exception as e:
                logger.error(f"发布说说异常: {e}")
                return False, ""

        def _image_to_base64(image_bytes: bytes) -> str:
            """将图片字节转换为base64字符串（仿照原版实现）"""
            pic_base64 = base64.b64encode(image_bytes)
            return str(pic_base64)[2:-1]  # 去掉 b'...' 的前缀和后缀

        def _get_picbo_and_richval(upload_result: dict) -> tuple:
            """从上传结果中提取图片的picbo和richval值（仿照原版实现）"""
            json_data = upload_result

            if "ret" not in json_data:
                raise Exception("获取图片picbo和richval失败")

            if json_data["ret"] != 0:
                raise Exception("上传图片失败")

            # 从URL中提取bo参数
            picbo_spt = json_data["data"]["url"].split("&bo=")
            if len(picbo_spt) < 2:
                raise Exception("上传图片失败")
            picbo = picbo_spt[1]

            # 构造richval - 完全按照原版格式
            richval = ",{},{},{},{},{},{},,{},{}".format(
                json_data["data"]["albumid"],
                json_data["data"]["lloc"],
                json_data["data"]["sloc"],
                json_data["data"]["type"],
                json_data["data"]["height"],
                json_data["data"]["width"],
                json_data["data"]["height"],
                json_data["data"]["width"],
            )

            return picbo, richval

        async def _upload_image(image_bytes: bytes, index: int) -> dict[str, str] | None:
            """上传图片到QQ空间（完全按照原版实现）"""
            try:
                upload_url = "https://up.qzone.qq.com/cgi-bin/upload/cgi_upload_image"

                # 完全按照原版构建请求数据
                post_data = {
                    "filename": "filename",
                    "zzpanelkey": "",
                    "uploadtype": "1",
                    "albumtype": "7",
                    "exttype": "0",
                    "skey": cookies.get("skey", ""),
                    "zzpaneluin": uin,
                    "p_uin": uin,
                    "uin": uin,
                    "p_skey": cookies.get("p_skey", ""),
                    "output_type": "json",
                    "qzonetoken": "",
                    "refer": "shuoshuo",
                    "charset": "utf-8",
                    "output_charset": "utf-8",
                    "upload_hd": "1",
                    "hd_width": "2048",
                    "hd_height": "10000",
                    "hd_quality": "96",
                    "backUrls": "http://upbak.photo.qzone.qq.com/cgi-bin/upload/cgi_upload_image,"
                    "http://119.147.64.75/cgi-bin/upload/cgi_upload_image",
                    "url": f"https://up.qzone.qq.com/cgi-bin/upload/cgi_upload_image?g_tk={gtk}",
                    "base64": "1",
                    "picfile": _image_to_base64(image_bytes),
                }

                headers = {"referer": f"https://user.qzone.qq.com/{uin}", "origin": "https://user.qzone.qq.com"}

                logger.info(f"开始上传图片 {index + 1}...")

                async with aiohttp.ClientSession(cookies=cookies) as session:
                    timeout = aiohttp.ClientTimeout(total=60)
                    async with session.post(upload_url, data=post_data, headers=headers, timeout=timeout) as response:
                        if response.status == 200:
                            resp_text = await response.text()
                            logger.info(f"图片上传响应状态码: {response.status}")
                            logger.info(f"图片上传响应内容前500字符: {resp_text[:500]}")

                            # 按照原版方式解析响应
                            start_idx = resp_text.find("{")
                            end_idx = resp_text.rfind("}") + 1
                            if start_idx != -1 and end_idx != -1:
                                json_str = resp_text[start_idx:end_idx]
                                try:
                                    upload_result = orjson.loads(json_str)
                                except orjson.JSONDecodeError:
                                    logger.error(f"图片上传响应JSON解析失败，原始响应: {resp_text}")
                                    return None

                                logger.debug(f"图片上传解析结果: {upload_result}")

                                if upload_result.get("ret") == 0:
                                    try:
                                        # 使用原版的参数提取逻辑
                                        picbo, richval = _get_picbo_and_richval(upload_result)
                                        logger.info(f"图片 {index + 1} 上传成功: picbo={picbo}")
                                        return {"pic_bo": picbo, "richval": richval}
                                    except Exception as e:
                                        logger.error(
                                            f"从上传结果中提取图片参数失败: {e}, 上传结果: {upload_result}",
                                            exc_info=True,
                                        )
                                        return None
                                else:
                                    logger.error(f"图片 {index + 1} 上传失败: {upload_result}")
                                    return None
                            else:
                                logger.error(f"无法从响应中提取JSON内容: {resp_text}")
                                return None
                        else:
                            error_text = await response.text()
                            logger.error(f"图片上传HTTP请求失败，状态码: {response.status}, 响应: {error_text[:200]}")
                            return None

            except Exception as e:
                logger.error(f"上传图片 {index + 1} 异常: {e}")
                return None

        async def _list_feeds(t_qq: str, num: int) -> list[dict]:
            """获取指定用户说说列表 (统一接口)"""
            try:
                logger.debug(f"_list_feeds 开始，t_qq={t_qq}, num={num}")
                # 统一使用 format=json 获取完整评论
                params = {
                    "g_tk": gtk,
                    "uin": t_qq,
                    "ftype": 0,
                    "sort": 0,
                    "pos": 0,
                    "num": num,
                    "replynum": 999,  # 尽量获取更多
                    "code_version": 1,
                    "format": "json",  # 关键：使用JSON格式
                    "need_comment": 1,
                }
                logger.debug(f"准备发送HTTP请求到 {self.LIST_URL}")
                res_text = await _request("GET", self.LIST_URL, params=params)
                logger.debug(f"HTTP请求返回，响应长度={len(res_text)}")
                json_data = orjson.loads(res_text)
                logger.debug(f"JSON解析成功，code={json_data.get('code')}")

                if json_data.get("code") != 0:
                    error_code = json_data.get("code")
                    error_message = json_data.get("message", "未知错误")
                    logger.warning(f"获取说说列表API返回错误: code={error_code}, message={error_message}")

                    # 将API错误信息抛出，让上层处理并反馈给用户
                    raise RuntimeError(f"QQ空间API错误: {error_message} (错误码: {error_code})")

                feeds_list = []
                my_name = json_data.get("logininfo", {}).get("name", "")
                total_msgs = len(json_data.get("msglist", []))
                logger.debug(f"[DEBUG] 从API获取到 {total_msgs} 条原始说说")

                for idx, msg in enumerate(json_data.get("msglist", [])):
                    msg_tid = msg.get("tid", "")
                    msg_content = msg.get("content", "")
                    msg_rt_con = msg.get("rt_con")
                    is_retweet = bool(msg_rt_con)

                    logger.debug(f"[DEBUG] 说说 {idx+1}/{total_msgs}: tid={msg_tid}, 是否转发={is_retweet}, content长度={len(msg_content)}")
                    if is_retweet:
                        logger.debug(f"[DEBUG] rt_con类型={type(msg_rt_con)}, 内容={msg_rt_con}")

                    # 当读取的是好友动态时，检查是否已评论过，如果是则跳过
                    is_friend_feed = str(t_qq) != str(uin)
                    if is_friend_feed:
                        commentlist_for_check = msg.get("commentlist")
                        is_commented = False
                        if isinstance(commentlist_for_check, list):
                            is_commented = any(
                                c.get("name") == my_name for c in commentlist_for_check if isinstance(c, dict)
                            )
                        if is_commented:
                            logger.debug(f"[DEBUG] 跳过已评论的说说: tid={msg_tid}, 是否转发={is_retweet}")
                            continue

                    # --- 安全地处理图片列表 ---
                    images = []
                    if "pic" in msg and isinstance(msg["pic"], list):
                        images = [pic.get("url1", "") for pic in msg["pic"] if pic.get("url1")]
                    elif "pictotal" in msg and isinstance(msg["pictotal"], list):
                        images = [pic.get("url1", "") for pic in msg["pictotal"] if pic.get("url1")]

                    # --- 解析完整评论列表 (包括二级评论) ---
                    # 复用公共解析函数（同时供接话闭环的详情解析使用），避免两份拷贝。
                    # 返回的字典包含既有字段（qq_account/nickname/content/comment_tid/parent_tid/
                    # create_time），下游逻辑照常读取；多出的 create_ts/t2_subtype 等键会被忽略。
                    comments = _parse_comment_tree(msg.get("commentlist"))

                    feeds_list.append(
                        {
                            "tid": msg.get("tid", ""),
                            "content": msg.get("content", ""),
                            "created_time": time.strftime(
                                "%Y-%m-%d %H:%M:%S", time.localtime(msg.get("created_time", 0))
                            ),
                            "rt_con": msg.get("rt_con", {}).get("content", "") if isinstance(msg.get("rt_con"), dict) else msg.get("rt_con", ""),
                            "images": images,
                            "comments": comments,
                        }
                    )

                logger.info(f"成功获取到 {len(feeds_list)} 条说说 from {t_qq} (使用统一JSON接口)")
                return feeds_list
            except RuntimeError:
                # QQ空间API业务错误，向上传播让调用者处理
                raise
            except Exception as e:
                # 其他异常（如网络错误、JSON解析错误等），记录后返回空列表
                logger.error(f"获取说说列表失败: {e}")
                return []

        async def _comment(t_qq: str, feed_id: str, text: str) -> tuple[bool, str | None]:
            """评论说说。

            返回 ``(是否成功, 新评论tid)``；新评论 tid 从响应体多候选字段中尽力取出，
            拿不到则为 None（接话闭环会在检测阶段用 uin==bot 兜底定位 bot 的楼，不依赖此值）。
            """
            try:
                data = {
                    "topicId": f"{t_qq}_{feed_id}__1",
                    "uin": uin,
                    "hostUin": t_qq,
                    "content": text,
                    "format": "fs",
                    "plat": "qzone",
                    "source": "ic",
                    "platformid": 52,
                    "ref": "feeds",
                }
                response_text = await _request("POST", self.COMMENT_URL, params={"g_tk": gtk}, data=data)

                # 解析响应检查业务状态
                try:
                    response_data = orjson.loads(response_text)
                    code = response_data.get("code", -1)
                    if code == 0:
                        # 顺手从响应体里取新评论 tid（顶层或 data 子级，多候选防御性取值）
                        new_tid: str | None = None
                        candidates: list[dict] = [response_data]
                        if isinstance(response_data.get("data"), dict):
                            candidates.append(response_data["data"])
                        for src in candidates:
                            for k in ("commentid", "commentId", "tid", "comment_tid"):
                                v = src.get(k)
                                if v:
                                    new_tid = str(v)
                                    break
                            if new_tid:
                                break
                        logger.info(f"评论API返回成功: feed_id={feed_id}")
                        return True, new_tid
                    else:
                        message = response_data.get("message", "未知错误")
                        logger.error(f"评论API返回失败: code={code}, message={message}, feed_id={feed_id}")
                        return False, None
                except orjson.JSONDecodeError:
                    logger.warning(f"评论API响应无法解析为JSON，假定成功: {response_text[:200]}")
                    return True, None
            except Exception as e:
                logger.error(f"评论说说异常: {e}")
                return False, None

        async def _like(t_qq: str, feed_id: str) -> bool:
            """点赞说说"""
            try:
                data = {
                    "opuin": uin,
                    "unikey": f"http://user.qzone.qq.com/{t_qq}/mood/{feed_id}",
                    "curkey": f"http://user.qzone.qq.com/{t_qq}/mood/{feed_id}",
                    "from": 1,
                    "appid": 311,
                    "typeid": 0,
                    "abstime": int(time.time()),
                    "fid": feed_id,
                    "active": 0,
                    "format": "json",
                    "fupdate": 1,
                }
                response_text = await _request("POST", self.DOLIKE_URL, params={"g_tk": gtk}, data=data)

                # 解析响应检查业务状态
                try:
                    response_data = orjson.loads(response_text)
                    code = response_data.get("code", -1)
                    if code == 0:
                        logger.debug(f"点赞API返回成功: feed_id={feed_id}")
                        return True
                    else:
                        message = response_data.get("message", "未知错误")
                        logger.warning(f"点赞API返回失败: code={code}, message={message}, feed_id={feed_id}")
                        return False
                except orjson.JSONDecodeError:
                    logger.warning(f"点赞API响应无法解析为JSON，假定成功: {response_text[:200]}")
                    return True
            except Exception as e:
                logger.error(f"点赞说说异常: {e}")
                return False

        async def _reply(fid, host_qq, target_name, content, comment_tid):
            """回复评论 - 修复为能正确提醒的回复格式"""
            try:
                # 修复回复逻辑：确保能正确提醒被回复的人
                data = {
                    "topicId": f"{host_qq}_{fid}__1",
                    "parent_tid": comment_tid,
                    "uin": uin,
                    "hostUin": host_qq,
                    "content": content,
                    "format": "fs",
                    "plat": "qzone",
                    "source": "ic",
                    "platformid": 52,
                    "ref": "feeds",
                    "richtype": "",
                    "richval": "",
                    "paramstr": "",
                }

                # 记录详细的请求参数用于调试
                logger.info(
                    f"子回复请求参数: topicId={data['topicId']}, parent_tid={data['parent_tid']}, content='{content[:50]}...'"
                )

                response_text = await _request("POST", self.REPLY_URL, params={"g_tk": gtk}, data=data)

                # 解析响应检查业务状态
                try:
                    response_data = orjson.loads(response_text)
                    code = response_data.get("code", -1)
                    if code == 0:
                        logger.info(f"回复API返回成功: fid={fid}, parent_tid={comment_tid}")
                        return True
                    else:
                        message = response_data.get("message", "未知错误")
                        logger.error(f"回复API返回失败: code={code}, message={message}, fid={fid}")
                        return False
                except orjson.JSONDecodeError:
                    logger.warning(f"回复API响应无法解析为JSON，假定成功: {response_text[:200]}")
                    return True
            except Exception as e:
                logger.error(f"回复评论异常: {e}")
                return False

        async def _get_feed_detail(host_qq: str, feed_id: str) -> dict | None:
            """拉取单条说说详情（含完整评论楼 + 楼中楼），供好友说说接话闭环检测回复。

            走 taotao.qq.com 代理域的 ``emotion_cgi_msgdetail_v6``（GET，与 msglist 同签名），
            跨用户可读。任何失败返回 None，由调用方跳过该 feed，绝不影响监控主流程。

            返回：``{"feed_id", "host_qq", "content"(原文), "comments"(_parse_comment_tree 结果)}``。
            """
            try:
                params = {
                    "g_tk": gtk,
                    "uin": host_qq,  # 说说作者（楼主）
                    "tid": feed_id,
                    "t1_source": 1,
                    "ftype": 0,
                    "sort": 0,
                    "pos": 0,
                    "num": 20,
                    "code_version": 1,
                    "format": "json",
                    "need_private_comment": 1,
                }
                res_text = await _request("GET", self.MSGDETAIL_URL, params=params)
                data = _loads_lenient(res_text)
                if not isinstance(data, dict) or data.get("code") != 0:
                    code = data.get("code") if isinstance(data, dict) else "N/A"
                    logger.debug(f"获取说说详情返回异常: host={host_qq}, tid={feed_id}, code={code}")
                    return None

                # commentlist / content 可能在顶层，也可能嵌在 data 子级，做防御性多层取值
                commentlist = data.get("commentlist")
                story_content = data.get("content", "")
                inner = data.get("data") if isinstance(data.get("data"), dict) else {}
                if isinstance(inner, dict):
                    if not isinstance(commentlist, list):
                        commentlist = inner.get("commentlist")
                    if not story_content:
                        story_content = inner.get("content", "")

                # 首轮诊断：若 code=0 却没找到 commentlist，打出响应顶层键，便于排查嵌套位置
                if not isinstance(commentlist, list):
                    logger.debug(
                        f"[接话] 说说详情未定位到 commentlist（host={host_qq}, tid={feed_id}）；"
                        f"响应顶层键={list(data.keys())}"
                    )

                return {
                    "feed_id": str(feed_id),
                    "host_qq": str(host_qq),
                    "content": story_content or "",
                    "comments": _parse_comment_tree(commentlist),
                }
            except Exception as e:
                logger.debug(f"获取说说详情失败: host={host_qq}, tid={feed_id}, err={e}")
                return None

        async def _monitor_list_feeds(num: int) -> list[dict]:
            """监控好友动态"""
            try:
                params = {
                    "uin": uin,
                    "scope": 0,
                    "view": 1,
                    "filter": "all",
                    "flag": 1,
                    "applist": "all",
                    "pagenum": 1,
                    "count": num,
                    "format": "json",
                    "g_tk": gtk,
                    "useutf8": 1,
                    "outputhtmlfeed": 1,
                }
                res_text = await _request("GET", self.ZONE_LIST_URL, params=params)

                # 处理不同的响应格式
                json_str = ""
                stripped_res_text = res_text.strip()
                if stripped_res_text.startswith("_Callback(") and stripped_res_text.endswith(");"):
                    json_str = stripped_res_text[len("_Callback(") : -2]
                elif stripped_res_text.startswith("{") and stripped_res_text.endswith("}"):
                    json_str = stripped_res_text
                else:
                    logger.warning(f"意外的响应格式: {res_text[:100]}...")
                    return []

                json_str = json_str.replace("undefined", "null").strip()

                # 解析JSON
                try:
                    json_data = json5.loads(json_str)
                except Exception as parse_error:
                    logger.error(f"JSON解析失败: {parse_error}, 原始数据: {json_str[:200]}...")
                    return []

                # 检查JSON数据类型
                if not isinstance(json_data, dict):
                    logger.warning(f"解析后的JSON数据不是字典类型: {type(json_data)}")
                    return []

                # 检查错误码（在try-except之外，让异常能向上传播）
                if json_data.get("code") != 0:
                    error_code = json_data.get("code")
                    error_msg = json_data.get("message", "未知错误")
                    logger.warning(f"QQ空间API返回错误: code={error_code}, message={error_msg}")
                    # 抛出异常以便上层的重试机制捕获
                    raise RuntimeError(f"QQ空间API错误: {error_msg} (错误码: {error_code})")

                feeds_data = []
                if isinstance(json_data, dict):
                    data_level1 = json_data.get("data")
                    if isinstance(data_level1, dict):
                        feeds_data = data_level1.get("data", [])

                feeds_list = []
                for feed in feeds_data:
                    if not feed or not isinstance(feed, dict):
                        continue

                    if str(feed.get("appid", "")) != "311":
                        continue

                    target_qq = str(feed.get("uin", ""))
                    tid = feed.get("key", "")
                    if not target_qq or not tid:
                        continue

                    if target_qq == str(uin):
                        continue

                    html_content = feed.get("html", "")
                    if not html_content:
                        continue

                    soup = bs4.BeautifulSoup(html_content, "html.parser")

                    # DEBUG: 查找所有可能的文本容器
                    all_divs = soup.find_all("div")
                    for i, div in enumerate(all_divs[:10]):  # 只看前10个
                        div_class = div.get("class", [])
                        div_text = div.get_text(strip=True)[:50] if div.get_text(strip=True) else ""
                    like_btn = soup.find("a", class_="qz_like_btn_v3")
                    is_liked = False
                    if isinstance(like_btn, bs4.Tag) and like_btn.get("data-islike") == "1":
                        is_liked = True

                    if is_liked:
                        continue

                    # 提取说说主体内容（兼容普通说说和转发说说）
                    text_div = soup.find("div", class_="f-info")
                    if not text_div:
                        text_div = soup.find("div", class_="qz_summary")
                    text = text_div.get_text(strip=True) if isinstance(text_div, bs4.Tag) else ""

                    # 提取转发内容（从f-item中提取完整内容，然后去除已提取的text部分）
                    rt_con = ""
                    f_item_div = soup.find("div", class_="f-item")
                    if f_item_div:
                        full_text = f_item_div.get_text(strip=True)
                        # 如果完整文本包含了说说主体，去除主体部分得到转发内容
                        if text and full_text.startswith(text):
                            rt_con = full_text[len(text):].strip()
                        elif full_text and full_text != text:
                            rt_con = full_text

                    # --- 借鉴原版插件的精确图片提取逻辑 ---
                    image_urls = []
                    img_box = soup.find("div", class_="img-box")
                    if isinstance(img_box, bs4.Tag):
                        for img in img_box.find_all("img"):
                            if isinstance(img, bs4.Tag):
                                src = img.get("src")
                                if src and isinstance(src, str) and "qzonestyle.gtimg.cn" not in src:
                                    image_urls.append(src)

                    # 视频封面也视为图片
                    video_thumb = soup.select_one("div.video-img img")
                    if isinstance(video_thumb, bs4.Tag) and "src" in video_thumb.attrs:
                        image_urls.append(video_thumb["src"])

                    # 去重
                    images = list(set(image_urls))

                    comments = []
                    comment_divs = soup.find_all("div", class_="f-single-comment")
                    for comment_div in comment_divs:
                        if not isinstance(comment_div, bs4.Tag):
                            continue
                        # --- 处理主评论 ---
                        author_a = comment_div.find("a", class_="f-nick")
                        content_span = comment_div.find("span", class_="f-re-con")

                        if isinstance(author_a, bs4.Tag) and isinstance(content_span, bs4.Tag):
                            comments.append(
                                {
                                    "qq_account": str(comment_div.get("data-uin", "")),
                                    "nickname": author_a.get_text(strip=True),
                                    "content": content_span.get_text(strip=True),
                                    "comment_tid": comment_div.get("data-tid", ""),
                                    "parent_tid": None,  # 主评论没有父ID
                                }
                            )

                        # --- 处理这条主评论下的所有回复 ---
                        reply_divs = comment_div.find_all("div", class_="f-single-re")
                        for reply_div in reply_divs:
                            if not isinstance(reply_div, bs4.Tag):
                                continue
                            reply_author_a = reply_div.find("a", class_="f-nick")
                            reply_content_span = reply_div.find("span", class_="f-re-con")

                            if isinstance(reply_author_a, bs4.Tag) and isinstance(reply_content_span, bs4.Tag):
                                comments.append(
                                    {
                                        "qq_account": str(reply_div.get("data-uin", "")),
                                        "nickname": reply_author_a.get_text(strip=True),
                                        "content": reply_content_span.get_text(strip=True).lstrip(
                                            ": "
                                        ),
                                        "comment_tid": reply_div.get("data-tid", ""),
                                        "parent_tid": reply_div.get(
                                            "data-parent-tid", comment_div.get("data-tid", "")
                                        ),
                                    }
                                )

                    feeds_list.append(
                        {"target_qq": target_qq, "tid": tid, "content": text, "rt_con": rt_con, "images": images, "comments": comments}
                    )
                logger.info(f"监控任务发现 {len(feeds_list)} 条未处理的新说说。")
                return feeds_list
            except Exception as e:
                # 检查是否是Cookie失效错误（-3000），如果是则重新抛出
                if "错误码: -3000" in str(e):
                    logger.warning("监控任务遇到Cookie失效错误，重新抛出异常以触发上层重试")
                    raise  # 重新抛出异常，让上层处理
                logger.error(f"监控好友动态失败: {e}")
                return []

        async def _get_count() -> dict | None:
            """获取未读计数（新访客/新评论/新赞/新动态）。字段名做防御性多候选取值。

            实测该端点当前返回 404；此处保留完整实现并在任何失败时返回 None，
            由调用方回退到「全量执行」，因此接口不可用也不会影响监控。
            """
            try:
                params = {"uin": uin, "g_tk": gtk, "format": "json"}
                res_text = await _request("GET", self.COUNT_URL, params=params)
                data = _loads_lenient(res_text)
                if not isinstance(data, dict):
                    return None
                # 计数可能直接在顶层，也可能在 data 子级
                payload = data.get("data") if isinstance(data.get("data"), dict) else data

                def _pick(*keys: str) -> int:
                    for k in keys:
                        v = payload.get(k)
                        if isinstance(v, bool):
                            continue
                        if isinstance(v, (int, float)):
                            return int(v)
                        if isinstance(v, str) and v.strip().isdigit():
                            return int(v)
                    return 0

                return {
                    "feed": _pick("newfeeds", "unreadfeeds", "feed", "feedcount", "friendfeeds"),
                    "comment": _pick("commentcount", "newcomment", "comment", "replycount"),
                    "visitor": _pick("newvisitornum", "newvisitor", "visitorcount", "visitor", "visit"),
                    "like": _pick("likecount", "newlike", "like", "praise", "praisecount"),
                }
            except Exception as e:
                logger.debug(f"获取未读计数失败（将回退全量）: {e}")
                return None

        async def _get_visitors(page: int = 1) -> list[dict]:
            """拉取空间访客列表（已只读实测：走代理域，_Callback 包裹，code=0）。

            返回 ``[{"uin": str, "name": str, "time": int}, ...]``；失败返回空列表。
            """
            try:
                params = {"uin": uin, "mask": 2, "g_tk": gtk, "page": page, "fupdate": 1, "clear": 1}
                res_text = await _request("GET", self.VISITOR_URL, params=params)
                data = _loads_lenient(res_text)
                if not isinstance(data, dict) or data.get("code") != 0:
                    code = data.get("code") if isinstance(data, dict) else "N/A"
                    logger.warning(f"获取访客列表返回异常: code={code}")
                    return []
                inner = data.get("data") if isinstance(data.get("data"), dict) else {}
                items = inner.get("items") or []
                visitors: list[dict] = []
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    v_uin = it.get("uin")
                    if not v_uin:
                        continue
                    visitors.append(
                        {
                            "uin": str(v_uin),
                            "name": it.get("name", "") or "",
                            "time": int(it.get("time", 0) or 0),
                        }
                    )
                logger.debug(f"获取到 {len(visitors)} 条访客记录")
                return visitors
            except Exception as e:
                logger.error(f"获取访客列表失败: {e}")
                return []

        async def _get_likers(feed_id: str) -> list[dict]:
            """获取某条自己说说的点赞者名单。返回 ``[{"uin": str, "name": str}, ...]``；失败返回空列表。

            使用与点赞相同的 mood unikey；点赞名单字段做多候选防御性取值。
            """
            try:
                unikey = f"http://user.qzone.qq.com/{uin}/mood/{feed_id}"
                params = {
                    "uin": uin,
                    "unikey": unikey,
                    "begin_uin": 0,
                    "query_count": 60,
                    "if_first_page": 1,
                    "fupdate": 1,
                    "g_tk": gtk,
                }
                res_text = await _request("GET", self.LIKE_LIST_URL, params=params)
                data = _loads_lenient(res_text)
                if not isinstance(data, dict):
                    return []
                inner = data.get("data") if isinstance(data.get("data"), dict) else data
                raw = None
                for key in ("like_uin_info", "likemans", "info", "list"):
                    if isinstance(inner.get(key), list):
                        raw = inner[key]
                        break
                if raw is None:
                    return []
                likers: list[dict] = []
                for it in raw:
                    if not isinstance(it, dict):
                        continue
                    l_uin = it.get("fuin") or it.get("uin")
                    if not l_uin:
                        continue
                    likers.append({"uin": str(l_uin), "name": it.get("nick") or it.get("name") or ""})
                return likers
            except Exception as e:
                logger.debug(f"获取点赞者名单失败: {e}")
                return []

        async def _repost(target_qq: str, feed_id: str, original_content: str, comment: str) -> tuple[bool, str]:
            """转发好友的（转发类）说说到自己空间并配一句短评（转发锐评功能）。

            ⚠️ 接口未经 live 验证：QQ空间转发走 ``emotion_cgi_publish_v6`` 同族 CGI，本实现以
            ``_publish``（约 L1327-1391）的参数骨架为基准，追加 ``rt_tid``/``rt_uin``/``rt_con``
            转发标识；参数形状靠通用 QZone API 知识 + 既有 publish 模式推断。首次开启由 sol 真机验证。

            :param target_qq: 被转发说说作者的 QQ（uin）
            :param feed_id: 被转发说说的 tid
            :param original_content: 被转发的原始内容（防御性携带进 rt_con）
            :param comment: bot 的转发配文（短评）
            :return: ``(是否成功, 新说说tid)``；任何失败仅日志，返回 ``(False, "")``，绝不外溢。
            """
            try:
                # 以 _publish 的字段为基准，追加转发标识字段
                post_data = {
                    "syn_tweet_verson": "1",
                    "paramstr": "1",
                    "who": "1",
                    "con": comment,  # 转发配文（bot 的短评）
                    "feedversion": "1",
                    "ver": "1",
                    "ugc_right": "1",
                    "to_sign": "0",
                    "hostuin": uin,
                    "code_version": "1",
                    "format": "json",
                    "qzreferrer": f"https://user.qzone.qq.com/{uin}",
                    # --- 相对 _publish 新增的转发标识（未经 live 验证）---
                    "rt_tid": str(feed_id),  # 被转发说说 tid
                    "rt_uin": str(target_qq),  # 被转发说说作者 uin
                    "rt_con": original_content or "",  # 被转发内容（部分接口需要，防御性携带）
                    "richtype": "",
                    "richval": "",
                }

                res_text = await _request("POST", self.EMOTION_FORWARD_URL, params={"g_tk": gtk}, data=post_data)

                # 多候选解析新 tid（不同接口版本字段名不一，防御性取值）
                new_tid = ""
                try:
                    result = orjson.loads(res_text)
                    if isinstance(result, dict):
                        candidates: list[dict] = [result]
                        if isinstance(result.get("data"), dict):
                            candidates.append(result["data"])
                        for src in candidates:
                            for k in ("tid", "t1_tid", "new_tid", "fid"):
                                v = src.get(k)
                                if v:
                                    new_tid = str(v)
                                    break
                            if new_tid:
                                break
                        # 拿不到 tid 且响应码非成功时，记录响应码便于首轮排查
                        code = result.get("code", result.get("ret", 0))
                        if not new_tid and code not in (0, None):
                            message = result.get("message", result.get("msg", "未知错误"))
                            logger.error(
                                f"[转发] 转发API返回失败: code={code}, message={message}, src_tid={feed_id}"
                            )
                except orjson.JSONDecodeError:
                    logger.warning(f"[转发] 转发API响应无法解析为JSON: {res_text[:200]}")

                if new_tid:
                    logger.info(f"[转发] 成功转发说说 作者={target_qq} 源tid={feed_id} 新tid={new_tid}")
                else:
                    logger.error(f"[转发] 转发未拿到新 tid（可能失败或接口形状不符），源tid={feed_id}")
                return bool(new_tid), new_tid
            except Exception as e:
                logger.error(f"[转发] 转发说说异常: {e}")
                return False, ""

        logger.debug("API客户端构造完成，返回包含11个方法的字典")
        return {
            "publish": _publish,
            "list_feeds": _list_feeds,
            "comment": _comment,
            "like": _like,
            "reply": _reply,
            "get_feed_detail": _get_feed_detail,
            "monitor_list_feeds": _monitor_list_feeds,
            "get_count": _get_count,
            "get_visitors": _get_visitors,
            "get_likers": _get_likers,
            "repost": _repost,
        }
