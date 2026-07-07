"""
社交闭环服务

在监控主循环之上编排三件「像人一样的社交」行为，全部由未读计数事件门驱动：

1. 计数门（run_gated_round）：先轻量拉取未读计数，仅对有增量的维度执行对应重活
   （好友动态 / 访客回访 / 回赞）。计数接口不可用时**回退全量**，绝不让监控停摆。
2. 访客回访（visitor_callback_round）：对 24h 内首次来访的新访客，复用现有
   ``read_and_process_feeds`` 完整链路（LLM 选择性评论 + 点赞 + 记人）回访其空间。
3. 回赞（like_back_round）：对给自己说说点赞的新点赞者，只给对方最新一条说说点赞
   （轻接触，不评论），并把对方注册进记人系统。

失败隔离原则：每个对外方法整体 try/except，任何异常只告警，绝不影响监控主流程。
去重 / 冷却使用 storage_api 本地存储（``data/plugin_data/*.json``），结构均为 ``{qq: last_ts}``。
"""

import asyncio
import random
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.common.logger import get_logger
from src.plugin_system.apis import config_api
from src.plugin_system.apis.storage_api import get_local_storage

if TYPE_CHECKING:
    from .person_memory_service import PersonMemoryService
    from .qzone_service import QZoneService

logger = get_logger("MaiZone.SocialLoop")

# 本地存储名（对应 data/plugin_data/*.json）
_VISITOR_STORAGE = "maizone_visitor_callback"
_LIKEBACK_STORAGE = "maizone_like_back"
_COUNTS_STORAGE = "maizone_unread_counts"


class SocialLoopService:
    """访客回访 + 回赞 + 未读计数事件门 的编排服务。"""

    def __init__(
        self,
        get_config: Callable,
        qzone_service: "QZoneService",
        person_memory: "PersonMemoryService | None" = None,
    ):
        self.get_config = get_config
        self.qzone_service = qzone_service
        self.person_memory = person_memory
        # 三个独立的本地存储：访客冷却、回赞冷却、上次未读计数
        self.visitor_store = get_local_storage(_VISITOR_STORAGE)
        self.likeback_store = get_local_storage(_LIKEBACK_STORAGE)
        self.counts_store = get_local_storage(_COUNTS_STORAGE)

    # ---------------- 工具方法 ----------------

    @staticmethod
    def _self_qq() -> str:
        """当前 bot 自己的 QQ 号（字符串），用于把自己从访客/点赞者中排除。"""
        return str(config_api.get_global_config("bot.qq_account", "") or "")

    @staticmethod
    def _load_map(store) -> dict:
        """从存储读出 ``{qq: last_ts}`` 映射（异常/脏数据时回退空字典）。"""
        data = store.get("data", {})
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _on_cooldown(cd_map: dict, qq: str, cooldown_seconds: float) -> bool:
        """判断某 qq 是否仍在冷却期内。"""
        last = cd_map.get(str(qq))
        if not isinstance(last, (int, float)):
            return False
        return (time.time() - last) < cooldown_seconds

    # ---------------- 1. 未读计数事件门 ----------------

    async def run_gated_round(self) -> None:
        """一轮「计数门」编排：拉计数 → 比较增量 → 只跑有增量的重活；计数不可用则回退全量。

        本方法由 MonitorService 的定时循环每轮调用一次，替代原先直接调用 monitor_feeds。
        整体 try/except，任何异常都不会向监控循环抛出。
        """
        try:
            use_gate = bool(self.get_config("social_loop.enable_unread_gate", True))
            current: dict | None = None
            if use_gate:
                current = await self.qzone_service.get_unread_counts()

            last = self._load_map(self.counts_store)
            delta = self._counts_delta(current, last)

            # 计数门日志：新增量 + 触发/跳过决策
            if delta["available"]:
                dv = max(0, current.get("visitor", 0) - int(last.get("visitor", 0) or 0))
                dc = max(0, current.get("comment", 0) - int(last.get("comment", 0) or 0))
                dl = max(0, current.get("like", 0) - int(last.get("like", 0) or 0))
                triggered = [
                    name
                    for name, on in (
                        ("好友动态", delta["feed"] or delta["comment"]),
                        ("访客回访", delta["visitor"]),
                        ("回赞", delta["like"]),
                    )
                    if on
                ]
                action = ("触发 " + "/".join(triggered)) if triggered else "本轮无增量，全部跳过"
                logger.info(f"[计数门] 新访客+{dv} 新评论+{dc} 新赞+{dl} → {action}")
            else:
                logger.info("[计数门] 计数接口不可用或未启用，本轮回退全量执行")

            # --- 好友动态（含自己说说评论回复）：现有 monitor_feeds ---
            if delta["feed"] or delta["comment"]:
                try:
                    await self.qzone_service.monitor_feeds()
                except Exception as e:
                    logger.error(f"[计数门] 执行好友动态监控失败: {e}")

            # --- 访客回访 ---
            if delta["visitor"]:
                try:
                    await self.visitor_callback_round()
                except Exception as e:
                    logger.error(f"[计数门] 访客回访轮次失败: {e}")

            # --- 回赞 ---
            if delta["like"]:
                try:
                    await self.like_back_round()
                except Exception as e:
                    logger.error(f"[计数门] 回赞轮次失败: {e}")

            # 仅当计数可用时，落盘本轮计数作为下次比较基准
            if isinstance(current, dict):
                self.counts_store.set("data", current)
        except Exception as e:
            logger.error(f"[计数门] 轮次编排异常（不影响监控主流程）: {e}")

    @staticmethod
    def _counts_delta(current: dict | None, last: dict) -> dict:
        """比较当前与上次计数，返回各维度是否「有新增」。

        - ``current`` 为 None（接口不可用/未启用）时，各维度一律 True 且 ``available=False``，
          即回退到全量执行（保持插件原有行为）。
        - 否则某维度当前值 > 上次值即视为有新增（首次运行 last 为空 → 有值即触发）。
        """
        if not isinstance(current, dict):
            return {"available": False, "feed": True, "comment": True, "visitor": True, "like": True}
        result = {"available": True}
        for key in ("feed", "comment", "visitor", "like"):
            cur = int(current.get(key, 0) or 0)
            prev = int(last.get(key, 0) or 0)
            result[key] = cur > prev
        return result

    # ---------------- 2. 访客回访闭环 ----------------

    async def visitor_callback_round(self) -> None:
        """对新访客发起回访（复用 read_and_process_feeds 完整链路）。整体失败隔离。"""
        try:
            if not self.get_config("social_loop.enable_visitor_callback", True):
                return

            visitors = await self.qzone_service.get_visitors()
            if not visitors:
                logger.info("[回访] 本轮无新访客")
                return

            cooldown_h = float(self.get_config("social_loop.visitor_callback_cooldown_hours", 24))
            cooldown_s = cooldown_h * 3600
            max_per_round = int(self.get_config("social_loop.visitor_callback_max_per_round", 2))
            delay_min = int(self.get_config("social_loop.visitor_callback_delay_min_seconds", 5))
            delay_max = int(self.get_config("social_loop.visitor_callback_delay_max_seconds", 30))

            self_qq = self._self_qq()
            cd_map = self._load_map(self.visitor_store)

            # 过滤：排除自己 + 冷却期内 + 去重（同一轮同 qq 只处理一次）
            fresh: list[dict] = []
            seen_this_round: set[str] = set()
            for v in visitors:
                qq = str(v.get("uin", ""))
                if not qq or qq == self_qq or qq in seen_this_round:
                    continue
                if self._on_cooldown(cd_map, qq, cooldown_s):
                    continue
                seen_this_round.add(qq)
                fresh.append(v)

            if not fresh:
                logger.info("[回访] 本轮无新访客")
                return

            targets = fresh[:max_per_round]
            for v in targets:
                qq = str(v.get("uin", ""))
                name = v.get("name", "") or qq
                # 先落冷却戳（即使回访失败也不在冷却期内反复重试，避免风控）
                cd_map[qq] = time.time()
                self.visitor_store.set("data", cd_map)

                # 回访前随机延迟，模拟真人节奏
                if delay_max > 0:
                    lo, hi = min(delay_min, delay_max), max(delay_min, delay_max)
                    await asyncio.sleep(random.uniform(lo, hi))

                logger.info(f"[回访] 访客 {name}({qq}) {cooldown_h:.0f}h内首次来访，前去回访")
                try:
                    # read_and_process_feeds 自带 LLM 选择性评论 + 点赞 + 记人集成
                    result = await self.qzone_service.read_and_process_feeds(qq, None)
                    if not result.get("success"):
                        logger.warning(f"[回访] 回访 {name}({qq}) 未成功: {result.get('message', '')}")
                except Exception as e:
                    logger.error(f"[回访] 回访 {name}({qq}) 异常: {e}")
        except Exception as e:
            logger.error(f"[回访] 访客回访轮次异常（不影响监控主流程）: {e}")

    # ---------------- 3. 回赞闭环 ----------------

    async def like_back_round(self) -> None:
        """对给自己说说点赞的新点赞者回赞其最新一条说说（只点赞不评论）。整体失败隔离。"""
        try:
            if not self.get_config("social_loop.enable_like_back", True):
                return

            feed_count = int(self.get_config("social_loop.like_back_recent_feed_count", 3))
            likers = await self.qzone_service.get_recent_likers(feed_count)
            if not likers:
                logger.info("[回赞] 本轮无新点赞者")
                return

            cooldown_h = float(self.get_config("social_loop.like_back_cooldown_hours", 48))
            cooldown_s = cooldown_h * 3600
            max_per_round = int(self.get_config("social_loop.like_back_max_per_round", 3))

            self_qq = self._self_qq()
            cd_map = self._load_map(self.likeback_store)

            fresh: list[dict] = []
            seen_this_round: set[str] = set()
            for lk in likers:
                qq = str(lk.get("uin", ""))
                if not qq or qq == self_qq or qq in seen_this_round:
                    continue
                if self._on_cooldown(cd_map, qq, cooldown_s):
                    continue
                seen_this_round.add(qq)
                fresh.append(lk)

            if not fresh:
                logger.info("[回赞] 本轮无新点赞者")
                return

            targets = fresh[:max_per_round]
            for lk in targets:
                qq = str(lk.get("uin", ""))
                name = lk.get("name", "") or qq
                # 先落冷却戳，避免失败时反复重试触发风控
                cd_map[qq] = time.time()
                self.likeback_store.set("data", cd_map)

                # 点赞间隔随机化，防风控
                await asyncio.sleep(random.uniform(2, 8))

                try:
                    result = await self.qzone_service.like_user_latest_feed(qq)
                    if result.get("success"):
                        logger.info(f"[回赞] {name}({qq}) 赞了你的说说，回赞了ta的最新动态")
                        # 记人：只注册对方（点赞太轻，不写 interaction 记录，避免记忆噪音）
                        if self.person_memory is not None:
                            try:
                                await self.person_memory.register_person(qq, name)
                            except Exception as e:
                                logger.warning(f"[回赞] 注册点赞者到记人系统失败(qq={qq}): {e}")
                    else:
                        logger.warning(f"[回赞] 回赞 {name}({qq}) 未成功: {result.get('message', '')}")
                except Exception as e:
                    logger.error(f"[回赞] 回赞 {name}({qq}) 异常: {e}")
        except Exception as e:
            logger.error(f"[回赞] 回赞轮次异常（不影响监控主流程）: {e}")
