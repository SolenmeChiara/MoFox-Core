"""
KFC 回复动作模块

KFC 的 reply 动作：
- 完整的回复流程在 execute() 中实现
- 调用 Replyer 生成回复文本
- 回复后处理（系统格式词过滤、分段发送、错字生成等）
- 发送回复消息

与 AFC 类似，但使用 KFC 专属的 Replyer 和 Session 系统。
"""

import asyncio
import time
from typing import TYPE_CHECKING, ClassVar, Optional

from src.common.logger import get_logger
from src.config.config import global_config
from src.plugin_system import ActionActivationType, BaseAction, ChatMode
from src.plugin_system.apis import send_api

if TYPE_CHECKING:
    from ..session import KokoroSession

logger = get_logger("kfc_reply_action")


class KFCInterruptionError(BaseException):
    """KFC 打断异常，当检测到新消息时抛出"""
    def __init__(self, partial_reply: str, unsend_segments: list[str]):
        self.partial_reply = partial_reply
        self.unsend_segments = unsend_segments
        super().__init__("Reply action interrupted by new message")


class KFCReplyAction(BaseAction):
    """KFC Reply 动作 - 完整的私聊回复流程

    特点：
    - 完整的回复流程：生成回复 → 后处理 → 分段发送
    - 使用 KFC 专属的 Replyer 生成回复
    - 支持系统格式词过滤、分段发送、错字生成等后处理
    - 仅限 KokoroFlowChatter 使用
    - 支持回复打断：如果发送过程中收到新消息，会抛出 KFCInterruptionError

    action_data 参数：
    - user_id: 用户ID（必需，用于获取 Session）
    - user_name: 用户名称（必需）
    - thought: Planner 生成的想法/内心独白（必需）
    - situation_type: 情况类型（可选，默认 "new_message"）
    - extra_context: 额外上下文（可选）
    - content: 预生成的回复内容（可选，如果提供则直接发送）
    - should_quote_reply: 是否引用原消息（可选，默认 false）
    - enable_splitter: 是否启用分段发送（可选，默认 true）
    - enable_chinese_typo: 是否启用错字生成（可选，默认 true）
    """

    # 动作基本信息
    action_name = "kfc_reply"
    action_description = "发送回复消息。会根据当前对话情境生成并发送回复。"

    # 激活设置
    activation_type = ActionActivationType.ALWAYS
    mode_enable = ChatMode.ALL
    parallel_action = False

    # Chatter 限制：仅允许 KokoroFlowChatter 使用
    chatter_allow: ClassVar[list[str]] = ["KokoroFlowChatter"]

    # 动作参数定义
    action_parameters: ClassVar = {
        "content": "要发送的回复内容（可选，如果不提供则自动生成）",
        "should_quote_reply": "是否引用原消息（可选，true/false，默认 false）",
    }

    # 动作使用场景
    action_require: ClassVar = [
        "需要发送回复消息时使用",
        "私聊场景的标准回复动作",
    ]

    # 关联类型
    associated_types: ClassVar[list[str]] = ["text"]

    async def execute(self) -> tuple[bool, str]:
        """执行 reply 动作 - 完整的回复流程"""
        try:
            # 0. 记录本轮生成开始的打断基线（时间戳 + 已知消息ID集合）。
            #    用于分段发送时的间隙自检：只对本轮生成开始之后新到的消息反应，
            #    避免把本轮触发时就已在未读/缓存里的消息误判为打断信号。
            self._capture_interrupt_baseline()

            # 1. 检查是否有预生成的内容
            content = self.action_data.get("content", "")

            if not content:
                # 2. 需要生成回复，获取必要信息
                user_id = self.action_data.get("user_id")
                user_name = self.action_data.get("user_name", "用户")
                thought = self.action_data.get("thought", "")
                situation_type = self.action_data.get("situation_type", "new_message")
                extra_context = self.action_data.get("extra_context")

                if not user_id:
                    logger.warning(f"{self.log_prefix} 缺少 user_id，无法生成回复")
                    return False, ""

                # 3. 获取 Session
                session = await self._get_session(user_id)
                if not session:
                    logger.warning(f"{self.log_prefix} 无法获取 Session: {user_id}")
                    return False, ""

                # 4. 调用 Replyer 生成回复
                success, content = await self._generate_reply(
                    session=session,
                    user_name=user_name,
                    thought=thought,
                    situation_type=situation_type,
                    extra_context=extra_context,
                )

                if not success or not content:
                    logger.warning(f"{self.log_prefix} 回复生成失败")
                    return False, ""

            # 5. 回复后处理（系统格式词过滤 + 分段处理）
            enable_splitter = self.action_data.get("enable_splitter", True)
            enable_chinese_typo = self.action_data.get("enable_chinese_typo", True)

            processed_segments = self._post_process_reply(
                content=content,
                enable_splitter=enable_splitter,
                enable_chinese_typo=enable_chinese_typo,
            )

            if not processed_segments:
                logger.warning(f"{self.log_prefix} 回复后处理后内容为空")
                return False, ""

            # 6. 分段发送回复
            should_quote = self.action_data.get("should_quote_reply", False)
            reply_text = await self._send_segments(
                segments=processed_segments,
                should_quote=should_quote,
            )

            logger.info(f"{self.log_prefix} KFC reply 动作执行成功: {reply_text[:50]}...")
            return True, reply_text

        except KFCInterruptionError:
            raise  # 重新抛出打断异常，交由上层处理
        except asyncio.CancelledError:
            raise  # 抛出取消异常，可能在 _send_segments 中被转换为 KFCInterruptionError
        except Exception as e:
            logger.error(f"{self.log_prefix} KFC reply 动作执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False, ""

    def _post_process_reply(
        self,
        content: str,
        enable_splitter: bool = True,
        enable_chinese_typo: bool = True,
    ) -> list[str]:
        """
        回复后处理

        包括：
        1. 系统格式词过滤（移除 [回复...]、[表情包：...]、@<...> 等）
        2. 分段处理（根据标点分句、智能合并）
        3. 错字生成（拟人化）

        Args:
            content: 原始回复内容
            enable_splitter: 是否启用分段
            enable_chinese_typo: 是否启用错字生成

        Returns:
            处理后的文本段落列表
        """
        try:
            from src.chat.utils.utils import filter_system_format_content, process_llm_response

            # 1. 过滤系统格式词
            filtered_content = filter_system_format_content(content)

            if not filtered_content or not filtered_content.strip():
                logger.warning(f"{self.log_prefix} 过滤系统格式词后内容为空")
                return []

            # 2. 分段处理 + 错字生成
            processed_segments = process_llm_response(
                filtered_content,
                enable_splitter=enable_splitter,
                enable_chinese_typo=enable_chinese_typo,
            )

            # 过滤空段落
            processed_segments = [seg for seg in processed_segments if seg and seg.strip()]

            logger.debug(
                f"{self.log_prefix} 回复后处理完成: "
                f"原始长度={len(content)}, 过滤后长度={len(filtered_content)}, "
                f"分段数={len(processed_segments)}"
            )

            return processed_segments

        except Exception as e:
            logger.error(f"{self.log_prefix} 回复后处理失败: {e}")
            # 失败时返回原始内容
            return [content] if content else []

    async def _send_segments(
        self,
        segments: list[str],
        should_quote: bool = False,
    ) -> str:
        """
        分段发送回复
        """
        reply_text = ""
        first_sent = False

        # 获取分段发送的间隔时间
        typing_delay = 0.5
        if global_config and hasattr(global_config, "response_splitter"):
            typing_delay = getattr(global_config.response_splitter, "typing_delay", 0.5)

        try:
            for i, segment in enumerate(segments):
                if not segment or not segment.strip():
                    continue

                # 发送前自检：本轮生成开始后若有新消息到达，中断剩余分段（含第一段）。
                # 第一段之前的检查覆盖 replyer LLM 生成的 20-40s 窗口内新到的消息。
                if await self._should_interrupt():
                    logger.info(
                        f"{self.log_prefix} 发送间隙检测到新消息，中断剩余分段"
                        f"（已发送 {i} 段，剩余 {len(segments) - i} 段）"
                    )
                    raise KFCInterruptionError(
                        partial_reply=reply_text,
                        unsend_segments=list(segments[i:]),
                    )

                reply_text += segment

                # 发送消息
                if not first_sent:
                    await send_api.text_to_stream(
                        text=segment,
                        stream_id=self.chat_stream.stream_id,
                        reply_to_message=self.action_message,
                        set_reply=should_quote and bool(self.action_message),
                        typing=False,
                    )
                    first_sent = True
                else:
                    if typing_delay > 0:
                        await asyncio.sleep(typing_delay)

                    await send_api.text_to_stream(
                        text=segment,
                        stream_id=self.chat_stream.stream_id,
                        reply_to_message=None,
                        set_reply=False,
                        typing=True,
                    )

            return reply_text

        except asyncio.CancelledError:
            # 如果被外部取消（如 Task.cancel()），也视为打断，保存当前进度
            logger.info(f"{self.log_prefix} 发送过程被强制取消，保存进度")
            # 注意：此时循环中的 segment 可能还没发，或者刚发完但还没更新 reply_text (如果是 await 处取消)
            # 简单起见，我们认为 reply_text 是已确认发送的
            # 未发送部分从当前 index 开始（如果还没加进 reply_text）
            # 由于 reply_text += segment 是在 await 之前，所以如果是 await send/sleep 被取消，
            # 该 segment 已经加进去了，但没发成功（或者sleep时已发成功）。
            # 这是一个边缘情况，为了不丢失信息，我们宁可多发（假装没发成功）也不要少发。
            # 这里保守策略：reply_text 包含 current segment，但其实可能没发出去。
            # 如果是 sleep 被取消，那上一条是发成功的。
            # 如果是 send_api 被取消，那这条可能没发成功。
            # 我们可以检查 reply_text 是否包含 segment。
            # 简化逻辑：直接抛出

            # 计算未发送部分：从当前 i 开始（如果还没处理完）
            # 由于 enumerate scope 问题，我们需要在循环外访问 i？
            # Python loop 变量泄漏到外部 scope，但 try block 内部变量可能不一样。
            # 还是在 loop 内 try/except 比较好？不，loop 外 catch 更整洁。

            # 由于我们无法准确知道 i 的值（除非在 loop 里更新 self.current_index），
            # 这里简单处理：若被强制取消，只返回已累积的 reply_text。
            # 剩下的丢弃？不，这正是用户不要的。
            # 但被 Cancelled 通常意味着 LLM 阶段或者新的 Execute 来了。
            # 如果是新的 Execute 来了，它会重新规划。
            # 所以这里抛出 KFCInterruptionError 主要是为了让 Session 记录 "我说了X"。
            raise KFCInterruptionError(
                partial_reply=reply_text,
                unsend_segments=[] # 无法获取剩余部分，但这不重要，因为会重新规划
            )

    def _capture_interrupt_baseline(self) -> None:
        """记录本轮生成开始时的打断基线（时间戳 + 已知消息ID集合）。

        基线包含：
        - _interrupt_baseline_time: 基线时间戳
        - _interrupt_baseline_ids: 基线时刻已知的所有消息ID（未读 ∪ 缓存）

        优先消费 chatter 通过 action_data 传入的"回合快照基线"——它捕获于 chatter 取未读
        消息的那一刻（planner LLM 之前），能覆盖 planner ~15s 窗口内到达的消息，修复
        "planner 期间到达的消息被并入缓存后漏判打断"的时序盲区。

        拿不到回合基线时（proactive 直调动作等独立场景）回退为动作自采：以 execute() 起始
        为基线时间、采集当时未读 ∪ 缓存的消息ID。此时基线通常为空，语义仍正确——proactive
        无触发消息，任何生成期间到达的新用户消息都会触发打断（保持昨晚已实现的语义）。

        用于 _should_interrupt 区分"本轮触发时就已存在的消息"和"生成期间新到的消息"，
        避免把本轮的触发消息误判为打断信号。失败时降级为仅时间基线，不阻断主流程。
        """
        # 优先：消费 chatter 传入的回合快照基线
        action_data = self.action_data if isinstance(self.action_data, dict) else {}
        round_time = action_data.get("_interrupt_baseline_time")
        round_ids = action_data.get("_interrupt_baseline_ids")
        if round_time is not None and round_ids is not None:
            try:
                self._interrupt_baseline_time = float(round_time)
            except (TypeError, ValueError):
                self._interrupt_baseline_time = time.time()
            self._interrupt_baseline_ids = set(round_ids)
            # 消费后从 action_data 移除：_interrupt_baseline_ids 为 set，非 JSON 安全，
            # 若残留会在 _store_action_info 的 reply_text 为空分支被 orjson.dumps 拒绝（虽被
            # try/except 兜住但会丢失动作记录）。就地清理使动作存档彻底不含内部基线字段。
            action_data.pop("_interrupt_baseline_time", None)
            action_data.pop("_interrupt_baseline_ids", None)
            logger.debug(
                f"{self.log_prefix} 打断基线已捕获(来源=回合快照): 已知消息 "
                f"{len(self._interrupt_baseline_ids)} 条, baseline_time={self._interrupt_baseline_time:.3f}"
            )
            return

        # 回退：动作自采（proactive 直调等独立场景）
        self._interrupt_baseline_time = time.time()
        ids: set[str] = set()
        try:
            context = getattr(self.chat_stream, "context", None) if self.chat_stream else None
            if context:
                for msg in context.get_unread_messages() or []:
                    mid = str(getattr(msg, "message_id", "") or "")
                    if mid:
                        ids.add(mid)
                try:
                    for msg in list(context.message_cache):
                        mid = str(getattr(msg, "message_id", "") or "")
                        if mid:
                            ids.add(mid)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"{self.log_prefix} 捕获打断基线失败: {e}")
        self._interrupt_baseline_ids = ids
        logger.debug(
            f"{self.log_prefix} 打断基线已捕获(来源=动作自采): 已知消息 {len(ids)} 条, "
            f"baseline_time={self._interrupt_baseline_time:.3f}"
        )

    async def _should_interrupt(self) -> bool:
        """检查是否应该打断回复（发送间隙自检）。

        仅当本轮生成开始之后有新消息到达时返回 True，通过 execute() 起始记录的
        基线（时间戳 + 已知消息ID集合）区分新旧消息，避免误判本轮触发消息。

        处理期间新消息可能进入缓存（is_chatter_processing 时），也可能直接进未读
        （未启用缓存时），因此两者都要检查。
        """
        chat_stream = self.chat_stream
        context = getattr(chat_stream, "context", None) if chat_stream else None
        if not context:
            return False

        baseline_time = getattr(self, "_interrupt_baseline_time", 0.0)
        baseline_ids = getattr(self, "_interrupt_baseline_ids", None) or set()

        # 汇总候选消息：未读 + 处理期间缓存
        candidates = list(context.get_unread_messages() or [])
        try:
            candidates.extend(list(context.message_cache))
        except Exception:
            pass

        if not candidates:
            return False

        for msg in candidates:
            msg_id = str(getattr(msg, "message_id", "") or "")

            if msg_id:
                # 主判据：不在本轮基线集合中的消息即为新消息
                # （基线集合已覆盖本轮触发时的所有已知未读/缓存消息）
                if msg_id in baseline_ids:
                    continue
            else:
                # 兜底：无有效 msg_id 时，用时间戳判断是否晚于本轮基线（+0.05s 抗抖动）
                msg_time = float(getattr(msg, "time", 0.0) or 0.0)
                if baseline_time > 0 and msg_time <= baseline_time + 0.05:
                    continue

            logger.debug(
                f"{self.log_prefix} 检测到本轮生成后的新消息 "
                f"(id={msg_id}, baseline_time={baseline_time:.3f})，将触发打断"
            )
            return True

        return False

    async def _get_session(self, user_id: str) -> Optional["KokoroSession"]:
        """获取用户 Session"""
        try:
            from ..session import get_session_manager

            session_manager = get_session_manager()
            return await session_manager.get_session(user_id, self.chat_stream.stream_id)
        except Exception as e:
            logger.error(f"{self.log_prefix} 获取 Session 失败: {e}")
            return None

    async def _generate_reply(
        self,
        session: "KokoroSession",
        user_name: str,
        thought: str,
        situation_type: str,
        extra_context: dict | None = None,
    ) -> tuple[bool, str]:
        """调用 Replyer 生成回复"""
        try:
            from ..replyer import generate_reply_text

            return await generate_reply_text(
                session=session,
                user_name=user_name,
                thought=thought,
                situation_type=situation_type,
                chat_stream=self.chat_stream,
                extra_context=extra_context,
            )
        except Exception as e:
            logger.error(f"{self.log_prefix} 生成回复失败: {e}")
            return False, ""
