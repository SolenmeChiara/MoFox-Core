"""AFC 按需看图动作模块

定义 view_image 动作：planner 在历史里看到某张图片（以 [图片id:xxx 描述：...] 形式出现）
且认为描述不足以判断、需要看清真实内容时，可输出该动作把图片"钉住"。被钉住的图会在同一轮
replyer 构建 prompt 时以真实图片形式注入，并在后续轮次持续出现，直到该消息滚出历史窗口自动失效。

该动作由 plan_executor 作为前置动作在回复生成之前 await 执行，以保证同轮可见。
仅限 AffinityFlowChatter 使用。
"""

import time
from typing import ClassVar

from src.chat.utils.chat_message_builder import (
    collect_pic_ids_from_context,
    prune_pinned_images,
    resolve_short_pic_id,
)
from src.common.logger import get_logger
from src.plugin_system import ActionActivationType, BaseAction, ChatMode

logger = get_logger("afc_view_image_action")

# 每个会话最多同时钉住的图片数（LRU 上限）
_MAX_PINNED_IMAGES = 3


class ViewImageAction(BaseAction):
    """View Image 动作 - 按需查看历史中的某张图片的真实内容"""

    # 动作基本信息
    action_name = "view_image"
    action_description = "查看历史消息中某张图片的真实内容。当图片的文字描述不足以让你判断、而你确实对该图片内容感兴趣时使用，被查看的图片会在你本轮回复时以真实图像形式呈现给你。"

    # 激活设置：始终可用，交由 planner 判断是否需要
    activation_type = ActionActivationType.ALWAYS
    mode_enable = ChatMode.ALL
    parallel_action = True  # 可与回复动作组合（由执行器保证在回复之前完成）

    # Chatter 限制：仅允许 AffinityFlowChatter 使用
    chatter_allow: ClassVar[list[str]] = ["AffinityFlowChatter"]

    # 动作参数定义
    action_parameters: ClassVar = {
        "image_id": "要查看的图片ID，来自历史消息中 [图片id:xxx ...] 里的那串 xxx（必需）",
        "reason": "想查看这张图片的原因（可选）",
    }

    # 动作使用场景
    action_require: ClassVar = [
        "历史消息里出现 [图片id:xxx 描述：...] 且你对该图片的具体内容感兴趣、仅凭描述无法判断时使用",
        "image_id 必须来自历史中真实出现过的 [图片id:xxx ...] 标记",
        "纯表情包、与当前话题无关的配图、描述已足够清楚的图片，不要使用",
        "同一张图片已经看过（已被钉住）时不要重复查看",
    ]

    # 关联类型
    associated_types: ClassVar[list[str]] = ["text"]

    async def execute(self) -> tuple[bool, str]:
        """把指定图片钉住到会话上下文的 pinned_images 集合中。"""
        try:
            raw_id = (self.action_data or {}).get("image_id", "")
            raw_id = str(raw_id).strip() if raw_id else ""
            if not raw_id:
                logger.warning(f"{self.log_prefix} [看图] 未提供 image_id，放弃")
                return False, ""

            stream_context = getattr(self.chat_stream, "context", None)
            if stream_context is None:
                logger.warning(f"{self.log_prefix} [看图] 无法获取会话上下文，放弃")
                return False, ""

            # 收集当前窗口内出现过的图片，用于短 id 解析与滚窗自愈
            window_pic_ids = collect_pic_ids_from_context(stream_context)
            resolved_id = resolve_short_pic_id(raw_id, window_pic_ids)
            if not resolved_id:
                logger.warning(
                    f"{self.log_prefix} [看图] 无法在当前窗口唯一定位图片 '{raw_id}'（未命中或歧义），放弃"
                )
                return False, ""

            pinned = stream_context.pinned_images
            # 先自愈：剔除已滚出窗口的旧钉住图，避免它们占用 LRU 名额
            prune_pinned_images(pinned, window_pic_ids)

            # 加入/刷新（重复查看仅刷新 LRU 时间，不额外占名额）
            pinned[resolved_id] = time.time()

            # LRU：超过上限踢掉最旧的
            if len(pinned) > _MAX_PINNED_IMAGES:
                overflow = len(pinned) - _MAX_PINNED_IMAGES
                for old_id, _ in sorted(pinned.items(), key=lambda kv: kv[1])[:overflow]:
                    del pinned[old_id]

            logger.info(f"{self.log_prefix} [看图] 钉住图片 {resolved_id[:8]}，当前钉住 {len(pinned)} 张")
            return True, ""

        except Exception as e:
            logger.error(f"{self.log_prefix} [看图] view_image 动作执行失败: {e}")
            return False, ""
