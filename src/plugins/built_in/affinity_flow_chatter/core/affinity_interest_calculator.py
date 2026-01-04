"""AffinityFlow 风格兴趣值计算组件

基于原有的 AffinityFlow 兴趣度评分系统，提供标准化的兴趣值计算功能
集成了语义兴趣度计算（TF-IDF + Logistic Regression）

2024.12 优化更新：
- 使用 FastScorer 优化评分（绕过 sklearn，纯 Python 字典计算）
- 支持批处理队列模式（高频群聊场景）
- 全局线程池避免重复创建 executor
- 更短的超时时间（2秒）
"""

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from src.common.logger import get_logger
from src.config.config import global_config
from src.plugin_system.base.base_interest_calculator import BaseInterestCalculator, InterestCalculationResult

if TYPE_CHECKING:
    from src.common.data_models.database_data_model import DatabaseMessages

logger = get_logger("affinity_interest_calculator")


class AffinityInterestCalculator(BaseInterestCalculator):
    """AffinityFlow 风格兴趣值计算组件"""

    # 直接定义类属性
    component_name = "affinity_interest_calculator"
    component_version = "1.0.0"
    component_description = "基于AffinityFlow逻辑的兴趣值计算组件，使用智能兴趣匹配和用户关系评分"

    def __init__(self):
        super().__init__()

        # 智能兴趣匹配配置（已在类属性中定义）

        # 从配置加载评分权重
        affinity_config = global_config.affinity_flow
        self.score_weights = {
            "semantic": 0.5,  # 语义兴趣度权重（核心维度）
            "relationship": affinity_config.relationship_weight,  # 关系分权重
            "mentioned": affinity_config.mention_bot_weight,  # 是否提及bot权重
        }

        # 语义兴趣度评分器（替代原有的 embedding 兴趣匹配）
        self.semantic_scorer = None
        self.use_semantic_scoring = True  # 必须启用
        self._semantic_initialized = False  # 防止重复初始化
        self.model_manager = None

        # 评分阈值
        self.reply_threshold = affinity_config.reply_action_interest_threshold  # 回复动作兴趣阈值
        self.mention_threshold = affinity_config.mention_bot_adjustment_threshold  # 提及bot后的调整阈值

        # 连续不回复概率提升
        self.no_reply_count = 0
        self.max_no_reply_count = affinity_config.max_no_reply_count
        self.reply_cooldown_reduction = affinity_config.reply_cooldown_reduction  # 回复后减少的不回复计数
        if self.max_no_reply_count > 0:
            self.probability_boost_per_no_reply = (
                affinity_config.no_reply_threshold_adjustment / self.max_no_reply_count
            )
        else:
            self.probability_boost_per_no_reply = 0.0  # 避免除以零的错误

        # 用户关系数据缓存
        self.user_relationships: dict[str, float] = {}  # user_id -> relationship_score

        # 回复后阈值降低机制
        self.enable_post_reply_boost = affinity_config.enable_post_reply_boost
        self.post_reply_boost_remaining = 0  # 剩余的回复后降低次数
        self.post_reply_threshold_reduction = affinity_config.post_reply_threshold_reduction
        self.post_reply_boost_max_count = affinity_config.post_reply_boost_max_count
        self.post_reply_boost_decay_rate = affinity_config.post_reply_boost_decay_rate

        logger.info("[Affinity兴趣计算器] 初始化完成（基于语义兴趣度 TF-IDF+LR）:")
        logger.info(f"  - 权重配置: {self.score_weights}")
        logger.info(f"  - 回复阈值: {self.reply_threshold}")
        logger.info(f"  - 语义评分: {self.use_semantic_scoring} (TF-IDF + Logistic Regression + FastScorer优化)")
        logger.info(f"  - 回复后连续对话: {self.enable_post_reply_boost}")
        logger.info(f"  - 回复冷却减少: {self.reply_cooldown_reduction}")
        logger.info(f"  - 最大不回复计数: {self.max_no_reply_count}")

        # 异步初始化语义评分器
        asyncio.create_task(self._initialize_semantic_scorer())

    async def execute(self, message: "DatabaseMessages") -> InterestCalculationResult:
        """执行AffinityFlow风格的兴趣值计算"""
        try:
            start_time = time.time()
            message_id = getattr(message, "message_id", "")
            content = getattr(message, "processed_plain_text", "")
            user_info = getattr(message, "user_info", None)
            if user_info and hasattr(user_info, "user_id"):
                user_id = user_info.user_id
            else:
                user_id = ""

            logger.debug(f"[Affinity兴趣计算] 开始处理消息 {message_id}")
            logger.debug(f"[Affinity兴趣计算] 消息内容: {content[:50]}...")
            logger.debug(f"[Affinity兴趣计算] 用户ID: {user_id}")

            # 1. 计算语义兴趣度（核心维度，替代原 embedding 兴趣匹配）
            semantic_score = await self._calculate_semantic_score(content)
            logger.debug(f"[Affinity兴趣计算] 语义兴趣度（TF-IDF+LR）: {semantic_score}")

            # 2. 计算关系分
            relationship_score = await self._calculate_relationship_score(user_id)
            logger.debug(f"[Affinity兴趣计算] 关系分: {relationship_score}")

            # 3. 计算提及分
            mentioned_score = self._calculate_mentioned_score(message, global_config.bot.nickname)
            logger.debug(f"[Affinity兴趣计算] 提及分: {mentioned_score}")

            # 4. 综合评分
            # 确保所有分数都是有效的 float 值
            semantic_score = float(semantic_score) if semantic_score is not None else 0.0
            relationship_score = float(relationship_score) if relationship_score is not None else 0.0
            mentioned_score = float(mentioned_score) if mentioned_score is not None else 0.0

            raw_total_score = (
                semantic_score * self.score_weights["semantic"]
                + relationship_score * self.score_weights["relationship"]
                + mentioned_score * self.score_weights["mentioned"]
            )

            # 限制总分上限为1.0，确保分数在合理范围内
            total_score = min(raw_total_score, 1.0)

            logger.debug(
                f"[Affinity兴趣计算] 综合得分计算: "
                f"{semantic_score:.3f}*{self.score_weights['semantic']} + "
                f"{relationship_score:.3f}*{self.score_weights['relationship']} + "
                f"{mentioned_score:.3f}*{self.score_weights['mentioned']} = {raw_total_score:.3f}"
            )

            if raw_total_score > 1.0:
                logger.debug(f"[Affinity兴趣计算] 原始分数 {raw_total_score:.3f} 超过1.0，已限制为 {total_score:.3f}")

            # 5. 考虑连续不回复的阈值调整
            adjusted_score = total_score
            adjusted_reply_threshold, adjusted_action_threshold = self._apply_threshold_adjustment()
            logger.debug(
                f"[Affinity兴趣计算] 连续不回复调整: 回复阈值 {self.reply_threshold:.3f} → {adjusted_reply_threshold:.3f}, "
                f"动作阈值 {global_config.affinity_flow.non_reply_action_interest_threshold:.3f} → {adjusted_action_threshold:.3f}"
            )

            # 6. 决定是否回复和执行动作
            should_reply = adjusted_score >= adjusted_reply_threshold
            should_take_action = adjusted_score >= adjusted_action_threshold

            logger.debug(
                f"[Affinity兴趣计算] 阈值判断: {adjusted_score:.3f} >= 回复阈值:{adjusted_reply_threshold:.3f}? = {should_reply}"
            )
            logger.debug(
                f"[Affinity兴趣计算] 阈值判断: {adjusted_score:.3f} >= 动作阈值:{adjusted_action_threshold:.3f}? = {should_take_action}"
            )

            calculation_time = time.time() - start_time

            logger.debug(
                f"Affinity兴趣值计算完成 - 消息 {message_id}: {adjusted_score:.3f} "
                f"(语义:{semantic_score:.2f}, 关系:{relationship_score:.2f}, 提及:{mentioned_score:.2f})"
            )

            return InterestCalculationResult(
                success=True,
                message_id=message_id,
                interest_value=adjusted_score,
                should_take_action=should_take_action,
                should_reply=should_reply,
                should_act=should_take_action,
                calculation_time=calculation_time,
            )

        except Exception as e:
            logger.error(f"Affinity兴趣值计算失败: {e}")
            return InterestCalculationResult(
                success=False, message_id=getattr(message, "message_id", ""), interest_value=0.0, error_message=str(e)
            )

    async def _calculate_relationship_score(self, user_id: str) -> float:
        """计算用户关系分"""
        if not user_id:
            return global_config.affinity_flow.base_relationship_score

        # 优先使用内存中的关系分
        if user_id in self.user_relationships:
            relationship_value = self.user_relationships[user_id]
            # 移除关系分上限，允许超过1.0，最终分数会被整体限制
            return relationship_value

        # 如果内存中没有，尝试从统一的评分API获取
        try:
            from src.plugin_system.apis import person_api

            relationship_data = await person_api.get_user_relationship_data(user_id)
            if relationship_data:
                relationship_score = relationship_data.get("relationship_score", global_config.affinity_flow.base_relationship_score)
                # 同时更新内存缓存
                self.user_relationships[user_id] = relationship_score
                return relationship_score
        except Exception as e:
            logger.debug(f"获取用户关系分失败: {e}")

        # 默认新用户的基础分
        return global_config.affinity_flow.base_relationship_score

    def _calculate_mentioned_score(self, message: "DatabaseMessages", bot_nickname: str) -> float:
        """计算提及分 - 区分强提及和弱提及

        强提及（被@、被回复、私聊）: 使用 strong_mention_interest_score
        弱提及（文本匹配名字/别名）: 使用 weak_mention_interest_score
        """
        from src.chat.utils.utils import is_mentioned_bot_in_message

        # 使用统一的提及检测函数
        is_mentioned, mention_type = is_mentioned_bot_in_message(message)

        if not is_mentioned:
            logger.debug("[提及分计算] 未提及机器人，返回0.0")
            return 0.0

        # mention_type: 0=未提及, 1=弱提及, 2=强提及
        if mention_type >= 2:
            # 强提及：被@、被回复、私聊
            score = global_config.affinity_flow.strong_mention_interest_score
            logger.debug(f"[提及分计算] 检测到强提及（@/回复/私聊），返回分值: {score}")
            return score
        elif mention_type >= 1:
            # 弱提及：文本匹配bot名字或别名
            score = global_config.affinity_flow.weak_mention_interest_score
            logger.debug(f"[提及分计算] 检测到弱提及（文本匹配），返回分值: {score}")
            return score
        else:
            logger.debug("[提及分计算] 未提及机器人，返回0.0")
            return 0.0

    def _apply_threshold_adjustment(self) -> tuple[float, float]:
        """应用阈值调整（包括连续不回复和回复后降低机制）

        Returns:
            tuple[float, float]: (调整后的回复阈值, 调整后的动作阈值)
        """
        # 基础阈值
        base_reply_threshold = self.reply_threshold
        base_action_threshold = global_config.affinity_flow.non_reply_action_interest_threshold

        total_reduction = 0.0

        # 1. 连续不回复的阈值降低
        if self.no_reply_count > 0 and self.no_reply_count < self.max_no_reply_count:
            no_reply_reduction = self.no_reply_count * self.probability_boost_per_no_reply
            total_reduction += no_reply_reduction
            logger.debug(f"[阈值调整] 连续不回复降低: {no_reply_reduction:.3f} (计数: {self.no_reply_count})")

        # 2. 回复后的阈值降低（使bot更容易连续对话）
        if self.enable_post_reply_boost and self.post_reply_boost_remaining > 0:
            # 计算衰减后的降低值
            decay_factor = self.post_reply_boost_decay_rate ** (
                self.post_reply_boost_max_count - self.post_reply_boost_remaining
            )
            post_reply_reduction = self.post_reply_threshold_reduction * decay_factor
            self.post_reply_boost_remaining -= 1
            total_reduction += post_reply_reduction
            logger.debug(
                f"[阈值调整] 回复后降低: {post_reply_reduction:.3f} "
                f"(剩余次数: {self.post_reply_boost_remaining}, 衰减: {decay_factor:.2f})"
            )

        # 应用总降低量
        adjusted_reply_threshold = max(0.0, base_reply_threshold - total_reduction)
        adjusted_action_threshold = max(0.0, base_action_threshold - total_reduction)

        return adjusted_reply_threshold, adjusted_action_threshold

    async def _initialize_semantic_scorer(self):
        """异步初始化语义兴趣度评分器（使用单例 + FastScorer优化）"""
        # 检查是否已初始化
        if self._semantic_initialized:
            logger.debug("[语义评分] 评分器已初始化，跳过")
            return

        if not self.use_semantic_scoring:
            logger.debug("[语义评分] 未启用语义兴趣度评分")
            return

        # 防止并发初始化（使用锁）
        if not hasattr(self, "_init_lock"):
            self._init_lock = asyncio.Lock()

        async with self._init_lock:
            # 双重检查
            if self._semantic_initialized:
                logger.debug("[语义评分] 评分器已在其他任务中初始化，跳过")
                return

            try:
                from src.chat.semantic_interest import get_semantic_scorer
                from src.chat.semantic_interest.runtime_scorer import ModelManager

                # 查找最新的模型文件
                model_dir = Path("data/semantic_interest/models")
                if not model_dir.exists():
                    logger.info(f"[语义评分] 模型目录不存在，已创建: {model_dir}")
                    model_dir.mkdir(parents=True, exist_ok=True)

                # 使用模型管理器（支持人设感知）
                if self.model_manager is None:
                    self.model_manager = ModelManager(model_dir)
                    logger.debug("[语义评分] 模型管理器已创建")

                # 获取人设信息
                persona_info = self._get_current_persona_info()

                # 先检查是否已有可用模型
                from src.chat.semantic_interest.auto_trainer import get_auto_trainer
                auto_trainer = get_auto_trainer()
                existing_model = auto_trainer.get_model_for_persona(persona_info)

                # 加载模型（自动选择合适的版本，使用单例 + FastScorer）
                try:
                    if existing_model and existing_model.exists():
                        # 直接加载已有模型
                        logger.info(f"[语义评分] 使用已有模型: {existing_model.name}")
                        scorer = await get_semantic_scorer(existing_model, use_async=True)
                    else:
                        # 使用 ModelManager 自动选择或训练
                        scorer = await self.model_manager.load_model(
                            version="auto",  # 自动选择或训练
                            persona_info=persona_info
                        )

                    self.semantic_scorer = scorer

                    logger.info("[语义评分] 语义兴趣度评分器初始化成功（FastScorer优化 + 单例）")

                    # 设置初始化标志
                    self._semantic_initialized = True

                    # 启动自动训练任务（每24小时检查一次）- 只在没有模型时或明确需要时启动
                    if not existing_model or not existing_model.exists():
                        await self.model_manager.start_auto_training(
                            persona_info=persona_info,
                            interval_hours=24
                        )
                    else:
                        logger.debug("[语义评分] 已有模型，跳过自动训练启动")

                except FileNotFoundError:
                    logger.warning("[语义评分] 未找到训练模型，将自动训练...")
                    # 触发首次训练
                    trained, model_path = await auto_trainer.auto_train_if_needed(
                        persona_info=persona_info,
                        force=True  # 强制训练
                    )
                    if trained and model_path:
                        # 使用单例获取评分器（默认启用 FastScorer）
                        self.semantic_scorer = await get_semantic_scorer(model_path)
                        logger.info("[语义评分] 首次训练完成，模型已加载（FastScorer优化 + 单例）")
                        # 设置初始化标志
                        self._semantic_initialized = True
                    else:
                        logger.error("[语义评分] 首次训练失败")
                        self.use_semantic_scoring = False

            except ImportError:
                logger.warning("[语义评分] 无法导入语义兴趣度模块，将禁用语义评分")
                self.use_semantic_scoring = False
            except Exception as e:
                logger.error(f"[语义评分] 初始化失败: {e}")
                self.use_semantic_scoring = False

    def _get_current_persona_info(self) -> dict[str, Any]:
        """获取当前人设信息
        
        Returns:
            人设信息字典
        """
        # 默认信息（至少包含名字）
        persona_info = {
            "name": global_config.bot.nickname,
            "interests": [],
            "dislikes": [],
            "personality": "",
        }

        # 优先从已生成的人设文件获取（Individuality 初始化时会生成）
        try:
            persona_file = Path("data/personality/personality_data.json")
            if persona_file.exists():
                data = orjson.loads(persona_file.read_bytes())
                personality_parts = [data.get("personality", ""), data.get("identity", "")]
                persona_info["personality"] = "，".join([p for p in personality_parts if p]).strip("，")
                if persona_info["personality"]:
                    return persona_info
        except Exception as e:
            logger.debug(f"[语义评分] 从文件获取人设信息失败: {e}")

        # 退化为配置中的人设描述
        try:
            personality_parts = []
            personality_core = getattr(global_config.personality, "personality_core", "")
            personality_side = getattr(global_config.personality, "personality_side", "")
            identity = getattr(global_config.personality, "identity", "")

            if personality_core:
                personality_parts.append(personality_core)
            if personality_side:
                personality_parts.append(personality_side)
            if identity:
                personality_parts.append(identity)

            persona_info["personality"] = "，".join(personality_parts) or "默认人设"
        except Exception as e:
            logger.debug(f"[语义评分] 使用配置获取人设信息失败: {e}")
            persona_info["personality"] = "默认人设"

        return persona_info

    async def _calculate_semantic_score(self, content: str) -> float:
        """计算语义兴趣度分数（优化版：FastScorer + 可选批处理 + 超时保护）

        Args:
            content: 消息文本

        Returns:
            语义兴趣度分数 [0.0, 1.0]
        """
        # 检查是否启用
        if not self.use_semantic_scoring:
            return 0.0

        # 检查评分器是否已加载
        if not self.semantic_scorer:
            return 0.0

        # 检查内容是否为空
        if not content or not content.strip():
            return 0.0

        try:
            score = await self.semantic_scorer.score_async(content, timeout=5.0)  # 本地Ollama需要更长超时

            logger.debug(f"[语义评分] 内容: '{content[:50]}...' -> 分数: {score:.3f}")
            return score

        except Exception as e:
            logger.warning(f"[语义评分] 计算失败: {e}")
            return 0.0

    async def reload_semantic_model(self):
        """重新加载语义兴趣度模型（支持热更新和人设检查）"""
        if not self.use_semantic_scoring:
            logger.info("[语义评分] 语义评分未启用，无需重载")
            return

        logger.info("[语义评分] 开始重新加载模型...")

        # 检查人设是否变化
        if hasattr(self, "model_manager") and self.model_manager:
            persona_info = self._get_current_persona_info()
            reloaded = await self.model_manager.check_and_reload_for_persona(persona_info)
            if reloaded:
                self.semantic_scorer = self.model_manager.get_scorer()

                logger.info("[语义评分] 模型重载完成（人设已更新）")
            else:
                logger.info("[语义评分] 人设未变化，无需重载")
        else:
            # 降级：简单重新初始化
            self._semantic_initialized = False
            await self._initialize_semantic_scorer()
            logger.info("[语义评分] 模型重载完成")

    def update_no_reply_count(self, replied: bool):
        """更新连续不回复计数"""
        if replied:
            self.no_reply_count = 0
        else:
            self.no_reply_count = min(self.no_reply_count + 1, self.max_no_reply_count)

    def on_reply_sent(self):
        """当机器人发送回复后调用，激活回复后阈值降低机制"""
        if self.enable_post_reply_boost and not self.post_reply_boost_remaining:
            # 重置回复后降低计数器
            self.post_reply_boost_remaining = self.post_reply_boost_max_count
            logger.debug(
                f"[回复后机制] 激活连续对话模式，阈值将在接下来 {self.post_reply_boost_max_count} 条消息中降低"
            )

        # 应用回复后减少不回复计数的功能
        if self.reply_cooldown_reduction > 0:
            old_count = self.no_reply_count
            self.no_reply_count = max(0, self.no_reply_count - self.reply_cooldown_reduction)
            logger.debug(
                f"[回复后机制] 应用回复冷却减少: 不回复计数 {old_count} → {self.no_reply_count} "
                f"(减少量: {self.reply_cooldown_reduction})"
            )

    def on_message_processed(self, replied: bool):
        """消息处理完成后调用，更新各种计数器

        Args:
            replied: 是否回复了此消息
        """
        # 更新不回复计数
        self.update_no_reply_count(replied)

        # 如果已回复，激活回复后降低机制
        if replied:
            self.on_reply_sent()
        else:
            # 如果没有回复，减少回复后降低剩余次数
            if self.post_reply_boost_remaining > 0:
                self.post_reply_boost_remaining -= 1
                logger.debug(
                    f"[回复后机制] 未回复消息，剩余降低次数: {self.post_reply_boost_remaining}"
                )

afc_interest_calculator = AffinityInterestCalculator()
