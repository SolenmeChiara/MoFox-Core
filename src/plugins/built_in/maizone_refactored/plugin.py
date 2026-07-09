"""
MaiZone（麦麦空间）- 重构版
"""

import asyncio

from src.common.logger import get_logger
from src.plugin_system import BasePlugin, ComponentInfo, register_plugin
from src.plugin_system.base.component_types import PermissionNodeField
from src.plugin_system.base.config_types import ConfigField

# 全局背景任务集合
_background_tasks = set()

from .actions.read_feed_action import ReadFeedAction
from .actions.send_feed_action import SendFeedAction
from .commands.send_feed_command import SendFeedCommand
from .services.comment_tracking_service import CommentTrackingService
from .services.content_service import ContentService
from .services.cookie_service import CookieService
from .services.image_service import ImageService
from .services.manager import register_service
from .services.monitor_service import MonitorService
from .services.own_thread_tracking_service import OwnThreadTrackingService
from .services.person_memory_service import PersonMemoryService
from .services.qzone_service import QZoneService
from .services.reply_tracker_service import ReplyTrackerService
from .services.repost_tracking_service import RepostTrackingService
from .services.scheduler_service import SchedulerService
from .services.social_loop_service import SocialLoopService

logger = get_logger("MaiZone.Plugin")


@register_plugin
class MaiZoneRefactoredPlugin(BasePlugin):
    plugin_name: str = "MaiZoneRefactored"
    plugin_version: str = "3.0.0"
    plugin_author: str = "Kilo Code"
    plugin_description: str = "重构版的MaiZone插件"
    config_file_name: str = "config.toml"
    enable_plugin: bool = True
    dependencies: list[str] = []
    python_dependencies: list[str] = []

    config_schema: dict = {
        "plugin": {"enable": ConfigField(type=bool, default=True, description="是否启用插件")},
        "models": {
            "text_model": ConfigField(type=str, default="maizone", description="生成文本的模型名称"),
        },
        "ai_image": {
            "enable_ai_image": ConfigField(type=bool, default=False, description="是否启用AI生成配图"),
            "provider": ConfigField(type=str, default="siliconflow", description="AI生图服务提供商（siliconflow/novelai）"),
            "image_number": ConfigField(type=int, default=1, description="生成图片数量（1-4张）"),
        },
        "siliconflow": {
            "api_key": ConfigField(type=str, default="", description="硅基流动API密钥"),
        },
        "novelai": {
            "api_key": ConfigField(type=str, default="", description="NovelAI官方API密钥"),
            "character_prompt": ConfigField(type=str, default="", description="Bot角色外貌描述（AI判断需要bot出镜时插入）"),
            "base_negative_prompt": ConfigField(type=str, default="nsfw, nude, explicit, sexual content, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality", description="基础负面提示词（禁止不良内容）"),
            "proxy_host": ConfigField(type=str, default="", description="代理服务器地址（如：127.0.0.1）"),
            "proxy_port": ConfigField(type=int, default=0, description="代理服务器端口（如：7890）"),
        },
        "send": {
            "permission": ConfigField(type=list, default=[], description="发送权限QQ号列表"),
            "permission_type": ConfigField(type=str, default="whitelist", description="权限类型"),
            "enable_reply": ConfigField(type=bool, default=True, description="完成后是否回复"),
        },
        "read": {
            "permission": ConfigField(type=list, default=[], description="阅读权限QQ号列表"),
            "permission_type": ConfigField(type=str, default="blacklist", description="权限类型"),
            "read_number": ConfigField(type=int, default=5, description="一次读取的说说数量"),
            "like_possibility": ConfigField(type=float, default=1.0, description="点赞概率"),
            "comment_possibility": ConfigField(type=float, default=0.3, description="评论概率（仅在关闭LLM选择时作为逐条随机回退使用）"),
        },
        "selection": {
            "enable_llm_selection": ConfigField(
                type=bool, default=True, description="是否启用LLM主动挑选值得评论的动态（关闭则沿用逐条随机评论）"
            ),
            "candidate_limit": ConfigField(type=int, default=8, description="一次交给LLM挑选的候选动态数量上限"),
            "max_comments_per_round": ConfigField(type=int, default=3, description="单轮最多评论的动态数量"),
        },
        "monitor": {
            "enable_auto_monitor": ConfigField(type=bool, default=False, description="是否启用自动监控"),
            "interval_minutes": ConfigField(type=int, default=10, description="监控间隔分钟数"),
            "enable_auto_reply": ConfigField(type=bool, default=False, description="是否启用自动回复自己说说的评论"),
            "enable_own_thread_reply": ConfigField(
                type=bool,
                default=True,
                description="是否启用自己说说「自楼接话」：拉完整评论楼发现别人回复 bot 评论的楼中楼/平铺@并定向接话"
                "（修 msglist 楼中楼盲区，默认开启；首轮基线播种不回复历史）",
            ),
            "own_thread_max_replies_per_round": ConfigField(
                type=int, default=3, description="自楼接话单轮最多接话总次数（跨所有自己说说）"
            ),
            "own_thread_per_feed_limit": ConfigField(
                type=int, default=2, description="同一条自己说说里最多接话次数（防无限对聊刷楼）"
            ),
            "own_thread_tracking_ttl_hours": ConfigField(
                type=int, default=168, description="一条自己说说追踪多久（小时），超时不再轮询其楼中楼回复"
            ),
            "own_thread_force_refresh_minutes": ConfigField(
                type=int,
                default=180,
                description="自楼接话「兜底强刷」阈值（分钟，0=关闭）：即使复合信号失灵，只要 msglist 快照里"
                "有 bot 自己的评论、且距上次拉详情超过此分钟数，就强制拉一次 msgdetail，"
                "保证楼中楼最迟这么久内被发现（最坏成本=bot 评论过的自己说说数 ≤5 每此周期各 1 次）",
            ),
        },
        "schedule": {
            "enable_schedule": ConfigField(type=bool, default=False, description="是否启用定时发送"),
            "random_interval_min_minutes": ConfigField(type=int, default=120, description="随机间隔分钟数下限"),
            "random_interval_max_minutes": ConfigField(type=int, default=135, description="随机间隔分钟数上限"),
            "forbidden_hours_start": ConfigField(type=int, default=2, description="禁止发送的开始小时(24小时制)"),
            "forbidden_hours_end": ConfigField(type=int, default=6, description="禁止发送的结束小时(24小时制)"),
        },
        "cookie": {
            "http_fallback_host": ConfigField(
                type=str, default="127.0.0.1", description="备用Cookie获取服务的主机地址"
            ),
            "http_fallback_port": ConfigField(type=int, default=9999, description="备用Cookie获取服务的端口"),
            "napcat_token": ConfigField(type=str, default="", description="Napcat服务的认证Token（可选）"),
        },
        "cross_context": {
            "user_id": ConfigField(type=str, default="", description="用于获取互通上下文的目标用户QQ号"),
        },
        "person_memory": {
            "enable_person_memory": ConfigField(
                type=bool,
                default=True,
                description="是否把空间互动对象注册进记人系统，并积累/调取空间互动记忆",
            ),
            "max_interaction_records": ConfigField(
                type=int, default=20, description="每个用户保留的空间互动记录条数上限"
            ),
        },
        "social_loop": {
            "enable_unread_gate": ConfigField(
                type=bool,
                default=True,
                description="未读计数事件门：仅当计数有增量时才跑对应重活（计数接口失败则回退全量，不影响监控）",
            ),
            "enable_visitor_callback": ConfigField(type=bool, default=True, description="是否启用访客回访闭环"),
            "visitor_callback_cooldown_hours": ConfigField(
                type=int, default=24, description="同一访客多久内不重复回访（小时）"
            ),
            "visitor_callback_max_per_round": ConfigField(
                type=int, default=2, description="每轮最多回访的新访客数"
            ),
            "visitor_callback_delay_min_seconds": ConfigField(
                type=int, default=5, description="回访前随机延迟下限（秒），模拟真人节奏"
            ),
            "visitor_callback_delay_max_seconds": ConfigField(
                type=int, default=30, description="回访前随机延迟上限（秒），模拟真人节奏"
            ),
            "enable_like_back": ConfigField(type=bool, default=True, description="是否启用回赞闭环"),
            "like_back_cooldown_hours": ConfigField(
                type=int, default=48, description="同一点赞者多久内不重复回赞（小时）"
            ),
            "like_back_max_per_round": ConfigField(type=int, default=3, description="每轮最多回赞的新点赞者数"),
            "like_back_recent_feed_count": ConfigField(
                type=int, default=3, description="检查点赞者时回看自己最近说说的条数"
            ),
        },
        "comment_reply": {
            "enable_comment_reply": ConfigField(
                type=bool,
                default=False,
                description="是否启用好友说说「接话闭环」：追踪 bot 评论过的好友说说，发现有人回复 bot 评论后定向接话（默认关闭，观察后再手动开启）",
            ),
            "tracking_ttl_hours": ConfigField(
                type=int, default=72, description="一条已评论好友说说追踪多久（小时），超时不再轮询回复"
            ),
            "max_replies_per_round": ConfigField(
                type=int, default=3, description="单轮接话检查里最多接话的总次数（跨所有追踪说说）"
            ),
            "per_feed_reply_limit": ConfigField(
                type=int, default=2, description="同一条说说里 bot 最多接话的次数（防无限对聊刷楼）"
            ),
        },
        "repost": {
            "enable_repost": ConfigField(
                type=bool,
                default=False,
                description="是否启用「转发锐评」：偶尔把好友的转发类说说转到自己空间并配一句短评。"
                "极度保守：默认关闭；仅转发本身就是转发的说说（social norm），需观察后再手动开启",
            ),
            "max_reposts_per_day": ConfigField(
                type=int, default=1, description="每天最多转发次数（保守建议 1）"
            ),
            "repost_cooldown_hours": ConfigField(
                type=int,
                default=24,
                description="两次转发尝试之间的冷却小时数：本冷却也限制「拉候选+LLM判定」这条昂贵路径的频率（保守建议 ≥24）",
            ),
            "per_friend_cooldown_days": ConfigField(
                type=int, default=7, description="转发过某好友的说说后，多少天内不再转发该好友（保守建议 7）"
            ),
            "candidate_scan_count": ConfigField(
                type=int, default=20, description="每次通过频率闸后，扫描好友动态时间线的条数（复用监控拉取，非新增常态请求）"
            ),
            "repost_delay_min_seconds": ConfigField(
                type=int, default=5, description="执行转发前随机延迟下限（秒），模拟真人节奏"
            ),
            "repost_delay_max_seconds": ConfigField(
                type=int, default=15, description="执行转发前随机延迟上限（秒），模拟真人节奏"
            ),
        },
    }

    permission_nodes: list[PermissionNodeField] = [
        PermissionNodeField(node_name="send_feed", description="是否可以使用机器人发送QQ空间说说"),
        PermissionNodeField(node_name="read_feed", description="是否可以使用机器人读取QQ空间说说"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def on_plugin_loaded(self):
        """插件加载完成后的回调，初始化服务并启动后台任务"""
        # --- 创建并注册所有服务实例 ---
        # 空间人物记忆服务：注册互动对象进记人系统 + 积累/调取空间互动记忆
        person_memory_service = PersonMemoryService(self.get_config)
        content_service = ContentService(self.get_config, person_memory=person_memory_service)
        image_service = ImageService(self.get_config)
        cookie_service = CookieService(self.get_config)
        reply_tracker_service = ReplyTrackerService()
        # 好友说说接话闭环：追踪 bot 评论过的好友说说，供 QZoneService.check_comment_replies 轮询
        comment_tracking_service = CommentTrackingService()
        # 自己说说「自楼接话」：基线播种 + 去重 + 防刷楼 + 便宜信号闸，供 _process_own_thread_replies
        own_thread_tracking_service = OwnThreadTrackingService()
        # 转发锐评风控追踪：永久转发记录 + 每日计数 + 好友级 / 尝试冷却
        repost_tracking_service = RepostTrackingService()

        qzone_service = QZoneService(
            self.get_config,
            content_service,
            image_service,
            cookie_service,
            reply_tracker_service,
            person_memory=person_memory_service,
            comment_tracking=comment_tracking_service,
            own_thread_tracking=own_thread_tracking_service,
        )
        scheduler_service = SchedulerService(self.get_config, qzone_service)
        # 社交闭环服务：计数门 + 访客回访 + 回赞 + 转发锐评，注入监控服务由其定时循环驱动
        social_loop_service = SocialLoopService(
            self.get_config,
            qzone_service,
            person_memory=person_memory_service,
            repost_tracking=repost_tracking_service,
        )
        monitor_service = MonitorService(
            self.get_config, qzone_service, social_loop_service=social_loop_service
        )

        register_service("qzone", qzone_service)
        register_service("reply_tracker", reply_tracker_service)
        register_service("get_config", self.get_config)

        logger.info("MaiZone重构版插件服务已注册。")

        # --- 启动后台任务 ---
        task1 = asyncio.create_task(scheduler_service.start())
        _background_tasks.add(task1)
        task1.add_done_callback(_background_tasks.discard)

        task2 = asyncio.create_task(monitor_service.start())
        _background_tasks.add(task2)
        task2.add_done_callback(_background_tasks.discard)

        logger.info("MaiZone后台监控和定时任务已启动。")

    def get_plugin_components(self) -> list[tuple[ComponentInfo, type]]:
        return [
            (SendFeedAction.get_action_info(), SendFeedAction),
            (ReadFeedAction.get_action_info(), ReadFeedAction),
            (SendFeedCommand.get_plus_command_info(), SendFeedCommand),
        ]
