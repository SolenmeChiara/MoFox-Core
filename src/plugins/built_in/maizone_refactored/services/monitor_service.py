"""
好友动态监控服务
"""

import asyncio
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING

from src.common.logger import get_logger

from .qzone_service import QZoneService

if TYPE_CHECKING:
    from .social_loop_service import SocialLoopService

logger = get_logger("MaiZone.MonitorService")


class MonitorService:
    """好友动态监控服务"""

    def __init__(
        self,
        get_config: Callable,
        qzone_service: QZoneService,
        social_loop_service: "SocialLoopService | None" = None,
    ):
        self.get_config = get_config
        self.qzone_service = qzone_service
        # 社交闭环服务：承载「计数门 + 访客回访 + 回赞」的编排。为空时退回原始全量监控。
        self.social_loop_service = social_loop_service
        self.is_running = False
        self.task = None

    async def start(self):
        """启动监控任务"""
        if self.is_running:
            return
        self.is_running = True
        self.task = asyncio.create_task(self._monitor_loop())
        logger.info("好友动态监控任务已启动")

    async def stop(self):
        """停止监控任务"""
        if not self.is_running:
            return
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("好友动态监控任务已停止")

    async def _monitor_loop(self):
        """监控任务主循环"""
        # 插件启动后，延迟一段时间再开始第一次监控
        await asyncio.sleep(60)

        while self.is_running:
            try:
                if not self.get_config("monitor.enable_auto_monitor", False):
                    await asyncio.sleep(60)
                    continue

                interval_minutes = self.get_config("monitor.interval_minutes", 10)

                # 优先走「计数门」编排（计数门内部会在有增量时调用 monitor_feeds/访客回访/回赞，
                # 计数不可用时回退全量）；未注入 social_loop 时退回原始全量监控，保证向后兼容。
                if self.social_loop_service is not None:
                    await self.social_loop_service.run_gated_round()
                else:
                    await self.qzone_service.monitor_feeds()

                logger.info(f"本轮监控完成，将在 {interval_minutes} 分钟后进行下一次检查。")
                await asyncio.sleep(interval_minutes * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控任务循环出错: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(300)
