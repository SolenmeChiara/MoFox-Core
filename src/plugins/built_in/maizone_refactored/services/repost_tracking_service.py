"""
转发锐评追踪服务（风控状态存储）

「转发锐评」功能极度保守，必须有一套永久 / 每日 / 好友级的风控闸门。本服务负责这些
闸门的状态持久化，接口刻意收窄，只服务于 ``SocialLoopService.repost_round`` 的编排：

- ``reposted_feeds``：**永久**记录 bot 转发过的说说 tid，同一条说说永不二次转发。
- ``friend_cooldown``：``{qq: 上次转发该好友说说的时间戳}``，同一好友转发后冷却若干天。
- ``daily_date`` / ``daily_count``：按自然日计的当日转发次数（跨日自动归零）。
- ``last_repost_ts``：上次**成功转发**时间戳（审计 / 日志用）。
- ``last_attempt_ts``：上次**转发尝试**时间戳，用于冷却闸限制昂贵路径（拉候选 + LLM 判定）
  的调用频率——即使当轮无候选或 LLM 判定 SKIP，也照样落尝试戳，绝不每轮重复拉取 / 调 LLM。

存储名 ``maizone_repost_tracking``（对应 ``data/plugin_data/*.json``）；``.set`` 由存储 API
在后台延迟落盘（非阻塞）。所有写入包裹 try/except，异常只告警，绝不影响监控 / 转发主流程。
"""

import time
from datetime import date
from typing import Any

from src.common.logger import get_logger
from src.plugin_system.apis.storage_api import get_local_storage

logger = get_logger("MaiZone.RepostTracking")

# 本地存储名（对应 data/plugin_data/maizone_repost_tracking.json）
_STORAGE_NAME = "maizone_repost_tracking"
# 永久转发记录的软上限，防止无界增长（时间线里不会再出现的老 feed 允许淘汰最旧的）
_MAX_REPOSTED_FEEDS = 1000


class RepostTrackingService:
    """转发锐评的永久 / 每日 / 好友级风控状态存储（窄接口）。"""

    def __init__(self):
        self.storage = get_local_storage(_STORAGE_NAME)
        data = self.storage.get("data", {})
        self.data: dict[str, Any] = data if isinstance(data, dict) else {}
        # 结构兜底
        if not isinstance(self.data.get("reposted_feeds"), list):
            self.data["reposted_feeds"] = []
        if not isinstance(self.data.get("friend_cooldown"), dict):
            self.data["friend_cooldown"] = {}

    # ---------------- 内部工具 ----------------

    def _persist(self) -> None:
        try:
            self.storage.set("data", self.data)
        except Exception as e:
            logger.error(f"[转发] 持久化转发追踪数据失败: {e}")

    @staticmethod
    def _today() -> str:
        return date.today().isoformat()

    # ---------------- 尝试冷却（限制拉候选 + LLM 的频率） ----------------

    def attempt_on_cooldown(self, cooldown_seconds: float) -> bool:
        """距上次转发尝试是否仍在冷却期内（True=还在冷却，应跳过）。"""
        last = self.data.get("last_attempt_ts")
        if not isinstance(last, (int, float)):
            return False
        return (time.time() - last) < cooldown_seconds

    def mark_attempt(self) -> None:
        """落一次「尝试戳」。在拉取候选 / 调 LLM 之前调用，保证冷却期内不重复走昂贵路径。"""
        try:
            self.data["last_attempt_ts"] = time.time()
            self._persist()
        except Exception as e:
            logger.warning(f"[转发] 记录转发尝试戳失败: {e}")

    # ---------------- 每日次数闸 ----------------

    def reposts_today(self) -> int:
        """今日（自然日）已成功转发的次数；跨日自动视为 0。"""
        if self.data.get("daily_date") != self._today():
            return 0
        c = self.data.get("daily_count", 0)
        return c if isinstance(c, int) else 0

    # ---------------- 永不二转闸 ----------------

    def already_reposted(self, feed_id: Any) -> bool:
        """这条说说是否已经被转发过（永久记录）。"""
        if not feed_id:
            return False
        feeds = self.data.get("reposted_feeds") or []
        return isinstance(feeds, list) and str(feed_id) in feeds

    # ---------------- 好友级冷却闸 ----------------

    def friend_on_cooldown(self, qq: Any, cooldown_seconds: float) -> bool:
        """转发过该好友说说后，该好友是否仍在冷却期内（True=还在冷却，应跳过）。"""
        cd = self.data.get("friend_cooldown") or {}
        last = cd.get(str(qq)) if isinstance(cd, dict) else None
        if not isinstance(last, (int, float)):
            return False
        return (time.time() - last) < cooldown_seconds

    # ---------------- 成功转发后落所有风控戳（动作前调用） ----------------

    def record_repost(self, feed_id: Any, qq: Any) -> None:
        """记录一次即将执行的转发：永久标记该 feed + 当日计数 +1 + 好友冷却戳 + 成功戳。

        按仓库风控约定，本方法在真正发起转发**网络请求之前**调用，确保即便请求失败也不会
        被下一轮重复转发。幂等：同一 feed 不会重复入永久表 / 重复计数。
        """
        try:
            fid = str(feed_id) if feed_id else ""
            now = time.time()

            # 1) 永久表（幂等 + 软上限淘汰最旧）
            feeds = self.data.get("reposted_feeds")
            if not isinstance(feeds, list):
                feeds = []
                self.data["reposted_feeds"] = feeds
            if fid and fid not in feeds:
                feeds.append(fid)
                if len(feeds) > _MAX_REPOSTED_FEEDS:
                    del feeds[:-_MAX_REPOSTED_FEEDS]

            # 2) 每日计数（跨日归零后 +1）
            today = self._today()
            if self.data.get("daily_date") != today:
                self.data["daily_date"] = today
                self.data["daily_count"] = 0
            self.data["daily_count"] = int(self.data.get("daily_count", 0) or 0) + 1

            # 3) 好友级冷却戳
            cd = self.data.get("friend_cooldown")
            if not isinstance(cd, dict):
                cd = {}
                self.data["friend_cooldown"] = cd
            if qq:
                cd[str(qq)] = now

            # 4) 成功转发戳（审计）
            self.data["last_repost_ts"] = now

            self._persist()
        except Exception as e:
            logger.warning(f"[转发] 记录转发风控戳失败(fid={feed_id}): {e}")
