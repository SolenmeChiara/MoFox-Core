"""
好友说说评论追踪服务（接话闭环）

bot 在好友说说下评论后，目前是「评论即失联」。本服务负责追踪 bot 评论过的好友说说，
配合 ``QZoneService.check_comment_replies`` 轮询发现「回复了 bot 评论」的新消息并定向接话。

存储名 ``maizone_comment_tracking``（对应 ``data/plugin_data/*.json``），结构：

    {
      "<feed_id>": {
        "host_qq": str,               # 楼主 QQ
        "host_name": str,             # 楼主昵称
        "bot_comment_tid": str|None,  # bot 评论的楼层序号（可选，评论时拿到就存，否则检测时回填）
        "bot_comment_content": str,   # bot 当时评了什么
        "commented_at": float,        # bot 评论的时间戳（TTL 依据）
        "seen_reply_keys": [str],     # 已处理过的回复去重 key（"<uin>_<tid>"）
        "replied_uins": [str],        # 本 feed 内 bot 已接过话的人（同一个人只接一次）
        "reply_count": int            # 本 feed 内 bot 的接话次数（防无限对聊刷楼）
      }
    }

失败隔离：所有方法内部对存储写入包裹 try/except，异常只告警，绝不影响评论 / 监控主流程。
"""

import time
from typing import Any

from src.common.logger import get_logger
from src.plugin_system.apis.storage_api import get_local_storage

logger = get_logger("MaiZone.CommentTracking")

# 本地存储名（对应 data/plugin_data/maizone_comment_tracking.json）
_STORAGE_NAME = "maizone_comment_tracking"
# 单个 feed 的 seen key 上限，防止长期活跃说说无界增长
_MAX_SEEN_KEYS = 200


class CommentTrackingService:
    """追踪 bot 评论过的好友说说，为接话闭环提供状态存储。"""

    def __init__(self):
        self.storage = get_local_storage(_STORAGE_NAME)
        data = self.storage.get("data", {})
        # 内存态：{feed_id: entry}
        self.tracked: dict[str, dict] = data if isinstance(data, dict) else {}

    # ---------------- 内部工具 ----------------

    def _persist(self) -> None:
        """把内存态写回存储（由存储 API 后台延迟落盘）。"""
        try:
            self.storage.set("data", self.tracked)
        except Exception as e:
            logger.error(f"持久化评论追踪数据失败: {e}")

    def _entry(self, feed_id: Any) -> dict | None:
        """取出某 feed 的追踪条目（不存在或脏数据返回 None）。"""
        entry = self.tracked.get(str(feed_id))
        return entry if isinstance(entry, dict) else None

    # ---------------- 写入 ----------------

    def add_commented_feed(
        self,
        feed_id: Any,
        host_qq: Any,
        host_name: Any,
        bot_comment_content: Any,
        bot_comment_tid: Any = None,
    ) -> None:
        """记录一条 bot 刚评论过的好友说说（幂等：已在追踪则只补齐可选 tid，不动 seen/replied 状态）。"""
        if not feed_id:
            return
        try:
            fid = str(feed_id)
            entry = self._entry(fid)
            if entry is None:
                self.tracked[fid] = {
                    "host_qq": str(host_qq or ""),
                    "host_name": str(host_name or ""),
                    "bot_comment_tid": str(bot_comment_tid) if bot_comment_tid else None,
                    "bot_comment_content": str(bot_comment_content or ""),
                    "commented_at": time.time(),
                    "seen_reply_keys": [],
                    "replied_uins": [],
                    "reply_count": 0,
                }
            elif bot_comment_tid and not entry.get("bot_comment_tid"):
                # 已在追踪：仅补齐当时没拿到的 bot 评论 tid
                entry["bot_comment_tid"] = str(bot_comment_tid)
            self._persist()
        except Exception as e:
            logger.warning(f"[接话] 记录已评论说说失败(fid={feed_id}): {e}")

    def update_bot_comment_tid(self, feed_id: Any, bot_comment_tid: Any) -> None:
        """检测阶段从详情里定位到 bot 评论后，回填当初没拿到的楼层序号。"""
        if not bot_comment_tid:
            return
        entry = self._entry(feed_id)
        if entry is not None and not entry.get("bot_comment_tid"):
            entry["bot_comment_tid"] = str(bot_comment_tid)
            self._persist()

    def mark_reply_seen(self, feed_id: Any, key: str) -> None:
        """标记某条回复已处理过（无论是否接话，都要标记，避免重复评估）。"""
        entry = self._entry(feed_id)
        if entry is None:
            return
        keys = entry.setdefault("seen_reply_keys", [])
        if not isinstance(keys, list):
            keys = []
            entry["seen_reply_keys"] = keys
        if key not in keys:
            keys.append(key)
            # 防无界增长：只保留最近 _MAX_SEEN_KEYS 条
            if len(keys) > _MAX_SEEN_KEYS:
                del keys[:-_MAX_SEEN_KEYS]
            self._persist()

    def record_reply(self, feed_id: Any, replier_uin: Any) -> None:
        """记录 bot 在本 feed 成功接了一次话（用于防刷楼上限 + 同一个人只接一次）。"""
        entry = self._entry(feed_id)
        if entry is None:
            return
        entry["reply_count"] = self.get_reply_count(feed_id) + 1
        uins = entry.setdefault("replied_uins", [])
        if not isinstance(uins, list):
            uins = []
            entry["replied_uins"] = uins
        if str(replier_uin) not in uins:
            uins.append(str(replier_uin))
        self._persist()

    def remove_feed(self, feed_id: Any) -> None:
        """从追踪表移除某 feed（如 bot 评论被删）。"""
        fid = str(feed_id)
        if fid in self.tracked:
            del self.tracked[fid]
            self._persist()

    # ---------------- 读取 ----------------

    def has_seen(self, feed_id: Any, key: str) -> bool:
        """某条回复是否已处理过。"""
        entry = self._entry(feed_id)
        if entry is None:
            return False
        keys = entry.get("seen_reply_keys") or []
        return isinstance(keys, list) and key in keys

    def has_replied_to(self, feed_id: Any, replier_uin: Any) -> bool:
        """本 feed 内 bot 是否已经接过这个人的话（同一个人只接一次）。"""
        entry = self._entry(feed_id)
        if entry is None:
            return False
        uins = entry.get("replied_uins") or []
        return isinstance(uins, list) and str(replier_uin) in uins

    def get_reply_count(self, feed_id: Any) -> int:
        """本 feed 内 bot 已接话次数。"""
        entry = self._entry(feed_id)
        if entry is None:
            return 0
        c = entry.get("reply_count", 0)
        return c if isinstance(c, int) else 0

    def get_active_feeds(self, ttl_seconds: float) -> list[tuple[str, dict]]:
        """返回 TTL 内仍需轮询的 (feed_id, entry) 列表。"""
        now = time.time()
        result: list[tuple[str, dict]] = []
        for fid, entry in self.tracked.items():
            if not isinstance(entry, dict):
                continue
            ts = entry.get("commented_at", 0)
            if not isinstance(ts, (int, float)):
                continue
            if now - ts <= ttl_seconds:
                result.append((fid, entry))
        return result

    # ---------------- 清理 ----------------

    def cleanup(self, ttl_seconds: float) -> None:
        """清理超过 TTL 的追踪条目。"""
        now = time.time()
        expired = []
        for fid, entry in self.tracked.items():
            if not isinstance(entry, dict):
                expired.append(fid)
                continue
            ts = entry.get("commented_at", 0)
            if not isinstance(ts, (int, float)) or now - ts > ttl_seconds:
                expired.append(fid)
        for fid in expired:
            del self.tracked[fid]
        if expired:
            self._persist()
            logger.info(f"[接话] 清理了 {len(expired)} 条过期的评论追踪记录")
