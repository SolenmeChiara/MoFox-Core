"""
自己说说「自楼接话」追踪服务

背景：``_reply_to_own_feed_comments`` 走 msglist_v6，对 ``commentlist`` 的楼中楼（``list_3``）返回
不完整/不可靠，导致「别人回复 bot 评论的楼中楼」进不了待回复候选，bot 从不接话。新增的
``QZoneService._process_own_thread_replies`` 改用 ``msgdetail_v6`` 拉完整评论楼来发现这类回复，
本服务负责其状态存储：基线播种、去重、防刷楼计数，以及「便宜信号闸」用的上次评论数记录。

存储名 ``maizone_own_thread_tracking``（对应 ``data/plugin_data/*.json``），结构：

    {
      "<feed_id>": {
        "last_cmtnum": int,           # 上次记录的「复合信号值」（cmtnum+Σreply_num，对楼中楼敏感；
                                      #   变化才拉 msgdetail。沿用旧字段名、不迁移旧数据）
        "last_detail_ts": float,      # 上次为本 feed 拉 msgdetail 的时间戳（兜底强刷判断依据）
        "baseline_seeded": bool,      # 是否已完成基线播种（首轮把现存候选全部标记已见、不回复）
        "seen_reply_keys": [str],     # 已处理过的回复去重 key（"<uin>_<tid>"）
        "replied_uins": [str],        # 本 feed 内 bot 已接过话的人（同一个人只接一次，防对喷循环）
        "reply_count": int,           # 本 feed 内 bot 的接话次数（防无限对聊刷楼）
        "updated_at": float           # 最后更新时间戳（TTL 清理依据）
      }
    }

失败隔离：所有写入包裹 try/except，异常只告警，绝不影响监控主流程。
"""

import time
from typing import Any

from src.common.logger import get_logger
from src.plugin_system.apis.storage_api import get_local_storage

logger = get_logger("MaiZone.OwnThreadTracking")

# 本地存储名（对应 data/plugin_data/maizone_own_thread_tracking.json）
_STORAGE_NAME = "maizone_own_thread_tracking"
# 单个 feed 的 seen key 上限，防止长期活跃说说无界增长
_MAX_SEEN_KEYS = 300


class OwnThreadTrackingService:
    """追踪 bot 自己说说的楼中楼接话状态（基线播种 + 去重 + 防刷楼 + 便宜信号闸）。"""

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
            logger.error(f"持久化自楼接话数据失败: {e}")

    def _entry(self, feed_id: Any) -> dict | None:
        """取出某 feed 的追踪条目（不存在或脏数据返回 None）。"""
        entry = self.tracked.get(str(feed_id))
        return entry if isinstance(entry, dict) else None

    def _ensure(self, feed_id: Any) -> dict:
        """取出或新建某 feed 的追踪条目（不落盘，由调用方决定何时 _persist）。"""
        fid = str(feed_id)
        entry = self._entry(fid)
        if entry is None:
            entry = {
                "last_cmtnum": -1,
                "last_detail_ts": 0.0,
                "baseline_seeded": False,
                "seen_reply_keys": [],
                "replied_uins": [],
                "reply_count": 0,
                "updated_at": time.time(),
            }
            self.tracked[fid] = entry
        return entry

    # ---------------- 便宜信号闸 / 基线 ----------------

    def is_tracked(self, feed_id: Any) -> bool:
        """该 feed 是否已在追踪表内（用于判断是否首次遇到 → 需基线播种）。"""
        return self._entry(feed_id) is not None

    def needs_baseline(self, feed_id: Any) -> bool:
        """该 feed 是否还没完成基线播种（首次遇到，或上次播种未成功）。"""
        entry = self._entry(feed_id)
        return entry is None or not entry.get("baseline_seeded", False)

    def get_last_cmtnum(self, feed_id: Any) -> int | None:
        """上次记录的「复合信号值」（字段名沿用 last_cmtnum，语义已改为信号值）；无记录返回 None。"""
        entry = self._entry(feed_id)
        if entry is None:
            return None
        v = entry.get("last_cmtnum", -1)
        return v if isinstance(v, int) else None

    def seed_baseline(self, feed_id: Any, signal: int, seen_keys: list[str]) -> None:
        """基线播种：把现存候选的 key 全部记为已见（不回复），并记录复合信号值。

        ``signal`` 为对楼中楼敏感的复合信号（cmtnum+Σreply_num），存入沿用的 last_cmtnum 字段。
        """
        try:
            entry = self._ensure(feed_id)
            keys = entry.setdefault("seen_reply_keys", [])
            if not isinstance(keys, list):
                keys = []
                entry["seen_reply_keys"] = keys
            for k in seen_keys:
                if k not in keys:
                    keys.append(k)
            if len(keys) > _MAX_SEEN_KEYS:
                del keys[:-_MAX_SEEN_KEYS]
            entry["baseline_seeded"] = True
            entry["last_cmtnum"] = int(signal) if isinstance(signal, int) else -1
            entry["updated_at"] = time.time()
            self._persist()
        except Exception as e:
            logger.warning(f"[自楼接话] 基线播种失败(fid={feed_id}): {e}")

    def update_cmtnum(self, feed_id: Any, signal: int) -> None:
        """记录本轮观察到的复合信号值，作为下次便宜信号闸的比较基准（字段名沿用 last_cmtnum）。"""
        entry = self._entry(feed_id)
        if entry is None:
            return
        entry["last_cmtnum"] = int(signal) if isinstance(signal, int) else -1
        entry["updated_at"] = time.time()
        self._persist()

    # ---------------- 兜底强刷（防复合信号也失灵）----------------

    def get_last_detail_ts(self, feed_id: Any) -> float | None:
        """上次为本 feed 拉 msgdetail 的时间戳；无记录返回 None（兜底强刷判断用）。"""
        entry = self._entry(feed_id)
        if entry is None:
            return None
        v = entry.get("last_detail_ts", 0.0)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    def mark_detail_fetched(self, feed_id: Any) -> None:
        """记录本轮实际为本 feed 拉过一次 msgdetail（用于兜底强刷的节流，避免每轮硬刷）。"""
        try:
            entry = self._ensure(feed_id)
            entry["last_detail_ts"] = time.time()
            entry["updated_at"] = time.time()
            self._persist()
        except Exception as e:
            logger.warning(f"[自楼接话] 记录拉详情时间失败(fid={feed_id}): {e}")

    # ---------------- 去重 / 防刷楼 ----------------

    def has_seen(self, feed_id: Any, key: str) -> bool:
        """某条回复是否已处理过。"""
        entry = self._entry(feed_id)
        if entry is None:
            return False
        keys = entry.get("seen_reply_keys") or []
        return isinstance(keys, list) and key in keys

    def mark_reply_seen(self, feed_id: Any, key: str) -> None:
        """标记某条回复已处理过（无论最终是否接话，都要标记，避免下轮重复评估）。"""
        try:
            entry = self._ensure(feed_id)
            keys = entry.setdefault("seen_reply_keys", [])
            if not isinstance(keys, list):
                keys = []
                entry["seen_reply_keys"] = keys
            if key not in keys:
                keys.append(key)
                if len(keys) > _MAX_SEEN_KEYS:
                    del keys[:-_MAX_SEEN_KEYS]
                entry["updated_at"] = time.time()
                self._persist()
        except Exception as e:
            logger.warning(f"[自楼接话] 标记已见失败(fid={feed_id}): {e}")

    def has_replied_to(self, feed_id: Any, replier_uin: Any) -> bool:
        """本 feed 内 bot 是否已经接过这个人的话（同一个人只接一次，防对喷循环）。"""
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

    def record_reply(self, feed_id: Any, replier_uin: Any) -> None:
        """记录 bot 在本 feed 成功接了一次话（用于防刷楼上限 + 同一个人只接一次）。"""
        try:
            entry = self._ensure(feed_id)
            entry["reply_count"] = self.get_reply_count(feed_id) + 1
            uins = entry.setdefault("replied_uins", [])
            if not isinstance(uins, list):
                uins = []
                entry["replied_uins"] = uins
            if str(replier_uin) not in uins:
                uins.append(str(replier_uin))
            entry["updated_at"] = time.time()
            self._persist()
        except Exception as e:
            logger.warning(f"[自楼接话] 记录接话失败(fid={feed_id}): {e}")

    # ---------------- 清理 ----------------

    def cleanup(self, ttl_seconds: float) -> None:
        """清理超过 TTL 的追踪条目（按 updated_at）。"""
        now = time.time()
        expired = []
        for fid, entry in self.tracked.items():
            if not isinstance(entry, dict):
                expired.append(fid)
                continue
            ts = entry.get("updated_at", 0)
            if not isinstance(ts, (int, float)) or now - ts > ttl_seconds:
                expired.append(fid)
        for fid in expired:
            del self.tracked[fid]
        if expired:
            self._persist()
            logger.info(f"[自楼接话] 清理了 {len(expired)} 条过期的追踪记录")
