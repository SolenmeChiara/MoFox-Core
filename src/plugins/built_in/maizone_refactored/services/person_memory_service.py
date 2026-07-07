"""
空间人物记忆服务

把 bot 在 QQ 空间评论过 / 回复过的人接入 bot 的「记人」系统，并让这些人随互动积累、
在下一次生成评论或回复时被重新调取，从而带上「我认识你」的连续性。

三件事：
1. 注册：互动成功后，把对方（platform="qq", user_id=对方QQ号）通过 person_info 的
   ``get_or_create_person`` 注册进记人系统（已存在则跳过，不触发取名 LLM，成本极低）。
2. 积累：为每个人维护一份轻量的「空间互动记录」滚动列表（时间、互动类型、对方内容摘要、
   bot 回应摘要），存放在插件本地存储（``data/plugin_data/maizone_person_memory.json``），
   不写 person_info 主表——零数据库迁移、零 LLM 成本，语义干净。
3. 调取：把互动记录格式化成一段「你与 ta 的空间往事」文本，交给 content_service 注入到
   缓存断点之后的动态区（候选说说内容之前）。

失败隔离：所有对外方法内部自带 try/except，任何异常只告警，绝不影响评论 / 回复主流程。
"""

import time
from collections.abc import Callable
from typing import Any

from src.common.logger import get_logger
from src.plugin_system.apis.storage_api import get_local_storage

logger = get_logger("MaiZone.PersonMemoryService")

# 本地存储名称（对应 data/plugin_data/maizone_person_memory.json）
_STORAGE_NAME = "maizone_person_memory"

# 代码级默认值兜底（配置缺失时使用）
_DEFAULT_ENABLE = True
_DEFAULT_MAX_RECORDS = 20
# 单条内容摘要的最大字数
_SUMMARY_LIMIT = 80


class PersonMemoryService:
    """空间人物记忆服务，负责注册 / 积累 / 调取三件事。"""

    def __init__(self, get_config: Callable):
        """
        :param get_config: 插件的配置读取函数（与其它 service 一致）。
        """
        self.get_config = get_config
        # 数据结构：{ "<qq号>": [ {time, type, their, mine, name}, ... ] }
        self.storage = get_local_storage(_STORAGE_NAME)

    # ---------------- 配置读取 ----------------

    def _enabled(self) -> bool:
        """记人功能总开关。"""
        try:
            return bool(self.get_config("person_memory.enable_person_memory", _DEFAULT_ENABLE))
        except Exception:
            return _DEFAULT_ENABLE

    def _max_records(self) -> int:
        """每人保留的互动记录条数上限（至少 1）。"""
        try:
            value = int(self.get_config("person_memory.max_interaction_records", _DEFAULT_MAX_RECORDS))
            return max(1, value)
        except Exception:
            return _DEFAULT_MAX_RECORDS

    # ---------------- 工具方法 ----------------

    @staticmethod
    def _summarize(text: Any, limit: int = _SUMMARY_LIMIT) -> str:
        """把任意文本压成单行、限长的摘要。"""
        if not text:
            return ""
        # 折叠所有空白（含换行）为单个空格
        collapsed = " ".join(str(text).split())
        if len(collapsed) <= limit:
            return collapsed
        return collapsed[: limit - 1] + "…"

    # ---------------- 1. 注册 ----------------

    async def register_person(self, qq: str | int | None, nickname: str | None = None) -> None:
        """把对方注册进 person_info 记人系统（已存在则跳过）。

        :param qq: 对方 QQ 号。
        :param nickname: 对方昵称（能拿到就带上，拿不到时回退为 QQ 号）。
        """
        if not self._enabled() or not qq:
            return
        try:
            # 与 kokoro / social_toolkit 等内置插件一致，直接使用 person_info 单例
            from src.person_info.person_info import get_person_info_manager

            name = (str(nickname).strip() if nickname else "") or str(qq)
            # get_or_create_person 内部：不存在则创建、已存在则直接返回，且不触发取名 LLM
            await get_person_info_manager().get_or_create_person(
                platform="qq",
                user_id=qq,
                nickname=name,
                user_cardname="",
            )
        except Exception as e:
            logger.warning(f"注册空间互动对象到记人系统失败(qq={qq}): {e}")

    # ---------------- 2. 积累 ----------------

    def add_interaction(
        self,
        qq: str | int | None,
        nickname: str | None,
        interaction_type: str,
        their_content: str | None,
        bot_reply: str | None,
    ) -> None:
        """追加一条空间互动记录，并按上限滚动裁剪。

        :param qq: 对方 QQ 号（作为存储 key）。
        :param nickname: 对方昵称（用于展示，可空）。
        :param interaction_type: 互动类型描述，如「评论了ta的说说」「回复了ta的评论」。
        :param their_content: 对方内容（对方说说 / 对方评论），会被压成 ≤80 字摘要。
        :param bot_reply: bot 的回应（评论 / 回复），会被压成 ≤80 字摘要。
        """
        if not self._enabled() or not qq:
            return
        try:
            key = str(qq)
            records = self.storage.get(key, []) or []
            if not isinstance(records, list):
                records = []

            record = {
                "time": time.strftime("%Y-%m-%d %H:%M", time.localtime()),
                "type": interaction_type,
                "their": self._summarize(their_content),
                "mine": self._summarize(bot_reply),
            }
            if nickname:
                record["name"] = str(nickname)

            records = [*records, record][-self._max_records() :]
            self.storage.set(key, records)
            logger.debug(f"已记录一条空间互动记忆(qq={qq}, type={interaction_type})")
        except Exception as e:
            logger.warning(f"追加空间互动记忆失败(qq={qq}): {e}")

    # ---------------- 3. 调取 ----------------

    def get_interaction_block(
        self, qq: str | int | None, display_name: str | None = None, limit: int | None = None
    ) -> str:
        """构建「你与 ta 的空间往事」文本块；查无记录时返回空串（首次互动语义自然）。

        :param qq: 对方 QQ 号。
        :param display_name: 展示用名称（优先使用；缺省时回退到记录里的最近昵称或「ta」）。
        :param limit: 本次最多引用的记录条数（缺省用配置上限）。
        :return: 可直接拼进 prompt 的文本块，或空串。
        """
        if not self._enabled() or not qq:
            return ""
        try:
            records = self.storage.get(str(qq), []) or []
            if not isinstance(records, list) or not records:
                return ""

            take = limit if (limit and limit > 0) else self._max_records()
            recent = records[-take:]

            # 展示名：入参优先 → 记录里最近的昵称 → 兜底「ta」
            name = display_name or ""
            if not name:
                for rec in reversed(recent):
                    if isinstance(rec, dict) and rec.get("name"):
                        name = str(rec["name"])
                        break
            if not name:
                name = "ta"

            lines = [f"# 你与 {name} 在QQ空间的过往互动（由远及近）"]
            for rec in recent:
                if not isinstance(rec, dict):
                    continue
                when = rec.get("time", "")
                itype = rec.get("type", "互动过")
                their = rec.get("their", "")
                mine = rec.get("mine", "")
                parts = [f"- [{when}] 你{itype}"]
                if their:
                    parts.append(f"；ta的内容：「{their}」")
                if mine:
                    parts.append(f"；你的回应：「{mine}」")
                lines.append("".join(parts))

            if len(lines) <= 1:
                return ""
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"读取空间互动记忆失败(qq={qq}): {e}")
            return ""
