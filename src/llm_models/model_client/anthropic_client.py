"""
Anthropic 原生客户端 - 直接调用 /v1/messages 端点

特性:
- 通过 httpx 直接走 Anthropic Messages API，不依赖 anthropic SDK
- 自动启用 prompt caching: 在 system 块上显式打 cache_control={"type":"ephemeral"}
  （MoFox 的 user 块每次都是全新拼接的动态内容，把缓存断点放系统块才会真正命中）
- 同时支持工具定义缓存: 工具列表是静态的，会在最后一个工具上打 cache_control
- 控制台日志显示「缓存命中 / 缓存写入 / 新输入」三个 token 数

注意: Anthropic 没有官方 embedding 和 audio transcription 端点，对应方法会抛 NotImplementedError。
"""

import asyncio
import hashlib
import time
from collections.abc import Callable
from typing import Any, ClassVar

import httpx
import orjson
from json_repair import repair_json

from src.common.logger import get_logger
from src.config.api_ada_configs import APIProvider, ModelInfo

from ..exceptions import (
    NetworkConnectionError,
    ReqAbortException,
    RespNotOkException,
    RespParseException,
)
from ..payload_content.message import CACHE_BREAKPOINT_MARKER, Message, RoleType
from ..payload_content.resp_format import RespFormat
from ..payload_content.tool_option import ToolCall, ToolOption, ToolOptionBuilder, ToolParamType
from .base_client import APIResponse, BaseClient, UsageRecord, client_registry

logger = get_logger("Anthropic客户端")

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_CACHE_TTL = "1h"  # 1h TTL：写入费 2x（5m 为 1.25x）但读只要 0.1x，bot 流量稀疏（间隔常超 5 分钟）时更划算；如需改回填 "5m"
MAX_CACHE_BREAKPOINTS = 4  # Anthropic 单次请求最多允许 4 个 cache_control 断点

# —— 缓存保活（keep-alive）——
# 周期性用 max_tokens=1 的迷你请求原样重放"已缓存前缀"，只付 0.1x 读取费即可刷新 TTL，
# 避免流量空窗后下一次真实请求重付 2x 写入费。缓存条目以内容前缀为键，生成参数不影响命中。
# 真实流量停止超过 HORIZON 后放弃保活，让缓存自然过期（防止对着无人聊天永远续命）。
CACHE_KEEPALIVE_ENABLED = True
CACHE_KEEPALIVE_CHECK_INTERVAL = 300  # 保活检查周期（秒）
CACHE_KEEPALIVE_REFRESH_MARGIN = 2700  # 距上次触达（真实请求或ping）多久后预热；须小于 TTL（1h=3600s）
CACHE_KEEPALIVE_HORIZON = 6 * 3600  # 该前缀的真实流量停止多久后放弃保活
CACHE_KEEPALIVE_MAX_ENTRIES = 16  # 同时保活的前缀数量上限（超出按最久未用淘汰）

# extra_params 中思考/努力程度相关的友好配置键（会被翻译成 Anthropic 原生参数，不会原样发给 API）
VALID_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# 这些模型家族已移除采样参数（temperature/top_p/top_k），发送会直接返回 400
_NO_SAMPLING_MODEL_KEYWORDS = ("claude-fable", "claude-mythos", "claude-opus-4-7", "claude-opus-4-8")


def _model_rejects_sampling_params(model_identifier: str) -> bool:
    """Fable 5 / Mythos 5 / Opus 4.7+ 不接受采样参数"""
    return any(keyword in model_identifier for keyword in _NO_SAMPLING_MODEL_KEYWORDS)


_PARAM_TYPE_MAP = {
    ToolParamType.STRING: "string",
    ToolParamType.INTEGER: "integer",
    ToolParamType.FLOAT: "number",
    ToolParamType.BOOLEAN: "boolean",
}


def _split_messages(messages: list[Message]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把 MoFox Message 列表拆成 Anthropic 的 (system_blocks, message_blocks)

    System 角色的消息合并成顶层 system 字段；其他消息按原顺序送进 messages。
    """
    system_blocks: list[dict[str, Any]] = []
    msg_blocks: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == RoleType.System:
            text = _flatten_text(msg.content)
            if text:
                system_blocks.append({"type": "text", "text": text})
            continue

        content_blocks = _build_content_blocks(msg)
        if not content_blocks:
            continue

        if msg.role == RoleType.Tool:
            # MoFox 的 Tool 角色对应 Anthropic 的 user 消息里的 tool_result 块
            if not msg.tool_call_id:
                raise ValueError("Tool 消息缺少 tool_call_id")
            tool_result_text = _flatten_text(msg.content) or ""
            msg_blocks.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": tool_result_text,
                }],
            })
            continue

        role = "user" if msg.role == RoleType.User else "assistant"
        msg_blocks.append({"role": role, "content": content_blocks})

    return system_blocks, msg_blocks


def _flatten_text(content: Any) -> str:
    """把 message.content 拍平成纯文本（忽略图片，并移除缓存断点标记）"""
    if isinstance(content, str):
        return content.replace(CACHE_BREAKPOINT_MARKER, "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.replace(CACHE_BREAKPOINT_MARKER, ""))
            # 图片元组在文本场景被忽略
        return "\n".join(parts)
    return ""


def _split_cached_text_blocks(text: str) -> list[dict[str, Any]]:
    """按缓存断点标记拆分文本为多个 content block。

    标记之前的片段会带上 cache_control，使「静态前缀」可以命中 prompt cache；
    标记之后的动态内容保持不缓存。无标记时退化为单个 text block。
    """
    if CACHE_BREAKPOINT_MARKER not in text:
        return [{"type": "text", "text": text}] if text else []

    blocks: list[dict[str, Any]] = []
    segments = text.split(CACHE_BREAKPOINT_MARKER)
    for i, seg in enumerate(segments):
        is_boundary = i < len(segments) - 1
        if seg.strip():
            block: dict[str, Any] = {"type": "text", "text": seg}
            if is_boundary:
                block["cache_control"] = _cache_control_payload()
            blocks.append(block)
        elif is_boundary and blocks:
            # 标记前是空白片段时，把断点挂到上一个已有的块上
            blocks[-1]["cache_control"] = _cache_control_payload()
    return blocks


def _build_content_blocks(msg: Message) -> list[dict[str, Any]]:
    """构造 Anthropic 风格的 content blocks（文本 + 图片）"""
    blocks: list[dict[str, Any]] = []
    if isinstance(msg.content, str):
        if msg.content:
            blocks.extend(_split_cached_text_blocks(msg.content))
        return blocks

    if isinstance(msg.content, list):
        for item in msg.content:
            if isinstance(item, str):
                if item:
                    blocks.extend(_split_cached_text_blocks(item))
            elif isinstance(item, tuple) and len(item) == 2:
                fmt, b64 = item
                media_type = f"image/{fmt.lower().replace('jpg', 'jpeg')}"
                blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                })
    return blocks


def _convert_tools(tool_options: list[ToolOption]) -> list[dict[str, Any]]:
    """把 MoFox ToolOption 转成 Anthropic tools 格式"""
    tools: list[dict[str, Any]] = []
    for opt in tool_options:
        tool: dict[str, Any] = {
            "name": opt.name,
            "description": opt.description,
        }
        if opt.params:
            properties: dict[str, Any] = {}
            required: list[str] = []
            for p in opt.params:
                schema: dict[str, Any] = {
                    "type": _PARAM_TYPE_MAP.get(p.param_type, "string"),
                    "description": p.description,
                }
                if p.enum_values:
                    schema["enum"] = p.enum_values
                properties[p.name] = schema
                if p.required:
                    required.append(p.name)
            tool["input_schema"] = {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        else:
            tool["input_schema"] = {"type": "object", "properties": {}}
        tools.append(tool)
    return tools


def _extract_thinking_options(
    extra: dict[str, Any], max_tokens: int, model_name: str
) -> tuple[dict[str, Any] | None, str | None, int]:
    """从 extra_params 中取出思考/努力程度相关的友好配置，翻译成 Anthropic 原生参数。

    支持的配置键（均会从 extra 中弹出，不会原样发给 API）:
    - enable_adaptive_thinking: bool  自适应思考（Claude Opus 4.6 / Sonnet 4.6 及更新模型）
    - enable_thinking: bool           传统扩展思考开关（旧模型，使用 budget_tokens）
    - thinking_budget: int            扩展思考的 token 预算（仅 enable_thinking=true 时生效，最小 1024）
    - effort: str                     努力程度 low/medium/high/xhigh/max（写入 output_config.effort）

    enable_adaptive_thinking 与 enable_thinking 在 Anthropic API 层面互斥，
    两者同时开启时优先使用自适应思考并告警。

    Returns:
        (thinking 字段, effort 等级, 调整后的 max_tokens)
    """
    adaptive = bool(extra.pop("enable_adaptive_thinking", False))
    enable = extra.pop("enable_thinking", None)
    budget = extra.pop("thinking_budget", None)
    effort = extra.pop("effort", None)

    thinking: dict[str, Any] | None = None
    if adaptive:
        if enable:
            logger.warning(
                f"[{model_name}] enable_adaptive_thinking 与 enable_thinking 同时开启（二者互斥），"
                "优先使用自适应思考(adaptive)"
            )
        thinking = {"type": "adaptive"}
    elif enable:
        budget_tokens = max(1024, int(budget or 1024))
        if budget_tokens >= max_tokens:
            # Anthropic 要求 budget_tokens < max_tokens，且思考 token 计入 max_tokens，这里自动扩容
            new_max = budget_tokens + max_tokens
            logger.info(
                f"[{model_name}] 思考预算({budget_tokens}) >= max_tokens({max_tokens})，"
                f"已自动将 max_tokens 提升至 {new_max} 以容纳思考输出"
            )
            max_tokens = new_max
        thinking = {"type": "enabled", "budget_tokens": budget_tokens}

    if effort is not None and effort not in VALID_EFFORT_LEVELS:
        logger.warning(f"[{model_name}] 无效的 effort 等级: {effort!r}（可选 {VALID_EFFORT_LEVELS}），已忽略")
        effort = None

    return thinking, effort, max_tokens


def _enforce_breakpoint_limit(body: dict[str, Any], limit: int = MAX_CACHE_BREAKPOINTS) -> None:
    """确保整个请求的 cache_control 断点数不超过 Anthropic 上限（4 个）。

    超限时从最早的消息断点开始移除（保留更深的前缀断点，命中收益更大）。
    """
    fixed = 0
    for section in ("tools", "system"):
        for block in body.get(section) or []:
            if isinstance(block, dict) and "cache_control" in block:
                fixed += 1

    msg_breakpoints: list[dict[str, Any]] = []
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            msg_breakpoints.extend(
                block for block in content if isinstance(block, dict) and "cache_control" in block
            )

    overflow = fixed + len(msg_breakpoints) - limit
    if overflow > 0:
        for block in msg_breakpoints[:overflow]:
            block.pop("cache_control", None)
        logger.debug(f"cache_control 断点超过 {limit} 个，已移除最早的 {overflow} 个消息断点")


def _parse_response(payload: dict[str, Any], model_info: ModelInfo) -> APIResponse:
    """解析 Anthropic /messages 响应"""
    api_resp = APIResponse()
    api_resp.raw_data = payload

    # Fable 5 等模型的安全分类器可能拒绝请求：HTTP 200 + stop_reason="refusal"，content 为空。
    # 这里输出告警便于排查；上层的空回复处理会自动故障转移到列表中的下一个模型。
    if payload.get("stop_reason") == "refusal":
        stop_details = payload.get("stop_details") or {}
        logger.warning(
            f"[{model_info.name}] 请求被模型安全分类器拒绝 (stop_reason=refusal, "
            f"category={stop_details.get('category')}), 将由空回复处理逻辑故障转移"
        )

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in payload.get("content", []) or []:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif btype == "tool_use":
            args = block.get("input") or {}
            if not isinstance(args, dict):
                try:
                    args = orjson.loads(repair_json(args if isinstance(args, str) else orjson.dumps(args).decode()))
                except Exception:
                    args = {}
            tool_calls.append(ToolCall(block.get("id", ""), block.get("name", ""), args))

    if text_parts:
        api_resp.content = "".join(text_parts)
    if thinking_parts:
        api_resp.reasoning_content = "\n".join(thinking_parts)
    if tool_calls:
        api_resp.tool_calls = tool_calls

    usage = payload.get("usage") or {}
    if usage:
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_create = int(usage.get("cache_creation_input_tokens") or 0)
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        prompt_tokens = cache_read + cache_create + input_tokens
        api_resp.usage = UsageRecord(
            model_name=model_info.name,
            provider_name=model_info.api_provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_create,
        )
        _log_cache_stats(model_info.name, cache_read, cache_create, input_tokens, output_tokens)

    return api_resp


def _log_cache_stats(model_name: str, cache_read: int, cache_create: int, input_tokens: int, output_tokens: int) -> None:
    """按用户要求输出「缓存命中…，目前缓存token…」"""
    cached_total = cache_read + cache_create
    if cache_read > 0:
        logger.info(
            f"[{model_name}] 缓存命中 {cache_read} tokens，目前缓存token共 {cached_total} 个 "
            f"（命中 {cache_read} / 本次写入 {cache_create}），新输入 {input_tokens}，输出 {output_tokens}"
        )
    elif cache_create > 0:
        logger.info(
            f"[{model_name}] 缓存命中 0 tokens（首次写入），目前缓存token共 {cache_create} 个，"
            f"新输入 {input_tokens}，输出 {output_tokens}"
        )
    else:
        logger.info(
            f"[{model_name}] 缓存命中 0 tokens（未触发缓存）｜本次输入 {input_tokens}，输出 {output_tokens}"
        )


@client_registry.register_client_class("anthropic")
class AnthropicClient(BaseClient):
    """Anthropic Messages API 客户端"""

    _global_client_cache: ClassVar[dict[tuple[int, int | None], httpx.AsyncClient]] = {}

    SUPPORTED_IMAGE_FORMATS: ClassVar[list[str]] = ["jpeg", "jpg", "png", "webp", "gif"]

    def __init__(self, api_provider: APIProvider):
        super().__init__(api_provider)
        self._config_hash = hash((api_provider.base_url, api_provider.get_api_key(), api_provider.timeout))

    @staticmethod
    def _get_loop_id() -> int | None:
        try:
            return id(asyncio.get_running_loop())
        except RuntimeError:
            return None

    def _get_http_client(self) -> httpx.AsyncClient:
        loop_id = self._get_loop_id()
        cache_key = (self._config_hash, loop_id)

        # 清理同一配置但不同事件循环的旧实例（与 OpenAI 客户端策略一致，由 GC 收尾）
        for k in [k for k in self._global_client_cache if k[0] == self._config_hash and k[1] != loop_id]:
            logger.debug(f"清理过期的 Anthropic httpx 客户端缓存 (loop_id={k[1]})")
            self._global_client_cache.pop(k, None)

        if cache_key in self._global_client_cache:
            return self._global_client_cache[cache_key]

        limits = httpx.Limits(max_keepalive_connections=50, max_connections=100, keepalive_expiry=30.0)
        client = httpx.AsyncClient(
            base_url=self.api_provider.base_url.rstrip("/"),
            timeout=httpx.Timeout(self.api_provider.timeout),
            limits=limits,
            headers={
                "x-api-key": self.api_provider.get_api_key(),
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )
        self._global_client_cache[cache_key] = client
        logger.debug(f"创建新的 Anthropic httpx 客户端 (base_url={self.api_provider.base_url}, loop_id={loop_id})")
        return client

    async def get_response(
        self,
        model_info: ModelInfo,
        message_list: list[Message],
        tool_options: list[ToolOption] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: RespFormat | None = None,
        stream_response_handler: Callable[..., Any] | None = None,
        async_response_parser: Callable[..., Any] | None = None,
        interrupt_flag: asyncio.Event | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        system_blocks, msg_blocks = _split_messages(message_list)

        # 关键：在 system 块的最后一个上打 cache_control，让静态系统提示词被缓存
        if system_blocks:
            system_blocks[-1] = {**system_blocks[-1], "cache_control": _cache_control_payload()}

        # 翻译 extra_params 中的思考/努力程度友好配置（enable_thinking / enable_adaptive_thinking / effort 等）
        extra = dict(extra_params) if extra_params else {}
        thinking, effort, max_tokens = _extract_thinking_options(extra, max_tokens, model_info.name)

        body: dict[str, Any] = {
            "model": model_info.model_identifier,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": msg_blocks,
        }
        if system_blocks:
            body["system"] = system_blocks
        if thinking:
            body["thinking"] = thinking
        if effort:
            body["output_config"] = {"effort": effort}

        if tool_options:
            tools = _convert_tools(tool_options)
            # 工具定义同样静态，在最后一个工具上打 cache_control
            if tools:
                tools[-1] = {**tools[-1], "cache_control": _cache_control_payload()}
            body["tools"] = tools

        # 合并剩余的模型 extra_params（如 metadata、原生 thinking 字典等）
        for k, v in extra.items():
            if k not in body:
                body[k] = v

        # 开启思考时 Anthropic 不允许自定义 temperature（必须省略），否则会返回 400
        body_thinking = body.get("thinking")
        if isinstance(body_thinking, dict) and body_thinking.get("type") in ("adaptive", "enabled"):
            if body.pop("temperature", None) is not None:
                logger.debug(f"[{model_info.name}] 已开启思考，自动移除 temperature 参数（二者不兼容）")

        # Fable 5 / Mythos 5 / Opus 4.7+ 已移除采样参数，发送会返回 400，自动剥离
        if _model_rejects_sampling_params(model_info.model_identifier):
            removed = [k for k in ("temperature", "top_p", "top_k") if body.pop(k, None) is not None]
            if removed:
                logger.debug(f"[{model_info.name}] 该模型不接受采样参数，已自动移除: {removed}")

        # 控制 cache_control 断点总数不超过 API 上限
        _enforce_breakpoint_limit(body)

        # 注册缓存前缀用于 TTL 保活（仅当 body 含 cache_control 断点时生效）
        if CACHE_KEEPALIVE_ENABLED:
            _cache_keepalive.register(self, body)

        client = self._get_http_client()

        try:
            req_task = asyncio.create_task(client.post("/messages", content=orjson.dumps(body)))
            while not req_task.done():
                if interrupt_flag and interrupt_flag.is_set():
                    req_task.cancel()
                    raise ReqAbortException("请求被外部信号中断")
                await asyncio.sleep(0.1)
            response: httpx.Response = req_task.result()
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as e:
            raise NetworkConnectionError() from e
        except httpx.HTTPError as e:
            raise NetworkConnectionError() from e

        if response.status_code != 200:
            try:
                err_payload = response.json()
                err_msg = err_payload.get("error", {}).get("message") or response.text
            except Exception:
                err_msg = response.text
            raise RespNotOkException(response.status_code, err_msg)

        try:
            payload = orjson.loads(response.content)
        except orjson.JSONDecodeError as e:
            raise RespParseException(response.text, "响应解析失败，无法解析 JSON") from e

        return _parse_response(payload, model_info)

    async def get_embedding(
        self,
        model_info: ModelInfo,
        embedding_input: str | list[str],
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        raise NotImplementedError("Anthropic 没有官方 embedding 接口，请使用其他 provider（如 Ollama bge-m3 或 OpenAI embedding）")

    async def get_audio_transcriptions(
        self,
        model_info: ModelInfo,
        audio_base64: str,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        raise NotImplementedError("Anthropic 没有官方音频转录接口，请使用 SiliconFlow SenseVoice 或其他 provider")

    def get_support_image_formats(self) -> list[str]:
        return list(self.SUPPORTED_IMAGE_FORMATS)


def _cache_control_payload() -> dict[str, Any]:
    """构造 cache_control 字段；TTL 默认 5m，可改成 1h"""
    if DEFAULT_CACHE_TTL == "5m":
        return {"type": "ephemeral"}
    return {"type": "ephemeral", "ttl": DEFAULT_CACHE_TTL}


def _extract_prefix_body(body: dict[str, Any]) -> dict[str, Any] | None:
    """
    从最终请求体中截取"到最后一个 cache_control 断点为止"的最小请求体。

    Anthropic 的缓存条目按内容前缀（tools → system → messages 渲染序）匹配，
    生成参数（max_tokens/thinking 等）不参与匹配——所以用这个截取体发一个
    max_tokens=1 的迷你请求，就能以 0.1x 读取费刷新前缀上所有断点的 TTL。
    无任何断点时返回 None。
    """
    tools = body.get("tools")
    system = body.get("system")
    messages = body.get("messages", [])

    last_mi, last_bi = -1, -1
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, list):
            for bi, blk in enumerate(content):
                if isinstance(blk, dict) and "cache_control" in blk:
                    last_mi, last_bi = mi, bi

    has_sys_bp = isinstance(system, list) and any(
        isinstance(b, dict) and "cache_control" in b for b in system
    )
    has_tool_bp = isinstance(tools, list) and any(
        isinstance(t, dict) and "cache_control" in t for t in tools
    )
    if last_mi < 0 and not has_sys_bp and not has_tool_bp:
        return None

    ping: dict[str, Any] = {"model": body["model"]}
    if tools:
        ping["tools"] = tools
    if system:
        ping["system"] = system
    if last_mi >= 0:
        msgs = [dict(m) for m in messages[: last_mi + 1]]
        last_msg = dict(msgs[-1])
        content = last_msg.get("content")
        if isinstance(content, list):
            last_msg["content"] = content[: last_bi + 1]
        msgs[-1] = last_msg
        ping["messages"] = msgs
    else:
        # 断点只在 system/tools 上：messages 不能为空，补一个极小的动态尾巴
        ping["messages"] = [{"role": "user", "content": "."}]
    return ping


class _CacheKeepAlive:
    """
    缓存前缀保活器。

    每次真实请求发出前把它的缓存前缀登记进来（真实请求本身就刷新了 TTL，
    所以登记同时重置计时）；后台任务周期检查，距上次触达超过 REFRESH_MARGIN
    的条目会收到一个 max_tokens=1 的迷你请求续命。前缀的真实流量停止超过
    HORIZON 后条目被移除，缓存自然过期。
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self._task: asyncio.Task | None = None

    def register(self, client: "AnthropicClient", body: dict[str, Any]) -> None:
        ping = _extract_prefix_body(body)
        if ping is None:
            return
        key = hashlib.md5(orjson.dumps(ping)).hexdigest()
        now = time.time()
        entry = self._entries.get(key)
        if entry is not None:
            # 真实请求命中同一前缀 = TTL 已被免费刷新
            entry["last_seen"] = now
            entry["last_touch"] = now
        else:
            if len(self._entries) >= CACHE_KEEPALIVE_MAX_ENTRIES:
                oldest = min(self._entries, key=lambda k: self._entries[k]["last_seen"])
                self._entries.pop(oldest, None)
            self._entries[key] = {
                "ping": ping,
                "client": client,
                "model": body.get("model", "?"),
                "last_seen": now,
                "last_touch": now,
                "ping_max_tokens": 1,
            }
        self._ensure_task()

    def _ensure_task(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._task = loop.create_task(self._loop(), name="anthropic-cache-keepalive")

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(CACHE_KEEPALIVE_CHECK_INTERVAL)
            try:
                now = time.time()
                for key in list(self._entries.keys()):
                    entry = self._entries.get(key)
                    if entry is None:
                        continue
                    if now - entry["last_seen"] > CACHE_KEEPALIVE_HORIZON:
                        logger.debug(f"缓存保活: [{entry['model']}] 前缀 {key[:8]} 超过保活时限，停止续命")
                        self._entries.pop(key, None)
                        continue
                    if now - entry["last_touch"] < CACHE_KEEPALIVE_REFRESH_MARGIN:
                        continue
                    await self._ping(key, entry)
            except Exception as e:  # 保活失败绝不能影响主流程
                logger.warning(f"缓存保活循环异常: {e}")

    async def _ping(self, key: str, entry: dict[str, Any]) -> None:
        client: AnthropicClient = entry["client"]
        body = dict(entry["ping"])
        body["max_tokens"] = entry["ping_max_tokens"]
        try:
            http = client._get_http_client()
            response = await http.post("/messages", content=orjson.dumps(body))
        except httpx.HTTPError as e:
            logger.warning(f"缓存保活: [{entry['model']}] 前缀 {key[:8]} ping 网络失败: {e}")
            return

        if response.status_code != 200:
            try:
                err_msg = response.json().get("error", {}).get("message", "") or response.text
            except Exception:
                err_msg = response.text
            # 个别模型可能拒绝 max_tokens=1（如思考强制开启的场景），放宽后下轮重试
            if response.status_code == 400 and "max_tokens" in err_msg and entry["ping_max_tokens"] == 1:
                entry["ping_max_tokens"] = 64
                logger.info(f"缓存保活: [{entry['model']}] 不接受 max_tokens=1，放宽为 64 后重试")
            else:
                logger.warning(f"缓存保活: [{entry['model']}] ping 失败 {response.status_code}: {err_msg[:120]}")
            return

        entry["last_touch"] = time.time()
        try:
            usage = orjson.loads(response.content).get("usage", {})
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_create = usage.get("cache_creation_input_tokens", 0)
        except Exception:
            cache_read = cache_create = 0
        if cache_create and not cache_read:
            # ping 到达时缓存已过期，本次相当于替下一个真实请求预付了重写费
            logger.info(f"缓存保活: [{entry['model']}] 前缀 {key[:8]} 已过期，重写 {cache_create} tokens")
        else:
            logger.info(f"缓存保活: [{entry['model']}] 前缀 {key[:8]} 续命成功，读取 {cache_read} tokens（0.1x费率）")


_cache_keepalive = _CacheKeepAlive()


# 防止未使用 import 告警（ToolOptionBuilder 留作扩展占位）
_ = ToolOptionBuilder
