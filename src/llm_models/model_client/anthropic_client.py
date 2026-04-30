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
from ..payload_content.message import Message, RoleType
from ..payload_content.resp_format import RespFormat
from ..payload_content.tool_option import ToolCall, ToolOption, ToolOptionBuilder, ToolParamType
from .base_client import APIResponse, BaseClient, UsageRecord, client_registry

logger = get_logger("Anthropic客户端")

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_CACHE_TTL = "5m"  # 5 分钟默认 TTL；如需更长改成 "1h"


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
    """把 message.content 拍平成纯文本（忽略图片）"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            # 图片元组在文本场景被忽略
        return "\n".join(parts)
    return ""


def _build_content_blocks(msg: Message) -> list[dict[str, Any]]:
    """构造 Anthropic 风格的 content blocks（文本 + 图片）"""
    blocks: list[dict[str, Any]] = []
    if isinstance(msg.content, str):
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        return blocks

    if isinstance(msg.content, list):
        for item in msg.content:
            if isinstance(item, str):
                if item:
                    blocks.append({"type": "text", "text": item})
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


def _parse_response(payload: dict[str, Any], model_info: ModelInfo) -> APIResponse:
    """解析 Anthropic /messages 响应"""
    api_resp = APIResponse()
    api_resp.raw_data = payload

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

        body: dict[str, Any] = {
            "model": model_info.model_identifier,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": msg_blocks,
        }
        if system_blocks:
            body["system"] = system_blocks

        if tool_options:
            tools = _convert_tools(tool_options)
            # 工具定义同样静态，在最后一个工具上打 cache_control
            if tools:
                tools[-1] = {**tools[-1], "cache_control": _cache_control_payload()}
            body["tools"] = tools

        # 合并模型 extra_params（如 thinking、metadata 等）
        if extra_params:
            for k, v in extra_params.items():
                if k not in body:
                    body[k] = v

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


# 防止未使用 import 告警（time/ToolOptionBuilder 留作扩展占位）
_ = time
_ = ToolOptionBuilder
