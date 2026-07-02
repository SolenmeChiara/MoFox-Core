import asyncio
import base64
import io
import math
from collections.abc import Callable, Coroutine
from typing import Any

import aiohttp
import orjson

from src.common.logger import get_logger
from src.config.api_ada_configs import APIProvider, ModelInfo

from ..exceptions import (
    NetworkConnectionError,
    ReqAbortException,
    RespNotOkException,
    RespParseException,
)
from ..payload_content.message import Message, RoleType
from ..payload_content.resp_format import RespFormat, RespFormatType
from ..payload_content.tool_option import ToolCall, ToolOption, ToolParam
from .base_client import APIResponse, BaseClient, UsageRecord, client_registry

logger = get_logger("AioHTTP-Gemini客户端")


# gemini_thinking参数(默认范围) - 旧版 thinking_budget
# 不同模型的思考预算范围配置
THINKING_BUDGET_LIMITS = {
    "gemini-2.5-flash": {"min": 1, "max": 24576, "can_disable": True},
    "gemini-2.5-flash-lite": {"min": 512, "max": 24576, "can_disable": True},
    "gemini-2.5-pro": {"min": 128, "max": 32768, "can_disable": False},
}
# 思维预算特殊值
THINKING_BUDGET_AUTO = -1  # 自动调整思考预算,由模型决定
THINKING_BUDGET_DISABLED = 0  # 禁用思考预算(如果模型允许禁用)

# 新版 thinking_level 参数
# 支持的思考等级（"minimal" 为 Gemini 3.5 系列新增；3.5 默认等级为 medium）
THINKING_LEVEL_MINIMAL = "minimal"
THINKING_LEVEL_LOW = "low"
THINKING_LEVEL_MEDIUM = "medium"
THINKING_LEVEL_HIGH = "high"
VALID_THINKING_LEVELS = [THINKING_LEVEL_MINIMAL, THINKING_LEVEL_LOW, THINKING_LEVEL_MEDIUM, THINKING_LEVEL_HIGH]

# Gemini 3.5 起官方弃用采样参数（temperature/topP/topK），强烈建议不要发送
_NO_SAMPLING_MODEL_PREFIXES = ("gemini-3.5",)

# 视频直传：inline_data 整个请求上限 20MB，base64 会膨胀约33%，原始视频取 14MB 以内走内联，
# 超过则走 Files API（免费层存储上限 2GB，文件 48 小时后自动清理）
VIDEO_INLINE_LIMIT_BYTES = 14 * 1024 * 1024


def _model_rejects_sampling_params(model_identifier: str) -> bool:
    """判断模型是否不应发送 temperature/topP/topK（Gemini 3.5 及以后）"""
    return model_identifier.startswith(_NO_SAMPLING_MODEL_PREFIXES)

gemini_safe_settings = [
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
]


def _l2_normalize(vector: list[float]) -> list[float]:
    """对向量做 L2 归一化（已归一化的向量重复归一化无害）"""
    if not vector:
        return vector
    norm = math.sqrt(sum(x * x for x in vector))
    if norm <= 0:
        return vector
    return [x / norm for x in vector]


def _format_to_mime_type(image_format: str) -> str:
    """
    将图片格式转换为正确的MIME类型

    Args:
        image_format (str): 图片格式 (如 'jpg', 'png' 等)

    Returns:
        str: 对应的MIME类型
    """
    format_mapping = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
        "heic": "image/heic",
        "heif": "image/heif",
    }

    return format_mapping.get(image_format.lower(), f"image/{image_format.lower()}")


def _convert_messages(messages: list[Message]) -> tuple[list[dict], list[str] | None]:
    """
    转换消息格式 - 将消息转换为Gemini REST API所需的格式
    :param messages: 消息列表
    :return: (contents, system_instructions)
    """

    def _convert_message_item(message: Message) -> dict:
        """转换单个消息格式"""
        # 转换角色名称
        if message.role == RoleType.Assistant:
            role = "model"
        elif message.role == RoleType.User:
            role = "user"
        else:
            raise ValueError(f"不支持的消息角色: {message.role}")

        # 转换内容
        parts = []
        if isinstance(message.content, str):
            parts.append({"text": message.content})
        elif isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, tuple):  # (format, base64_data)
                    parts.append({"inline_data": {"mime_type": _format_to_mime_type(item[0]), "data": item[1]}})
                elif isinstance(item, str):
                    parts.append({"text": item})
        else:
            raise RuntimeError("无法触及的代码：请使用MessageBuilder类构建消息对象")

        return {"role": role, "parts": parts}

    contents = []
    system_instructions = []

    for message in messages:
        if message.role == RoleType.System:
            if isinstance(message.content, str):
                system_instructions.append(message.content)
            else:
                raise ValueError("System消息不支持非文本内容")
        elif message.role == RoleType.Tool:
            # 工具调用结果处理
            if not message.tool_call_id:
                raise ValueError("工具调用消息缺少tool_call_id")
            contents.append({"role": "function", "parts": [{"text": str(message.content)}]})
        else:
            contents.append(_convert_message_item(message))

    return contents, system_instructions if system_instructions else None


def _convert_tool_options(tool_options: list[ToolOption]) -> list[dict]:
    """
    转换工具选项格式 - 将内部 ToolOption 对象列表转换为 Gemini REST API 所需的格式。

    Args:
        tool_options: 要转换的 ToolOption 对象列表。

    Returns:
        一个列表，其中每个字典都代表一个 Gemini API 的工具声明。
    """

    def _convert_tool_param(param: ToolParam) -> dict[str, Any]:
        """转换工具参数"""
        result: dict[str, Any] = {
            "type": param.param_type.value,
            "description": param.description,
        }
        if param.enum_values:
            result["enum"] = param.enum_values
        return result

    def _convert_tool_option_item(tool_option: ToolOption) -> dict[str, Any]:
        """转换单个工具选项"""
        function_declaration: dict[str, Any] = {
            "name": tool_option.name,
            "description": tool_option.description,
        }

        if tool_option.params:
            function_declaration["parameters"] = {
                "type": "object",
                "properties": {param.name: _convert_tool_param(param) for param in tool_option.params},
                "required": [param.name for param in tool_option.params if param.required],
            }

        return {"function_declarations": [function_declaration]}

    return [_convert_tool_option_item(tool_option) for tool_option in tool_options]


def _build_generation_config(
    max_tokens: int,
    temperature: float,
    thinking_budget: int | None = None,
    thinking_level: str | None = None,
    response_format: RespFormat | None = None,
    extra_params: dict | None = None,
    model_identifier: str = "",
) -> dict:
    """
    构建并返回 Gemini API 的 `generationConfig` 字典。

    此函数整合了多个参数，如最大输出 token 数、温度、思考配置(预算或等级)、响应格式和
    其他自定义参数，以创建一个符合 Gemini API 规范的配置对象。

    注意: thinking_budget 和 thinking_level 不能同时使用，否则会返回 400 错误。
    优先使用 thinking_level (新版)，如果未提供则使用 thinking_budget (旧版)。

    Args:
        max_tokens: 生成内容的最大 token 数。
        temperature: 控制生成文本的随机性，值越高越随机。
        thinking_budget: 模型的思考预算(旧版，与 thinking_level 互斥)。
        thinking_level: 模型的思考等级(新版，可选值: "low", "medium", "high")。
        response_format: 指定响应的格式，例如 JSON 对象或遵循特定 schema。
        extra_params: 一个包含其他要合并到配置中的参数的字典。

    Returns:
        一个包含完整 `generationConfig` 的字典。
    """
    config: dict[str, Any] = {"maxOutputTokens": max_tokens}
    # Gemini 3.5 起官方弃用采样参数，发送虽不报错但强烈不建议；旧模型保持原有行为
    if not _model_rejects_sampling_params(model_identifier):
        config["temperature"] = temperature
        config["topK"] = 1
        config["topP"] = 1

    # 处理思考配置 - 新版 thinking_level 优先于旧版 thinking_budget
    if thinking_level is not None:
        # 使用新版 thinkingLevel 参数
        if thinking_level in VALID_THINKING_LEVELS:
            config["thinkingConfig"] = {"thinkingLevel": thinking_level}
        else:
            logger.warning(f"无效的 thinking_level 值 {thinking_level}，有效值为: {VALID_THINKING_LEVELS}")
    elif thinking_budget is not None:
        # 使用旧版 thinkingBudget 参数
        config["thinkingConfig"] = {"includeThoughts": True, "thinkingBudget": thinking_budget}

    # 处理响应格式
    if response_format:
        if response_format.format_type == RespFormatType.JSON_OBJ:
            config["responseMimeType"] = "application/json"
        elif response_format.format_type == RespFormatType.JSON_SCHEMA:
            config["responseMimeType"] = "application/json"
            config["responseSchema"] = response_format.to_dict()

    # 合并额外参数
    if extra_params:
        # 拷贝一份以防修改原始字典
        safe_extra_params = extra_params.copy()
        # 移除已单独处理的参数
        safe_extra_params.pop("thinking_budget", None)
        safe_extra_params.pop("thinking_level", None)
        config.update(safe_extra_params)

    return config


class AiohttpGeminiStreamParser:
    """
    用于处理和解析来自 Gemini API 的流式（server-sent events）响应的专用解析器。

    该类会累积流式数据块，并从中提取内容、思考过程、工具调用和使用情况统计。
    """

    def __init__(self):
        """初始化解析器，创建用于存储解析数据的缓冲区。"""
        self.content_buffer = io.StringIO()  # 用于累积文本内容
        self.reasoning_buffer = io.StringIO()  # 用于累积思考过程内容
        self.tool_calls_buffer = []  # 用于存储工具调用信息
        self.usage_record = None  # 用于存储最终的使用情况统计

    def parse_chunk(self, chunk_text: str):
        """
        解析单个流式数据块（通常是一行 SSE 数据）。

        Args:
            chunk_text: 从流式响应中接收到的原始文本数据块。
        """
        try:
            if not chunk_text.strip():
                return

            # 移除data:前缀
            if chunk_text.startswith("data: "):
                chunk_text = chunk_text[6:].strip()

            if chunk_text == "[DONE]":
                return

            chunk_data = orjson.loads(chunk_text)

            # 解析候选项
            if chunk_data.get("candidates"):
                candidate = chunk_data["candidates"][0]

                # 解析内容（functionCall 与 thought 都挂在 content.parts 的各 part 上）
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if part.get("functionCall"):
                            func_call = part["functionCall"]
                            call_id = f"gemini_call_{len(self.tool_calls_buffer)}"
                            self.tool_calls_buffer.append(
                                {"id": call_id, "name": func_call.get("name", ""), "args": func_call.get("args", {})}
                            )
                        elif "text" in part:
                            # 带 thought 标记的 part 是思考内容，不能混进正文
                            if part.get("thought"):
                                self.reasoning_buffer.write(part["text"])
                            else:
                                self.content_buffer.write(part["text"])

            # 解析使用统计
            if "usageMetadata" in chunk_data:
                usage = chunk_data["usageMetadata"]
                self.usage_record = (
                    usage.get("promptTokenCount", 0),
                    usage.get("candidatesTokenCount", 0),
                    usage.get("totalTokenCount", 0),
                )

        except orjson.JSONDecodeError as e:
            logger.warning(f"解析流式数据块失败: {e}, 数据: {chunk_text}")
        except Exception as e:
            logger.error(f"处理流式数据块时出错: {e}")

    def get_response(self) -> APIResponse:
        """
        在所有数据块处理完毕后，获取最终的、结构化的 APIResponse 对象。

        此方法会整合所有缓冲区中的数据，并将其封装到一个 APIResponse 对象中。
        调用此方法后，内部缓冲区将被清理。

        Returns:
            一个包含从流中解析出的所有信息的 APIResponse 对象。
        """
        response = APIResponse()

        if self.content_buffer.tell() > 0:
            response.content = self.content_buffer.getvalue()

        if self.reasoning_buffer.tell() > 0:
            response.reasoning_content = self.reasoning_buffer.getvalue()

        if self.tool_calls_buffer:
            response.tool_calls = []
            for call_data in self.tool_calls_buffer:
                response.tool_calls.append(ToolCall(call_data["id"], call_data["name"], call_data["args"]))

        # 清理缓冲区
        self.content_buffer.close()
        self.reasoning_buffer.close()

        return response


async def _default_stream_response_handler(
    response: aiohttp.ClientResponse,
    interrupt_flag: asyncio.Event | None,
) -> tuple[APIResponse, tuple[int, int, int] | None]:
    """
    默认的流式响应处理器。

    此异步函数迭代处理 aiohttp 响应的每一行，使用 AiohttpGeminiStreamParser
    来解析它们，并处理中断信号。

    Args:
        response: aiohttp 的 ClientResponse 对象。
        interrupt_flag: 一个 asyncio.Event，用于发出中断请求的信号。

    Returns:
        一个元组，包含最终的 APIResponse 和使用情况记录。

    Raises:
        ReqAbortException: 如果请求被中断。
        RespParseException: 如果流式响应解析失败。
    """
    parser = AiohttpGeminiStreamParser()

    try:
        async for line in response.content:
            if interrupt_flag and interrupt_flag.is_set():
                raise ReqAbortException("请求被外部信号中断")

            line_text = line.decode("utf-8").strip()
            if line_text:
                parser.parse_chunk(line_text)

        api_response = parser.get_response()
        return api_response, parser.usage_record

    except Exception as e:
        if not isinstance(e, ReqAbortException):
            raise RespParseException(None, f"流式响应解析失败: {e}") from e
        raise


def _default_normal_response_parser(
    response_data: dict,
) -> tuple[APIResponse, tuple[int, int, int] | None]:
    """
    默认的非流式（普通）响应解析器。

    此函数解析一个完整的 JSON 响应体，并从中提取内容、工具调用和使用情况统计。

    Args:
        response_data: 已解析为字典的 JSON 响应数据。

    Returns:
        一个元组，包含最终的 APIResponse 和使用情况记录。

    Raises:
        RespParseException: 如果响应解析失败。
    """
    api_response = APIResponse()

    try:
        # 解析候选项
        if response_data.get("candidates"):
            candidate = response_data["candidates"][0]

            # 解析内容（functionCall 与 thought 都挂在 content.parts 的各 part 上）
            if "content" in candidate and "parts" in candidate["content"]:
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                tool_calls: list[ToolCall] = []
                for part in candidate["content"]["parts"]:
                    if part.get("functionCall"):
                        func_call = part["functionCall"]
                        tool_calls.append(
                            ToolCall(f"gemini_call_{len(tool_calls)}", func_call.get("name", ""), func_call.get("args", {}))
                        )
                    elif "text" in part:
                        # 带 thought 标记的 part 是思考内容，不能混进正文
                        if part.get("thought"):
                            reasoning_parts.append(part["text"])
                        else:
                            content_parts.append(part["text"])

                if content_parts:
                    api_response.content = "".join(content_parts)
                if reasoning_parts:
                    api_response.reasoning_content = "".join(reasoning_parts)
                if tool_calls:
                    api_response.tool_calls = tool_calls

        # 解析使用统计
        usage_record = None
        if "usageMetadata" in response_data:
            usage = response_data["usageMetadata"]
            usage_record = (
                usage.get("promptTokenCount", 0),
                usage.get("candidatesTokenCount", 0),
                usage.get("totalTokenCount", 0),
            )

        api_response.raw_data = response_data
        return api_response, usage_record

    except Exception as e:
        raise RespParseException(response_data, f"响应解析失败: {e}") from e


@client_registry.register_client_class("aiohttp_gemini")
class AiohttpGeminiClient(BaseClient):
    """
    一个使用 aiohttp 库与 Google Gemini API 进行异步通信的客户端。

    该客户端实现了 BaseClient 接口，提供了获取对话响应、处理流式数据、
    管理 API key 和端点等功能。它被设计为无状态的，每次请求都创建一个
    新的 aiohttp.ClientSession，以增强健壮性。
    """

    def __init__(self, api_provider: APIProvider):
        """
        初始化 AiohttpGeminiClient。

        Args:
            api_provider: 包含 API key 和可选的 base_url 的 APIProvider 对象。
        """
        super().__init__(api_provider)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.session: aiohttp.ClientSession | None = None  # 注意：此 session 不再被全局使用

        # 如果 API provider 中提供了自定义的 base_url，则覆盖默认值
        if api_provider.base_url:
            self.base_url = api_provider.base_url.rstrip("/")

    @staticmethod
    def clamp_thinking_budget(tb: int, model_id: str) -> int:
        """
        按模型限制思考预算范围，仅支持指定的模型（支持带数字后缀的新版本）
        """
        limits = None

        # 优先尝试精确匹配
        if model_id in THINKING_BUDGET_LIMITS:
            limits = THINKING_BUDGET_LIMITS[model_id]
        else:
            # 按 key 长度倒序，保证更长的（更具体的，如 -lite）优先
            sorted_keys = sorted(THINKING_BUDGET_LIMITS.keys(), key=len, reverse=True)
            for key in sorted_keys:
                # 必须满足：完全等于 或者 前缀匹配（带 "-" 边界）
                if model_id == key or model_id.startswith(f"{key}-"):
                    limits = THINKING_BUDGET_LIMITS[key]
                    break

        # 特殊值处理
        if tb == THINKING_BUDGET_AUTO:
            return THINKING_BUDGET_AUTO
        if tb == THINKING_BUDGET_DISABLED:
            if limits and limits.get("can_disable", False):
                return THINKING_BUDGET_DISABLED
            return limits["min"] if limits else THINKING_BUDGET_AUTO

        # 已知模型裁剪到范围
        if limits:
            return max(limits["min"], min(tb, limits["max"]))

        # 未知模型，返回动态模式
        logger.warning(f"模型 {model_id} 未在 THINKING_BUDGET_LIMITS 中定义，将使用动态模式 tb=-1 兼容。")
        return tb

    # 移除全局 session，全部请求都用 with aiohttp.ClientSession() as session:

    def _session_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36",
        }

    async def _request_json(self, method: str, endpoint: str, data: dict | None = None) -> dict:
        """
        发起非流式请求并返回已解析的 JSON 数据。

        关键：响应体必须在 ClientSession 的作用域【内】读完。
        旧实现把 response 对象返回到 with 块外再读 body，此时 session 已关闭、
        连接已被回收，读取会挂起或报错——这就是旧版"Gemini 客户端超时"的根源。

        - 网络层 `aiohttp.ClientError` 最多重试3次；
        - HTTP 状态码错误（4xx/5xx）立即失败，不重试。
        """
        api_key = self.api_provider.get_api_key()
        url = f"{self.base_url}/{endpoint}?key={api_key}"

        max_retries = 3
        last_exception = None

        for _attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=max(self.api_provider.timeout, 30)),
                    headers=self._session_headers(),
                ) as session:
                    if method.upper() == "POST":
                        response = await session.post(url, json=data, headers={"Accept": "application/json"})
                    else:
                        response = await session.get(url)

                    if response.status >= 400:
                        error_text = await response.text()
                        raise RespNotOkException(response.status, error_text)

                    # 在 session 作用域内读完并解析 body
                    return await response.json()

            except aiohttp.ClientError as e:
                last_exception = e
                await asyncio.sleep(1)  # 等待1秒后重试

        raise NetworkConnectionError() from last_exception

    async def _request_stream(
        self,
        endpoint: str,
        data: dict,
        stream_response_handler: Callable[
            [aiohttp.ClientResponse, asyncio.Event | None],
            Coroutine[Any, Any, tuple[APIResponse, tuple[int, int, int] | None]],
        ],
        interrupt_flag: asyncio.Event | None,
    ) -> tuple[APIResponse, tuple[int, int, int] | None]:
        """
        发起流式请求，并在 ClientSession 的作用域【内】完成整个流的消费。

        与 _request_json 同理：流必须在 session 存活期间读完，不能把 response 带出 with 块。
        """
        api_key = self.api_provider.get_api_key()
        url = f"{self.base_url}/{endpoint}?key={api_key}"

        max_retries = 3
        last_exception = None

        for _attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=300),
                    headers=self._session_headers(),
                ) as session:
                    response = await session.post(url, json=data, headers={"Accept": "text/event-stream"})

                    if response.status >= 400:
                        error_text = await response.text()
                        raise RespNotOkException(response.status, error_text)

                    # 在 session 作用域内消费完整个流
                    return await stream_response_handler(response, interrupt_flag)

            except aiohttp.ClientError as e:
                last_exception = e
                await asyncio.sleep(1)

        raise NetworkConnectionError() from last_exception

    async def get_response(
        self,
        model_info: ModelInfo,
        message_list: list[Message],
        tool_options: list[ToolOption] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: RespFormat | None = None,
        stream_response_handler: Callable[
            [aiohttp.ClientResponse, asyncio.Event | None],
            Coroutine[Any, Any, tuple[APIResponse, tuple[int, int, int] | None]],
        ]
        | None = None,
        async_response_parser: Callable[[dict], tuple[APIResponse, tuple[int, int, int] | None]] | None = None,
        interrupt_flag: asyncio.Event | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """
        获取一个完整的对话响应，支持流式和非流式模式。

        这是客户端的核心方法，它负责：
        1. 转换输入消息和工具选项。
        2. 构建请求体，包括生成配置。
        3. 根据模型信息决定是使用流式还是非流式端点。
        4. 发起请求并处理中断。
        5. 使用适当的处理器/解析器处理响应。
        6. 格式化最终的 APIResponse 对象，包括使用情况统计。

        Args:
            model_info: 包含模型标识符和配置的模型信息。
            message_list: 对话消息列表。
            tool_options: 可用的工具选项列表。
            max_tokens: 最大生成 token 数。
            temperature: 生成温度。
            response_format: 响应格式。
            stream_response_handler: 用于处理流式响应的可调用对象。
            async_response_parser: 用于解析非流式响应的可调用对象。
            interrupt_flag: 用于中断请求的 asyncio.Event。
            extra_params: 包含额外参数的字典，例如 'thinking_budget'。

        Returns:
            一个包含模型响应的 APIResponse 对象。
        """
        if stream_response_handler is None:
            stream_response_handler = _default_stream_response_handler

        if async_response_parser is None:
            async_response_parser = _default_normal_response_parser

        # 转换消息格式
        contents, system_instructions = _convert_messages(message_list)

        # 处理思考配置 - 优先使用新版 thinking_level，否则使用旧版 thinking_budget
        thinking_level = None
        thinking_budget = None

        if extra_params:
            # 优先检查新版 thinking_level
            if "thinking_level" in extra_params:
                level_value = extra_params.get("thinking_level", "").lower()
                if level_value in VALID_THINKING_LEVELS:
                    thinking_level = level_value
                else:
                    logger.warning(f"无效的 thinking_level 值 {level_value}，有效值为: {VALID_THINKING_LEVELS}")
            # 如果没有 thinking_level，则使用旧版 thinking_budget
            elif "thinking_budget" in extra_params:
                try:
                    tb = int(extra_params["thinking_budget"])
                    thinking_budget = self.clamp_thinking_budget(tb, model_info.model_identifier)
                except (ValueError, TypeError):
                    logger.warning(f"无效的 thinking_budget 值 {extra_params['thinking_budget']}，将使用默认动态模式")
                    thinking_budget = THINKING_BUDGET_AUTO

        # 构建请求体
        request_data = {
            "contents": contents,
            "generationConfig": _build_generation_config(
                max_tokens,
                temperature,
                thinking_budget=thinking_budget,
                thinking_level=thinking_level,
                response_format=response_format,
                extra_params=extra_params,
                model_identifier=model_info.model_identifier,
            ),
            "safetySettings": gemini_safe_settings,
        }

        # 添加系统指令
        if system_instructions:
            request_data["systemInstruction"] = {"parts": [{"text": instr} for instr in system_instructions]}

        # 添加工具定义
        if tool_options:
            request_data["tools"] = _convert_tool_options(tool_options)

        try:
            if model_info.force_stream_mode:
                # 流式请求（整个流在 _request_stream 的 session 作用域内消费）
                endpoint = f"models/{model_info.model_identifier}:streamGenerateContent"
                req_task = asyncio.create_task(
                    self._request_stream(endpoint, request_data, stream_response_handler, interrupt_flag)
                )

                while not req_task.done():
                    if interrupt_flag and interrupt_flag.is_set():
                        req_task.cancel()
                        raise ReqAbortException("请求被外部信号中断")
                    await asyncio.sleep(0.1)

                api_response, usage_record = req_task.result()

            else:
                # 普通请求（JSON 在 _request_json 的 session 作用域内解析完成）
                endpoint = f"models/{model_info.model_identifier}:generateContent"
                req_task = asyncio.create_task(self._request_json("POST", endpoint, request_data))

                while not req_task.done():
                    if interrupt_flag and interrupt_flag.is_set():
                        req_task.cancel()
                        raise ReqAbortException("请求被外部信号中断")
                    await asyncio.sleep(0.1)

                response_data = req_task.result()
                api_response, usage_record = async_response_parser(response_data)

                # 截断检测：Gemini 的思考 token 计入 maxOutputTokens，思考过多会挤掉可见输出
                _candidates = response_data.get("candidates") or []
                if _candidates and _candidates[0].get("finishReason") == "MAX_TOKENS":
                    logger.warning(
                        f"[{model_info.name}] 输出被 maxOutputTokens 截断（思考token也计入上限），"
                        f"建议调大 max_tokens 或将 thinking_level 调低"
                    )

        except (ReqAbortException, NetworkConnectionError, RespNotOkException, RespParseException):
            # 直接重抛项目定义的异常
            raise
        except Exception as e:
            logger.debug(str(e))
            # 其他异常转换为网络连接错误
            raise NetworkConnectionError() from e

        # 设置使用统计
        if usage_record:
            api_response.usage = UsageRecord(
                model_name=model_info.name,
                provider_name=model_info.api_provider,
                prompt_tokens=usage_record[0],
                completion_tokens=usage_record[1],
                total_tokens=usage_record[2],
            )

        return api_response

    # ---- 视频直传理解 ----

    def _upload_base_url(self) -> str:
        """推导 Files API 的上传端点根路径（…/upload/v1beta）"""
        root = self.base_url
        if root.endswith("/v1beta"):
            root = root[: -len("/v1beta")]
        return f"{root}/upload/v1beta"

    async def _upload_file_and_wait_active(
        self, data: bytes, mime_type: str, display_name: str = "mofox_video"
    ) -> tuple[str, str]:
        """通过 Files API 上传文件并等待其变为 ACTIVE 状态。

        使用官方的 resumable 协议：start 拿到上传URL → 一次性 upload+finalize → 轮询状态。

        Returns:
            (file_uri, file_name)：生成请求里用 uri，删除时用 name（形如 files/abc123）
        """
        api_key = self.api_provider.get_api_key()
        start_url = f"{self._upload_base_url()}/files?key={api_key}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600),
            headers=self._session_headers(),
        ) as session:
            # 第一步：发起 resumable 上传，拿到上传 URL
            start_resp = await session.post(
                start_url,
                json={"file": {"display_name": display_name}},
                headers={
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(len(data)),
                    "X-Goog-Upload-Header-Content-Type": mime_type,
                },
            )
            if start_resp.status >= 400:
                raise RespNotOkException(start_resp.status, await start_resp.text())
            upload_url = start_resp.headers.get("X-Goog-Upload-URL")
            if not upload_url:
                raise RespParseException(None, "Files API 未返回上传URL（X-Goog-Upload-URL）")

            # 第二步：上传全部字节并 finalize
            upload_resp = await session.post(
                upload_url,
                data=data,
                headers={
                    "Content-Length": str(len(data)),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                },
            )
            if upload_resp.status >= 400:
                raise RespNotOkException(upload_resp.status, await upload_resp.text())
            file_info = (await upload_resp.json()).get("file") or {}
            file_uri = file_info.get("uri", "")
            file_name = file_info.get("name", "")
            state = file_info.get("state", "")
            if not file_uri or not file_name:
                raise RespParseException(None, f"Files API 上传响应缺少 uri/name: {file_info}")

            # 第三步：等待视频处理完成（PROCESSING → ACTIVE）
            poll_url = f"{self.base_url}/{file_name}?key={api_key}"
            waited = 0.0
            while state == "PROCESSING" and waited < 180:
                await asyncio.sleep(2.0)
                waited += 2.0
                poll_resp = await session.get(poll_url)
                if poll_resp.status >= 400:
                    raise RespNotOkException(poll_resp.status, await poll_resp.text())
                file_info = await poll_resp.json()
                state = file_info.get("state", "")

            if state != "ACTIVE":
                raise RespParseException(None, f"Files API 文件未就绪（state={state}）")

            return file_uri, file_name

    async def _delete_file(self, file_name: str) -> None:
        """删除 Files API 上的文件（尽力而为，失败不抛出——文件48小时后也会自动清理）"""
        try:
            api_key = self.api_provider.get_api_key()
            url = f"{self.base_url}/{file_name}?key={api_key}"
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=self._session_headers(),
            ) as session:
                await session.delete(url)
        except Exception as e:
            logger.debug(f"删除 Files API 文件失败（将自动过期清理）: {e}")

    async def get_video_response(
        self,
        model_info: ModelInfo,
        prompt: str,
        video_bytes: bytes,
        mime_type: str = "video/mp4",
        max_tokens: int = 2048,
        low_resolution: bool = False,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """视频直传理解：把整个视频（含音轨）直接交给 Gemini 分析。

        - 小视频（<= 14MB）走 inline_data，单次请求完成
        - 大视频走 Files API：上传 → 等待 ACTIVE → 生成 → 尽力删除
        - low_resolution=True 时使用低媒体分辨率（约100 token/秒 vs 标准约300 token/秒）
        """
        uploaded_file_name: str | None = None
        if len(video_bytes) <= VIDEO_INLINE_LIMIT_BYTES:
            video_part: dict[str, Any] = {
                "inline_data": {"mime_type": mime_type, "data": base64.b64encode(video_bytes).decode()}
            }
        else:
            logger.info(
                f"视频大小 {len(video_bytes) / 1024 / 1024:.1f}MB 超过内联上限，走 Files API 上传"
            )
            file_uri, uploaded_file_name = await self._upload_file_and_wait_active(video_bytes, mime_type)
            video_part = {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}

        generation_config = _build_generation_config(
            max_tokens,
            0.3,
            response_format=None,
            extra_params=extra_params,
            model_identifier=model_info.model_identifier,
        )
        if low_resolution:
            generation_config["mediaResolution"] = "MEDIA_RESOLUTION_LOW"

        request_data = {
            "contents": [{"role": "user", "parts": [video_part, {"text": prompt}]}],
            "generationConfig": generation_config,
            "safetySettings": gemini_safe_settings,
        }

        try:
            endpoint = f"models/{model_info.model_identifier}:generateContent"
            response_data = await self._request_json("POST", endpoint, request_data)
            api_response, usage_record = _default_normal_response_parser(response_data)

            # 截断检测：Gemini 的思考 token 计入 maxOutputTokens，思考太多会把可见输出挤掉
            candidates = response_data.get("candidates") or []
            finish_reason = candidates[0].get("finishReason") if candidates else None
            if finish_reason == "MAX_TOKENS":
                logger.warning(
                    f"[{model_info.name}] 视频分析输出被 maxOutputTokens 截断"
                    f"（思考token也计入上限），建议调大 max_tokens 或将 thinking_level 调低至 minimal"
                )

            if usage_record:
                api_response.usage = UsageRecord(
                    model_name=model_info.name,
                    provider_name=model_info.api_provider,
                    prompt_tokens=usage_record[0],
                    completion_tokens=usage_record[1],
                    total_tokens=usage_record[2],
                )
            return api_response
        finally:
            if uploaded_file_name:
                await self._delete_file(uploaded_file_name)

    async def get_embedding(
        self,
        model_info: ModelInfo,
        embedding_input: str | list[str],
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """
        获取文本嵌入（gemini-embedding-2 / gemini-embedding-001）。

        - 单条输入走 :embedContent，批量输入走 :batchEmbedContents
        - 维度通过模型的 extra_params 中的 output_dimensionality 控制（128-3072，默认3072）
        - 截断维度后做一次 L2 归一化（embedding-2 服务端会归一化，001 不会；重复归一化无害）
        """
        extra = dict(extra_params) if extra_params else {}
        output_dim = extra.get("output_dimensionality") or extra.get("outputDimensionality")
        model_path = f"models/{model_info.model_identifier}"

        if isinstance(embedding_input, list):
            requests_payload = []
            for text in embedding_input:
                item: dict[str, Any] = {"model": model_path, "content": {"parts": [{"text": text}]}}
                if output_dim:
                    item["outputDimensionality"] = int(output_dim)
                requests_payload.append(item)
            endpoint = f"models/{model_info.model_identifier}:batchEmbedContents"
            response_data = await self._request_json("POST", endpoint, {"requests": requests_payload})
            vectors = [emb.get("values", []) for emb in response_data.get("embeddings", [])]
            embedding_result: list[list[float]] | list[float] = [_l2_normalize(v) for v in vectors]
        else:
            payload: dict[str, Any] = {"content": {"parts": [{"text": embedding_input}]}}
            if output_dim:
                payload["outputDimensionality"] = int(output_dim)
            endpoint = f"models/{model_info.model_identifier}:embedContent"
            response_data = await self._request_json("POST", endpoint, payload)
            embedding_result = _l2_normalize((response_data.get("embedding") or {}).get("values", []))

        api_response = APIResponse()
        api_response.embedding = embedding_result
        api_response.raw_data = response_data
        return api_response

    async def get_audio_transcriptions(
        self,
        model_info: ModelInfo,
        audio_base64: str,
        extra_params: dict[str, Any] | None = None,
    ) -> APIResponse:
        """
        使用 Gemini 模型获取音频文件的转录。

        此方法将 base64 编码的音频数据和预设的提示词发送到 Gemini API，
        并返回生成的文本转录。

        Args:
            model_info: 要使用的模型的信息。
            audio_base64: Base64 编码的 WAV 音频数据。
            extra_params: 传递给生成配置的额外参数。

        Returns:
            一个 APIResponse 对象，其 `content` 字段包含音频转录。

        Raises:
            NetworkConnectionError: 如果发生网络问题。
            RespNotOkException: 如果 API 返回错误状态码。
            RespParseException: 如果响应解析失败。
        """
        # 构建包含音频的内容
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": "Generate a transcript of the speech. The language of the transcript should match the language of the speech."
                    },
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_base64}},
                ],
            }
        ]

        request_data = {
            "contents": contents,
            "generationConfig": _build_generation_config(
                2048,
                0.1,
                thinking_budget=THINKING_BUDGET_AUTO,
                thinking_level=None,
                response_format=None,
                extra_params=extra_params,
                model_identifier=model_info.model_identifier,
            ),
            "safetySettings": gemini_safe_settings,
        }

        try:
            endpoint = f"models/{model_info.model_identifier}:generateContent"
            response_data = await self._request_json("POST", endpoint, request_data)

            api_response, usage_record = _default_normal_response_parser(response_data)

            if usage_record:
                api_response.usage = UsageRecord(
                    model_name=model_info.name,
                    provider_name=model_info.api_provider,
                    prompt_tokens=usage_record[0],
                    completion_tokens=usage_record[1],
                    total_tokens=usage_record[2],
                )

            return api_response

        except (NetworkConnectionError, RespNotOkException, RespParseException):
            raise
        except Exception as e:
            raise NetworkConnectionError() from e

    def get_support_image_formats(self) -> list[str]:
        """
        获取支持的图片格式

        gif 默认不在列表中（MessageBuilder 会将其切成最多4帧PNG）；
        开启 [emoji].gif_native_upload 后 gif 以原格式直传，Gemini 能完整感知动画。
        """
        formats = ["png", "jpg", "jpeg", "webp", "heic", "heif"]
        try:
            from src.config.config import global_config

            if global_config and getattr(global_config.emoji, "gif_native_upload", False):
                formats.append("gif")
        except Exception:
            pass
        return formats

    # 移除 __aenter__、__aexit__、__del__，不再持有全局 session
