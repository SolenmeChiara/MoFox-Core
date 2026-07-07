from typing import Any

# 常见Error Code Mapping (以OpenAI API为例)
error_code_mapping = {
    400: "参数不正确",
    401: "API-Key错误，认证失败，请检查/config/model_list.toml中的配置是否正确",
    402: "账号余额不足",
    403: "模型拒绝访问，可能需要实名或余额不足",
    404: "Not Found",
    413: "请求体过大，请尝试压缩图片或减少输入内容",
    429: "请求过于频繁，请稍后再试",
    500: "服务器内部故障",
    503: "服务器负载过高",
}


class NetworkConnectionError(Exception):
    """连接异常，常见于网络问题或服务器不可用"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "连接异常，请检查网络连接状态或URL是否正确"


class ReqAbortException(Exception):
    """请求异常退出，常见于请求被中断或取消"""

    def __init__(self, message: str | None = None):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message or "请求因未知原因异常终止"


class RespNotOkException(Exception):
    """请求响应异常，见于请求未能成功响应（非 '200 OK'）"""

    def __init__(self, status_code: int, message: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

    def __str__(self):
        if self.status_code in error_code_mapping:
            return error_code_mapping[self.status_code]
        elif self.message:
            return self.message
        else:
            return f"未知的异常响应代码：{self.status_code}"


class RespParseException(Exception):
    """响应解析错误，常见于响应格式不正确或解析方法不匹配"""

    def __init__(self, ext_info: Any, message: str | None = None):
        super().__init__(message)
        self.ext_info = ext_info
        self.message = message

    def __str__(self):
        return self.message or "解析响应内容时发生未知错误，请检查是否配置了正确的解析方法"


class ModelRefusalException(Exception):
    """模型安全分类器拒答（HTTP 200 + stop_reason="refusal"）。

    典型见于 Claude Fable 5 等带安全分类器的模型：响应体 stop_reason="refusal"、content 为空，
    并在 stop_details 里给出 category 等信息。**拒答对相同输入是确定性的**，重试纯属浪费，
    因此本异常语义为「不可重试」：utils_model 捕获后跳过内部空回复重试，直接把该模型计入失败列表
    （多模型集下照常故障转移），并把拒答标记与 category 带进上层错误信息。

    ``str(异常)`` 恒以 ``REFUSAL_TAG`` 开头并携带 ``category=...``，便于上层做字符串识别与日志 grep。
    """

    REFUSAL_TAG = "【模型拒答】"

    def __init__(self, model_name: str, category: Any = None, stop_details: dict | None = None):
        self.model_name = model_name
        self.category = category
        self.stop_details = stop_details or {}
        super().__init__(f"{self.REFUSAL_TAG}模型 {model_name} 被安全分类器拒答 (category={category})")


class PayLoadTooLargeError(Exception):
    """自定义异常类，用于处理请求体过大错误"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return "请求体过大，请尝试压缩图片或减少输入内容。"


class RequestAbortException(Exception):
    """自定义异常类，用于处理请求中断异常"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class PermissionDeniedException(Exception):
    """自定义异常类，用于处理访问拒绝的异常"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message
