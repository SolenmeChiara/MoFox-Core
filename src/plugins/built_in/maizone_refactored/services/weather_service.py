"""
天气服务模块

给「发说说」注入一段真实世界的天气背景：加拿大环境部（Environment and Climate Change Canada）
的 citypage 实时数据，取当天的实况温度 / 天况 / 预报文本 / 高低温 / 降水概率，外加当前生效中
的天气预警，拼成一小段提示词块交给 ContentService 插进内容生成提示词。

三条纪律：
1. **绝不阻断发说说**：接口超时、返回异常、JSON 结构变动，一律只记日志并返回空字符串，
   由调用方静默降级成「不带天气」。对外方法自带 try/except，不向上抛任何异常。
2. **带缓存**：成功结果按 ``weather.cache_ttl_minutes``（默认 30 分钟）缓存；失败也做
   **负缓存 300 秒**——接口宕机时不至于每次发说说都白等一次 10 秒超时。
3. **默认关闭**：``weather.enable`` 默认 False，不开启时第一行就返回，零成本零请求。

数据源（免 key 的公开 GET 接口，返回 GeoJSON）：
``https://api.weather.gc.ca/collections/citypageweather-realtime/items?lang=en&f=json&identifier=on-24``
其中 ``identifier`` 是 citypage 的城市代码，``on-24`` = 安大略省密西沙加。

已知的数据坑（实测）：
- ``currentConditions.windChill`` 夏天会残留脏值（实况 16.9°C 时返回 -3），**不使用该字段**。
- ``abbreviatedForecast.pop``（降水概率）**经常整个缺失**，不是恒有字段，必须可缺省。
- ``forecasts[0].temperatures.temperature[]`` 通常只有一项：白天时段只有 high、夜间时段只有
  low。因此白天取到 high 后会再从 ``forecasts[1]``（Tonight）补一个 low，显示成「最高 20°C /
  夜间最低 14°C」；夜里 ``forecasts[0]`` 自带 low，不做补取，避免误取到明天白天的温度。
- ``warnings`` 无预警时是空数组；各级子键都可能缺失，全程 ``.get()`` 兜底。
- JSON 全 UTF-8、时间戳全 UTC——提示词块里**不放预警的具体时刻**，省掉时区换算的坑。
"""

import time
from collections.abc import Callable
from typing import Any

import aiohttp

from src.common.logger import get_logger

logger = get_logger("MaiZone.WeatherService")

# 接口地址与请求参数（identifier 由配置提供，走 params 传参避免拼接注入）
_API_URL = "https://api.weather.gc.ca/collections/citypageweather-realtime/items"
_USER_AGENT = "MoFox-Bot weather module"
_REQUEST_TIMEOUT_SECONDS = 10

# 代码级默认值兜底（配置缺失时使用，与 plugin.py 的 config_schema 保持同值）
_DEFAULT_ENABLE = False
_DEFAULT_IDENTIFIER = "on-24"
_DEFAULT_CITY_NAME = "密西沙加"
_DEFAULT_CACHE_TTL_MINUTES = 30

# 失败负缓存秒数（硬编码）：接口宕机 / 结构异常时，这段时间内不再重复请求
_FAILURE_CACHE_SECONDS = 300

# 缓存条目上限（identifier 基本不变，正常只有 1 条；纯粹防御配置被反复改写）
_CACHE_MAX_SIZE = 8

# 克制护栏：预警这类高强度输入容易劫持文案，这句用来防「每条说说都在聊天气」
_HINT_LINE = "（天气只是背景信息，顺其自然，不必每次提及；无预警时可完全不提天气。）"


def _pick_en(node: Any) -> Any:
    """从 ``{"en": ..., "fr": ...}`` 形态的节点里取英文值；节点不是 dict 时返回 None。"""
    if isinstance(node, dict):
        return node.get("en")
    return None


def _fmt_temp(value: Any) -> str | None:
    """把温度值格式化成 ``16.9`` / ``20`` 这样的字符串；非数字返回 None。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.1f}"


def _extract_high_low(forecast: Any) -> tuple[Any, Any]:
    """
    从单条 forecast 的 ``temperatures.temperature[]`` 里取出 ``(high, low)``，取不到的为 None。

    每项用同级 ``class.en`` 区分 high / low；单个时段通常只有一项（白天只有 high、夜间只有 low）。
    """
    high: Any = None
    low: Any = None
    if not isinstance(forecast, dict):
        return high, low
    temps_node = forecast.get("temperatures")
    if not isinstance(temps_node, dict):
        return high, low
    temp_list = temps_node.get("temperature")
    if not isinstance(temp_list, list):
        return high, low
    for item in temp_list:
        if not isinstance(item, dict):
            continue
        kind = _pick_en(item.get("class"))
        value = _pick_en(item.get("value"))
        if kind == "high" and high is None:
            high = value
        elif kind == "low" and low is None:
            low = value
    return high, low


def _parse_citypage(data: Any) -> dict | None:
    """
    把 citypage 的原始 JSON 解析成扁平字典（纯函数，不依赖 aiohttp / 项目模块，便于离线单测）。

    :param data: ``response.json()`` 的原始返回，任何类型都能接受。
    :return: 解析结果字典；结构完全对不上（拿不到 ``features[0].properties``）时返回 None。
             字典里每个字段都可能是 None，调用方需自行容忍缺失。
    """
    if not isinstance(data, dict):
        return None

    features = data.get("features")
    if not isinstance(features, list) or not features:
        return None

    first = features[0]
    if not isinstance(first, dict):
        return None

    props = first.get("properties")
    if not isinstance(props, dict):
        return None

    # --- 实况 ---
    current = props.get("currentConditions")
    current = current if isinstance(current, dict) else {}
    temperature = None
    temperature_node = current.get("temperature")
    if isinstance(temperature_node, dict):
        temperature = _pick_en(temperature_node.get("value"))
    condition = _pick_en(current.get("condition"))

    # --- 今日预报（forecasts[0]，白天/夜间取决于当前时刻）---
    forecast_group = props.get("forecastGroup")
    forecast_group = forecast_group if isinstance(forecast_group, dict) else {}
    forecasts = forecast_group.get("forecasts")
    forecast_list = forecasts if isinstance(forecasts, list) else []
    today = forecast_list[0] if forecast_list and isinstance(forecast_list[0], dict) else {}

    period = None
    period_node = today.get("period")
    if isinstance(period_node, dict):
        period = _pick_en(period_node.get("textForecastName"))

    text_summary = _pick_en(today.get("textSummary"))

    # 高低温：单个时段通常只给一个数，需要往后一个时段补另一半
    high, low = _extract_high_low(today)

    # forecasts[0] 是白天时段（Today / This Afternoon）时只有 high，当晚最低温在 forecasts[1]
    # （Tonight）里，补过来才凑得齐「最高 X / 夜间最低 Y」。反过来若 forecasts[0] 已是夜间时段，
    # 它自带 low 就不会走这里，也就不会把 forecasts[1]（明天白天）的温度误当成今天的。
    # 只在已有 high 时才补：单独一个来自下一时段的 low 容易被误读成今天的最低温。
    low_is_overnight = False
    if high is not None and low is None and len(forecast_list) > 1:
        _, next_low = _extract_high_low(forecast_list[1])
        if next_low is not None:
            low = next_low
            low_is_overnight = True

    # 降水概率：实测经常整个缺失，属正常
    pop = None
    abbreviated = today.get("abbreviatedForecast")
    if isinstance(abbreviated, dict):
        pop = abbreviated.get("pop")
        if isinstance(pop, dict):
            pop = pop.get("en")

    # --- 预警（空数组 = 无预警）---
    warnings: list[dict] = []
    raw_warnings = props.get("warnings")
    if isinstance(raw_warnings, list):
        for item in raw_warnings:
            if not isinstance(item, dict):
                continue
            description = _pick_en(item.get("description"))
            if not description:
                continue
            warnings.append({"description": str(description), "type": _pick_en(item.get("type"))})

    return {
        "temperature": temperature,
        "condition": condition,
        "period": period,
        "text_summary": text_summary,
        "high": high,
        "low": low,
        # low 是从下一个（夜间）时段补来的，展示时要标明「夜间最低」，别混成今天白天的最低温
        "low_is_overnight": low_is_overnight,
        "pop": pop,
        "warnings": warnings,
    }


def _format_weather_block(parsed: dict | None, city_name: str) -> str:
    """
    把解析结果拼成提示词块（纯函数）。个别字段缺失时省略对应片段，而不是整体失败；
    一个可用片段都没有时返回空字符串（避免留下孤立标题）。
    """
    if not parsed:
        return ""

    lines: list[str] = []

    # --- 【今日天气】行 ---
    segments: list[str] = []
    temp_text = _fmt_temp(parsed.get("temperature"))
    condition = parsed.get("condition")
    if temp_text and condition:
        segments.append(f"当前 {temp_text}°C，{condition}")
    elif temp_text:
        segments.append(f"当前 {temp_text}°C")
    elif condition:
        segments.append(f"当前天况 {condition}")

    forecast_parts: list[str] = []
    text_summary = parsed.get("text_summary")
    if text_summary:
        period = parsed.get("period")
        head = f"今日预报（{period}）" if period else "今日预报"
        forecast_parts.append(f"{head}：{text_summary}")

    high_text = _fmt_temp(parsed.get("high"))
    low_text = _fmt_temp(parsed.get("low"))
    low_label = "夜间最低" if parsed.get("low_is_overnight") else "最低"
    if high_text and low_text:
        forecast_parts.append(f"最高 {high_text}°C / {low_label} {low_text}°C")
    elif high_text:
        forecast_parts.append(f"最高 {high_text}°C")
    elif low_text:
        forecast_parts.append(f"{low_label} {low_text}°C")

    pop = parsed.get("pop")
    if isinstance(pop, (int, float)) and not isinstance(pop, bool):
        forecast_parts.append(f"降水概率 {int(pop)}%")

    if forecast_parts:
        segments.append("，".join(forecast_parts))

    if segments:
        lines.append(f"【今日天气】{city_name}：{'；'.join(segments)}")

    # --- 【天气预警】行（无预警时整行省略，不放具体时刻）---
    warnings = parsed.get("warnings") or []
    warning_texts = []
    for warning in warnings:
        description = warning.get("description")
        if not description:
            continue
        warning_type = warning.get("type")
        warning_texts.append(f"{description}（{warning_type}）" if warning_type else str(description))
    if warning_texts:
        lines.append(f"【天气预警】{'；'.join(warning_texts)}")

    if not lines:
        return ""

    lines.append(_HINT_LINE)
    return "\n".join(lines)


class WeatherService:
    """天气背景服务：对外只有 ``get_weather_block()`` 一个方法，永不抛异常。"""

    def __init__(self, get_config: Callable):
        """
        :param get_config: 插件的配置读取函数（与其它 service 一致）。
        """
        self.get_config = get_config
        # identifier -> {"block": str, "last_updated": float, "ttl": float}
        self._cache: dict[str, dict] = {}

    async def get_weather_block(self) -> str:
        """
        取一段可直接塞进提示词的天气背景文本。

        :return: 提示词块；未启用 / 获取失败 / 数据不可用时统一返回空字符串。
        """
        try:
            if not self.get_config("weather.enable", _DEFAULT_ENABLE):
                return ""

            identifier = str(self.get_config("weather.identifier", _DEFAULT_IDENTIFIER) or _DEFAULT_IDENTIFIER)
            city_name = str(self.get_config("weather.city_name", _DEFAULT_CITY_NAME) or _DEFAULT_CITY_NAME)

            cached = self._get_from_cache(identifier)
            if cached is not None:
                # 命中缓存的路径本来完全静默，部署后无从确认「这条说说到底带没带天气」，补一行 debug
                logger.debug(f"天气缓存命中（{identifier}），本次{'带' if cached else '不带'}天气")
                return cached

            data = await self._fetch_raw(identifier)
            parsed = _parse_citypage(data) if data is not None else None
            block = _format_weather_block(parsed, city_name)

            if block:
                warning_count = len(parsed.get("warnings") or []) if parsed else 0
                temp_text = _fmt_temp(parsed.get("temperature")) if parsed else None
                logger.info(
                    f"天气获取成功：{city_name}（{identifier}）当前 {temp_text or '未知'}°C，生效预警 {warning_count} 条"
                )
                self._update_cache(identifier, block, self._cache_ttl_seconds())
            else:
                # 请求失败或数据不可用：负缓存，避免接口宕机时每次发说说都白等超时
                logger.warning(
                    f"天气数据不可用，本次说说不带天气（{identifier}），{_FAILURE_CACHE_SECONDS} 秒内不再重试"
                )
                self._update_cache(identifier, "", _FAILURE_CACHE_SECONDS)

            return block
        except Exception as e:
            # 兜底：任何意外都不许冒泡到发说说主流程
            logger.warning(f"获取天气背景信息异常，本次说说不带天气: {e}")
            return ""

    def _cache_ttl_seconds(self) -> float:
        """成功结果的缓存秒数，配置非法时回落到默认值。"""
        try:
            minutes = int(self.get_config("weather.cache_ttl_minutes", _DEFAULT_CACHE_TTL_MINUTES))
        except (TypeError, ValueError):
            minutes = _DEFAULT_CACHE_TTL_MINUTES
        return max(1, minutes) * 60

    def _get_from_cache(self, identifier: str) -> str | None:
        """命中且未过期则返回缓存块（可能是负缓存的空字符串），否则返回 None。"""
        entry = self._cache.get(identifier)
        if entry and time.time() - entry["last_updated"] < entry["ttl"]:
            return entry["block"]
        return None

    def _update_cache(self, identifier: str, block: str, ttl: float) -> None:
        """写缓存；条目数越界时整体清空（identifier 基本不变，不值得做 LRU）。"""
        if len(self._cache) >= _CACHE_MAX_SIZE and identifier not in self._cache:
            self._cache.clear()
        self._cache[identifier] = {"block": block, "last_updated": time.time(), "ttl": ttl}

    async def _fetch_raw(self, identifier: str) -> Any | None:
        """请求 citypage 接口，返回原始 JSON；任何失败都只记日志并返回 None。"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    _API_URL,
                    params={"lang": "en", "f": "json", "identifier": identifier},
                    headers={"User-Agent": _USER_AGENT},
                    timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS),
                ) as response:
                    if response.status != 200:
                        logger.warning(f"天气接口返回异常状态: {response.status}（identifier={identifier}）")
                        return None
                    # 接口 Content-Type 是 geo+json，需关掉 aiohttp 的类型校验
                    return await response.json(content_type=None)
        except aiohttp.ClientError as e:
            logger.warning(f"天气接口网络请求失败: {e}")
            return None
        except Exception as e:
            logger.warning(f"天气接口请求异常: {e}")
            return None
