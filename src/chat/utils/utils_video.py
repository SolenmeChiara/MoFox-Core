#!/usr/bin/env python3
"""纯 inkfox 视频关键帧分析工具

仅依赖 `inkfox.video` 提供的 Rust 扩展能力：
    - extract_keyframes_from_video
    - get_system_info

功能：
    - 关键帧提取 (base64, timestamp)
    - 批量 / 逐帧 LLM 描述
    - 自动模式 (<=3 帧批量，否则逐帧)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image
from sqlalchemy import select

from src.common.database.core import get_db_session
from src.common.database.core.models import Videos
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.llm_models.utils_model import LLMRequest

# 简易并发控制：同一 hash 只处理一次
_video_locks: dict[str, asyncio.Lock] = {}
_locks_guard = asyncio.Lock()

logger = get_logger("utils_video")

from inkfox import video  # type: ignore


class VideoAnalyzer:
    """基于 inkfox 的视频关键帧 + LLM 描述分析器"""

    def __init__(self) -> None:
        assert global_config is not None
        assert model_config is not None
        cfg = getattr(global_config, "video_analysis", object())
        self.max_frames: int = getattr(cfg, "max_frames", 20)
        self.frame_quality: int = getattr(cfg, "frame_quality", 85)
        self.max_image_size: int = getattr(cfg, "max_image_size", 600)
        self.enable_frame_timing: bool = getattr(cfg, "enable_frame_timing", True)
        self.use_simd: bool = getattr(cfg, "rust_use_simd", True)
        self.threads: int = getattr(cfg, "rust_threads", 0)
        self.ffmpeg_path: str = getattr(cfg, "ffmpeg_path", "ffmpeg")
        self.analysis_mode: str = getattr(cfg, "analysis_mode", "auto")
        self.direct_low_resolution: bool = getattr(cfg, "direct_low_resolution", False)
        self.frame_analysis_delay: float = 0.3

        # 人格与提示模板
        try:
            persona = global_config.personality
            self.personality_core = getattr(persona, "personality_core", "是一个积极向上的女大学生")
            self.personality_side = getattr(persona, "personality_side", "用一句话或几句话描述人格的侧面特点")
        except Exception:  # pragma: no cover
            self.personality_core = "是一个积极向上的女大学生"
            self.personality_side = "用一句话或几句话描述人格的侧面特点"

        self.batch_analysis_prompt = getattr(
            cfg,
            "batch_analysis_prompt",
            """请以第一人称视角阅读这些按时间顺序提取的关键帧。\n核心：{personality_core}\n人格：{personality_side}\n请详细描述视频(主题/人物与场景/动作与时间线/视觉风格/情绪氛围/特殊元素)。""",
        )

        try:
            self.video_llm = LLMRequest(
                model_set=model_config.model_task_config.video_analysis, request_type="video_analysis"
            )
        except Exception:
            self.video_llm = LLMRequest(model_set=model_config.model_task_config.vlm, request_type="vlm")

        self._log_system()

    # ---- 系统信息 ----
    def _log_system(self) -> None:
        try:
            info = video.get_system_info()  # type: ignore[attr-defined]
            logger.info(
                f"inkfox: threads={info.get('threads')} version={info.get('version')} simd={info.get('simd_supported')}"
            )
        except Exception as e:  # pragma: no cover
            logger.debug(f"获取系统信息失败: {e}")

    # ---- 关键帧提取 ----
    async def extract_keyframes(self, video_path: str) -> list[tuple[str, float]]:
        """提取关键帧并返回 (base64, timestamp_seconds) 列表"""
        with tempfile.TemporaryDirectory() as tmp:
            result = video.extract_keyframes_from_video(  # type: ignore[attr-defined]
                video_path=video_path,
                output_dir=tmp,
                max_keyframes=self.max_frames * 2,  # 先多抓一点再截断
                max_save=self.max_frames,
                ffmpeg_path=self.ffmpeg_path,
                use_simd=self.use_simd,
                threads=self.threads,
                verbose=False,
            )
            files = sorted(Path(tmp).glob("keyframe_*.jpg"))[: self.max_frames]
            total_ms = getattr(result, "total_time_ms", 0)
            frames: list[tuple[str, float]] = []
            for i, f in enumerate(files):
                img = Image.open(f).convert("RGB")
                if max(img.size) > self.max_image_size:
                    scale = self.max_image_size / max(img.size)
                    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=self.frame_quality)
                b64 = base64.b64encode(buf.getvalue()).decode()
                ts = (i / max(1, len(files) - 1)) * (total_ms / 1000.0) if total_ms else float(i)
                frames.append((b64, ts))
            return frames

    # ---- 批量分析 ----
    async def _analyze_batch(self, frames: list[tuple[str, float]], question: str | None) -> str:
        from src.llm_models.payload_content.message import MessageBuilder, RoleType

        prompt = self.batch_analysis_prompt.format(
            personality_core=self.personality_core, personality_side=self.personality_side
        )
        if question:
            prompt += f"\n用户关注: {question}"
        desc = [
            (f"第{i + 1}帧 (时间: {ts:.2f}s)" if self.enable_frame_timing else f"第{i + 1}帧")
            for i, (_b, ts) in enumerate(frames)
        ]
        prompt += "\n帧列表: " + ", ".join(desc)
        mb = MessageBuilder().set_role(RoleType.User).add_text_content(prompt)
        for b64, _ in frames:
            mb.add_image_content("jpeg", b64)
        message = mb.build()
        resp = await self.video_llm.execute_with_messages(
            message_list=[message],
            temperature=None,
            max_tokens=None,
        )
        return resp.content or "❌ 未获得响应"

    # ---- 逐帧分析 ----
    async def _analyze_sequential(self, frames: list[tuple[str, float]], question: str | None) -> str:
        results: list[str] = []
        for i, (b64, ts) in enumerate(frames):
            prompt = f"分析第{i + 1}帧" + (f" (时间: {ts:.2f}s)" if self.enable_frame_timing else "")
            if question:
                prompt += f"\n关注: {question}"
            try:
                text, _ = await self.video_llm.generate_response_for_image(
                    prompt=prompt, image_base64=b64, image_format="jpeg"
                )
                results.append(f"第{i + 1}帧: {text}")
            except Exception as e:  # pragma: no cover
                results.append(f"第{i + 1}帧: 失败 {e}")
            if i < len(frames) - 1:
                await asyncio.sleep(self.frame_analysis_delay)
        summary_prompt = "基于以下逐帧结果给出完整总结:\n\n" + "\n".join(results)
        try:
            final, _ = await self.video_llm.generate_response_for_image(
                prompt=summary_prompt, image_base64=frames[-1][0], image_format="jpeg"
            )
            return final
        except Exception:  # pragma: no cover
            return "\n".join(results)

    # ---- Gemini 视频直传 ----
    def _find_gemini_direct_model(self):
        """从视频分析任务的模型列表中找到第一个走 aiohttp_gemini 客户端的模型。

        Returns:
            (ModelInfo, AiohttpGeminiClient) 或 (None, None)
        """
        try:
            assert model_config is not None
            task = model_config.model_task_config.video_analysis
            for model_name in task.model_list:
                mi = model_config.models_dict.get(model_name)
                if not mi:
                    continue
                provider = model_config.api_providers_dict.get(mi.api_provider)
                if provider and provider.client_type == "aiohttp_gemini":
                    from src.llm_models.model_client.aiohttp_gemini_client import AiohttpGeminiClient

                    return mi, AiohttpGeminiClient(provider)
        except Exception as e:
            logger.warning(f"查找视频直传模型失败: {e}")
        return None, None

    async def _analyze_direct(self, video_bytes: bytes, question: str | None) -> str:
        """视频直传分析：整个视频（含音轨）直接交给 Gemini，不抽帧。

        失败时抛出异常，由调用方回退到抽帧方案。
        """
        model_info, client = self._find_gemini_direct_model()
        if not model_info or not client:
            raise RuntimeError(
                "utils_video 任务的模型列表中没有走 aiohttp_gemini 客户端的模型，无法使用视频直传"
            )

        prompt = (
            f"你将看到一段视频（包含画面和声音）。\n"
            f"你的人设核心：{self.personality_core}\n人格侧面：{self.personality_side}\n"
            f"请以第一人称、符合人设的口吻描述视频内容（主题/人物与场景/动作与时间线/对白或声音/情绪氛围/特殊元素）。\n"
            f"要求：直接开始描述正文，禁止自我介绍、禁止'我正在观看/审视'之类的过程性开场白，控制在300字以内。"
        )
        if question:
            prompt += f"\n用户关注: {question}"

        # Gemini 的思考 token 计入 maxOutputTokens，给足余量防止描述被思考挤掉而截断
        try:
            max_tokens = int(getattr(self.video_llm.model_for_task, "max_tokens", 2048)) or 2048
        except Exception:
            max_tokens = 2048
        max_tokens = max(max_tokens, 2048)

        resp = await client.get_video_response(
            model_info=model_info,
            prompt=prompt,
            video_bytes=video_bytes,
            mime_type="video/mp4",
            max_tokens=max_tokens,
            low_resolution=self.direct_low_resolution,
            extra_params=model_info.extra_params or None,
        )
        if not resp.content:
            raise RuntimeError("视频直传未获得响应内容")
        if resp.usage:
            logger.info(
                f"视频直传完成 (模型: {model_info.name}): "
                f"输入 {resp.usage.prompt_tokens} tokens, 输出 {resp.usage.completion_tokens} tokens"
            )
        return resp.content

    # ---- 主入口 ----
    async def analyze_video(self, video_path: str, question: str | None = None) -> tuple[bool, str]:
        if not os.path.exists(video_path):
            return False, "❌ 文件不存在"

        # 视频直传模式：整个视频交给 Gemini；失败时自动回退到抽帧方案
        if self.analysis_mode == "gemini_direct":
            try:
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                text = await self._analyze_direct(video_bytes, question)
                return True, text
            except Exception as e:
                logger.warning(f"视频直传失败，回退到抽帧分析: {e}")

        frames = await self.extract_keyframes(video_path)
        if not frames:
            return False, "❌ 未提取到关键帧"
        # 模式名映射：配置文件用 batch_frames/frame_by_frame，代码内部用 batch/sequential
        mode = {
            "batch_frames": "batch",
            "frame_by_frame": "sequential",
            "gemini_direct": "batch",  # 直传失败回退时按批量处理
        }.get(self.analysis_mode, self.analysis_mode)
        if mode == "auto":
            mode = "batch" if len(frames) <= 20 else "sequential"
        text = await (
            self._analyze_batch(frames, question) if mode == "batch" else self._analyze_sequential(frames, question)
        )
        return True, text

    async def analyze_video_from_bytes(
        self,
        video_bytes: bytes,
        filename: str | None = None,
        prompt: str | None = None,
        question: str | None = None,
    ) -> dict[str, str]:
        """从内存字节分析视频，兼容旧调用 (prompt / question 二选一) 返回 {"summary": str}."""
        if not video_bytes:
            return {"summary": "❌ 空视频数据"}
        # 兼容参数：prompt 优先，其次 question
        q = prompt if prompt is not None else question
        video_hash = hashlib.sha256(video_bytes).hexdigest()

        # 查缓存（注意：SQLAlchemy 2.0 的 Row 不支持用 Column 对象索引，必须用 select 指定列后按位置取值）
        try:
            async with get_db_session() as session:  # type: ignore
                row = (
                    await session.execute(
                        select(Videos.description, Videos.vlm_processed).where(Videos.video_hash == video_hash)
                    )
                ).first()
                if row and row[0] and row[1]:
                    logger.debug(f"视频分析缓存命中: {video_hash[:16]}...")
                    return {"summary": row[0]}
        except Exception as e:  # pragma: no cover
            logger.debug(f"视频缓存查询失败: {e}")

        # 获取锁避免重复处理
        async with _locks_guard:
            lock = _video_locks.get(video_hash)
            if lock is None:
                lock = asyncio.Lock()
                _video_locks[video_hash] = lock
        async with lock:
            # 双检：进入锁后再查一次，避免重复处理
            try:
                async with get_db_session() as session:  # type: ignore
                    row = (
                        await session.execute(
                            select(Videos.description, Videos.vlm_processed).where(Videos.video_hash == video_hash)
                        )
                    ).first()
                    if row and row[0] and row[1]:
                        logger.debug(f"视频分析缓存命中（锁内双检）: {video_hash[:16]}...")
                        return {"summary": row[0]}
            except Exception as e:  # pragma: no cover
                logger.debug(f"视频缓存双检查询失败: {e}")

            try:
                with tempfile.NamedTemporaryFile(delete=False) as fp:
                    fp.write(video_bytes)
                    temp_path = fp.name
                try:
                    ok, summary = await self.analyze_video(temp_path, q)
                    # 写入缓存（仅成功）
                    if ok:
                        try:
                            async with get_db_session() as session:  # type: ignore
                                await session.execute(
                                    Videos.__table__.insert().values(
                                        video_id="",
                                        video_hash=video_hash,
                                        description=summary,
                                        count=1,
                                        timestamp=time.time(),
                                        vlm_processed=True,
                                        duration=None,
                                        frame_count=None,
                                        fps=None,
                                        resolution=None,
                                        file_size=len(video_bytes),
                                    )
                                )
                                await session.commit()
                        except Exception as e:  # pragma: no cover
                            logger.warning(f"视频分析结果写入缓存失败（下次将重复分析）: {e}")
                    return {"summary": summary}
                finally:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:  # pragma: no cover
                            pass
            except Exception as e:  # pragma: no cover
                return {"summary": f"❌ 处理失败: {e}"}


# ---- 外部接口 ----
_INSTANCE: VideoAnalyzer | None = None


def get_video_analyzer() -> VideoAnalyzer:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = VideoAnalyzer()
    return _INSTANCE


def is_video_analysis_available() -> bool:
    return True


def get_video_analysis_status() -> dict[str, Any]:
    try:
        info = video.get_system_info()  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        return {"available": False, "error": str(e)}
    inst = get_video_analyzer()
    return {
        "available": True,
        "system": info,
        "modes": ["auto", "batch", "sequential"],
        "max_frames_default": inst.max_frames,
        "implementation": "inkfox",
    }
