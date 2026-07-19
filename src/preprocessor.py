"""在模型调用前准备文本、图片和视频文件。

文本文件会读取为 raw_text。图片会透传给 OCR 和视觉理解。视频会转换为 keyframes、
audio_path 和 duration_ms。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def preprocess_text(source_path: str | Path) -> dict[str, Any]:
    """读取文本文件，返回文本分析需要的原文。"""

    path = Path(source_path)
    return {
        "raw_text": path.read_text(encoding="utf-8"),
        "duration_ms": None,
    }


def preprocess_image(source_path: str | Path) -> dict[str, Any]:
    """图片文件在 MVP 阶段只透传原路径。"""

    path = Path(source_path)
    return {
        "image_path": str(path.resolve()),
        "duration_ms": None,
    }


def preprocess_video(source_path: str | Path) -> dict[str, Any]:
    """视频文件在 MVP 阶段返回模拟关键帧、模拟音频路径和模拟时长。"""

    path = Path(source_path)
    stem = path.with_suffix("")
    return {
        "keyframes": [str(stem.with_name(f"{stem.name}_frame_0001.jpg").resolve())],
        "audio_path": str(stem.with_suffix(".wav").resolve()),
        "duration_ms": 0,
    }


def preprocess_file(file_record: dict[str, Any]) -> dict[str, Any]:
    """根据文件类型选择对应的预处理方式。"""

    media_type = file_record["media_type"]
    source_path = file_record["source_path"]

    if media_type == "text":
        return preprocess_text(source_path)
    if media_type == "image":
        return preprocess_image(source_path)
    if media_type == "video":
        return preprocess_video(source_path)

    raise ValueError(f"不支持的文件类型: {media_type}")
