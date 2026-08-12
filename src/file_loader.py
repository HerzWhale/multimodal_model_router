"""读取输入文件并生成初始文件清单。

文件清单是流水线中的第一个结构化对象，用于在预处理和模型调用前记录基础文件信息。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from runtime_config import runtime_policy_section


def _configured_extensions(media_type: str) -> set[str]:
    """读取指定媒体类型的扩展名白名单。"""

    values = runtime_policy_section("file_extensions").get(media_type, [])
    if not isinstance(values, list):
        raise ValueError(f"file_extensions.{media_type} 必须是数组。")
    return {str(value).lower() for value in values}


TEXT_EXTENSIONS = _configured_extensions("text")
IMAGE_EXTENSIONS = _configured_extensions("image")
VIDEO_EXTENSIONS = _configured_extensions("video")


def detect_media_type(file_path: str | Path) -> str | None:
    """返回文件类型；不支持的文件返回 None。"""

    suffix = Path(file_path).suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def build_file_manifest(
    input_dir: str | Path,
    batch_id: str,
    created_at: str | None = None,
) -> list[dict[str, Any]]:
    """扫描输入目录并返回文件元数据记录。

    MVP 阶段会忽略不支持的文件类型。返回结果按路径排序，以保证 file_id 生成稳定。
    """

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    manifest_created_at = created_at or datetime.now().astimezone().isoformat(timespec="seconds")
    supported_files = sorted(
        path for path in input_path.rglob("*") if path.is_file() and detect_media_type(path)
    )

    manifest: list[dict[str, Any]] = []
    for index, path in enumerate(supported_files, start=1):
        manifest.append(
            {
                "batch_id": batch_id,
                "file_id": f"file_{index:04d}",
                "file_name": path.name,
                "source_path": str(path.resolve()),
                "media_type": detect_media_type(path),
                "file_size_bytes": path.stat().st_size,
                "created_at": manifest_created_at,
            }
        )

    return manifest
