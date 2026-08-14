"""在模型调用前准备文本、图片和视频文件。

文本文件会读取为 raw_text。图片会透传给 OCR 和视觉理解。视频 V1 只做本地预处理：
尽量读取视频元信息、抽取前段优先的代表关键帧，并在本机具备 ffmpeg 时抽取音频文件。
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from runtime_config import runtime_policy_section

VIDEO_POLICY = runtime_policy_section("video_preprocessing")
DEFAULT_MAX_KEYFRAMES = int(VIDEO_POLICY.get("default_max_keyframes", 3))
DEFAULT_AUDIO_SAMPLE_RATE_HZ = int(VIDEO_POLICY.get("audio_sample_rate_hz", 16000))
DEFAULT_AUDIO_CHANNELS = int(VIDEO_POLICY.get("audio_channels", 1))
DEFAULT_FFMPEG_TIMEOUT_SECONDS = int(VIDEO_POLICY.get("ffmpeg_timeout_seconds", 120))
DEFAULT_PROCESS_ERROR_PREVIEW_CHARS = int(VIDEO_POLICY.get("process_error_preview_chars", 300))
DEFAULT_EARLY_FRAME_RATIO = float(VIDEO_POLICY.get("early_frame_ratio", 0.05))
DEFAULT_KEYFRAME_SAMPLING_STRATEGY = str(VIDEO_POLICY.get("keyframe_sampling_strategy", "start_early_then_spaced"))
DEFAULT_MIN_KEYFRAMES_FOR_STABLE_EVIDENCE = int(VIDEO_POLICY.get("min_keyframes_for_stable_evidence", 3))
DEFAULT_REQUIRE_EARLY_KEYFRAME_FOR_STABLE_EVIDENCE = bool(
    VIDEO_POLICY.get("require_early_keyframe_for_stable_evidence", True)
)
DEFAULT_EARLY_EVIDENCE_WINDOW_MS = int(VIDEO_POLICY.get("early_evidence_window_ms", 3000))


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


def _load_cv2() -> Any | None:
    """按需加载 OpenCV；未安装时返回 None，避免把视频预处理变成强依赖。"""

    try:
        return importlib.import_module("cv2")
    except ImportError:
        return None


def _write_keyframe(cv2: Any, keyframe_path: Path, frame: Any) -> bool:
    """写入关键帧；OpenCV 直写失败时用 imencode 兜底，以兼容中文路径。"""

    if cv2.imwrite(str(keyframe_path), frame):
        return True
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return False
    keyframe_path.write_bytes(encoded.tobytes())
    return True


def _normalize_ffmpeg_candidate(raw_path: str | Path) -> str | None:
    """把用户显式提供的 ffmpeg 路径规范化为可执行文件路径。"""

    value = str(raw_path).strip().strip('"')
    if not value:
        return None

    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate)
    if candidate.is_dir():
        for executable_name in ("ffmpeg.exe", "ffmpeg"):
            executable_path = candidate / executable_name
            if executable_path.is_file():
                return str(executable_path)

    resolved_from_path = shutil.which(value)
    if resolved_from_path:
        return resolved_from_path
    return None


def _find_ffmpeg_executable(ffmpeg_path: str | Path | None = None) -> str | None:
    """查找本机 ffmpeg 可执行文件；优先使用显式路径，其次使用环境变量，最后查 PATH。"""

    if ffmpeg_path:
        resolved_path = _normalize_ffmpeg_candidate(ffmpeg_path)
        if resolved_path:
            return resolved_path

    env_ffmpeg_path = os.environ.get("FFMPEG_PATH")
    if env_ffmpeg_path:
        resolved_path = _normalize_ffmpeg_candidate(env_ffmpeg_path)
        if resolved_path:
            return resolved_path

    return shutil.which("ffmpeg")


def _preview_process_error(stderr: bytes) -> str:
    """截取本地命令错误输出，避免把过长日志写进结果文件。"""

    text = stderr.decode("utf-8", errors="replace").strip()
    if len(text) > DEFAULT_PROCESS_ERROR_PREVIEW_CHARS:
        return f"{text[:DEFAULT_PROCESS_ERROR_PREVIEW_CHARS]}..."
    return text


def _extract_audio_with_ffmpeg(
    source_path: Path,
    artifact_dir: Path,
    *,
    ffmpeg_path: str | Path | None = None,
    sample_rate_hz: int = DEFAULT_AUDIO_SAMPLE_RATE_HZ,
    channels: int = DEFAULT_AUDIO_CHANNELS,
) -> dict[str, Any]:
    """用 ffmpeg 抽取单声道 wav 音频；失败时返回可记录的状态和警告。"""

    resolved_ffmpeg_path = _find_ffmpeg_executable(ffmpeg_path)
    if resolved_ffmpeg_path is None:
        return {
            "audio_path": None,
            "audio_extraction_status": "dependency_missing",
            "warning_message": "本地未找到 ffmpeg，视频音频未提取。",
            "audio_extraction_method": "ffmpeg_wav",
        }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    audio_path = artifact_dir / f"{source_path.stem}_audio.wav"
    command = [
        resolved_ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate_hz),
        "-f",
        "wav",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=DEFAULT_FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "audio_path": None,
            "audio_extraction_status": "timeout",
            "warning_message": "ffmpeg 音频提取超时，视频音频未提取。",
            "audio_extraction_method": "ffmpeg_wav",
        }
    except OSError as exc:
        return {
            "audio_path": None,
            "audio_extraction_status": "failed",
            "warning_message": f"ffmpeg 音频提取启动失败：{exc}",
            "audio_extraction_method": "ffmpeg_wav",
        }

    if completed.returncode != 0:
        detail = _preview_process_error(completed.stderr)
        return {
            "audio_path": None,
            "audio_extraction_status": "failed",
            "warning_message": f"ffmpeg 音频提取失败。{detail}",
            "audio_extraction_method": "ffmpeg_wav",
        }
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        return {
            "audio_path": None,
            "audio_extraction_status": "empty_output",
            "warning_message": "ffmpeg 执行成功但未写出有效音频文件。",
            "audio_extraction_method": "ffmpeg_wav",
        }

    return {
        "audio_path": str(audio_path.resolve()),
        "audio_extraction_status": "extracted",
        "warning_message": None,
        "audio_extraction_method": "ffmpeg_wav",
    }


def _sample_keyframe_indices(frame_count: int | None, max_keyframes: int = DEFAULT_MAX_KEYFRAMES) -> list[int]:
    """选择关键帧下标：起始帧、早期帧，再补齐后段覆盖。"""

    if max_keyframes <= 0:
        return []
    if frame_count is None or frame_count <= 0:
        return [0]

    sample_count = min(frame_count, max_keyframes)
    if sample_count == 1:
        return [0]
    if sample_count == 2:
        return [0, frame_count - 1]

    early_index = max(1, int(round((frame_count - 1) * DEFAULT_EARLY_FRAME_RATIO)))
    raw_indices = [0, min(early_index, frame_count - 1)]
    remaining_count = sample_count - len(set(raw_indices))
    raw_indices.extend(
        int(round(position * (frame_count - 1) / remaining_count))
        for position in range(1, remaining_count + 1)
    )
    deduplicated_indices: list[int] = []
    for index in raw_indices:
        if index not in deduplicated_indices:
            deduplicated_indices.append(index)
    return deduplicated_indices


def _timestamp_ms_for_frame(frame_index: int, fps: float | None) -> int | None:
    """根据帧号和 FPS 估算关键帧在视频中的时间位置。"""

    if fps is None or fps <= 0:
        return None
    return int(round(frame_index / fps * 1000))


def _assess_video_evidence_stability(
    keyframe_metadata: list[dict[str, Any]],
    *,
    min_keyframes: int = DEFAULT_MIN_KEYFRAMES_FOR_STABLE_EVIDENCE,
    require_early_keyframe: bool = DEFAULT_REQUIRE_EARLY_KEYFRAME_FOR_STABLE_EVIDENCE,
    early_window_ms: int = DEFAULT_EARLY_EVIDENCE_WINDOW_MS,
) -> dict[str, Any]:
    """判断视频关键帧证据是否足够支撑后续分类。"""

    reasons: list[str] = []
    if len(keyframe_metadata) < min_keyframes:
        reasons.append(f"关键帧数量少于 {min_keyframes} 张。")
    timestamps = [
        item.get("timestamp_ms")
        for item in keyframe_metadata
        if isinstance(item.get("timestamp_ms"), int)
    ]
    if require_early_keyframe and not any(timestamp <= early_window_ms for timestamp in timestamps):
        reasons.append(f"缺少前 {early_window_ms}ms 内的早期关键帧。")
    return {
        "video_evidence_stability": "stable" if not reasons else "weak",
        "video_evidence_risk_reasons": reasons,
    }


def preprocess_video(
    source_path: str | Path,
    artifact_dir: str | Path | None = None,
    max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
    ffmpeg_path: str | Path | None = None,
) -> dict[str, Any]:
    """视频 V1 预处理。

    当前只做最小闭环：如果本地有 OpenCV 且视频可读，就抽取前段优先的代表关键帧；
    如果本地有 ffmpeg 且提供了产物目录，就抽取单声道 wav 音频文件。
    """

    path = Path(source_path)
    cv2 = _load_cv2()
    warnings: list[str] = []
    keyframes: list[str] = []
    keyframe_metadata: list[dict[str, Any]] = []
    duration_ms: int | None = None
    frame_count: int | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    keyframe_status = "not_attempted"
    audio_path: str | None = None
    audio_status = "not_attempted_no_artifact_dir"
    audio_extraction_method: str | None = None
    preprocess_status = "metadata_only"

    if cv2 is None:
        warnings.append("本地未安装 OpenCV，视频V1无法读取元信息或抽取关键帧。")
        preprocess_status = "failed"
    else:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                warnings.append("OpenCV 无法打开该视频，未能读取元信息或抽取关键帧。")
                preprocess_status = "failed"
                keyframe_status = "failed"
            else:
                raw_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                raw_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                raw_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                frame_count = raw_frame_count if raw_frame_count > 0 else None
                fps = round(raw_fps, 6) if raw_fps > 0 else None
                width = raw_width if raw_width > 0 else None
                height = raw_height if raw_height > 0 else None
                if frame_count is not None and fps:
                    duration_ms = int(round(frame_count / fps * 1000))

                sampled_indices = _sample_keyframe_indices(frame_count, max_keyframes=max_keyframes)
                if artifact_dir is not None and sampled_indices:
                    target_dir = Path(artifact_dir)
                    target_dir.mkdir(parents=True, exist_ok=True)

                    for keyframe_number, frame_index in enumerate(sampled_indices, start=1):
                        if hasattr(capture, "set") and hasattr(cv2, "CAP_PROP_POS_FRAMES"):
                            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        success, frame = capture.read()
                        keyframe_path = target_dir / f"{path.stem}_frame_{keyframe_number:04d}.jpg"
                        if success and frame is not None and _write_keyframe(cv2, keyframe_path, frame):
                            resolved_keyframe_path = str(keyframe_path.resolve())
                            keyframes.append(resolved_keyframe_path)
                            keyframe_metadata.append(
                                {
                                    "frame_index": keyframe_number,
                                    "source_frame_index": frame_index,
                                    "timestamp_ms": _timestamp_ms_for_frame(frame_index, fps),
                                    "path": resolved_keyframe_path,
                                }
                            )
                        else:
                            warnings.append(f"第 {keyframe_number} 张关键帧读取或写入失败，源帧号为 {frame_index}。")

                    if keyframes and len(keyframes) == len(sampled_indices):
                        keyframe_status = "extracted"
                        preprocess_status = "success"
                    elif keyframes:
                        keyframe_status = "partial_extracted"
                        preprocess_status = "partial_success"
                    else:
                        keyframe_status = "failed"
                        warnings.append("OpenCV 已打开视频，但没有写出可用关键帧文件。")
                elif artifact_dir is None and sampled_indices:
                    keyframe_status = "readable_not_written"
                    warnings.append("视频关键帧可定位，但未提供 artifact_dir，因此没有写出关键帧文件。")
                else:
                    keyframe_status = "failed"
                    warnings.append("OpenCV 已打开视频，但没有定位到可用关键帧。")
        finally:
            capture.release()

    if artifact_dir is None:
        warnings.append("未提供 artifact_dir，视频音频未写出为本地产物。")
    else:
        audio_result = _extract_audio_with_ffmpeg(path, Path(artifact_dir), ffmpeg_path=ffmpeg_path)
        audio_path = audio_result["audio_path"]
        audio_status = str(audio_result["audio_extraction_status"])
        audio_extraction_method = str(audio_result["audio_extraction_method"])
        if audio_result.get("warning_message"):
            warnings.append(str(audio_result["warning_message"]))

    evidence_stability = _assess_video_evidence_stability(keyframe_metadata)
    if evidence_stability["video_evidence_stability"] == "weak":
        warnings.extend(evidence_stability["video_evidence_risk_reasons"])

    video_preprocess = {
        "schema_version": "v1",
        "preprocess_status": preprocess_status,
        "source_path": str(path.resolve()),
        "artifact_dir": str(Path(artifact_dir).resolve()) if artifact_dir is not None else None,
        "keyframe_paths": keyframes,
        "keyframe_metadata": keyframe_metadata,
        "keyframe_count": len(keyframes),
        "max_keyframes": max_keyframes,
        "keyframe_sampling_strategy": DEFAULT_KEYFRAME_SAMPLING_STRATEGY,
        "keyframe_extraction_status": keyframe_status,
        "audio_path": audio_path,
        "audio_extraction_status": audio_status,
        "audio_extraction_method": audio_extraction_method,
        "audio_sample_rate_hz": DEFAULT_AUDIO_SAMPLE_RATE_HZ,
        "audio_channels": DEFAULT_AUDIO_CHANNELS,
        "duration_ms": duration_ms,
        "duration_source": "opencv_frame_count_fps" if duration_ms is not None else "unavailable",
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        **evidence_stability,
        "warning_messages": warnings,
    }
    return {
        "keyframes": keyframes,
        "keyframe_metadata": keyframe_metadata,
        "audio_path": audio_path,
        "duration_ms": duration_ms,
        "preprocessing_artifacts": video_preprocess,
    }


def preprocess_file(
    file_record: dict[str, Any],
    artifact_dir: str | Path | None = None,
    ffmpeg_path: str | Path | None = None,
    max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
) -> dict[str, Any]:
    """根据文件类型选择对应的预处理方式。"""

    media_type = file_record["media_type"]
    source_path = file_record["source_path"]

    if media_type == "text":
        return preprocess_text(source_path)
    if media_type == "image":
        return preprocess_image(source_path)
    if media_type == "video":
        return preprocess_video(source_path, artifact_dir=artifact_dir, ffmpeg_path=ffmpeg_path, max_keyframes=max_keyframes)

    raise ValueError(f"不支持的文件类型: {media_type}")
