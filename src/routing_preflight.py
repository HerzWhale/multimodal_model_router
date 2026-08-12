"""运行前路由策略预检查。

这个模块只读取本地配置和可选的估算输入，不调用 DeepSeek、PaddleOCR 或任何外部服务。
它的作用是在批处理开始前回答三个问题：
1. 当前任务是否都有明确路由；
2. 当前路由在预算、延迟、真实模型覆盖率约束下是否存在明显风险；
3. 哪些结论因为缺少预估用量或历史延迟而暂时不能判断。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from cost_latency_tracker import load_model_prices
from file_loader import build_file_manifest
from model_catalog import UNKNOWN_VALUE_TEXT, is_mock_model
from model_router import load_routing_rules
from report_generator import read_jsonl
from routing_policy import build_constraints, load_policy_config


DEFAULT_EXPECTED_TASK_TYPES = (
    "ocr",
    "visual_understanding",
    "speech_to_text",
    "text_analysis",
    "summary_merge",
)

DEFAULT_MAX_PRICE_AGE_DAYS = 7
TRUSTED_PRICE_CONFIDENCES = {
    "official_public_page",
    "call_level_reconciled",
    "period_level_reconciled",
}
EXPLAINABLE_PRICE_CONFIDENCES = {
    "mock_only",
    "local_external_api_zero",
}
LOCAL_RUNTIME_PROVIDERS = {"paddlepaddle"}
LOCAL_RUNTIME_MODEL_PREFIXES = ("PP-OCR",)


def _latency_runtime_type(provider: Any, model_name: Any) -> str:
    """把一次调用归入延迟来源类型：真实 API、本地运行或 mock 占位。"""

    model_text = str(model_name or "")
    provider_text = str(provider or "").lower()
    if is_mock_model(model_text):
        return "mock"
    if provider_text in LOCAL_RUNTIME_PROVIDERS or model_text.startswith(LOCAL_RUNTIME_MODEL_PREFIXES):
        return "local_runtime"
    return "real_api"


def _p95_or_unknown(values: list[float]) -> float | str:
    """有数据时返回 P95，没有数据时明确标记为当前数据未提供。"""

    if not values:
        return UNKNOWN_VALUE_TEXT
    return _p95(values)


def _latency_interpretation(task_type: str, stats: dict[str, Any]) -> str:
    """根据真实 API、本地运行和 mock 占比解释某个任务的延迟口径。"""

    if stats["mock_call_count"] and not stats["real_call_count"]:
        return "该任务当前只有 mock 延迟，不能用于判断真实供应商性能。"
    if stats["local_runtime_call_count"] and not stats["real_api_call_count"]:
        return "该任务延迟主要来自本地运行环境，例如本机 CPU 推理、模型加载、图片分辨率和版面复杂度。"
    if stats["real_api_call_count"] and not stats["local_runtime_call_count"] and not stats["mock_call_count"]:
        return "该任务延迟来自真实外部 API 调用，可作为受控样本下的供应商延迟证据，但不能直接外推为 SLA。"
    if stats["real_api_call_count"] and stats["mock_call_count"]:
        return "该任务同时包含真实 API 和 mock 延迟，整体 P95 只能用于发现风险，不能直接代表单一供应商表现。"
    if stats["local_runtime_call_count"] and stats["mock_call_count"]:
        return "该任务同时包含本地运行和 mock 延迟，应优先查看本地运行 P95，mock 0ms 不代表真实能力。"
    return "该任务延迟来源混合或样本不足，需要拆分后再解释。"


def build_workload_profile(
    input_dir: str | Path,
    *,
    include_files: list[str] | tuple[str, ...] | None = None,
    expected_frames_per_video: int = 3,
    expected_audio_seconds_per_video: int | None = None,
    expected_output_tokens_per_file: int = 300,
    estimated_evidence_tokens_per_image: int = 300,
    estimated_evidence_tokens_per_video: int = 800,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """基于输入目录生成运行前规模画像，不读取或调用任何模型。"""

    if expected_frames_per_video < 0:
        raise ValueError("expected_frames_per_video 不能小于 0")
    if expected_audio_seconds_per_video is not None and expected_audio_seconds_per_video < 0:
        raise ValueError("expected_audio_seconds_per_video 不能小于 0")
    if expected_output_tokens_per_file < 0:
        raise ValueError("expected_output_tokens_per_file 不能小于 0")
    if estimated_evidence_tokens_per_image < 0 or estimated_evidence_tokens_per_video < 0:
        raise ValueError("证据 token 估算值不能小于 0")

    manifest = build_file_manifest(
        input_dir,
        batch_id="preflight_workload",
        created_at=generated_at,
    )
    manifest = _filter_manifest_by_include_files(manifest, include_files)

    media_type_counts = {"text": 0, "image": 0, "video": 0}
    total_file_size_bytes = 0
    text_char_count = 0
    files: list[dict[str, Any]] = []

    for record in manifest:
        media_type = str(record["media_type"])
        media_type_counts[media_type] = media_type_counts.get(media_type, 0) + 1
        total_file_size_bytes += int(record.get("file_size_bytes", 0))
        files.append(
            {
                "file_name": record["file_name"],
                "media_type": media_type,
                "file_size_bytes": record.get("file_size_bytes", 0),
            }
        )
        if media_type == "text":
            text_char_count += _read_text_char_count(record["source_path"])

    text_file_count = media_type_counts.get("text", 0)
    image_file_count = media_type_counts.get("image", 0)
    video_file_count = media_type_counts.get("video", 0)
    total_files = len(manifest)
    estimated_video_frames = video_file_count * expected_frames_per_video
    estimated_ocr_images = image_file_count + estimated_video_frames
    estimated_visual_frames = image_file_count + estimated_video_frames
    estimated_raw_text_tokens = _estimate_tokens_from_char_count(text_char_count)
    estimated_evidence_tokens = (
        image_file_count * estimated_evidence_tokens_per_image
        + video_file_count * estimated_evidence_tokens_per_video
    )
    estimated_text_analysis_input_tokens = estimated_raw_text_tokens + estimated_evidence_tokens
    estimated_text_analysis_output_tokens = total_files * expected_output_tokens_per_file

    expected_units_by_task: dict[str, Any] = {}
    if estimated_ocr_images > 0:
        expected_units_by_task["ocr"] = {
            "unit_type": "image_count",
            "quantity": estimated_ocr_images,
        }
    if estimated_visual_frames > 0:
        expected_units_by_task["visual_understanding"] = {
            "unit_type": "frame_count",
            "quantity": estimated_visual_frames,
        }
    if video_file_count > 0 and expected_audio_seconds_per_video is not None:
        expected_units_by_task["speech_to_text"] = {
            "unit_type": "audio_seconds",
            "quantity": video_file_count * expected_audio_seconds_per_video,
        }
    if total_files > 0:
        expected_units_by_task["text_analysis"] = [
            {
                "unit_type": "input_tokens",
                "quantity": estimated_text_analysis_input_tokens,
            },
            {
                "unit_type": "output_tokens",
                "quantity": estimated_text_analysis_output_tokens,
            },
        ]

    warnings = _workload_profile_warnings(
        video_file_count=video_file_count,
        expected_audio_seconds_per_video=expected_audio_seconds_per_video,
        text_file_count=text_file_count,
        image_file_count=image_file_count,
        total_files=total_files,
    )

    return {
        "schema_version": "v1",
        "profile_type": "routing_preflight_workload",
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_dir": str(Path(input_dir)),
        "include_files": list(include_files or []),
        "total_files": total_files,
        "media_type_counts": media_type_counts,
        "total_file_size_bytes": total_file_size_bytes,
        "estimated_raw_text_char_count": text_char_count,
        "estimated_raw_text_tokens": estimated_raw_text_tokens,
        "expected_units_by_task": expected_units_by_task,
        "assumptions": {
            "expected_frames_per_video": expected_frames_per_video,
            "expected_audio_seconds_per_video": expected_audio_seconds_per_video
            if expected_audio_seconds_per_video is not None
            else UNKNOWN_VALUE_TEXT,
            "expected_output_tokens_per_file": expected_output_tokens_per_file,
            "estimated_evidence_tokens_per_image": estimated_evidence_tokens_per_image,
            "estimated_evidence_tokens_per_video": estimated_evidence_tokens_per_video,
            "summary_merge_units": "当前 workload 画像不默认生成汇总任务；只有长文本切分后需要跨片段汇总时才应显式提供。",
        },
        "files": files,
        "warning_messages": warnings,
    }


def _normalize_path_list(model_calls_paths: str | Path | list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    """把单个路径或路径列表统一成 Path 列表。"""

    if isinstance(model_calls_paths, (str, Path)):
        paths = [Path(model_calls_paths)]
    else:
        paths = [Path(path) for path in model_calls_paths]
    if not paths:
        raise ValueError("至少需要提供一个 model_calls.jsonl 路径")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"model_calls 文件不存在: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"model_calls 路径不是文件: {path}")
    return paths


def _as_optional_float(value: Any) -> float | None:
    """把值安全转换成浮点数；失败时返回 None。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _average(values: list[float]) -> float:
    """计算平均值；没有数据时返回 0。"""

    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _p95(values: list[float]) -> float:
    """计算 95 分位值；没有数据时返回 0。"""

    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = math.ceil(len(sorted_values) * 0.95) - 1
    return sorted_values[index]


def _latency_profile_warnings(
    *,
    task_latency_stats: dict[str, Any],
    skipped_records: int,
    source_count: int,
) -> list[str]:
    """生成历史延迟画像的风险提示。"""

    warnings: list[str] = []
    if not task_latency_stats:
        warnings.append("历史调用文件中没有可用延迟记录，P95延迟仍无法判断。")
    if skipped_records:
        warnings.append(f"有 {skipped_records} 条历史调用因缺少 task_type 或 latency_ms 被跳过。")
    if source_count > 1:
        warnings.append("历史延迟来自多个批次，批次输入规模和运行环境可能不同，不能等同于下一批真实延迟。")
    mock_tasks = [
        task_type
        for task_type, stats in task_latency_stats.items()
        if stats["mock_call_count"] > 0
    ]
    if mock_tasks:
        warnings.append(f"以下任务的历史延迟包含 mock 调用：{', '.join(mock_tasks)}；这些延迟不能代表真实供应商性能。")
    local_runtime_tasks = [
        task_type
        for task_type, stats in task_latency_stats.items()
        if stats.get("local_runtime_call_count", 0) > 0
    ]
    if local_runtime_tasks:
        warnings.append(f"以下任务包含本地运行延迟：{', '.join(local_runtime_tasks)}；这些延迟反映本机环境，不等同于云端 API SLA。")
    return warnings


def build_historical_latency_profile(
    model_calls_paths: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    status_filter: list[str] | tuple[str, ...] | None = ("success",),
    generated_at: str | None = None,
) -> dict[str, Any]:
    """从已有 model_calls.jsonl 生成任务级历史 P95 延迟画像。"""

    paths = _normalize_path_list(model_calls_paths)
    status_set = set(status_filter or [])
    latency_by_task: dict[str, list[float]] = {}
    real_call_count_by_task: dict[str, int] = {}
    real_api_call_count_by_task: dict[str, int] = {}
    local_runtime_call_count_by_task: dict[str, int] = {}
    mock_call_count_by_task: dict[str, int] = {}
    real_api_latency_by_task: dict[str, list[float]] = {}
    local_runtime_latency_by_task: dict[str, list[float]] = {}
    mock_latency_by_task: dict[str, list[float]] = {}
    models_by_task: dict[str, set[str]] = {}
    source_summaries: list[dict[str, Any]] = []
    skipped_records = 0

    for path in paths:
        records = read_jsonl(path)
        used_records = 0
        for record in records:
            if status_set and str(record.get("status")) not in status_set:
                continue
            task_type = str(record.get("task_type") or "")
            latency_ms = _as_optional_float(record.get("latency_ms"))
            if not task_type or latency_ms is None:
                skipped_records += 1
                continue
            latency_by_task.setdefault(task_type, []).append(latency_ms)
            models_by_task.setdefault(task_type, set()).add(
                f"{record.get('provider', UNKNOWN_VALUE_TEXT)}/{record.get('model_name', UNKNOWN_VALUE_TEXT)}"
            )
            runtime_type = _latency_runtime_type(record.get("provider"), record.get("model_name"))
            if runtime_type == "mock":
                mock_call_count_by_task[task_type] = mock_call_count_by_task.get(task_type, 0) + 1
                mock_latency_by_task.setdefault(task_type, []).append(latency_ms)
            elif runtime_type == "local_runtime":
                real_call_count_by_task[task_type] = real_call_count_by_task.get(task_type, 0) + 1
                local_runtime_call_count_by_task[task_type] = local_runtime_call_count_by_task.get(task_type, 0) + 1
                local_runtime_latency_by_task.setdefault(task_type, []).append(latency_ms)
            else:
                real_call_count_by_task[task_type] = real_call_count_by_task.get(task_type, 0) + 1
                real_api_call_count_by_task[task_type] = real_api_call_count_by_task.get(task_type, 0) + 1
                real_api_latency_by_task.setdefault(task_type, []).append(latency_ms)
            used_records += 1
        source_summaries.append(
            {
                "path": str(Path(path)),
                "total_records": len(records),
                "used_records": used_records,
            }
        )

    task_latency_stats = {}
    for task_type, values in sorted(latency_by_task.items()):
        stats = {
            "call_count": len(values),
            "real_call_count": real_call_count_by_task.get(task_type, 0),
            "real_api_call_count": real_api_call_count_by_task.get(task_type, 0),
            "local_runtime_call_count": local_runtime_call_count_by_task.get(task_type, 0),
            "mock_call_count": mock_call_count_by_task.get(task_type, 0),
            "avg_latency_ms": _average(values),
            "p95_latency_ms": _p95(values),
            "max_latency_ms": max(values),
            "real_api_p95_latency_ms": _p95_or_unknown(real_api_latency_by_task.get(task_type, [])),
            "local_runtime_p95_latency_ms": _p95_or_unknown(local_runtime_latency_by_task.get(task_type, [])),
            "mock_p95_latency_ms": _p95_or_unknown(mock_latency_by_task.get(task_type, [])),
            "models": sorted(models_by_task.get(task_type, set())),
        }
        stats["latency_interpretation"] = _latency_interpretation(task_type, stats)
        task_latency_stats[task_type] = stats

    warnings = _latency_profile_warnings(
        task_latency_stats=task_latency_stats,
        skipped_records=skipped_records,
        source_count=len(paths),
    )

    return {
        "schema_version": "v1",
        "profile_type": "routing_preflight_latency",
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_model_calls": source_summaries,
        "status_filter": list(status_filter or []),
        "historical_p95_latency_by_task_ms": {
            task_type: stats["p95_latency_ms"]
            for task_type, stats in task_latency_stats.items()
        },
        "task_latency_stats": task_latency_stats,
        "skipped_records": skipped_records,
        "warning_messages": warnings,
    }


def build_latency_bottleneck_analysis(
    latency_profile: dict[str, Any] | None,
    *,
    p95_latency_limit_ms: Any = None,
    task_latency_targets_ms: dict[str, Any] | None = None,
    expected_task_types: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """把历史延迟拆成真实 API、本地运行和 mock 三类，避免混用口径。"""

    raw_task_latency_stats = latency_profile.get("task_latency_stats", {}) if latency_profile else {}
    if expected_task_types is None:
        task_latency_stats = raw_task_latency_stats
    else:
        expected_task_set = {str(task_type) for task_type in expected_task_types}
        task_latency_stats = {
            task_type: stats
            for task_type, stats in raw_task_latency_stats.items()
            if task_type in expected_task_set
        }
    latency_limit = _as_optional_float(p95_latency_limit_ms)
    if latency_limit is not None and not math.isfinite(latency_limit):
        latency_limit = None
    normalized_task_targets = _normalize_task_latency_targets(task_latency_targets_ms)
    has_latency_limit = latency_limit is not None or bool(normalized_task_targets)

    real_api_slow_tasks: list[dict[str, Any]] = []
    local_runtime_slow_tasks: list[dict[str, Any]] = []
    mock_latency_unusable_tasks: list[dict[str, Any]] = []
    top_latency_tasks: list[dict[str, Any]] = []

    for task_type, stats in sorted(
        task_latency_stats.items(),
        key=lambda item: _safe_sort_latency(item[1].get("p95_latency_ms")),
        reverse=True,
    ):
        p95_latency = stats.get("p95_latency_ms", UNKNOWN_VALUE_TEXT)
        task_latency_limit = _latency_limit_for_task(task_type, normalized_task_targets, latency_limit)
        top_latency_tasks.append(
            {
                "task_type": task_type,
                "p95_latency_ms": p95_latency,
                "latency_target_ms": task_latency_limit if task_latency_limit is not None else UNKNOWN_VALUE_TEXT,
                "latency_interpretation": stats.get("latency_interpretation", UNKNOWN_VALUE_TEXT),
            }
        )

        real_api_p95 = _finite_optional_float(stats.get("real_api_p95_latency_ms"))
        if real_api_p95 is not None and _is_latency_over_limit(real_api_p95, task_latency_limit):
            real_api_slow_tasks.append(
                {
                    "task_type": task_type,
                    "p95_latency_ms": real_api_p95,
                    "latency_target_ms": task_latency_limit if task_latency_limit is not None else UNKNOWN_VALUE_TEXT,
                    "call_count": stats.get("real_api_call_count", 0),
                    "evidence_level": "real_external_api",
                    "reason": "真实外部 API 调用耗时超过当前 P95 目标，可能包含网络往返、供应商排队、模型生成和提示词长度影响。",
                }
            )

        local_runtime_p95 = _finite_optional_float(stats.get("local_runtime_p95_latency_ms"))
        if local_runtime_p95 is not None and _is_latency_over_limit(local_runtime_p95, task_latency_limit):
            local_runtime_slow_tasks.append(
                {
                    "task_type": task_type,
                    "p95_latency_ms": local_runtime_p95,
                    "latency_target_ms": task_latency_limit if task_latency_limit is not None else UNKNOWN_VALUE_TEXT,
                    "call_count": stats.get("local_runtime_call_count", 0),
                    "evidence_level": "real_local_runtime",
                    "reason": "本地运行耗时超过当前 P95 目标，主要反映本机 CPU、模型加载、图片分辨率、小字号和版面复杂度，不代表云端 OCR 供应商 SLA。",
                }
            )

        if int(stats.get("mock_call_count", 0) or 0) > 0:
            mock_latency_unusable_tasks.append(
                {
                    "task_type": task_type,
                    "mock_call_count": stats.get("mock_call_count", 0),
                    "mock_p95_latency_ms": stats.get("mock_p95_latency_ms", UNKNOWN_VALUE_TEXT),
                    "evidence_level": "mock_placeholder",
                    "reason": "mock 延迟只能说明代码分支跑通，不能用于证明真实模型速度，也不能用于供应商选型。",
                }
            )

    root_cause_summary = _latency_root_cause_summary(
        real_api_slow_tasks=real_api_slow_tasks,
        local_runtime_slow_tasks=local_runtime_slow_tasks,
        mock_latency_unusable_tasks=mock_latency_unusable_tasks,
        has_latency_limit=has_latency_limit,
        has_latency_data=bool(task_latency_stats),
    )
    recommended_next_actions = _latency_next_actions(
        real_api_slow_tasks=real_api_slow_tasks,
        local_runtime_slow_tasks=local_runtime_slow_tasks,
        mock_latency_unusable_tasks=mock_latency_unusable_tasks,
        has_latency_limit=has_latency_limit,
        has_latency_data=bool(task_latency_stats),
    )

    return {
        "analysis_type": "routing_preflight_latency_bottleneck",
        "p95_latency_limit_ms": latency_limit if latency_limit is not None else UNKNOWN_VALUE_TEXT,
        "task_latency_targets_ms": normalized_task_targets or {},
        "bottleneck_status": _latency_bottleneck_status(
            real_api_slow_tasks=real_api_slow_tasks,
            local_runtime_slow_tasks=local_runtime_slow_tasks,
            mock_latency_unusable_tasks=mock_latency_unusable_tasks,
            has_latency_limit=has_latency_limit,
            has_latency_data=bool(task_latency_stats),
        ),
        "real_api_slow_tasks": real_api_slow_tasks,
        "local_runtime_slow_tasks": local_runtime_slow_tasks,
        "mock_latency_unusable_tasks": mock_latency_unusable_tasks,
        "top_latency_tasks": top_latency_tasks[:5],
        "root_cause_summary": root_cause_summary,
        "recommended_next_actions": recommended_next_actions,
    }


def _finite_optional_float(value: Any) -> float | None:
    """安全读取有限浮点数；未知、NaN 和 Infinity 都不参与延迟阈值判断。"""

    number = _as_optional_float(value)
    if number is None or not math.isfinite(number):
        return None
    return number


def _normalize_task_latency_targets(task_latency_targets_ms: dict[str, Any] | None) -> dict[str, float]:
    """把任务级 P95 延迟目标标准化为可比较的数字字典。"""

    if not isinstance(task_latency_targets_ms, dict):
        return {}
    normalized: dict[str, float] = {}
    for task_type, raw_limit in task_latency_targets_ms.items():
        limit = _finite_optional_float(raw_limit)
        if limit is not None and limit >= 0:
            normalized[str(task_type)] = limit
    return normalized


def _latency_limit_for_task(
    task_type: str,
    task_latency_targets_ms: dict[str, float],
    global_latency_limit_ms: float | None,
) -> float | None:
    """优先读取任务级 P95 目标；没有配置时使用全局目标兜底。"""

    if task_type in task_latency_targets_ms:
        return task_latency_targets_ms[task_type]
    return global_latency_limit_ms


def _safe_sort_latency(value: Any) -> float:
    """给延迟排序使用的安全数值。"""

    number = _finite_optional_float(value)
    return number if number is not None else 0.0


def _is_latency_over_limit(value: float, latency_limit: float | None) -> bool:
    """判断延迟是否超过目标；没有目标时不硬判失败。"""

    if latency_limit is None:
        return False
    return value > latency_limit


def _latency_bottleneck_status(
    *,
    real_api_slow_tasks: list[dict[str, Any]],
    local_runtime_slow_tasks: list[dict[str, Any]],
    mock_latency_unusable_tasks: list[dict[str, Any]],
    has_latency_limit: bool,
    has_latency_data: bool,
) -> str:
    """生成延迟归因状态。"""

    if not has_latency_data or not has_latency_limit:
        return "unknown"
    if real_api_slow_tasks or local_runtime_slow_tasks:
        return "fail"
    if mock_latency_unusable_tasks:
        return "warning"
    return "pass"


def _latency_root_cause_summary(
    *,
    real_api_slow_tasks: list[dict[str, Any]],
    local_runtime_slow_tasks: list[dict[str, Any]],
    mock_latency_unusable_tasks: list[dict[str, Any]],
    has_latency_limit: bool,
    has_latency_data: bool,
) -> list[str]:
    """生成延迟慢因摘要。"""

    if not has_latency_data:
        return ["当前没有可用历史延迟数据，无法判断慢因。"]
    if not has_latency_limit:
        return ["当前没有 P95 延迟目标，只能展示历史延迟，不能判断是否阻塞。"]

    summary: list[str] = []
    if local_runtime_slow_tasks:
        summary.append("本地 OCR 慢主要来自 PaddleOCR 本机推理链路，受 CPU、模型加载、图片分辨率、小字号、低对比度和版面复杂度影响。")
    if real_api_slow_tasks:
        summary.append("真实 API 慢主要来自外部网络请求和供应商模型处理链路，可作为受控样本证据，但样本量不足时不能外推为稳定 SLA。")
    if mock_latency_unusable_tasks:
        summary.append("mock 任务的 0ms 或极低延迟只说明占位分支执行成功，不能证明视觉理解、语音识别或其他供应商真实速度。")
    if not summary:
        summary.append("当前历史延迟没有超过 P95 目标的真实或本地运行任务，但仍需注意样本规模和运行环境差异。")
    return summary


def _latency_next_actions(
    *,
    real_api_slow_tasks: list[dict[str, Any]],
    local_runtime_slow_tasks: list[dict[str, Any]],
    mock_latency_unusable_tasks: list[dict[str, Any]],
    has_latency_limit: bool,
    has_latency_data: bool,
) -> list[str]:
    """根据慢因给出下一步动作，避免直接扩大运行。"""

    if not has_latency_data:
        return ["先提供历史 model_calls.jsonl 或做受控小样本试跑，再判断延迟瓶颈。"]
    if not has_latency_limit:
        return ["先明确本轮 P95 延迟目标，再判断是否阻塞。"]

    actions: list[str] = []
    if local_runtime_slow_tasks:
        actions.append("OCR 先继续小批量受控运行，单独记录本地 OCR 耗时；不要把本地 PaddleOCR 延迟当作云端 OCR 供应商结论。")
    if real_api_slow_tasks:
        actions.append("真实 API 任务单独做小样本延迟复测，控制提示词长度、输入规模和调用时间窗口，避免和 OCR 慢因混在一起。")
    if mock_latency_unusable_tasks:
        actions.append("mock 任务暂不参与供应商延迟判断；如果要比较视觉理解或语音识别速度，必须先接入真实模型并单独授权试跑。")
    if not actions:
        actions.append("当前可以继续小批量扩大，但每轮只增加一个变量，例如只增加图片数量或只放开一个真实模型。")
    return actions


def apply_backend_overrides(
    routing_rules: dict[str, dict[str, str]],
    *,
    ocr_backend: str | None = None,
    text_analysis_backend: str | None = None,
) -> dict[str, dict[str, str]]:
    """按主流程的显式后端选择生成预检查用路由，不修改原始配置。"""

    updated_rules = {task_type: dict(rule) for task_type, rule in routing_rules.items()}

    if ocr_backend is not None:
        if ocr_backend not in {"mock", "paddleocr"}:
            raise ValueError(f"不支持的 OCR 后端: {ocr_backend}")
        if "ocr" in updated_rules:
            updated_rules["ocr"] = (
                {"provider": "paddlepaddle", "model_name": "PP-OCRv5_mobile"}
                if ocr_backend == "paddleocr"
                else {"provider": "doubao", "model_name": "mock-ocr"}
            )

    if text_analysis_backend is not None:
        if text_analysis_backend not in {"mock", "deepseek"}:
            raise ValueError(f"不支持的文本分析后端: {text_analysis_backend}")
        for task_type in ("text_analysis", "summary_merge"):
            if task_type not in updated_rules:
                continue
            updated_rules[task_type] = (
                {"provider": "deepseek", "model_name": "deepseek-v4-flash"}
                if text_analysis_backend == "deepseek"
                else {"provider": "deepseek", "model_name": "mock-text"}
            )

    return updated_rules


def _filter_manifest_by_include_files(
    manifest: list[dict[str, Any]],
    include_files: list[str] | tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    """按文件名过滤清单，用于只预检查指定输入文件。"""

    if not include_files:
        return manifest
    include_set = {item.strip() for item in include_files if item.strip()}
    available_names = {str(record["file_name"]) for record in manifest}
    missing_names = sorted(include_set - available_names)
    if missing_names:
        raise FileNotFoundError(f"include_files 中存在输入目录里找不到的文件: {', '.join(missing_names)}")
    return [record for record in manifest if str(record["file_name"]) in include_set]


def _read_text_char_count(source_path: str | Path) -> int:
    """读取文本文件字符数；无法按 UTF-8 解码时用忽略错误方式兜底。"""

    try:
        return len(Path(source_path).read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return len(Path(source_path).read_text(encoding="utf-8", errors="ignore"))


def _estimate_tokens_from_char_count(char_count: int) -> int:
    """用轻量近似把字符数转换成 token 数，避免运行前调用 tokenizer。"""

    if char_count <= 0:
        return 0
    return max(1, math.ceil(char_count / 2))


def _workload_profile_warnings(
    *,
    video_file_count: int,
    expected_audio_seconds_per_video: int | None,
    text_file_count: int,
    image_file_count: int,
    total_files: int,
) -> list[str]:
    """生成运行前规模画像的风险提示。"""

    warnings: list[str] = []
    if total_files == 0:
        warnings.append("输入目录中没有识别到支持的文本、图片或视频文件，预检查无法代表真实批处理。")
    if video_file_count > 0 and expected_audio_seconds_per_video is None:
        warnings.append("存在视频文件，但未提供 expected_audio_seconds_per_video，语音识别成本仍无法估算。")
    if image_file_count > 0:
        warnings.append("图片 OCR 与视觉理解的单位按图片张数估算；真实耗时还会受分辨率、小字号和版面复杂度影响。")
    if text_file_count > 0:
        warnings.append("文本 token 数使用字符数粗估；真实供应商计费 token 可能存在偏差。")
    return warnings


def _expected_task_types_from_workload_profile(workload_profile: dict[str, Any] | None) -> list[str] | None:
    """根据本批次输入类型推导实际会触发的任务类型。"""

    if workload_profile is None:
        return None

    media_counts = workload_profile.get("media_type_counts") or {}
    text_count = int(media_counts.get("text", 0) or 0)
    image_count = int(media_counts.get("image", 0) or 0)
    video_count = int(media_counts.get("video", 0) or 0)
    total_files = int(workload_profile.get("total_files", text_count + image_count + video_count) or 0)

    task_types: list[str] = []
    if image_count > 0 or video_count > 0:
        task_types.extend(["ocr", "visual_understanding"])
    if video_count > 0:
        task_types.append("speech_to_text")
    if total_files > 0 or text_count > 0:
        task_types.append("text_analysis")
    return task_types


def _has_positive_units(expected_units: Any) -> bool:
    """判断显式预估用量是否表示任务确实会执行。"""

    for unit in _normalize_units(expected_units):
        quantity = _find_unit_quantity([unit], str(unit.get("unit_type", "")))
        if quantity is not None and quantity > 0:
            return True
    return False


def build_preflight_report(
    *,
    routing_rules: dict[str, dict[str, str]],
    model_prices: dict[str, dict[str, Any]],
    policy_name: str = "balanced",
    policy_overrides: dict[str, Any] | None = None,
    expected_task_types: list[str] | tuple[str, ...] | None = None,
    expected_units_by_task: dict[str, Any] | None = None,
    historical_p95_latency_by_task_ms: dict[str, Any] | None = None,
    workload_profile: dict[str, Any] | None = None,
    latency_profile: dict[str, Any] | None = None,
    generated_at: str | None = None,
    max_price_age_days: int = DEFAULT_MAX_PRICE_AGE_DAYS,
    source_files: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成路由预检查报告。"""

    report_generated_at = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    constraints = build_constraints(policy_name, policy_overrides)
    units_by_task = expected_units_by_task or {}
    if expected_task_types is not None:
        task_types = list(expected_task_types)
    else:
        workload_task_types = _expected_task_types_from_workload_profile(workload_profile)
        task_types = workload_task_types if workload_task_types is not None else list(DEFAULT_EXPECTED_TASK_TYPES)
        for task_type, expected_units in units_by_task.items():
            if task_type not in task_types and _has_positive_units(expected_units):
                task_types.append(task_type)
    latency_by_task = historical_p95_latency_by_task_ms or {}

    current_route = [
        _build_route_entry(
            task_type,
            routing_rules,
            model_prices,
            expected_units=units_by_task.get(task_type),
            expected_p95_latency_ms=latency_by_task.get(task_type),
        )
        for task_type in task_types
    ]
    task_latency_target_checks = _build_task_latency_target_checks(current_route, constraints)
    route_summary = _summarize_route(
        current_route,
        task_latency_target_checks=task_latency_target_checks,
        task_latency_targets_ms=constraints.get("task_latency_targets_ms"),
    )
    price_catalog_profile = build_price_catalog_profile(
        current_route,
        generated_at=report_generated_at,
        max_price_age_days=max_price_age_days,
    )
    constraint_checks = _build_constraint_checks(route_summary, constraints)
    preflight_status = _overall_preflight_status(current_route, constraint_checks, price_catalog_profile)
    latency_bottleneck_analysis = build_latency_bottleneck_analysis(
        latency_profile,
        p95_latency_limit_ms=constraints.get("p95_latency_limit_ms"),
        task_latency_targets_ms=constraints.get("task_latency_targets_ms"),
        expected_task_types=task_types,
    )
    controlled_trial_plan = _build_controlled_trial_plan(
        preflight_status=preflight_status,
        route_summary=route_summary,
        constraint_checks=constraint_checks,
        workload_profile=workload_profile,
        latency_profile=latency_profile,
    )

    return {
        "schema_version": "v1",
        "report_type": "routing_preflight",
        "policy_name": policy_name,
        "generated_at": report_generated_at,
        "source_files": source_files or {},
        "expected_task_types": task_types,
        "workload_profile": workload_profile,
        "latency_profile": latency_profile,
        "latency_bottleneck_analysis": latency_bottleneck_analysis,
        "task_latency_target_checks": task_latency_target_checks,
        "price_catalog_profile": price_catalog_profile,
        "constraints": constraints,
        "current_route": current_route,
        "route_summary": route_summary,
        "constraint_checks": constraint_checks,
        "preflight_status": preflight_status,
        "blocking_reasons": _blocking_reasons(current_route, constraint_checks),
        "warning_messages": _warning_messages(current_route, constraint_checks, price_catalog_profile),
        "estimated_cost_scope": _estimated_cost_scope(route_summary),
        "controlled_trial_plan": controlled_trial_plan,
        "recommended_action": _recommended_action(preflight_status, route_summary, constraint_checks),
        "boundary_notes": _boundary_notes(),
        "field_notes": _field_notes(),
    }


def build_preflight_from_files(
    *,
    routing_rules_path: str | Path,
    model_prices_path: str | Path,
    policy_config_path: str | Path | None = None,
    policy_name: str = "balanced",
    expected_task_types: list[str] | tuple[str, ...] | None = None,
    expected_units_by_task: dict[str, Any] | None = None,
    historical_p95_latency_by_task_ms: dict[str, Any] | None = None,
    historical_model_calls_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    input_dir: str | Path | None = None,
    include_files: list[str] | tuple[str, ...] | None = None,
    expected_frames_per_video: int = 3,
    expected_audio_seconds_per_video: int | None = None,
    expected_output_tokens_per_file: int = 300,
    estimated_evidence_tokens_per_image: int = 300,
    estimated_evidence_tokens_per_video: int = 800,
    policy_overrides: dict[str, Any] | None = None,
    ocr_backend: str | None = None,
    text_analysis_backend: str | None = None,
    max_price_age_days: int = DEFAULT_MAX_PRICE_AGE_DAYS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """从本地配置文件生成预检查报告。"""

    routing_rules_file = Path(routing_rules_path)
    model_prices_file = Path(model_prices_path)
    routing_rules = load_routing_rules(routing_rules_file)
    model_prices = load_model_prices(model_prices_file)
    routing_rules = apply_backend_overrides(
        routing_rules,
        ocr_backend=ocr_backend,
        text_analysis_backend=text_analysis_backend,
    )

    loaded_policy_overrides = None
    policy_config_source = "code_defaults"
    if policy_config_path is not None:
        policy_config = load_policy_config(policy_config_path)
        loaded_policy_overrides = policy_config["policy_overrides"].get(policy_name)
        policy_config_source = policy_config["config_source"]
    merged_policy_overrides = dict(loaded_policy_overrides or {})
    if policy_overrides:
        merged_policy_overrides.update({key: value for key, value in policy_overrides.items() if value is not None})

    workload_profile = None
    merged_expected_units = expected_units_by_task
    if input_dir is not None:
        workload_profile = build_workload_profile(
            input_dir,
            include_files=include_files,
            expected_frames_per_video=expected_frames_per_video,
            expected_audio_seconds_per_video=expected_audio_seconds_per_video,
            expected_output_tokens_per_file=expected_output_tokens_per_file,
            estimated_evidence_tokens_per_image=estimated_evidence_tokens_per_image,
            estimated_evidence_tokens_per_video=estimated_evidence_tokens_per_video,
            generated_at=generated_at,
        )
        if merged_expected_units is None:
            merged_expected_units = workload_profile["expected_units_by_task"]

    latency_profile = None
    merged_historical_p95 = historical_p95_latency_by_task_ms
    if historical_model_calls_paths:
        latency_profile = build_historical_latency_profile(
            list(historical_model_calls_paths),
            generated_at=generated_at,
        )
        if merged_historical_p95 is None:
            merged_historical_p95 = latency_profile["historical_p95_latency_by_task_ms"]

    return build_preflight_report(
        routing_rules=routing_rules,
        model_prices=model_prices,
        policy_name=policy_name,
        policy_overrides=merged_policy_overrides,
        expected_task_types=expected_task_types,
        expected_units_by_task=merged_expected_units,
        historical_p95_latency_by_task_ms=merged_historical_p95,
        workload_profile=workload_profile,
        latency_profile=latency_profile,
        generated_at=generated_at,
        max_price_age_days=max_price_age_days,
        source_files={
            "routing_rules": str(routing_rules_file),
            "model_prices": str(model_prices_file),
            "policy_config": policy_config_source,
            "historical_model_calls": [str(Path(path)) for path in historical_model_calls_paths or []],
        },
    )


def write_preflight_reports(output_dir: str | Path, report: dict[str, Any]) -> dict[str, str]:
    """写入 JSON 和 Markdown 两种预检查报告。"""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "routing_preflight_report.json"
    markdown_path = path / "routing_preflight_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_preflight_markdown(report), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }


def _build_route_entry(
    task_type: str,
    routing_rules: dict[str, dict[str, str]],
    model_prices: dict[str, dict[str, Any]],
    *,
    expected_units: Any = None,
    expected_p95_latency_ms: Any = None,
) -> dict[str, Any]:
    """生成单个任务类型的预检查条目。"""

    if task_type not in routing_rules:
        return {
            "task_type": task_type,
            "provider": UNKNOWN_VALUE_TEXT,
            "model_name": UNKNOWN_VALUE_TEXT,
            "route_status": "missing",
            "is_mock": UNKNOWN_VALUE_TEXT,
            "price_status": "unknown",
            "price_source": UNKNOWN_VALUE_TEXT,
            "price_updated_at": UNKNOWN_VALUE_TEXT,
            "price_confidence": UNKNOWN_VALUE_TEXT,
            "pricing_summary": UNKNOWN_VALUE_TEXT,
            "estimated_cost_cny": UNKNOWN_VALUE_TEXT,
            "expected_p95_latency_ms": UNKNOWN_VALUE_TEXT,
            "latency_status": "unknown",
            "risk_notes": ["当前任务类型没有路由规则，批处理运行时无法选择模型。"],
        }

    rule = routing_rules[task_type]
    provider = rule.get("provider", UNKNOWN_VALUE_TEXT)
    model_name = rule.get("model_name", UNKNOWN_VALUE_TEXT)
    price_rule = model_prices.get(model_name)
    estimated_cost = _estimate_cost(price_rule, expected_units)
    latency_status = "known" if expected_p95_latency_ms is not None else "unknown"

    return {
        "task_type": task_type,
        "provider": provider,
        "model_name": model_name,
        "route_status": "configured",
        "is_mock": is_mock_model(model_name),
        "price_status": "known" if price_rule else "unknown",
        "price_source": price_rule.get("price_source", UNKNOWN_VALUE_TEXT) if price_rule else UNKNOWN_VALUE_TEXT,
        "price_updated_at": price_rule.get("price_updated_at", UNKNOWN_VALUE_TEXT) if price_rule else UNKNOWN_VALUE_TEXT,
        "price_confidence": price_rule.get("price_confidence", UNKNOWN_VALUE_TEXT) if price_rule else UNKNOWN_VALUE_TEXT,
        "pricing_summary": _pricing_summary(price_rule),
        "estimated_cost_cny": estimated_cost,
        "expected_p95_latency_ms": expected_p95_latency_ms if expected_p95_latency_ms is not None else UNKNOWN_VALUE_TEXT,
        "latency_status": latency_status,
        "risk_notes": _route_risk_notes(task_type, model_name, price_rule, expected_units, expected_p95_latency_ms),
    }


def _estimate_cost(price_rule: dict[str, Any] | None, expected_units: Any) -> float | str:
    """根据可选预估用量计算成本；缺少用量时不硬算。"""

    if not price_rule:
        return UNKNOWN_VALUE_TEXT
    normalized_units = _normalize_units(expected_units)
    if not normalized_units:
        return UNKNOWN_VALUE_TEXT

    total_cost = 0.0
    if "pricing_rules" in price_rule:
        for rule in price_rule["pricing_rules"]:
            quantity = _find_unit_quantity(normalized_units, str(rule["unit_type"]))
            if quantity is None:
                return UNKNOWN_VALUE_TEXT
            total_cost += quantity * float(rule["price_cny_per_unit"])
        return round(total_cost, 6)

    unit_type = str(price_rule.get("pricing_unit"))
    quantity = _find_unit_quantity(normalized_units, unit_type)
    if quantity is None:
        return UNKNOWN_VALUE_TEXT
    return round(quantity * float(price_rule["price_cny_per_unit"]), 6)


def _normalize_units(expected_units: Any) -> list[dict[str, Any]]:
    """把调用方传入的预估用量统一成列表结构。"""

    if expected_units is None:
        return []
    if isinstance(expected_units, dict):
        return [expected_units]
    if isinstance(expected_units, list):
        return [unit for unit in expected_units if isinstance(unit, dict)]
    return []


def _find_unit_quantity(units: list[dict[str, Any]], unit_type: str) -> float | None:
    """从预估用量中找到指定单位的数量。"""

    for unit in units:
        if unit.get("unit_type") == unit_type:
            try:
                return float(unit["quantity"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def _pricing_summary(price_rule: dict[str, Any] | None) -> str:
    """生成价格规则摘要。"""

    if not price_rule:
        return UNKNOWN_VALUE_TEXT
    if "pricing_rules" in price_rule:
        parts = [
            f"{rule['unit_type']}={rule['price_cny_per_unit']}元/单位"
            for rule in price_rule["pricing_rules"]
        ]
        return "；".join(parts)
    return f"{price_rule['pricing_unit']}={price_rule['price_cny_per_unit']}元/单位"


def _route_risk_notes(
    task_type: str,
    model_name: str,
    price_rule: dict[str, Any] | None,
    expected_units: Any,
    expected_p95_latency_ms: Any,
) -> list[str]:
    """生成单个任务路由的风险说明。"""

    notes: list[str] = []
    if is_mock_model(model_name):
        notes.append("当前路由仍是 mock，只能证明流程可走通，不能证明真实供应商质量、成本或延迟。")
    if price_rule is None:
        notes.append("当前价格表缺少该模型，无法进行预算预检查。")
    elif not _normalize_units(expected_units):
        notes.append("当前没有提供本任务的预估用量，只能展示单位价格，不能估算总成本。")
    if expected_p95_latency_ms is None:
        notes.append("当前没有提供本任务的历史或目标前估 P95 延迟，不能判断延迟约束。")
    if task_type in {"ocr", "visual_understanding", "speech_to_text"} and is_mock_model(model_name):
        notes.append("该上游证据提取任务仍是多模态质量瓶颈，不能把下游文本分析结果解释为完整真实多模态能力。")
    return notes


def build_price_catalog_profile(
    current_route: list[dict[str, Any]],
    *,
    generated_at: str | None = None,
    max_price_age_days: int = DEFAULT_MAX_PRICE_AGE_DAYS,
) -> dict[str, Any]:
    """生成路由预检查中的价格目录画像，不联网刷新价格。"""

    report_date = _parse_report_date(generated_at)
    checked_items = []
    warning_messages: list[str] = []

    for entry in current_route:
        if entry["route_status"] != "configured":
            continue
        item = _build_price_catalog_item(
            entry,
            report_date=report_date,
            max_price_age_days=max_price_age_days,
        )
        checked_items.append(item)

    stale_models = _unique_model_names(
        item
        for item in checked_items
        if item["price_freshness_status"] in {"stale", "missing_updated_at", "invalid_updated_at", "future_updated_at"}
    )
    untrusted_models = _unique_model_names(
        item
        for item in checked_items
        if item["price_confidence_status"] in {"unknown", "unverified"}
    )
    missing_price_models = _unique_model_names(item for item in checked_items if item["price_status"] == "unknown")

    if stale_models:
        warning_messages.append(
            f"以下模型的价格更新时间缺失、异常或超过 {max_price_age_days} 天：{', '.join(stale_models)}；成本预估前建议先刷新价格目录。"
        )
    if untrusted_models:
        warning_messages.append(
            f"以下模型的价格可信度不足：{', '.join(untrusted_models)}；成本结论只能作为工程估算，不能当作已验证扣费。"
        )
    if missing_price_models:
        warning_messages.append(
            f"以下模型缺少价格规则：{', '.join(missing_price_models)}；对应任务无法参与预算估算。"
        )

    price_catalog_status = "warning" if warning_messages else "pass"
    return {
        "profile_type": "routing_preflight_price_catalog",
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "max_price_age_days": max_price_age_days,
        "price_catalog_status": price_catalog_status,
        "checked_model_count": len(_unique_model_names(checked_items)),
        "stale_model_names": stale_models,
        "untrusted_model_names": untrusted_models,
        "missing_price_model_names": missing_price_models,
        "checked_items": checked_items,
        "warning_messages": warning_messages,
    }


def _build_price_catalog_item(
    entry: dict[str, Any],
    *,
    report_date: date,
    max_price_age_days: int,
) -> dict[str, Any]:
    """生成单个路由模型的价格目录检查项。"""

    price_updated_at = entry.get("price_updated_at")
    parsed_date = _parse_price_date(price_updated_at)
    age_days: int | str
    if parsed_date is None:
        age_days = UNKNOWN_VALUE_TEXT
        freshness_status = "missing_updated_at" if price_updated_at == UNKNOWN_VALUE_TEXT else "invalid_updated_at"
    else:
        age_days = (report_date - parsed_date).days
        if age_days < 0:
            freshness_status = "future_updated_at"
        elif age_days > max_price_age_days:
            freshness_status = "stale"
        else:
            freshness_status = "fresh"

    confidence = str(entry.get("price_confidence") or UNKNOWN_VALUE_TEXT)
    if confidence in TRUSTED_PRICE_CONFIDENCES:
        confidence_status = "trusted"
    elif confidence in EXPLAINABLE_PRICE_CONFIDENCES:
        confidence_status = "explainable"
    elif confidence in {UNKNOWN_VALUE_TEXT, "unknown", ""}:
        confidence_status = "unknown"
    else:
        confidence_status = "unverified"

    return {
        "task_type": entry["task_type"],
        "provider": entry["provider"],
        "model_name": entry["model_name"],
        "is_mock": entry["is_mock"],
        "price_status": entry["price_status"],
        "price_source": entry.get("price_source", UNKNOWN_VALUE_TEXT),
        "price_updated_at": price_updated_at,
        "price_age_days": age_days,
        "price_freshness_status": freshness_status,
        "price_confidence": entry.get("price_confidence", UNKNOWN_VALUE_TEXT),
        "price_confidence_status": confidence_status,
    }


def _parse_report_date(generated_at: str | None) -> date:
    """解析报告日期；解析失败时使用当前日期。"""

    if not generated_at:
        return datetime.now().astimezone().date()
    try:
        return datetime.fromisoformat(generated_at).date()
    except ValueError:
        return datetime.now().astimezone().date()


def _parse_price_date(value: Any) -> date | None:
    """解析价格更新时间；无法解析时返回 None。"""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value in {None, UNKNOWN_VALUE_TEXT, ""}:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def _unique_model_names(items: Any) -> list[str]:
    """从检查项中提取去重后的模型名称。"""

    names = {
        str(item.get("model_name"))
        for item in items
        if item.get("model_name") and item.get("model_name") != UNKNOWN_VALUE_TEXT
    }
    return sorted(names)


def _build_task_latency_target_checks(
    current_route: list[dict[str, Any]],
    constraints: dict[str, Any],
) -> list[dict[str, Any]]:
    """按任务类型检查 P95 延迟目标；没有任务级目标时使用全局目标兜底。"""

    global_latency_limit = _finite_optional_float(constraints.get("p95_latency_limit_ms"))
    task_latency_targets = _normalize_task_latency_targets(constraints.get("task_latency_targets_ms"))
    checks: list[dict[str, Any]] = []

    for entry in current_route:
        if entry.get("route_status") != "configured":
            continue
        task_type = str(entry["task_type"])
        observed = _finite_optional_float(entry.get("expected_p95_latency_ms"))
        limit = _latency_limit_for_task(task_type, task_latency_targets, global_latency_limit)
        target_source = "task_specific" if task_type in task_latency_targets else "global_fallback"
        is_mock_route = entry.get("is_mock") is True
        if is_mock_route:
            evidence_level = "mock_placeholder"
        elif entry.get("price_confidence") == "local_external_api_zero":
            evidence_level = "local_runtime_history"
        else:
            evidence_level = "non_mock_history"

        if limit is None:
            checks.append(
                {
                    "task_type": task_type,
                    "observed_p95_latency_ms": observed if observed is not None else UNKNOWN_VALUE_TEXT,
                    "target_p95_latency_ms": UNKNOWN_VALUE_TEXT,
                    "target_source": "missing",
                    "evidence_level": evidence_level,
                    "status": "unknown",
                    "target_ratio": UNKNOWN_VALUE_TEXT,
                    "reason": "当前没有任务级或全局 P95 延迟目标，不能判断该任务延迟是否满足约束。",
                }
            )
            continue

        if observed is None:
            checks.append(
                {
                    "task_type": task_type,
                    "observed_p95_latency_ms": UNKNOWN_VALUE_TEXT,
                    "target_p95_latency_ms": limit,
                    "target_source": target_source,
                    "evidence_level": evidence_level,
                    "status": "unknown",
                    "target_ratio": UNKNOWN_VALUE_TEXT,
                    "reason": "当前缺少该任务的历史 P95 延迟，不能判断是否满足任务级目标。",
                }
            )
            continue

        passed = observed <= limit
        status = "warning" if is_mock_route else ("pass" if passed else "fail")
        if is_mock_route:
            reason = "当前观察延迟来自 mock 占位，不能证明真实供应商 P95 延迟满足目标；可继续受控试跑，但不能作为真实性能结论。"
        elif passed:
            reason = "满足任务级 P95 延迟目标。"
        else:
            reason = "超过任务级 P95 延迟目标，扩大运行前需要处理。"
        checks.append(
            {
                "task_type": task_type,
                "observed_p95_latency_ms": round(observed, 6),
                "target_p95_latency_ms": round(limit, 6),
                "target_source": target_source,
                "evidence_level": evidence_level,
                "status": status,
                "target_ratio": round(observed / limit, 6) if limit > 0 else UNKNOWN_VALUE_TEXT,
                "reason": reason,
            }
        )

    return checks


def _summarize_task_latency_target_checks(
    checks: list[dict[str, Any]],
    task_latency_targets_ms: Any,
) -> dict[str, Any]:
    """汇总任务级延迟目标检查结果。"""

    statuses = {str(check["status"]) for check in checks}
    if "fail" in statuses:
        overall_status = "fail"
    elif "unknown" in statuses:
        overall_status = "unknown"
    elif "warning" in statuses:
        overall_status = "warning"
    else:
        overall_status = "pass"

    known_ratios = [
        float(check["target_ratio"])
        for check in checks
        if isinstance(check.get("target_ratio"), (int, float))
    ]
    return {
        "target_mode": "task_specific" if isinstance(task_latency_targets_ms, dict) and task_latency_targets_ms else "global_fallback",
        "overall_status": overall_status,
        "failed_task_types": [check["task_type"] for check in checks if check["status"] == "fail"],
        "unknown_task_types": [check["task_type"] for check in checks if check["status"] == "unknown"],
        "warning_task_types": [check["task_type"] for check in checks if check["status"] == "warning"],
        "max_target_ratio": round(max(known_ratios), 6) if known_ratios else UNKNOWN_VALUE_TEXT,
    }


def _summarize_route(
    current_route: list[dict[str, Any]],
    *,
    task_latency_target_checks: list[dict[str, Any]] | None = None,
    task_latency_targets_ms: Any = None,
) -> dict[str, Any]:
    """汇总当前路由结构。"""

    configured = [entry for entry in current_route if entry["route_status"] == "configured"]
    missing = [entry for entry in current_route if entry["route_status"] == "missing"]
    mock_entries = [entry for entry in configured if entry["is_mock"] is True]
    real_entries = [entry for entry in configured if entry["is_mock"] is False]
    known_costs = [entry["estimated_cost_cny"] for entry in configured if isinstance(entry["estimated_cost_cny"], (int, float))]
    unknown_cost_tasks = [entry["task_type"] for entry in configured if not isinstance(entry["estimated_cost_cny"], (int, float))]
    known_latencies = [
        float(entry["expected_p95_latency_ms"])
        for entry in configured
        if isinstance(entry["expected_p95_latency_ms"], (int, float))
    ]

    total_expected_tasks = len(current_route)
    real_coverage_rate = round(len(real_entries) / total_expected_tasks, 6) if total_expected_tasks else 0.0
    mock_coverage_rate = round(len(mock_entries) / total_expected_tasks, 6) if total_expected_tasks else 0.0

    return {
        "total_expected_task_types": total_expected_tasks,
        "configured_task_types": [entry["task_type"] for entry in configured],
        "missing_task_types": [entry["task_type"] for entry in missing],
        "real_task_types": [entry["task_type"] for entry in real_entries],
        "mock_task_types": [entry["task_type"] for entry in mock_entries],
        "real_coverage_rate": real_coverage_rate,
        "mock_coverage_rate": mock_coverage_rate,
        "estimated_total_cost_cny": round(sum(known_costs), 6) if known_costs and not unknown_cost_tasks else UNKNOWN_VALUE_TEXT,
        "cost_unknown_task_types": unknown_cost_tasks,
        "max_expected_p95_latency_ms": max(known_latencies) if known_latencies else UNKNOWN_VALUE_TEXT,
        "latency_unknown_task_types": [
            entry["task_type"]
            for entry in configured
            if not isinstance(entry["expected_p95_latency_ms"], (int, float))
        ],
        "task_latency_target_summary": _summarize_task_latency_target_checks(
            task_latency_target_checks or [],
            task_latency_targets_ms,
        ),
    }


def _build_constraint_checks(route_summary: dict[str, Any], constraints: dict[str, Any]) -> list[dict[str, Any]]:
    """基于路由汇总结果生成约束检查项。"""

    if constraints.get("task_latency_targets_ms"):
        latency_constraint_check = _task_latency_constraint_check(route_summary)
    else:
        latency_constraint_check = _constraint_check(
            "p95_latency_limit_ms",
            observed_value=route_summary["max_expected_p95_latency_ms"],
            limit_value=constraints.get("p95_latency_limit_ms"),
            pass_when_less_or_equal=True,
        )

    return [
        _constraint_check(
            "routing_rules_complete",
            observed_value=len(route_summary["missing_task_types"]),
            limit_value=0,
            pass_when_less_or_equal=True,
            unknown_when_limit_missing=False,
        ),
        _constraint_check(
            "budget_limit_cny",
            observed_value=route_summary["estimated_total_cost_cny"],
            limit_value=constraints.get("budget_limit_cny"),
            pass_when_less_or_equal=True,
        ),
        latency_constraint_check,
        _constraint_check(
            "min_real_coverage_rate",
            observed_value=route_summary["real_coverage_rate"],
            limit_value=constraints.get("min_real_coverage_rate"),
            pass_when_less_or_equal=False,
        ),
    ]


def _task_latency_constraint_check(route_summary: dict[str, Any]) -> dict[str, Any]:
    """把任务级 P95 延迟检查汇总成整体延迟约束。"""

    summary = route_summary.get("task_latency_target_summary", {})
    status = str(summary.get("overall_status", "unknown"))
    failed_tasks = summary.get("failed_task_types", [])
    unknown_tasks = summary.get("unknown_task_types", [])
    warning_tasks = summary.get("warning_task_types", [])
    max_target_ratio = summary.get("max_target_ratio", UNKNOWN_VALUE_TEXT)

    if status == "fail":
        reason = f"以下任务超过各自 P95 延迟目标：{', '.join(failed_tasks)}。"
    elif status == "unknown":
        reason = f"以下任务缺少历史 P95 或延迟目标：{', '.join(unknown_tasks)}。"
    elif status == "warning":
        reason = f"以下任务只有 mock 延迟证据，不能证明真实供应商 P95 延迟达标：{', '.join(warning_tasks)}。"
    else:
        reason = "当前所有已配置任务都满足各自 P95 延迟目标；任务级目标优先于全局兜底目标。"

    return {
        "constraint_name": "p95_latency_limit_ms",
        "observed_value": max_target_ratio,
        "limit_value": "task_latency_targets_ms",
        "status": status,
        "reason": reason,
    }


def _constraint_check(
    constraint_name: str,
    *,
    observed_value: Any,
    limit_value: Any,
    pass_when_less_or_equal: bool,
    unknown_when_limit_missing: bool = True,
) -> dict[str, Any]:
    """生成单项约束判断。"""

    if limit_value is None and unknown_when_limit_missing:
        return {
            "constraint_name": constraint_name,
            "observed_value": observed_value,
            "limit_value": UNKNOWN_VALUE_TEXT,
            "status": "unknown",
            "reason": "当前策略没有提供该约束上限或下限，因此不做硬性判断。",
        }
    if observed_value == UNKNOWN_VALUE_TEXT:
        return {
            "constraint_name": constraint_name,
            "observed_value": UNKNOWN_VALUE_TEXT,
            "limit_value": limit_value if limit_value is not None else UNKNOWN_VALUE_TEXT,
            "status": "unknown",
            "reason": "当前数据未提供，不能判断该约束是否满足。",
        }

    observed = float(observed_value)
    limit = float(limit_value)
    passed = observed <= limit if pass_when_less_or_equal else observed >= limit
    return {
        "constraint_name": constraint_name,
        "observed_value": round(observed, 6),
        "limit_value": round(limit, 6),
        "status": "pass" if passed else "fail",
        "reason": "满足约束。" if passed else "未满足约束，运行前需要处理。",
    }


def _overall_preflight_status(
    current_route: list[dict[str, Any]],
    constraint_checks: list[dict[str, Any]],
    price_catalog_profile: dict[str, Any] | None = None,
) -> str:
    """生成整体预检查状态。"""

    statuses = {check["status"] for check in constraint_checks}
    if "fail" in statuses:
        return "fail"
    if "warning" in statuses:
        return "warning"
    if price_catalog_profile and price_catalog_profile.get("price_catalog_status") == "warning":
        return "warning"
    if any(entry["is_mock"] is True for entry in current_route):
        return "warning"
    if "unknown" in statuses:
        return "warning"
    return "pass"


def _blocking_reasons(current_route: list[dict[str, Any]], constraint_checks: list[dict[str, Any]]) -> list[str]:
    """汇总阻塞继续运行的原因。"""

    reasons = []
    for entry in current_route:
        if entry["route_status"] == "missing":
            reasons.append(f"{entry['task_type']} 缺少路由规则。")
    for check in constraint_checks:
        if check["status"] == "fail":
            reasons.append(f"{check['constraint_name']} 未满足：{check['reason']}")
    return reasons


def _warning_messages(
    current_route: list[dict[str, Any]],
    constraint_checks: list[dict[str, Any]],
    price_catalog_profile: dict[str, Any] | None = None,
) -> list[str]:
    """汇总可以继续试跑但必须解释的风险。"""

    warnings: list[str] = []
    mock_task_types = [entry["task_type"] for entry in current_route if entry["is_mock"] is True]
    if mock_task_types:
        warnings.append(f"当前仍有 mock 任务：{', '.join(mock_task_types)}。这些任务不能证明真实模型能力。")
    if price_catalog_profile:
        warnings.extend(price_catalog_profile.get("warning_messages", []))
    unknown_checks = [check["constraint_name"] for check in constraint_checks if check["status"] == "unknown"]
    if unknown_checks:
        warnings.append(f"以下约束因为数据不足无法判断：{', '.join(unknown_checks)}。")
    warning_checks = [check["constraint_name"] for check in constraint_checks if check["status"] == "warning"]
    if warning_checks:
        warnings.append(f"以下约束存在非阻塞风险，不能解释成真实达标结论：{', '.join(warning_checks)}。")
    return warnings


def _estimated_cost_scope(route_summary: dict[str, Any]) -> str:
    """说明成本预估的可信范围。"""

    if route_summary["estimated_total_cost_cny"] == UNKNOWN_VALUE_TEXT:
        return "当前只能检查单位价格和缺失项；由于缺少部分任务的预估用量，不能估算整批总成本。"
    return "当前已基于传入的预估用量和本地价格表计算整批预估成本；该结果仍不是供应商真实账单。"


def _find_constraint_status(constraint_checks: list[dict[str, Any]], constraint_name: str) -> str:
    """查找某个约束的状态；找不到时返回 unknown。"""

    for check in constraint_checks:
        if check["constraint_name"] == constraint_name:
            return str(check["status"])
    return "unknown"


def _select_trial_files(workload_profile: dict[str, Any] | None, max_total_files: int = 3) -> list[str]:
    """从运行前规模画像中选出少量文件名，用于受控试跑建议。"""

    if not workload_profile:
        return []
    files = workload_profile.get("files") or []
    selected: list[str] = []

    def add_by_media_type(media_type: str, limit: int) -> None:
        candidates = [
            str(item.get("file_name"))
            for item in files
            if item.get("media_type") == media_type and item.get("file_name")
        ]
        for file_name in sorted(candidates)[:limit]:
            if len(selected) >= max_total_files:
                return
            selected.append(file_name)

    add_by_media_type("text", 1)
    add_by_media_type("image", 2)
    return selected


def _format_trial_include_files(selected_files: list[str]) -> str:
    """把受控试跑文件名转换成命令行参数片段。"""

    if not selected_files:
        return ""
    return f" --include-files {','.join(selected_files)}"


def _build_trial_commands(selected_files: list[str]) -> list[dict[str, Any]]:
    """生成受控试跑命令建议；这里只生成文本，不执行命令。"""

    include_arg = _format_trial_include_files(selected_files)
    text_only_files = [file_name for file_name in selected_files if Path(file_name).suffix.lower() in {".txt", ".md"}]
    text_include_arg = _format_trial_include_files(text_only_files)

    commands = [
        {
            "command_name": "offline_mock_trial",
            "requires_live_api": False,
            "command": (
                "python .\\src\\main.py --input-dir .\\input"
                f"{include_arg} --ocr-backend mock --text-analysis-backend mock --batch-id batch_controlled_mock_trial"
            ),
            "purpose": "先验证指定文件范围、文件分流、结果写入和报告生成，不触发真实模型。",
        },
        {
            "command_name": "local_ocr_trial",
            "requires_live_api": False,
            "command": (
                "python .\\src\\main.py --input-dir .\\input"
                f"{include_arg} --ocr-backend paddleocr --text-analysis-backend mock --batch-id batch_controlled_paddleocr_trial"
            ),
            "purpose": "只放开本地 PaddleOCR，观察 OCR 延迟和错误状态，不触发 DeepSeek API。",
        },
    ]
    if text_only_files:
        commands.append(
            {
                "command_name": "deepseek_text_trial",
                "requires_live_api": True,
                "command": (
                    "python .\\src\\main.py --input-dir .\\input"
                    f"{text_include_arg} --ocr-backend mock --text-analysis-backend deepseek --allow-live-api "
                    "--batch-id batch_controlled_deepseek_text_trial"
                ),
                "purpose": "只用少量文本文件验证 DeepSeek 文本分析延迟；必须单独授权 API 调用。",
            }
        )
    return commands


def _build_controlled_trial_plan(
    *,
    preflight_status: str,
    route_summary: dict[str, Any],
    constraint_checks: list[dict[str, Any]],
    workload_profile: dict[str, Any] | None,
    latency_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """基于预检查结果生成受控小样本试跑建议。"""

    budget_status = _find_constraint_status(constraint_checks, "budget_limit_cny")
    latency_status = _find_constraint_status(constraint_checks, "p95_latency_limit_ms")
    selected_files = _select_trial_files(workload_profile)
    total_files = int(workload_profile.get("total_files", 0)) if workload_profile else 0
    media_counts = workload_profile.get("media_type_counts", {}) if workload_profile else {}
    task_latency_stats = latency_profile.get("task_latency_stats", {}) if latency_profile else {}
    slow_tasks = [
        {
            "task_type": task_type,
            "p95_latency_ms": stats.get("p95_latency_ms", UNKNOWN_VALUE_TEXT),
            "real_call_count": stats.get("real_call_count", 0),
            "real_api_call_count": stats.get("real_api_call_count", 0),
            "local_runtime_call_count": stats.get("local_runtime_call_count", 0),
            "mock_call_count": stats.get("mock_call_count", 0),
            "real_api_p95_latency_ms": stats.get("real_api_p95_latency_ms", UNKNOWN_VALUE_TEXT),
            "local_runtime_p95_latency_ms": stats.get("local_runtime_p95_latency_ms", UNKNOWN_VALUE_TEXT),
            "mock_p95_latency_ms": stats.get("mock_p95_latency_ms", UNKNOWN_VALUE_TEXT),
            "latency_interpretation": stats.get("latency_interpretation", UNKNOWN_VALUE_TEXT),
        }
        for task_type, stats in sorted(
            task_latency_stats.items(),
            key=lambda item: float(item[1].get("p95_latency_ms") or 0),
            reverse=True,
        )
        if isinstance(stats.get("p95_latency_ms"), (int, float)) and float(stats.get("p95_latency_ms")) > 0
    ][:3]

    if latency_status == "fail" and budget_status == "pass":
        decision = "shrink_scope_before_running"
        reason = "当前预算约束通过，但 P95 延迟约束失败；不应直接跑完整 input，应先缩小范围定位慢点。"
    elif preflight_status == "fail":
        decision = "fix_blocking_constraints_first"
        reason = "当前存在硬性失败约束，应先处理阻塞项，再考虑受控试跑。"
    elif preflight_status == "warning":
        decision = "controlled_trial_allowed_with_boundaries"
        reason = "当前没有硬性失败，但存在 mock 或未知数据边界，只能做受控工程试跑。"
    else:
        decision = "controlled_trial_allowed"
        reason = "当前预检查通过，可以在授权范围内做小批量受控试跑。"

    max_image_files = min(int(media_counts.get("image", 0)), 2)
    return {
        "decision": decision,
        "reason": reason,
        "recommended_scope": {
            "max_total_files": min(total_files, 3) if total_files else 3,
            "max_text_files": min(int(media_counts.get("text", 0)), 1),
            "max_image_files": max_image_files,
            "max_video_files": 0,
            "max_live_api_files": 1,
            "scope_reason": "延迟阻塞未解除前，先用最多3个文件验证链路；暂不纳入视频，避免把 OCR、视频预处理和 mock 边界混在一起。",
        },
        "suggested_include_files": selected_files,
        "slow_task_evidence": slow_tasks,
        "trial_commands": _build_trial_commands(selected_files),
        "do_not_run": [
            "不要直接处理完整 input 目录。",
            "不要在同一轮试跑里同时放开 PaddleOCR 和 DeepSeek 大量真实调用。",
            "不要把 visual_understanding 或 speech_to_text 的 0ms mock 延迟解释为真实供应商性能。",
        ],
        "success_criteria": [
            "受控试跑能生成 batch_metadata、results、model_calls 和 batch_report。",
            "model_calls 中能分清真实调用和 mock 调用。",
            "OCR 慢点和 DeepSeek 慢点能分别观察，不能混成一个总耗时结论。",
            "如果小批量仍超过 P95 目标，不扩大运行范围。",
        ],
        "next_decision_after_trial": [
            "如果 OCR 仍明显超过延迟目标，保留本地 OCR 基线，但不要把它写成生产可用 OCR。",
            "如果 DeepSeek 文本分析仍超过延迟目标，文本链路可以继续做质量评估，但不承诺在线低延迟。",
            "如果受控试跑通过，再逐步增加文件数；每次只增加一个变量。",
        ],
    }


def _recommended_action(
    preflight_status: str,
    route_summary: dict[str, Any],
    constraint_checks: list[dict[str, Any]],
) -> str:
    """根据预检查状态给出行动建议。"""

    if preflight_status == "fail":
        failed_names = [check["constraint_name"] for check in constraint_checks if check["status"] == "fail"]
        return f"暂不建议直接扩大运行；先处理失败约束：{', '.join(failed_names)}。"
    if route_summary["mock_task_types"]:
        return "可以做受控工程试跑，但必须把 mock 任务边界写入报告；不能把结果解释为完整真实多模态能力。"
    return "当前预检查没有发现硬阻塞；可以在既定授权边界内进入受控批处理。"


def _boundary_notes() -> list[str]:
    """说明预检查不能替代的内容。"""

    return [
        "路由预检查不调用真实模型，因此不能产生新的质量结论。",
        "预算检查只有在提供预估用量时才有意义；缺少用量时不会硬算总成本。",
        "延迟检查只有在提供历史或目标前估 P95 延迟时才有意义；缺少延迟数据时只给出未知状态。",
        "本模块不会自动修改 routing_rules.yaml，也不会替代运行时模型路由器。",
    ]


def _field_notes() -> dict[str, str]:
    """解释预检查报告中的关键字段。"""

    return {
        "workload_profile": "运行前规模画像，用于在不调用模型的情况下统计输入文件规模，并估算各任务会消耗的计量单位。",
        "latency_profile": "历史延迟画像，用于从已有模型调用记录中提取任务级 P95 延迟，帮助运行前判断延迟约束。",
        "latency_bottleneck_analysis": "延迟阻塞归因，用于把慢因拆成真实外部 API、本地运行和 mock 占位三类，避免把不同来源的延迟混成一个结论。",
        "real_api_slow_tasks": "真实外部 API 慢任务，表示这些任务的历史真实网络调用 P95 超过当前目标，可用于定位供应商或请求链路风险。",
        "local_runtime_slow_tasks": "本地运行慢任务，表示这些任务的耗时来自本机运行环境，例如 PaddleOCR 本地推理，不能直接代表云端供应商 SLA。",
        "mock_latency_unusable_tasks": "mock 延迟不可用任务，表示这些任务虽然有延迟记录，但只是占位分支，不能用于供应商速度判断。",
        "latency_interpretation": "单个任务延迟口径解释，用于说明该任务的 P95 来自真实 API、本地运行、mock 还是混合来源。",
        "price_catalog_profile": "价格目录画像，用于检查本次路由涉及模型的价格来源、更新时间和可信度，避免过期或未验证价格直接支撑成本决策。",
        "max_price_age_days": "价格过期阈值，表示价格更新时间超过多少天后需要提示先刷新价格目录。",
        "price_freshness_status": "价格新鲜度状态，用于区分 fresh、stale、missing_updated_at、invalid_updated_at 和 future_updated_at。",
        "price_confidence_status": "价格可信度状态，用于区分官方价格、可解释的 mock 或本地零成本假设、未知价格和未验证价格。",
        "expected_units_by_task": "按任务类型整理的预估用量，用于把单位价格转换成整批预算预估；缺少该字段时不会硬算总成本。",
        "historical_p95_latency_by_task_ms": "按任务类型整理的历史 P95 延迟，用于把历史调用经验带入运行前延迟预检查。",
        "budget_limit_cny": "本次预算上限，用于判断当前模型组合在预估用量下是否可能超出人民币预算。",
        "p95_latency_limit_ms": "P95 延迟限制，用于判断最慢的高分位任务延迟是否超过业务目标。",
        "task_latency_targets_ms": "按任务类型配置的 P95 延迟目标，用于让 OCR、文本分析、视觉理解等不同任务使用不同延迟闸门；没有配置的任务使用全局 P95 目标兜底。",
        "task_latency_target_checks": "任务级延迟目标检查明细，用于记录每个任务观察到的 P95、目标 P95、目标来源、证据口径和通过状态。",
        "task_latency_target_summary": "任务级延迟目标汇总，用于把多条任务级延迟检查合并成整体预检查中的延迟约束结果。",
        "evidence_level": "证据口径，用于区分非mock历史延迟、本地运行历史延迟和mock占位延迟，避免把mock延迟误读成真实供应商性能。",
        "min_real_coverage_rate": "最低真实模型覆盖率，用于判断当前路线中真实模型任务占比是否过低。",
        "current_route": "当前每个任务类型会走向哪个供应商和模型，用于运行前核对实际模型组合。",
        "preflight_status": "预检查总状态；pass 表示未发现阻塞，warning 表示可试跑但有未知或 mock 风险，fail 表示不建议直接运行。",
        "blocking_reasons": "阻塞原因列表，用于说明为什么当前配置不应直接进入扩大运行。",
        "warning_messages": "风险提示列表，用于说明哪些地方可以继续试跑但不能过度解读。",
        "estimated_cost_scope": "成本估算范围说明，用于区分单位价格检查、预估成本和真实账单。",
        "controlled_trial_plan": "受控小样本试跑建议，用于在预算可接受但延迟失败或仍有 mock 边界时，说明下一轮应缩小到哪些范围、怎么试跑、哪些结论不能越界。",
        "suggested_include_files": "建议传给 `--include-files` 的文件名列表，用于只处理少量代表性文件，避免误跑完整输入目录。",
        "trial_commands": "受控试跑命令建议，只作为人工执行参考；报告生成本身不会执行这些命令，也不会触发模型调用。",
    }


def render_preflight_markdown(report: dict[str, Any]) -> str:
    """把预检查报告渲染为 Markdown。"""

    route_summary = report["route_summary"]
    lines = [
        "# 路由策略预检查报告",
        "",
        f"生成时间：{report['generated_at']}",
        "",
        "说明：本报告只读取本地配置和可选估算输入，不触发 DeepSeek、PaddleOCR 或任何外部模型调用。",
        "",
        f"策略名称：`{report['policy_name']}`",
        f"预检查状态：`{report['preflight_status']}`",
        "",
    ]
    if report.get("workload_profile"):
        profile = report["workload_profile"]
        media_counts = profile["media_type_counts"]
        lines.extend(
            [
                "## 0. 运行前规模画像",
                "",
                "| 指标 | 值 | 含义 |",
                "|---|---:|---|",
                f"| 输入文件总数 | {profile['total_files']} | 本次预检查纳入的文件数量 |",
                f"| 文本文件数 | {media_counts.get('text', 0)} | 会直接进入文本读取与文本分析的文件数 |",
                f"| 图片文件数 | {media_counts.get('image', 0)} | 会触发 OCR 和视觉理解的图片数 |",
                f"| 视频文件数 | {media_counts.get('video', 0)} | 会触发抽帧、语音识别、OCR 和视觉理解的视频数 |",
                f"| 文件总大小 | {profile['total_file_size_bytes']} bytes | 用于粗略判断批次规模 |",
                f"| 估算文本 token | {profile['estimated_raw_text_tokens']} | 基于文本字符数粗估，不等于供应商真实计费 token |",
                "",
                "预估任务用量：",
                "",
                "| 任务类型 | 预估用量 | 含义 |",
                "|---|---|---|",
            ]
        )
        for task_type, units in profile["expected_units_by_task"].items():
            lines.append(
                f"| {task_type} | {_format_units(units)} | 用于运行前预算估算的任务单位 |"
            )
        if profile["warning_messages"]:
            lines.extend(["", "画像风险提示："])
            lines.extend(f"- {item}" for item in profile["warning_messages"])
        lines.append("")

    if report.get("latency_profile"):
        latency_profile = report["latency_profile"]
        lines.extend(
            [
                "## 0.1 历史延迟画像",
                "",
                "| 任务类型 | 调用数 | 真实调用数 | 真实API调用数 | 本地运行调用数 | mock调用数 | 平均延迟 | P95延迟 | 真实API P95 | 本地运行 P95 | mock P95 | 最大延迟 | 模型 | 口径解释 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for task_type, stats in latency_profile["task_latency_stats"].items():
            lines.append(
                "| "
                f"{task_type} | "
                f"{stats['call_count']} | "
                f"{stats['real_call_count']} | "
                f"{stats.get('real_api_call_count', 0)} | "
                f"{stats.get('local_runtime_call_count', 0)} | "
                f"{stats['mock_call_count']} | "
                f"{_format_ms(stats['avg_latency_ms'])} | "
                f"{_format_ms(stats['p95_latency_ms'])} | "
                f"{_format_ms(stats.get('real_api_p95_latency_ms'))} | "
                f"{_format_ms(stats.get('local_runtime_p95_latency_ms'))} | "
                f"{_format_ms(stats.get('mock_p95_latency_ms'))} | "
                f"{_format_ms(stats['max_latency_ms'])} | "
                f"{', '.join(stats['models']) or UNKNOWN_VALUE_TEXT} | "
                f"{stats.get('latency_interpretation', UNKNOWN_VALUE_TEXT)} |"
            )
        if latency_profile["warning_messages"]:
            lines.extend(["", "延迟画像风险提示："])
            lines.extend(f"- {item}" for item in latency_profile["warning_messages"])
        lines.append("")

    if report.get("latency_bottleneck_analysis"):
        analysis = report["latency_bottleneck_analysis"]
        lines.extend(
            [
                "## 0.1.1 延迟阻塞归因",
                "",
                f"- 归因状态：`{analysis['bottleneck_status']}`",
                f"- P95 延迟目标：{_format_ms(analysis['p95_latency_limit_ms'])}",
                f"- 任务级 P95 目标：{_format_task_latency_targets(analysis.get('task_latency_targets_ms'))}",
                "",
            ]
        )
        if analysis.get("root_cause_summary"):
            lines.extend(["根因摘要：", ""])
            lines.extend(f"- {item}" for item in analysis["root_cause_summary"])
            lines.append("")

        if analysis.get("local_runtime_slow_tasks"):
            lines.extend(
                [
                    "本地运行慢任务：",
                    "",
                    "| 任务类型 | P95延迟 | 调用数 | 证据口径 | 原因解释 |",
                    "|---|---:|---:|---|---|",
                ]
            )
            for item in analysis["local_runtime_slow_tasks"]:
                lines.append(
                    "| "
                    f"{item['task_type']} | "
                    f"{_format_ms(item['p95_latency_ms'])} | "
                    f"{item['call_count']} | "
                    f"{item['evidence_level']} | "
                    f"{item['reason']} |"
                )
            lines.append("")

        if analysis.get("real_api_slow_tasks"):
            lines.extend(
                [
                    "真实 API 慢任务：",
                    "",
                    "| 任务类型 | P95延迟 | 调用数 | 证据口径 | 原因解释 |",
                    "|---|---:|---:|---|---|",
                ]
            )
            for item in analysis["real_api_slow_tasks"]:
                lines.append(
                    "| "
                    f"{item['task_type']} | "
                    f"{_format_ms(item['p95_latency_ms'])} | "
                    f"{item['call_count']} | "
                    f"{item['evidence_level']} | "
                    f"{item['reason']} |"
                )
            lines.append("")

        if analysis.get("mock_latency_unusable_tasks"):
            lines.extend(
                [
                    "mock 延迟不可用任务：",
                    "",
                    "| 任务类型 | mock调用数 | mock P95 | 证据口径 | 为什么不能用于判断 |",
                    "|---|---:|---:|---|---|",
                ]
            )
            for item in analysis["mock_latency_unusable_tasks"]:
                lines.append(
                    "| "
                    f"{item['task_type']} | "
                    f"{item['mock_call_count']} | "
                    f"{_format_ms(item['mock_p95_latency_ms'])} | "
                    f"{item['evidence_level']} | "
                    f"{item['reason']} |"
                )
            lines.append("")

        if analysis.get("recommended_next_actions"):
            lines.extend(["下一步建议：", ""])
            lines.extend(f"- {item}" for item in analysis["recommended_next_actions"])
            lines.append("")

    if report.get("task_latency_target_checks"):
        lines.extend(
            [
                "## 0.1.2 任务级延迟目标检查",
                "",
                "| 任务类型 | 观察到的P95延迟 | 目标P95延迟 | 目标来源 | 证据口径 | 状态 | 说明 |",
                "|---|---:|---:|---|---|---|---|",
            ]
        )
        for check in report["task_latency_target_checks"]:
            lines.append(
                "| "
                f"{check['task_type']} | "
                f"{_format_ms(check['observed_p95_latency_ms'])} | "
                f"{_format_ms(check['target_p95_latency_ms'])} | "
                f"{check['target_source']} | "
                f"{check.get('evidence_level', UNKNOWN_VALUE_TEXT)} | "
                f"{check['status']} | "
                f"{check['reason']} |"
            )
        lines.append("")

    if report.get("price_catalog_profile"):
        price_profile = report["price_catalog_profile"]
        lines.extend(
            [
                "## 0.2 价格目录画像",
                "",
                "| 指标 | 值 | 含义 |",
                "|---|---:|---|",
                f"| 检查模型数 | {price_profile['checked_model_count']} | 本次路由涉及并进入价格目录检查的模型数量 |",
                f"| 价格目录状态 | {price_profile['price_catalog_status']} | pass 表示未发现价格目录风险，warning 表示成本结论需要谨慎解释 |",
                f"| 价格过期阈值 | {price_profile['max_price_age_days']} 天 | 超过该天数会提示先刷新价格目录 |",
                "",
                "| 任务类型 | 模型 | 价格来源 | 更新时间 | 价格年龄 | 新鲜度 | 可信度 |",
                "|---|---|---|---|---:|---|---|",
            ]
        )
        for item in price_profile["checked_items"]:
            lines.append(
                "| "
                f"{item['task_type']} | "
                f"{item['model_name']} | "
                f"{item['price_source']} | "
                f"{item['price_updated_at']} | "
                f"{_format_value(item['price_age_days'])} | "
                f"{item['price_freshness_status']} | "
                f"{item['price_confidence_status']} |"
            )
        if price_profile["warning_messages"]:
            lines.extend(["", "价格目录风险提示："])
            lines.extend(f"- {item}" for item in price_profile["warning_messages"])
        lines.append("")

    lines.extend(
        [
            "## 1. 当前路由摘要",
            "",
            "| 指标 | 值 | 含义 |",
            "|---|---:|---|",
            f"| 预期任务类型数 | {route_summary['total_expected_task_types']} | 本次预检查覆盖的任务类型数量 |",
            f"| 真实任务类型 | {', '.join(route_summary['real_task_types']) or UNKNOWN_VALUE_TEXT} | 当前配置中非 mock 的任务类型 |",
            f"| mock 任务类型 | {', '.join(route_summary['mock_task_types']) or UNKNOWN_VALUE_TEXT} | 当前仍为占位流程的任务类型 |",
            f"| 缺失路由任务 | {', '.join(route_summary['missing_task_types']) or UNKNOWN_VALUE_TEXT} | 没有配置供应商和模型的任务类型 |",
            f"| 真实模型覆盖率 | {_format_percent(route_summary['real_coverage_rate'])} | 非 mock 任务占预期任务的比例 |",
            f"| 预估总成本 | {_format_cny(route_summary['estimated_total_cost_cny'])} | 基于传入预估用量计算；缺数据则不硬算 |",
            f"| 最大预估 P95 延迟 | {_format_ms(route_summary['max_expected_p95_latency_ms'])} | 当前可用延迟数据中的最高 P95 |",
            "",
            "## 2. 路由明细",
            "",
            "| 任务类型 | 供应商 | 模型 | mock? | 价格状态 | 预估成本 | P95 延迟 | 风险说明 |",
            "|---|---|---|---|---|---:|---:|---|",
        ]
    )
    for entry in report["current_route"]:
        lines.append(
            "| "
            f"{entry['task_type']} | "
            f"{entry['provider']} | "
            f"{entry['model_name']} | "
            f"{_format_bool(entry['is_mock'])} | "
            f"{entry['price_status']} | "
            f"{_format_cny(entry['estimated_cost_cny'])} | "
            f"{_format_ms(entry['expected_p95_latency_ms'])} | "
            f"{'；'.join(entry['risk_notes']) or '无'} |"
        )

    lines.extend(
        [
            "",
            "## 3. 约束检查",
            "",
            "| 约束 | 观测值 | 限制值 | 状态 | 说明 |",
            "|---|---:|---:|---|---|",
        ]
    )
    for check in report["constraint_checks"]:
        lines.append(
            f"| {check['constraint_name']} | {_format_value(check['observed_value'])} | "
            f"{_format_value(check['limit_value'])} | {check['status']} | {check['reason']} |"
        )

    lines.extend(["", "## 4. 阻塞与风险", ""])
    if report["blocking_reasons"]:
        lines.append("阻塞原因：")
        lines.extend(f"- {item}" for item in report["blocking_reasons"])
        lines.append("")
    if report["warning_messages"]:
        lines.append("风险提示：")
        lines.extend(f"- {item}" for item in report["warning_messages"])
        lines.append("")

    controlled_trial_plan = report.get("controlled_trial_plan")
    if controlled_trial_plan:
        scope = controlled_trial_plan["recommended_scope"]
        lines.extend(
            [
                "## 5. 受控小样本试跑建议",
                "",
                f"决策：`{controlled_trial_plan['decision']}`",
                "",
                f"原因：{controlled_trial_plan['reason']}",
                "",
                "| 范围项 | 建议值 | 含义 |",
                "|---|---:|---|",
                f"| 最大总文件数 | {scope['max_total_files']} | 延迟问题未定位前，本轮最多处理的文件数 |",
                f"| 最大文本文件数 | {scope['max_text_files']} | 用于观察文本分析链路和 DeepSeek 延迟 |",
                f"| 最大图片文件数 | {scope['max_image_files']} | 用于观察本地 OCR 延迟和错误状态 |",
                f"| 最大视频文件数 | {scope['max_video_files']} | 当前先不纳入视频，避免混入视频预处理和 mock 边界 |",
                f"| 最大真实 API 文件数 | {scope['max_live_api_files']} | 如需调用 DeepSeek，本轮最多纳入的文件数 |",
                "",
                f"范围理由：{scope['scope_reason']}",
                "",
                f"建议 include-files：`{', '.join(controlled_trial_plan['suggested_include_files']) or UNKNOWN_VALUE_TEXT}`",
                "",
            ]
        )
        if controlled_trial_plan["slow_task_evidence"]:
            lines.extend(
                [
                    "慢任务证据：",
                    "",
                    "| 任务类型 | P95延迟 | 真实API P95 | 本地运行 P95 | mock P95 | 真实API调用数 | 本地运行调用数 | mock调用数 | 含义 |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for item in controlled_trial_plan["slow_task_evidence"]:
                lines.append(
                    "| "
                    f"{item['task_type']} | "
                    f"{_format_ms(item['p95_latency_ms'])} | "
                    f"{_format_ms(item.get('real_api_p95_latency_ms'))} | "
                    f"{_format_ms(item.get('local_runtime_p95_latency_ms'))} | "
                    f"{_format_ms(item.get('mock_p95_latency_ms'))} | "
                    f"{item.get('real_api_call_count', 0)} | "
                    f"{item.get('local_runtime_call_count', 0)} | "
                    f"{item['mock_call_count']} | "
                    f"{item.get('latency_interpretation', '用于判断该任务是否是继续扩大运行前的延迟阻塞')} |"
                )
            lines.append("")
        lines.extend(["建议命令：", ""])
        for command in controlled_trial_plan["trial_commands"]:
            api_note = "需要真实 API 授权" if command["requires_live_api"] else "不需要真实 API"
            lines.extend(
                [
                    f"- `{command['command_name']}`（{api_note}）：{command['purpose']}",
                    "",
                    "```powershell",
                    command["command"],
                    "```",
                    "",
                ]
            )
        lines.extend(["不要做："])
        lines.extend(f"- {item}" for item in controlled_trial_plan["do_not_run"])
        lines.extend(["", "成功标准："])
        lines.extend(f"- {item}" for item in controlled_trial_plan["success_criteria"])
        lines.extend(["", "试跑后的判断："])
        lines.extend(f"- {item}" for item in controlled_trial_plan["next_decision_after_trial"])
        lines.append("")

    lines.extend(
        [
            "## 6. 建议动作",
            "",
            report["recommended_action"],
            "",
            "## 7. 边界说明",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["boundary_notes"])

    lines.extend(["", "## 8. 字段说明", "", "| 字段 | 含义与作用 |", "|---|---|"])
    for field_name, note in report["field_notes"].items():
        lines.append(f"| `{field_name}` | {note} |")

    lines.append("")
    return "\n".join(lines)


def _format_bool(value: Any) -> str:
    """把布尔值转换为中文显示。"""

    if value == UNKNOWN_VALUE_TEXT:
        return UNKNOWN_VALUE_TEXT
    return "是" if value else "否"


def _format_value(value: Any) -> str:
    """把数值或未知状态转成展示文本。"""

    if value in {None, UNKNOWN_VALUE_TEXT}:
        return UNKNOWN_VALUE_TEXT
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _format_cny(value: Any) -> str:
    """格式化人民币金额。"""

    if value in {None, UNKNOWN_VALUE_TEXT}:
        return UNKNOWN_VALUE_TEXT
    return f"{float(value):.6f} 元"


def _format_ms(value: Any) -> str:
    """格式化毫秒延迟。"""

    if value in {None, UNKNOWN_VALUE_TEXT}:
        return UNKNOWN_VALUE_TEXT
    return f"{float(value):.0f} ms"


def _format_task_latency_targets(value: Any) -> str:
    """格式化任务级 P95 延迟目标。"""

    if not isinstance(value, dict) or not value:
        return UNKNOWN_VALUE_TEXT
    parts = [f"{task_type}={_format_ms(limit)}" for task_type, limit in sorted(value.items())]
    return "；".join(parts)


def _format_percent(value: Any) -> str:
    """格式化比例。"""

    if value in {None, UNKNOWN_VALUE_TEXT}:
        return UNKNOWN_VALUE_TEXT
    return f"{float(value) * 100:.2f}%"


def _format_units(units: Any) -> str:
    """格式化预估用量。"""

    normalized_units = _normalize_units(units)
    if not normalized_units:
        return UNKNOWN_VALUE_TEXT
    parts = []
    for unit in normalized_units:
        parts.append(f"{unit.get('unit_type', UNKNOWN_VALUE_TEXT)}={unit.get('quantity', UNKNOWN_VALUE_TEXT)}")
    return "；".join(parts)


def _read_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    """读取可选 JSON 文件。"""

    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_task_types(raw_value: str | None) -> list[str] | None:
    """解析逗号分隔的任务类型。"""

    if not raw_value:
        return None
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _parse_include_files(raw_value: str | None) -> list[str] | None:
    """解析逗号分隔的指定文件名。"""

    if not raw_value:
        return None
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _parse_path_list_arg(raw_value: str | None) -> list[str] | None:
    """解析逗号分隔的路径列表。"""

    if not raw_value:
        return None
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _build_policy_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """把命令行约束参数转换为策略覆盖项。"""

    return {
        "budget_limit_cny": args.budget_limit_cny,
        "p95_latency_limit_ms": args.p95_latency_limit_ms,
        "min_real_coverage_rate": args.min_real_coverage_rate,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="生成运行前路由策略预检查报告。")
    parser.add_argument("--routing-rules", default="config/routing_rules.yaml", help="路由规则配置文件路径。")
    parser.add_argument("--model-prices", default="config/model_prices.yaml", help="模型价格配置文件路径。")
    parser.add_argument("--policy-config", default="config/routing_policy_config.yaml", help="策略约束配置文件路径。")
    parser.add_argument("--policy", default="balanced", help="要使用的策略名称，例如 balanced。")
    parser.add_argument("--output-dir", default="output/preflight", help="预检查报告输出目录。")
    parser.add_argument("--expected-task-types", help="逗号分隔的预期任务类型；不填时使用默认多模态任务集合。")
    parser.add_argument("--expected-units-json", help="可选预估用量 JSON 文件；用于预算预检查。")
    parser.add_argument("--historical-latency-json", help="可选历史 P95 延迟 JSON 文件；用于延迟预检查。")
    parser.add_argument("--historical-model-calls", help="可选历史 model_calls.jsonl 路径，多个路径用逗号分隔；用于自动提取任务级 P95 延迟。")
    parser.add_argument("--input-dir", help="可选输入目录；提供后会自动生成运行前规模画像和预估用量。")
    parser.add_argument("--include-files", help="可选文件名清单，用逗号分隔；用于只预检查部分输入文件。")
    parser.add_argument("--expected-frames-per-video", type=int, default=3, help="每个视频预计抽取的关键帧数量。")
    parser.add_argument("--expected-audio-seconds-per-video", type=int, help="每个视频预计进入语音识别的音频秒数。")
    parser.add_argument("--expected-output-tokens-per-file", type=int, default=300, help="每个文件预计生成的文本输出 token 数。")
    parser.add_argument("--estimated-evidence-tokens-per-image", type=int, default=300, help="每张图片上游证据预计转成的文本 token 数。")
    parser.add_argument("--estimated-evidence-tokens-per-video", type=int, default=800, help="每个视频上游证据预计转成的文本 token 数。")
    parser.add_argument("--max-price-age-days", type=int, default=DEFAULT_MAX_PRICE_AGE_DAYS, help="价格目录过期阈值；超过该天数会在预检查报告中提示先刷新价格目录。")
    parser.add_argument("--budget-limit-cny", type=float, help="本轮预检查使用的预算上限，单位人民币。")
    parser.add_argument("--p95-latency-limit-ms", type=float, help="本轮预检查使用的 P95 延迟上限，单位毫秒。")
    parser.add_argument("--min-real-coverage-rate", type=float, help="本轮预检查使用的最低真实模型覆盖率。")
    parser.add_argument("--ocr-backend", choices=["mock", "paddleocr"], help="按主流程规则模拟 OCR 后端选择。")
    parser.add_argument("--text-analysis-backend", choices=["mock", "deepseek"], help="按主流程规则模拟文本分析后端选择。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = _parse_args(argv if argv is not None else sys.argv[1:])
    report = build_preflight_from_files(
        routing_rules_path=args.routing_rules,
        model_prices_path=args.model_prices,
        policy_config_path=args.policy_config,
        policy_name=args.policy,
        expected_task_types=_parse_task_types(args.expected_task_types),
        expected_units_by_task=_read_optional_json(args.expected_units_json),
        historical_p95_latency_by_task_ms=_read_optional_json(args.historical_latency_json),
        historical_model_calls_paths=_parse_path_list_arg(args.historical_model_calls),
        input_dir=args.input_dir,
        include_files=_parse_include_files(args.include_files),
        expected_frames_per_video=args.expected_frames_per_video,
        expected_audio_seconds_per_video=args.expected_audio_seconds_per_video,
        expected_output_tokens_per_file=args.expected_output_tokens_per_file,
        estimated_evidence_tokens_per_image=args.estimated_evidence_tokens_per_image,
        estimated_evidence_tokens_per_video=args.estimated_evidence_tokens_per_video,
        policy_overrides=_build_policy_overrides_from_args(args),
        ocr_backend=args.ocr_backend,
        text_analysis_backend=args.text_analysis_backend,
        max_price_age_days=args.max_price_age_days,
    )
    output_paths = write_preflight_reports(args.output_dir, report)
    print(json.dumps(output_paths, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
