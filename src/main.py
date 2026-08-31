"""多模态模型路由 MVP 的命令行入口。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from cost_latency_tracker import load_model_prices
from file_loader import build_file_manifest
from model_clients import (
    DEFAULT_DEEPSEEK_MAX_TOKENS,
    DEFAULT_QWEN_OCR_MAX_TOKENS,
    DEFAULT_QWEN_OCR_MODEL_NAME,
    DEFAULT_QWEN_VL_MAX_IMAGE_SIDE,
    DEFAULT_QWEN_VL_MAX_TOKENS,
)
from model_router import (
    build_route_plan,
    load_route_plan,
    load_routing_rules,
    route_plan_backends_for_media,
    routing_rules_from_route_plan,
    validate_route_plan,
)
from pipeline_runner import run_file_pipeline
from preprocessor import DEFAULT_MAX_KEYFRAMES
from report_generator import generate_batch_report
from routing_preflight import build_preflight_from_files, write_preflight_reports
from runtime_config import runtime_policy_list
from result_writer import (
    ensure_batch_output_dir,
    write_batch_metadata,
    write_errors,
    write_json,
    write_model_calls,
    write_results,
    write_results_readable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OCR_BACKENDS = runtime_policy_list("runtime_backends", "ocr")
ALLOWED_VISION_BACKENDS = runtime_policy_list("runtime_backends", "vision_understanding")
ALLOWED_SPEECH_BACKENDS = runtime_policy_list("runtime_backends", "speech_to_text")
ALLOWED_TEXT_BACKENDS = runtime_policy_list("runtime_backends", "text_analysis")
DEFAULT_TEXT_ANALYSIS_EVIDENCE_CHAR_LIMIT = 6000


def _now_iso() -> str:
    """返回当前本地时间的 ISO 字符串。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_settings(settings_path: str | Path) -> dict[str, Any]:
    """读取运行配置文件。"""

    path = Path(settings_path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _setting(settings: dict[str, Any], dotted_key: str, legacy_key: str | None = None, default: Any = None) -> Any:
    """读取新分层配置；缺失时兼容旧扁平配置。"""

    current: Any = settings
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return settings.get(legacy_key, default) if legacy_key else default
        current = current[part]
    return current


def _pipeline_backend(
    settings: dict[str, Any],
    media_type: str,
    task_name: str,
    legacy_key: str,
    default: str = "mock",
) -> str:
    """读取某类输入在某个任务上选择的后端。"""

    return str(_setting(settings, f"pipelines.{media_type}.{task_name}", legacy_key, default))


def _pipeline_selection(
    settings: dict[str, Any],
    media_type: str,
    *,
    ocr_override: str | None = None,
    vision_override: str | None = None,
    speech_override: str | None = None,
    text_override: str | None = None,
) -> dict[str, str]:
    """返回单个媒体类型实际使用的任务后端。"""

    task_names = {
        "text": (None, None, None, "text_analysis"),
        "image": ("ocr", "vision_understanding", None, "text_analysis"),
        "video": ("keyframe_ocr", "keyframe_vision_understanding", "speech_to_text", "text_analysis"),
        "audio": (None, None, "speech_to_text", "text_analysis"),
    }
    if media_type not in task_names:
        raise ValueError(f"不支持的媒体类型：{media_type}")
    ocr_task, vision_task, speech_task, text_task = task_names[media_type]
    return {
        "ocr_backend": ocr_override
        or (_pipeline_backend(settings, media_type, ocr_task, "ocr_backend") if ocr_task else "mock"),
        "vision_understanding_backend": vision_override
        or (
            _pipeline_backend(settings, media_type, vision_task, "vision_understanding_backend")
            if vision_task
            else "mock"
        ),
        "speech_to_text_backend": speech_override
        or (
            _pipeline_backend(settings, media_type, speech_task, "speech_to_text_backend")
            if speech_task
            else "mock"
        ),
        "text_analysis_backend": text_override
        or _pipeline_backend(settings, media_type, text_task, "text_analysis_backend"),
    }


def _backend_setting(
    settings: dict[str, Any],
    backend_group: str,
    backend_name: str,
    key: str,
    legacy_key: str,
    default: Any = None,
) -> Any:
    """读取模型后端参数；兼容旧扁平键。"""

    return _setting(settings, f"backends.{backend_group}.{backend_name}.{key}", legacy_key, default)


def _resolve_path(project_root: Path, path_value: str | Path) -> Path:
    """把配置中的相对路径转换为项目内绝对路径。"""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def _load_route_decision(
    project_root: Path,
    report_path: str | Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """读取候选对照报告并生成紧凑、可审计的文本选路快照。"""

    path = _resolve_path(project_root, report_path).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("report_type") != "phase2_text_backend_comparison_gate":
        raise ValueError("路由决策报告类型无效。")
    candidate = report.get("recommended_candidate")
    status = report.get("recommendation_status")
    unmet_constraints = report.get("unmet_constraints")
    evaluations = report.get("candidate_evaluations")
    if report.get("overall_status") == "fail" or status not in {"pass", "warning"}:
        raise ValueError("路由决策报告没有可执行推荐。")
    if report.get("overall_status") != status:
        raise ValueError("路由决策报告的 overall_status 与 recommendation_status 不一致。")
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("路由决策报告缺少 recommended_candidate。")
    if not isinstance(unmet_constraints, list):
        raise ValueError("路由决策报告的 unmet_constraints 必须是列表。")
    if any(not isinstance(item, str) or not item for item in unmet_constraints):
        raise ValueError("路由决策报告的 unmet_constraints 只能包含非空字符串。")
    if status == "pass" and unmet_constraints:
        raise ValueError("pass 路由推荐不能包含 unmet_constraints。")
    if status == "warning" and not unmet_constraints:
        raise ValueError("warning 路由推荐必须说明 unmet_constraints。")
    selected_candidates = report.get("selected_candidates")
    if status == "pass" and (
        not isinstance(selected_candidates, list) or candidate not in selected_candidates
    ):
        raise ValueError("pass 路由推荐必须属于 selected_candidates。")
    evaluation = evaluations.get(candidate) if isinstance(evaluations, dict) else None
    if not isinstance(evaluation, dict):
        raise ValueError("路由决策报告缺少推荐候选的评估摘要。")
    text_backends = (settings.get("backends") or {}).get("text_analysis")
    if not isinstance(text_backends, dict) or candidate not in text_backends:
        raise ValueError(f"路由决策报告推荐了 settings 中不存在的文本后端：{candidate}")
    non_compared_tasks = sorted(
        task for task in (settings.get("backends") or {}) if task != "text_analysis"
    )
    return {
        "task_type": "text_analysis",
        "recommended_candidate": candidate,
        "recommendation_status": status,
        "unmet_constraints": list(unmet_constraints),
        "non_compared_tasks": non_compared_tasks,
        "evidence_source": str(path),
        "evidence_generated_at": report.get("generated_at"),
        "candidate_summary": {
            key: evaluation.get(key)
            for key in (
                "quality_pass",
                "latency_pass",
                "text_analysis_p95_latency_ms",
                "successful_text_call_count",
                "estimated_cost_cny",
            )
        },
    }


def _load_asr_audio_url_map(project_root: Path, path_value: str | Path | None) -> dict[str, str]:
    """读取文件名到远端音频 URL 的映射。"""

    if path_value is None:
        return {}
    path = _resolve_path(project_root, path_value)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in data.items()):
        raise ValueError("ASR 音频 URL 映射必须是 JSON 对象，键和值都必须是字符串。")
    return data


def _ensure_paddleocr_runtime_available() -> None:
    """在生成批次输出前检查本地 PaddleOCR 的两个必要包。"""

    missing_packages = [
        package_name
        for package_name in ("paddle", "paddleocr")
        if importlib.util.find_spec(package_name) is None
    ]
    if missing_packages:
        missing_text = "、".join(missing_packages)
        raise RuntimeError(
            f"缺少 PaddleOCR 运行依赖：{missing_text}。请先按 README 完成本地安装。"
        )


def _positive_int_setting(settings: dict[str, Any], key: str, default: int) -> int:
    """读取大于等于 1 的整数配置。"""

    value = int(settings.get(key, default))
    if value < 1:
        raise ValueError(f"{key} 必须是大于等于 1 的整数。")
    return value


def _positive_int_value(value: Any, name: str, default: int) -> int:
    """校验大于等于 1 的整数配置值。"""

    parsed = int(default if value is None else value)
    if parsed < 1:
        raise ValueError(f"{name} 必须是大于等于 1 的整数。")
    return parsed


def _bool_setting(settings: dict[str, Any], key: str, default: bool) -> bool:
    """读取布尔配置，拒绝容易误读的字符串。"""

    value = settings.get(key, default)
    return _bool_value(value, key)


def _bool_value(value: Any, name: str) -> bool:
    """校验布尔配置值。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} 必须是 true 或 false。")


def _filter_manifest_by_file_names(
    file_manifest: list[dict[str, Any]],
    include_file_names: list[str] | None,
) -> list[dict[str, Any]]:
    """按文件名保留本次明确指定的输入文件。"""

    if include_file_names is None:
        return file_manifest

    selected_file_names = {file_name.strip() for file_name in include_file_names if file_name.strip()}
    if not selected_file_names:
        raise ValueError("指定文件列表不能为空。")

    matched_file_names = {
        file_record["file_name"]
        for file_record in file_manifest
        if file_record["file_name"] in selected_file_names
    }
    missing_file_names = sorted(selected_file_names - matched_file_names)
    if missing_file_names:
        raise ValueError(f"指定文件未在输入目录中找到：{', '.join(missing_file_names)}")

    return [
        file_record
        for file_record in file_manifest
        if file_record["file_name"] in selected_file_names
    ]


def _runtime_type_for_model_call(model_call: dict[str, Any]) -> str:
    """根据单条调用记录判断它是真实 API、本地模型、mock 还是未知来源。"""

    provider = str(model_call.get("provider", ""))
    model_name = str(model_call.get("model_name", ""))
    if model_name.startswith("mock-"):
        return "mock"
    if provider == "paddlepaddle":
        return "local_model"
    if provider in {"deepseek", "qwen", "dashscope"}:
        return "live_api"
    return "unknown"


def _build_backend_runtime_summary(model_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """从实际模型调用记录汇总本批次真实 / 本地 / mock 后端组合。"""

    breakdown: dict[tuple[str, str, str, str, str | None], dict[str, Any]] = {}
    runtime_types_present: set[str] = set()
    for model_call in model_calls:
        runtime_type = _runtime_type_for_model_call(model_call)
        runtime_types_present.add(runtime_type)
        key = (
            str(model_call.get("task_type", "unknown_task")),
            str(model_call.get("provider", "unknown_provider")),
            str(model_call.get("model_name", "unknown_model")),
            runtime_type,
            model_call.get("response_model_name"),
        )
        if key not in breakdown:
            task_type, provider, model_name, _, response_model_name = key
            breakdown[key] = {
                "task_type": task_type,
                "provider": provider,
                "model_name": model_name,
                "response_model_name": response_model_name,
                "runtime_type": runtime_type,
                "call_count": 0,
            }
        breakdown[key]["call_count"] += 1

    sorted_breakdown = sorted(
        breakdown.values(),
        key=lambda item: (
            str(item["runtime_type"]),
            str(item["task_type"]),
            str(item["provider"]),
            str(item["model_name"]),
        ),
    )
    return {
        "evidence_source": "model_calls.jsonl",
        "runtime_types_present": sorted(runtime_types_present),
        "contains_live_api": "live_api" in runtime_types_present,
        "contains_local_model": "local_model" in runtime_types_present,
        "contains_mock": "mock" in runtime_types_present,
        "model_call_runtime_breakdown": sorted_breakdown,
    }


def _default_request_purpose(backend_runtime_summary: dict[str, Any]) -> str:
    """根据实际后端组合生成默认批次用途说明。"""

    labels = []
    if backend_runtime_summary["contains_live_api"]:
        labels.append("真实 API")
    if backend_runtime_summary["contains_local_model"]:
        labels.append("本地模型")
    if backend_runtime_summary["contains_mock"]:
        labels.append("mock")
    if not labels:
        labels.append("未产生模型调用")
    return f"受控{' + '.join(labels)}批处理验证"


def run_preflight(
    *,
    settings_path: str | Path = PROJECT_ROOT / "config" / "settings.yaml",
    routing_rules_path: str | Path | None = None,
    model_prices_path: str | Path | None = None,
    policy_config_path: str | Path | None = None,
    policy_name: str = "balanced",
    input_dir_override: str | Path | None = None,
    include_file_names: list[str] | None = None,
    ocr_backend_override: str | None = None,
    vision_understanding_backend_override: str | None = None,
    speech_to_text_backend_override: str | None = None,
    text_analysis_backend_override: str | None = None,
    batch_id: str | None = None,
    historical_model_calls_paths: list[str | Path] | None = None,
    max_keyframes: int | None = None,
    generated_at: str | None = None,
    route_decision_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """运行批处理前的路由预检查，不触发模型 API。"""

    settings_file = Path(settings_path)
    project_root = settings_file.resolve().parents[1]
    settings = load_settings(settings_file)
    input_dir = _resolve_path(project_root, input_dir_override or _setting(settings, "paths.input_dir", "input_dir"))
    output_dir = _resolve_path(project_root, _setting(settings, "paths.output_dir", "output_dir"))
    current_batch_id = batch_id or f"preflight_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    default_policy_config = policy_config_path or project_root / "config" / "routing_policy_config.yaml"
    if not Path(default_policy_config).exists():
        default_policy_config = PROJECT_ROOT / "config" / "routing_policy_config.yaml"
    uses_nested_routes = isinstance(settings.get("pipelines"), dict) and isinstance(settings.get("backends"), dict)
    selection_decisions: list[dict[str, Any]] = []
    if route_decision_report_path is not None:
        if not uses_nested_routes:
            raise ValueError("路由决策报告只支持双层 settings 配置。")
        if text_analysis_backend_override is not None:
            raise ValueError("路由决策报告不能与 text_analysis_backend_override 同时使用。")
        decision = _load_route_decision(project_root, route_decision_report_path, settings)
        text_analysis_backend_override = decision["recommended_candidate"]
        selection_decisions.append(decision)
    if uses_nested_routes:
        if routing_rules_path is not None:
            raise ValueError("双层 settings 路由预检查不接受 routing_rules_path；settings.yaml 是唯一事实来源。")
        draft_route_plan = build_route_plan(
            settings,
            preflight_status="warning",
            policy_name=policy_name,
            source_settings=str(settings_file.resolve()),
            generated_at=generated_at,
            ocr_backend=ocr_backend_override,
            vision_understanding_backend=vision_understanding_backend_override,
            speech_to_text_backend=speech_to_text_backend_override,
            text_analysis_backend=text_analysis_backend_override,
            selection_decisions=selection_decisions,
        )
        preflight_routing_rules, route_conflicts = routing_rules_from_route_plan(draft_route_plan)
        report = build_preflight_from_files(
            routing_rules_path=None,
            routing_rules_override=preflight_routing_rules,
            routing_source=str(settings_file.resolve()),
            model_prices_path=model_prices_path or project_root / "config" / "model_prices.yaml",
            policy_config_path=default_policy_config,
            policy_name=policy_name,
            input_dir=input_dir,
            include_files=include_file_names,
            expected_frames_per_video=max_keyframes or DEFAULT_MAX_KEYFRAMES,
            historical_model_calls_paths=historical_model_calls_paths,
            generated_at=generated_at,
        )
        if route_conflicts:
            report["warning_messages"] = [*report.get("warning_messages", []), *route_conflicts]
            if report["preflight_status"] == "pass":
                report["preflight_status"] = "warning"
                report["recommended_action"] = "路由计划可受控执行，但任务级预检查存在跨媒体代表路线限制，必须保留风险提示。"
        if selection_decisions and selection_decisions[0]["recommendation_status"] == "warning":
            decision_warning = (
                "文本后端来自 warning 推荐，未满足约束："
                + ", ".join(selection_decisions[0]["unmet_constraints"])
                + "。只能显式受控执行。"
            )
            report["warning_messages"] = [*report.get("warning_messages", []), decision_warning]
            if report["preflight_status"] == "pass":
                report["preflight_status"] = "warning"
            report["recommended_action"] = decision_warning
        report["route_plan"] = build_route_plan(
            settings,
            preflight_status=report["preflight_status"],
            policy_name=policy_name,
            source_settings=str(settings_file.resolve()),
            generated_at=generated_at,
            warning_messages=report.get("warning_messages"),
            ocr_backend=ocr_backend_override,
            vision_understanding_backend=vision_understanding_backend_override,
            speech_to_text_backend=speech_to_text_backend_override,
            text_analysis_backend=text_analysis_backend_override,
            selection_decisions=selection_decisions,
        )
    else:
        report = build_preflight_from_files(
            routing_rules_path=routing_rules_path or project_root / "config" / "routing_rules.yaml",
            model_prices_path=model_prices_path or project_root / "config" / "model_prices.yaml",
            policy_config_path=default_policy_config,
            policy_name=policy_name,
            input_dir=input_dir,
            include_files=include_file_names,
            expected_frames_per_video=max_keyframes or DEFAULT_MAX_KEYFRAMES,
            historical_model_calls_paths=historical_model_calls_paths,
            ocr_backend=ocr_backend_override or _pipeline_backend(settings, "image", "ocr", "ocr_backend"),
            vision_understanding_backend=vision_understanding_backend_override
            or _pipeline_backend(settings, "image", "vision_understanding", "vision_understanding_backend"),
            speech_to_text_backend=speech_to_text_backend_override
            or _pipeline_backend(settings, "video", "speech_to_text", "speech_to_text_backend"),
            text_analysis_backend=text_analysis_backend_override
            or _pipeline_backend(settings, "text", "text_analysis", "text_analysis_backend"),
            generated_at=generated_at,
        )
    paths = write_preflight_reports(output_dir / current_batch_id, report)
    return {
        "batch_id": current_batch_id,
        "preflight_status": report["preflight_status"],
        "recommended_action": report["recommended_action"],
        "report_paths": paths,
    }


def run_batch(
    *,
    settings_path: str | Path = PROJECT_ROOT / "config" / "settings.yaml",
    routing_rules_path: str | Path | None = None,
    model_prices_path: str | Path | None = None,
    input_dir_override: str | Path | None = None,
    ocr_backend_override: str | None = None,
    vision_understanding_backend_override: str | None = None,
    speech_to_text_backend_override: str | None = None,
    text_analysis_backend_override: str | None = None,
    defer_text_analysis: bool = False,
    allow_live_api: bool = False,
    deepseek_max_retries: int = 0,
    qwen_vl_max_retries: int = 0,
    batch_id: str | None = None,
    created_at: str | None = None,
    generated_at: str | None = None,
    include_file_names: list[str] | None = None,
    ffmpeg_path: str | Path | None = None,
    asr_audio_url_map_path: str | Path | None = None,
    max_keyframes: int | None = None,
    route_plan_path: str | Path | None = None,
) -> dict[str, Any]:
    """运行一次批处理。"""

    settings_file = Path(settings_path)
    project_root = settings_file.resolve().parents[1]
    settings = load_settings(settings_file)
    route_plan = None
    if route_plan_path is not None:
        if routing_rules_path is not None:
            raise ValueError("使用路由计划时不能同时指定旧 routing_rules_path。")
        if any(
            value is not None
            for value in (
                ocr_backend_override,
                vision_understanding_backend_override,
                speech_to_text_backend_override,
                text_analysis_backend_override,
            )
        ):
            raise ValueError("使用 --route-plan 时不能同时指定后端覆盖参数。")
        route_plan = load_route_plan(_resolve_path(project_root, route_plan_path))
        validate_route_plan(route_plan, settings)
    routing_rules = (
        {}
        if route_plan is not None
        else load_routing_rules(routing_rules_path or project_root / "config" / "routing_rules.yaml")
    )
    model_prices = load_model_prices(model_prices_path or project_root / "config" / "model_prices.yaml")

    current_batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_created_at = created_at or _now_iso()
    input_dir = _resolve_path(project_root, input_dir_override or _setting(settings, "paths.input_dir", "input_dir"))
    output_dir = _resolve_path(project_root, _setting(settings, "paths.output_dir", "output_dir"))
    file_manifest = build_file_manifest(input_dir, current_batch_id, created_at=batch_created_at)
    file_manifest = _filter_manifest_by_file_names(file_manifest, include_file_names)
    media_types = {str(record["media_type"]) for record in file_manifest}
    selected_pipelines = (
        {media_type: route_plan_backends_for_media(route_plan, media_type) for media_type in media_types}
        if route_plan is not None
        else {
            media_type: _pipeline_selection(
                settings,
                media_type,
                ocr_override=ocr_backend_override,
                vision_override=vision_understanding_backend_override,
                speech_override=speech_to_text_backend_override,
                text_override=text_analysis_backend_override,
            )
            for media_type in media_types
        }
    )
    active_selections = list(selected_pipelines.values())
    ocr_backends = {selection["ocr_backend"] for selection in active_selections}
    vision_backends = {selection["vision_understanding_backend"] for selection in active_selections}
    speech_backends = {selection["speech_to_text_backend"] for selection in active_selections}
    text_backends = {selection["text_analysis_backend"] for selection in active_selections}
    deepseek_max_tokens = _positive_int_value(
        _backend_setting(settings, "text_analysis", "deepseek", "max_tokens", "deepseek_max_tokens"),
        "deepseek_max_tokens",
        DEFAULT_DEEPSEEK_MAX_TOKENS,
    )
    deepseek_compact_mode = _bool_value(
        _backend_setting(settings, "text_analysis", "deepseek", "compact_mode", "deepseek_compact_mode", False),
        "deepseek_compact_mode",
    )
    text_analysis_evidence_char_limit = _positive_int_value(
        _backend_setting(
            settings,
            "text_analysis",
            "deepseek",
            "evidence_char_limit",
            "text_analysis_evidence_char_limit",
        ),
        "text_analysis_evidence_char_limit",
        DEFAULT_TEXT_ANALYSIS_EVIDENCE_CHAR_LIMIT,
    )
    qwen_vl_max_tokens = _positive_int_value(
        _backend_setting(settings, "vision_understanding", "qwen_vl", "max_tokens", "qwen_vl_max_tokens"),
        "qwen_vl_max_tokens",
        DEFAULT_QWEN_VL_MAX_TOKENS,
    )
    qwen_vl_max_image_side = _positive_int_value(
        _backend_setting(settings, "vision_understanding", "qwen_vl", "max_image_side", "qwen_vl_max_image_side"),
        "qwen_vl_max_image_side",
        DEFAULT_QWEN_VL_MAX_IMAGE_SIDE,
    )
    qwen_ocr_max_tokens = _positive_int_value(
        _backend_setting(settings, "ocr", "qwen_ocr", "max_tokens", "qwen_ocr_max_tokens"),
        "qwen_ocr_max_tokens",
        DEFAULT_QWEN_OCR_MAX_TOKENS,
    )
    qwen_ocr_max_image_side_value = _backend_setting(
        settings, "ocr", "qwen_ocr", "max_image_side", "qwen_ocr_max_image_side"
    )
    qwen_ocr_max_image_side = (
        None
        if qwen_ocr_max_image_side_value in {None, ""}
        else _positive_int_value(qwen_ocr_max_image_side_value, "qwen_ocr_max_image_side", 1)
    )
    for backend in ocr_backends:
        if backend not in ALLOWED_OCR_BACKENDS:
            raise ValueError(f"不支持的 OCR 后端：{backend}")
    for backend in vision_backends:
        if backend not in ALLOWED_VISION_BACKENDS:
            raise ValueError(f"不支持的视觉理解后端：{backend}")
    for backend in speech_backends:
        if backend not in ALLOWED_SPEECH_BACKENDS:
            raise ValueError(f"不支持的语音识别后端：{backend}")
    for backend in text_backends:
        if backend not in ALLOWED_TEXT_BACKENDS:
            raise ValueError(f"不支持的文本分析后端：{backend}")
    if "deepseek" in text_backends and not defer_text_analysis and not allow_live_api:
        raise PermissionError(
            "本次运行将访问 DeepSeek API。请显式选择 DeepSeek 后端并指定 --allow-live-api。"
        )
    if "qwen_ocr" in ocr_backends and not allow_live_api:
        raise PermissionError(
            "本次运行将访问 Qwen3.5-OCR API。请显式选择 qwen_ocr 后端并指定 --allow-live-api。"
        )
    if "qwen_vl" in vision_backends and not allow_live_api:
        raise PermissionError(
            "本次运行将访问 Qwen-VL API。请显式选择 Qwen-VL 后端并指定 --allow-live-api。"
        )
    if "dashscope_asr" in speech_backends and not allow_live_api:
        raise PermissionError(
            "本次运行将访问 DashScope ASR API。请显式选择 DashScope ASR 后端并指定 --allow-live-api。"
        )
    if deepseek_max_retries not in {0, 1}:
        raise ValueError("DeepSeek 最大重试次数只能是 0 或 1。")
    if deepseek_max_retries and ("deepseek" not in text_backends or defer_text_analysis):
        raise ValueError("只有显式选择 DeepSeek 后端时才能启用 API 重试。")
    if qwen_vl_max_retries not in {0, 1}:
        raise ValueError("Qwen-VL 最大重试次数只能是 0 或 1。")
    if qwen_vl_max_retries and "qwen_vl" not in vision_backends:
        raise ValueError("只有显式选择 Qwen-VL 后端时才能启用 Qwen-VL API 重试。")
    effective_max_keyframes = max_keyframes if max_keyframes is not None else DEFAULT_MAX_KEYFRAMES
    if effective_max_keyframes < 1:
        raise ValueError("视频关键帧数量必须大于等于 1。")

    deepseek_api_key_env = _backend_setting(
        settings, "text_analysis", "deepseek", "api_key_env", "deepseek_api_key_env", "DEEPSEEK_API_KEY"
    )
    deepseek_api_key = os.environ.get(deepseek_api_key_env) if "deepseek" in text_backends and not defer_text_analysis else None
    if "deepseek" in text_backends and not defer_text_analysis and not deepseek_api_key:
        raise RuntimeError(f"未读取到环境变量 {deepseek_api_key_env}，已在发送网络请求前停止运行。")
    qwen_vl_api_key_env = _backend_setting(
        settings, "vision_understanding", "qwen_vl", "api_key_env", "qwen_vl_api_key_env", "DASHSCOPE_API_KEY"
    )
    qwen_vl_api_key = (
        os.environ.get(qwen_vl_api_key_env)
        if "qwen_vl" in vision_backends
        else None
    )
    if "qwen_vl" in vision_backends and not qwen_vl_api_key:
        raise RuntimeError(f"未读取到环境变量 {qwen_vl_api_key_env}，已在发送网络请求前停止运行。")
    qwen_ocr_api_key_env = _backend_setting(
        settings, "ocr", "qwen_ocr", "api_key_env", "qwen_ocr_api_key_env", "DASHSCOPE_API_KEY"
    )
    qwen_ocr_api_key = os.environ.get(qwen_ocr_api_key_env) if "qwen_ocr" in ocr_backends else None
    if "qwen_ocr" in ocr_backends and not qwen_ocr_api_key:
        raise RuntimeError(f"未读取到环境变量 {qwen_ocr_api_key_env}，已在发送网络请求前停止运行。")
    dashscope_asr_api_key_env = _backend_setting(
        settings, "speech_to_text", "dashscope_asr", "api_key_env", "dashscope_asr_api_key_env", "DASHSCOPE_API_KEY"
    )
    dashscope_asr_api_key = os.environ.get(dashscope_asr_api_key_env) if "dashscope_asr" in speech_backends else None
    if "dashscope_asr" in speech_backends and not dashscope_asr_api_key:
        raise RuntimeError(f"未读取到环境变量 {dashscope_asr_api_key_env}，已在发送网络请求前停止运行。")
    asr_audio_url_map = _load_asr_audio_url_map(project_root, asr_audio_url_map_path) if "dashscope_asr" in speech_backends else None

    if "paddleocr" in ocr_backends:
        _ensure_paddleocr_runtime_available()
    results: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    batch_dir = output_dir / current_batch_id

    for file_record in file_manifest:
        file_backends = selected_pipelines[str(file_record["media_type"])]
        pipeline_output = run_file_pipeline(
            file_record,
            routing_rules,
            model_prices,
            route_plan=route_plan,
            ocr_backend=file_backends["ocr_backend"],
            vision_understanding_backend=file_backends["vision_understanding_backend"],
            speech_to_text_backend=file_backends["speech_to_text_backend"],
            text_analysis_backend=file_backends["text_analysis_backend"],
            defer_text_analysis=defer_text_analysis,
            deepseek_api_key=deepseek_api_key,
            deepseek_base_url=_backend_setting(
                settings, "text_analysis", "deepseek", "base_url", "deepseek_base_url", "https://api.deepseek.com"
            ),
            deepseek_model_name=_backend_setting(
                settings, "text_analysis", "deepseek", "model_name", "deepseek_model_name", "deepseek-v4-flash"
            ),
            deepseek_max_retries=deepseek_max_retries,
            deepseek_max_tokens=deepseek_max_tokens,
            deepseek_compact_mode=deepseek_compact_mode,
            text_analysis_evidence_char_limit=text_analysis_evidence_char_limit,
            qwen_ocr_api_key=qwen_ocr_api_key,
            qwen_ocr_base_url=_backend_setting(
                settings,
                "ocr",
                "qwen_ocr",
                "base_url",
                "qwen_ocr_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            qwen_ocr_model_name=_backend_setting(
                settings, "ocr", "qwen_ocr", "model_name", "qwen_ocr_model_name", DEFAULT_QWEN_OCR_MODEL_NAME
            ),
            qwen_ocr_max_tokens=qwen_ocr_max_tokens,
            qwen_ocr_max_image_side=qwen_ocr_max_image_side,
            qwen_vl_api_key=qwen_vl_api_key,
            qwen_vl_base_url=_backend_setting(
                settings,
                "vision_understanding",
                "qwen_vl",
                "base_url",
                "qwen_vl_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            qwen_vl_model_name=_backend_setting(
                settings, "vision_understanding", "qwen_vl", "model_name", "qwen_vl_model_name", "qwen-vl-plus"
            ),
            qwen_vl_max_retries=qwen_vl_max_retries,
            qwen_vl_max_tokens=qwen_vl_max_tokens,
            qwen_vl_max_image_side=qwen_vl_max_image_side,
            dashscope_asr_api_key=dashscope_asr_api_key,
            dashscope_asr_submit_url=_backend_setting(
                settings,
                "speech_to_text",
                "dashscope_asr",
                "submit_url",
                "dashscope_asr_submit_url",
                "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
            ),
            dashscope_asr_model_name=_backend_setting(
                settings, "speech_to_text", "dashscope_asr", "model_name", "dashscope_asr_model_name", "paraformer-v2"
            ),
            asr_audio_url_map=asr_audio_url_map,
            preprocess_artifact_dir=batch_dir / "preprocess_artifacts" / file_record["file_id"],
            ffmpeg_path=ffmpeg_path,
            max_keyframes=effective_max_keyframes,
        )
        results.append(pipeline_output["result"])
        model_calls.extend(pipeline_output["model_calls"])
        errors.extend(pipeline_output["errors"])

    backend_runtime_summary = _build_backend_runtime_summary(model_calls)
    batch_metadata = {
        "schema_version": "v1",
        "batch_id": current_batch_id,
        "created_by": settings.get("created_by", "local_user"),
        "team_name": settings.get("team_name", "content_ai_team"),
        "request_purpose": settings.get("request_purpose")
        or _default_request_purpose(backend_runtime_summary),
        "created_at": batch_created_at,
        "budget_limit_cny": _setting(settings, "runtime.default_budget_limit_cny", "default_budget_limit_cny"),
        "allow_partial_success": _setting(settings, "runtime.allow_partial_success", "allow_partial_success"),
        "target_output_format": _setting(settings, "runtime.target_output_format", "target_output_format"),
        "video_max_keyframes": effective_max_keyframes,
        "deepseek_compact_mode": deepseek_compact_mode,
        "text_analysis_evidence_char_limit": text_analysis_evidence_char_limit,
        "text_analysis_execution_mode": "deferred" if defer_text_analysis else "synchronous",
        "qwen_vl_max_image_side": qwen_vl_max_image_side,
        "qwen_ocr_max_image_side": qwen_ocr_max_image_side,
        "selected_backends": {
            "ocr_backend": next(iter(ocr_backends)) if len(ocr_backends) == 1 else "mixed_by_media",
            "vision_understanding_backend": (
                next(iter(vision_backends)) if len(vision_backends) == 1 else "mixed_by_media"
            ),
            "speech_to_text_backend": next(iter(speech_backends)) if len(speech_backends) == 1 else "mixed_by_media",
            "text_analysis_backend": next(iter(text_backends)) if len(text_backends) == 1 else "mixed_by_media",
        },
        "selected_pipelines": selected_pipelines,
        "route_plan_source": str(_resolve_path(project_root, route_plan_path)) if route_plan_path is not None else None,
        "route_plan": route_plan,
        "backend_runtime_summary": backend_runtime_summary,
        "cost_estimation": {
            "currency": "CNY",
            "price_config_path": "config/model_prices.yaml",
            "method": "按 input_units / output_units 与本地价格表逐项相乘后汇总",
            "contains_mock_estimates": backend_runtime_summary["contains_mock"],
            "bill_reconciled": False,
            "estimation_error_status": "unknown_until_bill_reconciliation",
            "note": "真实 API 成本依赖供应商返回用量和本地价格配置；未与供应商账单对账前，整体估算误差未知。mock 成本只用于流程占位，不代表真实账单。",
        },
    }

    write_batch_metadata(output_dir, current_batch_id, batch_metadata)
    write_results(output_dir, current_batch_id, results)
    write_results_readable(output_dir, current_batch_id, results)
    write_model_calls(output_dir, current_batch_id, model_calls)
    write_errors(output_dir, current_batch_id, errors)

    batch_report = generate_batch_report(
        batch_id=current_batch_id,
        results=results,
        model_calls=model_calls,
        errors=errors,
        budget_limit_cny=float(_setting(settings, "runtime.default_budget_limit_cny", "default_budget_limit_cny")),
        generated_at=generated_at,
    )
    batch_dir = ensure_batch_output_dir(output_dir, current_batch_id)
    write_json(batch_dir / "batch_report.json", batch_report)

    return {
        "batch_id": current_batch_id,
        "batch_dir": str(batch_dir),
        "total_files": len(results),
        "total_model_calls": len(model_calls),
        "total_errors": len(errors),
    }


def _parse_include_files_arg(value: str) -> list[str]:
    """解析逗号分隔的文件名列表。"""

    file_names = [file_name.strip() for file_name in value.split(",") if file_name.strip()]
    if not file_names:
        raise argparse.ArgumentTypeError("文件名列表不能为空。")
    return file_names


def _parse_path_list_arg(value: str) -> list[str]:
    """解析逗号分隔的路径列表。"""

    paths = [path.strip() for path in value.split(",") if path.strip()]
    if not paths:
        raise argparse.ArgumentTypeError("路径列表不能为空。")
    return paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行多模态批处理或受控评估输入批处理。")
    parser.add_argument(
        "--settings",
        default=str(PROJECT_ROOT / "config" / "settings.yaml"),
        help="配置文件路径，默认使用 config/settings.yaml。",
    )
    parser.add_argument(
        "--input-dir",
        help="显式指定本次输入目录；例如 evaluation/text_topic_small_set。",
    )
    parser.add_argument(
        "--include-files",
        type=_parse_include_files_arg,
        help="只处理指定文件名，多个文件用英文逗号分隔；例如 img_7.jpg,img_8.jpg,img_9.jpg。",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=ALLOWED_OCR_BACKENDS,
        help="显式指定图片或视频关键帧 OCR 后端；PaddleOCR 在本地运行，不需要 API 密钥。",
    )
    parser.add_argument(
        "--vision-backend",
        choices=ALLOWED_VISION_BACKENDS,
        help="显式指定图片或视频关键帧视觉理解后端；Qwen-VL 需要 API Key 和 --allow-live-api。",
    )
    parser.add_argument(
        "--speech-backend",
        choices=ALLOWED_SPEECH_BACKENDS,
        help="显式指定视频语音识别后端；DashScope ASR 需要 API Key、DashScope SDK/CLI 和 --allow-live-api。",
    )
    parser.add_argument(
        "--asr-audio-url-map",
        help="可选 JSON 文件路径，内容为文件名、file_id 或音频文件名到远端音频 URL 的映射；不提供时会自动上传本地音频到 DashScope 临时存储。",
    )
    parser.add_argument(
        "--text-analysis-backend",
        choices=ALLOWED_TEXT_BACKENDS,
        help="显式指定文本分析后端；评估流程离线验证时建议使用 mock。",
    )
    parser.add_argument(
        "--defer-text-analysis",
        action="store_true",
        help="只完成上游证据处理并写出 pending 结果，不调用文本模型。",
    )
    parser.add_argument(
        "--batch-id",
        help="显式指定批次 ID，便于复现实验输出。",
    )
    parser.add_argument(
        "--allow-live-api",
        action="store_true",
        help="明确允许本次运行访问 DeepSeek 或 Qwen-VL API；必须同时显式选择真实后端。",
    )
    parser.add_argument(
        "--max-api-retries",
        type=int,
        choices=[0, 1],
        default=0,
        help="DeepSeek 或 Qwen-VL 可重试错误的最大重试次数；默认 0，显式设置为 1 才允许重试一次。",
    )
    parser.add_argument(
        "--ffmpeg-path",
        help="显式指定 ffmpeg.exe 或 ffmpeg 所在目录；用于绕过 Windows PATH 未刷新的问题。",
    )
    parser.add_argument(
        "--max-keyframes",
        type=int,
        default=3,
        help="每个视频最多抽取多少张关键帧；默认 3，用于降低 OCR 和 Qwen-VL 调用次数。",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只生成运行前路由预检查报告，不执行批处理，不触发任何模型 API。",
    )
    parser.add_argument(
        "--preflight-policy",
        default="balanced",
        help="运行前路由预检查使用的策略名称，默认 balanced。",
    )
    parser.add_argument(
        "--historical-model-calls",
        type=_parse_path_list_arg,
        help="用于预检查延迟画像的历史 model_calls.jsonl 路径，多个路径用英文逗号分隔。",
    )
    parser.add_argument(
        "--route-plan",
        help="显式执行预检查生成的 route_plan.json；不能与后端覆盖参数同时使用。",
    )
    parser.add_argument(
        "--route-decision-report",
        help="预检查时读取候选对照报告并将推荐写入路由计划；只与 --preflight-only 一起使用。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = _parse_args(argv)
    real_api_backend_selected = (
        args.route_plan is not None
        or (args.text_analysis_backend == "deepseek" and not args.defer_text_analysis)
        or args.ocr_backend == "qwen_ocr"
        or args.vision_backend == "qwen_vl"
        or args.speech_backend == "dashscope_asr"
    )
    if args.route_plan and any(
        value is not None
        for value in (args.ocr_backend, args.vision_backend, args.speech_backend, args.text_analysis_backend)
    ):
        print("错误：--route-plan 不能与 --ocr-backend、--vision-backend、--speech-backend 或 --text-analysis-backend 同时使用。")
        return 2
    if args.route_plan and args.max_api_retries:
        print("错误：当前路由计划执行不接受 --max-api-retries；请在计划闭环稳定后再单独设计重试策略。")
        return 2
    if args.route_decision_report and not args.preflight_only:
        print("错误：--route-decision-report 只用于 --preflight-only 生成路由计划。")
        return 2
    if args.route_decision_report and args.text_analysis_backend:
        print("错误：--route-decision-report 不能与 --text-analysis-backend 同时使用。")
        return 2
    if args.allow_live_api and not real_api_backend_selected:
        print("错误：--allow-live-api 必须与 --ocr-backend qwen_ocr、--text-analysis-backend deepseek、--vision-backend qwen_vl 或 --speech-backend dashscope_asr 同时使用。")
        return 2
    if args.max_api_retries and args.text_analysis_backend != "deepseek" and args.vision_backend != "qwen_vl":
        print("错误：--max-api-retries 只能与 --text-analysis-backend deepseek 或 --vision-backend qwen_vl 同时使用。")
        return 2
    if args.max_api_retries and args.defer_text_analysis and args.vision_backend != "qwen_vl":
        print("错误：延后文本分析时不能为文本后端设置重试。")
        return 2

    try:
        if args.preflight_only:
            summary = run_preflight(
                settings_path=args.settings,
                input_dir_override=args.input_dir,
                ocr_backend_override=args.ocr_backend,
                vision_understanding_backend_override=args.vision_backend,
                speech_to_text_backend_override=args.speech_backend,
                text_analysis_backend_override=args.text_analysis_backend,
                batch_id=args.batch_id,
                policy_name=args.preflight_policy,
                include_file_names=args.include_files,
                historical_model_calls_paths=args.historical_model_calls,
                max_keyframes=args.max_keyframes,
                route_decision_report_path=args.route_decision_report,
            )
            print(summary)
            return 0

        summary = run_batch(
            settings_path=args.settings,
            input_dir_override=args.input_dir,
            ocr_backend_override=None
            if args.route_plan
            else args.ocr_backend or ("mock" if args.allow_live_api else None),
            vision_understanding_backend_override=None
            if args.route_plan
            else args.vision_backend or ("mock" if args.allow_live_api else None),
            speech_to_text_backend_override=None
            if args.route_plan
            else args.speech_backend or ("mock" if args.allow_live_api else None),
            text_analysis_backend_override=None
            if args.route_plan
            else args.text_analysis_backend or ("mock" if args.allow_live_api or args.ocr_backend == "paddleocr" else None),
            defer_text_analysis=args.defer_text_analysis,
            allow_live_api=args.allow_live_api,
            deepseek_max_retries=args.max_api_retries if args.text_analysis_backend == "deepseek" else 0,
            qwen_vl_max_retries=args.max_api_retries if args.vision_backend == "qwen_vl" else 0,
            batch_id=args.batch_id,
            include_file_names=args.include_files,
            ffmpeg_path=args.ffmpeg_path,
            asr_audio_url_map_path=args.asr_audio_url_map,
            max_keyframes=args.max_keyframes,
            route_plan_path=args.route_plan,
        )
    except (PermissionError, RuntimeError, ValueError) as error:
        print(f"错误：{error}")
        return 2

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
