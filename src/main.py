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
from model_clients import DEFAULT_DEEPSEEK_MAX_TOKENS, DEFAULT_QWEN_VL_MAX_TOKENS
from model_router import load_routing_rules
from pipeline_runner import run_file_pipeline
from preprocessor import DEFAULT_MAX_KEYFRAMES
from report_generator import generate_batch_report
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


def _now_iso() -> str:
    """返回当前本地时间的 ISO 字符串。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_settings(settings_path: str | Path) -> dict[str, Any]:
    """读取运行配置文件。"""

    path = Path(settings_path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_path(project_root: Path, path_value: str | Path) -> Path:
    """把配置中的相对路径转换为项目内绝对路径。"""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


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
) -> dict[str, Any]:
    """运行一次批处理。"""

    settings_file = Path(settings_path)
    project_root = settings_file.resolve().parents[1]
    settings = load_settings(settings_file)
    routing_rules = load_routing_rules(routing_rules_path or project_root / "config" / "routing_rules.yaml")
    model_prices = load_model_prices(model_prices_path or project_root / "config" / "model_prices.yaml")

    current_batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_created_at = created_at or _now_iso()
    input_dir = _resolve_path(project_root, input_dir_override or settings["input_dir"])
    output_dir = _resolve_path(project_root, settings["output_dir"])
    ocr_backend = ocr_backend_override or settings.get("ocr_backend", "mock")
    vision_understanding_backend = (
        vision_understanding_backend_override
        or settings.get("vision_understanding_backend", "mock")
    )
    speech_to_text_backend = speech_to_text_backend_override or settings.get("speech_to_text_backend", "mock")
    text_analysis_backend = text_analysis_backend_override or settings.get("text_analysis_backend", "mock")
    deepseek_max_tokens = _positive_int_setting(settings, "deepseek_max_tokens", DEFAULT_DEEPSEEK_MAX_TOKENS)
    qwen_vl_max_tokens = _positive_int_setting(settings, "qwen_vl_max_tokens", DEFAULT_QWEN_VL_MAX_TOKENS)
    if ocr_backend not in ALLOWED_OCR_BACKENDS:
        raise ValueError(f"不支持的 OCR 后端：{ocr_backend}")
    if vision_understanding_backend not in ALLOWED_VISION_BACKENDS:
        raise ValueError(f"不支持的视觉理解后端：{vision_understanding_backend}")
    if speech_to_text_backend not in ALLOWED_SPEECH_BACKENDS:
        raise ValueError(f"不支持的语音识别后端：{speech_to_text_backend}")
    if text_analysis_backend not in ALLOWED_TEXT_BACKENDS:
        raise ValueError(f"不支持的文本分析后端：{text_analysis_backend}")
    if text_analysis_backend == "deepseek" and not allow_live_api:
        raise PermissionError(
            "本次运行将访问 DeepSeek API。请显式选择 DeepSeek 后端并指定 --allow-live-api。"
        )
    if vision_understanding_backend == "qwen_vl" and not allow_live_api:
        raise PermissionError(
            "本次运行将访问 Qwen-VL API。请显式选择 Qwen-VL 后端并指定 --allow-live-api。"
        )
    if speech_to_text_backend == "dashscope_asr" and not allow_live_api:
        raise PermissionError(
            "本次运行将访问 DashScope ASR API。请显式选择 DashScope ASR 后端并指定 --allow-live-api。"
        )
    if deepseek_max_retries not in {0, 1}:
        raise ValueError("DeepSeek 最大重试次数只能是 0 或 1。")
    if deepseek_max_retries and text_analysis_backend != "deepseek":
        raise ValueError("只有显式选择 DeepSeek 后端时才能启用 API 重试。")
    if qwen_vl_max_retries not in {0, 1}:
        raise ValueError("Qwen-VL 最大重试次数只能是 0 或 1。")
    if qwen_vl_max_retries and vision_understanding_backend != "qwen_vl":
        raise ValueError("只有显式选择 Qwen-VL 后端时才能启用 Qwen-VL API 重试。")
    effective_max_keyframes = max_keyframes if max_keyframes is not None else DEFAULT_MAX_KEYFRAMES
    if effective_max_keyframes < 1:
        raise ValueError("视频关键帧数量必须大于等于 1。")

    deepseek_api_key_env = settings.get("deepseek_api_key_env", "DEEPSEEK_API_KEY")
    deepseek_api_key = os.environ.get(deepseek_api_key_env) if text_analysis_backend == "deepseek" else None
    if text_analysis_backend == "deepseek" and not deepseek_api_key:
        raise RuntimeError(f"未读取到环境变量 {deepseek_api_key_env}，已在发送网络请求前停止运行。")
    qwen_vl_api_key_env = settings.get("qwen_vl_api_key_env", "DASHSCOPE_API_KEY")
    qwen_vl_api_key = (
        os.environ.get(qwen_vl_api_key_env)
        if vision_understanding_backend == "qwen_vl"
        else None
    )
    if vision_understanding_backend == "qwen_vl" and not qwen_vl_api_key:
        raise RuntimeError(f"未读取到环境变量 {qwen_vl_api_key_env}，已在发送网络请求前停止运行。")
    dashscope_asr_api_key_env = settings.get("dashscope_asr_api_key_env", "DASHSCOPE_API_KEY")
    dashscope_asr_api_key = os.environ.get(dashscope_asr_api_key_env) if speech_to_text_backend == "dashscope_asr" else None
    if speech_to_text_backend == "dashscope_asr" and not dashscope_asr_api_key:
        raise RuntimeError(f"未读取到环境变量 {dashscope_asr_api_key_env}，已在发送网络请求前停止运行。")
    asr_audio_url_map = _load_asr_audio_url_map(project_root, asr_audio_url_map_path) if speech_to_text_backend == "dashscope_asr" else None

    if ocr_backend == "paddleocr":
        _ensure_paddleocr_runtime_available()

    file_manifest = build_file_manifest(input_dir, current_batch_id, created_at=batch_created_at)
    file_manifest = _filter_manifest_by_file_names(file_manifest, include_file_names)
    results: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    batch_dir = output_dir / current_batch_id

    for file_record in file_manifest:
        pipeline_output = run_file_pipeline(
            file_record,
            routing_rules,
            model_prices,
            ocr_backend=ocr_backend,
            vision_understanding_backend=vision_understanding_backend,
            speech_to_text_backend=speech_to_text_backend,
            text_analysis_backend=text_analysis_backend,
            deepseek_api_key=deepseek_api_key,
            deepseek_base_url=settings.get("deepseek_base_url", "https://api.deepseek.com"),
            deepseek_model_name=settings.get("deepseek_model_name", "deepseek-v4-flash"),
            deepseek_max_retries=deepseek_max_retries,
            deepseek_max_tokens=deepseek_max_tokens,
            qwen_vl_api_key=qwen_vl_api_key,
            qwen_vl_base_url=settings.get(
                "qwen_vl_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            qwen_vl_model_name=settings.get("qwen_vl_model_name", "qwen-vl-plus"),
            qwen_vl_max_retries=qwen_vl_max_retries,
            qwen_vl_max_tokens=qwen_vl_max_tokens,
            dashscope_asr_api_key=dashscope_asr_api_key,
            dashscope_asr_submit_url=settings.get(
                "dashscope_asr_submit_url",
                "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
            ),
            dashscope_asr_model_name=settings.get("dashscope_asr_model_name", "paraformer-v2"),
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
        "budget_limit_cny": settings["default_budget_limit_cny"],
        "allow_partial_success": settings["allow_partial_success"],
        "target_output_format": settings["target_output_format"],
        "video_max_keyframes": effective_max_keyframes,
        "selected_backends": {
            "ocr_backend": ocr_backend,
            "vision_understanding_backend": vision_understanding_backend,
            "speech_to_text_backend": speech_to_text_backend,
            "text_analysis_backend": text_analysis_backend,
        },
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
        budget_limit_cny=float(settings["default_budget_limit_cny"]),
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = _parse_args(argv)
    real_api_backend_selected = (
        args.text_analysis_backend == "deepseek"
        or args.vision_backend == "qwen_vl"
        or args.speech_backend == "dashscope_asr"
    )
    if args.allow_live_api and not real_api_backend_selected:
        print("错误：--allow-live-api 必须与 --text-analysis-backend deepseek、--vision-backend qwen_vl 或 --speech-backend dashscope_asr 同时使用。")
        return 2
    if args.max_api_retries and args.text_analysis_backend != "deepseek" and args.vision_backend != "qwen_vl":
        print("错误：--max-api-retries 只能与 --text-analysis-backend deepseek 或 --vision-backend qwen_vl 同时使用。")
        return 2

    try:
        summary = run_batch(
            settings_path=args.settings,
            input_dir_override=args.input_dir,
            ocr_backend_override=args.ocr_backend or ("mock" if args.allow_live_api else None),
            vision_understanding_backend_override=args.vision_backend
            or ("mock" if args.allow_live_api else None),
            speech_to_text_backend_override=args.speech_backend
            or ("mock" if args.allow_live_api else None),
            text_analysis_backend_override=args.text_analysis_backend
            or ("mock" if args.allow_live_api or args.ocr_backend == "paddleocr" else None),
            allow_live_api=args.allow_live_api,
            deepseek_max_retries=args.max_api_retries if args.text_analysis_backend == "deepseek" else 0,
            qwen_vl_max_retries=args.max_api_retries if args.vision_backend == "qwen_vl" else 0,
            batch_id=args.batch_id,
            include_file_names=args.include_files,
            ffmpeg_path=args.ffmpeg_path,
            asr_audio_url_map_path=args.asr_audio_url_map,
            max_keyframes=args.max_keyframes,
        )
    except (PermissionError, RuntimeError, ValueError) as error:
        print(f"错误：{error}")
        return 2

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
