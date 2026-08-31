"""只重跑已有批次的文本分析层，不重新调用 OCR 或视觉理解模型。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from cost_latency_tracker import build_model_call_record, load_model_prices
from model_clients import (
    DEFAULT_DEEPSEEK_MAX_TOKENS,
    DEFAULT_QWEN_TEXT_MAX_TOKENS,
    DeepSeekAttemptsExhausted,
    QwenTextAttemptsExhausted,
    deepseek_text_analysis_client,
    qwen_text_analysis_client,
)
from pipeline_runner import _analysis_input_units, _analysis_output_units, _models_used_from_calls, _now_iso
from report_generator import generate_batch_report, read_jsonl
from result_writer import (
    ensure_batch_output_dir,
    write_batch_metadata,
    write_errors,
    write_json,
    write_model_calls,
    write_results,
    write_results_readable,
)
from runtime_config import runtime_policy_section


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_ANALYSIS_EXECUTION_POLICY = runtime_policy_section("text_analysis_execution")
DEFERRED_TEXT_QUALITY_FLAG = str(TEXT_ANALYSIS_EXECUTION_POLICY["deferred_quality_flag"])
DEFERRED_TEXT_WARNING = str(TEXT_ANALYSIS_EXECUTION_POLICY["deferred_warning"])


def _load_settings(settings_path: str | Path) -> dict[str, Any]:
    """读取运行配置。"""

    return yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))


def _setting(settings: dict[str, Any], dotted_key: str, legacy_key: str | None = None, default: Any = None) -> Any:
    """读取新分层配置；缺失时兼容旧扁平配置。"""

    current: Any = settings
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return settings.get(legacy_key, default) if legacy_key else default
        current = current[part]
    return current


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


def _resolve_path(path_value: str | Path) -> Path:
    """把相对路径转换为项目内路径。"""

    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_reanalysis_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """从历史结果中提取文本分析证据；mock 音频转写不进入真实重分析。"""

    audio_transcript = record.get("audio_transcript")
    if isinstance(audio_transcript, str) and audio_transcript.startswith("模拟音频转写"):
        audio_transcript = None
    return {
        "raw_text": record.get("raw_text"),
        "ocr_text": record.get("ocr_text"),
        "visual_description": record.get("visual_description"),
        "audio_transcript": audio_transcript,
    }


def _evidence_text(evidence: dict[str, Any]) -> str:
    """拼接证据文本，用于 token 粗估兜底。"""

    return " ".join(str(value) for value in evidence.values() if value)


def _parse_include_files(value: str | None) -> set[str]:
    """解析只处理的文件名或文件 ID。"""

    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _positive_int(value: int, name: str) -> int:
    """校验命令行传入的正整数。"""

    if value < 1:
        raise ValueError(f"{name} 必须是大于等于 1 的整数。")
    return value


def _bool_setting(settings: dict[str, Any], key: str, default: bool) -> bool:
    """读取布尔配置。"""

    value = settings.get(key, default)
    return _bool_value(value, key)


def _bool_value(value: Any, name: str) -> bool:
    """校验布尔配置值。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} 必须是 true 或 false。")


def _filter_records(source_records: list[dict[str, Any]], include_files: set[str]) -> list[dict[str, Any]]:
    """按文件名或文件 ID 过滤历史结果。"""

    if not include_files:
        return source_records
    return [
        record
        for record in source_records
        if str(record.get("file_name")) in include_files or str(record.get("file_id")) in include_files
    ]


def _attempt_call_id(call_id: str, attempt_index: int, total_attempts: int) -> str:
    """单次尝试的调用 ID；只有多次尝试时才加后缀。"""

    return call_id if total_attempts == 1 else f"{call_id}_attempt_{attempt_index:02d}"


def _build_call(
    *,
    call_id: str,
    batch_id: str,
    record: dict[str, Any],
    model_name: str,
    provider: str,
    input_units: list[dict[str, Any]],
    output_units: list[dict[str, Any]],
    latency_ms: int,
    status: str,
    error_message: str | None,
    model_prices: dict[str, dict[str, Any]],
    response_model_name: str | None = None,
) -> dict[str, Any]:
    """生成本轮文本重分析的模型调用记录。"""

    return build_model_call_record(
        call_id=call_id,
        batch_id=batch_id,
        file_id=str(record["file_id"]),
        task_type="text_analysis",
        provider=provider,
        model_name=model_name,
        input_units=input_units,
        output_units=output_units,
        latency_ms=latency_ms,
        started_at=_now_iso(),
        status=status,
        error_message=error_message,
        model_prices=model_prices,
        response_model_name=response_model_name,
    )


def reanalyze_records(
    *,
    source_records: list[dict[str, Any]],
    batch_id: str,
    api_key: str,
    model_name: str,
    base_url: str,
    model_prices: dict[str, dict[str, Any]],
    max_retries: int,
    max_tokens: int,
    compact_mode: bool = False,
    text_analysis_backend: str = "deepseek",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """逐条调用指定文本后端，只更新分类、标签、摘要和业务用途。"""

    clients = {
        "deepseek": ("deepseek", deepseek_text_analysis_client),
        "qwen_text": ("qwen", qwen_text_analysis_client),
    }
    if text_analysis_backend not in clients:
        raise ValueError(f"不支持的文本分析后端：{text_analysis_backend}")
    provider, analysis_client = clients[text_analysis_backend]

    results: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for record_index, source_record in enumerate(source_records, start=1):
        evidence = build_reanalysis_evidence(source_record)
        evidence_text = _evidence_text(evidence)
        evidence_used = [key for key, value in evidence.items() if value]
        missing_evidence = ["audio_transcript"] if source_record.get("media_type") == "video" and not evidence.get("audio_transcript") else []
        call_id = f"{source_record['file_id']}_reanalyze_{record_index:04d}"
        result_record = dict(source_record)
        result_record["batch_id"] = batch_id
        result_record["source_batch_id"] = source_record.get("batch_id")
        result_record["processed_at"] = _now_iso()
        result_record["audio_transcript"] = evidence["audio_transcript"]
        result_record["evidence_used"] = evidence_used
        result_record["missing_evidence"] = missing_evidence
        record_calls: list[dict[str, Any]] = []

        try:
            analysis_result = analysis_client(
                evidence,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                max_retries=max_retries,
                max_tokens=max_tokens,
                compact_mode=compact_mode,
            )
            api_usage = analysis_result.pop("_api_usage", {})
            api_attempts = analysis_result.pop("_api_attempts", [])
            quality_flags = analysis_result.pop("_quality_flags", [])
            attempts = api_attempts or [{"status": "success", "latency_ms": 0, "api_usage": api_usage, "error_message": None}]
            for attempt_index, attempt in enumerate(attempts, start=1):
                attempt_usage = attempt.get("api_usage") or {}
                attempt_status = str(attempt.get("status") or "failed")
                call = _build_call(
                    call_id=_attempt_call_id(call_id, attempt_index, len(attempts)),
                    batch_id=batch_id,
                    record=source_record,
                    model_name=model_name,
                    provider=provider,
                    input_units=_analysis_input_units(attempt_usage, evidence_text),
                    output_units=_analysis_output_units(attempt_usage, analysis_result if attempt_status == "success" else None),
                    latency_ms=int(attempt.get("latency_ms") or 0),
                    status=attempt_status,
                    error_message=attempt.get("error_message"),
                    model_prices=model_prices,
                    response_model_name=attempt.get("response_model_name"),
                )
                record_calls.append(call)
            model_calls.extend(record_calls)
            result_record.update(analysis_result)
            old_quality_flags = [
                flag
                for flag in result_record.get("quality_flags", [])
                if flag not in {"text_analysis_failed", DEFERRED_TEXT_QUALITY_FLAG}
            ]
            result_record["quality_flags"] = list(dict.fromkeys([*old_quality_flags, *quality_flags]))
            result_record["warning_messages"] = [
                warning
                for warning in result_record.get("warning_messages", [])
                if warning not in {
                    "文本分析模型调用失败，无法产出有效分类、标签和摘要。",
                    DEFERRED_TEXT_WARNING,
                }
            ]
            if missing_evidence:
                result_record["processing_status"] = "partial_success"
                result_record["warning_messages"].append("本次文本重分析未使用 mock 音频转写，结果只基于 OCR 和视觉理解证据。")
            else:
                result_record["processing_status"] = "success"
            result_record["error_message"] = None
        except Exception as exc:
            failed_attempts = exc.attempts if isinstance(exc, (DeepSeekAttemptsExhausted, QwenTextAttemptsExhausted)) else []
            attempts = failed_attempts or [{"api_usage": {}, "latency_ms": 0, "error_message": str(exc), "status": "failed"}]
            for attempt_index, attempt in enumerate(attempts, start=1):
                call = _build_call(
                    call_id=_attempt_call_id(call_id, attempt_index, len(attempts)),
                    batch_id=batch_id,
                    record=source_record,
                    model_name=model_name,
                    provider=provider,
                    input_units=_analysis_input_units(attempt.get("api_usage") or {}, evidence_text),
                    output_units=_analysis_output_units(attempt.get("api_usage") or {}, None),
                    latency_ms=int(attempt.get("latency_ms") or 0),
                    status="failed",
                    error_message=str(attempt.get("error_message") or exc),
                    model_prices=model_prices,
                )
                record_calls.append(call)
            model_calls.extend(record_calls)
            last_call = record_calls[-1]
            errors.append(
                {
                    "batch_id": batch_id,
                    "file_id": source_record["file_id"],
                    "call_id": last_call["call_id"],
                    "error_level": "model_call",
                    "task_type": "text_analysis",
                    "error_message": last_call["error_message"],
                }
            )
            result_record["processing_status"] = "failed"
            result_record["topic"] = None
            result_record["secondary_topics"] = []
            result_record["tags"] = []
            result_record["summary"] = None
            result_record["business_use"] = None
            result_record["error_message"] = last_call["error_message"]

        result_record["call_ids"] = [call["call_id"] for call in record_calls]
        result_record["models_used"] = _models_used_from_calls(record_calls)
        result_record["processing_cost_cny"] = round(sum(float(call["cost_cny"]) for call in record_calls), 6)
        result_record["processing_time_ms"] = sum(int(call["latency_ms"]) for call in record_calls)
        results.append(result_record)

    return results, model_calls, errors


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="复用已有 OCR / 视觉 / ASR 证据，只重跑指定文本分析后端。")
    parser.add_argument("--source-batch-dir", required=True, help="已有批次目录，例如 output/batch_video_qwen_vl_4videos_review。")
    parser.add_argument("--batch-id", required=True, help="新输出批次 ID。")
    parser.add_argument("--settings", default="config/settings.yaml", help="配置文件路径。")
    parser.add_argument("--include-files", help="只处理指定文件名或 file_id，多个值用英文逗号分隔。")
    parser.add_argument("--text-analysis-backend", choices=["deepseek", "qwen_text"], default="deepseek", help="本次只重跑的文本分析后端。")
    parser.add_argument("--allow-live-api", action="store_true", help="必须显式授权，才会调用真实文本 API。")
    parser.add_argument("--max-api-retries", type=int, choices=[0, 1], default=0, help="真实文本 API 最多重试次数。")
    parser.add_argument(
        "--max-tokens",
        "--deepseek-max-tokens",
        dest="max_tokens",
        type=int,
        help="文本分析单次输出 token 上限；旧 DeepSeek 参数名仍兼容。",
    )
    args = parser.parse_args()

    if not args.allow_live_api:
        raise SystemExit("错误：本命令会调用真实文本 API，必须显式提供 --allow-live-api。")

    settings = _load_settings(_resolve_path(args.settings))
    is_qwen = args.text_analysis_backend == "qwen_text"
    backend_name = "qwen_text" if is_qwen else "deepseek"
    api_key_env = str(
        _backend_setting(
            settings,
            "text_analysis",
            backend_name,
            "api_key_env",
            "qwen_text_api_key_env" if is_qwen else "deepseek_api_key_env",
            "DASHSCOPE_API_KEY" if is_qwen else "DEEPSEEK_API_KEY",
        )
    )
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise SystemExit(f"错误：未读取到 {api_key_env}，已在发送网络请求前停止。")
    try:
        max_tokens = _positive_int(
            int(
                args.max_tokens
                or _backend_setting(
                    settings,
                    "text_analysis",
                    backend_name,
                    "max_tokens",
                    "qwen_text_max_tokens" if is_qwen else "deepseek_max_tokens",
                    DEFAULT_QWEN_TEXT_MAX_TOKENS if is_qwen else DEFAULT_DEEPSEEK_MAX_TOKENS,
                )
            ),
            "max_tokens",
        )
        compact_mode = _bool_value(
            _backend_setting(settings, "text_analysis", "deepseek", "compact_mode", "deepseek_compact_mode", False),
            "deepseek_compact_mode",
        )
    except ValueError as exc:
        raise SystemExit(f"错误：{exc}") from exc

    source_batch_dir = _resolve_path(args.source_batch_dir)
    source_records = read_jsonl(source_batch_dir / "results.jsonl")
    source_records = _filter_records(source_records, _parse_include_files(args.include_files))
    if not source_records:
        raise SystemExit("错误：没有匹配到需要重分析的文件。")
    model_prices = load_model_prices(_resolve_path("config/model_prices.yaml"))
    results, model_calls, errors = reanalyze_records(
        source_records=source_records,
        batch_id=args.batch_id,
        api_key=api_key,
        model_name=str(
            _backend_setting(
                settings,
                "text_analysis",
                backend_name,
                "model_name",
                "qwen_text_model_name" if is_qwen else "deepseek_model_name",
            )
        ),
        base_url=str(
            _backend_setting(
                settings,
                "text_analysis",
                backend_name,
                "base_url",
                "qwen_text_base_url" if is_qwen else "deepseek_base_url",
            )
        ),
        model_prices=model_prices,
        max_retries=args.max_api_retries,
        max_tokens=max_tokens,
        compact_mode=compact_mode,
        text_analysis_backend=args.text_analysis_backend,
    )

    output_dir = _resolve_path(_setting(settings, "paths.output_dir", "output_dir", "output"))
    batch_dir = ensure_batch_output_dir(output_dir, args.batch_id)
    write_batch_metadata(
        output_dir,
        args.batch_id,
        {
            "schema_version": "v1",
            "batch_id": args.batch_id,
            "source_batch_dir": str(source_batch_dir),
            "reanalyze_scope": "text_analysis_only",
            "text_analysis_backend": args.text_analysis_backend,
            "text_analysis_model_name": str(
                _backend_setting(
                    settings,
                    "text_analysis",
                    backend_name,
                    "model_name",
                    "qwen_text_model_name" if is_qwen else "deepseek_model_name",
                )
            ),
            "text_analysis_max_tokens": max_tokens,
            "text_analysis_compact_mode": compact_mode,
            "note": "本批次复用历史 OCR 和视觉理解证据，不重新调用 OCR、Qwen-VL 或 ASR。",
        },
    )
    write_results(output_dir, args.batch_id, results)
    write_results_readable(output_dir, args.batch_id, results)
    write_model_calls(output_dir, args.batch_id, model_calls)
    write_errors(output_dir, args.batch_id, errors)
    batch_report = generate_batch_report(
        batch_id=args.batch_id,
        results=results,
        model_calls=model_calls,
        errors=errors,
        budget_limit_cny=float(_setting(settings, "runtime.default_budget_limit_cny", "default_budget_limit_cny", 50)),
    )
    write_json(batch_dir / "batch_report.json", batch_report)
    print({"batch_id": args.batch_id, "batch_dir": str(batch_dir), "total_files": len(results), "total_model_calls": len(model_calls), "total_errors": len(errors)})


if __name__ == "__main__":
    main()
