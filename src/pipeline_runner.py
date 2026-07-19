"""为每个文件运行对应的处理流水线。

调度器根据 media_type 选择文本、图片或视频流水线，收集证据，判断 processing_status，
并返回供 result_writer 写入的文件级结果对象。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from cost_latency_tracker import build_model_call_record
from model_clients import (
    DEFAULT_DEEPSEEK_BASE_URL,
    deepseek_text_analysis_client,
    mock_asr_client,
    mock_ocr_client,
    mock_text_analysis_client,
    mock_vision_client,
)
from model_router import select_model
from preprocessor import preprocess_file


def _now_iso() -> str:
    """返回当前本地时间的 ISO 字符串。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _text_units(text: str) -> list[dict[str, Any]]:
    """把文本长度粗略转换为输入 token 数。"""

    return [{"unit_type": "input_tokens", "quantity": max(1, len(text) // 2)}]


def _output_text_chars(text: str) -> list[dict[str, Any]]:
    """把输出文字长度记录为字符数。"""

    return [{"unit_type": "text_chars", "quantity": len(text)}]


def _output_tokens(text: str) -> list[dict[str, Any]]:
    """把输出文字长度粗略转换为输出 token 数。"""

    return [{"unit_type": "output_tokens", "quantity": max(1, len(text) // 2)}]


def _build_success_call(
    *,
    call_index: int,
    file_record: dict[str, Any],
    task_type: str,
    input_units: list[dict[str, Any]],
    output_units: list[dict[str, Any]],
    routing_rules: dict[str, dict[str, str]],
    model_prices: dict[str, dict[str, Any]],
    latency_ms: int = 0,
    provider: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """生成一次成功的 mock 模型调用记录。"""

    selected = select_model(task_type, routing_rules)
    selected_provider = provider or selected["provider"]
    selected_model_name = model_name or selected["model_name"]
    return build_model_call_record(
        call_id=f"{file_record['file_id']}_call_{call_index:04d}",
        batch_id=file_record["batch_id"],
        file_id=file_record["file_id"],
        task_type=task_type,
        provider=selected_provider,
        model_name=selected_model_name,
        input_units=input_units,
        output_units=output_units,
        latency_ms=latency_ms,
        started_at=_now_iso(),
        status="success",
        error_message=None,
        model_prices=model_prices,
    )


def _build_failed_text_analysis_call(
    *,
    call_index: int,
    file_record: dict[str, Any],
    input_units: list[dict[str, Any]],
    output_units: list[dict[str, Any]],
    latency_ms: int,
    error_message: str,
    model_prices: dict[str, dict[str, Any]],
    provider: str,
    model_name: str,
) -> dict[str, Any]:
    """生成一次失败的文本分析模型调用记录。"""

    return build_model_call_record(
        call_id=f"{file_record['file_id']}_call_{call_index:04d}",
        batch_id=file_record["batch_id"],
        file_id=file_record["file_id"],
        task_type="text_analysis",
        provider=provider,
        model_name=model_name,
        input_units=input_units,
        output_units=output_units,
        latency_ms=latency_ms,
        started_at=_now_iso(),
        status="failed",
        error_message=error_message,
        model_prices=model_prices,
    )


def _analysis_input_units(api_usage: dict[str, int], evidence_text: str) -> list[dict[str, Any]]:
    """优先使用真实 API 返回的输入 token；没有时使用粗略估算。"""

    prompt_tokens = int(api_usage.get("prompt_tokens") or 0)
    if prompt_tokens > 0:
        return [{"unit_type": "input_tokens", "quantity": prompt_tokens}]
    return _text_units(evidence_text)


def _analysis_output_units(api_usage: dict[str, int], analysis_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """优先使用真实 API 返回的输出 token；没有时使用粗略估算。"""

    completion_tokens = int(api_usage.get("completion_tokens") or 0)
    if completion_tokens > 0:
        return [{"unit_type": "output_tokens", "quantity": completion_tokens}]
    if not analysis_result:
        return [{"unit_type": "output_tokens", "quantity": 0}]
    return _output_tokens(str(analysis_result["summary"]) + str(analysis_result["business_use"]))


def _models_used_from_calls(model_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
    """从模型调用明细中提取文件级模型使用摘要。"""

    return [
        {
            "call_id": str(call["call_id"]),
            "task_type": str(call["task_type"]),
            "provider": str(call["provider"]),
            "model_name": str(call["model_name"]),
            "status": str(call["status"]),
        }
        for call in model_calls
    ]


def run_file_pipeline(
    file_record: dict[str, Any],
    routing_rules: dict[str, dict[str, str]],
    model_prices: dict[str, dict[str, Any]],
    *,
    text_analysis_backend: str = "mock",
    deepseek_api_key: str | None = None,
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    deepseek_model_name: str = "deepseek-v4-flash",
) -> dict[str, Any]:
    """运行单个文件的 mock 处理流水线。"""

    pipeline_started_at = perf_counter()
    preprocessed = preprocess_file(file_record)
    media_type = file_record["media_type"]
    model_calls: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "raw_text": None,
        "ocr_text": None,
        "audio_transcript": None,
        "visual_description": None,
    }
    evidence_used: list[str] = []

    if media_type == "text":
        evidence["raw_text"] = preprocessed["raw_text"]
        evidence_used.append("raw_text")

    elif media_type == "image":
        image_path = preprocessed["image_path"]
        ocr_result = mock_ocr_client(image_path)
        vision_result = mock_vision_client(image_path)
        evidence.update(ocr_result)
        evidence.update(vision_result)
        evidence_used.extend(["ocr_text", "visual_description"])
        model_calls.append(
            _build_success_call(
                call_index=len(model_calls) + 1,
                file_record=file_record,
                task_type="ocr",
                input_units=[{"unit_type": "image_count", "quantity": 1}],
                output_units=_output_text_chars(ocr_result["ocr_text"]),
                routing_rules=routing_rules,
                model_prices=model_prices,
            )
        )
        model_calls.append(
            _build_success_call(
                call_index=len(model_calls) + 1,
                file_record=file_record,
                task_type="visual_understanding",
                input_units=[{"unit_type": "frame_count", "quantity": 1}],
                output_units=_output_tokens(vision_result["visual_description"]),
                routing_rules=routing_rules,
                model_prices=model_prices,
            )
        )

    elif media_type == "video":
        keyframe_path = preprocessed["keyframes"][0]
        audio_path = preprocessed["audio_path"]
        ocr_result = mock_ocr_client(keyframe_path)
        vision_result = mock_vision_client(keyframe_path)
        asr_result = mock_asr_client(audio_path)
        evidence.update(ocr_result)
        evidence.update(vision_result)
        evidence.update(asr_result)
        evidence_used.extend(["ocr_text", "visual_description", "audio_transcript"])
        model_calls.append(
            _build_success_call(
                call_index=len(model_calls) + 1,
                file_record=file_record,
                task_type="ocr",
                input_units=[{"unit_type": "image_count", "quantity": 1}],
                output_units=_output_text_chars(ocr_result["ocr_text"]),
                routing_rules=routing_rules,
                model_prices=model_prices,
            )
        )
        model_calls.append(
            _build_success_call(
                call_index=len(model_calls) + 1,
                file_record=file_record,
                task_type="visual_understanding",
                input_units=[{"unit_type": "frame_count", "quantity": 1}],
                output_units=_output_tokens(vision_result["visual_description"]),
                routing_rules=routing_rules,
                model_prices=model_prices,
            )
        )
        model_calls.append(
            _build_success_call(
                call_index=len(model_calls) + 1,
                file_record=file_record,
                task_type="speech_to_text",
                input_units=[{"unit_type": "audio_seconds", "quantity": preprocessed["duration_ms"] / 1000}],
                output_units=_output_text_chars(asr_result["audio_transcript"]),
                routing_rules=routing_rules,
                model_prices=model_prices,
            )
        )

    else:
        raise ValueError(f"不支持的文件类型: {media_type}")

    evidence_text = " ".join(str(value) for value in evidence.values() if value)
    analysis_started_at = perf_counter()
    api_usage: dict[str, int] = {}
    analysis_provider = "deepseek" if text_analysis_backend == "deepseek" else None
    analysis_model_name = deepseek_model_name if text_analysis_backend == "deepseek" else None

    try:
        if text_analysis_backend == "deepseek":
            analysis_result = deepseek_text_analysis_client(
                evidence,
                api_key=deepseek_api_key,
                model_name=deepseek_model_name,
                base_url=deepseek_base_url,
            )
            api_usage = analysis_result.pop("_api_usage", {})
        elif text_analysis_backend == "mock":
            analysis_result = mock_text_analysis_client(evidence)
        else:
            raise ValueError(f"不支持的文本分析后端: {text_analysis_backend}")
    except Exception as exc:
        latency_ms = int(round((perf_counter() - analysis_started_at) * 1000))
        input_units = _analysis_input_units(api_usage, evidence_text)
        output_units = _analysis_output_units(api_usage, None)
        error_message = str(exc)
        fallback_model = select_model("text_analysis", routing_rules)
        model_calls.append(
            _build_failed_text_analysis_call(
                call_index=len(model_calls) + 1,
                file_record=file_record,
                input_units=input_units,
                output_units=output_units,
                latency_ms=latency_ms,
                error_message=error_message,
                model_prices=model_prices,
                provider=analysis_provider or fallback_model["provider"],
                model_name=analysis_model_name or fallback_model["model_name"],
            )
        )
        result = {
            "schema_version": "v1",
            "batch_id": file_record["batch_id"],
            "file_id": file_record["file_id"],
            "file_name": file_record["file_name"],
            "media_type": media_type,
            "source_path": file_record["source_path"],
            "file_size_bytes": file_record["file_size_bytes"],
            "duration_ms": preprocessed.get("duration_ms"),
            "language": "unknown",
            "created_at": file_record["created_at"],
            "processed_at": _now_iso(),
            "raw_text": evidence["raw_text"],
            "ocr_text": evidence["ocr_text"],
            "audio_transcript": evidence["audio_transcript"],
            "visual_description": evidence["visual_description"],
            "topic": None,
            "secondary_topics": [],
            "tags": [],
            "summary": None,
            "business_use": None,
            "processing_status": "failed",
            "evidence_used": evidence_used,
            "missing_evidence": [],
            "quality_flags": ["text_analysis_failed"],
            "warning_messages": ["文本分析模型调用失败，无法产出有效分类、标签和摘要。"],
            "error_message": error_message,
            "call_ids": [call["call_id"] for call in model_calls],
            "models_used": _models_used_from_calls(model_calls),
            "processing_cost_cny": round(sum(float(call["cost_cny"]) for call in model_calls), 6),
            "processing_time_ms": int(round((perf_counter() - pipeline_started_at) * 1000)),
        }

        return {
            "result": result,
            "model_calls": model_calls,
            "errors": [
                {
                    "batch_id": file_record["batch_id"],
                    "file_id": file_record["file_id"],
                    "call_id": model_calls[-1]["call_id"],
                    "error_level": "model_call",
                    "task_type": "text_analysis",
                    "error_message": error_message,
                }
            ],
        }

    latency_ms = int(round((perf_counter() - analysis_started_at) * 1000))
    model_calls.append(
        _build_success_call(
            call_index=len(model_calls) + 1,
            file_record=file_record,
            task_type="text_analysis",
            input_units=_analysis_input_units(api_usage, evidence_text),
            output_units=_analysis_output_units(api_usage, analysis_result),
            routing_rules=routing_rules,
            model_prices=model_prices,
            latency_ms=latency_ms,
            provider=analysis_provider,
            model_name=analysis_model_name,
        )
    )

    result = {
        "schema_version": "v1",
        "batch_id": file_record["batch_id"],
        "file_id": file_record["file_id"],
        "file_name": file_record["file_name"],
        "media_type": media_type,
        "source_path": file_record["source_path"],
        "file_size_bytes": file_record["file_size_bytes"],
        "duration_ms": preprocessed.get("duration_ms"),
        "language": "unknown",
        "created_at": file_record["created_at"],
        "processed_at": _now_iso(),
        "raw_text": evidence["raw_text"],
        "ocr_text": evidence["ocr_text"],
        "audio_transcript": evidence["audio_transcript"],
        "visual_description": evidence["visual_description"],
        "topic": analysis_result["topic"],
        "secondary_topics": analysis_result["secondary_topics"],
        "tags": analysis_result["tags"],
        "summary": analysis_result["summary"],
        "business_use": analysis_result["business_use"],
        "processing_status": "success",
        "evidence_used": evidence_used,
        "missing_evidence": [],
        "quality_flags": [],
        "warning_messages": [],
        "error_message": None,
        "call_ids": [call["call_id"] for call in model_calls],
        "models_used": _models_used_from_calls(model_calls),
        "processing_cost_cny": round(sum(float(call["cost_cny"]) for call in model_calls), 6),
        "processing_time_ms": int(round((perf_counter() - pipeline_started_at) * 1000)),
    }

    return {
        "result": result,
        "model_calls": model_calls,
        "errors": [],
    }
