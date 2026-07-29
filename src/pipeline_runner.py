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
    DEFAULT_PADDLEOCR_MODEL_NAME,
    DeepSeekAttemptsExhausted,
    deepseek_text_analysis_client,
    mock_asr_client,
    mock_ocr_client,
    mock_text_analysis_client,
    mock_vision_client,
    paddleocr_client,
)
from model_router import select_model
from preprocessor import preprocess_file


LOW_QUALITY_OCR_FLAG = "low_quality_ocr_text"
LOW_QUALITY_OCR_WARNING = "OCR 返回了非空文字，但文本疑似乱码或过度碎片化，最终分析未把 OCR 文字作为可靠证据。"
STATUS_AFFECTING_QUALITY_FLAGS = {LOW_QUALITY_OCR_FLAG}
LOW_QUALITY_OCR_MIN_LINES = 6
LOW_QUALITY_OCR_MAX_VISIBLE_CHARS = 80
LOW_QUALITY_OCR_MIN_SHORT_LINE_RATIO = 0.7
LOW_QUALITY_OCR_MIN_SINGLE_CHAR_LINE_RATIO = 0.25
LOW_QUALITY_OCR_MIN_ASCII_NOISE_LINE_RATIO = 0.4


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


def _visible_text_length(text: str) -> int:
    """计算去掉空白后的可见字符数。"""

    return sum(1 for char in text if not char.isspace())


def _has_ascii_letter_or_digit(text: str) -> bool:
    """判断文本中是否包含英文字母或数字。"""

    return any(char.isascii() and char.isalnum() for char in text)


def _is_low_quality_ocr_text(ocr_text: str | None) -> bool:
    """用保守启发式判断 OCR 文本是否疑似乱码或过度碎片化。

    这个函数只判断明显不可用的 OCR 文字，不尝试替代人工基准评估。
    为降低误伤，它要求短行碎片、单字行和 ASCII 噪声同时明显存在。
    """

    if not ocr_text or not ocr_text.strip():
        return False

    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    if not lines:
        return False

    visible_length = _visible_text_length(ocr_text)
    short_lines = sum(1 for line in lines if _visible_text_length(line) <= 5)
    single_char_lines = sum(1 for line in lines if _visible_text_length(line) == 1)
    ascii_noise_lines = sum(1 for line in lines if _has_ascii_letter_or_digit(line))
    short_line_ratio = short_lines / len(lines)
    single_char_line_ratio = single_char_lines / len(lines)
    ascii_noise_line_ratio = ascii_noise_lines / len(lines)

    return (
        len(lines) >= LOW_QUALITY_OCR_MIN_LINES
        and visible_length <= LOW_QUALITY_OCR_MAX_VISIBLE_CHARS
        and short_line_ratio >= LOW_QUALITY_OCR_MIN_SHORT_LINE_RATIO
        and single_char_line_ratio >= LOW_QUALITY_OCR_MIN_SINGLE_CHAR_LINE_RATIO
        and ascii_noise_line_ratio >= LOW_QUALITY_OCR_MIN_ASCII_NOISE_LINE_RATIO
    )


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
    """生成一次成功的模型调用记录。"""

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


def _build_failed_call(
    *,
    call_index: int,
    file_record: dict[str, Any],
    task_type: str,
    input_units: list[dict[str, Any]],
    output_units: list[dict[str, Any]],
    routing_rules: dict[str, dict[str, str]],
    model_prices: dict[str, dict[str, Any]],
    error_message: str,
    latency_ms: int = 0,
    provider: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """生成一次失败的上游模型调用记录。"""

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
    ocr_backend: str = "mock",
    text_analysis_backend: str = "mock",
    deepseek_api_key: str | None = None,
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    deepseek_model_name: str = "deepseek-v4-flash",
    deepseek_max_retries: int = 0,
    fault_injection: dict[str, str] | None = None,
) -> dict[str, Any]:
    """运行单个文件处理流水线；本地 PaddleOCR 当前只用于图片文件。"""

    if ocr_backend not in {"mock", "paddleocr"}:
        raise ValueError(f"不支持的 OCR 后端: {ocr_backend}")
    pipeline_started_at = perf_counter()
    injected_failures = fault_injection or {}
    preprocessed = preprocess_file(file_record)
    media_type = file_record["media_type"]
    model_calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    recovered_errors: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "raw_text": None,
        "ocr_text": None,
        "audio_transcript": None,
        "visual_description": None,
    }
    evidence_used: list[str] = []
    missing_evidence: list[str] = []
    quality_flags: list[str] = []
    warning_messages: list[str] = []

    def record_upstream_failure(
        *,
        task_type: str,
        missing_field: str,
        quality_flag: str,
        warning_message: str,
        input_units: list[dict[str, Any]],
        output_units: list[dict[str, Any]],
        error_message: str | None = None,
        latency_ms: int = 0,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """记录一条上游分支失败，并保留故障追踪信息。"""

        resolved_error_message = error_message or injected_failures[task_type]
        failed_call = _build_failed_call(
            call_index=len(model_calls) + 1,
            file_record=file_record,
            task_type=task_type,
            input_units=input_units,
            output_units=output_units,
            routing_rules=routing_rules,
            model_prices=model_prices,
            error_message=resolved_error_message,
            latency_ms=latency_ms,
            provider=provider,
            model_name=model_name,
        )
        model_calls.append(failed_call)
        missing_evidence.append(missing_field)
        quality_flags.append(quality_flag)
        warning_messages.append(warning_message)
        errors.append(
            {
                "batch_id": file_record["batch_id"],
                "file_id": file_record["file_id"],
                "call_id": failed_call["call_id"],
                "error_level": "model_call",
                "task_type": task_type,
                "error_message": resolved_error_message,
            }
        )

    if media_type == "text":
        evidence["raw_text"] = preprocessed["raw_text"]
        evidence_used.append("raw_text")

    elif media_type == "image":
        image_path = preprocessed["image_path"]
        ocr_provider = "paddlepaddle" if ocr_backend == "paddleocr" else None
        ocr_model_name = DEFAULT_PADDLEOCR_MODEL_NAME if ocr_backend == "paddleocr" else None
        ocr_started_at = perf_counter()
        try:
            if "ocr" in injected_failures:
                raise RuntimeError(injected_failures["ocr"])
            if ocr_backend == "paddleocr":
                ocr_result = paddleocr_client(image_path)
            else:
                ocr_result = mock_ocr_client(image_path)
        except Exception as exc:
            record_upstream_failure(
                task_type="ocr",
                missing_field="ocr_text",
                quality_flag="ocr_failed",
                warning_message="OCR 分支失败，最终分析未使用图片文字证据。",
                input_units=[{"unit_type": "image_count", "quantity": 1}],
                output_units=_output_text_chars(""),
                error_message=str(exc),
                latency_ms=int(round((perf_counter() - ocr_started_at) * 1000)),
                provider=ocr_provider,
                model_name=ocr_model_name,
            )
        else:
            evidence.update(ocr_result)
            if ocr_result["ocr_text"]:
                if ocr_backend == "paddleocr" and _is_low_quality_ocr_text(ocr_result["ocr_text"]):
                    quality_flags.append(LOW_QUALITY_OCR_FLAG)
                    warning_messages.append(LOW_QUALITY_OCR_WARNING)
                else:
                    evidence_used.append("ocr_text")
            model_calls.append(
                _build_success_call(
                    call_index=len(model_calls) + 1,
                    file_record=file_record,
                    task_type="ocr",
                    input_units=[{"unit_type": "image_count", "quantity": 1}],
                    output_units=_output_text_chars(ocr_result["ocr_text"] or ""),
                    routing_rules=routing_rules,
                    model_prices=model_prices,
                    latency_ms=int(round((perf_counter() - ocr_started_at) * 1000)),
                    provider=ocr_provider,
                    model_name=ocr_model_name,
                )
            )

        if "visual_understanding" in injected_failures:
            record_upstream_failure(
                task_type="visual_understanding",
                missing_field="visual_description",
                quality_flag="visual_understanding_failed",
                warning_message="视觉理解分支失败，最终分析未使用画面描述证据。",
                input_units=[{"unit_type": "frame_count", "quantity": 1}],
                output_units=_output_tokens(""),
            )
        else:
            vision_result = mock_vision_client(image_path)
            evidence.update(vision_result)
            evidence_used.append("visual_description")
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

        if "ocr" in injected_failures:
            record_upstream_failure(
                task_type="ocr",
                missing_field="ocr_text",
                quality_flag="ocr_failed",
                warning_message="OCR 分支失败，最终分析未使用视频关键帧文字证据。",
                input_units=[{"unit_type": "image_count", "quantity": 1}],
                output_units=_output_text_chars(""),
            )
        else:
            ocr_result = mock_ocr_client(keyframe_path)
            evidence.update(ocr_result)
            evidence_used.append("ocr_text")
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

        if "visual_understanding" in injected_failures:
            record_upstream_failure(
                task_type="visual_understanding",
                missing_field="visual_description",
                quality_flag="visual_understanding_failed",
                warning_message="视觉理解分支失败，最终分析未使用视频画面描述证据。",
                input_units=[{"unit_type": "frame_count", "quantity": 1}],
                output_units=_output_tokens(""),
            )
        else:
            vision_result = mock_vision_client(keyframe_path)
            evidence.update(vision_result)
            evidence_used.append("visual_description")
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

        if "speech_to_text" in injected_failures:
            record_upstream_failure(
                task_type="speech_to_text",
                missing_field="audio_transcript",
                quality_flag="speech_to_text_failed",
                warning_message="语音识别分支失败，最终分析未使用音频转写证据。",
                input_units=[{"unit_type": "audio_seconds", "quantity": preprocessed["duration_ms"] / 1000}],
                output_units=_output_text_chars(""),
            )
        else:
            asr_result = mock_asr_client(audio_path)
            evidence.update(asr_result)
            evidence_used.append("audio_transcript")
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

    analysis_evidence = dict(evidence)
    if LOW_QUALITY_OCR_FLAG in quality_flags:
        analysis_evidence["ocr_text"] = None

    evidence_text = " ".join(str(value) for value in analysis_evidence.values() if value)
    analysis_started_at = perf_counter()
    api_usage: dict[str, int] = {}
    api_attempts: list[dict[str, Any]] = []
    analysis_quality_flags: list[str] = []
    analysis_provider = "deepseek" if text_analysis_backend == "deepseek" else None
    analysis_model_name = deepseek_model_name if text_analysis_backend == "deepseek" else None

    try:
        if "text_analysis" in injected_failures:
            raise RuntimeError(injected_failures["text_analysis"])
        if text_analysis_backend == "deepseek":
            analysis_result = deepseek_text_analysis_client(
                analysis_evidence,
                api_key=deepseek_api_key,
                model_name=deepseek_model_name,
                base_url=deepseek_base_url,
                max_retries=deepseek_max_retries,
            )
            api_usage = analysis_result.pop("_api_usage", {})
            api_attempts = analysis_result.pop("_api_attempts", [])
            analysis_quality_flags = analysis_result.pop("_quality_flags", [])
        elif text_analysis_backend == "mock":
            analysis_result = mock_text_analysis_client(analysis_evidence)
        else:
            raise ValueError(f"不支持的文本分析后端: {text_analysis_backend}")
    except Exception as exc:
        latency_ms = int(round((perf_counter() - analysis_started_at) * 1000))
        error_message = str(exc)
        fallback_model = select_model("text_analysis", routing_rules)
        failed_attempts = exc.attempts if isinstance(exc, DeepSeekAttemptsExhausted) else []
        if not failed_attempts:
            failed_attempts = [
                {
                    "status": "failed",
                    "latency_ms": latency_ms,
                    "api_usage": api_usage,
                    "error_message": error_message,
                }
            ]

        for attempt in failed_attempts:
            attempt_usage = attempt.get("api_usage") or {}
            failed_call = _build_failed_text_analysis_call(
                call_index=len(model_calls) + 1,
                file_record=file_record,
                input_units=_analysis_input_units(attempt_usage, evidence_text),
                output_units=_analysis_output_units(attempt_usage, None),
                latency_ms=int(attempt.get("latency_ms") or 0),
                error_message=str(attempt.get("error_message") or error_message),
                model_prices=model_prices,
                provider=analysis_provider or fallback_model["provider"],
                model_name=analysis_model_name or fallback_model["model_name"],
            )
            model_calls.append(failed_call)
            errors.append(
                {
                    "batch_id": file_record["batch_id"],
                    "file_id": file_record["file_id"],
                    "call_id": failed_call["call_id"],
                    "error_level": "model_call",
                    "task_type": "text_analysis",
                    "error_message": failed_call["error_message"],
                }
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
            "missing_evidence": missing_evidence,
            "quality_flags": [*quality_flags, "text_analysis_failed"],
            "warning_messages": [*warning_messages, "文本分析模型调用失败，无法产出有效分类、标签和摘要。"],
            "error_message": error_message,
            "call_ids": [call["call_id"] for call in model_calls],
            "models_used": _models_used_from_calls(model_calls),
            "processing_cost_cny": round(sum(float(call["cost_cny"]) for call in model_calls), 6),
            "processing_time_ms": int(round((perf_counter() - pipeline_started_at) * 1000)),
        }

        return {
            "result": result,
            "model_calls": model_calls,
            "errors": errors,
        }

    latency_ms = int(round((perf_counter() - analysis_started_at) * 1000))
    if text_analysis_backend == "deepseek" and api_attempts:
        for attempt in api_attempts:
            attempt_usage = attempt.get("api_usage") or {}
            if attempt.get("status") == "success":
                model_calls.append(
                    _build_success_call(
                        call_index=len(model_calls) + 1,
                        file_record=file_record,
                        task_type="text_analysis",
                        input_units=_analysis_input_units(attempt_usage, evidence_text),
                        output_units=_analysis_output_units(attempt_usage, analysis_result),
                        routing_rules=routing_rules,
                        model_prices=model_prices,
                        latency_ms=int(attempt.get("latency_ms") or 0),
                        provider=analysis_provider,
                        model_name=analysis_model_name,
                    )
                )
                continue

            failed_call = _build_failed_text_analysis_call(
                call_index=len(model_calls) + 1,
                file_record=file_record,
                input_units=_analysis_input_units(attempt_usage, evidence_text),
                output_units=_analysis_output_units(attempt_usage, None),
                latency_ms=int(attempt.get("latency_ms") or 0),
                error_message=str(attempt.get("error_message") or "DeepSeek 调用失败。"),
                model_prices=model_prices,
                provider=analysis_provider or "deepseek",
                model_name=analysis_model_name or deepseek_model_name,
            )
            model_calls.append(failed_call)
            recovered_errors.append(
                {
                    "batch_id": file_record["batch_id"],
                    "file_id": file_record["file_id"],
                    "call_id": failed_call["call_id"],
                    "error_level": "model_call",
                    "task_type": "text_analysis",
                    "error_message": failed_call["error_message"],
                }
            )
    else:
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

    quality_flags.extend(str(flag) for flag in analysis_quality_flags if flag not in quality_flags)
    has_status_affecting_quality_flag = any(flag in STATUS_AFFECTING_QUALITY_FLAGS for flag in quality_flags)
    processing_status = "partial_success" if missing_evidence or has_status_affecting_quality_flag else "success"
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
        "processing_status": processing_status,
        "evidence_used": evidence_used,
        "missing_evidence": missing_evidence,
        "quality_flags": quality_flags,
        "warning_messages": warning_messages,
        "error_message": "；".join(str(error["error_message"]) for error in errors) if errors else None,
        "call_ids": [call["call_id"] for call in model_calls],
        "models_used": _models_used_from_calls(model_calls),
        "processing_cost_cny": round(sum(float(call["cost_cny"]) for call in model_calls), 6),
        "processing_time_ms": int(round((perf_counter() - pipeline_started_at) * 1000)),
    }

    return {
        "result": result,
        "model_calls": model_calls,
        "errors": [*errors, *recovered_errors],
    }
