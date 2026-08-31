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
    DEFAULT_DEEPSEEK_MAX_TOKENS,
    DEFAULT_DASHSCOPE_ASR_MODEL_NAME,
    DEFAULT_DASHSCOPE_ASR_SUBMIT_URL,
    DEFAULT_PADDLEOCR_MODEL_NAME,
    DEFAULT_QWEN_OCR_MAX_TOKENS,
    DEFAULT_QWEN_OCR_MODEL_NAME,
    DEFAULT_QWEN_VL_BASE_URL,
    DEFAULT_QWEN_VL_MAX_IMAGE_SIDE,
    DEFAULT_QWEN_VL_MAX_TOKENS,
    DEFAULT_QWEN_VL_MODEL_NAME,
    DeepSeekAttemptsExhausted,
    QwenVLAttemptsExhausted,
    dashscope_asr_client,
    dashscope_upload_local_file,
    deepseek_text_analysis_client,
    mock_asr_client,
    mock_ocr_client,
    mock_text_analysis_client,
    mock_vision_client,
    paddleocr_client,
    qwen_ocr_client,
    qwen_vl_image_understanding_client,
)
from model_router import route_plan_backends_for_media, routing_rules_for_media, select_model
from preprocessor import preprocess_file
from runtime_config import runtime_policy_list, runtime_policy_section


LOW_QUALITY_OCR_FLAG = "low_quality_ocr_text"
LOW_QUALITY_OCR_WARNING = "OCR 返回了非空文字，但文本疑似乱码或过度碎片化，最终分析未把 OCR 文字作为可靠证据。"
VIDEO_OCR_KEYFRAME_FAILED_FLAG = "video_ocr_keyframe_failed"
VIDEO_VISUAL_KEYFRAME_FAILED_FLAG = "video_visual_keyframe_failed"
VIDEO_EVIDENCE_WEAK_FLAG = "video_evidence_weak"
TEXT_ANALYSIS_EVIDENCE_TRUNCATED_FLAG = "text_analysis_evidence_truncated"
TEXT_ANALYSIS_EVIDENCE_TRUNCATED_WARNING = "文本分析输入证据超过配置上限，系统已裁剪送入文本分析模型的证据副本；原始证据仍保留在输出结果中。"
TEXT_ANALYSIS_EVIDENCE_KEYS = ("raw_text", "ocr_text", "audio_transcript", "visual_description")
STATUS_AFFECTING_QUALITY_FLAGS = {
    LOW_QUALITY_OCR_FLAG,
    VIDEO_OCR_KEYFRAME_FAILED_FLAG,
    VIDEO_VISUAL_KEYFRAME_FAILED_FLAG,
    VIDEO_EVIDENCE_WEAK_FLAG,
}
OCR_QUALITY_GATE = runtime_policy_section("ocr_quality_gate")
TEXT_ANALYSIS_EXECUTION_POLICY = runtime_policy_section("text_analysis_execution")
ALLOWED_OCR_BACKENDS = runtime_policy_list("runtime_backends", "ocr")
ALLOWED_VISION_BACKENDS = runtime_policy_list("runtime_backends", "vision_understanding")
ALLOWED_SPEECH_BACKENDS = runtime_policy_list("runtime_backends", "speech_to_text")
LOW_QUALITY_OCR_MIN_LINES = int(OCR_QUALITY_GATE.get("min_lines", 6))
LOW_QUALITY_OCR_MAX_VISIBLE_CHARS = int(OCR_QUALITY_GATE.get("max_visible_chars", 80))
LOW_QUALITY_OCR_MIN_SHORT_LINE_RATIO = float(OCR_QUALITY_GATE.get("min_short_line_ratio", 0.7))
LOW_QUALITY_OCR_MIN_SINGLE_CHAR_LINE_RATIO = float(OCR_QUALITY_GATE.get("min_single_char_line_ratio", 0.25))
LOW_QUALITY_OCR_MIN_ASCII_NOISE_LINE_RATIO = float(OCR_QUALITY_GATE.get("min_ascii_noise_line_ratio", 0.4))
LOW_QUALITY_OCR_MIN_ASCII_FRAGMENT_LINES = int(OCR_QUALITY_GATE.get("min_ascii_fragment_lines", 30))
LOW_QUALITY_OCR_MIN_ASCII_FRAGMENT_RATIO = float(OCR_QUALITY_GATE.get("min_ascii_fragment_ratio", 0.6))
DEFERRED_TEXT_STATUS = str(TEXT_ANALYSIS_EXECUTION_POLICY["deferred_processing_status"])
DEFERRED_TEXT_QUALITY_FLAG = str(TEXT_ANALYSIS_EXECUTION_POLICY["deferred_quality_flag"])
DEFERRED_TEXT_WARNING = str(TEXT_ANALYSIS_EXECUTION_POLICY["deferred_warning"])
NO_EVIDENCE_STATUS = str(TEXT_ANALYSIS_EXECUTION_POLICY["no_evidence_processing_status"])
NO_EVIDENCE_QUALITY_FLAG = str(TEXT_ANALYSIS_EXECUTION_POLICY["no_evidence_quality_flag"])
NO_EVIDENCE_ERROR = str(TEXT_ANALYSIS_EXECUTION_POLICY["no_evidence_error"])


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


def _video_audio_seconds(preprocessed: dict[str, Any]) -> float:
    """根据视频预处理结果估算音频秒数；缺少时返回0，避免把未知时长硬算成真实音频用量。"""

    duration_ms = preprocessed.get("duration_ms")
    if isinstance(duration_ms, (int, float)) and duration_ms > 0:
        return round(float(duration_ms) / 1000, 6)
    return 0


def _audio_url_for_record(
    *,
    file_record: dict[str, Any],
    audio_path: str | Path,
    audio_url_map: dict[str, str] | None,
) -> str | None:
    """按 file_id、文件名或音频文件名查找远端音频 URL。"""

    if not audio_url_map:
        return None
    audio_name = Path(audio_path).name
    for key in (str(file_record.get("file_id")), str(file_record.get("file_name")), audio_name):
        value = audio_url_map.get(key)
        if value:
            return value
    return None


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
    ascii_fragment_lines = sum(
        1
        for line in lines
        if _visible_text_length(line) <= 3 and all(char.isascii() for char in line if not char.isspace())
    )
    ascii_fragment_ratio = ascii_fragment_lines / len(lines)

    if (
        ascii_fragment_lines >= LOW_QUALITY_OCR_MIN_ASCII_FRAGMENT_LINES
        and ascii_fragment_ratio >= LOW_QUALITY_OCR_MIN_ASCII_FRAGMENT_RATIO
    ):
        return True

    return (
        len(lines) >= LOW_QUALITY_OCR_MIN_LINES
        and visible_length <= LOW_QUALITY_OCR_MAX_VISIBLE_CHARS
        and short_line_ratio >= LOW_QUALITY_OCR_MIN_SHORT_LINE_RATIO
        and single_char_line_ratio >= LOW_QUALITY_OCR_MIN_SINGLE_CHAR_LINE_RATIO
        and ascii_noise_line_ratio >= LOW_QUALITY_OCR_MIN_ASCII_NOISE_LINE_RATIO
    )


def _limit_text_analysis_evidence(
    evidence: dict[str, Any],
    max_chars: int | None,
) -> tuple[dict[str, Any], bool]:
    """限制送入文本分析模型的证据长度，原始证据由结果对象继续保留。"""

    limited_evidence = dict(evidence)
    if max_chars is None:
        return limited_evidence, False
    if max_chars < 1:
        raise ValueError("文本分析证据字符上限必须大于等于 1。")

    active_keys = [
        key
        for key in TEXT_ANALYSIS_EVIDENCE_KEYS
        if isinstance(limited_evidence.get(key), str) and limited_evidence[key]
    ]
    if not active_keys:
        return limited_evidence, False

    per_key_limit = max(1, max_chars // len(active_keys))
    truncated = False
    for key in active_keys:
        value = limited_evidence[key]
        if len(value) <= per_key_limit:
            continue
        limited_evidence[key] = value[: max(0, per_key_limit - 1)] + "…"
        truncated = True
    return limited_evidence, truncated


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
    response_model_name: str | None = None,
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
        response_model_name=response_model_name,
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
    response_model_name: str | None = None,
    response_diagnostics: dict[str, Any] | None = None,
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
        response_model_name=response_model_name,
        response_diagnostics=response_diagnostics,
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
    response_model_name: str | None = None,
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
        response_model_name=response_model_name,
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


def _vision_input_units(api_usage: dict[str, int]) -> list[dict[str, Any]]:
    """优先使用真实视觉 API 返回的输入 token；失败或缺失时记为 0。"""

    return [{"unit_type": "input_tokens", "quantity": int(api_usage.get("prompt_tokens") or 0)}]


def _vision_output_units(api_usage: dict[str, int], visual_description: str | None) -> list[dict[str, Any]]:
    """优先使用真实视觉 API 返回的输出 token；没有时按描述文字粗略估算。"""

    completion_tokens = int(api_usage.get("completion_tokens") or 0)
    if completion_tokens > 0:
        return [{"unit_type": "output_tokens", "quantity": completion_tokens}]
    if not visual_description:
        return [{"unit_type": "output_tokens", "quantity": 0}]
    return _output_tokens(visual_description)


def _ocr_input_units(ocr_backend: str, api_usage: dict[str, int]) -> list[dict[str, Any]]:
    """云 OCR 使用真实输入 token，本地与 mock 继续按图片数计量。"""

    if ocr_backend == "qwen_ocr":
        return [{"unit_type": "input_tokens", "quantity": int(api_usage.get("prompt_tokens") or 0)}]
    return [{"unit_type": "image_count", "quantity": 1}]


def _ocr_output_units(ocr_backend: str, api_usage: dict[str, int], ocr_text: str | None) -> list[dict[str, Any]]:
    """云 OCR 使用真实输出 token，本地与 mock 继续按文字长度计量。"""

    if ocr_backend == "qwen_ocr":
        return [{"unit_type": "output_tokens", "quantity": int(api_usage.get("completion_tokens") or 0)}]
    return _output_text_chars(ocr_text or "")


def _run_ocr_backend(
    image_path: str | Path,
    ocr_backend: str,
    *,
    qwen_ocr_api_key: str | None,
    qwen_ocr_base_url: str,
    qwen_ocr_model_name: str,
    qwen_ocr_max_tokens: int,
    qwen_ocr_max_image_side: int | None,
) -> tuple[dict[str, Any], dict[str, int], str | None]:
    """让图片和视频关键帧共用同一 OCR 后端选择。"""

    if ocr_backend == "paddleocr":
        return paddleocr_client(image_path), {}, None
    if ocr_backend == "qwen_ocr":
        result = qwen_ocr_client(
            image_path,
            api_key=qwen_ocr_api_key,
            base_url=qwen_ocr_base_url,
            model_name=qwen_ocr_model_name,
            max_tokens=qwen_ocr_max_tokens,
            max_image_side=qwen_ocr_max_image_side,
        )
        return result, result.pop("_api_usage", {}), result.pop("_response_model_name", None)
    return mock_ocr_client(image_path), {}, None


def _build_qwen_vl_attempt_call(
    *,
    attempt: dict[str, Any],
    call_index: int,
    file_record: dict[str, Any],
    visual_description: str | None,
    routing_rules: dict[str, dict[str, str]],
    model_prices: dict[str, dict[str, Any]],
    provider: str,
    model_name: str,
) -> dict[str, Any]:
    """把 Qwen-VL 的单次尝试转换为模型调用记录。"""

    api_usage = attempt.get("api_usage") or {}
    response_model_name = attempt.get("response_model_name")
    if attempt.get("status") == "success":
        return _build_success_call(
            call_index=call_index,
            file_record=file_record,
            task_type="visual_understanding",
            input_units=_vision_input_units(api_usage),
            output_units=_vision_output_units(api_usage, visual_description),
            routing_rules=routing_rules,
            model_prices=model_prices,
            latency_ms=int(attempt.get("latency_ms") or 0),
            provider=provider,
            model_name=model_name,
            response_model_name=response_model_name,
        )

    return _build_failed_call(
        call_index=call_index,
        file_record=file_record,
        task_type="visual_understanding",
        input_units=_vision_input_units(api_usage),
        output_units=_vision_output_units(api_usage, None),
        routing_rules=routing_rules,
        model_prices=model_prices,
        error_message=str(attempt.get("error_message") or "Qwen-VL 调用失败。"),
        latency_ms=int(attempt.get("latency_ms") or 0),
        provider=provider,
        model_name=model_name,
        response_model_name=response_model_name,
    )


def _models_used_from_calls(model_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
    """从模型调用明细中提取文件级模型使用摘要。"""

    models_used = []
    for call in model_calls:
        item = {
            "call_id": str(call["call_id"]),
            "task_type": str(call["task_type"]),
            "provider": str(call["provider"]),
            "model_name": str(call["model_name"]),
            "status": str(call["status"]),
        }
        if call.get("response_model_name"):
            item["response_model_name"] = str(call["response_model_name"])
        models_used.append(item)
    return models_used


def _audio_missing_message(preprocessed: dict[str, Any]) -> str:
    """根据音频提取状态生成缺失音频证据的错误说明。"""

    artifacts = preprocessed.get("preprocessing_artifacts") or {}
    audio_status = artifacts.get("audio_extraction_status")
    if audio_status == "dependency_missing":
        return "本地未找到 ffmpeg，视频音频未提取。"
    if audio_status == "not_attempted_no_artifact_dir":
        return "未提供预处理产物目录，视频音频未写出为本地产物。"
    if audio_status == "timeout":
        return "ffmpeg 音频提取超时，视频音频未提取。"
    if audio_status == "empty_output":
        return "ffmpeg 执行成功但未写出有效音频文件。"
    if audio_status == "failed":
        return "视频音频提取失败。"
    return "视频V1尚未实现真实音频提取。"


def run_file_pipeline(
    file_record: dict[str, Any],
    routing_rules: dict[str, dict[str, str]],
    model_prices: dict[str, dict[str, Any]],
    *,
    route_plan: dict[str, Any] | None = None,
    ocr_backend: str = "mock",
    vision_understanding_backend: str = "mock",
    speech_to_text_backend: str = "mock",
    text_analysis_backend: str = "mock",
    defer_text_analysis: bool = False,
    deepseek_api_key: str | None = None,
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    deepseek_model_name: str = "deepseek-v4-flash",
    deepseek_max_retries: int = 0,
    deepseek_max_tokens: int = DEFAULT_DEEPSEEK_MAX_TOKENS,
    deepseek_compact_mode: bool = False,
    text_analysis_evidence_char_limit: int | None = None,
    qwen_ocr_api_key: str | None = None,
    qwen_ocr_base_url: str = DEFAULT_QWEN_VL_BASE_URL,
    qwen_ocr_model_name: str = DEFAULT_QWEN_OCR_MODEL_NAME,
    qwen_ocr_max_tokens: int = DEFAULT_QWEN_OCR_MAX_TOKENS,
    qwen_ocr_max_image_side: int | None = None,
    qwen_vl_api_key: str | None = None,
    qwen_vl_base_url: str = DEFAULT_QWEN_VL_BASE_URL,
    qwen_vl_model_name: str = DEFAULT_QWEN_VL_MODEL_NAME,
    qwen_vl_max_retries: int = 0,
    qwen_vl_max_tokens: int = DEFAULT_QWEN_VL_MAX_TOKENS,
    qwen_vl_max_image_side: int = DEFAULT_QWEN_VL_MAX_IMAGE_SIDE,
    dashscope_asr_api_key: str | None = None,
    dashscope_asr_submit_url: str = DEFAULT_DASHSCOPE_ASR_SUBMIT_URL,
    dashscope_asr_model_name: str = DEFAULT_DASHSCOPE_ASR_MODEL_NAME,
    asr_audio_url_map: dict[str, str] | None = None,
    preprocess_artifact_dir: str | Path | None = None,
    ffmpeg_path: str | Path | None = None,
    max_keyframes: int | None = None,
    fault_injection: dict[str, str] | None = None,
) -> dict[str, Any]:
    """运行单个文件处理流水线；图片和视频关键帧可按配置选择真实上游后端。"""

    media_type = str(file_record["media_type"])
    if route_plan is not None:
        planned_backends = route_plan_backends_for_media(route_plan, media_type)
        routing_rules = routing_rules_for_media(route_plan, media_type)
        ocr_backend = planned_backends["ocr_backend"]
        vision_understanding_backend = planned_backends["vision_understanding_backend"]
        speech_to_text_backend = planned_backends["speech_to_text_backend"]
        text_analysis_backend = planned_backends["text_analysis_backend"]

    if ocr_backend not in ALLOWED_OCR_BACKENDS:
        raise ValueError(f"不支持的 OCR 后端: {ocr_backend}")
    if vision_understanding_backend not in ALLOWED_VISION_BACKENDS:
        raise ValueError(f"不支持的视觉理解后端: {vision_understanding_backend}")
    if speech_to_text_backend not in ALLOWED_SPEECH_BACKENDS:
        raise ValueError(f"不支持的语音识别后端: {speech_to_text_backend}")
    if text_analysis_evidence_char_limit is not None and text_analysis_evidence_char_limit < 1:
        raise ValueError("文本分析证据字符上限必须大于等于 1。")
    pipeline_started_at = perf_counter()
    injected_failures = fault_injection or {}
    preprocessed = preprocess_file(
        file_record,
        artifact_dir=preprocess_artifact_dir,
        ffmpeg_path=ffmpeg_path,
        **({"max_keyframes": max_keyframes} if max_keyframes is not None else {}),
    )
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
        if missing_field not in missing_evidence:
            missing_evidence.append(missing_field)
        if quality_flag not in quality_flags:
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
        ocr_provider = "paddlepaddle" if ocr_backend == "paddleocr" else "qwen" if ocr_backend == "qwen_ocr" else None
        ocr_model_name = (
            DEFAULT_PADDLEOCR_MODEL_NAME if ocr_backend == "paddleocr" else qwen_ocr_model_name if ocr_backend == "qwen_ocr" else None
        )
        ocr_started_at = perf_counter()
        ocr_api_usage: dict[str, int] = {}
        ocr_response_model_name: str | None = None
        try:
            if "ocr" in injected_failures:
                raise RuntimeError(injected_failures["ocr"])
            ocr_result, ocr_api_usage, ocr_response_model_name = _run_ocr_backend(
                image_path,
                ocr_backend,
                qwen_ocr_api_key=qwen_ocr_api_key,
                qwen_ocr_base_url=qwen_ocr_base_url,
                qwen_ocr_model_name=qwen_ocr_model_name,
                qwen_ocr_max_tokens=qwen_ocr_max_tokens,
                qwen_ocr_max_image_side=qwen_ocr_max_image_side,
            )
        except Exception as exc:
            record_upstream_failure(
                task_type="ocr",
                missing_field="ocr_text",
                quality_flag="ocr_failed",
                warning_message="OCR 分支失败，最终分析未使用图片文字证据。",
                input_units=_ocr_input_units(ocr_backend, ocr_api_usage),
                output_units=_ocr_output_units(ocr_backend, ocr_api_usage, None),
                error_message=str(exc),
                latency_ms=int(round((perf_counter() - ocr_started_at) * 1000)),
                provider=ocr_provider,
                model_name=ocr_model_name,
            )
        else:
            evidence.update(ocr_result)
            if ocr_result["ocr_text"]:
                if ocr_backend != "mock" and _is_low_quality_ocr_text(ocr_result["ocr_text"]):
                    quality_flags.append(LOW_QUALITY_OCR_FLAG)
                    warning_messages.append(LOW_QUALITY_OCR_WARNING)
                else:
                    evidence_used.append("ocr_text")
            model_calls.append(
                _build_success_call(
                    call_index=len(model_calls) + 1,
                    file_record=file_record,
                    task_type="ocr",
                    input_units=_ocr_input_units(ocr_backend, ocr_api_usage),
                    output_units=_ocr_output_units(ocr_backend, ocr_api_usage, ocr_result["ocr_text"]),
                    routing_rules=routing_rules,
                    model_prices=model_prices,
                    latency_ms=int(round((perf_counter() - ocr_started_at) * 1000)),
                    provider=ocr_provider,
                    model_name=ocr_model_name,
                    response_model_name=ocr_response_model_name,
                )
            )

        vision_provider = "qwen" if vision_understanding_backend == "qwen_vl" else None
        vision_model_name = qwen_vl_model_name if vision_understanding_backend == "qwen_vl" else None
        vision_started_at = perf_counter()
        vision_api_usage: dict[str, int] = {}
        vision_response_model_name: str | None = None
        vision_api_attempts: list[dict[str, Any]] = []
        if "visual_understanding" in injected_failures:
            record_upstream_failure(
                task_type="visual_understanding",
                missing_field="visual_description",
                quality_flag="visual_understanding_failed",
                warning_message="视觉理解分支失败，最终分析未使用画面描述证据。",
                input_units=_vision_input_units(vision_api_usage)
                if vision_understanding_backend == "qwen_vl"
                else [{"unit_type": "frame_count", "quantity": 1}],
                output_units=_output_tokens(""),
                provider=vision_provider,
                model_name=vision_model_name,
            )
        else:
            try:
                if vision_understanding_backend == "qwen_vl":
                    vision_result = qwen_vl_image_understanding_client(
                        image_path,
                        api_key=qwen_vl_api_key,
                        model_name=qwen_vl_model_name,
                        base_url=qwen_vl_base_url,
                        max_retries=qwen_vl_max_retries,
                        max_tokens=qwen_vl_max_tokens,
                        max_image_side=qwen_vl_max_image_side,
                    )
                    vision_api_usage = vision_result.pop("_api_usage", {})
                    vision_response_model_name = vision_result.pop("_response_model_name", None)
                    vision_api_attempts = vision_result.pop("_api_attempts", [])
                else:
                    vision_result = mock_vision_client(image_path)
            except QwenVLAttemptsExhausted as exc:
                for attempt in exc.attempts:
                    failed_call = _build_qwen_vl_attempt_call(
                        attempt=attempt,
                        call_index=len(model_calls) + 1,
                        file_record=file_record,
                        visual_description=None,
                        routing_rules=routing_rules,
                        model_prices=model_prices,
                        provider=vision_provider or "qwen",
                        model_name=vision_model_name or qwen_vl_model_name,
                    )
                    model_calls.append(failed_call)
                    errors.append(
                        {
                            "batch_id": file_record["batch_id"],
                            "file_id": file_record["file_id"],
                            "call_id": failed_call["call_id"],
                            "error_level": "model_call",
                            "task_type": "visual_understanding",
                            "error_message": failed_call["error_message"],
                        }
                    )
                if "visual_description" not in missing_evidence:
                    missing_evidence.append("visual_description")
                if "visual_understanding_failed" not in quality_flags:
                    quality_flags.append("visual_understanding_failed")
                warning_messages.append("视觉理解分支重试后仍失败，最终分析未使用画面描述证据。")
            except Exception as exc:
                record_upstream_failure(
                    task_type="visual_understanding",
                    missing_field="visual_description",
                    quality_flag="visual_understanding_failed",
                    warning_message="视觉理解分支失败，最终分析未使用画面描述证据。",
                    input_units=_vision_input_units(vision_api_usage)
                    if vision_understanding_backend == "qwen_vl"
                    else [{"unit_type": "frame_count", "quantity": 1}],
                    output_units=_vision_output_units(vision_api_usage, None)
                    if vision_understanding_backend == "qwen_vl"
                    else _output_tokens(""),
                    error_message=str(exc),
                    latency_ms=int(round((perf_counter() - vision_started_at) * 1000)),
                    provider=vision_provider,
                    model_name=vision_model_name,
                )
            else:
                evidence.update(vision_result)
                evidence_used.append("visual_description")
                if vision_understanding_backend == "qwen_vl" and vision_api_attempts:
                    for attempt in vision_api_attempts:
                        attempt_call = _build_qwen_vl_attempt_call(
                            attempt=attempt,
                            call_index=len(model_calls) + 1,
                            file_record=file_record,
                            visual_description=vision_result["visual_description"],
                            routing_rules=routing_rules,
                            model_prices=model_prices,
                            provider=vision_provider or "qwen",
                            model_name=vision_model_name or qwen_vl_model_name,
                        )
                        model_calls.append(attempt_call)
                        if attempt_call["status"] == "failed":
                            recovered_errors.append(
                                {
                                    "batch_id": file_record["batch_id"],
                                    "file_id": file_record["file_id"],
                                    "call_id": attempt_call["call_id"],
                                    "error_level": "model_call",
                                    "task_type": "visual_understanding",
                                    "error_message": attempt_call["error_message"],
                                }
                            )
                else:
                    model_calls.append(
                        _build_success_call(
                            call_index=len(model_calls) + 1,
                            file_record=file_record,
                            task_type="visual_understanding",
                            input_units=_vision_input_units(vision_api_usage)
                            if vision_understanding_backend == "qwen_vl"
                            else [{"unit_type": "frame_count", "quantity": 1}],
                            output_units=_vision_output_units(
                                vision_api_usage,
                                vision_result["visual_description"],
                            )
                            if vision_understanding_backend == "qwen_vl"
                            else _output_tokens(vision_result["visual_description"]),
                            routing_rules=routing_rules,
                            model_prices=model_prices,
                            latency_ms=int(round((perf_counter() - vision_started_at) * 1000)),
                            provider=vision_provider,
                            model_name=vision_model_name,
                            response_model_name=vision_response_model_name,
                        )
                    )

    elif media_type == "video":
        keyframe_paths = preprocessed.get("keyframes") or []
        keyframe_count = len(keyframe_paths)
        audio_path = preprocessed.get("audio_path")
        preprocessing_artifacts = preprocessed.get("preprocessing_artifacts") or {}
        for warning in preprocessing_artifacts.get("warning_messages", []):
            if warning not in warning_messages:
                warning_messages.append(str(warning))
        if preprocessing_artifacts.get("video_evidence_stability") == "weak":
            if VIDEO_EVIDENCE_WEAK_FLAG not in quality_flags:
                quality_flags.append(VIDEO_EVIDENCE_WEAK_FLAG)

        ocr_provider = "paddlepaddle" if ocr_backend == "paddleocr" else "qwen" if ocr_backend == "qwen_ocr" else None
        ocr_model_name = (
            DEFAULT_PADDLEOCR_MODEL_NAME if ocr_backend == "paddleocr" else qwen_ocr_model_name if ocr_backend == "qwen_ocr" else None
        )
        if "ocr" in injected_failures:
            record_upstream_failure(
                task_type="ocr",
                missing_field="ocr_text",
                quality_flag="ocr_failed",
                warning_message="OCR 分支失败，最终分析未使用视频关键帧文字证据。",
                input_units=[{"unit_type": "image_count", "quantity": max(1, keyframe_count)}],
                output_units=_output_text_chars(""),
                provider=ocr_provider,
                model_name=ocr_model_name,
            )
        elif not keyframe_paths:
            record_upstream_failure(
                task_type="ocr",
                missing_field="ocr_text",
                quality_flag="video_keyframe_missing",
                warning_message="视频预处理未产出可用关键帧，最终分析未使用视频关键帧文字证据。",
                input_units=[{"unit_type": "image_count", "quantity": 0}],
                output_units=_output_text_chars(""),
                error_message="视频预处理未产出可用关键帧。",
                provider=ocr_provider,
                model_name=ocr_model_name,
            )
        else:
            usable_ocr_texts: list[str] = []
            raw_ocr_texts: list[str] = []
            low_quality_keyframes = 0
            for keyframe_index, keyframe_path in enumerate(keyframe_paths, start=1):
                ocr_started_at = perf_counter()
                ocr_api_usage: dict[str, int] = {}
                ocr_response_model_name: str | None = None
                try:
                    ocr_result, ocr_api_usage, ocr_response_model_name = _run_ocr_backend(
                        keyframe_path,
                        ocr_backend,
                        qwen_ocr_api_key=qwen_ocr_api_key,
                        qwen_ocr_base_url=qwen_ocr_base_url,
                        qwen_ocr_model_name=qwen_ocr_model_name,
                        qwen_ocr_max_tokens=qwen_ocr_max_tokens,
                        qwen_ocr_max_image_side=qwen_ocr_max_image_side,
                    )
                except Exception as exc:
                    failed_call = _build_failed_call(
                        call_index=len(model_calls) + 1,
                        file_record=file_record,
                        task_type="ocr",
                        input_units=_ocr_input_units(ocr_backend, ocr_api_usage),
                        output_units=_ocr_output_units(ocr_backend, ocr_api_usage, None),
                        routing_rules=routing_rules,
                        model_prices=model_prices,
                        error_message=str(exc),
                        latency_ms=int(round((perf_counter() - ocr_started_at) * 1000)),
                        provider=ocr_provider,
                        model_name=ocr_model_name,
                    )
                    model_calls.append(failed_call)
                    if VIDEO_OCR_KEYFRAME_FAILED_FLAG not in quality_flags:
                        quality_flags.append(VIDEO_OCR_KEYFRAME_FAILED_FLAG)
                    warning_messages.append(f"第 {keyframe_index} 张视频关键帧 OCR 失败，最终分析未使用该帧文字证据。")
                    errors.append(
                        {
                            "batch_id": file_record["batch_id"],
                            "file_id": file_record["file_id"],
                            "call_id": failed_call["call_id"],
                            "error_level": "model_call",
                            "task_type": "ocr",
                            "error_message": str(exc),
                        }
                    )
                    continue

                ocr_text = ocr_result["ocr_text"]
                if ocr_text:
                    frame_text = f"[关键帧 {keyframe_index}] {ocr_text}"
                    raw_ocr_texts.append(frame_text)
                    if ocr_backend != "mock" and _is_low_quality_ocr_text(ocr_text):
                        low_quality_keyframes += 1
                    else:
                        usable_ocr_texts.append(frame_text)
                model_calls.append(
                    _build_success_call(
                        call_index=len(model_calls) + 1,
                        file_record=file_record,
                        task_type="ocr",
                        input_units=_ocr_input_units(ocr_backend, ocr_api_usage),
                        output_units=_ocr_output_units(ocr_backend, ocr_api_usage, ocr_text),
                        routing_rules=routing_rules,
                        model_prices=model_prices,
                        latency_ms=int(round((perf_counter() - ocr_started_at) * 1000)),
                        provider=ocr_provider,
                        model_name=ocr_model_name,
                        response_model_name=ocr_response_model_name,
                    )
                )

            if usable_ocr_texts:
                evidence["ocr_text"] = "\n\n".join(usable_ocr_texts)
                evidence_used.append("ocr_text")
                if low_quality_keyframes:
                    warning_messages.append("部分视频关键帧 OCR 文本疑似低质量，已从下游文本分析证据中排除。")
            elif raw_ocr_texts:
                evidence["ocr_text"] = "\n\n".join(raw_ocr_texts)
                if LOW_QUALITY_OCR_FLAG not in quality_flags:
                    quality_flags.append(LOW_QUALITY_OCR_FLAG)
                warning_messages.append(LOW_QUALITY_OCR_WARNING)
            elif VIDEO_OCR_KEYFRAME_FAILED_FLAG in quality_flags and "ocr_text" not in missing_evidence:
                missing_evidence.append("ocr_text")

        vision_provider = "qwen" if vision_understanding_backend == "qwen_vl" else None
        vision_model_name = qwen_vl_model_name if vision_understanding_backend == "qwen_vl" else None
        vision_api_usage: dict[str, int] = {}
        if "visual_understanding" in injected_failures:
            record_upstream_failure(
                task_type="visual_understanding",
                missing_field="visual_description",
                quality_flag="visual_understanding_failed",
                warning_message="视觉理解分支失败，最终分析未使用视频画面描述证据。",
                input_units=_vision_input_units(vision_api_usage)
                if vision_understanding_backend == "qwen_vl"
                else [{"unit_type": "frame_count", "quantity": max(1, keyframe_count)}],
                output_units=_output_tokens(""),
                provider=vision_provider,
                model_name=vision_model_name,
            )
        elif not keyframe_paths:
            record_upstream_failure(
                task_type="visual_understanding",
                missing_field="visual_description",
                quality_flag="video_keyframe_missing",
                warning_message="视频预处理未产出可用关键帧，最终分析未使用视频画面描述证据。",
                input_units=_vision_input_units(vision_api_usage)
                if vision_understanding_backend == "qwen_vl"
                else [{"unit_type": "frame_count", "quantity": 0}],
                output_units=_vision_output_units(vision_api_usage, None)
                if vision_understanding_backend == "qwen_vl"
                else _output_tokens(""),
                error_message="视频预处理未产出可用关键帧。",
                provider=vision_provider,
                model_name=vision_model_name,
            )
        else:
            visual_descriptions: list[str] = []
            for keyframe_index, keyframe_path in enumerate(keyframe_paths, start=1):
                vision_started_at = perf_counter()
                vision_api_usage = {}
                vision_response_model_name: str | None = None
                vision_api_attempts: list[dict[str, Any]] = []
                try:
                    if vision_understanding_backend == "qwen_vl":
                        vision_result = qwen_vl_image_understanding_client(
                            keyframe_path,
                            api_key=qwen_vl_api_key,
                            model_name=qwen_vl_model_name,
                            base_url=qwen_vl_base_url,
                            max_retries=qwen_vl_max_retries,
                            max_tokens=qwen_vl_max_tokens,
                            max_image_side=qwen_vl_max_image_side,
                        )
                        vision_api_usage = vision_result.pop("_api_usage", {})
                        vision_response_model_name = vision_result.pop("_response_model_name", None)
                        vision_api_attempts = vision_result.pop("_api_attempts", [])
                    else:
                        vision_result = mock_vision_client(keyframe_path)
                except QwenVLAttemptsExhausted as exc:
                    for attempt in exc.attempts:
                        failed_call = _build_qwen_vl_attempt_call(
                            attempt=attempt,
                            call_index=len(model_calls) + 1,
                            file_record=file_record,
                            visual_description=None,
                            routing_rules=routing_rules,
                            model_prices=model_prices,
                            provider=vision_provider or "qwen",
                            model_name=vision_model_name or qwen_vl_model_name,
                        )
                        model_calls.append(failed_call)
                        errors.append(
                            {
                                "batch_id": file_record["batch_id"],
                                "file_id": file_record["file_id"],
                                "call_id": failed_call["call_id"],
                                "error_level": "model_call",
                                "task_type": "visual_understanding",
                                "error_message": failed_call["error_message"],
                            }
                        )
                    if VIDEO_VISUAL_KEYFRAME_FAILED_FLAG not in quality_flags:
                        quality_flags.append(VIDEO_VISUAL_KEYFRAME_FAILED_FLAG)
                    warning_messages.append(f"第 {keyframe_index} 张视频关键帧 Qwen-VL 重试后仍失败，最终分析未使用该帧画面描述证据。")
                    continue
                except Exception as exc:
                    failed_call = _build_failed_call(
                        call_index=len(model_calls) + 1,
                        file_record=file_record,
                        task_type="visual_understanding",
                        input_units=_vision_input_units(vision_api_usage)
                        if vision_understanding_backend == "qwen_vl"
                        else [{"unit_type": "frame_count", "quantity": 1}],
                        output_units=_vision_output_units(vision_api_usage, None)
                        if vision_understanding_backend == "qwen_vl"
                        else _output_tokens(""),
                        routing_rules=routing_rules,
                        model_prices=model_prices,
                        error_message=str(exc),
                        latency_ms=int(round((perf_counter() - vision_started_at) * 1000)),
                        provider=vision_provider,
                        model_name=vision_model_name,
                    )
                    model_calls.append(failed_call)
                    if VIDEO_VISUAL_KEYFRAME_FAILED_FLAG not in quality_flags:
                        quality_flags.append(VIDEO_VISUAL_KEYFRAME_FAILED_FLAG)
                    warning_messages.append(f"第 {keyframe_index} 张视频关键帧视觉理解失败，最终分析未使用该帧画面描述证据。")
                    errors.append(
                        {
                            "batch_id": file_record["batch_id"],
                            "file_id": file_record["file_id"],
                            "call_id": failed_call["call_id"],
                            "error_level": "model_call",
                            "task_type": "visual_understanding",
                            "error_message": str(exc),
                        }
                    )
                    continue

                visual_description = vision_result["visual_description"]
                visual_descriptions.append(f"[关键帧 {keyframe_index}] {visual_description}")
                if vision_understanding_backend == "qwen_vl" and vision_api_attempts:
                    for attempt in vision_api_attempts:
                        attempt_call = _build_qwen_vl_attempt_call(
                            attempt=attempt,
                            call_index=len(model_calls) + 1,
                            file_record=file_record,
                            visual_description=visual_description,
                            routing_rules=routing_rules,
                            model_prices=model_prices,
                            provider=vision_provider or "qwen",
                            model_name=vision_model_name or qwen_vl_model_name,
                        )
                        model_calls.append(attempt_call)
                        if attempt_call["status"] == "failed":
                            recovered_errors.append(
                                {
                                    "batch_id": file_record["batch_id"],
                                    "file_id": file_record["file_id"],
                                    "call_id": attempt_call["call_id"],
                                    "error_level": "model_call",
                                    "task_type": "visual_understanding",
                                    "error_message": attempt_call["error_message"],
                                }
                            )
                else:
                    model_calls.append(
                        _build_success_call(
                            call_index=len(model_calls) + 1,
                            file_record=file_record,
                            task_type="visual_understanding",
                            input_units=_vision_input_units(vision_api_usage)
                            if vision_understanding_backend == "qwen_vl"
                            else [{"unit_type": "frame_count", "quantity": 1}],
                            output_units=_vision_output_units(
                                vision_api_usage,
                                visual_description,
                            )
                            if vision_understanding_backend == "qwen_vl"
                            else _output_tokens(visual_description),
                            routing_rules=routing_rules,
                            model_prices=model_prices,
                            latency_ms=int(round((perf_counter() - vision_started_at) * 1000)),
                            provider=vision_provider,
                            model_name=vision_model_name,
                            response_model_name=vision_response_model_name,
                        )
                    )

            if visual_descriptions:
                evidence["visual_description"] = "\n\n".join(visual_descriptions)
                evidence_used.append("visual_description")
            elif VIDEO_VISUAL_KEYFRAME_FAILED_FLAG in quality_flags and "visual_description" not in missing_evidence:
                missing_evidence.append("visual_description")

        if "speech_to_text" in injected_failures:
            record_upstream_failure(
                task_type="speech_to_text",
                missing_field="audio_transcript",
                quality_flag="speech_to_text_failed",
                warning_message="语音识别分支失败，最终分析未使用音频转写证据。",
                input_units=[{"unit_type": "audio_seconds", "quantity": _video_audio_seconds(preprocessed)}],
                output_units=_output_text_chars(""),
            )
        elif audio_path is None:
            audio_error_message = _audio_missing_message(preprocessed)
            record_upstream_failure(
                task_type="speech_to_text",
                missing_field="audio_transcript",
                quality_flag="video_audio_not_extracted",
                warning_message="视频音频未成功提取，最终分析未使用音频转写证据。",
                input_units=[{"unit_type": "audio_seconds", "quantity": 0}],
                output_units=_output_text_chars(""),
                error_message=audio_error_message,
            )
        else:
            asr_started_at = perf_counter()
            asr_provider = "dashscope" if speech_to_text_backend == "dashscope_asr" else None
            asr_model_name = dashscope_asr_model_name if speech_to_text_backend == "dashscope_asr" else None
            asr_request_started = speech_to_text_backend != "dashscope_asr"
            try:
                if speech_to_text_backend == "dashscope_asr":
                    audio_url = _audio_url_for_record(
                        file_record=file_record,
                        audio_path=audio_path,
                        audio_url_map=asr_audio_url_map,
                    )
                    if not audio_url:
                        audio_url = dashscope_upload_local_file(
                            audio_path,
                            api_key=dashscope_asr_api_key,
                            model_name=dashscope_asr_model_name,
                        )
                    asr_request_started = True
                    asr_result = dashscope_asr_client(
                        audio_url,
                        api_key=dashscope_asr_api_key,
                        model_name=dashscope_asr_model_name,
                        submit_url=dashscope_asr_submit_url,
                    )
                    response_model_name = asr_result.pop("_response_model_name", dashscope_asr_model_name)
                    asr_result.pop("_api_usage", {})
                    asr_result.pop("_response_diagnostics", {})
                else:
                    asr_result = mock_asr_client(audio_path)
                    response_model_name = None
                evidence.update(asr_result)
                evidence_used.append("audio_transcript")
                model_calls.append(
                    _build_success_call(
                        call_index=len(model_calls) + 1,
                        file_record=file_record,
                        task_type="speech_to_text",
                        input_units=[{"unit_type": "audio_seconds", "quantity": _video_audio_seconds(preprocessed)}],
                        output_units=_output_text_chars(asr_result["audio_transcript"]),
                        routing_rules=routing_rules,
                        model_prices=model_prices,
                        latency_ms=int(round((perf_counter() - asr_started_at) * 1000)),
                        provider=asr_provider,
                        model_name=asr_model_name,
                        response_model_name=response_model_name,
                    )
                )
            except Exception as exc:
                record_upstream_failure(
                    task_type="speech_to_text",
                    missing_field="audio_transcript",
                    quality_flag="speech_to_text_failed",
                    warning_message="语音识别分支失败，最终分析未使用音频转写证据。",
                    input_units=[
                        {
                            "unit_type": "audio_seconds",
                            "quantity": _video_audio_seconds(preprocessed) if asr_request_started else 0,
                        }
                    ],
                    output_units=_output_text_chars(""),
                    error_message=str(exc),
                    latency_ms=int(round((perf_counter() - asr_started_at) * 1000)),
                    provider=asr_provider,
                    model_name=asr_model_name,
                )

    else:
        raise ValueError(f"不支持的文件类型: {media_type}")

    if defer_text_analysis:
        has_usable_evidence = any(
            isinstance(value, str) and bool(value.strip())
            for value in evidence.values()
        )
        deferred_quality_flags = list(
            dict.fromkeys([*quality_flags, DEFERRED_TEXT_QUALITY_FLAG if has_usable_evidence else NO_EVIDENCE_QUALITY_FLAG])
        )
        deferred_warnings = list(
            dict.fromkeys([*warning_messages, DEFERRED_TEXT_WARNING if has_usable_evidence else NO_EVIDENCE_ERROR])
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
            "preprocessing_artifacts": preprocessed.get("preprocessing_artifacts"),
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
            "processing_status": DEFERRED_TEXT_STATUS if has_usable_evidence else NO_EVIDENCE_STATUS,
            "evidence_used": evidence_used,
            "missing_evidence": missing_evidence,
            "quality_flags": deferred_quality_flags,
            "warning_messages": deferred_warnings,
            "error_message": (
                "；".join(str(error["error_message"]) for error in errors)
                if errors
                else None if has_usable_evidence else NO_EVIDENCE_ERROR
            ),
            "call_ids": [call["call_id"] for call in model_calls],
            "models_used": _models_used_from_calls(model_calls),
            "processing_cost_cny": round(sum(float(call["cost_cny"]) for call in model_calls), 6),
            "processing_time_ms": int(round((perf_counter() - pipeline_started_at) * 1000)),
        }
        return {"result": result, "model_calls": model_calls, "errors": errors}

    analysis_evidence = dict(evidence)
    if LOW_QUALITY_OCR_FLAG in quality_flags:
        analysis_evidence["ocr_text"] = None
    analysis_evidence, evidence_truncated = _limit_text_analysis_evidence(
        analysis_evidence,
        text_analysis_evidence_char_limit,
    )
    if evidence_truncated:
        if TEXT_ANALYSIS_EVIDENCE_TRUNCATED_FLAG not in quality_flags:
            quality_flags.append(TEXT_ANALYSIS_EVIDENCE_TRUNCATED_FLAG)
        if TEXT_ANALYSIS_EVIDENCE_TRUNCATED_WARNING not in warning_messages:
            warning_messages.append(TEXT_ANALYSIS_EVIDENCE_TRUNCATED_WARNING)

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
                max_tokens=deepseek_max_tokens,
                compact_mode=deepseek_compact_mode,
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
                response_diagnostics=attempt.get("response_diagnostics"),
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
            "preprocessing_artifacts": preprocessed.get("preprocessing_artifacts"),
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
                response_diagnostics=attempt.get("response_diagnostics"),
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
        "preprocessing_artifacts": preprocessed.get("preprocessing_artifacts"),
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
