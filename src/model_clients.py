"""封装 OCR、语音识别、视觉理解和文本分析的真实或模拟模型调用。

客户端只返回模型输出，不直接写入结果、模型调用日志或错误记录。
"""

from __future__ import annotations

import base64
from http import client as http_client
import json
import mimetypes
import re
from functools import lru_cache
from pathlib import Path
from time import sleep
from time import perf_counter
from typing import Any
from urllib.parse import urlparse
from urllib import error, request

from runtime_config import runtime_policy_section

TOPIC_POLICY = runtime_policy_section("topics")
TOPIC_VALUES = set(TOPIC_POLICY.get("values", []))
TOPIC_KEYWORDS = TOPIC_POLICY.get("mock_keywords", {})
DEEPSEEK_PROMPT_POLICY = runtime_policy_section("deepseek_prompt")

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_PADDLEOCR_MODEL_NAME = "PP-OCRv5_mobile"
DEFAULT_QWEN_VL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_VL_MODEL_NAME = "qwen-vl-plus"
DEFAULT_DASHSCOPE_ASR_MODEL_NAME = "paraformer-v2"
DEFAULT_DASHSCOPE_ASR_SUBMIT_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
DEFAULT_DEEPSEEK_MAX_TOKENS = 2500
DEFAULT_QWEN_VL_MAX_TOKENS = 500
DEEPSEEK_CONTENT_PREVIEW_CHARS = 120
DEEPSEEK_COMPACT_EVIDENCE_CHARS = 500

CONSERVATIVE_BUSINESS_USE = "可用于内容归档、检索和人工复核。"

HIGH_RISK_COMMERCIAL_USE_TERMS = (
    "品牌推广",
    "广告投放",
    "带货",
    "购买转化",
    "销售转化",
    "营销转化",
    "商品推广",
    "商业推广",
)

POSITIVE_COMMERCIAL_EVIDENCE_TERMS = (
    "本期是品牌合作",
    "品牌合作内容",
    "广告合作",
    "商家赞助",
    "官方推广",
    "购买链接",
    "购买入口",
    "下单",
    "优惠券",
    "促销",
    "带货",
)

NEGATED_COMMERCIAL_EVIDENCE_TERMS = (
    "不是品牌合作",
    "非品牌合作",
    "没有品牌合作",
    "不是广告",
    "非广告",
    "没有广告",
    "不是商家赞助",
    "没有购买链接",
    "没有购买入口",
    "不含购买链接",
    "没有推广",
)


class DeepSeekResponseError(RuntimeError):
    """表示一次 DeepSeek 请求产生了可识别的响应错误。"""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        retryable: bool,
        api_usage: dict[str, int] | None = None,
        response_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"[{error_code}] {message}")
        self.error_code = error_code
        self.retryable = retryable
        self.api_usage = api_usage or {}
        self.response_diagnostics = response_diagnostics or {}


class DashScopeASRResponseError(RuntimeError):
    """表示一次 DashScope ASR 请求产生了可识别的响应错误。"""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        api_usage: dict[str, int] | None = None,
        response_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"[{error_code}] {message}")
        self.error_code = error_code
        self.api_usage = api_usage or {}
        self.response_diagnostics = response_diagnostics or {}


class DeepSeekAttemptsExhausted(RuntimeError):
    """表示 DeepSeek 调用未成功，并保留每次尝试的计量信息。"""

    def __init__(self, last_error: DeepSeekResponseError, attempts: list[dict[str, Any]]) -> None:
        super().__init__(str(last_error))
        self.attempts = attempts


class PaddleOCRResponseError(RuntimeError):
    """表示 PaddleOCR 推理失败或返回结构不符合预期。"""


class QwenVLResponseError(RuntimeError):
    """表示一次 Qwen-VL 图片理解请求产生了可识别的响应错误。"""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        retryable: bool,
        api_usage: dict[str, int] | None = None,
        response_model_name: str | None = None,
    ) -> None:
        super().__init__(f"[{error_code}] {message}")
        self.error_code = error_code
        self.retryable = retryable
        self.api_usage = api_usage or {}
        self.response_model_name = response_model_name


class QwenVLAttemptsExhausted(RuntimeError):
    """表示 Qwen-VL 调用未成功，并保留每次尝试的计量信息。"""

    def __init__(self, last_error: QwenVLResponseError, attempts: list[dict[str, Any]]) -> None:
        super().__init__(str(last_error))
        self.attempts = attempts


def _require_positive_int(value: int, name: str) -> int:
    """校验模型生成长度等外部可调参数。"""

    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须是大于等于 1 的整数。")
    return value


def mock_ocr_client(image_path: str | Path) -> dict[str, str]:
    """模拟 OCR 调用，返回画面文字证据。"""

    file_name = Path(image_path).name
    return {"ocr_text": f"模拟 OCR 文字：{file_name}"}


def paddleocr_client(image_path: str | Path) -> dict[str, str | None]:
    """在本地调用 PaddleOCR，并返回标准化文字证据。"""

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"OCR 输入图片不存在：{path}")
    if path.stat().st_size == 0:
        raise ValueError("OCR 输入图片为空。")

    try:
        image_data = _decode_image_for_paddleocr(path)
        prediction = _create_paddleocr_engine().predict(image_data)
    except Exception as exc:
        raise PaddleOCRResponseError(f"PaddleOCR 本地推理失败：{exc}") from exc

    return _parse_paddleocr_prediction(prediction)


def _decode_image_for_paddleocr(path: Path) -> Any:
    """用二进制读取兼容中文路径的图片，并解码为 PaddleOCR 可接收的数组。"""

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("PaddleOCR 图片解码依赖未完整安装。") from exc

    encoded_image = np.fromfile(path, dtype=np.uint8)
    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if decoded_image is None:
        raise ValueError(f"OCR 输入文件不是可解码的图片：{path}")
    return decoded_image


@lru_cache(maxsize=1)
def _create_paddleocr_engine() -> Any:
    """延迟创建并复用本地 OCR 引擎，避免每张图片重复加载模型。"""

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "未安装 PaddleOCR 运行环境；请先按 README 安装 PaddlePaddle 和 PaddleOCR。"
        ) from exc
    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        enable_mkldnn=False,
    )


def _paddleocr_result_to_dict(result: Any) -> dict[str, Any]:
    """把 PaddleOCR Result 对象或字典统一转换为普通字典。"""

    if isinstance(result, dict):
        return result

    json_value = getattr(result, "json", None)
    if json_value is not None:
        json_value = json_value() if callable(json_value) else json_value
        if isinstance(json_value, str):
            try:
                json_value = json.loads(json_value)
            except json.JSONDecodeError as exc:
                raise PaddleOCRResponseError("PaddleOCR Result.json 不是有效 JSON。") from exc
        if isinstance(json_value, dict):
            return json_value

    raise PaddleOCRResponseError("PaddleOCR 单项结果无法转换为字典。")


def _parse_paddleocr_prediction(prediction: Any) -> dict[str, str | None]:
    """读取 PaddleOCR 的 rec_texts，并合并所有页面或图片的有效文字。"""

    if isinstance(prediction, dict):
        items = [prediction]
    else:
        try:
            items = list(prediction)
        except TypeError as exc:
            raise PaddleOCRResponseError("PaddleOCR predict 返回值不可迭代。") from exc

    cleaned_lines: list[str] = []
    for item in items:
        payload = _paddleocr_result_to_dict(item)
        if isinstance(payload.get("res"), dict):
            payload = payload["res"]
        recognized_texts = payload.get("rec_texts")
        if not isinstance(recognized_texts, list) or any(
            not isinstance(text, str) for text in recognized_texts
        ):
            raise PaddleOCRResponseError("PaddleOCR 结果中的 rec_texts 不是字符串数组。")
        cleaned_lines.extend(text.strip() for text in recognized_texts if text.strip())

    return {"ocr_text": "\n".join(cleaned_lines) if cleaned_lines else None}


def mock_asr_client(audio_path: str | Path) -> dict[str, str]:
    """模拟语音识别调用，返回音频转写证据。"""

    file_name = Path(audio_path).name
    return {"audio_transcript": f"模拟音频转写：{file_name}"}


def _read_json_response(api_request: request.Request, timeout_seconds: int) -> dict[str, Any]:
    """读取标准 JSON API 响应。"""

    try:
        with request.urlopen(api_request, timeout=timeout_seconds) as response:
            response_bytes = response.read()
    except error.HTTPError as exc:
        detail_bytes = exc.read()
        raise DashScopeASRResponseError(
            "dashscope_asr_http_error",
            f"DashScope ASR API 返回 HTTP {exc.code}，响应体长度 {len(detail_bytes)} 字节。",
        ) from exc
    except error.URLError as exc:
        raise DashScopeASRResponseError(
            "dashscope_asr_network_error",
            f"DashScope ASR API 网络连接失败：{exc.reason}",
        ) from exc

    try:
        response_text = response_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DashScopeASRResponseError(
            "dashscope_asr_response_invalid_encoding",
            f"DashScope ASR API 响应不是有效 UTF-8，响应体长度 {len(response_bytes)} 字节。",
        ) from exc

    try:
        response_data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise DashScopeASRResponseError(
            "dashscope_asr_response_invalid_json",
            f"DashScope ASR API 响应不是合法 JSON，长度 {len(response_text)} 字符。",
        ) from exc

    if not isinstance(response_data, dict):
        raise DashScopeASRResponseError(
            "dashscope_asr_response_invalid_schema",
            "DashScope ASR API 响应必须是 JSON 对象。",
        )
    return response_data


def dashscope_upload_local_file(
    file_path: str | Path,
    *,
    api_key: str | None,
    model_name: str = DEFAULT_DASHSCOPE_ASR_MODEL_NAME,
    timeout_seconds: int = 120,
) -> str:
    """上传本地音频到 DashScope 临时存储，并返回 oss:// URL。"""

    path = Path(file_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"ASR 本地音频文件不存在或为空：{path}")
    if not api_key:
        raise ValueError("缺少 DASHSCOPE_API_KEY，无法上传 ASR 音频文件。")

    try:
        from dashscope.utils.oss_utils import OssUtils
    except ImportError as exc:
        raise RuntimeError("未安装 DashScope SDK，无法自动上传本地 ASR 音频。请先安装 dashscope。") from exc

    try:
        oss_url, _ = OssUtils.upload(
            model=model_name,
            file_path=str(path),
            api_key=api_key,
        )
    except Exception as exc:
        message = str(exc).replace(api_key, "***") if api_key else str(exc)
        raise DashScopeASRResponseError(
            "dashscope_asr_upload_failed",
            f"DashScope ASR 本地音频上传失败：{message}",
        ) from exc
    if not isinstance(oss_url, str) or not oss_url.startswith("oss://"):
        raise DashScopeASRResponseError(
            "dashscope_asr_upload_missing_url",
            "DashScope ASR 本地音频上传未返回 oss:// URL。",
        )
    return oss_url


def _dashscope_task_url(submit_url: str, task_id: str) -> str:
    """根据提交接口生成任务查询接口。"""

    prefix = submit_url.split("/api/v1/", 1)[0]
    return f"{prefix}/api/v1/tasks/{task_id}"


def _extract_dashscope_output(response_data: dict[str, Any]) -> dict[str, Any]:
    """读取 DashScope 响应里的 output 对象。"""

    output = response_data.get("output")
    if not isinstance(output, dict):
        raise DashScopeASRResponseError(
            "dashscope_asr_missing_output",
            "DashScope ASR 响应缺少 output 对象。",
        )
    return output


def _extract_transcription_url(output: dict[str, Any]) -> str:
    """从任务结果中读取转写结果 URL。"""

    results = output.get("results")
    if not isinstance(results, list) or not results:
        raise DashScopeASRResponseError(
            "dashscope_asr_missing_results",
            "DashScope ASR 任务结果缺少 results。",
        )
    first = results[0]
    if not isinstance(first, dict):
        raise DashScopeASRResponseError(
            "dashscope_asr_invalid_result",
            "DashScope ASR results[0] 必须是对象。",
        )
    if first.get("subtask_status") not in {None, "SUCCEEDED"}:
        raise DashScopeASRResponseError(
            "dashscope_asr_subtask_failed",
            f"DashScope ASR 子任务未成功：{first.get('subtask_status')}",
        )
    transcription_url = first.get("transcription_url")
    if not isinstance(transcription_url, str) or not transcription_url.strip():
        raise DashScopeASRResponseError(
            "dashscope_asr_missing_transcription_url",
            "DashScope ASR 任务结果缺少 transcription_url。",
        )
    return transcription_url


def _extract_transcript_text(transcription_data: dict[str, Any]) -> str:
    """从 DashScope 转写 JSON 中读取文本。"""

    if isinstance(transcription_data.get("text"), str):
        text = transcription_data["text"].strip()
        if text:
            return text
    transcripts = transcription_data.get("transcripts")
    if isinstance(transcripts, list):
        lines = [
            item.get("text", "").strip()
            for item in transcripts
            if isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text", "").strip()
        ]
        if lines:
            return "\n".join(lines)
    sentences = transcription_data.get("sentences")
    if isinstance(sentences, list):
        lines = [
            item.get("text", "").strip()
            for item in sentences
            if isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text", "").strip()
        ]
        if lines:
            return "\n".join(lines)
    raise DashScopeASRResponseError(
        "dashscope_asr_transcript_empty",
        "DashScope ASR 转写结果没有可用文本。",
    )


def dashscope_asr_client(
    audio_url: str,
    *,
    api_key: str | None,
    model_name: str = DEFAULT_DASHSCOPE_ASR_MODEL_NAME,
    submit_url: str = DEFAULT_DASHSCOPE_ASR_SUBMIT_URL,
    timeout_seconds: int = 120,
    poll_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    """调用 DashScope Paraformer 录音文件识别，返回音频转写文本。"""

    parsed_audio_url = urlparse(audio_url)
    if parsed_audio_url.scheme not in {"http", "https", "oss"}:
        raise ValueError("DashScope ASR 需要可访问的 http/https 或 oss:// 音频 URL。")
    if not api_key:
        raise ValueError("缺少 DASHSCOPE_API_KEY，无法调用 DashScope ASR API。")

    submit_payload = {"model": model_name, "input": {"file_urls": [audio_url]}}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    if parsed_audio_url.scheme == "oss":
        headers["X-DashScope-OssResourceResolve"] = "enable"
    submit_request = request.Request(
        submit_url,
        data=json.dumps(submit_payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    submit_response = _read_json_response(submit_request, timeout_seconds)
    task_id = _extract_dashscope_output(submit_response).get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise DashScopeASRResponseError(
            "dashscope_asr_missing_task_id",
            "DashScope ASR 提交响应缺少 task_id。",
        )

    task_url = _dashscope_task_url(submit_url, task_id)
    deadline = perf_counter() + timeout_seconds
    task_output: dict[str, Any] | None = None
    while perf_counter() < deadline:
        task_request = request.Request(
            task_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
            },
            method="POST",
        )
        task_response = _read_json_response(task_request, timeout_seconds)
        output = _extract_dashscope_output(task_response)
        status = output.get("task_status")
        if status == "SUCCEEDED":
            task_output = output
            usage = task_response.get("usage") if isinstance(task_response.get("usage"), dict) else {}
            break
        if status in {"FAILED", "CANCELED"}:
            raise DashScopeASRResponseError(
                "dashscope_asr_task_failed",
                f"DashScope ASR 任务未成功：{status}",
            )
        sleep(poll_interval_seconds)
    else:
        raise DashScopeASRResponseError(
            "dashscope_asr_timeout",
            "DashScope ASR 任务等待超时。",
        )

    transcription_request = request.Request(_extract_transcription_url(task_output), method="GET")
    transcription_data = _read_json_response(transcription_request, timeout_seconds)
    transcript = _extract_transcript_text(transcription_data)
    return {
        "audio_transcript": transcript,
        "_api_usage": usage,
        "_response_model_name": model_name,
        "_response_diagnostics": {"task_id": task_id, "transcription_url_present": True},
    }


def mock_vision_client(image_path: str | Path) -> dict[str, str]:
    """模拟视觉理解调用，返回画面描述证据。"""

    file_name = Path(image_path).name
    return {"visual_description": f"模拟视觉描述：{file_name} 展示了一段待分析内容。"}


def qwen_vl_image_understanding_client(
    image_path: str | Path,
    *,
    api_key: str | None,
    model_name: str = DEFAULT_QWEN_VL_MODEL_NAME,
    base_url: str = DEFAULT_QWEN_VL_BASE_URL,
    timeout_seconds: int = 60,
    max_retries: int = 0,
    max_tokens: int = DEFAULT_QWEN_VL_MAX_TOKENS,
) -> dict[str, Any]:
    """调用 Qwen-VL 图片理解 API，返回标准化画面描述。"""

    if not api_key:
        raise ValueError("缺少 DASHSCOPE_API_KEY，无法调用 Qwen-VL API。")
    if max_retries not in {0, 1}:
        raise ValueError("Qwen-VL 最大重试次数只能是 0 或 1。")
    max_tokens = _require_positive_int(max_tokens, "Qwen-VL max_tokens")

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"视觉理解输入图片不存在：{path}")
    if path.stat().st_size == 0:
        raise ValueError("视觉理解输入图片为空。")

    payload = {
        "model": model_name,
        "messages": _build_qwen_vl_messages(path),
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": False,
    }
    api_request = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    attempts: list[dict[str, Any]] = []
    for attempt_index in range(max_retries + 1):
        attempt_started_at = perf_counter()
        try:
            result, api_usage, response_model_name = _perform_qwen_vl_request(api_request, timeout_seconds)
        except QwenVLResponseError as exc:
            attempts.append(
                {
                    "status": "failed",
                    "latency_ms": int(round((perf_counter() - attempt_started_at) * 1000)),
                    "api_usage": exc.api_usage,
                    "error_message": str(exc),
                    "response_model_name": exc.response_model_name,
                }
            )
            if exc.retryable and attempt_index < max_retries:
                continue
            if max_retries:
                raise QwenVLAttemptsExhausted(exc, attempts) from exc
            raise

        attempts.append(
            {
                "status": "success",
                "latency_ms": int(round((perf_counter() - attempt_started_at) * 1000)),
                "api_usage": api_usage,
                "error_message": None,
                "response_model_name": response_model_name,
            }
        )
        result["_api_usage"] = api_usage
        result["_response_model_name"] = response_model_name
        result["_api_attempts"] = attempts
        return result

    raise AssertionError("Qwen-VL 调用循环未返回结果。")


def _build_qwen_vl_messages(image_path: Path) -> list[dict[str, Any]]:
    """构造 Qwen-VL 图片理解提示词。"""

    system_prompt = """
你是内容平台 AI 团队的图片视觉理解助手。请只基于图片可见内容输出严格 json。
不要编造图片中看不到的信息，不要推断人物身份、品牌合作效果或业务结论。
如果图片文字很少，也要描述画面主体、布局、内容类型和可见元素。
visual_description 必须是中文，控制在 300 字以内。
""".strip()
    user_prompt = """
请描述这张图片的可见内容，用于后续内容分类、标签和摘要生成。
只输出以下 JSON：
{
  "visual_description": "图片画面描述"
}
""".strip()
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def _image_to_data_url(image_path: Path) -> str:
    """把本地图片转换成 OpenAI 兼容接口可接收的 Base64 Data URL。"""

    mime_type = mimetypes.guess_type(str(image_path))[0]
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"不支持的视觉理解图片类型：{image_path.suffix or '无后缀'}")

    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_image}"


def _perform_qwen_vl_request(
    api_request: request.Request,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, int], str | None]:
    """执行单次 Qwen-VL 请求，并解析、校验模型响应。"""

    try:
        with request.urlopen(api_request, timeout=timeout_seconds) as response:
            response_bytes = response.read()
    except error.HTTPError as exc:
        detail_bytes = exc.read()
        retryable = exc.code == 429 or exc.code >= 500
        raise QwenVLResponseError(
            "qwen_vl_http_error",
            f"Qwen-VL API 返回 HTTP {exc.code}，响应体长度 {len(detail_bytes)} 字节。",
            retryable=retryable,
        ) from exc
    except error.URLError as exc:
        raise QwenVLResponseError(
            "qwen_vl_network_error",
            f"Qwen-VL API 网络连接失败：{exc.reason}",
            retryable=True,
        ) from exc
    except http_client.RemoteDisconnected as exc:
        raise QwenVLResponseError(
            "qwen_vl_network_disconnected",
            "Qwen-VL API 网络连接被远端关闭。",
            retryable=True,
        ) from exc
    except (ConnectionResetError, ConnectionAbortedError) as exc:
        raise QwenVLResponseError(
            "qwen_vl_network_interrupted",
            f"Qwen-VL API 网络连接中断：{exc}",
            retryable=True,
        ) from exc
    except TimeoutError as exc:
        raise QwenVLResponseError(
            "qwen_vl_timeout",
            "Qwen-VL API 请求超时。",
            retryable=True,
        ) from exc

    try:
        response_text = response_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QwenVLResponseError(
            "qwen_vl_response_invalid_encoding",
            f"Qwen-VL API 响应不是有效 UTF-8，响应体长度 {len(response_bytes)} 字节。",
            retryable=True,
        ) from exc

    try:
        response_data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise QwenVLResponseError(
            "qwen_vl_response_invalid_json",
            f"Qwen-VL API 外层响应不是合法 JSON，长度 {len(response_text)} 字符。",
            retryable=True,
        ) from exc

    if not isinstance(response_data, dict):
        raise QwenVLResponseError(
            "qwen_vl_response_invalid_schema",
            "Qwen-VL API 外层响应必须是 JSON 对象。",
            retryable=True,
        )

    api_usage = _extract_api_usage(response_data)
    response_model_name = (
        str(response_data["model"]).strip()
        if response_data.get("model")
        else None
    )
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise QwenVLResponseError(
            "qwen_vl_response_missing_content",
            "Qwen-VL API 响应缺少 choices[0].message.content。",
            retryable=True,
            api_usage=api_usage,
            response_model_name=response_model_name,
        ) from exc

    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text"))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
        )
    if not isinstance(content, str) or not content.strip():
        raise QwenVLResponseError(
            "qwen_vl_content_empty",
            "Qwen-VL 模型内容为空。",
            retryable=True,
            api_usage=api_usage,
            response_model_name=response_model_name,
        )

    normalized_content = _strip_json_code_fence(content)
    try:
        vision_data = json.loads(normalized_content)
    except json.JSONDecodeError as exc:
        raise QwenVLResponseError(
            "qwen_vl_content_invalid_json",
            f"Qwen-VL 模型内容不是合法 JSON，长度 {len(content)} 字符。",
            retryable=True,
            api_usage=api_usage,
            response_model_name=response_model_name,
        ) from exc

    try:
        result = _normalize_vision_result(vision_data)
    except (TypeError, ValueError) as exc:
        raise QwenVLResponseError(
            "qwen_vl_content_invalid_schema",
            f"Qwen-VL 模型内容不符合视觉理解结构：{exc}",
            retryable=True,
            api_usage=api_usage,
            response_model_name=response_model_name,
        ) from exc
    return result, api_usage, response_model_name


def _normalize_vision_result(data: dict[str, Any]) -> dict[str, str]:
    """把真实视觉模型输出清洗成系统需要的固定结构。"""

    if not isinstance(data, dict):
        raise TypeError("视觉模型内容必须是 JSON 对象。")

    visual_description = str(data.get("visual_description") or "").strip()
    if not visual_description:
        raise ValueError("visual_description 不能为空。")
    return {"visual_description": visual_description[:300]}


def mock_text_analysis_client(evidence: dict[str, Any]) -> dict[str, Any]:
    """模拟文本分析调用，返回标准内容分析结果。"""

    available_text = " ".join(
        str(value)
        for key, value in evidence.items()
        if key in {"raw_text", "ocr_text", "audio_transcript", "visual_description"} and value
    )
    topic_scores = {
        topic: sum(1 for keyword in keywords if keyword in available_text)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    matched_topics = [topic for topic, score in sorted(topic_scores.items(), key=lambda item: item[1], reverse=True) if score > 0]
    topic = matched_topics[0] if matched_topics else "other"
    secondary_topics = matched_topics[1:3]
    tags = _build_mock_tags(available_text)

    return {
        "topic": topic,
        "secondary_topics": secondary_topics,
        "tags": tags,
        "summary": _build_mock_summary(available_text),
        "business_use": _build_mock_business_use(topic, tags),
    }


def deepseek_text_analysis_client(
    evidence: dict[str, Any],
    *,
    api_key: str | None,
    model_name: str = "deepseek-v4-flash",
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    timeout_seconds: int = 60,
    max_retries: int = 0,
    max_tokens: int = DEFAULT_DEEPSEEK_MAX_TOKENS,
) -> dict[str, Any]:
    """调用 DeepSeek API，返回标准化后的内容分析结果。"""

    if not api_key:
        raise ValueError("缺少 DEEPSEEK_API_KEY，无法调用 DeepSeek API。")
    if max_retries not in {0, 1}:
        raise ValueError("DeepSeek 最大重试次数只能是 0 或 1。")
    max_tokens = _require_positive_int(max_tokens, "DeepSeek max_tokens")

    payload = {
        "model": model_name,
        "messages": _build_deepseek_messages(evidence),
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": False,
    }

    attempts: list[dict[str, Any]] = []
    for attempt_index in range(max_retries + 1):
        attempt_started_at = perf_counter()
        api_request = request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            result, api_usage = _perform_deepseek_request(
                api_request,
                timeout_seconds,
                max_tokens=int(payload["max_tokens"]),
            )
        except DeepSeekResponseError as exc:
            attempts.append(
                {
                    "status": "failed",
                    "latency_ms": int(round((perf_counter() - attempt_started_at) * 1000)),
                    "api_usage": exc.api_usage,
                    "error_message": str(exc),
                    "response_diagnostics": exc.response_diagnostics,
                }
            )
            if exc.retryable and attempt_index < max_retries:
                if exc.error_code == "deepseek_content_empty" and exc.response_diagnostics.get("hit_max_tokens"):
                    payload["messages"] = _build_deepseek_messages(evidence, compact=True)
                continue
            raise DeepSeekAttemptsExhausted(exc, attempts) from exc

        attempts.append(
            {
                "status": "success",
                "latency_ms": int(round((perf_counter() - attempt_started_at) * 1000)),
                "api_usage": api_usage,
                "error_message": None,
            }
        )
        grounded_business_use, guard_applied = _ground_business_use(result["business_use"], evidence)
        result["business_use"] = grounded_business_use
        result["_quality_flags"] = ["business_use_grounded_fallback"] if guard_applied else []
        result["_api_usage"] = api_usage
        result["_api_attempts"] = attempts
        return result

    raise AssertionError("DeepSeek 调用循环未返回结果。")


def _perform_deepseek_request(
    api_request: request.Request,
    timeout_seconds: int,
    *,
    max_tokens: int | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """执行单次 DeepSeek 请求，并解析、校验模型响应。"""

    try:
        with request.urlopen(api_request, timeout=timeout_seconds) as response:
            response_bytes = response.read()
    except error.HTTPError as exc:
        detail_bytes = exc.read()
        retryable = exc.code == 429 or exc.code >= 500
        raise DeepSeekResponseError(
            "deepseek_http_error",
            f"DeepSeek API 返回 HTTP {exc.code}，响应体长度 {len(detail_bytes)} 字节。",
            retryable=retryable,
        ) from exc
    except error.URLError as exc:
        raise DeepSeekResponseError(
            "deepseek_network_error",
            f"DeepSeek API 网络连接失败：{exc.reason}",
            retryable=True,
        ) from exc
    except TimeoutError as exc:
        raise DeepSeekResponseError(
            "deepseek_timeout",
            "DeepSeek API 请求超时。",
            retryable=True,
        ) from exc

    try:
        response_text = response_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DeepSeekResponseError(
            "deepseek_response_invalid_encoding",
            f"DeepSeek API 响应不是有效 UTF-8，响应体长度 {len(response_bytes)} 字节。",
            retryable=True,
        ) from exc

    try:
        response_data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise DeepSeekResponseError(
            "deepseek_response_invalid_json",
            f"DeepSeek API 外层响应不是合法 JSON，长度 {len(response_text)} 字符，错误位置第 {exc.lineno} 行第 {exc.colno} 列。",
            retryable=True,
        ) from exc

    if not isinstance(response_data, dict):
        raise DeepSeekResponseError(
            "deepseek_response_invalid_schema",
            "DeepSeek API 外层响应必须是 JSON 对象。",
            retryable=True,
        )

    api_usage = _extract_api_usage(response_data)
    response_diagnostics = _build_deepseek_response_diagnostics(
        response_data,
        max_tokens=max_tokens,
    )
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DeepSeekResponseError(
            "deepseek_response_missing_content",
            "DeepSeek API 响应缺少 choices[0].message.content。",
            retryable=True,
            api_usage=api_usage,
            response_diagnostics=response_diagnostics,
        ) from exc

    response_diagnostics = _build_deepseek_response_diagnostics(
        response_data,
        content=content,
        max_tokens=max_tokens,
    )
    if not isinstance(content, str) or not content.strip():
        message = "DeepSeek 模型内容为空。"
        if response_diagnostics.get("hit_max_tokens"):
            message = "DeepSeek 模型内容为空；本次响应命中 max_tokens，可能需要提高输出长度或压缩输入证据。"
        raise DeepSeekResponseError(
            "deepseek_content_empty",
            message,
            retryable=True,
            api_usage=api_usage,
            response_diagnostics=response_diagnostics,
        )

    normalized_content = _strip_json_code_fence(content)
    try:
        analysis_data = json.loads(normalized_content)
    except json.JSONDecodeError as exc:
        raise DeepSeekResponseError(
            "deepseek_content_invalid_json",
            f"DeepSeek 模型内容不是合法 JSON，长度 {len(content)} 字符，错误位置第 {exc.lineno} 行第 {exc.colno} 列。",
            retryable=True,
            api_usage=api_usage,
            response_diagnostics=response_diagnostics,
        ) from exc

    try:
        result = _normalize_analysis_result(analysis_data)
    except (TypeError, ValueError) as exc:
        raise DeepSeekResponseError(
            "deepseek_content_invalid_schema",
            f"DeepSeek 模型内容不符合结果结构：{exc}",
            retryable=True,
            api_usage=api_usage,
            response_diagnostics=response_diagnostics,
        ) from exc
    return result, api_usage


def _build_deepseek_response_diagnostics(
    response_data: dict[str, Any],
    *,
    content: Any = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """提取 DeepSeek 响应诊断信息，不保存完整原始响应。"""

    choices = response_data.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(first_choice, dict):
        first_choice = {}

    message = first_choice.get("message")
    if not isinstance(message, dict):
        message = {}

    if content is None and "content" in message:
        content = message.get("content")

    api_usage = _extract_api_usage(response_data)
    completion_tokens = int(api_usage.get("completion_tokens") or 0)
    raw_content_length = len(content) if isinstance(content, str) else None
    stripped_content_length = len(content.strip()) if isinstance(content, str) else None

    return {
        "response_model_name": str(response_data.get("model")).strip() if response_data.get("model") else None,
        "finish_reason": str(first_choice.get("finish_reason")).strip()
        if first_choice.get("finish_reason")
        else None,
        "message_keys": sorted(str(key) for key in message.keys()),
        "raw_content_length": raw_content_length,
        "stripped_content_length": stripped_content_length,
        "content_preview": _preview_deepseek_content(content),
        "completion_tokens": completion_tokens,
        "max_tokens": max_tokens,
        "hit_max_tokens": max_tokens is not None and completion_tokens >= max_tokens,
    }


def _preview_deepseek_content(content: Any) -> str | None:
    """生成短预览用于排查，不记录完整模型输出。"""

    if not isinstance(content, str):
        return None
    preview = (
        content[:DEEPSEEK_CONTENT_PREVIEW_CHARS]
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    if len(content) > DEEPSEEK_CONTENT_PREVIEW_CHARS:
        return f"{preview}..."
    return preview


def _strip_json_code_fence(content: str) -> str:
    """移除完整包裹 JSON 的 Markdown 代码块标记。"""

    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return stripped
    if lines[0].strip().lower() not in {"```", "```json"}:
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _build_deepseek_messages(evidence: dict[str, Any], *, compact: bool = False) -> list[dict[str, str]]:
    """构造 DeepSeek 文本分析提示词。"""

    system_prompt = str(DEEPSEEK_PROMPT_POLICY["system"]).strip()
    if compact:
        system_prompt += "\n\n" + str(DEEPSEEK_PROMPT_POLICY["compact_retry_suffix"]).strip()
        prompt_evidence = _compact_deepseek_evidence(evidence)
    else:
        prompt_evidence = {
            "raw_text": evidence.get("raw_text"),
            "ocr_text": evidence.get("ocr_text"),
            "audio_transcript": evidence.get("audio_transcript"),
            "visual_description": evidence.get("visual_description"),
        }
    user_prompt = json.dumps(prompt_evidence, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请分析以下证据并输出 json：\n{user_prompt}"},
    ]


def _compact_deepseek_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """压缩证据，给 max_tokens 空响应后的重试使用。"""

    return {
        "raw_text": _compact_text(evidence.get("raw_text")),
        "ocr_text": _compact_text(evidence.get("ocr_text")),
        "audio_transcript": _compact_text(evidence.get("audio_transcript")),
        "visual_description": _compact_text(evidence.get("visual_description")),
    }


def _compact_text(value: Any) -> Any:
    """保留证据开头，避免短视频前几秒信息被截掉。"""

    if not isinstance(value, str) or len(value) <= DEEPSEEK_COMPACT_EVIDENCE_CHARS:
        return value
    return value[:DEEPSEEK_COMPACT_EVIDENCE_CHARS] + "……"


def _normalize_analysis_result(data: dict[str, Any]) -> dict[str, Any]:
    """把真实模型输出清洗成系统需要的固定结构。"""

    if not isinstance(data, dict):
        raise TypeError("模型内容必须是 JSON 对象。")

    required_fields = {"topic", "secondary_topics", "tags", "summary", "business_use"}
    missing_fields = sorted(field for field in required_fields if field not in data)
    if missing_fields:
        raise ValueError(f"缺少必要字段：{', '.join(missing_fields)}。")

    topic = str(data.get("topic") or "").strip()
    if topic not in TOPIC_VALUES:
        raise ValueError(f"topic 不属于既定分类：{topic or '空值'}。")

    if not isinstance(data.get("secondary_topics"), list):
        raise TypeError("secondary_topics 必须是数组。")
    if not isinstance(data.get("tags"), list):
        raise TypeError("tags 必须是数组。")

    summary = str(data.get("summary") or "").strip()
    business_use = str(data.get("business_use") or "").strip()
    if not summary:
        raise ValueError("summary 不能为空。")
    if not business_use:
        raise ValueError("business_use 不能为空。")

    secondary_topics = _normalize_topic_list(data.get("secondary_topics"), topic)
    tags = _normalize_text_list(data.get("tags"), limit=5)

    return {
        "topic": topic,
        "secondary_topics": secondary_topics,
        "tags": tags,
        "summary": summary[:300],
        "business_use": business_use,
    }


def _ground_business_use(business_use: str, evidence: dict[str, Any]) -> tuple[str, bool]:
    """在缺少商业证据时，拦截高风险商业用途并返回保守用途。"""

    contains_high_risk_use = any(term in business_use for term in HIGH_RISK_COMMERCIAL_USE_TERMS)
    if not contains_high_risk_use:
        return business_use, False

    evidence_text = " ".join(str(value) for value in evidence.values() if value)
    cleaned_evidence = re.sub(
        r"(?:没有|不含|并无|未提供)[^。；！？]*?(?=，?(?:但|不过|然而)|[。；！？]|$)",
        "",
        evidence_text,
    )
    for negated_term in NEGATED_COMMERCIAL_EVIDENCE_TERMS:
        cleaned_evidence = cleaned_evidence.replace(negated_term, "")

    has_positive_evidence = any(term in cleaned_evidence for term in POSITIVE_COMMERCIAL_EVIDENCE_TERMS)
    if has_positive_evidence:
        return business_use, False
    return CONSERVATIVE_BUSINESS_USE, True


def _normalize_topic_list(value: Any, topic: str) -> list[str]:
    """清洗副分类列表，确保最多两个且不重复主分类。"""

    topics = _normalize_text_list(value, limit=5)
    filtered = []
    for item in topics:
        if item in TOPIC_VALUES and item != topic and item not in filtered:
            filtered.append(item)
    return filtered[:2]


def _normalize_text_list(value: Any, *, limit: int) -> list[str]:
    """把模型返回的列表型字段清洗成字符串列表。"""

    if value is None:
        return []
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []

    result = []
    for item in candidates:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result[:limit]


def _extract_api_usage(response_data: dict[str, Any]) -> dict[str, int]:
    """从 API 响应中提取 token 用量。"""

    usage = response_data.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _build_mock_tags(text: str) -> list[str]:
    """根据证据文本生成最多五个模拟标签。"""

    candidates = [
        ("AI", "AI团队"),
        ("多模态", "多模态处理"),
        ("素材", "素材结构化"),
        ("模型", "模型调用"),
        ("成本", "成本核算"),
        ("延迟", "延迟统计"),
        ("供应商", "供应商对比"),
        ("JSONL", "JSONL输出"),
        ("预算", "预算控制"),
        ("检索", "内容检索"),
    ]
    tags = [tag for keyword, tag in candidates if keyword in text]
    return tags[:5] or ["内容分析"]


def _build_mock_summary(text: str) -> str:
    """根据证据文本生成克制的模拟摘要。"""

    cleaned = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        return "当前没有足够证据生成内容摘要。"
    preview = cleaned[:120]
    return f"基于现有文本证据，该内容主要涉及：{preview}"


def _build_mock_business_use(topic: str, tags: list[str]) -> str:
    """根据主分类和标签生成模拟业务用途说明。"""

    if topic == "technology":
        return "可用于技术素材归档、模型调用流程验证、内容检索和批次统计分析。"
    if "成本核算" in tags or "供应商对比" in tags:
        return "可用于模型成本核算、供应商对比和预算控制分析。"
    return "可用于内容归档、素材检索和结构化结果验证。"
