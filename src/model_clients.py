"""封装 OCR、语音识别、视觉理解和文本分析的真实或模拟模型调用。

客户端只返回模型输出，不直接写入结果、模型调用日志或错误记录。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib import error, request


TOPIC_VALUES = {
    "news",
    "entertainment",
    "knowledge",
    "lifestyle",
    "technology",
    "sports_health",
    "finance_business",
    "ads_marketing",
    "other",
}

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_PADDLEOCR_MODEL_NAME = "PP-OCRv5_mobile"

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
    ) -> None:
        super().__init__(f"[{error_code}] {message}")
        self.error_code = error_code
        self.retryable = retryable
        self.api_usage = api_usage or {}


class DeepSeekAttemptsExhausted(RuntimeError):
    """表示 DeepSeek 调用未成功，并保留每次尝试的计量信息。"""

    def __init__(self, last_error: DeepSeekResponseError, attempts: list[dict[str, Any]]) -> None:
        super().__init__(str(last_error))
        self.attempts = attempts


class PaddleOCRResponseError(RuntimeError):
    """表示 PaddleOCR 推理失败或返回结构不符合预期。"""


TOPIC_KEYWORDS = {
    "ads_marketing": ["广告", "带货", "推广", "品牌宣传", "种草"],
    "news": ["新闻", "报道", "突发", "发布会", "事件"],
    "finance_business": ["财经", "商业", "成本", "预算", "供应商", "投资", "财报"],
    "technology": ["AI", "模型", "系统", "数据", "技术", "软件", "手机", "多模态"],
    "sports_health": ["体育", "健身", "运动", "健康"],
    "entertainment": ["娱乐", "明星", "综艺", "影视", "游戏", "搞笑"],
    "lifestyle": ["日常", "vlog", "美食", "旅行", "穿搭", "自拍"],
    "knowledge": ["科普", "教程", "知识", "学习", "教育"],
}


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


def mock_vision_client(image_path: str | Path) -> dict[str, str]:
    """模拟视觉理解调用，返回画面描述证据。"""

    file_name = Path(image_path).name
    return {"visual_description": f"模拟视觉描述：{file_name} 展示了一段待分析内容。"}


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
) -> dict[str, Any]:
    """调用 DeepSeek API，返回标准化后的内容分析结果。"""

    if not api_key:
        raise ValueError("缺少 DEEPSEEK_API_KEY，无法调用 DeepSeek API。")
    if max_retries not in {0, 1}:
        raise ValueError("DeepSeek 最大重试次数只能是 0 或 1。")

    payload = {
        "model": model_name,
        "messages": _build_deepseek_messages(evidence),
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 800,
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
            result, api_usage = _perform_deepseek_request(api_request, timeout_seconds)
        except DeepSeekResponseError as exc:
            attempts.append(
                {
                    "status": "failed",
                    "latency_ms": int(round((perf_counter() - attempt_started_at) * 1000)),
                    "api_usage": exc.api_usage,
                    "error_message": str(exc),
                }
            )
            if exc.retryable and attempt_index < max_retries:
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
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DeepSeekResponseError(
            "deepseek_response_missing_content",
            "DeepSeek API 响应缺少 choices[0].message.content。",
            retryable=True,
            api_usage=api_usage,
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise DeepSeekResponseError(
            "deepseek_content_empty",
            "DeepSeek 模型内容为空。",
            retryable=True,
            api_usage=api_usage,
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
        ) from exc

    try:
        result = _normalize_analysis_result(analysis_data)
    except (TypeError, ValueError) as exc:
        raise DeepSeekResponseError(
            "deepseek_content_invalid_schema",
            f"DeepSeek 模型内容不符合结果结构：{exc}",
            retryable=True,
            api_usage=api_usage,
        ) from exc
    return result, api_usage


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


def _build_deepseek_messages(evidence: dict[str, Any]) -> list[dict[str, str]]:
    """构造 DeepSeek 文本分析提示词。"""

    system_prompt = """
你是内容平台 AI 团队的内容结构化助手。请基于用户提供的证据生成严格 json。
不要编造证据中没有的信息；如果证据不足，可以保守分类。

可选 topic 只能从以下 9 类中选择一个：
news, entertainment, knowledge, lifestyle, technology, sports_health, finance_business, ads_marketing, other。

主分类判断顺序如下。必须从上到下判断；命中更高优先级规则后，将其他交叉领域放入 secondary_topics，不要同时输出多个主分类：
1. ads_marketing：明确品牌广告、商业合作、带货、官方推广、购买引导或营销转化。
2. news：以报道具体事件为核心，包含事件进展、时间线、当事方或官方回应、公共变化等新闻要素；即使事件发生在娱乐、科技等领域，也优先选择 news，并把相关领域放入 secondary_topics。
3. finance_business：财经分析、公司经营、商业思维、投资理财、财报、营收、股价、市场规模或商业风险；即使提到 AI、芯片、软件，也优先选择 finance_business。
4. technology：以 AI、数码、软件、硬件、汽车科技的功能发布、技术教程、产品能力或工程实现为核心。
5. sports_health：以体育赛事、运动、健身、身体健康或健康方法为核心。
6. entertainment：以影视、综艺、明星、游戏、表演或搞笑内容为核心，且不是在报道一项具体新闻事件。
7. lifestyle：以个人经历、vlog、自拍、美食、旅行、穿搭、消费体验或日常分享为核心。
8. knowledge：面向一般受众解释可迁移的概念、原理、历史、因果关系或方法，且没有更强的财经、科技、体育健康等领域归属。
9. other：以上八类都不符合时使用，包括没有明确主题领域的组织通知、办事流程或局部协作信息；不要为了避免 other 而强行选择相邻类别。

补充边界规则：
- 先判断内容主要业务场景，再判断内容对象词。
- 领域优先于讲解形式：财经讲解选择 finance_business，科技教程选择 technology，健身教学选择 sports_health；不能因为采用讲解、教程或说明形式就直接选择 knowledge。
- lifestyle 必须以个人生活体验或日常分享为核心；不能因为内容涉及生活用品或日常场所，就把组织通知、办事流程或协作规则归为 lifestyle。
- knowledge 必须包含面向一般受众的知识解释；单纯告知时间、地点、登记步骤或操作规定不等于知识科普。
- other 是有效主分类。证据确实不符合前八类时应选择 other，不得用 secondary_topics 代替主分类判断。

business_use 证据规则：
- 只能说明基于现有证据可以直接支持的业务动作，不得编造用户增长、收入提升、转化效果或算法效果。
- 只有证据明确出现品牌合作、广告合作、购买入口、下单、促销或带货时，才能建议品牌推广、广告投放、购买转化或带货用途。
- 仅仅出现商品、食品、设备或服务名称，不足以证明内容适合商业推广。
- 如果证据没有明确业务背景，使用“可用于内容归档、检索和人工复核”这类保守用途。

请输出以下 json 字段：
{
  "topic": "主分类，只能一个",
  "secondary_topics": ["副分类，最多两个，不能和 topic 重复"],
  "tags": ["关键词，最多五个，尽量使用名词或短语"],
  "summary": "300 字以内摘要",
  "business_use": "这份结构化结果可以支持的业务用途"
}
""".strip()
    user_prompt = json.dumps(
        {
            "raw_text": evidence.get("raw_text"),
            "ocr_text": evidence.get("ocr_text"),
            "audio_transcript": evidence.get("audio_transcript"),
            "visual_description": evidence.get("visual_description"),
        },
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请分析以下证据并输出 json：\n{user_prompt}"},
    ]


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
        raise ValueError(f"topic 不属于既定九类：{topic or '空值'}。")

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
