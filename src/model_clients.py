"""封装 OCR、语音识别、视觉理解和文本分析的真实或模拟模型调用。

客户端只返回模型输出，不直接写入结果、模型调用日志或错误记录。
"""

from __future__ import annotations

import json
from pathlib import Path
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
) -> dict[str, Any]:
    """调用 DeepSeek API，返回标准化后的内容分析结果。"""

    if not api_key:
        raise ValueError("缺少 DEEPSEEK_API_KEY，无法调用 DeepSeek API。")

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

    try:
        with request.urlopen(api_request, timeout=timeout_seconds) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API 请求失败：HTTP {exc.code}，{detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"DeepSeek API 网络连接失败：{exc.reason}") from exc

    content = response_data["choices"][0]["message"]["content"]
    analysis_data = json.loads(content)
    result = _normalize_analysis_result(analysis_data)
    result["_api_usage"] = _extract_api_usage(response_data)
    return result


def _build_deepseek_messages(evidence: dict[str, Any]) -> list[dict[str, str]]:
    """构造 DeepSeek 文本分析提示词。"""

    system_prompt = """
你是内容平台 AI 团队的内容结构化助手。请基于用户提供的证据生成严格 json。
不要编造证据中没有的信息；如果证据不足，可以保守分类。

可选 topic 只能从以下 9 类中选择一个：
news, entertainment, knowledge, lifestyle, technology, sports_health, finance_business, ads_marketing, other。

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

    topic = str(data.get("topic") or "other")
    if topic not in TOPIC_VALUES:
        topic = "other"

    secondary_topics = _normalize_topic_list(data.get("secondary_topics"), topic)
    tags = _normalize_text_list(data.get("tags"), limit=5)
    summary = str(data.get("summary") or "").strip()[:300]
    business_use = str(data.get("business_use") or "").strip()

    return {
        "topic": topic,
        "secondary_topics": secondary_topics,
        "tags": tags,
        "summary": summary or "真实模型未能生成有效摘要。",
        "business_use": business_use or "可用于内容归档、素材检索和结构化结果验证。",
    }


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
