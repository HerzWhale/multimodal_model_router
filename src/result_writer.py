"""把批次输出写入 output/{batch_id}/。

本模块写入 batch_metadata.json、results.jsonl、model_calls.jsonl、errors.jsonl
和 batch_report.json，但不分析内容，也不计算统计指标。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_batch_output_dir(output_dir: str | Path, batch_id: str) -> Path:
    """确保批次输出目录存在，并返回该目录路径。"""

    batch_dir = Path(output_dir) / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def write_json(file_path: str | Path, data: dict[str, Any]) -> Path:
    """把一个字典写入 JSON 文件。"""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_jsonl(file_path: str | Path, records: list[dict[str, Any]]) -> Path:
    """把多条字典记录写入标准 JSONL 文件，每条记录占一行。"""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")
    return path


def write_batch_metadata(output_dir: str | Path, batch_id: str, metadata: dict[str, Any]) -> Path:
    """写入批次元数据文件。"""

    batch_dir = ensure_batch_output_dir(output_dir, batch_id)
    return write_json(batch_dir / "batch_metadata.json", metadata)


def write_results(output_dir: str | Path, batch_id: str, records: list[dict[str, Any]]) -> Path:
    """写入文件级最终结果。"""

    batch_dir = ensure_batch_output_dir(output_dir, batch_id)
    return write_jsonl(batch_dir / "results.jsonl", records)


def write_results_readable(output_dir: str | Path, batch_id: str, records: list[dict[str, Any]]) -> Path:
    """写入方便人工阅读的文件级结果说明。"""

    batch_dir = ensure_batch_output_dir(output_dir, batch_id)
    path = batch_dir / "results_readable.md"
    sections = [
        "# 文件级分析结果解读",
        "",
        f"批次编号：{batch_id}",
        "",
        "说明：本文件用于人工阅读；机器读取请使用 `results.jsonl`。",
        "",
    ]

    for index, record in enumerate(records, start=1):
        sections.extend(_format_readable_result(index, record))

    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def _format_readable_result(index: int, record: dict[str, Any]) -> list[str]:
    """把一条文件级结果转换成 Markdown 段落。"""

    return [
        "---",
        "",
        f"## 结果 {index}：{record.get('file_id')} | {record.get('file_name')} | {record.get('media_type')}",
        "",
        f"- 处理状态：{record.get('processing_status')}",
        f"- 主分类：{record.get('topic')}",
        f"- 副分类：{_format_list(record.get('secondary_topics'))}",
        f"- 关键词：{_format_list(record.get('tags'))}",
        f"- 摘要：{record.get('summary')}",
        f"- 业务用途：{record.get('business_use')}",
        f"- 使用证据：{_format_list(record.get('evidence_used'))}",
        f"- 缺失证据：{_format_list(record.get('missing_evidence'))}",
        f"- 使用模型：{_format_models_used(record.get('models_used'))}",
        f"- 关联模型调用：{_format_list(record.get('call_ids'))}",
        f"- 文件处理成本：{record.get('processing_cost_cny')} 元",
        f"- 文件处理耗时：{record.get('processing_time_ms')} ms",
        f"- 错误信息：{record.get('error_message')}",
        f"- 风险提示：{_format_list(record.get('warning_messages'))}",
        "",
    ]


def _format_list(value: Any) -> str:
    """把列表值转换成适合阅读的短文本。"""

    if value is None:
        return "无"
    if isinstance(value, list):
        return "、".join(str(item) for item in value) if value else "无"
    return str(value)


def _format_models_used(value: Any) -> str:
    """把模型使用摘要转换成适合阅读的短文本。"""

    if not value:
        return "无"
    if not isinstance(value, list):
        return str(value)

    model_items = []
    for item in value:
        if not isinstance(item, dict):
            model_items.append(str(item))
            continue
        task_type = item.get("task_type", "unknown_task")
        provider = item.get("provider", "unknown_provider")
        model_name = item.get("model_name", "unknown_model")
        status = item.get("status", "unknown_status")
        call_id = item.get("call_id", "unknown_call")
        model_items.append(f"{task_type}: {provider}/{model_name}（{status}, {call_id}）")
    return "；".join(model_items)


def write_model_calls(output_dir: str | Path, batch_id: str, records: list[dict[str, Any]]) -> Path:
    """写入模型调用明细。"""

    batch_dir = ensure_batch_output_dir(output_dir, batch_id)
    return write_jsonl(batch_dir / "model_calls.jsonl", records)


def write_errors(output_dir: str | Path, batch_id: str, records: list[dict[str, Any]]) -> Path:
    """写入错误索引。"""

    batch_dir = ensure_batch_output_dir(output_dir, batch_id)
    return write_jsonl(batch_dir / "errors.jsonl", records)
