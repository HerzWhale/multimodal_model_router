"""批次成本与延迟瓶颈诊断。

只读取已有 batch_report.json 和 model_calls.jsonl，不调用任何模型 API。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


MISSING_VALUE_TEXT = "当前数据未提供"


def read_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON 文件。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_objects(path: str | Path) -> list[dict[str, Any]]:
    """读取项目中的缩进式 JSONL。"""

    content = Path(path).read_text(encoding="utf-8").strip()
    if not content:
        return []
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(content):
        while index < len(content) and content[index].isspace():
            index += 1
        if index >= len(content):
            break
        item, index = decoder.raw_decode(content, index)
        if isinstance(item, dict):
            records.append(item)
    return records


def generate_bottleneck_report(
    batch_report: dict[str, Any],
    model_calls: list[dict[str, Any]],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """生成批次瓶颈诊断报告。"""

    successful_calls = [call for call in model_calls if call.get("status") == "success"]
    cost_by_task = _sum_by(successful_calls, "task_type", "cost_cny")
    cost_by_provider = _sum_by(successful_calls, "provider", "cost_cny")
    latency_by_task = _avg_by(successful_calls, "task_type", "latency_ms")
    latency_by_provider = _avg_by(successful_calls, "provider", "latency_ms")
    slowest_call = max(successful_calls, key=lambda call: _as_float(call.get("latency_ms")), default={})
    costliest_call = max(successful_calls, key=lambda call: _as_float(call.get("cost_cny")), default={})
    slowest_file_id = _slowest_file_id(successful_calls)

    return {
        "schema_version": "v1",
        "report_type": "batch_bottleneck_report",
        "batch_id": batch_report.get("batch_id", MISSING_VALUE_TEXT),
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_files": {
            "batch_report": "batch_report.json",
            "model_calls": "model_calls.jsonl",
        },
        "batch_overview": {
            "total_files": batch_report.get("file_stats", {}).get("total_files", MISSING_VALUE_TEXT),
            "success_rate": batch_report.get("file_stats", {}).get("success_rate", MISSING_VALUE_TEXT),
            "total_cost_cny": batch_report.get("cost_stats", {}).get("total_cost_cny", MISSING_VALUE_TEXT),
            "cost_confidence": batch_report.get("cost_stats", {}).get("cost_confidence", MISSING_VALUE_TEXT),
            "avg_model_latency_ms": batch_report.get("latency_stats", {}).get(
                "avg_model_latency_ms",
                MISSING_VALUE_TEXT,
            ),
            "p95_model_latency_ms": batch_report.get("latency_stats", {}).get(
                "p95_model_latency_ms",
                MISSING_VALUE_TEXT,
            ),
            "model_call_count": len(model_calls),
        },
        "cost_bottleneck": {
            "top_task_type": _top_name(cost_by_task),
            "top_provider": _top_name(cost_by_provider),
            "cost_by_task_type": cost_by_task,
            "cost_by_provider": cost_by_provider,
            "costliest_call": _call_summary(costliest_call),
            "cost_note": "成本仍为本地价格表估算，未完成供应商账单对账。",
        },
        "latency_bottleneck": {
            "top_task_type": _top_name(latency_by_task),
            "top_provider": _top_name(latency_by_provider),
            "avg_latency_by_task_type_ms": latency_by_task,
            "avg_latency_by_provider_ms": latency_by_provider,
            "slowest_call": _call_summary(slowest_call),
            "slowest_file_id": slowest_file_id,
        },
        "runtime_mix": _runtime_mix(successful_calls),
        "recommendations": _recommendations(cost_by_task, latency_by_task, batch_report),
        "field_notes": _field_notes(),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """渲染 Markdown 报告。"""

    overview = report["batch_overview"]
    cost = report["cost_bottleneck"]
    latency = report["latency_bottleneck"]
    return "\n".join(
        [
            "# 批次成本与延迟瓶颈诊断报告",
            "",
            f"- 批次编号：{report['batch_id']}",
            f"- 文件数：{overview['total_files']}",
            f"- 成功率：{_percent(overview['success_rate'])}",
            f"- 总估算成本：{overview['total_cost_cny']} 元",
            f"- 成本可信度：{overview['cost_confidence']}",
            f"- 平均模型延迟：{overview['avg_model_latency_ms']} ms",
            f"- P95 模型延迟：{overview['p95_model_latency_ms']} ms",
            f"- 模型调用数：{overview['model_call_count']}",
            "",
            "## 核心结论",
            "",
            f"- 成本最高任务：{cost['top_task_type']}",
            f"- 成本最高供应商：{cost['top_provider']}",
            f"- 延迟最高任务：{latency['top_task_type']}",
            f"- 延迟最高供应商：{latency['top_provider']}",
            f"- 最慢文件：{latency['slowest_file_id']}",
            "",
            "## 成本分布",
            "",
            _dict_table("任务类型", "成本/元", cost["cost_by_task_type"]),
            "",
            _dict_table("供应商", "成本/元", cost["cost_by_provider"]),
            "",
            "## 延迟分布",
            "",
            _dict_table("任务类型", "平均延迟/ms", latency["avg_latency_by_task_type_ms"]),
            "",
            _dict_table("供应商", "平均延迟/ms", latency["avg_latency_by_provider_ms"]),
            "",
            "## 建议",
            "",
            *[f"- {item}" for item in report["recommendations"]],
            "",
            "## 字段说明",
            "",
            "| 字段 | 含义与作用 |",
            "|---|---|",
            *[f"| `{key}` | {value} |" for key, value in report["field_notes"].items()],
            "",
        ]
    )


def write_reports(batch_dir: str | Path) -> dict[str, str]:
    """从批次目录生成 JSON 和 Markdown 报告。"""

    directory = Path(batch_dir)
    report = generate_bottleneck_report(
        read_json(directory / "batch_report.json"),
        read_json_objects(directory / "model_calls.jsonl"),
    )
    json_path = directory / "bottleneck_report.json"
    markdown_path = directory / "bottleneck_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _sum_by(calls: list[dict[str, Any]], key: str, value_key: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for call in calls:
        totals[str(call.get(key) or "unknown")] += _as_float(call.get(value_key))
    return dict(sorted((name, round(value, 6)) for name, value in totals.items()))


def _avg_by(calls: list[dict[str, Any]], key: str, value_key: str) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    for call in calls:
        totals[str(call.get(key) or "unknown")].append(_as_float(call.get(value_key)))
    return dict(sorted((name, round(sum(values) / len(values), 6)) for name, values in totals.items() if values))


def _slowest_file_id(calls: list[dict[str, Any]]) -> str:
    totals: dict[str, float] = defaultdict(float)
    for call in calls:
        totals[str(call.get("file_id") or "unknown")] += _as_float(call.get("latency_ms"))
    return max(totals, key=totals.get, default=MISSING_VALUE_TEXT)


def _top_name(values: dict[str, float]) -> str:
    return max(values, key=values.get) if values else MISSING_VALUE_TEXT


def _call_summary(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "call_id": call.get("call_id", MISSING_VALUE_TEXT),
        "file_id": call.get("file_id", MISSING_VALUE_TEXT),
        "task_type": call.get("task_type", MISSING_VALUE_TEXT),
        "provider": call.get("provider", MISSING_VALUE_TEXT),
        "model_name": call.get("model_name", MISSING_VALUE_TEXT),
        "cost_cny": call.get("cost_cny", 0),
        "latency_ms": call.get("latency_ms", 0),
    }


def _runtime_mix(calls: list[dict[str, Any]]) -> dict[str, Any]:
    local_providers = {"paddlepaddle"}
    local_count = sum(1 for call in calls if str(call.get("provider")) in local_providers)
    return {
        "local_model_call_count": local_count,
        "live_api_call_count": len(calls) - local_count,
        "mock_call_count": sum(1 for call in calls if str(call.get("model_name") or "").startswith("mock")),
    }


def _recommendations(
    cost_by_task: dict[str, float],
    latency_by_task: dict[str, float],
    batch_report: dict[str, Any],
) -> list[str]:
    items = []
    top_cost_task = _top_name(cost_by_task)
    top_latency_task = _top_name(latency_by_task)
    if top_cost_task != MISSING_VALUE_TEXT:
        items.append(f"优先复核 {top_cost_task} 的价格口径，因为它贡献了最高成本。")
    if top_latency_task != MISSING_VALUE_TEXT:
        items.append(f"优先优化 {top_latency_task} 的耗时，因为它是平均延迟最高的任务。")
    if batch_report.get("cost_stats", {}).get("cost_confidence") != "reconciled":
        items.append("成本仍未完成供应商账单对账，不能把估算成本当作真实扣费。")
    return items or ["当前数据不足，暂不输出优化建议。"]


def _field_notes() -> dict[str, str]:
    return {
        "cost_cny": "单次模型调用成本，用于定位成本最高的任务和供应商。",
        "latency_ms": "单次模型调用耗时，用于定位最慢任务、供应商和文件。",
        "task_type": "模型调用任务类型，用于比较 OCR、视觉理解、语音识别和文本分析。",
        "provider": "模型供应商，用于比较本地模型和真实 API 的成本与延迟。",
        "cost_confidence": "成本可信度，用于说明当前成本是估算还是已完成账单对账。",
        "slowest_file_id": "累计模型调用耗时最高的文件，用于定位拖慢整批处理的输入。",
    }


def _as_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return MISSING_VALUE_TEXT


def _dict_table(name_header: str, value_header: str, values: dict[str, float]) -> str:
    lines = [f"| {name_header} | {value_header} |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(values.items(), key=lambda item: item[1], reverse=True))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("用法: python .\\src\\bottleneck_report.py output\\batch_xxx")
        return 2
    print(json.dumps(write_reports(args[0]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
