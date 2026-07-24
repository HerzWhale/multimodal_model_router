"""根据明细输出文件生成 batch_report.json。

报告生成器读取 results.jsonl、model_calls.jsonl 和 errors.jsonl，
生成文件、成本、延迟、错误和质量统计。
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def read_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件并返回记录列表，兼容单行和缩进两种写法。"""

    path = Path(file_path)
    content = path.read_text(encoding="utf-8").strip()
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

        record, index = decoder.raw_decode(content, index)
        if isinstance(record, list):
            records.extend(record)
        else:
            records.append(record)

    return records


def _rate(count: int, total: int) -> float:
    """计算比例；总数为 0 时返回 0。"""

    if total == 0:
        return 0.0
    return round(count / total, 6)


def _average(values: list[float]) -> float:
    """计算平均值；没有数据时返回 0。"""

    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _p95(values: list[float]) -> float:
    """计算 95 分位值；没有数据时返回 0。"""

    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = math.ceil(len(sorted_values) * 0.95) - 1
    return sorted_values[index]


def _sum_by(records: list[dict[str, Any]], group_key: str, value_key: str) -> dict[str, float]:
    """按指定字段分组求和。"""

    totals: dict[str, float] = defaultdict(float)
    for record in records:
        totals[str(record[group_key])] += float(record.get(value_key, 0))
    return {key: round(value, 6) for key, value in sorted(totals.items())}


def _latency_by(records: list[dict[str, Any]], group_key: str) -> dict[str, dict[str, float]]:
    """按指定字段分组统计平均延迟和 95 分位延迟。"""

    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record[group_key])].append(float(record.get("latency_ms", 0)))

    return {
        key: {
            "avg_latency_ms": _average(values),
            "p95_latency_ms": _p95(values),
        }
        for key, values in sorted(grouped.items())
    }


def _top_error_messages(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """统计高频错误信息。"""

    counter = Counter(str(error["error_message"]) for error in errors)
    return [
        {
            "message": message,
            "count": count,
        }
        for message, count in counter.most_common(5)
    ]


def _quality_flags_count(results: list[dict[str, Any]]) -> dict[str, int]:
    """统计质量风险标签出现次数。"""

    counter: Counter[str] = Counter()
    for result in results:
        counter.update(result.get("quality_flags", []))
    return dict(counter)


def _partial_success_reasons(results: list[dict[str, Any]]) -> dict[str, int]:
    """统计部分成功记录中的质量风险原因。"""

    counter: Counter[str] = Counter()
    for result in results:
        if result.get("processing_status") == "partial_success":
            counter.update(result.get("quality_flags", []))
    return dict(counter)


def generate_batch_report(
    *,
    batch_id: str,
    results: list[dict[str, Any]],
    model_calls: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    budget_limit_cny: float,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """根据文件结果、模型调用和错误索引生成批次报告。"""

    total_files = len(results)
    success_files = sum(1 for result in results if result.get("processing_status") == "success")
    partial_success_files = sum(1 for result in results if result.get("processing_status") == "partial_success")
    failed_files = sum(1 for result in results if result.get("processing_status") == "failed")
    skipped_files = sum(1 for result in results if result.get("processing_status") == "skipped")

    total_cost_cny = round(sum(float(call.get("cost_cny", 0)) for call in model_calls), 6)
    avg_cost_per_success_file_cny = round(total_cost_cny / success_files, 6) if success_files else 0.0
    model_latencies = [float(call.get("latency_ms", 0)) for call in model_calls]
    processing_times = [float(result.get("processing_time_ms", 0)) for result in results]
    slowest_file = max(results, key=lambda result: result.get("processing_time_ms", 0), default=None)

    return {
        "schema_version": "v1",
        "batch_id": batch_id,
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "file_stats": {
            "total_files": total_files,
            "success_files": success_files,
            "partial_success_files": partial_success_files,
            "failed_files": failed_files,
            "skipped_files": skipped_files,
            "success_rate": _rate(success_files, total_files),
            "failure_rate": _rate(failed_files, total_files),
            "partial_success_rate": _rate(partial_success_files, total_files),
        },
        "cost_stats": {
            "budget_limit_cny": budget_limit_cny,
            "total_cost_cny": total_cost_cny,
            "avg_cost_per_file_cny": _rate(total_cost_cny, total_files),
            "avg_cost_per_success_file_cny": avg_cost_per_success_file_cny,
            "cost_by_task_type": _sum_by(model_calls, "task_type", "cost_cny"),
            "cost_by_provider": _sum_by(model_calls, "provider", "cost_cny"),
            "budget_used_rate": round(total_cost_cny / budget_limit_cny, 6) if budget_limit_cny else 0.0,
        },
        "latency_stats": {
            "total_processing_time_ms": round(sum(processing_times), 6),
            "avg_processing_time_per_file_ms": _average(processing_times),
            "avg_model_latency_ms": _average(model_latencies),
            "p95_model_latency_ms": _p95(model_latencies),
            "latency_by_task_type": _latency_by(model_calls, "task_type"),
            "latency_by_provider": _latency_by(model_calls, "provider"),
            "slowest_file_id": slowest_file.get("file_id") if slowest_file else None,
        },
        "error_quality_stats": {
            "total_errors": len(errors),
            "errors_by_level": dict(Counter(str(error["error_level"]) for error in errors)),
            "errors_by_task_type": dict(Counter(str(error["task_type"]) for error in errors if error.get("task_type"))),
            "top_error_messages": _top_error_messages(errors),
            "quality_flags_count": _quality_flags_count(results),
            "partial_success_reasons": _partial_success_reasons(results),
        },
    }


def generate_batch_report_from_files(
    *,
    batch_dir: str | Path,
    batch_id: str,
    budget_limit_cny: float,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """从批次输出目录读取明细文件并生成批次报告。"""

    path = Path(batch_dir)
    return generate_batch_report(
        batch_id=batch_id,
        results=read_jsonl(path / "results.jsonl"),
        model_calls=read_jsonl(path / "model_calls.jsonl"),
        errors=read_jsonl(path / "errors.jsonl"),
        budget_limit_cny=budget_limit_cny,
        generated_at=generated_at,
    )
