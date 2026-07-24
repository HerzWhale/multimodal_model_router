"""根据既有模型调用记录生成模型目录。

模型目录不是供应商宣传表，而是从本项目已经记录到的调用明细中抽取可比较信息。
当前没有真实观测的数据必须保留为“当前数据未提供”，避免把 mock 流程包装成真实模型能力。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


UNKNOWN_VALUE_TEXT = "当前数据未提供"


def is_mock_model(model_name: Any) -> bool:
    """判断模型名称是否属于 mock 占位模型。"""

    return str(model_name or "").lower().startswith("mock")


def as_float(value: Any, default: float = 0.0) -> float:
    """把输入值安全转换为浮点数。"""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile_95(values: list[float]) -> float | str:
    """计算 P95；没有有效数据时返回缺失说明。"""

    if not values:
        return UNKNOWN_VALUE_TEXT
    sorted_values = sorted(values)
    index = math.ceil(len(sorted_values) * 0.95) - 1
    return sorted_values[index]


def _collect_unit_types(calls: list[dict[str, Any]], unit_field: str) -> list[str] | str:
    """收集输入或输出用量单位。"""

    unit_types: set[str] = set()
    for call in calls:
        units = call.get(unit_field)
        if not isinstance(units, list):
            continue
        for unit in units:
            if isinstance(unit, dict) and unit.get("unit_type"):
                unit_types.add(str(unit["unit_type"]))

    return sorted(unit_types) if unit_types else UNKNOWN_VALUE_TEXT


def _quality_evidence_status(task_type: str, is_mock: bool) -> str:
    """描述某类模型调用当前能提供的质量证据。"""

    if is_mock:
        return "mock_only"
    if task_type == "text_analysis":
        return "real_text_analysis_observed"
    return "real_call_observed"


def _risk_notes(task_type: str, model_name: str, is_mock: bool, latency_values: list[float]) -> list[str]:
    """生成模型目录中的风险说明。"""

    notes: list[str] = []
    if is_mock:
        notes.append("该模型当前为 mock 或占位流程，只能证明链路跑通，不能证明真实供应商质量。")
        if all(value == 0 for value in latency_values):
            notes.append("latency_ms 为 0 是 mock 流程记录，不代表真实模型延迟。")
    else:
        notes.append("该模型已有真实调用记录，可用于观察本批次成本和延迟。")
        notes.append("当前数据未提供人工标注质量分数，不能推出 Accuracy、F1、PSNR、SSIM 或 VMAF 等质量结论。")

    if task_type in {"ocr", "speech_to_text", "visual_understanding"} and is_mock:
        notes.append("该上游证据提取任务是多模态质量瓶颈，后续应优先替换为真实模型做小样本验证。")

    if not model_name:
        notes.append("当前数据未提供 model_name，无法定位具体模型。")

    return notes


def build_model_catalog(model_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从 model_calls 生成按 provider、model_name、task_type 聚合的模型目录。

    返回列表中的每一项代表一个“模型-任务”组合，而不是单次调用。
    """

    grouped_calls: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for call in model_calls:
        provider = str(call.get("provider") or UNKNOWN_VALUE_TEXT)
        model_name = str(call.get("model_name") or UNKNOWN_VALUE_TEXT)
        task_type = str(call.get("task_type") or UNKNOWN_VALUE_TEXT)
        grouped_calls[(provider, model_name, task_type)].append(call)

    catalog: list[dict[str, Any]] = []
    for (provider, model_name, task_type), calls in sorted(grouped_calls.items()):
        mock_flag = is_mock_model(model_name)
        cost_values = [as_float(call.get("cost_cny")) for call in calls]
        latency_values = [as_float(call.get("latency_ms")) for call in calls if call.get("latency_ms") is not None]
        total_cost = round(sum(cost_values), 6)
        call_count = len(calls)

        catalog.append(
            {
                "model_id": f"{provider}/{model_name}/{task_type}",
                "provider": provider,
                "model_name": model_name,
                "task_type": task_type,
                "is_real": not mock_flag,
                "is_mock": mock_flag,
                "call_count": call_count,
                "total_cost_cny": total_cost,
                "avg_cost_per_call_cny": round(total_cost / call_count, 6) if call_count else UNKNOWN_VALUE_TEXT,
                "avg_latency_ms": round(sum(latency_values) / len(latency_values), 6) if latency_values else UNKNOWN_VALUE_TEXT,
                "p95_latency_ms": percentile_95(latency_values),
                "input_unit_types": _collect_unit_types(calls, "input_units"),
                "output_unit_types": _collect_unit_types(calls, "output_units"),
                "observed_statuses": sorted({str(call.get("status") or UNKNOWN_VALUE_TEXT) for call in calls}),
                "quality_evidence_status": _quality_evidence_status(task_type, mock_flag),
                "risk_notes": _risk_notes(task_type, model_name, mock_flag, latency_values),
                "data_source": "model_calls.jsonl",
            }
        )

    return catalog


def summarize_catalog(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总模型目录中的真实与 mock 覆盖情况。"""

    total_calls = sum(int(entry.get("call_count") or 0) for entry in catalog)
    real_calls = sum(int(entry.get("call_count") or 0) for entry in catalog if entry.get("is_real"))
    mock_calls = sum(int(entry.get("call_count") or 0) for entry in catalog if entry.get("is_mock"))

    return {
        "model_entry_count": len(catalog),
        "total_model_calls": total_calls,
        "real_model_calls": real_calls,
        "mock_model_calls": mock_calls,
        "real_coverage_rate": round(real_calls / total_calls, 6) if total_calls else 0.0,
        "mock_coverage_rate": round(mock_calls / total_calls, 6) if total_calls else 0.0,
        "real_task_types": sorted({entry["task_type"] for entry in catalog if entry.get("is_real")}),
        "mock_task_types": sorted({entry["task_type"] for entry in catalog if entry.get("is_mock")}),
    }
