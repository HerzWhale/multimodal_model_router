"""多供应商通用成本对账工具。

本模块只读取已有 model_calls.jsonl 和手工账单 CSV，不调用任何供应商 API。
它的目标是把系统估算成本与供应商后台账单拆开记录，避免把估算值误读成真实扣费。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from report_generator import read_jsonl


MISSING_VALUE_TEXT = "当前数据未提供"
TEMPLATE_COLUMNS = [
    "provider",
    "model_name",
    "response_model_name",
    "billing_start_at",
    "billing_end_at",
    "estimated_call_count",
    "estimated_cost_cny",
    "billed_cost_cny",
    "billing_granularity",
    "bill_source",
    "matching_method",
    "note",
]


class BillingValidationError(ValueError):
    """账单数据校验错误。"""


def _as_float(value: Any, default: float = 0.0) -> float:
    """把值安全转换为浮点数。"""

    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _blank_to_none(value: Any) -> str | None:
    """把空字符串转换为空值。"""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_time(value: Any) -> datetime | None:
    """解析 ISO 时间；无法解析时返回空值。"""

    text = _blank_to_none(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _parse_billed_cost(value: Any, record_index: int) -> float | None:
    """解析供应商实际扣费；空值表示未对账，非法值直接拒绝。"""

    text = _blank_to_none(value)
    if text is None:
        return None
    try:
        amount = float(text)
    except ValueError as exc:
        raise BillingValidationError(
            f"账单第 {record_index} 条的 billed_cost_cny 非法：必须是非负有限数字，当前值为 {text!r}。"
        ) from exc
    if not math.isfinite(amount) or amount < 0:
        raise BillingValidationError(
            f"账单第 {record_index} 条的 billed_cost_cny 非法：必须是非负有限数字，当前值为 {text!r}。"
        )
    return amount


def _runtime_type_for_call(model_call: dict[str, Any]) -> str:
    """判断单次调用是真实 API、本地模型、mock 还是未知类型。"""

    provider = str(model_call.get("provider") or "")
    model_name = str(model_call.get("model_name") or "")
    if model_name.startswith("mock-"):
        return "mock"
    if provider == "paddlepaddle":
        return "local_model"
    if provider in {"deepseek", "qwen", "doubao", "tongyi"}:
        return "live_api"
    return "unknown"


def _is_billable_runtime(runtime_type: str) -> bool:
    """判断该运行类型是否应该进入供应商账单对账。"""

    return runtime_type == "live_api"


def _call_in_billing_window(
    model_call: dict[str, Any],
    billing_start_at: str | None,
    billing_end_at: str | None,
) -> bool:
    """判断模型调用是否落入账单时间窗口。"""

    started_at = _parse_time(model_call.get("started_at"))
    start_time = _parse_time(billing_start_at)
    end_time = _parse_time(billing_end_at)
    if started_at is None:
        return True
    if start_time is not None and started_at < start_time:
        return False
    if end_time is not None and started_at > end_time:
        return False
    return True


def _cost_delta_rate(delta_cny: float, estimated_cost_cny: float) -> float | None:
    """计算成本偏差比例；估算值为 0 时不硬算。"""

    if estimated_cost_cny == 0:
        return None
    return round(delta_cny / estimated_cost_cny, 6)


def _cost_delta_reason(
    *,
    estimated_cost_cny: float,
    billed_cost_cny: float | None,
    note: str | None,
    bill_source: str | None,
) -> str:
    """给成本偏差生成机器可读原因，不把备注当成结构化结论。"""

    if billed_cost_cny is None:
        return "unverified"
    delta = round(billed_cost_cny - estimated_cost_cny, 6)
    if delta == 0:
        return "matched"

    reason_text = f"{note or ''} {bill_source or ''}".lower()
    adjustment_keywords = ["免费", "额度", "优惠", "抵扣", "coupon", "credit", "free_quota"]
    if any(keyword in reason_text for keyword in adjustment_keywords):
        return "billing_adjustment"
    if delta < 0:
        return "billed_lower_than_estimate"
    return "billed_higher_than_estimate"


def _confidence_from_billing(
    *,
    billed_cost_cny: float | None,
    billing_granularity: str | None,
    matched_call_count: int,
) -> str:
    """根据账单粒度生成成本可信度状态。"""

    if billed_cost_cny is None:
        return "unverified"
    granularity = (billing_granularity or "").strip().lower()
    if granularity in {"call", "call_level", "single_call"} and matched_call_count == 1:
        return "call_level_reconciled"
    return "period_level_reconciled"


def _group_key(record: dict[str, Any]) -> tuple[str, str, str | None]:
    """生成供应商、请求模型和响应模型维度的分组键。"""

    return (
        str(record.get("provider") or "unknown_provider"),
        str(record.get("model_name") or "unknown_model"),
        _blank_to_none(record.get("response_model_name")),
    )


def group_estimated_model_costs(model_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按供应商和模型汇总系统估算成本。"""

    groups: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for model_call in model_calls:
        provider, model_name, response_model_name = _group_key(model_call)
        key = (provider, model_name, response_model_name)
        runtime_type = _runtime_type_for_call(model_call)
        if key not in groups:
            groups[key] = {
                "provider": provider,
                "model_name": model_name,
                "response_model_name": response_model_name,
                "runtime_type": runtime_type,
                "estimated_call_count": 0,
                "estimated_cost_cny": 0.0,
                "call_ids": [],
                "started_at_values": [],
            }
        groups[key]["estimated_call_count"] += 1
        groups[key]["estimated_cost_cny"] += _as_float(model_call.get("cost_cny"))
        groups[key]["call_ids"].append(str(model_call.get("call_id") or "unknown_call"))
        if model_call.get("started_at"):
            groups[key]["started_at_values"].append(str(model_call["started_at"]))

    result = []
    for group in groups.values():
        started_at_values = sorted(group.pop("started_at_values"))
        group["estimated_cost_cny"] = round(group["estimated_cost_cny"], 6)
        group["first_started_at"] = started_at_values[0] if started_at_values else None
        group["last_started_at"] = started_at_values[-1] if started_at_values else None
        group["requires_bill_reconciliation"] = _is_billable_runtime(group["runtime_type"])
        result.append(group)

    return sorted(
        result,
        key=lambda item: (
            not item["requires_bill_reconciliation"],
            str(item["provider"]),
            str(item["model_name"]),
        ),
    )


def build_billing_template(model_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """根据真实 API 调用生成手工账单对账模板。"""

    template_rows = []
    for group in group_estimated_model_costs(model_calls):
        if not group["requires_bill_reconciliation"]:
            continue
        template_rows.append(
            {
                "provider": group["provider"],
                "model_name": group["model_name"],
                "response_model_name": group["response_model_name"] or "",
                "billing_start_at": group["first_started_at"] or "",
                "billing_end_at": group["last_started_at"] or "",
                "estimated_call_count": group["estimated_call_count"],
                "estimated_cost_cny": f"{group['estimated_cost_cny']:.6f}",
                "billed_cost_cny": "",
                "billing_granularity": "period",
                "bill_source": "manual_entry",
                "matching_method": "provider_model_time_window",
                "note": "请从供应商后台填入该模型在对应时间窗口内的实际扣费；没有账单前保持为空。",
            }
        )
    return template_rows


def write_billing_template(file_path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """写入 Excel 友好的 UTF-8 BOM CSV 模板。"""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TEMPLATE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in TEMPLATE_COLUMNS})
    return path


def read_billing_records(file_path: str | Path) -> list[dict[str, Any]]:
    """读取手工账单 CSV。"""

    path = Path(file_path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _billing_identity(record: dict[str, Any]) -> tuple[str, str, str | None]:
    """生成账单记录身份，用于判断是否重复。"""

    return _group_key(record)


def _billing_windows_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """判断两条账单记录的时间窗口是否重叠。"""

    left_start = _parse_time(left.get("billing_start_at"))
    left_end = _parse_time(left.get("billing_end_at"))
    right_start = _parse_time(right.get("billing_start_at"))
    right_end = _parse_time(right.get("billing_end_at"))
    if left_end is not None and right_start is not None and left_end < right_start:
        return False
    if right_end is not None and left_start is not None and right_end < left_start:
        return False
    return True


def _validate_billing_records(billing_records: list[dict[str, Any]]) -> list[float | None]:
    """校验手工账单记录，并返回已解析的真实扣费金额。"""

    parsed_costs = [
        _parse_billed_cost(record.get("billed_cost_cny"), index)
        for index, record in enumerate(billing_records, start=1)
    ]
    for left_index, left_record in enumerate(billing_records):
        if not _blank_to_none(left_record.get("provider")) or not _blank_to_none(left_record.get("model_name")):
            continue
        for right_index in range(left_index + 1, len(billing_records)):
            right_record = billing_records[right_index]
            if _billing_identity(left_record) != _billing_identity(right_record):
                continue
            if _billing_windows_overlap(left_record, right_record):
                provider, model_name, response_model_name = _billing_identity(left_record)
                response_name = response_model_name or "无"
                raise BillingValidationError(
                    "账单记录重复：第 {left} 条和第 {right} 条具有相同供应商、模型、响应模型且时间窗口重叠；"
                    "当前版本拒绝重复记录，请先合并或缩小时间窗口。供应商={provider}，模型={model}，响应模型={response}。".format(
                        left=left_index + 1,
                        right=right_index + 1,
                        provider=provider,
                        model=model_name,
                        response=response_name,
                    )
                )
    return parsed_costs


def _billing_record_matches_group(record: dict[str, Any], group: dict[str, Any]) -> bool:
    """判断账单记录是否匹配某个供应商模型组。"""

    if str(record.get("provider") or "") != str(group["provider"]):
        return False
    if str(record.get("model_name") or "") != str(group["model_name"]):
        return False
    response_model_name = _blank_to_none(record.get("response_model_name"))
    if response_model_name and response_model_name != _blank_to_none(group.get("response_model_name")):
        return False
    return True


def _matching_calls_for_bill(
    model_calls: list[dict[str, Any]],
    billing_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """按供应商、模型、响应模型和时间窗口匹配调用记录。"""

    provider = str(billing_record.get("provider") or "")
    model_name = str(billing_record.get("model_name") or "")
    response_model_name = _blank_to_none(billing_record.get("response_model_name"))
    matched = []
    for model_call in model_calls:
        if str(model_call.get("provider") or "") != provider:
            continue
        if str(model_call.get("model_name") or "") != model_name:
            continue
        if response_model_name and _blank_to_none(model_call.get("response_model_name")) != response_model_name:
            continue
        if not _call_in_billing_window(
            model_call,
            _blank_to_none(billing_record.get("billing_start_at")),
            _blank_to_none(billing_record.get("billing_end_at")),
        ):
            continue
        matched.append(model_call)
    return matched


def reconcile_costs(
    model_calls: list[dict[str, Any]],
    billing_records: list[dict[str, Any]],
    *,
    batch_id: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """对比系统估算成本和手工账单成本。"""

    parsed_billed_costs = _validate_billing_records(billing_records)
    billable_groups = [
        group
        for group in group_estimated_model_costs(model_calls)
        if group["requires_bill_reconciliation"]
    ]
    excluded_groups = [
        {
            **group,
            "exclude_reason": "该组不是真实 API 账单对象；mock 只用于流程占位，本地模型外部API成本为0但不含本机资源成本。",
        }
        for group in group_estimated_model_costs(model_calls)
        if not group["requires_bill_reconciliation"]
    ]
    reconciliation_items = []
    matched_billing_indexes: set[int] = set()

    for group in billable_groups:
        key = _group_key(group)
        billing_matches = [
            (index, record)
            for index, record in enumerate(billing_records)
            if _billing_record_matches_group(record, group)
        ]
        if len(billing_matches) > 1:
            raise BillingValidationError(
                "账单记录重复：供应商={provider}、模型={model}、响应模型={response} 匹配到多条账单记录；"
                "当前版本每个模型组只支持一条账单记录，请先合并账单后再对账。".format(
                    provider=group["provider"],
                    model=group["model_name"],
                    response=group["response_model_name"] or "无",
                )
            )
        billing_index = billing_matches[0][0] if billing_matches else None
        billing_record = billing_matches[0][1] if billing_matches else None
        billed_cost = parsed_billed_costs[billing_index] if billing_index is not None else None
        if billing_record:
            matched_billing_indexes.add(billing_index)
            matched_calls = _matching_calls_for_bill(model_calls, billing_record)
        else:
            matched_calls = [
                model_call
                for model_call in model_calls
                if _group_key(model_call) == key
            ]

        estimated_cost = round(sum(_as_float(call.get("cost_cny")) for call in matched_calls), 6)
        estimated_call_count = len(matched_calls)
        delta = round(billed_cost - estimated_cost, 6) if billed_cost is not None else None
        bill_source = billing_record.get("bill_source") if billing_record else None
        note = billing_record.get("note") if billing_record else "未提供供应商账单，本项仍为未验证估算。"
        reconciliation_items.append(
            {
                "provider": group["provider"],
                "model_name": group["model_name"],
                "response_model_name": group["response_model_name"],
                "runtime_type": group["runtime_type"],
                "estimated_call_count": estimated_call_count,
                "estimated_cost_cny": estimated_cost,
                "billed_cost_cny": billed_cost,
                "cost_delta_cny": delta,
                "cost_delta_rate": _cost_delta_rate(delta, estimated_cost) if delta is not None else None,
                "cost_delta_reason": _cost_delta_reason(
                    estimated_cost_cny=estimated_cost,
                    billed_cost_cny=billed_cost,
                    note=note,
                    bill_source=bill_source,
                ),
                "billing_granularity": billing_record.get("billing_granularity") if billing_record else None,
                "bill_source": bill_source,
                "matching_method": billing_record.get("matching_method") if billing_record else "provider_model_time_window",
                "bill_reconciled": billed_cost is not None,
                "cost_confidence": _confidence_from_billing(
                    billed_cost_cny=billed_cost,
                    billing_granularity=billing_record.get("billing_granularity") if billing_record else None,
                    matched_call_count=estimated_call_count,
                ),
                "matched_call_ids": [str(call.get("call_id") or "unknown_call") for call in matched_calls],
                "note": note,
            }
        )

    unmatched_billing_records = [
        billing_record
        for index, billing_record in enumerate(billing_records)
        if index not in matched_billing_indexes
        and _blank_to_none(billing_record.get("provider"))
        and _blank_to_none(billing_record.get("model_name"))
    ]
    total_estimated = round(sum(_as_float(item["estimated_cost_cny"]) for item in reconciliation_items), 6)
    total_billed = round(sum(_as_float(item["billed_cost_cny"]) for item in reconciliation_items if item["billed_cost_cny"] is not None), 6)
    reconciled_items = [item for item in reconciliation_items if item["bill_reconciled"]]
    total_delta = round(total_billed - sum(_as_float(item["estimated_cost_cny"]) for item in reconciled_items), 6)

    confidence_counts: dict[str, int] = {}
    delta_reason_counts: dict[str, int] = {}
    for item in reconciliation_items:
        confidence = str(item["cost_confidence"])
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        delta_reason = str(item["cost_delta_reason"])
        delta_reason_counts[delta_reason] = delta_reason_counts.get(delta_reason, 0) + 1

    return {
        "schema_version": "v1",
        "report_type": "cost_reconciliation",
        "batch_id": batch_id,
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_files": {
            "estimated_calls": "model_calls.jsonl",
            "billing_records": "manual_billing_csv",
        },
        "summary": {
            "billable_group_count": len(billable_groups),
            "reconciled_group_count": len(reconciled_items),
            "unverified_group_count": len(reconciliation_items) - len(reconciled_items),
            "total_estimated_cost_cny": total_estimated,
            "total_billed_cost_cny": total_billed if reconciled_items else None,
            "total_cost_delta_cny": total_delta if reconciled_items else None,
            "total_cost_delta_rate": _cost_delta_rate(total_delta, sum(_as_float(item["estimated_cost_cny"]) for item in reconciled_items)) if reconciled_items else None,
            "confidence_counts": confidence_counts,
            "cost_delta_reason_counts": delta_reason_counts,
            "bill_reconciled": bool(reconciled_items) and len(reconciled_items) == len(reconciliation_items),
            "estimation_error_status": "known_for_reconciled_items" if reconciled_items else "unknown_until_bill_reconciliation",
        },
        "reconciliation_items": reconciliation_items,
        "excluded_groups": excluded_groups,
        "unmatched_billing_records": unmatched_billing_records,
        "field_notes": {
            "provider": "供应商名称，用于把系统调用记录和供应商账单聚合到同一来源。",
            "model_name": "请求模型名称，用于按模型维度匹配账单。",
            "response_model_name": "服务端响应模型名称，用于在供应商返回模型别名时辅助核对。",
            "estimated_cost_cny": "系统根据用量和本地价格表计算出的估算成本，单位人民币。",
            "billed_cost_cny": "供应商后台显示或账单导出的实际扣费，单位人民币。",
            "cost_delta_cny": "实际扣费减去系统估算值后的差额，单位人民币。",
            "cost_delta_rate": "差额相对估算值的比例；估算值为0时不硬算。",
            "cost_delta_reason": "成本偏差原因，用于区分未验证、已匹配、账单优惠抵扣、实际扣费低于估算和实际扣费高于估算。",
            "billing_granularity": "账单粒度，例如单次调用、小时级、日级或模型级。",
            "bill_source": "真实扣费来源，例如供应商控制台人工查看或供应商账单导出文件。",
            "matching_method": "系统调用记录与账单记录的匹配方式，例如按供应商、模型和时间窗口匹配。",
            "bill_reconciled": "是否已经填入供应商账单金额并完成对账。",
            "cost_confidence": "成本可信度状态，用于区分未验证、单次调用级对账和时间段级对账。",
            "matched_call_ids": "实际参与本条账单核对的模型调用编号列表，用于从对账结果反查调用明细。",
            "unmatched_billing_records": "没有匹配到本批次模型调用的账单记录，用于提示账单时间窗口或模型名称可能填错。",
        },
    }


def render_reconciliation_markdown(report: dict[str, Any]) -> str:
    """把成本对账报告渲染成 Markdown。"""

    summary = report["summary"]
    lines = [
        "# 成本对账报告",
        "",
        f"批次编号：{report.get('batch_id') or MISSING_VALUE_TEXT}",
        "",
        "说明：本报告只比较已有 `model_calls.jsonl` 和手工账单 CSV，不调用任何供应商 API。",
        "",
        "## 1. 总览",
        "",
        "| 指标 | 数值 | 含义 |",
        "|---|---:|---|",
        f"| 需对账模型组数 | {summary['billable_group_count']} | 真实 API 供应商/模型组合数量 |",
        f"| 已对账组数 | {summary['reconciled_group_count']} | 已填入供应商实际扣费的组合数量 |",
        f"| 未验证组数 | {summary['unverified_group_count']} | 仍只有系统估算的组合数量 |",
        f"| 估算成本合计 | {_format_cny(summary['total_estimated_cost_cny'])} | 需对账真实 API 的系统估算成本 |",
        f"| 实际扣费合计 | {_format_optional_cny(summary['total_billed_cost_cny'])} | 已对账项目的供应商账单成本 |",
        f"| 总偏差 | {_format_optional_cny(summary['total_cost_delta_cny'])} | 实际扣费减估算成本 |",
        f"| 总偏差率 | {_format_optional_rate(summary['total_cost_delta_rate'])} | 总偏差相对估算成本的比例 |",
        f"| 误差状态 | {summary['estimation_error_status']} | 未对账前不得宣称误差小 |",
        "",
        "## 2. 对账明细",
        "",
        "| 供应商 | 模型 | 响应模型 | 估算调用数 | 估算成本 | 实际扣费 | 偏差 | 偏差率 | 差异原因 | 账单粒度 | 账单来源 | 可信度 | 备注 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for item in report["reconciliation_items"]:
        lines.append(
            "| {provider} | {model_name} | {response_model_name} | {estimated_call_count} | {estimated_cost} | {billed_cost} | {delta} | {delta_rate} | {delta_reason} | {granularity} | {bill_source} | {confidence} | {note} |".format(
                provider=item["provider"],
                model_name=item["model_name"],
                response_model_name=item.get("response_model_name") or "无",
                estimated_call_count=item["estimated_call_count"],
                estimated_cost=_format_cny(item["estimated_cost_cny"]),
                billed_cost=_format_optional_cny(item["billed_cost_cny"]),
                delta=_format_optional_cny(item["cost_delta_cny"]),
                delta_rate=_format_optional_rate(item["cost_delta_rate"]),
                delta_reason=item["cost_delta_reason"],
                granularity=item.get("billing_granularity") or MISSING_VALUE_TEXT,
                bill_source=item.get("bill_source") or MISSING_VALUE_TEXT,
                confidence=item["cost_confidence"],
                note=item.get("note") or "",
            )
        )

    if report["excluded_groups"]:
        lines.extend(
            [
                "",
                "## 3. 排除对账项",
                "",
                "| 供应商 | 模型 | 类型 | 估算成本 | 排除原因 |",
                "|---|---|---|---:|---|",
            ]
        )
        for group in report["excluded_groups"]:
            lines.append(
                f"| {group['provider']} | {group['model_name']} | {group['runtime_type']} | {_format_cny(group['estimated_cost_cny'])} | {group['exclude_reason']} |"
            )

    if report["unmatched_billing_records"]:
        lines.extend(["", "## 4. 未匹配账单记录", ""])
        for billing_record in report["unmatched_billing_records"]:
            lines.append(
                f"- {billing_record.get('provider')}/{billing_record.get('model_name')}：账单记录没有匹配到本批次模型调用。"
            )

    lines.extend(
        [
            "",
            "## 字段说明",
            "",
            "| 字段 | 含义与作用 |",
            "|---|---|",
        ]
    )
    for field_name, note in report["field_notes"].items():
        lines.append(f"| `{field_name}` | {note} |")
    lines.append("")
    return "\n".join(lines)


def write_reconciliation_reports(
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    report: dict[str, Any],
) -> dict[str, str]:
    """写入成本对账 JSON 和 Markdown 报告。"""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_output.write_text(render_reconciliation_markdown(report), encoding="utf-8")
    return {"json": str(json_output), "markdown": str(markdown_output)}


def _format_cny(value: float) -> str:
    """格式化人民币金额。"""

    return f"{value:.6f} 元"


def _format_optional_cny(value: Any) -> str:
    """格式化可空人民币金额。"""

    if value is None:
        return MISSING_VALUE_TEXT
    return _format_cny(_as_float(value))


def _format_optional_rate(value: Any) -> str:
    """格式化可空比例。"""

    if value is None:
        return MISSING_VALUE_TEXT
    return f"{_as_float(value) * 100:.2f}%"


def build_template_from_batch(batch_dir: str | Path, output_csv: str | Path) -> Path:
    """从批次目录生成手工账单 CSV 模板。"""

    path = Path(batch_dir)
    rows = build_billing_template(read_jsonl(path / "model_calls.jsonl"))
    return write_billing_template(output_csv, rows)


def build_reconciliation_from_files(
    *,
    batch_dir: str | Path,
    billing_csv: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """从批次目录和手工账单 CSV 生成成本对账报告。"""

    path = Path(batch_dir)
    batch_metadata_path = path / "batch_metadata.json"
    batch_id = None
    if batch_metadata_path.exists():
        batch_id = json.loads(batch_metadata_path.read_text(encoding="utf-8")).get("batch_id")
    return reconcile_costs(
        read_jsonl(path / "model_calls.jsonl"),
        read_billing_records(billing_csv),
        batch_id=batch_id,
        generated_at=generated_at,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="生成或执行多供应商成本对账。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser("template", help="根据批次模型调用生成手工账单模板。")
    template_parser.add_argument("batch_dir", help="批次输出目录。")
    template_parser.add_argument("output_csv", help="输出的手工账单模板 CSV。")

    reconcile_parser = subparsers.add_parser("reconcile", help="根据手工账单 CSV 生成对账报告。")
    reconcile_parser.add_argument("batch_dir", help="批次输出目录。")
    reconcile_parser.add_argument("billing_csv", help="手工账单 CSV。")
    reconcile_parser.add_argument("output_json", help="输出 JSON 报告路径。")
    reconcile_parser.add_argument("output_md", help="输出 Markdown 报告路径。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = _parse_args(argv)
    try:
        if args.command == "template":
            path = build_template_from_batch(args.batch_dir, args.output_csv)
            print(json.dumps({"template_csv": str(path)}, ensure_ascii=False, indent=2))
            return 0

        report = build_reconciliation_from_files(
            batch_dir=args.batch_dir,
            billing_csv=args.billing_csv,
        )
        output_paths = write_reconciliation_reports(
            json_path=args.output_json,
            markdown_path=args.output_md,
            report=report,
        )
        print(json.dumps(output_paths, ensure_ascii=False, indent=2))
        return 0
    except BillingValidationError as exc:
        print(
            json.dumps(
                {
                    "error_type": "billing_validation_error",
                    "error_message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
