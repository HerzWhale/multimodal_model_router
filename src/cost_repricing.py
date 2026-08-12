"""按当前价格目录重算历史批次成本。

本模块只读取已有 model_calls.jsonl 和本地价格目录，不重跑模型，不访问供应商 API。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from cost_latency_tracker import build_price_metadata, calculate_cost_cny, load_model_prices
from report_generator import read_jsonl


MISSING_VALUE_TEXT = "当前数据未提供"


def _as_float(value: Any, default: float = 0.0) -> float:
    """把值转换为浮点数；失败时返回默认值。"""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _delta_rate(delta_cny: float, recorded_cost_cny: float) -> float | None:
    """计算重算成本相对历史记录成本的变化比例。"""

    if recorded_cost_cny == 0:
        return None
    return round(delta_cny / recorded_cost_cny, 6)


def _call_status(delta_cny: float | None) -> str:
    """生成单条调用的重算状态。"""

    if delta_cny is None:
        return "not_repriced"
    if delta_cny == 0:
        return "unchanged"
    return "changed"


def reprice_model_calls(
    model_calls: list[dict[str, Any]],
    model_prices: dict[str, dict[str, Any]],
    *,
    batch_id: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """按当前价格目录重算模型调用成本。"""

    items = []
    for model_call in model_calls:
        model_name = str(model_call.get("model_name") or "")
        recorded_cost = round(_as_float(model_call.get("cost_cny")), 6)
        current_cost = None
        delta = None
        error_message = None
        price_metadata = {
            "cost_estimation_method": MISSING_VALUE_TEXT,
            "price_source": MISSING_VALUE_TEXT,
            "price_updated_at": None,
            "price_confidence": "missing_price",
        }

        if model_name not in model_prices:
            error_message = f"当前价格目录缺少模型：{model_name}"
        else:
            try:
                current_cost = calculate_cost_cny(
                    model_name,
                    model_call.get("input_units") or [],
                    model_prices,
                    model_call.get("output_units") or [],
                )
                delta = round(current_cost - recorded_cost, 6)
                price_metadata = build_price_metadata(model_name, model_prices)
            except (KeyError, ValueError, TypeError) as exc:
                error_message = str(exc)

        items.append(
            {
                "call_id": model_call.get("call_id"),
                "file_id": model_call.get("file_id"),
                "task_type": model_call.get("task_type"),
                "provider": model_call.get("provider"),
                "model_name": model_name,
                "response_model_name": model_call.get("response_model_name"),
                "recorded_cost_cny": recorded_cost,
                "current_estimated_cost_cny": current_cost,
                "cost_delta_cny": delta,
                "cost_delta_rate": _delta_rate(delta, recorded_cost) if delta is not None else None,
                "reprice_status": _call_status(delta),
                "error_message": error_message,
                **price_metadata,
            }
        )

    repriced_items = [item for item in items if item["current_estimated_cost_cny"] is not None]
    changed_items = [item for item in repriced_items if item["reprice_status"] == "changed"]
    total_recorded = round(sum(item["recorded_cost_cny"] for item in repriced_items), 6)
    total_current = round(sum(_as_float(item["current_estimated_cost_cny"]) for item in repriced_items), 6)
    total_delta = round(total_current - total_recorded, 6)

    return {
        "schema_version": "v1",
        "report_type": "cost_repricing",
        "batch_id": batch_id,
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": {
            "model_call_count": len(items),
            "repriced_call_count": len(repriced_items),
            "changed_call_count": len(changed_items),
            "not_repriced_call_count": len(items) - len(repriced_items),
            "total_recorded_cost_cny": total_recorded,
            "total_current_estimated_cost_cny": total_current,
            "total_cost_delta_cny": total_delta,
            "total_cost_delta_rate": _delta_rate(total_delta, total_recorded),
        },
        "reprice_items": items,
        "field_notes": {
            "recorded_cost_cny": "历史模型调用记录中原本保存的成本估算值。",
            "current_estimated_cost_cny": "按当前价格目录和历史调用用量重新计算出的成本估算值。",
            "cost_delta_cny": "当前重算成本减去历史记录成本后的金额差。",
            "cost_delta_rate": "成本变化比例，用于观察价格目录更新对历史批次成本的影响。",
            "reprice_status": "重算状态，用于区分成本未变化、已变化或无法重算。",
            "price_source": "当前重算使用的价格来源。",
            "price_updated_at": "当前重算使用的价格更新时间。",
            "price_confidence": "当前重算使用的价格可信度。",
        },
    }


def build_reprice_report_from_batch(
    *,
    batch_dir: str | Path,
    model_prices_path: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """从批次目录和价格目录生成重算报告。"""

    batch_path = Path(batch_dir)
    batch_id = None
    metadata_path = batch_path / "batch_metadata.json"
    if metadata_path.exists():
        batch_id = json.loads(metadata_path.read_text(encoding="utf-8")).get("batch_id")
    return reprice_model_calls(
        read_jsonl(batch_path / "model_calls.jsonl"),
        load_model_prices(model_prices_path),
        batch_id=batch_id,
        generated_at=generated_at,
    )


def render_reprice_markdown(report: dict[str, Any]) -> str:
    """把成本重算报告渲染为 Markdown。"""

    summary = report["summary"]
    lines = [
        "# 成本重算报告",
        "",
        f"批次编号：{report.get('batch_id') or MISSING_VALUE_TEXT}",
        "",
        "说明：本报告只读取历史模型调用记录和当前价格目录，不重跑模型，不调用供应商 API。",
        "",
        "## 1. 总览",
        "",
        "| 指标 | 数值 | 含义 |",
        "|---|---:|---|",
        f"| 模型调用数 | {summary['model_call_count']} | 历史批次中的模型调用数量 |",
        f"| 已重算调用数 | {summary['repriced_call_count']} | 当前价格目录可覆盖的调用数量 |",
        f"| 成本变化调用数 | {summary['changed_call_count']} | 当前重算成本与历史记录不同的调用数量 |",
        f"| 未重算调用数 | {summary['not_repriced_call_count']} | 缺少价格或用量导致无法重算的调用数量 |",
        f"| 历史记录成本 | {_format_cny(summary['total_recorded_cost_cny'])} | 历史 model_calls.jsonl 中保存的成本合计 |",
        f"| 当前重算成本 | {_format_cny(summary['total_current_estimated_cost_cny'])} | 按当前价格目录重新计算的成本合计 |",
        f"| 成本变化金额 | {_format_cny(summary['total_cost_delta_cny'])} | 当前重算成本减去历史记录成本 |",
        f"| 成本变化比例 | {_format_rate(summary['total_cost_delta_rate'])} | 成本变化金额相对历史记录成本的比例 |",
        "",
        "## 2. 调用明细",
        "",
        "| call_id | 模型 | 历史成本 | 当前重算成本 | 变化金额 | 变化比例 | 状态 | 价格来源 | 价格可信度 |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for item in report["reprice_items"]:
        lines.append(
            "| {call_id} | {model_name} | {recorded_cost} | {current_cost} | {delta} | {delta_rate} | {status} | {source} | {confidence} |".format(
                call_id=item.get("call_id") or MISSING_VALUE_TEXT,
                model_name=item.get("model_name") or MISSING_VALUE_TEXT,
                recorded_cost=_format_cny(item["recorded_cost_cny"]),
                current_cost=_format_optional_cny(item["current_estimated_cost_cny"]),
                delta=_format_optional_cny(item["cost_delta_cny"]),
                delta_rate=_format_rate(item["cost_delta_rate"]),
                status=item["reprice_status"],
                source=item["price_source"],
                confidence=item["price_confidence"],
            )
        )

    lines.extend(["", "## 字段说明", "", "| 字段 | 含义与作用 |", "|---|---|"])
    for field_name, note in report["field_notes"].items():
        lines.append(f"| `{field_name}` | {note} |")
    lines.append("")
    return "\n".join(lines)


def write_reprice_reports(
    *,
    report: dict[str, Any],
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, str]:
    """写入成本重算 JSON 和 Markdown 报告。"""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_output.write_text(render_reprice_markdown(report), encoding="utf-8")
    return {"json": str(json_output), "markdown": str(markdown_output)}


def _format_cny(value: float) -> str:
    """格式化人民币金额。"""

    return f"{value:.6f} 元"


def _format_optional_cny(value: Any) -> str:
    """格式化可空人民币金额。"""

    if value is None:
        return MISSING_VALUE_TEXT
    return _format_cny(_as_float(value))


def _format_rate(value: Any) -> str:
    """格式化比例。"""

    if value is None:
        return MISSING_VALUE_TEXT
    return f"{_as_float(value) * 100:.2f}%"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="按当前价格目录重算历史批次成本。")
    parser.add_argument("batch_dir", help="历史批次输出目录。")
    parser.add_argument("model_prices_path", help="当前模型价格目录。")
    parser.add_argument("output_json", help="输出 JSON 报告路径。")
    parser.add_argument("output_md", help="输出 Markdown 报告路径。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = _parse_args(argv)
    try:
        report = build_reprice_report_from_batch(
            batch_dir=args.batch_dir,
            model_prices_path=args.model_prices_path,
        )
        output_paths = write_reprice_reports(
            report=report,
            json_path=args.output_json,
            markdown_path=args.output_md,
        )
        print(json.dumps(output_paths, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"error_type": "cost_repricing_error", "error_message": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
