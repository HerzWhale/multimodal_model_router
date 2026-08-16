"""把已有批次证据合成为技术负责人可读的决策摘要。

只读取已经生成的 JSON 报告，不调用任何模型 API。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


MISSING = "当前数据未提供"


def read_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON 文件。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_decision_summary(
    batch_report: dict[str, Any],
    bottleneck_report: dict[str, Any],
    cost_report: dict[str, Any],
    eval_report: dict[str, Any],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """生成批次级决策摘要。"""

    file_stats = batch_report.get("file_stats", {})
    latency_stats = batch_report.get("latency_stats", {})
    cost_summary = cost_report.get("summary", {})
    cost_items = cost_report.get("reconciliation_items", [])
    zero_billed_items = [
        item
        for item in cost_items
        if _float(item.get("estimated_cost_cny")) > 0 and _float(item.get("billed_cost_cny")) == 0
    ]

    success_rate = _float(file_stats.get("success_rate"))
    accuracy = _float(eval_report.get("accuracy"))
    macro_f1 = _float(eval_report.get("macro_f1"))
    bill_reconciled = bool(cost_summary.get("bill_reconciled"))

    readiness = "controlled_batch_ready" if success_rate == 1 and accuracy == 1 and bill_reconciled else "needs_review"

    return {
        "schema_version": "v1",
        "report_type": "decision_summary",
        "batch_id": batch_report.get("batch_id", MISSING),
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_files": {
            "batch_report": "batch_report.json",
            "bottleneck_report": "bottleneck_report.json",
            "cost_reconciliation_report": "cost_reconciliation_report_hour.json",
            "video_topic_eval_report": "video_topic_eval_report.json",
        },
        "readiness": {
            "status": readiness,
            "meaning": "当前批次可作为受控小批量试跑证据" if readiness == "controlled_batch_ready" else "当前批次仍需复核",
        },
        "quality_summary": {
            "total_files": file_stats.get("total_files", MISSING),
            "success_rate": success_rate,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "prediction_coverage": _float(eval_report.get("prediction_coverage")),
            "quality_note": "样本量较小，只能证明本批次回归表现，不能证明泛化能力。",
        },
        "cost_summary": {
            "total_estimated_cost_cny": cost_summary.get("total_estimated_cost_cny", MISSING),
            "total_billed_cost_cny": cost_summary.get("total_billed_cost_cny", MISSING),
            "total_cost_delta_cny": cost_summary.get("total_cost_delta_cny", MISSING),
            "total_cost_delta_rate": cost_summary.get("total_cost_delta_rate", MISSING),
            "cost_confidence": cost_summary.get("confidence_counts", {}),
            "billing_granularity": sorted({str(item.get("billing_granularity") or MISSING) for item in cost_items}),
            "zero_billed_live_api_models": [
                _model_key(item) for item in zero_billed_items
            ],
            "cost_note": "当前是小时级/周期级对账，不是单次调用级精确对账；后台扣费为 0 的模型需按免费额度或抵扣口径解释。",
        },
        "latency_summary": {
            "avg_model_latency_ms": latency_stats.get("avg_model_latency_ms", MISSING),
            "p95_model_latency_ms": latency_stats.get("p95_model_latency_ms", MISSING),
            "top_latency_task_type": bottleneck_report.get("latency_bottleneck", {}).get("top_task_type", MISSING),
            "top_latency_provider": bottleneck_report.get("latency_bottleneck", {}).get("top_provider", MISSING),
            "slowest_file_id": bottleneck_report.get("latency_bottleneck", {}).get("slowest_file_id", MISSING),
        },
        "decision": _decision_items(bottleneck_report, zero_billed_items),
        "field_notes": field_notes(),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """渲染 Markdown 决策报告。"""

    quality = report["quality_summary"]
    cost = report["cost_summary"]
    latency = report["latency_summary"]
    return "\n".join(
        [
            "# 批次决策摘要报告",
            "",
            f"- 批次编号：{report['batch_id']}",
            f"- 当前状态：{report['readiness']['meaning']}",
            "",
            "## 1. 质量结论",
            "",
            f"- 文件数：{quality['total_files']}",
            f"- 成功率：{_percent(quality['success_rate'])}",
            f"- 主分类 Accuracy：{_percent(quality['accuracy'])}",
            f"- Macro-F1：{_percent(quality['macro_f1'])}",
            f"- 预测覆盖率：{_percent(quality['prediction_coverage'])}",
            f"- 注意：{quality['quality_note']}",
            "",
            "## 2. 成本结论",
            "",
            f"- 系统估算成本：{cost['total_estimated_cost_cny']} 元",
            f"- 后台实际扣费：{cost['total_billed_cost_cny']} 元",
            f"- 成本偏差：{cost['total_cost_delta_cny']} 元",
            f"- 成本偏差率：{_percent(cost['total_cost_delta_rate'])}",
            f"- 账单粒度：{', '.join(cost['billing_granularity'])}",
            f"- 后台扣费为 0 的真实 API 模型：{', '.join(cost['zero_billed_live_api_models']) or '无'}",
            f"- 注意：{cost['cost_note']}",
            "",
            "## 3. 延迟结论",
            "",
            f"- 平均模型延迟：{latency['avg_model_latency_ms']} ms",
            f"- P95 模型延迟：{latency['p95_model_latency_ms']} ms",
            f"- 延迟最高任务：{latency['top_latency_task_type']}",
            f"- 延迟最高供应商：{latency['top_latency_provider']}",
            f"- 最慢文件：{latency['slowest_file_id']}",
            "",
            "## 4. 技术负责人决策建议",
            "",
            *[f"- {item}" for item in report["decision"]],
            "",
            "## 5. 字段说明",
            "",
            "| 字段 | 含义与作用 |",
            "|---|---|",
            *[f"| `{key}` | {value} |" for key, value in report["field_notes"].items()],
            "",
        ]
    )


def write_reports(batch_dir: str | Path) -> dict[str, str]:
    """从批次目录读取已有报告并写出决策摘要。"""

    directory = Path(batch_dir)
    report = build_decision_summary(
        read_json(directory / "batch_report.json"),
        read_json(directory / "bottleneck_report.json"),
        read_json(directory / "cost_reconciliation_report_hour.json"),
        read_json(directory / "video_topic_eval_report.json"),
    )
    json_path = directory / "decision_summary_report.json"
    markdown_path = directory / "decision_summary_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def field_notes() -> dict[str, str]:
    """报告字段说明。"""

    return {
        "success_rate": "文件级成功率，用于判断本批次处理链路是否稳定跑通。",
        "accuracy": "人工评估中的主分类准确率，用于判断模型输出是否命中人工标准答案。",
        "macro_f1": "各分类 F1 的平均值，用于观察小样本中是否存在明显弱类。",
        "total_estimated_cost_cny": "系统根据本地价格表和模型用量估算出的理论成本，单位人民币。",
        "total_billed_cost_cny": "供应商后台或账单记录显示的实际扣费，单位人民币。",
        "total_cost_delta_cny": "实际扣费减去估算成本后的差值，用于判断估算与账单差异。",
        "billing_granularity": "账单核对粒度，例如小时级或周期级；当前不能代表单次调用级精确对账。",
        "zero_billed_live_api_models": "估算成本大于 0 但后台扣费为 0 的真实 API 模型，用于标记免费额度或抵扣口径。",
        "p95_model_latency_ms": "模型调用延迟的 P95，用于判断尾部慢调用是否影响批处理体验。",
        "slowest_file_id": "累计模型调用耗时最高的文件编号，用于定位拖慢批次的输入样本。",
    }


def _decision_items(bottleneck_report: dict[str, Any], zero_billed_items: list[dict[str, Any]]) -> list[str]:
    latency = bottleneck_report.get("latency_bottleneck", {})
    cost = bottleneck_report.get("cost_bottleneck", {})
    items = [
        "当前批次可以作为受控小批量视频链路证据，但样本量仍不足以证明大规模稳定性。",
        f"延迟优先时，先处理 {latency.get('top_task_type', MISSING)} / {latency.get('top_provider', MISSING)}。",
        f"成本优先时，先复核 {cost.get('top_task_type', MISSING)} / {cost.get('top_provider', MISSING)} 的价格和账单口径。",
        "成本报告只能按小时级或周期级解释，不应宣传为单次调用级精确成本。",
    ]
    if zero_billed_items:
        items.append("存在后台扣费为 0 的真实 API 模型，应单独标注免费额度或抵扣口径，不能说模型没有成本。")
    return items


def _model_key(item: dict[str, Any]) -> str:
    return f"{item.get('provider', MISSING)}/{item.get('model_name', MISSING)}"


def _float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return MISSING


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("用法: python .\\src\\decision_summary.py output\\batch_xxx")
        return 2
    print(json.dumps(write_reports(args[0]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
