"""基于既有 Demo 批次的路由策略离线模拟器。

该模块只读取 batch_report.json 和 model_calls.jsonl，不触发 DeepSeek 或任何外部 API。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from model_catalog import UNKNOWN_VALUE_TEXT, build_model_catalog, summarize_catalog
from model_strategy_advisor import read_json, read_json_objects, write_json, write_markdown
from routing_policy import evaluate_routing_policy, list_policy_names, load_policy_config, simulate_budget_expansion


def simulate_routing_policies(
    batch_report: dict[str, Any],
    model_calls: list[dict[str, Any]],
    *,
    generated_at: str | None = None,
    constraints_override: dict[str, dict[str, Any]] | None = None,
    budget_expansion_multipliers: tuple[int, ...] = (2, 5, 10),
    config_source: str = "code_defaults",
) -> dict[str, Any]:
    """对同一批次调用记录运行多种路由策略模拟。"""

    catalog = build_model_catalog(model_calls)
    catalog_summary = summarize_catalog(catalog)
    policy_results = {}
    for policy_name in list_policy_names():
        policy_results[policy_name] = evaluate_routing_policy(
            policy_name,
            catalog,
            batch_report,
            constraints_override=(constraints_override or {}).get(policy_name),
        )

    return {
        "schema_version": "v1",
        "report_type": "routing_policy_simulation",
        "batch_id": batch_report.get("batch_id", UNKNOWN_VALUE_TEXT),
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_files": {
            "batch_report": "batch_report.json",
            "model_calls": "model_calls.jsonl",
        },
        "policy_config": {
            "config_source": config_source,
            "budget_expansion_multipliers": list(budget_expansion_multipliers),
            "constraints_override": constraints_override or {},
        },
        "field_notes": _field_notes(),
        "current_call_structure": _current_call_structure(batch_report, model_calls, catalog_summary),
        "model_catalog": catalog,
        "policy_results": policy_results,
        "budget_expansion_simulation": simulate_budget_expansion(
            batch_report,
            catalog,
            multipliers=budget_expansion_multipliers,
        ),
        "technical_lead_recommendation": _technical_lead_recommendation(policy_results, catalog_summary),
        "insufficient_data_and_boundaries": _insufficient_data_and_boundaries(catalog_summary),
    }


def _field_notes() -> dict[str, str]:
    """解释模拟报告中新出现的关键字段。"""

    return {
        "policy_name": "路由策略名称，用于区分成本优先、延迟优先、质量优先和平衡策略。",
        "constraint_status": "约束满足状态，用于判断当前模型组合是否符合某种业务目标。",
        "real_coverage_rate": "真实模型调用占全部模型调用的比例，用于衡量结果有多少来自真实 API 证据。",
        "mock_coverage_rate": "mock 调用占全部模型调用的比例，用于提示哪些链路仍是占位流程。",
        "budget_multiplier": "预算扩展倍数，用于模拟预算变为当前预算 2 倍、5 倍、10 倍后的升级优先级。",
        "upgrade_priority": "建议优先从 mock 替换为真实模型的任务列表，用于指导下一步接入顺序。",
        "missing_data_notes": "缺失字段说明，用于提醒哪些结论当前不能从 Demo 数据中推断。",
        "config_source": "策略约束配置来源，用于追踪本次模拟使用的是代码默认值还是 YAML 配置文件。",
    }


def _current_call_structure(
    batch_report: dict[str, Any],
    model_calls: list[dict[str, Any]],
    catalog_summary: dict[str, Any],
) -> dict[str, Any]:
    """生成当前调用结构摘要。"""

    return {
        "total_files": batch_report.get("file_stats", {}).get("total_files", UNKNOWN_VALUE_TEXT),
        "total_cost_cny": batch_report.get("cost_stats", {}).get("total_cost_cny", UNKNOWN_VALUE_TEXT),
        "budget_limit_cny": batch_report.get("cost_stats", {}).get("budget_limit_cny", UNKNOWN_VALUE_TEXT),
        "avg_processing_time_per_file_ms": batch_report.get("latency_stats", {}).get(
            "avg_processing_time_per_file_ms",
            UNKNOWN_VALUE_TEXT,
        ),
        "p95_model_latency_ms": batch_report.get("latency_stats", {}).get(
            "p95_model_latency_ms",
            UNKNOWN_VALUE_TEXT,
        ),
        "model_call_count": len(model_calls),
        "real_model_calls": catalog_summary["real_model_calls"],
        "mock_model_calls": catalog_summary["mock_model_calls"],
        "real_coverage_rate": catalog_summary["real_coverage_rate"],
        "mock_coverage_rate": catalog_summary["mock_coverage_rate"],
        "real_task_types": catalog_summary["real_task_types"],
        "mock_task_types": catalog_summary["mock_task_types"],
    }


def _technical_lead_recommendation(
    policy_results: dict[str, dict[str, Any]],
    catalog_summary: dict[str, Any],
) -> dict[str, Any]:
    """从技术负责人视角给出最终建议。"""

    quality_status = policy_results["quality_first"]["constraint_status"]
    balanced_status = policy_results["balanced"]["constraint_status"]
    if catalog_summary["mock_model_calls"] > 0:
        decision = "建议采用平衡策略作为下一阶段试点路线，并优先补真实 OCR 与 ASR 小样本验证。"
    elif balanced_status == "pass":
        decision = "当前组合适合作为试点默认组合，但仍需要持续观察成本和延迟。"
    else:
        decision = "当前数据不足以直接给出默认组合，应先补齐约束数据。"

    return {
        "recommended_policy": "balanced",
        "decision": decision,
        "reason": (
            "成本优先和延迟优先能控制工程风险，但质量优先暴露了真实覆盖率不足。"
            f" 当前质量优先状态为 {quality_status}，平衡策略状态为 {balanced_status}。"
        ),
        "next_live_test_focus": ["真实 OCR", "真实 ASR", "真实视觉理解"],
    }


def _insufficient_data_and_boundaries(catalog_summary: dict[str, Any]) -> list[str]:
    """列出当前不能推断的内容。"""

    notes = [
        "当前数据未提供人工标注质量结果，不能计算 Accuracy、F1、PSNR、SSIM 或 VMAF。",
        "当前数据未提供真实 OCR、ASR、视觉理解调用结果，不能评价真实图片或视频理解质量。",
        "mock 模型的 cost_cny 和 latency_ms 只能用于流程演示，不能等同于供应商真实账单或真实延迟。",
        "当前只有一个 Demo 批次，不能推出长期稳定性、峰值并发能力或供应商横向优劣。",
    ]
    if catalog_summary["mock_model_calls"] == 0:
        notes.append("当前没有 mock 调用，但仍需要更多批次验证稳定性。")
    return notes


def render_simulation_markdown(report: dict[str, Any]) -> str:
    """把路由策略模拟报告渲染为 Markdown。"""

    current = report["current_call_structure"]
    policy_config = report.get("policy_config", {})
    lines = [
        "# 路由策略模拟报告",
        "",
        f"批次编号：{current.get('batch_id', report.get('batch_id', UNKNOWN_VALUE_TEXT))}",
        "",
        "说明：本报告只基于已有 Demo 批次离线生成，没有重新跑批处理，也没有触发任何外部模型 API。",
        "",
        f"策略配置来源：`{policy_config.get('config_source', UNKNOWN_VALUE_TEXT)}`",
        "",
        f"预算扩展倍数：{_format_multiplier_list(policy_config.get('budget_expansion_multipliers'))}",
        "",
        "## 1. 当前模型调用结构",
        "",
        "| 指标 | 数值 | 含义 |",
        "|---|---:|---|",
        f"| 文件数 | {_format_value(current.get('total_files'))} | 本批次处理的输入文件数量 |",
        f"| 模型调用次数 | {_format_value(current.get('model_call_count'))} | 本批次记录到的模型调用总数 |",
        f"| 总成本 | {_format_cny(current.get('total_cost_cny'))} | 本批次记录到的模型调用成本合计 |",
        f"| 预算上限 | {_format_cny(current.get('budget_limit_cny'))} | 本批次配置的人民币预算上限 |",
        f"| 平均文件耗时 | {_format_ms(current.get('avg_processing_time_per_file_ms'))} | 文件级处理耗时平均值 |",
        f"| P95 模型延迟 | {_format_ms(current.get('p95_model_latency_ms'))} | 模型调用层面的 95 分位延迟 |",
        f"| 真实模型调用数 | {_format_value(current.get('real_model_calls'))} | 来自真实模型 API 的调用次数 |",
        f"| mock 调用数 | {_format_value(current.get('mock_model_calls'))} | 来自占位流程的调用次数 |",
        f"| 真实模型覆盖率 | {_format_percent(current.get('real_coverage_rate'))} | 真实调用占全部模型调用的比例 |",
        f"| mock 覆盖率 | {_format_percent(current.get('mock_coverage_rate'))} | mock 调用占全部模型调用的比例 |",
        "",
        "## 2. 模型目录",
        "",
        "| provider/model/task | 真实? | mock? | 调用数 | 成本 | P95 延迟 | 证据状态 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for entry in report["model_catalog"]:
        lines.append(
            "| "
            f"{entry['model_id']} | "
            f"{'是' if entry['is_real'] else '否'} | "
            f"{'是' if entry['is_mock'] else '否'} | "
            f"{entry['call_count']} | "
            f"{_format_cny(entry['total_cost_cny'])} | "
            f"{_format_ms(entry['p95_latency_ms'])} | "
            f"{entry['quality_evidence_status']} |"
        )

    lines.extend(["", "## 3. 不同策略下的推荐方案", ""])
    for policy_name, result in report["policy_results"].items():
        lines.extend(
            [
                f"### {result['display_name']}（{policy_name}）",
                "",
                f"- 目标：{result['primary_objective']}",
                f"- 约束状态：`{result['constraint_status']}`",
                f"- 取舍解释：{result['recommendation']['tradeoff_explanation']}",
                "- 推荐组合：",
            ]
        )
        lines.extend(f"  - {item}" for item in result["recommendation"]["recommended_combo"])
        if result["recommendation"]["risk_warnings"]:
            lines.append("- 风险提示：")
            lines.extend(f"  - {item}" for item in result["recommendation"]["risk_warnings"])
        lines.extend(["", "约束检查：", "", "| 约束 | 观测值 | 限制值 | 状态 | 说明 |", "|---|---:|---:|---|---|"])
        for check in result["constraint_checks"]:
            lines.append(
                f"| {check['constraint_name']} | {_format_value(check['observed_value'])} | "
                f"{_format_value(check['limit_value'])} | {check['status']} | {check['reason']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 4. 预算扩展模拟",
            "",
            "| 预算倍数 | 扩展后预算 | 建议替换优先级 | 说明 |",
            "|---:|---:|---|---|",
        ]
    )
    for scenario in report["budget_expansion_simulation"]:
        priorities = "、".join(item["recommended_replacement"] for item in scenario["upgrade_priority"]) or UNKNOWN_VALUE_TEXT
        lines.append(
            f"| {scenario['budget_multiplier']}x | {_format_cny(scenario['expanded_budget_cny'])} | "
            f"{priorities} | {scenario['recommendation']} |"
        )

    recommendation = report["technical_lead_recommendation"]
    lines.extend(
        [
            "",
            "## 5. 技术负责人视角下的最终建议",
            "",
            f"- 推荐策略：`{recommendation['recommended_policy']}`",
            f"- 决策建议：{recommendation['decision']}",
            f"- 原因：{recommendation['reason']}",
            f"- 下一步 live test 重点：{'、'.join(recommendation['next_live_test_focus'])}",
            "",
            "## 6. 当前数据不足和不可推断内容",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["insufficient_data_and_boundaries"])

    lines.extend(["", "## 7. 字段说明", "", "| 字段 | 含义与作用 |", "|---|---|"])
    for field_name, note in report["field_notes"].items():
        lines.append(f"| `{field_name}` | {note} |")

    lines.append("")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    """把数值或缺失信息转成展示文本。"""

    if value in {None, UNKNOWN_VALUE_TEXT}:
        return UNKNOWN_VALUE_TEXT
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _format_cny(value: Any) -> str:
    """格式化人民币金额。"""

    if value in {None, UNKNOWN_VALUE_TEXT}:
        return UNKNOWN_VALUE_TEXT
    return f"{float(value):.6f} 元"


def _format_ms(value: Any) -> str:
    """格式化毫秒延迟。"""

    if value in {None, UNKNOWN_VALUE_TEXT}:
        return UNKNOWN_VALUE_TEXT
    return f"{_format_value(float(value))} ms"


def _format_percent(value: Any) -> str:
    """格式化比例。"""

    if value in {None, UNKNOWN_VALUE_TEXT}:
        return UNKNOWN_VALUE_TEXT
    return f"{float(value) * 100:.2f}%"


def _format_multiplier_list(value: Any) -> str:
    """格式化预算扩展倍数列表。"""

    if not value:
        return UNKNOWN_VALUE_TEXT
    if isinstance(value, list):
        return "、".join(f"{item}x" for item in value)
    return str(value)


def build_simulation_from_files(
    batch_dir: str | Path,
    *,
    generated_at: str | None = None,
    policy_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """从指定批次目录读取文件并生成模拟报告。"""

    path = Path(batch_dir)
    policy_config = None
    if policy_config_path is not None:
        policy_config = load_policy_config(policy_config_path)

    return simulate_routing_policies(
        read_json(path / "batch_report.json"),
        read_json_objects(path / "model_calls.jsonl"),
        generated_at=generated_at,
        constraints_override=policy_config["policy_overrides"] if policy_config else None,
        budget_expansion_multipliers=policy_config["budget_expansion_multipliers"] if policy_config else (2, 5, 10),
        config_source=policy_config["config_source"] if policy_config else "code_defaults",
    )


def write_simulation_reports(batch_dir: str | Path, report: dict[str, Any]) -> dict[str, str]:
    """写入路由策略模拟报告 JSON 和 Markdown。"""

    path = Path(batch_dir)
    json_path = write_json(path / "routing_policy_simulation.json", report)
    markdown_path = write_markdown(path / "routing_policy_simulation.md", render_simulation_markdown(report))
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    """命令行入口：为指定批次目录生成路由策略模拟报告。"""

    args = argv if argv is not None else sys.argv[1:]
    if len(args) not in {1, 2}:
        print("用法: python .\\src\\strategy_simulator.py output\\batch_xxx [config\\routing_policy_config.yaml]")
        return 2

    batch_dir = Path(args[0])
    policy_config_path = Path(args[1]) if len(args) == 2 else None
    report = build_simulation_from_files(batch_dir, policy_config_path=policy_config_path)
    output_paths = write_simulation_reports(batch_dir, report)
    print(json.dumps(output_paths, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
