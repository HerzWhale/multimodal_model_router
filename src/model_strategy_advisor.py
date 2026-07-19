"""基于已有批次输出生成模型组合建议报告。

本模块只读取 batch_report.json 和 model_calls.jsonl，不重新运行批处理，
也不触发任何外部模型 API。它的目标是把成本、延迟、真实/mock 边界
整理成技术负责人能用于决策的报告。
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


MISSING_VALUE_TEXT = "当前数据未提供"


def read_json(file_path: str | Path) -> dict[str, Any]:
    """读取 JSON 文件。"""

    path = Path(file_path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_objects(file_path: str | Path) -> list[dict[str, Any]]:
    """读取连续 JSON 对象文件，兼容缩进式 JSONL。"""

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


def write_json(file_path: str | Path, data: dict[str, Any]) -> Path:
    """写入 JSON 文件。"""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_markdown(file_path: str | Path, content: str) -> Path:
    """写入 Markdown 文件。"""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _get_nested(data: dict[str, Any], path: list[str], missing_notes: list[str]) -> Any:
    """从嵌套字典中读取字段；缺失时记录说明并返回 None。"""

    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            missing_notes.append(f"当前数据未提供：{'.'.join(path)}")
            return None
        current = current[key]
    return current


def _as_float(value: Any, default: float = 0.0) -> float:
    """把值安全转换为浮点数。"""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_text(value: Any, default: str = MISSING_VALUE_TEXT) -> str:
    """把值安全转换为文本。"""

    if value is None:
        return default
    text = str(value)
    return text if text else default


def _is_mock_call(call: dict[str, Any]) -> bool:
    """判断一次模型调用是否为 mock 调用。"""

    model_name = str(call.get("model_name") or "").lower()
    return model_name.startswith("mock")


def _p95(values: list[float]) -> float:
    """计算 95 分位值；没有数据时返回 0。"""

    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = math.ceil(len(sorted_values) * 0.95) - 1
    return sorted_values[index]


def _sum_by(calls: list[dict[str, Any]], key: str) -> dict[str, float]:
    """按指定字段汇总成本。"""

    totals: dict[str, float] = defaultdict(float)
    for call in calls:
        group_name = str(call.get(key) or "unknown")
        totals[group_name] += _as_float(call.get("cost_cny"))
    return {name: round(value, 6) for name, value in sorted(totals.items())}


def _sum_by_model(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按供应商和模型汇总成本。"""

    totals: dict[tuple[str, str], float] = defaultdict(float)
    for call in calls:
        provider = str(call.get("provider") or "unknown")
        model_name = str(call.get("model_name") or "unknown")
        totals[(provider, model_name)] += _as_float(call.get("cost_cny"))

    return [
        {
            "provider": provider,
            "model_name": model_name,
            "cost_cny": round(cost, 6),
        }
        for (provider, model_name), cost in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def _add_share(items: list[dict[str, Any]], total_cost: float) -> list[dict[str, Any]]:
    """给成本项补充占比。"""

    result = []
    for item in items:
        cost = _as_float(item.get("cost_cny"))
        new_item = dict(item)
        new_item["cost_share"] = round(cost / total_cost, 6) if total_cost else 0.0
        result.append(new_item)
    return result


def _slowest_calls(calls: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """按延迟从高到低返回最慢调用。"""

    sorted_calls = sorted(calls, key=lambda call: _as_float(call.get("latency_ms")), reverse=True)
    result = []
    for call in sorted_calls[:limit]:
        result.append(
            {
                "call_id": call.get("call_id"),
                "file_id": call.get("file_id"),
                "task_type": call.get("task_type"),
                "provider": call.get("provider"),
                "model_name": call.get("model_name"),
                "latency_ms": _as_float(call.get("latency_ms")),
                "is_mock": _is_mock_call(call),
            }
        )
    return result


def _call_boundary(calls: list[dict[str, Any]], *, is_mock: bool) -> dict[str, Any]:
    """汇总真实或 mock 调用边界。"""

    selected = [call for call in calls if _is_mock_call(call) is is_mock]
    task_types = sorted({str(call.get("task_type") or "unknown") for call in selected})
    models = sorted({f"{call.get('provider')}/{call.get('model_name')}" for call in selected})
    return {
        "count": len(selected),
        "task_types": task_types,
        "models": models,
        "cost_cny": round(sum(_as_float(call.get("cost_cny")) for call in selected), 6),
    }


def _deepseek_latency(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 DeepSeek 调用延迟。"""

    deepseek_calls = [call for call in calls if str(call.get("provider") or "").lower() == "deepseek"]
    values = [_as_float(call.get("latency_ms")) for call in deepseek_calls]
    return {
        "call_count": len(deepseek_calls),
        "avg_latency_ms": round(sum(values) / len(values), 6) if values else None,
        "p95_latency_ms": _p95(values) if values else None,
        "max_latency_ms": max(values) if values else None,
    }


def generate_strategy_report(
    batch_report: dict[str, Any],
    model_calls: list[dict[str, Any]],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """根据批次报告和模型调用明细生成策略建议。"""

    missing_notes: list[str] = []
    total_files = _get_nested(batch_report, ["file_stats", "total_files"], missing_notes)
    success_rate = _get_nested(batch_report, ["file_stats", "success_rate"], missing_notes)
    total_cost_cny = _get_nested(batch_report, ["cost_stats", "total_cost_cny"], missing_notes)
    avg_latency_ms = _get_nested(batch_report, ["latency_stats", "avg_processing_time_per_file_ms"], missing_notes)
    p95_latency_ms = _get_nested(batch_report, ["latency_stats", "p95_model_latency_ms"], missing_notes)

    total_cost = _as_float(total_cost_cny)
    cost_by_task_type = _sum_by(model_calls, "task_type")
    cost_by_provider = _sum_by(model_calls, "provider")
    cost_by_model = _sum_by_model(model_calls)
    mock_boundary = _call_boundary(model_calls, is_mock=True)
    real_boundary = _call_boundary(model_calls, is_mock=False)
    deepseek_cost = _as_float(cost_by_provider.get("deepseek"))
    mock_cost = _as_float(mock_boundary["cost_cny"])
    slowest_calls = _slowest_calls(model_calls)
    deepseek_latency = _deepseek_latency(model_calls)
    p95_value = _as_float(p95_latency_ms)
    slowest_call = slowest_calls[0] if slowest_calls else None

    p95_driver = MISSING_VALUE_TEXT
    if slowest_call and p95_value and _as_float(slowest_call.get("latency_ms")) >= p95_value:
        p95_driver = (
            f"P95 延迟主要由 {slowest_call.get('provider')}/"
            f"{slowest_call.get('model_name')} 的 {slowest_call.get('task_type')} 调用拉高。"
        )
    elif slowest_call:
        p95_driver = "当前数据能识别最慢调用，但无法确认 P95 是否由单次调用拉高。"

    return {
        "schema_version": "v1",
        "report_type": "model_strategy",
        "batch_id": batch_report.get("batch_id"),
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_files": {
            "batch_report": "batch_report.json",
            "model_calls": "model_calls.jsonl",
        },
        "field_notes": {
            "batch_id": "批次唯一标识，用来定位本报告分析的是哪一次批处理。",
            "model_call_count": "模型调用次数，用来衡量本批次实际触发了多少次模型任务。",
            "cost_cny": "成本金额，单位人民币，用于成本核算和预算判断。",
            "latency_ms": "延迟时间，单位毫秒，用于分析模型调用或文件处理耗时。",
            "is_mock": "是否为 mock 调用，用于区分真实模型调用和占位调用。",
            "cost_share": "成本占比，用于判断某个任务、供应商或模型是否构成主要成本来源。",
        },
        "batch_overview": {
            "total_files": total_files,
            "success_rate": success_rate,
            "total_cost_cny": total_cost_cny,
            "avg_processing_time_ms": avg_latency_ms,
            "p95_model_latency_ms": p95_latency_ms,
            "model_call_count": len(model_calls),
        },
        "cost_analysis": {
            "cost_by_task_type": cost_by_task_type,
            "cost_by_provider": cost_by_provider,
            "top_cost_tasks": _add_share(
                [{"task_type": key, "cost_cny": value} for key, value in sorted(cost_by_task_type.items(), key=lambda item: item[1], reverse=True)],
                total_cost,
            ),
            "top_cost_models": _add_share(cost_by_model, total_cost),
            "deepseek_cost_cny": round(deepseek_cost, 6),
            "deepseek_cost_share": round(deepseek_cost / total_cost, 6) if total_cost else 0.0,
            "mock_cost_cny": round(mock_cost, 6),
            "mock_cost_share": round(mock_cost / total_cost, 6) if total_cost else 0.0,
            "mock_cost_counted": mock_cost > 0,
            "cost_reliability": "当前总成本可证明成本核算链路已经跑通；但 mock OCR、mock 视觉理解和 mock 语音识别的成本不是实际供应商账单，真实预算判断应优先看 DeepSeek 文本分析部分。",
        },
        "latency_analysis": {
            "slowest_calls": slowest_calls,
            "deepseek_latency": deepseek_latency,
            "p95_driver": p95_driver,
            "current_bottleneck": "当前延迟瓶颈是 DeepSeek 文本分析；mock 上游调用延迟为 0，不能代表真实 OCR、视觉理解或语音识别延迟。",
        },
        "quality_boundary": {
            "real_model_calls": real_boundary,
            "mock_model_calls": mock_boundary,
            "can_prove": [
                "系统可以证明文件级批处理、模型调用记录、成本核算、延迟统计和统一输出链路已经跑通。",
                "系统可以证明 DeepSeek 文本分析已产生真实调用记录，并记录了成本和延迟。",
            ],
            "cannot_prove": [
                "当前不能证明真实 OCR、视觉理解或语音识别质量。",
                "当前不能用 mock 图片/视频上游结果评价真实多模态模型效果。",
                "当前不能证明多个真实供应商之间的质量、成本或延迟对比。",
            ],
        },
        "recommendations": {
            "current_combo": {
                "suitable_for": "适合文本为主、图像和视频上游仍处于流程验证阶段的内部 Demo 或方案评估。",
                "reason": "当前组合能展示统一输入、统一输出、真实文本分析和成本延迟追踪，但还不能承担真实图片/视频理解任务。",
            },
            "budget_sensitive": {
                "recommendation": "优先保留低成本文本分析和批次级成本统计，先不要扩大真实多模态调用范围；对图片/视频只抽小样本做 live test。",
                "reason": "本批次 DeepSeek 成本占比较低，但 mock 上游成本不等于真实账单，直接全量接入真实多模态可能放大预算风险。",
            },
            "latency_sensitive": {
                "recommendation": "优先优化或异步化文本分析，并为图片/视频真实上游模型设置单独 P95 目标。",
                "reason": "当前 P95 主要来自 DeepSeek 文本分析；mock 上游延迟为 0，暂不能估算真实图片/视频处理等待时间。",
            },
            "quality_first": {
                "recommendation": "下一步接入真实 OCR、ASR 和视觉理解模型，并建立小样本人工标注集验证质量。",
                "reason": "只有真实上游证据进入文本分析，才能判断图片和视频最终分类、摘要和业务用途是否可信。",
            },
            "replace_mock_upstream": {
                "recommended": True,
                "reason": "建议逐步把图片/视频上游从 mock 替换为真实 OCR、ASR 和视觉理解模型；否则项目的多模态质量结论始终停留在流程验证层。",
            },
        },
        "roadmap": {
            "low_cost_route": [
                "保留 DeepSeek 文本分析真实调用。",
                "图片/视频先抽样接入真实上游模型，控制 live test 数量。",
                "继续用批次报告观察成本增长曲线。",
            ],
            "balanced_route": [
                "接入一个真实 OCR 和一个真实 ASR，优先覆盖图片文字和视频音频。",
                "保留 mock 视觉理解作为兜底路径。",
                "增加失败和部分成功样例，观察证据缺失对结果可信度的影响。",
            ],
            "high_quality_route": [
                "接入真实 OCR、真实 ASR 和真实视觉理解模型。",
                "为每类任务建立人工评估样本。",
                "把模型路由从固定规则升级为按成本、延迟和质量目标选择组合。",
            ],
            "live_test_route": [
                "新增受保护 live test 开关，默认关闭。",
                "只有显式设置环境变量和测试标记时才调用真实 API。",
                "记录 live test 的实际费用、延迟和失败情况。",
            ],
            "interview_talk_track": [
                "先讲原问题：内容平台批量使用大模型时，成本、延迟、模型链路很难追踪。",
                "再讲系统能力：统一输入输出、调用明细、批次报告和策略建议。",
                "主动讲边界：当前真实调用是 DeepSeek 文本分析，图片/视频上游仍是 mock。",
                "最后讲下一步：接入真实上游模型，让策略报告从流程判断升级为真实模型组合决策。",
            ],
        },
        "missing_data_notes": sorted(set(missing_notes)),
    }


def render_strategy_markdown(report: dict[str, Any]) -> str:
    """把策略报告渲染为 Markdown。"""

    overview = report["batch_overview"]
    cost = report["cost_analysis"]
    latency = report["latency_analysis"]
    boundary = report["quality_boundary"]
    recommendations = report["recommendations"]
    roadmap = report["roadmap"]

    lines = [
        "# 模型组合策略报告",
        "",
        f"批次编号：{_as_text(report.get('batch_id'))}",
        "",
        "说明：本报告只基于已有 `batch_report.json` 和 `model_calls.jsonl` 生成，不重新运行批处理，也不触发任何外部模型 API。",
        "",
        "## 1. 批次概览",
        "",
        "| 指标 | 数值 | 含义 |",
        "|---|---:|---|",
        f"| 文件数 | {_as_text(overview.get('total_files'))} | 本批次处理的输入文件数量 |",
        f"| 成功率 | {_format_rate(overview.get('success_rate'))} | 文件级结果成功生成比例 |",
        f"| 总成本 | {_format_cny(overview.get('total_cost_cny'))} | 本批次记录到的模型调用成本合计 |",
        f"| 平均文件处理耗时 | {_format_ms(overview.get('avg_processing_time_ms'))} | 文件级处理耗时平均值 |",
        f"| P95 模型调用延迟 | {_format_ms(overview.get('p95_model_latency_ms'))} | 单次模型调用的 95 分位延迟 |",
        f"| 模型调用次数 | {overview.get('model_call_count')} | 本批次实际记录的模型调用条数 |",
        "",
        "## 2. 成本分析",
        "",
        f"- DeepSeek 成本：{_format_cny(cost.get('deepseek_cost_cny'))}，占总成本 {_format_percent(cost.get('deepseek_cost_share'))}。",
        f"- Mock 调用成本：{_format_cny(cost.get('mock_cost_cny'))}，占总成本 {_format_percent(cost.get('mock_cost_share'))}。",
        f"- Mock 调用是否计入成本：{'是' if cost.get('mock_cost_counted') else '否'}。",
        f"- 成本可信度结论：{cost.get('cost_reliability')}",
        "",
        "按任务类型的成本：",
        "",
        "| 任务类型 | 成本 | 占比 |",
        "|---|---:|---:|",
    ]
    for item in cost["top_cost_tasks"]:
        lines.append(f"| {item.get('task_type')} | {_format_cny(item.get('cost_cny'))} | {_format_percent(item.get('cost_share'))} |")

    lines.extend(
        [
            "",
            "按模型的成本：",
            "",
            "| 供应商 | 模型 | 成本 | 占比 |",
            "|---|---|---:|---:|",
        ]
    )
    for item in cost["top_cost_models"]:
        lines.append(
            f"| {item.get('provider')} | {item.get('model_name')} | {_format_cny(item.get('cost_cny'))} | {_format_percent(item.get('cost_share'))} |"
        )

    lines.extend(
        [
            "",
            "## 3. 延迟分析",
            "",
            f"- DeepSeek 平均延迟：{_format_ms(latency['deepseek_latency'].get('avg_latency_ms'))}。",
            f"- DeepSeek P95 延迟：{_format_ms(latency['deepseek_latency'].get('p95_latency_ms'))}。",
            f"- P95 来源判断：{latency.get('p95_driver')}",
            f"- 当前瓶颈：{latency.get('current_bottleneck')}",
            "",
            "最慢调用：",
            "",
            "| call_id | file_id | task_type | provider/model | latency_ms | mock? |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for call in latency["slowest_calls"]:
        provider_model = f"{call.get('provider')}/{call.get('model_name')}"
        lines.append(
            f"| {call.get('call_id')} | {call.get('file_id')} | {call.get('task_type')} | {provider_model} | {_format_number(call.get('latency_ms'))} | {'是' if call.get('is_mock') else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 4. 质量与可信度边界",
            "",
            f"- 真实模型调用数：{boundary['real_model_calls'].get('count')}。",
            f"- Mock 模型调用数：{boundary['mock_model_calls'].get('count')}。",
            f"- 真实任务类型：{_join_list(boundary['real_model_calls'].get('task_types'))}。",
            f"- Mock 任务类型：{_join_list(boundary['mock_model_calls'].get('task_types'))}。",
            "",
            "当前 Demo 能证明：",
        ]
    )
    lines.extend(f"- {item}" for item in boundary["can_prove"])
    lines.append("")
    lines.append("当前 Demo 不能证明：")
    lines.extend(f"- {item}" for item in boundary["cannot_prove"])

    lines.extend(
        [
            "",
            "## 5. 模型组合建议",
            "",
            f"- 当前组合适用场景：{recommendations['current_combo']['suitable_for']}",
            f"- 当前组合原因：{recommendations['current_combo']['reason']}",
            f"- 预算敏感建议：{recommendations['budget_sensitive']['recommendation']}",
            f"- 延迟敏感建议：{recommendations['latency_sensitive']['recommendation']}",
            f"- 质量优先建议：{recommendations['quality_first']['recommendation']}",
            f"- 是否建议替换 mock 上游：{'是' if recommendations['replace_mock_upstream']['recommended'] else '否'}。{recommendations['replace_mock_upstream']['reason']}",
            "",
            "## 6. 后续路线",
            "",
            "低成本路线：",
        ]
    )
    lines.extend(f"- {item}" for item in roadmap["low_cost_route"])
    lines.append("")
    lines.append("平衡路线：")
    lines.extend(f"- {item}" for item in roadmap["balanced_route"])
    lines.append("")
    lines.append("高质量路线：")
    lines.extend(f"- {item}" for item in roadmap["high_quality_route"])
    lines.append("")
    lines.append("Live test 路线：")
    lines.extend(f"- {item}" for item in roadmap["live_test_route"])
    lines.append("")
    lines.append("面试讲述建议：")
    lines.extend(f"- {item}" for item in roadmap["interview_talk_track"])

    lines.extend(
        [
            "",
            "## 7. 字段说明",
            "",
            "| 字段 | 含义与作用 |",
            "|---|---|",
        ]
    )
    for field_name, note in report["field_notes"].items():
        lines.append(f"| `{field_name}` | {note} |")

    if report.get("missing_data_notes"):
        lines.extend(["", "## 8. 数据缺失说明", ""])
        lines.extend(f"- {item}" for item in report["missing_data_notes"])

    lines.append("")
    return "\n".join(lines)


def _format_number(value: Any) -> str:
    """格式化数字。"""

    if value is None:
        return MISSING_VALUE_TEXT
    try:
        number = float(value)
    except (TypeError, ValueError):
        return MISSING_VALUE_TEXT
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _format_rate(value: Any) -> str:
    """格式化比例。"""

    if value is None:
        return MISSING_VALUE_TEXT
    return f"{_as_float(value) * 100:.2f}%"


def _format_percent(value: Any) -> str:
    """格式化占比。"""

    if value is None:
        return MISSING_VALUE_TEXT
    return f"{_as_float(value) * 100:.2f}%"


def _format_cny(value: Any) -> str:
    """格式化人民币金额。"""

    if value is None:
        return MISSING_VALUE_TEXT
    return f"{_as_float(value):.6f} 元"


def _format_ms(value: Any) -> str:
    """格式化毫秒。"""

    if value is None:
        return MISSING_VALUE_TEXT
    return f"{_format_number(value)} ms"


def _join_list(value: Any) -> str:
    """格式化列表。"""

    if not value:
        return "无"
    if isinstance(value, list):
        return "、".join(str(item) for item in value)
    return str(value)


def build_strategy_report_from_files(
    batch_dir: str | Path,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """从批次目录读取文件并生成策略报告。"""

    path = Path(batch_dir)
    return generate_strategy_report(
        read_json(path / "batch_report.json"),
        read_json_objects(path / "model_calls.jsonl"),
        generated_at=generated_at,
    )


def write_strategy_reports(batch_dir: str | Path, report: dict[str, Any]) -> dict[str, str]:
    """写入策略报告 JSON 和 Markdown。"""

    path = Path(batch_dir)
    json_path = write_json(path / "model_strategy_report.json", report)
    markdown_path = write_markdown(path / "model_strategy_report.md", render_strategy_markdown(report))
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }


def main(argv: list[str] | None = None) -> int:
    """命令行入口：为指定批次目录生成策略报告。"""

    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("用法: python .\\src\\model_strategy_advisor.py output\\batch_xxx")
        return 2

    batch_dir = Path(args[0])
    report = build_strategy_report_from_files(batch_dir)
    output_paths = write_strategy_reports(batch_dir, report)
    print(json.dumps(output_paths, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
