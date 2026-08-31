"""DeepSeek 文本分析延迟离线诊断。"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from report_generator import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIGH_OUTPUT_TOKEN_THRESHOLD = 800


def diagnose_text_analysis_latency(
    config_path: str | Path = PROJECT_ROOT / "config" / "phase2_benchmark.yaml",
    *,
    project_root: str | Path = PROJECT_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """基于二期基准批次诊断文本分析延迟，不调用任何模型。"""

    root = Path(project_root)
    config = yaml.safe_load(_resolve_path(root, config_path).read_text(encoding="utf-8"))
    batch_dir = _resolve_path(root, config["baseline"]["batch_dir"])
    target_ms = config["criteria"]["preflight"]["max_task_p95_latency_ms"]["text_analysis"]
    calls = [
        call
        for call in read_jsonl(batch_dir / "model_calls.jsonl")
        if call.get("task_type") == "text_analysis" and call.get("status") == "success"
    ]

    call_rows = [_call_row(call, target_ms) for call in calls]
    latencies = [row["latency_ms"] for row in call_rows if isinstance(row["latency_ms"], (int, float))]
    p95_latency_ms = _p95(latencies)
    slow_calls = [row for row in call_rows if row["latency_status"] == "fail"]
    diagnosis = _diagnose(call_rows, target_ms)

    return {
        "schema_version": "v1",
        "report_type": "text_analysis_latency_diagnosis",
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "batch_dir": str(batch_dir),
        "target_p95_latency_ms": target_ms,
        "observed_p95_latency_ms": p95_latency_ms,
        "overall_status": "pass" if p95_latency_ms <= target_ms else "fail",
        "call_count": len(call_rows),
        "slow_call_count": len(slow_calls),
        "calls": call_rows,
        "diagnosis": diagnosis,
        "recommended_next_actions": _recommended_actions(diagnosis),
        "field_notes": _field_notes(),
    }


def write_reports(report: dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> dict[str, str]:
    """写入 JSON 和 Markdown 诊断报告。"""

    json_file = Path(json_path)
    markdown_file = Path(markdown_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    markdown_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_file.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_file), "markdown": str(markdown_file)}


def render_markdown(report: dict[str, Any]) -> str:
    """生成面向人工复核的 Markdown 报告。"""

    lines = [
        "# 文本分析延迟诊断报告",
        "",
        f"- 总状态：`{report['overall_status']}`",
        f"- 目标 P95：{report['target_p95_latency_ms']}ms",
        f"- 实际 P95：{report['observed_p95_latency_ms']}ms",
        f"- 慢调用数：{report['slow_call_count']} / {report['call_count']}",
        "",
        "## 调用明细",
        "",
        "| file_id | latency_ms | input_tokens | output_tokens | latency_status | 诊断 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in report["calls"]:
        lines.append(
            f"| {row['file_id']} | {row['latency_ms']} | {row['input_tokens']} | "
            f"{row['output_tokens']} | {row['latency_status']} | {'; '.join(row['risk_flags']) or '无'} |"
        )
    lines.extend(["", "## 诊断结论", ""])
    for item in report["diagnosis"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 下一步", ""])
    for item in report["recommended_next_actions"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## 字段说明",
            "",
            "| 字段 | 含义 |",
            "|---|---|",
            "| `latency_ms` | 单次文本分析调用耗时，用于判断是否超过当前 P95 目标 |",
            "| `input_tokens` | 输入 token 数，用于判断是否是输入证据过长造成慢调用 |",
            "| `output_tokens` | 输出 token 数，用于判断是否是模型生成内容过长造成慢调用 |",
            "| `latency_status` | 单次调用是否超过当前文本分析目标 |",
        ]
    )
    return "\n".join(lines) + "\n"


def _call_row(call: dict[str, Any], target_ms: int | float) -> dict[str, Any]:
    input_tokens = _unit_quantity(call.get("input_units"), "input_tokens")
    output_tokens = _unit_quantity(call.get("output_units"), "output_tokens")
    latency_ms = call.get("latency_ms")
    return {
        "call_id": call.get("call_id"),
        "file_id": call.get("file_id"),
        "provider": call.get("provider"),
        "model_name": call.get("model_name"),
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_cny": call.get("cost_cny"),
        "latency_status": "fail" if isinstance(latency_ms, (int, float)) and latency_ms > target_ms else "pass",
        "risk_flags": [],
    }


def _diagnose(rows: list[dict[str, Any]], target_ms: int | float) -> list[str]:
    if not rows:
        return ["当前没有可诊断的成功文本分析调用。"]

    output_values = [row["output_tokens"] for row in rows if isinstance(row["output_tokens"], (int, float))]
    input_values = [row["input_tokens"] for row in rows if isinstance(row["input_tokens"], (int, float))]
    median_output = _median(output_values)
    median_input = _median(input_values)
    slow_rows = [row for row in rows if row["latency_status"] == "fail"]

    notes = []
    if slow_rows:
        notes.append(f"当前文本分析 P95 未达标：存在 {len(slow_rows)} 次调用超过 {target_ms}ms。")
    for row in slow_rows:
        has_high_output = row["output_tokens"] >= HIGH_OUTPUT_TOKEN_THRESHOLD
        has_relative_output_pressure = median_output and row["output_tokens"] > median_output * 1.3
        if has_high_output or has_relative_output_pressure:
            row["risk_flags"].append("output_token_pressure")
            reason = (
                f"达到高输出阈值 {HIGH_OUTPUT_TOKEN_THRESHOLD}"
                if has_high_output
                else f"高于中位数 {median_output}"
            )
            notes.append(f"{row['file_id']} 是慢调用，输出 token 为 {row['output_tokens']}，{reason}，更像输出生成过长导致延迟升高。")
        if median_input and row["input_tokens"] > median_input * 1.3:
            row["risk_flags"].append("input_token_pressure")
            notes.append(
                f"{row['file_id']} 输入 token 为 {row['input_tokens']}，明显高于中位数 {median_input}，需要继续压缩输入证据。"
            )
        if not row["risk_flags"]:
            row["risk_flags"].append("external_api_variance")
            notes.append(f"{row['file_id']} 慢调用没有明显 token 异常，可能来自网络、供应商排队或模型服务波动。")
    if not slow_rows:
        notes.append("当前文本分析调用均未超过目标。")
    return notes


def _recommended_actions(diagnosis: list[str]) -> list[str]:
    joined = " ".join(diagnosis)
    if "输出 token" in joined:
        return [
            "如果已经收紧 DeepSeek 输出结构但 P95 仍超过 8000ms，不要继续压低 max_tokens 到 1000。",
            "下一步只做同批次小样本复测或文本模型对照，判断慢因是供应商波动还是模型生成速度上限。",
            "如果模型对照仍无法稳定低于 8000ms，应把文本分析改为异步 SLA 口径，而不是强行宣传同步达标。"
        ]
    return [
        "下一步做同样输入下的文本模型对照或异步化口径判断。",
        "不要扩大视频样本，避免把供应商波动和样本差异混在一起。"
    ]


def _unit_quantity(units: Any, unit_type: str) -> int | float | None:
    if not isinstance(units, list):
        return None
    for item in units:
        if isinstance(item, dict) and item.get("unit_type") == unit_type:
            return item.get("quantity")
    return None


def _p95(values: list[int | float]) -> int | float:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def _median(values: list[int | float]) -> int | float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _field_notes() -> dict[str, str]:
    return {
        "latency_ms": "单次文本分析调用耗时，用于判断是否超过当前 P95 目标。",
        "input_tokens": "输入 token 数，用于判断输入证据是否过长。",
        "output_tokens": "输出 token 数，用于判断生成内容是否过长。",
        "risk_flags": "机器可读风险标签，用于标记慢调用更像输入过长、输出过长还是外部 API 波动。",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="诊断二期基准中的文本分析延迟慢因。")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "phase2_benchmark.yaml"), help="二期基准配置文件路径。")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="项目根目录。")
    parser.add_argument("--output-json", default=str(PROJECT_ROOT / "output" / "text_analysis_latency_diagnosis_current.json"), help="输出 JSON 报告路径。")
    parser.add_argument("--output-md", default=str(PROJECT_ROOT / "output" / "text_analysis_latency_diagnosis_current.md"), help="输出 Markdown 报告路径。")
    args = parser.parse_args(argv)

    report = diagnose_text_analysis_latency(args.config, project_root=args.project_root)
    paths = write_reports(report, args.output_json, args.output_md)
    print(json.dumps({"overall_status": report["overall_status"], "paths": paths}, ensure_ascii=False, indent=2))
    return 0 if report["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
