"""二期真实链路稳定化离线出闸检查。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from report_generator import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_phase2_gate(
    config_path: str | Path = PROJECT_ROOT / "config" / "phase2_benchmark.yaml",
    *,
    project_root: str | Path = PROJECT_ROOT,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """读取二期基准配置和已有输出，返回机器可读出闸结果。"""

    root = Path(project_root)
    config_file = _resolve_path(root, config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    generated = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")

    deferred_gate = config.get("deferred_text_gate", {})
    if deferred_gate.get("upstream_batch_dir") and deferred_gate.get("completion_batch_dir"):
        return _evaluate_deferred_text_gate(config, root, generated)

    comparison = config.get("text_backend_comparison", {})
    if comparison.get("candidate_batches"):
        return _evaluate_text_backend_comparison(config, root, generated)

    checks: list[dict[str, Any]] = []
    baseline = config["baseline"]
    criteria = config["criteria"]
    batch_dir = _resolve_path(root, baseline["batch_dir"])
    preflight_path = _resolve_path(root, baseline["preflight_report"])

    required_files = ["batch_report.json", "results.jsonl", "model_calls.jsonl"]
    missing = [name for name in required_files if not (batch_dir / name).exists()]
    checks.append(_check("batch_files_exist", not missing, missing, [], "二期基准批次必须包含必要输出文件。"))

    batch_report: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    if not missing:
        batch_report = _read_json(batch_dir / "batch_report.json")
        results = read_jsonl(batch_dir / "results.jsonl")
        model_calls = read_jsonl(batch_dir / "model_calls.jsonl")
        _check_batch_report(batch_report, results, criteria, checks)
        _check_baseline_files(results, baseline["files"], checks)
        _check_model_calls(model_calls, criteria, checks)

    if preflight_path.exists():
        _check_preflight(_read_json(preflight_path), criteria["preflight"], checks)
    else:
        checks.append(_check("preflight_report_exists", False, str(preflight_path), "existing file", "缺少二期生产候选预检查报告。"))

    overall_status = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    return {
        "schema_version": "v1",
        "report_type": "phase2_gate",
        "phase_name": config["phase_name"],
        "generated_at": generated,
        "overall_status": overall_status,
        "batch_dir": str(batch_dir),
        "preflight_report": str(preflight_path),
        "checks": checks,
        "field_notes": _field_notes(),
    }


def _evaluate_deferred_text_gate(
    config: dict[str, Any],
    root: Path,
    generated_at: str,
) -> dict[str, Any]:
    """检查两阶段文本分析的状态转换、调用隔离和源批次追踪。"""

    gate = config["deferred_text_gate"]
    upstream_dir = _resolve_path(root, gate["upstream_batch_dir"])
    completion_dir = _resolve_path(root, gate["completion_batch_dir"])
    expected_files = gate["files"]
    expected_count = len(expected_files)
    checks: list[dict[str, Any]] = []

    required_files = ["results.jsonl", "model_calls.jsonl", "batch_report.json"]
    upstream_missing = [name for name in required_files if not (upstream_dir / name).exists()]
    completion_missing = [name for name in required_files if not (completion_dir / name).exists()]
    checks.append(_check("upstream_batch_files_exist", not upstream_missing, upstream_missing, [], "第一阶段必须保留完整输出。"))
    checks.append(_check("completion_batch_files_exist", not completion_missing, completion_missing, [], "第二阶段必须保留完整输出。"))

    if not upstream_missing and not completion_missing:
        upstream_results = read_jsonl(upstream_dir / "results.jsonl")
        upstream_calls = read_jsonl(upstream_dir / "model_calls.jsonl")
        completion_results = read_jsonl(completion_dir / "results.jsonl")
        completion_calls = read_jsonl(completion_dir / "model_calls.jsonl")
        upstream_batch_ids = {item.get("batch_id") for item in upstream_results}
        upstream_batch_id = next(iter(upstream_batch_ids)) if len(upstream_batch_ids) == 1 else None

        checks.append(_check("pending_file_count", sum(item.get("processing_status") == "pending" for item in upstream_results) == expected_count, sum(item.get("processing_status") == "pending" for item in upstream_results), expected_count, "第一阶段必须把所有基准文件写为待文本分析。"))
        upstream_text_calls = [call for call in upstream_calls if call.get("task_type") == "text_analysis"]
        checks.append(_check("upstream_text_call_count", not upstream_text_calls, len(upstream_text_calls), 0, "第一阶段不得调用文本模型。"))

        completion_non_text_calls = [call.get("call_id") for call in completion_calls if call.get("task_type") != "text_analysis"]
        checks.append(_check("completion_upstream_call_count", not completion_non_text_calls, len(completion_non_text_calls), 0, "第二阶段不得重跑 OCR、视觉理解或 ASR。"))
        successful_calls = [call for call in completion_calls if call.get("task_type") == "text_analysis" and call.get("status") == "success"]
        checks.append(_check("completion_text_call_count", len(successful_calls) == expected_count, len(successful_calls), expected_count, "第二阶段每个基准文件必须有一次成功文本调用。"))
        successful_by_file: dict[str, int] = {}
        for call in successful_calls:
            file_id = str(call.get("file_id"))
            successful_by_file[file_id] = successful_by_file.get(file_id, 0) + 1
        duplicates = sorted(file_id for file_id, count in successful_by_file.items() if count > 1)
        checks.append(_check("duplicate_successful_calls", not duplicates, duplicates, [], "同一文件不得在同一完成批次中重复成功调用。"))

        completion_by_file = {item.get("file_id"): item for item in completion_results}
        for expected in expected_files:
            file_id = expected["file_id"]
            result = completion_by_file.get(file_id)
            checks.append(_check(f"{file_id}_exists", result is not None, file_id, "present", "完成批次必须包含基准文件。"))
            if result is None:
                continue
            checks.append(_check(f"{file_id}_source_batch_id", result.get("source_batch_id") == upstream_batch_id, result.get("source_batch_id"), upstream_batch_id, "最终结果必须可追溯到第一阶段批次。"))
            checks.append(_check(f"{file_id}_topic", result.get("topic") == expected["expected_topic"], result.get("topic"), expected["expected_topic"], "两阶段执行不得导致主分类回退。"))
            checks.append(_check(f"{file_id}_secondary_topics", (result.get("secondary_topics") or []) == expected.get("expected_secondary_topics", []), result.get("secondary_topics") or [], expected.get("expected_secondary_topics", []), "两阶段执行不得导致副分类回退。"))

    overall_status = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    return {
        "schema_version": "v1",
        "report_type": "phase2_deferred_text_gate",
        "phase_name": config["phase_name"],
        "generated_at": generated_at,
        "overall_status": overall_status,
        "upstream_batch_dir": str(upstream_dir),
        "completion_batch_dir": str(completion_dir),
        "checks": checks,
        "field_notes": _field_notes(),
    }


def _evaluate_text_backend_comparison(
    config: dict[str, Any],
    root: Path,
    generated_at: str,
) -> dict[str, Any]:
    """检查同步文本后端对照证据；无合格候选时返回 warning。"""

    comparison = config["text_backend_comparison"]
    expected_files = comparison["files"]
    max_latency_ms = comparison["max_text_analysis_p95_latency_ms"]
    required_calls = comparison["first_round_calls_per_candidate"]
    max_total_cost = comparison["max_estimated_cost_cny"]
    selection_priority = comparison.get(
        "warning_selection_priority",
        ["quality", "latency", "estimated_cost"],
    )
    allowed_priorities = {"quality", "latency", "estimated_cost"}
    if (
        not isinstance(selection_priority, list)
        or not selection_priority
        or len(set(selection_priority)) != len(selection_priority)
        or set(selection_priority) != allowed_priorities
    ):
        raise ValueError(
            "warning_selection_priority 必须且只能包含 quality、latency、estimated_cost。"
        )
    checks: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, Any]] = {}
    total_cost = 0.0

    for candidate_name, candidate in comparison["candidate_batches"].items():
        batch_dir = _resolve_path(root, candidate["batch_dir"])
        required_files = ["batch_report.json", "results.jsonl", "model_calls.jsonl"]
        missing = [name for name in required_files if not (batch_dir / name).exists()]
        checks.append(
            _check(
                f"{candidate_name}_batch_files_exist",
                not missing,
                missing,
                [],
                "文本后端候选必须保留完整对照证据。",
            )
        )
        if missing:
            continue

        batch_report = _read_json(batch_dir / "batch_report.json")
        results = read_jsonl(batch_dir / "results.jsonl")
        model_calls = read_jsonl(batch_dir / "model_calls.jsonl")
        text_calls = [call for call in model_calls if call.get("task_type") == "text_analysis"]
        successful_calls = [call for call in text_calls if call.get("status") == "success"]
        call_count_ok = len(successful_calls) == required_calls
        checks.append(
            _check(
                f"{candidate_name}_successful_call_count",
                call_count_ok,
                len(successful_calls),
                required_calls,
                "首轮对照必须完成配置数量的真实文本调用。",
            )
        )

        results_by_file = {item.get("file_id"): item for item in results}
        regressions: list[str] = []
        for expected in expected_files:
            result = results_by_file.get(expected["file_id"])
            if result is None:
                regressions.append(f"{expected['file_id']}:missing")
                continue
            if result.get("processing_status") != expected["expected_processing_status"]:
                regressions.append(f"{expected['file_id']}:processing_status")
            if result.get("topic") != expected["expected_topic"]:
                regressions.append(f"{expected['file_id']}:topic")
            if (result.get("secondary_topics") or []) != expected.get("expected_secondary_topics", []):
                regressions.append(f"{expected['file_id']}:secondary_topics")

        latency_ms = (
            batch_report.get("latency_stats", {})
            .get("latency_by_task_type", {})
            .get("text_analysis", {})
            .get("p95_latency_ms")
        )
        quality_pass = not regressions
        latency_pass = isinstance(latency_ms, (int, float)) and latency_ms <= max_latency_ms
        estimated_cost = float(batch_report.get("cost_stats", {}).get("total_cost_cny", 0))
        total_cost += estimated_cost
        evaluations[candidate_name] = {
            "batch_dir": str(batch_dir),
            "quality_pass": quality_pass,
            "quality_regressions": regressions,
            "latency_pass": latency_pass,
            "text_analysis_p95_latency_ms": latency_ms,
            "successful_text_call_count": len(successful_calls),
            "estimated_cost_cny": estimated_cost,
            "selected": call_count_ok and quality_pass and latency_pass,
        }

    checks.append(
        _check(
            "max_total_estimated_cost_cny",
            total_cost <= max_total_cost,
            round(total_cost, 6),
            max_total_cost,
            "受控对照的总估算成本不得超过配置护栏。",
        )
    )
    evidence_complete = all(check["status"] == "pass" for check in checks)
    selected_candidates = [name for name, item in evaluations.items() if item["selected"]]
    overall_status = "fail" if not evidence_complete else "pass" if selected_candidates else "warning"
    recommended_candidate = None
    recommendation_status = "fail"
    unmet_constraints = ["evidence_complete"] if not evidence_complete else []
    if evidence_complete and evaluations:
        candidate_pool = selected_candidates or list(evaluations)
        recommended_candidate = min(
            candidate_pool,
            key=lambda name: _candidate_rank(name, evaluations[name], selection_priority),
        )
        recommendation_status = "pass" if recommended_candidate in selected_candidates else "warning"
        recommended = evaluations[recommended_candidate]
        unmet_constraints = [
            constraint
            for constraint, passed in (
                ("quality", recommended["quality_pass"]),
                ("p95_latency_ms", recommended["latency_pass"]),
            )
            if not passed
        ]
    return {
        "schema_version": "v1",
        "report_type": "phase2_text_backend_comparison_gate",
        "phase_name": config["phase_name"],
        "generated_at": generated_at,
        "overall_status": overall_status,
        "selected_candidates": selected_candidates,
        "recommended_candidate": recommended_candidate,
        "recommendation_status": recommendation_status,
        "unmet_constraints": unmet_constraints,
        "selection_priority": selection_priority,
        "candidate_evaluations": evaluations,
        "checks": checks,
        "field_notes": _field_notes(),
    }


def _candidate_rank(
    candidate_name: str,
    evaluation: dict[str, Any],
    selection_priority: list[str],
) -> tuple[Any, ...]:
    """按配置顺序生成确定性候选排序，不构造虚假综合分。"""

    values = {
        "quality": 0 if evaluation["quality_pass"] else 1,
        "latency": 0 if evaluation["latency_pass"] else 1,
        "estimated_cost": float(evaluation["estimated_cost_cny"]),
    }
    return tuple(values[item] for item in selection_priority) + (candidate_name,)


def _check_batch_report(
    batch_report: dict[str, Any],
    results: list[dict[str, Any]],
    criteria: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    file_stats = batch_report.get("file_stats", {})
    cost_stats = batch_report.get("cost_stats", {})
    error_stats = batch_report.get("error_quality_stats", {})

    checks.append(
        _check(
            "min_total_files",
            len(results) >= criteria["min_total_files"],
            len(results),
            criteria["min_total_files"],
            "二期基准样本数量不能低于配置要求。",
        )
    )
    checks.append(
        _check(
            "max_failed_files",
            file_stats.get("failed_files", 0) <= criteria["max_failed_files"],
            file_stats.get("failed_files", 0),
            criteria["max_failed_files"],
            "二期基准不允许出现未解释的文件级失败。",
        )
    )
    checks.append(
        _check(
            "min_success_rate",
            file_stats.get("success_rate", 0) >= criteria["min_success_rate"],
            file_stats.get("success_rate", 0),
            criteria["min_success_rate"],
            "二期基准成功率必须达到配置要求。",
        )
    )
    checks.append(
        _check(
            "max_total_errors",
            error_stats.get("total_errors", 0) <= criteria["max_total_errors"],
            error_stats.get("total_errors", 0),
            criteria["max_total_errors"],
            "二期基准不允许出现未收束错误。",
        )
    )
    checks.append(
        _check(
            "max_budget_used_rate",
            cost_stats.get("budget_used_rate", 0) <= criteria["max_budget_used_rate"],
            cost_stats.get("budget_used_rate", 0),
            criteria["max_budget_used_rate"],
            "二期基准成本不能超过预算使用率上限。",
        )
    )


def _check_baseline_files(
    results: list[dict[str, Any]],
    expected_files: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    results_by_file = {item.get("file_id"): item for item in results}
    for expected in expected_files:
        file_id = expected["file_id"]
        result = results_by_file.get(file_id)
        checks.append(_check(f"{file_id}_exists", result is not None, file_id, "present", "基准样本必须存在于 results.jsonl。"))
        if result is None:
            continue
        checks.append(
            _check(
                f"{file_id}_processing_status",
                result.get("processing_status") == expected["expected_processing_status"],
                result.get("processing_status"),
                expected["expected_processing_status"],
                "基准样本处理状态不能回退。",
            )
        )
        checks.append(
            _check(
                f"{file_id}_topic",
                result.get("topic") == expected["expected_topic"],
                result.get("topic"),
                expected["expected_topic"],
                "基准样本主分类不能回退。",
            )
        )
        if "expected_secondary_topics" in expected:
            observed = result.get("secondary_topics") or []
            checks.append(
                _check(
                    f"{file_id}_secondary_topics",
                    observed == expected["expected_secondary_topics"],
                    observed,
                    expected["expected_secondary_topics"],
                    "基准样本副分类不能回退。",
                )
            )


def _check_model_calls(model_calls: list[dict[str, Any]], criteria: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    success_task_types = {call.get("task_type") for call in model_calls if call.get("status") == "success"}
    missing_tasks = sorted(set(criteria["required_task_types"]) - success_task_types)
    checks.append(_check("required_task_types", not missing_tasks, missing_tasks, [], "基准批次必须包含必要任务类型的成功调用。"))

    if criteria.get("require_no_mock_calls"):
        mock_calls = [call.get("call_id") for call in model_calls if _is_mock_call(call)]
        checks.append(_check("require_no_mock_calls", not mock_calls, mock_calls, [], "二期基准不能依赖 mock 模型调用。"))


def _check_preflight(preflight: dict[str, Any], criteria: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    acceptable_statuses = set(criteria["acceptable_statuses"])
    checks.append(
        _check(
            "preflight_status",
            preflight.get("preflight_status") in acceptable_statuses,
            preflight.get("preflight_status"),
            sorted(acceptable_statuses),
            "二期出闸要求生产候选预检查不能存在硬失败。",
        )
    )
    checks.append(
        _check(
            "preflight_policy_name",
            preflight.get("policy_name") == criteria["expected_policy_name"],
            preflight.get("policy_name"),
            criteria["expected_policy_name"],
            "预检查必须使用二期指定策略。",
        )
    )

    route_summary = preflight.get("route_summary", {})
    checks.append(
        _check(
            "min_real_coverage_rate",
            route_summary.get("real_coverage_rate", 0) >= criteria["min_real_coverage_rate"],
            route_summary.get("real_coverage_rate", 0),
            criteria["min_real_coverage_rate"],
            "二期基准必须达到真实模型覆盖率要求。",
        )
    )
    checks.append(
        _check(
            "max_mock_coverage_rate",
            route_summary.get("mock_coverage_rate", 1) <= criteria["max_mock_coverage_rate"],
            route_summary.get("mock_coverage_rate", 1),
            criteria["max_mock_coverage_rate"],
            "二期基准不能依赖 mock 覆盖率。",
        )
    )

    stats = preflight.get("latency_profile", {}).get("task_latency_stats", {})
    for task_type, limit in criteria["max_task_p95_latency_ms"].items():
        observed = stats.get(task_type, {}).get("p95_latency_ms")
        checks.append(
            _check(
                f"{task_type}_p95_latency_ms",
                isinstance(observed, (int, float)) and observed <= limit,
                observed,
                limit,
                "任务级 P95 延迟必须低于二期出闸标准。",
            )
        )


def _is_mock_call(call: dict[str, Any]) -> bool:
    model_name = str(call.get("model_name", ""))
    provider = str(call.get("provider", ""))
    return call.get("is_mock") is True or model_name.startswith("mock-") or provider == "mock"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _check(name: str, passed: bool, observed: Any, expected: Any, reason: str) -> dict[str, Any]:
    return {
        "check_name": name,
        "status": "pass" if passed else "fail",
        "observed": observed,
        "expected": expected,
        "reason": reason,
    }


def _field_notes() -> dict[str, str]:
    return {
        "overall_status": "二期 gate 总状态；pass 表示达标，warning 表示证据完整但未选出候选，fail 表示证据或硬门槛缺失。",
        "checks": "逐项检查结果，用于定位是样本、分类、成本、真实覆盖率还是延迟未达标。",
        "selected_candidates": "同时通过质量和延迟硬门槛的同步文本后端列表。",
        "recommended_candidate": "证据完整时最终建议的文本后端；可能是全部达标候选，也可能是带约束缺口的 warning 候选。",
        "recommendation_status": "推荐状态；pass 表示候选全部达标，warning 表示存在已披露约束缺口，fail 表示证据不足不能推荐。",
        "unmet_constraints": "推荐候选没有满足的约束，用于解释 warning，避免把折中方案写成合格方案。",
        "selection_priority": "warning 候选的确定性排序顺序；不代表加权质量分数。",
        "candidate_evaluations": "各文本后端的调用数、分类回退、P95延迟和估算成本检查。",
        "preflight_status": "生产候选预检查状态，用于判断当前路线是否存在硬阻塞。",
        "latency_ms": "单次调用耗时；本 gate 使用任务级 P95 延迟判断出闸。",
        "topic": "主分类，用于检查基准样本是否发生分类回退。",
        "secondary_topics": "副分类，用于检查交叉领域判断是否发生回退。",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查二期真实链路稳定化是否达到当前出闸标准。")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "phase2_benchmark.yaml"), help="二期基准配置文件路径。")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="项目根目录。")
    parser.add_argument("--output", help="可选输出 JSON 文件路径；由程序写入 UTF-8，避免 PowerShell 重定向编码问题。")
    args = parser.parse_args(argv)

    report = evaluate_phase2_gate(args.config, project_root=args.project_root)
    output_text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        _resolve_path(Path(args.project_root), args.output).write_text(output_text + "\n", encoding="utf-8")
    print(output_text)
    return 0 if report["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
