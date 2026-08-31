"""phase2_gate 的离线测试。"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from phase2_gate import evaluate_phase2_gate, main as cli_main  # noqa: E402


class Phase2GateTest(unittest.TestCase):
    def test_comparison_gate_rejects_invalid_warning_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_comparison_case(root)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["text_backend_comparison"]["warning_selection_priority"] = ["quality", "quality"]
            config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "warning_selection_priority"):
                evaluate_phase2_gate(config_path, project_root=root)

    def test_deferred_gate_passes_when_stages_are_isolated_and_linked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_deferred_case(root)

            report = evaluate_phase2_gate(config_path, project_root=root, generated_at="2026-08-26T00:00:00+08:00")

        self.assertEqual(report["overall_status"], "pass")

    def test_deferred_gate_rejects_duplicate_successful_text_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_deferred_case(root, duplicate_success=True)

            report = evaluate_phase2_gate(config_path, project_root=root, generated_at="2026-08-26T00:00:00+08:00")

        self.assertEqual(report["overall_status"], "fail")
        failed = {item["check_name"] for item in report["checks"] if item["status"] == "fail"}
        self.assertIn("duplicate_successful_calls", failed)

    def test_comparison_gate_warns_when_no_candidate_passes_quality_and_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_comparison_case(root)

            report = evaluate_phase2_gate(config_path, project_root=root, generated_at="2026-08-26T00:00:00+08:00")

        self.assertEqual(report["overall_status"], "warning")
        self.assertEqual(report["selected_candidates"], [])
        self.assertEqual(report["recommended_candidate"], "deepseek")
        self.assertEqual(report["recommendation_status"], "warning")
        self.assertEqual(report["unmet_constraints"], ["p95_latency_ms"])
        self.assertTrue(report["candidate_evaluations"]["deepseek"]["quality_pass"])
        self.assertFalse(report["candidate_evaluations"]["deepseek"]["latency_pass"])
        self.assertFalse(report["candidate_evaluations"]["qwen_text"]["quality_pass"])
        self.assertTrue(report["candidate_evaluations"]["qwen_text"]["latency_pass"])

    def test_comparison_gate_passes_when_candidate_meets_both_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_comparison_case(root, qwen_secondary=[])

            report = evaluate_phase2_gate(config_path, project_root=root, generated_at="2026-08-26T00:00:00+08:00")

        self.assertEqual(report["overall_status"], "pass")
        self.assertEqual(report["selected_candidates"], ["qwen_text"])
        self.assertEqual(report["recommended_candidate"], "qwen_text")
        self.assertEqual(report["recommendation_status"], "pass")
        self.assertEqual(report["unmet_constraints"], [])

    def test_comparison_gate_fails_when_candidate_evidence_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_comparison_case(root)
            (root / "output" / "qwen" / "model_calls.jsonl").unlink()

            report = evaluate_phase2_gate(config_path, project_root=root, generated_at="2026-08-26T00:00:00+08:00")

        self.assertEqual(report["overall_status"], "fail")
        self.assertIsNone(report["recommended_candidate"])
        self.assertEqual(report["recommendation_status"], "fail")
        self.assertEqual(report["unmet_constraints"], ["evidence_complete"])
        failed = {item["check_name"] for item in report["checks"] if item["status"] == "fail"}
        self.assertIn("qwen_text_batch_files_exist", failed)

    def test_gate_passes_when_batch_and_preflight_meet_criteria(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_case(root)

            report = evaluate_phase2_gate(config_path, project_root=root, generated_at="2026-08-20T00:00:00+08:00")

        self.assertEqual(report["overall_status"], "pass")
        self.assertTrue(all(check["status"] == "pass" for check in report["checks"]))

    def test_gate_fails_on_topic_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_case(root, topic="news")

            report = evaluate_phase2_gate(config_path, project_root=root, generated_at="2026-08-20T00:00:00+08:00")

        self.assertEqual(report["overall_status"], "fail")
        failed = {check["check_name"] for check in report["checks"] if check["status"] == "fail"}
        self.assertIn("file_0001_topic", failed)

    def test_gate_fails_on_text_analysis_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_case(root, text_analysis_p95=9000)

            report = evaluate_phase2_gate(config_path, project_root=root, generated_at="2026-08-20T00:00:00+08:00")

        self.assertEqual(report["overall_status"], "fail")
        failed = {check["check_name"] for check in report["checks"] if check["status"] == "fail"}
        self.assertIn("text_analysis_p95_latency_ms", failed)

    def test_cli_returns_nonzero_when_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_case(root, preflight_status="fail")
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = cli_main(["--config", str(config_path), "--project-root", str(root)])

        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["overall_status"], "fail")
        self.assertIn("preflight_status", {check["check_name"] for check in report["checks"] if check["status"] == "fail"})

    def test_cli_writes_utf8_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_case(root)
            output_path = root / "phase2_gate.json"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = cli_main(["--config", str(config_path), "--project-root", str(root), "--output", str(output_path)])

            self.assertEqual(exit_code, 0)
            saved_report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["overall_status"], "pass")


def _write_case(
    root: Path,
    *,
    topic: str = "other",
    text_analysis_p95: int = 7000,
    preflight_status: str = "warning",
) -> Path:
    batch_dir = root / "output" / "batch"
    preflight_dir = root / "output" / "preflight"
    batch_dir.mkdir(parents=True)
    preflight_dir.mkdir(parents=True)

    _write_json(
        batch_dir / "batch_report.json",
        {
            "file_stats": {"total_files": 1, "failed_files": 0, "success_rate": 1.0},
            "cost_stats": {"budget_used_rate": 0.001},
            "error_quality_stats": {"total_errors": 0},
        },
    )
    _write_jsonl(
        batch_dir / "results.jsonl",
        [
            {
                "file_id": "file_0001",
                "processing_status": "success",
                "topic": topic,
                "secondary_topics": [],
            }
        ],
    )
    _write_jsonl(
        batch_dir / "model_calls.jsonl",
        [
            {
                "call_id": "call_0001",
                "file_id": "file_0001",
                "task_type": "text_analysis",
                "provider": "deepseek",
                "model_name": "deepseek-v4-flash",
                "status": "success",
            }
        ],
    )
    _write_json(
        preflight_dir / "routing_preflight_report.json",
        {
            "policy_name": "production_sla",
            "preflight_status": preflight_status,
            "route_summary": {"real_coverage_rate": 1.0, "mock_coverage_rate": 0.0},
            "latency_profile": {
                "task_latency_stats": {
                    "text_analysis": {"p95_latency_ms": text_analysis_p95},
                }
            },
        },
    )
    config = {
        "schema_version": "v1",
        "phase_name": "phase2_test",
        "baseline": {
            "batch_dir": "output/batch",
            "preflight_report": "output/preflight/routing_preflight_report.json",
            "files": [
                {
                    "file_id": "file_0001",
                    "expected_processing_status": "success",
                    "expected_topic": "other",
                    "expected_secondary_topics": [],
                }
            ],
        },
        "criteria": {
            "min_total_files": 1,
            "max_failed_files": 0,
            "min_success_rate": 1.0,
            "max_total_errors": 0,
            "max_budget_used_rate": 0.01,
            "require_no_mock_calls": True,
            "required_task_types": ["text_analysis"],
            "preflight": {
                "expected_policy_name": "production_sla",
                "acceptable_statuses": ["pass", "warning"],
                "min_real_coverage_rate": 1.0,
                "max_mock_coverage_rate": 0.0,
                "max_task_p95_latency_ms": {"text_analysis": 8000},
            },
        },
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path


def _write_comparison_case(root: Path, *, qwen_secondary: list[str] | None = None) -> Path:
    for name, provider, latency, secondary in [
        ("deepseek", "deepseek", 10000, []),
        ("qwen", "qwen", 6000, ["gaming"] if qwen_secondary is None else qwen_secondary),
    ]:
        batch_dir = root / "output" / name
        batch_dir.mkdir(parents=True)
        _write_json(
            batch_dir / "batch_report.json",
            {
                "cost_stats": {"total_cost_cny": 0.01},
                "latency_stats": {"latency_by_task_type": {"text_analysis": {"p95_latency_ms": latency}}},
            },
        )
        _write_jsonl(
            batch_dir / "results.jsonl",
            [{"file_id": "file_0001", "processing_status": "success", "topic": "technology", "secondary_topics": secondary}],
        )
        _write_jsonl(
            batch_dir / "model_calls.jsonl",
            [{"call_id": f"{name}_call", "task_type": "text_analysis", "provider": provider, "status": "success"}],
        )
    config = {
        "schema_version": "v1",
        "phase_name": "phase2_comparison_test",
        "text_backend_comparison": {
            "candidate_batches": {
                "deepseek": {"batch_dir": "output/deepseek"},
                "qwen_text": {"batch_dir": "output/qwen"},
            },
            "files": [{
                "file_id": "file_0001",
                "expected_processing_status": "success",
                "expected_topic": "technology",
                "expected_secondary_topics": [],
            }],
            "first_round_calls_per_candidate": 1,
            "max_estimated_cost_cny": 0.1,
            "max_text_analysis_p95_latency_ms": 8000,
            "warning_selection_priority": ["quality", "latency", "estimated_cost"],
        },
    }
    config_path = root / "comparison.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path


def _write_deferred_case(root: Path, *, duplicate_success: bool = False) -> Path:
    upstream_dir = root / "output" / "upstream"
    completion_dir = root / "output" / "completion"
    upstream_dir.mkdir(parents=True)
    completion_dir.mkdir(parents=True)
    _write_jsonl(upstream_dir / "results.jsonl", [{"batch_id": "batch_upstream", "file_id": "file_0001", "processing_status": "pending"}])
    _write_jsonl(upstream_dir / "model_calls.jsonl", [{"call_id": "ocr_1", "file_id": "file_0001", "task_type": "ocr", "status": "success"}])
    _write_json(upstream_dir / "batch_report.json", {"file_stats": {"total_files": 1, "pending_files": 1}})
    _write_jsonl(completion_dir / "results.jsonl", [{"batch_id": "batch_completion", "source_batch_id": "batch_upstream", "file_id": "file_0001", "processing_status": "success", "topic": "other", "secondary_topics": []}])
    calls = [{"call_id": "text_1", "file_id": "file_0001", "task_type": "text_analysis", "status": "success"}]
    if duplicate_success:
        calls.append({"call_id": "text_2", "file_id": "file_0001", "task_type": "text_analysis", "status": "success"})
    _write_jsonl(completion_dir / "model_calls.jsonl", calls)
    _write_json(completion_dir / "batch_report.json", {"file_stats": {"total_files": 1, "success_files": 1}})
    config = {
        "schema_version": "v1",
        "phase_name": "phase2_deferred_test",
        "deferred_text_gate": {
            "upstream_batch_dir": "output/upstream",
            "completion_batch_dir": "output/completion",
            "files": [{"file_id": "file_0001", "expected_topic": "other", "expected_secondary_topics": []}],
        },
    }
    config_path = root / "deferred.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False, indent=2) + "\n" for record in records), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
