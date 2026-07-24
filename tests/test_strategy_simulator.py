"""strategy_simulator 的离线测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from strategy_simulator import (  # noqa: E402
    build_simulation_from_files,
    render_simulation_markdown,
    simulate_routing_policies,
    write_simulation_reports,
)


def _batch_report() -> dict:
    """构造批次报告。"""

    return {
        "batch_id": "batch_test",
        "file_stats": {
            "total_files": 2,
        },
        "cost_stats": {
            "budget_limit_cny": 10.0,
            "total_cost_cny": 2.0,
        },
        "latency_stats": {
            "avg_processing_time_per_file_ms": 500,
            "p95_model_latency_ms": 2500,
        },
    }


def _model_calls() -> list[dict]:
    """构造模型调用明细。"""

    return [
        {
            "call_id": "call_001",
            "task_type": "visual_understanding",
            "provider": "qwen",
            "model_name": "mock-vision",
            "cost_cny": 1.0,
            "latency_ms": 0,
            "input_units": [{"unit_type": "frame_count", "quantity": 1}],
            "output_units": [{"unit_type": "output_tokens", "quantity": 10}],
            "status": "success",
        },
        {
            "call_id": "call_002",
            "task_type": "text_analysis",
            "provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "cost_cny": 1.0,
            "latency_ms": 2500,
            "input_units": [{"unit_type": "input_tokens", "quantity": 200}],
            "output_units": [{"unit_type": "output_tokens", "quantity": 60}],
            "status": "success",
        },
    ]


class StrategySimulatorTest(unittest.TestCase):
    def test_simulation_contains_all_policy_results(self) -> None:
        report = simulate_routing_policies(_batch_report(), _model_calls(), generated_at="2026-07-19T10:00:00+08:00")

        self.assertEqual(report["report_type"], "routing_policy_simulation")
        self.assertEqual(set(report["policy_results"]), {"budget_first", "latency_first", "quality_first", "balanced"})
        self.assertEqual(report["current_call_structure"]["real_model_calls"], 1)
        self.assertEqual(report["current_call_structure"]["mock_model_calls"], 1)

    def test_policy_results_are_not_hardcoded_text_only(self) -> None:
        report = simulate_routing_policies(
            _batch_report(),
            _model_calls(),
            constraints_override={"quality_first": {"min_real_coverage_rate": 0.9}},
        )

        quality_result = report["policy_results"]["quality_first"]
        self.assertEqual(quality_result["constraint_status"], "fail")
        self.assertEqual(quality_result["constraint_checks"][2]["observed_value"], 0.5)
        self.assertEqual(quality_result["constraint_checks"][2]["limit_value"], 0.9)

    def test_markdown_report_generation(self) -> None:
        report = simulate_routing_policies(_batch_report(), _model_calls(), generated_at="2026-07-19T10:00:00+08:00")
        markdown = render_simulation_markdown(report)

        self.assertIn("# 路由策略模拟报告", markdown)
        self.assertIn("## 3. 不同策略下的推荐方案", markdown)
        self.assertIn("预算扩展模拟", markdown)
        self.assertIn("当前数据未提供人工标注质量结果", markdown)

    def test_json_and_markdown_files_are_written(self) -> None:
        report = simulate_routing_policies(_batch_report(), _model_calls(), generated_at="2026-07-19T10:00:00+08:00")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_paths = write_simulation_reports(tmp_dir, report)
            json_path = Path(output_paths["json"])
            markdown_path = Path(output_paths["markdown"])

            saved_report = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(saved_report["report_type"], "routing_policy_simulation")
        self.assertTrue(json_path.name.endswith(".json"))
        self.assertIn("技术负责人视角", markdown)

    def test_build_simulation_from_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_dir = Path(tmp_dir)
            (batch_dir / "batch_report.json").write_text(json.dumps(_batch_report(), ensure_ascii=False), encoding="utf-8")
            (batch_dir / "model_calls.jsonl").write_text(
                "\n".join(json.dumps(call, ensure_ascii=False, indent=2) for call in _model_calls()),
                encoding="utf-8",
            )

            report = build_simulation_from_files(batch_dir, generated_at="2026-07-19T10:00:00+08:00")

        self.assertEqual(report["batch_id"], "batch_test")
        self.assertEqual(report["budget_expansion_simulation"][1]["budget_multiplier"], 5)

    def test_build_simulation_with_policy_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_dir = Path(tmp_dir)
            config_path = batch_dir / "routing_policy_config.yaml"
            (batch_dir / "batch_report.json").write_text(json.dumps(_batch_report(), ensure_ascii=False), encoding="utf-8")
            (batch_dir / "model_calls.jsonl").write_text(
                "\n".join(json.dumps(call, ensure_ascii=False, indent=2) for call in _model_calls()),
                encoding="utf-8",
            )
            config_path.write_text(
                "\n".join(
                    [
                        "schema_version: v1",
                        "policies:",
                        "  balanced:",
                        "    p95_latency_limit_ms: 3000",
                        "    min_real_coverage_rate: 0.5",
                        "budget_expansion_multipliers:",
                        "  - 3",
                        "  - 6",
                    ]
                ),
                encoding="utf-8",
            )

            report = build_simulation_from_files(
                batch_dir,
                generated_at="2026-07-19T10:00:00+08:00",
                policy_config_path=config_path,
            )

        self.assertEqual(report["policy_results"]["balanced"]["constraint_status"], "pass")
        self.assertEqual(report["policy_config"]["budget_expansion_multipliers"], [3, 6])
        self.assertIn("routing_policy_config.yaml", report["policy_config"]["config_source"])

    def test_missing_fields_are_rendered_as_missing_data(self) -> None:
        report = simulate_routing_policies({}, [], generated_at="2026-07-19T10:00:00+08:00")
        markdown = render_simulation_markdown(report)

        self.assertIn("当前数据未提供", markdown)
        self.assertEqual(report["current_call_structure"]["total_files"], "当前数据未提供")


if __name__ == "__main__":
    unittest.main()
