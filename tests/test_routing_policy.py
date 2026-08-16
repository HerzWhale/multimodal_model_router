"""routing_policy 的离线测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model_catalog import build_model_catalog, summarize_catalog  # noqa: E402
from routing_policy import (  # noqa: E402
    evaluate_routing_policy,
    get_policy_definition,
    list_policy_names,
    load_policy_config,
    normalize_policy_config,
    simulate_budget_expansion,
)


def _batch_report() -> dict:
    """构造批次报告。"""

    return {
        "batch_id": "batch_test",
        "cost_stats": {
            "budget_limit_cny": 10.0,
            "total_cost_cny": 2.0,
        },
        "latency_stats": {
            "p95_model_latency_ms": 2500,
        },
    }


def _model_calls() -> list[dict]:
    """构造模型调用明细。"""

    return [
        {
            "task_type": "ocr",
            "provider": "doubao",
            "model_name": "mock-ocr",
            "cost_cny": 1.0,
            "latency_ms": 0,
            "input_units": [{"unit_type": "image_count", "quantity": 1}],
            "output_units": [{"unit_type": "text_chars", "quantity": 10}],
            "status": "success",
        },
        {
            "task_type": "text_analysis",
            "provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "cost_cny": 1.0,
            "latency_ms": 2500,
            "input_units": [{"unit_type": "input_tokens", "quantity": 100}],
            "output_units": [{"unit_type": "output_tokens", "quantity": 50}],
            "status": "success",
        },
    ]


class RoutingPolicyTest(unittest.TestCase):
    def test_policy_names_and_definition(self) -> None:
        policy_names = list_policy_names()

        self.assertEqual(policy_names, ["budget_first", "latency_first", "quality_first", "balanced", "production_sla"])
        self.assertEqual(get_policy_definition("budget_first")["display_name"], "成本优先")
        self.assertEqual(get_policy_definition("production_sla")["display_name"], "生产 SLA 候选")

    def test_unknown_policy_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            get_policy_definition("unknown_policy")

    def test_cost_constraint_logic(self) -> None:
        catalog = build_model_catalog(_model_calls())
        result = evaluate_routing_policy(
            "budget_first",
            catalog,
            _batch_report(),
            constraints_override={"budget_limit_cny": 1.0},
        )

        budget_check = result["constraint_checks"][0]
        self.assertEqual(budget_check["constraint_name"], "budget_limit_cny")
        self.assertEqual(budget_check["status"], "fail")
        self.assertEqual(result["constraint_status"], "fail")

    def test_latency_constraint_logic(self) -> None:
        catalog = build_model_catalog(_model_calls())
        result = evaluate_routing_policy("latency_first", catalog, _batch_report())

        latency_check = result["constraint_checks"][1]
        self.assertEqual(latency_check["constraint_name"], "p95_latency_limit_ms")
        self.assertEqual(latency_check["status"], "fail")
        self.assertIn("DeepSeek", result["recommendation"]["recommended_combo"][0])

    def test_production_sla_policy_has_strict_defaults(self) -> None:
        constraints = get_policy_definition("production_sla")["default_constraints"]

        self.assertEqual(constraints["min_real_coverage_rate"], 1.0)
        self.assertEqual(constraints["task_latency_targets_ms"]["visual_understanding"], 10000)
        self.assertEqual(constraints["task_latency_targets_ms"]["text_analysis"], 8000)

    def test_mock_and_real_model_boundary(self) -> None:
        catalog = build_model_catalog(_model_calls())
        summary = summarize_catalog(catalog)

        self.assertEqual(summary["real_model_calls"], 1)
        self.assertEqual(summary["mock_model_calls"], 1)
        self.assertEqual(summary["real_coverage_rate"], 0.5)
        self.assertEqual(summary["mock_task_types"], ["ocr"])

    def test_missing_fields_are_handled(self) -> None:
        catalog = build_model_catalog([{"model_name": "mock-ocr"}])
        result = evaluate_routing_policy("balanced", catalog, {})

        self.assertTrue(result["missing_data_notes"])
        self.assertEqual(result["constraint_checks"][0]["status"], "unknown")
        self.assertEqual(result["constraint_status"], "fail")
        self.assertEqual(result["constraint_checks"][2]["status"], "fail")

    def test_budget_expansion_logic(self) -> None:
        catalog = build_model_catalog(_model_calls())
        scenarios = simulate_budget_expansion(_batch_report(), catalog)

        self.assertEqual([scenario["budget_multiplier"] for scenario in scenarios], [2, 5, 10])
        self.assertEqual(scenarios[0]["expanded_budget_cny"], 20.0)
        self.assertEqual(scenarios[0]["upgrade_priority"][0]["task_type"], "ocr")
        self.assertIn("不编造真实 OCR", scenarios[0]["assumption"])

    def test_normalize_policy_config(self) -> None:
        config = normalize_policy_config(
            {
                "schema_version": "v1",
                "policies": {
                    "balanced": {
                        "p95_latency_limit_ms": 3000,
                        "min_real_coverage_rate": 0.5,
                    }
                },
                "budget_expansion_multipliers": [3, 6],
            }
        )

        self.assertEqual(config["policy_overrides"]["balanced"]["p95_latency_limit_ms"], 3000)
        self.assertEqual(config["policy_overrides"]["balanced"]["min_real_coverage_rate"], 0.5)
        self.assertEqual(config["budget_expansion_multipliers"], (3, 6))

    def test_load_policy_config_from_yaml_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "routing_policy_config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "schema_version: v1",
                        "policies:",
                        "  latency_first:",
                        "    p95_latency_limit_ms: 4000",
                        "    min_real_coverage_rate: 0.3",
                        "budget_expansion_multipliers:",
                        "  - 2",
                        "  - 4",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_policy_config(config_path)

        self.assertEqual(config["policy_overrides"]["latency_first"]["p95_latency_limit_ms"], 4000)
        self.assertEqual(config["policy_overrides"]["latency_first"]["min_real_coverage_rate"], 0.3)
        self.assertEqual(config["budget_expansion_multipliers"], (2, 4))

    def test_invalid_policy_config_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            normalize_policy_config({"policies": {"unknown_policy": {}}})


if __name__ == "__main__":
    unittest.main()
