"""model_router 的测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model_router import (
    build_route_plan,
    load_routing_rules,
    route_plan_backends_for_media,
    routing_rules_for_media,
    select_model,
    validate_route_plan,
)


class ModelRouterTest(unittest.TestCase):
    def _settings(self) -> dict:
        return {
            "pipelines": {
                "text": {"text_analysis": "mock"},
                "image": {"ocr": "mock", "vision_understanding": "mock", "text_analysis": "mock"},
                "video": {
                    "keyframe_ocr": "mock",
                    "keyframe_vision_understanding": "mock",
                    "speech_to_text": "mock",
                    "text_analysis": "mock",
                },
            },
            "backends": {
                "ocr": {"mock": {"provider": "local", "model_name": "mock-ocr"}},
                "vision_understanding": {"mock": {"provider": "local", "model_name": "mock-vision"}},
                "speech_to_text": {"mock": {"provider": "local", "model_name": "mock-asr"}},
                "text_analysis": {
                    "mock": {"provider": "local", "model_name": "mock-text"},
                    "deepseek": {"provider": "deepseek", "model_name": "deepseek-v4-flash"},
                },
            },
        }

    def test_load_routing_rules(self) -> None:
        config_path = PROJECT_ROOT / "config" / "routing_rules.yaml"

        routing_rules = load_routing_rules(config_path)

        self.assertEqual(routing_rules["ocr"]["provider"], "doubao")
        self.assertEqual(routing_rules["text_analysis"]["model_name"], "mock-text")

    def test_select_model(self) -> None:
        routing_rules = {
            "ocr": {
                "provider": "doubao",
                "model_name": "mock-ocr",
            }
        }

        selected = select_model("ocr", routing_rules)

        self.assertEqual(selected["provider"], "doubao")
        self.assertEqual(selected["model_name"], "mock-ocr")

    def test_select_model_rejects_unknown_task_type(self) -> None:
        with self.assertRaises(KeyError):
            select_model("unknown_task", {})

    def test_build_route_plan_resolves_media_pipelines_and_models(self) -> None:
        plan = build_route_plan(
            self._settings(),
            preflight_status="warning",
            policy_name="balanced",
            source_settings="config/settings.yaml",
            generated_at="2026-08-30T12:00:00+08:00",
        )

        self.assertFalse(plan["requires_live_api"])
        self.assertEqual(plan["resolved_backends"]["ocr.mock"]["model_name"], "mock-ocr")
        self.assertEqual(route_plan_backends_for_media(plan, "video")["speech_to_text_backend"], "mock")
        self.assertEqual(routing_rules_for_media(plan, "image")["text_analysis"]["provider"], "local")
        validate_route_plan(plan, self._settings())

    def test_build_route_plan_marks_live_api_override(self) -> None:
        decision = {
            "task_type": "text_analysis",
            "recommended_candidate": "deepseek",
            "recommendation_status": "warning",
            "unmet_constraints": ["p95_latency_ms"],
            "non_compared_tasks": ["ocr", "vision_understanding", "speech_to_text"],
            "evidence_source": "output/route_decision.json",
            "candidate_summary": {"quality_pass": True, "latency_pass": False},
        }
        plan = build_route_plan(
            self._settings(),
            preflight_status="warning",
            policy_name="balanced",
            source_settings="config/settings.yaml",
            text_analysis_backend="deepseek",
            selection_decisions=[decision],
        )

        self.assertTrue(plan["requires_live_api"])
        self.assertEqual(plan["selected_pipelines"]["video"]["text_analysis"], "deepseek")
        self.assertEqual(plan["selection_decisions"], [decision])
        validate_route_plan(plan, self._settings())

    def test_build_route_plan_marks_qwen_ocr_as_live_api(self) -> None:
        settings = self._settings()
        settings["backends"]["ocr"]["qwen_ocr"] = {
            "provider": "qwen",
            "model_name": "qwen3.5-ocr",
            "runtime_type": "live_api",
        }
        plan = build_route_plan(
            settings,
            preflight_status="warning",
            policy_name="balanced",
            source_settings="config/settings.yaml",
            ocr_backend="qwen_ocr",
        )

        self.assertTrue(plan["requires_live_api"])
        self.assertEqual(plan["selected_pipelines"]["image"]["ocr"], "qwen_ocr")
        self.assertEqual(plan["resolved_backends"]["ocr.qwen_ocr"]["model_name"], "qwen3.5-ocr")
        validate_route_plan(plan, settings)

    def test_validate_route_plan_rejects_decision_pipeline_mismatch(self) -> None:
        settings = self._settings()
        decision = {
            "task_type": "text_analysis",
            "recommended_candidate": "deepseek",
            "recommendation_status": "warning",
            "unmet_constraints": ["p95_latency_ms"],
        }
        plan = build_route_plan(
            settings,
            preflight_status="warning",
            policy_name="balanced",
            source_settings="config/settings.yaml",
            selection_decisions=[decision],
        )

        with self.assertRaisesRegex(ValueError, "pipeline 不一致"):
            validate_route_plan(plan, settings)

    def test_validate_route_plan_rejects_warning_decision_disguised_as_pass(self) -> None:
        settings = self._settings()
        plan = build_route_plan(
            settings,
            preflight_status="pass",
            policy_name="balanced",
            source_settings="config/settings.yaml",
            text_analysis_backend="deepseek",
            selection_decisions=[{
                "task_type": "text_analysis",
                "recommended_candidate": "deepseek",
                "recommendation_status": "warning",
                "unmet_constraints": ["p95_latency_ms"],
            }],
        )

        with self.assertRaisesRegex(ValueError, "不能写入 pass"):
            validate_route_plan(plan, settings)

    def test_validate_route_plan_rejects_fail_and_config_drift(self) -> None:
        settings = self._settings()
        failed_plan = build_route_plan(
            settings,
            preflight_status="fail",
            policy_name="balanced",
            source_settings="config/settings.yaml",
        )
        with self.assertRaisesRegex(ValueError, "fail"):
            validate_route_plan(failed_plan, settings)

        warning_plan = build_route_plan(
            settings,
            preflight_status="warning",
            policy_name="balanced",
            source_settings="config/settings.yaml",
        )
        settings["backends"]["ocr"]["mock"]["model_name"] = "changed-model"
        with self.assertRaisesRegex(ValueError, "配置漂移"):
            validate_route_plan(warning_plan, settings)


if __name__ == "__main__":
    unittest.main()
