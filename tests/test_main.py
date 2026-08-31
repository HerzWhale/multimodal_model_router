"""main 的测试。"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from main import main as cli_main
from main import _build_backend_runtime_summary
from main import _pipeline_selection
from main import run_batch
from main import run_preflight
from model_router import build_route_plan


def _read_json_objects(file_path: Path) -> list[dict]:
    """兼容读取标准 JSONL 和历史缩进式连续 JSON 对象。"""

    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    decoder = json.JSONDecoder()
    records: list[dict] = []
    index = 0
    while index < len(content):
        while index < len(content) and content[index].isspace():
            index += 1
        if index >= len(content):
            break
        record, index = decoder.raw_decode(content, index)
        if isinstance(record, dict):
            records.append(record)
    return records


class MainTest(unittest.TestCase):
    def _write_nested_mock_settings(self, root: Path) -> Path:
        """写入可生成和执行路由计划的最小双层配置。"""

        config_dir = root / "config"
        input_dir = root / "input"
        config_dir.mkdir()
        input_dir.mkdir()
        (input_dir / "demo.txt").write_text("这是一段 AI 工具教程", encoding="utf-8")
        settings = yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
        settings["paths"] = {"input_dir": "input", "output_dir": "output"}
        settings_path = config_dir / "settings.yaml"
        settings_path.write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return settings_path

    def _write_route_decision(
        self,
        root: Path,
        *,
        candidate: str = "deepseek",
        status: str = "warning",
    ) -> Path:
        path = root / "route_decision.json"
        report = {
            "report_type": "phase2_text_backend_comparison_gate",
            "generated_at": "2026-08-31T10:00:00+08:00",
            "overall_status": status,
            "selected_candidates": [candidate] if status == "pass" else [],
            "recommended_candidate": candidate,
            "recommendation_status": status,
            "unmet_constraints": [] if status == "pass" else ["p95_latency_ms"],
            "candidate_evaluations": {
                candidate: {
                    "quality_pass": True,
                    "latency_pass": status == "pass",
                    "text_analysis_p95_latency_ms": 10702,
                    "successful_text_call_count": 3,
                    "estimated_cost_cny": 0.010127,
                }
            },
        }
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _write_settings(
        self,
        root: Path,
        *,
        backend: str,
        ocr_backend: str = "mock",
        vision_backend: str = "mock",
    ) -> Path:
        """写入主入口测试所需的最小配置。"""

        config_dir = root / "config"
        input_dir = root / "input"
        config_dir.mkdir()
        input_dir.mkdir()
        (input_dir / "demo.txt").write_text("这是一段 AI 工具教程", encoding="utf-8")
        (config_dir / "routing_rules.yaml").write_text(
            (PROJECT_ROOT / "config" / "routing_rules.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (config_dir / "model_prices.yaml").write_text(
            (PROJECT_ROOT / "config" / "model_prices.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        settings_path = config_dir / "settings.yaml"
        settings_path.write_text(
            "\n".join(
                    [
                        "input_dir: input",
                        "output_dir: output",
                        f"ocr_backend: {ocr_backend}",
                        f"vision_understanding_backend: {vision_backend}",
                        f"text_analysis_backend: {backend}",
                        "deepseek_api_key_env: TEST_DEEPSEEK_API_KEY",
                        "deepseek_max_tokens: 1500",
                        "deepseek_compact_mode: false",
                        "qwen_vl_api_key_env: TEST_DASHSCOPE_API_KEY",
                        "qwen_vl_max_tokens: 500",
                        "qwen_vl_max_image_side: 960",
                        "default_budget_limit_cny: 50",
                        "target_output_format: jsonl",
                    "allow_partial_success: true",
                ]
            ),
            encoding="utf-8",
        )
        return settings_path

    def test_run_batch_writes_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config"
            input_dir = root / "input"
            config_dir.mkdir()
            input_dir.mkdir()
            (input_dir / "demo.txt").write_text("这是一段 AI 工具教程", encoding="utf-8")
            settings_path = config_dir / "settings.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        "input_dir: input",
                        "output_dir: output",
                        "use_mock_models: true",
                        "default_budget_limit_cny: 50",
                        "target_output_format: jsonl",
                        "allow_partial_success: true",
                    ]
                ),
                encoding="utf-8",
            )

            summary = run_batch(
                settings_path=settings_path,
                routing_rules_path=PROJECT_ROOT / "config" / "routing_rules.yaml",
                model_prices_path=PROJECT_ROOT / "config" / "model_prices.yaml",
                batch_id="batch_test",
                created_at="2026-07-14T10:00:00+08:00",
                generated_at="2026-07-14T10:01:00+08:00",
            )

            batch_dir = Path(summary["batch_dir"])

            self.assertEqual(summary["total_files"], 1)
            self.assertTrue((batch_dir / "batch_metadata.json").exists())
            self.assertTrue((batch_dir / "results.jsonl").exists())
            self.assertTrue((batch_dir / "results_readable.md").exists())
            self.assertTrue((batch_dir / "model_calls.jsonl").exists())
            self.assertTrue((batch_dir / "errors.jsonl").exists())
            self.assertTrue((batch_dir / "batch_report.json").exists())

            for file_name in ["results.jsonl", "model_calls.jsonl", "errors.jsonl"]:
                records = _read_json_objects(batch_dir / file_name)
                self.assertIsInstance(records, list)

            report = json.loads((batch_dir / "batch_report.json").read_text(encoding="utf-8"))
            metadata = json.loads((batch_dir / "batch_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(report["file_stats"]["total_files"], 1)
            self.assertEqual(report["batch_id"], "batch_test")
            self.assertEqual(metadata["selected_backends"]["ocr_backend"], "mock")
            self.assertEqual(metadata["selected_backends"]["vision_understanding_backend"], "mock")
            self.assertEqual(metadata["selected_backends"]["speech_to_text_backend"], "mock")
            self.assertEqual(metadata["selected_backends"]["text_analysis_backend"], "mock")
            self.assertEqual(metadata["video_max_keyframes"], 3)
            self.assertEqual(metadata["request_purpose"], "受控mock批处理验证")
            self.assertTrue(metadata["backend_runtime_summary"]["contains_mock"])
            self.assertFalse(metadata["backend_runtime_summary"]["contains_live_api"])
            self.assertTrue(metadata["cost_estimation"]["contains_mock_estimates"])
            self.assertEqual(
                metadata["cost_estimation"]["estimation_error_status"],
                "unknown_until_bill_reconciliation",
            )

    def test_run_batch_deferred_text_analysis_does_not_require_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_settings(root, backend="deepseek")

            summary = run_batch(
                settings_path=settings_path,
                defer_text_analysis=True,
                batch_id="batch_deferred",
            )

            batch_dir = Path(summary["batch_dir"])
            results = _read_json_objects(batch_dir / "results.jsonl")
            model_calls = _read_json_objects(batch_dir / "model_calls.jsonl")
            metadata = json.loads((batch_dir / "batch_metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(results[0]["processing_status"], "pending")
        self.assertEqual(model_calls, [])
        self.assertEqual(metadata["text_analysis_execution_mode"], "deferred")

    def test_cli_input_dir_override_keeps_evaluation_separate_from_default_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config"
            default_input_dir = root / "input"
            evaluation_input_dir = root / "evaluation" / "text_topic_small_set"
            config_dir.mkdir()
            default_input_dir.mkdir()
            evaluation_input_dir.mkdir(parents=True)
            (config_dir / "routing_rules.yaml").write_text(
                (PROJECT_ROOT / "config" / "routing_rules.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (config_dir / "model_prices.yaml").write_text(
                (PROJECT_ROOT / "config" / "model_prices.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            (default_input_dir / "default_input.txt").write_text("默认业务输入，不应进入本次评估批次。", encoding="utf-8")
            (evaluation_input_dir / "eval_01.txt").write_text("标题：AI 工具评测\n口播：这是一条评估样本。", encoding="utf-8")
            (evaluation_input_dir / "eval_02.txt").write_text("标题：城市交通新闻\n字幕：用于评估分类边界。", encoding="utf-8")

            settings_path = config_dir / "settings.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        "input_dir: input",
                        "output_dir: output",
                        "use_mock_models: true",
                        "text_analysis_backend: deepseek",
                        "default_budget_limit_cny: 50",
                        "target_output_format: jsonl",
                        "allow_partial_success: true",
                    ]
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = cli_main(
                    [
                        "--settings",
                        str(settings_path),
                        "--input-dir",
                        str(evaluation_input_dir),
                        "--text-analysis-backend",
                        "mock",
                        "--batch-id",
                        "batch_eval_cli",
                    ]
                )

            batch_dir = root / "output" / "batch_eval_cli"
            results = _read_json_objects(batch_dir / "results.jsonl")
            file_names = {record["file_name"] for record in results}

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(results), 2)
            self.assertEqual(file_names, {"eval_01.txt", "eval_02.txt"})
            self.assertNotIn("default_input.txt", file_names)

    def test_run_batch_include_file_names_only_processes_selected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_settings(root, backend="mock")
            input_dir = root / "input"
            (input_dir / "skip.txt").write_text("这条不应进入本次批次。", encoding="utf-8")
            (input_dir / "target.txt").write_text("这条是本次指定处理文件。", encoding="utf-8")

            summary = run_batch(
                settings_path=settings_path,
                batch_id="batch_include_files",
                include_file_names=["target.txt"],
            )

            batch_dir = Path(summary["batch_dir"])
            results = _read_json_objects(batch_dir / "results.jsonl")
            file_names = {record["file_name"] for record in results}

            self.assertEqual(summary["total_files"], 1)
            self.assertEqual(file_names, {"target.txt"})

    def test_run_batch_include_file_names_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(Path(tmp_dir), backend="mock")

            with self.assertRaisesRegex(ValueError, "指定文件未在输入目录中找到"):
                run_batch(
                    settings_path=settings_path,
                    batch_id="batch_missing_include_file",
                    include_file_names=["missing.txt"],
                )

    def test_run_preflight_writes_report_without_running_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(Path(tmp_dir), backend="mock")

            summary = run_preflight(
                settings_path=settings_path,
                batch_id="preflight_test",
                include_file_names=["demo.txt"],
            )

            report_dir = Path(tmp_dir) / "output" / "preflight_test"
            self.assertIn(summary["preflight_status"], {"pass", "warning", "fail"})
            self.assertTrue((report_dir / "routing_preflight_report.json").exists())
            self.assertTrue((report_dir / "routing_preflight_report.md").exists())
            self.assertFalse((report_dir / "results.jsonl").exists())

    def test_nested_preflight_writes_executable_route_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_nested_mock_settings(root)
            summary = run_preflight(
                settings_path=settings_path,
                model_prices_path=PROJECT_ROOT / "config" / "model_prices.yaml",
                policy_config_path=PROJECT_ROOT / "config" / "routing_policy_config.yaml",
                batch_id="preflight_route_plan",
                include_file_names=["demo.txt"],
                generated_at="2026-08-30T12:00:00+08:00",
            )

            route_plan_path = Path(summary["report_paths"]["route_plan"])
            route_plan = json.loads(route_plan_path.read_text(encoding="utf-8"))

        self.assertTrue(route_plan_path.name == "route_plan.json")
        self.assertEqual(route_plan["preflight_status"], summary["preflight_status"])
        self.assertEqual(route_plan["selected_pipelines"]["text"]["text_analysis"], "mock")
        self.assertFalse(route_plan["requires_live_api"])

    def test_nested_preflight_applies_compact_route_decision_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_nested_mock_settings(root)
            decision_path = self._write_route_decision(root)
            summary = run_preflight(
                settings_path=settings_path,
                model_prices_path=PROJECT_ROOT / "config" / "model_prices.yaml",
                policy_config_path=PROJECT_ROOT / "config" / "routing_policy_config.yaml",
                batch_id="preflight_route_decision",
                include_file_names=["demo.txt"],
                route_decision_report_path=decision_path,
            )
            route_plan = json.loads(
                Path(summary["report_paths"]["route_plan"]).read_text(encoding="utf-8")
            )

        self.assertEqual(route_plan["preflight_status"], "warning")
        self.assertEqual(route_plan["selected_pipelines"]["text"]["text_analysis"], "deepseek")
        self.assertTrue(route_plan["requires_live_api"])
        decision = route_plan["selection_decisions"][0]
        self.assertEqual(decision["recommended_candidate"], "deepseek")
        self.assertEqual(decision["unmet_constraints"], ["p95_latency_ms"])
        self.assertEqual(
            decision["non_compared_tasks"],
            ["ocr", "speech_to_text", "vision_understanding"],
        )
        self.assertNotIn("candidate_evaluations", decision)

    def test_nested_preflight_rejects_unknown_route_decision_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_nested_mock_settings(root)
            decision_path = self._write_route_decision(root, candidate="missing_backend")

            with self.assertRaisesRegex(ValueError, "不存在的文本后端"):
                run_preflight(
                    settings_path=settings_path,
                    route_decision_report_path=decision_path,
                    batch_id="preflight_unknown_route_decision",
                    include_file_names=["demo.txt"],
                )

    def test_nested_preflight_rejects_legacy_routing_rules_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_nested_mock_settings(Path(tmp_dir))
            with self.assertRaisesRegex(ValueError, "唯一事实来源"):
                run_preflight(
                    settings_path=settings_path,
                    routing_rules_path=PROJECT_ROOT / "config" / "routing_rules.yaml",
                    model_prices_path=PROJECT_ROOT / "config" / "model_prices.yaml",
                    policy_config_path=PROJECT_ROOT / "config" / "routing_policy_config.yaml",
                    batch_id="preflight_legacy_source_rejected",
                    include_file_names=["demo.txt"],
                )

    def test_run_batch_consumes_warning_route_plan_through_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_nested_mock_settings(root)
            settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
            route_plan = build_route_plan(
                settings,
                preflight_status="warning",
                policy_name="balanced",
                source_settings=str(settings_path),
                generated_at="2026-08-30T12:00:00+08:00",
            )
            route_plan_path = root / "route_plan.json"
            route_plan_path.write_text(json.dumps(route_plan, ensure_ascii=False, indent=2), encoding="utf-8")

            summary = run_batch(
                settings_path=settings_path,
                model_prices_path=PROJECT_ROOT / "config" / "model_prices.yaml",
                route_plan_path=route_plan_path,
                batch_id="batch_route_plan",
                created_at="2026-08-30T12:01:00+08:00",
                generated_at="2026-08-30T12:02:00+08:00",
            )
            batch_dir = Path(summary["batch_dir"])
            metadata = json.loads((batch_dir / "batch_metadata.json").read_text(encoding="utf-8"))
            calls = _read_json_objects(batch_dir / "model_calls.jsonl")

        self.assertEqual(metadata["route_plan"]["preflight_status"], "warning")
        self.assertEqual(metadata["selected_pipelines"]["text"]["text_analysis_backend"], "mock")
        self.assertEqual(calls[0]["provider"], "local")
        self.assertEqual(calls[0]["model_name"], "mock-text")

    def test_run_batch_rejects_failed_or_drifted_route_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_nested_mock_settings(root)
            settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
            route_plan = build_route_plan(
                settings,
                preflight_status="fail",
                policy_name="balanced",
                source_settings=str(settings_path),
            )
            route_plan_path = root / "route_plan.json"
            route_plan_path.write_text(json.dumps(route_plan, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fail"):
                run_batch(
                    settings_path=settings_path,
                    model_prices_path=PROJECT_ROOT / "config" / "model_prices.yaml",
                    route_plan_path=route_plan_path,
                    batch_id="batch_failed_plan",
                )

            route_plan["preflight_status"] = "warning"
            route_plan_path.write_text(json.dumps(route_plan, ensure_ascii=False), encoding="utf-8")
            settings["pipelines"]["text"]["text_analysis"] = "deepseek"
            settings_path.write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "配置漂移"):
                run_batch(
                    settings_path=settings_path,
                    model_prices_path=PROJECT_ROOT / "config" / "model_prices.yaml",
                    route_plan_path=route_plan_path,
                    batch_id="batch_drifted_plan",
                )

    def test_run_batch_rejects_route_plan_with_backend_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_nested_mock_settings(root)
            settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
            route_plan = build_route_plan(
                settings,
                preflight_status="warning",
                policy_name="balanced",
                source_settings=str(settings_path),
            )
            route_plan_path = root / "route_plan.json"
            route_plan_path.write_text(json.dumps(route_plan, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "不能同时指定后端覆盖参数"):
                run_batch(
                    settings_path=settings_path,
                    model_prices_path=PROJECT_ROOT / "config" / "model_prices.yaml",
                    route_plan_path=route_plan_path,
                    ocr_backend_override="mock",
                    batch_id="batch_conflicting_plan",
                )

    def test_live_route_plan_still_requires_explicit_api_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_nested_mock_settings(root)
            settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
            route_plan = build_route_plan(
                settings,
                preflight_status="warning",
                policy_name="balanced",
                source_settings=str(settings_path),
                text_analysis_backend="deepseek",
            )
            route_plan_path = root / "route_plan.json"
            route_plan_path.write_text(json.dumps(route_plan, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(PermissionError, "--allow-live-api"):
                run_batch(
                    settings_path=settings_path,
                    model_prices_path=PROJECT_ROOT / "config" / "model_prices.yaml",
                    route_plan_path=route_plan_path,
                    batch_id="batch_live_plan_blocked",
                )

    def test_repository_default_ocr_backend_falls_back_to_paddle_after_qwen_gate_failure(self) -> None:
        settings = yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))

        self.assertEqual(settings["pipelines"]["image"]["ocr"], "paddleocr")
        self.assertEqual(settings["pipelines"]["video"]["keyframe_ocr"], "paddleocr")
        self.assertEqual(settings["pipelines"]["image"]["vision_understanding"], "mock")
        self.assertEqual(settings["pipelines"]["text"]["text_analysis"], "mock")
        self.assertEqual(settings["pipelines"]["video"]["speech_to_text"], "mock")

    def test_run_batch_rejects_qwen_ocr_without_live_permission(self) -> None:
        with self.assertRaisesRegex(PermissionError, "Qwen3.5-OCR"):
            run_batch(
                settings_path=PROJECT_ROOT / "config" / "settings.yaml",
                input_dir_override=PROJECT_ROOT / "input",
                include_file_names=["img_1.png"],
                ocr_backend_override="qwen_ocr",
                vision_understanding_backend_override="mock",
                speech_to_text_backend_override="mock",
                text_analysis_backend_override="mock",
                batch_id="batch_qwen_ocr_blocked",
            )

    @patch.dict(os.environ, {}, clear=True)
    def test_qwen_ocr_with_permission_but_without_key_stops_before_output(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "DASHSCOPE_API_KEY"):
            run_batch(
                settings_path=PROJECT_ROOT / "config" / "settings.yaml",
                input_dir_override=PROJECT_ROOT / "input",
                include_file_names=["img_1.png"],
                ocr_backend_override="qwen_ocr",
                vision_understanding_backend_override="mock",
                speech_to_text_backend_override="mock",
                text_analysis_backend_override="mock",
                allow_live_api=True,
                batch_id="batch_qwen_ocr_missing_key",
            )
        self.assertFalse((PROJECT_ROOT / "output" / "batch_qwen_ocr_missing_key").exists())

    def test_pipeline_selection_distinguishes_media_chains(self) -> None:
        settings = {
            "pipelines": {
                "text": {"text_analysis": "deepseek"},
                "image": {"ocr": "paddleocr", "vision_understanding": "qwen_vl", "text_analysis": "deepseek"},
                "video": {
                    "keyframe_ocr": "mock",
                    "keyframe_vision_understanding": "qwen_vl",
                    "speech_to_text": "dashscope_asr",
                    "text_analysis": "qwen_text",
                },
            }
        }

        self.assertEqual(_pipeline_selection(settings, "text")["vision_understanding_backend"], "mock")
        self.assertEqual(_pipeline_selection(settings, "image")["ocr_backend"], "paddleocr")
        self.assertEqual(
            _pipeline_selection(settings, "video"),
            {
                "ocr_backend": "mock",
                "vision_understanding_backend": "qwen_vl",
                "speech_to_text_backend": "dashscope_asr",
                "text_analysis_backend": "qwen_text",
            },
        )

    @patch("main.run_file_pipeline")
    def test_run_batch_reads_nested_settings(self, mock_pipeline) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config"
            input_dir = root / "input"
            config_dir.mkdir()
            input_dir.mkdir()
            (input_dir / "demo.txt").write_text("这是一段 AI 工具教程", encoding="utf-8")
            (input_dir / "demo.png").write_bytes(b"not-decoded-because-pipeline-is-mocked")
            (config_dir / "routing_rules.yaml").write_text(
                (PROJECT_ROOT / "config" / "routing_rules.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (config_dir / "model_prices.yaml").write_text(
                (PROJECT_ROOT / "config" / "model_prices.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            settings_path = config_dir / "settings.yaml"
            settings_path.write_text(
                """
paths:
  input_dir: input
  output_dir: output
runtime:
  default_budget_limit_cny: 50
  target_output_format: jsonl
  allow_partial_success: true
pipelines:
  text:
    text_analysis: deepseek
  image:
    ocr: mock
    vision_understanding: qwen_vl
  video:
    speech_to_text: mock
backends:
  text_analysis:
    deepseek:
      api_key_env: TEST_DEEPSEEK_API_KEY
      base_url: https://deepseek.test
      model_name: deepseek-test
      max_tokens: 2200
      compact_mode: true
      evidence_char_limit: 1234
  vision_understanding:
    qwen_vl:
      api_key_env: TEST_DASHSCOPE_API_KEY
      base_url: https://qwen.test
      model_name: qwen-vl-test
      max_tokens: 333
      max_image_side: 444
""",
                encoding="utf-8",
            )
            mock_pipeline.return_value = {
                "result": {
                    "schema_version": "v1",
                    "batch_id": "batch_nested",
                    "file_id": "file_001",
                    "file_name": "demo.txt",
                    "media_type": "text",
                    "processing_status": "success",
                    "processing_cost_cny": 0,
                    "processing_time_ms": 0,
                    "quality_flags": [],
                    "warning_messages": [],
                    "models_used": [],
                },
                "model_calls": [],
                "errors": [],
            }
            old_deepseek = os.environ.get("TEST_DEEPSEEK_API_KEY")
            old_dashscope = os.environ.get("TEST_DASHSCOPE_API_KEY")
            os.environ["TEST_DEEPSEEK_API_KEY"] = "test-deepseek"
            os.environ["TEST_DASHSCOPE_API_KEY"] = "test-dashscope"
            try:
                run_batch(settings_path=settings_path, allow_live_api=True, batch_id="batch_nested")
            finally:
                if old_deepseek is None:
                    os.environ.pop("TEST_DEEPSEEK_API_KEY", None)
                else:
                    os.environ["TEST_DEEPSEEK_API_KEY"] = old_deepseek
                if old_dashscope is None:
                    os.environ.pop("TEST_DASHSCOPE_API_KEY", None)
                else:
                    os.environ["TEST_DASHSCOPE_API_KEY"] = old_dashscope

        calls_by_media = {call.args[0]["media_type"]: call.kwargs for call in mock_pipeline.call_args_list}
        text_kwargs = calls_by_media["text"]
        image_kwargs = calls_by_media["image"]
        self.assertEqual(text_kwargs["text_analysis_backend"], "deepseek")
        self.assertEqual(text_kwargs["vision_understanding_backend"], "mock")
        self.assertEqual(image_kwargs["vision_understanding_backend"], "qwen_vl")
        self.assertEqual(text_kwargs["deepseek_model_name"], "deepseek-test")
        self.assertEqual(text_kwargs["deepseek_base_url"], "https://deepseek.test")
        self.assertEqual(text_kwargs["deepseek_max_tokens"], 2200)
        self.assertEqual(text_kwargs["text_analysis_evidence_char_limit"], 1234)
        self.assertEqual(image_kwargs["qwen_vl_model_name"], "qwen-vl-test")
        self.assertEqual(image_kwargs["qwen_vl_base_url"], "https://qwen.test")
        self.assertEqual(image_kwargs["qwen_vl_max_tokens"], 333)
        self.assertEqual(image_kwargs["qwen_vl_max_image_side"], 444)

    def test_backend_runtime_summary_distinguishes_live_local_and_mock_calls(self) -> None:
        model_calls = [
            {
                "task_type": "ocr",
                "provider": "doubao",
                "model_name": "mock-ocr",
                "response_model_name": None,
            },
            {
                "task_type": "ocr",
                "provider": "paddlepaddle",
                "model_name": "PP-OCRv5_mobile",
                "response_model_name": None,
            },
            {
                "task_type": "visual_understanding",
                "provider": "qwen",
                "model_name": "qwen-vl-plus",
                "response_model_name": "qwen-vl-plus",
            },
        ]

        summary = _build_backend_runtime_summary(model_calls)
        runtime_types = {
            item["runtime_type"]
            for item in summary["model_call_runtime_breakdown"]
        }
        qwen_items = [
            item
            for item in summary["model_call_runtime_breakdown"]
            if item["model_name"] == "qwen-vl-plus"
        ]

        self.assertTrue(summary["contains_live_api"])
        self.assertTrue(summary["contains_local_model"])
        self.assertTrue(summary["contains_mock"])
        self.assertEqual(runtime_types, {"live_api", "local_model", "mock"})
        self.assertEqual(qwen_items[0]["response_model_name"], "qwen-vl-plus")

    @patch("main.run_file_pipeline")
    def test_run_batch_passes_generation_limits_from_settings(self, mock_pipeline) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_settings(root, backend="deepseek", vision_backend="qwen_vl")
            settings_text = settings_path.read_text(encoding="utf-8")
            settings_path.write_text(
                settings_text.replace("deepseek_max_tokens: 1500", "deepseek_max_tokens: 3000")
                .replace("deepseek_compact_mode: false", "deepseek_compact_mode: true")
                .replace("qwen_vl_max_tokens: 500", "qwen_vl_max_tokens: 700")
                .replace("qwen_vl_max_image_side: 960", "qwen_vl_max_image_side: 480"),
                encoding="utf-8",
            )
            mock_pipeline.return_value = {
                "result": {
                    "schema_version": "v1",
                    "batch_id": "batch_limits",
                    "file_id": "file_001",
                    "file_name": "demo.txt",
                    "media_type": "text",
                    "processing_status": "success",
                    "processing_cost_cny": 0,
                    "processing_time_ms": 0,
                    "quality_flags": [],
                    "warning_messages": [],
                    "models_used": [],
                },
                "model_calls": [],
                "errors": [],
            }
            original_deepseek_key = os.environ.get("TEST_DEEPSEEK_API_KEY")
            original_dashscope_key = os.environ.get("TEST_DASHSCOPE_API_KEY")
            os.environ["TEST_DEEPSEEK_API_KEY"] = "test-deepseek"
            os.environ["TEST_DASHSCOPE_API_KEY"] = "test-dashscope"
            try:
                run_batch(settings_path=settings_path, allow_live_api=True, batch_id="batch_limits")
            finally:
                if original_deepseek_key is None:
                    os.environ.pop("TEST_DEEPSEEK_API_KEY", None)
                else:
                    os.environ["TEST_DEEPSEEK_API_KEY"] = original_deepseek_key
                if original_dashscope_key is None:
                    os.environ.pop("TEST_DASHSCOPE_API_KEY", None)
                else:
                    os.environ["TEST_DASHSCOPE_API_KEY"] = original_dashscope_key

        self.assertEqual(mock_pipeline.call_args.kwargs["deepseek_max_tokens"], 3000)
        self.assertTrue(mock_pipeline.call_args.kwargs["deepseek_compact_mode"])
        self.assertEqual(mock_pipeline.call_args.kwargs["qwen_vl_max_tokens"], 700)
        self.assertEqual(mock_pipeline.call_args.kwargs["qwen_vl_max_image_side"], 480)

    def test_run_batch_rejects_deepseek_without_live_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(Path(tmp_dir), backend="deepseek")

            with self.assertRaisesRegex(PermissionError, "--allow-live-api"):
                run_batch(settings_path=settings_path, batch_id="batch_live_blocked")

    def test_run_batch_rejects_qwen_vl_without_live_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(
                Path(tmp_dir),
                backend="mock",
                vision_backend="qwen_vl",
            )
            (Path(tmp_dir) / "input" / "demo.png").write_bytes(b"image")

            with self.assertRaisesRegex(PermissionError, "--allow-live-api"):
                run_batch(
                    settings_path=settings_path,
                    include_file_names=["demo.png"],
                    batch_id="batch_qwen_vl_blocked",
                )

    def test_run_batch_rejects_dashscope_asr_without_live_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(Path(tmp_dir), backend="mock")

            with self.assertRaisesRegex(PermissionError, "--allow-live-api"):
                run_batch(
                    settings_path=settings_path,
                    speech_to_text_backend_override="dashscope_asr",
                    batch_id="batch_asr_blocked",
                )

    @patch("main._ensure_paddleocr_runtime_available")
    def test_run_batch_allows_paddleocr_without_live_permission(self, mock_runtime_check) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(
                Path(tmp_dir),
                backend="mock",
                ocr_backend="paddleocr",
            )
            (Path(tmp_dir) / "input" / "demo.png").write_bytes(b"image")

            summary = run_batch(
                settings_path=settings_path,
                include_file_names=["demo.png"],
                batch_id="batch_local_ocr",
            )

        self.assertEqual(summary["total_files"], 1)
        mock_runtime_check.assert_called_once_with()

    def test_cli_live_permission_requires_explicit_real_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(Path(tmp_dir), backend="mock")
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = cli_main(["--settings", str(settings_path), "--allow-live-api"])

            self.assertEqual(exit_code, 2)
            self.assertIn("--speech-backend dashscope_asr", stdout.getvalue())
            self.assertFalse((Path(tmp_dir) / "output").exists())

    @patch("main.run_batch")
    @patch("main.run_preflight")
    def test_cli_preflight_only_does_not_run_batch(self, mock_run_preflight, mock_run_batch) -> None:
        mock_run_preflight.return_value = {
            "batch_id": "preflight_cli",
            "preflight_status": "warning",
            "recommended_action": "受控试跑",
            "report_paths": {},
        }

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main(
                [
                    "--preflight-only",
                    "--batch-id",
                    "preflight_cli",
                    "--ocr-backend",
                    "paddleocr",
                    "--vision-backend",
                    "qwen_vl",
                    "--speech-backend",
                    "dashscope_asr",
                    "--text-analysis-backend",
                    "deepseek",
                    "--historical-model-calls",
                    "output/demo/model_calls.jsonl",
                ]
            )

        self.assertEqual(exit_code, 0)
        mock_run_batch.assert_not_called()
        call_kwargs = mock_run_preflight.call_args.kwargs
        self.assertEqual(call_kwargs["ocr_backend_override"], "paddleocr")
        self.assertEqual(call_kwargs["vision_understanding_backend_override"], "qwen_vl")
        self.assertEqual(call_kwargs["speech_to_text_backend_override"], "dashscope_asr")
        self.assertEqual(call_kwargs["text_analysis_backend_override"], "deepseek")
        self.assertEqual(call_kwargs["historical_model_calls_paths"], ["output/demo/model_calls.jsonl"])

    @patch("main.run_batch")
    @patch("main.run_preflight")
    def test_cli_route_decision_is_preflight_only(self, mock_run_preflight, mock_run_batch) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["--route-decision-report", "decision.json"])

        self.assertEqual(exit_code, 2)
        self.assertIn("只用于 --preflight-only", stdout.getvalue())
        mock_run_preflight.assert_not_called()
        mock_run_batch.assert_not_called()

    @patch("main.run_batch")
    def test_cli_paddleocr_keeps_unselected_text_backend_mock(self, mock_run_batch) -> None:
        mock_run_batch.return_value = {"batch_id": "batch_safe"}

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main(["--ocr-backend", "paddleocr"])

        self.assertEqual(exit_code, 0)
        call_kwargs = mock_run_batch.call_args.kwargs
        self.assertEqual(call_kwargs["ocr_backend_override"], "paddleocr")
        self.assertEqual(call_kwargs["text_analysis_backend_override"], "mock")

    @patch("main.run_batch")
    def test_cli_include_files_passes_selected_file_names(self, mock_run_batch) -> None:
        mock_run_batch.return_value = {"batch_id": "batch_safe"}

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main(
                ["--include-files", "img_7.jpg,img_8.jpg,img_9.jpg"]
            )

        self.assertEqual(exit_code, 0)
        call_kwargs = mock_run_batch.call_args.kwargs
        self.assertEqual(call_kwargs["include_file_names"], ["img_7.jpg", "img_8.jpg", "img_9.jpg"])

    @patch("main.run_batch")
    def test_cli_ffmpeg_path_passes_explicit_path(self, mock_run_batch) -> None:
        mock_run_batch.return_value = {"batch_id": "batch_safe"}

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main(["--ffmpeg-path", "D:\\tools\\ffmpeg\\bin\\ffmpeg.exe"])

        self.assertEqual(exit_code, 0)
        call_kwargs = mock_run_batch.call_args.kwargs
        self.assertEqual(call_kwargs["ffmpeg_path"], "D:\\tools\\ffmpeg\\bin\\ffmpeg.exe")
        self.assertEqual(call_kwargs["max_keyframes"], 3)

    @patch("main.run_batch")
    def test_cli_max_keyframes_passes_explicit_value(self, mock_run_batch) -> None:
        mock_run_batch.return_value = {"batch_id": "batch_safe"}

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main(["--max-keyframes", "2"])

        self.assertEqual(exit_code, 0)
        call_kwargs = mock_run_batch.call_args.kwargs
        self.assertEqual(call_kwargs["max_keyframes"], 2)

    def test_run_batch_rejects_invalid_max_keyframes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(Path(tmp_dir), backend="mock")

            with self.assertRaisesRegex(ValueError, "视频关键帧数量必须大于等于 1"):
                run_batch(settings_path=settings_path, max_keyframes=0)

    def test_run_batch_rejects_invalid_deepseek_compact_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(Path(tmp_dir), backend="mock")
            settings_path.write_text(
                settings_path.read_text(encoding="utf-8").replace(
                    "deepseek_compact_mode: false",
                    "deepseek_compact_mode: maybe",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "deepseek_compact_mode"):
                run_batch(settings_path=settings_path, batch_id="batch_invalid_bool")

    @patch("main.run_batch")
    def test_cli_deepseek_keeps_unselected_ocr_backend_mock(self, mock_run_batch) -> None:
        mock_run_batch.return_value = {"batch_id": "batch_safe"}

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main(
                ["--text-analysis-backend", "deepseek", "--allow-live-api"]
            )

        self.assertEqual(exit_code, 0)
        call_kwargs = mock_run_batch.call_args.kwargs
        self.assertEqual(call_kwargs["ocr_backend_override"], "mock")
        self.assertEqual(call_kwargs["vision_understanding_backend_override"], "mock")
        self.assertEqual(call_kwargs["text_analysis_backend_override"], "deepseek")

    @patch("main.run_batch")
    def test_cli_qwen_vl_keeps_unselected_other_backends_mock(self, mock_run_batch) -> None:
        mock_run_batch.return_value = {"batch_id": "batch_safe"}

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main(["--vision-backend", "qwen_vl", "--allow-live-api"])

        self.assertEqual(exit_code, 0)
        call_kwargs = mock_run_batch.call_args.kwargs
        self.assertEqual(call_kwargs["ocr_backend_override"], "mock")
        self.assertEqual(call_kwargs["vision_understanding_backend_override"], "qwen_vl")
        self.assertEqual(call_kwargs["text_analysis_backend_override"], "mock")

    def test_cli_api_retry_requires_explicit_real_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_settings(root, backend="mock")
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = cli_main(
                    [
                        "--settings",
                        str(settings_path),
                        "--max-api-retries",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("--text-analysis-backend deepseek 或 --vision-backend qwen_vl", stdout.getvalue())
            self.assertFalse((root / "output").exists())

    @patch("main.run_batch")
    def test_cli_qwen_vl_retry_passes_qwen_retry_only(self, mock_run_batch) -> None:
        mock_run_batch.return_value = {"batch_id": "batch_safe"}

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main(
                [
                    "--vision-backend",
                    "qwen_vl",
                    "--allow-live-api",
                    "--max-api-retries",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 0)
        call_kwargs = mock_run_batch.call_args.kwargs
        self.assertEqual(call_kwargs["deepseek_max_retries"], 0)
        self.assertEqual(call_kwargs["qwen_vl_max_retries"], 1)

    @patch("main.run_batch")
    def test_cli_deepseek_retry_passes_deepseek_retry_only(self, mock_run_batch) -> None:
        mock_run_batch.return_value = {"batch_id": "batch_safe"}

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main(
                [
                    "--text-analysis-backend",
                    "deepseek",
                    "--allow-live-api",
                    "--max-api-retries",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 0)
        call_kwargs = mock_run_batch.call_args.kwargs
        self.assertEqual(call_kwargs["deepseek_max_retries"], 1)
        self.assertEqual(call_kwargs["qwen_vl_max_retries"], 0)

    def test_run_batch_rejects_more_than_one_api_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(Path(tmp_dir), backend="mock")

            with self.assertRaisesRegex(ValueError, "最大重试次数只能是 0 或 1"):
                run_batch(
                    settings_path=settings_path,
                    deepseek_max_retries=2,
                    batch_id="batch_retry_invalid",
                )

    def test_run_batch_rejects_more_than_one_qwen_vl_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(
                Path(tmp_dir),
                backend="mock",
                vision_backend="qwen_vl",
            )

            with self.assertRaisesRegex(ValueError, "Qwen-VL 最大重试次数只能是 0 或 1"):
                run_batch(
                    settings_path=settings_path,
                    allow_live_api=True,
                    qwen_vl_max_retries=2,
                    batch_id="batch_qwen_retry_invalid",
                )

    def test_deepseek_with_permission_but_without_key_stops_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_settings(root, backend="mock")
            stdout = io.StringIO()
            original_api_key = os.environ.pop("TEST_DEEPSEEK_API_KEY", None)

            try:
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli_main(
                        [
                            "--settings",
                            str(settings_path),
                            "--text-analysis-backend",
                            "deepseek",
                            "--allow-live-api",
                        ]
                    )
            finally:
                if original_api_key is not None:
                    os.environ["TEST_DEEPSEEK_API_KEY"] = original_api_key

            self.assertEqual(exit_code, 2)
            self.assertIn("已在发送网络请求前停止运行", stdout.getvalue())
            self.assertFalse((root / "output").exists())

    def test_qwen_vl_with_permission_but_without_key_stops_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_settings(root, backend="mock")
            stdout = io.StringIO()
            original_api_key = os.environ.pop("TEST_DASHSCOPE_API_KEY", None)

            try:
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli_main(
                        [
                            "--settings",
                            str(settings_path),
                            "--vision-backend",
                            "qwen_vl",
                            "--allow-live-api",
                        ]
                    )
            finally:
                if original_api_key is not None:
                    os.environ["TEST_DASHSCOPE_API_KEY"] = original_api_key

            self.assertEqual(exit_code, 2)
            self.assertIn("已在发送网络请求前停止运行", stdout.getvalue())
            self.assertFalse((root / "output").exists())

    @patch("main.importlib.util.find_spec", return_value=None)
    def test_paddleocr_without_runtime_stops_before_output(self, _mock_find_spec) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_settings(root, backend="mock")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli_main(
                    [
                        "--settings",
                        str(settings_path),
                        "--ocr-backend",
                        "paddleocr",
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("缺少 PaddleOCR 运行依赖", stdout.getvalue())
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
