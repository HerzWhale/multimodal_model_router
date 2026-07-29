"""routing_preflight 的离线测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model_catalog import UNKNOWN_VALUE_TEXT  # noqa: E402
from routing_preflight import (  # noqa: E402
    apply_backend_overrides,
    build_historical_latency_profile,
    build_preflight_from_files,
    build_preflight_report,
    build_workload_profile,
    render_preflight_markdown,
    write_preflight_reports,
)


def _routing_rules() -> dict[str, dict[str, str]]:
    """构造测试用路由规则。"""

    return {
        "ocr": {"provider": "doubao", "model_name": "mock-ocr"},
        "visual_understanding": {"provider": "qwen", "model_name": "mock-vision"},
        "speech_to_text": {"provider": "doubao", "model_name": "mock-asr"},
        "text_analysis": {"provider": "deepseek", "model_name": "mock-text"},
        "summary_merge": {"provider": "deepseek", "model_name": "mock-text"},
    }


def _model_prices() -> dict[str, dict]:
    """构造测试用价格表。"""

    return {
        "mock-ocr": {"provider": "doubao", "pricing_unit": "image_count", "price_cny_per_unit": 0.01},
        "mock-vision": {"provider": "qwen", "pricing_unit": "frame_count", "price_cny_per_unit": 0.01},
        "mock-asr": {"provider": "doubao", "pricing_unit": "audio_seconds", "price_cny_per_unit": 0.0005},
        "mock-text": {"provider": "deepseek", "pricing_unit": "input_tokens", "price_cny_per_unit": 0.000002},
        "PP-OCRv5_mobile": {"provider": "paddlepaddle", "pricing_unit": "image_count", "price_cny_per_unit": 0.0},
        "deepseek-v4-flash": {
            "provider": "deepseek",
            "pricing_rules": [
                {"unit_type": "input_tokens", "price_cny_per_unit": 0.000001008},
                {"unit_type": "output_tokens", "price_cny_per_unit": 0.000002016},
            ],
        },
    }


class RoutingPreflightTest(unittest.TestCase):
    def _create_sample_input_dir(self, root: Path) -> Path:
        """创建测试用输入目录。"""

        input_dir = root / "input"
        text_dir = input_dir / "sample_text"
        image_dir = input_dir / "sample_images"
        video_dir = input_dir / "sample_videos"
        text_dir.mkdir(parents=True)
        image_dir.mkdir(parents=True)
        video_dir.mkdir(parents=True)
        (text_dir / "sample.txt").write_text("这是一个内容平台测试文本。", encoding="utf-8")
        (image_dir / "sample.jpg").write_bytes(b"fake image bytes")
        (image_dir / "second.png").write_bytes(b"fake image bytes")
        (video_dir / "sample.mp4").write_bytes(b"fake video bytes")
        return input_dir

    def _write_model_calls(self, path: Path, records: list[dict]) -> None:
        """写入测试用模型调用记录。"""

        path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
            encoding="utf-8",
        )

    def test_default_mock_routes_fail_real_coverage_gate(self) -> None:
        report = build_preflight_report(
            routing_rules=_routing_rules(),
            model_prices=_model_prices(),
            policy_name="balanced",
            generated_at="2026-07-28T10:00:00+08:00",
        )

        self.assertEqual(report["report_type"], "routing_preflight")
        self.assertEqual(report["preflight_status"], "fail")
        self.assertEqual(report["route_summary"]["real_coverage_rate"], 0.0)
        self.assertEqual(report["constraint_checks"][3]["constraint_name"], "min_real_coverage_rate")
        self.assertEqual(report["constraint_checks"][3]["status"], "fail")
        self.assertIn("ocr", report["route_summary"]["mock_task_types"])

    def test_backend_overrides_match_main_runtime_choices(self) -> None:
        updated_rules = apply_backend_overrides(
            _routing_rules(),
            ocr_backend="paddleocr",
            text_analysis_backend="deepseek",
        )

        self.assertEqual(updated_rules["ocr"]["provider"], "paddlepaddle")
        self.assertEqual(updated_rules["ocr"]["model_name"], "PP-OCRv5_mobile")
        self.assertEqual(updated_rules["text_analysis"]["model_name"], "deepseek-v4-flash")
        self.assertEqual(updated_rules["summary_merge"]["model_name"], "deepseek-v4-flash")
        self.assertEqual(_routing_rules()["ocr"]["model_name"], "mock-ocr")

    def test_workload_profile_counts_media_types_and_expected_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = self._create_sample_input_dir(Path(tmp_dir))
            profile = build_workload_profile(
                input_dir,
                expected_frames_per_video=3,
                expected_audio_seconds_per_video=60,
                generated_at="2026-07-28T10:00:00+08:00",
            )

        self.assertEqual(profile["profile_type"], "routing_preflight_workload")
        self.assertEqual(profile["total_files"], 4)
        self.assertEqual(profile["media_type_counts"], {"text": 1, "image": 2, "video": 1})
        self.assertEqual(profile["expected_units_by_task"]["ocr"]["quantity"], 5)
        self.assertEqual(profile["expected_units_by_task"]["visual_understanding"]["quantity"], 5)
        self.assertEqual(profile["expected_units_by_task"]["speech_to_text"]["quantity"], 60)
        self.assertEqual(profile["expected_units_by_task"]["text_analysis"][1]["quantity"], 1200)
        self.assertEqual(profile["expected_units_by_task"]["summary_merge"][0]["quantity"], 0)

    def test_workload_profile_can_limit_scope_with_include_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = self._create_sample_input_dir(Path(tmp_dir))
            profile = build_workload_profile(input_dir, include_files=["sample.jpg"])

        self.assertEqual(profile["total_files"], 1)
        self.assertEqual(profile["media_type_counts"], {"text": 0, "image": 1, "video": 0})
        self.assertEqual(profile["expected_units_by_task"]["ocr"]["quantity"], 1)
        self.assertEqual(profile["include_files"], ["sample.jpg"])

    def test_video_without_audio_seconds_keeps_speech_cost_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = self._create_sample_input_dir(Path(tmp_dir))
            profile = build_workload_profile(input_dir)
            report = build_preflight_report(
                routing_rules=_routing_rules(),
                model_prices=_model_prices(),
                policy_name="balanced",
                policy_overrides={"min_real_coverage_rate": 0.0},
                expected_units_by_task=profile["expected_units_by_task"],
                workload_profile=profile,
            )

        self.assertNotIn("speech_to_text", profile["expected_units_by_task"])
        self.assertIn("speech_to_text", report["route_summary"]["cost_unknown_task_types"])
        self.assertEqual(report["route_summary"]["estimated_total_cost_cny"], UNKNOWN_VALUE_TEXT)
        self.assertTrue(any("语音识别成本仍无法估算" in item for item in profile["warning_messages"]))

    def test_preflight_from_input_dir_estimates_total_cost_when_units_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_dir = self._create_sample_input_dir(tmp_path)
            routing_path = tmp_path / "routing_rules.yaml"
            prices_path = tmp_path / "model_prices.yaml"
            routing_path.write_text(
                "\n".join(
                    [
                        "routing_rules:",
                        "  ocr:",
                        "    provider: doubao",
                        "    model_name: mock-ocr",
                        "  visual_understanding:",
                        "    provider: qwen",
                        "    model_name: mock-vision",
                        "  speech_to_text:",
                        "    provider: doubao",
                        "    model_name: mock-asr",
                        "  text_analysis:",
                        "    provider: deepseek",
                        "    model_name: deepseek-v4-flash",
                        "  summary_merge:",
                        "    provider: deepseek",
                        "    model_name: deepseek-v4-flash",
                    ]
                ),
                encoding="utf-8",
            )
            prices_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  mock-ocr:",
                        "    provider: doubao",
                        "    pricing_unit: image_count",
                        "    price_cny_per_unit: 0.01",
                        "  mock-vision:",
                        "    provider: qwen",
                        "    pricing_unit: frame_count",
                        "    price_cny_per_unit: 0.01",
                        "  mock-asr:",
                        "    provider: doubao",
                        "    pricing_unit: audio_seconds",
                        "    price_cny_per_unit: 0.0005",
                        "  deepseek-v4-flash:",
                        "    provider: deepseek",
                        "    pricing_rules:",
                        "      - unit_type: input_tokens",
                        "        price_cny_per_unit: 0.000001008",
                        "      - unit_type: output_tokens",
                        "        price_cny_per_unit: 0.000002016",
                    ]
                ),
                encoding="utf-8",
            )

            report = build_preflight_from_files(
                routing_rules_path=routing_path,
                model_prices_path=prices_path,
                policy_config_path=None,
                policy_name="balanced",
                input_dir=input_dir,
                expected_audio_seconds_per_video=60,
                policy_overrides={"budget_limit_cny": 1.0, "min_real_coverage_rate": 0.0},
            )

        self.assertIsNotNone(report["workload_profile"])
        self.assertIsInstance(report["route_summary"]["estimated_total_cost_cny"], float)
        self.assertEqual(report["constraint_checks"][1]["constraint_name"], "budget_limit_cny")
        self.assertEqual(report["constraint_checks"][1]["status"], "pass")
        self.assertIn("运行前规模画像", render_preflight_markdown(report))

    def test_historical_latency_profile_groups_p95_by_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_calls_path = Path(tmp_dir) / "model_calls.jsonl"
            self._write_model_calls(
                model_calls_path,
                [
                    {"task_type": "ocr", "provider": "paddlepaddle", "model_name": "PP-OCRv5_mobile", "latency_ms": 100, "status": "success"},
                    {"task_type": "ocr", "provider": "paddlepaddle", "model_name": "PP-OCRv5_mobile", "latency_ms": 200, "status": "success"},
                    {"task_type": "ocr", "provider": "paddlepaddle", "model_name": "PP-OCRv5_mobile", "latency_ms": 3000, "status": "success"},
                    {"task_type": "text_analysis", "provider": "deepseek", "model_name": "deepseek-v4-flash", "latency_ms": 500, "status": "success"},
                    {"task_type": "ocr", "provider": "paddlepaddle", "model_name": "PP-OCRv5_mobile", "latency_ms": 9000, "status": "failed"},
                ],
            )

            profile = build_historical_latency_profile(
                model_calls_path,
                generated_at="2026-07-28T10:00:00+08:00",
            )

        self.assertEqual(profile["profile_type"], "routing_preflight_latency")
        self.assertEqual(profile["task_latency_stats"]["ocr"]["call_count"], 3)
        self.assertEqual(profile["historical_p95_latency_by_task_ms"]["ocr"], 3000.0)
        self.assertEqual(profile["historical_p95_latency_by_task_ms"]["text_analysis"], 500.0)
        self.assertEqual(profile["source_model_calls"][0]["used_records"], 4)

    def test_historical_latency_profile_reports_mock_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_calls_path = Path(tmp_dir) / "model_calls.jsonl"
            self._write_model_calls(
                model_calls_path,
                [
                    {"task_type": "visual_understanding", "provider": "qwen", "model_name": "mock-vision", "latency_ms": 0, "status": "success"},
                    {"task_type": "speech_to_text", "provider": "doubao", "model_name": "mock-asr", "latency_ms": 0, "status": "success"},
                    {"task_type": "ocr", "provider": "paddlepaddle", "model_name": "PP-OCRv5_mobile", "latency_ms": "bad", "status": "success"},
                ],
            )

            profile = build_historical_latency_profile(model_calls_path)

        self.assertEqual(profile["task_latency_stats"]["visual_understanding"]["mock_call_count"], 1)
        self.assertEqual(profile["skipped_records"], 1)
        self.assertTrue(any("mock 调用" in item for item in profile["warning_messages"]))

    def test_preflight_from_model_calls_uses_historical_latency_for_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            routing_path = tmp_path / "routing_rules.yaml"
            prices_path = tmp_path / "model_prices.yaml"
            model_calls_path = tmp_path / "model_calls.jsonl"
            routing_path.write_text(
                "\n".join(
                    [
                        "routing_rules:",
                        "  ocr:",
                        "    provider: paddlepaddle",
                        "    model_name: PP-OCRv5_mobile",
                    ]
                ),
                encoding="utf-8",
            )
            prices_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  PP-OCRv5_mobile:",
                        "    provider: paddlepaddle",
                        "    pricing_unit: image_count",
                        "    price_cny_per_unit: 0.0",
                    ]
                ),
                encoding="utf-8",
            )
            self._write_model_calls(
                model_calls_path,
                [
                    {"task_type": "ocr", "provider": "paddlepaddle", "model_name": "PP-OCRv5_mobile", "latency_ms": 1000, "status": "success"},
                    {"task_type": "ocr", "provider": "paddlepaddle", "model_name": "PP-OCRv5_mobile", "latency_ms": 5000, "status": "success"},
                ],
            )

            report = build_preflight_from_files(
                routing_rules_path=routing_path,
                model_prices_path=prices_path,
                policy_config_path=None,
                policy_name="balanced",
                expected_task_types=["ocr"],
                historical_model_calls_paths=[model_calls_path],
                policy_overrides={"p95_latency_limit_ms": 3000, "min_real_coverage_rate": 0.0},
            )

        self.assertIsNotNone(report["latency_profile"])
        self.assertEqual(report["route_summary"]["max_expected_p95_latency_ms"], 5000.0)
        self.assertEqual(report["constraint_checks"][2]["constraint_name"], "p95_latency_limit_ms")
        self.assertEqual(report["constraint_checks"][2]["status"], "fail")
        self.assertEqual(report["preflight_status"], "fail")
        self.assertIn("历史延迟画像", render_preflight_markdown(report))

    def test_missing_task_type_blocks_preflight(self) -> None:
        report = build_preflight_report(
            routing_rules={"text_analysis": {"provider": "deepseek", "model_name": "deepseek-v4-flash"}},
            model_prices=_model_prices(),
            policy_name="balanced",
            expected_task_types=["text_analysis", "ocr"],
        )

        self.assertEqual(report["preflight_status"], "fail")
        self.assertEqual(report["route_summary"]["missing_task_types"], ["ocr"])
        self.assertTrue(any("ocr" in reason for reason in report["blocking_reasons"]))

    def test_budget_constraint_uses_expected_units_when_available(self) -> None:
        report = build_preflight_report(
            routing_rules={"text_analysis": {"provider": "deepseek", "model_name": "deepseek-v4-flash"}},
            model_prices=_model_prices(),
            policy_name="budget_first",
            policy_overrides={"budget_limit_cny": 1.0, "min_real_coverage_rate": 0.25},
            expected_task_types=["text_analysis"],
            expected_units_by_task={
                "text_analysis": [
                    {"unit_type": "input_tokens", "quantity": 1_000_000},
                    {"unit_type": "output_tokens", "quantity": 1_000_000},
                ]
            },
        )

        self.assertEqual(report["route_summary"]["estimated_total_cost_cny"], 3.024)
        self.assertEqual(report["constraint_checks"][1]["constraint_name"], "budget_limit_cny")
        self.assertEqual(report["constraint_checks"][1]["status"], "fail")
        self.assertEqual(report["preflight_status"], "fail")

    def test_latency_constraint_uses_known_p95_and_fails_when_over_limit(self) -> None:
        report = build_preflight_report(
            routing_rules={"ocr": {"provider": "paddlepaddle", "model_name": "PP-OCRv5_mobile"}},
            model_prices=_model_prices(),
            policy_name="latency_first",
            policy_overrides={"p95_latency_limit_ms": 2000, "min_real_coverage_rate": 0.25},
            expected_task_types=["ocr"],
            historical_p95_latency_by_task_ms={"ocr": 5000},
        )

        self.assertEqual(report["constraint_checks"][2]["constraint_name"], "p95_latency_limit_ms")
        self.assertEqual(report["constraint_checks"][2]["status"], "fail")
        self.assertEqual(report["preflight_status"], "fail")

    def test_unknown_budget_and_latency_are_not_fabricated(self) -> None:
        report = build_preflight_report(
            routing_rules={"ocr": {"provider": "paddlepaddle", "model_name": "PP-OCRv5_mobile"}},
            model_prices=_model_prices(),
            policy_name="balanced",
            policy_overrides={"min_real_coverage_rate": 0.25},
            expected_task_types=["ocr"],
        )

        self.assertEqual(report["route_summary"]["estimated_total_cost_cny"], UNKNOWN_VALUE_TEXT)
        self.assertEqual(report["route_summary"]["max_expected_p95_latency_ms"], UNKNOWN_VALUE_TEXT)
        self.assertEqual(report["preflight_status"], "warning")
        self.assertTrue(report["warning_messages"])

    def test_missing_price_is_reported_without_crashing(self) -> None:
        report = build_preflight_report(
            routing_rules={"text_analysis": {"provider": "deepseek", "model_name": "unknown-real-model"}},
            model_prices={},
            policy_name="balanced",
            policy_overrides={"min_real_coverage_rate": 0.25},
            expected_task_types=["text_analysis"],
        )

        self.assertEqual(report["current_route"][0]["price_status"], "unknown")
        self.assertEqual(report["route_summary"]["estimated_total_cost_cny"], UNKNOWN_VALUE_TEXT)
        self.assertEqual(report["preflight_status"], "warning")

    def test_json_and_markdown_reports_are_written(self) -> None:
        report = build_preflight_report(
            routing_rules={"text_analysis": {"provider": "deepseek", "model_name": "deepseek-v4-flash"}},
            model_prices=_model_prices(),
            policy_name="balanced",
            policy_overrides={"min_real_coverage_rate": 0.25},
            expected_task_types=["text_analysis"],
            generated_at="2026-07-28T10:00:00+08:00",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_paths = write_preflight_reports(tmp_dir, report)
            saved_report = json.loads(Path(output_paths["json"]).read_text(encoding="utf-8"))
            markdown = Path(output_paths["markdown"]).read_text(encoding="utf-8")

        self.assertEqual(saved_report["report_type"], "routing_preflight")
        self.assertIn("# 路由策略预检查报告", markdown)
        self.assertIn("字段说明", markdown)

    def test_build_preflight_from_files_reads_config_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            routing_path = tmp_path / "routing_rules.yaml"
            prices_path = tmp_path / "model_prices.yaml"
            policy_path = tmp_path / "routing_policy_config.yaml"
            routing_path.write_text(
                "\n".join(
                    [
                        "routing_rules:",
                        "  ocr:",
                        "    provider: doubao",
                        "    model_name: mock-ocr",
                        "  text_analysis:",
                        "    provider: deepseek",
                        "    model_name: mock-text",
                    ]
                ),
                encoding="utf-8",
            )
            prices_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  mock-ocr:",
                        "    provider: doubao",
                        "    pricing_unit: image_count",
                        "    price_cny_per_unit: 0.01",
                        "  PP-OCRv5_mobile:",
                        "    provider: paddlepaddle",
                        "    pricing_unit: image_count",
                        "    price_cny_per_unit: 0.0",
                        "  deepseek-v4-flash:",
                        "    provider: deepseek",
                        "    pricing_rules:",
                        "      - unit_type: input_tokens",
                        "        price_cny_per_unit: 0.000001008",
                        "      - unit_type: output_tokens",
                        "        price_cny_per_unit: 0.000002016",
                    ]
                ),
                encoding="utf-8",
            )
            policy_path.write_text(
                "\n".join(
                    [
                        "schema_version: v1",
                        "policies:",
                        "  balanced:",
                        "    min_real_coverage_rate: 0.5",
                        "    p95_latency_limit_ms: 3000",
                        "budget_expansion_multipliers:",
                        "  - 2",
                    ]
                ),
                encoding="utf-8",
            )

            report = build_preflight_from_files(
                routing_rules_path=routing_path,
                model_prices_path=prices_path,
                policy_config_path=policy_path,
                policy_name="balanced",
                expected_task_types=["ocr", "text_analysis"],
                ocr_backend="paddleocr",
                text_analysis_backend="deepseek",
            )

        self.assertEqual(report["route_summary"]["real_coverage_rate"], 1.0)
        self.assertEqual(report["preflight_status"], "warning")
        self.assertEqual(report["source_files"]["policy_config"], str(policy_path))

    def test_markdown_contains_no_claim_of_real_api_execution(self) -> None:
        report = build_preflight_report(
            routing_rules={"text_analysis": {"provider": "deepseek", "model_name": "deepseek-v4-flash"}},
            model_prices=_model_prices(),
            policy_name="balanced",
            expected_task_types=["text_analysis"],
        )

        markdown = render_preflight_markdown(report)

        self.assertIn("不触发 DeepSeek", markdown)
        self.assertIn("不能产生新的质量结论", markdown)

    def test_controlled_trial_plan_shrinks_scope_when_budget_passes_but_latency_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = self._create_sample_input_dir(Path(tmp_dir))
            profile = build_workload_profile(
                input_dir,
                expected_audio_seconds_per_video=60,
                generated_at="2026-07-28T10:00:00+08:00",
            )
            report = build_preflight_report(
                routing_rules=apply_backend_overrides(
                    _routing_rules(),
                    ocr_backend="paddleocr",
                    text_analysis_backend="deepseek",
                ),
                model_prices=_model_prices(),
                policy_name="balanced",
                policy_overrides={
                    "budget_limit_cny": 50,
                    "p95_latency_limit_ms": 3500,
                    "min_real_coverage_rate": 0.5,
                },
                expected_units_by_task=profile["expected_units_by_task"],
                historical_p95_latency_by_task_ms={"ocr": 28261, "text_analysis": 7112},
                workload_profile=profile,
                latency_profile={
                    "task_latency_stats": {
                        "ocr": {
                            "p95_latency_ms": 28261,
                            "real_call_count": 3,
                            "mock_call_count": 2,
                        },
                        "text_analysis": {
                            "p95_latency_ms": 7112,
                            "real_call_count": 20,
                            "mock_call_count": 3,
                        },
                    }
                },
            )

        plan = report["controlled_trial_plan"]
        self.assertEqual(report["preflight_status"], "fail")
        self.assertEqual(plan["decision"], "shrink_scope_before_running")
        self.assertEqual(plan["recommended_scope"]["max_total_files"], 3)
        self.assertEqual(plan["recommended_scope"]["max_video_files"], 0)
        self.assertLessEqual(len(plan["suggested_include_files"]), 3)
        self.assertEqual(plan["slow_task_evidence"][0]["task_type"], "ocr")

    def test_controlled_trial_plan_keeps_live_api_separated_from_local_ocr(self) -> None:
        report = build_preflight_report(
            routing_rules=apply_backend_overrides(
                _routing_rules(),
                ocr_backend="paddleocr",
                text_analysis_backend="deepseek",
            ),
            model_prices=_model_prices(),
            policy_name="balanced",
            policy_overrides={"budget_limit_cny": 50, "p95_latency_limit_ms": 3500, "min_real_coverage_rate": 0.5},
            expected_task_types=["ocr", "text_analysis"],
            workload_profile={
                "total_files": 3,
                "media_type_counts": {"text": 1, "image": 2, "video": 0},
                "files": [
                    {"file_name": "sample.txt", "media_type": "text"},
                    {"file_name": "img_a.jpg", "media_type": "image"},
                    {"file_name": "img_b.jpg", "media_type": "image"},
                ],
            },
            historical_p95_latency_by_task_ms={"ocr": 5000, "text_analysis": 7000},
        )

        commands = report["controlled_trial_plan"]["trial_commands"]
        local_ocr_command = next(command for command in commands if command["command_name"] == "local_ocr_trial")
        deepseek_command = next(command for command in commands if command["command_name"] == "deepseek_text_trial")

        self.assertFalse(local_ocr_command["requires_live_api"])
        self.assertIn("--text-analysis-backend mock", local_ocr_command["command"])
        self.assertTrue(deepseek_command["requires_live_api"])
        self.assertIn("--ocr-backend mock", deepseek_command["command"])
        self.assertIn("--allow-live-api", deepseek_command["command"])

    def test_markdown_renders_controlled_trial_plan(self) -> None:
        report = build_preflight_report(
            routing_rules=apply_backend_overrides(
                _routing_rules(),
                ocr_backend="paddleocr",
                text_analysis_backend="deepseek",
            ),
            model_prices=_model_prices(),
            policy_name="balanced",
            policy_overrides={"budget_limit_cny": 50, "p95_latency_limit_ms": 3500, "min_real_coverage_rate": 0.5},
            expected_task_types=["ocr", "text_analysis"],
            expected_units_by_task={
                "ocr": {"unit_type": "image_count", "quantity": 1},
                "text_analysis": [
                    {"unit_type": "input_tokens", "quantity": 100},
                    {"unit_type": "output_tokens", "quantity": 100},
                ],
            },
            workload_profile={
                "total_files": 2,
                "media_type_counts": {"text": 1, "image": 1, "video": 0},
                "total_file_size_bytes": 100,
                "estimated_raw_text_tokens": 10,
                "expected_units_by_task": {},
                "warning_messages": [],
                "files": [
                    {"file_name": "sample.txt", "media_type": "text"},
                    {"file_name": "img_a.jpg", "media_type": "image"},
                ],
            },
            historical_p95_latency_by_task_ms={"ocr": 5000, "text_analysis": 7000},
        )

        markdown = render_preflight_markdown(report)

        self.assertIn("受控小样本试跑建议", markdown)
        self.assertIn("shrink_scope_before_running", markdown)
        self.assertIn("offline_mock_trial", markdown)
        self.assertIn("controlled_trial_plan", markdown)


if __name__ == "__main__":
    unittest.main()
