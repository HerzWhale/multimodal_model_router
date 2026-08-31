"""report_generator 的测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from report_generator import generate_batch_report, generate_batch_report_from_files, read_jsonl


class ReportGeneratorTest(unittest.TestCase):
    def test_read_jsonl_keeps_historical_multiline_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.jsonl"
            path.write_text(
                '{\n  "file_id": "file_001"\n}\n{\n  "file_id": "file_002"\n}\n',
                encoding="utf-8",
            )

            records = read_jsonl(path)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["file_id"], "file_001")

    def test_read_jsonl_supports_standard_one_record_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.jsonl"
            path.write_text(
                '{"file_id": "file_001", "tags": ["中文"]}\n'
                '{"file_id": "file_002", "error_message": null}\n',
                encoding="utf-8",
            )

            records = read_jsonl(path)

        self.assertEqual([record["file_id"] for record in records], ["file_001", "file_002"])
        self.assertEqual(records[0]["tags"], ["中文"])
        self.assertIsNone(records[1]["error_message"])

    def test_generate_batch_report(self) -> None:
        results = [
            {
                "file_id": "file_001",
                "processing_status": "success",
                "quality_flags": [],
                "processing_time_ms": 100,
            },
            {
                "file_id": "file_002",
                "processing_status": "partial_success",
                "quality_flags": ["missing_audio"],
                "processing_time_ms": 300,
            },
            {
                "file_id": "file_003",
                "processing_status": "failed",
                "quality_flags": ["no_evidence"],
                "processing_time_ms": 50,
            },
        ]
        model_calls = [
            {
                "task_type": "ocr",
                "provider": "doubao",
                "cost_cny": 0.1,
                "latency_ms": 100,
            },
            {
                "task_type": "text_analysis",
                "provider": "deepseek",
                "cost_cny": 0.2,
                "latency_ms": 300,
            },
        ]
        errors = [
            {
                "error_level": "model_call",
                "task_type": "speech_to_text",
                "error_message": "音频格式不支持",
            }
        ]

        report = generate_batch_report(
            batch_id="batch_001",
            results=results,
            model_calls=model_calls,
            errors=errors,
            budget_limit_cny=10,
            generated_at="2026-07-14T10:00:00+08:00",
        )

        self.assertEqual(report["file_stats"]["total_files"], 3)
        self.assertEqual(report["file_stats"]["partial_success_files"], 1)
        self.assertEqual(report["cost_stats"]["total_cost_cny"], 0.3)
        self.assertEqual(report["latency_stats"]["slowest_file_id"], "file_002")
        self.assertEqual(report["error_quality_stats"]["total_errors"], 1)
        self.assertEqual(report["error_quality_stats"]["quality_flags_count"]["missing_audio"], 1)

    def test_generate_batch_report_counts_pending_files(self) -> None:
        report = generate_batch_report(
            batch_id="batch_pending",
            results=[
                {"file_id": "file_001", "processing_status": "pending", "processing_time_ms": 10, "quality_flags": ["text_analysis_deferred"]},
                {"file_id": "file_002", "processing_status": "success", "processing_time_ms": 20, "quality_flags": []},
            ],
            model_calls=[],
            errors=[],
            budget_limit_cny=10,
            generated_at="2026-08-26T00:00:00+08:00",
        )

        stats = report["file_stats"]
        self.assertEqual(stats["total_files"], 2)
        self.assertEqual(stats["pending_files"], 1)
        self.assertEqual(stats["pending_rate"], 0.5)
        counted = sum(stats[key] for key in ["success_files", "partial_success_files", "failed_files", "skipped_files", "pending_files"])
        self.assertEqual(counted, stats["total_files"])

    def test_generate_batch_report_from_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_dir = Path(tmp_dir)
            (batch_dir / "results.jsonl").write_text(
                json.dumps(
                    {
                        "file_id": "file_001",
                        "processing_status": "success",
                        "quality_flags": [],
                        "processing_time_ms": 100,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (batch_dir / "model_calls.jsonl").write_text(
                json.dumps(
                    {
                        "task_type": "ocr",
                        "provider": "doubao",
                        "cost_cny": 0.1,
                        "latency_ms": 100,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (batch_dir / "errors.jsonl").write_text("", encoding="utf-8")

            report = generate_batch_report_from_files(
                batch_dir=batch_dir,
                batch_id="batch_001",
                budget_limit_cny=10,
                generated_at="2026-07-14T10:00:00+08:00",
            )

        self.assertEqual(report["batch_id"], "batch_001")
        self.assertEqual(report["file_stats"]["success_files"], 1)

    def test_avg_cost_per_success_file_is_zero_without_success_files(self) -> None:
        report = generate_batch_report(
            batch_id="batch_failure_demo",
            results=[
                {
                    "file_id": "file_001",
                    "processing_status": "partial_success",
                    "quality_flags": ["ocr_failed"],
                    "processing_time_ms": 100,
                },
                {
                    "file_id": "file_002",
                    "processing_status": "failed",
                    "quality_flags": ["text_analysis_failed"],
                    "processing_time_ms": 50,
                },
            ],
            model_calls=[
                {
                    "task_type": "ocr",
                    "provider": "doubao",
                    "cost_cny": 0.1,
                    "latency_ms": 0,
                }
            ],
            errors=[],
            budget_limit_cny=10,
            generated_at="2026-07-14T10:00:00+08:00",
        )

        self.assertEqual(report["file_stats"]["success_files"], 0)
        self.assertEqual(report["cost_stats"]["avg_cost_per_success_file_cny"], 0.0)

    def test_recovered_retry_keeps_file_success_and_counts_both_attempts(self) -> None:
        report = generate_batch_report(
            batch_id="batch_retry_recovered",
            results=[
                {
                    "file_id": "file_001",
                    "processing_status": "success",
                    "quality_flags": [],
                    "processing_time_ms": 1700,
                }
            ],
            model_calls=[
                {
                    "task_type": "text_analysis",
                    "provider": "deepseek",
                    "status": "failed",
                    "cost_cny": 0.001,
                    "latency_ms": 800,
                },
                {
                    "task_type": "text_analysis",
                    "provider": "deepseek",
                    "status": "success",
                    "cost_cny": 0.002,
                    "latency_ms": 900,
                },
            ],
            errors=[
                {
                    "error_level": "model_call",
                    "task_type": "text_analysis",
                    "error_message": "第一次响应无法解析，第二次调用成功。",
                }
            ],
            budget_limit_cny=10,
            generated_at="2026-07-22T16:00:00+08:00",
        )

        self.assertEqual(report["file_stats"]["success_files"], 1)
        self.assertEqual(report["error_quality_stats"]["total_errors"], 1)
        self.assertEqual(report["cost_stats"]["total_cost_cny"], 0.003)
        self.assertEqual(report["latency_stats"]["avg_model_latency_ms"], 850.0)

    def test_cost_stats_separates_live_local_and_mock_costs(self) -> None:
        report = generate_batch_report(
            batch_id="batch_cost_scope",
            results=[
                {
                    "file_id": "file_001",
                    "processing_status": "success",
                    "quality_flags": [],
                    "processing_time_ms": 1000,
                }
            ],
            model_calls=[
                {
                    "task_type": "visual_understanding",
                    "provider": "qwen",
                    "model_name": "qwen-vl-plus",
                    "cost_cny": 0.005,
                    "latency_ms": 3000,
                },
                {
                    "task_type": "ocr",
                    "provider": "paddlepaddle",
                    "model_name": "PP-OCRv5_mobile",
                    "cost_cny": 0.0,
                    "latency_ms": 5000,
                },
                {
                    "task_type": "speech_to_text",
                    "provider": "doubao",
                    "model_name": "mock-asr",
                    "cost_cny": 0.15,
                    "latency_ms": 0,
                },
            ],
            errors=[],
            budget_limit_cny=10,
            generated_at="2026-08-09T14:00:00+08:00",
        )

        cost_stats = report["cost_stats"]
        self.assertEqual(cost_stats["total_cost_cny"], 0.005)
        self.assertEqual(cost_stats["recorded_total_cost_cny"], 0.155)
        self.assertEqual(cost_stats["live_api_cost_cny"], 0.005)
        self.assertEqual(cost_stats["local_model_cost_cny"], 0.0)
        self.assertEqual(cost_stats["mock_cost_cny"], 0.15)
        self.assertEqual(cost_stats["cost_by_runtime_type"], {"live_api": 0.005, "local_model": 0.0, "mock": 0.15})
        self.assertEqual(cost_stats["cost_confidence"], "estimated_unreconciled")
        self.assertFalse(cost_stats["billing_reconciled"])
        self.assertIn("不等同供应商后台真实扣费", cost_stats["cost_scope_note"])
        self.assertIn("免费额度抵扣", cost_stats["cost_estimation_note"])
        self.assertEqual(cost_stats["cost_by_task_type"], {"visual_understanding": 0.005})
        self.assertEqual(
            cost_stats["recorded_cost_by_task_type"],
            {"ocr": 0.0, "speech_to_text": 0.15, "visual_understanding": 0.005},
        )

    def test_cost_confidence_is_not_applicable_without_live_api_cost(self) -> None:
        report = generate_batch_report(
            batch_id="batch_local_only",
            results=[
                {
                    "file_id": "file_001",
                    "processing_status": "success",
                    "quality_flags": [],
                    "processing_time_ms": 1000,
                }
            ],
            model_calls=[
                {
                    "task_type": "ocr",
                    "provider": "paddlepaddle",
                    "model_name": "PP-OCRv5_mobile",
                    "cost_cny": 0.0,
                    "latency_ms": 5000,
                }
            ],
            errors=[],
            budget_limit_cny=10,
            generated_at="2026-08-10T12:00:00+08:00",
        )

        self.assertEqual(report["cost_stats"]["total_cost_cny"], 0.0)
        self.assertEqual(report["cost_stats"]["cost_confidence"], "not_applicable_no_live_api_cost")
        self.assertFalse(report["cost_stats"]["billing_reconciled"])


if __name__ == "__main__":
    unittest.main()
