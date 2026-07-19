"""model_strategy_advisor 的离线测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model_strategy_advisor import (  # noqa: E402
    build_strategy_report_from_files,
    generate_strategy_report,
    render_strategy_markdown,
    write_strategy_reports,
)


def _batch_report() -> dict:
    """构造最小批次报告。"""

    return {
        "batch_id": "batch_test",
        "file_stats": {
            "total_files": 2,
            "success_rate": 1.0,
        },
        "cost_stats": {
            "total_cost_cny": 1.5,
        },
        "latency_stats": {
            "avg_processing_time_per_file_ms": 200,
            "p95_model_latency_ms": 500,
        },
    }


def _model_calls() -> list[dict]:
    """构造模型调用明细。"""

    return [
        {
            "call_id": "call_001",
            "file_id": "file_001",
            "task_type": "ocr",
            "provider": "doubao",
            "model_name": "mock-ocr",
            "cost_cny": 1.0,
            "latency_ms": 0,
        },
        {
            "call_id": "call_002",
            "file_id": "file_001",
            "task_type": "text_analysis",
            "provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "cost_cny": 0.5,
            "latency_ms": 500,
        },
    ]


class ModelStrategyAdvisorTest(unittest.TestCase):
    def test_cost_summary(self) -> None:
        report = generate_strategy_report(_batch_report(), _model_calls(), generated_at="2026-07-18T10:00:00+08:00")

        self.assertEqual(report["batch_overview"]["model_call_count"], 2)
        self.assertEqual(report["cost_analysis"]["deepseek_cost_cny"], 0.5)
        self.assertEqual(report["cost_analysis"]["mock_cost_cny"], 1.0)
        self.assertAlmostEqual(report["cost_analysis"]["deepseek_cost_share"], 0.333333)
        self.assertTrue(report["cost_analysis"]["mock_cost_counted"])

    def test_latency_bottleneck_identification(self) -> None:
        report = generate_strategy_report(_batch_report(), _model_calls(), generated_at="2026-07-18T10:00:00+08:00")

        slowest = report["latency_analysis"]["slowest_calls"][0]

        self.assertEqual(slowest["call_id"], "call_002")
        self.assertEqual(slowest["provider"], "deepseek")
        self.assertIn("DeepSeek", report["latency_analysis"]["current_bottleneck"])
        self.assertIn("P95", report["latency_analysis"]["p95_driver"])

    def test_real_and_mock_boundary(self) -> None:
        report = generate_strategy_report(_batch_report(), _model_calls(), generated_at="2026-07-18T10:00:00+08:00")

        self.assertEqual(report["quality_boundary"]["real_model_calls"]["count"], 1)
        self.assertEqual(report["quality_boundary"]["mock_model_calls"]["count"], 1)
        self.assertEqual(report["quality_boundary"]["real_model_calls"]["task_types"], ["text_analysis"])
        self.assertEqual(report["quality_boundary"]["mock_model_calls"]["task_types"], ["ocr"])
        self.assertIn("当前不能证明真实 OCR", report["quality_boundary"]["cannot_prove"][0])

    def test_missing_fields_are_handled(self) -> None:
        report = generate_strategy_report({}, [{"model_name": "mock-ocr"}], generated_at="2026-07-18T10:00:00+08:00")

        self.assertIsNone(report["batch_overview"]["total_files"])
        self.assertEqual(report["batch_overview"]["model_call_count"], 1)
        self.assertTrue(report["missing_data_notes"])
        self.assertIn("当前数据未提供：file_stats.total_files", report["missing_data_notes"])

    def test_json_and_markdown_report_generation(self) -> None:
        report = generate_strategy_report(_batch_report(), _model_calls(), generated_at="2026-07-18T10:00:00+08:00")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_paths = write_strategy_reports(tmp_dir, report)
            json_path = Path(output_paths["json"])
            markdown_path = Path(output_paths["markdown"])

            saved_report = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(saved_report["report_type"], "model_strategy")
        self.assertIn("# 模型组合策略报告", markdown)
        self.assertIn("## 5. 模型组合建议", markdown)
        self.assertIn("字段说明", markdown)

    def test_build_strategy_report_from_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_dir = Path(tmp_dir)
            (batch_dir / "batch_report.json").write_text(json.dumps(_batch_report(), ensure_ascii=False), encoding="utf-8")
            (batch_dir / "model_calls.jsonl").write_text(
                "\n".join(json.dumps(call, ensure_ascii=False, indent=2) for call in _model_calls()),
                encoding="utf-8",
            )

            report = build_strategy_report_from_files(batch_dir, generated_at="2026-07-18T10:00:00+08:00")

        self.assertEqual(report["batch_id"], "batch_test")
        self.assertEqual(report["cost_analysis"]["top_cost_tasks"][0]["task_type"], "ocr")

    def test_markdown_uses_missing_text_for_absent_values(self) -> None:
        report = generate_strategy_report({}, [], generated_at="2026-07-18T10:00:00+08:00")

        markdown = render_strategy_markdown(report)

        self.assertIn("当前数据未提供", markdown)


if __name__ == "__main__":
    unittest.main()
