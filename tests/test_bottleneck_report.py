"""bottleneck_report 的离线测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from bottleneck_report import generate_bottleneck_report, render_markdown, write_reports  # noqa: E402


def _batch_report() -> dict:
    return {
        "batch_id": "batch_test",
        "file_stats": {"total_files": 2, "success_rate": 1.0},
        "cost_stats": {"total_cost_cny": 0.3, "cost_confidence": "estimated_unreconciled"},
        "latency_stats": {"avg_model_latency_ms": 100, "p95_model_latency_ms": 200},
    }


def _model_calls() -> list[dict]:
    return [
        {
            "call_id": "call_1",
            "file_id": "file_1",
            "task_type": "ocr",
            "provider": "paddlepaddle",
            "model_name": "PP-OCRv5_mobile",
            "cost_cny": 0,
            "latency_ms": 300,
            "status": "success",
        },
        {
            "call_id": "call_2",
            "file_id": "file_2",
            "task_type": "speech_to_text",
            "provider": "dashscope",
            "model_name": "paraformer-v2",
            "cost_cny": 0.2,
            "latency_ms": 100,
            "status": "success",
        },
        {
            "call_id": "call_3",
            "file_id": "file_2",
            "task_type": "text_analysis",
            "provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "cost_cny": 0.1,
            "latency_ms": 500,
            "status": "success",
        },
        {
            "call_id": "call_4",
            "file_id": "file_2",
            "task_type": "text_analysis",
            "provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "cost_cny": 9,
            "latency_ms": 9999,
            "status": "failed",
        },
    ]


class BottleneckReportTest(unittest.TestCase):
    def test_generate_bottleneck_report_ignores_failed_calls(self) -> None:
        report = generate_bottleneck_report(_batch_report(), _model_calls(), generated_at="2026-08-13T00:00:00+08:00")

        self.assertEqual(report["batch_id"], "batch_test")
        self.assertEqual(report["cost_bottleneck"]["top_task_type"], "speech_to_text")
        self.assertEqual(report["cost_bottleneck"]["top_provider"], "dashscope")
        self.assertEqual(report["latency_bottleneck"]["top_task_type"], "text_analysis")
        self.assertEqual(report["latency_bottleneck"]["slowest_file_id"], "file_2")
        self.assertEqual(report["runtime_mix"]["local_model_call_count"], 1)
        self.assertEqual(report["runtime_mix"]["live_api_call_count"], 2)

    def test_render_markdown_contains_core_sections(self) -> None:
        report = generate_bottleneck_report(_batch_report(), _model_calls(), generated_at="2026-08-13T00:00:00+08:00")
        markdown = render_markdown(report)

        self.assertIn("批次成本与延迟瓶颈诊断报告", markdown)
        self.assertIn("成本最高任务：speech_to_text", markdown)
        self.assertIn("延迟最高任务：text_analysis", markdown)
        self.assertIn("成本仍未完成供应商账单对账", markdown)

    def test_write_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_dir = Path(tmp_dir)
            (batch_dir / "batch_report.json").write_text(json.dumps(_batch_report(), ensure_ascii=False), encoding="utf-8")
            (batch_dir / "model_calls.jsonl").write_text(
                "\n".join(json.dumps(call, ensure_ascii=False, indent=2) for call in _model_calls()),
                encoding="utf-8",
            )

            paths = write_reports(batch_dir)
            saved_report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            saved_markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

            self.assertEqual(saved_report["report_type"], "batch_bottleneck_report")
            self.assertIn("批次成本与延迟瓶颈诊断报告", saved_markdown)


if __name__ == "__main__":
    unittest.main()
