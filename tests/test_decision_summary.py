"""decision_summary 的离线测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from decision_summary import build_decision_summary, render_markdown, write_reports  # noqa: E402


def _batch_report() -> dict:
    return {
        "batch_id": "batch_test",
        "file_stats": {"total_files": 4, "success_rate": 1.0},
        "latency_stats": {"avg_model_latency_ms": 1000, "p95_model_latency_ms": 2000},
    }


def _bottleneck_report() -> dict:
    return {
        "latency_bottleneck": {
            "top_task_type": "text_analysis",
            "top_provider": "deepseek",
            "slowest_file_id": "file_0008",
        },
        "cost_bottleneck": {
            "top_task_type": "speech_to_text",
            "top_provider": "dashscope",
        },
    }


def _cost_report() -> dict:
    return {
        "summary": {
            "bill_reconciled": True,
            "total_estimated_cost_cny": 0.066553,
            "total_billed_cost_cny": 0.02,
            "total_cost_delta_cny": -0.046553,
            "total_cost_delta_rate": -0.699488,
            "confidence_counts": {"period_level_reconciled": 3},
        },
        "reconciliation_items": [
            {
                "provider": "dashscope",
                "model_name": "paraformer-v2",
                "estimated_cost_cny": 0.036014,
                "billed_cost_cny": 0,
                "billing_granularity": "hour",
            },
            {
                "provider": "deepseek",
                "model_name": "deepseek-v4-flash",
                "estimated_cost_cny": 0.016104,
                "billed_cost_cny": 0.02,
                "billing_granularity": "hour",
            },
        ],
    }


def _eval_report() -> dict:
    return {
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "prediction_coverage": 1.0,
    }


class DecisionSummaryTest(unittest.TestCase):
    def test_build_decision_summary_marks_controlled_batch_ready(self) -> None:
        report = build_decision_summary(
            _batch_report(),
            _bottleneck_report(),
            _cost_report(),
            _eval_report(),
            generated_at="2026-08-14T00:00:00+08:00",
        )

        self.assertEqual(report["report_type"], "decision_summary")
        self.assertEqual(report["readiness"]["status"], "controlled_batch_ready")
        self.assertEqual(report["quality_summary"]["accuracy"], 1.0)
        self.assertEqual(report["latency_summary"]["top_latency_task_type"], "text_analysis")
        self.assertEqual(report["cost_summary"]["zero_billed_live_api_models"], ["dashscope/paraformer-v2"])

    def test_markdown_keeps_cost_granularity_boundary(self) -> None:
        report = build_decision_summary(
            _batch_report(),
            _bottleneck_report(),
            _cost_report(),
            _eval_report(),
            generated_at="2026-08-14T00:00:00+08:00",
        )
        markdown = render_markdown(report)

        self.assertIn("批次决策摘要报告", markdown)
        self.assertIn("小时级或周期级", markdown)
        self.assertIn("dashscope/paraformer-v2", markdown)
        self.assertIn("不能代表单次调用级精确对账", markdown)

    def test_write_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_dir = Path(tmp_dir)
            files = {
                "batch_report.json": _batch_report(),
                "bottleneck_report.json": _bottleneck_report(),
                "cost_reconciliation_report_hour.json": _cost_report(),
                "video_topic_eval_report.json": _eval_report(),
            }
            for name, data in files.items():
                (batch_dir / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            paths = write_reports(batch_dir)
            saved_report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            saved_markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

            self.assertEqual(saved_report["batch_id"], "batch_test")
            self.assertIn("技术负责人决策建议", saved_markdown)


if __name__ == "__main__":
    unittest.main()
