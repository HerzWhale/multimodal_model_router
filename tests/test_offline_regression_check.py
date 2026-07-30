"""offline_regression_check 的离线测试。"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from offline_regression_check import main as cli_main  # noqa: E402
from offline_regression_check import run_offline_regression_check  # noqa: E402


class OfflineRegressionCheckTest(unittest.TestCase):
    def test_offline_regression_check_passes_without_real_models(self) -> None:
        official_output = PROJECT_ROOT / "output" / "offline_regression_mock_batch"

        report = run_offline_regression_check(
            project_root=PROJECT_ROOT,
            run_unit_tests=False,
            generated_at="2026-07-30T10:00:00+08:00",
        )

        self.assertEqual(report["overall_status"], "pass")
        self.assertFalse(report["boundary"]["calls_deepseek_api"])
        self.assertFalse(report["boundary"]["runs_real_paddleocr"])
        self.assertFalse(report["boundary"]["uses_cloud_ocr"])
        self.assertFalse(report["boundary"]["writes_official_output"])
        self.assertTrue(report["boundary"]["uses_temporary_output"])
        self.assertEqual([step["step_name"] for step in report["steps"]], ["mock_batch_smoke", "routing_preflight_smoke"])
        self.assertEqual(report["steps"][0]["total_files"], 3)
        self.assertEqual(report["steps"][0]["total_errors"], 0)
        self.assertEqual(report["steps"][1]["total_files"], 3)
        self.assertEqual(report["steps"][1]["constraint_statuses"]["budget_limit_cny"], "pass")
        self.assertIn("p95_latency_limit_ms", report["steps"][1]["constraint_statuses"])
        self.assertFalse(official_output.exists())

    def test_cli_outputs_json_and_can_skip_unit_tests(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = cli_main(["--skip-unit-tests", "--project-root", str(PROJECT_ROOT)])

        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["check_name"], "offline_regression_check")
        self.assertEqual(report["overall_status"], "pass")
        self.assertEqual([step["step_name"] for step in report["steps"]], ["mock_batch_smoke", "routing_preflight_smoke"])

    def test_field_notes_explain_report_fields(self) -> None:
        report = run_offline_regression_check(
            project_root=PROJECT_ROOT,
            run_unit_tests=False,
            generated_at="2026-07-30T10:00:00+08:00",
        )

        self.assertIn("overall_status", report["field_notes"])
        self.assertIn("boundary", report["field_notes"])
        self.assertIn("preflight_status", report["field_notes"])


if __name__ == "__main__":
    unittest.main()
