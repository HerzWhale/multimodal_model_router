"""main 的测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from main import run_batch


class MainTest(unittest.TestCase):
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

            report = json.loads((batch_dir / "batch_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["file_stats"]["total_files"], 1)
            self.assertEqual(report["batch_id"], "batch_test")


if __name__ == "__main__":
    unittest.main()
