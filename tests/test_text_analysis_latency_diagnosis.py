"""text_analysis_latency_diagnosis 的离线测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from text_analysis_latency_diagnosis import diagnose_text_analysis_latency, render_markdown, write_reports  # noqa: E402


class TextAnalysisLatencyDiagnosisTest(unittest.TestCase):
    def test_diagnoses_output_token_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_case(root)

            report = diagnose_text_analysis_latency(config_path, project_root=root, generated_at="2026-08-20T00:00:00+08:00")

        self.assertEqual(report["overall_status"], "fail")
        slow = [row for row in report["calls"] if row["latency_status"] == "fail"]
        self.assertEqual(slow[0]["file_id"], "file_0002")
        self.assertIn("output_token_pressure", slow[0]["risk_flags"])
        self.assertIn("输出 token", " ".join(report["diagnosis"]))

    def test_diagnoses_multiple_high_output_slow_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_case(
                root,
                [
                    _call("file_0001", 12262, 2073, 1378),
                    _call("file_0002", 12999, 1984, 1517),
                    _call("file_0003", 3525, 2338, 274),
                ],
            )

            report = diagnose_text_analysis_latency(config_path, project_root=root, generated_at="2026-08-21T00:00:00+08:00")

        slow_rows = [row for row in report["calls"] if row["latency_status"] == "fail"]
        self.assertEqual([row["file_id"] for row in slow_rows], ["file_0001", "file_0002"])
        self.assertTrue(all("output_token_pressure" in row["risk_flags"] for row in slow_rows))

    def test_render_markdown_contains_core_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_case(root)
            report = diagnose_text_analysis_latency(config_path, project_root=root, generated_at="2026-08-20T00:00:00+08:00")

        markdown = render_markdown(report)
        self.assertIn("文本分析延迟诊断报告", markdown)
        self.assertIn("output_token_pressure", markdown)
        self.assertIn("下一步", markdown)

    def test_write_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = _write_case(root)
            report = diagnose_text_analysis_latency(config_path, project_root=root, generated_at="2026-08-20T00:00:00+08:00")
            paths = write_reports(report, root / "diagnosis.json", root / "diagnosis.md")

            saved_report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            saved_markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertEqual(saved_report["report_type"], "text_analysis_latency_diagnosis")
        self.assertIn("文本分析延迟诊断报告", saved_markdown)


def _write_case(root: Path, calls: list[dict] | None = None) -> Path:
    batch_dir = root / "output" / "batch"
    batch_dir.mkdir(parents=True)
    calls = calls or [
        _call("file_0001", 6000, 2000, 600),
        _call("file_0002", 11000, 1900, 1300),
        _call("file_0003", 7000, 2200, 800),
    ]
    (batch_dir / "model_calls.jsonl").write_text(
        "".join(json.dumps(call, ensure_ascii=False, indent=2) + "\n" for call in calls),
        encoding="utf-8",
    )
    config = {
        "baseline": {"batch_dir": "output/batch"},
        "criteria": {"preflight": {"max_task_p95_latency_ms": {"text_analysis": 8000}}},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return config_path


def _call(file_id: str, latency_ms: int, input_tokens: int, output_tokens: int) -> dict:
    return {
        "call_id": f"{file_id}_call",
        "file_id": file_id,
        "task_type": "text_analysis",
        "provider": "deepseek",
        "model_name": "deepseek-v4-flash",
        "status": "success",
        "latency_ms": latency_ms,
        "input_units": [{"unit_type": "input_tokens", "quantity": input_tokens}],
        "output_units": [{"unit_type": "output_tokens", "quantity": output_tokens}],
        "cost_cny": 0.01,
    }


if __name__ == "__main__":
    unittest.main()
