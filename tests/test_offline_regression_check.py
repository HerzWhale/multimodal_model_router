"""offline_regression_check 的离线测试。"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from offline_regression_check import main as cli_main  # noqa: E402
from offline_regression_check import check_batch_completeness  # noqa: E402
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
        self.assertFalse(report["boundary"]["calls_qwen_vl_api"])
        self.assertFalse(report["boundary"]["runs_real_paddleocr"])
        self.assertFalse(report["boundary"]["uses_cloud_ocr"])
        self.assertFalse(report["boundary"]["writes_official_output"])
        self.assertTrue(report["boundary"]["uses_temporary_output"])
        self.assertEqual([step["step_name"] for step in report["steps"]], ["mock_batch_smoke", "routing_preflight_smoke"])
        self.assertEqual(report["steps"][0]["total_files"], 3)
        self.assertEqual(report["steps"][0]["expected_video_v0_errors"], 1)
        self.assertEqual(report["steps"][0]["unexpected_errors"], 0)
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
        self.assertIn("batch_completeness_check", report["field_notes"])

    def test_batch_completeness_check_passes_for_complete_video_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_dir = _write_complete_video_batch(Path(tmp_dir))

            report = check_batch_completeness(batch_dir)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["total_files"], 1)
        self.assertEqual(report["total_model_calls"], 4)
        self.assertEqual(report["issues"], [])

    def test_batch_completeness_check_fails_when_expected_task_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_dir = _write_complete_video_batch(Path(tmp_dir), omit_task_type="visual_understanding")

            report = check_batch_completeness(batch_dir)

        self.assertEqual(report["status"], "fail")
        self.assertIn("缺少成功的 visual_understanding 模型调用", "\n".join(report["issues"]))

    def test_batch_completeness_check_can_require_no_mock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_dir = _write_complete_video_batch(Path(tmp_dir), contains_mock=True)

            report = check_batch_completeness(batch_dir, require_no_mock=True)

        self.assertEqual(report["status"], "fail")
        self.assertIn("包含 mock 调用", "\n".join(report["issues"]))

    def test_cli_can_check_existing_batch_dir_without_unit_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_dir = _write_complete_video_batch(Path(tmp_dir))
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = cli_main(["--skip-unit-tests", "--project-root", str(PROJECT_ROOT), "--check-batch-dir", str(batch_dir)])

        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["overall_status"], "pass")
        self.assertEqual(report["steps"][-1]["step_name"], "batch_completeness_check")


def _write_complete_video_batch(
    temp_root: Path,
    omit_task_type: str | None = None,
    contains_mock: bool = False,
) -> Path:
    batch_dir = temp_root / "batch_complete_video"
    batch_dir.mkdir()
    batch_id = "batch_complete_video"
    file_id = "file_0001"
    call_ids = [f"{file_id}_call_{index:04d}" for index in range(1, 5)]
    tasks = ["ocr", "visual_understanding", "speech_to_text", "text_analysis"]
    calls = []
    for call_id, task_type in zip(call_ids, tasks):
        if task_type == omit_task_type:
            continue
        calls.append(
            {
                "call_id": call_id,
                "batch_id": batch_id,
                "file_id": file_id,
                "task_type": task_type,
                "provider": "mock_provider",
                "model_name": "mock_model",
                "cost_cny": 0.001,
                "latency_ms": 100,
                "status": "success",
                "error_message": None,
            }
        )
    result = {
        "schema_version": "v1",
        "batch_id": batch_id,
        "file_id": file_id,
        "file_name": "example.mp4",
        "media_type": "video",
        "preprocessing_artifacts": {
            "preprocess_status": "success",
            "keyframe_count": 3,
            "keyframe_metadata": [{"frame_index": 1}, {"frame_index": 2}, {"frame_index": 3}],
            "audio_extraction_status": "extracted",
        },
        "topic": "other",
        "secondary_topics": [],
        "tags": ["合唱"],
        "summary": "一个合唱视频。",
        "processing_status": "success",
        "evidence_used": ["ocr_text", "visual_description", "audio_transcript"],
        "missing_evidence": [],
        "call_ids": [call["call_id"] for call in calls],
        "processing_cost_cny": 0.004,
        "processing_time_ms": 400,
        "warning_messages": [],
        "error_message": None,
    }
    metadata = {
        "schema_version": "v1",
        "batch_id": batch_id,
        "selected_backends": {
            "ocr_backend": "paddleocr",
            "vision_understanding_backend": "qwen_vl",
            "speech_to_text_backend": "dashscope_asr",
            "text_analysis_backend": "deepseek",
        },
        "backend_runtime_summary": {"contains_mock": contains_mock},
    }
    batch_report = {
        "schema_version": "v1",
        "batch_id": batch_id,
        "file_stats": {"total_files": 1, "success_files": 1},
        "cost_stats": {"total_cost_cny": 0.004},
        "latency_stats": {"p95_model_latency_ms": 100},
    }
    (batch_dir / "batch_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (batch_dir / "batch_report.json").write_text(json.dumps(batch_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (batch_dir / "results.jsonl").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (batch_dir / "model_calls.jsonl").write_text(
        "\n".join(json.dumps(call, ensure_ascii=False, indent=2) for call in calls) + "\n",
        encoding="utf-8",
    )
    return batch_dir


if __name__ == "__main__":
    unittest.main()
