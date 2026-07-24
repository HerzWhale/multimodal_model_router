"""pipeline_runner 的测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cost_latency_tracker import load_model_prices
from model_clients import DeepSeekAttemptsExhausted, DeepSeekResponseError
from model_router import load_routing_rules
from pipeline_runner import run_file_pipeline


class PipelineRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.routing_rules = load_routing_rules(PROJECT_ROOT / "config" / "routing_rules.yaml")
        self.model_prices = load_model_prices(PROJECT_ROOT / "config" / "model_prices.yaml")

    def _file_record(self, path: Path, media_type: str) -> dict[str, object]:
        return {
            "batch_id": "batch_001",
            "file_id": "file_001",
            "file_name": path.name,
            "source_path": str(path),
            "media_type": media_type,
            "file_size_bytes": path.stat().st_size,
            "created_at": "2026-07-14T10:00:00+08:00",
        }

    def test_run_text_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.txt"
            path.write_text("这是一段 AI 工具教程", encoding="utf-8")

            output = run_file_pipeline(self._file_record(path, "text"), self.routing_rules, self.model_prices)

        result = output["result"]
        self.assertEqual(result["processing_status"], "success")
        self.assertEqual(result["evidence_used"], ["raw_text"])
        self.assertEqual(len(output["model_calls"]), 1)
        self.assertEqual(output["model_calls"][0]["task_type"], "text_analysis")
        self.assertEqual(result["models_used"][0]["model_name"], output["model_calls"][0]["model_name"])

    def test_run_image_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.png"
            path.write_bytes(b"fake-image")

            output = run_file_pipeline(self._file_record(path, "image"), self.routing_rules, self.model_prices)

        result = output["result"]
        self.assertEqual(result["processing_status"], "success")
        self.assertEqual(result["evidence_used"], ["ocr_text", "visual_description"])
        self.assertEqual(len(output["model_calls"]), 3)
        self.assertEqual([call["task_type"] for call in output["model_calls"]], ["ocr", "visual_understanding", "text_analysis"])
        self.assertEqual([model["task_type"] for model in result["models_used"]], ["ocr", "visual_understanding", "text_analysis"])

    @patch("pipeline_runner.paddleocr_client")
    def test_image_pipeline_uses_local_paddleocr_backend(self, mock_ocr) -> None:
        mock_ocr.return_value = {"ocr_text": "标题\n正文内容"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.png"
            path.write_bytes(b"image")
            output = run_file_pipeline(
                self._file_record(path, "image"),
                self.routing_rules,
                self.model_prices,
                ocr_backend="paddleocr",
            )

        result = output["result"]
        ocr_call = output["model_calls"][0]
        self.assertEqual(result["ocr_text"], "标题\n正文内容")
        self.assertIn("ocr_text", result["evidence_used"])
        self.assertEqual(ocr_call["provider"], "paddlepaddle")
        self.assertEqual(ocr_call["model_name"], "PP-OCRv5_mobile")
        self.assertEqual(ocr_call["cost_cny"], 0.0)
        mock_ocr.assert_called_once_with(str(path))

    @patch("pipeline_runner.paddleocr_client")
    def test_image_pipeline_keeps_success_when_paddleocr_finds_no_text(self, mock_ocr) -> None:
        mock_ocr.return_value = {"ocr_text": None}

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "no_text.png"
            path.write_bytes(b"image")
            output = run_file_pipeline(
                self._file_record(path, "image"),
                self.routing_rules,
                self.model_prices,
                ocr_backend="paddleocr",
            )

        result = output["result"]
        self.assertEqual(result["processing_status"], "success")
        self.assertIsNone(result["ocr_text"])
        self.assertNotIn("ocr_text", result["evidence_used"])
        self.assertEqual(result["missing_evidence"], [])
        self.assertEqual(output["model_calls"][0]["status"], "success")

    @patch("pipeline_runner.paddleocr_client")
    def test_image_pipeline_records_paddleocr_failure_as_partial_success(self, mock_ocr) -> None:
        mock_ocr.side_effect = RuntimeError("PaddleOCR 暂时不可用")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "failed.png"
            path.write_bytes(b"image")
            output = run_file_pipeline(
                self._file_record(path, "image"),
                self.routing_rules,
                self.model_prices,
                ocr_backend="paddleocr",
            )

        result = output["result"]
        failed_call = output["model_calls"][0]
        self.assertEqual(result["processing_status"], "partial_success")
        self.assertEqual(result["missing_evidence"], ["ocr_text"])
        self.assertEqual(result["quality_flags"], ["ocr_failed"])
        self.assertEqual(failed_call["provider"], "paddlepaddle")
        self.assertEqual(failed_call["model_name"], "PP-OCRv5_mobile")
        self.assertEqual(failed_call["status"], "failed")
        self.assertIn("PaddleOCR 暂时不可用", result["error_message"])

    def test_run_video_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            path.write_bytes(b"fake-video")

            output = run_file_pipeline(self._file_record(path, "video"), self.routing_rules, self.model_prices)

        result = output["result"]
        self.assertEqual(result["processing_status"], "success")
        self.assertEqual(result["evidence_used"], ["ocr_text", "visual_description", "audio_transcript"])
        self.assertEqual(len(output["model_calls"]), 4)
        self.assertEqual(result["call_ids"], [call["call_id"] for call in output["model_calls"]])
        self.assertEqual(len(result["models_used"]), 4)

    @patch("pipeline_runner.paddleocr_client")
    def test_paddleocr_backend_does_not_claim_local_video_ocr(self, mock_local_ocr) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            path.write_bytes(b"fake-video")
            output = run_file_pipeline(
                self._file_record(path, "video"),
                self.routing_rules,
                self.model_prices,
                ocr_backend="paddleocr",
            )

        ocr_call = output["model_calls"][0]
        mock_local_ocr.assert_not_called()
        self.assertEqual(ocr_call["provider"], "doubao")
        self.assertEqual(ocr_call["model_name"], "mock-ocr")

    def test_image_pipeline_partial_success_when_ocr_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.png"
            path.write_bytes(b"fake-image")

            output = run_file_pipeline(
                self._file_record(path, "image"),
                self.routing_rules,
                self.model_prices,
                fault_injection={"ocr": "演示用 OCR 失败"},
            )

        result = output["result"]
        self.assertEqual(result["processing_status"], "partial_success")
        self.assertEqual(result["evidence_used"], ["visual_description"])
        self.assertEqual(result["missing_evidence"], ["ocr_text"])
        self.assertEqual(result["quality_flags"], ["ocr_failed"])
        self.assertTrue(result["warning_messages"])
        self.assertIn("OCR", result["warning_messages"][0])
        self.assertIn("演示用 OCR 失败", result["error_message"])
        self.assertEqual([call["status"] for call in output["model_calls"]], ["failed", "success", "success"])
        self.assertEqual([call["task_type"] for call in output["model_calls"]], ["ocr", "visual_understanding", "text_analysis"])
        self.assertEqual(len(output["errors"]), 1)
        self.assertEqual(output["errors"][0]["task_type"], "ocr")

    def test_text_pipeline_failed_when_text_analysis_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.txt"
            path.write_text("这是一段 AI 工具教程", encoding="utf-8")

            output = run_file_pipeline(
                self._file_record(path, "text"),
                self.routing_rules,
                self.model_prices,
                fault_injection={"text_analysis": "演示用文本分析失败"},
            )

        result = output["result"]
        self.assertEqual(result["processing_status"], "failed")
        self.assertIsNone(result["topic"])
        self.assertEqual(result["quality_flags"], ["text_analysis_failed"])
        self.assertTrue(result["warning_messages"])
        self.assertIn("演示用文本分析失败", result["error_message"])
        self.assertEqual([call["status"] for call in output["model_calls"]], ["failed"])
        self.assertEqual(output["model_calls"][0]["task_type"], "text_analysis")
        self.assertEqual(len(output["errors"]), 1)
        self.assertEqual(output["errors"][0]["task_type"], "text_analysis")

    @patch("pipeline_runner.deepseek_text_analysis_client")
    def test_deepseek_retry_records_each_attempt_and_keeps_file_success(self, mock_deepseek) -> None:
        mock_deepseek.return_value = {
            "topic": "technology",
            "secondary_topics": ["knowledge"],
            "tags": ["AI工程"],
            "summary": "技术内容摘要。",
            "business_use": "可用于技术内容归档。",
            "_api_usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
            "_api_attempts": [
                {
                    "status": "failed",
                    "latency_ms": 800,
                    "api_usage": {"prompt_tokens": 120, "completion_tokens": 5, "total_tokens": 125},
                    "error_message": "[deepseek_content_invalid_json] 模型内容不是合法 JSON。",
                },
                {
                    "status": "success",
                    "latency_ms": 900,
                    "api_usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
                    "error_message": None,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.txt"
            path.write_text("这是一段 AI 工具教程", encoding="utf-8")
            output = run_file_pipeline(
                self._file_record(path, "text"),
                self.routing_rules,
                self.model_prices,
                text_analysis_backend="deepseek",
                deepseek_api_key="test-key",
                deepseek_max_retries=1,
            )

        result = output["result"]
        calls = output["model_calls"]
        self.assertEqual(result["processing_status"], "success")
        self.assertIsNone(result["error_message"])
        self.assertEqual(result["warning_messages"], [])
        self.assertEqual([call["status"] for call in calls], ["failed", "success"])
        self.assertEqual([call["call_id"] for call in calls], ["file_001_call_0001", "file_001_call_0002"])
        self.assertEqual(result["call_ids"], ["file_001_call_0001", "file_001_call_0002"])
        self.assertEqual(result["processing_cost_cny"], round(sum(call["cost_cny"] for call in calls), 6))
        self.assertEqual(len(output["errors"]), 1)
        self.assertEqual(output["errors"][0]["call_id"], "file_001_call_0001")
        mock_deepseek.assert_called_once_with(
            {"raw_text": "这是一段 AI 工具教程", "ocr_text": None, "audio_transcript": None, "visual_description": None},
            api_key="test-key",
            model_name="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            max_retries=1,
        )

    @patch("pipeline_runner.deepseek_text_analysis_client")
    def test_deepseek_exhausted_retries_records_both_failed_attempts(self, mock_deepseek) -> None:
        last_error = DeepSeekResponseError(
            "deepseek_content_invalid_json",
            "模型内容不是合法 JSON。",
            retryable=True,
        )
        mock_deepseek.side_effect = DeepSeekAttemptsExhausted(
            last_error,
            [
                {
                    "status": "failed",
                    "latency_ms": 700,
                    "api_usage": {"prompt_tokens": 100, "completion_tokens": 3},
                    "error_message": str(last_error),
                },
                {
                    "status": "failed",
                    "latency_ms": 750,
                    "api_usage": {"prompt_tokens": 100, "completion_tokens": 4},
                    "error_message": str(last_error),
                },
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.txt"
            path.write_text("这是一段 AI 工具教程", encoding="utf-8")
            output = run_file_pipeline(
                self._file_record(path, "text"),
                self.routing_rules,
                self.model_prices,
                text_analysis_backend="deepseek",
                deepseek_api_key="test-key",
                deepseek_max_retries=1,
            )

        self.assertEqual(output["result"]["processing_status"], "failed")
        self.assertEqual([call["status"] for call in output["model_calls"]], ["failed", "failed"])
        self.assertEqual(len(output["errors"]), 2)
        self.assertEqual(output["result"]["call_ids"], ["file_001_call_0001", "file_001_call_0002"])

    @patch("pipeline_runner.deepseek_text_analysis_client")
    def test_business_use_guard_flag_reaches_file_result(self, mock_deepseek) -> None:
        """业务用途降级标记应进入文件结果，但不把成功处理误判为失败。"""

        mock_deepseek.return_value = {
            "topic": "sports_health",
            "secondary_topics": [],
            "tags": ["马拉松", "补给"],
            "summary": "内容介绍马拉松补给方法。",
            "business_use": "可用于内容归档、检索和人工复核。",
            "_quality_flags": ["business_use_grounded_fallback"],
            "_api_usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
            "_api_attempts": [
                {
                    "status": "success",
                    "latency_ms": 600,
                    "api_usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
                    "error_message": None,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.txt"
            path.write_text("这是一份马拉松补给建议。", encoding="utf-8")
            output = run_file_pipeline(
                self._file_record(path, "text"),
                self.routing_rules,
                self.model_prices,
                text_analysis_backend="deepseek",
                deepseek_api_key="test-key",
            )

        result = output["result"]
        self.assertEqual(result["processing_status"], "success")
        self.assertEqual(result["business_use"], "可用于内容归档、检索和人工复核。")
        self.assertIn("business_use_grounded_fallback", result["quality_flags"])
        self.assertEqual(result["warning_messages"], [])
        self.assertIsNone(result["error_message"])


if __name__ == "__main__":
    unittest.main()
