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
from model_clients import (
    DEFAULT_QWEN_VL_MAX_IMAGE_SIDE,
    DEFAULT_QWEN_VL_MAX_TOKENS,
    DeepSeekAttemptsExhausted,
    DeepSeekResponseError,
    QwenVLAttemptsExhausted,
    QwenVLResponseError,
)
from model_router import load_routing_rules
from pipeline_runner import LOW_QUALITY_OCR_FLAG, VIDEO_EVIDENCE_WEAK_FLAG, _is_low_quality_ocr_text, run_file_pipeline


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

    def _video_preprocess_without_artifacts(self, path: Path) -> dict[str, object]:
        return {
            "keyframes": [],
            "audio_path": None,
            "duration_ms": None,
            "preprocessing_artifacts": {
                "schema_version": "v1",
                "preprocess_status": "failed",
                "source_path": str(path),
                "artifact_dir": None,
                "keyframe_paths": [],
                "keyframe_count": 0,
                "keyframe_extraction_status": "failed",
                "audio_path": None,
                "audio_extraction_status": "not_implemented",
                "duration_ms": None,
                "duration_source": "unavailable",
                "frame_count": None,
                "fps": None,
                "width": None,
                "height": None,
                "video_evidence_stability": "weak",
                "video_evidence_risk_reasons": ["关键帧数量少于 3 张。"],
                "warning_messages": ["视频V1测试固定返回：未产出关键帧，未提取音频。"],
            },
        }

    def _video_preprocess_with_keyframe(self, path: Path, keyframe_path: Path) -> dict[str, object]:
        return {
            "keyframes": [str(keyframe_path)],
            "audio_path": None,
            "duration_ms": 2000,
            "preprocessing_artifacts": {
                "schema_version": "v1",
                "preprocess_status": "success",
                "source_path": str(path),
                "artifact_dir": str(keyframe_path.parent),
                "keyframe_paths": [str(keyframe_path)],
                "keyframe_count": 1,
                "keyframe_extraction_status": "extracted",
                "audio_path": None,
                "audio_extraction_status": "not_implemented",
                "duration_ms": 2000,
                "duration_source": "opencv_frame_count_fps",
                "video_evidence_stability": "weak",
                "video_evidence_risk_reasons": ["关键帧数量少于 3 张。"],
                "warning_messages": ["视频V1尚未实现真实音频提取。"],
            },
        }

    def _video_preprocess_with_keyframe_and_audio(
        self,
        path: Path,
        keyframe_path: Path,
        audio_path: Path,
    ) -> dict[str, object]:
        return {
            "keyframes": [str(keyframe_path)],
            "audio_path": str(audio_path),
            "duration_ms": 2000,
            "preprocessing_artifacts": {
                "schema_version": "v1",
                "preprocess_status": "success",
                "source_path": str(path),
                "artifact_dir": str(keyframe_path.parent),
                "keyframe_paths": [str(keyframe_path)],
                "keyframe_count": 1,
                "keyframe_extraction_status": "extracted",
                "audio_path": str(audio_path),
                "audio_extraction_status": "extracted",
                "audio_extraction_method": "ffmpeg_wav",
                "audio_sample_rate_hz": 16000,
                "audio_channels": 1,
                "duration_ms": 2000,
                "duration_source": "opencv_frame_count_fps",
                "video_evidence_stability": "weak",
                "video_evidence_risk_reasons": ["关键帧数量少于 3 张。"],
                "warning_messages": [],
            },
        }

    def _video_preprocess_with_keyframes(self, path: Path, keyframe_paths: list[Path]) -> dict[str, object]:
        keyframe_metadata = [
            {
                "frame_index": index,
                "source_frame_index": (index - 1) * 50,
                "timestamp_ms": (index - 1) * 2000,
                "path": str(keyframe_path),
            }
            for index, keyframe_path in enumerate(keyframe_paths, start=1)
        ]
        return {
            "keyframes": [str(keyframe_path) for keyframe_path in keyframe_paths],
            "keyframe_metadata": keyframe_metadata,
            "audio_path": None,
            "duration_ms": 6000,
            "preprocessing_artifacts": {
                "schema_version": "v1",
                "preprocess_status": "success",
                "source_path": str(path),
                "artifact_dir": str(keyframe_paths[0].parent),
                "keyframe_paths": [str(keyframe_path) for keyframe_path in keyframe_paths],
                "keyframe_metadata": keyframe_metadata,
                "keyframe_count": len(keyframe_paths),
                "max_keyframes": 5,
                "keyframe_sampling_strategy": "start_early_then_spaced",
                "keyframe_extraction_status": "extracted",
                "audio_path": None,
                "audio_extraction_status": "not_implemented",
                "duration_ms": 6000,
                "duration_source": "opencv_frame_count_fps",
                "video_evidence_stability": "stable",
                "video_evidence_risk_reasons": [],
                "warning_messages": ["视频V1尚未实现真实音频提取。"],
            },
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

    @patch("pipeline_runner.qwen_vl_image_understanding_client")
    def test_image_pipeline_uses_qwen_vl_backend(self, mock_vision) -> None:
        mock_vision.return_value = {
            "visual_description": "图片展示一张中文信息图，包含标题、图表和参数说明。",
            "_api_usage": {"prompt_tokens": 300, "completion_tokens": 80, "total_tokens": 380},
            "_response_model_name": "qwen-vl-plus",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.png"
            path.write_bytes(b"image")
            output = run_file_pipeline(
                self._file_record(path, "image"),
                self.routing_rules,
                self.model_prices,
                vision_understanding_backend="qwen_vl",
                qwen_vl_api_key="test-key",
            )

        result = output["result"]
        vision_call = output["model_calls"][1]
        self.assertEqual(result["processing_status"], "success")
        self.assertEqual(result["visual_description"], "图片展示一张中文信息图，包含标题、图表和参数说明。")
        self.assertIn("visual_description", result["evidence_used"])
        self.assertEqual(vision_call["provider"], "qwen")
        self.assertEqual(vision_call["model_name"], "qwen-vl-plus")
        self.assertEqual(vision_call["response_model_name"], "qwen-vl-plus")
        self.assertEqual(vision_call["input_units"], [{"unit_type": "input_tokens", "quantity": 300}])
        self.assertEqual(vision_call["output_units"], [{"unit_type": "output_tokens", "quantity": 80}])
        self.assertEqual(vision_call["cost_cny"], 0.0004)
        self.assertEqual(result["models_used"][1]["response_model_name"], "qwen-vl-plus")
        mock_vision.assert_called_once_with(
            str(path),
            api_key="test-key",
            model_name="qwen-vl-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            max_retries=0,
            max_tokens=DEFAULT_QWEN_VL_MAX_TOKENS,
            max_image_side=DEFAULT_QWEN_VL_MAX_IMAGE_SIDE,
        )

    @patch("pipeline_runner.qwen_vl_image_understanding_client")
    def test_image_pipeline_records_qwen_vl_failure_as_partial_success(self, mock_vision) -> None:
        mock_vision.side_effect = RuntimeError("Qwen-VL 暂时不可用")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "failed_vision.png"
            path.write_bytes(b"image")
            output = run_file_pipeline(
                self._file_record(path, "image"),
                self.routing_rules,
                self.model_prices,
                vision_understanding_backend="qwen_vl",
                qwen_vl_api_key="test-key",
            )

        result = output["result"]
        failed_call = output["model_calls"][1]
        self.assertEqual(result["processing_status"], "partial_success")
        self.assertEqual(result["missing_evidence"], ["visual_description"])
        self.assertEqual(result["quality_flags"], ["visual_understanding_failed"])
        self.assertEqual(failed_call["provider"], "qwen")
        self.assertEqual(failed_call["model_name"], "qwen-vl-plus")
        self.assertEqual(failed_call["status"], "failed")
        self.assertIn("Qwen-VL 暂时不可用", result["error_message"])

    def test_low_quality_ocr_gate_flags_fragmented_noise(self) -> None:
        bad_ocr_text = "1n22\nS\nluiin22\n治\nluiS\n灯\nutin22\n6l\ncadiae  sota\n自药m"
        good_ocr_text = "食影双修\n专注于影视解说视频的创作\n《功夫女足》首评来了！"

        self.assertTrue(_is_low_quality_ocr_text(bad_ocr_text))
        self.assertFalse(_is_low_quality_ocr_text(good_ocr_text))

    def test_low_quality_ocr_gate_does_not_flag_normal_short_lines(self) -> None:
        normal_short_lines = "新闻资讯\n娱乐休闲\n知识科普\n生活日常\n科技数码\n体育健康\n财经商业\n广告营销\n其他"

        self.assertFalse(_is_low_quality_ocr_text(normal_short_lines))

    @patch("pipeline_runner.paddleocr_client")
    def test_image_pipeline_marks_low_quality_paddleocr_as_partial_success(self, mock_ocr) -> None:
        mock_ocr.return_value = {
            "ocr_text": "1n22\nS\nluiin22\n治\nluiS\n灯\nutin22\n6l\ncadiae  sota\n自药m"
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad_ocr.png"
            path.write_bytes(b"image")
            output = run_file_pipeline(
                self._file_record(path, "image"),
                self.routing_rules,
                self.model_prices,
                ocr_backend="paddleocr",
            )

        result = output["result"]
        self.assertEqual(result["processing_status"], "partial_success")
        self.assertIn(LOW_QUALITY_OCR_FLAG, result["quality_flags"])
        self.assertTrue(result["warning_messages"])
        self.assertNotIn("ocr_text", result["evidence_used"])
        self.assertEqual(result["missing_evidence"], [])
        self.assertEqual(output["errors"], [])
        self.assertEqual(output["model_calls"][0]["status"], "success")
        self.assertIn("1n22", result["ocr_text"])
        self.assertNotIn("1n22", result["summary"])

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

            with patch("pipeline_runner.preprocess_file", return_value=self._video_preprocess_without_artifacts(path)):
                output = run_file_pipeline(self._file_record(path, "video"), self.routing_rules, self.model_prices)

        result = output["result"]
        self.assertEqual(result["processing_status"], "partial_success")
        self.assertEqual(result["evidence_used"], [])
        self.assertEqual(result["missing_evidence"], ["ocr_text", "visual_description", "audio_transcript"])
        self.assertIn("video_keyframe_missing", result["quality_flags"])
        self.assertIn("video_audio_not_extracted", result["quality_flags"])
        self.assertEqual(result["preprocessing_artifacts"]["preprocess_status"], "failed")
        self.assertEqual(result["preprocessing_artifacts"]["audio_extraction_status"], "not_implemented")
        self.assertEqual(len(output["model_calls"]), 4)
        self.assertEqual([call["status"] for call in output["model_calls"]], ["failed", "failed", "failed", "success"])
        self.assertEqual(result["call_ids"], [call["call_id"] for call in output["model_calls"]])
        self.assertEqual(len(result["models_used"]), 4)

    def test_run_video_pipeline_passes_explicit_ffmpeg_path_to_preprocessor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            ffmpeg_path = Path(tmp_dir) / "ffmpeg.exe"
            path.write_bytes(b"fake-video")

            with patch(
                "pipeline_runner.preprocess_file",
                return_value=self._video_preprocess_without_artifacts(path),
            ) as mock_preprocess:
                run_file_pipeline(
                    self._file_record(path, "video"),
                    self.routing_rules,
                    self.model_prices,
                    ffmpeg_path=ffmpeg_path,
                )

        call_kwargs = mock_preprocess.call_args.kwargs
        self.assertEqual(call_kwargs["ffmpeg_path"], ffmpeg_path)

    @patch("pipeline_runner.paddleocr_client")
    def test_paddleocr_backend_does_not_run_when_video_keyframe_missing(self, mock_local_ocr) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            path.write_bytes(b"fake-video")
            with patch("pipeline_runner.preprocess_file", return_value=self._video_preprocess_without_artifacts(path)):
                output = run_file_pipeline(
                    self._file_record(path, "video"),
                    self.routing_rules,
                    self.model_prices,
                    ocr_backend="paddleocr",
                )

        ocr_call = output["model_calls"][0]
        mock_local_ocr.assert_not_called()
        self.assertEqual(ocr_call["provider"], "paddlepaddle")
        self.assertEqual(ocr_call["model_name"], "PP-OCRv5_mobile")
        self.assertEqual(ocr_call["status"], "failed")
        self.assertEqual(ocr_call["cost_cny"], 0.0)

    @patch("pipeline_runner.paddleocr_client")
    def test_video_pipeline_uses_paddleocr_on_extracted_keyframe(self, mock_local_ocr) -> None:
        mock_local_ocr.return_value = {"ocr_text": "视频关键帧标题\n视频关键帧正文"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            keyframe_path = Path(tmp_dir) / "demo_frame_0001.jpg"
            path.write_bytes(b"fake-video")
            keyframe_path.write_bytes(b"fake-keyframe")

            with patch("pipeline_runner.preprocess_file", return_value=self._video_preprocess_with_keyframe(path, keyframe_path)):
                output = run_file_pipeline(
                    self._file_record(path, "video"),
                    self.routing_rules,
                    self.model_prices,
                    ocr_backend="paddleocr",
                )

        result = output["result"]
        ocr_call = output["model_calls"][0]
        self.assertEqual(result["processing_status"], "partial_success")
        self.assertEqual(result["ocr_text"], "[关键帧 1] 视频关键帧标题\n视频关键帧正文")
        self.assertIn("ocr_text", result["evidence_used"])
        self.assertEqual(result["missing_evidence"], ["audio_transcript"])
        self.assertEqual(ocr_call["provider"], "paddlepaddle")
        self.assertEqual(ocr_call["model_name"], "PP-OCRv5_mobile")
        self.assertEqual(ocr_call["status"], "success")
        self.assertEqual(ocr_call["cost_cny"], 0.0)
        mock_local_ocr.assert_called_once_with(str(keyframe_path))

    @patch("pipeline_runner.qwen_vl_image_understanding_client")
    def test_video_pipeline_uses_qwen_vl_on_extracted_keyframe(self, mock_vision) -> None:
        mock_vision.return_value = {
            "visual_description": "关键帧展示一张视频信息图，包含标题、人物和数据说明。",
            "_api_usage": {"prompt_tokens": 300, "completion_tokens": 80, "total_tokens": 380},
            "_response_model_name": "qwen-vl-plus",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            keyframe_path = Path(tmp_dir) / "demo_frame_0001.jpg"
            path.write_bytes(b"fake-video")
            keyframe_path.write_bytes(b"fake-keyframe")

            with patch("pipeline_runner.preprocess_file", return_value=self._video_preprocess_with_keyframe(path, keyframe_path)):
                output = run_file_pipeline(
                    self._file_record(path, "video"),
                    self.routing_rules,
                    self.model_prices,
                    vision_understanding_backend="qwen_vl",
                    qwen_vl_api_key="test-key",
                )

        result = output["result"]
        vision_call = output["model_calls"][1]
        self.assertEqual(result["processing_status"], "partial_success")
        self.assertEqual(result["visual_description"], "[关键帧 1] 关键帧展示一张视频信息图，包含标题、人物和数据说明。")
        self.assertIn("visual_description", result["evidence_used"])
        self.assertEqual(result["missing_evidence"], ["audio_transcript"])
        self.assertEqual(vision_call["provider"], "qwen")
        self.assertEqual(vision_call["model_name"], "qwen-vl-plus")
        self.assertEqual(vision_call["response_model_name"], "qwen-vl-plus")
        self.assertEqual(vision_call["input_units"], [{"unit_type": "input_tokens", "quantity": 300}])
        self.assertEqual(vision_call["output_units"], [{"unit_type": "output_tokens", "quantity": 80}])
        self.assertEqual(vision_call["cost_cny"], 0.0004)
        mock_vision.assert_called_once_with(
            str(keyframe_path),
            api_key="test-key",
            model_name="qwen-vl-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            max_retries=0,
            max_tokens=DEFAULT_QWEN_VL_MAX_TOKENS,
            max_image_side=DEFAULT_QWEN_VL_MAX_IMAGE_SIDE,
        )

    @patch("pipeline_runner.qwen_vl_image_understanding_client")
    def test_video_qwen_vl_retry_records_failed_and_success_attempts_per_keyframe(self, mock_vision) -> None:
        mock_vision.return_value = {
            "visual_description": "关键帧展示一张视频信息图，包含标题、人物和数据说明。",
            "_api_usage": {"prompt_tokens": 300, "completion_tokens": 80, "total_tokens": 380},
            "_response_model_name": "qwen-vl-plus",
            "_api_attempts": [
                {
                    "status": "failed",
                    "latency_ms": 5000,
                    "api_usage": {},
                    "error_message": "[qwen_vl_network_disconnected] Qwen-VL API 网络连接被远端关闭。",
                    "response_model_name": None,
                },
                {
                    "status": "success",
                    "latency_ms": 4200,
                    "api_usage": {"prompt_tokens": 300, "completion_tokens": 80, "total_tokens": 380},
                    "error_message": None,
                    "response_model_name": "qwen-vl-plus",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            keyframe_path = Path(tmp_dir) / "demo_frame_0001.jpg"
            path.write_bytes(b"fake-video")
            keyframe_path.write_bytes(b"fake-keyframe")

            with patch("pipeline_runner.preprocess_file", return_value=self._video_preprocess_with_keyframe(path, keyframe_path)):
                output = run_file_pipeline(
                    self._file_record(path, "video"),
                    self.routing_rules,
                    self.model_prices,
                    vision_understanding_backend="qwen_vl",
                    qwen_vl_api_key="test-key",
                    qwen_vl_max_retries=1,
                )

        result = output["result"]
        visual_calls = [
            call for call in output["model_calls"]
            if call["task_type"] == "visual_understanding"
        ]
        self.assertEqual([call["status"] for call in visual_calls], ["failed", "success"])
        self.assertEqual(visual_calls[1]["response_model_name"], "qwen-vl-plus")
        self.assertIn("visual_description", result["evidence_used"])
        self.assertNotIn("video_visual_keyframe_failed", result["quality_flags"])
        self.assertIn("audio_transcript", result["missing_evidence"])
        self.assertIsNotNone(result["visual_description"])
        self.assertEqual(len([error for error in output["errors"] if error["task_type"] == "visual_understanding"]), 1)
        mock_vision.assert_called_once_with(
            str(keyframe_path),
            api_key="test-key",
            model_name="qwen-vl-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            max_retries=1,
            max_tokens=DEFAULT_QWEN_VL_MAX_TOKENS,
            max_image_side=DEFAULT_QWEN_VL_MAX_IMAGE_SIDE,
        )

    @patch("pipeline_runner.qwen_vl_image_understanding_client")
    def test_video_qwen_vl_exhausted_retry_keeps_other_pipeline_compensation(self, mock_vision) -> None:
        last_error = QwenVLResponseError(
            "qwen_vl_network_disconnected",
            "Qwen-VL API 网络连接被远端关闭。",
            retryable=True,
        )
        mock_vision.side_effect = QwenVLAttemptsExhausted(
            last_error,
            [
                {
                    "status": "failed",
                    "latency_ms": 5100,
                    "api_usage": {},
                    "error_message": str(last_error),
                    "response_model_name": None,
                },
                {
                    "status": "failed",
                    "latency_ms": 5200,
                    "api_usage": {},
                    "error_message": str(last_error),
                    "response_model_name": None,
                },
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            keyframe_path = Path(tmp_dir) / "demo_frame_0001.jpg"
            path.write_bytes(b"fake-video")
            keyframe_path.write_bytes(b"fake-keyframe")

            with patch("pipeline_runner.preprocess_file", return_value=self._video_preprocess_with_keyframe(path, keyframe_path)):
                output = run_file_pipeline(
                    self._file_record(path, "video"),
                    self.routing_rules,
                    self.model_prices,
                    vision_understanding_backend="qwen_vl",
                    qwen_vl_api_key="test-key",
                    qwen_vl_max_retries=1,
                )

        result = output["result"]
        visual_calls = [
            call for call in output["model_calls"]
            if call["task_type"] == "visual_understanding"
        ]
        self.assertEqual(result["processing_status"], "partial_success")
        self.assertEqual([call["status"] for call in visual_calls], ["failed", "failed"])
        self.assertIn("visual_description", result["missing_evidence"])
        self.assertIn("video_visual_keyframe_failed", result["quality_flags"])
        self.assertIn("Qwen-VL", result["error_message"])
        self.assertEqual(len([error for error in output["errors"] if error["task_type"] == "visual_understanding"]), 2)

    def test_video_pipeline_uses_extracted_keyframe_but_keeps_audio_missing_in_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            keyframe_path = Path(tmp_dir) / "demo_frame_0001.jpg"
            path.write_bytes(b"fake-video")
            keyframe_path.write_bytes(b"fake-keyframe")
            preprocessed = self._video_preprocess_with_keyframe(path, keyframe_path)

            with patch("pipeline_runner.preprocess_file", return_value=preprocessed):
                output = run_file_pipeline(self._file_record(path, "video"), self.routing_rules, self.model_prices)

        result = output["result"]
        self.assertEqual(result["processing_status"], "partial_success")
        self.assertEqual(result["evidence_used"], ["ocr_text", "visual_description"])
        self.assertEqual(result["missing_evidence"], ["audio_transcript"])
        self.assertIn("video_audio_not_extracted", result["quality_flags"])
        self.assertEqual([call["task_type"] for call in output["model_calls"]], ["ocr", "visual_understanding", "speech_to_text", "text_analysis"])
        self.assertEqual([call["status"] for call in output["model_calls"]], ["success", "success", "failed", "success"])
        speech_call = output["model_calls"][2]
        self.assertEqual(speech_call["input_units"], [{"unit_type": "audio_seconds", "quantity": 0}])
        self.assertEqual(speech_call["cost_cny"], 0.0)

    def test_video_pipeline_uses_extracted_audio_in_mock_asr_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            keyframe_path = Path(tmp_dir) / "demo_frame_0001.jpg"
            audio_path = Path(tmp_dir) / "demo_audio.wav"
            path.write_bytes(b"fake-video")
            keyframe_path.write_bytes(b"fake-keyframe")
            audio_path.write_bytes(b"fake-wav")
            preprocessed = self._video_preprocess_with_keyframe_and_audio(path, keyframe_path, audio_path)

            with patch("pipeline_runner.preprocess_file", return_value=preprocessed):
                output = run_file_pipeline(self._file_record(path, "video"), self.routing_rules, self.model_prices)

        result = output["result"]
        self.assertEqual(result["processing_status"], "partial_success")
        self.assertEqual(result["evidence_used"], ["ocr_text", "visual_description", "audio_transcript"])
        self.assertEqual(result["missing_evidence"], [])
        self.assertIn(VIDEO_EVIDENCE_WEAK_FLAG, result["quality_flags"])
        self.assertNotIn("video_audio_not_extracted", result["quality_flags"])
        self.assertEqual(result["audio_transcript"], "模拟音频转写：demo_audio.wav")
        self.assertEqual([call["task_type"] for call in output["model_calls"]], ["ocr", "visual_understanding", "speech_to_text", "text_analysis"])
        self.assertEqual([call["status"] for call in output["model_calls"]], ["success", "success", "success", "success"])
        speech_call = output["model_calls"][2]
        self.assertEqual(speech_call["input_units"], [{"unit_type": "audio_seconds", "quantity": 2.0}])

    @patch("pipeline_runner.dashscope_asr_client")
    def test_video_pipeline_uses_dashscope_asr_when_url_is_configured(self, mock_asr) -> None:
        mock_asr.return_value = {
            "audio_transcript": "这是真实语音转写。",
            "_api_usage": {"duration": 2},
            "_response_model_name": "paraformer-v2",
            "_response_diagnostics": {},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            keyframe_path = Path(tmp_dir) / "demo_frame_0001.jpg"
            audio_path = Path(tmp_dir) / "demo_audio.wav"
            path.write_bytes(b"fake-video")
            keyframe_path.write_bytes(b"fake-keyframe")
            audio_path.write_bytes(b"fake-wav")
            preprocessed = self._video_preprocess_with_keyframe_and_audio(path, keyframe_path, audio_path)

            with patch("pipeline_runner.preprocess_file", return_value=preprocessed):
                output = run_file_pipeline(
                    self._file_record(path, "video"),
                    self.routing_rules,
                    self.model_prices,
                    speech_to_text_backend="dashscope_asr",
                    dashscope_asr_api_key="test-key",
                    asr_audio_url_map={"demo.mp4": "https://example.test/demo.wav"},
                )

        result = output["result"]
        self.assertEqual(result["audio_transcript"], "这是真实语音转写。")
        self.assertEqual(result["missing_evidence"], [])
        speech_call = [call for call in output["model_calls"] if call["task_type"] == "speech_to_text"][0]
        self.assertEqual(speech_call["provider"], "dashscope")
        self.assertEqual(speech_call["model_name"], "paraformer-v2")
        self.assertEqual(speech_call["status"], "success")
        self.assertEqual(speech_call["cost_cny"], 0.00016)
        mock_asr.assert_called_once_with(
            "https://example.test/demo.wav",
            api_key="test-key",
            model_name="paraformer-v2",
            submit_url="https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
        )

    @patch("pipeline_runner.dashscope_asr_client")
    @patch("pipeline_runner.dashscope_upload_local_file")
    def test_video_pipeline_uploads_local_audio_when_url_map_is_missing(self, mock_upload, mock_asr) -> None:
        mock_upload.return_value = "oss://dashscope-test/demo.wav"
        mock_asr.return_value = {
            "audio_transcript": "这是上传后识别出的语音。",
            "_api_usage": {"duration": 2},
            "_response_model_name": "paraformer-v2",
            "_response_diagnostics": {},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            keyframe_path = Path(tmp_dir) / "demo_frame_0001.jpg"
            audio_path = Path(tmp_dir) / "demo_audio.wav"
            path.write_bytes(b"fake-video")
            keyframe_path.write_bytes(b"fake-keyframe")
            audio_path.write_bytes(b"fake-wav")
            preprocessed = self._video_preprocess_with_keyframe_and_audio(path, keyframe_path, audio_path)

            with patch("pipeline_runner.preprocess_file", return_value=preprocessed):
                output = run_file_pipeline(
                    self._file_record(path, "video"),
                    self.routing_rules,
                    self.model_prices,
                    speech_to_text_backend="dashscope_asr",
                    dashscope_asr_api_key="test-key",
                    asr_audio_url_map={},
                )

        result = output["result"]
        self.assertEqual(result["audio_transcript"], "这是上传后识别出的语音。")
        self.assertEqual(result["missing_evidence"], [])
        mock_upload.assert_called_once_with(
            str(audio_path),
            api_key="test-key",
            model_name="paraformer-v2",
        )
        mock_asr.assert_called_once_with(
            "oss://dashscope-test/demo.wav",
            api_key="test-key",
            model_name="paraformer-v2",
            submit_url="https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription",
        )

    @patch("pipeline_runner.dashscope_asr_client")
    @patch("pipeline_runner.dashscope_upload_local_file")
    def test_video_pipeline_records_missing_asr_upload_dependency_without_calling_api(self, mock_upload, mock_asr) -> None:
        mock_upload.side_effect = RuntimeError("未安装 DashScope SDK/CLI")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            keyframe_path = Path(tmp_dir) / "demo_frame_0001.jpg"
            audio_path = Path(tmp_dir) / "demo_audio.wav"
            path.write_bytes(b"fake-video")
            keyframe_path.write_bytes(b"fake-keyframe")
            audio_path.write_bytes(b"fake-wav")
            preprocessed = self._video_preprocess_with_keyframe_and_audio(path, keyframe_path, audio_path)

            with patch("pipeline_runner.preprocess_file", return_value=preprocessed):
                output = run_file_pipeline(
                    self._file_record(path, "video"),
                    self.routing_rules,
                    self.model_prices,
                    speech_to_text_backend="dashscope_asr",
                    dashscope_asr_api_key="test-key",
                    asr_audio_url_map={},
                )

        result = output["result"]
        self.assertEqual(result["processing_status"], "partial_success")
        self.assertEqual(result["missing_evidence"], ["audio_transcript"])
        self.assertIn(VIDEO_EVIDENCE_WEAK_FLAG, result["quality_flags"])
        speech_call = [call for call in output["model_calls"] if call["task_type"] == "speech_to_text"][0]
        self.assertEqual(speech_call["status"], "failed")
        self.assertEqual(speech_call["input_units"], [{"unit_type": "audio_seconds", "quantity": 0}])
        self.assertEqual(speech_call["cost_cny"], 0.0)
        self.assertIn("DashScope SDK/CLI", speech_call["error_message"])
        mock_asr.assert_not_called()

    def test_video_pipeline_reports_ffmpeg_dependency_missing_when_audio_not_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            keyframe_path = Path(tmp_dir) / "demo_frame_0001.jpg"
            path.write_bytes(b"fake-video")
            keyframe_path.write_bytes(b"fake-keyframe")
            preprocessed = self._video_preprocess_with_keyframe(path, keyframe_path)
            preprocessed["preprocessing_artifacts"]["audio_extraction_status"] = "dependency_missing"

            with patch("pipeline_runner.preprocess_file", return_value=preprocessed):
                output = run_file_pipeline(self._file_record(path, "video"), self.routing_rules, self.model_prices)

        result = output["result"]
        self.assertEqual(result["processing_status"], "partial_success")
        self.assertEqual(result["missing_evidence"], ["audio_transcript"])
        self.assertIn("video_audio_not_extracted", result["quality_flags"])
        self.assertIn("ffmpeg", result["error_message"])

    @patch("pipeline_runner.mock_vision_client")
    @patch("pipeline_runner.mock_ocr_client")
    def test_video_pipeline_aggregates_multiple_keyframes(self, mock_ocr, mock_vision) -> None:
        def fake_ocr(keyframe_path: str) -> dict[str, str]:
            return {"ocr_text": f"OCR文字-{Path(keyframe_path).stem}"}

        def fake_vision(keyframe_path: str) -> dict[str, str]:
            return {"visual_description": f"画面描述-{Path(keyframe_path).stem}"}

        mock_ocr.side_effect = fake_ocr
        mock_vision.side_effect = fake_vision

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            keyframe_paths = [
                Path(tmp_dir) / "demo_frame_0001.jpg",
                Path(tmp_dir) / "demo_frame_0002.jpg",
                Path(tmp_dir) / "demo_frame_0003.jpg",
            ]
            path.write_bytes(b"fake-video")
            for keyframe_path in keyframe_paths:
                keyframe_path.write_bytes(b"fake-keyframe")

            with patch("pipeline_runner.preprocess_file", return_value=self._video_preprocess_with_keyframes(path, keyframe_paths)):
                output = run_file_pipeline(self._file_record(path, "video"), self.routing_rules, self.model_prices)

        result = output["result"]
        self.assertEqual(result["processing_status"], "partial_success")
        self.assertIn("[关键帧 1] OCR文字-demo_frame_0001", result["ocr_text"])
        self.assertIn("[关键帧 3] OCR文字-demo_frame_0003", result["ocr_text"])
        self.assertIn("[关键帧 1] 画面描述-demo_frame_0001", result["visual_description"])
        self.assertIn("[关键帧 3] 画面描述-demo_frame_0003", result["visual_description"])
        self.assertEqual(result["evidence_used"], ["ocr_text", "visual_description"])
        self.assertEqual(result["missing_evidence"], ["audio_transcript"])
        self.assertEqual(
            [call["task_type"] for call in output["model_calls"]],
            [
                "ocr",
                "ocr",
                "ocr",
                "visual_understanding",
                "visual_understanding",
                "visual_understanding",
                "speech_to_text",
                "text_analysis",
            ],
        )
        self.assertEqual(mock_ocr.call_count, 3)
        self.assertEqual(mock_vision.call_count, 3)

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
                deepseek_max_tokens=3000,
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
            max_tokens=3000,
            compact_mode=False,
        )

    @patch("pipeline_runner.deepseek_text_analysis_client")
    def test_deepseek_receives_limited_evidence_and_result_keeps_original_text(self, mock_deepseek) -> None:
        mock_deepseek.return_value = {
            "topic": "technology",
            "secondary_topics": [],
            "tags": ["输入压缩"],
            "summary": "文本分析证据已完成受控压缩。",
            "business_use": "可用于内容归档、检索和人工复核。",
            "_api_usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            "_api_attempts": [
                {
                    "status": "success",
                    "latency_ms": 500,
                    "api_usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                    "error_message": None,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "long.txt"
            original_text = "甲" * 120
            path.write_text(original_text, encoding="utf-8")
            output = run_file_pipeline(
                self._file_record(path, "text"),
                self.routing_rules,
                self.model_prices,
                text_analysis_backend="deepseek",
                deepseek_api_key="test-key",
                deepseek_compact_mode=True,
                text_analysis_evidence_char_limit=20,
            )

        sent_evidence = mock_deepseek.call_args.args[0]
        result = output["result"]
        self.assertEqual(result["raw_text"], original_text)
        self.assertEqual(len(sent_evidence["raw_text"]), 20)
        self.assertTrue(sent_evidence["raw_text"].endswith("…"))
        self.assertEqual(result["processing_status"], "success")
        self.assertIn("text_analysis_evidence_truncated", result["quality_flags"])
        self.assertTrue(any("裁剪" in warning for warning in result["warning_messages"]))
        self.assertTrue(mock_deepseek.call_args.kwargs["compact_mode"])

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
                    "response_diagnostics": {"finish_reason": "length", "hit_max_tokens": False},
                },
                {
                    "status": "failed",
                    "latency_ms": 750,
                    "api_usage": {"prompt_tokens": 100, "completion_tokens": 4},
                    "error_message": str(last_error),
                    "response_diagnostics": {"finish_reason": "length", "hit_max_tokens": False},
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
        self.assertEqual(output["model_calls"][0]["response_diagnostics"]["finish_reason"], "length")
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
