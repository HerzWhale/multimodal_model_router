"""model_clients 的测试。"""

from __future__ import annotations

import base64
import io
from http import client as http_client
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib import error
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model_clients import (
    DEFAULT_DEEPSEEK_MAX_TOKENS,
    DEFAULT_QWEN_OCR_MODEL_NAME,
    DEFAULT_QWEN_VL_MAX_IMAGE_SIDE,
    DEFAULT_QWEN_VL_MAX_TOKENS,
    DEFAULT_QWEN_TEXT_MODEL_NAME,
    DeepSeekAttemptsExhausted,
    PaddleOCRResponseError,
    QwenVLAttemptsExhausted,
    QwenVLResponseError,
    QwenTextAttemptsExhausted,
    QwenOCRResponseError,
    TOPIC_VALUES,
    _create_paddleocr_engine,
    _build_deepseek_messages,
    _build_qwen_vl_messages,
    dashscope_asr_client,
    dashscope_upload_local_file,
    deepseek_text_analysis_client,
    mock_asr_client,
    mock_ocr_client,
    mock_text_analysis_client,
    mock_vision_client,
    paddleocr_client,
    qwen_vl_image_understanding_client,
    qwen_ocr_client,
    qwen_text_analysis_client,
)


class _FakeResponse:
    """模拟 urllib 返回的响应对象。"""

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class _FakePaddleResult:
    """模拟 PaddleOCR Result 对象的 json 属性。"""

    def __init__(self, payload: object) -> None:
        self.json = payload


def _analysis_content(
    *,
    topic: str = "technology",
    include_summary: bool = True,
    business_use: str = "可用于技术内容归档。",
) -> str:
    """构造 DeepSeek 模型内容。"""

    data = {
        "topic": topic,
        "secondary_topics": ["knowledge"],
        "tags": ["AI工程"],
        "summary": "这是一条技术内容摘要。",
        "business_use": business_use,
    }
    if not include_summary:
        data.pop("summary")
    return json.dumps(data, ensure_ascii=False)


def _api_response(
    content: str,
    *,
    usage: dict[str, int] | None = None,
    finish_reason: str | None = None,
    model: str | None = None,
) -> _FakeResponse:
    """构造包含 token 用量的 DeepSeek API 外层响应。"""

    body = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    if model:
        body["model"] = model
    return _FakeResponse(json.dumps(body, ensure_ascii=False).encode("utf-8"))


def _json_response(data: dict) -> _FakeResponse:
    """构造普通 JSON 响应。"""

    return _FakeResponse(json.dumps(data, ensure_ascii=False).encode("utf-8"))


def _qwen_vl_api_response(content: str) -> _FakeResponse:
    """构造包含 token 用量的 Qwen-VL API 外层响应。"""

    body = {
        "model": "qwen-vl-plus",
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 300, "completion_tokens": 80, "total_tokens": 380},
    }
    return _FakeResponse(json.dumps(body, ensure_ascii=False).encode("utf-8"))


def _qwen_ocr_api_response(content: object) -> _FakeResponse:
    body = {
        "model": DEFAULT_QWEN_OCR_MODEL_NAME,
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 240, "completion_tokens": 24, "total_tokens": 264},
    }
    return _FakeResponse(json.dumps(body, ensure_ascii=False).encode("utf-8"))


class ModelClientsTest(unittest.TestCase):
    def test_qwen_ocr_client_requires_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY"):
            qwen_ocr_client("missing.png", api_key=None)

    @patch("model_clients.request.urlopen")
    def test_qwen_ocr_client_sends_original_image_and_records_usage(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _qwen_ocr_api_response("第一行\n第二行")
        original = b"original-image-bytes"
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "demo.png"
            image_path.write_bytes(original)
            result = qwen_ocr_client(image_path, api_key="test-key")

        payload = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        data_url = payload["messages"][0]["content"][0]["image_url"]["url"]
        self.assertEqual(payload["model"], DEFAULT_QWEN_OCR_MODEL_NAME)
        self.assertEqual(base64.b64decode(data_url.split(",", 1)[1]), original)
        self.assertEqual(result["ocr_text"], "第一行\n第二行")
        self.assertEqual(result["_api_usage"]["total_tokens"], 264)
        self.assertEqual(result["_response_model_name"], DEFAULT_QWEN_OCR_MODEL_NAME)

    @patch("model_clients.request.urlopen")
    def test_qwen_ocr_client_rejects_invalid_outer_json(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse(b"not-json")
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "demo.png"
            image_path.write_bytes(b"image")
            with self.assertRaisesRegex(QwenOCRResponseError, "qwen_ocr_response_invalid_json"):
                qwen_ocr_client(image_path, api_key="test-key")

    @patch("model_clients.request.urlopen")
    def test_qwen_ocr_client_rejects_empty_content(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _qwen_ocr_api_response("")
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "demo.png"
            image_path.write_bytes(b"image")
            with self.assertRaisesRegex(QwenOCRResponseError, "qwen_ocr_content_empty"):
                qwen_ocr_client(image_path, api_key="test-key")

    def test_mock_ocr_client(self) -> None:
        result = mock_ocr_client("demo.png")

        self.assertIn("ocr_text", result)
        self.assertIn("demo.png", result["ocr_text"])

    def test_paddleocr_client_rejects_missing_image(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "不存在"):
            paddleocr_client("missing.png")

    @patch("model_clients._decode_image_for_paddleocr", return_value="decoded-image")
    @patch("model_clients._create_paddleocr_engine")
    def test_paddleocr_client_merges_recognized_lines(
        self,
        mock_create_engine,
        _mock_decode_image,
    ) -> None:
        mock_create_engine.return_value.predict.return_value = [
            {"res": {"rec_texts": [" 第一行 ", "", "第二行"]}}
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "demo.png"
            image_path.write_bytes(b"real-image-bytes")
            result = paddleocr_client(image_path)

        self.assertEqual(result["ocr_text"], "第一行\n第二行")
        mock_create_engine.return_value.predict.assert_called_once_with("decoded-image")

    @patch("model_clients._decode_image_for_paddleocr", return_value="decoded-image")
    @patch("model_clients._create_paddleocr_engine")
    def test_paddleocr_client_reads_result_object_json(
        self,
        mock_create_engine,
        _mock_decode_image,
    ) -> None:
        mock_create_engine.return_value.predict.return_value = [
            _FakePaddleResult({"res": {"rec_texts": ["识别文字"]}})
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "result_object.png"
            image_path.write_bytes(b"image")
            result = paddleocr_client(image_path)

        self.assertEqual(result["ocr_text"], "识别文字")

    @patch("model_clients._decode_image_for_paddleocr", return_value="decoded-image")
    @patch("model_clients._create_paddleocr_engine")
    def test_paddleocr_client_returns_null_when_no_text(
        self,
        mock_create_engine,
        _mock_decode_image,
    ) -> None:
        mock_create_engine.return_value.predict.return_value = [{"res": {"rec_texts": []}}]

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "no_text.png"
            image_path.write_bytes(b"image")
            result = paddleocr_client(image_path)

        self.assertIsNone(result["ocr_text"])

    @patch("model_clients._decode_image_for_paddleocr", return_value="decoded-image")
    @patch("model_clients._create_paddleocr_engine")
    def test_paddleocr_client_wraps_inference_failure(
        self,
        mock_create_engine,
        _mock_decode_image,
    ) -> None:
        mock_create_engine.return_value.predict.side_effect = RuntimeError("模型加载失败")

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "bad.png"
            image_path.write_bytes(b"bad-image")
            with self.assertRaisesRegex(PaddleOCRResponseError, "模型加载失败"):
                paddleocr_client(image_path)

    @patch("model_clients._decode_image_for_paddleocr", return_value="decoded-image")
    @patch("model_clients._create_paddleocr_engine")
    def test_paddleocr_client_rejects_invalid_rec_texts(
        self,
        mock_create_engine,
        _mock_decode_image,
    ) -> None:
        mock_create_engine.return_value.predict.return_value = [
            {"res": {"rec_texts": "不是数组"}}
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "bad_schema.png"
            image_path.write_bytes(b"image")
            with self.assertRaisesRegex(PaddleOCRResponseError, "rec_texts"):
                paddleocr_client(image_path)

    def test_paddleocr_engine_disables_mkldnn_on_windows_cpu_path(self) -> None:
        fake_constructor = Mock(return_value=object())
        _create_paddleocr_engine.cache_clear()

        with patch.dict(
            sys.modules,
            {"paddleocr": SimpleNamespace(PaddleOCR=fake_constructor)},
        ):
            _create_paddleocr_engine()

        fake_constructor.assert_called_once_with(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False,
        )
        _create_paddleocr_engine.cache_clear()

    def test_mock_asr_client(self) -> None:
        result = mock_asr_client("demo.wav")

        self.assertIn("audio_transcript", result)
        self.assertIn("demo.wav", result["audio_transcript"])

    @patch("model_clients.request.urlopen")
    def test_dashscope_asr_client_reads_transcript_url(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = [
            _json_response({"output": {"task_id": "task_001"}}),
            _json_response(
                {
                    "output": {
                        "task_status": "SUCCEEDED",
                        "results": [
                            {
                                "subtask_status": "SUCCEEDED",
                                "transcription_url": "https://example.test/transcript.json",
                            }
                        ],
                    },
                    "usage": {"duration": 2},
                }
            ),
            _json_response({"transcripts": [{"text": "第一句。"}, {"text": "第二句。"}]}),
        ]

        result = dashscope_asr_client(
            "https://example.test/demo.wav",
            api_key="test-key",
            poll_interval_seconds=0,
        )

        self.assertEqual(result["audio_transcript"], "第一句。\n第二句。")
        self.assertEqual(result["_api_usage"], {"duration": 2})
        self.assertEqual(result["_response_model_name"], "paraformer-v2")
        self.assertEqual(mock_urlopen.call_count, 3)

    @patch("model_clients.request.urlopen")
    def test_dashscope_asr_client_accepts_oss_url(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = [
            _json_response({"output": {"task_id": "task_001"}}),
            _json_response(
                {
                    "output": {
                        "task_status": "SUCCEEDED",
                        "results": [
                            {
                                "subtask_status": "SUCCEEDED",
                                "transcription_url": "https://example.test/transcript.json",
                            }
                        ],
                    },
                    "usage": {"duration": 2},
                }
            ),
            _json_response({"text": "转写成功。"}),
        ]

        result = dashscope_asr_client(
            "oss://dashscope-test/demo.wav",
            api_key="test-key",
            poll_interval_seconds=0,
        )

        submit_request = mock_urlopen.call_args_list[0].args[0]
        headers = {key.lower(): value for key, value in submit_request.header_items()}
        self.assertEqual(result["audio_transcript"], "转写成功。")
        self.assertEqual(headers["x-dashscope-ossresourceresolve"], "enable")

    @patch("dashscope.utils.oss_utils.OssUtils.upload")
    def test_dashscope_upload_local_file_reads_oss_url(self, mock_upload) -> None:
        mock_upload.return_value = ("oss://dashscope-test/demo.wav", {"policy": "fake"})
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "demo.wav"
            audio_path.write_bytes(b"fake-wav")

            result = dashscope_upload_local_file(audio_path, api_key="test-key")

        self.assertEqual(result, "oss://dashscope-test/demo.wav")
        mock_upload.assert_called_once()
        self.assertEqual(mock_upload.call_args.kwargs["model"], "paraformer-v2")
        self.assertEqual(mock_upload.call_args.kwargs["api_key"], "test-key")

    @patch("dashscope.utils.oss_utils.OssUtils.upload")
    def test_dashscope_upload_local_file_reports_sdk_error_without_api_key(self, mock_upload) -> None:
        mock_upload.side_effect = RuntimeError("bad key test-key-secret")
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "demo.wav"
            audio_path.write_bytes(b"fake-wav")

            with self.assertRaisesRegex(Exception, "bad key \\*\\*\\*-secret"):
                dashscope_upload_local_file(audio_path, api_key="test-key")

    def test_mock_vision_client(self) -> None:
        result = mock_vision_client("frame.jpg")

        self.assertIn("visual_description", result)
        self.assertIn("frame.jpg", result["visual_description"])

    def test_qwen_vl_client_requires_api_key(self) -> None:
        with self.assertRaises(ValueError):
            qwen_vl_image_understanding_client("demo.png", api_key=None)

    def test_qwen_vl_prompt_uses_base64_image_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "demo.png"
            image_path.write_bytes(b"image-bytes")

            messages = _build_qwen_vl_messages(image_path)

        image_part = messages[1]["content"][0]
        text_part = messages[1]["content"][1]
        self.assertEqual(image_part["type"], "image_url")
        self.assertTrue(image_part["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertIn("visual_description", text_part["text"])

    @patch("model_clients.request.urlopen")
    def test_qwen_vl_client_parses_valid_json_and_usage(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _qwen_vl_api_response(
            json.dumps({"visual_description": "图片展示一张内容平台信息图。"}, ensure_ascii=False)
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "demo.png"
            image_path.write_bytes(b"image-bytes")
            result = qwen_vl_image_understanding_client(
                image_path,
                api_key="test-key",
                model_name="qwen-vl-plus",
            )

        api_request = mock_urlopen.call_args.args[0]
        payload = json.loads(api_request.data.decode("utf-8"))
        image_url = payload["messages"][1]["content"][0]["image_url"]["url"]

        self.assertEqual(result["visual_description"], "图片展示一张内容平台信息图。")
        self.assertEqual(result["_api_usage"]["total_tokens"], 380)
        self.assertEqual(result["_response_model_name"], "qwen-vl-plus")
        self.assertEqual([item["status"] for item in result["_api_attempts"]], ["success"])
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        self.assertEqual(payload["max_tokens"], DEFAULT_QWEN_VL_MAX_TOKENS)

    @patch("model_clients.request.urlopen")
    def test_qwen_vl_client_accepts_custom_max_tokens(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _qwen_vl_api_response(
            json.dumps({"visual_description": "图片展示一张内容平台信息图。"}, ensure_ascii=False)
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "demo.png"
            image_path.write_bytes(b"image-bytes")
            qwen_vl_image_understanding_client(image_path, api_key="test-key", max_tokens=321)

        payload = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["max_tokens"], 321)

    @patch("model_clients.request.urlopen")
    def test_qwen_vl_client_limits_image_side_before_request(self, mock_urlopen) -> None:
        from PIL import Image

        mock_urlopen.return_value = _qwen_vl_api_response(
            json.dumps({"visual_description": "图片已压缩后发送。"}, ensure_ascii=False)
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "large.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            qwen_vl_image_understanding_client(image_path, api_key="test-key", max_image_side=50)

        payload = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        data_url = payload["messages"][1]["content"][0]["image_url"]["url"]
        image_bytes = base64.b64decode(data_url.split(",", 1)[1])
        with Image.open(io.BytesIO(image_bytes)) as image:
            self.assertEqual(max(image.size), 50)

    def test_qwen_vl_client_rejects_invalid_max_tokens_before_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            qwen_vl_image_understanding_client("missing.png", api_key="test-key", max_tokens=0)

    def test_qwen_vl_client_rejects_invalid_max_image_side_before_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_image_side"):
            qwen_vl_image_understanding_client("missing.png", api_key="test-key", max_image_side=0)

    @patch("model_clients.request.urlopen")
    def test_qwen_vl_client_rejects_invalid_json_content(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _qwen_vl_api_response("不是 JSON")

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "demo.png"
            image_path.write_bytes(b"image-bytes")
            with self.assertRaisesRegex(QwenVLResponseError, "qwen_vl_content_invalid_json"):
                qwen_vl_image_understanding_client(image_path, api_key="test-key")

    @patch("model_clients.request.urlopen")
    def test_qwen_vl_client_retries_remote_disconnected_once_then_succeeds(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = [
            http_client.RemoteDisconnected("Remote end closed connection without response"),
            _qwen_vl_api_response(
                json.dumps({"visual_description": "关键帧展示内容平台短视频画面。"}, ensure_ascii=False)
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.jpg"
            image_path.write_bytes(b"image-bytes")
            result = qwen_vl_image_understanding_client(
                image_path,
                api_key="test-key",
                max_retries=1,
            )

        self.assertEqual(result["visual_description"], "关键帧展示内容平台短视频画面。")
        self.assertEqual([item["status"] for item in result["_api_attempts"]], ["failed", "success"])
        self.assertIn("qwen_vl_network_disconnected", result["_api_attempts"][0]["error_message"])
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("model_clients.request.urlopen")
    def test_qwen_vl_client_does_not_retry_authentication_error(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = error.HTTPError(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            401,
            "Unauthorized",
            None,
            io.BytesIO(b"secret detail"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.jpg"
            image_path.write_bytes(b"image-bytes")
            with self.assertRaises(QwenVLAttemptsExhausted) as context:
                qwen_vl_image_understanding_client(
                    image_path,
                    api_key="test-key",
                    max_retries=1,
                )

        self.assertIn("HTTP 401", str(context.exception))
        self.assertNotIn("secret detail", str(context.exception))
        self.assertEqual(len(context.exception.attempts), 1)
        self.assertEqual(mock_urlopen.call_count, 1)

    def test_mock_text_analysis_client(self) -> None:
        result = mock_text_analysis_client({"raw_text": "这是一段 AI 工具教程"})

        self.assertIn(result["topic"], TOPIC_VALUES)
        self.assertIn("AI团队", result["tags"])
        self.assertTrue(result["summary"])
        self.assertTrue(result["business_use"])

    def test_mock_text_analysis_client_uses_content_keywords(self) -> None:
        result = mock_text_analysis_client(
            {
                "raw_text": "AI 团队需要处理多模态素材，记录模型调用成本、延迟和供应商表现。"
            }
        )

        self.assertEqual(result["topic"], "technology")
        self.assertIn("finance_business", result["secondary_topics"])
        self.assertIn("多模态处理", result["tags"])
        self.assertIn("成本核算", result["tags"])
        self.assertIn("现有文本证据", result["summary"])

    def test_deepseek_text_analysis_client_requires_api_key(self) -> None:
        with self.assertRaises(ValueError):
            deepseek_text_analysis_client({"raw_text": "AI 内容分析"}, api_key=None)

    def test_qwen_text_analysis_client_requires_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY"):
            qwen_text_analysis_client({"raw_text": "AI 内容分析"}, api_key=None)

    @patch("model_clients.request.urlopen")
    def test_qwen_text_client_uses_fixed_model_and_records_response(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _api_response(
            _analysis_content(),
            model=DEFAULT_QWEN_TEXT_MODEL_NAME,
        )

        result = qwen_text_analysis_client({"raw_text": "AI 内容分析"}, api_key="test-key")

        payload = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["model"], DEFAULT_QWEN_TEXT_MODEL_NAME)
        self.assertFalse(payload["enable_thinking"])
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["max_completion_tokens"], 1600)
        self.assertEqual(result["_api_usage"]["total_tokens"], 150)
        self.assertEqual(result["_api_attempts"][0]["response_model_name"], DEFAULT_QWEN_TEXT_MODEL_NAME)

    @patch("model_clients.request.urlopen")
    def test_qwen_text_client_rejects_invalid_json(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _api_response("不是 JSON")

        with self.assertRaises(QwenTextAttemptsExhausted) as context:
            qwen_text_analysis_client({"raw_text": "AI 内容分析"}, api_key="test-key")

        self.assertIn("qwen_text_content_invalid_json", str(context.exception))

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_parses_valid_json_and_tracks_attempt(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _api_response(_analysis_content(), model="deepseek-v4-flash")

        result = deepseek_text_analysis_client({"raw_text": "AI 内容分析"}, api_key="test-key")

        self.assertEqual(result["topic"], "technology")
        self.assertEqual(result["_api_usage"]["total_tokens"], 150)
        self.assertEqual([item["status"] for item in result["_api_attempts"]], ["success"])
        self.assertEqual(result["_api_attempts"][0]["response_model_name"], "deepseek-v4-flash")
        sent_payload = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(sent_payload["max_tokens"], DEFAULT_DEEPSEEK_MAX_TOKENS)

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_accepts_custom_max_tokens(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _api_response(_analysis_content())

        deepseek_text_analysis_client({"raw_text": "AI 内容分析"}, api_key="test-key", max_tokens=3000)

        sent_payload = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(sent_payload["max_tokens"], 3000)

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_compact_mode_uses_compact_prompt_first(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _api_response(_analysis_content())

        deepseek_text_analysis_client(
            {"visual_description": "前几秒有效内容。" * 200},
            api_key="test-key",
            compact_mode=True,
        )

        sent_payload = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertIn("紧凑输出要求", sent_payload["messages"][0]["content"])
        self.assertIn("……", sent_payload["messages"][1]["content"])

    def test_deepseek_client_rejects_invalid_max_tokens_before_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            deepseek_text_analysis_client({"raw_text": "AI 内容分析"}, api_key="test-key", max_tokens=0)

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_accepts_complete_json_code_fence(self, mock_urlopen) -> None:
        fenced_content = f"```json\n{_analysis_content()}\n```"
        mock_urlopen.return_value = _api_response(fenced_content)

        result = deepseek_text_analysis_client({"raw_text": "AI 内容分析"}, api_key="test-key")

        self.assertEqual(result["topic"], "technology")

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_classifies_invalid_json_without_raw_content(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _api_response("这不是 JSON 内容")

        with self.assertRaises(DeepSeekAttemptsExhausted) as context:
            deepseek_text_analysis_client({"raw_text": "AI 内容分析"}, api_key="test-key")

        self.assertIn("deepseek_content_invalid_json", str(context.exception))
        self.assertIn("长度", str(context.exception))
        self.assertNotIn("这不是 JSON 内容", str(context.exception))
        self.assertEqual(len(context.exception.attempts), 1)

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_classifies_empty_content(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _api_response("")

        with self.assertRaises(DeepSeekAttemptsExhausted) as context:
            deepseek_text_analysis_client({"raw_text": "AI 内容分析"}, api_key="test-key")

        self.assertIn("deepseek_content_empty", str(context.exception))

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_records_diagnostics_when_empty_content_hits_max_tokens(
        self,
        mock_urlopen,
    ) -> None:
        mock_urlopen.return_value = _api_response(
            "   \n\t",
            usage={"prompt_tokens": 1091, "completion_tokens": DEFAULT_DEEPSEEK_MAX_TOKENS, "total_tokens": 2591},
            finish_reason="length",
            model="deepseek-v4-flash",
        )

        with self.assertRaises(DeepSeekAttemptsExhausted) as context:
            deepseek_text_analysis_client({"raw_text": "AI 内容分析"}, api_key="test-key")

        attempt = context.exception.attempts[0]
        diagnostics = attempt["response_diagnostics"]
        self.assertIn("max_tokens", str(context.exception))
        self.assertEqual(attempt["api_usage"]["completion_tokens"], DEFAULT_DEEPSEEK_MAX_TOKENS)
        self.assertEqual(diagnostics["response_model_name"], "deepseek-v4-flash")
        self.assertEqual(diagnostics["finish_reason"], "length")
        self.assertEqual(diagnostics["raw_content_length"], 5)
        self.assertEqual(diagnostics["stripped_content_length"], 0)
        self.assertEqual(diagnostics["completion_tokens"], DEFAULT_DEEPSEEK_MAX_TOKENS)
        self.assertEqual(diagnostics["max_tokens"], DEFAULT_DEEPSEEK_MAX_TOKENS)
        self.assertTrue(diagnostics["hit_max_tokens"])
        self.assertIn("content", diagnostics["message_keys"])
        self.assertEqual(diagnostics["content_preview"], "   \\n\\t")

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_retries_once_then_succeeds(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = [
            _api_response("不是 JSON"),
            _api_response(_analysis_content()),
        ]

        result = deepseek_text_analysis_client(
            {"raw_text": "AI 内容分析"},
            api_key="test-key",
            max_retries=1,
        )

        self.assertEqual([item["status"] for item in result["_api_attempts"]], ["failed", "success"])
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_compacts_prompt_after_empty_max_tokens(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = [
            _api_response(
                "   ",
                usage={"prompt_tokens": 1000, "completion_tokens": DEFAULT_DEEPSEEK_MAX_TOKENS, "total_tokens": 2500},
                finish_reason="length",
            ),
            _api_response(_analysis_content()),
        ]

        result = deepseek_text_analysis_client(
            {"visual_description": "前几秒有效内容。" * 200},
            api_key="test-key",
            max_retries=1,
        )

        first_payload = json.loads(mock_urlopen.call_args_list[0].args[0].data.decode("utf-8"))
        second_payload = json.loads(mock_urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual(result["topic"], "technology")
        self.assertEqual([item["status"] for item in result["_api_attempts"]], ["failed", "success"])
        self.assertNotIn("紧凑输出要求", first_payload["messages"][0]["content"])
        self.assertIn("紧凑输出要求", second_payload["messages"][0]["content"])
        self.assertIn("……", second_payload["messages"][1]["content"])

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_does_not_retry_authentication_error(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = error.HTTPError(
            "https://api.deepseek.com/chat/completions",
            401,
            "Unauthorized",
            None,
            io.BytesIO(b"secret detail"),
        )

        with self.assertRaises(DeepSeekAttemptsExhausted) as context:
            deepseek_text_analysis_client(
                {"raw_text": "AI 内容分析"},
                api_key="test-key",
                max_retries=1,
            )

        self.assertIn("HTTP 401", str(context.exception))
        self.assertNotIn("secret detail", str(context.exception))
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_rejects_missing_required_field(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _api_response(_analysis_content(include_summary=False))

        with self.assertRaises(DeepSeekAttemptsExhausted) as context:
            deepseek_text_analysis_client({"raw_text": "AI 内容分析"}, api_key="test-key")

        self.assertIn("deepseek_content_invalid_schema", str(context.exception))
        self.assertIn("summary", str(context.exception))

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_rejects_topic_outside_taxonomy(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _api_response(_analysis_content(topic="invalid_topic"))

        with self.assertRaises(DeepSeekAttemptsExhausted) as context:
            deepseek_text_analysis_client({"raw_text": "AI 内容分析"}, api_key="test-key")

        self.assertIn("deepseek_content_invalid_schema", str(context.exception))
        self.assertIn("topic 不属于既定分类", str(context.exception))

    def test_deepseek_prompt_prefers_business_context_over_tech_terms(self) -> None:
        messages = _build_deepseek_messages(
            {
                "raw_text": "AI 芯片公司发布财报，营收增长但投资风险上升。",
            }
        )

        system_prompt = messages[0]["content"]

        self.assertIn("先判断内容主要业务场景", system_prompt)
        self.assertIn("即使提到 AI、芯片、软件，也优先选择 finance_business", system_prompt)
        self.assertIn("功能发布、技术教程、产品能力或工程实现为核心", system_prompt)

    def test_deepseek_prompt_contains_complete_topic_policy(self) -> None:
        """检查真实模型提示词实现已经确认的主分类规则。"""

        messages = _build_deepseek_messages({"raw_text": "用于检查提示词规则。"})
        system_prompt = messages[0]["content"]

        expected_policy_fragments = [
            "1. ads_marketing：内容整体以品牌广告",
            "2. news：以报道具体事件为核心",
            "3. finance_business：财经分析",
            "4. technology：以 AI、数码、软件、硬件、汽车科技",
            "5. gaming：以游戏玩法、剧情、主播实况",
            "6. sports_health：以体育赛事、运动、健身",
            "7. entertainment：以娱乐圈、影视作品、综艺节目、明星艺人、粉丝互动、演唱会、线下见面会",
            "8. humor：以搞笑段子、生活趣事、意外反转",
            "9. lifestyle：以个人经历、vlog、自拍、美食、旅行、穿搭",
            "10. knowledge：面向一般受众解释可迁移的概念",
            "11. other：以上十类都不符合时使用",
            "领域优先于讲解形式",
            "secondary_topics 也必须遵守领域优先规则",
            "不能只因为内容采用讲解、教程、科普或历史梳理形式",
            "不要为了避免 other 而强行选择相邻类别",
        ]

        for fragment in expected_policy_fragments:
            self.assertIn(fragment, system_prompt)

    def test_deepseek_prompt_handles_video_secondary_topic_boundaries(self) -> None:
        """检查视频证据中的背景词不会被误当成主分类或副分类。"""

        messages = _build_deepseek_messages({"visual_description": "用于检查视频分类边界规则。"})
        system_prompt = messages[0]["content"]

        expected_policy_fragments = [
            "secondary_topics 必须是内容实质讨论对象",
            "平台界面、搜索页、点赞评论数据、测试工具名、背景道具或单个对象词",
            "游戏如果只是手机、芯片或设备性能测试的负载，不应加入 gaming 或 entertainment",
            "手机屏幕、应用界面、搜索页、二维码、点赞评论数、播放控件等平台 UI 证据",
            "普通音乐、合唱、校园晚会、舞台记录或线下演出片段",
            "secondary_topics 也不要填 entertainment",
            "突发趣事、滑稽反转或搞笑效果",
            "topic 应选择 humor",
        ]

        for fragment in expected_policy_fragments:
            self.assertIn(fragment, system_prompt)

        for file_name in ["例子.mp4", "例子2.mp4", "例子3.mp4"]:
            self.assertNotIn(file_name, system_prompt)

    def test_deepseek_prompt_handles_ads_and_entertainment_boundaries(self) -> None:
        messages = _build_deepseek_messages({"visual_description": "用于检查广告植入和娱乐活动边界。"})
        system_prompt = messages[0]["content"]

        self.assertIn("仅在长视频或评论内容中插入一段口播广告", system_prompt)
        self.assertIn("不应把整条内容主分类改为 ads_marketing", system_prompt)
        self.assertIn("明星艺人、粉丝互动、演唱会、线下见面会、舞台活动", system_prompt)
        self.assertIn("即使中间出现广告植入，也不应让 ads_marketing 覆盖主体分类", system_prompt)

    def test_deepseek_prompt_keeps_plain_ceremony_record_out_of_knowledge(self) -> None:
        messages = _build_deepseek_messages({"visual_description": "用于检查普通民俗婚礼现场记录边界。"})
        system_prompt = messages[0]["content"]

        self.assertIn("仅记录某个民俗、婚礼、活动、仪式或现场片段", system_prompt)
        self.assertIn("不应因为出现“传统”“习俗”“文化”等词就归为 knowledge", system_prompt)
        self.assertIn("普通仪式记录或缺少实质信息增量的现场片段", system_prompt)
        self.assertNotIn("抖音2026810-211642", system_prompt)

    def test_deepseek_prompt_does_not_copy_evaluation_answers(self) -> None:
        """检查提示词使用通用边界规则，而不是复制现有错例答案。"""

        messages = _build_deepseek_messages({"raw_text": "用于检查提示词规则。"})
        system_prompt = messages[0]["content"]

        evaluation_specific_phrases = [
            "校园失物",
            "社区共享工具",
            "电影节红毯",
            "控糖配料表",
        ]

        for phrase in evaluation_specific_phrases:
            self.assertNotIn(phrase, system_prompt)

    def test_deepseek_prompt_requires_business_use_evidence(self) -> None:
        """检查提示词要求业务用途建立在输入证据之上。"""

        messages = _build_deepseek_messages({"raw_text": "用于检查业务用途规则。"})
        system_prompt = messages[0]["content"]

        self.assertIn("不得编造用户增长、收入提升、转化效果或算法效果", system_prompt)
        self.assertIn("只有证据明确出现品牌合作、广告合作、购买入口、下单、促销或带货时", system_prompt)
        self.assertIn("可用于内容归档、检索和人工复核", system_prompt)

    def test_deepseek_prompt_limits_normal_output_length(self) -> None:
        """检查正常调用也有短输出约束，避免文本分析结果过长。"""

        messages = _build_deepseek_messages({"raw_text": "用于检查输出长度规则。"})
        system_prompt = messages[0]["content"]

        self.assertIn("只输出一个 JSON 对象", system_prompt)
        self.assertIn("summary 控制在 120 字以内", system_prompt)
        self.assertIn("business_use 控制在 40 字以内", system_prompt)
        self.assertIn("不输出 Markdown、解释过程、证据逐段复述或额外字段", system_prompt)

    def test_deepseek_compact_prompt_uses_stricter_output_length(self) -> None:
        """检查紧凑模式进一步收紧输出，供视频批次降延迟使用。"""

        messages = _build_deepseek_messages({"raw_text": "用于检查紧凑输出规则。"}, compact=True)
        system_prompt = messages[0]["content"]

        self.assertIn("紧凑输出要求", system_prompt)
        self.assertIn("summary 控制在 80 字以内", system_prompt)
        self.assertIn("business_use 控制在 30 字以内", system_prompt)
        self.assertIn("不要复述证据细节", system_prompt)

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_replaces_unsupported_commercial_use(self, mock_urlopen) -> None:
        """没有商业证据时，高风险商业用途必须降级为保守用途。"""

        mock_urlopen.return_value = _api_response(
            _analysis_content(business_use="可关联运动营养品进行品牌推广。")
        )

        result = deepseek_text_analysis_client(
            {"raw_text": "这是一份马拉松补给与配速建议。"},
            api_key="test-key",
        )

        self.assertEqual(result["business_use"], "可用于内容归档、检索和人工复核。")
        self.assertEqual(result["_quality_flags"], ["business_use_grounded_fallback"])

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_keeps_commercial_use_with_explicit_evidence(self, mock_urlopen) -> None:
        """输入明确包含合作和购买证据时，可以保留相应商业用途。"""

        commercial_use = "可用于品牌推广和购买转化。"
        mock_urlopen.return_value = _api_response(_analysis_content(business_use=commercial_use))

        result = deepseek_text_analysis_client(
            {"raw_text": "本期是品牌合作内容，视频下方提供购买链接和优惠券。"},
            api_key="test-key",
        )

        self.assertEqual(result["business_use"], commercial_use)
        self.assertEqual(result["_quality_flags"], [])

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_does_not_treat_negated_terms_as_commercial_evidence(self, mock_urlopen) -> None:
        """否定表达不能被误判为支持广告或转化建议的正向证据。"""

        mock_urlopen.return_value = _api_response(
            _analysis_content(business_use="可用于广告投放和商品推广。")
        )

        result = deepseek_text_analysis_client(
            {"raw_text": "这不是广告，也没有品牌合作、购买入口或推广。"},
            api_key="test-key",
        )

        self.assertEqual(result["business_use"], "可用于内容归档、检索和人工复核。")
        self.assertEqual(result["_quality_flags"], ["business_use_grounded_fallback"])

    @patch("model_clients.request.urlopen")
    def test_deepseek_client_keeps_noncommercial_business_use(self, mock_urlopen) -> None:
        """普通归档和检索用途不应被防护逻辑改写。"""

        business_use = "可用于体育健康内容归档和人工复核。"
        mock_urlopen.return_value = _api_response(_analysis_content(business_use=business_use))

        result = deepseek_text_analysis_client(
            {"raw_text": "这是一份马拉松补给建议。"},
            api_key="test-key",
        )

        self.assertEqual(result["business_use"], business_use)
        self.assertEqual(result["_quality_flags"], [])


if __name__ == "__main__":
    unittest.main()
