"""model_clients 的测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model_clients import (
    TOPIC_VALUES,
    deepseek_text_analysis_client,
    mock_asr_client,
    mock_ocr_client,
    mock_text_analysis_client,
    mock_vision_client,
)


class ModelClientsTest(unittest.TestCase):
    def test_mock_ocr_client(self) -> None:
        result = mock_ocr_client("demo.png")

        self.assertIn("ocr_text", result)
        self.assertIn("demo.png", result["ocr_text"])

    def test_mock_asr_client(self) -> None:
        result = mock_asr_client("demo.wav")

        self.assertIn("audio_transcript", result)
        self.assertIn("demo.wav", result["audio_transcript"])

    def test_mock_vision_client(self) -> None:
        result = mock_vision_client("frame.jpg")

        self.assertIn("visual_description", result)
        self.assertIn("frame.jpg", result["visual_description"])

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


if __name__ == "__main__":
    unittest.main()
