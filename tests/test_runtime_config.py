"""runtime_policy 配置读取测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from file_loader import detect_media_type  # noqa: E402
from model_clients import TOPIC_VALUES, _build_deepseek_messages  # noqa: E402
from preprocessor import DEFAULT_MAX_KEYFRAMES  # noqa: E402
from runtime_config import runtime_policy_section  # noqa: E402


class RuntimeConfigTest(unittest.TestCase):
    def test_file_extensions_are_loaded_from_runtime_policy(self) -> None:
        self.assertEqual(detect_media_type("a.flv"), "video")
        self.assertEqual(detect_media_type("a.jsonl"), "text")
        self.assertEqual(detect_media_type("a.tiff"), "image")

    def test_video_defaults_are_loaded_from_runtime_policy(self) -> None:
        self.assertEqual(DEFAULT_MAX_KEYFRAMES, runtime_policy_section("video_preprocessing")["default_max_keyframes"])

    def test_topic_values_are_loaded_from_runtime_policy(self) -> None:
        configured_topics = set(runtime_policy_section("topics")["values"])
        self.assertEqual(TOPIC_VALUES, configured_topics)

    def test_deepseek_prompt_is_loaded_from_runtime_policy(self) -> None:
        messages = _build_deepseek_messages({"raw_text": "手机游戏性能测试"})
        system_prompt = messages[0]["content"]
        self.assertIn("gaming", system_prompt)
        self.assertIn("游戏如果只是手机、芯片或设备性能测试的负载", system_prompt)


if __name__ == "__main__":
    unittest.main()
