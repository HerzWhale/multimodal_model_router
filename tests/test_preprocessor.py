"""preprocessor 的测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocessor import preprocess_file, preprocess_image, preprocess_text, preprocess_video


class PreprocessorTest(unittest.TestCase):
    def test_preprocess_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.txt"
            path.write_text("测试文本", encoding="utf-8")

            result = preprocess_text(path)

        self.assertEqual(result["raw_text"], "测试文本")
        self.assertIsNone(result["duration_ms"])

    def test_preprocess_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.png"
            path.write_bytes(b"fake-image")

            result = preprocess_image(path)

        self.assertTrue(result["image_path"].endswith("demo.png"))
        self.assertIsNone(result["duration_ms"])

    def test_preprocess_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            path.write_bytes(b"fake-video")

            result = preprocess_video(path)

        self.assertEqual(len(result["keyframes"]), 1)
        self.assertTrue(result["keyframes"][0].endswith("demo_frame_0001.jpg"))
        self.assertTrue(result["audio_path"].endswith("demo.wav"))
        self.assertEqual(result["duration_ms"], 0)

    def test_preprocess_file_dispatches_by_media_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.txt"
            path.write_text("分发测试", encoding="utf-8")

            result = preprocess_file({"media_type": "text", "source_path": str(path)})

        self.assertEqual(result["raw_text"], "分发测试")


if __name__ == "__main__":
    unittest.main()
