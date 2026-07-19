"""file_loader 的测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from file_loader import build_file_manifest, detect_media_type


class FileLoaderTest(unittest.TestCase):
    def test_detect_media_type(self) -> None:
        self.assertEqual(detect_media_type("demo.txt"), "text")
        self.assertEqual(detect_media_type("demo.PNG"), "image")
        self.assertEqual(detect_media_type("demo.mp4"), "video")
        self.assertIsNone(detect_media_type("demo.exe"))

    def test_build_file_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.txt").write_text("hello", encoding="utf-8")
            (root / "b.png").write_bytes(b"fake-image")
            (root / "c.mp4").write_bytes(b"fake-video")
            (root / "ignore.exe").write_bytes(b"ignored")

            manifest = build_file_manifest(
                root,
                batch_id="batch_test",
                created_at="2026-07-14T10:00:00+08:00",
            )

        self.assertEqual(len(manifest), 3)
        self.assertEqual([record["file_id"] for record in manifest], ["file_0001", "file_0002", "file_0003"])
        self.assertEqual({record["media_type"] for record in manifest}, {"text", "image", "video"})
        self.assertTrue(all(record["batch_id"] == "batch_test" for record in manifest))
        self.assertTrue(all(record["created_at"] == "2026-07-14T10:00:00+08:00" for record in manifest))
        self.assertTrue(all(record["file_size_bytes"] > 0 for record in manifest))


if __name__ == "__main__":
    unittest.main()
