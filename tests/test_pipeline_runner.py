"""pipeline_runner 的测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cost_latency_tracker import load_model_prices
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


if __name__ == "__main__":
    unittest.main()
