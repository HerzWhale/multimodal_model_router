"""result_writer 的测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from result_writer import (
    ensure_batch_output_dir,
    write_batch_metadata,
    write_errors,
    write_json,
    write_jsonl,
    write_model_calls,
    write_results,
    write_results_readable,
)


def decode_concatenated_json_objects(content: str) -> list[dict]:
    """把连续排列的多个 JSON 对象解析成列表。"""

    decoder = json.JSONDecoder()
    records = []
    index = 0

    while index < len(content):
        while index < len(content) and content[index].isspace():
            index += 1
        if index >= len(content):
            break

        record, index = decoder.raw_decode(content, index)
        records.append(record)

    return records


class ResultWriterTest(unittest.TestCase):
    def test_ensure_batch_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_dir = ensure_batch_output_dir(tmp_dir, "batch_001")

            self.assertTrue(batch_dir.exists())
            self.assertEqual(batch_dir.name, "batch_001")

    def test_write_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = write_json(Path(tmp_dir) / "demo.json", {"name": "测试"})
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["name"], "测试")

    def test_write_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = write_jsonl(
                Path(tmp_dir) / "demo.jsonl",
                [{"file_id": "file_001"}, {"file_id": "file_002"}],
            )
            content = path.read_text(encoding="utf-8")
            records = decode_concatenated_json_objects(content)

        self.assertIn('{\n  "file_id"', content)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["file_id"], "file_001")

    def test_write_batch_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            write_batch_metadata(tmp_dir, "batch_001", {"batch_id": "batch_001"})
            write_results(tmp_dir, "batch_001", [{"file_id": "file_001"}])
            write_results_readable(
                tmp_dir,
                "batch_001",
                [
                    {
                        "file_id": "file_001",
                        "file_name": "demo.txt",
                        "media_type": "text",
                        "processing_status": "success",
                        "topic": "technology",
                        "secondary_topics": [],
                        "tags": ["AI"],
                        "summary": "测试摘要",
                        "business_use": "测试用途",
                        "evidence_used": ["raw_text"],
                        "missing_evidence": [],
                        "call_ids": ["call_001"],
                        "models_used": [
                            {
                                "call_id": "call_001",
                                "task_type": "text_analysis",
                                "provider": "deepseek",
                                "model_name": "deepseek-v4-flash",
                                "status": "success",
                            }
                        ],
                        "processing_cost_cny": 0.01,
                        "processing_time_ms": 100,
                        "error_message": None,
                        "warning_messages": [],
                    }
                ],
            )
            write_model_calls(tmp_dir, "batch_001", [{"call_id": "call_001"}])
            write_errors(tmp_dir, "batch_001", [{"error_message": "错误"}])

            batch_dir = Path(tmp_dir) / "batch_001"

            self.assertTrue((batch_dir / "batch_metadata.json").exists())
            self.assertTrue((batch_dir / "results.jsonl").exists())
            self.assertTrue((batch_dir / "results_readable.md").exists())
            self.assertTrue((batch_dir / "model_calls.jsonl").exists())
            self.assertTrue((batch_dir / "errors.jsonl").exists())

            readable_content = (batch_dir / "results_readable.md").read_text(encoding="utf-8")
            self.assertIn("结果 1：file_001 | demo.txt | text", readable_content)
            self.assertIn("text_analysis: deepseek/deepseek-v4-flash", readable_content)


if __name__ == "__main__":
    unittest.main()
