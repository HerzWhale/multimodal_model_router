"""rapidocr_candidate_evaluator 的离线测试。"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from rapidocr_candidate_evaluator import (  # noqa: E402
    _parse_rapidocr_result,
    check_rapidocr_dependency,
    run_rapidocr_candidate_evaluation,
    write_rapidocr_candidate_report,
)


def _gold_row(file_name: str, segment_id: str, gold_text: str) -> dict[str, str]:
    """构造一条OCR人工基准记录。"""

    return {
        "file_name": file_name,
        "segment_id": segment_id,
        "segment_type": "diagram_label",
        "gold_text": gold_text,
        "is_required": "true",
        "evaluation_scope": "business_content",
        "reviewer_note": "RapidOCR候选测试",
    }


def _write_gold(path: Path) -> None:
    """写出测试用OCR人工基准。"""

    rows = [
        _gold_row("img_7.jpg", "title", "SPEC 2017"),
        _gold_row("img_7.jpg", "cpu", "Kirin 9010"),
        _gold_row("img_8.jpg", "core", "TaiShan V121"),
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RapidOcrCandidateEvaluatorTest(unittest.TestCase):
    def test_dependency_missing_report_does_not_fabricate_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            gold_path = tmp_path / "gold.csv"
            _write_gold(gold_path)

            with patch(
                "rapidocr_candidate_evaluator.check_rapidocr_dependency",
                return_value={
                    "status": "missing",
                    "packages": {"rapidocr": False, "rapidocr_onnxruntime": False, "onnxruntime": False},
                    "install_hint": "pip install rapidocr onnxruntime",
                },
            ):
                report = run_rapidocr_candidate_evaluation(
                    image_paths=[tmp_path / "img_7.jpg"],
                    gold_path=gold_path,
                )

        self.assertEqual(report["gate_decision"]["status"], "dependency_missing")
        self.assertIsNone(report["overview"]["overall_exact_segment_recall"])
        self.assertIsNone(report["overview"]["ocr_p95_latency_ms"])
        self.assertEqual(report["file_metrics"][0]["status"], "skipped_dependency_missing")

    def test_fake_ocr_client_generates_candidate_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            gold_path = tmp_path / "gold.csv"
            image_7 = tmp_path / "img_7.jpg"
            image_8 = tmp_path / "img_8.jpg"
            image_7.write_bytes(b"fake")
            image_8.write_bytes(b"fake")
            _write_gold(gold_path)

            def fake_ocr_client(path: str | Path) -> dict[str, str]:
                if Path(path).name == "img_7.jpg":
                    return {"ocr_text": "SPEC 2017\nKirin 9010"}
                return {"ocr_text": "TaiShan V121"}

            report = run_rapidocr_candidate_evaluation(
                image_paths=[image_7, image_8],
                gold_path=gold_path,
                ocr_client=fake_ocr_client,
            )

        self.assertEqual(report["gate_decision"]["status"], "passed")
        self.assertEqual(report["overview"]["successful_files"], 2)
        self.assertEqual(report["overview"]["overall_exact_segment_recall"], 1.0)
        self.assertEqual(report["overview"]["overall_character_error_rate"], 0.0)
        self.assertEqual(report["overview"]["ocr_external_api_cost_cny"], 0.0)

    def test_parse_rapidocr_result_supports_common_shapes(self) -> None:
        tuple_result = ([[[0, 0], [1, 1]], "第一行", 0.99], [[[0, 1], [1, 2]], ("第二行", 0.98)])
        dict_result = {"rec_texts": ["第三行", "第四行"]}

        parsed_tuple = _parse_rapidocr_result([*tuple_result])
        parsed_dict = _parse_rapidocr_result(dict_result)

        self.assertEqual(parsed_tuple["ocr_text"], "第一行\n第二行")
        self.assertEqual(parsed_dict["ocr_text"], "第三行\n第四行")

    def test_write_report_outputs_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            gold_path = tmp_path / "gold.csv"
            image_7 = tmp_path / "img_7.jpg"
            output_json_path = tmp_path / "rapidocr_eval.json"
            output_markdown_path = tmp_path / "rapidocr_eval.md"
            image_7.write_bytes(b"fake")
            _write_gold(gold_path)

            def fake_ocr_client(path: str | Path) -> dict[str, str]:
                return {"ocr_text": "SPEC 2017\nKirin 9010"}

            json_path, markdown_path = write_rapidocr_candidate_report(
                image_paths=[image_7],
                gold_path=gold_path,
                output_json_path=output_json_path,
                output_markdown_path=output_markdown_path,
                ocr_client=fake_ocr_client,
            )
            report = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(report["report_name"], "rapidocr_candidate_evaluation")
        self.assertIn("RapidOCR 候选后端评估报告", markdown)
        self.assertIn("rapidocr_onnxruntime_local", markdown)

    def test_dependency_check_returns_expected_fields(self) -> None:
        dependency = check_rapidocr_dependency()

        self.assertIn(dependency["status"], {"available", "missing"})
        self.assertIn("rapidocr", dependency["packages"])
        self.assertIn("install_hint", dependency)


if __name__ == "__main__":
    unittest.main()
