"""image_ocr_evaluator 的离线测试。"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from image_ocr_evaluator import (  # noqa: E402
    evaluate_image_ocr,
    levenshtein_distance,
    normalize_ocr_text,
    read_ocr_gold_rows,
    write_image_ocr_report,
)


def _gold_row(
    segment_id: str,
    gold_text: str,
    *,
    is_required: str = "true",
) -> dict[str, str]:
    """构造一条OCR人工基准。"""

    return {
        "file_name": "sample.png",
        "segment_id": segment_id,
        "segment_type": "content_caption",
        "gold_text": gold_text,
        "is_required": is_required,
        "evaluation_scope": "business_content",
        "reviewer_note": "测试文字块",
    }


class ImageOcrEvaluatorTest(unittest.TestCase):
    def test_normalize_ocr_text_removes_whitespace_and_unifies_width(self) -> None:
        self.assertEqual(normalize_ocr_text(" Ｐ４\n年会不能停 "), "P4年会不能停")

    def test_levenshtein_distance_counts_missing_characters(self) -> None:
        self.assertEqual(levenshtein_distance("功夫女足", "功夫"), 2)

    def test_exact_segment_recall_requires_distinct_ocr_lines(self) -> None:
        report = evaluate_image_ocr(
            file_name="sample.png",
            ocr_text="年会不能停",
            gold_rows=[
                _gold_row("card_01", "年会不能停"),
                _gold_row("card_02", "年会不能停"),
            ],
        )

        self.assertEqual(report["required_segment_count"], 2)
        self.assertEqual(report["exact_segment_count"], 1)
        self.assertEqual(report["exact_segment_recall"], 0.5)

    def test_distinct_segments_can_share_one_ocr_line(self) -> None:
        report = evaluate_image_ocr(
            file_name="sample.png",
            ocr_text="#周星驰#一口气看完系列#影视解说",
            gold_rows=[
                _gold_row("topic_01", "#周星驰"),
                _gold_row("topic_02", "#一口气看完系列"),
                _gold_row("topic_03", "#影视解说"),
            ],
        )

        self.assertEqual(report["required_segment_count"], 3)
        self.assertEqual(report["exact_segment_count"], 3)
        self.assertEqual(report["exact_segment_recall"], 1.0)
        self.assertEqual(report["character_error_rate"], 0.0)

    def test_character_error_rate_uses_best_continuous_window(self) -> None:
        report = evaluate_image_ocr(
            file_name="sample.png",
            ocr_text="《功夫》首评来了！#电影功",
            gold_rows=[_gold_row("caption", "《功夫女足》首评来了！")],
        )

        detail = report["details"][0]
        self.assertFalse(detail["is_exact_match"])
        self.assertEqual(detail["comparison_text"], "《功夫》首评来了!")
        self.assertEqual(detail["edit_distance"], 2)
        self.assertEqual(report["total_edit_distance"], 2)

    def test_optional_segments_do_not_enter_metrics(self) -> None:
        report = evaluate_image_ocr(
            file_name="sample.png",
            ocr_text="账号名称",
            gold_rows=[
                _gold_row("required", "账号名称"),
                _gold_row("optional", "点赞数量", is_required="false"),
            ],
        )

        self.assertEqual(report["required_segment_count"], 1)
        self.assertEqual(report["exact_segment_count"], 1)
        self.assertEqual(len(report["details"]), 1)

    def test_read_project_gold_file_keeps_confirmed_twenty_segments(self) -> None:
        rows = read_ocr_gold_rows(PROJECT_ROOT / "evaluation" / "image_ocr_gold.csv")
        image_rows = [row for row in rows if row["file_name"] == "img_1.png"]

        self.assertEqual(len(image_rows), 20)
        self.assertEqual(len({row["segment_id"] for row in image_rows}), 20)
        self.assertTrue(all(row["evaluation_scope"] == "business_content" for row in image_rows))

    def test_read_gold_rejects_missing_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            gold_path = Path(tmp_dir) / "gold.csv"
            gold_path.write_text("file_name,gold_text\nsample.png,账号名称\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "缺少字段"):
                read_ocr_gold_rows(gold_path)

    def test_read_gold_rejects_duplicate_segment_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            gold_path = Path(tmp_dir) / "gold.csv"
            with gold_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(_gold_row("", "")))
                writer.writeheader()
                writer.writerow(_gold_row("name", "账号名称"))
                writer.writerow(_gold_row("name", "重复名称"))

            with self.assertRaisesRegex(ValueError, "重复文字块"):
                read_ocr_gold_rows(gold_path)

    def test_write_report_rejects_missing_image_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            results_path = tmp_path / "results.jsonl"
            gold_path = tmp_path / "gold.csv"
            results_path.write_text("", encoding="utf-8")
            with gold_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(_gold_row("", "")))
                writer.writeheader()
                writer.writerow(_gold_row("name", "账号名称"))

            with self.assertRaisesRegex(ValueError, "没有找到图片"):
                write_image_ocr_report(
                    results_path=results_path,
                    gold_path=gold_path,
                    file_name="sample.png",
                    output_path=tmp_path / "report.json",
                )

    def test_write_image_ocr_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            results_path = tmp_path / "results.jsonl"
            gold_path = tmp_path / "gold.csv"
            output_path = tmp_path / "report.json"
            results_path.write_text(
                json.dumps(
                    {
                        "file_name": "sample.png",
                        "media_type": "image",
                        "ocr_text": "账号名称",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with gold_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(_gold_row("", "")))
                writer.writeheader()
                writer.writerow(_gold_row("name", "账号名称"))

            written_path = write_image_ocr_report(
                results_path=results_path,
                gold_path=gold_path,
                file_name="sample.png",
                output_path=output_path,
            )
            report = json.loads(written_path.read_text(encoding="utf-8"))

        self.assertEqual(report["exact_segment_recall"], 1.0)
        self.assertEqual(report["character_error_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
