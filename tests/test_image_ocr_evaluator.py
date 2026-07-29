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
    analyze_image_ocr_errors,
    build_image_ocr_gate_report,
    evaluate_image_ocr,
    levenshtein_distance,
    normalize_ocr_text,
    read_ocr_gold_rows,
    write_image_ocr_error_analysis,
    write_image_ocr_gate_report,
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


def _detail(
    segment_id: str,
    segment_type: str,
    gold_text: str,
    comparison_text: str,
    edit_distance: int,
    *,
    is_exact_match: bool = False,
) -> dict:
    """构造一条OCR评估明细。"""

    return {
        "segment_id": segment_id,
        "segment_type": segment_type,
        "gold_text": gold_text,
        "matched_ocr_text": comparison_text,
        "comparison_text": comparison_text,
        "is_exact_match": is_exact_match,
        "gold_character_count": len(normalize_ocr_text(gold_text)),
        "edit_distance": edit_distance,
        "character_error_rate": round(edit_distance / len(normalize_ocr_text(gold_text)), 6),
        "reviewer_note": "测试明细",
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

    def test_error_analysis_groups_errors_and_builds_gate_decision(self) -> None:
        evaluation_report = {
            "schema_version": "v1",
            "file_name": "img_9.jpg",
            "required_segment_count": 3,
            "exact_segment_count": 1,
            "exact_segment_recall": 0.333333,
            "character_error_rate": 0.25,
            "details": [
                _detail(
                    "title",
                    "diagram_title",
                    "麒麟9010",
                    "麒麟9010",
                    0,
                    is_exact_match=True,
                ),
                _detail(
                    "tlb",
                    "tlb_size",
                    "L1D TLB 128 pages",
                    "128pages",
                    6,
                ),
                _detail(
                    "prf",
                    "pipeline_module",
                    "Integer PRF 158 Entry",
                    "IntegerPRF",
                    8,
                ),
            ],
        }
        batch_summary = {
            "batch_id": "batch_test",
            "file_metrics": [{"file_name": "img_9.jpg", "ocr_latency_ms": 28261}],
        }

        analysis = analyze_image_ocr_errors(evaluation_report, batch_summary=batch_summary)

        self.assertEqual(analysis["file_name"], "img_9.jpg")
        self.assertEqual(analysis["overview"]["error_segment_count"], 2)
        self.assertEqual(analysis["overview"]["ocr_latency_ms"], 28261)
        self.assertEqual(analysis["gate_decision"]["status"], "not_passed")
        self.assertIn("继续留在图片OCR功能内", analysis["gate_decision"]["next_action"])
        self.assertEqual(
            [row["segment_type"] for row in analysis["error_by_segment_type"][:2]],
            ["pipeline_module", "tlb_size"],
        )
        self.assertEqual(
            {row["error_bucket"] for row in analysis["error_buckets"]},
            {"label_retained_value_lost", "value_retained_label_lost"},
        )

    def test_write_image_ocr_error_analysis_outputs_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            evaluation_report_path = tmp_path / "eval.json"
            batch_summary_path = tmp_path / "summary.json"
            output_json_path = tmp_path / "analysis.json"
            output_markdown_path = tmp_path / "analysis.md"
            evaluation_report_path.write_text(
                json.dumps(
                    {
                        "schema_version": "v1",
                        "file_name": "img_9.jpg",
                        "required_segment_count": 1,
                        "exact_segment_count": 0,
                        "exact_segment_recall": 0.0,
                        "character_error_rate": 0.5,
                        "details": [
                            _detail(
                                "buffer",
                                "buffer_size",
                                "Re-order Buffer 504 Entry",
                                "Re-orderBuffer",
                                8,
                            )
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            batch_summary_path.write_text(
                json.dumps(
                    {
                        "batch_id": "batch_test",
                        "file_metrics": [{"file_name": "img_9.jpg", "ocr_latency_ms": 3000}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            json_path, markdown_path = write_image_ocr_error_analysis(
                evaluation_report_path=evaluation_report_path,
                batch_summary_path=batch_summary_path,
                output_json_path=output_json_path,
                output_markdown_path=output_markdown_path,
            )

            analysis = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(analysis["analysis_name"], "image_ocr_error_analysis")
        self.assertIn("图片 OCR 错误归因报告", markdown)
        self.assertIn("not_passed", markdown)

    def test_batch_gate_report_blocks_next_feature_when_quality_or_latency_fails(self) -> None:
        summary_report = {
            "batch_id": "batch_test",
            "evaluated_files": 2,
            "total_required_segment_count": 12,
            "overall_exact_segment_recall": 0.75,
            "overall_character_error_rate": 0.12,
            "ocr_avg_latency_ms": 12000,
            "ocr_p95_latency_ms": 25000,
            "ocr_cost_cny": 0.0,
            "file_metrics": [
                {
                    "file_name": "img_ok.jpg",
                    "required_segment_count": 5,
                    "exact_segment_count": 5,
                    "exact_segment_recall": 1.0,
                    "character_error_rate": 0.0,
                    "ocr_latency_ms": 1800,
                },
                {
                    "file_name": "img_9.jpg",
                    "required_segment_count": 7,
                    "exact_segment_count": 4,
                    "exact_segment_recall": 0.571429,
                    "character_error_rate": 0.2,
                    "ocr_latency_ms": 25000,
                },
            ],
        }
        error_analysis_report = {
            "file_name": "img_9.jpg",
            "gate_decision": {"status": "not_passed", "next_action": "继续留在图片OCR功能内。"},
            "error_by_segment_type": [
                {"segment_type": "buffer_size", "error_segments": 3, "character_error_rate": 0.3}
            ],
            "error_buckets": [
                {
                    "error_bucket": "label_retained_value_lost",
                    "segment_count": 3,
                    "description": "标签识别到但数值缺失。",
                }
            ],
        }

        report = build_image_ocr_gate_report(
            summary_report,
            error_analysis_report=error_analysis_report,
        )

        self.assertEqual(report["gate_decision"]["status"], "not_passed")
        self.assertEqual(report["gate_decision"]["blocking_files"], ["img_9.jpg"])
        self.assertIn("不要开启ASR", report["recommendations"][0])
        self.assertEqual(report["weak_sample_analysis"]["file_name"], "img_9.jpg")

    def test_batch_gate_report_handles_missing_metric_without_hard_calculation(self) -> None:
        summary_report = {
            "batch_id": "batch_missing",
            "file_metrics": [
                {
                    "file_name": "img_missing.jpg",
                    "required_segment_count": 1,
                    "exact_segment_count": 1,
                    "exact_segment_recall": 1.0,
                    "character_error_rate": 0.0,
                }
            ],
        }

        report = build_image_ocr_gate_report(summary_report)

        self.assertEqual(report["gate_decision"]["status"], "not_passed")
        self.assertIsNone(report["batch_overview"]["ocr_p95_latency_ms"])
        self.assertTrue(
            any("缺少批次OCR P95延迟" in reason for reason in report["gate_decision"]["reasons"])
        )

    def test_write_image_ocr_gate_report_outputs_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            summary_path = tmp_path / "summary.json"
            analysis_path = tmp_path / "analysis.json"
            output_json_path = tmp_path / "gate.json"
            output_markdown_path = tmp_path / "gate.md"
            summary_path.write_text(
                json.dumps(
                    {
                        "batch_id": "batch_test",
                        "evaluated_files": 1,
                        "total_required_segment_count": 1,
                        "overall_exact_segment_recall": 1.0,
                        "overall_character_error_rate": 0.0,
                        "ocr_avg_latency_ms": 1000,
                        "ocr_p95_latency_ms": 1000,
                        "ocr_cost_cny": 0.0,
                        "file_metrics": [
                            {
                                "file_name": "img_ok.jpg",
                                "required_segment_count": 1,
                                "exact_segment_count": 1,
                                "exact_segment_recall": 1.0,
                                "character_error_rate": 0.0,
                                "ocr_latency_ms": 1000,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            analysis_path.write_text(
                json.dumps({"file_name": "img_ok.jpg"}, ensure_ascii=False),
                encoding="utf-8",
            )

            json_path, markdown_path = write_image_ocr_gate_report(
                summary_report_path=summary_path,
                error_analysis_path=analysis_path,
                output_json_path=output_json_path,
                output_markdown_path=output_markdown_path,
            )

            report = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(report["report_name"], "image_ocr_batch_gate_report")
        self.assertIn("图片 OCR 批次级闸门报告", markdown)
        self.assertIn("passed", markdown)


if __name__ == "__main__":
    unittest.main()
