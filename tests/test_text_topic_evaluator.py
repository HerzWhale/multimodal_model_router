"""text_topic_evaluator 的离线测试。"""

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

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
TEXT_TOPIC_SAMPLE_DIR = EVALUATION_DIR / "text_topic_small_set"
TEXT_TOPIC_GOLD_PATH = EVALUATION_DIR / "text_topic_gold.csv"
EXPECTED_TOPIC_SET = {
    "news",
    "finance_business",
    "ads_marketing",
    "technology",
    "sports_health",
    "entertainment",
    "lifestyle",
    "knowledge",
    "other",
}

from text_topic_evaluator import (  # noqa: E402
    apply_gold_topics,
    evaluate_topic_metrics,
    extract_text_topic_rows,
    read_gold_topic_rows,
    read_annotation_rows,
    render_evaluation_markdown,
    write_annotation_template,
    write_evaluation_report,
)


def _records() -> list[dict]:
    """构造文件级结果。"""

    return [
        {
            "batch_id": "batch_test",
            "file_id": "file_001",
            "file_name": "sample.txt",
            "media_type": "text",
            "topic": "technology",
            "secondary_topics": ["knowledge"],
            "summary": "这是一段技术内容。",
            "raw_text": "这是一段关于 AI 工程化和模型路由的文本。",
        },
        {
            "batch_id": "batch_test",
            "file_id": "file_002",
            "file_name": "image.png",
            "media_type": "image",
            "topic": "other",
        },
    ]


class TextTopicEvaluatorTest(unittest.TestCase):
    def test_extract_only_text_topic_rows(self) -> None:
        rows = extract_text_topic_rows(_records())

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["file_id"], "file_001")
        self.assertEqual(rows[0]["predicted_topic"], "technology")
        self.assertEqual(rows[0]["predicted_secondary_topics"], "knowledge")
        self.assertEqual(rows[0]["gold_topic"], "")

    def test_extract_keeps_missing_prediction_empty(self) -> None:
        records = _records()
        records[0]["topic"] = None

        rows = extract_text_topic_rows(records)

        self.assertEqual(rows[0]["predicted_topic"], "")

    def test_write_annotation_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = Path(tmp_dir) / "results.jsonl"
            template_path = Path(tmp_dir) / "text_topic_eval_template.csv"
            results_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False, indent=2) for record in _records()),
                encoding="utf-8",
            )

            write_annotation_template(results_path, template_path)
            rows = read_annotation_rows(template_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["predicted_topic"], "technology")
        self.assertEqual(rows[0]["gold_topic"], "")

    def test_apply_gold_topics_by_file_name(self) -> None:
        template_rows = extract_text_topic_rows(_records())
        merged_rows = apply_gold_topics(
            template_rows,
            [
                {
                    "file_name": "sample.txt",
                    "gold_topic": "technology",
                    "reviewer_note": "标准答案匹配",
                }
            ],
        )

        self.assertEqual(merged_rows[0]["gold_topic"], "technology")
        self.assertEqual(merged_rows[0]["reviewer_note"], "标准答案匹配")

    def test_write_annotation_template_with_gold_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = Path(tmp_dir) / "results.jsonl"
            gold_path = Path(tmp_dir) / "gold.csv"
            template_path = Path(tmp_dir) / "text_topic_eval_template.csv"
            results_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False, indent=2) for record in _records()),
                encoding="utf-8",
            )
            gold_path.write_text(
                "\n".join(
                    [
                        "file_name,gold_topic,reviewer_note",
                        "sample.txt,technology,标准答案匹配",
                    ]
                ),
                encoding="utf-8",
            )

            write_annotation_template(results_path, template_path, gold_path)
            rows = read_annotation_rows(template_path)

        self.assertEqual(rows[0]["gold_topic"], "technology")
        self.assertEqual(rows[0]["reviewer_note"], "标准答案匹配")

    def test_read_gold_topic_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            gold_path = Path(tmp_dir) / "gold.csv"
            gold_path.write_text(
                "\n".join(
                    [
                        "file_name,gold_topic,reviewer_note",
                        "sample.txt,technology,标准答案匹配",
                    ]
                ),
                encoding="utf-8",
            )

            rows = read_gold_topic_rows(gold_path)

        self.assertEqual(rows[0]["file_name"], "sample.txt")
        self.assertEqual(rows[0]["gold_topic"], "technology")

    def test_evaluate_topic_metrics(self) -> None:
        report = evaluate_topic_metrics(
            [
                {
                    "file_id": "file_001",
                    "file_name": "sample.txt",
                    "predicted_topic": "technology",
                    "gold_topic": "technology",
                    "reviewer_note": "正确",
                },
                {
                    "file_id": "file_002",
                    "file_name": "sample2.txt",
                    "predicted_topic": "news",
                    "gold_topic": "finance_business",
                    "reviewer_note": "应归为财经",
                },
                {
                    "file_id": "file_003",
                    "file_name": "sample3.txt",
                    "predicted_topic": "other",
                    "gold_topic": "",
                    "reviewer_note": "",
                },
            ]
        )

        self.assertEqual(report["evaluated_count"], 2)
        self.assertEqual(report["valid_prediction_count"], 2)
        self.assertEqual(report["correct_count"], 1)
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["valid_prediction_accuracy"], 0.5)
        self.assertEqual(report["prediction_coverage"], 1.0)
        self.assertEqual(report["macro_f1"], 0.333333)
        self.assertEqual(report["missing_label_file_ids"], ["file_003"])

    def test_macro_f1_exposes_imbalanced_class_error(self) -> None:
        report = evaluate_topic_metrics(
            [
                {"file_id": "1", "predicted_topic": "technology", "gold_topic": "technology"},
                {"file_id": "2", "predicted_topic": "technology", "gold_topic": "technology"},
                {"file_id": "3", "predicted_topic": "technology", "gold_topic": "news"},
                {"file_id": "4", "predicted_topic": "news", "gold_topic": "news"},
            ]
        )

        self.assertEqual(report["accuracy"], 0.75)
        self.assertEqual(report["macro_f1"], 0.733333)
        self.assertEqual(report["evaluated_labels"], ["news", "technology"])
        metrics_by_topic = {item["topic"]: item for item in report["per_class_metrics"]}
        self.assertEqual(metrics_by_topic["technology"]["support"], 2)
        self.assertEqual(metrics_by_topic["technology"]["precision"], 0.666667)
        self.assertEqual(metrics_by_topic["news"]["recall"], 0.5)

    def test_invalid_prediction_is_not_treated_as_a_business_class(self) -> None:
        report = evaluate_topic_metrics(
            [
                {"file_id": "1", "predicted_topic": "unknown_topic", "gold_topic": "news"},
            ]
        )

        self.assertEqual(report["evaluated_labels"], ["news"])
        self.assertEqual(report["macro_f1"], 0.0)
        self.assertEqual(report["valid_prediction_count"], 0)
        self.assertEqual(report["invalid_prediction_count"], 1)
        self.assertEqual(report["invalid_prediction_file_ids"], ["1"])
        metrics_by_topic = {item["topic"]: item for item in report["per_class_metrics"]}
        self.assertNotIn("unknown_topic", metrics_by_topic)
        self.assertEqual(metrics_by_topic["news"]["false_negative"], 1)

    def test_missing_prediction_is_separated_from_classification_quality(self) -> None:
        report = evaluate_topic_metrics(
            [
                {"file_id": "1", "predicted_topic": "technology", "gold_topic": "technology"},
                {"file_id": "2", "predicted_topic": "", "gold_topic": "news"},
            ]
        )

        self.assertEqual(report["evaluated_count"], 2)
        self.assertEqual(report["valid_prediction_count"], 1)
        self.assertEqual(report["missing_prediction_count"], 1)
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["valid_prediction_accuracy"], 1.0)
        self.assertEqual(report["prediction_coverage"], 0.5)
        self.assertEqual(report["missing_prediction_file_ids"], ["2"])

    def test_write_evaluation_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            annotation_path = Path(tmp_dir) / "template.csv"
            json_path = Path(tmp_dir) / "report.json"
            markdown_path = Path(tmp_dir) / "report.md"
            with annotation_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["file_id", "file_name", "predicted_topic", "gold_topic", "reviewer_note"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "file_id": "file_001",
                        "file_name": "sample.txt",
                        "predicted_topic": "technology",
                        "gold_topic": "technology",
                        "reviewer_note": "正确",
                    }
                )

            output_paths = write_evaluation_report(annotation_path, json_path, markdown_path)
            saved_report = json.loads(Path(output_paths["json"]).read_text(encoding="utf-8"))
            markdown = Path(output_paths["markdown"]).read_text(encoding="utf-8")

        self.assertEqual(saved_report["accuracy"], 1.0)
        self.assertEqual(saved_report["macro_f1"], 1.0)
        self.assertIn("# 文本主分类人工评估报告", markdown)
        self.assertIn("Accuracy", markdown)
        self.assertIn("Macro-F1", markdown)
        self.assertIn("分类级指标", markdown)

    def test_render_missing_label_report(self) -> None:
        report = evaluate_topic_metrics(
            [
                {
                    "file_id": "file_001",
                    "file_name": "sample.txt",
                    "predicted_topic": "technology",
                    "gold_topic": "",
                    "reviewer_note": "",
                }
            ]
        )
        markdown = render_evaluation_markdown(report)

        self.assertEqual(report["accuracy"], "当前数据未提供")
        self.assertEqual(report["macro_f1"], "当前数据未提供")
        self.assertIn("请先填写 gold_topic", markdown)

    def test_repository_text_topic_gold_matches_sample_files(self) -> None:
        """检查项目内置文本评估样本和人工标准答案一一对应。"""

        sample_files = sorted(path.name for path in TEXT_TOPIC_SAMPLE_DIR.glob("*.txt"))
        gold_rows = read_gold_topic_rows(TEXT_TOPIC_GOLD_PATH)
        gold_file_names = sorted(row["file_name"] for row in gold_rows)
        gold_topics = {row["gold_topic"] for row in gold_rows}

        self.assertEqual(len(sample_files), 18)
        self.assertEqual(sample_files, gold_file_names)
        self.assertEqual(gold_topics, EXPECTED_TOPIC_SET)

    def test_repository_text_topic_gold_has_utf8_bom_for_excel(self) -> None:
        """检查人工标准答案可被 Excel 自动识别为 UTF-8。"""

        raw_bytes = TEXT_TOPIC_GOLD_PATH.read_bytes()
        gold_rows = read_gold_topic_rows(TEXT_TOPIC_GOLD_PATH)

        self.assertTrue(raw_bytes.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(len(gold_rows), 18)
        self.assertEqual(
            set(gold_rows[0]),
            {"file_name", "gold_topic", "reviewer_note"},
        )
        self.assertTrue(gold_rows[0]["reviewer_note"])

    def test_repository_text_topic_samples_are_production_like(self) -> None:
        """检查文本评估样本具备短视频/图文平台内容的基本结构。"""

        platform_markers = [
            "标题：",
            "口播",
            "字幕",
            "图文说明",
            "评论区",
            "视频文案",
            "视频脚本",
            "讲解稿",
            "评论",
        ]

        for sample_path in TEXT_TOPIC_SAMPLE_DIR.glob("*.txt"):
            content = sample_path.read_text(encoding="utf-8").strip()
            marker_count = sum(1 for marker in platform_markers if marker in content)

            self.assertGreaterEqual(len(content), 220, sample_path.name)
            self.assertGreaterEqual(marker_count, 2, sample_path.name)

    def test_repository_samples_do_not_leak_gold_labels(self) -> None:
        """检查模型可见样本中不包含人工分类答案。"""

        forbidden_phrases = ["应优先归入", "归入科技数码", "归入娱乐休闲", "属于广告营销", "属于体育健康"]
        for sample_path in TEXT_TOPIC_SAMPLE_DIR.glob("*.txt"):
            content = sample_path.read_text(encoding="utf-8")
            self.assertNotIn("运营备注", content, sample_path.name)
            for phrase in forbidden_phrases:
                self.assertNotIn(phrase, content, sample_path.name)

    def test_new_long_samples_have_enough_content(self) -> None:
        """检查新增难例达到更接近长口播稿的内容量。"""

        long_sample_names = {
            "15_technology_long_local_ai_workflow.txt",
            "16_entertainment_long_variety_production.txt",
            "17_other_long_campus_lost_found.txt",
            "18_other_long_community_coordination.txt",
        }
        for file_name in long_sample_names:
            content = (TEXT_TOPIC_SAMPLE_DIR / file_name).read_text(encoding="utf-8").strip()
            self.assertGreaterEqual(len(content), 800, file_name)


if __name__ == "__main__":
    unittest.main()
