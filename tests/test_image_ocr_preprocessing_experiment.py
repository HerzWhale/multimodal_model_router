"""image_ocr_preprocessing_experiment 的离线测试。"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from image_ocr_preprocessing_experiment import (  # noqa: E402
    build_preprocess_variants,
    run_latency_profile,
    run_preprocessing_experiment,
    write_latency_profile_report,
    write_preprocessing_experiment_report,
)


def _gold_row(segment_id: str, gold_text: str) -> dict[str, str]:
    """构造一条OCR人工基准。"""

    return {
        "file_name": "sample.jpg",
        "segment_id": segment_id,
        "segment_type": "diagram_label",
        "gold_text": gold_text,
        "is_required": "true",
        "evaluation_scope": "business_content",
        "reviewer_note": "测试预处理实验",
    }


def _write_gold(path: Path) -> None:
    """写出测试用人工基准。"""

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(_gold_row("", "")))
        writer.writeheader()
        writer.writerow(_gold_row("left_label", "Integer PRF"))
        writer.writerow(_gold_row("right_value", "222 Entry"))


class ImageOcrPreprocessingExperimentTest(unittest.TestCase):
    def test_build_preprocess_variants_creates_scaled_and_split_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "sample.jpg"
            variants_dir = tmp_path / "variants"
            Image.new("RGB", (20, 10), color="white").save(image_path)

            variants = build_preprocess_variants(
                image_path=image_path,
                variants_dir=variants_dir,
            )

            with Image.open(variants_dir / "sample_full_2x.png") as full_image:
                full_size = full_image.size
            with Image.open(variants_dir / "sample_left_2x.png") as left_image:
                left_size = left_image.size
            with Image.open(variants_dir / "sample_right_2x.png") as right_image:
                right_size = right_image.size

        self.assertEqual([variant["variant_name"] for variant in variants], ["full_image_2x", "vertical_halves_2x"])
        self.assertEqual(full_size, (40, 20))
        self.assertEqual(left_size, (20, 20))
        self.assertEqual(right_size, (20, 20))

    def test_run_preprocessing_experiment_compares_variants_with_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "sample.jpg"
            gold_path = tmp_path / "gold.csv"
            baseline_path = tmp_path / "baseline.json"
            summary_path = tmp_path / "summary.json"
            Image.new("RGB", (20, 10), color="white").save(image_path)
            _write_gold(gold_path)
            baseline_path.write_text(
                json.dumps(
                    {
                        "file_name": "sample.jpg",
                        "required_segment_count": 2,
                        "exact_segment_count": 1,
                        "exact_segment_recall": 0.5,
                        "character_error_rate": 0.5,
                        "total_edit_distance": 8,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            summary_path.write_text(
                json.dumps(
                    {
                        "file_metrics": [
                            {"file_name": "sample.jpg", "ocr_latency_ms": 3000}
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            def fake_ocr_client(path: str | Path) -> dict[str, str]:
                file_name = Path(path).name
                if "left" in file_name:
                    return {"ocr_text": "Integer PRF"}
                if "right" in file_name:
                    return {"ocr_text": "222 Entry"}
                return {"ocr_text": "Integer PRF"}

            report = run_preprocessing_experiment(
                image_path=image_path,
                gold_path=gold_path,
                baseline_report_path=baseline_path,
                batch_summary_path=summary_path,
                variants_dir=tmp_path / "variants",
                ocr_client=fake_ocr_client,
            )

        self.assertEqual(report["experiment_name"], "image_ocr_preprocessing_experiment")
        self.assertEqual(report["decision"]["best_variant"], "vertical_halves_2x")
        self.assertEqual(report["decision"]["status"], "passed")
        self.assertEqual(report["variants"][1]["exact_segment_recall"], 1.0)
        self.assertEqual(report["variants"][1]["character_error_rate"], 0.0)

    def test_write_preprocessing_experiment_report_outputs_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "sample.jpg"
            gold_path = tmp_path / "gold.csv"
            baseline_path = tmp_path / "baseline.json"
            summary_path = tmp_path / "summary.json"
            output_json_path = tmp_path / "experiment.json"
            output_markdown_path = tmp_path / "experiment.md"
            Image.new("RGB", (20, 10), color="white").save(image_path)
            _write_gold(gold_path)
            baseline_path.write_text(
                json.dumps(
                    {
                        "file_name": "sample.jpg",
                        "required_segment_count": 2,
                        "exact_segment_count": 1,
                        "exact_segment_recall": 0.5,
                        "character_error_rate": 0.5,
                        "total_edit_distance": 8,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            summary_path.write_text(
                json.dumps(
                    {"file_metrics": [{"file_name": "sample.jpg", "ocr_latency_ms": 3000}]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            def fake_ocr_client(path: str | Path) -> dict[str, str]:
                return {"ocr_text": "Integer PRF\n222 Entry"}

            json_path, markdown_path = write_preprocessing_experiment_report(
                image_path=image_path,
                gold_path=gold_path,
                baseline_report_path=baseline_path,
                batch_summary_path=summary_path,
                output_json_path=output_json_path,
                output_markdown_path=output_markdown_path,
                ocr_client=fake_ocr_client,
            )
            report = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(report["file_name"], "sample.jpg")
        self.assertIn("图片 OCR 预处理实验报告", markdown)
        self.assertIn("full_image_2x", markdown)

    def test_run_latency_profile_splits_engine_decode_predict_and_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "sample.jpg"
            Image.new("RGB", (20, 10), color="white").save(image_path)

            class FakeEngine:
                def predict(self, image: object) -> dict[str, object]:
                    return {"rec_texts": ["Integer PRF", "222 Entry"]}

            report = run_latency_profile(
                image_path=image_path,
                repeat_count=2,
                engine_factory=lambda: FakeEngine(),
                image_decoder=lambda path: {"path": str(path)},
                prediction_parser=lambda prediction: {"ocr_text": "\n".join(prediction["rec_texts"])},
            )

        self.assertEqual(report["profile_name"], "image_ocr_latency_profile")
        self.assertEqual(report["repeat_count"], 2)
        self.assertEqual(len(report["attempts"]), 2)
        self.assertIn("engine_create_ms", report)
        self.assertIn("predict_ms", report["attempts"][0])
        self.assertIn("main_bottleneck", report["decision"])

    def test_run_latency_profile_rejects_zero_repeat_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.jpg"
            Image.new("RGB", (20, 10), color="white").save(image_path)

            with self.assertRaisesRegex(ValueError, "repeat_count"):
                run_latency_profile(image_path=image_path, repeat_count=0)

    def test_write_latency_profile_report_outputs_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "sample.jpg"
            output_json_path = tmp_path / "latency.json"
            output_markdown_path = tmp_path / "latency.md"
            Image.new("RGB", (20, 10), color="white").save(image_path)

            class FakeEngine:
                def predict(self, image: object) -> dict[str, object]:
                    return {"rec_texts": ["Integer PRF"]}

            json_path, markdown_path = write_latency_profile_report(
                image_path=image_path,
                output_json_path=output_json_path,
                output_markdown_path=output_markdown_path,
                repeat_count=1,
                engine_factory=lambda: FakeEngine(),
                image_decoder=lambda path: {"path": str(path)},
                prediction_parser=lambda prediction: {"ocr_text": "\n".join(prediction["rec_texts"])},
            )
            report = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(report["file_name"], "sample.jpg")
        self.assertIn("图片 OCR 延迟拆分报告", markdown)
        self.assertIn("模型推理", markdown)


if __name__ == "__main__":
    unittest.main()
