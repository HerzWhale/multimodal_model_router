"""cost_repricing 的离线测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cost_repricing import (  # noqa: E402
    build_reprice_report_from_batch,
    render_reprice_markdown,
    reprice_model_calls,
    write_reprice_reports,
)


def _model_prices() -> dict:
    """构造当前价格目录。"""

    return {
        "qwen-vl-plus": {
            "price_source": "official_public_price_page",
            "price_updated_at": "2026-08-01",
            "price_confidence": "official_public_page",
            "pricing_rules": [
                {"unit_type": "input_tokens", "price_cny_per_unit": 0.0000008},
                {"unit_type": "output_tokens", "price_cny_per_unit": 0.000002},
            ],
        },
        "mock-ocr": {
            "price_source": "local_mock_assumption",
            "price_updated_at": "2026-08-01",
            "price_confidence": "mock_only",
            "pricing_unit": "image_count",
            "price_cny_per_unit": 0.01,
        },
    }


def _model_calls() -> list[dict]:
    """构造历史模型调用记录。"""

    return [
        {
            "call_id": "call_qwen_001",
            "file_id": "file_001",
            "task_type": "visual_understanding",
            "provider": "qwen",
            "model_name": "qwen-vl-plus",
            "response_model_name": "qwen-vl-plus",
            "input_units": [{"unit_type": "input_tokens", "quantity": 1231}],
            "output_units": [{"unit_type": "output_tokens", "quantity": 309}],
            "cost_cny": 0.003237,
        },
        {
            "call_id": "call_ocr_001",
            "file_id": "file_001",
            "task_type": "ocr",
            "provider": "doubao",
            "model_name": "mock-ocr",
            "input_units": [{"unit_type": "image_count", "quantity": 1}],
            "output_units": [],
            "cost_cny": 0.01,
        },
    ]


class CostRepricingTest(unittest.TestCase):
    def test_reprice_model_calls_compares_recorded_and_current_cost(self) -> None:
        report = reprice_model_calls(
            _model_calls(),
            _model_prices(),
            batch_id="batch_test",
            generated_at="2026-08-01T10:00:00+08:00",
        )
        by_call = {item["call_id"]: item for item in report["reprice_items"]}

        self.assertEqual(by_call["call_qwen_001"]["current_estimated_cost_cny"], 0.001603)
        self.assertEqual(by_call["call_qwen_001"]["cost_delta_cny"], -0.001634)
        self.assertEqual(by_call["call_qwen_001"]["reprice_status"], "changed")
        self.assertEqual(by_call["call_qwen_001"]["price_confidence"], "official_public_page")
        self.assertEqual(by_call["call_ocr_001"]["reprice_status"], "unchanged")
        self.assertEqual(report["summary"]["changed_call_count"], 1)

    def test_reprice_model_calls_handles_missing_price(self) -> None:
        report = reprice_model_calls(
            [
                {
                    "call_id": "call_unknown",
                    "model_name": "unknown-model",
                    "input_units": [],
                    "output_units": [],
                    "cost_cny": 0.1,
                }
            ],
            {},
            batch_id="batch_test",
            generated_at="2026-08-01T10:00:00+08:00",
        )
        item = report["reprice_items"][0]

        self.assertEqual(item["reprice_status"], "not_repriced")
        self.assertEqual(item["price_confidence"], "missing_price")
        self.assertIn("缺少模型", item["error_message"])

    def test_build_reprice_report_from_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            batch_dir = root / "batch"
            batch_dir.mkdir()
            (batch_dir / "batch_metadata.json").write_text(
                json.dumps({"batch_id": "batch_from_files"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (batch_dir / "model_calls.jsonl").write_text(
                "\n".join(json.dumps(call, ensure_ascii=False) for call in _model_calls()),
                encoding="utf-8",
            )
            prices_path = root / "model_prices.yaml"
            prices_path.write_text(
                """
models:
  qwen-vl-plus:
    price_source: official_public_price_page
    price_updated_at: "2026-08-01"
    price_confidence: official_public_page
    pricing_rules:
      - unit_type: input_tokens
        price_cny_per_unit: 0.0000008
      - unit_type: output_tokens
        price_cny_per_unit: 0.000002
  mock-ocr:
    price_source: local_mock_assumption
    price_updated_at: "2026-08-01"
    price_confidence: mock_only
    pricing_unit: image_count
    price_cny_per_unit: 0.01
""",
                encoding="utf-8",
            )

            report = build_reprice_report_from_batch(
                batch_dir=batch_dir,
                model_prices_path=prices_path,
                generated_at="2026-08-01T10:00:00+08:00",
            )

        self.assertEqual(report["batch_id"], "batch_from_files")
        self.assertEqual(report["summary"]["repriced_call_count"], 2)

    def test_markdown_and_json_reports_are_written(self) -> None:
        report = reprice_model_calls(
            _model_calls(),
            _model_prices(),
            batch_id="batch_test",
            generated_at="2026-08-01T10:00:00+08:00",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = write_reprice_reports(
                report=report,
                json_path=Path(tmp_dir) / "cost_reprice_report.json",
                markdown_path=Path(tmp_dir) / "cost_reprice_report.md",
            )
            saved_report = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertEqual(saved_report["report_type"], "cost_repricing")
        self.assertIn("# 成本重算报告", markdown)
        self.assertIn("call_qwen_001", markdown)
        self.assertIn("official_public_price_page", markdown)

    def test_render_markdown_contains_field_notes(self) -> None:
        markdown = render_reprice_markdown(
            reprice_model_calls(
                _model_calls(),
                _model_prices(),
                batch_id="batch_test",
                generated_at="2026-08-01T10:00:00+08:00",
            )
        )

        self.assertIn("字段说明", markdown)
        self.assertIn("current_estimated_cost_cny", markdown)


if __name__ == "__main__":
    unittest.main()
