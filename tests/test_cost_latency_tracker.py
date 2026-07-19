"""cost_latency_tracker 的测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cost_latency_tracker import build_model_call_record, calculate_cost_cny, load_model_prices


class CostLatencyTrackerTest(unittest.TestCase):
    def test_load_model_prices(self) -> None:
        config_path = PROJECT_ROOT / "config" / "model_prices.yaml"

        model_prices = load_model_prices(config_path)

        self.assertEqual(model_prices["mock-ocr"]["provider"], "doubao")
        self.assertEqual(model_prices["mock-asr"]["pricing_unit"], "audio_seconds")

    def test_calculate_cost_cny(self) -> None:
        model_prices = {
            "mock-asr": {
                "pricing_unit": "audio_seconds",
                "price_cny_per_unit": 0.0005,
            }
        }

        cost = calculate_cost_cny(
            "mock-asr",
            [{"unit_type": "audio_seconds", "quantity": 120}],
            model_prices,
        )

        self.assertEqual(cost, 0.06)

    def test_calculate_cost_cny_with_input_and_output_tokens(self) -> None:
        model_prices = {
            "deepseek-v4-flash": {
                "pricing_rules": [
                    {
                        "unit_type": "input_tokens",
                        "price_cny_per_unit": 0.000001,
                    },
                    {
                        "unit_type": "output_tokens",
                        "price_cny_per_unit": 0.000002,
                    },
                ],
            }
        }

        cost = calculate_cost_cny(
            "deepseek-v4-flash",
            [{"unit_type": "input_tokens", "quantity": 1000}],
            model_prices,
            [{"unit_type": "output_tokens", "quantity": 500}],
        )

        self.assertEqual(cost, 0.002)

    def test_build_model_call_record(self) -> None:
        model_prices = {
            "mock-ocr": {
                "pricing_unit": "image_count",
                "price_cny_per_unit": 0.01,
            }
        }

        record = build_model_call_record(
            call_id="call_001",
            batch_id="batch_001",
            file_id="file_001",
            task_type="ocr",
            provider="doubao",
            model_name="mock-ocr",
            input_units=[{"unit_type": "image_count", "quantity": 3}],
            output_units=[{"unit_type": "text_chars", "quantity": 120}],
            latency_ms=850,
            started_at="2026-07-14T10:00:00+08:00",
            status="success",
            error_message=None,
            model_prices=model_prices,
        )

        self.assertEqual(record["cost_cny"], 0.03)
        self.assertEqual(record["status"], "success")
        self.assertIsNone(record["error_message"])

    def test_build_model_call_record_rejects_unknown_status(self) -> None:
        with self.assertRaises(ValueError):
            build_model_call_record(
                call_id="call_001",
                batch_id="batch_001",
                file_id="file_001",
                task_type="ocr",
                provider="doubao",
                model_name="mock-ocr",
                input_units=[{"unit_type": "image_count", "quantity": 1}],
                output_units=[],
                latency_ms=1,
                started_at="2026-07-14T10:00:00+08:00",
                status="partial_success",
                error_message=None,
                model_prices={
                    "mock-ocr": {
                        "pricing_unit": "image_count",
                        "price_cny_per_unit": 0.01,
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
