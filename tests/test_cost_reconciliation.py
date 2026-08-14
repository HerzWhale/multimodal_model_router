"""cost_reconciliation 的离线测试。"""

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

from cost_reconciliation import (  # noqa: E402
    BillingValidationError,
    build_billing_template,
    build_reconciliation_from_files,
    build_template_from_batch,
    group_estimated_model_costs,
    reconcile_costs,
    render_reconciliation_markdown,
    write_billing_template,
    write_reconciliation_reports,
)


def _model_calls() -> list[dict]:
    """构造包含多供应商、mock 和本地模型的调用明细。"""

    return [
        {
            "call_id": "call_qwen_001",
            "provider": "qwen",
            "model_name": "qwen-vl-plus",
            "response_model_name": "qwen-vl-plus",
            "task_type": "visual_understanding",
            "cost_cny": 0.003237,
            "started_at": "2026-07-30T15:24:51+08:00",
        },
        {
            "call_id": "call_deepseek_001",
            "provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "response_model_name": None,
            "task_type": "text_analysis",
            "cost_cny": 0.001,
            "started_at": "2026-07-30T15:25:00+08:00",
        },
        {
            "call_id": "call_deepseek_002",
            "provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "response_model_name": None,
            "task_type": "text_analysis",
            "cost_cny": 0.002,
            "started_at": "2026-07-30T15:26:00+08:00",
        },
        {
            "call_id": "call_dashscope_001",
            "provider": "dashscope",
            "model_name": "paraformer-v2",
            "response_model_name": "paraformer-v2",
            "task_type": "speech_to_text",
            "cost_cny": 0.004,
            "started_at": "2026-07-30T15:27:00+08:00",
        },
        {
            "call_id": "call_mock_001",
            "provider": "doubao",
            "model_name": "mock-ocr",
            "task_type": "ocr",
            "cost_cny": 0.01,
            "started_at": "2026-07-30T15:24:41+08:00",
        },
        {
            "call_id": "call_local_001",
            "provider": "paddlepaddle",
            "model_name": "PP-OCRv5_mobile",
            "task_type": "ocr",
            "cost_cny": 0.0,
            "started_at": "2026-07-30T15:24:42+08:00",
        },
    ]


class CostReconciliationTest(unittest.TestCase):
    def test_group_estimated_model_costs_marks_billable_and_excluded_groups(self) -> None:
        groups = group_estimated_model_costs(_model_calls())
        by_model = {group["model_name"]: group for group in groups}

        self.assertTrue(by_model["qwen-vl-plus"]["requires_bill_reconciliation"])
        self.assertTrue(by_model["deepseek-v4-flash"]["requires_bill_reconciliation"])
        self.assertTrue(by_model["paraformer-v2"]["requires_bill_reconciliation"])
        self.assertFalse(by_model["mock-ocr"]["requires_bill_reconciliation"])
        self.assertFalse(by_model["PP-OCRv5_mobile"]["requires_bill_reconciliation"])
        self.assertEqual(by_model["deepseek-v4-flash"]["estimated_call_count"], 2)
        self.assertEqual(by_model["deepseek-v4-flash"]["estimated_cost_cny"], 0.003)

    def test_billing_template_only_contains_live_api_groups(self) -> None:
        rows = build_billing_template(_model_calls())
        model_names = {row["model_name"] for row in rows}

        self.assertEqual(model_names, {"qwen-vl-plus", "deepseek-v4-flash", "paraformer-v2"})
        self.assertNotIn("mock-ocr", model_names)
        self.assertEqual(rows[0]["billed_cost_cny"], "")
        self.assertEqual({row["billing_granularity"] for row in rows}, {"hour"})

    def test_reconcile_call_level_bill(self) -> None:
        report = reconcile_costs(
            _model_calls(),
            [
                {
                    "provider": "qwen",
                    "model_name": "qwen-vl-plus",
                    "response_model_name": "qwen-vl-plus",
                    "billing_start_at": "2026-07-30T15:24:00+08:00",
                    "billing_end_at": "2026-07-30T15:25:00+08:00",
                    "billed_cost_cny": "0.003300",
                    "billing_granularity": "call",
                    "bill_source": "manual_entry",
                    "matching_method": "provider_model_time_window",
                    "note": "单次调用级对账。",
                }
            ],
            batch_id="batch_test",
            generated_at="2026-07-30T16:00:00+08:00",
        )
        qwen_item = [
            item
            for item in report["reconciliation_items"]
            if item["model_name"] == "qwen-vl-plus"
        ][0]

        self.assertEqual(qwen_item["cost_confidence"], "call_level_reconciled")
        self.assertEqual(qwen_item["billed_cost_cny"], 0.0033)
        self.assertEqual(qwen_item["cost_delta_cny"], 0.000063)
        self.assertEqual(qwen_item["cost_delta_reason"], "billed_higher_than_estimate")
        self.assertEqual(qwen_item["matched_call_ids"], ["call_qwen_001"])
        self.assertEqual(report["summary"]["estimation_error_status"], "known_for_reconciled_items")

    def test_reconcile_period_level_bill_and_unverified_group(self) -> None:
        report = reconcile_costs(
            _model_calls(),
            [
                {
                    "provider": "deepseek",
                    "model_name": "deepseek-v4-flash",
                    "response_model_name": "",
                    "billing_start_at": "2026-07-30T15:00:00+08:00",
                    "billing_end_at": "2026-07-30T16:00:00+08:00",
                    "billed_cost_cny": "0.003100",
                    "billing_granularity": "hour",
                    "bill_source": "manual_entry",
                    "matching_method": "provider_model_time_window",
                    "note": "小时级账单。",
                }
            ],
            batch_id="batch_test",
            generated_at="2026-07-30T16:00:00+08:00",
        )
        by_model = {item["model_name"]: item for item in report["reconciliation_items"]}

        self.assertEqual(by_model["deepseek-v4-flash"]["cost_confidence"], "period_level_reconciled")
        self.assertEqual(by_model["deepseek-v4-flash"]["estimated_call_count"], 2)
        self.assertEqual(by_model["qwen-vl-plus"]["cost_confidence"], "unverified")
        self.assertEqual(by_model["paraformer-v2"]["cost_confidence"], "unverified")
        self.assertEqual(report["summary"]["reconciled_group_count"], 1)
        self.assertEqual(report["summary"]["unverified_group_count"], 2)
        self.assertFalse(report["summary"]["bill_reconciled"])

    def test_blank_billed_cost_keeps_group_unverified(self) -> None:
        report = reconcile_costs(
            _model_calls(),
            [
                {
                    "provider": "qwen",
                    "model_name": "qwen-vl-plus",
                    "response_model_name": "qwen-vl-plus",
                    "billed_cost_cny": "",
                    "billing_granularity": "period",
                }
            ],
            batch_id="batch_test",
            generated_at="2026-07-30T16:00:00+08:00",
        )
        qwen_item = [
            item
            for item in report["reconciliation_items"]
            if item["model_name"] == "qwen-vl-plus"
        ][0]

        self.assertFalse(qwen_item["bill_reconciled"])
        self.assertEqual(qwen_item["cost_confidence"], "unverified")
        self.assertEqual(qwen_item["cost_delta_reason"], "unverified")
        self.assertIsNone(qwen_item["cost_delta_rate"])

    def test_free_quota_bill_gets_billing_adjustment_reason(self) -> None:
        report = reconcile_costs(
            _model_calls(),
            [
                {
                    "provider": "qwen",
                    "model_name": "qwen-vl-plus",
                    "response_model_name": "qwen-vl-plus",
                    "billing_start_at": "2026-07-30T15:24:00+08:00",
                    "billing_end_at": "2026-07-30T15:25:00+08:00",
                    "billed_cost_cny": "0.00",
                    "billing_granularity": "period",
                    "bill_source": "dashscope_console_free_quota",
                    "matching_method": "provider_model_time_window",
                    "note": "供应商后台显示本次调用走免费额度。",
                }
            ],
            batch_id="batch_test",
            generated_at="2026-07-30T16:00:00+08:00",
        )
        qwen_item = [
            item
            for item in report["reconciliation_items"]
            if item["model_name"] == "qwen-vl-plus"
        ][0]

        self.assertEqual(qwen_item["cost_delta_reason"], "billing_adjustment")
        self.assertEqual(report["summary"]["cost_delta_reason_counts"]["billing_adjustment"], 1)

    def test_non_numeric_billed_cost_is_rejected(self) -> None:
        with self.assertRaisesRegex(BillingValidationError, "billed_cost_cny 非法"):
            reconcile_costs(
                _model_calls(),
                [
                    {
                        "provider": "qwen",
                        "model_name": "qwen-vl-plus",
                        "response_model_name": "qwen-vl-plus",
                        "billed_cost_cny": "not-a-number",
                    }
                ],
            )

    def test_negative_billed_cost_is_rejected(self) -> None:
        with self.assertRaisesRegex(BillingValidationError, "非负有限数字"):
            reconcile_costs(
                _model_calls(),
                [
                    {
                        "provider": "qwen",
                        "model_name": "qwen-vl-plus",
                        "response_model_name": "qwen-vl-plus",
                        "billed_cost_cny": "-0.01",
                    }
                ],
            )

    def test_nan_and_infinity_billed_cost_are_rejected(self) -> None:
        for invalid_cost in ["NaN", "Infinity", "-Infinity"]:
            with self.subTest(invalid_cost=invalid_cost):
                with self.assertRaisesRegex(BillingValidationError, "非负有限数字"):
                    reconcile_costs(
                        _model_calls(),
                        [
                            {
                                "provider": "qwen",
                                "model_name": "qwen-vl-plus",
                                "response_model_name": "qwen-vl-plus",
                                "billed_cost_cny": invalid_cost,
                            }
                        ],
                    )

    def test_overlapping_duplicate_billing_records_are_rejected(self) -> None:
        with self.assertRaisesRegex(BillingValidationError, "账单记录重复"):
            reconcile_costs(
                _model_calls(),
                [
                    {
                        "provider": "qwen",
                        "model_name": "qwen-vl-plus",
                        "response_model_name": "qwen-vl-plus",
                        "billing_start_at": "2026-07-30T15:00:00+08:00",
                        "billing_end_at": "2026-07-30T16:00:00+08:00",
                        "billed_cost_cny": "0.0033",
                    },
                    {
                        "provider": "qwen",
                        "model_name": "qwen-vl-plus",
                        "response_model_name": "qwen-vl-plus",
                        "billing_start_at": "2026-07-30T15:30:00+08:00",
                        "billing_end_at": "2026-07-30T16:30:00+08:00",
                        "billed_cost_cny": "0.0034",
                    },
                ],
            )

    def test_unmatched_billing_record_is_preserved(self) -> None:
        report = reconcile_costs(
            _model_calls(),
            [
                {
                    "provider": "unknown_provider",
                    "model_name": "unknown_model",
                    "billed_cost_cny": "1.0",
                }
            ],
            batch_id="batch_test",
            generated_at="2026-07-30T16:00:00+08:00",
        )

        self.assertEqual(len(report["unmatched_billing_records"]), 1)

    def test_json_markdown_and_csv_files_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_path = write_billing_template(root / "billing.csv", build_billing_template(_model_calls()))
            report = reconcile_costs(_model_calls(), [], batch_id="batch_test", generated_at="2026-07-30T16:00:00+08:00")
            output_paths = write_reconciliation_reports(
                json_path=root / "cost_reconciliation.json",
                markdown_path=root / "cost_reconciliation.md",
                report=report,
            )
            saved_report = json.loads(Path(output_paths["json"]).read_text(encoding="utf-8"))
            markdown = Path(output_paths["markdown"]).read_text(encoding="utf-8")
            with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(saved_report["report_type"], "cost_reconciliation")
        self.assertIn("# 成本对账报告", markdown)
        self.assertIn("字段说明", markdown)
        self.assertEqual(len(rows), 3)

    def test_build_from_batch_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_dir = Path(tmp_dir) / "batch_test"
            batch_dir.mkdir()
            (batch_dir / "batch_metadata.json").write_text(
                json.dumps({"batch_id": "batch_from_files"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (batch_dir / "model_calls.jsonl").write_text(
                "\n".join(json.dumps(call, ensure_ascii=False) for call in _model_calls()),
                encoding="utf-8",
            )
            template_path = build_template_from_batch(batch_dir, batch_dir / "manual_billing.csv")
            report = build_reconciliation_from_files(
                batch_dir=batch_dir,
                billing_csv=template_path,
                generated_at="2026-07-30T16:00:00+08:00",
            )

        self.assertEqual(report["batch_id"], "batch_from_files")
        self.assertEqual(report["summary"]["billable_group_count"], 3)
        self.assertEqual(report["summary"]["estimation_error_status"], "unknown_until_bill_reconciliation")

    def test_markdown_marks_unknown_error_before_bill_reconciliation(self) -> None:
        report = reconcile_costs(
            _model_calls(),
            [],
            batch_id="batch_test",
            generated_at="2026-07-30T16:00:00+08:00",
        )

        markdown = render_reconciliation_markdown(report)

        self.assertIn("unknown_until_bill_reconciliation", markdown)
        self.assertIn("未对账前不得宣称误差小", markdown)

    def test_markdown_shows_bill_source_and_note(self) -> None:
        report = reconcile_costs(
            _model_calls(),
            [
                {
                    "provider": "qwen",
                    "model_name": "qwen-vl-plus",
                    "response_model_name": "qwen-vl-plus",
                    "billing_start_at": "2026-07-30T15:24:00+08:00",
                    "billing_end_at": "2026-07-30T15:25:00+08:00",
                    "billed_cost_cny": "0.00",
                    "billing_granularity": "period",
                    "bill_source": "dashscope_console_free_quota",
                    "matching_method": "provider_model_time_window",
                    "note": "供应商后台显示本次调用走免费额度。",
                }
            ],
            batch_id="batch_test",
            generated_at="2026-07-30T16:00:00+08:00",
        )

        markdown = render_reconciliation_markdown(report)

        self.assertIn("dashscope_console_free_quota", markdown)
        self.assertIn("供应商后台显示本次调用走免费额度", markdown)
        self.assertIn("差异原因", markdown)
        self.assertIn("billing_adjustment", markdown)


if __name__ == "__main__":
    unittest.main()
