"""price_catalog_updater 的离线测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from price_catalog_updater import (  # noqa: E402
    parse_aliyun_qwen_vl_plus,
    parse_deepseek_v4_flash,
    preflight_price_update,
    refresh_price_catalog,
    write_refresh_report,
)


ALIYUN_SAMPLE = """
<table>
<tr><td>输入</td><td>0.8</td><td>每百万 tokens</td></tr>
<tr><td>输出</td><td>2</td><td>每百万 tokens</td></tr>
<tr><td>输入（缓存命中）</td><td>0.16</td><td>每百万 tokens</td></tr>
</table>
"""


DEEPSEEK_SAMPLE = """
<table>
<tr><td>模型</td><td>deepseek-v4-flash</td><td>deepseek-v4-pro</td></tr>
<tr><td>百万tokens输入（缓存命中）</td><td>0.02元</td><td>0.025元</td></tr>
<tr><td>百万tokens输入（缓存未命中）</td><td>1元</td><td>3元</td></tr>
<tr><td>百万tokens输出</td><td>2元</td><td>6元</td></tr>
</table>
"""


def _config_text() -> str:
    """生成临时价格配置。"""

    return """
models:
  qwen-vl-plus:
    provider: qwen
    price_source: local_manual_config
    price_updated_at: "2026-08-01"
    price_confidence: unverified_manual_config
    price_fetch:
      enabled: true
      source_url: https://help.aliyun.com/zh/model-studio/qwen-vl-plus
      parser: aliyun_qwen_vl_plus
    pricing_rules:
      - unit_type: input_tokens
        price_cny_per_unit: 0.0000015
      - unit_type: output_tokens
        price_cny_per_unit: 0.0000045
  deepseek-v4-flash:
    provider: deepseek
    price_source: local_manual_config
    price_updated_at: "2026-08-01"
    price_confidence: unverified_manual_config
    price_fetch:
      enabled: true
      source_url: https://api-docs.deepseek.com/zh-cn/quick_start/pricing
      parser: deepseek_v4_flash
    pricing_rules:
      - unit_type: input_tokens
        price_cny_per_unit: 0.000001008
      - unit_type: output_tokens
        price_cny_per_unit: 0.000002016
"""


def _fake_fetcher(url: str) -> str:
    """根据URL返回假官方页面。"""

    if "aliyun" in url:
        return ALIYUN_SAMPLE
    if "deepseek" in url:
        return DEEPSEEK_SAMPLE
    raise AssertionError(f"未预期的URL: {url}")


class PriceCatalogUpdaterTest(unittest.TestCase):
    def test_parse_aliyun_qwen_vl_plus_price(self) -> None:
        result = parse_aliyun_qwen_vl_plus(ALIYUN_SAMPLE)

        self.assertEqual(result["currency"], "CNY")
        self.assertEqual(result["pricing_rules"][0]["price_cny_per_unit"], 0.0000008)
        self.assertEqual(result["pricing_rules"][1]["price_cny_per_unit"], 0.000002)

    def test_parse_deepseek_v4_flash_price_uses_cache_miss_input(self) -> None:
        result = parse_deepseek_v4_flash(DEEPSEEK_SAMPLE)

        self.assertEqual(result["currency"], "CNY")
        self.assertEqual(result["pricing_rules"][0]["price_cny_per_unit"], 0.000001)
        self.assertEqual(result["pricing_rules"][1]["price_cny_per_unit"], 0.000002)

    def test_refresh_without_apply_only_reports_candidate_prices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "model_prices.yaml"
            config_path.write_text(_config_text(), encoding="utf-8")

            report = refresh_price_catalog(
                config_path,
                apply_changes=False,
                fetcher=_fake_fetcher,
                generated_at="2026-08-01T10:00:00+08:00",
            )
            saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(report["refreshed_model_count"], 2)
        self.assertEqual(report["applied_model_count"], 0)
        self.assertEqual(report["preflight_failed_count"], 1)
        self.assertTrue(report["refresh_items"][0]["changed"])
        self.assertEqual(
            saved_config["models"]["qwen-vl-plus"]["pricing_rules"][0]["price_cny_per_unit"],
            0.0000015,
        )

    def test_refresh_with_apply_updates_cny_prices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "model_prices.yaml"
            config_path.write_text(_config_text(), encoding="utf-8")

            report = refresh_price_catalog(
                config_path,
                apply_changes=True,
                allow_large_change=True,
                fetcher=_fake_fetcher,
                generated_at="2026-08-01T10:00:00+08:00",
            )
            saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(report["applied_model_count"], 2)
        self.assertEqual(report["preflight_warning_count"], 1)
        self.assertEqual(saved_config["models"]["qwen-vl-plus"]["price_source"], "official_public_price_page")
        self.assertEqual(saved_config["models"]["qwen-vl-plus"]["price_confidence"], "official_public_page")
        self.assertEqual(
            saved_config["models"]["deepseek-v4-flash"]["pricing_rules"][0]["price_cny_per_unit"],
            0.000001,
        )

    def test_preflight_blocks_large_price_change_without_explicit_allow(self) -> None:
        old_rules = [
            {"unit_type": "input_tokens", "price_cny_per_unit": 0.000001},
            {"unit_type": "output_tokens", "price_cny_per_unit": 0.000001},
        ]
        new_rules = [
            {"unit_type": "input_tokens", "price_cny_per_unit": 0.000003},
            {"unit_type": "output_tokens", "price_cny_per_unit": 0.000001},
        ]

        result = preflight_price_update(old_rules, new_rules)

        self.assertEqual(result["preflight_status"], "fail")
        self.assertEqual(result["preflight_checks"][-1]["check_name"], "large_price_change")

    def test_preflight_allows_large_price_change_with_warning_when_explicit(self) -> None:
        old_rules = [
            {"unit_type": "input_tokens", "price_cny_per_unit": 0.000001},
            {"unit_type": "output_tokens", "price_cny_per_unit": 0.000001},
        ]
        new_rules = [
            {"unit_type": "input_tokens", "price_cny_per_unit": 0.000003},
            {"unit_type": "output_tokens", "price_cny_per_unit": 0.000001},
        ]

        result = preflight_price_update(old_rules, new_rules, allow_large_change=True)

        self.assertEqual(result["preflight_status"], "warning")
        self.assertEqual(result["preflight_checks"][-1]["status"], "warning")

    def test_preflight_rejects_missing_required_unit(self) -> None:
        result = preflight_price_update(
            old_rules=[],
            new_rules=[{"unit_type": "input_tokens", "price_cny_per_unit": 0.000001}],
        )

        self.assertEqual(result["preflight_status"], "fail")
        self.assertEqual(result["preflight_checks"][0]["check_name"], "required_units")

    def test_preflight_rejects_non_positive_and_non_finite_price(self) -> None:
        for invalid_price in (0, -0.000001, float("nan"), float("inf")):
            result = preflight_price_update(
                old_rules=[],
                new_rules=[
                    {"unit_type": "input_tokens", "price_cny_per_unit": invalid_price},
                    {"unit_type": "output_tokens", "price_cny_per_unit": 0.000001},
                ],
            )

            self.assertEqual(result["preflight_status"], "fail")

    def test_preflight_rejects_non_numeric_price(self) -> None:
        result = preflight_price_update(
            old_rules=[],
            new_rules=[
                {"unit_type": "input_tokens", "price_cny_per_unit": "not-a-number"},
                {"unit_type": "output_tokens", "price_cny_per_unit": 0.000001},
            ],
        )

        self.assertEqual(result["preflight_status"], "fail")
        self.assertEqual(result["preflight_checks"][0]["check_name"], "numeric_price")

    def test_refresh_rejects_zero_candidate_price(self) -> None:
        zero_price_sample = """
        <table>
        <tr><td>输入</td><td>0</td><td>每百万 tokens</td></tr>
        <tr><td>输出</td><td>2</td><td>每百万 tokens</td></tr>
        </table>
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "model_prices.yaml"
            config_path.write_text(_config_text(), encoding="utf-8")

            report = refresh_price_catalog(
                config_path,
                apply_changes=True,
                fetcher=lambda _url: zero_price_sample,
                generated_at="2026-08-01T10:00:00+08:00",
            )
            saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(report["applied_model_count"], 0)
        self.assertEqual(report["preflight_failed_count"], 1)
        self.assertEqual(report["refresh_items"][0]["status"], "preflight_failed")
        self.assertEqual(
            saved_config["models"]["qwen-vl-plus"]["pricing_rules"][0]["price_cny_per_unit"],
            0.0000015,
        )

    def test_refresh_report_file_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = write_refresh_report(
                {"report_type": "price_catalog_refresh", "refresh_items": []},
                Path(tmp_dir) / "price_report.json",
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(data["report_type"], "price_catalog_refresh")


if __name__ == "__main__":
    unittest.main()
