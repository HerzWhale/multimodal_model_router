"""从官方公开价格页刷新本地模型价格目录。

本模块只访问公开文档页，不登录供应商控制台，不读取真实账单，不调用付费模型 API。
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

import yaml


FETCH_USER_AGENT = "multimodal-model-router-price-updater/1.0"
PRICE_CONFIDENCE_OFFICIAL_PAGE = "official_public_page"
PRICE_SOURCE_AUTO_FETCH = "official_public_price_page"
MAX_PRICE_CHANGE_RATE = 0.5
REQUIRED_TOKEN_UNITS = {"input_tokens", "output_tokens"}


class PriceCatalogUpdateError(ValueError):
    """价格目录刷新错误。"""


def _today() -> str:
    """返回本地日期字符串，用于记录价格目录更新时间。"""

    return datetime.now().astimezone().date().isoformat()


def fetch_url_text(url: str, timeout_seconds: int = 20) -> str:
    """读取公开网页文本。"""

    request = Request(url, headers={"User-Agent": FETCH_USER_AGENT})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8", errors="replace")


def _plain_text(raw_html: str) -> str:
    """把网页 HTML 转成便于正则解析的纯文本。"""

    text = html.unescape(raw_html)
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def _cny_per_token(price_per_million_tokens: str) -> float:
    """把每百万 token 的人民币价格转换成单 token 价格。"""

    return round(float(price_per_million_tokens) / 1_000_000, 12)


def _token_rules(input_price_per_token: float, output_price_per_token: float) -> list[dict[str, Any]]:
    """生成输入和输出 token 的价格规则。"""

    return [
        {"unit_type": "input_tokens", "price_cny_per_unit": input_price_per_token},
        {"unit_type": "output_tokens", "price_cny_per_unit": output_price_per_token},
    ]


def parse_aliyun_qwen_vl_plus(raw_html: str) -> dict[str, Any]:
    """解析阿里云百炼 qwen-vl-plus 官方人民币价格。"""

    text = _plain_text(raw_html)
    input_match = re.search(r"(?:^|\s)输入\s+([0-9]+(?:\.[0-9]+)?)\s+每百万\s*tokens", text)
    output_match = re.search(r"(?:^|\s)输出\s+([0-9]+(?:\.[0-9]+)?)\s+每百万\s*tokens", text)
    if not input_match or not output_match:
        raise PriceCatalogUpdateError("未能从阿里云 qwen-vl-plus 页面解析输入/输出 token 价格。")

    return {
        "currency": "CNY",
        "pricing_rules": _token_rules(
            _cny_per_token(input_match.group(1)),
            _cny_per_token(output_match.group(1)),
        ),
        "raw_price_unit": "CNY_per_million_tokens",
    }


def parse_deepseek_v4_flash(raw_html: str) -> dict[str, Any]:
    """解析 DeepSeek V4 Flash 官方人民币价格，输入价格使用缓存未命中口径。"""

    text = _plain_text(raw_html)
    input_match = re.search(
        r"百万tokens输入（缓存未命中）\s+([0-9]+(?:\.[0-9]+)?)元\s+[0-9]+(?:\.[0-9]+)?元",
        text,
    )
    output_match = re.search(
        r"百万tokens输出\s+([0-9]+(?:\.[0-9]+)?)元\s+[0-9]+(?:\.[0-9]+)?元",
        text,
    )
    if not input_match or not output_match:
        raise PriceCatalogUpdateError("未能从 DeepSeek 页面解析 deepseek-v4-flash 输入/输出 token 价格。")

    return {
        "currency": "CNY",
        "pricing_rules": _token_rules(
            _cny_per_token(input_match.group(1)),
            _cny_per_token(output_match.group(1)),
        ),
        "raw_price_unit": "CNY_per_million_tokens_cache_miss_input",
    }


PARSERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "aliyun_qwen_vl_plus": parse_aliyun_qwen_vl_plus,
    "deepseek_v4_flash": parse_deepseek_v4_flash,
}


def _pricing_rules_changed(old_rules: list[dict[str, Any]], new_rules: list[dict[str, Any]]) -> bool:
    """判断价格规则是否发生变化。"""

    return json.dumps(old_rules, sort_keys=True) != json.dumps(new_rules, sort_keys=True)


def _rules_by_unit(pricing_rules: list[dict[str, Any]]) -> dict[str, float]:
    """按计费单位整理价格规则。"""

    result = {}
    for rule in pricing_rules:
        unit_type = str(rule.get("unit_type") or "")
        price = float(rule.get("price_cny_per_unit"))
        result[unit_type] = price
    return result


def preflight_price_update(
    old_rules: list[dict[str, Any]],
    new_rules: list[dict[str, Any]],
    *,
    allow_large_change: bool = False,
    max_change_rate: float = MAX_PRICE_CHANGE_RATE,
) -> dict[str, Any]:
    """在写回价格目录前检查候选价格是否安全。"""

    checks = []
    status = "pass"
    try:
        new_by_unit = _rules_by_unit(new_rules)
    except (TypeError, ValueError) as exc:
        return {
            "preflight_status": "fail",
            "preflight_checks": [
                {
                    "check_name": "numeric_price",
                    "status": "fail",
                    "message": f"候选价格不是有效数字：{exc}",
                }
            ],
        }

    missing_units = sorted(REQUIRED_TOKEN_UNITS - set(new_by_unit))
    if missing_units:
        status = "fail"
        checks.append(
            {
                "check_name": "required_units",
                "status": "fail",
                "message": "候选价格缺少必要计费单位：" + "、".join(missing_units),
            }
        )
    else:
        checks.append(
            {
                "check_name": "required_units",
                "status": "pass",
                "message": "候选价格包含输入和输出 token 价格。",
            }
        )

    invalid_prices = [
        unit_type
        for unit_type, price in new_by_unit.items()
        if not math.isfinite(price) or price <= 0
    ]
    if invalid_prices:
        status = "fail"
        checks.append(
            {
                "check_name": "positive_price",
                "status": "fail",
                "message": "候选价格必须为正数：" + "、".join(sorted(invalid_prices)),
            }
        )
    else:
        checks.append(
            {
                "check_name": "positive_price",
                "status": "pass",
                "message": "候选价格均为正数。",
            }
        )

    old_by_unit = _rules_by_unit(old_rules) if old_rules else {}
    large_changes = []
    for unit_type, new_price in new_by_unit.items():
        old_price = old_by_unit.get(unit_type)
        if old_price is None or old_price <= 0:
            continue
        change_rate = round((new_price - old_price) / old_price, 6)
        if abs(change_rate) > max_change_rate:
            large_changes.append(
                {
                    "unit_type": unit_type,
                    "old_price_cny_per_unit": old_price,
                    "new_price_cny_per_unit": new_price,
                    "change_rate": change_rate,
                }
            )

    if large_changes:
        check_status = "warning" if allow_large_change and status != "fail" else "fail"
        if check_status == "fail":
            status = "fail"
        elif status == "pass":
            status = "warning"
        checks.append(
            {
                "check_name": "large_price_change",
                "status": check_status,
                "message": "候选价格变化超过阈值，默认需要人工确认。",
                "max_change_rate": max_change_rate,
                "large_changes": large_changes,
            }
        )
    else:
        checks.append(
            {
                "check_name": "large_price_change",
                "status": "pass",
                "message": "候选价格变化未超过阈值。",
                "max_change_rate": max_change_rate,
            }
        )

    return {
        "preflight_status": status,
        "preflight_checks": checks,
    }


def refresh_price_catalog(
    config_path: str | Path,
    *,
    apply_changes: bool = False,
    allow_large_change: bool = False,
    fetcher: Callable[[str], str] = fetch_url_text,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """刷新价格目录；默认只生成候选结果，只有 apply_changes=True 才写回配置。"""

    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    models = data.get("models", {})
    refresh_items = []
    today = _today()

    for model_name, model_config in models.items():
        fetch_config = model_config.get("price_fetch") or {}
        if not fetch_config.get("enabled"):
            continue

        parser_name = fetch_config.get("parser")
        source_url = fetch_config.get("source_url")
        if parser_name not in PARSERS:
            raise PriceCatalogUpdateError(f"{model_name} 的价格解析器不存在: {parser_name}")
        if not source_url:
            raise PriceCatalogUpdateError(f"{model_name} 缺少 price_fetch.source_url")

        item: dict[str, Any] = {
            "model_name": model_name,
            "provider": model_config.get("provider"),
            "source_url": source_url,
            "parser": parser_name,
            "status": "pending",
            "applied": False,
        }
        try:
            fetched = PARSERS[parser_name](fetcher(source_url))
            old_rules = model_config.get("pricing_rules", [])
            new_rules = fetched["pricing_rules"]
            changed = _pricing_rules_changed(old_rules, new_rules)
            preflight = preflight_price_update(
                old_rules,
                new_rules,
                allow_large_change=allow_large_change,
            )
            status = "preflight_failed" if preflight["preflight_status"] == "fail" else "fetched"
            item.update(
                {
                    "status": status,
                    "currency": fetched["currency"],
                    "raw_price_unit": fetched["raw_price_unit"],
                    "old_pricing_rules": old_rules,
                    "fetched_pricing_rules": new_rules,
                    "changed": changed,
                    **preflight,
                    "price_source": PRICE_SOURCE_AUTO_FETCH,
                    "price_updated_at": today,
                    "price_confidence": PRICE_CONFIDENCE_OFFICIAL_PAGE,
                }
            )
            if status == "preflight_failed":
                item["error_message"] = "价格刷新预检查未通过，未写回本地价格目录。"
            if apply_changes and fetched["currency"] == "CNY" and status != "preflight_failed":
                model_config["pricing_rules"] = new_rules
                model_config["price_source"] = PRICE_SOURCE_AUTO_FETCH
                model_config["price_updated_at"] = today
                model_config["price_confidence"] = PRICE_CONFIDENCE_OFFICIAL_PAGE
                item["applied"] = True
        except Exception as exc:
            item.update({"status": "failed", "error_message": str(exc), "changed": False})
        refresh_items.append(item)

    if apply_changes:
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    return {
        "schema_version": "v1",
        "report_type": "price_catalog_refresh",
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "config_path": str(path),
        "apply_changes": apply_changes,
        "allow_large_change": allow_large_change,
        "refreshed_model_count": len(refresh_items),
        "applied_model_count": sum(1 for item in refresh_items if item.get("applied")),
        "failed_model_count": sum(1 for item in refresh_items if item.get("status") in {"failed", "preflight_failed"}),
        "preflight_failed_count": sum(1 for item in refresh_items if item.get("status") == "preflight_failed"),
        "preflight_warning_count": sum(1 for item in refresh_items if item.get("preflight_status") == "warning"),
        "refresh_items": refresh_items,
        "field_notes": {
            "price_source": "价格来源，用于说明模型单价来自官方公开页、人工配置还是mock假设。",
            "price_updated_at": "价格目录更新时间，用于判断本地价格是否可能过期。",
            "price_confidence": "价格可信度，用于区分官方公开页抓取、未验证手工配置和mock价格。",
            "old_pricing_rules": "刷新前本地价格规则，用于审查价格变化。",
            "fetched_pricing_rules": "从官方公开页解析出的候选价格规则。",
            "preflight_status": "价格刷新预检查状态，用于判断候选价格是否允许写回。",
            "preflight_checks": "价格刷新预检查明细，用于说明必要字段、正数价格和大幅变化检查是否通过。",
            "allow_large_change": "是否允许超过阈值的大幅价格变化写回配置；默认不允许，只有人工确认后才应开启。",
            "applied": "是否已经把候选价格写回本地配置。",
        },
    }


def write_refresh_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """写入价格刷新报告。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="从官方公开页刷新模型价格目录。")
    parser.add_argument("config_path", help="模型价格配置文件。")
    parser.add_argument("output_json", help="价格刷新报告输出路径。")
    parser.add_argument("--apply", action="store_true", help="把官方公开页解析出的 CNY 价格写回配置。")
    parser.add_argument("--allow-large-change", action="store_true", help="允许超过50%的价格变化写回配置。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = _parse_args(argv)
    try:
        report = refresh_price_catalog(
            args.config_path,
            apply_changes=args.apply,
            allow_large_change=args.allow_large_change,
        )
        output_path = write_refresh_report(report, args.output_json)
        print(json.dumps({"report": str(output_path)}, ensure_ascii=False, indent=2))
        return 0 if report["failed_model_count"] == 0 else 2
    except PriceCatalogUpdateError as exc:
        print(
            json.dumps(
                {"error_type": "price_catalog_update_error", "error_message": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
