"""记录模型调用成本、耗时、状态和技术错误。

每次模型调用都应经过这个模块，系统才能生成模型调用明细和批次级成本/延迟统计。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


VALID_STATUSES = {"success", "failed", "skipped"}
DEFAULT_PRICE_SOURCE = "unspecified"
DEFAULT_PRICE_UPDATED_AT = None
DEFAULT_PRICE_CONFIDENCE = "unknown"


def load_model_prices(config_path: str | Path) -> dict[str, dict[str, Any]]:
    """从 YAML 配置文件中读取模型价格表。"""

    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["models"]


def _find_quantity(units: list[dict[str, Any]], unit_type: str) -> float:
    """从用量列表中找到指定单位的数量。"""

    for item in units:
        if item["unit_type"] == unit_type:
            return float(item["quantity"])
    raise ValueError(f"缺少计价单位: {unit_type}")


def calculate_cost_cny(
    model_name: str,
    input_units: list[dict[str, Any]],
    model_prices: dict[str, dict[str, Any]],
    output_units: list[dict[str, Any]] | None = None,
) -> float:
    """根据模型价格表和输入用量计算单次调用成本。"""

    price_rule = model_prices[model_name]
    if "pricing_rules" in price_rule:
        all_units = input_units + (output_units or [])
        total_cost = 0.0
        for rule in price_rule["pricing_rules"]:
            quantity = _find_quantity(all_units, rule["unit_type"])
            total_cost += quantity * float(rule["price_cny_per_unit"])
        return round(total_cost, 6)

    pricing_unit = price_rule["pricing_unit"]
    quantity = _find_quantity(input_units, pricing_unit)
    return round(quantity * float(price_rule["price_cny_per_unit"]), 6)


def build_price_metadata(model_name: str, model_prices: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """生成价格目录元数据，说明本次成本估算依据是否可靠。"""

    price_rule = model_prices[model_name]
    return {
        "cost_estimation_method": "price_catalog",
        "price_source": price_rule.get("price_source", DEFAULT_PRICE_SOURCE),
        "price_updated_at": price_rule.get("price_updated_at", DEFAULT_PRICE_UPDATED_AT),
        "price_confidence": price_rule.get("price_confidence", DEFAULT_PRICE_CONFIDENCE),
    }


def build_model_call_record(
    *,
    call_id: str,
    batch_id: str,
    file_id: str,
    task_type: str,
    provider: str,
    model_name: str,
    input_units: list[dict[str, Any]],
    output_units: list[dict[str, Any]],
    latency_ms: int,
    started_at: str,
    status: str,
    error_message: str | None,
    model_prices: dict[str, dict[str, Any]],
    response_model_name: str | None = None,
    response_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成一条模型调用记录。"""

    if status not in VALID_STATUSES:
        raise ValueError(f"不支持的调用状态: {status}")

    record = {
        "call_id": call_id,
        "batch_id": batch_id,
        "file_id": file_id,
        "task_type": task_type,
        "provider": provider,
        "model_name": model_name,
        "response_model_name": response_model_name,
        "input_units": input_units,
        "output_units": output_units,
        "cost_cny": calculate_cost_cny(model_name, input_units, model_prices, output_units),
        **build_price_metadata(model_name, model_prices),
        "latency_ms": latency_ms,
        "started_at": started_at,
        "status": status,
        "error_message": error_message,
    }
    if response_diagnostics:
        record["response_diagnostics"] = response_diagnostics
    return record
