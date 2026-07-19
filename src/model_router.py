"""为每种任务类型选择供应商和模型名称。

MVP 使用 config/routing_rules.yaml 中的固定路由规则。后续版本可根据成本、延迟、预算和模型能力进行动态路由。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_routing_rules(config_path: str | Path) -> dict[str, dict[str, str]]:
    """从 YAML 配置文件中读取固定路由规则。"""

    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["routing_rules"]


def select_model(task_type: str, routing_rules: dict[str, dict[str, str]]) -> dict[str, str]:
    """根据任务类型返回供应商和模型名称。"""

    if task_type not in routing_rules:
        raise KeyError(f"未配置的任务类型: {task_type}")

    rule = routing_rules[task_type]
    return {
        "provider": rule["provider"],
        "model_name": rule["model_name"],
    }
