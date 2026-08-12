"""读取运行策略配置。

这里集中放会影响运行结果的可调项：文件类型、后端白名单、视频预处理、OCR 闸门和分类规则。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_POLICY_PATH = PROJECT_ROOT / "config" / "runtime_policy.yaml"


@lru_cache(maxsize=1)
def load_runtime_policy() -> dict[str, Any]:
    """读取运行策略配置。"""

    return yaml.safe_load(RUNTIME_POLICY_PATH.read_text(encoding="utf-8"))


def runtime_policy_section(name: str) -> dict[str, Any]:
    """返回指定配置段。"""

    section = load_runtime_policy().get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"runtime_policy.yaml 的 {name} 必须是对象。")
    return section


def runtime_policy_list(section_name: str, key: str) -> list[str]:
    """返回指定配置列表。"""

    values = runtime_policy_section(section_name).get(key, [])
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise ValueError(f"runtime_policy.yaml 的 {section_name}.{key} 必须是字符串数组。")
    return values
