"""事前模型路由策略与约束评估。

本模块不调用任何外部模型，只把已有批次数据放进不同业务目标下评估：
预算优先、延迟优先、质量优先和平衡策略。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from model_catalog import UNKNOWN_VALUE_TEXT, as_float, summarize_catalog


POLICY_DEFINITIONS: dict[str, dict[str, Any]] = {
    "budget_first": {
        "display_name": "成本优先",
        "primary_objective": "在预算范围内尽量保持流程可用，优先控制真实模型调用范围。",
        "default_constraints": {
            "budget_limit_cny": None,
            "p95_latency_limit_ms": None,
            "min_real_coverage_rate": 0.25,
        },
    },
    "latency_first": {
        "display_name": "延迟优先",
        "primary_objective": "优先压低 P95 延迟，识别慢调用并给出减慢风险处理方案。",
        "default_constraints": {
            "budget_limit_cny": None,
            "p95_latency_limit_ms": 2000,
            "min_real_coverage_rate": 0.25,
        },
    },
    "quality_first": {
        "display_name": "质量优先",
        "primary_objective": "优先提高真实模型覆盖率，并指出 mock 上游对质量结论的限制。",
        "default_constraints": {
            "budget_limit_cny": None,
            "p95_latency_limit_ms": None,
            "min_real_coverage_rate": 0.7,
        },
    },
    "balanced": {
        "display_name": "平衡策略",
        "primary_objective": "在成本、延迟和真实模型覆盖率之间做折中，适合内容平台试点。",
        "default_constraints": {
            "budget_limit_cny": None,
            "p95_latency_limit_ms": 3500,
            "min_real_coverage_rate": 0.4,
        },
    },
}


UPSTREAM_UPGRADE_PRIORITY = [
    {
        "task_type": "ocr",
        "recommended_replacement": "真实 OCR 模型",
        "reason": "OCR 直接决定图片文字和视频关键帧文字证据是否可靠，优先级最高。",
    },
    {
        "task_type": "speech_to_text",
        "recommended_replacement": "真实 ASR 语音识别模型",
        "reason": "ASR 决定视频音频证据是否可用，适合在视频样本增加后优先验证。",
    },
    {
        "task_type": "visual_understanding",
        "recommended_replacement": "真实视觉理解模型",
        "reason": "视觉理解补足非文字画面信息，是提高图片/视频主题判断质量的关键。",
    },
]

ALLOWED_CONSTRAINT_KEYS = {
    "budget_limit_cny",
    "p95_latency_limit_ms",
    "min_real_coverage_rate",
}


def list_policy_names() -> list[str]:
    """返回当前支持的策略名称。"""

    return list(POLICY_DEFINITIONS.keys())


def get_policy_definition(policy_name: str) -> dict[str, Any]:
    """读取策略定义，策略不存在时抛出清晰错误。"""

    if policy_name not in POLICY_DEFINITIONS:
        raise ValueError(f"不支持的路由策略: {policy_name}")
    return POLICY_DEFINITIONS[policy_name]


def build_constraints(policy_name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """根据策略默认值和用户传入值生成约束。"""

    definition = get_policy_definition(policy_name)
    constraints = dict(definition["default_constraints"])
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                constraints[key] = value
    return constraints


def load_policy_config(config_path: str | Path) -> dict[str, Any]:
    """从 YAML 文件读取路由策略约束配置。"""

    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return normalize_policy_config(data, source_path=path)


def normalize_policy_config(data: dict[str, Any] | None, *, source_path: str | Path | None = None) -> dict[str, Any]:
    """校验并标准化路由策略配置。"""

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("路由策略配置必须是 YAML 字典结构。")

    raw_policies = data.get("policies", {})
    if raw_policies is None:
        raw_policies = {}
    if not isinstance(raw_policies, dict):
        raise ValueError("policies 必须是字典结构。")

    unknown_policies = sorted(set(raw_policies) - set(POLICY_DEFINITIONS))
    if unknown_policies:
        raise ValueError(f"发现不支持的策略配置: {', '.join(unknown_policies)}")

    policy_overrides: dict[str, dict[str, Any]] = {}
    for policy_name in list_policy_names():
        raw_constraints = raw_policies.get(policy_name) or {}
        if not isinstance(raw_constraints, dict):
            raise ValueError(f"{policy_name} 的约束配置必须是字典结构。")

        unknown_constraints = sorted(set(raw_constraints) - ALLOWED_CONSTRAINT_KEYS)
        if unknown_constraints:
            raise ValueError(f"{policy_name} 包含不支持的约束字段: {', '.join(unknown_constraints)}")

        policy_overrides[policy_name] = {
            key: raw_constraints[key]
            for key in ALLOWED_CONSTRAINT_KEYS
            if key in raw_constraints
        }

    multipliers = data.get("budget_expansion_multipliers", [2, 5, 10])
    if not isinstance(multipliers, list) or not multipliers:
        raise ValueError("budget_expansion_multipliers 必须是非空列表。")

    normalized_multipliers: list[int] = []
    for value in multipliers:
        multiplier = int(value)
        if multiplier <= 0:
            raise ValueError("budget_expansion_multipliers 中的倍数必须大于 0。")
        normalized_multipliers.append(multiplier)

    return {
        "schema_version": data.get("schema_version", UNKNOWN_VALUE_TEXT),
        "description": data.get("description", UNKNOWN_VALUE_TEXT),
        "config_source": str(source_path) if source_path is not None else "inline",
        "policy_overrides": policy_overrides,
        "budget_expansion_multipliers": tuple(normalized_multipliers),
        "notes": data.get("notes", []),
    }


def _get_nested(data: dict[str, Any], path: list[str], missing_notes: list[str]) -> Any:
    """读取嵌套字段；缺失时记录说明。"""

    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            missing_notes.append(f"当前数据未提供：{'.'.join(path)}")
            return UNKNOWN_VALUE_TEXT
        current = current[key]
    return current


def _constraint_result(
    *,
    name: str,
    observed_value: Any,
    limit_value: Any,
    pass_when_less_or_equal: bool = True,
) -> dict[str, Any]:
    """生成单项约束检查结果。"""

    if observed_value == UNKNOWN_VALUE_TEXT or limit_value in {None, UNKNOWN_VALUE_TEXT}:
        return {
            "constraint_name": name,
            "observed_value": observed_value,
            "limit_value": limit_value if limit_value is not None else UNKNOWN_VALUE_TEXT,
            "status": "unknown",
            "reason": "当前数据未提供，不能判断该约束是否满足。",
        }

    observed = as_float(observed_value)
    limit = as_float(limit_value)
    passed = observed <= limit if pass_when_less_or_equal else observed >= limit
    relation = "不超过" if pass_when_less_or_equal else "不低于"
    return {
        "constraint_name": name,
        "observed_value": round(observed, 6),
        "limit_value": round(limit, 6),
        "status": "pass" if passed else "fail",
        "reason": f"观测值 {observed:.6f} {'满足' if passed else '未满足'} {relation} {limit:.6f} 的约束。",
    }


def _overall_status(checks: list[dict[str, Any]]) -> str:
    """汇总多项约束的整体状态。"""

    statuses = {check.get("status") for check in checks}
    if "fail" in statuses:
        return "fail"
    if "unknown" in statuses:
        return "partial_unknown"
    return "pass"


def _top_cost_entries(catalog: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """识别主要成本来源。"""

    return sorted(catalog, key=lambda entry: as_float(entry.get("total_cost_cny")), reverse=True)[:limit]


def _slowest_entries(catalog: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """识别最慢模型任务。"""

    return sorted(catalog, key=lambda entry: as_float(entry.get("p95_latency_ms")), reverse=True)[:limit]


def _strategy_recommendation(policy_name: str, catalog_summary: dict[str, Any], status: str) -> dict[str, Any]:
    """根据策略输出组合建议和取舍说明。"""

    has_mock = catalog_summary.get("mock_model_calls", 0) > 0
    if policy_name == "budget_first":
        return {
            "recommended_combo": [
                "保留 DeepSeek 文本分析真实调用",
                "OCR、视觉理解、语音识别继续小样本验证，不直接全量替换",
                "把真实多模态模型接入放到受控 live test 中逐步扩大",
            ],
            "tradeoff_explanation": "成本优先时，当前批次成本很低，但 mock 上游成本不能代表真实账单；因此重点是限制真实多模态调用范围。",
            "risk_warnings": ["mock 成本不是供应商真实成本"] if has_mock else [],
        }
    if policy_name == "latency_first":
        return {
            "recommended_combo": [
                "优先优化或异步化 DeepSeek 文本分析",
                "为 OCR、ASR、视觉理解单独设置 P95 延迟预算",
                "对慢调用文件做缓存、队列化或拆分处理",
            ],
            "tradeoff_explanation": "延迟优先时，当前可观测瓶颈集中在真实文本分析；mock 上游延迟为 0，不能外推真实多模态延迟。",
            "risk_warnings": ["mock latency_ms 为 0 不代表真实延迟"] if has_mock else [],
        }
    if policy_name == "quality_first":
        return {
            "recommended_combo": [
                "优先把 OCR 替换为真实模型",
                "随后接入真实 ASR",
                "再接入真实视觉理解模型，并建立小样本人工标注校验",
            ],
            "tradeoff_explanation": "质量优先时，最大问题不是文本分析，而是图片/视频上游证据仍是 mock，不能支撑真实质量判断。",
            "risk_warnings": ["当前不能证明真实 OCR、ASR、视觉理解质量"] if has_mock else [],
        }
    if policy_name == "balanced":
        return {
            "recommended_combo": [
                "保持 DeepSeek 文本分析作为真实分析底座",
                "先接入真实 OCR 和 ASR 的小样本验证",
                "保留成本、延迟、真实覆盖率三项指标作为试点门槛",
            ],
            "tradeoff_explanation": "平衡策略适合内容平台试点：先证明关键证据链可靠，再逐步扩大真实多模态调用比例。",
            "risk_warnings": ["当前真实覆盖率仍不足，适合试点不适合宣称全真实多模态平台"] if status != "pass" else [],
        }
    return {
        "recommended_combo": [],
        "tradeoff_explanation": UNKNOWN_VALUE_TEXT,
        "risk_warnings": [],
    }


def evaluate_routing_policy(
    policy_name: str,
    catalog: list[dict[str, Any]],
    batch_report: dict[str, Any],
    *,
    constraints_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在指定策略下评估当前批次模型组合。"""

    definition = get_policy_definition(policy_name)
    constraints = build_constraints(policy_name, constraints_override)
    missing_notes: list[str] = []
    total_cost_cny = _get_nested(batch_report, ["cost_stats", "total_cost_cny"], missing_notes)
    fallback_budget = _get_nested(batch_report, ["cost_stats", "budget_limit_cny"], missing_notes)
    p95_latency_ms = _get_nested(batch_report, ["latency_stats", "p95_model_latency_ms"], missing_notes)
    catalog_summary = summarize_catalog(catalog)
    budget_limit = constraints.get("budget_limit_cny") if constraints.get("budget_limit_cny") is not None else fallback_budget

    checks = [
        _constraint_result(
            name="budget_limit_cny",
            observed_value=total_cost_cny,
            limit_value=budget_limit,
            pass_when_less_or_equal=True,
        ),
        _constraint_result(
            name="p95_latency_limit_ms",
            observed_value=p95_latency_ms,
            limit_value=constraints.get("p95_latency_limit_ms"),
            pass_when_less_or_equal=True,
        ),
        _constraint_result(
            name="min_real_coverage_rate",
            observed_value=catalog_summary["real_coverage_rate"],
            limit_value=constraints.get("min_real_coverage_rate"),
            pass_when_less_or_equal=False,
        ),
    ]
    status = _overall_status(checks)
    recommendation = _strategy_recommendation(policy_name, catalog_summary, status)

    return {
        "policy_name": policy_name,
        "display_name": definition["display_name"],
        "primary_objective": definition["primary_objective"],
        "constraints": constraints,
        "constraint_checks": checks,
        "constraint_status": status,
        "catalog_summary": catalog_summary,
        "top_cost_model_tasks": _top_cost_entries(catalog),
        "slowest_model_tasks": _slowest_entries(catalog),
        "recommendation": recommendation,
        "missing_data_notes": sorted(set(missing_notes)),
    }


def simulate_budget_expansion(
    batch_report: dict[str, Any],
    catalog: list[dict[str, Any]],
    *,
    multipliers: tuple[int, ...] = (2, 5, 10),
) -> list[dict[str, Any]]:
    """模拟预算扩大后最值得替换的 mock 上游模块。

    这里不估算真实供应商价格，只给出基于当前 mock 风险的替换优先级。
    """

    base_budget = batch_report.get("cost_stats", {}).get("budget_limit_cny", UNKNOWN_VALUE_TEXT)
    mock_task_types = {entry["task_type"] for entry in catalog if entry.get("is_mock")}
    scenarios: list[dict[str, Any]] = []

    for multiplier in multipliers:
        expanded_budget = (
            round(as_float(base_budget) * multiplier, 6)
            if base_budget != UNKNOWN_VALUE_TEXT
            else UNKNOWN_VALUE_TEXT
        )
        upgrade_priority = [
            item for item in UPSTREAM_UPGRADE_PRIORITY if item["task_type"] in mock_task_types
        ]
        scenarios.append(
            {
                "budget_multiplier": multiplier,
                "base_budget_cny": base_budget,
                "expanded_budget_cny": expanded_budget,
                "upgrade_priority": upgrade_priority,
                "assumption": "只模拟预算空间扩大后的优先级，不编造真实 OCR、ASR、视觉模型价格或质量结果。",
                "recommendation": _budget_expansion_recommendation(multiplier, upgrade_priority),
            }
        )

    return scenarios


def _budget_expansion_recommendation(multiplier: int, upgrade_priority: list[dict[str, Any]]) -> str:
    """根据预算倍数生成谨慎建议。"""

    if not upgrade_priority:
        return "当前没有发现 mock 上游任务，预算扩展优先级当前数据未提供。"
    if multiplier == 2:
        return "建议先选择一个最高风险上游任务做小样本真实接入，优先验证证据质量与真实延迟。"
    if multiplier == 5:
        return "建议覆盖前两个上游任务，并继续保留调用级成本与 P95 延迟门槛。"
    return "可以考虑把 OCR、ASR、视觉理解全部纳入受控 live test，但仍需逐步放量，不能一次性全量替换。"
