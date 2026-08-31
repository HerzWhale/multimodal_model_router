"""构造、校验并消费显式路由计划；旧固定规则仅保留兼容。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROUTE_PLAN_SCHEMA_VERSION = "v1"
PIPELINE_TASK_GROUPS = {
    "ocr": "ocr",
    "keyframe_ocr": "ocr",
    "vision_understanding": "vision_understanding",
    "keyframe_vision_understanding": "vision_understanding",
    "speech_to_text": "speech_to_text",
    "text_analysis": "text_analysis",
}
PIPELINE_BACKEND_KEYS = {
    "ocr": ("ocr", "keyframe_ocr", "ocr_backend"),
    "vision_understanding": (
        "vision_understanding",
        "keyframe_vision_understanding",
        "vision_understanding_backend",
    ),
    "speech_to_text": ("speech_to_text", "speech_to_text", "speech_to_text_backend"),
    "text_analysis": ("text_analysis", "text_analysis", "text_analysis_backend"),
}
ROUTING_TASK_TYPES = {
    "ocr": "ocr",
    "vision_understanding": "visual_understanding",
    "speech_to_text": "speech_to_text",
    "text_analysis": "text_analysis",
}


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


def _settings_fingerprint(settings: dict[str, Any]) -> str:
    """为会影响路由的配置生成稳定指纹。"""

    relevant = {
        "pipelines": settings.get("pipelines"),
        "backends": settings.get("backends"),
    }
    payload = json.dumps(relevant, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runtime_type(provider: str, model_name: str, backend: dict[str, Any]) -> str:
    """根据配置判断后端是 mock、本地模型还是真实 API。"""

    configured = backend.get("runtime_type")
    if configured:
        return str(configured)
    if model_name.startswith("mock-") or provider == "local":
        return "mock"
    if provider == "paddlepaddle":
        return "local_model"
    return "live_api"


def _apply_pipeline_overrides(
    pipelines: dict[str, Any],
    *,
    ocr_backend: str | None = None,
    vision_understanding_backend: str | None = None,
    speech_to_text_backend: str | None = None,
    text_analysis_backend: str | None = None,
) -> dict[str, dict[str, str]]:
    """复制媒体处理链，并应用预检查时显式选择的全局后端。"""

    selected = {
        str(media_type): {str(task): str(backend) for task, backend in pipeline.items()}
        for media_type, pipeline in pipelines.items()
        if isinstance(pipeline, dict)
    }
    overrides = {
        "ocr": ocr_backend,
        "vision_understanding": vision_understanding_backend,
        "speech_to_text": speech_to_text_backend,
        "text_analysis": text_analysis_backend,
    }
    for pipeline in selected.values():
        for task_name in tuple(pipeline):
            group = PIPELINE_TASK_GROUPS.get(task_name)
            if group and overrides[group] is not None:
                pipeline[task_name] = str(overrides[group])
    return selected


def build_route_plan(
    settings: dict[str, Any],
    *,
    preflight_status: str,
    policy_name: str,
    source_settings: str,
    generated_at: str | None = None,
    warning_messages: list[str] | None = None,
    ocr_backend: str | None = None,
    vision_understanding_backend: str | None = None,
    speech_to_text_backend: str | None = None,
    text_analysis_backend: str | None = None,
    selection_decisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """从 settings 的媒体处理链和后端目录构造可执行路由计划。"""

    pipelines = settings.get("pipelines")
    backends = settings.get("backends")
    if not isinstance(pipelines, dict) or not pipelines:
        raise ValueError("settings.yaml 缺少非空 pipelines 配置。")
    if not isinstance(backends, dict) or not backends:
        raise ValueError("settings.yaml 缺少非空 backends 配置。")

    selected_pipelines = _apply_pipeline_overrides(
        pipelines,
        ocr_backend=ocr_backend,
        vision_understanding_backend=vision_understanding_backend,
        speech_to_text_backend=speech_to_text_backend,
        text_analysis_backend=text_analysis_backend,
    )
    resolved_backends: dict[str, dict[str, str]] = {}
    requires_live_api = False
    for pipeline in selected_pipelines.values():
        for task_name, backend_name in pipeline.items():
            backend_group = PIPELINE_TASK_GROUPS.get(task_name)
            if backend_group is None:
                raise ValueError(f"路由计划包含不支持的任务：{task_name}")
            group = backends.get(backend_group)
            backend = group.get(backend_name) if isinstance(group, dict) else None
            if not isinstance(backend, dict):
                raise ValueError(f"未在 backends.{backend_group} 中配置后端：{backend_name}")
            provider = str(backend.get("provider") or "")
            model_name = str(backend.get("model_name") or "")
            if not provider or not model_name:
                raise ValueError(f"后端 {backend_group}.{backend_name} 缺少 provider 或 model_name。")
            runtime_type = _runtime_type(provider, model_name, backend)
            requires_live_api = requires_live_api or runtime_type == "live_api"
            key = f"{backend_group}.{backend_name}"
            resolved_backends[key] = {
                "backend_group": backend_group,
                "backend_name": backend_name,
                "provider": provider,
                "model_name": model_name,
                "runtime_type": runtime_type,
            }

    return {
        "schema_version": ROUTE_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_settings": source_settings,
        "settings_fingerprint": _settings_fingerprint(settings),
        "preflight_status": preflight_status,
        "policy_name": policy_name,
        "selected_pipelines": selected_pipelines,
        "resolved_backends": resolved_backends,
        "requires_live_api": requires_live_api,
        "warning_messages": list(warning_messages or []),
        "selection_decisions": json.loads(json.dumps(selection_decisions or [], ensure_ascii=False)),
    }


def validate_route_plan(route_plan: dict[str, Any], settings: dict[str, Any]) -> None:
    """拒绝失败、未知版本或与当前 settings 漂移的计划。"""

    if route_plan.get("schema_version") != ROUTE_PLAN_SCHEMA_VERSION:
        raise ValueError("不支持的路由计划 schema_version。")
    status = route_plan.get("preflight_status")
    if status not in {"pass", "warning", "fail"}:
        raise ValueError("路由计划 preflight_status 必须是 pass、warning 或 fail。")
    if status == "fail":
        raise ValueError("路由计划预检查状态为 fail，禁止执行。")
    if route_plan.get("settings_fingerprint") != _settings_fingerprint(settings):
        raise ValueError("路由计划与当前 settings.yaml 已发生配置漂移，请重新生成预检查计划。")

    selected_pipelines = route_plan.get("selected_pipelines")
    resolved_backends = route_plan.get("resolved_backends")
    if not isinstance(selected_pipelines, dict) or not isinstance(resolved_backends, dict):
        raise ValueError("路由计划缺少 selected_pipelines 或 resolved_backends。")

    # 计划可以包含显式后端覆盖，因此只校验引用的后端快照，不要求 pipeline 等于默认选择。
    current_backends = settings.get("backends") or {}
    for key, snapshot in resolved_backends.items():
        if not isinstance(snapshot, dict):
            raise ValueError(f"路由计划后端快照格式错误：{key}")
        group_name = snapshot.get("backend_group")
        backend_name = snapshot.get("backend_name")
        group = current_backends.get(group_name) if isinstance(current_backends, dict) else None
        backend = group.get(backend_name) if isinstance(group, dict) else None
        if not isinstance(backend, dict):
            raise ValueError(f"路由计划引用了当前配置中不存在的后端：{key}")
        current_snapshot = {
            "backend_group": str(group_name),
            "backend_name": str(backend_name),
            "provider": str(backend.get("provider") or ""),
            "model_name": str(backend.get("model_name") or ""),
            "runtime_type": _runtime_type(
                str(backend.get("provider") or ""),
                str(backend.get("model_name") or ""),
                backend,
            ),
        }
        if snapshot != current_snapshot:
            raise ValueError(f"路由计划后端快照与当前配置不一致：{key}")

    # 确保 pipeline 中的每个引用都存在于已校验快照中。
    for media_type, pipeline in selected_pipelines.items():
        if not isinstance(pipeline, dict):
            raise ValueError(f"路由计划中的 {media_type} pipeline 格式错误。")
        for task_name, backend_name in pipeline.items():
            group_name = PIPELINE_TASK_GROUPS.get(str(task_name))
            if not group_name or f"{group_name}.{backend_name}" not in resolved_backends:
                raise ValueError(f"路由计划中的任务后端没有有效快照：{media_type}.{task_name}")

    selection_decisions = route_plan.get("selection_decisions", [])
    if not isinstance(selection_decisions, list):
        raise ValueError("路由计划 selection_decisions 必须是列表。")
    for decision in selection_decisions:
        if not isinstance(decision, dict):
            raise ValueError("路由计划 selection_decisions 包含非法记录。")
        task_type = decision.get("task_type")
        candidate = decision.get("recommended_candidate")
        recommendation_status = decision.get("recommendation_status")
        unmet_constraints = decision.get("unmet_constraints")
        if recommendation_status not in {"pass", "warning"}:
            raise ValueError("路由计划中的 recommendation_status 必须是 pass 或 warning。")
        if not isinstance(task_type, str) or not isinstance(candidate, str):
            raise ValueError("路由计划推荐缺少 task_type 或 recommended_candidate。")
        if not isinstance(unmet_constraints, list):
            raise ValueError("路由计划推荐的 unmet_constraints 必须是列表。")
        if any(not isinstance(item, str) or not item for item in unmet_constraints):
            raise ValueError("路由计划推荐的 unmet_constraints 只能包含非空字符串。")
        if recommendation_status == "pass" and unmet_constraints:
            raise ValueError("pass 推荐不能包含 unmet_constraints。")
        if recommendation_status == "warning" and not unmet_constraints:
            raise ValueError("warning 推荐必须包含 unmet_constraints。")
        if recommendation_status == "warning" and status != "warning":
            raise ValueError("warning 推荐不能写入 pass 路由计划。")
        if not isinstance(decision.get("non_compared_tasks", []), list):
            raise ValueError("路由计划推荐的 non_compared_tasks 必须是列表。")
        if not isinstance(decision.get("evidence_source", ""), str):
            raise ValueError("路由计划推荐的 evidence_source 必须是字符串。")
        if not isinstance(decision.get("candidate_summary", {}), dict):
            raise ValueError("路由计划推荐的 candidate_summary 必须是对象。")
        group = current_backends.get(task_type) if isinstance(current_backends, dict) else None
        if not isinstance(group, dict) or candidate not in group:
            raise ValueError(f"路由计划推荐了当前配置中不存在的后端：{task_type}.{candidate}")
        for media_type, pipeline in selected_pipelines.items():
            for task_name, backend_name in pipeline.items():
                if PIPELINE_TASK_GROUPS.get(str(task_name)) == task_type and backend_name != candidate:
                    raise ValueError(
                        f"路由计划推荐与实际 pipeline 不一致：{media_type}.{task_name}"
                    )

    expected_live_api = any(
        item.get("runtime_type") == "live_api" for item in resolved_backends.values()
    )
    if bool(route_plan.get("requires_live_api")) != expected_live_api:
        raise ValueError("路由计划 requires_live_api 与后端快照不一致。")


def load_route_plan(path: str | Path) -> dict[str, Any]:
    """读取路由计划 JSON。"""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("路由计划必须是 JSON 对象。")
    return data


def route_plan_backends_for_media(route_plan: dict[str, Any], media_type: str) -> dict[str, str]:
    """返回某种媒体在流水线内部实际使用的四类后端。"""

    pipelines = route_plan.get("selected_pipelines")
    pipeline = pipelines.get(media_type) if isinstance(pipelines, dict) else None
    if not isinstance(pipeline, dict):
        raise ValueError(f"路由计划未配置媒体类型：{media_type}")
    result: dict[str, str] = {}
    for group_name, (regular_task, video_task, output_key) in PIPELINE_BACKEND_KEYS.items():
        task_name = video_task if media_type == "video" else regular_task
        result[output_key] = str(pipeline.get(task_name, "mock"))
    return result


def routing_rules_for_media(route_plan: dict[str, Any], media_type: str) -> dict[str, dict[str, str]]:
    """把媒体路由计划转换成调用记录所需的任务模型映射。"""

    pipeline_backends = route_plan_backends_for_media(route_plan, media_type)
    resolved = route_plan["resolved_backends"]
    rules: dict[str, dict[str, str]] = {}
    for group_name, (_, _, output_key) in PIPELINE_BACKEND_KEYS.items():
        backend_name = pipeline_backends[output_key]
        snapshot = resolved.get(f"{group_name}.{backend_name}")
        if snapshot is None:
            continue
        rules[ROUTING_TASK_TYPES[group_name]] = {
            "provider": str(snapshot["provider"]),
            "model_name": str(snapshot["model_name"]),
        }
    if "text_analysis" in rules:
        rules["summary_merge"] = dict(rules["text_analysis"])
    return rules


def routing_rules_from_route_plan(route_plan: dict[str, Any]) -> tuple[dict[str, dict[str, str]], list[str]]:
    """为现有任务级预检查生成代表路由，并报告跨媒体冲突。"""

    candidates: dict[str, list[tuple[str, dict[str, str]]]] = {}
    pipelines = route_plan.get("selected_pipelines") or {}
    for media_type in sorted(pipelines):
        rules = routing_rules_for_media(route_plan, str(media_type))
        for task_type, rule in rules.items():
            candidates.setdefault(task_type, []).append((str(media_type), rule))

    rules: dict[str, dict[str, str]] = {}
    conflicts: list[str] = []
    for task_type, options in candidates.items():
        rules[task_type] = dict(options[0][1])
        distinct = {(item[1]["provider"], item[1]["model_name"]) for item in options}
        if len(distinct) > 1:
            conflicts.append(f"任务 {task_type} 在不同媒体 pipeline 中选择了多个后端，任务级预检查只展示代表路线。")
    return rules, conflicts
