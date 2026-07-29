"""OCR 后端取舍判断工具。

本模块只读取已有 OCR 闸门报告和延迟拆分报告，生成是否继续使用
PaddleOCR、以及下一步应该评估哪类 OCR 方案的建议。
它不调用任何 OCR 模型，不访问外部 API，也不把候选方案接入主流程。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def default_ocr_candidate_catalog() -> list[dict[str, Any]]:
    """返回当前最小 OCR 候选方案目录。"""

    return [
        {
            "backend_id": "paddleocr_local",
            "backend_name": "PaddleOCR 本地 CPU",
            "deployment_type": "local",
            "integration_status": "current_backend",
            "source_url": "https://github.com/PaddlePaddle/PaddleOCR",
            "known_from_project": True,
            "why_consider": "已经接入项目，已有五张正式图片和151段人工业务文字评估证据。",
            "fit_for": ["无需外部API", "数据不出本机", "继续复用现有工程链路"],
            "risks": [
                "当前关键帧批次质量闸门未通过。",
                "当前本地CPU延迟远高于2秒目标。",
                "Paddle底层推理器对中文模型缓存路径仍不稳定。",
            ],
            "next_test": "仅在接受当前质量与延迟边界时继续保留；否则作为基线对照。",
        },
        {
            "backend_id": "rapidocr_onnxruntime_local",
            "backend_name": "RapidOCR / ONNXRuntime 本地方案",
            "deployment_type": "local",
            "integration_status": "candidate_for_next_eval",
            "source_url": "https://github.com/RapidAI/RapidOCR",
            "known_from_project": False,
            "why_consider": "同属本地OCR路线，主打ONNXRuntime等推理后端，可作为PaddlePaddle运行时的轻量替代候选。",
            "fit_for": ["不想引入外部API费用", "希望规避PaddlePaddle运行时问题", "需要先比较本地推理延迟"],
            "risks": [
                "项目中尚未安装和验证，质量、延迟和中文结构图效果未知。",
                "可能仍需要下载模型权重并消耗本机CPU或GPU。",
                "不能在未实测前宣称优于PaddleOCR。",
            ],
            "next_test": "用同一批 `img_7.jpg`、`img_8.jpg`、`img_9.jpg` 和同一份人工基准做离线对照评估。",
        },
        {
            "backend_id": "tesseract_local",
            "backend_name": "Tesseract 本地 OCR",
            "deployment_type": "local",
            "integration_status": "reference_baseline",
            "source_url": "https://github.com/tesseract-ocr/tesseract",
            "known_from_project": False,
            "why_consider": "成熟开源OCR基线，适合做低成本对照组，帮助判断问题是否来自PaddleOCR特有链路。",
            "fit_for": ["离线基线", "纯文本或较规则图片", "快速做可替代性下限测试"],
            "risks": [
                "复杂中文信息图和多栏布局效果可能不稳定。",
                "Windows安装与语言包配置可能增加环境成本。",
                "不应作为当前最优候选，只适合做对照。",
            ],
            "next_test": "只选一张弱样本 `img_9.jpg` 做最小对照，不先纳入主流程。",
        },
        {
            "backend_id": "cloud_ocr_service",
            "backend_name": "服务化 OCR（百度 / 腾讯 / 阿里等）",
            "deployment_type": "cloud_api",
            "integration_status": "candidate_after_local_eval",
            "source_url": "https://cloud.baidu.com/product/ocr",
            "known_from_project": False,
            "why_consider": "如果生产目标强依赖延迟、稳定性和服务可用性，云OCR可能比本地CPU更接近工程要求。",
            "fit_for": ["延迟和稳定性优先", "可接受外部API费用", "可处理数据合规与密钥管理"],
            "risks": [
                "需要API Key、网络、费用和数据合规评估。",
                "当前项目尚未接入，不得把云OCR能力写成已实现。",
                "不同服务的价格、QPS、识别能力需要单独查证和小样本实测。",
            ],
            "next_test": "如果用户授权，再选择一家服务做3张关键帧小样本live test，并记录成本、延迟和质量。",
        },
    ]


def build_ocr_backend_advice(
    *,
    gate_report: dict[str, Any],
    latency_profile: dict[str, Any] | None = None,
    candidate_catalog: list[dict[str, Any]] | None = None,
    candidate_evaluation_reports: list[dict[str, Any]] | None = None,
    monthly_api_budget_cny: float = 50.0,
    privacy_required: bool = False,
) -> dict[str, Any]:
    """基于已有证据生成 OCR 后端取舍建议。"""

    catalog = candidate_catalog or default_ocr_candidate_catalog()
    candidate_evaluations = _summarize_candidate_evaluations(candidate_evaluation_reports or [])
    catalog = _attach_candidate_evidence(catalog, candidate_evaluations)
    thresholds = gate_report.get("thresholds", {})
    overview = gate_report.get("batch_overview", {})
    gate_decision = gate_report.get("gate_decision", {})
    current_status = gate_decision.get("status", "unknown")
    quality_failed = _is_quality_failed(thresholds, overview)
    latency_failed = _is_latency_failed(thresholds, overview)
    p95_latency_ms = _to_number(overview.get("ocr_p95_latency_ms"))
    exact_recall = _to_number(overview.get("overall_exact_segment_recall"))
    character_error_rate = _to_number(overview.get("overall_character_error_rate"))

    latency_bottleneck = _extract_latency_bottleneck(latency_profile)
    switch_signal = _build_switch_signal(
        current_status=current_status,
        quality_failed=quality_failed,
        latency_failed=latency_failed,
    )
    ordered_candidates = _rank_candidates(
        catalog=catalog,
        switch_signal=switch_signal,
        quality_failed=quality_failed,
        latency_failed=latency_failed,
        privacy_required=privacy_required,
        monthly_api_budget_cny=monthly_api_budget_cny,
    )
    decision = _build_decision(
        switch_signal=switch_signal,
        ordered_candidates=ordered_candidates,
        quality_failed=quality_failed,
        latency_failed=latency_failed,
        latency_bottleneck=latency_bottleneck,
        candidate_evaluations=candidate_evaluations,
        privacy_required=privacy_required,
        monthly_api_budget_cny=monthly_api_budget_cny,
    )

    return {
        "schema_version": "v1",
        "report_name": "ocr_backend_advice",
        "current_backend": "paddleocr_local",
        "input_evidence": {
            "gate_report_name": gate_report.get("report_name"),
            "latency_profile_name": latency_profile.get("profile_name") if latency_profile else None,
        },
        "current_metrics": {
            "overall_exact_segment_recall": exact_recall,
            "overall_character_error_rate": character_error_rate,
            "ocr_p95_latency_ms": p95_latency_ms,
            "gate_status": current_status,
            "quality_failed": quality_failed,
            "latency_failed": latency_failed,
            "latency_bottleneck": latency_bottleneck,
        },
        "candidate_evaluations": candidate_evaluations,
        "constraints": {
            "min_exact_segment_recall": thresholds.get("min_exact_segment_recall"),
            "max_character_error_rate": thresholds.get("max_character_error_rate"),
            "max_latency_ms": thresholds.get("max_latency_ms"),
            "monthly_api_budget_cny": monthly_api_budget_cny,
            "privacy_required": privacy_required,
        },
        "decision": decision,
        "candidate_catalog": ordered_candidates,
        "boundary_notes": [
            "本报告只基于已有PaddleOCR评估证据、已有候选评估报告和公开候选方案信息生成取舍建议。",
            "已评估候选只代表当前样本上的本地对照结果，不等于已接入主流程。",
            "如果后续要验证服务化OCR，必须先确认API Key、费用、网络和数据合规风险。",
        ],
        "field_notes": {
            "backend_id": "OCR候选后端的唯一标识，用来区分当前后端和待评估后端。",
            "switch_signal": "是否需要从当前PaddleOCR转向替代方案评估的判断信号。",
            "evaluation_order": "下一步建议评估的OCR候选顺序，只表示测试优先级，不表示已接入。",
            "quality_failed": "质量是否未达当前闸门，用来判断是否需要寻找识别质量更稳的方案。",
            "latency_failed": "延迟是否未达当前闸门，用来判断是否需要寻找更快或服务化的方案。",
            "latency_bottleneck": "当前延迟瓶颈位置，用来判断优化方向是模型推理、解码、解析还是引擎创建。",
            "candidate_evaluations": "候选OCR后端的已评估结果摘要，用来避免重复推荐已经实测未通过的后端。",
        },
    }


def write_ocr_backend_advice_report(
    *,
    gate_report_path: str | Path,
    latency_profile_path: str | Path | None,
    output_json_path: str | Path,
    output_markdown_path: str | Path,
    candidate_evaluation_paths: list[str | Path] | None = None,
    monthly_api_budget_cny: float = 50.0,
    privacy_required: bool = False,
) -> tuple[Path, Path]:
    """读取已有报告并写出OCR后端判断报告。"""

    gate_report = json.loads(Path(gate_report_path).read_text(encoding="utf-8"))
    latency_profile = None
    if latency_profile_path:
        latency_profile = json.loads(Path(latency_profile_path).read_text(encoding="utf-8"))
    candidate_evaluation_reports = []
    for path in candidate_evaluation_paths or []:
        candidate_evaluation_reports.append(json.loads(Path(path).read_text(encoding="utf-8")))

    report = build_ocr_backend_advice(
        gate_report=gate_report,
        latency_profile=latency_profile,
        candidate_evaluation_reports=candidate_evaluation_reports,
        monthly_api_budget_cny=monthly_api_budget_cny,
        privacy_required=privacy_required,
    )
    json_path = Path(output_json_path)
    markdown_path = Path(output_markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _is_quality_failed(thresholds: dict[str, Any], overview: dict[str, Any]) -> bool | None:
    """判断质量闸门是否失败，缺字段时返回 None。"""

    recall = _to_number(overview.get("overall_exact_segment_recall"))
    error_rate = _to_number(overview.get("overall_character_error_rate"))
    min_recall = _to_number(thresholds.get("min_exact_segment_recall"))
    max_error_rate = _to_number(thresholds.get("max_character_error_rate"))
    if recall is None or error_rate is None or min_recall is None or max_error_rate is None:
        return None
    return recall < min_recall or error_rate > max_error_rate


def _is_latency_failed(thresholds: dict[str, Any], overview: dict[str, Any]) -> bool | None:
    """判断延迟闸门是否失败，缺字段时返回 None。"""

    latency = _to_number(overview.get("ocr_p95_latency_ms"))
    max_latency = _to_number(thresholds.get("max_latency_ms"))
    if latency is None or max_latency is None:
        return None
    return latency > max_latency


def _extract_latency_bottleneck(latency_profile: dict[str, Any] | None) -> str:
    """从延迟拆分报告中提取瓶颈阶段。"""

    if not latency_profile:
        return "current_data_not_provided"
    decision = latency_profile.get("decision", {})
    return str(decision.get("main_bottleneck") or "current_data_not_provided")


def _summarize_candidate_evaluations(
    candidate_evaluation_reports: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """提取候选OCR评估摘要，缺字段时不编造指标。"""

    summaries: dict[str, dict[str, Any]] = {}
    for report in candidate_evaluation_reports:
        backend_id = report.get("backend_id")
        if not isinstance(backend_id, str) or not backend_id:
            continue
        overview = report.get("overview", {})
        gate_decision = report.get("gate_decision", {})
        dependency = report.get("dependency", {})
        summaries[backend_id] = {
            "report_name": report.get("report_name"),
            "dependency_status": dependency.get("status"),
            "gate_status": gate_decision.get("status"),
            "overall_exact_segment_recall": _to_number(overview.get("overall_exact_segment_recall")),
            "overall_character_error_rate": _to_number(overview.get("overall_character_error_rate")),
            "ocr_p95_latency_ms": _to_number(overview.get("ocr_p95_latency_ms")),
            "ocr_external_api_cost_cny": _to_number(overview.get("ocr_external_api_cost_cny")),
            "next_action": gate_decision.get("next_action"),
        }
    return summaries


def _attach_candidate_evidence(
    catalog: list[dict[str, Any]],
    candidate_evaluations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """把已完成的候选评估结果附加到候选目录。"""

    updated_catalog = []
    for candidate in catalog:
        candidate = dict(candidate)
        evaluation = candidate_evaluations.get(candidate["backend_id"])
        if evaluation:
            candidate["known_from_project"] = True
            candidate["candidate_evaluation"] = evaluation
            if evaluation.get("gate_status") == "passed":
                candidate["integration_status"] = "evaluated_passed_not_integrated"
                candidate["next_test"] = "候选后端已通过当前闸门；如要接入主流程，需单独做集成改造和回归测试。"
            elif evaluation.get("gate_status") == "not_passed":
                candidate["integration_status"] = "evaluated_not_passed"
                candidate["next_test"] = "候选后端已实测未通过当前闸门，不应接入主流程。"
            elif evaluation.get("gate_status") == "dependency_missing":
                candidate["integration_status"] = "dependency_missing"
                candidate["next_test"] = "依赖仍缺失，不能得出质量或延迟结论。"
        updated_catalog.append(candidate)
    return updated_catalog


def _build_switch_signal(
    *,
    current_status: str,
    quality_failed: bool | None,
    latency_failed: bool | None,
) -> str:
    """生成是否需要评估替代OCR的判断信号。"""

    if current_status == "passed" and quality_failed is False and latency_failed is False:
        return "keep_current_backend"
    if quality_failed is None and latency_failed is None:
        return "need_more_evidence"
    if quality_failed or latency_failed or current_status == "not_passed":
        return "evaluate_alternative_backends"
    return "keep_current_backend"


def _rank_candidates(
    *,
    catalog: list[dict[str, Any]],
    switch_signal: str,
    quality_failed: bool | None,
    latency_failed: bool | None,
    privacy_required: bool,
    monthly_api_budget_cny: float,
) -> list[dict[str, Any]]:
    """根据当前约束给OCR候选排序。"""

    scored = []
    for candidate in catalog:
        score = _score_candidate(
            candidate=candidate,
            switch_signal=switch_signal,
            quality_failed=quality_failed,
            latency_failed=latency_failed,
            privacy_required=privacy_required,
            monthly_api_budget_cny=monthly_api_budget_cny,
        )
        scored.append({**candidate, "priority_score": score})
    return sorted(scored, key=lambda item: (-int(item["priority_score"]), item["backend_id"]))


def _score_candidate(
    *,
    candidate: dict[str, Any],
    switch_signal: str,
    quality_failed: bool | None,
    latency_failed: bool | None,
    privacy_required: bool,
    monthly_api_budget_cny: float,
) -> int:
    """给单个OCR候选生成简单优先级分数。"""

    backend_id = candidate["backend_id"]
    candidate_evaluation = candidate.get("candidate_evaluation", {})
    if candidate_evaluation.get("gate_status") == "not_passed":
        return 15
    if candidate_evaluation.get("gate_status") == "passed":
        return 95

    if switch_signal == "keep_current_backend":
        return 100 if backend_id == "paddleocr_local" else 20
    if switch_signal == "need_more_evidence":
        return 70 if backend_id == "paddleocr_local" else 40

    score = 0
    if backend_id == "rapidocr_onnxruntime_local":
        score += 75
        if latency_failed:
            score += 20
        if privacy_required or monthly_api_budget_cny <= 0:
            score += 10
    elif backend_id == "cloud_ocr_service":
        score += 60
        if latency_failed:
            score += 20
        if quality_failed:
            score += 10
        if privacy_required:
            score -= 40
        if monthly_api_budget_cny <= 0:
            score -= 50
    elif backend_id == "tesseract_local":
        score += 35
        if privacy_required or monthly_api_budget_cny <= 0:
            score += 10
    elif backend_id == "paddleocr_local":
        score += 25
    return max(score, 0)


def _build_decision(
    *,
    switch_signal: str,
    ordered_candidates: list[dict[str, Any]],
    quality_failed: bool | None,
    latency_failed: bool | None,
    latency_bottleneck: str,
    candidate_evaluations: dict[str, dict[str, Any]],
    privacy_required: bool,
    monthly_api_budget_cny: float,
) -> dict[str, Any]:
    """生成面向技术负责人的判断结论。"""

    recommended = ordered_candidates[0] if ordered_candidates else None
    reasons: list[str] = []
    if quality_failed:
        reasons.append("当前PaddleOCR关键帧批次质量闸门未通过，需要评估替代方案。")
    if latency_failed:
        reasons.append("当前PaddleOCR关键帧批次延迟闸门未通过，需要评估更快的本地或服务化方案。")
    if latency_bottleneck in {"first_predict", "avg_warm_predict", "avg_warm_attempt_total"}:
        reasons.append("延迟拆分显示主要瓶颈在模型推理，单纯优化文件读取或结果写入意义不大。")
    rapidocr_evaluation = candidate_evaluations.get("rapidocr_onnxruntime_local")
    if rapidocr_evaluation and rapidocr_evaluation.get("gate_status") == "not_passed":
        reasons.append("RapidOCR候选后端已完成同批样本实测，但质量和延迟仍未通过当前闸门，不应接入主流程。")
    if privacy_required:
        reasons.append("当前设置要求数据不出本机，因此服务化OCR只能作为后置候选。")
    if monthly_api_budget_cny <= 0:
        reasons.append("当前API预算为0，因此云OCR只能作为后续授权后的候选。")
    if not reasons:
        reasons.append("当前数据不足或闸门未显示明确失败，应先补齐评估证据。")

    evaluation_order = [candidate["backend_id"] for candidate in ordered_candidates]
    next_action = "继续使用当前PaddleOCR。"
    if switch_signal == "evaluate_alternative_backends" and recommended:
        if rapidocr_evaluation and rapidocr_evaluation.get("gate_status") == "not_passed":
            next_action = (
                "RapidOCR已实测未通过当前闸门；不要接入主流程。"
                f"如继续追求生产可用OCR，下一轮只能在用户授权后小样本评估 `{recommended['backend_id']}`；"
                "如暂不授权外部API，则保留PaddleOCR作为当前本地基线。"
            )
        else:
            next_action = f"下一轮只评估 `{recommended['backend_id']}`，使用同一批图片和同一份人工基准做对照。"
    elif switch_signal == "need_more_evidence":
        next_action = "先补齐缺失的OCR质量或延迟字段，再判断是否切换后端。"

    return {
        "switch_signal": switch_signal,
        "recommended_next_backend_id": recommended["backend_id"] if recommended else None,
        "evaluation_order": evaluation_order,
        "reasons": reasons,
        "next_action": next_action,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    """生成OCR后端判断Markdown报告。"""

    metrics = report["current_metrics"]
    decision = report["decision"]
    constraints = report["constraints"]
    candidate_rows = "\n".join(
        "| {backend_id} | {deployment_type} | {integration_status} | {priority_score} | {next_test} |".format(
            **candidate
        )
        for candidate in report["candidate_catalog"]
    )
    evaluation_rows = _render_candidate_evaluation_rows(report.get("candidate_evaluations", {}))
    reasons = "\n".join(f"- {reason}" for reason in decision["reasons"])
    sources = "\n".join(
        f"- `{candidate['backend_id']}`：{candidate['source_url']}"
        for candidate in report["candidate_catalog"]
    )

    return f"""# OCR 后端取舍判断报告

## 一、当前结论

- 当前后端：`{report['current_backend']}`
- 切换判断：`{decision['switch_signal']}`
- 下一优先评估后端：`{decision['recommended_next_backend_id']}`
- 下一步：{decision['next_action']}

## 二、当前PaddleOCR证据

| 指标 | 数值 | 作用 |
|---|---:|---|
| 完整段落召回率 | {_format_rate(metrics.get('overall_exact_segment_recall'))} | 判断OCR是否漏掉关键业务文字 |
| 字符错误率 | {_format_rate(metrics.get('overall_character_error_rate'))} | 判断OCR错字、漏字和多字程度 |
| OCR P95延迟 | {_format_ms(metrics.get('ocr_p95_latency_ms'))} | 判断批次慢调用是否超过目标 |
| 质量是否失败 | {metrics.get('quality_failed')} | 判断是否需要寻找识别更稳的方案 |
| 延迟是否失败 | {metrics.get('latency_failed')} | 判断是否需要寻找更快的方案 |
| 延迟瓶颈 | `{metrics.get('latency_bottleneck')}` | 判断优化方向 |

当前阈值：完整段落召回率 ≥ {_format_rate(constraints.get('min_exact_segment_recall'))}，字符错误率 ≤ {_format_rate(constraints.get('max_character_error_rate'))}，OCR延迟 ≤ {_format_ms(constraints.get('max_latency_ms'))}。

## 三、判断理由

{reasons}

## 四、候选后端排序

| 后端ID | 部署类型 | 当前状态 | 优先级分数 | 下一步测试 |
|---|---|---|---:|---|
{candidate_rows}

## 五、已评估候选后端

| 后端ID | 依赖状态 | 闸门状态 | 完整段落召回率 | 字符错误率 | OCR P95延迟 | 外部API成本 |
|---|---|---|---:|---:|---:|---:|
{evaluation_rows}

## 六、信息来源

{sources}

## 七、边界说明

- 本报告只基于已有PaddleOCR评估证据、已有候选评估报告和公开候选方案信息生成取舍建议。
- 已评估候选只代表当前样本上的本地对照结果，不等于已接入主流程。
- 如果后续要验证服务化OCR，必须先确认API Key、费用、网络和数据合规风险。

## 八、字段说明

| 字段 | 含义与作用 |
|---|---|
| `backend_id` | OCR候选后端的唯一标识，用来区分当前后端和待评估后端 |
| `switch_signal` | 是否需要从当前PaddleOCR转向替代方案评估的判断信号 |
| `evaluation_order` | 下一步建议评估的OCR候选顺序，只表示测试优先级，不表示已接入 |
| `quality_failed` | 质量是否未达当前闸门，用来判断是否需要寻找识别质量更稳的方案 |
| `latency_failed` | 延迟是否未达当前闸门，用来判断是否需要寻找更快或服务化的方案 |
| `latency_bottleneck` | 当前延迟瓶颈位置，用来判断优化方向是模型推理、解码、解析还是引擎创建 |
| `candidate_evaluations` | 候选OCR后端的已评估结果摘要，用来避免重复推荐已经实测未通过的后端 |
"""


def _render_candidate_evaluation_rows(candidate_evaluations: dict[str, dict[str, Any]]) -> str:
    """把候选评估摘要渲染为 Markdown 表格行。"""

    if not candidate_evaluations:
        return "| 当前数据未提供 | 当前数据未提供 | 当前数据未提供 | 当前数据未提供 | 当前数据未提供 | 当前数据未提供 | 当前数据未提供 |"
    rows = []
    for backend_id, evaluation in candidate_evaluations.items():
        rows.append(
            "| {backend_id} | {dependency_status} | {gate_status} | {recall} | {error_rate} | {latency} | {cost} |".format(
                backend_id=backend_id,
                dependency_status=evaluation.get("dependency_status") or "当前数据未提供",
                gate_status=evaluation.get("gate_status") or "当前数据未提供",
                recall=_format_rate(evaluation.get("overall_exact_segment_recall")),
                error_rate=_format_rate(evaluation.get("overall_character_error_rate")),
                latency=_format_ms(evaluation.get("ocr_p95_latency_ms")),
                cost=_format_cny(evaluation.get("ocr_external_api_cost_cny")),
            )
        )
    return "\n".join(rows)


def _format_rate(value: Any) -> str:
    """把比例格式化为百分比。"""

    return f"{value:.2%}" if isinstance(value, (int, float)) else "当前数据未提供"


def _format_ms(value: Any) -> str:
    """把毫秒格式化为文本。"""

    return f"{int(value)}ms" if isinstance(value, (int, float)) else "当前数据未提供"


def _format_cny(value: Any) -> str:
    """把人民币金额格式化为文本。"""

    return f"{value:.4f}元" if isinstance(value, (int, float)) else "当前数据未提供"


def _to_number(value: Any) -> float | None:
    """把数字字段转换为 float，失败时返回 None。"""

    if isinstance(value, (int, float)):
        return float(value)
    return None


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 4:
        print(
            "用法: python .\\src\\ocr_backend_advisor.py "
            "image_ocr_gate_report_keyframes.json "
            "image_ocr_latency_profile_img_9.json "
            "ocr_backend_advice.json "
            "ocr_backend_advice.md "
            "[candidate_eval.json ...]"
        )
        return 2

    json_path, markdown_path = write_ocr_backend_advice_report(
        gate_report_path=args[0],
        latency_profile_path=args[1],
        output_json_path=args[2],
        output_markdown_path=args[3],
        candidate_evaluation_paths=args[4:],
    )
    print(
        json.dumps(
            {"json_report": str(json_path), "markdown_report": str(markdown_path)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
