"""图片 OCR 业务文字评估工具。

本模块只评估人工确认的业务文字块，不把按钮、点赞数、时长等界面文字
计入质量指标，也不要求复杂页面中的文字按固定阅读顺序输出。
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from report_generator import read_jsonl


GOLD_FIELDS = {
    "file_name",
    "segment_id",
    "segment_type",
    "gold_text",
    "is_required",
    "evaluation_scope",
    "reviewer_note",
}

ERROR_BUCKET_DESCRIPTIONS = {
    "missing_or_unmatched_text": "没有找到可用OCR片段，通常表示该文字块被漏识别或匹配窗口过弱。",
    "value_retained_label_lost": "数值被识别到，但标签或模块名称缺失，常见于小字号指标说明。",
    "label_retained_value_lost": "标签或模块名称被识别到，但容量、页数、Entry等数值信息缺失。",
    "partial_text_match": "只匹配到较短片段，说明文字块被截断或被复杂布局拆散。",
    "character_substitution_or_layout_noise": "有接近片段但存在字符替换、符号误读或布局噪声。",
}


def normalize_ocr_text(value: str) -> str:
    """统一字符宽度并移除空白，保留标点和业务字符。"""

    normalized = unicodedata.normalize("NFKC", value or "")
    return "".join(character for character in normalized if not character.isspace())


def read_ocr_gold_rows(gold_path: str | Path) -> list[dict[str, str]]:
    """读取并校验图片 OCR 人工正确文本。"""

    path = Path(gold_path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        field_names = set(reader.fieldnames or [])
        if not GOLD_FIELDS.issubset(field_names):
            missing_fields = sorted(GOLD_FIELDS - field_names)
            raise ValueError(f"OCR人工基准缺少字段：{', '.join(missing_fields)}")
        rows = [dict(row) for row in reader]

    seen_keys: set[tuple[str, str]] = set()
    for row in rows:
        file_name = str(row.get("file_name") or "").strip()
        segment_id = str(row.get("segment_id") or "").strip()
        gold_text = str(row.get("gold_text") or "").strip()
        if not file_name or not segment_id or not gold_text:
            raise ValueError("OCR人工基准中的 file_name、segment_id 和 gold_text 不允许为空。")

        key = (file_name, segment_id)
        if key in seen_keys:
            raise ValueError(f"OCR人工基准存在重复文字块：{file_name} / {segment_id}")
        seen_keys.add(key)

        required_value = str(row.get("is_required") or "").strip().lower()
        if required_value not in {"true", "false"}:
            raise ValueError(f"is_required 只能填写 true 或 false：{file_name} / {segment_id}")

    return rows


def find_image_ocr_text(results_path: str | Path, file_name: str) -> str:
    """从文件级结果中找到指定图片的 OCR 文字。"""

    matched_records = [
        record
        for record in read_jsonl(results_path)
        if record.get("file_name") == file_name and record.get("media_type") == "image"
    ]
    if not matched_records:
        raise ValueError(f"结果文件中没有找到图片：{file_name}")
    if len(matched_records) > 1:
        raise ValueError(f"结果文件中存在多个同名图片结果：{file_name}")

    ocr_text = matched_records[0].get("ocr_text")
    if not isinstance(ocr_text, str) or not ocr_text.strip():
        raise ValueError(f"图片没有可评估的OCR文字：{file_name}")
    return ocr_text


def levenshtein_distance(source: str, target: str) -> int:
    """计算两个字符串之间的最小编辑距离。"""

    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for source_index, source_character in enumerate(source, start=1):
        current_row = [source_index]
        for target_index, target_character in enumerate(target, start=1):
            insertion_cost = current_row[target_index - 1] + 1
            deletion_cost = previous_row[target_index] + 1
            substitution_cost = previous_row[target_index - 1] + (
                0 if source_character == target_character else 1
            )
            current_row.append(min(insertion_cost, deletion_cost, substitution_cost))
        previous_row = current_row
    return previous_row[-1]


def evaluate_image_ocr(
    *,
    file_name: str,
    ocr_text: str,
    gold_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """按独立业务文字块计算精确召回率和字符错误率。"""

    selected_rows = [row for row in gold_rows if row.get("file_name") == file_name]
    if not selected_rows:
        raise ValueError(f"人工基准中没有找到图片：{file_name}")

    required_rows = [
        row
        for row in selected_rows
        if str(row.get("is_required") or "").strip().lower() == "true"
    ]
    if not required_rows:
        raise ValueError(f"图片没有必须评估的文字块：{file_name}")

    normalized_lines = [
        normalize_ocr_text(line)
        for line in ocr_text.splitlines()
        if normalize_ocr_text(line)
    ]
    details_by_segment: dict[str, dict[str, Any]] = {}
    occupied_spans: dict[int, list[tuple[int, int]]] = {
        line_index: []
        for line_index in range(len(normalized_lines))
    }

    rows_by_length = sorted(
        required_rows,
        key=lambda row: len(normalize_ocr_text(row["gold_text"])),
        reverse=True,
    )
    unmatched_rows: list[dict[str, str]] = []

    for row in rows_by_length:
        gold_text = normalize_ocr_text(row["gold_text"])
        exact_candidates: list[tuple[int, int, int, int]] = []
        for line_index, ocr_line in enumerate(normalized_lines):
            search_start = 0
            while True:
                match_start = ocr_line.find(gold_text, search_start)
                if match_start < 0:
                    break
                match_end = match_start + len(gold_text)
                if _span_is_available(
                    occupied_spans[line_index],
                    match_start,
                    match_end,
                ):
                    exact_candidates.append(
                        (
                            len(ocr_line) - len(gold_text),
                            line_index,
                            match_start,
                            match_end,
                        )
                    )
                search_start = match_start + 1

        if not exact_candidates:
            unmatched_rows.append(row)
            continue

        _, matched_index, match_start, match_end = min(exact_candidates)
        occupied_spans[matched_index].append((match_start, match_end))
        details_by_segment[row["segment_id"]] = _build_detail(
            row=row,
            matched_ocr_text=normalized_lines[matched_index],
            comparison_text=gold_text,
            edit_distance=0,
            is_exact_match=True,
        )

    for row in unmatched_rows:
        gold_text = normalize_ocr_text(row["gold_text"])
        best_match: tuple[int, int, int, int, str] | None = None
        for line_index, ocr_line in enumerate(normalized_lines):
            for fragment_start, fragment_text in _available_fragments(
                ocr_line,
                occupied_spans[line_index],
            ):
                comparison_text, distance = _best_comparison_window(
                    gold_text,
                    fragment_text,
                )
                local_start = fragment_text.find(comparison_text)
                match_start = fragment_start + max(local_start, 0)
                match_end = match_start + len(comparison_text)
                candidate = (
                    distance,
                    line_index,
                    match_start,
                    match_end,
                    comparison_text,
                )
                if best_match is None or candidate < best_match:
                    best_match = candidate

        if best_match is None:
            details_by_segment[row["segment_id"]] = _build_detail(
                row=row,
                matched_ocr_text="",
                comparison_text="",
                edit_distance=len(gold_text),
                is_exact_match=False,
            )
            continue

        edit_distance, matched_index, match_start, match_end, comparison_text = best_match
        if match_end > match_start:
            occupied_spans[matched_index].append((match_start, match_end))
        details_by_segment[row["segment_id"]] = _build_detail(
            row=row,
            matched_ocr_text=normalized_lines[matched_index],
            comparison_text=comparison_text,
            edit_distance=edit_distance,
            is_exact_match=False,
        )

    ordered_details = [details_by_segment[row["segment_id"]] for row in required_rows]
    exact_segment_count = sum(1 for detail in ordered_details if detail["is_exact_match"])
    total_gold_characters = sum(detail["gold_character_count"] for detail in ordered_details)
    total_edit_distance = sum(detail["edit_distance"] for detail in ordered_details)

    evaluation_scopes = sorted(
        {
            str(row.get("evaluation_scope") or "").strip()
            for row in required_rows
            if row.get("evaluation_scope")
        }
    )
    return {
        "schema_version": "v1",
        "metric_name": "image_ocr_business_text",
        "file_name": file_name,
        "evaluation_scopes": evaluation_scopes,
        "required_segment_count": len(required_rows),
        "exact_segment_count": exact_segment_count,
        "exact_segment_recall": round(exact_segment_count / len(required_rows), 6),
        "total_gold_characters": total_gold_characters,
        "total_edit_distance": total_edit_distance,
        "character_error_rate": round(total_edit_distance / total_gold_characters, 6),
        "details": ordered_details,
        "field_notes": {
            "exact_segment_recall": "完整识别的必选业务文字块占比；同一OCR行可以匹配多个互不重叠的文字块，重复文字必须对应不同字符范围。",
            "character_error_rate": "各业务文字块与最佳OCR片段的编辑距离之和除以人工正确字符总数；不统计界面噪声。",
            "matched_ocr_text": "与人工文字块配对的完整OCR行，用于人工复核匹配来源。",
            "comparison_text": "计算字符编辑距离时使用的OCR片段，避免把相邻且已排除的话题标签计入错误。",
            "edit_distance": "把OCR片段修改成人工正确文字所需的最少插入、删除和替换次数。",
        },
    }


def write_image_ocr_report(
    *,
    results_path: str | Path,
    gold_path: str | Path,
    file_name: str,
    output_path: str | Path,
) -> Path:
    """读取现有结果与人工基准并写出JSON评估报告。"""

    report = evaluate_image_ocr(
        file_name=file_name,
        ocr_text=find_image_ocr_text(results_path, file_name),
        gold_rows=read_ocr_gold_rows(gold_path),
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def analyze_image_ocr_errors(
    evaluation_report: dict[str, Any],
    *,
    batch_summary: dict[str, Any] | None = None,
    min_exact_segment_recall: float = 0.9,
    max_character_error_rate: float = 0.05,
    max_latency_ms: int = 2000,
) -> dict[str, Any]:
    """基于已有OCR评估报告生成错误归因和闸门判断。"""

    details = evaluation_report.get("details")
    if not isinstance(details, list):
        raise ValueError("OCR评估报告缺少 details 数组。")

    file_name = str(evaluation_report.get("file_name") or "")
    if not file_name:
        raise ValueError("OCR评估报告缺少 file_name。")

    latency_ms = _find_ocr_latency_ms(batch_summary, file_name)
    details_with_bucket = [
        {
            **detail,
            "error_bucket": "exact_match"
            if detail.get("is_exact_match")
            else _classify_error_bucket(detail),
        }
        for detail in details
    ]
    error_details = [
        detail
        for detail in details_with_bucket
        if detail.get("error_bucket") != "exact_match"
    ]

    overview = {
        "required_segment_count": evaluation_report.get("required_segment_count"),
        "exact_segment_count": evaluation_report.get("exact_segment_count"),
        "error_segment_count": len(error_details),
        "exact_segment_recall": evaluation_report.get("exact_segment_recall"),
        "character_error_rate": evaluation_report.get("character_error_rate"),
        "ocr_latency_ms": latency_ms,
    }

    return {
        "schema_version": "v1",
        "analysis_name": "image_ocr_error_analysis",
        "file_name": file_name,
        "batch_id": batch_summary.get("batch_id") if isinstance(batch_summary, dict) else None,
        "overview": overview,
        "thresholds": {
            "min_exact_segment_recall": min_exact_segment_recall,
            "max_character_error_rate": max_character_error_rate,
            "max_latency_ms": max_latency_ms,
        },
        "gate_decision": _build_ocr_gate_decision(
            overview,
            min_exact_segment_recall=min_exact_segment_recall,
            max_character_error_rate=max_character_error_rate,
            max_latency_ms=max_latency_ms,
        ),
        "error_by_segment_type": _summarize_errors_by_segment_type(details_with_bucket),
        "error_buckets": _summarize_error_buckets(error_details),
        "top_error_segments": _top_error_segments(error_details),
        "interpretation": _build_error_interpretation(file_name, details_with_bucket, latency_ms),
        "boundary_notes": [
            "本报告只解释已有OCR评估结果，不重新运行PaddleOCR，也不调用DeepSeek。",
            "错误归因来自分段人工基准与OCR文本的比较，是工程诊断，不是模型泛化质量结论。",
            "视觉理解、语音识别和视频真实处理仍未接入，不能把本报告解释为完整多模态质量评估。",
        ],
        "field_notes": {
            "error_bucket": "对未完整命中文字块的启发式错误归因，用于定位问题类型。",
            "error_by_segment_type": "按人工文字块类型聚合错误，用于判断哪类内容最容易失败。",
            "gate_decision": "基于当前MVP阈值给出的是否继续留在OCR功能内的判断。",
        },
    }


def write_image_ocr_error_analysis(
    *,
    evaluation_report_path: str | Path,
    batch_summary_path: str | Path | None,
    output_json_path: str | Path,
    output_markdown_path: str | Path,
) -> tuple[Path, Path]:
    """读取已有评估报告并写出JSON与Markdown错误归因报告。"""

    evaluation_report = json.loads(Path(evaluation_report_path).read_text(encoding="utf-8"))
    batch_summary = (
        json.loads(Path(batch_summary_path).read_text(encoding="utf-8"))
        if batch_summary_path
        else None
    )
    analysis = analyze_image_ocr_errors(evaluation_report, batch_summary=batch_summary)

    json_path = Path(output_json_path)
    markdown_path = Path(output_markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_error_analysis_markdown(analysis), encoding="utf-8")
    return json_path, markdown_path


def build_image_ocr_gate_report(
    summary_report: dict[str, Any],
    *,
    error_analysis_report: dict[str, Any] | None = None,
    min_exact_segment_recall: float = 0.9,
    max_character_error_rate: float = 0.05,
    max_latency_ms: int = 2000,
) -> dict[str, Any]:
    """基于已有OCR汇总和错误归因生成批次级功能闸门报告。"""

    file_metrics = summary_report.get("file_metrics")
    if not isinstance(file_metrics, list) or not file_metrics:
        raise ValueError("OCR汇总报告缺少 file_metrics 数组。")

    file_checks = [
        _build_file_gate_check(
            file_metric,
            min_exact_segment_recall=min_exact_segment_recall,
            max_character_error_rate=max_character_error_rate,
            max_latency_ms=max_latency_ms,
        )
        for file_metric in file_metrics
        if isinstance(file_metric, dict)
    ]
    if not file_checks:
        raise ValueError("OCR汇总报告中没有可判断的图片指标。")

    quality_blocking_files = [
        check["file_name"]
        for check in file_checks
        if check["quality_status"] == "not_passed"
    ]
    latency_blocking_files = [
        check["file_name"]
        for check in file_checks
        if check["latency_status"] == "not_passed"
    ]
    blocking_files = sorted(set(quality_blocking_files + latency_blocking_files))
    batch_reasons = _build_batch_gate_reasons(
        summary_report,
        min_exact_segment_recall=min_exact_segment_recall,
        max_character_error_rate=max_character_error_rate,
        max_latency_ms=max_latency_ms,
        quality_blocking_files=quality_blocking_files,
        latency_blocking_files=latency_blocking_files,
    )

    gate_status = "passed" if not batch_reasons else "not_passed"
    weak_sample = _extract_weak_sample_summary(error_analysis_report)
    recommendations = _build_batch_gate_recommendations(
        gate_status=gate_status,
        blocking_files=blocking_files,
        latency_blocking_files=latency_blocking_files,
        weak_sample=weak_sample,
    )

    return {
        "schema_version": "v1",
        "report_name": "image_ocr_batch_gate_report",
        "batch_id": summary_report.get("batch_id"),
        "thresholds": {
            "min_exact_segment_recall": min_exact_segment_recall,
            "max_character_error_rate": max_character_error_rate,
            "max_latency_ms": max_latency_ms,
        },
        "batch_overview": {
            "evaluated_files": summary_report.get("evaluated_files", len(file_checks)),
            "total_required_segment_count": summary_report.get("total_required_segment_count"),
            "overall_exact_segment_recall": summary_report.get("overall_exact_segment_recall"),
            "overall_character_error_rate": summary_report.get("overall_character_error_rate"),
            "ocr_avg_latency_ms": summary_report.get("ocr_avg_latency_ms"),
            "ocr_p95_latency_ms": summary_report.get("ocr_p95_latency_ms"),
            "ocr_cost_cny": summary_report.get("ocr_cost_cny"),
        },
        "gate_decision": {
            "status": gate_status,
            "reasons": batch_reasons,
            "blocking_files": blocking_files,
            "next_action": "可以考虑进入下一个功能。"
            if gate_status == "passed"
            else "继续留在图片OCR功能内，优先做弱样本和延迟瓶颈的最小实验。",
        },
        "file_checks": file_checks,
        "weak_sample_analysis": weak_sample,
        "recommendations": recommendations,
        "boundary_notes": [
            "本报告只读取已有OCR评估和错误归因结果，不重新运行PaddleOCR，也不调用DeepSeek。",
            "本报告判断的是当前关键帧OCR评估批次是否通过MVP闸门，不代表OCR模型线上泛化质量。",
            "视觉理解、语音识别和视频真实处理仍未接入，不能把本报告解释为完整多模态质量评估。",
        ],
        "field_notes": {
            "gate_decision": "批次级闸门判断，用来决定是否可以离开图片OCR功能进入下一能力。",
            "blocking_files": "导致批次闸门未通过的图片文件名列表，用来定位需要继续打磨的样本。",
            "exact_segment_recall": "完整识别的必选业务文字块占比，用于判断OCR是否漏掉关键文字。",
            "character_error_rate": "分段编辑距离除以人工正确字符总数，用于判断文字错漏程度。",
            "ocr_p95_latency_ms": "OCR调用耗时的95分位，用于判断延迟目标是否被慢样本拉高。",
        },
    }


def write_image_ocr_gate_report(
    *,
    summary_report_path: str | Path,
    error_analysis_path: str | Path | None,
    output_json_path: str | Path,
    output_markdown_path: str | Path,
) -> tuple[Path, Path]:
    """读取已有OCR报告并写出批次级JSON与Markdown闸门报告。"""

    summary_report = json.loads(Path(summary_report_path).read_text(encoding="utf-8"))
    error_analysis_report = (
        json.loads(Path(error_analysis_path).read_text(encoding="utf-8"))
        if error_analysis_path
        else None
    )
    report = build_image_ocr_gate_report(
        summary_report,
        error_analysis_report=error_analysis_report,
    )

    json_path = Path(output_json_path)
    markdown_path = Path(output_markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_gate_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _span_is_available(
    occupied_spans: list[tuple[int, int]],
    candidate_start: int,
    candidate_end: int,
) -> bool:
    """判断候选字符范围是否与已经匹配的范围重叠。"""

    return all(
        candidate_end <= occupied_start or candidate_start >= occupied_end
        for occupied_start, occupied_end in occupied_spans
    )


def _available_fragments(
    ocr_line: str,
    occupied_spans: list[tuple[int, int]],
) -> list[tuple[int, str]]:
    """返回一条OCR文字中尚未被其他基准文字占用的连续片段。"""

    fragments: list[tuple[int, str]] = []
    cursor = 0
    for occupied_start, occupied_end in sorted(occupied_spans):
        if cursor < occupied_start:
            fragments.append((cursor, ocr_line[cursor:occupied_start]))
        cursor = max(cursor, occupied_end)
    if cursor < len(ocr_line):
        fragments.append((cursor, ocr_line[cursor:]))
    return [(start, text) for start, text in fragments if text]


def _best_comparison_window(gold_text: str, ocr_line: str) -> tuple[str, int]:
    """在一条OCR文字中寻找与人工文字块最接近的连续片段。"""

    if not ocr_line:
        return "", len(gold_text)
    if gold_text in ocr_line:
        return gold_text, 0

    length_delta = max(2, round(len(gold_text) * 0.25))
    minimum_length = max(1, len(gold_text) - length_delta)
    maximum_length = min(len(ocr_line), len(gold_text) + length_delta)
    if minimum_length > maximum_length:
        return ocr_line, levenshtein_distance(gold_text, ocr_line)

    best_text = ocr_line
    best_distance = levenshtein_distance(gold_text, ocr_line)
    for window_length in range(minimum_length, maximum_length + 1):
        for start_index in range(0, len(ocr_line) - window_length + 1):
            candidate_text = ocr_line[start_index : start_index + window_length]
            distance = levenshtein_distance(gold_text, candidate_text)
            if (distance, len(candidate_text), candidate_text) < (
                best_distance,
                len(best_text),
                best_text,
            ):
                best_text = candidate_text
                best_distance = distance
    return best_text, best_distance


def _build_detail(
    *,
    row: dict[str, str],
    matched_ocr_text: str,
    comparison_text: str,
    edit_distance: int,
    is_exact_match: bool,
) -> dict[str, Any]:
    """生成一个业务文字块的评估明细。"""

    gold_text = normalize_ocr_text(row["gold_text"])
    return {
        "segment_id": row["segment_id"],
        "segment_type": row["segment_type"],
        "gold_text": row["gold_text"],
        "matched_ocr_text": matched_ocr_text,
        "comparison_text": comparison_text,
        "is_exact_match": is_exact_match,
        "gold_character_count": len(gold_text),
        "edit_distance": edit_distance,
        "character_error_rate": round(edit_distance / len(gold_text), 6),
        "reviewer_note": row.get("reviewer_note") or "",
    }


def _classify_error_bucket(detail: dict[str, Any]) -> str:
    """给单个未完整命中的文字块做启发式错误分类。"""

    gold_text = normalize_ocr_text(str(detail.get("gold_text") or ""))
    comparison_text = normalize_ocr_text(str(detail.get("comparison_text") or ""))
    if not comparison_text:
        return "missing_or_unmatched_text"

    gold_digit_sequences = set(re.findall(r"\d+", gold_text))
    comparison_digit_sequences = set(re.findall(r"\d+", comparison_text))
    has_shared_digits = bool(gold_digit_sequences & comparison_digit_sequences)

    if gold_digit_sequences and has_shared_digits and len(comparison_text) < len(gold_text) * 0.75:
        return "value_retained_label_lost"
    if gold_digit_sequences and not has_shared_digits:
        return "label_retained_value_lost"
    if len(comparison_text) < len(gold_text) * 0.6:
        return "partial_text_match"
    return "character_substitution_or_layout_noise"


def _find_ocr_latency_ms(batch_summary: dict[str, Any] | None, file_name: str) -> int | None:
    """从OCR汇总报告中读取指定图片的OCR延迟。"""

    if not isinstance(batch_summary, dict):
        return None
    for file_metric in batch_summary.get("file_metrics", []):
        if isinstance(file_metric, dict) and file_metric.get("file_name") == file_name:
            latency = file_metric.get("ocr_latency_ms")
            return latency if isinstance(latency, int) else None
    return None


def _build_ocr_gate_decision(
    overview: dict[str, Any],
    *,
    min_exact_segment_recall: float,
    max_character_error_rate: float,
    max_latency_ms: int,
) -> dict[str, Any]:
    """根据MVP阈值生成OCR功能闸门判断。"""

    reasons: list[str] = []
    exact_segment_recall = overview.get("exact_segment_recall")
    character_error_rate = overview.get("character_error_rate")
    latency_ms = overview.get("ocr_latency_ms")

    if not isinstance(exact_segment_recall, (int, float)):
        reasons.append("缺少完整段落召回率，无法判断OCR质量。")
    elif exact_segment_recall < min_exact_segment_recall:
        reasons.append(
            f"完整段落召回率为{exact_segment_recall:.2%}，低于当前MVP观察阈值{min_exact_segment_recall:.0%}。"
        )

    if not isinstance(character_error_rate, (int, float)):
        reasons.append("缺少字符错误率，无法判断文字准确性。")
    elif character_error_rate > max_character_error_rate:
        reasons.append(
            f"字符错误率为{character_error_rate:.2%}，高于当前MVP观察阈值{max_character_error_rate:.0%}。"
        )

    if not isinstance(latency_ms, int):
        reasons.append("缺少OCR延迟，无法判断性能目标。")
    elif latency_ms > max_latency_ms:
        reasons.append(f"OCR延迟为{latency_ms}ms，高于图片任务{max_latency_ms}ms目标。")

    return {
        "status": "passed" if not reasons else "not_passed",
        "reasons": reasons,
        "next_action": "可以考虑进入下一个功能。"
        if not reasons
        else "继续留在图片OCR功能内，先解释弱样本和延迟瓶颈。",
    }


def _build_file_gate_check(
    file_metric: dict[str, Any],
    *,
    min_exact_segment_recall: float,
    max_character_error_rate: float,
    max_latency_ms: int,
) -> dict[str, Any]:
    """把单张图片指标转换成闸门判断。"""

    file_name = str(file_metric.get("file_name") or "unknown")
    overview = {
        "exact_segment_recall": file_metric.get("exact_segment_recall"),
        "character_error_rate": file_metric.get("character_error_rate"),
        "ocr_latency_ms": file_metric.get("ocr_latency_ms"),
    }
    decision = _build_ocr_gate_decision(
        overview,
        min_exact_segment_recall=min_exact_segment_recall,
        max_character_error_rate=max_character_error_rate,
        max_latency_ms=max_latency_ms,
    )
    quality_reasons = [
        reason
        for reason in decision["reasons"]
        if "召回率" in reason or "字符错误率" in reason
    ]
    latency_reasons = [
        reason
        for reason in decision["reasons"]
        if "延迟" in reason
    ]

    return {
        "file_name": file_name,
        "required_segment_count": file_metric.get("required_segment_count"),
        "exact_segment_count": file_metric.get("exact_segment_count"),
        "exact_segment_recall": file_metric.get("exact_segment_recall"),
        "character_error_rate": file_metric.get("character_error_rate"),
        "ocr_latency_ms": file_metric.get("ocr_latency_ms"),
        "status": decision["status"],
        "quality_status": "passed" if not quality_reasons else "not_passed",
        "latency_status": "passed" if not latency_reasons else "not_passed",
        "blocking_reasons": decision["reasons"],
    }


def _build_batch_gate_reasons(
    summary_report: dict[str, Any],
    *,
    min_exact_segment_recall: float,
    max_character_error_rate: float,
    max_latency_ms: int,
    quality_blocking_files: list[str],
    latency_blocking_files: list[str],
) -> list[str]:
    """生成批次级闸门不通过原因。"""

    reasons: list[str] = []
    overall_recall = summary_report.get("overall_exact_segment_recall")
    overall_error_rate = summary_report.get("overall_character_error_rate")
    p95_latency = summary_report.get("ocr_p95_latency_ms")

    if isinstance(overall_recall, (int, float)) and overall_recall < min_exact_segment_recall:
        reasons.append(
            f"批次整体完整段落召回率为{overall_recall:.2%}，低于当前MVP观察阈值{min_exact_segment_recall:.0%}。"
        )
    elif not isinstance(overall_recall, (int, float)):
        reasons.append("缺少批次整体完整段落召回率，无法判断OCR质量。")

    if isinstance(overall_error_rate, (int, float)) and overall_error_rate > max_character_error_rate:
        reasons.append(
            f"批次整体字符错误率为{overall_error_rate:.2%}，高于当前MVP观察阈值{max_character_error_rate:.0%}。"
        )
    elif not isinstance(overall_error_rate, (int, float)):
        reasons.append("缺少批次整体字符错误率，无法判断文字准确性。")

    if isinstance(p95_latency, (int, float)) and p95_latency > max_latency_ms:
        reasons.append(f"批次OCR P95延迟为{int(p95_latency)}ms，高于图片任务{max_latency_ms}ms目标。")
    elif not isinstance(p95_latency, (int, float)):
        reasons.append("缺少批次OCR P95延迟，无法判断性能目标。")

    if quality_blocking_files:
        reasons.append(f"质量未通过图片：{', '.join(quality_blocking_files)}。")
    if latency_blocking_files:
        reasons.append(f"延迟未通过图片：{', '.join(latency_blocking_files)}。")
    return reasons


def _extract_weak_sample_summary(
    error_analysis_report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """从单图错误归因中抽取最小弱样本摘要。"""

    if not isinstance(error_analysis_report, dict):
        return None
    worst_segment_types = [
        {
            "segment_type": row.get("segment_type"),
            "error_segments": row.get("error_segments"),
            "character_error_rate": row.get("character_error_rate"),
        }
        for row in error_analysis_report.get("error_by_segment_type", [])[:3]
        if isinstance(row, dict)
    ]
    major_error_buckets = [
        {
            "error_bucket": row.get("error_bucket"),
            "segment_count": row.get("segment_count"),
            "description": row.get("description"),
        }
        for row in error_analysis_report.get("error_buckets", [])[:3]
        if isinstance(row, dict)
    ]
    return {
        "file_name": error_analysis_report.get("file_name"),
        "gate_status": (error_analysis_report.get("gate_decision") or {}).get("status")
        if isinstance(error_analysis_report.get("gate_decision"), dict)
        else None,
        "worst_segment_types": worst_segment_types,
        "major_error_buckets": major_error_buckets,
        "next_action": (error_analysis_report.get("gate_decision") or {}).get("next_action")
        if isinstance(error_analysis_report.get("gate_decision"), dict)
        else None,
    }


def _build_batch_gate_recommendations(
    *,
    gate_status: str,
    blocking_files: list[str],
    latency_blocking_files: list[str],
    weak_sample: dict[str, Any] | None,
) -> list[str]:
    """给出与闸门结果一致的下一步建议。"""

    if gate_status == "passed":
        return ["OCR当前批次通过闸门，可以再评估是否进入下一功能。"]

    recommendations = [
        "不要开启ASR、视觉理解或视频真实处理；先把图片OCR的质量和延迟边界说清楚。",
    ]
    weak_file = weak_sample.get("file_name") if isinstance(weak_sample, dict) else None
    if weak_file and weak_file in blocking_files:
        recommendations.append(
            f"下一轮只围绕{weak_file}做一个受控预处理实验，例如裁剪、放大或分区OCR。"
        )
    if latency_blocking_files:
        recommendations.append(
            "把OCR延迟拆成冷启动、模型加载和单图推理三部分，否则无法解释为什么本地CPU耗时远高于2秒目标。"
        )
    recommendations.append(
        "报告中继续保留真实/Mock边界：当前只证明PaddleOCR图片文字提取，不证明完整图片理解。"
    )
    return recommendations


def _summarize_errors_by_segment_type(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按人工文字块类型汇总命中与错误。"""

    summary: dict[str, dict[str, Any]] = {}
    for detail in details:
        segment_type = str(detail.get("segment_type") or "unknown")
        item = summary.setdefault(
            segment_type,
            {
                "segment_type": segment_type,
                "total_segments": 0,
                "exact_segments": 0,
                "error_segments": 0,
                "total_gold_characters": 0,
                "total_edit_distance": 0,
                "example_error_segment_ids": [],
            },
        )
        item["total_segments"] += 1
        item["total_gold_characters"] += int(detail.get("gold_character_count") or 0)
        item["total_edit_distance"] += int(detail.get("edit_distance") or 0)
        if detail.get("error_bucket") == "exact_match":
            item["exact_segments"] += 1
        else:
            item["error_segments"] += 1
            if len(item["example_error_segment_ids"]) < 3:
                item["example_error_segment_ids"].append(detail.get("segment_id"))

    rows: list[dict[str, Any]] = []
    for item in summary.values():
        total_segments = item["total_segments"]
        total_gold_characters = item["total_gold_characters"]
        rows.append(
            {
                **item,
                "error_segment_rate": round(item["error_segments"] / total_segments, 6),
                "character_error_rate": round(item["total_edit_distance"] / total_gold_characters, 6)
                if total_gold_characters
                else None,
            }
        )
    return sorted(
        rows,
        key=lambda item: (-item["error_segments"], -item["total_edit_distance"], item["segment_type"]),
    )


def _summarize_error_buckets(error_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按错误归因类型汇总未完整命中的文字块。"""

    summary: dict[str, dict[str, Any]] = {}
    for detail in error_details:
        bucket = str(detail.get("error_bucket") or "unknown")
        item = summary.setdefault(
            bucket,
            {
                "error_bucket": bucket,
                "description": ERROR_BUCKET_DESCRIPTIONS.get(bucket, "未分类错误。"),
                "segment_count": 0,
                "total_edit_distance": 0,
                "example_segment_ids": [],
            },
        )
        item["segment_count"] += 1
        item["total_edit_distance"] += int(detail.get("edit_distance") or 0)
        if len(item["example_segment_ids"]) < 5:
            item["example_segment_ids"].append(detail.get("segment_id"))
    return sorted(
        summary.values(),
        key=lambda item: (-item["segment_count"], -item["total_edit_distance"], item["error_bucket"]),
    )


def _top_error_segments(error_details: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    """返回编辑距离最高的错误文字块，便于人工复核。"""

    top_details = sorted(
        error_details,
        key=lambda detail: (-int(detail.get("edit_distance") or 0), str(detail.get("segment_id") or "")),
    )[:limit]
    return [
        {
            "segment_id": detail.get("segment_id"),
            "segment_type": detail.get("segment_type"),
            "gold_text": detail.get("gold_text"),
            "comparison_text": detail.get("comparison_text"),
            "matched_ocr_text": detail.get("matched_ocr_text"),
            "edit_distance": detail.get("edit_distance"),
            "character_error_rate": detail.get("character_error_rate"),
            "error_bucket": detail.get("error_bucket"),
        }
        for detail in top_details
    ]


def _build_error_interpretation(
    file_name: str,
    details: list[dict[str, Any]],
    latency_ms: int | None,
) -> list[str]:
    """生成简短人工解读，避免只有机器指标。"""

    by_type = _summarize_errors_by_segment_type(details)
    worst_types = [item["segment_type"] for item in by_type[:3] if item["error_segments"]]
    interpretation = [
        f"{file_name} 的错误主要集中在：{', '.join(worst_types)}。"
        if worst_types
        else f"{file_name} 没有明显错误集中类型。",
        "该图属于小字号、双栏结构图，OCR更容易把模块标签、容量数值和相邻文字拆散或压缩。",
    ]
    if latency_ms is not None:
        interpretation.append(f"本图OCR延迟为{latency_ms}ms，高于图片任务2秒目标。")
    return interpretation


def _render_error_analysis_markdown(analysis: dict[str, Any]) -> str:
    """把错误归因JSON渲染为人工可读Markdown。"""

    overview = analysis["overview"]
    gate_decision = analysis["gate_decision"]
    type_rows = "\n".join(
        "| {segment_type} | {total_segments} | {error_segments} | {error_segment_rate:.2%} | {total_edit_distance} | {character_error_rate:.2%} |".format(
            **row
        )
        for row in analysis["error_by_segment_type"]
    )
    bucket_rows = "\n".join(
        f"| {row['error_bucket']} | {row['segment_count']} | {row['total_edit_distance']} | {row['description']} |"
        for row in analysis["error_buckets"]
    )
    top_rows = "\n".join(
        f"| {row['segment_id']} | {row['segment_type']} | {row['edit_distance']} | {row['error_bucket']} | {row['gold_text']} | {row['comparison_text']} |"
        for row in analysis["top_error_segments"]
    )
    reasons = "\n".join(f"- {reason}" for reason in gate_decision["reasons"]) or "- 当前未发现阻断原因。"
    interpretation = "\n".join(f"- {item}" for item in analysis["interpretation"])

    return f"""# 图片 OCR 错误归因报告：{analysis['file_name']}

## 一、指标概览

- 批次：`{analysis.get('batch_id')}`
- 必选文字块数：{overview['required_segment_count']}
- 完整命中文字块数：{overview['exact_segment_count']}
- 未完整命中文字块数：{overview['error_segment_count']}
- 完整段落召回率：{overview['exact_segment_recall']:.2%}
- 字符错误率：{overview['character_error_rate']:.2%}
- OCR延迟：{overview['ocr_latency_ms']}ms

## 二、闸门判断

结论：`{gate_decision['status']}`

{reasons}

下一步：{gate_decision['next_action']}

## 三、按文字块类型聚合

| 文字块类型 | 总段数 | 错误段数 | 错误段占比 | 编辑距离合计 | 字符错误率 |
|---|---:|---:|---:|---:|---:|
{type_rows}

## 四、按错误类型聚合

| 错误类型 | 段数 | 编辑距离合计 | 解释 |
|---|---:|---:|---|
{bucket_rows}

## 五、编辑距离最高的错误段

| segment_id | segment_type | edit_distance | error_bucket | gold_text | comparison_text |
|---|---|---:|---|---|---|
{top_rows}

## 六、人工解读

{interpretation}

## 七、边界说明

- 本报告只解释已有OCR评估结果，不重新运行PaddleOCR，也不调用DeepSeek。
- 错误归因来自分段人工基准与OCR文本的比较，是工程诊断，不是模型泛化质量结论。
- 视觉理解、语音识别和视频真实处理仍未接入，不能把本报告解释为完整多模态质量评估。
"""


def _render_gate_markdown(report: dict[str, Any]) -> str:
    """把批次级OCR闸门报告渲染为人工可读Markdown。"""

    overview = report["batch_overview"]
    gate_decision = report["gate_decision"]
    thresholds = report["thresholds"]
    file_rows = "\n".join(
        "| {file_name} | {required_segment_count} | {recall} | {error_rate} | {latency} | {status} |".format(
            file_name=row.get("file_name"),
            required_segment_count=row.get("required_segment_count"),
            recall=_format_rate(row.get("exact_segment_recall")),
            error_rate=_format_rate(row.get("character_error_rate")),
            latency=_format_ms(row.get("ocr_latency_ms")),
            status=row.get("status"),
        )
        for row in report["file_checks"]
    )
    reasons = "\n".join(f"- {reason}" for reason in gate_decision["reasons"]) or "- 当前未发现阻断原因。"
    recommendations = "\n".join(f"- {item}" for item in report["recommendations"])
    weak_sample = report.get("weak_sample_analysis") or {}
    weak_types = "\n".join(
        f"- {item['segment_type']}：错误段数 {item['error_segments']}，字符错误率 {_format_rate(item['character_error_rate'])}"
        for item in weak_sample.get("worst_segment_types", [])
    ) or "- 当前没有单图错误归因摘要。"
    weak_buckets = "\n".join(
        f"- {item['error_bucket']}：{item['segment_count']} 段，{item['description']}"
        for item in weak_sample.get("major_error_buckets", [])
    ) or "- 当前没有错误类型摘要。"

    return f"""# 图片 OCR 批次级闸门报告

## 一、批次概览

- 批次：`{report.get('batch_id')}`
- 评估图片数：{overview.get('evaluated_files')}
- 必选文字块总数：{overview.get('total_required_segment_count')}
- 整体完整段落召回率：{_format_rate(overview.get('overall_exact_segment_recall'))}
- 整体字符错误率：{_format_rate(overview.get('overall_character_error_rate'))}
- OCR平均延迟：{_format_ms(overview.get('ocr_avg_latency_ms'))}
- OCR P95延迟：{_format_ms(overview.get('ocr_p95_latency_ms'))}
- OCR外部API成本：{overview.get('ocr_cost_cny')}元

## 二、当前闸门阈值

- 完整段落召回率最低要求：{thresholds['min_exact_segment_recall']:.0%}
- 字符错误率最高要求：{thresholds['max_character_error_rate']:.0%}
- 图片OCR延迟目标：{thresholds['max_latency_ms']}ms

## 三、闸门判断

结论：`{gate_decision['status']}`

{reasons}

下一步：{gate_decision['next_action']}

## 四、逐图检查

| 图片 | 必选文字块数 | 完整段落召回率 | 字符错误率 | OCR延迟ms | 状态 |
|---|---:|---:|---:|---:|---|
{file_rows}

## 五、弱样本摘要

弱样本：`{weak_sample.get('file_name')}`

主要问题文字块类型：

{weak_types}

主要错误类型：

{weak_buckets}

## 六、建议

{recommendations}

## 七、边界说明

- 本报告只读取已有OCR评估和错误归因结果，不重新运行PaddleOCR，也不调用DeepSeek。
- 本报告判断的是当前关键帧OCR评估批次是否通过MVP闸门，不代表OCR模型线上泛化质量。
- 视觉理解、语音识别和视频真实处理仍未接入，不能把本报告解释为完整多模态质量评估。
"""


def _format_rate(value: Any) -> str:
    """把比例格式化为百分比，缺失时明确标注。"""

    return f"{value:.2%}" if isinstance(value, (int, float)) else "当前数据未提供"


def _format_ms(value: Any) -> str:
    """把毫秒值格式化为文本，缺失时明确标注。"""

    return f"{int(value)}ms" if isinstance(value, (int, float)) else "当前数据未提供"


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = argv if argv is not None else sys.argv[1:]
    if len(args) == 5 and args[0] == "evaluate":
        output_path = write_image_ocr_report(
            results_path=args[1],
            gold_path=args[2],
            file_name=args[3],
            output_path=args[4],
        )
        print(json.dumps({"report": str(output_path)}, ensure_ascii=False, indent=2))
        return 0

    if len(args) == 5 and args[0] == "analyze":
        json_path, markdown_path = write_image_ocr_error_analysis(
            evaluation_report_path=args[1],
            batch_summary_path=args[2],
            output_json_path=args[3],
            output_markdown_path=args[4],
        )
        print(
            json.dumps(
                {"json_report": str(json_path), "markdown_report": str(markdown_path)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if len(args) == 5 and args[0] == "gate":
        json_path, markdown_path = write_image_ocr_gate_report(
            summary_report_path=args[1],
            error_analysis_path=args[2],
            output_json_path=args[3],
            output_markdown_path=args[4],
        )
        print(
            json.dumps(
                {"json_report": str(json_path), "markdown_report": str(markdown_path)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if len(args) != 5:
        print(
            "用法: python .\\src\\image_ocr_evaluator.py evaluate "
            "results.jsonl image_ocr_gold.csv img_1.png report.json\n"
            "或: python .\\src\\image_ocr_evaluator.py analyze "
            "image_ocr_eval_img_9.json image_ocr_eval_summary.json analysis.json analysis.md\n"
            "或: python .\\src\\image_ocr_evaluator.py gate "
            "image_ocr_eval_summary.json image_ocr_error_analysis_img_9.json gate.json gate.md"
        )
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
