"""图片 OCR 业务文字评估工具。

本模块只评估人工确认的业务文字块，不把按钮、点赞数、时长等界面文字
计入质量指标，也不要求复杂页面中的文字按固定阅读顺序输出。
"""

from __future__ import annotations

import csv
import json
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


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 5 or args[0] != "evaluate":
        print(
            "用法: python .\\src\\image_ocr_evaluator.py evaluate "
            "results.jsonl image_ocr_gold.csv img_1.png report.json"
        )
        return 2

    output_path = write_image_ocr_report(
        results_path=args[1],
        gold_path=args[2],
        file_name=args[3],
        output_path=args[4],
    )
    print(json.dumps({"report": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
