"""文本主分类人工评估工具。

本模块只做一件事：把文本文件的 topic 预测结果导出为人工标注表，
并在人工填写 gold_topic 后计算 Accuracy、Macro-F1 和分类级指标。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


MISSING_VALUE_TEXT = "当前数据未提供"

VALID_TOPIC_VALUES = {
    "news",
    "finance_business",
    "ads_marketing",
    "technology",
    "sports_health",
    "entertainment",
    "lifestyle",
    "knowledge",
    "other",
}

TEMPLATE_FIELDS = [
    "batch_id",
    "file_id",
    "file_name",
    "predicted_topic",
    "predicted_secondary_topics",
    "summary",
    "raw_text_preview",
    "gold_topic",
    "reviewer_note",
]

GOLD_FIELDS = [
    "file_name",
    "gold_topic",
    "reviewer_note",
]


def read_json_objects(file_path: str | Path) -> list[dict[str, Any]]:
    """读取连续 JSON 对象文件，兼容项目中缩进后的 JSONL。"""

    path = Path(file_path)
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(content):
        while index < len(content) and content[index].isspace():
            index += 1
        if index >= len(content):
            break
        record, index = decoder.raw_decode(content, index)
        if isinstance(record, dict):
            records.append(record)
    return records


def extract_text_topic_rows(records: list[dict[str, Any]], *, preview_chars: int = 160) -> list[dict[str, str]]:
    """从文件级结果中提取文本主分类人工标注行。"""

    rows: list[dict[str, str]] = []
    for record in records:
        if record.get("media_type") != "text":
            continue

        raw_text = str(record.get("raw_text") or "")
        rows.append(
            {
                "batch_id": _as_text(record.get("batch_id")),
                "file_id": _as_text(record.get("file_id")),
                "file_name": _as_text(record.get("file_name")),
                "predicted_topic": _optional_text(record.get("topic")),
                "predicted_secondary_topics": _format_list(record.get("secondary_topics")),
                "summary": _as_text(record.get("summary")),
                "raw_text_preview": _clean_preview(raw_text, preview_chars),
                "gold_topic": "",
                "reviewer_note": "",
            }
        )
    return rows


def read_gold_topic_rows(gold_path: str | Path) -> list[dict[str, str]]:
    """读取文本主分类标准答案表。"""

    path = Path(gold_path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def apply_gold_topics(
    template_rows: list[dict[str, str]],
    gold_rows: list[dict[str, str]],
    *,
    match_key: str = "file_name",
) -> list[dict[str, str]]:
    """把标准答案表合并进人工标注模板。"""

    gold_by_key = {
        str(row.get(match_key) or ""): row
        for row in gold_rows
        if row.get(match_key)
    }

    merged_rows: list[dict[str, str]] = []
    for row in template_rows:
        merged_row = dict(row)
        gold_row = gold_by_key.get(str(row.get(match_key) or ""))
        if gold_row:
            merged_row["gold_topic"] = str(gold_row.get("gold_topic") or "")
            merged_row["reviewer_note"] = str(gold_row.get("reviewer_note") or "")
        merged_rows.append(merged_row)
    return merged_rows


def write_annotation_template(
    results_path: str | Path,
    output_path: str | Path,
    gold_path: str | Path | None = None,
) -> Path:
    """从 results.jsonl 生成文本主分类人工标注模板。"""

    rows = extract_text_topic_rows(read_json_objects(results_path))
    if gold_path is not None:
        rows = apply_gold_topics(rows, read_gold_topic_rows(gold_path))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TEMPLATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_annotation_rows(annotation_path: str | Path) -> list[dict[str, str]]:
    """读取人工标注表。"""

    path = Path(annotation_path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def evaluate_topic_metrics(annotation_rows: list[dict[str, str]]) -> dict[str, Any]:
    """根据人工标注表计算文本主分类指标。"""

    evaluated_details: list[dict[str, Any]] = []
    missing_label_file_ids: list[str] = []
    missing_prediction_file_ids: list[str] = []
    invalid_prediction_file_ids: list[str] = []
    valid_prediction_count = 0

    for row in annotation_rows:
        file_id = str(row.get("file_id") or "")
        predicted_topic = str(row.get("predicted_topic") or "").strip()
        gold_topic = str(row.get("gold_topic") or "").strip()

        if not gold_topic:
            missing_label_file_ids.append(file_id)
            continue

        prediction_available = predicted_topic not in {"", MISSING_VALUE_TEXT}
        prediction_valid = prediction_available and predicted_topic in VALID_TOPIC_VALUES
        if not prediction_available:
            missing_prediction_file_ids.append(file_id)
        elif not prediction_valid:
            invalid_prediction_file_ids.append(file_id)
        else:
            valid_prediction_count += 1

        is_correct = predicted_topic == gold_topic
        evaluated_details.append(
            {
                "file_id": file_id,
                "file_name": row.get("file_name"),
                "predicted_topic": predicted_topic,
                "gold_topic": gold_topic,
                "is_correct": is_correct,
                "reviewer_note": row.get("reviewer_note") or "",
            }
        )

    evaluated_count = len(evaluated_details)
    correct_count = sum(1 for item in evaluated_details if item["is_correct"])
    evaluated_labels = sorted(
        {
            topic
            for item in evaluated_details
            for topic in (item["gold_topic"], item["predicted_topic"])
            if topic in VALID_TOPIC_VALUES
        }
    )
    raw_per_class_metrics = _calculate_per_class_metrics(evaluated_details, evaluated_labels)
    macro_f1 = (
        round(sum(item["f1"] for item in raw_per_class_metrics) / len(raw_per_class_metrics), 6)
        if raw_per_class_metrics
        else MISSING_VALUE_TEXT
    )
    per_class_metrics = [
        {
            **item,
            "precision": round(item["precision"], 6),
            "recall": round(item["recall"], 6),
            "f1": round(item["f1"], 6),
        }
        for item in raw_per_class_metrics
    ]

    return {
        "schema_version": "v3",
        "metric_name": "text_topic_classification",
        "total_template_rows": len(annotation_rows),
        "evaluated_count": evaluated_count,
        "missing_label_count": len(missing_label_file_ids),
        "valid_prediction_count": valid_prediction_count,
        "missing_prediction_count": len(missing_prediction_file_ids),
        "invalid_prediction_count": len(invalid_prediction_file_ids),
        "correct_count": correct_count,
        "accuracy": round(correct_count / evaluated_count, 6) if evaluated_count else MISSING_VALUE_TEXT,
        "valid_prediction_accuracy": (
            round(correct_count / valid_prediction_count, 6)
            if valid_prediction_count
            else MISSING_VALUE_TEXT
        ),
        "prediction_coverage": (
            round(valid_prediction_count / evaluated_count, 6)
            if evaluated_count
            else MISSING_VALUE_TEXT
        ),
        "macro_f1": macro_f1,
        "evaluated_labels": evaluated_labels,
        "per_class_metrics": per_class_metrics,
        "missing_label_file_ids": missing_label_file_ids,
        "missing_prediction_file_ids": missing_prediction_file_ids,
        "invalid_prediction_file_ids": invalid_prediction_file_ids,
        "details": evaluated_details,
        "field_notes": {
            "gold_topic": "人工标注的正确主分类，用于和模型预测的 predicted_topic 对比。",
            "predicted_topic": "模型输出的主分类，用于衡量文本分类结果是否命中人工标签。",
            "reviewer_note": "人工评审备注，用于记录分类正确或错误的判断依据。",
            "accuracy": "端到端文本主分类准确率，计算方式为 correct_count / evaluated_count；调用失败或无有效预测按未命中计算。",
            "valid_prediction_accuracy": "仅在有效九分类预测中的准确率，用于把分类判断能力与调用可用性分开观察。",
            "prediction_coverage": "有效九分类预测数占已标注样本数的比例，用于衡量模型调用和结构解析是否稳定。",
            "macro_f1": "九类业务标签中本批次实际出现分类的 F1 简单平均；无结果和非法预测会造成对应真实分类漏报，但不会被当作新分类。",
            "precision": "预测为某分类的样本中，真正属于该分类的比例。",
            "recall": "人工标注为某分类的样本中，被模型正确识别的比例。",
            "f1": "单个分类 Precision 与 Recall 的调和平均，用于综合衡量误报和漏报。",
            "support": "人工标准答案中属于某分类的样本数，用于判断该分类证据量。",
            "evaluated_labels": "人工标签或模型预测中实际出现的分类，用于说明本次指标覆盖范围。",
            "evaluated_count": "已经填写 gold_topic 并纳入统计的文本样本数。",
            "valid_prediction_count": "成功产出九类范围内 predicted_topic 的样本数。",
            "missing_prediction_count": "没有产出 predicted_topic 的已标注样本数，常用于识别调用或解析失败。",
            "invalid_prediction_count": "产出了内容但不属于九类允许值的样本数，用于发现输出约束失效。",
        },
    }


def _calculate_per_class_metrics(
    evaluated_details: list[dict[str, Any]],
    labels: list[str],
) -> list[dict[str, Any]]:
    """计算每个分类的 Precision、Recall、F1 和样本数。"""

    metrics: list[dict[str, Any]] = []
    for label in labels:
        true_positive = sum(
            1
            for item in evaluated_details
            if item["gold_topic"] == label and item["predicted_topic"] == label
        )
        false_positive = sum(
            1
            for item in evaluated_details
            if item["gold_topic"] != label and item["predicted_topic"] == label
        )
        false_negative = sum(
            1
            for item in evaluated_details
            if item["gold_topic"] == label and item["predicted_topic"] != label
        )
        support = sum(1 for item in evaluated_details if item["gold_topic"] == label)
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        metrics.append(
            {
                "topic": label,
                "support": support,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return metrics


def _safe_ratio(numerator: float, denominator: float) -> float:
    """安全计算比例，分母为零时返回 0。"""

    return numerator / denominator if denominator else 0.0


def write_evaluation_report(annotation_path: str | Path, json_output_path: str | Path, markdown_output_path: str | Path) -> dict[str, str]:
    """读取人工标注表并写入 JSON 和 Markdown 评估报告。"""

    report = evaluate_topic_metrics(read_annotation_rows(annotation_path))
    json_path = Path(json_output_path)
    markdown_path = Path(markdown_output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_evaluation_markdown(report), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }


def render_evaluation_markdown(report: dict[str, Any]) -> str:
    """把文本主分类评估结果渲染为 Markdown。"""

    lines = [
        "# 文本主分类人工评估报告",
        "",
        "说明：本报告只评估文本文件的 `topic` 主分类，不评估摘要、关键词、副分类、图片或视频结果。",
        "",
        "| 指标 | 数值 | 含义 |",
        "|---|---:|---|",
        f"| 模板行数 | {report['total_template_rows']} | 需要人工判断的文本样本数 |",
        f"| 已评估样本数 | {report['evaluated_count']} | 已填写 gold_topic 的样本数 |",
        f"| 缺少标签样本数 | {report['missing_label_count']} | 尚未填写 gold_topic 的样本数 |",
        f"| 有效预测数 | {report['valid_prediction_count']} | 成功产出九类范围内 predicted_topic 的样本数 |",
        f"| 缺少预测数 | {report['missing_prediction_count']} | 调用或解析失败导致没有 predicted_topic 的样本数 |",
        f"| 非法预测数 | {report['invalid_prediction_count']} | predicted_topic 不属于九类允许值的样本数 |",
        f"| 预测正确数 | {report['correct_count']} | predicted_topic 与 gold_topic 相同的样本数 |",
        f"| 端到端 Accuracy | {_format_metric(report['accuracy'])} | 以全部已标注样本为分母，无有效预测按未命中计算 |",
        f"| 有效预测 Accuracy | {_format_metric(report['valid_prediction_accuracy'])} | 仅衡量成功产出九类预测的样本 |",
        f"| 预测覆盖率 | {_format_metric(report['prediction_coverage'])} | 有效预测数占已标注样本数的比例 |",
        f"| Macro-F1 | {_format_metric(report['macro_f1'])} | 九类业务标签中本批次实际出现分类 F1 的简单平均 |",
        "",
        "## 分类级指标",
        "",
        "| topic | support | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]

    if report["per_class_metrics"]:
        for item in report["per_class_metrics"]:
            lines.append(
                f"| {item['topic']} | {item['support']} | {_format_metric(item['precision'])} | "
                f"{_format_metric(item['recall'])} | {_format_metric(item['f1'])} |"
            )
    else:
        lines.append("| 当前数据未提供 | 0 | 当前数据未提供 | 当前数据未提供 | 当前数据未提供 |")

    lines.extend(
        [
        "",
        "## 明细",
        "",
        "| file_id | file_name | predicted_topic | gold_topic | 是否正确 | 备注 |",
        "|---|---|---|---|---|---|",
        ]
    )

    if report["details"]:
        for item in report["details"]:
            lines.append(
                f"| {item['file_id']} | {item.get('file_name')} | {item['predicted_topic']} | "
                f"{item['gold_topic']} | {'是' if item['is_correct'] else '否'} | {item.get('reviewer_note') or ''} |"
            )
    else:
        lines.append("| 当前数据未提供 | 当前数据未提供 | 当前数据未提供 | 当前数据未提供 | 当前数据未提供 | 请先填写 gold_topic |")

    if report["missing_label_file_ids"]:
        lines.extend(["", "## 尚未标注的文件", ""])
        lines.extend(f"- {file_id}" for file_id in report["missing_label_file_ids"])

    if report["missing_prediction_file_ids"]:
        lines.extend(["", "## 没有有效预测的文件", ""])
        lines.extend(f"- {file_id}" for file_id in report["missing_prediction_file_ids"])

    if report["invalid_prediction_file_ids"]:
        lines.extend(["", "## 预测值不符合九类约束的文件", ""])
        lines.extend(f"- {file_id}" for file_id in report["invalid_prediction_file_ids"])

    lines.extend(["", "## 字段说明", "", "| 字段 | 含义与作用 |", "|---|---|"])
    for field_name, note in report["field_notes"].items():
        lines.append(f"| `{field_name}` | {note} |")

    lines.append("")
    return "\n".join(lines)


def _as_text(value: Any) -> str:
    """把值转换成文本，缺失时使用统一说明。"""

    if value is None:
        return MISSING_VALUE_TEXT
    text = str(value).strip()
    return text if text else MISSING_VALUE_TEXT


def _optional_text(value: Any) -> str:
    """把可选值转换成文本，缺失时保留为空字符串。"""

    if value is None:
        return ""
    return str(value).strip()


def _format_list(value: Any) -> str:
    """把列表字段压平成适合 CSV 阅读的短文本。"""

    if not value:
        return ""
    if isinstance(value, list):
        return "、".join(str(item) for item in value)
    return str(value)


def _clean_preview(raw_text: str, preview_chars: int) -> str:
    """生成单行文本预览。"""

    compact_text = " ".join(raw_text.split())
    if len(compact_text) <= preview_chars:
        return compact_text
    return compact_text[:preview_chars] + "..."


def _format_metric(value: Any) -> str:
    """格式化评估指标。"""

    if value == MISSING_VALUE_TEXT:
        return MISSING_VALUE_TEXT
    return f"{float(value) * 100:.2f}%"


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("用法: python .\\src\\text_topic_evaluator.py template results.jsonl template.csv [gold.csv]")
        print("或: python .\\src\\text_topic_evaluator.py evaluate template.csv report.json report.md")
        return 2

    command = args[0]
    if command == "template" and len(args) in {3, 4}:
        gold_path = args[3] if len(args) == 4 else None
        output_path = write_annotation_template(args[1], args[2], gold_path)
        print(json.dumps({"template": str(output_path)}, ensure_ascii=False, indent=2))
        return 0

    if command == "evaluate" and len(args) == 4:
        output_paths = write_evaluation_report(args[1], args[2], args[3])
        print(json.dumps(output_paths, ensure_ascii=False, indent=2))
        return 0

    print("参数不正确。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
