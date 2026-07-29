"""RapidOCR 候选后端评估工具。

本模块只用于下一步 OCR 候选验证：用同一批图片和同一份人工基准，
对照 RapidOCR / ONNXRuntime 路线是否值得继续。
它不接入主流水线，不调用 DeepSeek，也不访问云 OCR API。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from image_ocr_evaluator import evaluate_image_ocr, read_ocr_gold_rows


CandidateOcrClient = Callable[[str | Path], dict[str, str | None]]


def check_rapidocr_dependency() -> dict[str, Any]:
    """检查 RapidOCR 候选后端依赖是否存在。"""

    packages = {
        "rapidocr": importlib.util.find_spec("rapidocr") is not None,
        "rapidocr_onnxruntime": importlib.util.find_spec("rapidocr_onnxruntime") is not None,
        "onnxruntime": importlib.util.find_spec("onnxruntime") is not None,
    }
    runnable = packages["rapidocr"] or packages["rapidocr_onnxruntime"]
    return {
        "status": "available" if runnable else "missing",
        "packages": packages,
        "install_hint": "pip install rapidocr onnxruntime",
    }


def rapidocr_client(image_path: str | Path) -> dict[str, str | None]:
    """调用 RapidOCR 或 rapidocr_onnxruntime，并返回标准化 OCR 文字。"""

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"OCR 输入图片不存在：{path}")
    if path.stat().st_size == 0:
        raise ValueError("OCR 输入图片为空。")

    engine = _create_rapidocr_engine()
    result = engine(str(path))
    return _parse_rapidocr_result(result)


def run_rapidocr_candidate_evaluation(
    *,
    image_paths: list[str | Path],
    gold_path: str | Path,
    ocr_client: CandidateOcrClient | None = None,
    min_exact_segment_recall: float = 0.9,
    max_character_error_rate: float = 0.05,
    max_latency_ms: int = 2000,
) -> dict[str, Any]:
    """运行 RapidOCR 候选评估。"""

    dependency = check_rapidocr_dependency()
    if ocr_client is None and dependency["status"] != "available":
        return _build_dependency_missing_report(
            image_paths=image_paths,
            dependency=dependency,
            min_exact_segment_recall=min_exact_segment_recall,
            max_character_error_rate=max_character_error_rate,
            max_latency_ms=max_latency_ms,
        )

    client = ocr_client or rapidocr_client
    gold_rows = read_ocr_gold_rows(gold_path)
    file_metrics = [
        _evaluate_one_image(
            image_path=Path(image_path),
            gold_rows=gold_rows,
            ocr_client=client,
        )
        for image_path in image_paths
    ]
    overview = _build_overview(file_metrics)
    decision = _build_gate_decision(
        overview=overview,
        file_metrics=file_metrics,
        min_exact_segment_recall=min_exact_segment_recall,
        max_character_error_rate=max_character_error_rate,
        max_latency_ms=max_latency_ms,
    )

    return {
        "schema_version": "v1",
        "report_name": "rapidocr_candidate_evaluation",
        "backend_id": "rapidocr_onnxruntime_local",
        "dependency": dependency,
        "thresholds": {
            "min_exact_segment_recall": min_exact_segment_recall,
            "max_character_error_rate": max_character_error_rate,
            "max_latency_ms": max_latency_ms,
        },
        "overview": overview,
        "gate_decision": decision,
        "file_metrics": file_metrics,
        "boundary_notes": [
            "本报告只评估 RapidOCR 候选后端，不把候选后端接入主流水线。",
            "本报告不调用 DeepSeek，不调用云 OCR API，不产生外部 API 费用。",
            "如果依赖缺失，报告只说明当前无法运行，不编造质量或延迟结果。",
        ],
        "field_notes": {
            "backend_id": "OCR候选后端标识，用来区分当前PaddleOCR和待评估RapidOCR。",
            "dependency": "本地依赖状态，用来判断本轮能否真实运行候选后端。",
            "exact_segment_recall": "完整识别的必选业务文字块占比，用于判断候选OCR是否漏掉关键文字。",
            "character_error_rate": "分段编辑距离除以人工正确字符总数，用于判断候选OCR错漏字程度。",
            "ocr_latency_ms": "单张图片OCR耗时，用于判断候选后端是否接近延迟目标。",
            "gate_decision": "候选后端闸门判断，用来决定是否值得继续接入主流程。",
        },
    }


def write_rapidocr_candidate_report(
    *,
    image_paths: list[str | Path],
    gold_path: str | Path,
    output_json_path: str | Path,
    output_markdown_path: str | Path,
    ocr_client: CandidateOcrClient | None = None,
) -> tuple[Path, Path]:
    """写出 RapidOCR 候选评估 JSON 和 Markdown 报告。"""

    report = run_rapidocr_candidate_evaluation(
        image_paths=image_paths,
        gold_path=gold_path,
        ocr_client=ocr_client,
    )
    json_path = Path(output_json_path)
    markdown_path = Path(output_markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _create_rapidocr_engine() -> Any:
    """创建 RapidOCR 引擎，兼容不同包名。"""

    try:
        from rapidocr import RapidOCR
    except ImportError:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError("未安装 RapidOCR 候选后端；请先安装 rapidocr 或 rapidocr_onnxruntime。") from exc
    return RapidOCR()


def _parse_rapidocr_result(result: Any) -> dict[str, str | None]:
    """把 RapidOCR 常见返回结构转换为统一 OCR 文字。"""

    text_lines: list[str] = []

    if isinstance(result, dict):
        text_lines.extend(_extract_texts_from_mapping(result))
    elif hasattr(result, "txts"):
        value = getattr(result, "txts")
        text_lines.extend(str(item).strip() for item in value if str(item).strip())
    elif hasattr(result, "to_markdown"):
        value = result.to_markdown()
        if isinstance(value, str):
            text_lines.extend(line.strip() for line in value.splitlines() if line.strip())
    elif isinstance(result, tuple) and result:
        text_lines.extend(_extract_texts_from_sequence(result[0]))
    elif isinstance(result, list):
        text_lines.extend(_extract_texts_from_sequence(result))

    return {"ocr_text": "\n".join(text_lines) if text_lines else None}


def _extract_texts_from_mapping(value: dict[str, Any]) -> list[str]:
    """从字典结构中提取文字列表。"""

    for key in ("txts", "texts", "rec_texts"):
        texts = value.get(key)
        if isinstance(texts, list):
            return [str(item).strip() for item in texts if str(item).strip()]
    results = value.get("results")
    if isinstance(results, list):
        return _extract_texts_from_sequence(results)
    return []


def _extract_texts_from_sequence(value: Any) -> list[str]:
    """从列表或嵌套列表中提取 OCR 文字。"""

    text_lines: list[str] = []
    if not isinstance(value, list):
        return text_lines
    for item in value:
        extracted = _extract_text_from_item(item)
        if extracted:
            text_lines.append(extracted)
    return text_lines


def _extract_text_from_item(item: Any) -> str | None:
    """从单条OCR结果中提取文字。"""

    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, dict):
        for key in ("text", "txt", "rec_text"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    if isinstance(item, (list, tuple)):
        for part in item:
            if isinstance(part, str) and part.strip():
                return part.strip()
            if (
                isinstance(part, (list, tuple))
                and part
                and isinstance(part[0], str)
                and part[0].strip()
            ):
                return part[0].strip()
    return None


def _evaluate_one_image(
    *,
    image_path: Path,
    gold_rows: list[dict[str, str]],
    ocr_client: CandidateOcrClient,
) -> dict[str, Any]:
    """评估单张图片。"""

    started_at = perf_counter()
    try:
        result = ocr_client(image_path)
        latency_ms = int(round((perf_counter() - started_at) * 1000))
        ocr_text = result.get("ocr_text") or ""
        if not ocr_text.strip():
            raise ValueError("候选OCR没有返回可评估文字。")
        evaluation = evaluate_image_ocr(
            file_name=image_path.name,
            ocr_text=ocr_text,
            gold_rows=gold_rows,
        )
        return {
            "file_name": image_path.name,
            "status": "success",
            "ocr_latency_ms": latency_ms,
            "ocr_text_char_count": len(ocr_text),
            "required_segment_count": evaluation["required_segment_count"],
            "exact_segment_count": evaluation["exact_segment_count"],
            "exact_segment_recall": evaluation["exact_segment_recall"],
            "total_gold_characters": evaluation["total_gold_characters"],
            "character_error_rate": evaluation["character_error_rate"],
            "total_edit_distance": evaluation["total_edit_distance"],
            "error_message": None,
        }
    except Exception as exc:
        latency_ms = int(round((perf_counter() - started_at) * 1000))
        return {
            "file_name": image_path.name,
            "status": "failed",
            "ocr_latency_ms": latency_ms,
            "ocr_text_char_count": 0,
            "required_segment_count": None,
            "exact_segment_count": None,
            "exact_segment_recall": None,
            "total_gold_characters": None,
            "character_error_rate": None,
            "total_edit_distance": None,
            "error_message": str(exc),
        }


def _build_overview(file_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总候选后端评估结果。"""

    successful = [item for item in file_metrics if item.get("status") == "success"]
    total_required = sum(int(item.get("required_segment_count") or 0) for item in successful)
    total_exact = sum(int(item.get("exact_segment_count") or 0) for item in successful)
    total_gold_chars = sum(int(item.get("total_gold_characters") or 0) for item in successful)
    total_edit_distance = sum(int(item.get("total_edit_distance") or 0) for item in successful)

    latencies = [
        int(item["ocr_latency_ms"])
        for item in successful
        if isinstance(item.get("ocr_latency_ms"), int)
    ]
    return {
        "evaluated_files": len(file_metrics),
        "successful_files": len(successful),
        "failed_files": len(file_metrics) - len(successful),
        "total_required_segment_count": total_required,
        "total_exact_segment_count": total_exact,
        "overall_exact_segment_recall": round(total_exact / total_required, 6) if total_required else None,
        "overall_character_error_rate": (
            round(total_edit_distance / total_gold_chars, 6) if total_gold_chars else None
        ),
        "ocr_avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "ocr_p95_latency_ms": max(latencies) if latencies else None,
        "ocr_external_api_cost_cny": 0.0,
    }


def _build_gate_decision(
    *,
    overview: dict[str, Any],
    file_metrics: list[dict[str, Any]],
    min_exact_segment_recall: float,
    max_character_error_rate: float,
    max_latency_ms: int,
) -> dict[str, Any]:
    """生成候选后端闸门判断。"""

    if overview["successful_files"] == 0:
        return {
            "status": "not_runnable",
            "reasons": ["候选OCR没有任何图片成功产出可评估结果。"],
            "next_action": "先解决候选后端安装或调用问题，不进入主流程。",
        }

    reasons: list[str] = []
    recall = overview.get("overall_exact_segment_recall")
    error_rate = overview.get("overall_character_error_rate")
    p95_latency = overview.get("ocr_p95_latency_ms")
    if isinstance(recall, (int, float)) and recall < min_exact_segment_recall:
        reasons.append(f"整体完整段落召回率为{recall:.2%}，低于{min_exact_segment_recall:.0%}阈值。")
    if isinstance(error_rate, (int, float)) and error_rate > max_character_error_rate:
        reasons.append(f"整体字符错误率为{error_rate:.2%}，高于{max_character_error_rate:.0%}阈值。")
    if isinstance(p95_latency, (int, float)) and p95_latency > max_latency_ms:
        reasons.append(f"OCR P95延迟为{int(p95_latency)}ms，高于{max_latency_ms}ms目标。")

    failed_files = [item["file_name"] for item in file_metrics if item.get("status") != "success"]
    if failed_files:
        reasons.append(f"存在未成功评估图片：{', '.join(failed_files)}。")

    status = "passed" if not reasons else "not_passed"
    next_action = (
        "候选后端通过当前OCR闸门，可考虑进入更大样本验证。"
        if status == "passed"
        else "候选后端未通过当前OCR闸门，不应接入主流程。"
    )
    return {"status": status, "reasons": reasons, "next_action": next_action}


def _build_dependency_missing_report(
    *,
    image_paths: list[str | Path],
    dependency: dict[str, Any],
    min_exact_segment_recall: float,
    max_character_error_rate: float,
    max_latency_ms: int,
) -> dict[str, Any]:
    """构造依赖缺失报告，不编造评估结果。"""

    return {
        "schema_version": "v1",
        "report_name": "rapidocr_candidate_evaluation",
        "backend_id": "rapidocr_onnxruntime_local",
        "dependency": dependency,
        "thresholds": {
            "min_exact_segment_recall": min_exact_segment_recall,
            "max_character_error_rate": max_character_error_rate,
            "max_latency_ms": max_latency_ms,
        },
        "overview": {
            "evaluated_files": len(image_paths),
            "successful_files": 0,
            "failed_files": 0,
            "overall_exact_segment_recall": None,
            "overall_character_error_rate": None,
            "ocr_avg_latency_ms": None,
            "ocr_p95_latency_ms": None,
            "ocr_external_api_cost_cny": 0.0,
        },
        "gate_decision": {
            "status": "dependency_missing",
            "reasons": [
                "当前环境未安装 rapidocr、rapidocr_onnxruntime 或 onnxruntime，无法真实运行候选OCR。",
            ],
            "next_action": "先安装 RapidOCR 候选依赖，再用同一批图片和人工基准重新运行本报告。",
        },
        "file_metrics": [
            {
                "file_name": Path(image_path).name,
                "status": "skipped_dependency_missing",
                "ocr_latency_ms": None,
                "exact_segment_recall": None,
                "character_error_rate": None,
                "error_message": "RapidOCR候选依赖缺失。",
            }
            for image_path in image_paths
        ],
        "boundary_notes": [
            "本报告没有真实运行 RapidOCR，因此不能产生质量、延迟或效果结论。",
            "本报告不调用 DeepSeek，不调用云 OCR API，不产生外部 API 费用。",
        ],
        "field_notes": {
            "backend_id": "OCR候选后端标识，用来区分当前PaddleOCR和待评估RapidOCR。",
            "dependency": "本地依赖状态，用来判断本轮能否真实运行候选后端。",
            "gate_decision": "候选后端闸门判断，用来决定是否值得继续接入主流程。",
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    """生成候选评估 Markdown 报告。"""

    overview = report["overview"]
    gate = report["gate_decision"]
    metric_rows = "\n".join(
        "| {file_name} | {status} | {exact_segment_recall} | {character_error_rate} | {ocr_latency_ms} | {error_message} |".format(
            file_name=item.get("file_name"),
            status=item.get("status"),
            exact_segment_recall=_format_rate(item.get("exact_segment_recall")),
            character_error_rate=_format_rate(item.get("character_error_rate")),
            ocr_latency_ms=_format_ms(item.get("ocr_latency_ms")),
            error_message=item.get("error_message"),
        )
        for item in report["file_metrics"]
    )
    reasons = "\n".join(f"- {reason}" for reason in gate["reasons"])

    return f"""# RapidOCR 候选后端评估报告

## 一、当前结论

- 候选后端：`{report['backend_id']}`
- 依赖状态：`{report['dependency']['status']}`
- 闸门判断：`{gate['status']}`
- 下一步：{gate['next_action']}

## 二、批次概览

| 指标 | 数值 | 作用 |
|---|---:|---|
| 成功评估图片数 | {overview.get('successful_files')} | 判断候选后端是否跑通 |
| 完整段落召回率 | {_format_rate(overview.get('overall_exact_segment_recall'))} | 判断是否漏掉关键业务文字 |
| 字符错误率 | {_format_rate(overview.get('overall_character_error_rate'))} | 判断错漏字程度 |
| OCR平均延迟 | {_format_ms(overview.get('ocr_avg_latency_ms'))} | 判断平均耗时 |
| OCR P95延迟 | {_format_ms(overview.get('ocr_p95_latency_ms'))} | 判断慢调用 |
| 外部API成本 | {overview.get('ocr_external_api_cost_cny')}元 | 本地候选不产生外部API费用 |

## 三、判断理由

{reasons}

## 四、逐图结果

| 图片 | 状态 | 完整段落召回率 | 字符错误率 | OCR延迟 | 错误 |
|---|---|---:|---:|---:|---|
{metric_rows}

## 五、边界说明

- 本报告只用于 RapidOCR 候选后端评估，不接入主流水线。
- 依赖缺失时不会编造质量、延迟或效果结论。
- 本报告不调用 DeepSeek，不调用云 OCR API，不产生外部 API 费用。
"""


def _format_rate(value: Any) -> str:
    """把比例格式化为百分比。"""

    return f"{value:.2%}" if isinstance(value, (int, float)) else "当前数据未提供"


def _format_ms(value: Any) -> str:
    """把毫秒格式化为文本。"""

    return f"{int(value)}ms" if isinstance(value, (int, float)) else "当前数据未提供"


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 4:
        print(
            "用法: python .\\src\\rapidocr_candidate_evaluator.py "
            "image_ocr_gold.csv rapidocr_candidate_eval.json rapidocr_candidate_eval.md "
            "img_7.jpg img_8.jpg img_9.jpg"
        )
        return 2

    gold_path = args[0]
    output_json_path = args[1]
    output_markdown_path = args[2]
    image_paths = args[3:]
    json_path, markdown_path = write_rapidocr_candidate_report(
        image_paths=image_paths,
        gold_path=gold_path,
        output_json_path=output_json_path,
        output_markdown_path=output_markdown_path,
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
