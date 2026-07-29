"""图片 OCR 最小实验工具。

本模块只服务于当前 OCR 功能闸门：用已有人工基准比较原图、整图放大和
左右分区放大后的 PaddleOCR 结果，并拆分本地 OCR 延迟来源。
它不调用 DeepSeek，也不接入新的模型能力。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from PIL import Image

from image_ocr_evaluator import evaluate_image_ocr, read_ocr_gold_rows
from model_clients import (
    _create_paddleocr_engine,
    _decode_image_for_paddleocr,
    _parse_paddleocr_prediction,
    paddleocr_client,
)


OcrClient = Callable[[str | Path], dict[str, str | None]]
EngineFactory = Callable[[], Any]
ImageDecoder = Callable[[Path], Any]
PredictionParser = Callable[[Any], dict[str, str | None]]


def build_preprocess_variants(
    *,
    image_path: str | Path,
    variants_dir: str | Path,
) -> list[dict[str, Any]]:
    """生成当前最小实验需要的预处理图片变体。"""

    source_path = Path(image_path)
    output_dir = Path(variants_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as source_image:
        image = source_image.convert("RGB")
        width, height = image.size
        resample = getattr(Image, "Resampling", Image).BICUBIC

        full_2x_path = output_dir / f"{source_path.stem}_full_2x.png"
        image.resize((width * 2, height * 2), resample=resample).save(full_2x_path)

        middle_x = width // 2
        left_path = output_dir / f"{source_path.stem}_left_2x.png"
        right_path = output_dir / f"{source_path.stem}_right_2x.png"
        image.crop((0, 0, middle_x, height)).resize(
            (middle_x * 2, height * 2),
            resample=resample,
        ).save(left_path)
        image.crop((middle_x, 0, width, height)).resize(
            ((width - middle_x) * 2, height * 2),
            resample=resample,
        ).save(right_path)

    return [
        {
            "variant_name": "full_image_2x",
            "description": "整张图片放大2倍，用于观察小字号文字是否因整体放大而更容易识别。",
            "image_paths": [str(full_2x_path)],
        },
        {
            "variant_name": "vertical_halves_2x",
            "description": "按左右两半分区，每个分区放大2倍后分别OCR，再合并文字结果。",
            "image_paths": [str(left_path), str(right_path)],
        },
    ]


def run_preprocessing_experiment(
    *,
    image_path: str | Path,
    gold_path: str | Path,
    baseline_report_path: str | Path,
    batch_summary_path: str | Path,
    variants_dir: str | Path,
    ocr_client: OcrClient = paddleocr_client,
    min_exact_segment_recall: float = 0.9,
    max_character_error_rate: float = 0.05,
    max_latency_ms: int = 2000,
) -> dict[str, Any]:
    """运行受控预处理实验并返回结构化报告。"""

    source_path = Path(image_path)
    baseline_report = json.loads(Path(baseline_report_path).read_text(encoding="utf-8"))
    batch_summary = json.loads(Path(batch_summary_path).read_text(encoding="utf-8"))
    gold_rows = read_ocr_gold_rows(gold_path)
    variants = build_preprocess_variants(image_path=source_path, variants_dir=variants_dir)

    variant_results = [
        _run_variant_ocr(
            variant=variant,
            file_name=source_path.name,
            gold_rows=gold_rows,
            ocr_client=ocr_client,
        )
        for variant in variants
    ]
    baseline = _build_baseline_summary(baseline_report, batch_summary)
    decision = _build_experiment_decision(
        baseline=baseline,
        variant_results=variant_results,
        min_exact_segment_recall=min_exact_segment_recall,
        max_character_error_rate=max_character_error_rate,
        max_latency_ms=max_latency_ms,
    )

    return {
        "schema_version": "v1",
        "experiment_name": "image_ocr_preprocessing_experiment",
        "file_name": source_path.name,
        "source_image": str(source_path),
        "baseline": baseline,
        "thresholds": {
            "min_exact_segment_recall": min_exact_segment_recall,
            "max_character_error_rate": max_character_error_rate,
            "max_latency_ms": max_latency_ms,
        },
        "variants": variant_results,
        "decision": decision,
        "boundary_notes": [
            "本实验只调用本地PaddleOCR，不调用DeepSeek，也不新增视觉理解、语音识别或视频处理能力。",
            "本实验只比较同一张图片在不同预处理方式下的OCR结果，不能证明PaddleOCR线上泛化质量。",
            "本实验产生的预处理图片是实验中间产物，用于复现本次OCR输入，不应混入正式业务输入目录。",
        ],
        "field_notes": {
            "variant_name": "预处理方案名称，用来区分原图、整图放大和左右分区放大。",
            "exact_segment_recall": "完整识别的必选业务文字块占比，用来判断预处理是否减少漏识别。",
            "character_error_rate": "分段编辑距离除以人工正确字符总数，用来判断预处理是否减少错字和漏字。",
            "ocr_latency_ms": "当前变体所有OCR调用耗时合计，用来判断预处理是否带来不可接受的性能成本。",
            "decision": "基于质量和延迟阈值给出的实验结论，用来决定是否继续优化该预处理方向。",
        },
    }


def write_preprocessing_experiment_report(
    *,
    image_path: str | Path,
    gold_path: str | Path,
    baseline_report_path: str | Path,
    batch_summary_path: str | Path,
    output_json_path: str | Path,
    output_markdown_path: str | Path,
    ocr_client: OcrClient = paddleocr_client,
) -> tuple[Path, Path]:
    """运行实验并写出JSON与Markdown报告。"""

    json_path = Path(output_json_path)
    markdown_path = Path(output_markdown_path)
    variants_dir = json_path.with_suffix("")
    report = run_preprocessing_experiment(
        image_path=image_path,
        gold_path=gold_path,
        baseline_report_path=baseline_report_path,
        batch_summary_path=batch_summary_path,
        variants_dir=variants_dir,
        ocr_client=ocr_client,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def run_latency_profile(
    *,
    image_path: str | Path,
    repeat_count: int = 2,
    engine_factory: EngineFactory = _create_paddleocr_engine,
    image_decoder: ImageDecoder = _decode_image_for_paddleocr,
    prediction_parser: PredictionParser = _parse_paddleocr_prediction,
) -> dict[str, Any]:
    """拆分本地PaddleOCR延迟来源。"""

    if repeat_count < 1:
        raise ValueError("repeat_count 必须至少为1。")

    source_path = Path(image_path)
    if not source_path.is_file():
        raise ValueError(f"图片文件不存在：{source_path}")

    total_started_at = perf_counter()
    engine_started_at = perf_counter()
    engine = engine_factory()
    engine_create_ms = int(round((perf_counter() - engine_started_at) * 1000))

    attempts: list[dict[str, Any]] = []
    for attempt_index in range(1, repeat_count + 1):
        decode_started_at = perf_counter()
        decoded_image = image_decoder(source_path)
        decode_ms = int(round((perf_counter() - decode_started_at) * 1000))

        predict_started_at = perf_counter()
        prediction = engine.predict(decoded_image)
        predict_ms = int(round((perf_counter() - predict_started_at) * 1000))

        parse_started_at = perf_counter()
        parsed_result = prediction_parser(prediction)
        parse_ms = int(round((perf_counter() - parse_started_at) * 1000))

        ocr_text = parsed_result.get("ocr_text") or ""
        attempts.append(
            {
                "attempt_index": attempt_index,
                "decode_ms": decode_ms,
                "predict_ms": predict_ms,
                "parse_ms": parse_ms,
                "attempt_total_ms": decode_ms + predict_ms + parse_ms,
                "ocr_text_char_count": len(ocr_text),
            }
        )

    total_ms = int(round((perf_counter() - total_started_at) * 1000))
    decision = _build_latency_profile_decision(
        engine_create_ms=engine_create_ms,
        attempts=attempts,
    )

    return {
        "schema_version": "v1",
        "profile_name": "image_ocr_latency_profile",
        "file_name": source_path.name,
        "source_image": str(source_path),
        "repeat_count": repeat_count,
        "engine_create_ms": engine_create_ms,
        "attempts": attempts,
        "total_profile_ms": total_ms,
        "decision": decision,
        "boundary_notes": [
            "本报告只拆分本地PaddleOCR延迟来源，不调用DeepSeek，也不新增视觉理解、语音识别或视频处理能力。",
            "engine_create_ms 包含PaddleOCR引擎创建、模型加载和可能的模型源检查/缓存读取，不能直接等同于线上服务冷启动。",
            "predict_ms 是同一引擎下单次图片推理耗时，更接近热启动后的单图OCR成本。",
        ],
        "field_notes": {
            "engine_create_ms": "本地OCR引擎创建耗时，用于观察模型加载和初始化开销。",
            "decode_ms": "图片读取和解码耗时，用于判断是否慢在文件读取或图像解码。",
            "predict_ms": "OCR模型推理耗时，用于判断核心瓶颈是否在模型识别。",
            "parse_ms": "PaddleOCR结果解析耗时，用于判断后处理是否形成明显开销。",
            "attempt_total_ms": "单次图片解码、推理和解析的合计耗时，不包含引擎创建。",
        },
    }


def write_latency_profile_report(
    *,
    image_path: str | Path,
    output_json_path: str | Path,
    output_markdown_path: str | Path,
    repeat_count: int = 2,
    engine_factory: EngineFactory = _create_paddleocr_engine,
    image_decoder: ImageDecoder = _decode_image_for_paddleocr,
    prediction_parser: PredictionParser = _parse_paddleocr_prediction,
) -> tuple[Path, Path]:
    """运行延迟拆分并写出JSON与Markdown报告。"""

    report = run_latency_profile(
        image_path=image_path,
        repeat_count=repeat_count,
        engine_factory=engine_factory,
        image_decoder=image_decoder,
        prediction_parser=prediction_parser,
    )
    json_path = Path(output_json_path)
    markdown_path = Path(output_markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_latency_profile_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _run_variant_ocr(
    *,
    variant: dict[str, Any],
    file_name: str,
    gold_rows: list[dict[str, str]],
    ocr_client: OcrClient,
) -> dict[str, Any]:
    """对一个预处理变体运行OCR并计算评估指标。"""

    started_at = perf_counter()
    ocr_text_parts: list[str] = []
    try:
        for image_path in variant["image_paths"]:
            result = ocr_client(image_path)
            if result.get("ocr_text"):
                ocr_text_parts.append(str(result["ocr_text"]))
        latency_ms = int(round((perf_counter() - started_at) * 1000))
        ocr_text = "\n".join(ocr_text_parts)
        evaluation = evaluate_image_ocr(
            file_name=file_name,
            ocr_text=ocr_text,
            gold_rows=gold_rows,
        )
        return {
            **variant,
            "status": "success",
            "ocr_latency_ms": latency_ms,
            "ocr_text_char_count": len(ocr_text),
            "required_segment_count": evaluation["required_segment_count"],
            "exact_segment_count": evaluation["exact_segment_count"],
            "exact_segment_recall": evaluation["exact_segment_recall"],
            "character_error_rate": evaluation["character_error_rate"],
            "total_edit_distance": evaluation["total_edit_distance"],
            "error_message": None,
        }
    except Exception as exc:
        latency_ms = int(round((perf_counter() - started_at) * 1000))
        return {
            **variant,
            "status": "failed",
            "ocr_latency_ms": latency_ms,
            "ocr_text_char_count": 0,
            "required_segment_count": None,
            "exact_segment_count": None,
            "exact_segment_recall": None,
            "character_error_rate": None,
            "total_edit_distance": None,
            "error_message": str(exc),
        }


def _build_baseline_summary(
    baseline_report: dict[str, Any],
    batch_summary: dict[str, Any],
) -> dict[str, Any]:
    """从已有单图评估和批次汇总中提取原图基线指标。"""

    file_name = baseline_report.get("file_name")
    latency_ms = None
    for file_metric in batch_summary.get("file_metrics", []):
        if isinstance(file_metric, dict) and file_metric.get("file_name") == file_name:
            latency_ms = file_metric.get("ocr_latency_ms")
            break
    return {
        "variant_name": "baseline_original",
        "file_name": file_name,
        "required_segment_count": baseline_report.get("required_segment_count"),
        "exact_segment_count": baseline_report.get("exact_segment_count"),
        "exact_segment_recall": baseline_report.get("exact_segment_recall"),
        "character_error_rate": baseline_report.get("character_error_rate"),
        "total_edit_distance": baseline_report.get("total_edit_distance"),
        "ocr_latency_ms": latency_ms,
    }


def _build_experiment_decision(
    *,
    baseline: dict[str, Any],
    variant_results: list[dict[str, Any]],
    min_exact_segment_recall: float,
    max_character_error_rate: float,
    max_latency_ms: int,
) -> dict[str, Any]:
    """根据基线和变体结果生成实验结论。"""

    successful_variants = [
        variant
        for variant in variant_results
        if variant.get("status") == "success"
        and isinstance(variant.get("exact_segment_recall"), (int, float))
        and isinstance(variant.get("character_error_rate"), (int, float))
    ]
    if not successful_variants:
        return {
            "status": "failed",
            "best_variant": None,
            "reasons": ["所有预处理变体均未成功生成可评估OCR结果。"],
            "next_action": "先排查本地PaddleOCR或预处理图片生成问题。",
        }

    best_variant = sorted(
        successful_variants,
        key=lambda item: (
            -float(item["exact_segment_recall"]),
            float(item["character_error_rate"]),
            int(item.get("ocr_latency_ms") or 0),
        ),
    )[0]
    baseline_recall = float(baseline.get("exact_segment_recall") or 0)
    baseline_error_rate = float(baseline.get("character_error_rate") or 1)
    best_recall = float(best_variant["exact_segment_recall"])
    best_error_rate = float(best_variant["character_error_rate"])
    best_latency = int(best_variant.get("ocr_latency_ms") or 0)

    reasons: list[str] = []
    if best_recall > baseline_recall:
        reasons.append(f"最佳变体完整段落召回率从{baseline_recall:.2%}提升到{best_recall:.2%}。")
    else:
        reasons.append(f"最佳变体完整段落召回率未超过原图基线{baseline_recall:.2%}。")
    if best_error_rate < baseline_error_rate:
        reasons.append(f"最佳变体字符错误率从{baseline_error_rate:.2%}下降到{best_error_rate:.2%}。")
    else:
        reasons.append(f"最佳变体字符错误率未低于原图基线{baseline_error_rate:.2%}。")

    quality_passed = best_recall >= min_exact_segment_recall and best_error_rate <= max_character_error_rate
    latency_passed = best_latency <= max_latency_ms
    if quality_passed and latency_passed:
        status = "passed"
        next_action = "可以把该预处理方案作为图片OCR候选方案继续小样本验证。"
    elif quality_passed:
        status = "quality_passed_latency_failed"
        reasons.append(f"最佳变体OCR延迟为{best_latency}ms，高于{max_latency_ms}ms目标。")
        next_action = "先拆分冷启动、模型加载和单图推理延迟，再判断是否接受该方案。"
    elif best_recall > baseline_recall or best_error_rate < baseline_error_rate:
        status = "improved_but_not_passed"
        next_action = "预处理方向有一定价值，但不能直接通过闸门；下一步只做一个更窄的分区或裁剪实验。"
    else:
        status = "not_improved"
        next_action = "当前预处理方向不应继续扩大，优先分析OCR模型对小字号结构图的能力边界。"

    return {
        "status": status,
        "best_variant": best_variant["variant_name"],
        "quality_passed": quality_passed,
        "latency_passed": latency_passed,
        "reasons": reasons,
        "next_action": next_action,
    }


def _build_latency_profile_decision(
    *,
    engine_create_ms: int,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    """根据延迟拆分结果生成结论。"""

    if not attempts:
        return {
            "main_bottleneck": "unknown",
            "reasons": ["没有成功记录任何OCR尝试。"],
            "next_action": "先排查本地PaddleOCR调用是否成功。",
        }

    first_attempt = attempts[0]
    warm_attempts = attempts[1:] or attempts
    avg_warm_predict_ms = sum(int(item["predict_ms"]) for item in warm_attempts) / len(warm_attempts)
    avg_warm_attempt_total_ms = sum(int(item["attempt_total_ms"]) for item in warm_attempts) / len(warm_attempts)
    first_predict_ms = int(first_attempt["predict_ms"])

    phase_costs = {
        "engine_create": engine_create_ms,
        "first_predict": first_predict_ms,
        "avg_warm_predict": int(round(avg_warm_predict_ms)),
        "avg_warm_attempt_total": int(round(avg_warm_attempt_total_ms)),
    }
    main_bottleneck = max(phase_costs, key=phase_costs.get)
    reasons = [
        f"引擎创建耗时为{engine_create_ms}ms。",
        f"首次图片推理耗时为{first_predict_ms}ms。",
        f"热启动平均图片推理耗时为{avg_warm_predict_ms:.0f}ms。",
        f"热启动平均单图总耗时为{avg_warm_attempt_total_ms:.0f}ms。",
    ]
    if avg_warm_attempt_total_ms > 2000:
        reasons.append("即使不计引擎创建，热启动单图耗时仍高于2秒目标。")
        next_action = "下一轮不要先扩展新功能，应评估是否接受本地CPU延迟，或改用更轻量/服务化OCR方案。"
    else:
        next_action = "热启动单图耗时接近目标，可继续观察批量吞吐和并发策略。"

    return {
        "main_bottleneck": main_bottleneck,
        "phase_costs": phase_costs,
        "reasons": reasons,
        "next_action": next_action,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    """生成面向人工复核的Markdown报告。"""

    baseline = report["baseline"]
    decision = report["decision"]
    variant_rows = "\n".join(
        "| {variant_name} | {status} | {exact_segment_recall} | {character_error_rate} | {ocr_latency_ms} | {total_edit_distance} |".format(
            variant_name=variant.get("variant_name"),
            status=variant.get("status"),
            exact_segment_recall=_format_rate(variant.get("exact_segment_recall")),
            character_error_rate=_format_rate(variant.get("character_error_rate")),
            ocr_latency_ms=_format_ms(variant.get("ocr_latency_ms")),
            total_edit_distance=variant.get("total_edit_distance"),
        )
        for variant in report["variants"]
    )
    reasons = "\n".join(f"- {reason}" for reason in decision["reasons"])
    generated_images = "\n".join(
        f"- `{path}`"
        for variant in report["variants"]
        for path in variant.get("image_paths", [])
    )

    return f"""# 图片 OCR 预处理实验报告：{report['file_name']}

## 一、实验目标

只围绕 `img_9.jpg` 验证两种最小预处理是否能改善小字号结构图 OCR：整图放大2倍、左右分区后各自放大2倍。

## 二、原图基线

- 完整段落召回率：{_format_rate(baseline.get('exact_segment_recall'))}
- 字符错误率：{_format_rate(baseline.get('character_error_rate'))}
- OCR延迟：{_format_ms(baseline.get('ocr_latency_ms'))}
- 编辑距离合计：{baseline.get('total_edit_distance')}

## 三、实验结果

| 变体 | 状态 | 完整段落召回率 | 字符错误率 | OCR延迟 | 编辑距离合计 |
|---|---|---:|---:|---:|---:|
{variant_rows}

## 四、实验结论

结论：`{decision['status']}`

最佳变体：`{decision['best_variant']}`

{reasons}

下一步：{decision['next_action']}

## 五、生成的预处理图片

{generated_images}

## 六、边界说明

- 本实验只调用本地PaddleOCR，不调用DeepSeek，也不新增视觉理解、语音识别或视频处理能力。
- 本实验只比较同一张图片在不同预处理方式下的OCR结果，不能证明PaddleOCR线上泛化质量。
- 本实验产生的预处理图片是实验中间产物，用于复现本次OCR输入，不应混入正式业务输入目录。
"""


def _render_latency_profile_markdown(report: dict[str, Any]) -> str:
    """生成延迟拆分Markdown报告。"""

    decision = report["decision"]
    attempt_rows = "\n".join(
        "| {attempt_index} | {decode_ms}ms | {predict_ms}ms | {parse_ms}ms | {attempt_total_ms}ms | {ocr_text_char_count} |".format(
            **attempt
        )
        for attempt in report["attempts"]
    )
    reasons = "\n".join(f"- {reason}" for reason in decision["reasons"])

    return f"""# 图片 OCR 延迟拆分报告：{report['file_name']}

## 一、实验目标

拆分本地 PaddleOCR 在 `img_9.jpg` 上的耗时来源，区分引擎创建、图片解码、模型推理和结果解析。

## 二、总体结果

- 引擎创建耗时：{report['engine_create_ms']}ms
- 重复推理次数：{report['repeat_count']}
- 全部 profile 总耗时：{report['total_profile_ms']}ms
- 主要瓶颈：`{decision['main_bottleneck']}`

## 三、逐次调用拆分

| 次数 | 图片解码 | 模型推理 | 结果解析 | 单次合计 | OCR文字数 |
|---:|---:|---:|---:|---:|---:|
{attempt_rows}

## 四、结论

{reasons}

下一步：{decision['next_action']}

## 五、边界说明

- 本报告只拆分本地PaddleOCR延迟来源，不调用DeepSeek，也不新增视觉理解、语音识别或视频处理能力。
- `engine_create_ms` 包含PaddleOCR引擎创建、模型加载和可能的模型源检查/缓存读取，不能直接等同于线上服务冷启动。
- `predict_ms` 是同一引擎下单次图片推理耗时，更接近热启动后的单图OCR成本。
"""


def _format_rate(value: Any) -> str:
    """把比例格式化为百分比，缺失时明确标注。"""

    return f"{value:.2%}" if isinstance(value, (int, float)) else "当前数据未提供"


def _format_ms(value: Any) -> str:
    """把毫秒格式化为文本，缺失时明确标注。"""

    return f"{int(value)}ms" if isinstance(value, (int, float)) else "当前数据未提供"


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = argv if argv is not None else sys.argv[1:]
    if len(args) == 7 and args[0] == "run":
        json_path, markdown_path = write_preprocessing_experiment_report(
            image_path=args[1],
            gold_path=args[2],
            baseline_report_path=args[3],
            batch_summary_path=args[4],
            output_json_path=args[5],
            output_markdown_path=args[6],
        )
        print(
            json.dumps(
                {"json_report": str(json_path), "markdown_report": str(markdown_path)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if len(args) == 5 and args[0] == "profile":
        json_path, markdown_path = write_latency_profile_report(
            image_path=args[1],
            output_json_path=args[2],
            output_markdown_path=args[3],
            repeat_count=int(args[4]),
        )
        print(
            json.dumps(
                {"json_report": str(json_path), "markdown_report": str(markdown_path)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(
        "用法: python .\\src\\image_ocr_preprocessing_experiment.py run "
        "img_9.jpg image_ocr_gold.csv image_ocr_eval_img_9.json "
        "image_ocr_eval_summary.json experiment.json experiment.md\n"
        "或: python .\\src\\image_ocr_preprocessing_experiment.py profile "
        "img_9.jpg latency_profile.json latency_profile.md 2"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
