"""多模态模型路由 MVP 的命令行入口。"""

from __future__ import annotations

import argparse
import importlib.util
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from cost_latency_tracker import load_model_prices
from file_loader import build_file_manifest
from model_router import load_routing_rules
from pipeline_runner import run_file_pipeline
from report_generator import generate_batch_report
from result_writer import (
    ensure_batch_output_dir,
    write_batch_metadata,
    write_errors,
    write_json,
    write_model_calls,
    write_results,
    write_results_readable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    """返回当前本地时间的 ISO 字符串。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_settings(settings_path: str | Path) -> dict[str, Any]:
    """读取运行配置文件。"""

    path = Path(settings_path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_path(project_root: Path, path_value: str | Path) -> Path:
    """把配置中的相对路径转换为项目内绝对路径。"""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def _ensure_paddleocr_runtime_available() -> None:
    """在生成批次输出前检查本地 PaddleOCR 的两个必要包。"""

    missing_packages = [
        package_name
        for package_name in ("paddle", "paddleocr")
        if importlib.util.find_spec(package_name) is None
    ]
    if missing_packages:
        missing_text = "、".join(missing_packages)
        raise RuntimeError(
            f"缺少 PaddleOCR 运行依赖：{missing_text}。请先按 README 完成本地安装。"
        )


def _filter_manifest_by_file_names(
    file_manifest: list[dict[str, Any]],
    include_file_names: list[str] | None,
) -> list[dict[str, Any]]:
    """按文件名保留本次明确指定的输入文件。"""

    if include_file_names is None:
        return file_manifest

    selected_file_names = {file_name.strip() for file_name in include_file_names if file_name.strip()}
    if not selected_file_names:
        raise ValueError("指定文件列表不能为空。")

    matched_file_names = {
        file_record["file_name"]
        for file_record in file_manifest
        if file_record["file_name"] in selected_file_names
    }
    missing_file_names = sorted(selected_file_names - matched_file_names)
    if missing_file_names:
        raise ValueError(f"指定文件未在输入目录中找到：{', '.join(missing_file_names)}")

    return [
        file_record
        for file_record in file_manifest
        if file_record["file_name"] in selected_file_names
    ]


def run_batch(
    *,
    settings_path: str | Path = PROJECT_ROOT / "config" / "settings.yaml",
    routing_rules_path: str | Path | None = None,
    model_prices_path: str | Path | None = None,
    input_dir_override: str | Path | None = None,
    ocr_backend_override: str | None = None,
    text_analysis_backend_override: str | None = None,
    allow_live_api: bool = False,
    deepseek_max_retries: int = 0,
    batch_id: str | None = None,
    created_at: str | None = None,
    generated_at: str | None = None,
    include_file_names: list[str] | None = None,
) -> dict[str, Any]:
    """运行一次批处理。"""

    settings_file = Path(settings_path)
    project_root = settings_file.resolve().parents[1]
    settings = load_settings(settings_file)
    routing_rules = load_routing_rules(routing_rules_path or project_root / "config" / "routing_rules.yaml")
    model_prices = load_model_prices(model_prices_path or project_root / "config" / "model_prices.yaml")

    current_batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_created_at = created_at or _now_iso()
    input_dir = _resolve_path(project_root, input_dir_override or settings["input_dir"])
    output_dir = _resolve_path(project_root, settings["output_dir"])
    ocr_backend = ocr_backend_override or settings.get("ocr_backend", "mock")
    text_analysis_backend = text_analysis_backend_override or settings.get("text_analysis_backend", "mock")
    if ocr_backend not in {"mock", "paddleocr"}:
        raise ValueError(f"不支持的 OCR 后端：{ocr_backend}")
    if text_analysis_backend not in {"mock", "deepseek"}:
        raise ValueError(f"不支持的文本分析后端：{text_analysis_backend}")
    if text_analysis_backend == "deepseek" and not allow_live_api:
        raise PermissionError(
            "本次运行将访问 DeepSeek API。请显式选择 DeepSeek 后端并指定 --allow-live-api。"
        )
    if deepseek_max_retries not in {0, 1}:
        raise ValueError("DeepSeek 最大重试次数只能是 0 或 1。")
    if deepseek_max_retries and text_analysis_backend != "deepseek":
        raise ValueError("只有显式选择 DeepSeek 后端时才能启用 API 重试。")

    deepseek_api_key_env = settings.get("deepseek_api_key_env", "DEEPSEEK_API_KEY")
    deepseek_api_key = os.environ.get(deepseek_api_key_env) if text_analysis_backend == "deepseek" else None
    if text_analysis_backend == "deepseek" and not deepseek_api_key:
        raise RuntimeError(f"未读取到环境变量 {deepseek_api_key_env}，已在发送网络请求前停止运行。")

    if ocr_backend == "paddleocr":
        _ensure_paddleocr_runtime_available()

    file_manifest = build_file_manifest(input_dir, current_batch_id, created_at=batch_created_at)
    file_manifest = _filter_manifest_by_file_names(file_manifest, include_file_names)
    results: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for file_record in file_manifest:
        pipeline_output = run_file_pipeline(
            file_record,
            routing_rules,
            model_prices,
            ocr_backend=ocr_backend,
            text_analysis_backend=text_analysis_backend,
            deepseek_api_key=deepseek_api_key,
            deepseek_base_url=settings.get("deepseek_base_url", "https://api.deepseek.com"),
            deepseek_model_name=settings.get("deepseek_model_name", "deepseek-v4-flash"),
            deepseek_max_retries=deepseek_max_retries,
        )
        results.append(pipeline_output["result"])
        model_calls.extend(pipeline_output["model_calls"])
        errors.extend(pipeline_output["errors"])

    batch_metadata = {
        "schema_version": "v1",
        "batch_id": current_batch_id,
        "created_by": settings.get("created_by", "local_user"),
        "team_name": settings.get("team_name", "content_ai_team"),
        "request_purpose": settings.get("request_purpose", "本地 mock 批处理验证"),
        "created_at": batch_created_at,
        "budget_limit_cny": settings["default_budget_limit_cny"],
        "allow_partial_success": settings["allow_partial_success"],
        "target_output_format": settings["target_output_format"],
    }

    write_batch_metadata(output_dir, current_batch_id, batch_metadata)
    write_results(output_dir, current_batch_id, results)
    write_results_readable(output_dir, current_batch_id, results)
    write_model_calls(output_dir, current_batch_id, model_calls)
    write_errors(output_dir, current_batch_id, errors)

    batch_report = generate_batch_report(
        batch_id=current_batch_id,
        results=results,
        model_calls=model_calls,
        errors=errors,
        budget_limit_cny=float(settings["default_budget_limit_cny"]),
        generated_at=generated_at,
    )
    batch_dir = ensure_batch_output_dir(output_dir, current_batch_id)
    write_json(batch_dir / "batch_report.json", batch_report)

    return {
        "batch_id": current_batch_id,
        "batch_dir": str(batch_dir),
        "total_files": len(results),
        "total_model_calls": len(model_calls),
        "total_errors": len(errors),
    }


def _parse_include_files_arg(value: str) -> list[str]:
    """解析逗号分隔的文件名列表。"""

    file_names = [file_name.strip() for file_name in value.split(",") if file_name.strip()]
    if not file_names:
        raise argparse.ArgumentTypeError("文件名列表不能为空。")
    return file_names


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行多模态批处理或受控评估输入批处理。")
    parser.add_argument(
        "--settings",
        default=str(PROJECT_ROOT / "config" / "settings.yaml"),
        help="配置文件路径，默认使用 config/settings.yaml。",
    )
    parser.add_argument(
        "--input-dir",
        help="显式指定本次输入目录；例如 evaluation/text_topic_small_set。",
    )
    parser.add_argument(
        "--include-files",
        type=_parse_include_files_arg,
        help="只处理指定文件名，多个文件用英文逗号分隔；例如 img_7.jpg,img_8.jpg,img_9.jpg。",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["mock", "paddleocr"],
        help="显式指定图片 OCR 后端；PaddleOCR 在本地运行，不需要 API 密钥。",
    )
    parser.add_argument(
        "--text-analysis-backend",
        choices=["mock", "deepseek"],
        help="显式指定文本分析后端；评估流程离线验证时建议使用 mock。",
    )
    parser.add_argument(
        "--batch-id",
        help="显式指定批次 ID，便于复现实验输出。",
    )
    parser.add_argument(
        "--allow-live-api",
        action="store_true",
        help="明确允许本次运行访问 DeepSeek API；必须同时显式选择 DeepSeek 后端。",
    )
    parser.add_argument(
        "--max-api-retries",
        type=int,
        choices=[0, 1],
        default=0,
        help="DeepSeek 可重试错误的最大重试次数；默认 0，显式设置为 1 才允许重试一次。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = _parse_args(argv)
    if args.allow_live_api and args.text_analysis_backend != "deepseek":
        print("错误：--allow-live-api 必须与 --text-analysis-backend deepseek 同时使用。")
        return 2
    if args.max_api_retries and args.text_analysis_backend != "deepseek":
        print("错误：--max-api-retries 只能与 --text-analysis-backend deepseek 同时使用。")
        return 2

    try:
        summary = run_batch(
            settings_path=args.settings,
            input_dir_override=args.input_dir,
            ocr_backend_override=args.ocr_backend or ("mock" if args.allow_live_api else None),
            text_analysis_backend_override=args.text_analysis_backend
            or ("mock" if args.allow_live_api or args.ocr_backend == "paddleocr" else None),
            allow_live_api=args.allow_live_api,
            deepseek_max_retries=args.max_api_retries,
            batch_id=args.batch_id,
            include_file_names=args.include_files,
        )
    except (PermissionError, RuntimeError, ValueError) as error:
        print(f"错误：{error}")
        return 2

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
