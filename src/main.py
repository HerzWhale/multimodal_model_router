"""多模态模型路由 MVP 的命令行入口。"""

from __future__ import annotations

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


def run_batch(
    *,
    settings_path: str | Path = PROJECT_ROOT / "config" / "settings.yaml",
    routing_rules_path: str | Path | None = None,
    model_prices_path: str | Path | None = None,
    batch_id: str | None = None,
    created_at: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """运行一次最小 mock 批处理。"""

    settings_file = Path(settings_path)
    project_root = settings_file.resolve().parents[1]
    settings = load_settings(settings_file)
    routing_rules = load_routing_rules(routing_rules_path or project_root / "config" / "routing_rules.yaml")
    model_prices = load_model_prices(model_prices_path or project_root / "config" / "model_prices.yaml")

    current_batch_id = batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_created_at = created_at or _now_iso()
    input_dir = _resolve_path(project_root, settings["input_dir"])
    output_dir = _resolve_path(project_root, settings["output_dir"])
    text_analysis_backend = settings.get("text_analysis_backend", "mock")
    deepseek_api_key_env = settings.get("deepseek_api_key_env", "DEEPSEEK_API_KEY")
    deepseek_api_key = os.environ.get(deepseek_api_key_env) if text_analysis_backend == "deepseek" else None

    file_manifest = build_file_manifest(input_dir, current_batch_id, created_at=batch_created_at)
    results: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for file_record in file_manifest:
        pipeline_output = run_file_pipeline(
            file_record,
            routing_rules,
            model_prices,
            text_analysis_backend=text_analysis_backend,
            deepseek_api_key=deepseek_api_key,
            deepseek_base_url=settings.get("deepseek_base_url", "https://api.deepseek.com"),
            deepseek_model_name=settings.get("deepseek_model_name", "deepseek-v4-flash"),
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


if __name__ == "__main__":
    summary = run_batch()
    print(summary)
