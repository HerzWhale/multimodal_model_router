"""受保护的离线回归检查入口。

这个脚本只验证现有核心链路是否仍能跑通，不调用 DeepSeek，不运行真实 PaddleOCR，
也不向项目正式 output 目录写入新批次。所有运行产物都放在临时目录中。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from main import run_batch
from routing_preflight import build_preflight_from_files, write_preflight_reports


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_GENERATED_AT = "2026-07-30T10:00:00+08:00"


def run_offline_regression_check(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_unit_tests: bool = True,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """运行受保护离线回归检查，并返回机器可读结果。"""

    root = Path(project_root)
    current_generated_at = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    steps: list[dict[str, Any]] = []

    boundary = {
        "calls_deepseek_api": False,
        "runs_real_paddleocr": False,
        "uses_cloud_ocr": False,
        "writes_official_output": False,
        "uses_temporary_output": True,
    }

    if run_unit_tests:
        steps.append(_run_unit_tests(root))

    with tempfile.TemporaryDirectory(prefix="mmr_offline_regression_") as tmp_dir:
        temp_root = Path(tmp_dir)
        sample_input_dir = _create_sample_input_dir(temp_root)
        steps.append(_run_mock_batch_smoke(root, temp_root, sample_input_dir, current_generated_at))
        steps.append(_run_routing_preflight_smoke(root, temp_root, sample_input_dir, current_generated_at))

    overall_status = "pass" if all(step["status"] == "pass" for step in steps) else "fail"
    return {
        "schema_version": "v1",
        "check_name": "offline_regression_check",
        "generated_at": current_generated_at,
        "overall_status": overall_status,
        "boundary": boundary,
        "steps": steps,
        "field_notes": _field_notes(),
    }


def _create_sample_input_dir(temp_root: Path) -> Path:
    """创建三类媒体的最小 mock 输入样本。"""

    input_dir = temp_root / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "regression_text.txt").write_text(
        "这是一次离线回归检查文本，用于验证文本分流和mock文本分析链路。",
        encoding="utf-8",
    )
    (input_dir / "regression_image.png").write_bytes(b"fake image bytes")
    (input_dir / "regression_video.mp4").write_bytes(b"fake video bytes")
    return input_dir


def _write_settings(temp_root: Path) -> Path:
    """写入只使用 mock 后端的临时配置。"""

    config_dir = temp_root / "config"
    config_dir.mkdir(parents=True)
    settings_path = config_dir / "settings.yaml"
    settings_path.write_text(
        "\n".join(
            [
                "input_dir: input",
                "output_dir: output",
                "ocr_backend: mock",
                "text_analysis_backend: mock",
                "deepseek_api_key_env: OFFLINE_REGRESSION_DEEPSEEK_API_KEY",
                "default_budget_limit_cny: 50",
                "target_output_format: jsonl",
                "allow_partial_success: true",
            ]
        ),
        encoding="utf-8",
    )
    return settings_path


def _run_unit_tests(project_root: Path) -> dict[str, Any]:
    """运行完整离线单元测试。"""

    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    completed = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "step_name": "unit_tests",
        "status": "pass" if completed.returncode == 0 else "fail",
        "command": " ".join(command),
        "return_code": completed.returncode,
        "summary": _last_non_empty_line(completed.stderr or completed.stdout),
    }


def _run_mock_batch_smoke(
    project_root: Path,
    temp_root: Path,
    sample_input_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    """在临时目录中跑一次三文件 mock 批处理。"""

    settings_path = _write_settings(temp_root)
    summary = run_batch(
        settings_path=settings_path,
        routing_rules_path=project_root / "config" / "routing_rules.yaml",
        model_prices_path=project_root / "config" / "model_prices.yaml",
        input_dir_override=sample_input_dir,
        ocr_backend_override="mock",
        text_analysis_backend_override="mock",
        allow_live_api=False,
        batch_id="offline_regression_mock_batch",
        created_at=generated_at,
        generated_at=generated_at,
    )
    batch_dir = Path(summary["batch_dir"])
    expected_files = [
        "batch_metadata.json",
        "results.jsonl",
        "results_readable.md",
        "model_calls.jsonl",
        "errors.jsonl",
        "batch_report.json",
    ]
    missing_files = [file_name for file_name in expected_files if not (batch_dir / file_name).exists()]
    status = "pass" if summary["total_files"] == 3 and summary["total_errors"] == 0 and not missing_files else "fail"
    return {
        "step_name": "mock_batch_smoke",
        "status": status,
        "batch_id": summary["batch_id"],
        "total_files": summary["total_files"],
        "total_model_calls": summary["total_model_calls"],
        "total_errors": summary["total_errors"],
        "missing_output_files": missing_files,
        "output_scope": "temporary_directory",
    }


def _run_routing_preflight_smoke(
    project_root: Path,
    temp_root: Path,
    sample_input_dir: Path,
    generated_at: str,
) -> dict[str, Any]:
    """在临时目录中生成一次 routing preflight 报告。"""

    model_calls_path = _write_sample_model_calls(temp_root, generated_at)
    report = build_preflight_from_files(
        routing_rules_path=project_root / "config" / "routing_rules.yaml",
        model_prices_path=project_root / "config" / "model_prices.yaml",
        policy_config_path=project_root / "config" / "routing_policy_config.yaml",
        policy_name="balanced",
        input_dir=sample_input_dir,
        expected_audio_seconds_per_video=60,
        historical_model_calls_paths=[model_calls_path],
        policy_overrides={"budget_limit_cny": 50.0},
        ocr_backend="paddleocr",
        text_analysis_backend="deepseek",
        generated_at=generated_at,
    )
    output_paths = write_preflight_reports(temp_root / "preflight_output", report)
    checks = {check["constraint_name"]: check["status"] for check in report["constraint_checks"]}
    status = "pass" if Path(output_paths["json"]).exists() and Path(output_paths["markdown"]).exists() else "fail"
    return {
        "step_name": "routing_preflight_smoke",
        "status": status,
        "preflight_status": report["preflight_status"],
        "total_files": report["workload_profile"]["total_files"],
        "estimated_total_cost_cny": report["route_summary"]["estimated_total_cost_cny"],
        "max_expected_p95_latency_ms": report["route_summary"]["max_expected_p95_latency_ms"],
        "constraint_statuses": checks,
        "output_scope": "temporary_directory",
    }


def _write_sample_model_calls(temp_root: Path, generated_at: str) -> Path:
    """写入预检查用的最小历史模型调用记录。"""

    path = temp_root / "historical_model_calls.jsonl"
    records = [
        {
            "call_id": "offline_call_ocr_0001",
            "batch_id": "offline_regression_history",
            "file_id": "file_image",
            "task_type": "ocr",
            "provider": "paddlepaddle",
            "model_name": "PP-OCRv5_mobile",
            "input_units": [{"unit_type": "image_count", "quantity": 1}],
            "output_units": [{"unit_type": "image_count", "quantity": 1}],
            "cost_cny": 0.0,
            "latency_ms": 5000,
            "started_at": generated_at,
            "status": "success",
            "error_message": None,
        },
        {
            "call_id": "offline_call_text_0001",
            "batch_id": "offline_regression_history",
            "file_id": "file_text",
            "task_type": "text_analysis",
            "provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "input_units": [{"unit_type": "input_tokens", "quantity": 100}],
            "output_units": [{"unit_type": "output_tokens", "quantity": 100}],
            "cost_cny": 0.0003,
            "latency_ms": 100,
            "started_at": generated_at,
            "status": "success",
            "error_message": None,
        },
    ]
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")
    return path


def _last_non_empty_line(text: str) -> str:
    """提取命令输出中最后一行非空文本。"""

    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _field_notes() -> dict[str, str]:
    """返回本报告中的关键字段说明。"""

    return {
        "overall_status": "本次离线回归检查总状态；只有全部步骤通过才为pass。",
        "boundary": "安全边界说明，用来确认本入口不会触发真实API或真实PaddleOCR。",
        "steps": "逐项检查结果列表，用来定位是哪条核心链路失败。",
        "mock_batch_smoke": "临时目录中的三文件mock批处理，用来验证主流水线、输出文件和错误统计。",
        "routing_preflight_smoke": "临时目录中的路由预检查，用来验证预算、延迟和真实覆盖率检查仍可生成报告。",
        "preflight_status": "路由预检查自己的业务判断状态；fail可能表示策略阻塞，不等于脚本执行失败。",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行受保护的离线回归检查。")
    parser.add_argument(
        "--skip-unit-tests",
        action="store_true",
        help="跳过完整单元测试，只验证mock批处理和routing preflight核心路径。",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="项目根目录，默认自动识别当前仓库。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = _parse_args(argv)
    report = run_offline_regression_check(
        project_root=args.project_root,
        run_unit_tests=not args.skip_unit_tests,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
