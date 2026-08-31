"""受保护的离线回归检查入口。

这个脚本只验证现有核心链路是否仍能跑通，不调用 DeepSeek 或 Qwen-VL，不运行真实 PaddleOCR，
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
    check_batch_dir: str | Path | None = None,
    require_no_mock: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """运行受保护离线回归检查，并返回机器可读结果。"""

    root = Path(project_root)
    current_generated_at = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    steps: list[dict[str, Any]] = []

    boundary = {
        "calls_deepseek_api": False,
        "calls_qwen_vl_api": False,
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

    if check_batch_dir:
        steps.append(check_batch_completeness(check_batch_dir, require_no_mock=require_no_mock))

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


def check_batch_completeness(batch_dir: str | Path, *, require_no_mock: bool = False) -> dict[str, Any]:
    """检查一个已生成批次的输出、证据链和模型调用是否闭环。"""

    path = Path(batch_dir)
    required_files = ["batch_metadata.json", "batch_report.json", "results.jsonl", "model_calls.jsonl"]
    missing_files = [file_name for file_name in required_files if not (path / file_name).exists()]
    if missing_files:
        return {
            "step_name": "batch_completeness_check",
            "status": "fail",
            "batch_dir": str(path),
            "issues": [f"缺少必要输出文件: {file_name}" for file_name in missing_files],
        }

    issues: list[str] = []
    try:
        metadata = json.loads((path / "batch_metadata.json").read_text(encoding="utf-8"))
        batch_report = json.loads((path / "batch_report.json").read_text(encoding="utf-8"))
        results = _read_json_objects(path / "results.jsonl")
        model_calls = _read_json_objects(path / "model_calls.jsonl")
    except Exception as exc:
        return {
            "step_name": "batch_completeness_check",
            "status": "fail",
            "batch_dir": str(path),
            "issues": [f"读取或解析批次输出失败: {exc}"],
        }

    file_stats = batch_report.get("file_stats") if isinstance(batch_report.get("file_stats"), dict) else {}
    expected_total_files = file_stats.get("total_files")
    if isinstance(expected_total_files, int) and expected_total_files != len(results):
        issues.append(f"results.jsonl 文件数 {len(results)} 与 batch_report total_files {expected_total_files} 不一致")

    call_by_id = {call.get("call_id"): call for call in model_calls if isinstance(call.get("call_id"), str)}
    calls_by_file: dict[str, list[dict[str, Any]]] = {}
    for call in model_calls:
        file_id = call.get("file_id")
        if isinstance(file_id, str):
            calls_by_file.setdefault(file_id, []).append(call)
        _check_model_call(call, issues)

    seen_file_ids: set[str] = set()
    selected_backends = metadata.get("selected_backends") if isinstance(metadata.get("selected_backends"), dict) else {}
    selected_pipelines = metadata.get("selected_pipelines") if isinstance(metadata.get("selected_pipelines"), dict) else {}
    for result in results:
        file_id = result.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            issues.append("存在缺少 file_id 的文件级结果")
            continue
        if file_id in seen_file_ids:
            issues.append(f"{file_id} 重复出现在 results.jsonl 中")
        seen_file_ids.add(file_id)
        media_type = str(result.get("media_type"))
        media_backends = selected_pipelines.get(media_type)
        if not isinstance(media_backends, dict):
            media_backends = selected_backends
        _check_result_record(result, call_by_id, calls_by_file.get(file_id, []), media_backends, issues)

    cost_stats = batch_report.get("cost_stats") if isinstance(batch_report.get("cost_stats"), dict) else {}
    latency_stats = batch_report.get("latency_stats") if isinstance(batch_report.get("latency_stats"), dict) else {}
    if not isinstance(cost_stats.get("total_cost_cny"), (int, float)):
        issues.append("batch_report 缺少 total_cost_cny 成本汇总")
    if not isinstance(latency_stats.get("p95_model_latency_ms"), (int, float)):
        issues.append("batch_report 缺少 p95_model_latency_ms 延迟汇总")

    runtime_summary = metadata.get("backend_runtime_summary")
    contains_mock = None
    if isinstance(runtime_summary, dict):
        contains_mock = runtime_summary.get("contains_mock")
    if require_no_mock and contains_mock is True:
        issues.append("当前批次包含 mock 调用，不能作为全真实链路证据")

    return {
        "step_name": "batch_completeness_check",
        "status": "pass" if not issues else "fail",
        "batch_dir": str(path),
        "batch_id": metadata.get("batch_id"),
        "total_files": len(results),
        "total_model_calls": len(model_calls),
        "contains_mock": contains_mock,
        "issues": issues,
    }


def _check_model_call(call: dict[str, Any], issues: list[str]) -> None:
    """检查单条模型调用记录是否能支撑追踪和成本延迟复盘。"""

    call_id = call.get("call_id") or "unknown_call"
    for field_name in ["call_id", "file_id", "task_type", "provider", "model_name", "status"]:
        if not call.get(field_name):
            issues.append(f"{call_id} 缺少模型调用字段 {field_name}")
    if not isinstance(call.get("cost_cny"), (int, float)):
        issues.append(f"{call_id} 缺少 cost_cny 成本记录")
    if not isinstance(call.get("latency_ms"), (int, float)):
        issues.append(f"{call_id} 缺少 latency_ms 延迟记录")


def _check_result_record(
    result: dict[str, Any],
    call_by_id: dict[str, dict[str, Any]],
    file_calls: list[dict[str, Any]],
    selected_backends: dict[str, Any],
    issues: list[str],
) -> None:
    """检查单个文件级结果是否有统一输出、证据链和必要模型调用。"""

    file_id = str(result.get("file_id"))
    for field_name in ["file_name", "media_type", "processing_status", "call_ids", "topic", "tags", "summary", "evidence_used", "missing_evidence"]:
        if field_name not in result:
            issues.append(f"{file_id} 缺少文件结果字段 {field_name}")
    if not isinstance(result.get("processing_cost_cny"), (int, float)):
        issues.append(f"{file_id} 缺少 processing_cost_cny 文件成本")
    if not isinstance(result.get("processing_time_ms"), (int, float)):
        issues.append(f"{file_id} 缺少 processing_time_ms 文件耗时")

    call_ids = result.get("call_ids")
    if not isinstance(call_ids, list) or not call_ids:
        issues.append(f"{file_id} 没有关联任何 call_ids")
    else:
        missing_call_ids = [call_id for call_id in call_ids if call_id not in call_by_id]
        if missing_call_ids:
            issues.append(f"{file_id} 引用了不存在的模型调用: {', '.join(map(str, missing_call_ids))}")

    if result.get("processing_status") == "success":
        if result.get("missing_evidence"):
            issues.append(f"{file_id} 状态为 success 但仍存在 missing_evidence")
        if not result.get("topic") or not result.get("tags") or not result.get("summary"):
            issues.append(f"{file_id} 状态为 success 但缺少 topic、tags 或 summary")

    media_type = result.get("media_type")
    expected_tasks = _expected_tasks_for_result(str(media_type), selected_backends)
    successful_tasks = {call.get("task_type") for call in file_calls if call.get("status") == "success"}
    for task_type in expected_tasks:
        if task_type not in successful_tasks:
            issues.append(f"{file_id} 缺少成功的 {task_type} 模型调用")

    if media_type == "video":
        _check_video_preprocessing(result, file_id, issues)


def _expected_tasks_for_result(media_type: str, selected_backends: dict[str, Any]) -> list[str]:
    """根据文件类型和批次后端选择，判断文件应出现哪些任务调用。"""

    tasks = ["text_analysis"]
    if media_type in {"image", "video"}:
        tasks.extend(["ocr", "visual_understanding"])
    if media_type == "video":
        speech_backend = selected_backends.get("speech_to_text_backend")
        if speech_backend not in {None, "none", "disabled"}:
            tasks.append("speech_to_text")
    return tasks


def _check_video_preprocessing(result: dict[str, Any], file_id: str, issues: list[str]) -> None:
    """检查视频是否至少完成关键帧和音频预处理记录。"""

    artifacts = result.get("preprocessing_artifacts")
    if not isinstance(artifacts, dict):
        issues.append(f"{file_id} 缺少视频 preprocessing_artifacts")
        return
    keyframe_count = artifacts.get("keyframe_count")
    if not isinstance(keyframe_count, int) or keyframe_count < 1:
        issues.append(f"{file_id} 没有有效关键帧")
    keyframe_metadata = artifacts.get("keyframe_metadata")
    if isinstance(keyframe_count, int) and isinstance(keyframe_metadata, list) and len(keyframe_metadata) != keyframe_count:
        issues.append(f"{file_id} keyframe_metadata 数量与 keyframe_count 不一致")
    if artifacts.get("audio_extraction_status") != "extracted":
        issues.append(f"{file_id} 音频未成功抽取")


def _read_json_objects(path: Path) -> list[dict[str, Any]]:
    """读取普通 JSONL 或多行缩进 JSONL。"""

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    decoder = json.JSONDecoder()
    index = 0
    records: list[dict[str, Any]] = []
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        record, next_index = decoder.raw_decode(text, index)
        if not isinstance(record, dict):
            raise ValueError(f"{path.name} 中存在非对象 JSON 记录")
        records.append(record)
        index = next_index
    return records


def _create_sample_input_dir(temp_root: Path) -> Path:
    """创建三类媒体的最小 mock 输入样本。"""

    input_dir = temp_root / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "regression_text.txt").write_text(
        "这是一次离线回归检查文本，用于验证文本分流和mock文本分析链路。",
        encoding="utf-8",
    )
    (input_dir / "regression_image.png").write_bytes(b"fake image bytes")
    _write_minimal_video(input_dir / "regression_video.avi")
    return input_dir


def _write_minimal_video(path: Path) -> None:
    """写入一个最小有效视频，避免真实视频预处理把假 mp4 误判成读取失败。"""

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            1.0,
            (16, 16),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV VideoWriter 未打开")
        writer.write(frame)
        writer.release()
    except Exception:
        path.write_bytes(b"fake video bytes")


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
                "vision_understanding_backend: mock",
                "text_analysis_backend: mock",
                "deepseek_api_key_env: OFFLINE_REGRESSION_DEEPSEEK_API_KEY",
                "qwen_vl_api_key_env: OFFLINE_REGRESSION_DASHSCOPE_API_KEY",
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
        vision_understanding_backend_override="mock",
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
    batch_report = json.loads((batch_dir / "batch_report.json").read_text(encoding="utf-8"))
    quality_flags_count = batch_report.get("error_quality_stats", {}).get("quality_flags_count", {})
    expected_video_v0_errors = int(quality_flags_count.get("video_audio_not_extracted") or 0)
    unexpected_errors = int(summary["total_errors"]) - expected_video_v0_errors
    status = "pass" if summary["total_files"] == 3 and unexpected_errors == 0 and not missing_files else "fail"
    return {
        "step_name": "mock_batch_smoke",
        "status": status,
        "batch_id": summary["batch_id"],
        "total_files": summary["total_files"],
        "total_model_calls": summary["total_model_calls"],
        "total_errors": summary["total_errors"],
        "expected_video_v0_errors": expected_video_v0_errors,
        "unexpected_errors": unexpected_errors,
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
        "boundary": "安全边界说明，用来确认本入口不会触发真实API、真实Qwen-VL或真实PaddleOCR。",
        "steps": "逐项检查结果列表，用来定位是哪条核心链路失败。",
        "mock_batch_smoke": "临时目录中的三文件mock批处理，用来验证主流水线、输出文件和错误统计。",
        "routing_preflight_smoke": "临时目录中的路由预检查，用来验证预算、延迟和真实覆盖率检查仍可生成报告。",
        "preflight_status": "路由预检查自己的业务判断状态；fail可能表示策略阻塞，不等于脚本执行失败。",
        "batch_completeness_check": "已有批次输出完整性验收；检查文件级结果、模型调用、成本延迟、视频预处理和证据链是否闭环。",
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
    parser.add_argument(
        "--check-batch-dir",
        default=None,
        help="检查一个已有批次目录的输出完整性；只读现有文件，不触发模型调用。",
    )
    parser.add_argument(
        "--require-no-mock",
        action="store_true",
        help="检查已有批次时要求不包含 mock 调用；用于判断批次能否作为全真实链路证据。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    args = _parse_args(argv)
    report = run_offline_regression_check(
        project_root=args.project_root,
        run_unit_tests=not args.skip_unit_tests,
        check_batch_dir=args.check_batch_dir,
        require_no_mock=args.require_no_mock,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
