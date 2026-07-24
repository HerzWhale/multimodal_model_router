"""main 的测试。"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from main import main as cli_main
from main import run_batch


def _read_json_objects(file_path: Path) -> list[dict]:
    """兼容读取标准 JSONL 和历史缩进式连续 JSON 对象。"""

    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    decoder = json.JSONDecoder()
    records: list[dict] = []
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


class MainTest(unittest.TestCase):
    def _write_settings(
        self,
        root: Path,
        *,
        backend: str,
        ocr_backend: str = "mock",
    ) -> Path:
        """写入主入口测试所需的最小配置。"""

        config_dir = root / "config"
        input_dir = root / "input"
        config_dir.mkdir()
        input_dir.mkdir()
        (input_dir / "demo.txt").write_text("这是一段 AI 工具教程", encoding="utf-8")
        (config_dir / "routing_rules.yaml").write_text(
            (PROJECT_ROOT / "config" / "routing_rules.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (config_dir / "model_prices.yaml").write_text(
            (PROJECT_ROOT / "config" / "model_prices.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        settings_path = config_dir / "settings.yaml"
        settings_path.write_text(
            "\n".join(
                [
                    "input_dir: input",
                    "output_dir: output",
                    f"ocr_backend: {ocr_backend}",
                    f"text_analysis_backend: {backend}",
                    "deepseek_api_key_env: TEST_DEEPSEEK_API_KEY",
                    "default_budget_limit_cny: 50",
                    "target_output_format: jsonl",
                    "allow_partial_success: true",
                ]
            ),
            encoding="utf-8",
        )
        return settings_path

    def test_run_batch_writes_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config"
            input_dir = root / "input"
            config_dir.mkdir()
            input_dir.mkdir()
            (input_dir / "demo.txt").write_text("这是一段 AI 工具教程", encoding="utf-8")
            settings_path = config_dir / "settings.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        "input_dir: input",
                        "output_dir: output",
                        "use_mock_models: true",
                        "default_budget_limit_cny: 50",
                        "target_output_format: jsonl",
                        "allow_partial_success: true",
                    ]
                ),
                encoding="utf-8",
            )

            summary = run_batch(
                settings_path=settings_path,
                routing_rules_path=PROJECT_ROOT / "config" / "routing_rules.yaml",
                model_prices_path=PROJECT_ROOT / "config" / "model_prices.yaml",
                batch_id="batch_test",
                created_at="2026-07-14T10:00:00+08:00",
                generated_at="2026-07-14T10:01:00+08:00",
            )

            batch_dir = Path(summary["batch_dir"])

            self.assertEqual(summary["total_files"], 1)
            self.assertTrue((batch_dir / "batch_metadata.json").exists())
            self.assertTrue((batch_dir / "results.jsonl").exists())
            self.assertTrue((batch_dir / "results_readable.md").exists())
            self.assertTrue((batch_dir / "model_calls.jsonl").exists())
            self.assertTrue((batch_dir / "errors.jsonl").exists())
            self.assertTrue((batch_dir / "batch_report.json").exists())

            for file_name in ["results.jsonl", "model_calls.jsonl", "errors.jsonl"]:
                lines = (batch_dir / file_name).read_text(encoding="utf-8").splitlines()
                for line in lines:
                    json.loads(line)

            report = json.loads((batch_dir / "batch_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["file_stats"]["total_files"], 1)
            self.assertEqual(report["batch_id"], "batch_test")

    def test_cli_input_dir_override_keeps_evaluation_separate_from_default_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "config"
            default_input_dir = root / "input"
            evaluation_input_dir = root / "evaluation" / "text_topic_small_set"
            config_dir.mkdir()
            default_input_dir.mkdir()
            evaluation_input_dir.mkdir(parents=True)
            (config_dir / "routing_rules.yaml").write_text(
                (PROJECT_ROOT / "config" / "routing_rules.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (config_dir / "model_prices.yaml").write_text(
                (PROJECT_ROOT / "config" / "model_prices.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            (default_input_dir / "default_input.txt").write_text("默认业务输入，不应进入本次评估批次。", encoding="utf-8")
            (evaluation_input_dir / "eval_01.txt").write_text("标题：AI 工具评测\n口播：这是一条评估样本。", encoding="utf-8")
            (evaluation_input_dir / "eval_02.txt").write_text("标题：城市交通新闻\n字幕：用于评估分类边界。", encoding="utf-8")

            settings_path = config_dir / "settings.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        "input_dir: input",
                        "output_dir: output",
                        "use_mock_models: true",
                        "text_analysis_backend: deepseek",
                        "default_budget_limit_cny: 50",
                        "target_output_format: jsonl",
                        "allow_partial_success: true",
                    ]
                ),
                encoding="utf-8",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = cli_main(
                    [
                        "--settings",
                        str(settings_path),
                        "--input-dir",
                        str(evaluation_input_dir),
                        "--text-analysis-backend",
                        "mock",
                        "--batch-id",
                        "batch_eval_cli",
                    ]
                )

            batch_dir = root / "output" / "batch_eval_cli"
            results = _read_json_objects(batch_dir / "results.jsonl")
            file_names = {record["file_name"] for record in results}

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(results), 2)
            self.assertEqual(file_names, {"eval_01.txt", "eval_02.txt"})
            self.assertNotIn("default_input.txt", file_names)

    def test_run_batch_include_file_names_only_processes_selected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_settings(root, backend="mock")
            input_dir = root / "input"
            (input_dir / "skip.txt").write_text("这条不应进入本次批次。", encoding="utf-8")
            (input_dir / "target.txt").write_text("这条是本次指定处理文件。", encoding="utf-8")

            summary = run_batch(
                settings_path=settings_path,
                batch_id="batch_include_files",
                include_file_names=["target.txt"],
            )

            batch_dir = Path(summary["batch_dir"])
            results = _read_json_objects(batch_dir / "results.jsonl")
            file_names = {record["file_name"] for record in results}

            self.assertEqual(summary["total_files"], 1)
            self.assertEqual(file_names, {"target.txt"})

    def test_run_batch_include_file_names_rejects_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(Path(tmp_dir), backend="mock")

            with self.assertRaisesRegex(ValueError, "指定文件未在输入目录中找到"):
                run_batch(
                    settings_path=settings_path,
                    batch_id="batch_missing_include_file",
                    include_file_names=["missing.txt"],
                )

    def test_repository_default_backend_is_mock(self) -> None:
        settings = (PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8")

        self.assertIn("ocr_backend: mock", settings)
        self.assertIn("text_analysis_backend: mock", settings)

    def test_run_batch_rejects_deepseek_without_live_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(Path(tmp_dir), backend="deepseek")

            with self.assertRaisesRegex(PermissionError, "--allow-live-api"):
                run_batch(settings_path=settings_path, batch_id="batch_live_blocked")

    @patch("main._ensure_paddleocr_runtime_available")
    def test_run_batch_allows_paddleocr_without_live_permission(self, mock_runtime_check) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(
                Path(tmp_dir),
                backend="mock",
                ocr_backend="paddleocr",
            )

            summary = run_batch(settings_path=settings_path, batch_id="batch_local_ocr")

        self.assertEqual(summary["total_files"], 1)
        mock_runtime_check.assert_called_once_with()

    def test_cli_live_permission_requires_explicit_real_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(Path(tmp_dir), backend="mock")
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = cli_main(["--settings", str(settings_path), "--allow-live-api"])

            self.assertEqual(exit_code, 2)
            self.assertIn("必须与 --text-analysis-backend deepseek 同时使用", stdout.getvalue())
            self.assertFalse((Path(tmp_dir) / "output").exists())

    @patch("main.run_batch")
    def test_cli_paddleocr_keeps_unselected_text_backend_mock(self, mock_run_batch) -> None:
        mock_run_batch.return_value = {"batch_id": "batch_safe"}

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main(["--ocr-backend", "paddleocr"])

        self.assertEqual(exit_code, 0)
        call_kwargs = mock_run_batch.call_args.kwargs
        self.assertEqual(call_kwargs["ocr_backend_override"], "paddleocr")
        self.assertEqual(call_kwargs["text_analysis_backend_override"], "mock")

    @patch("main.run_batch")
    def test_cli_include_files_passes_selected_file_names(self, mock_run_batch) -> None:
        mock_run_batch.return_value = {"batch_id": "batch_safe"}

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main(
                ["--include-files", "img_7.jpg,img_8.jpg,img_9.jpg"]
            )

        self.assertEqual(exit_code, 0)
        call_kwargs = mock_run_batch.call_args.kwargs
        self.assertEqual(call_kwargs["include_file_names"], ["img_7.jpg", "img_8.jpg", "img_9.jpg"])

    @patch("main.run_batch")
    def test_cli_deepseek_keeps_unselected_ocr_backend_mock(self, mock_run_batch) -> None:
        mock_run_batch.return_value = {"batch_id": "batch_safe"}

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = cli_main(
                ["--text-analysis-backend", "deepseek", "--allow-live-api"]
            )

        self.assertEqual(exit_code, 0)
        call_kwargs = mock_run_batch.call_args.kwargs
        self.assertEqual(call_kwargs["ocr_backend_override"], "mock")
        self.assertEqual(call_kwargs["text_analysis_backend_override"], "deepseek")

    def test_cli_api_retry_requires_explicit_deepseek_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_settings(root, backend="mock")
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = cli_main(
                    [
                        "--settings",
                        str(settings_path),
                        "--max-api-retries",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("只能与 --text-analysis-backend deepseek 同时使用", stdout.getvalue())
            self.assertFalse((root / "output").exists())

    def test_run_batch_rejects_more_than_one_api_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_path = self._write_settings(Path(tmp_dir), backend="mock")

            with self.assertRaisesRegex(ValueError, "最大重试次数只能是 0 或 1"):
                run_batch(
                    settings_path=settings_path,
                    deepseek_max_retries=2,
                    batch_id="batch_retry_invalid",
                )

    def test_deepseek_with_permission_but_without_key_stops_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_settings(root, backend="mock")
            stdout = io.StringIO()
            original_api_key = os.environ.pop("TEST_DEEPSEEK_API_KEY", None)

            try:
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli_main(
                        [
                            "--settings",
                            str(settings_path),
                            "--text-analysis-backend",
                            "deepseek",
                            "--allow-live-api",
                        ]
                    )
            finally:
                if original_api_key is not None:
                    os.environ["TEST_DEEPSEEK_API_KEY"] = original_api_key

            self.assertEqual(exit_code, 2)
            self.assertIn("已在发送网络请求前停止运行", stdout.getvalue())
            self.assertFalse((root / "output").exists())

    @patch("main.importlib.util.find_spec", return_value=None)
    def test_paddleocr_without_runtime_stops_before_output(self, _mock_find_spec) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            settings_path = self._write_settings(root, backend="mock")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = cli_main(
                    [
                        "--settings",
                        str(settings_path),
                        "--ocr-backend",
                        "paddleocr",
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("缺少 PaddleOCR 运行依赖", stdout.getvalue())
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
