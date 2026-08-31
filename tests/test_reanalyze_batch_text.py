"""reanalyze_batch_text 的离线测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from reanalyze_batch_text import (
    _backend_setting,
    _filter_records,
    _parse_include_files,
    _setting,
    build_reanalysis_evidence,
    reanalyze_records,
)


MODEL_PRICES = {
    "deepseek-v4-flash": {
        "provider": "deepseek",
        "pricing_rules": [
            {"unit_type": "input_tokens", "price_cny_per_unit": 0.000001},
            {"unit_type": "output_tokens", "price_cny_per_unit": 0.000002},
        ],
    },
    "qwen-plus-2025-12-01": {
        "provider": "qwen",
        "pricing_rules": [
            {"unit_type": "input_tokens", "price_cny_per_unit": 0.0000008},
            {"unit_type": "output_tokens", "price_cny_per_unit": 0.000002},
        ],
    },
}


def _source_record() -> dict:
    """构造历史视频结果。"""

    return {
        "batch_id": "batch_old",
        "file_id": "file_0001",
        "file_name": "例子.mp4",
        "media_type": "video",
        "processing_status": "success",
        "raw_text": None,
        "ocr_text": "合唱音乐视频字幕",
        "visual_description": "多人合唱舞台画面",
        "audio_transcript": "模拟音频转写：例子_audio.wav",
        "quality_flags": [],
        "warning_messages": [],
        "topic": "technology",
        "secondary_topics": [],
        "tags": ["旧标签"],
        "summary": "旧摘要",
        "business_use": "旧用途",
    }


def _failed_source_record_with_real_audio() -> dict:
    """构造文本分析失败、但上游证据完整的历史视频结果。"""

    record = _source_record()
    record["processing_status"] = "failed"
    record["audio_transcript"] = "真实音频转写文本"
    record["quality_flags"] = ["text_analysis_failed"]
    record["warning_messages"] = ["文本分析模型调用失败，无法产出有效分类、标签和摘要。"]
    record["error_message"] = "[deepseek_content_empty] DeepSeek 模型内容为空。"
    return record


class ReanalyzeBatchTextTest(unittest.TestCase):
    def test_nested_settings_helpers_keep_legacy_fallback(self) -> None:
        settings = {
            "output_dir": "old_output",
            "deepseek_model_name": "old-deepseek",
            "paths": {"output_dir": "new_output"},
            "backends": {
                "text_analysis": {
                    "deepseek": {"model_name": "new-deepseek"},
                }
            },
        }

        self.assertEqual(_setting(settings, "paths.output_dir", "output_dir"), "new_output")
        self.assertEqual(
            _backend_setting(settings, "text_analysis", "deepseek", "model_name", "deepseek_model_name"),
            "new-deepseek",
        )
        self.assertEqual(
            _backend_setting(settings, "text_analysis", "deepseek", "base_url", "deepseek_base_url", "fallback"),
            "fallback",
        )

    def test_build_reanalysis_evidence_drops_mock_audio(self) -> None:
        evidence = build_reanalysis_evidence(_source_record())

        self.assertEqual(evidence["ocr_text"], "合唱音乐视频字幕")
        self.assertEqual(evidence["visual_description"], "多人合唱舞台画面")
        self.assertIsNone(evidence["audio_transcript"])

    @patch("reanalyze_batch_text.deepseek_text_analysis_client")
    def test_reanalyze_records_updates_text_fields_only(self, mock_deepseek) -> None:
        mock_deepseek.return_value = {
            "topic": "other",
            "secondary_topics": [],
            "tags": ["合唱", "舞台"],
            "summary": "多人合唱音乐视频。",
            "business_use": "可用于内容归档、检索和人工复核。",
            "_api_usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "_api_attempts": [{"status": "success", "latency_ms": 1234, "api_usage": {"prompt_tokens": 100, "completion_tokens": 50}}],
            "_quality_flags": [],
        }

        results, model_calls, errors = reanalyze_records(
            source_records=[_source_record()],
            batch_id="batch_new",
            api_key="test-key",
            model_name="deepseek-v4-flash",
            base_url="https://example.test",
            model_prices=MODEL_PRICES,
            max_retries=0,
            max_tokens=3000,
            compact_mode=True,
        )

        self.assertEqual(errors, [])
        self.assertEqual(results[0]["topic"], "other")
        self.assertEqual(results[0]["source_batch_id"], "batch_old")
        self.assertEqual(results[0]["audio_transcript"], None)
        self.assertEqual(results[0]["evidence_used"], ["ocr_text", "visual_description"])
        self.assertEqual(results[0]["missing_evidence"], ["audio_transcript"])
        self.assertEqual(results[0]["processing_status"], "partial_success")
        self.assertEqual(model_calls[0]["provider"], "deepseek")
        self.assertEqual(model_calls[0]["latency_ms"], 1234)
        self.assertEqual(model_calls[0]["cost_cny"], 0.0002)
        self.assertEqual(mock_deepseek.call_args.kwargs["max_tokens"], 3000)
        self.assertTrue(mock_deepseek.call_args.kwargs["compact_mode"])

    @patch("reanalyze_batch_text.qwen_text_analysis_client")
    def test_reanalyze_records_supports_qwen_without_upstream_calls(self, mock_qwen) -> None:
        mock_qwen.return_value = {
            "topic": "other",
            "secondary_topics": [],
            "tags": ["合唱"],
            "summary": "多人合唱音乐视频。",
            "business_use": "可用于内容归档。",
            "_api_usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "_api_attempts": [{
                "status": "success",
                "latency_ms": 900,
                "api_usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "response_model_name": "qwen-plus-2025-12-01",
            }],
            "_quality_flags": [],
        }

        results, model_calls, errors = reanalyze_records(
            source_records=[_source_record()],
            batch_id="batch_qwen",
            api_key="test-key",
            model_name="qwen-plus-2025-12-01",
            base_url="https://example.test",
            model_prices=MODEL_PRICES,
            max_retries=0,
            max_tokens=1600,
            compact_mode=True,
            text_analysis_backend="qwen_text",
        )

        self.assertEqual(errors, [])
        self.assertEqual(results[0]["topic"], "other")
        self.assertEqual(model_calls[0]["provider"], "qwen")
        self.assertEqual(model_calls[0]["model_name"], "qwen-plus-2025-12-01")
        self.assertEqual(model_calls[0]["response_model_name"], "qwen-plus-2025-12-01")
        self.assertEqual(model_calls[0]["latency_ms"], 900)
        self.assertEqual(model_calls[0]["cost_cny"], 0.00018)
        mock_qwen.assert_called_once()

    @patch("reanalyze_batch_text.deepseek_text_analysis_client")
    def test_reanalyze_records_keeps_failed_and_success_attempts(self, mock_deepseek) -> None:
        mock_deepseek.return_value = {
            "topic": "other",
            "secondary_topics": [],
            "tags": ["合唱"],
            "summary": "多人合唱音乐视频。",
            "business_use": "可用于内容归档。",
            "_api_usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "_api_attempts": [
                {"status": "failed", "latency_ms": 1000, "api_usage": {"prompt_tokens": 100, "completion_tokens": 0}, "error_message": "[deepseek_content_empty] DeepSeek 模型内容为空。"},
                {"status": "success", "latency_ms": 1200, "api_usage": {"prompt_tokens": 100, "completion_tokens": 50}, "error_message": None},
            ],
            "_quality_flags": [],
        }

        results, model_calls, errors = reanalyze_records(
            source_records=[_source_record()],
            batch_id="batch_new",
            api_key="test-key",
            model_name="deepseek-v4-flash",
            base_url="https://example.test",
            model_prices=MODEL_PRICES,
            max_retries=1,
            max_tokens=1500,
        )

        self.assertEqual(errors, [])
        self.assertEqual([call["status"] for call in model_calls], ["failed", "success"])
        self.assertEqual(results[0]["call_ids"], ["file_0001_reanalyze_0001_attempt_01", "file_0001_reanalyze_0001_attempt_02"])
        self.assertEqual(results[0]["processing_cost_cny"], 0.0003)
        self.assertEqual(results[0]["processing_time_ms"], 2200)

    @patch("reanalyze_batch_text.deepseek_text_analysis_client")
    def test_reanalyze_records_success_overrides_old_text_failure(self, mock_deepseek) -> None:
        mock_deepseek.return_value = {
            "topic": "other",
            "secondary_topics": [],
            "tags": ["合唱"],
            "summary": "多人合唱音乐视频。",
            "business_use": "可用于内容归档。",
            "_api_usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "_api_attempts": [
                {"status": "success", "latency_ms": 1200, "api_usage": {"prompt_tokens": 100, "completion_tokens": 50}, "error_message": None},
            ],
            "_quality_flags": [],
        }

        results, _, errors = reanalyze_records(
            source_records=[_failed_source_record_with_real_audio()],
            batch_id="batch_new",
            api_key="test-key",
            model_name="deepseek-v4-flash",
            base_url="https://example.test",
            model_prices=MODEL_PRICES,
            max_retries=0,
            max_tokens=1500,
        )

        self.assertEqual(errors, [])
        self.assertEqual(results[0]["processing_status"], "success")
        self.assertEqual(results[0]["error_message"], None)
        self.assertEqual(results[0]["quality_flags"], [])
        self.assertEqual(results[0]["warning_messages"], [])
        self.assertEqual(results[0]["missing_evidence"], [])

    @patch("reanalyze_batch_text.deepseek_text_analysis_client")
    def test_reanalyze_records_clears_deferred_state_after_success(self, mock_deepseek) -> None:
        source = _failed_source_record_with_real_audio()
        source["processing_status"] = "pending"
        source["quality_flags"] = ["text_analysis_deferred"]
        source["warning_messages"] = ["上游证据已写出，文本分析已延后到第二阶段执行。"]
        mock_deepseek.return_value = {
            "topic": "other",
            "secondary_topics": [],
            "tags": ["合唱"],
            "summary": "多人合唱音乐视频。",
            "business_use": "可用于内容归档。",
            "_api_usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "_api_attempts": [{"status": "success", "latency_ms": 1200, "api_usage": {"prompt_tokens": 100, "completion_tokens": 50}}],
            "_quality_flags": [],
        }

        results, model_calls, errors = reanalyze_records(
            source_records=[source],
            batch_id="batch_completed",
            api_key="test-key",
            model_name="deepseek-v4-flash",
            base_url="https://example.test",
            model_prices=MODEL_PRICES,
            max_retries=0,
            max_tokens=1500,
        )

        self.assertEqual(errors, [])
        self.assertEqual(results[0]["processing_status"], "success")
        self.assertEqual(results[0]["source_batch_id"], "batch_old")
        self.assertNotIn("text_analysis_deferred", results[0]["quality_flags"])
        self.assertEqual(results[0]["warning_messages"], [])
        self.assertEqual([call["task_type"] for call in model_calls], ["text_analysis"])

    def test_filter_records_accepts_file_name_or_file_id(self) -> None:
        records = [
            {**_source_record(), "file_id": "file_0001", "file_name": "例子.mp4"},
            {**_source_record(), "file_id": "file_0002", "file_name": "例子1.mp4"},
        ]

        include_files = _parse_include_files("例子.mp4,file_0002")
        filtered = _filter_records(records, include_files)

        self.assertEqual([record["file_id"] for record in filtered], ["file_0001", "file_0002"])


if __name__ == "__main__":
    unittest.main()
