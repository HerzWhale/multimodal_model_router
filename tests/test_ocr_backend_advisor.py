"""ocr_backend_advisor 的离线测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ocr_backend_advisor import (  # noqa: E402
    build_ocr_backend_advice,
    default_ocr_candidate_catalog,
    write_ocr_backend_advice_report,
)


def _failed_gate_report() -> dict[str, object]:
    """构造未通过OCR闸门的测试报告。"""

    return {
        "report_name": "image_ocr_batch_gate_report",
        "thresholds": {
            "min_exact_segment_recall": 0.9,
            "max_character_error_rate": 0.05,
            "max_latency_ms": 2000,
        },
        "batch_overview": {
            "overall_exact_segment_recall": 0.78,
            "overall_character_error_rate": 0.11,
            "ocr_p95_latency_ms": 28261,
        },
        "gate_decision": {"status": "not_passed"},
    }


def _latency_profile() -> dict[str, object]:
    """构造OCR延迟拆分测试报告。"""

    return {
        "profile_name": "image_ocr_latency_profile",
        "decision": {"main_bottleneck": "first_predict"},
    }


def _failed_rapidocr_candidate_report() -> dict[str, object]:
    """构造RapidOCR候选后端未通过闸门的测试报告。"""

    return {
        "report_name": "rapidocr_candidate_evaluation",
        "backend_id": "rapidocr_onnxruntime_local",
        "dependency": {"status": "available"},
        "overview": {
            "overall_exact_segment_recall": 0.83,
            "overall_character_error_rate": 0.106,
            "ocr_p95_latency_ms": 4294,
            "ocr_external_api_cost_cny": 0.0,
        },
        "gate_decision": {
            "status": "not_passed",
            "next_action": "候选后端未通过当前OCR闸门，不应接入主流程。",
        },
    }


class OcrBackendAdvisorTest(unittest.TestCase):
    def test_failed_paddleocr_gate_recommends_local_onnx_candidate_first(self) -> None:
        report = build_ocr_backend_advice(
            gate_report=_failed_gate_report(),
            latency_profile=_latency_profile(),
        )

        self.assertEqual(report["decision"]["switch_signal"], "evaluate_alternative_backends")
        self.assertEqual(report["decision"]["recommended_next_backend_id"], "rapidocr_onnxruntime_local")
        self.assertTrue(report["current_metrics"]["quality_failed"])
        self.assertTrue(report["current_metrics"]["latency_failed"])
        self.assertEqual(report["current_metrics"]["latency_bottleneck"], "first_predict")

    def test_privacy_requirement_pushes_cloud_candidate_later(self) -> None:
        report = build_ocr_backend_advice(
            gate_report=_failed_gate_report(),
            latency_profile=_latency_profile(),
            privacy_required=True,
        )
        order = report["decision"]["evaluation_order"]

        self.assertLess(order.index("rapidocr_onnxruntime_local"), order.index("cloud_ocr_service"))
        self.assertIn("数据不出本机", " ".join(report["decision"]["reasons"]))

    def test_missing_metrics_do_not_fabricate_decision(self) -> None:
        report = build_ocr_backend_advice(
            gate_report={
                "report_name": "image_ocr_batch_gate_report",
                "thresholds": {},
                "batch_overview": {},
                "gate_decision": {"status": "unknown"},
            },
            latency_profile=None,
        )

        self.assertEqual(report["decision"]["switch_signal"], "need_more_evidence")
        self.assertIsNone(report["current_metrics"]["quality_failed"])
        self.assertIsNone(report["current_metrics"]["latency_failed"])
        self.assertEqual(report["current_metrics"]["latency_bottleneck"], "current_data_not_provided")

    def test_candidate_catalog_marks_unknown_candidates_as_not_project_evidence(self) -> None:
        catalog = default_ocr_candidate_catalog()
        by_id = {candidate["backend_id"]: candidate for candidate in catalog}

        self.assertTrue(by_id["paddleocr_local"]["known_from_project"])
        self.assertFalse(by_id["rapidocr_onnxruntime_local"]["known_from_project"])
        self.assertFalse(by_id["cloud_ocr_service"]["known_from_project"])

    def test_failed_rapidocr_candidate_is_not_recommended_again(self) -> None:
        report = build_ocr_backend_advice(
            gate_report=_failed_gate_report(),
            latency_profile=_latency_profile(),
            candidate_evaluation_reports=[_failed_rapidocr_candidate_report()],
        )
        by_id = {candidate["backend_id"]: candidate for candidate in report["candidate_catalog"]}

        self.assertEqual(report["candidate_evaluations"]["rapidocr_onnxruntime_local"]["gate_status"], "not_passed")
        self.assertEqual(by_id["rapidocr_onnxruntime_local"]["integration_status"], "evaluated_not_passed")
        self.assertEqual(report["decision"]["recommended_next_backend_id"], "cloud_ocr_service")
        self.assertIn("RapidOCR候选后端已完成同批样本实测", " ".join(report["decision"]["reasons"]))
        self.assertIn("不要接入主流程", report["decision"]["next_action"])

    def test_write_report_outputs_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            gate_path = tmp_path / "gate.json"
            latency_path = tmp_path / "latency.json"
            output_json_path = tmp_path / "advice.json"
            output_markdown_path = tmp_path / "advice.md"
            rapidocr_path = tmp_path / "rapidocr.json"
            gate_path.write_text(json.dumps(_failed_gate_report(), ensure_ascii=False), encoding="utf-8")
            latency_path.write_text(json.dumps(_latency_profile(), ensure_ascii=False), encoding="utf-8")
            rapidocr_path.write_text(json.dumps(_failed_rapidocr_candidate_report(), ensure_ascii=False), encoding="utf-8")

            json_path, markdown_path = write_ocr_backend_advice_report(
                gate_report_path=gate_path,
                latency_profile_path=latency_path,
                output_json_path=output_json_path,
                output_markdown_path=output_markdown_path,
                candidate_evaluation_paths=[rapidocr_path],
            )
            report = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(report["report_name"], "ocr_backend_advice")
        self.assertIn("OCR 后端取舍判断报告", markdown)
        self.assertIn("rapidocr_onnxruntime_local", markdown)
        self.assertIn("已评估候选后端", markdown)


if __name__ == "__main__":
    unittest.main()
