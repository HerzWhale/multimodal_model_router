"""model_router 的测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model_router import load_routing_rules, select_model


class ModelRouterTest(unittest.TestCase):
    def test_load_routing_rules(self) -> None:
        config_path = PROJECT_ROOT / "config" / "routing_rules.yaml"

        routing_rules = load_routing_rules(config_path)

        self.assertEqual(routing_rules["ocr"]["provider"], "doubao")
        self.assertEqual(routing_rules["text_analysis"]["model_name"], "mock-text")

    def test_select_model(self) -> None:
        routing_rules = {
            "ocr": {
                "provider": "doubao",
                "model_name": "mock-ocr",
            }
        }

        selected = select_model("ocr", routing_rules)

        self.assertEqual(selected["provider"], "doubao")
        self.assertEqual(selected["model_name"], "mock-ocr")

    def test_select_model_rejects_unknown_task_type(self) -> None:
        with self.assertRaises(KeyError):
            select_model("unknown_task", {})


if __name__ == "__main__":
    unittest.main()
