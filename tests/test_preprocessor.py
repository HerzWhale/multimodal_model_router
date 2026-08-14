"""preprocessor 的测试。"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocessor import (
    _assess_video_evidence_stability,
    _find_ffmpeg_executable,
    _sample_keyframe_indices,
    preprocess_file,
    preprocess_image,
    preprocess_text,
    preprocess_video,
)


class PreprocessorTest(unittest.TestCase):
    def test_preprocess_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.txt"
            path.write_text("测试文本", encoding="utf-8")

            result = preprocess_text(path)

        self.assertEqual(result["raw_text"], "测试文本")
        self.assertIsNone(result["duration_ms"])

    def test_preprocess_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.png"
            path.write_bytes(b"fake-image")

            result = preprocess_image(path)

        self.assertTrue(result["image_path"].endswith("demo.png"))
        self.assertIsNone(result["duration_ms"])

    def test_preprocess_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            path.write_bytes(b"fake-video")

            with patch("preprocessor._load_cv2", return_value=None):
                result = preprocess_video(path)

        self.assertEqual(result["keyframes"], [])
        self.assertIsNone(result["audio_path"])
        self.assertIsNone(result["duration_ms"])
        self.assertEqual(result["preprocessing_artifacts"]["preprocess_status"], "failed")
        self.assertEqual(result["preprocessing_artifacts"]["audio_extraction_status"], "not_attempted_no_artifact_dir")

    def test_sample_keyframe_indices_keeps_early_content(self) -> None:
        self.assertEqual(_sample_keyframe_indices(50), [0, 2, 49])
        self.assertEqual(_sample_keyframe_indices(50, max_keyframes=5), [0, 2, 16, 33, 49])
        self.assertEqual(_sample_keyframe_indices(3), [0, 1, 2])

    def test_assess_video_evidence_stability_requires_enough_keyframes_and_early_frame(self) -> None:
        stable = _assess_video_evidence_stability(
            [
                {"timestamp_ms": 0},
                {"timestamp_ms": 1200},
                {"timestamp_ms": 5000},
            ]
        )
        weak = _assess_video_evidence_stability([{"timestamp_ms": 5000}])

        self.assertEqual(stable["video_evidence_stability"], "stable")
        self.assertEqual(stable["video_evidence_risk_reasons"], [])
        self.assertEqual(weak["video_evidence_stability"], "weak")
        self.assertTrue(weak["video_evidence_risk_reasons"])

    def test_preprocess_video_extracts_multiple_keyframes_when_opencv_is_available(self) -> None:
        class FakeCapture:
            def __init__(self) -> None:
                self.current_frame_index = 0

            def isOpened(self) -> bool:
                return True

            def get(self, property_id: int) -> float:
                return {
                    1: 50,
                    2: 25,
                    3: 1920,
                    4: 1080,
                }[property_id]

            def set(self, property_id: int, value: int) -> bool:
                self.current_frame_index = value
                return True

            def read(self) -> tuple[bool, object]:
                return True, {"frame_index": self.current_frame_index}

            def release(self) -> None:
                return None

        class FakeCv2:
            CAP_PROP_FRAME_COUNT = 1
            CAP_PROP_FPS = 2
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4
            CAP_PROP_POS_FRAMES = 5

            def VideoCapture(self, source_path: str) -> FakeCapture:
                return FakeCapture()

            def imwrite(self, target_path: str, frame: object) -> bool:
                Path(target_path).write_bytes(b"fake-keyframe")
                return True

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            artifact_dir = Path(tmp_dir) / "artifacts"
            path.write_bytes(b"fake-video")

            with (
                patch("preprocessor._load_cv2", return_value=FakeCv2()),
                patch("preprocessor._find_ffmpeg_executable", return_value=None),
            ):
                result = preprocess_video(path, artifact_dir=artifact_dir)

        self.assertEqual(len(result["keyframes"]), 3)
        self.assertIsNone(result["audio_path"])
        self.assertTrue(result["keyframes"][0].endswith("demo_frame_0001.jpg"))
        self.assertTrue(result["keyframes"][2].endswith("demo_frame_0003.jpg"))
        self.assertEqual(result["duration_ms"], 2000)
        self.assertEqual(result["preprocessing_artifacts"]["preprocess_status"], "success")
        self.assertEqual(result["preprocessing_artifacts"]["keyframe_extraction_status"], "extracted")
        self.assertEqual(result["preprocessing_artifacts"]["keyframe_sampling_strategy"], "start_early_then_spaced")
        self.assertEqual(result["preprocessing_artifacts"]["max_keyframes"], 3)
        self.assertEqual(result["preprocessing_artifacts"]["video_evidence_stability"], "stable")
        self.assertEqual(result["preprocessing_artifacts"]["audio_extraction_status"], "dependency_missing")
        self.assertEqual(
            [item["source_frame_index"] for item in result["preprocessing_artifacts"]["keyframe_metadata"]],
            [0, 2, 49],
        )
        self.assertEqual(
            [item["timestamp_ms"] for item in result["preprocessing_artifacts"]["keyframe_metadata"]],
            [0, 80, 1960],
        )
        self.assertEqual(result["preprocessing_artifacts"]["frame_count"], 50)
        self.assertEqual(result["preprocessing_artifacts"]["fps"], 25)

    def test_preprocess_video_falls_back_when_opencv_imwrite_fails(self) -> None:
        class FakeEncodedImage:
            def tobytes(self) -> bytes:
                return b"encoded-keyframe"

        class FakeCapture:
            def __init__(self) -> None:
                self.current_frame_index = 0

            def isOpened(self) -> bool:
                return True

            def get(self, property_id: int) -> float:
                return {
                    1: 25,
                    2: 25,
                    3: 720,
                    4: 1280,
                }[property_id]

            def set(self, property_id: int, value: int) -> bool:
                self.current_frame_index = value
                return True

            def read(self) -> tuple[bool, object]:
                return True, {"frame_index": self.current_frame_index}

            def release(self) -> None:
                return None

        class FakeCv2:
            CAP_PROP_FRAME_COUNT = 1
            CAP_PROP_FPS = 2
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4
            CAP_PROP_POS_FRAMES = 5

            def VideoCapture(self, source_path: str) -> FakeCapture:
                return FakeCapture()

            def imwrite(self, target_path: str, frame: object) -> bool:
                return False

            def imencode(self, extension: str, frame: object) -> tuple[bool, FakeEncodedImage]:
                return True, FakeEncodedImage()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "中文视频.mp4"
            artifact_dir = Path(tmp_dir) / "中文产物"
            path.write_bytes(b"fake-video")

            with (
                patch("preprocessor._load_cv2", return_value=FakeCv2()),
                patch("preprocessor._find_ffmpeg_executable", return_value=None),
            ):
                result = preprocess_video(path, artifact_dir=artifact_dir)

            keyframe_path = Path(result["keyframes"][0])
            keyframe_bytes = keyframe_path.read_bytes()

        self.assertEqual(keyframe_bytes, b"encoded-keyframe")
        self.assertEqual(result["preprocessing_artifacts"]["keyframe_extraction_status"], "extracted")

    def test_preprocess_video_extracts_audio_with_ffmpeg_when_available(self) -> None:
        class FakeCapture:
            def isOpened(self) -> bool:
                return True

            def get(self, property_id: int) -> float:
                return {
                    1: 25,
                    2: 25,
                    3: 720,
                    4: 1280,
                }[property_id]

            def set(self, property_id: int, value: int) -> bool:
                return True

            def read(self) -> tuple[bool, object]:
                return True, {"frame": "demo"}

            def release(self) -> None:
                return None

        class FakeCv2:
            CAP_PROP_FRAME_COUNT = 1
            CAP_PROP_FPS = 2
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4
            CAP_PROP_POS_FRAMES = 5

            def VideoCapture(self, source_path: str) -> FakeCapture:
                return FakeCapture()

            def imwrite(self, target_path: str, frame: object) -> bool:
                Path(target_path).write_bytes(b"fake-keyframe")
                return True

        def fake_ffmpeg_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
            Path(command[-1]).write_bytes(b"fake-wav")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.mp4"
            artifact_dir = Path(tmp_dir) / "artifacts"
            path.write_bytes(b"fake-video")

            with (
                patch("preprocessor._load_cv2", return_value=FakeCv2()),
                patch("preprocessor._find_ffmpeg_executable", return_value="ffmpeg"),
                patch("preprocessor.subprocess.run", side_effect=fake_ffmpeg_run),
            ):
                result = preprocess_video(path, artifact_dir=artifact_dir)

            audio_path = Path(result["audio_path"])
            artifacts = result["preprocessing_artifacts"]
            self.assertTrue(audio_path.name.endswith("_audio.wav"))
            self.assertEqual(audio_path.read_bytes(), b"fake-wav")
            self.assertEqual(artifacts["audio_path"], str(audio_path))
            self.assertEqual(artifacts["audio_extraction_status"], "extracted")
            self.assertEqual(artifacts["audio_extraction_method"], "ffmpeg_wav")
            self.assertEqual(artifacts["audio_sample_rate_hz"], 16000)
            self.assertEqual(artifacts["audio_channels"], 1)

    def test_find_ffmpeg_executable_accepts_explicit_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ffmpeg_path = Path(tmp_dir) / "ffmpeg.exe"
            ffmpeg_path.write_text("fake ffmpeg", encoding="utf-8")

            resolved = _find_ffmpeg_executable(ffmpeg_path)

        self.assertEqual(resolved, str(ffmpeg_path))

    def test_find_ffmpeg_executable_accepts_explicit_directory_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ffmpeg_path = Path(tmp_dir) / "ffmpeg.exe"
            ffmpeg_path.write_text("fake ffmpeg", encoding="utf-8")

            resolved = _find_ffmpeg_executable(Path(tmp_dir))

        self.assertEqual(resolved, str(ffmpeg_path))

    def test_find_ffmpeg_executable_accepts_env_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ffmpeg_path = Path(tmp_dir) / "ffmpeg.exe"
            ffmpeg_path.write_text("fake ffmpeg", encoding="utf-8")

            with patch.dict(os.environ, {"FFMPEG_PATH": str(ffmpeg_path)}):
                resolved = _find_ffmpeg_executable()

        self.assertEqual(resolved, str(ffmpeg_path))

    def test_preprocess_file_dispatches_by_media_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "demo.txt"
            path.write_text("分发测试", encoding="utf-8")

            result = preprocess_file({"media_type": "text", "source_path": str(path)})

        self.assertEqual(result["raw_text"], "分发测试")


if __name__ == "__main__":
    unittest.main()
