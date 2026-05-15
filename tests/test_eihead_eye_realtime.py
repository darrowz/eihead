from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: str):
    module_path = REPO_ROOT / path
    assert module_path.exists(), f"missing module under test: {path}"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


realtime = _load_module("eihead_eye_realtime_under_test", "eihead/eye/realtime.py")

RealtimeDetection = realtime.RealtimeDetection
RealtimeEyePipeline = realtime.RealtimeEyePipeline
RealtimeVisionFrame = realtime.RealtimeVisionFrame
CompatStaticFrameSource = realtime.CompatStaticFrameSource


def test_realtime_pipeline_consumes_injected_stream_and_reports_live_status() -> None:
    clock = StepClock([10.00, 10.25, 10.50, 10.75])
    source = SequenceFrameSource(
        [
            RealtimeVisionFrame(frame_id="f1", timestamp=10.00, source="usb_camera", payload=b"1"),
            RealtimeVisionFrame(frame_id="f2", timestamp=10.25, source="usb_camera", payload=b"2"),
        ]
    )
    detector = FakeDetector(
        [
            RealtimeDetection(label="face", confidence=0.87, bbox=(0.1, 0.2, 0.3, 0.4)),
            RealtimeDetection(label="person", confidence=0.66, bbox=(0.0, 0.0, 1.0, 1.0)),
        ]
    )

    pipeline = RealtimeEyePipeline(
        frame_source=source,
        detector=detector,
        mode="realtime_stream",
        clock=clock,
    )

    status = pipeline.run(max_frames=2)

    assert status.mode == "realtime_stream"
    assert status.frame_count == 2
    assert status.detection_count == 4
    assert status.fps == 4.0
    assert status.last_frame_age == 0.5
    assert status.top_detection == RealtimeDetection(
        label="face",
        confidence=0.87,
        bbox=(0.1, 0.2, 0.3, 0.4),
    )
    assert status.backend == "fake_detector"
    assert status.status == "not_wired"
    assert status.status_reason == "not_wired"
    assert status.placeholder is True
    assert status.not_wired is True
    assert "detector is not wired" in status.message


def test_realtime_pipeline_reports_stale_reason_boxes_scores_and_readiness() -> None:
    clock = StepClock([100.0, 100.1, 104.5])
    source = SequenceFrameSource(
        [
            RealtimeVisionFrame(
                frame_id="late-frame",
                timestamp=100.0,
                source="usb_camera",
                width=640,
                height=480,
            )
        ]
    )
    detector = WiredDetector(
        [
            RealtimeDetection(label="person", confidence=0.91, bbox=(0.1, 0.2, 0.3, 0.4)),
            RealtimeDetection(
                label="face",
                confidence=0.73,
                bbox={"x_min": 0.5, "y_min": 0.1, "x_max": 0.7, "y_max": 0.6},
            ),
        ]
    )
    pipeline = RealtimeEyePipeline(
        frame_source=source,
        detector=detector,
        clock=clock,
        max_frame_age_s=2.0,
    )

    status = pipeline.run(max_frames=1)
    payload = status.to_dict()

    assert status.status == "stale"
    assert status.stream_ready is False
    assert status.stale is True
    assert status.degraded is False
    assert status.status_reason == "last_frame_stale"
    assert status.stale_reason == "last frame age 4.5s exceeds 2.0s"
    assert payload["readiness"] == {"ready": False, "reason": "last_frame_stale"}
    assert payload["detection_boxes"] == [
        {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
        {"x_min": 0.5, "y_min": 0.1, "x_max": 0.7, "y_max": 0.6},
    ]
    assert payload["detection_scores"] == [0.91, 0.73]


def test_process_next_reports_frame_age_at_observation_time() -> None:
    clock = StepClock([200.0, 202.5])
    source = SequenceFrameSource(
        [
            RealtimeVisionFrame(
                frame_id="poll-frame",
                timestamp=200.0,
                source="usb_camera",
            )
        ]
    )
    pipeline = RealtimeEyePipeline(
        frame_source=source,
        detector=WiredDetector([]),
        clock=clock,
    )

    status = pipeline.process_next()

    assert status["last_frame_age"] == 2.5
    assert status["stream_ready"] is True


def test_placeholder_pipeline_reports_not_wired_realtime_status() -> None:
    pipeline = RealtimeEyePipeline.placeholder()

    status = pipeline.run(max_frames=1)
    status_payload = status.to_dict()

    assert status.mode == "realtime_stream"
    assert status.frame_count == 0
    assert status.detection_count == 0
    assert status.fps == 0.0
    assert status.last_frame_age is None
    assert status.top_detection is None
    assert status.backend == "not_wired"
    assert status.placeholder is True
    assert status.not_wired is True
    for key in (
        "fps",
        "frame_count",
        "last_frame_age",
        "detection_count",
        "top_detection",
        "backend",
        "placeholder",
        "not_wired",
    ):
        assert key in status_payload


def test_run_max_frames_limits_each_call_instead_of_lifetime_total() -> None:
    clock = StepClock([30.00, 30.10, 30.20, 30.30, 30.40, 30.50, 30.60])
    source = SequenceFrameSource(
        [
            RealtimeVisionFrame(frame_id=f"f{index}", timestamp=30.0 + (index * 0.1), source="usb_camera")
            for index in range(4)
        ]
    )
    detector = FakeDetector([RealtimeDetection(label="face", confidence=0.8, bbox=(0.1, 0.2, 0.3, 0.4))])
    pipeline = RealtimeEyePipeline(frame_source=source, detector=detector, clock=clock)

    first_status = pipeline.run(max_frames=2)
    second_status = pipeline.run(max_frames=2)

    assert first_status.frame_count == 2
    assert second_status.frame_count == 4
    assert second_status.last_frame_id == "f3"


def test_raw_detection_with_non_numeric_class_id_does_not_break_status_serialization() -> None:
    clock = StepClock([40.0, 40.1, 40.2])
    source = SequenceFrameSource([RealtimeVisionFrame(frame_id="f1", timestamp=40.0, source="usb_camera")])
    detector = FakeDetector(
        [
            {
                "label": "face",
                "score": 0.91,
                "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                "class_id": "not-a-number",
            }
        ]
    )
    pipeline = RealtimeEyePipeline(frame_source=source, detector=detector, clock=clock)

    status = pipeline.run(max_frames=1).to_dict()

    assert status["top_detection"]["label"] == "face"
    assert "class_id" not in status["top_detection"]


def test_static_frames_are_explicit_compat_sources() -> None:
    source = CompatStaticFrameSource(
        frame_id="still-1",
        frame_path="fixtures/still.jpg",
        captured_at_ts=20.0,
    )

    frame = source.next_frame()

    assert frame is not None
    assert frame.frame_id == "still-1"
    assert frame.timestamp == 20.0
    assert frame.source == "compat_static_frame"
    assert frame.mode == "compat_static_frame"
    assert frame.metadata == {"frame_path": "fixtures/still.jpg"}

    with pytest.raises(ValueError, match="compat_static_frame source"):
        RealtimeVisionFrame(
            frame_id="bad-static",
            timestamp=20.0,
            source="usb_camera",
            mode="compat_static_frame",
            payload=b"still",
        )


def test_compat_static_pipeline_status_is_not_reported_as_realtime_ok() -> None:
    clock = StepClock([50.0, 50.1, 50.2])
    source = CompatStaticFrameSource(frame_id="still-2", captured_at_ts=50.0)
    detector = FakeDetector([RealtimeDetection(label="person", confidence=0.7, bbox=(0.1, 0.2, 0.3, 0.4))])
    pipeline = RealtimeEyePipeline(frame_source=source, detector=detector, clock=clock)

    status = pipeline.run(max_frames=1)

    assert status.status == "compat_static"
    assert status.mode == "compat_static_frame"
    assert status.compatibility_mode is True
    assert status.stream_ready is False
    assert status.status_reason == "compat_static_frame_test_only"
    assert status.to_dict()["compatibility_static_image"] == {
        "active": True,
        "mode": "compat_static_frame",
        "test_only": True,
    }
    assert "realtime stream remains primary" in status.message


class StepClock:
    def __init__(self, values: list[float]) -> None:
        self._values = values
        self._index = 0

    def __call__(self) -> float:
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return value


class SequenceFrameSource:
    source_name = "usb_camera"

    def __init__(self, frames: list[object]) -> None:
        self._frames = frames

    def frames(self):
        return iter(self._frames)


class FakeDetector:
    backend = "fake_detector"
    placeholder = True
    not_wired = True

    def __init__(self, detections: list[object]) -> None:
        self._detections = detections

    def detect(self, frame: object):
        return list(self._detections)


class WiredDetector(FakeDetector):
    placeholder = False
    not_wired = False
