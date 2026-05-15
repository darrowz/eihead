from __future__ import annotations

from eihead.eye.realtime import RealtimeEyeStatus
from eihead.eye.service import RealtimeEyeService


def test_realtime_eye_service_polls_adapter_and_exposes_latest_observation() -> None:
    adapter = FakeAdapter(
        initial_status={
            "mode": "realtime_stream",
            "status": "waiting_for_frame",
            "backend": "fake_realtime",
            "stream_ready": False,
            "placeholder": False,
            "not_wired": False,
        },
        polled_statuses=[
            {
                "mode": "realtime_stream",
                "status": "ok",
                "backend": "fake_realtime",
                "stream_ready": True,
                "placeholder": False,
                "not_wired": False,
                "last_frame_id": "frame-42",
                "width": 640,
                "height": 480,
                "last_frame_captured_at_ts": 123.25,
                "detections": [
                    {
                        "label": "person",
                        "score": 0.91,
                        "confidence": 0.91,
                        "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                        "track_id": "track-1",
                    }
                ],
                "detection_boxes": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}],
                "detection_scores": [0.91],
                "top_detection": {
                    "label": "person",
                    "score": 0.91,
                    "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                    "track_id": "track-1",
                },
            }
        ],
    )
    service = RealtimeEyeService(adapter=adapter)

    status = service.poll_once()
    observation = service.latest_observation()

    assert adapter.poll_count == 1
    assert service.latest_status == status
    assert status["status"] == "ok"
    assert observation["kind"] == "realtime_vision_observation"
    assert observation["mode"] == "realtime_stream"
    assert observation["frame_id"] == "frame-42"
    assert observation["width"] == 640
    assert observation["height"] == 480
    assert observation["detections"] == status["detections"]
    assert observation["boxes"] == status["detection_boxes"]
    assert observation["scores"] == status["detection_scores"]
    assert observation["tracked_target"] == {
        "label": "person",
        "score": 0.91,
        "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
        "track_id": "track-1",
    }
    assert observation["captured_at_ts"] == 123.25
    assert observation["status"] == "ok"
    assert observation["scene_id"].startswith("scene_rt_")
    assert observation["scene"]["sceneId"] == observation["scene_id"]
    assert "person" in observation["scene_summary"]
    assert observation["tracks"][0]["label"] == "person"
    assert observation["events"][0]["eventType"] == "appeared"


def test_realtime_eye_service_poll_runs_bounded_loop_without_threads() -> None:
    adapter = FakeAdapter(
        initial_status={"status": "waiting_for_frame"},
        polled_statuses=[
            {"status": "waiting_for_frame", "last_frame_id": ""},
            {"status": "ok", "last_frame_id": "frame-2"},
        ],
    )
    service = RealtimeEyeService(adapter=adapter, sleep=lambda _seconds: None)

    status = service.poll(max_polls=2, interval_s=0.01)

    assert adapter.poll_count == 2
    assert status["last_frame_id"] == "frame-2"
    assert service.latest_observation()["frame_id"] == "frame-2"


def test_realtime_eye_service_can_wrap_pipeline_process_next() -> None:
    pipeline = FakePipeline(
        initial_status={"status": "waiting_for_frame", "mode": "realtime_stream"},
        processed_status={"status": "ok", "mode": "realtime_stream", "last_frame_id": "pipeline-frame"},
    )
    service = RealtimeEyeService(adapter=pipeline)

    status = service.poll_once()

    assert pipeline.process_count == 1
    assert status["status"] == "ok"
    assert service.latest_observation()["frame_id"] == "pipeline-frame"


def test_realtime_eye_service_status_and_observation_honestly_preserve_not_wired() -> None:
    not_wired_status = RealtimeEyeStatus(
        status="not_wired",
        mode="realtime_stream",
        backend="gstreamer_hailo",
        placeholder=True,
        not_wired=True,
        stream_ready=False,
        status_reason="not_wired",
        not_wired_reason="realtime frame reader is not wired",
        message="realtime frame reader is not wired",
    )
    service = RealtimeEyeService(adapter=FakeAdapter(initial_status=not_wired_status, polled_statuses=[]))

    status = service.status()
    observation = service.latest_observation()

    assert status["status"] == "not_wired"
    assert status["not_wired"] is True
    assert status["stream_ready"] is False
    assert observation["status"] == "not_wired"
    assert observation["tracked_target"] is None
    assert observation["detections"] == []
    assert observation["boxes"] == []
    assert observation["scores"] == []
    assert observation["scene"]["metadata"]["reason"] == "not_wired"
    assert observation["events"] == []


def test_realtime_eye_service_preserves_placeholder_before_scene_bridge() -> None:
    service = RealtimeEyeService(
        adapter=FakeAdapter(
            initial_status={
                "status": "waiting_for_frame",
                "mode": "realtime_stream",
                "placeholder": True,
                "stream_ready": False,
                "last_frame_id": "placeholder-frame",
                "detections": [
                    {
                        "label": "person",
                        "score": 0.91,
                        "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                    }
                ],
            },
            polled_statuses=[],
        )
    )

    observation = service.latest_observation()

    assert observation["placeholder"] is True
    assert observation["scene_bridge"]["live"] is False
    assert observation["scene_bridge"]["reason"] == "placeholder"
    assert observation["events"] == []


def test_realtime_eye_service_scene_cache_respects_live_state_changes() -> None:
    adapter = FakeAdapter(
        initial_status={"status": "waiting_for_frame"},
        polled_statuses=[
            {
                "status": "ok",
                "mode": "realtime_stream",
                "stream_ready": True,
                "last_frame_id": "same-frame",
                "last_frame_captured_at_ts": 123.25,
                "detections": [
                    {
                        "label": "person",
                        "score": 0.91,
                        "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                    }
                ],
            },
            {
                "status": "ok",
                "mode": "realtime_stream",
                "stream_ready": False,
                "stale": True,
                "last_frame_id": "same-frame",
                "last_frame_captured_at_ts": 123.25,
                "detections": [
                    {
                        "label": "person",
                        "score": 0.91,
                        "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                    }
                ],
            },
        ],
    )
    service = RealtimeEyeService(adapter=adapter)

    service.poll_once()
    live = service.latest_observation()
    service.poll_once()
    stale = service.latest_observation()

    assert live["scene_bridge"]["reason"] == "live"
    assert stale["scene_bridge"]["live"] is False
    assert stale["scene_bridge"]["reason"] == "stale"


def test_realtime_eye_service_preserves_degraded_poll_status() -> None:
    service = RealtimeEyeService(
        adapter=FakeAdapter(
            initial_status={"status": "waiting_for_frame"},
            polled_statuses=[
                {
                    "status": "degraded",
                    "mode": "realtime_stream",
                    "degraded": True,
                    "stream_ready": False,
                    "status_reason": "detection_reader_failed",
                    "degraded_reason": "realtime detection reader failed",
                }
            ],
        )
    )

    status = service.poll_once()
    observation = service.latest_observation()

    assert status["status"] == "degraded"
    assert status["degraded"] is True
    assert status["status_reason"] == "detection_reader_failed"
    assert observation["status"] == "degraded"
    assert observation["stream_ready"] is False
    assert observation["degraded"] is True


def test_realtime_eye_service_scene_bridge_diagnostics_are_monitor_consumable() -> None:
    service = RealtimeEyeService(
        adapter=FakeAdapter(
            initial_status={
                "mode": "realtime_stream",
                "status": "ok",
                "backend": "fake_realtime",
                "stream_ready": True,
                "placeholder": False,
                "not_wired": False,
                "last_frame_id": "frame-diag",
                "last_frame_captured_at_ts": 123.25,
                "last_frame_age": 0.07,
                "fps": 14.2,
                "detections": [
                    {
                        "label": "person",
                        "score": 0.91,
                        "confidence": 0.91,
                        "bbox": {"x_min": 0.36, "y_min": 0.20, "x_max": 0.62, "y_max": 0.86},
                    }
                ],
            },
            polled_statuses=[],
        )
    )

    observation = service.latest_observation()
    diagnostics = observation["scene_bridge"]

    assert observation["fps"] == 14.2
    assert observation["last_frame_age"] == 0.07
    assert diagnostics["fps"] == 14.2
    assert diagnostics["frame_age"] == 0.07
    assert diagnostics["track_count"] == 1
    assert diagnostics["stable_target"]["label"] == "person"
    assert diagnostics["event_count"] == 1
    assert diagnostics["last_event"]["eventType"] == "appeared"


def test_realtime_eye_service_is_exported_from_package() -> None:
    from eihead.eye import RealtimeEyeService as ExportedService

    assert ExportedService is RealtimeEyeService


class FakeAdapter:
    def __init__(self, *, initial_status: object, polled_statuses: list[object]) -> None:
        self._status = initial_status
        self._polled_statuses = list(polled_statuses)
        self.poll_count = 0

    def status(self) -> object:
        return self._status

    def poll(self) -> object:
        self.poll_count += 1
        if self._polled_statuses:
            self._status = self._polled_statuses.pop(0)
        return self._status


class FakePipeline:
    def __init__(self, *, initial_status: object, processed_status: object) -> None:
        self._status = initial_status
        self._processed_status = processed_status
        self.process_count = 0

    def status(self) -> object:
        return self._status

    def process_next(self) -> object:
        self.process_count += 1
        self._status = self._processed_status
        return self._status
