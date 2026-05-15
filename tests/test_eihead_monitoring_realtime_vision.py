from __future__ import annotations

from eihead.eye import RealtimeEyeService
from eihead.monitoring import build_realtime_vision_payload
from eihead.eye import RealtimeVisionSceneBridge


def test_realtime_vision_payload_exposes_visual_overlay_for_box_diagnostics() -> None:
    payload = build_realtime_vision_payload(
        {
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "status": "tracking",
            "frame_id": "frame-overlay-1",
            "width": 640,
            "height": 480,
            "fps": 30.0,
            "detections": [
                {
                    "label": "person",
                    "score": 0.91,
                    "bbox": {"x_min": 160, "y_min": 120, "x_max": 320, "y_max": 360},
                },
                {
                    "label": "dog",
                    "confidence": 0.6,
                    "bbox": {"x_min": 0.6, "y_min": 0.25, "x_max": 0.8, "y_max": 0.75},
                },
                {
                    "label": "face",
                    "score": 0.5,
                    "bbox": {"x": 320, "y": 120, "w": 160, "h": 120},
                },
            ],
        },
        timestamp=1000.0,
        source="eye_realtime",
    )

    assert payload["overlay"] == payload["visual_diagnostic"]
    assert payload["overlay"]["frame"] == {
        "width": 640,
        "height": 480,
        "frame_id": "frame-overlay-1",
        "image_available": False,
        "image_message": "no live frame image yet",
    }
    assert payload["overlay"]["stream_ready"] is True
    assert payload["overlay"]["normalized_boxes"] == [
        {
            "label": "person",
            "score": 0.91,
            "score_label": "person 0.91",
            "x_min": 0.25,
            "y_min": 0.25,
            "x_max": 0.5,
            "y_max": 0.75,
        },
        {
            "label": "dog",
            "score": 0.6,
            "score_label": "dog 0.60",
            "x_min": 0.6,
            "y_min": 0.25,
            "x_max": 0.8,
            "y_max": 0.75,
        },
        {
            "label": "face",
            "score": 0.5,
            "score_label": "face 0.50",
            "x_min": 0.5,
            "y_min": 0.25,
            "x_max": 0.75,
            "y_max": 0.5,
        },
    ]
    assert payload["overlay"]["score_labels"] == ["person 0.91", "dog 0.60", "face 0.50"]
    assert payload["overlay"]["top_target"] == {
        "label": "person",
        "score": 0.91,
        "score_label": "person 0.91",
        "center": {"x": 0.375, "y": 0.5},
        "error": {"x": -0.125, "y": 0.0},
    }


def test_realtime_vision_payload_overlay_accepts_protocol_normalized_list_xywh_bbox() -> None:
    payload = build_realtime_vision_payload(
        {
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "status": "tracking",
            "frame_id": "frame-list-box",
            "width": 640,
            "height": 480,
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.91,
                    "bbox": [0.62, 0.35, 0.12, 0.18],
                },
            ],
        },
        timestamp=1000.0,
        source="eye_realtime",
    )

    assert payload["overlay"]["normalized_boxes"] == [
        {
            "label": "person",
            "score": 0.91,
            "score_label": "person 0.91",
            "x_min": 0.62,
            "y_min": 0.35,
            "x_max": 0.74,
            "y_max": 0.53,
        }
    ]
    assert payload["overlay"]["top_target"]["center"] == {"x": 0.68, "y": 0.44}


def test_realtime_vision_payload_overlay_honors_explicit_normalized_list_xyxy_bbox() -> None:
    payload = build_realtime_vision_payload(
        {
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "status": "tracking",
            "frame_id": "frame-list-box-xyxy",
            "width": 640,
            "height": 480,
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.91,
                    "bbox": [0.62, 0.35, 0.74, 0.53],
                    "bboxFormat": "xyxy",
                },
            ],
        },
        timestamp=1000.0,
        source="eye_realtime",
    )

    assert payload["overlay"]["normalized_boxes"][0] == {
        "label": "person",
        "score": 0.91,
        "score_label": "person 0.91",
        "x_min": 0.62,
        "y_min": 0.35,
        "x_max": 0.74,
        "y_max": 0.53,
    }
    assert payload["overlay"]["top_target"]["center"] == {"x": 0.68, "y": 0.44}


def test_realtime_vision_payload_surfaces_detection_level_multimodal_availability() -> None:
    payload = build_realtime_vision_payload(
        {
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "status": "tracking",
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.91,
                    "bbox": {"x_min": 0.2, "y_min": 0.2, "x_max": 0.4, "y_max": 0.8},
                    "attributes": {
                        "pose": {"available": True, "summary": "standing"},
                        "clipLabels": [{"label": "person at desk"}],
                        "semanticLabels": [{"label": "workspace"}],
                        "depth": {"status": "waiting", "reason": "depth sensor offline"},
                        "distance": {"status": "present", "summary": "target 0.8m"},
                        "trackingDiagnostics": {"status": "present", "summary": "stable 94%"},
                    },
                }
            ],
        },
        timestamp=1000.0,
        source="eye_realtime",
    )

    assert payload["multimodal_availability"] == {
        "pose": {"status": "present", "summary": "standing"},
        "clip": {"status": "present", "summary": "1 label(s)"},
        "semantic": {"status": "present", "summary": "1 label(s)"},
        "depth": {"status": "waiting", "summary": "depth sensor offline"},
        "distance": {"status": "present", "summary": "target 0.8m"},
        "tracking": {"status": "present", "summary": "stable 94%"},
    }


def test_realtime_vision_payload_promotes_scene_event_track_and_target_summaries() -> None:
    payload = build_realtime_vision_payload(
        {
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "status": "tracking",
            "frame_id": "frame-scene-1",
            "sceneSnapshot": {
                "scene_id": "scene-legacy-1",
                "summary": "person near doorway",
            },
            "sceneGraphSummary": "person near doorway with dog",
            "events": [
                {"event_type": "person_entered", "score": 0.88},
                {"type": "dog_seen"},
            ],
            "tracks": [
                {"track_id": "track-person-1", "label": "person", "score": 0.93},
                {"id": "track-dog-1", "label": "dog", "confidence": 0.64},
            ],
            "detections": [
                {
                    "label": "person",
                    "score": 0.93,
                    "bbox": {"x_min": 0.2, "y_min": 0.25, "x_max": 0.5, "y_max": 0.85},
                },
                {
                    "label": "dog",
                    "score": 0.64,
                    "bbox": {"x_min": 0.6, "y_min": 0.3, "x_max": 0.8, "y_max": 0.65},
                },
            ],
        },
        timestamp=1000.0,
        source="eye_realtime",
    )

    assert payload["scene_id"] == "scene-legacy-1"
    assert payload["scene_summary"] == "person near doorway with dog"
    assert payload["event_count"] == 2
    assert payload["event_summary"] == "person_entered 0.88, dog_seen"
    assert payload["track_count"] == 2
    assert payload["track_summary"] == "person 0.93, dog 0.64"
    assert payload["score_labels"] == ["person 0.93", "dog 0.64"]
    assert payload["target_center"] == {"x": 0.35, "y": 0.55}
    assert payload["target_error"] == {"x": -0.15, "y": 0.05}
    assert payload["target_score_label"] == "person 0.93"
    assert payload["diagnostic"]["scene"]["scene_id"] == "scene-legacy-1"
    assert payload["diagnostic"]["scene"]["summary"] == "person near doorway with dog"


def test_realtime_vision_payload_accepts_normalized_scene_bridge_output() -> None:
    payload = build_realtime_vision_payload(
        {
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "status": "tracking",
            "scene": {
                "id": "scene-bridge-1",
                "summary": "bridge summary",
                "events": [{"name": "gaze_locked"}],
                "tracks": [{"id": "bridge-track-1", "label": "face"}],
            },
            "target": {
                "label": "face",
                "score": 0.86,
                "center": {"x": 0.52, "y": 0.48},
                "error": {"x": 0.02, "y": -0.02},
            },
        },
        timestamp=1000.0,
        source="eye_realtime",
    )

    assert payload["scene_id"] == "scene-bridge-1"
    assert payload["scene_summary"] == "bridge summary"
    assert payload["events"] == {"count": 1, "items": [{"name": "gaze_locked"}], "summary": "gaze_locked"}
    assert payload["tracks"] == {"count": 1, "items": [{"id": "bridge-track-1", "label": "face"}], "summary": "face"}
    assert payload["target_center"] == {"x": 0.52, "y": 0.48}
    assert payload["target_error"] == {"x": 0.02, "y": -0.02}
    assert payload["target_score_label"] == "face 0.86"


def test_realtime_vision_payload_accepts_direct_scene_bridge_result() -> None:
    bridge_result = RealtimeVisionSceneBridge().update(
        {
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "status": "ok",
            "stream_ready": True,
            "frame_id": "frame-bridge-direct",
            "observed_at": "2026-05-06T01:00:00.000+08:00",
            "detections": [
                {
                    "label": "person",
                    "score": 0.92,
                    "bbox": {"x_min": 0.2, "y_min": 0.2, "x_max": 0.4, "y_max": 0.7},
                }
            ],
        }
    )

    payload = build_realtime_vision_payload(
        bridge_result,
        timestamp=1000.0,
        source="scene_bridge",
    )

    assert payload["wired"] is True
    assert payload["scene_id"] == bridge_result["latest_scene_id"]
    assert payload["scene_summary"] == bridge_result["sceneGraphSummary"]
    assert payload["event_count"] >= 1
    assert "appeared" in payload["event_summary"]
    assert payload["track_count"] == 1
    assert payload["track_summary"] == "person 0.92"
    assert payload["target_score_label"] == "person 0.92"


def test_realtime_vision_payload_reads_scene_augmented_service_observation() -> None:
    service = RealtimeEyeService(
        adapter=_FakeAdapter(
            initial_status={
                "kind": "realtime_vision_observation",
                "mode": "realtime_stream",
                "status": "ok",
                "stream_ready": True,
                "last_frame_id": "service-frame",
                "last_frame_captured_at_ts": 123.25,
                "detections": [
                    {
                        "label": "person",
                        "score": 0.93,
                        "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                    }
                ],
            }
        )
    )

    payload = build_realtime_vision_payload(service, timestamp=200.0, source="eye_realtime")

    assert payload["wired"] is True
    assert payload["scene_id"].startswith("scene_rt_")
    assert "person" in payload["scene_summary"]
    assert payload["event_count"] >= 1
    assert payload["track_count"] == 1
    assert payload["target_score_label"] == "person 0.93"


def test_realtime_vision_payload_marks_wired_waiting_stream_as_degraded() -> None:
    payload = build_realtime_vision_payload(
        {
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "status": "waiting_for_frame",
            "stream_ready": False,
            "not_wired": False,
            "message": "no realtime frame available",
        },
        timestamp=1000.0,
        source="eye_realtime",
    )

    assert payload["status"] == "degraded"
    assert payload["wired"] is True
    assert payload["not_wired"] is False
    assert payload["stream_ready"] is False
    assert payload["degraded"] is True
    assert payload["degraded_reason"] == "no realtime frame available"


def test_realtime_vision_payload_keeps_non_live_scene_bridge_outputs_unwired() -> None:
    bridge = RealtimeVisionSceneBridge()

    for overrides in (
        {"mode": "compat_static_frame", "status": "compat_static", "compatibility_mode": True},
        {"status": "static"},
    ):
        bridge_result = bridge.update(
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime_stream",
                "frame_id": "non-live-bridge",
                "observed_at": "2026-05-06T01:00:00.000+08:00",
                "detections": [
                    {
                        "label": "person",
                        "score": 0.92,
                        "bbox": {"x_min": 0.2, "y_min": 0.2, "x_max": 0.4, "y_max": 0.7},
                    }
                ],
                **overrides,
            }
        )

        payload = build_realtime_vision_payload(bridge_result, timestamp=1000.0, source="scene_bridge")

        assert payload["wired"] is False
        assert payload["status"] == "compat_static"
        assert payload["event_count"] == 0
        assert payload["track_count"] == 0


class _FakeAdapter:
    def __init__(self, *, initial_status: object) -> None:
        self._status = initial_status

    def status(self) -> object:
        return self._status
