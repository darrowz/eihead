from __future__ import annotations

from eihead.monitoring.realtime_vision import build_realtime_vision_payload


def test_monitor_payload_surfaces_native_eye_status_reasons_and_readiness() -> None:
    payload = build_realtime_vision_payload(
        {
            "schema": "eihead.eye.realtime_status.v1",
            "mode": "realtime_stream",
            "status": "degraded",
            "backend": "gstreamer_hailo",
            "last_frame_id": "frame-7",
            "last_frame_age": 0.44,
            "stream_ready": False,
            "stale": False,
            "degraded": True,
            "status_reason": "detection_reader_failed",
            "degraded_reason": "realtime detection reader failed: RuntimeError: parser down",
            "readiness": {"ready": False, "reason": "detection_reader_failed"},
            "detections": [
                {
                    "label": "person",
                    "score": 0.92,
                    "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
                }
            ],
            "detection_boxes": [
                {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
            ],
            "detection_scores": [0.92],
            "placeholder": False,
            "not_wired": False,
            "compatibility_mode": False,
        },
        timestamp=123.0,
        source="eye_realtime",
    )

    assert payload["status"] == "degraded"
    assert payload["wired"] is True
    assert payload["stream_ready"] is False
    assert payload["stale"] is False
    assert payload["degraded"] is True
    assert payload["status_reason"] == "detection_reader_failed"
    assert payload["degraded_reason"] == "realtime detection reader failed: RuntimeError: parser down"
    assert payload["readiness"] == {"ready": False, "reason": "detection_reader_failed"}
    assert payload["boxes"] == [
        {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
    ]
    assert payload["scores"] == [0.92]
    assert payload["diagnostic"]["stream_ready"] is False
    assert payload["diagnostic"]["degraded"] is True
    assert payload["diagnostic"]["status_reason"] == "detection_reader_failed"


def test_monitor_payload_surfaces_tracking_scene_and_multimodal_features() -> None:
    payload = build_realtime_vision_payload(
        {
            "schema": "eihead.eye.realtime_status.v1",
            "mode": "realtime_stream",
            "status": "ok",
            "tracking_stability": {"state": "stable", "score": 0.82},
            "switch_count": 3,
            "lost_count": 1,
            "reacquired_count": 2,
            "scene": {"summary": "person holding mug", "objects": [{"track_id": "person-1", "label": "person"}]},
            "scene_graph_summary": "person holding mug",
            "pose": {"available": True, "summary": "standing"},
            "clip": {"available": False, "reason": "embedding pending"},
            "semanticLabels": [{"label": "person_with_mug"}],
            "depth": {"status": "waiting", "reason": "depth sensor offline"},
            "distance": {"status": "present", "summary": "target 0.8m"},
            "trackingDiagnostics": {"status": "present", "summary": "stable 94%"},
        },
        timestamp=123.0,
        source="eye_realtime",
    )

    assert payload["tracking_stability"]["state"] == "stable"
    assert payload["tracking_switch_count"] == 3
    assert payload["tracking_lost_count"] == 1
    assert payload["tracking_reacquired_count"] == 2
    assert payload["scene_graph_summary"] == "person holding mug"
    assert payload["multimodal_availability"] == {
        "pose": {"status": "present", "summary": "standing"},
        "clip": {"status": "waiting", "summary": "embedding pending"},
        "semantic": {"status": "present", "summary": "1 label(s)"},
        "depth": {"status": "waiting", "summary": "depth sensor offline"},
        "distance": {"status": "present", "summary": "target 0.8m"},
        "tracking": {"status": "present", "summary": "stable 94%"},
    }
