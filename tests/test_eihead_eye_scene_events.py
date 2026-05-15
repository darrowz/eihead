from __future__ import annotations

import pytest

from eihead.eye import RealtimeVisionSceneBridge


def _observation(
    *,
    frame_id: str,
    observed_at: str,
    detections: list[dict[str, object]],
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "realtime_vision_observation",
        "mode": "realtime_stream",
        "status": "ok",
        "stream_ready": True,
        "not_wired": False,
        "placeholder": False,
        "stale": False,
        "compatibility_mode": False,
        "frame_id": frame_id,
        "observed_at": observed_at,
        "detections": detections,
        "frame_payload": b"not-for-protocol",
        "image": "raw-image-data",
    }
    payload.update(overrides)
    return payload


def _det(label: str, bbox: tuple[float, float, float, float], confidence: float) -> dict[str, object]:
    x_min, y_min, x_max, y_max = bbox
    return {
        "label": label,
        "confidence": confidence,
        "bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
    }


def test_scene_bridge_emits_protocol_scene_and_events_without_raw_frames() -> None:
    bridge = RealtimeVisionSceneBridge()

    result = bridge.update(
        _observation(
            frame_id="frame-001",
            observed_at="2026-05-05T10:00:00.000+08:00",
            detections=[_det("person", (0.10, 0.20, 0.30, 0.70), 0.91)],
        )
    )

    scene = result["scene_snapshot"]

    assert result["live"] is True
    assert result["latest_scene_id"] == scene["sceneId"]
    assert result["sceneGraphSummary"] == scene["summary"]
    assert result["object_count"] == 1
    assert result["track_count"] == 1
    assert scene["observedAt"] == "2026-05-05T10:00:00.000+08:00"
    assert scene["metadata"]["frameId"] == "frame-001"
    assert scene["objects"][0]["label"] == "person"
    assert scene["objects"][0]["confidence"] == 0.91
    assert scene["objects"][0]["bbox"] == {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.7}
    assert result["event_contents"]
    assert result["events"] == result["event_contents"]
    assert result["event_contents"][0]["sceneId"] == result["latest_scene_id"]
    assert "frame_payload" not in str(result)
    assert "raw-image-data" not in str(result)


def test_scene_bridge_keeps_simulator_track_ids_stable_across_frames() -> None:
    bridge = RealtimeVisionSceneBridge(move_threshold=0.08)

    first = bridge.update(
        _observation(
            frame_id="frame-001",
            observed_at="2026-05-05T10:00:00.000+08:00",
            detections=[_det("cup", (0.10, 0.20, 0.20, 0.35), 0.87)],
        )
    )
    second = bridge.update(
        _observation(
            frame_id="frame-002",
            observed_at="2026-05-05T10:00:00.100+08:00",
            detections=[_det("cup", (0.32, 0.22, 0.42, 0.37), 0.88)],
        )
    )

    first_track = first["scene_snapshot"]["objects"][0]["trackId"]
    second_track = second["scene_snapshot"]["objects"][0]["trackId"]

    assert first_track == second_track
    assert second["scene_snapshot"]["objects"][0]["label"] == "cup"
    assert second["scene_snapshot"]["objects"][0]["confidence"] == 0.88
    assert second["scene_snapshot"]["metadata"]["frameId"] == "frame-002"
    assert second["event_contents"][0]["eventType"] == "moved"
    assert second["event_contents"][0]["subject"]["trackId"] == first_track


def test_scene_bridge_keeps_snapshot_but_suppresses_unchanged_frame_events() -> None:
    bridge = RealtimeVisionSceneBridge(move_threshold=0.08, max_missing_frames=1)

    first = bridge.update(
        _observation(
            frame_id="frame-001",
            observed_at="2026-05-05T10:00:00.000+08:00",
            detections=[_det("person", (0.40, 0.20, 0.60, 0.80), 0.91)],
            fps=12.5,
            last_frame_age=0.04,
        )
    )
    second = bridge.update(
        _observation(
            frame_id="frame-002",
            observed_at="2026-05-05T10:00:00.100+08:00",
            detections=[_det("person", (0.405, 0.205, 0.605, 0.805), 0.90)],
            fps=12.5,
            last_frame_age=0.03,
        )
    )

    track_id = first["scene_snapshot"]["objects"][0]["trackId"]

    assert second["scene_snapshot"]["objects"][0]["trackId"] == track_id
    assert second["scene_snapshot"]["objects"][0]["temporalState"] == "stationary"
    assert second["event_contents"] == []
    assert second["event_count"] == 0
    assert second["last_event"] is None
    assert second["stable_target"]["trackId"] == track_id
    assert second["diagnostics"]["fps"] == 12.5
    assert second["diagnostics"]["frame_age"] == 0.03
    assert second["diagnostics"]["track_count"] == 1


def test_scene_bridge_exposes_hailo_tracking_diagnostics_for_trace_consumers() -> None:
    bridge = RealtimeVisionSceneBridge()

    result = bridge.update(
        _observation(
            frame_id="frame-hailo-001",
            observed_at="2026-05-05T10:00:00.000+08:00",
            detections=[_det("person", (0.40, 0.20, 0.60, 0.80), 0.91)],
            fps=15.0,
            frame_age_ms=88.0,
            hailo_metadata={"device": "hailo8l", "model": "yolov8n"},
            soak_summary={"track_id_switch_count": 0, "target_stability_ratio": 1.0},
        )
    )

    diagnostics = result["diagnostics"]

    assert diagnostics["p95_frame_age_ms"] == 88.0
    assert diagnostics["track_id_switch_count"] == 0
    assert diagnostics["target_stability_ratio"] == 1.0
    assert diagnostics["hailo_metadata"] == {"device": "hailo8l", "model": "yolov8n"}
    assert diagnostics["trace"]["kind"] == "vision_tracking_diagnostics"
    assert result["scene_snapshot"]["metadata"]["hailo"]["device"] == "hailo8l"
    assert result["scene_snapshot"]["metadata"]["soak_summary"]["target_stability_ratio"] == 1.0


def test_scene_bridge_preserves_multimodal_detection_fields_in_scene_and_event_metadata() -> None:
    bridge = RealtimeVisionSceneBridge()

    result = bridge.update(
        _observation(
            frame_id="frame-mm-001",
            observed_at="2026-05-05T10:00:00.000+08:00",
            detections=[
                {
                    "label": "person",
                    "confidence": 0.93,
                    "bbox": {"x_min": 0.20, "y_min": 0.18, "x_max": 0.58, "y_max": 0.92},
                    "pose": {
                        "keypoints": [
                            {"name": "right_wrist", "x": 0.61, "y": 0.58, "confidence": 0.76},
                        ]
                    },
                    "clip_labels": ["person"],
                    "semantic_labels": ["human"],
                    "depth_m": 0.72,
                    "distance_band": "near",
                    "looking_at_device": True,
                    "source": "hailo",
                    "model_id": "pose-clip-placeholder",
                    "provenance": {"device": "hailo8l"},
                },
                {
                    "label": "phone",
                    "confidence": 0.84,
                    "bbox": {"x_min": 0.58, "y_min": 0.52, "x_max": 0.68, "y_max": 0.66},
                    "semantic_labels": ["device"],
                    "depth_m": 0.78,
                },
            ],
        )
    )

    person = next(item for item in result["scene_snapshot"]["objects"] if item["label"] == "person")
    event_types = {event["eventType"] for event in result["event_contents"]}

    assert person["pose"]["keypoints"] == [{"name": "right_wrist", "x": 0.61, "y": 0.58, "confidence": 0.76}]
    assert person["clip_labels"] == [{"label": "person"}]
    assert person["semantic_labels"] == ["human"]
    assert person["depth_m"] == 0.72
    assert person["distance_band"] == "near"
    assert person["source"] == "hailo"
    assert person["model_id"] == "pose-clip-placeholder"
    assert person["provenance"]["device"] == "hailo8l"
    assert "looking_at_device" in event_types
    assert "hand_near_object" in event_types
    assert result["event_summary"]


def test_scene_bridge_uses_track_id_for_multimodal_matching_and_unique_object_event_ids() -> None:
    bridge = RealtimeVisionSceneBridge()

    result = bridge.update(
        _observation(
            frame_id="frame-multi-person",
            observed_at="2026-05-05T10:00:00.000+08:00",
            detections=[
                {
                    "label": "person",
                    "confidence": 0.90,
                    "bbox": {"x_min": 0.10, "y_min": 0.20, "x_max": 0.30, "y_max": 0.80},
                    "trackId": "person-A",
                    "model_id": "pose-A",
                    "pose": {"keypoints": [{"name": "right_wrist", "x": 0.50, "y": 0.48}]},
                },
                {
                    "label": "person",
                    "confidence": 0.89,
                    "bbox": {"x_min": 0.32, "y_min": 0.20, "x_max": 0.52, "y_max": 0.80},
                    "trackId": "person-B",
                    "model_id": "pose-B",
                    "pose": {"keypoints": [{"name": "right_wrist", "x": 0.70, "y": 0.48}]},
                },
                {
                    "label": "cup",
                    "confidence": 0.80,
                    "bbox": {"x_min": 0.48, "y_min": 0.45, "x_max": 0.55, "y_max": 0.56},
                    "trackId": "cup-1",
                },
                {
                    "label": "phone",
                    "confidence": 0.81,
                    "bbox": {"x_min": 0.68, "y_min": 0.45, "x_max": 0.75, "y_max": 0.56},
                    "trackId": "phone-1",
                },
            ],
        )
    )

    people = {item["sourceTrackId"]: item for item in result["scene_snapshot"]["objects"] if item["label"] == "person"}
    hand_events = [event for event in result["event_contents"] if event["eventType"] == "hand_near_object"]

    assert people["person-A"]["model_id"] == "pose-A"
    assert people["person-B"]["model_id"] == "pose-B"
    assert len(hand_events) >= 2
    assert len({event["eventId"] for event in hand_events}) == len(hand_events)
    assert {event["details"]["objectId"] for event in hand_events} >= {"cup-001", "phone-001"}


def test_scene_bridge_accepts_protocol_bbox_depth_distance_and_clip_labels() -> None:
    bridge = RealtimeVisionSceneBridge()

    result = bridge.update(
        _observation(
            frame_id="frame-protocol-001",
            observed_at="2026-05-05T10:00:00.000+08:00",
            detections=[
                {
                    "label": "person",
                    "confidence": 0.93,
                    "bbox": {"x": 0.20, "y": 0.18, "w": 0.38, "h": 0.74},
                    "clipLabels": [{"label": "person at desk", "score": 0.84}],
                    "semanticLabels": [{"label": "workspace", "confidence": 0.79}],
                    "depth": {"median": 0.72, "unit": "m"},
                    "distance": {"fromCameraM": 0.72},
                }
            ],
        )
    )

    person = result["scene_snapshot"]["objects"][0]

    assert person["bbox"] == {"x_min": 0.2, "y_min": 0.18, "x_max": 0.58, "y_max": 0.92}
    assert person["clip_labels"][0]["label"] == "person at desk"
    assert person["semantic_labels"][0] == "workspace"
    assert person["depth_m"] == 0.72
    assert person["distance_band"] == "near"


def test_scene_bridge_accepts_protocol_list_bbox_and_tracking_diagnostics() -> None:
    bridge = RealtimeVisionSceneBridge()

    result = bridge.update(
        _observation(
            frame_id="frame-list-bbox",
            observed_at="2026-05-05T10:00:00.000+08:00",
            detections=[
                {
                    "label": "person",
                    "score": 0.91,
                    "bbox": [0.62, 0.35, 0.12, 0.18],
                    "trackingDiagnostics": {
                        "trackIdSwitchCount": 0,
                        "targetStabilityRatio": 1.0,
                    },
                }
            ],
        )
    )

    person = result["scene_snapshot"]["objects"][0]

    assert person["bbox"] == {"x_min": 0.62, "y_min": 0.35, "x_max": 0.74, "y_max": 0.53}
    assert person["trackingDiagnostics"]["targetStabilityRatio"] == 1.0
    assert result["scene_snapshot"]["trackingDiagnostics"]["trackIdSwitchCount"] == 0
    assert result["event_contents"][0]["trackingDiagnostics"]["targetStabilityRatio"] == 1.0


def test_scene_bridge_accepts_latest_status_dicts_from_realtime_eye_service() -> None:
    bridge = RealtimeVisionSceneBridge()

    result = bridge.update(
        {
            "mode": "realtime_stream",
            "status": "ok",
            "stream_ready": True,
            "not_wired": False,
            "placeholder": False,
            "stale": False,
            "last_frame_id": "frame-003",
            "last_frame_captured_at_ts": 1777975200.125,
            "detections": [_det("book", (0.66, 0.58, 0.82, 0.72), 0.86)],
            "payload": "must-not-leak",
        }
    )

    assert result["live"] is True
    assert result["frame_id"] == "frame-003"
    assert result["scene_snapshot"]["observedAt"] == "2026-05-05T10:00:00.125+00:00"
    assert result["scene_snapshot"]["objects"][0]["label"] == "book"
    assert "must-not-leak" not in str(result)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"not_wired": True, "status": "not_wired", "stream_ready": False}, "not_wired"),
        ({"placeholder": True, "backend": "placeholder"}, "placeholder"),
        ({"mode": "compat_static_frame", "status": "compat_static", "compatibility_mode": True}, "compat_static"),
        ({"status": "static"}, "static"),
        ({"stale": True, "status": "stale"}, "stale"),
    ],
)
def test_scene_bridge_marks_non_live_observations_without_durable_events(
    overrides: dict[str, object],
    reason: str,
) -> None:
    bridge = RealtimeVisionSceneBridge()

    result = bridge.update(
        _observation(
            frame_id="non-live-frame",
            observed_at="2026-05-05T10:00:00.000+08:00",
            detections=[_det("person", (0.10, 0.20, 0.30, 0.70), 0.91)],
            **overrides,
        )
    )

    assert result["live"] is False
    assert result["reason"] == reason
    assert result["latest_scene_id"] == ""
    assert result["scene_snapshot"]["objects"] == []
    assert result["event_contents"] == []
    assert result["events"] == []
    assert result["object_count"] == 0
    assert result["track_count"] == 0
