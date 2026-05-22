from __future__ import annotations

from pathlib import Path

from eihead.eye import (
    FaceIdentityMatcher,
    GStreamerHailoRealtimeConfig,
    JsonIdentityRegistry,
    StaticFaceEmbeddingProvider,
)
from eihead.eye.vision_loop import build_vision_state_payload, _identity_observations_from_evidence


def test_vision_state_payload_includes_realtime_evidence_metadata(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    config = GStreamerHailoRealtimeConfig(camera_device="/dev/video42", hailo_device="/dev/hailo0")
    evidence = {
        "frame": {
            "path": "/tmp/eibrain-vision/evidence/native-1-frame.jpg",
            "frame_id": "native-1",
            "captured_at_ts": 123.0,
        },
        "face_crops": [
            {
                "path": "/tmp/eibrain-vision/evidence/native-1-face-0.jpg",
                "frame_id": "native-1",
                "label": "face",
            }
        ],
    }

    payload = build_vision_state_payload(
        {
            "status": "tracking",
            "stream_ready": True,
            "not_wired": False,
            "last_frame_id": "native-1",
            "detections": [{"label": "face", "score": 0.8}],
        },
        config=config,
        config_path="/etc/eihead/eihead.honjia.yaml",
        state_path=state_path,
        interval_s=0.1,
        updated_at_ts=124.0,
        pid=123,
        evidence=evidence,
    )

    assert payload["evidence"] == evidence
    assert payload["status_payload"]["evidence"] == evidence
    assert payload["evidence"]["frame"]["path"].startswith("/tmp/eibrain-vision/evidence/")
    assert payload["evidence"]["frame"]["path"] != "/tmp/eibrain-vision/latest.jpg"


def test_vision_state_payload_includes_visual_identity_observations(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    config = GStreamerHailoRealtimeConfig(camera_device="/dev/video42", hailo_device="/dev/hailo0")
    identity_observations = [
        {
            "known": True,
            "person_id": "person-darrow",
            "display_name": "Darrow",
            "confidence": 0.97,
        }
    ]

    payload = build_vision_state_payload(
        {"status": "tracking", "stream_ready": True, "not_wired": False, "last_frame_id": "native-1"},
        config=config,
        config_path="/etc/eihead/eihead.honjia.yaml",
        state_path=state_path,
        interval_s=0.1,
        updated_at_ts=124.0,
        pid=123,
        identity_observations=identity_observations,
    )

    assert payload["identity_count"] == 1
    assert payload["identity_observations"] == identity_observations
    assert payload["status_payload"]["identity_observations"] == identity_observations


def test_identity_observations_match_face_crops_and_record_known_memory(tmp_path: Path) -> None:
    registry = JsonIdentityRegistry(tmp_path / "people.json")
    registry.enroll_or_update(person_id="person-darrow", display_name="Darrow", embeddings=[[1.0, 0.0, 0.0]])
    matcher = FaceIdentityMatcher(
        registry=registry,
        embedding_provider=StaticFaceEmbeddingProvider({"frame-1:face:0": [0.99, 0.01, 0.0]}, provider_id="test"),
        threshold=0.85,
    )
    memory = _FakeMemoryAdapter()

    observations = _identity_observations_from_evidence(
        {
            "status": "tracking",
            "last_frame_id": "frame-1",
            "last_frame_captured_at_ts": 123.0,
        },
        evidence={
            "face_crops": [
                {
                    "path": str(tmp_path / "frame-1-face-0.jpg"),
                    "frame_id": "frame-1",
                    "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.6},
                    "mime_type": "image/jpeg",
                }
            ]
        },
        matcher=matcher,
        memory_adapter=memory,
    )

    assert observations[0]["known"] is True
    assert observations[0]["person_id"] == "person-darrow"
    assert observations[0]["display_name"] == "Darrow"
    assert observations[0]["frame_id"] == "frame-1"
    assert observations[0]["crop"]["path"].endswith("frame-1-face-0.jpg")
    assert observations[0]["memory"] == {"status": "sent", "memory_id": "mem-1"}
    assert memory.observations[0]["person_id"] == "person-darrow"


class _FakeMemoryAdapter:
    def __init__(self) -> None:
        self.observations: list[dict[str, object]] = []

    def ingest_identity_observation(self, observation: dict[str, object]) -> dict[str, str]:
        self.observations.append(dict(observation))
        return {"status": "sent", "memory_id": "mem-1"}
