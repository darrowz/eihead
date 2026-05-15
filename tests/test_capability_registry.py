from __future__ import annotations

import json

from eihead.monitoring import build_status_snapshot, snapshot_to_json
from eihead.services import (
    CapabilityRegistry,
    DEGRADED,
    OFFLINE,
    ONLINE,
    UNKNOWN,
    manifest_from_config,
    manifest_to_eiprotocol_event,
    manifest_to_json,
)


def test_manifest_marks_default_device_paths_as_unverified_without_live_probe() -> None:
    existing_paths = {"/dev/video0", "/dev/hailo0", "/dev/i2c-1"}

    registry = CapabilityRegistry(
        {"node_id": "honjia"},
        path_exists=existing_paths.__contains__,
        clock=_clock(100.0, 100.015, 100.030, 100.045, 100.060, 100.075, 100.090, 100.105, 100.120, 100.135),
    )

    manifest = registry.manifest()

    assert manifest["schema"] == "eihead.capability_manifest.v1"
    assert manifest["node_id"] == "honjia"
    assert manifest["capabilities"]["camera"]["status"] == UNKNOWN
    assert manifest["capabilities"]["hailo"]["status"] == UNKNOWN
    assert manifest["capabilities"]["i2c"]["status"] == UNKNOWN
    assert manifest["capabilities"]["neck"]["status"] == UNKNOWN
    assert manifest["capabilities"]["camera"]["details"]["hardware_verified"] is False
    assert manifest["capabilities"]["camera"]["details"]["reason"] == "all_declared_paths_exist_unverified"
    assert manifest["capabilities"]["camera"]["latency_ms"] >= 0
    assert manifest["capabilities"]["camera"]["last_ok_ts"] is None


def test_manifest_supports_declarative_software_capabilities() -> None:
    manifest = manifest_from_config(
        {
            "node_id": "honjia",
            "capabilities": {
                "asr": {
                    "enabled": True,
                    "provider": "sherpa-onnx",
                    "model": "streaming-zipformer",
                    "limits": {"streaming": True, "languages": ["zh"]},
                },
                "tts": {
                    "status": "degraded",
                    "provider": "minimax",
                    "error": "quota warning",
                    "limits": {"streaming": True},
                },
                "vision_backend": {
                    "enabled": True,
                    "backend": "hailo",
                    "limits": {"realtime": True, "max_fps": 15},
                },
            },
        },
        path_exists=lambda _path: False,
        clock=_clock(200.0),
    )

    asr = manifest["capabilities"]["asr"]
    tts = manifest["capabilities"]["tts"]
    vision = manifest["capabilities"]["vision_backend"]

    assert asr["status"] == UNKNOWN
    assert asr["details"]["provider"] == "sherpa-onnx"
    assert asr["limits"]["streaming"] is True
    assert tts["status"] == DEGRADED
    assert tts["error"] == "quota warning"
    assert vision["details"]["backend"] == "hailo"
    assert vision["limits"]["max_fps"] == 15


def test_manifest_reports_missing_and_partially_available_paths() -> None:
    registry = CapabilityRegistry(
        {
            "capabilities": {
                "camera": {"paths": ["/dev/video0", "/dev/video1"]},
                "microphone": {"path": "/dev/snd/pcmC2D0c"},
                "speaker": {"enabled": False},
            }
        },
        path_exists=lambda path: path in {"/dev/video0"},
        clock=_clock(300.0),
    )

    manifest = registry.manifest()

    assert manifest["capabilities"]["camera"]["status"] == DEGRADED
    assert manifest["capabilities"]["camera"]["details"]["available_paths"] == ["/dev/video0"]
    assert "/dev/video1" in manifest["capabilities"]["camera"]["error"]
    assert manifest["capabilities"]["microphone"]["status"] == OFFLINE
    assert manifest["capabilities"]["speaker"]["status"] == OFFLINE
    assert manifest["capabilities"]["speaker"]["error"] == "disabled"


def test_status_snapshot_summarizes_manifest_for_monitoring() -> None:
    manifest = {
        "schema": "eihead.capability_manifest.v1",
        "node_id": "honjia",
        "generated_at_ts": 400.0,
        "capabilities": {
            "camera": {"status": ONLINE},
            "tts": {"status": DEGRADED},
            "microphone": {"status": OFFLINE},
        },
    }

    snapshot = build_status_snapshot(manifest, clock=lambda: 401.0)

    assert snapshot["schema"] == "eihead.status_snapshot.v1"
    assert snapshot["node_id"] == "honjia"
    assert snapshot["overall_status"] == DEGRADED
    assert snapshot["summary"] == {ONLINE: 1, DEGRADED: 1, OFFLINE: 1, "total": 3}
    assert snapshot["manifest_generated_at_ts"] == 400.0


def test_manifest_and_snapshot_are_json_serializable() -> None:
    registry = CapabilityRegistry({"capabilities": {"asr": {"enabled": True}}}, clock=_clock(500.0))
    manifest_json = manifest_to_json(registry.manifest())
    snapshot_json = snapshot_to_json(build_status_snapshot(registry, clock=lambda: 501.0))

    assert json.loads(manifest_json)["capabilities"]["asr"]["status"] == UNKNOWN
    assert json.loads(snapshot_json)["overall_status"] == DEGRADED


def test_eiprotocol_conversion_preserves_plain_manifest_shape() -> None:
    existing_paths = {"/dev/video0", "/dev/hailo0", "/dev/i2c-1"}
    manifest = CapabilityRegistry(
        {"node_id": "honjia"},
        path_exists=existing_paths.__contains__,
        clock=_clock(600.0),
    ).manifest()
    original = json.loads(json.dumps(manifest))

    manifest_to_eiprotocol_event(
        manifest,
        event_id="evt_capability_registry",
        request_id="req_capability_registry",
        sequence=11,
        time="2026-05-04T10:31:00+08:00",
    )

    assert manifest == original
    assert set(manifest) == {"schema", "node_id", "generated_at_ts", "capabilities"}
    assert set(manifest["capabilities"]["camera"]) == {
        "name",
        "kind",
        "status",
        "latency_ms",
        "last_ok_ts",
        "error",
        "limits",
        "details",
    }
    assert manifest["capabilities"]["neck"]["limits"] == {"pan_deg": [0, 180], "tilt_deg": None}


def test_registry_eiprotocol_manifest_round_trips_for_honjia_devices() -> None:
    from eiprotocol import EventEnvelope, validate_event

    existing_paths = {"/dev/video0", "/dev/hailo0", "/dev/i2c-1"}
    registry = CapabilityRegistry(
        {"node_id": "honjia"},
        path_exists=existing_paths.__contains__,
        clock=_clock(700.0),
    )

    event = registry.eiprotocol_manifest(
        event_id="evt_capability_registry",
        request_id="req_capability_registry",
        sequence=12,
        time="2026-05-04T10:31:00+08:00",
    )
    payload = json.loads(event.to_json())
    restored = EventEnvelope.from_dict(payload)
    capabilities = {item["capabilityId"]: item for item in payload["content"]["capabilities"]}

    assert payload["name"] == "ei.capability.manifest.report"
    assert payload["source"]["domain"] == "eihead"
    assert payload["source"]["instanceId"] == "honjia"
    assert payload["source"]["deviceId"] == "honjia"
    assert payload["content"]["device"]["deviceId"] == "honjia"
    assert capabilities["camera.front"]["devicePath"] == "/dev/video0"
    assert capabilities["accelerator.hailo"]["devicePath"] == "/dev/hailo0"
    assert capabilities["bus.i2c"]["devicePath"] == "/dev/i2c-1"
    assert capabilities["microphone.default"]["kind"] == "audio_input"
    assert capabilities["speaker.default"]["kind"] == "audio_output"
    assert capabilities["neck.pan"]["limits"] == {
        "axis": "pan",
        "minAngle": 0,
        "maxAngle": 180,
        "tiltSupported": False,
    }
    assert restored.to_dict() == payload
    assert validate_event(restored) == []


def _clock(*values: float):
    timestamps = list(values) or [0.0]

    def tick() -> float:
        if len(timestamps) > 1:
            return timestamps.pop(0)
        return timestamps[0]

    return tick
