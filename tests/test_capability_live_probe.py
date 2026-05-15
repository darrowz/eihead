from __future__ import annotations

from typing import Any

from eihead.services import CapabilityRegistry


def test_live_probe_overrides_static_path_status_and_records_truth_metadata() -> None:
    calls: list[dict[str, Any]] = []

    def fake_probe(name: str, *, config: dict[str, Any], static_status: dict[str, Any]) -> dict[str, Any] | None:
        calls.append({"name": name, "config": config, "static_status": static_status})
        if name == "camera":
            return {
                "status": "live",
                "source": "fake-camera-live-probe",
                "checked_at": 1234.5,
                "reason": "frame_capture_ok",
                "hardware_verified": True,
                "details": {"fps": 30},
            }
        if name == "microphone":
            return {
                "status": "unknown",
                "source": "fake-audio-live-probe",
                "checked_at": 1234.75,
                "reason": "probe_not_configured",
                "hardware_verified": False,
            }
        return None

    registry = CapabilityRegistry(
        {
            "node_id": "honjia-live-test",
            "capabilities": {
                "camera": {"paths": ["/missing/video0"]},
                "microphone": {"enabled": True, "limits": {"channels": 2}},
                "asr": {"enabled": True, "provider": "fake-asr"},
            },
        },
        path_exists=lambda _path: False,
        probe=fake_probe,
        clock=_clock(1000.0, 1000.01, 1000.02, 1000.03, 1000.04, 1000.05, 1000.06),
    )

    manifest = registry.manifest()

    camera = manifest["capabilities"]["camera"]
    assert camera["status"] == "live"
    assert camera["error"] is None
    assert camera["details"]["source"] == "fake-camera-live-probe"
    assert camera["details"]["checked_at"] == 1234.5
    assert camera["details"]["last_checked"] == 1234.5
    assert camera["details"]["reason"] == "frame_capture_ok"
    assert camera["details"]["hardware_verified"] is True
    assert camera["details"]["fps"] == 30
    assert camera["details"]["paths"] == ["/missing/video0"]

    microphone = manifest["capabilities"]["microphone"]
    assert microphone["status"] == "unknown"
    assert microphone["error"] == "probe_not_configured"
    assert microphone["details"]["source"] == "fake-audio-live-probe"
    assert microphone["details"]["hardware_verified"] is False
    assert microphone["limits"]["channels"] == 2

    asr = manifest["capabilities"]["asr"]
    assert asr["status"] == "unknown"
    assert asr["details"]["source"] == "static_config"
    assert asr["details"]["hardware_verified"] is False
    assert asr["details"]["reason"] == "declared_without_live_probe"

    camera_call = next(call for call in calls if call["name"] == "camera")
    assert camera_call["static_status"]["status"] == "offline"
    assert camera_call["static_status"]["details"]["hardware_verified"] is False


def _clock(*values: float):
    timestamps = list(values) or [0.0]

    def tick() -> float:
        if len(timestamps) > 1:
            return timestamps.pop(0)
        return timestamps[0]

    return tick
