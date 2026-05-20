from __future__ import annotations

from contextlib import contextmanager
import json
import threading
import time
from typing import Any, Iterator
from urllib import request

import pytest

from eihead.protocol import RealtimeVisionObservation, VisionObservation
from eihead.runtime.app import HeadRuntimeApp
from eihead.runtime.http_api import create_server


class FakeBodyRuntime:
    def snapshot(self) -> dict[str, object]:
        return {"node_id": "honjia-test"}


class RealtimeObservationRuntime(FakeBodyRuntime):
    def vision_realtime(self) -> RealtimeVisionObservation:
        return RealtimeVisionObservation(
            ts=1.0,
            source="eihead.honjia.eye.realtime",
            stream_id="front-main",
            status="tracking",
            frame_id="live-1",
        )


class MappingRealtimeRuntime(FakeBodyRuntime):
    eye_realtime = {
        "kind": "realtime_vision_observation",
        "mode": "realtime",
        "primary_mode": True,
        "stream_id": "front-main",
        "status": "tracking",
    }


class AppHookHeadRuntime(HeadRuntimeApp):
    def eye_realtime(self) -> dict[str, object]:
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime",
            "primary_mode": True,
            "stream_id": "app-hook",
            "status": "tracking",
        }


class SnapshotEyeRealtimeRuntime(FakeBodyRuntime):
    def snapshot(self) -> dict[str, object]:
        return {
            "node_id": "honjia-test",
            "organs": {
                "eye": {
                    "realtime_vision": {
                        "kind": "realtime_vision_observation",
                        "mode": "realtime_stream",
                        "primary_mode": True,
                        "stream_id": "snapshot-eye",
                        "status": "tracking",
                    }
                }
            },
        }


class SnapshotBodyRuntimeRealtimeRuntime(FakeBodyRuntime):
    def snapshot(self) -> dict[str, object]:
        return {
            "node_id": "honjia-test",
            "body_runtime": {
                "vision_realtime": {
                    "kind": "realtime_vision_observation",
                    "mode": "realtime",
                    "primary_mode": True,
                    "stream_id": "snapshot-body-runtime",
                    "status": "tracking",
                }
            },
        }


class SnapshotStaticCompatRuntime(FakeBodyRuntime):
    def snapshot(self) -> dict[str, object]:
        return {
            "node_id": "honjia-test",
            "organs": {
                "eye": {
                    "realtime_vision": {
                        "kind": "vision_observation",
                        "mode": "compat/static",
                        "frame_id": "still-1",
                        "status": "tracking",
                    }
                }
            },
        }


class ToDictRealtimePayload:
    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime",
            "primary_mode": True,
            "stream_id": "front-main",
            "status": "tracking",
        }


class ToDictRealtimeRuntime(FakeBodyRuntime):
    def latest_realtime_vision(self) -> ToDictRealtimePayload:
        return ToDictRealtimePayload()


class LegacyVisionStateRuntime(FakeBodyRuntime):
    def snapshot(self) -> dict[str, object]:
        return {
            "node_id": "honjia-test",
            "vision_state": {
                "schema": "eibrain.vision_state.v2",
                "frame_path": "/tmp/eibrain-vision/latest.jpg",
                "status": "tracking",
                "detections": [{"label": "person", "score": 0.9}],
            },
        }


class ReplaySimulatorVisionStateRuntime(FakeBodyRuntime):
    def snapshot(self) -> dict[str, object]:
        return {
            "node_id": "honjia-test",
            "vision_state": {
                "schema": "eibrain.vision_state.v2",
                "source": "vision_replay_simulator",
                "simulated": True,
                "replay": True,
                "frame_id": "sim-frame-7",
                "status": "tracking",
                "fps": 12.5,
                "latency_ms": 22.0,
                "last_frame_age": 0.18,
                "detections": [{"label": "person", "score": 0.88}],
                "tracks": [{"track_id": "person-1", "label": "person", "age_frames": 4}],
                "events": [{"event_type": "track_started", "track_id": "person-1"}],
            },
        }


class LiveVisionStateRuntime(FakeBodyRuntime):
    def snapshot(self) -> dict[str, object]:
        captured_at = time.time()
        return {
            "node_id": "honjia-test",
            "organs": {
                "eye": {
                    "subfunctions": {
                        "camera": {
                            "details": {
                                "driver": "vision_state",
                                "source": "vision_state",
                                "status": "live",
                                "backend": "gstreamer_hailo",
                                "frame_path": "/tmp/eibrain-vision/latest.jpg",
                                "frame_captured_at_ts": captured_at,
                                "state_age_s": 0.2,
                                "fps": 10.0,
                                "detections": [{"label": "person", "score": 0.91}],
                                "scene": {"summary": "person in frame"},
                                "events": [{"event_type": "track_started"}],
                            }
                        }
                    }
                }
            },
        }


class StaticVisionObservationRuntime(FakeBodyRuntime):
    def vision_realtime(self) -> VisionObservation:
        return VisionObservation(ts=2.0, source="eihead.honjia.eye.compat", frame_id="still-1")


class MultiHookRuntime(FakeBodyRuntime):
    eye_realtime = {
        "kind": "realtime_vision_observation",
        "mode": "realtime_stream",
        "status": "not_wired",
        "not_wired": True,
    }

    def vision_realtime(self) -> dict[str, object]:
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime",
            "primary_mode": True,
            "stream_id": "front-main",
            "status": "tracking",
        }


class ExpiredRealtimePayloadRuntime(FakeBodyRuntime):
    def vision_realtime(self) -> dict[str, object]:
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime",
            "status": "tracking",
            "timestamp": 100.0,
            "stream_id": "front-main",
        }


class MillisecondRealtimePayloadRuntime(FakeBodyRuntime):
    def vision_realtime(self) -> dict[str, object]:
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime",
            "status": "tracking",
            "timestamp": 1_714_800_001_234.0,
            "stream_id": "front-main",
            "not_wired": "false",
        }


LIVE_ADAPTER_STATUS = {
    "schema": "eihead.eye.realtime_status.v1",
    "mode": "realtime_stream",
    "status": "tracking",
    "backend": "gstreamer_hailo",
    "frame_count": 8,
    "detection_count": 2,
    "fps": 28.0,
    "last_frame_id": "frame-88",
    "placeholder": "false",
    "not_wired": "false",
    "compatibility_mode": "false",
    "message": "live adapter status",
}


class AdapterPayload:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = dict(payload)

    def to_dict(self) -> dict[str, object]:
        return dict(self._payload)


class LatestStatusAdapterRuntime(FakeBodyRuntime):
    class Adapter:
        latest_status = AdapterPayload(LIVE_ADAPTER_STATUS)

    eye_realtime = Adapter()


class StatusMethodAdapterRuntime(FakeBodyRuntime):
    class Adapter:
        def status(self) -> AdapterPayload:
            return AdapterPayload(LIVE_ADAPTER_STATUS)

    eye_realtime = Adapter()


class PollMethodAdapterRuntime(FakeBodyRuntime):
    class Adapter:
        def poll(self) -> AdapterPayload:
            return AdapterPayload(LIVE_ADAPTER_STATUS)

    eye_realtime = Adapter()


class NativeEyeService:
    def __init__(self) -> None:
        self.poll_calls = 0
        self.status_calls = 0

    def poll(self) -> dict[str, object]:
        self.poll_calls += 1
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "primary_mode": True,
            "stream_id": "native-eye-service",
            "status": "tracking",
            "frame_id": "live-native-1",
        }

    def status(self) -> dict[str, object]:
        self.status_calls += 1
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "status": "not_wired",
            "not_wired": True,
        }


class NativeEyeHttpService:
    def __init__(self) -> None:
        self.poll_calls = 0

    def poll(self) -> dict[str, object]:
        self.poll_calls += 1
        return {
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "primary_mode": True,
            "stream_id": "native-eye-http",
            "status": "tracking",
            "frame_id": "native-http-1",
            "backend": "gstreamer_hailo",
            "camera_device": "/dev/video42",
            "hailo_device": "/dev/hailo0",
            "stream_ready": True,
            "readiness_message": "native stream ready",
        }


class ToDictAdapterRuntime(FakeBodyRuntime):
    class Adapter:
        def to_dict(self) -> dict[str, object]:
            return dict(LIVE_ADAPTER_STATUS)

    eye_realtime = Adapter()


class NotWiredAdapterRuntime(FakeBodyRuntime):
    class Adapter:
        latest_status = AdapterPayload(
            {
                **LIVE_ADAPTER_STATUS,
                "status": "not_wired",
                "placeholder": "true",
                "not_wired": "true",
            }
        )

    eye_realtime = Adapter()


class CompatStaticAdapterRuntime(FakeBodyRuntime):
    class Adapter:
        latest_status = AdapterPayload(
            {
                **LIVE_ADAPTER_STATUS,
                "mode": "compat/static",
                "compatibility_mode": "true",
            }
        )

    eye_realtime = Adapter()


@pytest.mark.parametrize(
    ("payload", "case_name"),
    [
        (
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime_stream",
                "status": "not_wired",
                "not_wired": True,
            },
            "not_wired",
        ),
        (
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime_stream",
                "status": "tracking",
                "placeholder": True,
            },
            "placeholder",
        ),
        (
            {
                "kind": "realtime_vision_observation",
                "mode": "compat/static",
                "status": "tracking",
            },
            "compat_static_mode",
        ),
        (
            {
                "kind": "vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "primary_mode": False,
            },
            "vision_observation",
        ),
        (
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "compatibility_mode": True,
            },
            "compatibility_mode",
        ),
    ],
)
def test_head_runtime_rejects_non_live_realtime_vision_payloads(
    payload: dict[str, Any],
    case_name: str,
) -> None:
    class Runtime(FakeBodyRuntime):
        def vision_realtime(self) -> dict[str, Any]:
            return payload

    runtime = HeadRuntimeApp(body_runtime=Runtime(), config_path=f"config/{case_name}.yaml")

    assert runtime.vision_realtime() is None


def test_head_runtime_passes_through_realtime_vision_observation() -> None:
    runtime = HeadRuntimeApp(body_runtime=RealtimeObservationRuntime(), config_path="config/test.yaml")

    observation = runtime.vision_realtime()

    assert isinstance(observation, RealtimeVisionObservation)
    assert observation.stream_id == "front-main"


def test_head_runtime_passes_through_realtime_mapping() -> None:
    runtime = HeadRuntimeApp(body_runtime=MappingRealtimeRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() == MappingRealtimeRuntime.eye_realtime


def test_head_runtime_accepts_app_level_eye_realtime_hook() -> None:
    runtime = AppHookHeadRuntime(body_runtime=FakeBodyRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() == {
        "kind": "realtime_vision_observation",
        "mode": "realtime",
        "primary_mode": True,
        "stream_id": "app-hook",
        "status": "tracking",
    }


def test_head_runtime_accepts_native_provider_eye_realtime_payload() -> None:
    runtime = HeadRuntimeApp(
        body_runtime=FakeBodyRuntime(),
        config_path="config/test.yaml",
        native_providers={
            "eye": {
                "status": "wired",
                "details": {
                    "realtime_vision": {
                        "kind": "realtime_vision_observation",
                        "mode": "realtime",
                        "primary_mode": True,
                        "stream_id": "native-eye",
                        "status": "tracking",
                    }
                },
            },
            "ear": {"status": "wired"},
            "mouth": {"status": "wired"},
            "neck": {"status": "wired"},
        },
        neck_servo_adapter=object(),
    )

    assert runtime.vision_realtime() == {
        "kind": "realtime_vision_observation",
        "mode": "realtime",
        "primary_mode": True,
        "stream_id": "native-eye",
        "status": "tracking",
    }


def test_head_runtime_polls_native_eye_service_object_before_status() -> None:
    eye_service = NativeEyeService()
    runtime = HeadRuntimeApp(
        body_runtime=FakeBodyRuntime(),
        config_path="config/test.yaml",
        native_providers={
            "eye": eye_service,
            "ear": {"status": "wired"},
            "mouth": {"status": "wired"},
            "neck": {"status": "wired"},
        },
        neck_servo_adapter=object(),
    )

    assert runtime.vision_realtime() == {
        "kind": "realtime_vision_observation",
        "mode": "realtime_stream",
        "primary_mode": True,
        "stream_id": "native-eye-service",
        "status": "tracking",
        "frame_id": "live-native-1",
    }
    assert eye_service.poll_calls == 1
    assert eye_service.status_calls == 0


def test_head_runtime_http_realtime_endpoint_uses_native_eye_service_payload() -> None:
    eye_service = NativeEyeHttpService()
    runtime = HeadRuntimeApp(
        body_runtime=FakeBodyRuntime(),
        config_path="config/test.yaml",
        native_providers={
            "eye": eye_service,
            "ear": {"status": "wired"},
            "mouth": {"status": "wired"},
            "neck": {"status": "wired"},
        },
        neck_servo_adapter=object(),
    )

    with _running_server(runtime, clock=lambda: 791.0) as base_url:
        status_code, payload = _read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["source"] == "vision_realtime"
    assert payload["observation"]["stream_id"] == "native-eye-http"
    assert payload["backend"] == "gstreamer_hailo"
    assert payload["devices"]["camera_device"] == "/dev/video42"
    assert payload["devices"]["hailo_device"] == "/dev/hailo0"
    assert payload["stream_ready"] is True
    assert payload["readiness_message"] == "native stream ready"
    assert eye_service.poll_calls == 1


def test_head_runtime_accepts_snapshot_eye_realtime_payload() -> None:
    runtime = HeadRuntimeApp(body_runtime=SnapshotEyeRealtimeRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() == {
        "kind": "realtime_vision_observation",
        "mode": "realtime_stream",
        "primary_mode": True,
        "stream_id": "snapshot-eye",
        "status": "tracking",
    }


def test_head_runtime_accepts_snapshot_body_runtime_realtime_payload() -> None:
    runtime = HeadRuntimeApp(body_runtime=SnapshotBodyRuntimeRealtimeRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() == {
        "kind": "realtime_vision_observation",
        "mode": "realtime",
        "primary_mode": True,
        "stream_id": "snapshot-body-runtime",
        "status": "tracking",
    }


def test_head_runtime_passes_through_realtime_to_dict_payload() -> None:
    runtime = HeadRuntimeApp(body_runtime=ToDictRealtimeRuntime(), config_path="config/test.yaml")

    observation = runtime.vision_realtime()

    assert isinstance(observation, ToDictRealtimePayload)
    assert observation.to_dict()["stream_id"] == "front-main"


def test_head_runtime_does_not_promote_legacy_vision_state_snapshot_to_realtime() -> None:
    runtime = HeadRuntimeApp(body_runtime=LegacyVisionStateRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() is None


def test_head_runtime_promotes_replay_simulator_vision_state_to_realtime_monitoring() -> None:
    runtime = HeadRuntimeApp(body_runtime=ReplaySimulatorVisionStateRuntime(), config_path="config/test.yaml")

    payload = runtime.vision_realtime()

    assert isinstance(payload, dict)
    assert payload["kind"] == "realtime_vision_observation"
    assert payload["mode"] == "realtime_stream"
    assert payload["primary_mode"] is True
    assert payload["source"] == "vision_replay_simulator"
    assert payload["simulated"] is True
    assert payload["replay"] is True
    assert payload["frame_id"] == "sim-frame-7"
    assert payload["tracks"][0]["track_id"] == "person-1"
    assert payload["events"][0]["event_type"] == "track_started"


def test_head_runtime_promotes_live_vision_state_to_realtime_monitoring() -> None:
    runtime = HeadRuntimeApp(body_runtime=LiveVisionStateRuntime(), config_path="config/test.yaml")

    payload = runtime.vision_realtime()

    assert isinstance(payload, dict)
    assert payload["kind"] == "realtime_vision_observation"
    assert payload["mode"] == "realtime_stream"
    assert payload["source"] == "vision_state_live"
    assert payload["status"] == "tracking"
    assert payload["stream_ready"] is True
    assert payload["backend"] == "gstreamer_hailo"
    assert payload["frame_id"].startswith("vision-state-")
    assert payload["detections"] == [{"label": "person", "score": 0.91}]


def test_head_runtime_http_realtime_endpoint_exposes_live_vision_state() -> None:
    runtime = HeadRuntimeApp(body_runtime=LiveVisionStateRuntime(), config_path="config/test.yaml")

    with _running_server(runtime, clock=time.time) as base_url:
        status_code, payload = _read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["backend"] == "gstreamer_hailo"
    assert payload["fps"] == 10.0
    assert payload["detections_summary"] == "person 0.91"
    assert payload["source_freshness"]["state"] == "healthy"


def test_head_runtime_http_realtime_endpoint_exposes_simulator_diagnostics() -> None:
    runtime = HeadRuntimeApp(body_runtime=ReplaySimulatorVisionStateRuntime(), config_path="config/test.yaml")

    with _running_server(runtime, clock=lambda: 1200.0) as base_url:
        status_code, payload = _read_json(f"{base_url}/api/vision/realtime")

    assert status_code == 200
    assert payload["status"] == "wired"
    assert payload["source"] == "vision_realtime"
    assert payload["source_freshness"]["state"] == "simulated"
    assert payload["latency_ms"] == 22.0
    assert payload["tracks"]["count"] == 1
    assert payload["events"]["count"] == 1
    assert payload["detections_summary"] == "person 0.88"


def test_head_runtime_http_exposes_voice_and_eivoice_runtime_diagnostics() -> None:
    class VoiceRuntime:
        def start(self) -> None:
            return None

        def voice_status(self) -> dict[str, Any]:
            return {
                "status": "ready",
                "ear": {"status": "listening"},
                "mouth": {"status": "idle"},
                "voice_dialogue": {
                    "running": True,
                    "phase": "listening",
                    "last_transcript": "你好鸿途",
                    "last_reply": "我在。",
                    "last_stage_latency_ms": {"listen_asr": 11.0, "dialogue": 22.0, "speak": 33.0, "total": 66.0},
                },
                "realtime_audio": {"enabled": True, "running": True},
                "readiness_message": "runtime voice diagnostics are attached",
            }

        def status(self) -> dict[str, Any]:
            return {
                "state": "running",
                "conversation_state": "listening",
                "audio_frontend": {
                    "aec": {"enabled": True, "available": True},
                    "ns": {"enabled": True, "available": True},
                    "vad": {"enabled": True, "available": True},
                    "loopback": {"enabled": True, "available": True},
                },
                "transport": {"transport": "openclaw_realtime", "state": "connected"},
                "openclaw_ws": {
                    "connected": True,
                    "url": "ws://honxin-gateway",
                    "session_state": "ready",
                },
            }

    runtime = HeadRuntimeApp(
        body_runtime=FakeBodyRuntime(),
        config_path="config/test.yaml",
        voice_runtime=VoiceRuntime(),
    )

    with _running_server(runtime, clock=lambda: 456.0) as base_url:
        voice_status, voice_payload = _read_json(f"{base_url}/api/voice/realtime")
        audio_status, audio_payload = _read_json(f"{base_url}/api/audio/realtime")
        eivoice_status, eivoice_payload = _read_json(f"{base_url}/api/eivoice/runtime")

    assert voice_status == 200
    assert voice_payload["schema"] == "eihead.monitor.voice_realtime.v1"
    assert voice_payload["voice_chain"]["last_asr_text"] == "你好鸿途"
    assert voice_payload["voice_chain"]["last_tts_text"] == "我在。"
    assert voice_payload["voice_chain"]["latency_ms"]["dialogue"] == 22.0
    assert audio_status == 200
    assert audio_payload["aliases"] == ["audio.realtime"]
    assert eivoice_status == 200
    assert eivoice_payload["eivoiceRuntime"]["state"] == "running"
    assert eivoice_payload["eivoiceRuntime"]["openclawWs"]["sessionState"] == "ready"


def test_head_runtime_does_not_promote_snapshot_static_compat_payload_to_realtime() -> None:
    runtime = HeadRuntimeApp(body_runtime=SnapshotStaticCompatRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() is None


def test_head_runtime_rejects_static_vision_observation() -> None:
    runtime = HeadRuntimeApp(body_runtime=StaticVisionObservationRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() is None


def test_head_runtime_uses_later_live_hook_when_earlier_hook_is_not_live() -> None:
    runtime = HeadRuntimeApp(body_runtime=MultiHookRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() == {
        "kind": "realtime_vision_observation",
        "mode": "realtime",
        "primary_mode": True,
        "stream_id": "front-main",
        "status": "tracking",
    }


@pytest.mark.parametrize(
    "runtime_cls",
    [
        LatestStatusAdapterRuntime,
        StatusMethodAdapterRuntime,
        PollMethodAdapterRuntime,
        ToDictAdapterRuntime,
    ],
)
def test_head_runtime_accepts_live_adapter_payload_forms(runtime_cls: type[FakeBodyRuntime]) -> None:
    runtime = HeadRuntimeApp(body_runtime=runtime_cls(), config_path="config/test.yaml")

    payload = runtime.vision_realtime()

    assert _payload_dict(payload) == LIVE_ADAPTER_STATUS


def test_head_runtime_rejects_not_wired_adapter_status_with_string_booleans() -> None:
    runtime = HeadRuntimeApp(body_runtime=NotWiredAdapterRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() is None


def test_head_runtime_rejects_stale_realtime_vision_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = HeadRuntimeApp(
        body_runtime=ExpiredRealtimePayloadRuntime(),
        config_path="config/test.yaml",
        realtime_vision_max_age_seconds=2.0,
    )
    _monotonic_timestamp = 120.0

    # Validate freshness window boundary explicitly.
    monkeypatch.setattr(time, "time", lambda: _monotonic_timestamp)

    assert runtime.vision_realtime() is None


def test_head_runtime_accepts_ms_realtime_timestamp_without_explicit_ms_field(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = HeadRuntimeApp(
        body_runtime=MillisecondRealtimePayloadRuntime(),
        config_path="config/test.yaml",
        realtime_vision_max_age_seconds=1200.0,
    )

    _monotonic_timestamp = 1_714_800_002.1
    monkeypatch.setattr(time, "time", lambda: _monotonic_timestamp)

    payload = runtime.vision_realtime()

    assert isinstance(payload, dict)
    assert payload["timestamp"] == 1_714_800_001_234.0


def test_head_runtime_rejects_compat_static_adapter_status() -> None:
    runtime = HeadRuntimeApp(body_runtime=CompatStaticAdapterRuntime(), config_path="config/test.yaml")

    assert runtime.vision_realtime() is None


def _payload_dict(payload: Any) -> dict[str, object]:
    assert payload is not None
    if isinstance(payload, dict):
        return payload
    assert hasattr(payload, "to_dict")
    return payload.to_dict()


@contextmanager
def _running_server(app: Any, **kwargs: Any) -> Iterator[str]:
    server = create_server(app, host="127.0.0.1", port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def _read_json(url: str) -> tuple[int, dict[str, Any]]:
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=2.0) as response:
        return response.status, json.loads(response.read().decode("utf-8"))
