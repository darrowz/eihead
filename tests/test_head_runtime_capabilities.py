from __future__ import annotations

from eihead.protocol import (
    ActionExecuted,
    MoveHeadAction,
    PlaySpeechAction,
    SpeechPlaybackCompleted,
    StopSpeechAction,
)
from eihead.runtime.app import HeadRuntimeApp


class _BodyRuntime:
    def __init__(self) -> None:
        self.dispatched: list[object] = []

    def snapshot(self) -> dict[str, object]:
        return {
            "node_id": "honjia-test",
            "organ_count": 4,
            "capabilities": {"voice": True, "vision": True, "neck": True},
        }

    def dispatch_actions(self, actions: list[object]) -> list[object]:
        self.dispatched.extend(actions)
        action = actions[0]
        if isinstance(action, PlaySpeechAction):
            return [
                SpeechPlaybackCompleted(
                    ts=action.ts,
                    source="mouth.tts_playback",
                    status="ok",
                    session_id=action.session_id,
                    actor_id=action.actor_id,
                    target_id=action.target_id,
                )
            ]
        if isinstance(action, (MoveHeadAction, StopSpeechAction)):
            return [
                ActionExecuted(
                    ts=action.ts,
                    source="neck.motor" if isinstance(action, MoveHeadAction) else "mouth.tts_playback",
                    status="ok",
                    session_id=action.session_id,
                    actor_id=action.actor_id,
                    target_id=action.target_id,
                    action_kind=action.kind,
                    details={"target_angle": getattr(action, "target_angle", None)},
                )
            ]
        return []


class _FrameOnlyRuntime:
    def snapshot(self) -> dict[str, object]:
        return {"node_id": "honjia-frame"}

    def latest_visual_frame_path(self) -> str:
        return "/tmp/honjia/latest.jpg"


class _SnapshotOnlyRuntime:
    def snapshot(self) -> dict[str, object]:
        return {"node_id": "honjia-snapshot-only"}


class _NativeVoiceRuntime:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.spoken: list[str] = []

    def start(self) -> None:
        self.started += 1

    def stop(self) -> dict[str, object]:
        self.stopped += 1
        return {"status": "stopped", "success": True}

    def speak(self, text: str) -> dict[str, object]:
        self.spoken.append(text)
        return {
            "status": "ok",
            "success": True,
            "details": {"text_preview": text[:20], "playback_elapsed_ms": 12.5},
        }

    def status(self) -> dict[str, object]:
        return {
            "schema": "eihead.eivoice_runtime.diagnostics.v1",
            "state": "running",
            "health": "healthy",
            "running": True,
            "audio_frontend": {"vad": {"enabled": True}},
        }

    def voice_status(self) -> dict[str, object]:
        return {
            "status": "ready",
            "voice_dialogue": {"enabled": True, "running": True, "phase": "listening"},
            "realtime_audio": {"enabled": True, "running": True},
            "readiness_message": "native realtime voice loop is attached",
        }


def test_capabilities_returns_status_snapshot_shape_without_hardware() -> None:
    runtime = HeadRuntimeApp(body_runtime=_BodyRuntime(), config_path="config/test.yaml")

    payload = runtime.capabilities()

    assert payload["command"] == "capabilities"
    assert payload["runtime"] == "eihead"
    assert payload["node_id"] == "honjia-test"
    assert payload["body_runtime_node_id"] == "honjia-test"
    assert payload["summary"]["total"] == len(payload["capabilities"])
    assert payload["overall_status"] != "online"
    assert payload["capabilities"]["neck"]["limits"]["tilt_deg"] is None


def test_capabilities_marks_static_path_checks_as_unverified_status_source() -> None:
    runtime = HeadRuntimeApp(body_runtime=_BodyRuntime(), config_path="config/test.yaml")

    payload = runtime.capabilities()

    for capability in payload["capabilities"].values():
        details = capability["details"]
        assert details["hardware_verified"] is False
        assert "checked_at" in details
        assert "last_checked" in details
        assert details["source"] in {
            "path_exists",
            "static_config",
            "config_disabled",
            "not_configured",
            "native_provider",
        }

    assert payload["capabilities"]["camera"]["status"] == "unknown"
    assert payload["capabilities"]["camera"]["details"]["source"] == "native_provider"
    assert payload["capabilities"]["asr"]["details"]["reason"] == "native_provider_status"
    assert payload["capabilities"]["asr"]["details"]["native_provider"] == "ear"


def test_capabilities_prefers_native_provider_truth_over_static_registry() -> None:
    runtime = HeadRuntimeApp(
        body_runtime=_BodyRuntime(),
        config_path="config/test.yaml",
        neck_servo_adapter=object(),
        native_providers={
            "eye": {
                "status": "wired",
                "provider": "fake-eye-adapter",
                "source": "native-eye-probe",
                "checked_at": 111.25,
                "hardware_verified": True,
                "details": {"fps": 30},
            },
            "ear": {"status": "unknown", "reason": "ear_probe_not_ready"},
            "mouth": {"status": "degraded", "reason": "tts_queue_backpressure"},
            "neck": {
                "status": "wired",
                "provider": "fake-neck-servo",
                "reason": "adapter_injected",
                "hardware_verified": False,
            },
        },
    )

    payload = runtime.capabilities()

    camera = payload["capabilities"]["camera"]
    assert camera["status"] == "live"
    assert camera["details"]["source"] == "native-eye-probe"
    assert camera["details"]["hardware_verified"] is True
    assert camera["details"]["native_provider"] == "eye"
    assert camera["details"]["native_fps"] == 30

    neck = payload["capabilities"]["neck"]
    assert neck["status"] == "online"
    assert neck["details"]["hardware_verified"] is False
    assert neck["details"]["native_provider"] == "neck"

    assert payload["capabilities"]["microphone"]["status"] == "unknown"
    assert payload["capabilities"]["speaker"]["status"] == "degraded"


def test_handle_action_speak_delegates_to_body_runtime_dispatch() -> None:
    body_runtime = _BodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)

    outcome = runtime.handle_action(
        {"type": "speak", "text": "你好鸿途", "session_id": "s1", "actor_id": "darrow"},
        trace_id="trace-voice",
    )

    assert outcome["status"] == "accepted"
    assert outcome["success"] is True
    assert outcome["trace_id"] == "trace-voice"
    assert isinstance(body_runtime.dispatched[0], PlaySpeechAction)
    assert body_runtime.dispatched[0].text == "你好鸿途"
    assert outcome["details"]["delegate_outcomes"][0]["kind"] == "speech_playback_completed"


def test_handle_action_move_head_defaults_to_yaw_and_keeps_horizontal_only() -> None:
    body_runtime = _BodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)

    outcome = runtime.handle_action({"type": "move_head", "angle": 112, "target_name": "speaker"})

    assert outcome["status"] == "accepted"
    assert outcome["details"]["axis"] == "yaw"
    assert isinstance(body_runtime.dispatched[0], MoveHeadAction)
    assert body_runtime.dispatched[0].target_angle == 112
    assert body_runtime.dispatched[0].target_name == "speaker"


def test_handle_action_move_head_suppresses_jitter_within_min_angle_delta() -> None:
    body_runtime = _BodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime, ptz_min_angle_delta=4.0)

    first_outcome = runtime.handle_action({"type": "move_head", "angle": 112})
    second_outcome = runtime.handle_action({"type": "move_head", "angle": 115})
    third_outcome = runtime.handle_action({"type": "move_head", "angle": 119})

    assert first_outcome["status"] == "accepted"
    assert second_outcome["status"] == "skipped"
    assert second_outcome["details"]["reason"] == "ptz_jitter_suppressed"
    assert second_outcome["details"]["min_angle_delta"] == 4.0
    assert third_outcome["status"] == "accepted"
    assert len(body_runtime.dispatched) == 2


def test_handle_action_rejects_non_yaw_axis_without_dispatching() -> None:
    body_runtime = _BodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)

    outcome = runtime.handle_action({"type": "move_head", "axis": "pitch", "angle": 30})

    assert outcome["status"] == "unsupported"
    assert outcome["success"] is False
    assert outcome["details"]["axis"] == "pitch"
    assert body_runtime.dispatched == []


def test_handle_action_stop_speech_delegates_to_body_runtime_dispatch() -> None:
    body_runtime = _BodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)

    outcome = runtime.handle_action({"type": "stop_speech", "trace_id": "trace-stop"})

    assert outcome["status"] == "accepted"
    assert outcome["trace_id"] == "trace-stop"
    assert isinstance(body_runtime.dispatched[0], StopSpeechAction)
    assert outcome["details"]["delegate_outcomes"][0]["action_kind"] == "stop_speech_action"


def test_handle_action_capture_frame_uses_latest_visual_frame_path_fallback() -> None:
    runtime = HeadRuntimeApp(body_runtime=_FrameOnlyRuntime())

    outcome = runtime.handle_action({"type": "capture_frame"})

    assert outcome["status"] == "accepted"
    assert outcome["success"] is True
    assert outcome["delegated"] is True
    assert outcome["details"]["source"] == "latest_visual_frame_path"
    assert outcome["details"]["frame_path"] == "/tmp/honjia/latest.jpg"


def test_handle_action_without_dispatcher_returns_structured_skipped_outcome() -> None:
    runtime = HeadRuntimeApp(body_runtime=_SnapshotOnlyRuntime())

    outcome = runtime.handle_action({"type": "speak", "text": "hello"})

    assert outcome["status"] == "not_wired"
    assert outcome["success"] is False
    assert outcome["details"]["reason"] == "native_provider_unavailable"


def test_head_runtime_starts_and_prefers_native_voice_runtime_status() -> None:
    voice_runtime = _NativeVoiceRuntime()
    runtime = HeadRuntimeApp(
        body_runtime=_SnapshotOnlyRuntime(),
        native_voice_status={"status": "degraded", "voice_dialogue": {"running": False}},
        voice_runtime=voice_runtime,
    )

    assert voice_runtime.started == 1
    assert runtime.voice_status()["voice_dialogue"]["running"] is True
    assert runtime.eivoice_runtime_status()["state"] == "running"


def test_handle_action_speak_uses_native_voice_runtime_without_body_dispatcher() -> None:
    voice_runtime = _NativeVoiceRuntime()
    runtime = HeadRuntimeApp(body_runtime=_SnapshotOnlyRuntime(), voice_runtime=voice_runtime)

    outcome = runtime.handle_action({"type": "speak", "text": "hello honjia"}, trace_id="trace-native-voice")

    assert outcome["status"] == "accepted"
    assert outcome["success"] is True
    assert outcome["delegated"] is True
    assert outcome["trace_id"] == "trace-native-voice"
    assert outcome["details"]["provider"] == "native_voice_runtime"
    assert voice_runtime.spoken == ["hello honjia"]


def test_handle_action_stop_speech_uses_native_voice_runtime_without_body_dispatcher() -> None:
    voice_runtime = _NativeVoiceRuntime()
    runtime = HeadRuntimeApp(body_runtime=_SnapshotOnlyRuntime(), voice_runtime=voice_runtime)

    outcome = runtime.handle_action({"type": "stop_speech"})

    assert outcome["status"] == "stopped"
    assert outcome["success"] is True
    assert voice_runtime.stopped == 1
