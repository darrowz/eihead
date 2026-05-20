from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from eihead.runtime.app import HeadRuntimeApp
from eihead.monitoring.neck import build_neck_diagnostics_from_app
from eihead.monitoring.voice import build_voice_diagnostics_from_app
from eihead.runtime.native_providers import build_native_provider_statuses
from eihead.eye.vision_loop import build_vision_state_payload, write_vision_state
from eihead.runtime.config import load_eihead_config
from eihead.runtime.native_services import (
    SafeSubprocessEyeAdapter,
    StateFileEyeAdapter,
    build_native_voice_runtime,
    native_voice_loop_config_from_eihead_config,
)
from eihead.eye import GStreamerHailoRealtimeConfig


class FakeBodyRuntime:
    def snapshot(self) -> dict[str, object]:
        return {"node_id": "honjia-native-test", "organ_count": 4}


class FakeNativeProbe:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, provider_name: str, *, config: Any, environ: dict[str, str]) -> dict[str, object]:
        self.calls.append(provider_name)
        if provider_name == "eye":
            return {
                "status": "wired",
                "provider": environ["EIHEAD_TEST_EYE_PROVIDER"],
                "reason": "probe_reported_wired",
            }
        if provider_name == "ear":
            return {"status": "unknown", "reason": f"config_node={config.node_id}"}
        if provider_name == "mouth":
            return {"status": "unavailable", "reason": "tts_backend_disabled"}
        raise AssertionError(f"unexpected hardware probe for {provider_name}")


class FakeNeckAdapter:
    def apply_plan(self, plan: dict[str, object]) -> dict[str, object]:
        return {"status": "ok", "plan_status": plan.get("status")}


class FakeEyeService:
    def status(self) -> dict[str, object]:
        return {
            "schema": "eihead.eye.realtime_status.v1",
            "status": "tracking",
            "mode": "realtime_stream",
            "provider": "native-eye-service",
            "backend": "gstreamer_hailo",
            "camera_device": "/dev/video42",
            "hailo_device": "/dev/hailo0",
            "stream_ready": True,
            "not_wired": False,
            "readiness_message": "camera and hailo ready",
        }


class FakeNativeEyeAdapter:
    def __init__(self) -> None:
        self.poll_count = 0

    def status(self) -> dict[str, object]:
        return self._payload()

    def poll(self) -> dict[str, object]:
        self.poll_count += 1
        return self._payload()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "eihead.eye.realtime_status.v1",
            "kind": "realtime_vision_observation",
            "mode": "realtime_stream",
            "status": "tracking",
            "backend": "gstreamer_hailo",
            "stream_ready": True,
            "not_wired": False,
            "last_frame_id": "native-1",
            "camera_device": "/dev/video42",
            "hailo_device": "/dev/hailo0",
        }


class FakeNativeVoiceRuntime:
    def __init__(self) -> None:
        self.started = 0

    def start(self) -> None:
        self.started += 1

    def voice_status(self) -> dict[str, object]:
        return {
            "status": "ready",
            "voice_dialogue": {"enabled": True, "running": True, "phase": "listening"},
            "realtime_audio": {"enabled": True, "running": True},
        }

    def status(self) -> dict[str, object]:
        return {"state": "running", "health": "healthy", "running": True}


def test_from_config_path_reports_native_provider_boundaries_without_hardware(tmp_path: Path) -> None:
    config_path = tmp_path / "eihead.honjia.yaml"
    body_config_path = tmp_path / "eibrain.honjia.yaml"
    config_path.write_text(
        "\n".join(
            [
                "node_id: honjia-native-test",
                "legacy:",
                f"  body_runtime_config_path: {body_config_path.as_posix()}",
                "native_providers:",
                "  eye:",
                "    enabled: true",
                "  ear:",
                "    enabled: true",
                "  mouth:",
                "    enabled: false",
                "  neck:",
                "    enabled: true",
            ]
        ),
        encoding="utf-8",
    )
    def fake_factory(path: str) -> FakeBodyRuntime:
        raise AssertionError(f"legacy factory should not be called: {path}")

    probe = FakeNativeProbe()

    runtime = HeadRuntimeApp.from_config_path(
        str(config_path),
        body_runtime_factory=fake_factory,
        native_provider_probe=probe,
        native_environ={"EIHEAD_TEST_EYE_PROVIDER": "fake-eye-adapter"},
    )

    snapshot = runtime.snapshot()
    native_providers = snapshot["native_providers"]

    assert snapshot["body_runtime"] == {}
    assert native_providers["eye"] == {
        "status": "wired",
        "provider": "fake-eye-adapter",
        "reason": "probe_reported_wired",
    }
    assert native_providers["ear"]["status"] == "unknown"
    assert native_providers["ear"]["reason"] == "config_node=honjia-native-test"
    assert native_providers["mouth"]["status"] == "unavailable"
    assert native_providers["mouth"]["reason"] == "tts_backend_disabled"
    assert native_providers["neck"]["status"] == "unavailable"
    assert native_providers["neck"]["reason"] == "neck_servo_adapter_missing"
    assert "neck" not in probe.calls


def test_injected_neck_adapter_can_be_reported_as_wired_by_probe(tmp_path: Path) -> None:
    config_path = tmp_path / "eihead.honjia.yaml"
    config_path.write_text("node_id: honjia-native-test\n", encoding="utf-8")

    def fake_factory(path: str) -> FakeBodyRuntime:
        raise AssertionError(f"legacy factory should not be called: {path}")

    def probe(provider_name: str, *, config: Any, environ: dict[str, str]) -> dict[str, object]:
        if provider_name == "neck":
            return {"status": "wired", "provider": "fake-neck-servo", "reason": "adapter_injected"}
        return {"status": "unknown", "reason": "probe_not_configured"}

    runtime = HeadRuntimeApp.from_config_path(
        str(config_path),
        body_runtime_factory=fake_factory,
        native_provider_probe=probe,
        neck_servo_adapter=FakeNeckAdapter(),
    )

    assert runtime.snapshot()["native_providers"]["neck"] == {
        "status": "wired",
        "provider": "fake-neck-servo",
        "reason": "adapter_injected",
    }


def test_env_can_report_neck_wired_before_adapter_injection() -> None:
    statuses = build_native_provider_statuses(
        config=None,
        environ={
            "EIHEAD_NATIVE_NECK_STATUS": "wired",
            "EIHEAD_NATIVE_NECK_PROVIDER": "raspbot-i2c",
            "EIHEAD_NATIVE_NECK_REASON": "verified_i2c_bus",
            "EIHEAD_NATIVE_NECK_HARDWARE_VERIFIED": "true",
        },
    )

    assert statuses["neck"] == {
        "status": "wired",
        "provider": "raspbot-i2c",
        "reason": "verified_i2c_bus",
        "hardware_verified": True,
    }


def test_normalize_keeps_explicit_neck_status_without_adapter() -> None:
    runtime = HeadRuntimeApp(
        body_runtime=FakeBodyRuntime(),
        native_providers={
            "neck": {
                "status": "wired",
                "provider": "raspbot-i2c",
                "reason": "verified_i2c_bus",
            }
        },
    )

    assert runtime.status()["native_providers"]["neck"] == {
        "status": "wired",
        "provider": "raspbot-i2c",
        "reason": "verified_i2c_bus",
    }


def test_direct_runtime_construction_marks_uninjected_neck_unavailable() -> None:
    runtime = HeadRuntimeApp(body_runtime=FakeBodyRuntime(), config_path="config/test.yaml")

    native_providers = runtime.status()["native_providers"]

    assert native_providers["eye"]["status"] == "unknown"
    assert native_providers["ear"]["status"] == "unknown"
    assert native_providers["mouth"]["status"] == "unknown"
    assert native_providers["neck"] == {
        "status": "unavailable",
        "reason": "neck_servo_adapter_missing",
    }


def test_native_provider_probe_can_report_degraded_with_truthful_metadata() -> None:
    def probe(provider_name: str, *, config: Any, environ: dict[str, str]) -> dict[str, object]:
        if provider_name == "eye":
            return {
                "status": "degraded",
                "provider": "fake-eye-adapter",
                "source": "fake-native-live-probe",
                "checked_at": 5678.5,
                "reason": "low_frame_rate",
                "hardware_verified": True,
                "details": {"fps": 2},
            }
        return {"status": "unknown", "source": "fake-native-live-probe", "reason": "not_checked"}

    statuses = build_native_provider_statuses(
        config=None,
        environ={},
        probe=probe,
        neck_servo_adapter=FakeNeckAdapter(),
    )

    assert statuses["eye"] == {
        "status": "degraded",
        "provider": "fake-eye-adapter",
        "reason": "low_frame_rate",
        "source": "fake-native-live-probe",
        "checked_at": 5678.5,
        "last_checked": 5678.5,
        "hardware_verified": True,
        "details": {"fps": 2},
    }


def test_native_provider_normalizes_wired_but_waiting_stream_as_degraded() -> None:
    runtime = HeadRuntimeApp(
        native_providers={
            "eye": {
                "status": "waiting_for_frame",
                "not_wired": False,
                "stream_ready": False,
                "provider": "native-eye-service",
            }
        },
    )

    assert runtime.status()["native_providers"]["eye"]["status"] == "degraded"


def test_native_eye_service_status_feeds_status_and_capability_readiness() -> None:
    runtime = HeadRuntimeApp(
        body_runtime=FakeBodyRuntime(),
        config_path="config/test.yaml",
        native_providers={
            "eye": FakeEyeService(),
            "ear": {"status": "unknown"},
            "mouth": {"status": "unknown"},
            "neck": {"status": "wired"},
        },
        neck_servo_adapter=FakeNeckAdapter(),
    )

    status_payload = runtime.status()
    eye_status = status_payload["native_providers"]["eye"]

    assert eye_status["status"] == "wired"
    assert eye_status["provider"] == "native-eye-service"
    assert eye_status["details"]["backend"] == "gstreamer_hailo"
    assert eye_status["details"]["camera_device"] == "/dev/video42"
    assert eye_status["details"]["hailo_device"] == "/dev/hailo0"
    assert eye_status["details"]["stream_ready"] is True
    assert eye_status["details"]["not_wired"] is False
    assert eye_status["details"]["readiness_message"] == "camera and hailo ready"

    capabilities = runtime.capabilities()["capabilities"]
    assert capabilities["camera"]["details"]["native_camera_device"] == "/dev/video42"
    assert capabilities["camera"]["details"]["native_stream_ready"] is True
    assert capabilities["hailo"]["details"]["native_hailo_device"] == "/dev/hailo0"
    assert capabilities["hailo"]["details"]["native_stream_ready"] is True
    assert capabilities["vision_backend"]["details"]["native_backend"] == "gstreamer_hailo"
    assert capabilities["vision_backend"]["details"]["native_hailo_device"] == "/dev/hailo0"


def test_from_config_path_wires_realtime_eye_service_from_honjia_config(tmp_path: Path) -> None:
    config_path = tmp_path / "eihead.honjia.yaml"
    config_path.write_text(
        "\n".join(
            [
                "node_id: honjia",
                "devices:",
                "  camera:",
                "    path: /dev/video42",
                "  hailo:",
                "    path: /dev/hailo0",
                "    hef_path: /opt/models/personface.hef",
                "    postprocess_so_path: /opt/hailo/libpost.so",
                "    postprocess_config_path: /opt/hailo/personface.json",
                "    postprocess_function: filter",
                "    score_threshold: 0.45",
                "    inference_width: 640",
                "    inference_height: 640",
                "    inference_format: RGB",
                "    labels: [person, face]",
                "capabilities:",
                "  software:",
                "    vision_backend:",
                "      enabled: true",
                "      backend: hailo",
                "      limits:",
                "        realtime: true",
                "        max_fps: 15",
            ]
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def adapter_factory(config: object) -> FakeNativeEyeAdapter:
        captured["config"] = config
        return FakeNativeEyeAdapter()

    runtime = HeadRuntimeApp.from_config_path(
        str(config_path),
        native_eye_adapter_factory=adapter_factory,
        native_environ={},
    )

    gstreamer_config = captured["config"]
    assert getattr(gstreamer_config, "camera_device") == "/dev/video42"
    assert getattr(gstreamer_config, "hailo_device") == "/dev/hailo0"
    assert getattr(gstreamer_config, "hef_path") == "/opt/models/personface.hef"
    assert getattr(gstreamer_config, "postprocess_config_path") == "/opt/hailo/personface.json"
    assert getattr(gstreamer_config, "labels") == ("person", "face")
    assert getattr(gstreamer_config, "score_threshold") == 0.45
    assert getattr(gstreamer_config, "framerate") == 15
    assert getattr(gstreamer_config, "inference_width") == 640
    assert getattr(gstreamer_config, "inference_height") == 640
    assert getattr(gstreamer_config, "inference_format") == "RGB"
    assert getattr(gstreamer_config, "sample_timeout_s") == 5.0

    payload = runtime.vision_realtime()
    assert payload is not None
    assert payload["status"] == "tracking"
    assert payload["last_frame_id"] == "native-1"
    assert payload["devices"]["camera_device"] == "/dev/video42"


def test_safe_subprocess_eye_adapter_degrades_without_crashing_parent() -> None:
    class Completed:
        returncode = -11
        stdout = ""
        stderr = "native plugin crashed"

    adapter = SafeSubprocessEyeAdapter(
        GStreamerHailoRealtimeConfig(camera_device="/dev/video42"),
        runner=lambda *_args, **_kwargs: Completed(),
    )

    payload = adapter.poll()

    assert payload["status"] == "degraded"
    assert payload["not_wired"] is False
    assert payload["stream_ready"] is False
    assert payload["subprocess"]["returncode"] == -11
    assert payload["pipeline"]["camera_device"] == "/dev/video42"


def test_state_file_eye_adapter_reads_persistent_vision_loop_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    config = GStreamerHailoRealtimeConfig(camera_device="/dev/video42", hailo_device="/dev/hailo0")
    write_vision_state(
        state_path,
        build_vision_state_payload(
            {
                "schema": "eihead.eye.realtime_status.v1",
                "kind": "realtime_vision_observation",
                "mode": "realtime_stream",
                "status": "tracking",
                "backend": "gstreamer_hailo",
                "stream_ready": True,
                "not_wired": False,
                "last_frame_id": "native-42",
                "detections": [{"label": "person", "score": 0.91}],
            },
            config=config,
            config_path="/etc/eihead/eihead.honjia.yaml",
            state_path=state_path,
            interval_s=0.1,
            updated_at_ts=100.0,
            pid=123,
        ),
    )

    adapter = StateFileEyeAdapter(config, state_path=state_path, clock=lambda: 101.0)
    payload = adapter.poll()

    assert payload["status"] == "tracking"
    assert payload["stream_ready"] is True
    assert payload["not_wired"] is False
    assert payload["last_frame_id"] == "native-42"
    assert payload["state_file"]["age_s"] == 1.0
    assert payload["state_file"]["path"] == str(state_path)
    assert payload["pipeline"]["camera_device"] == "/dev/video42"


def test_state_file_eye_adapter_reports_stale_state_without_polling_camera(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    config = GStreamerHailoRealtimeConfig(camera_device="/dev/video42")
    write_vision_state(
        state_path,
        build_vision_state_payload(
            {"status": "tracking", "stream_ready": True, "not_wired": False},
            config=config,
            config_path="/etc/eihead/eihead.honjia.yaml",
            state_path=state_path,
            interval_s=0.1,
            updated_at_ts=10.0,
            pid=123,
        ),
    )

    adapter = StateFileEyeAdapter(config, state_path=state_path, max_age_s=3.0, clock=lambda: 14.5)
    payload = adapter.poll()

    assert payload["status"] == "degraded"
    assert payload["status_reason"] == "vision_state_stale"
    assert payload["not_wired"] is False
    assert payload["stream_ready"] is False
    assert payload["state_file"]["age_s"] == 4.5


def test_honjia_runtime_reads_configured_persistent_vision_state_by_default(tmp_path: Path) -> None:
    state_path = tmp_path / "vision" / "state.json"
    config_path = tmp_path / "eihead.honjia.yaml"
    config_path.write_text(
        "\n".join(
            [
                "node_id: honjia",
                "devices:",
                "  camera:",
                "    path: /dev/video42",
                f"    state_path: {state_path.as_posix()}",
                "  hailo:",
                "    path: /dev/hailo0",
                "capabilities:",
                "  software:",
                "    vision_backend:",
                "      enabled: true",
                "      backend: hailo",
                "      limits:",
                "        realtime: true",
            ]
        ),
        encoding="utf-8",
    )
    write_vision_state(
        state_path,
        build_vision_state_payload(
            {
                "status": "tracking",
                "stream_ready": True,
                "not_wired": False,
                "last_frame_id": "state-backed-frame",
            },
            config=GStreamerHailoRealtimeConfig(camera_device="/dev/video42"),
            config_path=str(config_path),
            state_path=state_path,
            interval_s=0.1,
            updated_at_ts=time.time(),
            pid=123,
        ),
    )

    runtime = HeadRuntimeApp.from_config_path(str(config_path), native_environ={})
    payload = runtime.vision_realtime()

    assert payload is not None
    assert payload["status"] == "tracking"
    assert payload["stream_ready"] is True
    assert payload["last_frame_id"] == "state-backed-frame"
    assert payload["state_file"]["path"] == str(state_path)


def test_head_runtime_exposes_native_neck_status_for_monitoring() -> None:
    runtime = HeadRuntimeApp(
        body_runtime=FakeBodyRuntime(),
        native_providers={"neck": {"status": "wired", "provider": "fake-neck"}},
        neck_servo_adapter=FakeNeckAdapter(),
    )

    payload = build_neck_diagnostics_from_app(runtime, timestamp=10.0)

    assert payload["status"] == "wired"
    assert payload["not_wired"] is False
    assert payload["current_angle"] == 90
    assert payload["target_angle"] == 90
    assert payload["axis_support"]["pan"]["supported"] is True
    assert payload["axis_support"]["tilt"]["reason"] == "tilt_not_supported"


def test_from_config_path_exposes_degraded_voice_diagnostics_from_honjia_config(tmp_path: Path) -> None:
    config_path = tmp_path / "eihead.honjia.yaml"
    config_path.write_text(
        "\n".join(
            [
                "node_id: honjia",
                "devices:",
                "  microphone:",
                "    path: /dev/snd",
                "    device: plughw:CARD=U4K,DEV=0",
                "  speaker:",
                "    device: default",
                "capabilities:",
                "  software:",
                "    asr:",
                "      enabled: true",
                "      provider: sherpa_onnx",
                "      model: sherpa-onnx-streaming",
                "    tts:",
                "      enabled: true",
                "      provider: minimax",
                "      model: speech-2.8-hd",
                "      fallback_provider: piper",
                "      piper_model_path: /models/piper/zh_CN-huayan-medium.onnx",
            ]
        ),
        encoding="utf-8",
    )

    runtime = HeadRuntimeApp.from_config_path(str(config_path), native_environ={})
    payload = build_voice_diagnostics_from_app(runtime, timestamp=11.0)

    assert payload["status"] == "degraded"
    assert payload["not_wired"] is False
    assert payload["source"] in {"voice_realtime", "voice_status"}
    assert payload["ear"]["provider"] == "sherpa_onnx"
    assert payload["mouth"]["backend"] == "minimax"
    assert payload["realtime_audio"]["enabled"] is False


def test_native_voice_loop_config_reads_honjia_audio_devices_and_lstm_model_type(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "secret-minimax")
    config_path = tmp_path / "eihead.honjia.yaml"
    config_path.write_text(
        "\n".join(
            [
                "node_id: honjia",
                "devices:",
                "  microphone:",
                "    device: plughw:CARD=U4K,DEV=0",
                "    sample_rate: 16000",
                "    channels: 1",
                "  speaker:",
                "    device: plughw:CARD=SPA3700,DEV=0",
                "capabilities:",
                "  software:",
                "    asr:",
                "      enabled: true",
                "      provider: sherpa_onnx",
                "      model: sherpa-onnx-streaming",
                "      model_dir: /home/darrow/eibrain/models/asr/sherpa-onnx-streaming",
                "      model_type: lstm",
                "      vad_rms_threshold: 0.13",
                "      vad_min_voice_ms: 600",
                "      vad_end_silence_ms: 900",
                "      max_utterance_ms: 5000",
                "      limits:",
                "        streaming: true",
                "    tts:",
                "      enabled: true",
                "      provider: minimax",
                "      model: speech-2.8-hd",
                "      api_base_url: https://api.minimaxi.com",
                "      voice_id: female-shaonv",
                "      audio_format: wav",
                "      sample_rate: 32000",
                "      fallback_provider: piper",
                "      piper_command: /usr/local/bin/piper",
                "      piper_model_path: /models/piper/zh_CN-huayan-medium.onnx",
                "      piper_config_path: /models/piper/zh_CN-huayan-medium.onnx.json",
                "      playback_echo_cooldown_ms: 1200",
                "    dialogue:",
                "      enabled: true",
                "      provider: eibrain_subprocess",
                "      command: /opt/eihead/current/.venv/bin/python",
                "      module: apps.cognitive_runtime",
                "      cwd: /home/darrow/dev-project/eibrain",
                "      config_path: /home/darrow/dev-project/eibrain/config/eibrain.honjia.yaml",
                "      pythonpath: /home/darrow/dev-project/eibrain:/dev-project/eiprotocol",
                "      timeout_s: 55",
                "      session_id: honjia-voice",
                "      actor_id: darrow",
                "      wake_word_required: true",
                "      wake_words: [你好鸿途, 你好宏图]",
                "      end_phrases: [结束对话]",
                "      wake_ack_text: 我在。",
                "      end_ack_text: 好的，结束对话。",
            ]
        ),
        encoding="utf-8",
    )

    config = load_eihead_config(config_path)
    loop_config = native_voice_loop_config_from_eihead_config(config)
    runtime = build_native_voice_runtime(config)

    assert loop_config.microphone_device == "plughw:CARD=U4K,DEV=0"
    assert loop_config.speaker_device == "plughw:CARD=SPA3700,DEV=0"
    assert loop_config.sample_rate == 16000
    assert loop_config.channels == 1
    assert loop_config.vad_rms_threshold == 0.13
    assert loop_config.vad_min_voice_ms == 600
    assert loop_config.vad_end_silence_ms == 900
    assert loop_config.max_utterance_ms == 5000
    assert loop_config.asr_model_dir == "/home/darrow/eibrain/models/asr/sherpa-onnx-streaming"
    assert loop_config.asr_model_type == "lstm"
    assert loop_config.tts_backend == "minimax"
    assert loop_config.tts_fallback_provider == "piper"
    assert loop_config.piper_command == "/usr/local/bin/piper"
    assert loop_config.piper_model_path == "/models/piper/zh_CN-huayan-medium.onnx"
    assert loop_config.piper_config_path == "/models/piper/zh_CN-huayan-medium.onnx.json"
    assert loop_config.playback_echo_cooldown_ms == 1200
    assert loop_config.minimax_api_key == "secret-minimax"
    assert loop_config.minimax_api_base_url == "https://api.minimaxi.com"
    assert loop_config.minimax_model == "speech-2.8-hd"
    assert loop_config.minimax_voice_id == "female-shaonv"
    assert loop_config.minimax_audio_format == "wav"
    assert loop_config.minimax_sample_rate == 32000
    assert loop_config.dialogue_backend == "eibrain_subprocess"
    assert loop_config.dialogue_command == "/opt/eihead/current/.venv/bin/python"
    assert loop_config.dialogue_module == "apps.cognitive_runtime"
    assert loop_config.dialogue_cwd == "/home/darrow/dev-project/eibrain"
    assert loop_config.dialogue_config_path == "/home/darrow/dev-project/eibrain/config/eibrain.honjia.yaml"
    assert loop_config.dialogue_pythonpath == "/home/darrow/dev-project/eibrain:/dev-project/eiprotocol"
    assert loop_config.dialogue_timeout_s == 55
    assert loop_config.dialogue_session_id == "honjia-voice"
    assert loop_config.dialogue_actor_id == "darrow"
    assert loop_config.wake_word_required is True
    assert loop_config.wake_words == ("你好鸿途", "你好宏图")
    assert loop_config.end_phrases == ("结束对话",)
    assert loop_config.wake_ack_text == "我在。"
    assert loop_config.end_ack_text == "好的，结束对话。"
    assert runtime is not None


def test_native_voice_runtime_is_not_attached_without_sherpa_model_dir() -> None:
    from eihead.runtime.config import parse_eihead_config

    config = parse_eihead_config(
        {
            "node_id": "honjia",
            "capabilities": {
                "software": {
                    "asr": {
                        "enabled": True,
                        "provider": "sherpa_onnx",
                        "limits": {"streaming": True},
                    }
                }
            },
        }
    )

    assert build_native_voice_runtime(config) is None


def test_from_config_path_attaches_native_voice_runtime_from_honjia_config(tmp_path: Path, monkeypatch: Any) -> None:
    voice_runtime = FakeNativeVoiceRuntime()

    def fake_build_native_voice_runtime(config: object) -> FakeNativeVoiceRuntime:
        return voice_runtime

    monkeypatch.setattr(
        "eihead.runtime.app.build_native_voice_runtime",
        fake_build_native_voice_runtime,
    )
    config_path = tmp_path / "eihead.honjia.yaml"
    config_path.write_text(
        "\n".join(
            [
                "node_id: honjia",
                "capabilities:",
                "  software:",
                "    asr:",
                "      enabled: true",
                "      provider: sherpa_onnx",
                "      model_dir: /models/asr",
                "      model_type: lstm",
                "      limits:",
                "        streaming: true",
            ]
        ),
        encoding="utf-8",
    )

    runtime = HeadRuntimeApp.from_config_path(str(config_path), native_environ={})

    assert voice_runtime.started == 1
    assert runtime.voice_status()["voice_dialogue"]["running"] is True
    assert runtime.eivoice_runtime_status()["state"] == "running"


def test_native_voice_runtime_can_select_openclaw_realtime_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "eihead.honjia.yaml"
    config_path.write_text(
        "\n".join(
            [
                "node_id: honjia",
                "devices:",
                "  microphone:",
                "    device: plughw:CARD=U4K,DEV=0",
                "  speaker:",
                "    device: plughw:CARD=SPA3700,DEV=0",
                "capabilities:",
                "  software:",
                "    dialogue:",
                "      enabled: true",
                "      provider: openclaw_realtime",
                "      transport_provider: openclaw_realtime",
                "      ws_url: wss://openclaw.example/ws",
                "      fallback_transport_provider: legacy_native",
                "      session_id: honjia-voice",
            ]
        ),
        encoding="utf-8",
    )

    config = load_eihead_config(config_path)
    loop_config = native_voice_loop_config_from_eihead_config(config)
    runtime = build_native_voice_runtime(config)

    assert loop_config.transport_provider == "openclaw_realtime"
    assert loop_config.openclaw_ws_url == "wss://openclaw.example/ws"
    assert loop_config.fallback_transport_provider == "legacy_native"
    assert runtime is not None

    status = runtime.status()

    assert status["transport"]["transport"] == "openclaw_realtime"
    assert status["openclaw_ws"]["connected"] is False
    assert status["openclaw_ws"]["url"] == "wss://openclaw.example/ws"
    assert status["openclaw_ws"]["last_error"] == ""
    assert status["openclaw_ws"]["last_rx_ms"] is None
    assert status["openclaw_ws"]["last_tx_ms"] is None
    assert status["openclaw_ws"]["session_state"] == "idle"
