from __future__ import annotations

from contextlib import contextmanager
import json
import threading
from typing import Any, Iterator
from urllib import request

from eihead.monitoring.eivoice_runtime import build_eivoice_runtime_panel
from eihead.monitoring.voice import build_voice_diagnostics_from_app
from eihead.monitoring.web import create_server


def test_empty_status_waits_for_runtime_state() -> None:
    panel = build_eivoice_runtime_panel({})

    assert panel["state"] == "waiting"
    assert panel["conversationState"] == "unknown"
    assert panel["health"] == "waiting"
    assert panel["warnings"] == ["runtime state is missing"]
    assert panel["droppedTotal"] == 0


def test_normal_four_queues_are_healthy_with_fill_ratios() -> None:
    panel = build_eivoice_runtime_panel(
        {
            "state": "running",
            "conversation_state": "Idle",
            "queues": {
                "opus_encode_queue": {"depth": 1, "capacity": 4, "policy": "drop_oldest"},
                "ws_send_queue": {"depth": 2, "capacity": 4, "policy": "drop_newest"},
                "opus_decode_queue": {"depth": 0, "capacity": 5},
                "audio_playback_queue": {"depth": 3, "capacity": 6},
            },
            "audio_frontend": {
                "aec": {"enabled": True, "available": True},
                "ns": {"enabled": True},
                "vad": {"enabled": True},
                "loopback": {"enabled": True},
            },
            "wakeword": {"enabled": True, "state": "armed"},
        }
    )

    assert panel["health"] == "healthy"
    assert panel["state"] == "running"
    assert panel["conversationState"] == "Idle"
    assert panel["queueSummary"] == {
        "count": 4,
        "totalDepth": 6,
        "totalCapacity": 19,
        "maxFillRatio": 0.5,
    }
    assert panel["queues"]["opus_encode_queue"]["fillRatio"] == 0.25
    assert panel["queues"]["ws_send_queue"]["fillRatio"] == 0.5
    assert panel["queues"]["opus_decode_queue"]["fillRatio"] == 0.0
    assert panel["queues"]["audio_playback_queue"]["fillRatio"] == 0.5
    assert panel["droppedTotal"] == 0
    assert panel["audioFrontend"]["aec"] == {"enabled": True, "available": True}
    assert panel["wakeword"] == {"enabled": True, "state": "armed"}
    assert panel["warnings"] == []


def test_drop_oldest_and_drop_newest_counts_degrade_health() -> None:
    panel = build_eivoice_runtime_panel(
        {
            "state": "running",
            "queues": {
                "opus_encode_queue": {
                    "depth": 4,
                    "capacity": 4,
                    "dropped_oldest": 2,
                    "dropped_newest": 1,
                }
            },
        }
    )

    queue = panel["queues"]["opus_encode_queue"]
    assert panel["health"] == "degraded"
    assert panel["droppedTotal"] == 3
    assert queue["droppedOldest"] == 2
    assert queue["droppedNewest"] == 1
    assert "queue drops detected: 3" in panel["warnings"]


def test_acoustic_frontend_aec_unavailable_degrades_health() -> None:
    panel = build_eivoice_runtime_panel(
        {
            "state": "running",
            "acousticFrontend": {
                "aec": {"enabled": True, "available": False},
                "ns": {"enabled": True, "available": True},
                "vad": {"enabled": True, "available": True},
                "loopback": {"enabled": False},
            },
        }
    )

    assert panel["health"] == "degraded"
    assert panel["audioFrontend"]["aec"] == {"enabled": True, "available": False}
    assert "AEC unavailable" in panel["warnings"]


def test_audio_frontend_panel_preserves_loopback_route_and_last_capture() -> None:
    panel = build_eivoice_runtime_panel(
        {
            "state": "running",
            "audio_frontend": {
                "aec": {"enabled": True, "available": False},
                "loopback": {"enabled": True, "available": True},
                "devices": {
                    "capture": "alsa_input.usb-UGREEN_Camera_4K.analog-stereo",
                    "playback": "alsa_output.usb-Philips_SPA3700.analog-stereo",
                    "loopback": "alsa_output.usb-Philips_SPA3700.analog-stereo.monitor",
                },
                "audio_format": {"sample_rate": 48000, "frame_ms": 60, "channels": 2},
                "aec_backend": "pipewire-monitor",
                "aec_status": "unavailable",
                "last_capture": {
                    "playback_reference_available": True,
                    "reference_age_ms": 42.0,
                    "reference_matched_by": "pipewire-target",
                    "fallback_reason": "aec_unavailable",
                    "loopback_reference": {
                        "ready": False,
                        "state": "aec_unavailable",
                        "reason": "aec_unavailable",
                        "reference_age_ms": 42.0,
                        "matched_by": "pipewire-target",
                        "max_age_ms": 240,
                    },
                },
            },
        }
    )

    assert panel["audioFrontend"]["devices"]["playback"] == "alsa_output.usb-Philips_SPA3700.analog-stereo"
    assert panel["audioFrontend"]["audioFormat"] == {"sampleRate": 48000, "frameMs": 60, "channels": 2}
    assert panel["audioFrontend"]["aecBackend"] == "pipewire-monitor"
    assert panel["audioFrontend"]["aecStatus"] == "unavailable"
    assert panel["audioFrontend"]["lastCapture"]["playbackReferenceAvailable"] is True
    assert panel["audioFrontend"]["lastCapture"]["referenceAgeMs"] == 42.0
    assert panel["audioFrontend"]["lastCapture"]["referenceMatchedBy"] == "pipewire-target"
    assert panel["audioFrontend"]["lastCapture"]["loopbackReference"]["ready"] is False
    assert panel["audioFrontend"]["lastCapture"]["loopbackReference"]["state"] == "aec_unavailable"
    assert panel["audioFrontend"]["lastCapture"]["loopbackReference"]["reason"] == "aec_unavailable"


def test_audio_frontend_readiness_boolean_false_degrades_health() -> None:
    from eihead.devices.audio import evaluate_audio_frontend_readiness

    readiness = evaluate_audio_frontend_readiness(
        capture_device="hw:U4K",
        loopback_device=None,
        supports_aec=False,
        supports_ns=False,
        supports_vad=True,
    )

    panel = build_eivoice_runtime_panel(
        {
            "state": "running",
            "audio_frontend": readiness.to_dict(),
        }
    )

    assert panel["health"] == "degraded"
    assert panel["audioFrontend"]["aec"] == {"enabled": False}
    assert "AEC unavailable" in panel["warnings"]


def test_runtime_status_without_audio_frontend_is_not_reported_healthy() -> None:
    from eihead.eivoice_runtime import EiVoiceRuntimeCore

    runtime = EiVoiceRuntimeCore()
    runtime.state_machine.wake_detected()
    runtime.wakeword_buffer.append(_audio_frame())

    panel = build_eivoice_runtime_panel(runtime.status())

    assert panel["state"] == "conversation"
    assert panel["conversationState"] == "conversation"
    assert panel["wakeword"]["depth"] == 1
    assert panel["health"] == "degraded"
    assert "audio frontend readiness is missing" in panel["warnings"]


def test_transport_status_is_exposed_and_degrades_on_reconnect_errors() -> None:
    panel = build_eivoice_runtime_panel(
        {
            "state": "conversation",
            "audio_frontend": {
                "aec": {"enabled": True, "available": True},
                "ns": {"enabled": True, "available": True},
                "vad": {"enabled": True, "available": True},
                "loopback": {"enabled": True, "available": True},
            },
            "transport": {
                "transport": "fake_websocket",
                "state": "reconnect_wait",
                "heartbeat": {"awaiting_pong": True, "timed_out": True, "latency_ms": 1250},
                "reconnect": {"attempt": 2, "backoff_s": 4.0, "ready": False, "reason": "heartbeat_timeout"},
                "last_error": {"kind": "TimeoutError", "message": "pong timeout", "context": "heartbeat"},
            },
        }
    )

    assert panel["transport"]["name"] == "fake_websocket"
    assert panel["transport"]["state"] == "reconnect_wait"
    assert panel["transport"]["heartbeat"]["timed_out"] is True
    assert panel["transport"]["reconnect"]["attempt"] == 2
    assert panel["health"] == "degraded"
    assert "transport reconnect_wait" in panel["warnings"]
    assert "transport error: TimeoutError heartbeat" in panel["warnings"]


def test_voice_diagnostics_uses_tts_playback_as_mouth_authority() -> None:
    class App:
        def voice_status(self) -> dict[str, Any]:
            return {
                "mouth": {
                    "backend": "minimax",
                    "model": "play-model",
                    "voice_id": "play-voice",
                    "text_preview": "playback preview",
                    "tts_playback": {
                        "status": "playing",
                    },
                    "tts_plan": {
                        "status": "stopped",
                        "provider": "minimax",
                        "model": "plan-model",
                        "voice_id": "plan-voice",
                        "text_preview": "plan preview",
                    },
                }
            }

    payload = build_voice_diagnostics_from_app(App(), timestamp=1.0)
    mouth = payload.get("mouth")

    assert isinstance(mouth, dict)
    assert mouth["status"] == "playing"
    assert mouth["backend"] == "minimax"
    assert mouth["model"] == "play-model"
    assert mouth["voice_id"] == "play-voice"
    assert mouth["text_preview"] == "playback preview"
    assert "mouth: playing" in payload["readiness_message"]


def test_voice_diagnostics_does_not_infer_mouth_status_from_tts_plan_alone() -> None:
    class App:
        def voice_status(self) -> dict[str, Any]:
            return {
                "mouth": {
                    "backend": "noop",
                    "tts_playback": {
                        "provider": "noop",
                    },
                    "tts_plan": {
                        "status": "playing",
                        "model": "plan-model",
                        "voice_id": "plan-voice",
                        "text": "plan text",
                        "provider": "minimax",
                        "text_preview": "plan preview",
                    },
                }
            }

    payload = build_voice_diagnostics_from_app(App(), timestamp=2.0)
    mouth = payload.get("mouth")

    assert isinstance(mouth, dict)
    assert mouth["status"] == ""
    assert mouth["model"] == ""
    assert mouth["voice_id"] == ""
    assert mouth["text_preview"] == ""


def test_voice_diagnostics_uses_eivoice_runtime_when_voice_status_returns_none() -> None:
    class App:
        def voice_status(self) -> None:
            return None

        def status(self) -> dict[str, Any]:
            return {
                "eivoice_runtime": {
                    "state": "running",
                    "conversation_state": "Conversation",
                    "audio_frontend": {
                        "aec": {"enabled": True, "available": True},
                        "ns": {"enabled": True, "available": True},
                        "vad": {"enabled": True, "available": True},
                        "loopback": {"enabled": True, "available": True},
                        "devices": {"capture": "plughw:CARD=U4K,DEV=0"},
                    },
                    "asr": {"enabled": True, "provider": "sherpa_onnx", "status": "ready"},
                    "transport": {
                        "transport": "openai_realtime",
                        "state": "connected",
                        "heartbeat": {"latency_ms": 24},
                    },
                }
            }

    payload = build_voice_diagnostics_from_app(App(), timestamp=3.0)

    assert payload["source"] == "eivoice_runtime"
    assert payload["status"] == "degraded"
    assert payload["not_wired"] is False
    assert payload["ear"]["provider"] == "sherpa_onnx"
    assert payload["mouth"]["state"] == "not_wired"
    assert payload["realtime_audio"]["running"] is True
    assert payload["observation"]["eivoice_runtime"]["transport"]["state"] == "connected"


class RuntimePanelApp:
    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "ok",
            "runtime": "eihead",
            "node_id": "honjia-test",
            "eivoice_runtime": {
                "state": "running",
                "conversation_state": "Conversation",
                "queues": {
                    "opus_encode_queue": {"depth": 1, "capacity": 2},
                    "ws_send_queue": {"depth": 0, "capacity": 2},
                    "opus_decode_queue": {"depth": 0, "capacity": 2},
                    "audio_playback_queue": {"depth": 0, "capacity": 2},
                },
                "transport": {
                    "transport": "fake_websocket",
                    "state": "connected",
                    "heartbeat": {"latency_ms": 24},
                    "reconnect": {"attempt": 0},
                },
            },
        }

    def capabilities(self) -> dict[str, Any]:
        return {"capabilities": {}}


class AttachedVoiceRuntimeApp(RuntimePanelApp):
    def voice_status(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "ear": {
                "status": "listening",
                "provider": "sherpa_onnx",
                "capture": {"status": "running", "details": {"device": "plughw:CARD=U4K,DEV=0"}},
                "asr": {"enabled": True, "provider": "sherpa_onnx", "provider_state": "ready"},
            },
            "mouth": {
                "status": "idle",
                "backend": "espeak",
                "tts_playback": {"status": "ready", "details": {"device": "plughw:CARD=SPA3700,DEV=0"}},
            },
            "voice_dialogue": {"enabled": True, "running": True, "phase": "listening"},
            "realtime_audio": {"enabled": True, "running": True, "audio_level": 0.02},
            "readiness_message": "native realtime voice loop is attached",
        }

    def eivoice_runtime_status(self) -> dict[str, Any]:
        return {
            "state": "running",
            "conversation_state": "listening",
            "health": "healthy",
            "running": True,
            "audio_frontend": {"vad": {"enabled": True, "state": "listening"}},
        }


@contextmanager
def running_server(app: Any) -> Iterator[str]:
    server = create_server(app, host="127.0.0.1", port=0, clock=lambda: 123.0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def read_text(url: str) -> str:
    with request.urlopen(url, timeout=2.0) as response:
        return response.read().decode("utf-8")


def read_json(url: str) -> dict[str, Any]:
    return json.loads(read_text(url))


def _audio_frame():
    from eihead.eivoice_runtime import AudioFrame

    return AudioFrame(pcm=b"x")


def test_web_exposes_eivoice_runtime_panel() -> None:
    with running_server(RuntimePanelApp()) as base_url:
        body = read_text(f"{base_url}/")
        payload = read_json(f"{base_url}/api/eivoice/runtime")

    assert "EIVoice Runtime" in body
    assert "Transport state" in body
    assert "fake_websocket / connected" in body
    assert "eivoiceRuntime" in body
    assert payload["eivoiceRuntime"]["state"] == "running"
    assert payload["eivoiceRuntime"]["conversationState"] == "Conversation"
    assert payload["eivoiceRuntime"]["transport"]["name"] == "fake_websocket"


def test_web_voice_realtime_reports_attached_native_runtime_as_live() -> None:
    with running_server(AttachedVoiceRuntimeApp()) as base_url:
        payload = read_json(f"{base_url}/api/voice/realtime")
        runtime_payload = read_json(f"{base_url}/api/eivoice/runtime")

    assert payload["status"] in {"ready", "ok", "wired"}
    assert payload["not_wired"] is False
    assert payload["realtime_audio"]["running"] is True
    assert payload["dialogue"]["running"] is True
    assert "native realtime voice loop is attached" in payload["readiness_message"]
    assert runtime_payload["eivoiceRuntime"]["state"] == "running"
