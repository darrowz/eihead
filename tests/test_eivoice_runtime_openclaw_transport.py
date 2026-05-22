from __future__ import annotations

from array import array
import base64
import json
import subprocess
import time

import pytest

from eihead.eivoice_runtime import AudioFrame, EiVoiceRuntimeRunner, NoOpAcousticFrontend
from eihead.eivoice_runtime.native_loop import NativeVoiceLoopConfig
from eihead.eivoice_runtime.openclaw_transport import OpenClawRealtimeTransport
from eihead.runtime.openclaw_runtime import (
    AplayPcmPlaybackSink,
    OpenClawPlaybackEchoGate,
    OpenClawRealtimeRuntime,
    _session_config,
)


class ManualClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeWebSocket:
    def __init__(self, incoming: list[object] | None = None) -> None:
        self.incoming = list(incoming or [])
        self.sent: list[object] = []
        self.closed = False
        self.timeout: float | None = None
        self.timeout_history: list[float] = []

    def send(self, payload: object) -> None:
        self.sent.append(payload)

    def recv(self) -> object:
        if not self.incoming:
            raise TimeoutError("no message available")
        return self.incoming.pop(0)

    def settimeout(self, value: float) -> None:
        self.timeout = value
        self.timeout_history.append(value)

    def close(self) -> None:
        self.closed = True


class SendTimeoutGuardWebSocket(FakeWebSocket):
    def send(self, payload: object) -> None:
        if self.timeout is not None and self.timeout < 1.0:
            raise TimeoutError(f"send used receive timeout: {self.timeout}")
        super().send(payload)


class EmptyCaptureSource:
    def read_frame(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def status(self) -> dict[str, object]:
        return {"running": False}


class RaisingCaptureSource:
    def read_frame(self) -> None:
        raise AssertionError("capture should not run while output is active")

    def stop(self) -> None:
        return None

    def status(self) -> dict[str, object]:
        return {"running": False}


class FakePlaybackSink:
    def __init__(self) -> None:
        self.played: list[object] = []
        self.active = False

    def play(self, frame: object) -> None:
        self.played.append(frame)

    def stop(self) -> None:
        return None

    def status(self) -> dict[str, object]:
        return {"running": False, "active": self.active}


class FakeSegmentTranscriber:
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls: list[list[AudioFrame]] = []

    def transcribe(self, frames: list[AudioFrame]) -> str:
        self.calls.append(list(frames))
        return self.texts.pop(0) if self.texts else ""

    def status(self) -> dict[str, object]:
        return {"provider": "fake_asr", "state": "ready", "calls": len(self.calls)}


class FakeConnectedTextTransport:
    def __init__(self) -> None:
        self.sent_texts: list[str] = []

    def status(self) -> dict[str, object]:
        return {
            "state": "connected",
            "connection": {"state": "connected"},
            "openclaw_ws": {
                "connected": True,
                "url": "wss://openclaw.example/realtime",
                "session_state": "ready",
                "session_id": "relay-1",
                "last_user_transcript": "",
                "last_assistant_transcript": "你好，我是鸿途。",
                "latency_ms": {},
            },
        }

    def close(self, reason: str | None = None) -> None:
        return None

    def cancel_output(self, reason: str) -> bool:
        return True

    def send_text(self, text: str) -> bool:
        self.sent_texts.append(text)
        return True


class FakePopenProcess:
    class _Stdin:
        def __init__(self) -> None:
            self.payload = b""
            self.closed = False

        def write(self, payload: bytes) -> int:
            self.payload += payload
            return len(payload)

        def flush(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class _Stderr:
        def read(self) -> bytes:
            return b""

    def __init__(self) -> None:
        self.stdin = self._Stdin()
        self.stderr = self._Stderr()
        self.terminated = False
        self.killed = False

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


def _pcm_constant(amplitude: int, samples: int = 160) -> bytes:
    return array("h", [int(amplitude)] * int(samples)).tobytes()


def test_openclaw_transport_connects_and_sends_audio_chunk_with_default_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCLAW_REALTIME_TOKEN", "env-token")
    socket = FakeWebSocket(
        incoming=[
            {"type": "event", "event": "connect.challenge", "payload": {"nonce": "nonce-1"}},
            {
                "type": "res",
                "id": "1",
                "ok": True,
                "payload": {
                    "auth": {"role": "operator", "scopes": ["operator.read", "operator.write"]}
                },
            },
            {
                "type": "res",
                "id": "2",
                "ok": True,
                "payload": {
                    "provider": "openai",
                    "transport": "gateway-relay",
                    "relaySessionId": "relay-1",
                    "audio": {
                        "inputEncoding": "pcm16",
                        "inputSampleRateHz": 24000,
                        "outputEncoding": "pcm16",
                        "outputSampleRateHz": 24000,
                    },
                },
            },
            {"type": "event", "event": "talk.event", "payload": {"relaySessionId": "relay-1", "type": "ready"}},
        ]
    )
    captured: dict[str, object] = {}

    def _connect(url: str, *, header: list[str], timeout: float) -> FakeWebSocket:
        captured["url"] = url
        captured["header"] = list(header)
        captured["timeout"] = timeout
        return socket

    transport = OpenClawRealtimeTransport(
        url="wss://openclaw.example/realtime",
        websocket_factory=_connect,
        headers={"X-Trace-Id": "trace-1"},
        clock=ManualClock(10.0),
        session_config={"sessionKey": "honjia-voice", "voice": "Zephyr", "provider": "openai"},
        client_platform="test",
        client_device_family="unit",
    )

    transport.connect()
    accepted = transport.send_event(
        {
            "uid": "user-1",
            "mid": "mid-1",
            "contentType": "AUDIO_CHUNK",
            "content": {
                "eventType": "AUDIO_CHUNK",
                "index": 7,
                "audioBase64": "AQID",
                "durationMs": 40,
                "sampleRateHz": 24000,
                "channels": 1,
            },
        }
    )

    assert accepted is True
    assert captured == {
        "url": "wss://openclaw.example/realtime",
        "header": [
            "X-Trace-Id: trace-1",
        ],
        "timeout": 10.0,
    }
    sent = [json.loads(str(item)) for item in socket.sent]
    assert sent == [
        {
            "type": "req",
            "id": "1",
            "method": "connect",
            "params": {
                "minProtocol": 4,
                "maxProtocol": 4,
                "client": {
                    "id": "cli",
                    "version": "eihead",
                    "platform": "test",
                    "mode": "cli",
                    "deviceFamily": "unit",
                },
                "role": "operator",
                "scopes": ["operator.read", "operator.write"],
                "caps": ["tool-events"],
                "auth": {"token": "env-token"},
                "userAgent": "eihead-openclaw-realtime",
                "locale": "zh-CN",
            },
        },
        {
            "type": "req",
            "id": "2",
            "method": "talk.session.create",
            "params": {
                "sessionKey": "honjia-voice",
                "mode": "realtime",
                "transport": "gateway-relay",
                "brain": "agent-consult",
                "provider": "openai",
                "voice": "Zephyr",
            },
        },
        {
            "type": "req",
            "id": "3",
            "method": "talk.session.appendAudio",
            "params": {
                "sessionId": "relay-1",
                "audioBase64": "AQID",
                "timestamp": 10000,
            },
        },
    ]
    assert transport.status()["connection"]["state"] == "connected"
    assert 2.0 in socket.timeout_history
    assert socket.timeout_history[-1] == 2.0
    transport.close()
    assert json.loads(str(socket.sent[-1])) == {
        "type": "req",
        "id": "4",
        "method": "talk.session.close",
        "params": {"sessionId": "relay-1"},
    }
    assert socket.closed is True


def test_openclaw_transport_sends_recognized_text_to_realtime_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCLAW_REALTIME_TOKEN", "env-token")
    socket = FakeWebSocket(
        incoming=[
            {"type": "event", "event": "connect.challenge", "payload": {"nonce": "nonce-1"}},
            {
                "type": "res",
                "id": "1",
                "ok": True,
                "payload": {
                    "auth": {"role": "operator", "scopes": ["operator.read", "operator.write"]}
                },
            },
            {
                "type": "res",
                "id": "2",
                "ok": True,
                "payload": {
                    "provider": "openai",
                    "transport": "gateway-relay",
                    "relaySessionId": "relay-1",
                    "audio": {
                        "inputEncoding": "pcm16",
                        "inputSampleRateHz": 24000,
                        "outputEncoding": "pcm16",
                        "outputSampleRateHz": 24000,
                    },
                },
            },
            {"type": "event", "event": "talk.event", "payload": {"relaySessionId": "relay-1", "type": "ready"}},
        ]
    )
    transport = OpenClawRealtimeTransport(
        url="wss://openclaw.example/realtime",
        websocket_factory=lambda url, *, header, timeout: socket,
        clock=ManualClock(20.0),
        client_platform="test",
        client_device_family="unit",
    )

    transport.connect()
    accepted = transport.send_text("  介绍下你自己  ")

    assert accepted is True
    sent = [json.loads(str(item)) for item in socket.sent]
    assert sent[-1] == {
        "type": "req",
        "id": "3",
        "method": "talk.session.sendText",
        "params": {
            "sessionId": "relay-1",
            "text": "介绍下你自己",
        },
    }


def test_openclaw_transport_can_use_configured_gateway_protocol_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCLAW_REALTIME_TOKEN", "env-token")
    socket = FakeWebSocket(
        incoming=[
            {"type": "event", "event": "connect.challenge", "payload": {"nonce": "nonce-1"}},
            {"type": "res", "id": "1", "ok": True, "payload": {"auth": {"role": "operator"}}},
            {
                "type": "res",
                "id": "2",
                "ok": True,
                "payload": {
                    "provider": "openai",
                    "transport": "gateway-relay",
                    "relaySessionId": "relay-1",
                    "audio": {
                        "inputEncoding": "pcm16",
                        "inputSampleRateHz": 24000,
                        "outputEncoding": "pcm16",
                        "outputSampleRateHz": 24000,
                    },
                },
            },
            {"type": "event", "event": "talk.event", "payload": {"relaySessionId": "relay-1", "type": "ready"}},
        ]
    )
    transport = OpenClawRealtimeTransport(
        url="wss://openclaw.example/realtime",
        websocket_factory=lambda url, *, header, timeout: socket,
        protocol="3",
    )

    transport.connect()

    connect = json.loads(str(socket.sent[0]))
    assert connect["params"]["minProtocol"] == 3
    assert connect["params"]["maxProtocol"] == 3


def test_openclaw_transport_reconnects_instead_of_sending_audio_to_ended_relay_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCLAW_REALTIME_TOKEN", "env-token")
    clock = ManualClock(20.0)
    socket = FakeWebSocket(
        incoming=[
            {"type": "event", "event": "connect.challenge", "payload": {"nonce": "nonce-1"}},
            {"type": "res", "id": "1", "ok": True, "payload": {"auth": {"role": "operator"}}},
            {
                "type": "res",
                "id": "2",
                "ok": True,
                "payload": {
                    "provider": "openai",
                    "transport": "gateway-relay",
                    "relaySessionId": "relay-1",
                    "audio": {
                        "inputEncoding": "pcm16",
                        "inputSampleRateHz": 24000,
                        "outputEncoding": "pcm16",
                        "outputSampleRateHz": 24000,
                    },
                },
            },
            {"type": "event", "event": "talk.event", "payload": {"relaySessionId": "relay-1", "type": "ready"}},
        ]
    )
    transport = OpenClawRealtimeTransport(
        url="wss://openclaw.example/realtime",
        websocket_factory=lambda url, *, header, timeout: socket,
        clock=clock,
    )

    transport.connect()
    transport._session_state = "ended"

    accepted = transport.send_event(
        {
            "uid": "user-1",
            "mid": "mid-1",
            "contentType": "AUDIO_CHUNK",
            "content": {
                "eventType": "AUDIO_CHUNK",
                "index": 8,
                "audioBase64": "AQID",
                "durationMs": 40,
                "sampleRateHz": 24000,
                "channels": 1,
            },
        }
    )

    sent = [json.loads(str(item)) for item in socket.sent]
    assert accepted is False
    assert socket.closed is True
    assert [item["method"] for item in sent] == ["connect", "talk.session.create", "talk.session.close"]
    assert transport.status()["connection"]["state"] == "reconnect_wait"
    assert transport.status()["reconnect"]["reason"] == "relay_session_ended"
    assert transport.status()["openclaw_ws"]["session_state"] == "ended"


def test_openclaw_device_auth_uses_epoch_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    private_key = ed25519.Ed25519PrivateKey.generate()
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    public_key_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")
    captured: dict[str, str] = {}

    def _signer(private_key_text: str, payload: str) -> str:
        captured["private_key"] = private_key_text
        captured["payload"] = payload
        return "signature-1"

    monkeypatch.setattr("eihead.eivoice_runtime.openclaw_transport.time.time", lambda: 1_700_000_000.25)
    transport = OpenClawRealtimeTransport(
        url="wss://openclaw.example/realtime",
        clock=ManualClock(10.0),
        device_identity={
            "deviceId": "device-1",
            "privateKeyPem": private_key_pem,
            "publicKeyPem": public_key_pem,
        },
        device_signer=_signer,
        client_platform="LINUX",
        client_device_family="Pi",
    )

    device = transport._build_connect_device(nonce="nonce-1", token="token-1")

    assert device is not None
    assert device["id"] == "device-1"
    assert device["signedAt"] == 1_700_000_000_250
    assert device["signature"] == "signature-1"
    assert captured["private_key"] == private_key_pem
    assert "|1700000000250|" in captured["payload"]
    assert captured["payload"].endswith("|linux|pi")


def test_openclaw_transport_receives_audio_delta_and_updates_status() -> None:
    clock = ManualClock(20.0)
    audio_base64 = base64.b64encode(b"\0" * 19200).decode("ascii")
    socket = FakeWebSocket(
        incoming=[
            {"type": "event", "event": "connect.challenge", "payload": {"nonce": "nonce-1"}},
            {"type": "res", "id": "1", "ok": True, "payload": {"auth": {"role": "operator"}}},
            {
                "type": "res",
                "id": "2",
                "ok": True,
                "payload": {
                    "provider": "openai",
                    "transport": "gateway-relay",
                    "relaySessionId": "relay-1",
                    "audio": {
                        "inputEncoding": "pcm16",
                        "inputSampleRateHz": 24000,
                        "outputEncoding": "pcm16",
                        "outputSampleRateHz": 24000,
                    },
                },
            },
            {"type": "event", "event": "talk.event", "payload": {"relaySessionId": "relay-1", "type": "ready"}},
            json.dumps(
                {
                    "type": "event",
                    "event": "talk.event",
                    "payload": {
                        "relaySessionId": "relay-1",
                        "type": "audio",
                        "audioBase64": audio_base64,
                    },
                }
            ),
        ]
    )

    transport = OpenClawRealtimeTransport(
        url="wss://openclaw.example/realtime",
        token="explicit-token",
        websocket_factory=lambda url, *, header, timeout: socket,
        clock=clock,
        receive_timeout=2.5,
    )

    transport.connect()
    clock.advance(0.75)

    audio_event = transport.receive_event()

    assert socket.timeout == 2.5
    assert audio_event == {
        "contentType": "AUDIO_CHUNK",
            "content": {
                "eventType": "AUDIO_CHUNK",
                "audioBase64": audio_base64,
                "index": 0,
                "durationMs": 400,
                "sampleRateHz": 24000,
                "channels": 1,
                "metadata": {
                    "source": "openclaw_realtime",
                    "messageType": "talk.event.audio",
                },
            },
        }
    status = transport.status()
    assert status["openclaw_ws"]["session_state"] == "speaking"
    assert status["openclaw_ws"]["last_rx_ms"] == 20750


def test_openclaw_transport_drains_append_audio_acks_before_audio_event() -> None:
    socket = FakeWebSocket(
        incoming=[
            {"type": "event", "event": "connect.challenge", "payload": {"nonce": "nonce-1"}},
            {"type": "res", "id": "1", "ok": True, "payload": {"auth": {"role": "operator"}}},
            {
                "type": "res",
                "id": "2",
                "ok": True,
                "payload": {"relaySessionId": "relay-1", "audio": {"inputSampleRateHz": 24000}},
            },
            {"type": "event", "event": "talk.event", "payload": {"relaySessionId": "relay-1", "type": "ready"}},
            {"type": "res", "id": "3", "ok": True, "payload": {"ok": True}},
            {"type": "res", "id": "4", "ok": True, "payload": {"ok": True}},
            {
                "type": "event",
                "event": "talk.event",
                "payload": {
                    "relaySessionId": "relay-1",
                    "type": "audio",
                    "audioBase64": "BAUG",
                },
            },
        ]
    )
    transport = OpenClawRealtimeTransport(
        url="wss://openclaw.example/realtime",
        token="token",
        websocket_factory=lambda url, *, header, timeout: socket,
    )

    transport.connect()

    event = transport.receive_event()

    assert event is not None
    assert event["contentType"] == "AUDIO_CHUNK"
    assert event["content"]["audioBase64"] == "BAUG"
    assert transport.status()["openclaw_ws"]["last_audio_rx_ms"] is not None


def test_openclaw_transport_uses_send_timeout_after_short_receive_poll() -> None:
    socket = SendTimeoutGuardWebSocket(
        incoming=[
            {"type": "event", "event": "connect.challenge", "payload": {"nonce": "nonce-1"}},
            {"type": "res", "id": "1", "ok": True, "payload": {"auth": {"role": "operator"}}},
            {
                "type": "res",
                "id": "2",
                "ok": True,
                "payload": {"relaySessionId": "relay-1", "audio": {"inputSampleRateHz": 24000}},
            },
            {"type": "event", "event": "talk.event", "payload": {"relaySessionId": "relay-1", "type": "ready"}},
        ]
    )
    transport = OpenClawRealtimeTransport(
        url="wss://openclaw.example/realtime",
        token="token",
        websocket_factory=lambda url, *, header, timeout: socket,
        send_timeout=1.5,
        receive_timeout=0.02,
    )

    transport.connect()
    assert transport.receive_event() is None

    accepted = transport.send_event(
        {
            "contentType": "AUDIO_CHUNK",
            "content": {"eventType": "AUDIO_CHUNK", "audioBase64": "AQID"},
        }
    )

    assert accepted is True
    assert socket.timeout_history[-1] == 1.5


def test_openclaw_transport_send_event_without_connection_records_error() -> None:
    transport = OpenClawRealtimeTransport(url="wss://openclaw.example/realtime", token="token")

    accepted = transport.send_event(
        {
            "contentType": "AUDIO_CHUNK",
            "content": {"eventType": "AUDIO_CHUNK", "audioBase64": "AQID"},
        }
    )

    assert accepted is False
    status = transport.status()
    assert status["connection"]["state"] == "idle"
    assert status["last_error"]["context"] == "send_event"
    assert status["last_error"]["kind"] == "RuntimeError"


def test_openclaw_transport_cancel_output_sends_gateway_request() -> None:
    socket = FakeWebSocket(
        incoming=[
            {"type": "event", "event": "connect.challenge", "payload": {"nonce": "nonce-1"}},
            {"type": "res", "id": "1", "ok": True, "payload": {"auth": {"role": "operator"}}},
            {
                "type": "res",
                "id": "2",
                "ok": True,
                "payload": {
                    "relaySessionId": "relay-1",
                    "audio": {
                        "inputSampleRateHz": 24000,
                        "outputSampleRateHz": 24000,
                    },
                },
            },
            {"type": "event", "event": "talk.event", "payload": {"relaySessionId": "relay-1", "type": "ready"}},
        ]
    )
    transport = OpenClawRealtimeTransport(
        url="wss://openclaw.example/realtime",
        token="token",
        websocket_factory=lambda url, *, header, timeout: socket,
    )

    transport.connect()

    assert transport.cancel_output("barge-in") is True
    payload = json.loads(socket.sent[-1])
    assert payload == {
        "type": "req",
        "id": "3",
        "method": "talk.session.cancelOutput",
        "params": {"sessionId": "relay-1", "reason": "barge-in"},
    }
    assert transport.status()["openclaw_ws"]["session_state"] == "interrupted"


def test_openclaw_playback_sink_reports_active_audio_window() -> None:
    clock = ManualClock(10.0)
    sink = AplayPcmPlaybackSink(device="default", active_grace_s=0.1, clock=clock)
    sink._ensure_process = lambda *, sample_rate, channels: None  # type: ignore[method-assign]

    sink.play(AudioFrame(pcm=_pcm_constant(1000), duration_ms=200, sample_rate_hz=16000, channels=1))

    assert sink.status()["active"] is True
    clock.advance(0.31)
    assert sink.status()["active"] is False


def test_openclaw_playback_sink_keeps_initial_window_when_starting_aplay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = ManualClock(20.0)
    processes: list[FakePopenProcess] = []

    def fake_popen(*args: object, **kwargs: object) -> FakePopenProcess:
        process = FakePopenProcess()
        processes.append(process)
        return process

    monkeypatch.setattr("eihead.runtime.openclaw_runtime.subprocess.Popen", fake_popen)
    sink = AplayPcmPlaybackSink(device="default", active_grace_s=0.1, clock=clock)

    sink.play(AudioFrame(pcm=_pcm_constant(1000), duration_ms=200, sample_rate_hz=16000, channels=1))

    status = sink.status()
    assert status["active"] is True
    assert status["queued_audio_until_s"] == pytest.approx(20.2)
    assert status["active_until_s"] == pytest.approx(20.3)
    assert processes[0].stdin.payload


def test_openclaw_playback_sink_accumulates_queued_audio_without_stacking_grace() -> None:
    clock = ManualClock(10.0)
    sink = AplayPcmPlaybackSink(device="default", active_grace_s=0.1, clock=clock)
    sink._ensure_process = lambda *, sample_rate, channels: None  # type: ignore[method-assign]

    sink.play(AudioFrame(pcm=_pcm_constant(1000), duration_ms=200, sample_rate_hz=16000, channels=1))
    clock.advance(0.05)
    sink.play(AudioFrame(pcm=_pcm_constant(1000), duration_ms=200, sample_rate_hz=16000, channels=1))

    assert sink.status()["active_until_s"] == pytest.approx(10.5)


def test_openclaw_runtime_uses_long_playback_grace_to_prevent_speaker_echo() -> None:
    runtime = OpenClawRealtimeRuntime(
        NativeVoiceLoopConfig(
            openclaw_ws_url="wss://openclaw.example/realtime",
            playback_echo_cooldown_ms=350,
        ),
        transport_factory=lambda config: OpenClawRealtimeTransport(
            url=config.openclaw_ws_url,
            token="token",
            websocket_factory=lambda url, *, header, timeout: FakeWebSocket(),
        ),
        capture_source=EmptyCaptureSource(),
    )

    assert isinstance(runtime.playback_sink, AplayPcmPlaybackSink)
    assert runtime.playback_sink.active_grace_s >= 4.0


def test_openclaw_echo_gate_suppresses_playback_echo_and_allows_barge_in() -> None:
    sink = FakePlaybackSink()
    sink.active = True
    barge_ins: list[dict[str, object]] = []
    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: barge_ins.append(dict(payload)),
        barge_in_enabled=True,
        rms_threshold=0.1,
        peak_threshold=0.2,
        consecutive_frames=2,
    )

    quiet = AudioFrame(pcm=_pcm_constant(500), duration_ms=120, sample_rate_hz=16000, channels=1)
    loud = AudioFrame(pcm=_pcm_constant(10000), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(quiet) is None
    assert gate.process_capture(loud) is None
    assert gate.process_capture(loud) is loud
    readiness = gate.readiness()

    assert len(barge_ins) == 1
    assert readiness["playbackGate"]["suppressedFrames"] == 2
    assert readiness["playbackGate"]["bargeInCount"] == 1
    assert readiness["playbackGate"]["bargeInEnabled"] is True
    assert readiness["playbackGate"]["lastBargeIn"]["reason"] == "barge-in"


def test_openclaw_echo_gate_defaults_to_suppressing_loud_playback_without_barge_in() -> None:
    sink = FakePlaybackSink()
    sink.active = True
    barge_ins: list[dict[str, object]] = []
    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: barge_ins.append(dict(payload)),
        rms_threshold=0.1,
        peak_threshold=0.2,
        consecutive_frames=2,
    )

    loud = AudioFrame(pcm=_pcm_constant(12000), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(loud) is None
    assert gate.process_capture(loud) is None
    readiness = gate.readiness()

    assert barge_ins == []
    assert readiness["vad"]["state"] == "echo_suppression_only"
    assert readiness["playbackGate"]["bargeInEnabled"] is False
    assert readiness["playbackGate"]["suppressedFrames"] == 2
    assert readiness["playbackGate"]["bargeInCount"] == 0


def test_openclaw_echo_gate_local_vad_drops_idle_noise_before_upstream() -> None:
    sink = FakePlaybackSink()
    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: None,
        local_vad_enabled=True,
        local_vad_rms_threshold=0.1,
        local_vad_peak_threshold=0.2,
    )

    quiet = AudioFrame(pcm=_pcm_constant(500), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(quiet) is None
    readiness = gate.readiness()

    assert readiness["vad"]["enabled"] is True
    assert readiness["vad"]["available"] is True
    assert readiness["vad"]["state"] == "local_vad_ready"
    assert readiness["localVad"]["droppedFrames"] == 1
    assert readiness["localVad"]["passedFrames"] == 0


def test_openclaw_echo_gate_local_vad_passes_speech_and_trailing_silence() -> None:
    sink = FakePlaybackSink()
    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: None,
        local_vad_enabled=True,
        local_vad_rms_threshold=0.1,
        local_vad_peak_threshold=0.2,
        local_vad_hangover_frames=2,
    )
    speech = AudioFrame(pcm=_pcm_constant(10000), duration_ms=120, sample_rate_hz=16000, channels=1)
    quiet = AudioFrame(pcm=_pcm_constant(500), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(speech) is speech
    assert gate.process_capture(quiet) is quiet
    assert gate.process_capture(quiet) is quiet
    assert gate.process_capture(quiet) is None
    readiness = gate.readiness()

    assert readiness["localVad"]["passedFrames"] == 3
    assert readiness["localVad"]["droppedFrames"] == 1
    assert readiness["localVad"]["active"] is False


def test_openclaw_echo_gate_local_vad_caps_long_voice_segments() -> None:
    sink = FakePlaybackSink()
    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: None,
        local_vad_enabled=True,
        local_vad_rms_threshold=0.1,
        local_vad_peak_threshold=0.2,
        local_vad_max_frames=3,
    )
    speech = AudioFrame(pcm=_pcm_constant(10000), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(speech) is speech
    assert gate.process_capture(speech) is speech
    assert gate.process_capture(speech) is speech
    assert gate.process_capture(speech) is None
    readiness = gate.readiness()

    assert readiness["localVad"]["passedFrames"] == 3
    assert readiness["localVad"]["droppedFrames"] == 1
    assert readiness["localVad"]["maxFrames"] == 3


def test_openclaw_echo_gate_local_wake_gate_drops_background_speech_before_upstream() -> None:
    sink = FakePlaybackSink()
    events: list[dict[str, object]] = []
    transcriber = FakeSegmentTranscriber("旁边电视的声音")
    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: None,
        local_transcriber=transcriber,
        wake_word_required=True,
        wake_words=("你好鸿途", "你好宏图"),
        end_phrases=("结束对话",),
        on_gate_event=lambda payload: events.append(dict(payload)),
        local_vad_enabled=True,
        local_vad_rms_threshold=0.1,
        local_vad_peak_threshold=0.2,
        local_vad_hangover_frames=1,
    )
    speech = AudioFrame(pcm=_pcm_constant(10000), duration_ms=120, sample_rate_hz=16000, channels=1)
    quiet = AudioFrame(pcm=_pcm_constant(500), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(quiet) is None
    readiness = gate.readiness()

    assert len(transcriber.calls) == 1
    assert events == []
    assert readiness["localWakeGate"]["state"] == "armed"
    assert readiness["localWakeGate"]["conversationActive"] is False
    assert readiness["localWakeGate"]["lastTranscript"] == "旁边电视的声音"
    assert readiness["localWakeGate"]["lastGateReason"] == "wake_word_required"
    assert readiness["localWakeGate"]["droppedSegments"] == 1
    assert readiness["localVad"]["passedFrames"] == 0


def test_openclaw_echo_gate_local_wake_gate_activates_then_streams_next_utterance() -> None:
    sink = FakePlaybackSink()
    events: list[dict[str, object]] = []
    transcriber = FakeSegmentTranscriber("你好鸿途", "今天天气怎么样")
    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: None,
        local_transcriber=transcriber,
        wake_word_required=True,
        wake_words=("你好鸿途", "你好宏图"),
        end_phrases=("结束对话",),
        on_gate_event=lambda payload: events.append(dict(payload)),
        local_vad_enabled=True,
        local_vad_rms_threshold=0.1,
        local_vad_peak_threshold=0.2,
        local_vad_hangover_frames=1,
    )
    speech = AudioFrame(pcm=_pcm_constant(10000), duration_ms=120, sample_rate_hz=16000, channels=1)
    quiet = AudioFrame(pcm=_pcm_constant(500), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(quiet) is speech
    readiness = gate.readiness()

    assert events[0]["type"] == "wake_detected"
    assert events[0]["reply_text"] == "我在。"
    assert readiness["localWakeGate"]["state"] == "active"
    assert readiness["localWakeGate"]["conversationActive"] is True
    assert readiness["localWakeGate"]["lastTranscript"] == "今天天气怎么样"
    assert readiness["localWakeGate"]["lastGateReason"] == "active_utterance_replayed"


def test_openclaw_echo_gate_local_wake_gate_replays_wake_segment_when_it_has_remainder() -> None:
    sink = FakePlaybackSink()
    events: list[dict[str, object]] = []
    transcriber = FakeSegmentTranscriber("你好鸿途介绍一下")
    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: None,
        local_transcriber=transcriber,
        wake_word_required=True,
        wake_words=("你好鸿途",),
        end_phrases=("结束对话",),
        on_gate_event=lambda payload: events.append(dict(payload)),
        local_vad_enabled=True,
        local_vad_rms_threshold=0.1,
        local_vad_peak_threshold=0.2,
        local_vad_hangover_frames=1,
    )
    speech = AudioFrame(pcm=_pcm_constant(10000), duration_ms=120, sample_rate_hz=16000, channels=1)
    quiet = AudioFrame(pcm=_pcm_constant(500), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    first_replay = gate.process_capture(quiet)
    second_replay = gate.process_capture(quiet)
    readiness = gate.readiness()

    assert first_replay is speech
    assert second_replay is not None
    assert events[0]["type"] == "wake_detected"
    assert events[0]["remainder"] == "介绍一下"
    assert readiness["localWakeGate"]["state"] == "active"
    assert readiness["localWakeGate"]["conversationActive"] is True
    assert readiness["localWakeGate"]["lastGateReason"] == "wake_remainder_replayed"
    assert readiness["localWakeGate"]["lastStatus"] == "conversation_active"
    assert readiness["localVad"]["passedFrames"] == 2


def test_openclaw_echo_gate_keeps_conversation_active_after_wake_remainder_text_is_sent() -> None:
    sink = FakePlaybackSink()
    events: list[dict[str, object]] = []
    transcriber = FakeSegmentTranscriber("鸿途介绍一下你自己")

    def handle_event(payload: object) -> bool:
        event = dict(payload) if isinstance(payload, dict) else {}
        events.append(event)
        return event.get("type") == "wake_remainder_detected"

    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: None,
        local_transcriber=transcriber,
        wake_word_required=True,
        wake_words=("鸿途",),
        end_phrases=("结束对话",),
        on_gate_event=handle_event,
        local_vad_enabled=True,
        local_vad_rms_threshold=0.1,
        local_vad_peak_threshold=0.2,
        local_vad_hangover_frames=1,
    )
    speech = AudioFrame(pcm=_pcm_constant(10000), duration_ms=120, sample_rate_hz=16000, channels=1)
    quiet = AudioFrame(pcm=_pcm_constant(500), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(quiet) is None
    readiness = gate.readiness()

    assert [event["type"] for event in events] == ["wake_detected", "wake_remainder_detected"]
    assert events[-1]["text"] == "介绍一下你自己"
    assert readiness["localWakeGate"]["state"] == "active"
    assert readiness["localWakeGate"]["conversationActive"] is True
    assert readiness["localWakeGate"]["lastGateReason"] == "wake_remainder_sent_text"
    assert readiness["localWakeGate"]["lastStatus"] == "conversation_active"
    assert readiness["localWakeGate"]["wakeWords"] == ["鸿途"]
    assert readiness["localWakeGate"]["replayFrames"] == 0


def test_openclaw_echo_gate_accepts_greeting_before_short_wake_word() -> None:
    sink = FakePlaybackSink()
    events: list[dict[str, object]] = []
    transcriber = FakeSegmentTranscriber("你好鸿途介绍一下你自己")

    def handle_event(payload: object) -> bool:
        event = dict(payload) if isinstance(payload, dict) else {}
        events.append(event)
        return event.get("type") == "wake_remainder_detected"

    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: None,
        local_transcriber=transcriber,
        wake_word_required=True,
        wake_words=("鸿途",),
        end_phrases=("结束对话",),
        on_gate_event=handle_event,
        local_vad_enabled=True,
        local_vad_rms_threshold=0.1,
        local_vad_peak_threshold=0.2,
        local_vad_hangover_frames=1,
    )
    speech = AudioFrame(pcm=_pcm_constant(10000), duration_ms=120, sample_rate_hz=16000, channels=1)
    quiet = AudioFrame(pcm=_pcm_constant(500), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(quiet) is None

    assert [event["type"] for event in events] == ["wake_detected", "wake_remainder_detected"]
    assert events[0]["remainder"] == "介绍一下你自己"
    assert events[-1]["text"] == "介绍一下你自己"
    assert gate.readiness()["localWakeGate"]["conversationActive"] is True


def test_openclaw_echo_gate_local_wake_gate_buffers_active_utterance_until_asr_accepts() -> None:
    sink = FakePlaybackSink()
    events: list[dict[str, object]] = []
    transcriber = FakeSegmentTranscriber("你好鸿途", "介绍下你自己")
    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: None,
        local_transcriber=transcriber,
        wake_word_required=True,
        wake_words=("你好鸿途",),
        end_phrases=("结束对话",),
        on_gate_event=lambda payload: events.append(dict(payload)),
        local_vad_enabled=True,
        local_vad_rms_threshold=0.1,
        local_vad_peak_threshold=0.2,
        local_vad_hangover_frames=1,
    )
    speech = AudioFrame(pcm=_pcm_constant(10000), duration_ms=120, sample_rate_hz=16000, channels=1)
    quiet = AudioFrame(pcm=_pcm_constant(500), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(quiet) is None

    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    first_replay = gate.process_capture(quiet)
    second_replay = gate.process_capture(quiet)
    readiness = gate.readiness()

    assert first_replay is speech
    assert second_replay is not None
    assert readiness["localWakeGate"]["state"] == "active"
    assert readiness["localWakeGate"]["conversationActive"] is True
    assert readiness["localWakeGate"]["lastTranscript"] == "介绍下你自己"
    assert readiness["localWakeGate"]["lastGateReason"] == "active_utterance_replayed"
    assert readiness["localWakeGate"]["lastStatus"] == "conversation_active"
    assert readiness["localVad"]["passedFrames"] == 2


def test_openclaw_echo_gate_sends_active_utterance_text_when_event_handler_accepts() -> None:
    sink = FakePlaybackSink()
    events: list[dict[str, object]] = []
    transcriber = FakeSegmentTranscriber("你好鸿途", "介绍下你自己")

    def handle_event(payload: object) -> bool:
        event = dict(payload) if isinstance(payload, dict) else {}
        events.append(event)
        return event.get("type") == "active_utterance_detected"

    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: None,
        local_transcriber=transcriber,
        wake_word_required=True,
        wake_words=("你好鸿途",),
        end_phrases=("结束对话",),
        on_gate_event=handle_event,
        local_vad_enabled=True,
        local_vad_rms_threshold=0.1,
        local_vad_peak_threshold=0.2,
        local_vad_hangover_frames=1,
    )
    speech = AudioFrame(pcm=_pcm_constant(10000), duration_ms=120, sample_rate_hz=16000, channels=1)
    quiet = AudioFrame(pcm=_pcm_constant(500), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(quiet) is None

    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(quiet) is None
    readiness = gate.readiness()

    assert [event["type"] for event in events] == ["wake_detected", "active_utterance_detected"]
    assert events[-1]["text"] == "介绍下你自己"
    assert readiness["localWakeGate"]["state"] == "active"
    assert readiness["localWakeGate"]["conversationActive"] is True
    assert readiness["localWakeGate"]["lastGateReason"] == "active_utterance_sent_text"
    assert readiness["localWakeGate"]["lastStatus"] == "conversation_active"
    assert readiness["localWakeGate"]["replayFrames"] == 0
    assert readiness["localVad"]["passedFrames"] == 0


def test_openclaw_echo_gate_local_wake_gate_rejects_short_active_noise_before_upstream() -> None:
    sink = FakePlaybackSink()
    transcriber = FakeSegmentTranscriber("你好鸿途", "我。")
    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: None,
        local_transcriber=transcriber,
        wake_word_required=True,
        wake_words=("你好鸿途",),
        end_phrases=("结束对话",),
        local_vad_enabled=True,
        local_vad_rms_threshold=0.1,
        local_vad_peak_threshold=0.2,
        local_vad_hangover_frames=1,
    )
    speech = AudioFrame(pcm=_pcm_constant(10000), duration_ms=120, sample_rate_hz=16000, channels=1)
    quiet = AudioFrame(pcm=_pcm_constant(500), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(quiet) is None

    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(quiet) is None
    readiness = gate.readiness()

    assert readiness["localWakeGate"]["state"] == "active"
    assert readiness["localWakeGate"]["lastTranscript"] == "我。"
    assert readiness["localWakeGate"]["lastGateReason"] == "active_transcript_rejected"
    assert readiness["localVad"]["passedFrames"] == 0


def test_openclaw_echo_gate_local_wake_gate_end_phrase_returns_to_sleep() -> None:
    sink = FakePlaybackSink()
    events: list[dict[str, object]] = []
    transcriber = FakeSegmentTranscriber("你好鸿途", "结束对话")
    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: None,
        local_transcriber=transcriber,
        wake_word_required=True,
        wake_words=("你好鸿途",),
        end_phrases=("结束对话",),
        on_gate_event=lambda payload: events.append(dict(payload)),
        local_vad_enabled=True,
        local_vad_rms_threshold=0.1,
        local_vad_peak_threshold=0.2,
        local_vad_hangover_frames=1,
    )
    speech = AudioFrame(pcm=_pcm_constant(10000), duration_ms=120, sample_rate_hz=16000, channels=1)
    quiet = AudioFrame(pcm=_pcm_constant(500), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(speech) is None
    assert gate.process_capture(quiet) is None
    assert gate.process_capture(quiet) is None
    readiness = gate.readiness()

    assert [event["type"] for event in events] == ["wake_detected", "end_phrase_detected"]
    assert events[-1]["reply_text"] == "好的，结束对话。"
    assert readiness["localWakeGate"]["state"] == "armed"
    assert readiness["localWakeGate"]["conversationActive"] is False
    assert readiness["localWakeGate"]["lastTranscript"] == "结束对话"
    assert readiness["localWakeGate"]["lastGateReason"] == "end_phrase"


def test_openclaw_runtime_enables_local_vad_for_gateway_audio() -> None:
    runtime = OpenClawRealtimeRuntime(
        NativeVoiceLoopConfig(openclaw_ws_url="wss://openclaw.example/realtime"),
        transport_factory=lambda config: OpenClawRealtimeTransport(
            url=config.openclaw_ws_url,
            token="token",
            websocket_factory=lambda url, *, header, timeout: FakeWebSocket(),
        ),
        capture_source=EmptyCaptureSource(),
    )

    readiness = runtime.audio_frontend.readiness()

    assert readiness["vad"]["state"] == "local_vad_ready"
    assert readiness["localVad"]["enabled"] is True


def test_openclaw_runtime_wires_local_wake_gate_when_wake_word_required() -> None:
    runtime = OpenClawRealtimeRuntime(
        NativeVoiceLoopConfig(
            openclaw_ws_url="wss://openclaw.example/realtime",
            asr_model_dir="/models/asr/sherpa-onnx-streaming",
            wake_word_required=True,
            wake_words=("你好鸿途", "你好宏图"),
            end_phrases=("结束对话",),
        ),
        transport_factory=lambda config: OpenClawRealtimeTransport(
            url=config.openclaw_ws_url,
            token="token",
            websocket_factory=lambda url, *, header, timeout: FakeWebSocket(),
        ),
        capture_source=EmptyCaptureSource(),
    )

    readiness = runtime.audio_frontend.readiness()

    assert readiness["localWakeGate"]["enabled"] is True
    assert readiness["localWakeGate"]["state"] == "armed"
    assert readiness["localWakeGate"]["wakeWords"] == ["你好鸿途", "你好宏图"]
    assert readiness["localWakeGate"]["transcriber"]["provider"] == "sherpa_onnx"


def test_openclaw_runtime_local_wake_event_plays_configured_piper_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[dict[str, object]] = []

    def fake_run(command: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command_list = list(command) if isinstance(command, list) else [str(command)]
        if command_list and command_list[0] in {"/usr/local/bin/piper", "aplay"}:
            commands.append({"command": command_list, "input": kwargs.get("input")})
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("eihead.runtime.openclaw_runtime.subprocess.run", fake_run)
    runtime = OpenClawRealtimeRuntime(
        NativeVoiceLoopConfig(
            openclaw_ws_url="wss://openclaw.example/realtime",
            asr_model_dir="/models/asr/sherpa-onnx-streaming",
            wake_word_required=True,
            piper_command="/usr/local/bin/piper",
            piper_model_path="/models/piper/zh_CN-huayan-medium.onnx",
            piper_config_path="/models/piper/zh_CN-huayan-medium.onnx.json",
            speaker_device="plughw:CARD=SPA3700,DEV=0",
            playback_echo_cooldown_ms=1200,
        ),
        transport_factory=lambda config: OpenClawRealtimeTransport(
            url=config.openclaw_ws_url,
            token="token",
            websocket_factory=lambda url, *, header, timeout: FakeWebSocket(),
        ),
        capture_source=EmptyCaptureSource(),
    )

    runtime._handle_local_gate_event({"type": "wake_detected", "reply_text": "我在。"})

    assert commands[0]["command"][:5] == [
        "/usr/local/bin/piper",
        "--model",
        "/models/piper/zh_CN-huayan-medium.onnx",
        "--config",
        "/models/piper/zh_CN-huayan-medium.onnx.json",
    ]
    assert commands[0]["input"] == "我在。"
    assert commands[1]["command"][:4] == ["aplay", "-q", "-D", "plughw:CARD=SPA3700,DEV=0"]
    assert runtime._is_remote_output_active() is True


def test_openclaw_runtime_local_gate_event_sends_text_to_openclaw() -> None:
    transport = FakeConnectedTextTransport()
    runtime = OpenClawRealtimeRuntime(
        NativeVoiceLoopConfig(
            openclaw_ws_url="wss://openclaw.example/realtime",
            wake_word_required=True,
        ),
        transport_factory=lambda config: transport,  # type: ignore[return-value]
        capture_source=EmptyCaptureSource(),
        playback_sink=FakePlaybackSink(),
    )

    accepted = runtime._handle_local_gate_event(
        {"type": "active_utterance_detected", "text": "介绍下你自己", "asr_ms": 101.2}
    )

    assert accepted is True
    assert transport.sent_texts == ["介绍下你自己"]
    assert runtime._last_local_gate_event is not None
    assert runtime._last_local_gate_event["openclaw_text_sent"] is True
    assert runtime._last_local_gate_event["text_send_ms"] >= 0


def test_openclaw_runtime_status_uses_local_wake_gate_for_sleep_state_and_asr_latency() -> None:
    transport = FakeConnectedTextTransport()
    runtime = OpenClawRealtimeRuntime(
        NativeVoiceLoopConfig(
            openclaw_ws_url="wss://openclaw.example/realtime",
            wake_word_required=True,
        ),
        transport_factory=lambda config: transport,  # type: ignore[return-value]
        capture_source=EmptyCaptureSource(),
        playback_sink=FakePlaybackSink(),
    )
    runtime._started = True
    runtime.audio_frontend.local_transcriber = FakeSegmentTranscriber()
    runtime.audio_frontend._conversation_active = False
    runtime.audio_frontend._local_gate_last_transcript = "旁边电视的声音"
    runtime.audio_frontend._local_gate_last_reason = "wake_word_required"
    runtime.audio_frontend._local_gate_last_status = "waiting_for_wake_word"
    runtime.audio_frontend._local_gate_last_asr_ms = 1860.2

    status = runtime.status()

    assert status["voice_dialogue"]["conversation_active"] is False
    assert status["voice_dialogue"]["phase"] == "sleeping"
    assert status["voice_dialogue"]["last_transcript"] == "旁边电视的声音"
    assert status["voice_dialogue"]["last_gate_reason"] == "wake_word_required"
    assert status["voice_dialogue"]["last_stage_latency_ms"]["listen_asr"] == 1860.2
    assert status["wakeword"]["state"] == "armed"
    assert status["wakeword"]["last_gate_reason"] == "wake_word_required"


def test_openclaw_echo_gate_suppresses_when_remote_session_is_speaking() -> None:
    sink = FakePlaybackSink()
    sink.active = False
    remote_speaking = True
    gate = OpenClawPlaybackEchoGate(
        playback_sink=sink,
        on_barge_in=lambda payload: None,
        is_output_active=lambda: remote_speaking,
    )

    loud = AudioFrame(pcm=_pcm_constant(12000), duration_ms=120, sample_rate_hz=16000, channels=1)

    assert gate.process_capture(loud) is None
    readiness = gate.readiness()

    assert readiness["playbackGate"]["outputActive"] is True
    assert readiness["playbackGate"]["suppressedFrames"] == 1
    assert readiness["playbackGate"]["bargeInCount"] == 0


def test_openclaw_transport_receive_event_integrates_with_runtime_runner() -> None:
    socket = FakeWebSocket(
        incoming=[
            {"type": "event", "event": "connect.challenge", "payload": {"nonce": "nonce-1"}},
            {"type": "res", "id": "1", "ok": True, "payload": {"auth": {"role": "operator"}}},
            {
                "type": "res",
                "id": "2",
                "ok": True,
                "payload": {
                    "provider": "openai",
                    "transport": "gateway-relay",
                    "relaySessionId": "relay-1",
                    "audio": {
                        "inputEncoding": "pcm16",
                        "inputSampleRateHz": 24000,
                        "outputEncoding": "pcm16",
                        "outputSampleRateHz": 24000,
                    },
                },
            },
            {"type": "event", "event": "talk.event", "payload": {"relaySessionId": "relay-1", "type": "ready"}},
            {
                "type": "event",
                "event": "talk.event",
                "payload": {
                    "relaySessionId": "relay-1",
                    "type": "audio",
                    "audioBase64": "QkFTRTY0",
                },
            },
        ]
    )
    transport = OpenClawRealtimeTransport(
        url="wss://openclaw.example/realtime",
        token="token",
        websocket_factory=lambda url, *, header, timeout: socket,
    )
    playback = FakePlaybackSink()
    runner = EiVoiceRuntimeRunner(
        capture_source=EmptyCaptureSource(),
        audio_frontend=NoOpAcousticFrontend(),
        playback_sink=playback,
        transport=transport,
    )

    transport.connect()

    assert runner.step_receive() is True
    assert runner.step_decode() is True
    assert runner.step_playback() is True
    frame = playback.played[0]
    assert frame.pcm == b"BASE64"
    assert frame.sequence == 0
    assert frame.duration_ms == 60
    assert transport.status()["openclaw_ws"]["last_audio_rx_ms"] is not None


def test_openclaw_runtime_prioritizes_output_without_blocking_on_capture() -> None:
    socket = FakeWebSocket(
        incoming=[
            {"type": "event", "event": "connect.challenge", "payload": {"nonce": "nonce-1"}},
            {"type": "res", "id": "1", "ok": True, "payload": {"auth": {"role": "operator"}}},
            {
                "type": "res",
                "id": "2",
                "ok": True,
                "payload": {
                    "provider": "openai",
                    "transport": "gateway-relay",
                    "relaySessionId": "relay-1",
                    "audio": {
                        "inputEncoding": "pcm16",
                        "inputSampleRateHz": 24000,
                        "outputEncoding": "pcm16",
                        "outputSampleRateHz": 24000,
                    },
                },
            },
            {"type": "event", "event": "talk.event", "payload": {"relaySessionId": "relay-1", "type": "ready"}},
            {
                "type": "event",
                "event": "talk.event",
                "payload": {
                    "relaySessionId": "relay-1",
                    "type": "audio",
                    "audioBase64": "QkFTRTY0",
                },
            },
        ]
    )

    def _transport_factory(config: NativeVoiceLoopConfig) -> OpenClawRealtimeTransport:
        transport = OpenClawRealtimeTransport(
            url=config.openclaw_ws_url,
            token="token",
            websocket_factory=lambda url, *, header, timeout: socket,
        )
        return transport

    playback = FakePlaybackSink()
    runtime = OpenClawRealtimeRuntime(
        NativeVoiceLoopConfig(openclaw_ws_url="wss://openclaw.example/realtime"),
        transport_factory=_transport_factory,
        capture_source=RaisingCaptureSource(),
        playback_sink=playback,
    )

    assert runtime._ensure_connected() is True
    assert runtime.transport is not None
    runtime.transport._session_state = "speaking"
    runtime.transport._last_audio_rx_ms = int(time.monotonic() * 1000)

    result = runtime._step_runtime_once()

    assert result == {"playback": True, "decode": True, "receive": True}
    assert len(playback.played) == 1


def test_openclaw_runtime_retries_after_connect_failure() -> None:
    clock = ManualClock(100.0)
    sockets = [
        FakeWebSocket(
            incoming=[
                {"type": "event", "event": "connect.challenge", "payload": {"nonce": "nonce-2"}},
                {"type": "res", "id": "1", "ok": True, "payload": {"auth": {"role": "operator"}}},
                {
                    "type": "res",
                    "id": "2",
                    "ok": True,
                    "payload": {
                        "provider": "openai",
                        "transport": "gateway-relay",
                        "relaySessionId": "relay-2",
                        "audio": {
                            "inputEncoding": "pcm16",
                            "inputSampleRateHz": 24000,
                            "outputEncoding": "pcm16",
                            "outputSampleRateHz": 24000,
                        },
                    },
                },
                {"type": "event", "event": "talk.event", "payload": {"relaySessionId": "relay-2", "type": "ready"}},
            ]
        )
    ]
    connect_attempts = 0

    def _connect(url: str, *, header: list[str], timeout: float) -> FakeWebSocket:
        nonlocal connect_attempts
        connect_attempts += 1
        if connect_attempts == 1:
            raise TimeoutError("first connect failed")
        return sockets.pop(0)

    def _transport_factory(config: NativeVoiceLoopConfig) -> OpenClawRealtimeTransport:
        return OpenClawRealtimeTransport(
            url=config.openclaw_ws_url,
            token="token",
            websocket_factory=_connect,
            clock=clock,
        )

    runtime = OpenClawRealtimeRuntime(
        NativeVoiceLoopConfig(openclaw_ws_url="wss://openclaw.example/realtime"),
        transport_factory=_transport_factory,
        capture_source=EmptyCaptureSource(),
        playback_sink=FakePlaybackSink(),
    )

    assert runtime._ensure_connected() is False
    clock.advance(1.1)
    assert runtime._ensure_connected() is True
    assert connect_attempts == 2
    assert runtime.status()["openclaw_ws"]["connected"] is True


def test_openclaw_runtime_reconnects_ended_session_and_rearms_local_wake_gate() -> None:
    clock = ManualClock(200.0)

    def _socket(relay_id: str, *, connect_id: str, session_id: str) -> FakeWebSocket:
        return FakeWebSocket(
            incoming=[
                {"type": "event", "event": "connect.challenge", "payload": {"nonce": f"nonce-{relay_id}"}},
                {"type": "res", "id": connect_id, "ok": True, "payload": {"auth": {"role": "operator"}}},
                {
                    "type": "res",
                    "id": session_id,
                    "ok": True,
                    "payload": {
                        "provider": "openai",
                        "transport": "gateway-relay",
                        "relaySessionId": relay_id,
                        "audio": {
                            "inputEncoding": "pcm16",
                            "inputSampleRateHz": 24000,
                            "outputEncoding": "pcm16",
                            "outputSampleRateHz": 24000,
                        },
                    },
                },
                {"type": "event", "event": "talk.event", "payload": {"relaySessionId": relay_id, "type": "ready"}},
            ]
        )

    sockets = [
        _socket("relay-1", connect_id="1", session_id="2"),
        _socket("relay-2", connect_id="4", session_id="5"),
    ]

    def _transport_factory(config: NativeVoiceLoopConfig) -> OpenClawRealtimeTransport:
        return OpenClawRealtimeTransport(
            url=config.openclaw_ws_url,
            token="token",
            websocket_factory=lambda url, *, header, timeout: sockets.pop(0),
            clock=clock,
        )

    runtime = OpenClawRealtimeRuntime(
        NativeVoiceLoopConfig(
            openclaw_ws_url="wss://openclaw.example/realtime",
            wake_word_required=True,
        ),
        transport_factory=_transport_factory,
        capture_source=EmptyCaptureSource(),
        playback_sink=FakePlaybackSink(),
    )

    runtime.audio_frontend.local_transcriber = FakeSegmentTranscriber("你好鸿途")
    assert runtime._ensure_connected() is True
    assert runtime.transport is not None
    runtime.transport._session_state = "ended"
    runtime.audio_frontend._conversation_active = True
    runtime.audio_frontend._local_gate_last_status = "conversation_active"
    runtime.audio_frontend._local_gate_segment_frames.append(
        AudioFrame(pcm=_pcm_constant(10000), duration_ms=120, sample_rate_hz=16000, channels=1)
    )

    assert runtime._ensure_connected() is False
    readiness = runtime.audio_frontend.readiness()

    assert readiness["localWakeGate"]["state"] == "armed"
    assert readiness["localWakeGate"]["conversationActive"] is False
    assert readiness["localWakeGate"]["lastGateReason"] == "openclaw_session_ended"
    assert readiness["localWakeGate"]["segmentFrames"] == 0
    assert runtime.transport.status()["connection"]["state"] == "reconnect_wait"
    assert runtime.transport.status()["reconnect"]["reason"] == "relay_session_ended"

    clock.advance(1.1)

    assert runtime._ensure_connected() is True
    assert runtime.transport.status()["openclaw_ws"]["session_state"] == "ready"


def test_openclaw_runtime_session_config_includes_realtime_provider() -> None:
    payload = _session_config(
        NativeVoiceLoopConfig(
            dialogue_session_id="honjia-voice",
            openclaw_provider="openai",
            openclaw_model="gpt-realtime-2",
            openclaw_voice="cedar",
        )
    )

    assert payload["sessionKey"] == "honjia-voice"
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-realtime-2"
    assert payload["voice"] == "cedar"
