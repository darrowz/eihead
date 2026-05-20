from __future__ import annotations

from array import array
import json
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
                "payload": {"auth": {"role": "operator", "scopes": ["operator.read", "operator.write"]}},
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
                        "audioBase64": "BAUG",
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
                "audioBase64": "BAUG",
                "index": 0,
                "durationMs": 60,
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


def test_openclaw_playback_sink_does_not_stack_grace_per_audio_frame() -> None:
    clock = ManualClock(10.0)
    sink = AplayPcmPlaybackSink(device="default", active_grace_s=0.1, clock=clock)
    sink._ensure_process = lambda *, sample_rate, channels: None  # type: ignore[method-assign]

    sink.play(AudioFrame(pcm=_pcm_constant(1000), duration_ms=200, sample_rate_hz=16000, channels=1))
    clock.advance(0.05)
    sink.play(AudioFrame(pcm=_pcm_constant(1000), duration_ms=200, sample_rate_hz=16000, channels=1))

    assert sink.status()["active_until_s"] == pytest.approx(10.35)


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
