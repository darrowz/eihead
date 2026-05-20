from __future__ import annotations

import json

import pytest

from eihead.eivoice_runtime import EiVoiceRuntimeRunner, NoOpAcousticFrontend
from eihead.eivoice_runtime.openclaw_transport import OpenClawRealtimeTransport


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

    def send(self, payload: object) -> None:
        self.sent.append(payload)

    def recv(self) -> object:
        if not self.incoming:
            raise TimeoutError("no message available")
        return self.incoming.pop(0)

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def close(self) -> None:
        self.closed = True


class EmptyCaptureSource:
    def read_frame(self) -> None:
        return None


class FakePlaybackSink:
    def __init__(self) -> None:
        self.played: list[object] = []

    def play(self, frame: object) -> None:
        self.played.append(frame)

    def stop(self) -> None:
        return None


def test_openclaw_transport_connects_and_sends_audio_chunk_with_default_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCLAW_REALTIME_TOKEN", "env-token")
    socket = FakeWebSocket(incoming=[{"type": "session.ready", "sessionId": "session-1"}])
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
        session_config={"sessionKey": "honjia-voice", "voice": "Zephyr"},
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
            "Authorization: Bearer env-token",
            "Sec-WebSocket-Protocol: openclaw.realtime.v1",
            "X-Trace-Id: trace-1",
        ],
        "timeout": 10.0,
    }
    assert socket.sent == [
        json.dumps(
            {
                "sessionKey": "honjia-voice",
                "voice": "Zephyr",
                "type": "session.config",
                "apiKey": "env-token",
            }
        ),
        json.dumps(
            {
                "type": "audio.append",
                "data": "AQID",
                "sequence": 7,
                "duration_ms": 40,
                "sample_rate_hz": 24000,
                "channels": 1,
                "uid": "user-1",
                "mid": "mid-1",
            }
        )
    ]
    assert transport.status()["connection"]["state"] == "connected"


def test_openclaw_transport_receives_audio_delta_and_updates_pong_state() -> None:
    clock = ManualClock(20.0)
    socket = FakeWebSocket(
        incoming=[
            {"type": "session.ready", "sessionId": "session-1"},
            json.dumps(
                {
                    "type": "audio.delta",
                    "data": "BAUG",
                    "sequence": 3,
                    "duration_ms": 80,
                    "sample_rate_hz": 24000,
                    "channels": 1,
                }
            ),
            {"type": "pong"},
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
    clock.advance(10.0)
    transport.send_ping(uid="user-1", mid="mid-1")
    clock.advance(0.75)

    audio_event = transport.receive_event()
    pong_event = transport.receive_event()

    assert socket.timeout == 2.5
    assert audio_event == {
        "contentType": "AUDIO_CHUNK",
        "content": {
            "eventType": "AUDIO_CHUNK",
            "audioBase64": "BAUG",
            "index": 3,
            "durationMs": 80,
            "sampleRateHz": 24000,
            "channels": 1,
            "metadata": {
                "source": "openclaw_realtime",
                "messageType": "audio.delta",
            },
        },
    }
    assert pong_event == {"contentType": "PONG", "content": {"eventType": "PONG"}}
    assert transport.status()["heartbeat"]["awaiting_pong"] is False
    assert transport.status()["heartbeat"]["latency_ms"] == 750


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


def test_openclaw_transport_receive_event_integrates_with_runtime_runner() -> None:
    socket = FakeWebSocket(
        incoming=[
            {"type": "session.ready", "sessionId": "session-1"},
            {
                "type": "audio.delta",
                "data": "QkFTRTY0",
                "sequence": 4,
                "duration_ms": 20,
                "sample_rate_hz": 16000,
                "channels": 1,
            }
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
    assert frame.sequence == 4
    assert frame.duration_ms == 20
