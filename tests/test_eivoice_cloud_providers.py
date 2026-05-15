from __future__ import annotations

import base64
import json

from eihead.eivoice_runtime import AudioFrame, StreamingAsrSession, StreamingTtsRequest, StreamingTtsSession


class FakeAsrTransport:
    def __init__(self, *responses: dict[str, object]) -> None:
        self.responses = list(responses)
        self.sent: list[dict[str, object]] = []
        self.headers: list[dict[str, str]] = []
        self.cancelled = False
        self.closed = False

    def send_json(self, payload: dict[str, object], *, headers: dict[str, str], timeout_s: float) -> None:
        assert timeout_s == 7.5
        self.sent.append(payload)
        self.headers.append(headers)

    def receive_json(self) -> dict[str, object] | None:
        if not self.responses:
            return None
        return self.responses.pop(0)

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True


class FakeTtsTransport:
    def __init__(self, *messages: object) -> None:
        self.messages = list(messages)
        self.requests: list[dict[str, object]] = []
        self.headers: list[dict[str, str]] = []
        self.cancelled = False
        self.closed = False

    def stream_json(
        self,
        payload: dict[str, object],
        *,
        headers: dict[str, str],
        timeout_s: float,
    ):
        assert timeout_s == 7.5
        self.requests.append(payload)
        self.headers.append(headers)
        yield from self.messages

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True


def test_cloud_provider_config_loads_from_env_and_redacts_secret() -> None:
    from eihead.eivoice_runtime import CloudProviderConfig

    config = CloudProviderConfig.from_env(
        "minimax",
        env={
            "EIVOICE_MINIMAX_API_KEY": "super-secret",
            "EIVOICE_MINIMAX_BASE_URL": "https://tts.example",
            "EIVOICE_MINIMAX_MODEL": "speech-test",
            "EIVOICE_MINIMAX_VOICE_ID": "voice-test",
            "EIVOICE_MINIMAX_TIMEOUT": "7.5",
        },
    )

    diagnostics = config.diagnostics()

    assert config.api_key == "super-secret"
    assert config.base_url == "https://tts.example"
    assert config.model == "speech-test"
    assert config.voice_id == "voice-test"
    assert config.timeout_s == 7.5
    assert diagnostics["api_key"] == "s***t"
    assert "super-secret" not in repr(config)
    assert "super-secret" not in str(diagnostics)


def test_cloud_provider_config_falls_back_to_plain_provider_env_names() -> None:
    from eihead.eivoice_runtime import CloudProviderConfig

    config = CloudProviderConfig.from_env(
        "minimax",
        env={
            "MINIMAX_API_KEY": "legacy-secret",
            "MINIMAX_BASE_URL": "wss://legacy.example",
            "MINIMAX_MODEL": "speech-legacy",
            "MINIMAX_VOICE_ID": "voice-legacy",
            "MINIMAX_TIMEOUT": "6.25",
        },
    )

    assert config.api_key == "legacy-secret"
    assert config.base_url == "wss://legacy.example"
    assert config.model == "speech-legacy"
    assert config.voice_id == "voice-legacy"
    assert config.timeout_s == 6.25


def test_cloud_provider_config_prefers_eivoice_env_over_plain_provider_env_names() -> None:
    from eihead.eivoice_runtime import CloudProviderConfig

    config = CloudProviderConfig.from_env(
        "dashscope",
        env={
            "DASHSCOPE_API_KEY": "legacy-secret",
            "DASHSCOPE_MODEL": "legacy-model",
            "EIVOICE_DASHSCOPE_API_KEY": "new-secret",
            "EIVOICE_DASHSCOPE_MODEL": "new-model",
        },
    )

    assert config.api_key == "new-secret"
    assert config.model == "new-model"


def test_dashscope_asr_provider_encodes_audio_frame_and_maps_partial_final() -> None:
    from eihead.eivoice_runtime import CloudProviderConfig, DashScopeStreamingAsrProvider

    transport = FakeAsrTransport(
        {"event": "result", "text": "ni", "is_final": False, "request_id": "req-asr"},
        {"event": "result", "text": "ni hao", "is_final": True, "request_id": "req-asr", "confidence": 0.93},
    )
    provider = DashScopeStreamingAsrProvider(
        CloudProviderConfig(
            provider="dashscope",
            api_key="secret-asr",
            base_url="wss://asr.example",
            model="paraformer-test",
            timeout_s=7.5,
        ),
        transport=transport,
    )

    results = list(
        provider.accept_frame(
            AudioFrame(
                pcm=b"pcm-frame",
                payload=b"opus-frame",
                duration_ms=40,
                sample_rate_hz=16000,
                channels=1,
                sequence=12,
            )
        )
    )

    assert transport.sent == [
        {
            "type": "audio_frame",
            "provider": "dashscope",
            "model": "paraformer-test",
            "sequence": 12,
            "duration_ms": 40,
            "sample_rate_hz": 16000,
            "channels": 1,
            "audio_format": "opus",
            "audio_base64": base64.b64encode(b"opus-frame").decode("ascii"),
        }
    ]
    assert transport.headers[0]["Authorization"] == "Bearer secret-asr"
    assert [(result.text, result.final) for result in results] == [("ni", False), ("ni hao", True)]
    assert results[1].confidence == 0.93
    assert results[1].metadata["request_id"] == "req-asr"
    assert provider.diagnostics()["state"] == "finalized"
    assert "secret-asr" not in str(provider.diagnostics())


def test_dashscope_asr_provider_maps_official_nested_result_shape() -> None:
    from eihead.eivoice_runtime import CloudProviderConfig, DashScopeStreamingAsrProvider

    transport = FakeAsrTransport(
        {
            "header": {"event": "result-generated", "task_id": "task-1"},
            "payload": {
                "output": {
                    "sentence": {
                        "text": "你好鸿途",
                        "sentence_end": True,
                    }
                },
                "usage": {"duration": 1},
            },
        },
    )
    provider = DashScopeStreamingAsrProvider(
        CloudProviderConfig(provider="dashscope", api_key="secret-asr", model="fun-asr-realtime", timeout_s=7.5),
        transport=transport,
    )

    results = list(provider.accept_frame(AudioFrame(pcm=b"pcm", sequence=1)))

    assert [(result.text, result.final) for result in results] == [("你好鸿途", True)]
    assert results[0].metadata["request_id"] == "task-1"


def test_asr_provider_reports_errors_and_supports_cancel_close() -> None:
    from eihead.eivoice_runtime import CloudProviderConfig, DashScopeStreamingAsrProvider

    transport = FakeAsrTransport(
        {"event": "error", "code": "BadRequest", "message": "bad request"},
    )
    provider = DashScopeStreamingAsrProvider(
        CloudProviderConfig(
            provider="dashscope",
            api_key="secret-asr",
            base_url="wss://asr.example",
            model="paraformer-test",
            timeout_s=7.5,
        ),
        transport=transport,
    )

    assert list(provider.accept_frame(AudioFrame(pcm=b"pcm", sequence=1))) == []
    cancel_status = provider.cancel("barge_in")
    provider.close()
    diagnostics = provider.diagnostics()

    assert cancel_status["cancelled"] is True
    assert transport.cancelled is True
    assert transport.closed is True
    assert diagnostics["state"] == "closed"
    assert diagnostics["errors"][-1]["kind"] == "provider_error"
    assert diagnostics["errors"][-1]["code"] == "BadRequest"
    assert "secret-asr" not in str(diagnostics)


def test_asr_provider_redacts_secret_from_transport_exception() -> None:
    from eihead.eivoice_runtime import CloudProviderConfig, DashScopeStreamingAsrProvider

    class _FailingTransport(FakeAsrTransport):
        def send_json(self, payload: dict[str, object], *, headers: dict[str, str], timeout_s: float) -> None:
            raise RuntimeError(f"bad auth {headers['Authorization']}")

    provider = DashScopeStreamingAsrProvider(
        CloudProviderConfig(provider="dashscope", api_key="secret-asr", model="fun-asr-realtime", timeout_s=7.5),
        transport=_FailingTransport(),
    )

    assert list(provider.accept_frame(AudioFrame(pcm=b"pcm", sequence=1))) == []
    diagnostics = provider.diagnostics()

    assert "secret-asr" not in str(diagnostics)
    assert "Bearer s***r" in str(diagnostics)


def test_streaming_asr_session_bridges_provider_cancel_close_status() -> None:
    from eihead.eivoice_runtime import CloudProviderConfig, DashScopeStreamingAsrProvider

    transport = FakeAsrTransport()
    provider = DashScopeStreamingAsrProvider(
        CloudProviderConfig(provider="dashscope", api_key="secret-asr", timeout_s=7.5),
        transport=transport,
    )
    session = StreamingAsrSession(session_id="session-1", round_id="round-1", provider=provider)

    cancel_status = session.cancel("barge_in")
    session.close()
    status = session.status()

    assert cancel_status["cancelled"] is True
    assert transport.cancelled is True
    assert transport.closed is True
    assert status["provider_state"] == "closed"


def test_minimax_tts_provider_maps_audio_chunks_and_tracks_request_status() -> None:
    from eihead.eivoice_runtime import CloudProviderConfig, MiniMaxStreamingTtsProvider

    first_audio = base64.b64encode(b"audio-one").decode("ascii")
    transport = FakeTtsTransport(
        {"event": "start", "request_id": "req-tts"},
        {"event": "audio", "audio_base64": first_audio, "index": 4, "duration_ms": 30},
        b"audio-two",
        {"event": "complete", "request_id": "req-tts"},
    )
    provider = MiniMaxStreamingTtsProvider(
        CloudProviderConfig(
            provider="minimax",
            api_key="secret-tts",
            base_url="wss://tts.example",
            model="speech-test",
            voice_id="voice-test",
            timeout_s=7.5,
        ),
        transport=transport,
    )

    chunks = list(
        provider.stream(
            StreamingTtsRequest(
                text="hello",
                round_id="round-tts",
                voice_code="override-voice",
                emotion="warm",
                speed=1.1,
                volume=0.7,
                sample_rate_hz=24000,
                channels=1,
                audio_format="pcm16",
            )
        )
    )

    assert transport.requests[0]["type"] == "tts_request"
    assert transport.requests[0]["provider"] == "minimax"
    assert transport.requests[0]["model"] == "speech-test"
    assert transport.requests[0]["voice_id"] == "override-voice"
    assert transport.requests[0]["text"] == "hello"
    assert transport.headers[0]["Authorization"] == "Bearer secret-tts"
    assert [chunk.payload for chunk in chunks] == [b"audio-one", b"audio-two"]
    assert [chunk.index for chunk in chunks] == [4, 5]
    assert chunks[0].duration_ms == 30

    status = provider.status()

    assert status["state"] == "complete"
    assert status["request_id"] == "req-tts"
    assert status["first_chunk"] is True
    assert status["chunks_produced"] == 2
    assert status["last_error"] is None
    assert "secret-tts" not in str(status)


def test_minimax_tts_provider_maps_official_websocket_audio_shape() -> None:
    from eihead.eivoice_runtime import CloudProviderConfig, MiniMaxStreamingTtsProvider

    transport = FakeTtsTransport(
        {"event": "task_started", "trace_id": "trace-minimax"},
        {
            "data": {"audio": "52494646"},
            "is_final": True,
            "trace_id": "trace-minimax",
            "extra_info": {"audio_sample_rate": 32000, "audio_channel": 1, "audio_format": "mp3"},
        },
    )
    provider = MiniMaxStreamingTtsProvider(
        CloudProviderConfig(provider="minimax", api_key="secret-tts", model="speech-2.8-turbo", timeout_s=7.5),
        transport=transport,
    )

    chunks = list(provider.stream(StreamingTtsRequest(text="你好", round_id="round-1")))

    assert [chunk.payload for chunk in chunks] == [b"RIFF"]
    assert provider.status()["request_id"] == "trace-minimax"


def test_tts_provider_reports_error_and_supports_cancel_close() -> None:
    from eihead.eivoice_runtime import CloudProviderConfig, MiniMaxStreamingTtsProvider

    transport = FakeTtsTransport(
        {"event": "error", "code": "Unauthorized", "message": "bad token"},
    )
    provider = MiniMaxStreamingTtsProvider(
        CloudProviderConfig(
            provider="minimax",
            api_key="secret-tts",
            base_url="wss://tts.example",
            model="speech-test",
            voice_id="voice-test",
            timeout_s=7.5,
        ),
        transport=transport,
    )

    assert list(provider.stream(StreamingTtsRequest(text="hello", round_id="round-tts"))) == []
    cancel_status = provider.cancel("interrupt")
    provider.close()
    status = provider.status()

    assert cancel_status["cancelled"] is True
    assert transport.cancelled is True
    assert transport.closed is True
    assert status["state"] == "closed"
    assert status["last_error"]["kind"] == "provider_error"
    assert status["last_error"]["code"] == "Unauthorized"
    assert "secret-tts" not in str(status)


def test_tts_provider_redacts_secret_from_transport_exception() -> None:
    from eihead.eivoice_runtime import CloudProviderConfig, MiniMaxStreamingTtsProvider

    class _FailingTransport(FakeTtsTransport):
        def stream_json(self, payload: dict[str, object], *, headers: dict[str, str], timeout_s: float):
            raise RuntimeError(f"bad auth {headers['Authorization']}")

    provider = MiniMaxStreamingTtsProvider(
        CloudProviderConfig(provider="minimax", api_key="secret-tts", model="speech-test", timeout_s=7.5),
        transport=_FailingTransport(),
    )

    assert list(provider.stream(StreamingTtsRequest(text="hello", round_id="round-tts"))) == []
    status = provider.status()

    assert "secret-tts" not in str(status)
    assert "Bearer s***s" in str(status)


def test_tts_session_interrupt_survives_provider_cancel_error() -> None:
    class _CancelFails:
        def stream(self, request: StreamingTtsRequest):
            yield from ()

        def cancel(self, reason: str = "cancelled") -> None:
            raise RuntimeError("cancel boom")

        def status(self) -> dict[str, object]:
            return {"provider_state": "streaming"}

    session = StreamingTtsSession(provider=_CancelFails())
    list(session.synthesize(text="hello", round_id="round-tts"))

    result = session.interrupt(round_id="round-tts", reason="barge_in")

    assert result["cancelled"] is True
    assert result["reason"] == "barge_in"
    assert "cancel boom" in str(session.status(round_id="round-tts")["last_error"])


class _FakeWebSocket:
    def __init__(self, *messages: object) -> None:
        self.messages = list(messages)
        self.sent: list[object] = []
        self.closed = False
        self.timeout: float | None = None

    def send(self, payload: object, opcode: object | None = None) -> None:
        self.sent.append(payload)

    def recv(self) -> object:
        if not self.messages:
            raise TimeoutError("no message")
        value = self.messages.pop(0)
        return json.dumps(value) if isinstance(value, dict) else value

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def close(self) -> None:
        self.closed = True


def test_dashscope_websocket_transport_sends_run_task_audio_and_finish() -> None:
    from eihead.eivoice_runtime.cloud_providers import DashScopeWebSocketAsrTransport

    sockets: list[_FakeWebSocket] = []

    def _connect(url: str, *, header: list[str], timeout: float):
        socket = _FakeWebSocket(
            {"header": {"event": "task-started", "task_id": "task-ws"}},
            {
                "header": {"event": "result-generated", "task_id": "task-ws"},
                "payload": {"output": {"sentence": {"text": "hello", "sentence_end": False}}},
            },
        )
        sockets.append(socket)
        return socket

    transport = DashScopeWebSocketAsrTransport(websocket_factory=_connect, task_id_factory=lambda: "task-ws")
    payload = {
        "model": "fun-asr-realtime",
        "sample_rate_hz": 16000,
        "audio_format": "pcm16",
        "audio_base64": base64.b64encode(b"pcm").decode("ascii"),
    }

    transport.send_json(
        payload,
        headers={"Authorization": "Bearer secret-asr"},
        timeout_s=7.5,
        url="wss://asr.example",
    )
    message = transport.receive_json()
    transport.close()

    assert json.loads(sockets[0].sent[0])["header"]["action"] == "run-task"
    assert sockets[0].sent[1] == b"pcm"
    assert json.loads(sockets[0].sent[2])["header"]["action"] == "finish-task"
    assert message["header"]["event"] == "result-generated"
    assert sockets[0].closed is True


def test_minimax_websocket_transport_sends_task_start_continue_finish() -> None:
    from eihead.eivoice_runtime.cloud_providers import MiniMaxWebSocketTtsTransport

    sockets: list[_FakeWebSocket] = []

    def _connect(url: str, *, header: list[str], timeout: float):
        socket = _FakeWebSocket(
            {"event": "connected_success", "trace_id": "trace-1"},
            {"event": "task_started", "trace_id": "trace-1"},
            {"data": {"audio": "52494646"}, "is_final": True, "trace_id": "trace-1"},
        )
        sockets.append(socket)
        return socket

    transport = MiniMaxWebSocketTtsTransport(websocket_factory=_connect)
    messages = list(
        transport.stream_json(
            {
                "model": "speech-2.8-turbo",
                "text": "hello",
                "voice_id": "female-shaonv",
                "speed": 1.0,
                "volume": 1.0,
                "sample_rate_hz": 32000,
                "channels": 1,
                "audio_format": "mp3",
            },
            headers={"Authorization": "Bearer secret-tts"},
            timeout_s=7.5,
            url="wss://tts.example",
        )
    )
    transport.close()

    assert json.loads(sockets[0].sent[0])["event"] == "task_start"
    assert json.loads(sockets[0].sent[1]) == {"event": "task_continue", "text": "hello"}
    assert messages[-1]["data"]["audio"] == "52494646"
    assert json.loads(sockets[0].sent[-1])["event"] == "task_finish"
