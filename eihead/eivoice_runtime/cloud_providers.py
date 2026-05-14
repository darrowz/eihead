from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from .asr import AsrProviderResult
from .core import AudioFrame
from .tts import StreamingTtsAudioChunk, StreamingTtsRequest


class AsrJsonTransport(Protocol):
    def send_json(
        self,
        payload: dict[str, object],
        *,
        headers: dict[str, str],
        timeout_s: float,
        url: str | None = None,
    ) -> None:
        ...

    def receive_json(self) -> dict[str, object] | None:
        ...


class TtsJsonStreamTransport(Protocol):
    def stream_json(
        self,
        payload: dict[str, object],
        *,
        headers: dict[str, str],
        timeout_s: float,
        url: str | None = None,
    ) -> Iterable[dict[str, object] | bytes]:
        ...


@dataclass(frozen=True, repr=False)
class CloudProviderConfig:
    provider: str
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    voice_id: str = ""
    timeout_s: float = 30.0

    @classmethod
    def from_env(
        cls,
        provider: str,
        *,
        env: Mapping[str, str] | None = None,
        prefix: str = "EIVOICE",
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        voice_id: str | None = None,
        timeout_s: float | None = None,
    ) -> "CloudProviderConfig":
        source = os.environ if env is None else env
        key_prefix = f"{prefix}_{provider}".upper().replace("-", "_")
        fallback_prefix = str(provider).upper().replace("-", "_")
        timeout_value = timeout_s
        if timeout_value is None:
            timeout_value = _safe_float(_first_env(source, f"{key_prefix}_TIMEOUT", f"{fallback_prefix}_TIMEOUT"), default=30.0)
        return cls(
            provider=str(provider),
            api_key=str(api_key if api_key is not None else _first_env(source, f"{key_prefix}_API_KEY", f"{fallback_prefix}_API_KEY")),
            base_url=str(
                base_url if base_url is not None else _first_env(source, f"{key_prefix}_BASE_URL", f"{fallback_prefix}_BASE_URL")
            ),
            model=str(model if model is not None else _first_env(source, f"{key_prefix}_MODEL", f"{fallback_prefix}_MODEL")),
            voice_id=str(
                voice_id if voice_id is not None else _first_env(source, f"{key_prefix}_VOICE_ID", f"{fallback_prefix}_VOICE_ID")
            ),
            timeout_s=float(timeout_value),
        )

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def diagnostics(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "voice_id": self.voice_id,
            "timeout_s": self.timeout_s,
            "api_key": _redact_secret(self.api_key),
        }

    def __repr__(self) -> str:
        return (
            "CloudProviderConfig("
            f"provider={self.provider!r}, base_url={self.base_url!r}, model={self.model!r}, "
            f"voice_id={self.voice_id!r}, timeout_s={self.timeout_s!r}, "
            f"api_key={_redact_secret(self.api_key)!r})"
        )


class DashScopeStreamingAsrProvider:
    provider_name = "dashscope"

    def __init__(self, config: CloudProviderConfig, *, transport: AsrJsonTransport) -> None:
        self.config = config
        self.transport = transport
        self._state = "idle"
        self._frames_sent = 0
        self._results_received = 0
        self._request_id: str | None = None
        self._errors: list[dict[str, object]] = []

    @property
    def state(self) -> str:
        return self._state

    def can_accept_frame(self) -> bool:
        return self._state not in {"cancelled", "closed"}

    def accept_frame(self, frame: AudioFrame) -> Iterable[AsrProviderResult]:
        if not self.can_accept_frame():
            return []
        payload = _asr_audio_frame_payload(self.config, frame)
        try:
            _send_json(
                self.transport,
                payload,
                headers=self.config.headers(),
                timeout_s=self.config.timeout_s,
                url=self.config.base_url or None,
            )
        except BaseException as exc:
            self._record_exception(exc, context="send_json")
            return []
        self._frames_sent += 1
        self._state = "streaming"
        return self._drain_results()

    def cancel(self, reason: str = "cancelled") -> dict[str, object]:
        self._state = "cancelled"
        _call_optional(self.transport, "cancel")
        return {"cancelled": True, "reason": str(reason or "cancelled"), "provider": self.provider_name}

    def close(self) -> None:
        self._state = "closed"
        _call_optional(self.transport, "close")

    def status(self) -> dict[str, object]:
        return self.diagnostics()

    def diagnostics(self) -> dict[str, object]:
        return {
            "provider": self.provider_name,
            "state": self._state,
            "request_id": self._request_id,
            "frames_sent": self._frames_sent,
            "results_received": self._results_received,
            "errors": [dict(error) for error in self._errors],
            "config": self.config.diagnostics(),
        }

    def _drain_results(self) -> list[AsrProviderResult]:
        results: list[AsrProviderResult] = []
        while True:
            try:
                message = self.transport.receive_json()
            except BaseException as exc:
                self._record_exception(exc, context="receive_json")
                break
            if message is None:
                break
            result = self._result_from_message(message)
            if result is not None:
                results.append(result)
        return results

    def _result_from_message(self, message: Mapping[str, object]) -> AsrProviderResult | None:
        header = _mapping_or_empty(message.get("header"))
        event = _lower_str(_first_present(message, "event", "type", "name", default=header.get("event")))
        request_id = _string_or_none(
            _first_present(
                message,
                "request_id",
                "requestId",
                "task_id",
                "taskId",
                "trace_id",
                "traceId",
                default=header.get("task_id") or header.get("taskId"),
            )
        )
        if request_id:
            self._request_id = request_id
        if event in {"error", "failed", "task-failed"} or "error" in message:
            self._record_provider_error(message)
            return None
        text = _asr_text_from_message(message)
        if text is None:
            return None
        final = _bool_from_provider_message(message)
        self._results_received += 1
        self._state = "finalized" if final else "streaming"
        metadata: dict[str, object] = {"provider": self.provider_name}
        if self._request_id:
            metadata["request_id"] = self._request_id
        return AsrProviderResult(
            text=text,
            final=final,
            provider_state=self._state,
            confidence=_safe_float(_first_present(message, "confidence", "score"), default=None),
            metadata=metadata,
        )

    def _record_provider_error(self, message: Mapping[str, object]) -> None:
        self._state = "error"
        header = _mapping_or_empty(message.get("header"))
        error_payload = _first_present(message, "error", "base_resp", default=header if header else None)
        payload = error_payload if isinstance(error_payload, Mapping) else message
        self._errors.append(
            {
                "kind": "provider_error",
                "code": _string_or_none(_first_present(payload, "code", "status_code", "statusCode", "error_code"))
                or "unknown",
                "message": _redact_text(
                    _string_or_none(_first_present(payload, "message", "status_msg", "statusMsg", "error_message")) or "",
                    self.config.api_key,
                ),
            }
        )

    def _record_exception(self, exc: BaseException, *, context: str) -> None:
        self._state = "error"
        self._errors.append({"kind": type(exc).__name__, "message": _redact_text(str(exc), self.config.api_key), "context": context})


class MiniMaxStreamingTtsProvider:
    provider_name = "minimax"

    def __init__(self, config: CloudProviderConfig, *, transport: TtsJsonStreamTransport) -> None:
        self.config = config
        self.transport = transport
        self._state = "idle"
        self._request_id: str | None = None
        self._chunks_produced = 0
        self._first_chunk = False
        self._last_error: dict[str, object] | None = None

    def stream(self, request: StreamingTtsRequest) -> Iterator[StreamingTtsAudioChunk]:
        payload = _tts_request_payload(self.config, request)
        self._state = "requesting"
        self._request_id = None
        self._chunks_produced = 0
        self._first_chunk = False
        self._last_error = None
        try:
            messages = _stream_json(
                self.transport,
                payload,
                headers=self.config.headers(),
                timeout_s=self.config.timeout_s,
                url=self.config.base_url or None,
            )
            yield from self._chunks_from_messages(messages, request)
        except BaseException as exc:
            self._record_exception(exc, context="stream_json")

    def cancel(self, reason: str = "cancelled") -> dict[str, object]:
        self._state = "cancelled"
        _call_optional(self.transport, "cancel")
        return {
            "cancelled": True,
            "reason": str(reason or "cancelled"),
            "provider": self.provider_name,
            "request_id": self._request_id,
        }

    def close(self) -> None:
        self._state = "closed"
        _call_optional(self.transport, "close")

    def status(self) -> dict[str, object]:
        return {
            "provider": self.provider_name,
            "state": self._state,
            "request_id": self._request_id,
            "first_chunk": self._first_chunk,
            "chunks_produced": self._chunks_produced,
            "last_error": dict(self._last_error) if self._last_error is not None else None,
            "config": self.config.diagnostics(),
        }

    def _chunks_from_messages(
        self,
        messages: Iterable[dict[str, object] | bytes],
        request: StreamingTtsRequest,
    ) -> Iterator[StreamingTtsAudioChunk]:
        next_index = 0
        self._state = "streaming"
        for message in messages:
            if isinstance(message, (bytes, bytearray)):
                payload = bytes(message)
                chunk_index = next_index
                duration_ms = 60
            elif isinstance(message, Mapping):
                event = _lower_str(_first_present(message, "event", "type", "name"))
                request_id = _string_or_none(
                    _first_present(message, "request_id", "requestId", "trace_id", "traceId", "session_id", "sessionId")
                )
                if request_id:
                    self._request_id = request_id
                if event in {"start", "started"}:
                    continue
                if event in {"complete", "completed", "end", "finished"}:
                    self._state = "complete"
                    continue
                if event in {"error", "failed"} or "error" in message:
                    self._record_provider_error(message)
                    continue
                payload = _audio_payload_from_message(message)
                if payload is None:
                    continue
                chunk_index = _safe_int(_first_present(message, "index", "chunk_index", "chunkIndex"), default=next_index)
                duration_ms = _safe_int(_first_present(message, "duration_ms", "durationMs"), default=60)
            else:
                continue
            self._first_chunk = True
            self._chunks_produced += 1
            next_index = int(chunk_index) + 1
            yield StreamingTtsAudioChunk(
                payload=payload,
                index=int(chunk_index),
                duration_ms=int(duration_ms),
                sample_rate_hz=request.sample_rate_hz,
                channels=request.channels,
                audio_format=request.audio_format,
            )
        if self._state == "streaming":
            self._state = "complete"

    def _record_provider_error(self, message: Mapping[str, object]) -> None:
        self._state = "error"
        error_payload = _first_present(message, "error", "base_resp")
        payload = error_payload if isinstance(error_payload, Mapping) else message
        self._last_error = {
            "kind": "provider_error",
            "code": _string_or_none(_first_present(payload, "code", "status_code", "statusCode")) or "unknown",
            "message": _redact_text(
                _string_or_none(_first_present(payload, "message", "status_msg", "statusMsg")) or "",
                self.config.api_key,
            ),
        }

    def _record_exception(self, exc: BaseException, *, context: str) -> None:
        self._state = "error"
        self._last_error = {"kind": type(exc).__name__, "message": _redact_text(str(exc), self.config.api_key), "context": context}


class DashScopeWebSocketAsrTransport:
    """Minimal DashScope realtime ASR WebSocket transport.

    It follows DashScope's run-task -> binary audio -> result-generated ->
    finish-task flow while keeping network code behind a fakeable boundary.
    """

    default_url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

    def __init__(
        self,
        *,
        websocket_factory: WebSocketFactory | None = None,
        task_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.websocket_factory = websocket_factory or _default_websocket_factory
        self.task_id_factory = task_id_factory or _task_id
        self._ws: Any | None = None
        self._task_id: str | None = None
        self._started = False
        self._closed = False
        self._pending: list[dict[str, object]] = []

    def send_json(
        self,
        payload: dict[str, object],
        *,
        headers: dict[str, str],
        timeout_s: float,
        url: str | None = None,
    ) -> None:
        self._ensure_started(payload, headers=headers, timeout_s=timeout_s, url=url or self.default_url)
        audio = base64.b64decode(str(payload.get("audio_base64") or ""))
        if hasattr(self._ws, "send_binary"):
            self._ws.send_binary(audio)
        else:
            self._ws.send(audio)

    def receive_json(self) -> dict[str, object] | None:
        if self._pending:
            return self._pending.pop(0)
        if self._ws is None:
            return None
        try:
            message = _recv_ws_json(self._ws)
        except BaseException as exc:
            if _is_timeout(exc):
                return None
            raise
        return message

    def cancel(self) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ws is None:
            return
        if self._started and self._task_id:
            try:
                self._ws.send(json.dumps({"header": {"action": "finish-task", "task_id": self._task_id}, "payload": {}}))
            except BaseException:
                pass
        try:
            self._ws.close()
        except BaseException:
            pass

    def _ensure_started(
        self,
        payload: Mapping[str, object],
        *,
        headers: dict[str, str],
        timeout_s: float,
        url: str,
    ) -> None:
        if self._started:
            return
        self._task_id = self.task_id_factory()
        self._ws = self.websocket_factory(url, header=_ws_headers(headers), timeout=timeout_s)
        _set_ws_timeout(self._ws, min(max(float(timeout_s), 0.5), 5.0))
        command = _dashscope_run_task_payload(payload, task_id=self._task_id)
        self._ws.send(json.dumps(command))
        while True:
            message = _recv_ws_json(self._ws)
            event = _lower_str(_first_present(_mapping_or_empty(message.get("header")), "event"))
            if event == "task-started":
                self._started = True
                return
            if event in {"task-failed", "error"}:
                raise RuntimeError(str(message))
            self._pending.append(message)


class MiniMaxWebSocketTtsTransport:
    """Minimal MiniMax T2A WebSocket transport with fakeable connection setup."""

    default_url = "wss://api.minimax.io/ws/v1/t2a_v2"

    def __init__(self, *, websocket_factory: WebSocketFactory | None = None) -> None:
        self.websocket_factory = websocket_factory or _default_websocket_factory
        self._ws: Any | None = None
        self._closed = False

    def stream_json(
        self,
        payload: dict[str, object],
        *,
        headers: dict[str, str],
        timeout_s: float,
        url: str | None = None,
    ) -> Iterable[dict[str, object] | bytes]:
        self._ws = self.websocket_factory(url or self.default_url, header=_ws_headers(headers), timeout=timeout_s)
        _set_ws_timeout(self._ws, min(max(float(timeout_s), 0.5), 10.0))
        self._ws.send(json.dumps(_minimax_task_start_payload(payload)))
        self._wait_for_minimax_started()
        self._ws.send(json.dumps({"event": "task_continue", "text": str(payload.get("text") or "")}))
        while True:
            message = _recv_ws_json(self._ws)
            yield message
            event = _lower_str(_first_present(message, "event", "type", "name"))
            if bool(message.get("is_final")) or event in {"task_finished", "task-finished", "task_failed", "task-failed"}:
                break

    def cancel(self) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ws is None:
            return
        try:
            self._ws.send(json.dumps({"event": "task_finish"}))
        except BaseException:
            pass
        try:
            self._ws.close()
        except BaseException:
            pass

    def _wait_for_minimax_started(self) -> None:
        if self._ws is None:
            raise RuntimeError("websocket not connected")
        while True:
            message = _recv_ws_json(self._ws)
            event = _lower_str(_first_present(message, "event", "type", "name"))
            if event == "task_started":
                return
            if event in {"task_failed", "task-failed", "error", "failed"}:
                raise RuntimeError(str(message))


def _asr_audio_frame_payload(config: CloudProviderConfig, frame: AudioFrame) -> dict[str, object]:
    audio = frame.payload or frame.pcm
    return {
        "type": "audio_frame",
        "provider": config.provider,
        "model": config.model,
        "sequence": frame.sequence,
        "duration_ms": frame.duration_ms,
        "sample_rate_hz": frame.sample_rate_hz,
        "channels": frame.channels,
        "audio_format": "opus" if frame.payload else "pcm16",
        "audio_base64": base64.b64encode(audio).decode("ascii"),
    }


def _tts_request_payload(config: CloudProviderConfig, request: StreamingTtsRequest) -> dict[str, object]:
    return {
        "type": "tts_request",
        "provider": config.provider,
        "model": config.model,
        "text": request.text,
        "round_id": request.round_id,
        "voice_id": request.voice_code or config.voice_id,
        "emotion": request.emotion,
        "speed": request.speed,
        "volume": request.volume,
        "sample_rate_hz": request.sample_rate_hz,
        "channels": request.channels,
        "audio_format": request.audio_format,
        "stream": True,
    }


def _audio_payload_from_message(message: Mapping[str, object]) -> bytes | None:
    data = _mapping_or_empty(message.get("data"))
    if "audio_base64" in message or "audioBase64" in message:
        return _decode_audio_text(_first_present(message, "audio_base64", "audioBase64"), encoding="base64")
    if "audio_hex" in message or "audioHex" in message:
        return _decode_audio_text(_first_present(message, "audio_hex", "audioHex"), encoding="hex")
    if "audio" in data:
        return _decode_audio_text(data.get("audio"), encoding="hex")
    value = _first_present(message, "audio", "payload")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        encoding = str(message.get("audio_encoding") or message.get("audioEncoding") or "hex").lower()
        return _decode_audio_text(value, encoding=encoding)
    return None


def _decode_audio_text(value: object, *, encoding: str) -> bytes | None:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return b""
    if encoding in {"hex", "hexadecimal"}:
        try:
            return bytes.fromhex(text)
        except ValueError:
            return base64.b64decode(text)
    try:
        return base64.b64decode(text)
    except ValueError:
        return bytes.fromhex(text)


def _asr_text_from_message(message: Mapping[str, object]) -> str | None:
    for key in ("text", "transcript", "sentence"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    result = message.get("result")
    if isinstance(result, Mapping):
        text = _text_from_mapping(result)
        if text:
            return text
    payload = message.get("payload")
    if isinstance(payload, Mapping):
        text = _text_from_mapping(payload)
        if text:
            return text
        output = payload.get("output")
        if isinstance(output, Mapping):
            text = _text_from_mapping(output)
            if text:
                return text
            sentence = output.get("sentence")
            if isinstance(sentence, Mapping):
                return _text_from_mapping(sentence)
    return None


def _bool_from_provider_message(message: Mapping[str, object]) -> bool:
    value = _first_present(message, "is_final", "isFinal", "final", "sentence_end", "sentenceEnd")
    if value is None:
        payload = _mapping_or_empty(message.get("payload"))
        output = _mapping_or_empty(payload.get("output"))
        sentence = _mapping_or_empty(output.get("sentence"))
        value = _first_present(sentence, "sentence_end", "sentenceEnd")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "is_final", "final"}:
            return True
        if normalized in {"false", "0", "no", "partial"}:
            return False
    text_type = _lower_str(_first_present(message, "text_type", "textType"))
    return text_type in {"is_final", "final"}


def _text_from_mapping(mapping: Mapping[str, object]) -> str | None:
    for key in ("text", "transcript", "sentence"):
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, Mapping):
            nested = _text_from_mapping(value)
            if nested:
                return nested
    return None


def _first_present(mapping: Mapping[str, object], *keys: str, default: object | None = None) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _first_env(mapping: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return ""


def _redact_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 2:
        return "*" * len(value)
    return f"{value[0]}***{value[-1]}"


def _call_optional(target: object, method_name: str) -> None:
    method = getattr(target, method_name, None)
    if callable(method):
        method()


def _send_json(
    transport: AsrJsonTransport,
    payload: dict[str, object],
    *,
    headers: dict[str, str],
    timeout_s: float,
    url: str | None,
) -> None:
    try:
        transport.send_json(payload, headers=headers, timeout_s=timeout_s, url=url)
    except TypeError:
        transport.send_json(payload, headers=headers, timeout_s=timeout_s)


def _stream_json(
    transport: TtsJsonStreamTransport,
    payload: dict[str, object],
    *,
    headers: dict[str, str],
    timeout_s: float,
    url: str | None,
) -> Iterable[dict[str, object] | bytes]:
    try:
        return transport.stream_json(payload, headers=headers, timeout_s=timeout_s, url=url)
    except TypeError:
        return transport.stream_json(payload, headers=headers, timeout_s=timeout_s)


WebSocketFactory = Callable[[str], Any]


def _default_websocket_factory(url: str, *, header: list[str], timeout: float) -> Any:
    try:
        import websocket  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on honjia optional runtime package.
        raise RuntimeError("websocket-client package is required for real streaming providers") from exc
    return websocket.create_connection(url, header=header, timeout=timeout)


def _ws_headers(headers: Mapping[str, str]) -> list[str]:
    return [f"{name}: {value}" for name, value in headers.items() if value]


def _set_ws_timeout(ws: Any, timeout_s: float) -> None:
    settimeout = getattr(ws, "settimeout", None)
    if callable(settimeout):
        settimeout(float(timeout_s))


def _recv_ws_json(ws: Any) -> dict[str, object]:
    raw = ws.recv()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        value = json.loads(raw)
    elif isinstance(raw, Mapping):
        value = raw
    else:
        raise ValueError("websocket message must be JSON text or mapping")
    if not isinstance(value, Mapping):
        raise ValueError("websocket JSON message must be an object")
    return dict(value)


def _is_timeout(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    return isinstance(exc, TimeoutError) or "timeout" in name or "timedout" in name


def _task_id() -> str:
    return uuid4().hex[:32]


def _dashscope_run_task_payload(payload: Mapping[str, object], *, task_id: str) -> dict[str, object]:
    audio_format = str(payload.get("audio_format") or "pcm").lower()
    if audio_format == "pcm16":
        audio_format = "pcm"
    return {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": str(payload.get("model") or "fun-asr-realtime"),
            "parameters": {
                "format": audio_format,
                "sample_rate": int(_safe_int(payload.get("sample_rate_hz"), default=16000)),
                "semantic_punctuation_enabled": False,
                "max_sentence_silence": int(_safe_int(payload.get("max_sentence_silence"), default=800)),
            },
            "input": {},
        },
    }


def _minimax_task_start_payload(payload: Mapping[str, object]) -> dict[str, object]:
    audio_format = str(payload.get("audio_format") or "mp3").lower()
    if audio_format == "pcm16":
        audio_format = "pcm"
    return {
        "event": "task_start",
        "model": str(payload.get("model") or "speech-2.8-turbo"),
        "language_boost": str(payload.get("language_boost") or "Chinese"),
        "voice_setting": {
            "voice_id": str(payload.get("voice_id") or "female-shaonv"),
            "speed": float(_safe_float(payload.get("speed"), default=1.0) or 1.0),
            "vol": float(_safe_float(payload.get("volume"), default=1.0) or 1.0),
            "pitch": float(_safe_float(payload.get("pitch"), default=0.0) or 0.0),
        },
        "audio_setting": {
            "sample_rate": int(_safe_int(payload.get("sample_rate_hz"), default=32000)),
            "bitrate": int(_safe_int(payload.get("bitrate"), default=128000)),
            "format": audio_format,
            "channel": int(_safe_int(payload.get("channels"), default=1)),
        },
    }


def _mapping_or_empty(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _redact_text(text: str, *secrets: str) -> str:
    redacted = str(text)
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, _redact_secret(secret))
    return redacted


def _safe_float(value: object, *, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _lower_str(value: object) -> str:
    return str(value or "").strip().lower()


__all__ = [
    "AsrJsonTransport",
    "CloudProviderConfig",
    "DashScopeStreamingAsrProvider",
    "DashScopeWebSocketAsrTransport",
    "MiniMaxStreamingTtsProvider",
    "MiniMaxWebSocketTtsTransport",
    "TtsJsonStreamTransport",
]
