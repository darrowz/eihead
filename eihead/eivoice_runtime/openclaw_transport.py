"""OpenClaw realtime transport adapter for the EiVoice runtime."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from eihead.eivoice_runtime.joyinside_voice import JoyInsideVoiceEvent

from .transport import InMemoryVoiceStreamTransport

WebSocketFactory = Callable[[str], Any]
EnvelopeEncoder = Callable[[Mapping[str, Any]], dict[str, Any]]
EnvelopeDecoder = Callable[[Mapping[str, Any]], dict[str, Any] | None]


class OpenClawRealtimeTransport(InMemoryVoiceStreamTransport):
    """Thin websocket transport that adapts EiVoice events to OpenClaw realtime JSON."""

    def __init__(
        self,
        *,
        url: str,
        token: str | None = None,
        token_env_var: str = "OPENCLAW_REALTIME_TOKEN",
        headers: Mapping[str, str] | None = None,
        protocol: str = "openclaw.realtime.v1",
        session_config: Mapping[str, Any] | None = None,
        connect_timeout: float = 10.0,
        receive_timeout: float = 0.02,
        session_ready_timeout: float = 15.0,
        heartbeat_interval: float = 10.0,
        websocket_factory: WebSocketFactory | None = None,
        encode_event: EnvelopeEncoder | None = None,
        decode_message: EnvelopeDecoder | None = None,
        clock: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            clock=clock,
            heartbeat_interval_s=heartbeat_interval,
            transport_name="openclaw_realtime",
            **kwargs,
        )
        self.url = str(url)
        self.token = token
        self.token_env_var = str(token_env_var)
        self.headers = dict(headers or {})
        self.protocol = str(protocol or "")
        self.session_config = dict(session_config or {})
        self.connect_timeout = float(connect_timeout)
        self.receive_timeout = float(receive_timeout)
        self.session_ready_timeout = float(session_ready_timeout)
        self.websocket_factory = websocket_factory or _default_websocket_factory
        self.encode_event = encode_event or default_openclaw_encode_event
        self.decode_message = decode_message or default_openclaw_decode_message
        self._ws: Any | None = None
        self._session_state = "idle"
        self._last_rx_ms: int | None = None
        self._last_tx_ms: int | None = None
        self._last_user_transcript = ""
        self._last_assistant_transcript = ""
        self._session_id = ""

    def connect(self) -> None:
        if not self.url:
            raise ValueError("url is required")
        if self._ws is not None:
            return
        self._set_state("connecting")
        try:
            ws = self.websocket_factory(
                self.url,
                header=_ws_headers(self._connection_headers()),
                timeout=self.connect_timeout,
            )
            _set_ws_timeout(ws, self.receive_timeout)
            self._send_session_config(ws)
            self._wait_for_session_ready(ws)
        except Exception as exc:
            self.record_error(exc, context="connect")
            self._session_state = "error"
            self.schedule_reconnect("connect_error")
            raise
        self._ws = ws
        self._session_state = "ready"
        self.mark_connected()

    def reconnect(self) -> None:
        self._close_socket()
        self.connect()

    def close(self, reason: str | None = None) -> None:
        self._close_socket()
        with self._lock:
            now = self._clock()
            self._connection_state = "closed"
            self._reconnect_reason = reason
            self._last_transition_at = now
            self._last_activity_at = now
            self._awaiting_pong = False

    def send_event(self, event: Mapping[str, Any] | JoyInsideVoiceEvent) -> bool:
        payload = _event_dict(event)
        ws = self._ws
        if ws is None:
            self.record_error(RuntimeError("websocket not connected"), context="send_event")
            return False
        try:
            message = self.encode_event(payload)
            ws.send(json.dumps(message))
        except Exception as exc:
            self.record_error(exc, context="send_event")
            self._session_state = "error"
            self._close_socket()
            self.schedule_reconnect("send_error")
            return False
        with self._lock:
            self._last_tx_ms = int(round(self._clock() * 1000))
            self._last_activity_at = self._clock()
        return True

    def receive_event(self, *, block: bool = False, timeout: float | None = None) -> dict[str, Any] | None:
        ws = self._ws
        if ws is None:
            return None
        timeout_s = self.receive_timeout if timeout is None else float(timeout)
        if not block and timeout is None:
            timeout_s = self.receive_timeout
        try:
            _set_ws_timeout(ws, timeout_s)
            message = _recv_ws_json(ws)
        except Exception as exc:
            if _is_timeout(exc):
                return None
            self.record_error(exc, context="receive_event")
            self._session_state = "error"
            self._close_socket()
            self.schedule_reconnect("receive_error")
            return None
        event = self.decode_message(message)
        self._record_message(message, event)
        if event is None:
            return None
        with self._lock:
            self._last_rx_ms = int(round(self._clock() * 1000))
            self._last_activity_at = self._clock()
        if _event_type(event) == "PONG":
            self.record_pong(event)
        return event

    def drain_inbound_events(self) -> list[dict[str, Any]]:
        drained: list[dict[str, Any]] = []
        while True:
            event = self.receive_event(timeout=0.0)
            if event is None:
                return drained
            drained.append(event)

    def status(self) -> dict[str, Any]:
        status = super().status()
        status["endpoint"] = {
            "url": self.url,
            "connect_timeout_s": self.connect_timeout,
            "receive_timeout_s": self.receive_timeout,
            "session_ready_timeout_s": self.session_ready_timeout,
            "headers": sorted(self.headers.keys()),
            "protocol": self.protocol,
            "token_env_var": self.token_env_var,
            "has_token": bool(self._resolved_token()),
        }
        status["socket_connected"] = self._ws is not None
        status["openclaw_ws"] = {
            "connected": self._ws is not None and status["state"] == "connected",
            "url": self.url,
            "last_error": _error_message(status.get("last_error")),
            "last_rx_ms": self._last_rx_ms,
            "last_tx_ms": self._last_tx_ms,
            "session_state": self._session_state,
            "session_id": self._session_id,
            "last_user_transcript": self._last_user_transcript,
            "last_assistant_transcript": self._last_assistant_transcript,
        }
        return status

    def _connection_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        token = self._resolved_token()
        if token and "Authorization" not in self.headers:
            headers["Authorization"] = f"Bearer {token}"
        if self.protocol and "Sec-WebSocket-Protocol" not in self.headers:
            headers["Sec-WebSocket-Protocol"] = self.protocol
        headers.update(self.headers)
        return headers

    def _resolved_token(self) -> str | None:
        return self.token or os.getenv(self.token_env_var) or None

    def _set_state(self, state: str) -> None:
        with self._lock:
            now = self._clock()
            self._connection_state = state
            self._last_transition_at = now
            self._last_activity_at = now

    def _send_session_config(self, ws: Any) -> None:
        token = self._resolved_token()
        payload = dict(self.session_config)
        payload["type"] = "session.config"
        if token and not payload.get("apiKey"):
            payload["apiKey"] = token
        ws.send(json.dumps(payload))
        with self._lock:
            self._last_tx_ms = int(round(self._clock() * 1000))
            self._session_state = "configuring"

    def _wait_for_session_ready(self, ws: Any) -> None:
        _set_ws_timeout(ws, self.session_ready_timeout)
        while True:
            message = _recv_ws_json(ws)
            self._record_message(message, None)
            message_type = str(message.get("type") or "").strip()
            if message_type == "session.ready":
                self._session_id = str(message.get("sessionId") or message.get("session_id") or "")
                self._session_state = "ready"
                _set_ws_timeout(ws, self.receive_timeout)
                return
            if message_type == "error":
                code = message.get("code")
                detail = message.get("message") or message.get("error") or "OpenClaw session config failed"
                raise RuntimeError(f"OpenClaw realtime error {code}: {detail}")

    def _record_message(self, message: Mapping[str, Any], event: Mapping[str, Any] | None) -> None:
        message_type = str(message.get("type") or "").strip()
        now_ms = int(round(self._clock() * 1000))
        if message_type:
            self._session_state = _session_state_from_message(message_type, self._session_state)
        if message_type in {"transcript.delta", "transcript.done"}:
            role = str(message.get("role") or "").strip()
            text = str(message.get("text") or "").strip()
            if role == "user" and text:
                self._last_user_transcript = text
            if role == "assistant" and text:
                self._last_assistant_transcript = text
        if event is not None and _event_type(event) in {"AUDIO_CHUNK", "AUDIO"}:
            self._last_rx_ms = now_ms
        elif message_type:
            self._last_rx_ms = now_ms

    def _close_socket(self) -> None:
        ws, self._ws = self._ws, None
        if ws is None:
            return
        close = getattr(ws, "close", None)
        if callable(close):
            close()


def default_openclaw_encode_event(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(event))
    content = _mapping(payload.get("content"))
    event_type = _event_type(payload)
    if event_type in {"AUDIO", "AUDIO_CHUNK"}:
        encoded = (
            content.get("audioBase64")
            or content.get("audio_base64")
            or payload.get("audioBase64")
            or payload.get("audio_base64")
        )
        if not encoded:
            raise ValueError("audio event missing audioBase64 payload")
        sample_rate_hz = _optional_int(
            content.get("sampleRateHz"),
            content.get("sample_rate_hz"),
            payload.get("sampleRateHz"),
        )
        openclaw_audio = str(encoded)
        if sample_rate_hz and sample_rate_hz != 24000:
            openclaw_audio = _resample_pcm16_base64(openclaw_audio, source_rate=sample_rate_hz, target_rate=24000)
        return {
            "type": "audio.append",
            "data": openclaw_audio,
            "sequence": _optional_int(content.get("index"), payload.get("index")),
            "duration_ms": _optional_int(content.get("durationMs"), content.get("duration_ms"), payload.get("durationMs")),
            "sample_rate_hz": 24000,
            "channels": _optional_int(content.get("channels"), payload.get("channels")),
            "uid": payload.get("uid"),
            "mid": payload.get("mid"),
        }
    if event_type == "PING":
        return {
            "type": "ping",
            "uid": payload.get("uid"),
            "mid": payload.get("mid"),
            "timestamp": content.get("timestamp") or payload.get("timestamp"),
        }
    return {
        "type": "client.event",
        "event": payload,
    }


def default_openclaw_decode_message(message: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = deepcopy(dict(message))
    message_type = str(
        payload.get("type") or payload.get("eventType") or payload.get("contentType") or ""
    ).strip()
    normalized_type = message_type.lower()
    if normalized_type == "pong":
        return {"contentType": "PONG", "content": {"eventType": "PONG"}}
    encoded = (
        payload.get("delta")
        or payload.get("data")
        or payload.get("audio")
        or payload.get("audioBase64")
        or payload.get("audio_base64")
    )
    if encoded:
        return {
            "contentType": "AUDIO_CHUNK",
            "content": {
                "eventType": "AUDIO_CHUNK",
                "audioBase64": str(encoded),
                "index": _optional_int(payload.get("sequence"), payload.get("index")) or 0,
                "durationMs": _optional_int(payload.get("duration_ms"), payload.get("durationMs")) or 60,
                "sampleRateHz": _optional_int(payload.get("sample_rate_hz"), payload.get("sampleRateHz")) or 16000,
                "channels": _optional_int(payload.get("channels")) or 1,
                    "metadata": {
                        "source": "openclaw_realtime",
                        "messageType": message_type or "audio",
                    },
                },
        }
    if "content" in payload or "contentType" in payload:
        return payload
    if message_type:
        return {
            "contentType": "OPENCLAW_EVENT",
            "content": {
                "eventType": message_type,
                "payload": payload,
            },
        }
    return None


def _default_websocket_factory(url: str, *, header: list[str], timeout: float) -> Any:
    try:
        import websocket  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("websocket-client package is required for OpenClaw realtime transport") from exc
    return websocket.create_connection(url, header=header, timeout=timeout)


def _ws_headers(headers: Mapping[str, str]) -> list[str]:
    return [f"{name}: {value}" for name, value in headers.items() if value]


def _set_ws_timeout(ws: Any, timeout_s: float) -> None:
    settimeout = getattr(ws, "settimeout", None)
    if callable(settimeout):
        settimeout(float(timeout_s))


def _recv_ws_json(ws: Any) -> dict[str, Any]:
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


def _event_dict(event: Mapping[str, Any] | JoyInsideVoiceEvent) -> dict[str, Any]:
    if isinstance(event, JoyInsideVoiceEvent):
        return event.to_dict()
    if isinstance(event, Mapping):
        return deepcopy(dict(event))
    raise TypeError("voice transport events must be mappings or JoyInsideVoiceEvent instances")


def _event_type(event: Mapping[str, Any]) -> str:
    content = _mapping(event.get("content"))
    return str(
        content.get("eventType")
        or event.get("eventType")
        or event.get("contentType")
        or event.get("type")
        or ""
    ).upper()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_int(*values: Any) -> int | None:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _resample_pcm16_base64(encoded: str, *, source_rate: int, target_rate: int) -> str:
    if source_rate <= 0 or target_rate <= 0 or source_rate == target_rate:
        return encoded
    raw = base64.b64decode(encoded)
    if len(raw) < 2:
        return encoded
    sample_count = len(raw) // 2
    source = [int.from_bytes(raw[index * 2 : index * 2 + 2], "little", signed=True) for index in range(sample_count)]
    target_count = max(1, int(round(sample_count * target_rate / source_rate)))
    ratio = source_rate / target_rate
    out = bytearray(target_count * 2)
    for index in range(target_count):
        src_pos = index * ratio
        src_index = int(src_pos)
        frac = src_pos - src_index
        sample_a = source[min(src_index, sample_count - 1)]
        sample_b = source[min(src_index + 1, sample_count - 1)]
        sample = int(round(sample_a * (1.0 - frac) + sample_b * frac))
        sample = max(-32768, min(32767, sample))
        out[index * 2 : index * 2 + 2] = int(sample).to_bytes(2, "little", signed=True)
    return base64.b64encode(bytes(out)).decode("ascii")


def _session_state_from_message(message_type: str, current: str) -> str:
    if message_type == "session.ready":
        return "ready"
    if message_type == "session.ended":
        return "ended"
    if message_type == "turn.started":
        return "listening"
    if message_type == "turn.ended":
        return "ready"
    if message_type == "audio.delta":
        return "speaking"
    if message_type == "error":
        return "error"
    return current


def _error_message(value: Any) -> str:
    payload = _mapping(value)
    return str(payload.get("message") or payload.get("kind") or "")
