"""OpenClaw realtime transport adapter for the EiVoice runtime."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform as platform_module
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from eihead.eivoice_runtime.joyinside_voice import JoyInsideVoiceEvent

from .transport import InMemoryVoiceStreamTransport

WebSocketFactory = Callable[[str], Any]
EnvelopeEncoder = Callable[[Mapping[str, Any]], dict[str, Any]]
EnvelopeDecoder = Callable[[Mapping[str, Any]], dict[str, Any] | None]
DeviceSigner = Callable[[str, str], str]

GATEWAY_PROTOCOL_VERSION = 4
DEFAULT_CLIENT_ID = "cli"
DEFAULT_CLIENT_MODE = "cli"
DEFAULT_CLIENT_VERSION = "eihead"
DEFAULT_SCOPES = ["operator.read", "operator.write"]
DEFAULT_CAPS = ["tool-events"]


class OpenClawRealtimeTransport(InMemoryVoiceStreamTransport):
    """Thin websocket transport that adapts EiVoice audio to OpenClaw Talk relay RPC."""

    def __init__(
        self,
        *,
        url: str,
        token: str | None = None,
        token_env_var: str = "OPENCLAW_REALTIME_TOKEN",
        headers: Mapping[str, str] | None = None,
        protocol: str = "",
        session_config: Mapping[str, Any] | None = None,
        client_id: str = DEFAULT_CLIENT_ID,
        client_mode: str = DEFAULT_CLIENT_MODE,
        client_version: str = DEFAULT_CLIENT_VERSION,
        client_platform: str | None = None,
        client_device_family: str | None = None,
        locale: str = "zh-CN",
        device_identity: Mapping[str, Any] | None = None,
        device_identity_path: str | None = None,
        device_signer: DeviceSigner | None = None,
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
        self.client_id = str(client_id or DEFAULT_CLIENT_ID)
        self.client_mode = str(client_mode or DEFAULT_CLIENT_MODE)
        self.client_version = str(client_version or DEFAULT_CLIENT_VERSION)
        self.client_platform = _normalize_device_auth_metadata(
            client_platform or _default_client_platform()
        )
        self.client_device_family = _normalize_device_auth_metadata(
            client_device_family or platform_module.machine()
        )
        self.locale = str(locale or "zh-CN")
        self.device_identity = _resolve_device_identity(device_identity, device_identity_path)
        self.device_signer = device_signer or _default_device_signer
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
        self._relay_session_id = ""
        self._request_id = 0
        self._input_sample_rate_hz = 24000
        self._output_sample_rate_hz = 24000
        self._connected_scopes: list[str] = []

    def connect(self) -> None:
        if not self.url:
            raise ValueError("url is required")
        if self._ws is not None:
            return
        self._set_state("connecting")
        ws: Any | None = None
        try:
            ws = self.websocket_factory(
                self.url,
                header=_ws_headers(self._connection_headers()),
                timeout=self.connect_timeout,
            )
            _set_ws_timeout(ws, self.receive_timeout)
            nonce = self._wait_for_gateway_challenge(ws)
            self._connect_gateway(ws, nonce=nonce)
            self._create_relay_session(ws)
            self._wait_for_relay_ready(ws)
        except Exception as exc:
            if ws is not None:
                self._send_close_session(ws)
                _safe_close_ws(ws)
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
        if not self._relay_session_id:
            self.record_error(RuntimeError("OpenClaw relay session is not ready"), context="send_event")
            return False
        try:
            audio = self.encode_event(payload)
            audio_base64 = str(audio.get("audioBase64") or audio.get("audio_base64") or audio.get("data") or "")
            if not audio_base64:
                raise ValueError("audio event missing audioBase64 payload")
            self._send_request(
                ws,
                "talk.session.appendAudio",
                {
                    "sessionId": self._relay_session_id,
                    "audioBase64": audio_base64,
                    "timestamp": int(round(self._clock() * 1000)),
                },
            )
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

    def cancel_output(self, reason: str = "barge-in") -> bool:
        ws = self._ws
        if ws is None:
            self.record_error(RuntimeError("websocket not connected"), context="cancel_output")
            return False
        if not self._relay_session_id:
            self.record_error(RuntimeError("OpenClaw relay session is not ready"), context="cancel_output")
            return False
        try:
            self._send_request(
                ws,
                "talk.session.cancelOutput",
                {
                    "sessionId": self._relay_session_id,
                    "reason": str(reason or "barge-in"),
                },
            )
        except Exception as exc:
            self.record_error(exc, context="cancel_output")
            return False
        self._session_state = "interrupted"
        return True

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
        status["provider"] = str(self.session_config.get("provider") or "")
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
        return dict(self.headers)

    def _resolved_token(self) -> str | None:
        return self.token or os.getenv(self.token_env_var) or None

    def _set_state(self, state: str) -> None:
        with self._lock:
            now = self._clock()
            self._connection_state = state
            self._last_transition_at = now
            self._last_activity_at = now

    def _wait_for_gateway_challenge(self, ws: Any) -> str:
        _set_ws_timeout(ws, self.session_ready_timeout)
        while True:
            try:
                message = _recv_ws_json(ws)
            except Exception as exc:
                if _is_timeout(exc):
                    if self.device_identity:
                        raise RuntimeError("OpenClaw gateway connect challenge timed out") from exc
                    return ""
                raise
            self._record_message(message, None)
            if str(message.get("type") or "") == "event" and str(message.get("event") or "") == "connect.challenge":
                payload = _mapping(message.get("payload"))
                return str(payload.get("nonce") or "")

    def _connect_gateway(self, ws: Any, *, nonce: str) -> None:
        params: dict[str, Any] = {
            "minProtocol": GATEWAY_PROTOCOL_VERSION,
            "maxProtocol": GATEWAY_PROTOCOL_VERSION,
            "client": {
                "id": self.client_id,
                "version": self.client_version,
                "platform": self.client_platform,
                "mode": self.client_mode,
                "deviceFamily": self.client_device_family,
            },
            "role": "operator",
            "scopes": list(DEFAULT_SCOPES),
            "caps": list(DEFAULT_CAPS),
            "userAgent": "eihead-openclaw-realtime",
            "locale": self.locale,
        }
        token = self._resolved_token()
        if token:
            params["auth"] = {"token": token}
        device = self._build_connect_device(nonce=nonce, token=token)
        if device:
            params["device"] = device
        response = self._request_response(ws, "connect", params)
        auth = _mapping(response.get("auth"))
        scopes = auth.get("scopes")
        self._connected_scopes = [str(item) for item in scopes] if isinstance(scopes, list) else []
        self._session_state = "gateway_connected"

    def _create_relay_session(self, ws: Any) -> None:
        params = self._session_create_params()
        response = self._request_response(ws, "talk.session.create", params)
        self._relay_session_id = str(response.get("relaySessionId") or response.get("sessionId") or "")
        if not self._relay_session_id:
            raise RuntimeError("OpenClaw talk.session.create did not return relaySessionId")
        self._session_id = self._relay_session_id
        audio = _mapping(response.get("audio"))
        self._input_sample_rate_hz = _optional_int(audio.get("inputSampleRateHz")) or 24000
        self._output_sample_rate_hz = _optional_int(audio.get("outputSampleRateHz")) or 24000
        self._session_state = "session_created"

    def _wait_for_relay_ready(self, ws: Any) -> None:
        _set_ws_timeout(ws, self.session_ready_timeout)
        while True:
            message = _recv_ws_json(ws)
            self._record_message(message, None)
            if _is_relay_event(message, self._relay_session_id, "ready"):
                self._session_state = "ready"
                _set_ws_timeout(ws, self.receive_timeout)
                return
            if _is_relay_event(message, self._relay_session_id, "error"):
                payload = _mapping(message.get("payload"))
                raise RuntimeError(str(payload.get("message") or "OpenClaw realtime relay failed"))

    def _request_response(self, ws: Any, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        _set_ws_timeout(ws, self.session_ready_timeout)
        request_id = self._send_request(ws, method, dict(params))
        while True:
            message = _recv_ws_json(ws)
            self._record_message(message, None)
            if str(message.get("type") or "") != "res" or str(message.get("id") or "") != request_id:
                continue
            if message.get("ok") is True:
                payload = message.get("payload")
                return dict(payload) if isinstance(payload, Mapping) else {}
            error = _mapping(message.get("error"))
            raise RuntimeError(str(error.get("message") or f"OpenClaw gateway request failed: {method}"))

    def _send_request(self, ws: Any, method: str, params: Mapping[str, Any]) -> str:
        self._request_id += 1
        request_id = str(self._request_id)
        ws.send(json.dumps({"type": "req", "id": request_id, "method": method, "params": dict(params)}))
        with self._lock:
            self._last_tx_ms = int(round(self._clock() * 1000))
            self._last_activity_at = self._clock()
        return request_id

    def _session_create_params(self) -> dict[str, Any]:
        session_key = str(self.session_config.get("sessionKey") or self.session_config.get("session_key") or "honjia")
        params: dict[str, Any] = {
            "sessionKey": session_key,
            "mode": "realtime",
            "transport": "gateway-relay",
            "brain": "agent-consult",
        }
        for source, target in (
            ("provider", "provider"),
            ("model", "model"),
            ("voice", "voice"),
            ("instructions", "instructions"),
        ):
            value = self.session_config.get(source)
            if value:
                params[target] = value
        return params

    def _build_connect_device(self, *, nonce: str, token: str | None) -> dict[str, Any] | None:
        identity = self.device_identity
        if not identity:
            return None
        private_key_pem = str(identity.get("privateKeyPem") or identity.get("private_key_pem") or "")
        public_key_pem = str(identity.get("publicKeyPem") or identity.get("public_key_pem") or "")
        if not private_key_pem or not public_key_pem:
            return None
        public_key = _public_key_raw_base64url(public_key_pem)
        device_id = str(identity.get("deviceId") or identity.get("device_id") or _device_id_from_public_key(public_key))
        signed_at = int(round(time.time() * 1000))
        payload = _device_auth_payload_v3(
            device_id=device_id,
            client_id=self.client_id,
            client_mode=self.client_mode,
            role="operator",
            scopes=DEFAULT_SCOPES,
            signed_at_ms=signed_at,
            token=token or "",
            nonce=nonce,
            platform=self.client_platform,
            device_family=self.client_device_family,
        )
        return {
            "id": device_id,
            "publicKey": public_key,
            "signature": self.device_signer(private_key_pem, payload),
            "signedAt": signed_at,
            "nonce": nonce,
        }

    def _record_message(self, message: Mapping[str, Any], event: Mapping[str, Any] | None) -> None:
        message_type = str(message.get("type") or "").strip()
        now_ms = int(round(self._clock() * 1000))
        relay_event = _relay_event_payload(message)
        if relay_event and (not self._relay_session_id or str(relay_event.get("relaySessionId") or "") == self._relay_session_id):
            relay_type = str(relay_event.get("type") or "").strip()
            if relay_type:
                self._session_state = _session_state_from_message(relay_type, self._session_state)
            if relay_type == "transcript":
                role = str(relay_event.get("role") or "").strip()
                text = str(relay_event.get("text") or "").strip()
                if role == "user" and text:
                    self._last_user_transcript = text
                if role == "assistant" and text:
                    self._last_assistant_transcript = text
        elif message_type:
            self._session_state = _session_state_from_message(message_type, self._session_state)
        if event is not None and _event_type(event) in {"AUDIO_CHUNK", "AUDIO"}:
            self._last_rx_ms = now_ms
        elif message_type:
            self._last_rx_ms = now_ms

    def _close_socket(self) -> None:
        ws, self._ws = self._ws, None
        if ws is None:
            return
        self._send_close_session(ws)
        _safe_close_ws(ws)

    def _send_close_session(self, ws: Any) -> None:
        relay_session_id = self._relay_session_id
        if not relay_session_id:
            return
        try:
            self._send_request(ws, "talk.session.close", {"sessionId": relay_session_id})
        except Exception:
            pass
        self._relay_session_id = ""


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
            "audioBase64": openclaw_audio,
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
    relay_payload = _relay_event_payload(payload)
    if relay_payload:
        relay_type = str(relay_payload.get("type") or "").strip()
        if relay_type == "audio":
            encoded = relay_payload.get("audioBase64") or relay_payload.get("audio_base64")
            if encoded:
                return {
                    "contentType": "AUDIO_CHUNK",
                    "content": {
                        "eventType": "AUDIO_CHUNK",
                        "audioBase64": str(encoded),
                        "index": _optional_int(relay_payload.get("sequence"), relay_payload.get("index")) or 0,
                        "durationMs": _optional_int(relay_payload.get("durationMs"), relay_payload.get("duration_ms")) or 60,
                        "sampleRateHz": 24000,
                        "channels": 1,
                        "metadata": {
                            "source": "openclaw_realtime",
                            "messageType": "talk.event.audio",
                        },
                    },
                }
        if relay_type:
            return {
                "contentType": "OPENCLAW_EVENT",
                "content": {
                    "eventType": relay_type,
                    "payload": relay_payload,
                },
            }
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


def _safe_close_ws(ws: Any) -> None:
    close = getattr(ws, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


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


def _relay_event_payload(message: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(message.get("type") or "") != "event" or str(message.get("event") or "") != "talk.event":
        return None
    payload = message.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _is_relay_event(message: Mapping[str, Any], relay_session_id: str, event_type: str) -> bool:
    payload = _relay_event_payload(message)
    if not payload:
        return False
    return (
        str(payload.get("relaySessionId") or "") == relay_session_id
        and str(payload.get("type") or "") == event_type
    )


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


def _resolve_device_identity(
    explicit: Mapping[str, Any] | None,
    device_identity_path: str | None,
) -> dict[str, Any] | None:
    if explicit:
        return dict(explicit)
    candidate = (
        device_identity_path
        or os.getenv("OPENCLAW_DEVICE_IDENTITY_PATH")
        or str(Path.home() / ".openclaw" / "identity" / "device.json")
    )
    try:
        path = Path(candidate)
        if not path.is_file():
            return None
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _public_key_raw_base64url(public_key_pem: str) -> str:
    try:
        from cryptography.hazmat.primitives import serialization
    except ImportError as exc:  # pragma: no cover - depends on target runtime.
        raise RuntimeError("cryptography package is required for OpenClaw device identity signing") from exc
    public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return _base64url(raw)


def _device_id_from_public_key(public_key_base64url: str) -> str:
    padded = public_key_base64url + "=" * ((4 - len(public_key_base64url) % 4) % 4)
    raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    return hashlib.sha256(raw).hexdigest()


def _default_client_platform() -> str:
    system = platform_module.system().strip().lower()
    if system == "windows":
        return "win32"
    if system == "darwin":
        return "darwin"
    if system == "linux":
        return "linux"
    return system


def _normalize_device_auth_metadata(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return "".join(chr(ord(char) + 32) if "A" <= char <= "Z" else char for char in text)


def _default_device_signer(private_key_pem: str, payload: str) -> str:
    try:
        from cryptography.hazmat.primitives import serialization
    except ImportError as exc:  # pragma: no cover - depends on target runtime.
        raise RuntimeError("cryptography package is required for OpenClaw device identity signing") from exc
    private_key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
    signature = private_key.sign(payload.encode("utf-8"))
    return _base64url(signature)


def _device_auth_payload_v3(
    *,
    device_id: str,
    client_id: str,
    client_mode: str,
    role: str,
    scopes: list[str],
    signed_at_ms: int,
    token: str,
    nonce: str,
    platform: str,
    device_family: str,
) -> str:
    return "|".join(
        [
            "v3",
            device_id,
            client_id,
            client_mode,
            role,
            ",".join(scopes),
            str(signed_at_ms),
            token or "",
            nonce,
            _normalize_device_auth_metadata(platform),
            _normalize_device_auth_metadata(device_family),
        ]
    )

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
    normalized = message_type.lower()
    if normalized in {"session.ready", "ready"}:
        return "ready"
    if normalized in {"session.ended", "close", "session.closed"}:
        return "ended"
    if normalized == "turn.started":
        return "listening"
    if normalized == "turn.ended":
        return "ready"
    if normalized in {"audio.delta", "audio", "audio_chunk"}:
        return "speaking"
    if normalized in {"error", "session.error"}:
        return "error"
    return current


def _error_message(value: Any) -> str:
    payload = _mapping(value)
    return str(payload.get("message") or payload.get("kind") or "")
