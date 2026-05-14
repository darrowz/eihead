"""Lightweight HTTP client for the eihead runtime."""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib import request
from urllib.error import HTTPError, URLError


JsonObject = dict[str, Any]


class HeadClientError(RuntimeError):
    """Raised when the eihead HTTP client cannot complete a request."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        url: str,
        status_code: int | None = None,
        trace_id: str | None = None,
        response_body: str = "",
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.url = url
        self.status_code = status_code
        self.trace_id = trace_id
        self.response_body = response_body

    def to_dict(self) -> JsonObject:
        payload: JsonObject = {
            "ok": False,
            "kind": self.kind,
            "message": str(self),
            "url": self.url,
        }
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        if self.trace_id:
            payload["trace_id"] = self.trace_id
        if self.response_body:
            payload["response_body"] = self.response_body
        return payload


class HeadClient:
    """HTTP JSON client for eihead status, capabilities, and actions.

    This class intentionally has no dependency on eibrain runtime config so it
    can be introduced without changing the current honjia execution path.
    """

    def __init__(self, base_url: str, *, timeout: float = 3.0, trace_id: str | None = None) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.trace_id = trace_id

    def get_status(self, *, trace_id: str | None = None) -> JsonObject:
        return self._request_json("GET", "/status", trace_id=trace_id)

    def get_capabilities(self, *, trace_id: str | None = None) -> JsonObject:
        return self._request_json("GET", "/capabilities", trace_id=trace_id)

    def post_action(self, action: Mapping[str, Any], *, trace_id: str | None = None) -> JsonObject:
        if not isinstance(action, Mapping):
            raise TypeError("action must be a mapping")
        effective_trace_id = self._effective_trace_id(trace_id)
        payload: JsonObject = {"action": dict(action)}
        if effective_trace_id:
            payload["trace_id"] = effective_trace_id
        return self._request_json("POST", "/actions", payload=payload, trace_id=trace_id)

    def post_event(self, event: Any, *, trace_id: str | None = None) -> JsonObject:
        event_payload = self._event_to_dict(event)
        effective_trace_id = self._effective_trace_id(trace_id)
        payload: JsonObject = {"event": event_payload}
        if effective_trace_id:
            payload["trace_id"] = effective_trace_id
        return self._request_json("POST", "/events", payload=payload, trace_id=trace_id)

    def speak(self, text: str, *, trace_id: str | None = None, **params: Any) -> JsonObject:
        if not text.strip():
            raise ValueError("text must not be empty")
        action = self._compact({"type": "speak", "text": text, **params})
        return self.post_action(action, trace_id=trace_id)

    def move_head(self, angle: float, *, axis: str = "yaw", trace_id: str | None = None, **params: Any) -> JsonObject:
        action = self._compact({"type": "move_head", "axis": axis, "angle": angle, **params})
        return self.post_action(action, trace_id=trace_id)

    def stop_speech(self, *, trace_id: str | None = None, **params: Any) -> JsonObject:
        action = self._compact({"type": "stop_speech", **params})
        return self.post_action(action, trace_id=trace_id)

    def capture_frame(self, *, trace_id: str | None = None, **params: Any) -> JsonObject:
        action = self._compact({"type": "capture_frame", **params})
        return self.post_action(action, trace_id=trace_id)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: JsonObject | None = None,
        trace_id: str | None = None,
    ) -> JsonObject:
        url = self._url(path)
        effective_trace_id = self._effective_trace_id(trace_id)
        headers = {"Accept": "application/json"}
        data: bytes | None = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if effective_trace_id:
            headers["X-Trace-Id"] = effective_trace_id

        req = request.Request(url, data=data, method=method.upper(), headers=headers)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                status_code = self._status_code(response)
                response_body = self._decode(response.read())
        except HTTPError as exc:
            response_body = self._read_error_body(exc)
            raise HeadClientError(
                f"eihead HTTP error {exc.code} for {url}: {exc.reason}",
                kind="http_error",
                url=url,
                status_code=exc.code,
                trace_id=effective_trace_id,
                response_body=response_body,
            ) from exc
        except TimeoutError as exc:
            raise HeadClientError(
                f"eihead request timed out for {url}",
                kind="timeout",
                url=url,
                trace_id=effective_trace_id,
            ) from exc
        except URLError as exc:
            kind = "timeout" if self._is_timeout_error(exc) else "network_error"
            raise HeadClientError(
                f"eihead {kind} for {url}: {exc.reason}",
                kind=kind,
                url=url,
                trace_id=effective_trace_id,
            ) from exc
        except OSError as exc:
            raise HeadClientError(
                f"eihead network_error for {url}: {exc}",
                kind="network_error",
                url=url,
                trace_id=effective_trace_id,
            ) from exc

        if status_code >= 400:
            raise HeadClientError(
                f"eihead HTTP error {status_code} for {url}",
                kind="http_error",
                url=url,
                status_code=status_code,
                trace_id=effective_trace_id,
                response_body=response_body,
            )
        if not response_body.strip():
            return {}
        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise HeadClientError(
                f"eihead returned invalid JSON for {url}: {exc.msg}",
                kind="invalid_json",
                url=url,
                status_code=status_code,
                trace_id=effective_trace_id,
                response_body=response_body,
            ) from exc
        if not isinstance(decoded, dict):
            raise HeadClientError(
                f"eihead returned non-object JSON for {url}",
                kind="invalid_json",
                url=url,
                status_code=status_code,
                trace_id=effective_trace_id,
                response_body=response_body,
            )
        return decoded

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _effective_trace_id(self, trace_id: str | None) -> str | None:
        return trace_id if trace_id is not None else self.trace_id

    @staticmethod
    def _compact(payload: Mapping[str, Any]) -> JsonObject:
        return {key: value for key, value in payload.items() if value is not None}

    @staticmethod
    def _event_to_dict(event: Any) -> JsonObject:
        to_dict = getattr(event, "to_dict", None)
        if callable(to_dict):
            event = to_dict()
            if not isinstance(event, Mapping):
                raise TypeError("event.to_dict() must return a mapping")
        elif not isinstance(event, Mapping):
            raise TypeError("event must be a mapping")
        return dict(event)

    @staticmethod
    def _decode(raw: bytes) -> str:
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _status_code(response: Any) -> int:
        status = getattr(response, "status", None)
        if status is not None:
            return int(status)
        getcode = getattr(response, "getcode", None)
        if callable(getcode):
            return int(getcode())
        return 200

    @classmethod
    def _read_error_body(cls, exc: HTTPError) -> str:
        try:
            return cls._decode(exc.read())
        except Exception:
            return ""

    @staticmethod
    def _is_timeout_error(exc: URLError) -> bool:
        reason = getattr(exc, "reason", "")
        return isinstance(reason, TimeoutError) or "timed out" in str(reason).lower()


__all__ = ["HeadClient", "HeadClientError"]
