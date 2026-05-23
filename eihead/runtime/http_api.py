"""Standard-library HTTP JSON API for the eihead runtime."""

from __future__ import annotations

import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from eihead.monitoring.eivoice_runtime import build_eivoice_runtime_panel, eivoice_runtime_status_from_app
from eihead.monitoring.realtime_vision import realtime_vision_payload_from_app
from eihead.monitoring.voice import build_voice_diagnostics_from_app


JsonObject = dict[str, Any]
Clock = Callable[[], float]
DEFAULT_MAX_REQUEST_BYTES = 1024 * 1024
UNHEALTHY_STATES = {
    "blocked",
    "degraded",
    "error",
    "failed",
    "not_wired",
    "offline",
    "stale",
    "unavailable",
    "unhealthy",
    "unknown",
}


class HeadHttpApiError(RuntimeError):
    """Structured API error that should be rendered as JSON."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = code
        self.details = dict(details or {})


class HeadHttpServer:
    """Small lifecycle wrapper around ``ThreadingHTTPServer``."""

    def __init__(self, server: ThreadingHTTPServer) -> None:
        self._server = server
        self._serving = False

    @property
    def server_address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    @property
    def httpd(self) -> ThreadingHTTPServer:
        return self._server

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        self._serving = True
        try:
            self._server.serve_forever(poll_interval=poll_interval)
        finally:
            self._serving = False

    def shutdown(self) -> None:
        if self._serving:
            self._server.shutdown()

    def server_close(self) -> None:
        self._server.server_close()

    def close(self) -> None:
        self.shutdown()
        self.server_close()

    def __enter__(self) -> "HeadHttpServer":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class _ThreadingHeadHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def create_handler(
    app: Any,
    *,
    clock: Clock | None = None,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    log_requests: bool = False,
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to an injectable eihead runtime app."""

    _validate_runtime_app(app)
    runtime_app = app
    now = clock or time.time
    max_body_size = int(max_request_bytes)

    if max_body_size <= 0:
        raise ValueError("max_request_bytes must be positive")

    class HeadHttpApiHandler(BaseHTTPRequestHandler):
        server_version = "eihead-runtime/0.1"
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def send_error(
            self,
            code: int,
            message: str | None = None,
            explain: str | None = None,
        ) -> None:
            reason = message or _http_phrase(code)
            self._write_error(int(code), _error_code_for_status(int(code)), reason)

        def log_message(self, format: str, *args: Any) -> None:
            if log_requests:
                super().log_message(format, *args)

        def _dispatch(self, method: str) -> None:
            try:
                status_code, payload = self._route(method)
                self._write_json(status_code, payload)
            except HeadHttpApiError as exc:
                self._write_error(exc.status_code, exc.code, str(exc), details=exc.details)
            except Exception as exc:
                self._write_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "eihead runtime request failed",
                    details={"exception": exc.__class__.__name__},
                )

        def _route(self, method: str) -> tuple[int, JsonObject]:
            path = _normalize_path(self.path)
            if method == "GET":
                if path == "/health":
                    return self._handle_health()
                if path in {"/status", "/status.json", "/api/status"}:
                    return HTTPStatus.OK, self._call_mapping("status")
                if path == "/capabilities":
                    return HTTPStatus.OK, self._call_mapping("capabilities")
                if path in {"/api/vision/realtime", "/api/eye/realtime"}:
                    return HTTPStatus.OK, realtime_vision_payload_from_app(runtime_app, timestamp=now())
                if path in {"/api/voice/realtime", "/api/audio/realtime"}:
                    return HTTPStatus.OK, build_voice_diagnostics_from_app(runtime_app, timestamp=now())
                if path == "/api/eivoice/runtime":
                    return HTTPStatus.OK, _eivoice_runtime_payload(runtime_app)
                if path in {"/api/neck/status", "/api/neck/realtime"}:
                    return HTTPStatus.OK, self._call_mapping("neck_status")
                if path in {"/actions", "/events"}:
                    raise HeadHttpApiError(
                        HTTPStatus.METHOD_NOT_ALLOWED,
                        "method_not_allowed",
                        f"GET is not supported for {path}",
                    )
                raise HeadHttpApiError(HTTPStatus.NOT_FOUND, "not_found", f"unknown path: {path}")

            if method == "POST":
                if path == "/actions":
                    return HTTPStatus.OK, self._handle_action()
                if path == "/events":
                    return HTTPStatus.OK, self._handle_event()
                if path in {
                    "/health",
                    "/status",
                    "/status.json",
                    "/api/status",
                    "/capabilities",
                    "/api/vision/realtime",
                    "/api/eye/realtime",
                    "/api/voice/realtime",
                    "/api/audio/realtime",
                    "/api/eivoice/runtime",
                    "/api/neck/status",
                    "/api/neck/realtime",
                }:
                    raise HeadHttpApiError(
                        HTTPStatus.METHOD_NOT_ALLOWED,
                        "method_not_allowed",
                        f"POST is not supported for {path}",
                    )
                raise HeadHttpApiError(HTTPStatus.NOT_FOUND, "not_found", f"unknown path: {path}")

            raise HeadHttpApiError(
                HTTPStatus.METHOD_NOT_ALLOWED,
                "method_not_allowed",
                f"method is not supported: {method}",
            )

        def _handle_health(self) -> tuple[int, JsonObject]:
            health_fn = getattr(runtime_app, "health", None)
            if callable(health_fn):
                payload = _as_json_object(health_fn(), "app.health()")
            else:
                status_payload = self._call_mapping("status")
                payload = _health_from_status(status_payload, now())
            if not _is_healthy(payload) and "ok" not in payload:
                payload = {**payload, "ok": False}
            return (HTTPStatus.OK if _is_healthy(payload) else HTTPStatus.SERVICE_UNAVAILABLE), payload

        def _handle_action(self) -> JsonObject:
            request_payload = self._read_json_body()
            action = request_payload.get("action")
            if not isinstance(action, Mapping):
                raise HeadHttpApiError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_action",
                    "POST /actions requires object field 'action'",
                )

            trace_id = request_payload.get("trace_id")
            if trace_id is not None and not isinstance(trace_id, str):
                raise HeadHttpApiError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_trace_id",
                    "trace_id must be a string when provided",
                )

            result = runtime_app.handle_action(dict(action), trace_id=trace_id)
            if result is None:
                return {"ok": True, "accepted": True}
            return _as_json_object(result, "app.handle_action()")

        def _handle_event(self) -> JsonObject:
            request_payload = self._read_json_body()
            event = request_payload.get("event")
            if not isinstance(event, Mapping):
                raise HeadHttpApiError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_event",
                    "POST /events requires object field 'event'",
                )

            trace_id = request_payload.get("trace_id")
            if trace_id is not None and not isinstance(trace_id, str):
                raise HeadHttpApiError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_trace_id",
                    "trace_id must be a string when provided",
                )

            handle_event = getattr(runtime_app, "handle_event", None)
            if not callable(handle_event):
                raise HeadHttpApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "event_handler_not_wired",
                    "runtime app does not expose handle_event()",
                    details={
                        "accepted": False,
                        "status": "not_wired",
                        "reason": "runtime_app_handle_event_unavailable",
                        "trace_id": trace_id,
                    },
                )

            result = handle_event(dict(event), trace_id=trace_id)
            return _as_json_object(result, "app.handle_event()")

        def _call_mapping(self, method_name: str) -> JsonObject:
            method = getattr(runtime_app, method_name)
            return _as_json_object(method(), f"app.{method_name}()")

        def _read_json_body(self) -> JsonObject:
            length_header = self.headers.get("Content-Length")
            if not length_header:
                raise HeadHttpApiError(HTTPStatus.BAD_REQUEST, "empty_body", "request body is required")
            try:
                content_length = int(length_header)
            except ValueError as exc:
                raise HeadHttpApiError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_content_length",
                    "Content-Length must be an integer",
                ) from exc

            if content_length <= 0:
                raise HeadHttpApiError(HTTPStatus.BAD_REQUEST, "empty_body", "request body is required")
            if content_length > max_body_size:
                raise HeadHttpApiError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request_too_large",
                    f"request body exceeds {max_body_size} bytes",
                )

            raw = self.rfile.read(content_length)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HeadHttpApiError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_encoding",
                    "request body must be UTF-8 JSON",
                ) from exc

            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise HeadHttpApiError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_json",
                    "request body must be valid JSON",
                    details={"line": exc.lineno, "column": exc.colno},
                ) from exc
            if not isinstance(payload, Mapping):
                raise HeadHttpApiError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_json_object",
                    "request body must be a JSON object",
                    details={"payload_type": type(payload).__name__},
                )
            return dict(payload)

        def _write_json(self, status_code: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(int(status_code))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _write_error(
            self,
            status_code: int,
            code: str,
            message: str,
            *,
            details: Mapping[str, Any] | None = None,
        ) -> None:
            error: JsonObject = {
                "code": code,
                "message": message,
                "status_code": int(status_code),
            }
            if details:
                error["details"] = dict(details)
            self._write_json(int(status_code), {"ok": False, "error": error})

    return HeadHttpApiHandler


def create_server(
    app: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    clock: Clock | None = None,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    log_requests: bool = False,
) -> HeadHttpServer:
    """Create, but do not start, an eihead HTTP API server."""

    handler = create_handler(
        app,
        clock=clock,
        max_request_bytes=max_request_bytes,
        log_requests=log_requests,
    )
    return HeadHttpServer(_ThreadingHeadHttpServer((host, int(port)), handler))


def serve(
    app: Any,
    *,
    host: str = "0.0.0.0",
    port: int = 18081,
    poll_interval: float = 0.5,
    clock: Clock | None = None,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    log_requests: bool = False,
) -> None:
    """Run the HTTP API until ``shutdown()`` or process termination."""

    with create_server(
        app,
        host=host,
        port=port,
        clock=clock,
        max_request_bytes=max_request_bytes,
        log_requests=log_requests,
    ) as server:
        server.serve_forever(poll_interval=poll_interval)


def _validate_runtime_app(app: Any) -> None:
    missing = [name for name in ("status", "capabilities", "handle_action") if not callable(getattr(app, name, None))]
    if missing:
        raise TypeError(f"eihead runtime app is missing required callables: {', '.join(missing)}")


def _normalize_path(raw_path: str) -> str:
    path = urlsplit(raw_path).path or "/"
    if path != "/":
        path = path.rstrip("/")
    return path or "/"


def _as_json_object(payload: Any, source: str) -> JsonObject:
    if isinstance(payload, Mapping):
        return dict(payload)
    raise HeadHttpApiError(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "invalid_runtime_payload",
        f"{source} must return a JSON object",
        details={"payload_type": type(payload).__name__},
    )


def _health_from_status(status_payload: Mapping[str, Any], timestamp: float) -> JsonObject:
    state = str(status_payload.get("status", "ok")).lower()
    ok = status_payload.get("ok") is not False and state not in UNHEALTHY_STATES
    payload: JsonObject = {
        "ok": ok,
        "status": "ok" if ok else state,
        "runtime": status_payload.get("runtime", "eihead"),
        "source": "status",
        "checked_at_ts": timestamp,
    }
    for key in ("node_id", "node_role", "checks", "check_details", "native_providers"):
        if key in status_payload:
            payload[key] = status_payload[key]
    return payload


def _is_healthy(payload: Mapping[str, Any]) -> bool:
    state = str(payload.get("status", "ok")).lower()
    if payload.get("ok") is False:
        return False
    return state not in UNHEALTHY_STATES


def _eivoice_runtime_payload(app: Any) -> JsonObject:
    return {"eivoiceRuntime": build_eivoice_runtime_panel(eivoice_runtime_status_from_app(app))}


def _http_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "HTTP error"


def _error_code_for_status(status_code: int) -> str:
    if status_code == HTTPStatus.NOT_FOUND:
        return "not_found"
    if status_code == HTTPStatus.METHOD_NOT_ALLOWED:
        return "method_not_allowed"
    if status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
        return "request_too_large"
    if status_code == HTTPStatus.BAD_REQUEST:
        return "bad_request"
    if status_code >= 500:
        return "internal_error"
    return "http_error"


__all__ = [
    "DEFAULT_MAX_REQUEST_BYTES",
    "HeadHttpApiError",
    "HeadHttpServer",
    "create_handler",
    "create_server",
    "serve",
]
