"""Minimal eihead-native monitoring Web/API.

This module intentionally uses only the Python standard library so honjia can
serve a small diagnostics surface without depending on ``apps.operator_console``.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import html
import json
import os
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlsplit

from .eivoice_runtime import build_eivoice_runtime_panel, eivoice_runtime_status_from_app
from .neck import build_neck_diagnostics_from_app
from .realtime_vision import realtime_vision_payload_from_app
from .voice import build_voice_diagnostics_from_app
from .voice_test import run_voice_manual_test


JsonObject = dict[str, Any]
Clock = Callable[[], float]
ACTION_LOG_ATTRS = (
    "recent_actions",
    "action_log",
    "actions_log",
    "recent_action_log",
    "execution_log",
)
EVENT_LOG_ATTRS = (
    "recent_events",
)
EVENT_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "runtime",
        "status",
        "wired",
        "source",
        "captured_at_ts",
        "count",
        "events",
        "recent_events",
        "items",
        "actions",
    }
)
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


class EiheadMonitorError(RuntimeError):
    """Structured monitor error rendered as JSON."""

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


class EiheadMonitorServer:
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

    def __enter__(self) -> "EiheadMonitorServer":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class _ThreadingMonitorServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def create_handler(
    app: Any,
    *,
    clock: Clock | None = None,
    log_requests: bool = False,
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to an injectable eihead runtime app."""

    _validate_monitor_app(app)
    runtime_app = app
    now = clock or time.time

    class EiheadMonitorHandler(BaseHTTPRequestHandler):
        server_version = "eihead-monitor/0.1"
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
                if method == "GET":
                    self._route_get()
                    return
                if method == "POST":
                    self._route_post()
                    return
                else:
                    raise EiheadMonitorError(
                        HTTPStatus.METHOD_NOT_ALLOWED,
                        "method_not_allowed",
                        f"{method} is not supported by eihead monitor",
                    )
            except EiheadMonitorError as exc:
                self._write_error(exc.status_code, exc.code, str(exc), details=exc.details)
            except Exception as exc:
                self._write_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "eihead monitor request failed",
                    details={"exception": exc.__class__.__name__},
                )

        def _route_get(self) -> None:
            path = _normalize_path(self.path)
            if path == "/":
                if _env_truthy("EIHEAD_MONITOR_LIGHTWEIGHT_ROOT"):
                    self._write_html(HTTPStatus.OK, _render_lightweight_index(now()))
                    return
                self._write_html(HTTPStatus.OK, _render_index(runtime_app, now()))
                return
            if path in {"/health", "/healthz"}:
                status_code, payload = _health_payload(runtime_app, now())
                self._write_json(status_code, payload)
                return
            if path in {"/api/status", "/status.json"}:
                self._write_json(HTTPStatus.OK, _call_json_object(runtime_app, "status"))
                return
            if path == "/api/capabilities":
                self._write_json(HTTPStatus.OK, _call_json_object(runtime_app, "capabilities"))
                return
            if path in {"/api/vision/realtime", "/api/eye/realtime"}:
                self._write_json(
                    HTTPStatus.OK,
                    realtime_vision_payload_from_app(runtime_app, timestamp=now()),
                )
                return
            if path in {"/api/voice/realtime", "/api/audio/realtime"}:
                proxied = _runtime_proxy_payload(path)
                if proxied is not None:
                    self._write_json(HTTPStatus.OK, proxied)
                    return
                self._write_json(
                    HTTPStatus.OK,
                    build_voice_diagnostics_from_app(runtime_app, timestamp=now()),
                )
                return
            if path == "/api/eivoice/runtime":
                proxied = _runtime_proxy_payload(path)
                self._write_json(HTTPStatus.OK, proxied if proxied is not None else _eivoice_runtime_payload(runtime_app))
                return
            if path in {"/api/neck/status", "/api/neck/realtime"}:
                proxied = _runtime_proxy_payload(path)
                if proxied is not None:
                    self._write_json(
                        HTTPStatus.OK,
                        _neck_diagnostics_from_runtime_proxy(proxied, timestamp=now()),
                    )
                    return
                self._write_json(
                    HTTPStatus.OK,
                    build_neck_diagnostics_from_app(runtime_app, timestamp=now()),
                )
                return
            if path in {"/api/actions/recent", "/api/recent-actions"}:
                self._write_json(HTTPStatus.OK, _recent_actions_payload(runtime_app, now()))
                return
            if path in {"/api/events/recent", "/api/recent-events"}:
                self._write_json(HTTPStatus.OK, _recent_events_payload(runtime_app, now()))
                return
            raise EiheadMonitorError(HTTPStatus.NOT_FOUND, "not_found", f"unknown path: {path}")

        def _route_post(self) -> None:
            path = _normalize_path(self.path)
            if path == "/api/voice/test":
                payload = self._read_json_body()
                self._write_json(
                    HTTPStatus.OK,
                    run_voice_manual_test(runtime_app, payload, timestamp=now()),
                )
                return
            raise EiheadMonitorError(HTTPStatus.NOT_FOUND, "not_found", f"unknown path: {path}")

        def _read_json_body(self) -> JsonObject:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise EiheadMonitorError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_content_length",
                    "Content-Length must be an integer",
                ) from exc
            if length <= 0:
                return {}
            if length > 65536:
                raise EiheadMonitorError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "request_too_large",
                    "request body is too large",
                )
            raw_body = self.rfile.read(length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EiheadMonitorError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_json",
                    "request body must be a JSON object",
                ) from exc
            if not isinstance(payload, Mapping):
                raise EiheadMonitorError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_json",
                    "request body must be a JSON object",
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

        def _write_html(self, status_code: int, body_text: str) -> None:
            body = body_text.encode("utf-8")
            self.send_response(int(status_code))
            self.send_header("Content-Type", "text/html; charset=utf-8")
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

    return EiheadMonitorHandler


def create_server(
    app: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    clock: Clock | None = None,
    log_requests: bool = False,
) -> EiheadMonitorServer:
    """Create, but do not start, an eihead native monitor server."""

    handler = create_handler(app, clock=clock, log_requests=log_requests)
    return EiheadMonitorServer(_ThreadingMonitorServer((host, int(port)), handler))


def serve(
    app: Any,
    *,
    host: str = "0.0.0.0",
    port: int = 18080,
    poll_interval: float = 0.5,
    clock: Clock | None = None,
    log_requests: bool = False,
) -> None:
    """Run the native monitor until ``shutdown()`` or process termination."""

    with create_server(app, host=host, port=port, clock=clock, log_requests=log_requests) as server:
        server.serve_forever(poll_interval=poll_interval)


def _validate_monitor_app(app: Any) -> None:
    missing = [name for name in ("status", "capabilities") if not callable(getattr(app, name, None))]
    if missing:
        raise TypeError(f"eihead monitor app is missing required callables: {', '.join(missing)}")


def _normalize_path(raw_path: str) -> str:
    path = urlsplit(raw_path).path or "/"
    if path != "/":
        path = path.rstrip("/")
    return path or "/"


def _call_json_object(app: Any, method_name: str) -> JsonObject:
    method = getattr(app, method_name)
    payload = method()
    if isinstance(payload, Mapping):
        return dict(payload)
    raise EiheadMonitorError(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "invalid_runtime_payload",
        f"app.{method_name}() must return a JSON object",
        details={"payload_type": type(payload).__name__},
    )


def _health_payload(app: Any, timestamp: float) -> tuple[int, JsonObject]:
    health_fn = getattr(app, "health", None)
    if callable(health_fn):
        payload = health_fn()
        if not isinstance(payload, Mapping):
            raise EiheadMonitorError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "invalid_runtime_payload",
                "app.health() must return a JSON object",
                details={"payload_type": type(payload).__name__},
            )
        health = dict(payload)
    else:
        status_payload = _call_json_object(app, "status")
        state = str(status_payload.get("status", status_payload.get("overall_status", "ok"))).lower()
        ok = status_payload.get("ok") is not False and state not in UNHEALTHY_STATES
        health = {
            "ok": ok,
            "status": "ok" if ok else state,
            "runtime": status_payload.get("runtime", "eihead"),
            "node_id": status_payload.get("node_id", "honjia"),
            "source": "status",
            "checked_at_ts": timestamp,
        }
    return (HTTPStatus.OK if _is_healthy(health) else HTTPStatus.SERVICE_UNAVAILABLE), health


def _recent_actions_payload(app: Any, timestamp: float) -> JsonObject:
    for attr_name in ACTION_LOG_ATTRS:
        if not hasattr(app, attr_name):
            continue
        source = getattr(app, attr_name)
        try:
            raw_log = source() if callable(source) else source
        except Exception as exc:
            raise EiheadMonitorError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "recent_actions_failed",
                f"failed to read action log from app.{attr_name}",
                details={"exception": exc.__class__.__name__, "source": attr_name},
            ) from exc

        actions, extra = _coerce_action_log(raw_log)
        return {
            "schema": "eihead.monitor.recent_actions.v1",
            "runtime": "eihead",
            "status": "wired",
            "wired": True,
            "source": attr_name,
            "captured_at_ts": timestamp,
            "count": len(actions),
            "actions": actions,
            **extra,
        }

    return {
        "schema": "eihead.monitor.recent_actions.v1",
        "runtime": "eihead",
        "status": "not_wired",
        "wired": False,
        "source": None,
        "captured_at_ts": timestamp,
        "count": 0,
        "actions": [],
        "message": "runtime app does not expose recent_actions or action_log",
    }


def _recent_events_payload(app: Any, timestamp: float) -> JsonObject:
    for attr_name in EVENT_LOG_ATTRS:
        if not hasattr(app, attr_name):
            continue
        source = getattr(app, attr_name)
        try:
            raw_log = source() if callable(source) else source
        except Exception as exc:
            raise EiheadMonitorError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "recent_events_failed",
                f"failed to read event log from app.{attr_name}",
                details={"exception": exc.__class__.__name__, "source": attr_name},
            ) from exc

        events, extra = _coerce_event_log(raw_log)
        return {
            "schema": "eihead.monitor.recent_events.v1",
            "runtime": "eihead",
            "status": "wired",
            "wired": True,
            "source": attr_name,
            "captured_at_ts": timestamp,
            "count": len(events),
            "events": events,
            **extra,
        }

    event_journal = getattr(app, "event_journal", None)
    recent = getattr(event_journal, "recent", None)
    if callable(recent):
        try:
            raw_log = recent()
        except Exception as exc:
            raise EiheadMonitorError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "recent_events_failed",
                "failed to read event log from app.event_journal.recent",
                details={"exception": exc.__class__.__name__, "source": "event_journal.recent"},
            ) from exc

        events, extra = _coerce_event_log(raw_log)
        return {
            "schema": "eihead.monitor.recent_events.v1",
            "runtime": "eihead",
            "status": "wired",
            "wired": True,
            "source": "event_journal.recent",
            "captured_at_ts": timestamp,
            "count": len(events),
            "events": events,
            **extra,
        }

    return {
        "schema": "eihead.monitor.recent_events.v1",
        "runtime": "eihead",
        "status": "not_wired",
        "wired": False,
        "source": None,
        "captured_at_ts": timestamp,
        "count": 0,
        "events": [],
        "message": "runtime app does not expose recent_events or event_journal.recent",
    }


def _eivoice_runtime_payload(app: Any) -> JsonObject:
    return {"eivoiceRuntime": build_eivoice_runtime_panel(_eivoice_runtime_status_from_app(app))}


def _runtime_proxy_payload(path: str) -> JsonObject | None:
    if not _runtime_proxy_enabled(path):
        return None
    base_url = os.environ.get("EIHEAD_RUNTIME_URL", "").strip().rstrip("/")
    if not base_url:
        return None
    timeout_s = _env_float("EIHEAD_MONITOR_PROXY_TIMEOUT_S", 1.0)
    url = f"{base_url}{path}"
    try:
        with urlrequest.urlopen(url, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except (OSError, TimeoutError, urlerror.URLError):
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, Mapping):
        return dict(payload)
    return None


def _runtime_proxy_enabled(path: str) -> bool:
    if path in {"/api/neck/status", "/api/neck/realtime"}:
        return _env_truthy("EIHEAD_MONITOR_PROXY_RUNTIME_NECK")
    return _env_truthy("EIHEAD_MONITOR_PROXY_RUNTIME_VOICE")


def _neck_diagnostics_from_runtime_proxy(payload: Mapping[str, Any], *, timestamp: float) -> JsonObject:
    class RuntimeProxyNeckApp:
        def neck_status(self) -> Mapping[str, Any]:
            return payload

        def neck_realtime(self) -> Mapping[str, Any]:
            return payload

    return build_neck_diagnostics_from_app(RuntimeProxyNeckApp(), timestamp=timestamp)


def _eivoice_runtime_status_from_app(app: Any) -> JsonObject:
    return eivoice_runtime_status_from_app(app)


def _coerce_action_log(raw_log: Any) -> tuple[list[JsonObject], JsonObject]:
    extra: JsonObject = {}
    if raw_log is None:
        return [], extra

    if isinstance(raw_log, Mapping):
        for key in ("actions", "recent_actions", "items", "events"):
            if key in raw_log:
                actions = _coerce_action_items(raw_log[key])
                extra = {str(k): _json_ready(v) for k, v in raw_log.items() if k != key}
                return actions, extra
        return [_serialize_item(raw_log)], extra

    return _coerce_action_items(raw_log), extra


def _coerce_event_log(raw_log: Any) -> tuple[list[JsonObject], JsonObject]:
    extra: JsonObject = {}
    if raw_log is None:
        return [], extra

    if isinstance(raw_log, Mapping):
        for key in ("events", "recent_events", "items", "actions"):
            if key in raw_log:
                events = _coerce_action_items(raw_log[key])
                extra = {str(k): _json_ready(v) for k, v in raw_log.items() if str(k) not in EVENT_PAYLOAD_KEYS}
                return events, extra
        return [_serialize_item(raw_log)], extra

    return _coerce_action_items(raw_log), extra


def _coerce_action_items(items: Any) -> list[JsonObject]:
    if items is None:
        return []
    if isinstance(items, (str, bytes)) or isinstance(items, Mapping):
        return [_serialize_item(items)]
    try:
        iterator = iter(items)
    except TypeError:
        return [_serialize_item(items)]
    return [_serialize_item(item) for item in iterator]


def _serialize_item(item: Any) -> JsonObject:
    if isinstance(item, Mapping):
        return {str(k): _json_ready(v) for k, v in item.items()}
    if hasattr(item, "to_dict") and callable(item.to_dict):
        payload = item.to_dict()
        if isinstance(payload, Mapping):
            return {str(k): _json_ready(v) for k, v in payload.items()}
    if is_dataclass(item):
        return {str(k): _json_ready(v) for k, v in asdict(item).items()}
    return {"value": _json_ready(item), "payload_type": type(item).__name__}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        return _json_ready(asdict(value))
    return str(value)


def _render_index(app: Any, timestamp: float) -> str:
    status = _safe_payload(lambda: _call_json_object(app, "status"))
    capabilities = _safe_payload(lambda: _call_json_object(app, "capabilities"))
    realtime = _safe_payload(lambda: realtime_vision_payload_from_app(app, timestamp=timestamp))
    voice = _safe_payload(lambda: build_voice_diagnostics_from_app(app, timestamp=timestamp))
    eivoice_runtime = _safe_payload(lambda: _eivoice_runtime_payload(app))
    neck = _safe_payload(lambda: build_neck_diagnostics_from_app(app, timestamp=timestamp))
    recent = _safe_payload(lambda: _recent_actions_payload(app, timestamp))
    recent_events = _safe_payload(lambda: _recent_events_payload(app, timestamp))

    node_id = _display_value(status.get("node_id") or capabilities.get("node_id") or "honjia")
    overall = _display_value(
        status.get("overall_status")
        or status.get("status")
        or capabilities.get("overall_status")
        or "unknown"
    )
    realtime_state = _display_value(realtime.get("status", "unknown"))
    voice_state = _display_value(voice.get("status", "unknown"))
    eivoice_panel = (
        eivoice_runtime.get("eivoiceRuntime")
        if isinstance(eivoice_runtime.get("eivoiceRuntime"), Mapping)
        else {}
    )
    eivoice_runtime_state = _display_value(eivoice_panel.get("state", "unknown"))
    eivoice_runtime_health = _display_value(eivoice_panel.get("health", "unknown"))
    neck_state = _display_value(neck.get("status", "unknown"))
    recent_state = _display_value(recent.get("status", "unknown"))
    recent_events_state = _display_value(recent_events.get("status", "unknown"))
    realtime_diagnostic = realtime.get("diagnostic") if isinstance(realtime.get("diagnostic"), Mapping) else {}
    visual_overlay = realtime.get("overlay")
    if not isinstance(visual_overlay, Mapping):
        visual_overlay = realtime.get("visual_diagnostic")
    if not isinstance(visual_overlay, Mapping):
        visual_overlay = {}
    vision_status = _display_value(realtime_diagnostic.get("status") or realtime.get("status", "unknown"))
    vision_fps = _display_value(_metric_value(realtime_diagnostic.get("fps")))
    vision_top_detection = _display_value(_top_detection_summary(realtime_diagnostic.get("top_detection")))
    vision_frame_age = _display_value(_metric_value(realtime_diagnostic.get("last_frame_age"), suffix="s"))
    vision_backend = _display_value(realtime_diagnostic.get("backend") or "unknown")
    vision_frame_interval = _display_value(_metric_value(realtime.get("frame_interval_ms"), suffix="ms"))
    vision_jitter_guard = _display_value(realtime.get("jitter_guard") if realtime.get("jitter_guard") is not None else "unknown")
    vision_top_k = _display_value(_metric_value(realtime.get("top_k")))
    vision_score_threshold = _display_value(_metric_value(realtime.get("score_threshold")))
    vision_hooks_used = _display_value(_hooks_used_summary(realtime.get("hooks_used")))
    vision_pipeline = _display_value(_pipeline_summary(realtime.get("pipeline")))
    vision_devices = _display_value(_devices_summary(realtime.get("devices")))
    vision_readiness = _display_value(realtime.get("readiness_message") or "unknown")
    vision_parse_errors = _display_value(_metric_value(realtime.get("parse_error_count")))
    vision_overlay_frame = _display_value(_overlay_frame_summary(visual_overlay.get("frame")))
    vision_overlay_image = _display_value(_overlay_image_message(visual_overlay.get("frame")))
    vision_overlay_boxes = _display_value(_overlay_boxes_summary(visual_overlay.get("normalized_boxes")))
    vision_overlay_scores = _display_value(_overlay_scores_summary(visual_overlay.get("score_labels")))
    vision_overlay_top_target = _display_value(_overlay_top_target_summary(visual_overlay.get("top_target")))
    vision_overlay_stream_ready = _display_value(
        visual_overlay.get("stream_ready")
        if visual_overlay.get("stream_ready") is not None
        else realtime.get("stream_ready", "unknown")
    )
    vision_scene_id = _display_value(realtime.get("scene_id") or "unknown")
    vision_scene_summary = _display_value(realtime.get("scene_summary") or "none")
    vision_event_count = _display_value(_metric_value(realtime.get("event_count")))
    vision_event_summary = _display_value(realtime.get("event_summary") or "none")
    vision_track_count = _display_value(_metric_value(realtime.get("track_count")))
    vision_track_summary = _display_value(realtime.get("track_summary") or "none")
    vision_target_center = _display_value(_point_summary(realtime.get("target_center")))
    vision_target_error = _display_value(_point_summary(realtime.get("target_error")))
    vision_target_score_label = _display_value(realtime.get("target_score_label") or "none")
    vision_score_labels = _display_value(_overlay_scores_summary(realtime.get("score_labels")))
    voice_ear = _display_value(_voice_component_summary(voice.get("ear"), kind="ear"))
    voice_mouth = _display_value(_voice_component_summary(voice.get("mouth"), kind="mouth"))
    voice_dialogue = _display_value(_voice_dialogue_summary(voice.get("dialogue")))
    voice_latency = _display_value(_metric_value(_voice_latency_total_ms(voice.get("latency")), suffix="ms"))
    voice_bottleneck = _display_value(_voice_bottleneck_summary(voice.get("bottleneck")))
    voice_last_turn = _display_value(_voice_last_turn_summary(voice.get("last_turn")))
    voice_round = _display_value(_voice_round_summary(voice.get("round")))
    voice_scheduler = _display_value(_voice_scheduler_summary(voice.get("scheduler")))
    voice_fast_think = _display_value(_voice_realtime_component_summary(voice.get("fast_think")))
    voice_slow_reasoner = _display_value(_voice_realtime_component_summary(voice.get("slow_reasoner")))
    voice_arbiter = _display_value(_voice_realtime_component_summary(voice.get("arbiter")))
    voice_speech_action_plan = _display_value(_voice_realtime_component_summary(voice.get("speech_action_plan")))
    voice_proactive_activity = _display_value(_voice_realtime_component_summary(voice.get("proactive_activity")))
    voice_interruption = _display_value(_voice_interruption_summary(voice.get("interruption")))
    voice_microfeedback = _display_value(_voice_microfeedback_summary(voice.get("microfeedback")))
    voice_closed_loop = _display_value(_voice_closed_loop_summary(voice.get("closed_loop_state")))
    voice_realtime_audio = _display_value(_voice_realtime_audio_summary(voice.get("realtime_audio")))
    voice_event_count = _display_value(_metric_value(voice.get("event_count")))
    voice_last_reply_delta = _display_value(voice.get("last_reply_delta") or "unknown")
    voice_first_reply_token = _display_value(
        _metric_value(_voice_latency_stage_ms(voice.get("latency"), "first_reply_token"), suffix="ms")
    )
    voice_first_speech = _display_value(
        _metric_value(_voice_latency_stage_ms(voice.get("latency"), "first_speech"), suffix="ms")
    )
    voice_cancellation_chain = _display_value(_voice_cancellation_chain_summary(voice.get("cancellation_chain")))
    voice_chain_readiness = _display_value(_voice_chain_readiness_summary(voice.get("voice_chain_readiness")))
    voice_chain_bottleneck = _display_value(_voice_chain_bottleneck_summary(voice.get("voice_chain_readiness")))
    voice_readiness = _display_value(voice.get("readiness_message") or "unknown")
    voice_heard_text = _display_value(_voice_heard_text(voice))
    voice_reply_text = _display_value(_voice_reply_text(voice))
    voice_dialogue_engine = _display_value(_voice_dialogue_engine_summary(voice))
    voice_protocol_event = _display_value(_voice_protocol_event_summary(voice))
    voice_latency_breakdown = _display_value(_voice_latency_breakdown_summary(voice))
    voice_optimization = _display_value(_voice_optimization_summary(voice.get("optimization")))
    voice_tts_playback = _display_value(_voice_tts_playback_summary(voice))
    voice_mic_vad = _display_value(_voice_mic_vad_summary(voice))
    voice_asr_detail = _display_value(_voice_asr_detail_summary(voice))
    voice_chain_state = _display_value(_voice_chain_state_summary(voice.get("voice_chain")))
    voice_chain_steps = _display_value(_voice_chain_steps_summary(voice.get("voice_chain")))
    voice_chain_asr = _display_value(_voice_chain_text(voice.get("voice_chain"), "last_asr_text"))
    voice_chain_tts = _display_value(_voice_chain_text(voice.get("voice_chain"), "last_tts_text"))
    voice_openclaw_ws = _display_value(_openclaw_ws_summary(voice.get("openclaw_ws")))
    voice_openclaw_error = _display_value(_openclaw_ws_error_summary(voice.get("openclaw_ws")))
    voice_playback_gate = _display_value(_voice_playback_gate_summary(voice))
    eivoice_conversation = _display_value(eivoice_panel.get("conversationState", "unknown"))
    eivoice_dropped_total = _display_value(_metric_value(eivoice_panel.get("droppedTotal")))
    eivoice_queue_summary = eivoice_panel.get("queueSummary") if isinstance(eivoice_panel.get("queueSummary"), Mapping) else {}
    eivoice_queue_fill = _display_value(_metric_value(eivoice_queue_summary.get("maxFillRatio")))
    eivoice_warnings = eivoice_panel.get("warnings") if isinstance(eivoice_panel.get("warnings"), list) else []
    eivoice_warning_text = _display_value(", ".join(str(item) for item in eivoice_warnings) or "none")
    eivoice_transport = eivoice_panel.get("transport") if isinstance(eivoice_panel.get("transport"), Mapping) else {}
    eivoice_transport_state = _display_value(
        f"{eivoice_transport.get('name', 'unknown')} / {eivoice_transport.get('state', 'unknown')}"
    )
    eivoice_transport_heartbeat = _display_value(
        _metric_value(
            _first_mapping_value(eivoice_transport, "heartbeat").get("latency_ms"),
            suffix="ms",
        )
    )
    eivoice_transport_reconnect = _display_value(
        _metric_value(_first_mapping_value(eivoice_transport, "reconnect").get("attempt"))
    )
    eivoice_openclaw_ws = _display_value(_openclaw_ws_summary(eivoice_panel.get("openclawWs")))
    eivoice_local_vad = _display_value(_eivoice_local_vad_summary(eivoice_panel))
    eivoice_local_wake_gate = _display_value(_eivoice_local_wake_gate_summary(eivoice_panel))
    neck_current_angle = _display_value(_metric_value(neck.get("current_angle"), suffix="deg"))
    neck_target_angle = _display_value(_metric_value(neck.get("target_angle"), suffix="deg"))
    neck_will_move = _display_value(neck.get("will_move") if neck.get("will_move") is not None else "unknown")
    neck_suppressed = _display_value(neck.get("suppressed") if neck.get("suppressed") is not None else "unknown")
    neck_suppression_reason = _display_value(neck.get("suppression_reason") or "unknown")
    neck_servo = _display_value(_neck_servo_summary(neck.get("servo")))
    neck_axis_support = _display_value(_neck_axis_support_summary(neck.get("axis_support")))
    neck_motion_evidence = _display_value(_neck_motion_evidence_summary(neck.get("motion_evidence")))
    neck_readiness = _display_value(neck.get("readiness_message") or "unknown")

    status_json = _json_for_html(status)
    capabilities_json = _json_for_html(capabilities)
    realtime_json = _json_for_html(realtime)
    voice_json = _json_for_html(voice)
    eivoice_runtime_json = _json_for_html(eivoice_runtime)
    neck_json = _json_for_html(neck)
    recent_json = _json_for_html(recent)
    recent_events_json = _json_for_html(recent_events)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>eihead native monitor</title>
  <style>
    body {{ margin: 0; font: 15px/1.5 sans-serif; background: #f7f3ea; color: #17201a; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 28px 20px 42px; }}
    header {{ border-bottom: 3px solid #5f8f7a; margin-bottom: 20px; }}
    h1 {{ margin: 0 0 6px; font-size: 32px; }}
    .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .card {{ background: #fffaf0; border: 1px solid #d7cbb3; border-radius: 14px; padding: 16px; }}
    .label {{ color: #5c6b61; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; }}
    .metric {{ display: block; margin-top: 4px; font-size: 22px; font-weight: 700; }}
    .detail {{ display: block; margin-top: 8px; overflow-wrap: anywhere; }}
    code, pre {{ background: #10231a; color: #d9f3df; border-radius: 10px; }}
    code {{ padding: 1px 5px; }}
    pre {{ overflow: auto; padding: 14px; }}
    a {{ color: #315f4c; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>eihead native monitor</h1>
      <p>node <strong>{node_id}</strong> · status <strong>{overall}</strong> · realtime vision <strong>{realtime_state}</strong> · voice <strong>{voice_state}</strong> · eivoice runtime <strong>{eivoice_runtime_health}</strong> · neck <strong>{neck_state}</strong> · actions <strong>{recent_state}</strong> · events <strong>{recent_events_state}</strong></p>
    </header>
    <section class="grid">
      <div class="card"><div class="label">Status API</div><a href="/api/status">/api/status</a></div>
      <div class="card"><div class="label">Capabilities API</div><a href="/api/capabilities">/api/capabilities</a></div>
      <div class="card"><div class="label">Realtime Vision API</div><a href="/api/vision/realtime">/api/vision/realtime</a></div>
      <div class="card"><div class="label">Voice Diagnostics API</div><a href="/api/voice/realtime">/api/voice/realtime</a></div>
      <div class="card"><div class="label">EIVoice Runtime API</div><a href="/api/eivoice/runtime">/api/eivoice/runtime</a></div>
      <div class="card"><div class="label">Neck Diagnostics API</div><a href="/api/neck/status">/api/neck/status</a></div>
      <div class="card"><div class="label">Recent Actions API</div><a href="/api/actions/recent">/api/actions/recent</a></div>
      <div class="card"><div class="label">Recent Events API</div><a href="/api/events/recent">/api/events/recent</a></div>
      <div class="card"><div class="label">Health API</div><a href="/health">/health</a></div>
    </section>
    <h2>Realtime Vision Diagnostic</h2>
    <section class="grid">
      <div class="card"><div class="label">Status</div><span class="metric">{vision_status}</span></div>
      <div class="card"><div class="label">FPS</div><span class="metric">{vision_fps}</span></div>
      <div class="card"><div class="label">Top detection</div><span class="metric">{vision_top_detection}</span></div>
      <div class="card"><div class="label">Frame age</div><span class="metric">{vision_frame_age}</span></div>
      <div class="card"><div class="label">Backend</div><span class="metric">{vision_backend}</span></div>
      <div class="card"><div class="label">Frame interval</div><span class="metric">{vision_frame_interval}</span></div>
      <div class="card"><div class="label">Jitter guard</div><span class="metric">{vision_jitter_guard}</span></div>
      <div class="card"><div class="label">Top K</div><span class="metric">{vision_top_k}</span></div>
      <div class="card"><div class="label">Score threshold</div><span class="metric">{vision_score_threshold}</span></div>
      <div class="card"><div class="label">Hooks used</div><span class="metric">{vision_hooks_used}</span></div>
      <div class="card"><div class="label">Pipeline</div><span class="metric">{vision_pipeline}</span></div>
      <div class="card"><div class="label">Devices</div><span class="metric">{vision_devices}</span></div>
      <div class="card"><div class="label">Readiness</div><span class="metric">{vision_readiness}</span></div>
      <div class="card"><div class="label">Parse errors</div><span class="metric">{vision_parse_errors}</span></div>
      <div class="card"><div class="label">Scene</div><span class="metric">{vision_scene_id}</span></div>
      <div class="card"><div class="label">Scene summary</div><span class="metric">{vision_scene_summary}</span></div>
      <div class="card"><div class="label">Event count</div><span class="metric">{vision_event_count}</span></div>
      <div class="card"><div class="label">Event summary</div><span class="metric">{vision_event_summary}</span></div>
      <div class="card"><div class="label">Track count</div><span class="metric">{vision_track_count}</span></div>
      <div class="card"><div class="label">Track summary</div><span class="metric">{vision_track_summary}</span></div>
      <div class="card"><div class="label">Target center</div><span class="metric">{vision_target_center}</span></div>
      <div class="card"><div class="label">Target error</div><span class="metric">{vision_target_error}</span></div>
      <div class="card"><div class="label">Target score</div><span class="metric">{vision_target_score_label}</span></div>
      <div class="card"><div class="label">Score labels</div><span class="metric">{vision_score_labels}</span></div>
    </section>
    <p>Realtime JSON below includes <code>boxes</code> and <code>scores</code> for direct visual diagnostics.</p>
    <h2>Visual Overlay</h2>
    <section class="grid">
      <div class="card"><div class="label">Frame size</div><span class="metric">{vision_overlay_frame}</span></div>
      <div class="card"><div class="label">Frame image</div><span class="metric">{vision_overlay_image}</span></div>
      <div class="card"><div class="label">Normalized boxes</div><span class="metric">{vision_overlay_boxes}</span></div>
      <div class="card"><div class="label">Score labels</div><span class="metric">{vision_overlay_scores}</span></div>
      <div class="card"><div class="label">Top target</div><span class="metric">{vision_overlay_top_target}</span></div>
      <div class="card"><div class="label">Stream readiness</div><span class="metric">{vision_overlay_stream_ready}</span></div>
    </section>
    <h2>Neck Diagnostics</h2>
    <section class="grid">
      <div class="card"><div class="label">Status</div><span class="metric">{neck_state}</span></div>
      <div class="card"><div class="label">Current angle</div><span class="metric">{neck_current_angle}</span></div>
      <div class="card"><div class="label">Target angle</div><span class="metric">{neck_target_angle}</span></div>
      <div class="card"><div class="label">Current action will move</div><span class="metric">{neck_will_move}</span></div>
      <div class="card"><div class="label">Motion evidence</div><span class="metric">{neck_motion_evidence}</span></div>
      <div class="card"><div class="label">Suppressed</div><span class="metric">{neck_suppressed}</span></div>
      <div class="card"><div class="label">Suppression reason</div><span class="metric">{neck_suppression_reason}</span></div>
      <div class="card"><div class="label">Servo</div><span class="metric">{neck_servo}</span></div>
      <div class="card"><div class="label">Axis support</div><span class="metric">{neck_axis_support}</span></div>
      <div class="card"><div class="label">Readiness</div><span class="metric">{neck_readiness}</span></div>
    </section>
    <h2>Voice Diagnostics</h2>
    <section class="grid">
      <div class="card"><div class="label">Status</div><span class="metric">{voice_state}</span></div>
      <div class="card hot"><div class="label">听到内容</div><span class="metric">{voice_heard_text}</span></div>
      <div class="card hot"><div class="label">回答内容</div><span class="metric">{voice_reply_text}</span></div>
      <div class="card hot"><div class="label">语音链路明细</div><span class="metric">链路状态：{voice_chain_state}</span><span class="detail">步骤：{voice_chain_steps}</span><span class="detail">最后 ASR：{voice_chain_asr}</span><span class="detail">最后 TTS：{voice_chain_tts}</span></div>
      <div class="card"><div class="label">对话引擎</div><span class="metric">{voice_dialogue_engine}</span></div>
      <div class="card"><div class="label">协议事件</div><span class="metric">{voice_protocol_event}</span></div>
      <div class="card"><div class="label">耗时拆分</div><span class="metric">{voice_latency_breakdown}</span></div>
      <div class="card hot"><div class="label">性能优化</div><span class="metric">{voice_optimization}</span></div>
      <div class="card"><div class="label">TTS 播放</div><span class="metric">{voice_tts_playback}</span></div>
      <div class="card"><div class="label">麦克风/VAD</div><span class="metric">{voice_mic_vad}</span></div>
      <div class="card"><div class="label">ASR 识别</div><span class="metric">{voice_asr_detail}</span></div>
      <div class="card"><div class="label">OpenClaw WS</div><span class="metric">{voice_openclaw_ws}</span></div>
      <div class="card"><div class="label">OpenClaw 错误</div><span class="metric">{voice_openclaw_error}</span></div>
      <div class="card"><div class="label">回声门控</div><span class="metric">{voice_playback_gate}</span></div>
      <div class="card"><div class="label">Ear</div><span class="metric">{voice_ear}</span></div>
      <div class="card"><div class="label">Mouth</div><span class="metric">{voice_mouth}</span></div>
      <div class="card"><div class="label">Dialogue</div><span class="metric">{voice_dialogue}</span></div>
      <div class="card"><div class="label">Latency</div><span class="metric">{voice_latency}</span></div>
      <div class="card"><div class="label">Bottleneck</div><span class="metric">{voice_bottleneck}</span></div>
      <div class="card"><div class="label">Last turn</div><span class="metric">{voice_last_turn}</span></div>
      <div class="card"><div class="label">Round</div><span class="metric">{voice_round}</span></div>
      <div class="card"><div class="label">Scheduler</div><span class="metric">{voice_scheduler}</span></div>
      <div class="card"><div class="label">Fast think</div><span class="metric">{voice_fast_think}</span></div>
      <div class="card"><div class="label">Slow reasoner</div><span class="metric">{voice_slow_reasoner}</span></div>
      <div class="card"><div class="label">Arbiter</div><span class="metric">{voice_arbiter}</span></div>
      <div class="card"><div class="label">Speech/action plan</div><span class="metric">{voice_speech_action_plan}</span></div>
      <div class="card"><div class="label">Proactive activity</div><span class="metric">{voice_proactive_activity}</span></div>
      <div class="card"><div class="label">Interrupts</div><span class="metric">{voice_interruption}</span></div>
      <div class="card"><div class="label">Microfeedback</div><span class="metric">{voice_microfeedback}</span></div>
      <div class="card"><div class="label">Closed loop</div><span class="metric">{voice_closed_loop}</span></div>
      <div class="card"><div class="label">Realtime audio</div><span class="metric">{voice_realtime_audio}</span></div>
      <div class="card"><div class="label">Realtime events</div><span class="metric">{voice_event_count}</span></div>
      <div class="card"><div class="label">Last reply delta</div><span class="metric">{voice_last_reply_delta}</span></div>
      <div class="card"><div class="label">First reply token</div><span class="metric">{voice_first_reply_token}</span></div>
      <div class="card"><div class="label">First speech</div><span class="metric">{voice_first_speech}</span></div>
      <div class="card"><div class="label">Cancellation chain</div><span class="metric">{voice_cancellation_chain}</span></div>
      <div class="card"><div class="label">Voice chain readiness</div><span class="metric">{voice_chain_readiness}</span></div>
      <div class="card"><div class="label">Voice chain bottleneck</div><span class="metric">{voice_chain_bottleneck}</span></div>
      <div class="card"><div class="label">Readiness</div><span class="metric">{voice_readiness}</span></div>
    </section>
    <h2>EIVoice Runtime</h2>
    <section class="grid">
      <div class="card"><div class="label">Health</div><span class="metric">{eivoice_runtime_health}</span></div>
      <div class="card"><div class="label">State</div><span class="metric">{eivoice_runtime_state}</span></div>
      <div class="card"><div class="label">Conversation</div><span class="metric">{eivoice_conversation}</span></div>
      <div class="card"><div class="label">Dropped total</div><span class="metric">{eivoice_dropped_total}</span></div>
      <div class="card"><div class="label">Max queue fill</div><span class="metric">{eivoice_queue_fill}</span></div>
      <div class="card"><div class="label">Transport state</div><span class="metric">{eivoice_transport_state}</span></div>
      <div class="card"><div class="label">OpenClaw WS</div><span class="metric">{eivoice_openclaw_ws}</span></div>
      <div class="card"><div class="label">Transport heartbeat</div><span class="metric">{eivoice_transport_heartbeat}</span></div>
      <div class="card"><div class="label">Reconnect attempts</div><span class="metric">{eivoice_transport_reconnect}</span></div>
      <div class="card"><div class="label">Local VAD</div><span class="metric">{eivoice_local_vad}</span></div>
      <div class="card"><div class="label">Local wake gate</div><span class="metric">{eivoice_local_wake_gate}</span></div>
      <div class="card"><div class="label">Warnings</div><span class="metric">{eivoice_warning_text}</span></div>
    </section>
    <h2>Status</h2>
    <pre>{status_json}</pre>
    <h2>Capabilities</h2>
    <pre>{capabilities_json}</pre>
    <h2>Realtime Vision</h2>
    <pre>{realtime_json}</pre>
    <h2>Voice</h2>
    <pre>{voice_json}</pre>
    <h2>eivoiceRuntime</h2>
    <pre>{eivoice_runtime_json}</pre>
    <h2>Neck</h2>
    <pre>{neck_json}</pre>
    <h2>Recent Actions</h2>
    <pre>{recent_json}</pre>
    <h2>Recent Events</h2>
    <pre>{recent_events_json}</pre>
  </main>
</body>
</html>
"""


def _render_lightweight_index(timestamp: float) -> str:
    generated_at = _display_value(timestamp)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>eihead native monitor</title>
  <style>
    :root {{
      --canvas: #101010;
      --surface: #151515;
      --surface-soft: #1a1a1a;
      --hairline: #3d3a39;
      --ink: #f2f2f2;
      --body: #bdbdbd;
      --muted: #8b949e;
      --green: #00d992;
      --green-soft: #2fd6a1;
      --warn: #f4c95d;
      --bad: #ff6b6b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 15px/1.55 Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--canvas);
      color: var(--ink);
    }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 32px 24px 44px; }}
    header {{
      display: grid;
      gap: 18px;
      border-bottom: 1px dashed rgba(79, 93, 117, 0.55);
      padding-bottom: 24px;
      margin-bottom: 22px;
    }}
    h1 {{ margin: 0; font-size: clamp(32px, 4vw, 56px); line-height: 1.03; font-weight: 400; letter-spacing: 0; }}
    h2 {{ margin: 32px 0 14px; font-size: 22px; line-height: 1.25; font-weight: 650; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--body); max-width: 860px; }}
    a {{ color: var(--green-soft); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .eyebrow {{
      color: var(--green);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 2.2px;
      text-transform: uppercase;
    }}
    .topline {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      border: 1px solid var(--hairline);
      border-radius: 999px;
      padding: 4px 10px;
      color: var(--body);
      font-size: 13px;
      white-space: nowrap;
    }}
    .pill.good {{ border-color: rgba(0, 217, 146, 0.45); color: var(--green); }}
    .pill.warn {{ border-color: rgba(244, 201, 93, 0.5); color: var(--warn); }}
    .pill.bad {{ border-color: rgba(255, 107, 107, 0.55); color: var(--bad); }}
    .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }}
    .card {{
      min-width: 0;
      background: var(--surface);
      border: 1px solid var(--hairline);
      border-radius: 8px;
      padding: 18px;
    }}
    .card.hot {{ border-color: rgba(0, 217, 146, 0.65); box-shadow: 0 0 0 1px rgba(0, 217, 146, 0.1) inset; }}
    .label {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 1.6px;
      text-transform: uppercase;
    }}
    .metric {{
      display: block;
      margin-top: 7px;
      color: var(--ink);
      font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 21px;
      font-weight: 650;
      line-height: 1.2;
      overflow-wrap: anywhere;
    }}
    .hint {{ display: block; margin-top: 8px; color: var(--muted); font-size: 13px; }}
    .rows {{ display: grid; gap: 8px; margin-top: 10px; }}
    .row {{
      display: grid;
      gap: 8px;
      grid-template-columns: minmax(110px, .75fr) minmax(0, 1.4fr);
      align-items: baseline;
      border-top: 1px solid rgba(61, 58, 57, 0.75);
      padding-top: 8px;
    }}
    .row span:first-child {{ color: var(--muted); font-size: 13px; }}
    .row span:last-child {{
      color: var(--ink);
      font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      overflow-wrap: anywhere;
    }}
    .endpoint-bar {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .endpoint-bar a {{
      border: 1px solid var(--hairline);
      border-radius: 6px;
      padding: 8px 10px;
      background: var(--surface);
      color: var(--green-soft);
      font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
    }}
    @media (max-width: 860px) {{
      main {{ padding: 24px 16px 36px; }}
      .row {{ grid-template-columns: 1fr; }}
      .metric {{ font-size: 18px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">HONJIA LIVE DIAGNOSTICS</div>
      <h1>eihead 真机监控台</h1>
      <p>lightweight shell · generated {generated_at} · 页面会读取实时 API，并展示可判断的中文结论、具体数据和证据。</p>
      <div class="topline">
        <span class="pill" id="health-pill">健康：读取中</span>
        <span class="pill" id="vision-pill">视觉：读取中</span>
        <span class="pill" id="neck-pill">脖子：读取中</span>
        <span class="pill" id="voice-pill">语音：读取中</span>
      </div>
    </header>

    <h2>具体数据</h2>
    <section class="grid">
      <div class="card"><div class="label">Web/API</div><span class="metric" id="health">读取中</span><span class="hint" id="health-hint">/health</span></div>
      <div class="card"><div class="label">视觉实时流</div><span class="metric" id="vision">读取中</span><span class="hint" id="vision-hint">/api/vision/realtime</span></div>
      <div class="card"><div class="label">脖子/云台</div><span class="metric" id="neck">读取中</span><span class="hint" id="neck-hint">/api/neck/status</span></div>
      <div class="card"><div class="label">语音链路</div><span class="metric" id="voice">读取中</span><span class="hint" id="voice-hint">/api/voice/realtime</span></div>
    </section>

    <h2>证据</h2>
    <section class="grid">
      <div class="card"><div class="label">视觉证据</div><div class="rows" id="vision-evidence"></div></div>
      <div class="card"><div class="label">脖子证据</div><div class="rows" id="neck-evidence"></div></div>
      <div class="card hot"><div class="label">语音链路明细</div><span class="metric" id="voice-chain-state">读取中</span><span class="hint" id="voice-chain-steps">/api/voice/realtime</span><div class="rows" id="voice-chain-evidence"></div></div>
      <div class="card"><div class="label">语音证据</div><div class="rows" id="voice-evidence"></div></div>
      <div class="card"><div class="label">运行证据</div><div class="rows" id="health-evidence"></div></div>
    </section>

    <h2>原始接口</h2>
    <div class="endpoint-bar">
      <a href="/health">/health</a>
      <a href="/status.json">/status.json</a>
      <a href="/api/vision/realtime">/api/vision/realtime</a>
      <a href="/api/neck/status">/api/neck/status</a>
      <a href="/api/voice/realtime">/api/voice/realtime</a>
      <a href="/api/voice/test">/api/voice/test</a>
      <a href="/api/capabilities">/api/capabilities</a>
    </div>
  </main>
  <script>
    const timeoutSignal = (ms) => {{
      const controller = new AbortController();
      setTimeout(() => controller.abort(), ms);
      return controller.signal;
    }};
    async function loadJson(path) {{
      const response = await fetch(path, {{ cache: 'no-store', signal: timeoutSignal(3500) }});
      return await response.json();
    }}
    function setText(id, text) {{
      document.getElementById(id).textContent = text || 'unknown';
    }}
    function text(value, fallback = '未知') {{
      if (value === null || value === undefined || value === '') return fallback;
      if (typeof value === 'boolean') return value ? '是' : '否';
      return String(value);
    }}
    function metric(value, unit = '') {{
      if (value === null || value === undefined || value === '') return '未知';
      return `${{value}}${{unit}}`;
    }}
    function statusTone(status) {{
      const normalized = String(status || '').toLowerCase();
      if (['ok', 'online', 'healthy', 'ready', 'wired', 'tracking', 'running', 'live'].includes(normalized)) return 'good';
      if (['not_wired', 'offline', 'error', 'failed', 'blocked', 'unavailable'].includes(normalized)) return 'bad';
      if (['degraded', 'stale', 'unknown', 'timeout'].includes(normalized)) return 'warn';
      return 'warn';
    }}
    function setPill(id, label, status) {{
      const el = document.getElementById(id);
      const tone = statusTone(status);
      el.className = `pill ${{tone}}`;
      el.textContent = `${{label}}：${{text(status)}}`;
    }}
    function setRows(id, rows) {{
      const root = document.getElementById(id);
      root.innerHTML = '';
      rows.forEach(([label, value]) => {{
        const row = document.createElement('div');
        row.className = 'row';
        const left = document.createElement('span');
        const right = document.createElement('span');
        left.textContent = label;
        right.textContent = text(value);
        row.append(left, right);
        root.append(row);
      }});
    }}
    function first(...values) {{
      return values.find((value) => value !== null && value !== undefined && value !== '');
    }}
    function providerSummary(health, key) {{
      const providers = health.native_providers || {{}};
      const provider = providers[key] || {{}};
      return `${{text(provider.status)}} / ${{text(provider.provider)}} / ${{text(provider.reason)}}`;
    }}
    function sourceFreshness(vision) {{
      const freshness = vision.source_freshness || (vision.diagnostic || {{}}).source_freshness || {{}};
      if (!freshness.state && freshness.age_s === undefined) return '未知';
      return `${{text(freshness.state)}} · age=${{metric(freshness.age_s, 's')}}`;
    }}
    function neckMotionEvidence(neck) {{
      const evidence = neck.motion_evidence || {{}};
      if (evidence.verified === true) {{
        const servo = evidence.servo_id ? `S${{evidence.servo_id}}` : 'S?';
        const axis = evidence.axis === 'pan' ? '水平' : text(evidence.axis);
        return `已确认：${{servo}} ${{axis}}舵机现场观察到转动`;
      }}
      if (evidence.verified === false) return '未确认：没有运动证据';
      return '未知：没有运动证据';
    }}
    function currentActionWillMove(neck) {{
      if (neck.will_move === null || neck.will_move === undefined) return '无当前动作';
      return neck.will_move;
    }}
    function voiceReadiness(voice) {{
      const chain = voice.voice_chain_readiness || {{}};
      return first(voice.readiness_message, chain.readinessMessage, chain.summary, '未知');
    }}
    function playbackGateSummary(voice) {{
      const observation = voice.observation || {{}};
      const runtime = observation.eivoice_runtime || observation.eivoiceRuntime || {{}};
      const frontend = runtime.audio_frontend || runtime.audioFrontend || {{}};
      const gate = frontend.playbackGate || frontend.playback_gate || {{}};
      if (!gate || Object.keys(gate).length === 0) return '未上报';
      const muted = gate.muted === true ? '压制中' : '未压制';
      const autoBarge = gate.bargeInEnabled === true || gate.barge_in_enabled === true ? '自动打断开' : '自动打断关';
      const outputActive = gate.outputActive === true || gate.output_active === true ? '输出中' : '输出空闲';
      const suppressed = first(gate.suppressedFrames, gate.suppressed_frames, 0);
      const barge = first(gate.bargeInCount, gate.barge_in_count, 0);
      const rms = first(gate.lastRms, gate.last_rms);
      const peak = first(gate.lastPeak, gate.last_peak);
      return `${{muted}} / ${{outputActive}} / ${{autoBarge}} / 回声帧=${{metric(suppressed)}} / 打断=${{metric(barge)}} / rms=${{metric(rms)}} / peak=${{metric(peak)}}`;
    }}
    function optimizationSummary(optimization) {{
      const latency = optimization.latency_ms || {{}};
      const bottleneck = optimization.bottleneck || {{}};
      const wake = optimization.wakeword || {{}};
      const audio = optimization.realtime_audio || {{}};
      const parts = [];
      if (bottleneck.stage) parts.push(`瓶颈=${{bottleneck.stage}} ${{metric(bottleneck.latency_ms, 'ms')}}`);
      if (latency.listen_asr !== undefined) parts.push(`ASR=${{metric(latency.listen_asr, 'ms')}}`);
      if (latency.dialogue !== undefined) parts.push(`eibrain=${{metric(latency.dialogue, 'ms')}}`);
      if (latency.speak !== undefined) parts.push(`TTS=${{metric(latency.speak, 'ms')}}`);
      if (latency.total !== undefined) parts.push(`总=${{metric(latency.total, 'ms')}}`);
      if (wake.state || wake.last_gate_reason) parts.push(`唤醒=${{text(wake.state || wake.last_gate_reason)}}`);
      if (audio.audio_level !== undefined) parts.push(`level=${{metric(audio.audio_level)}}`);
      if (audio.rms_dbfs !== undefined) parts.push(`rms=${{metric(audio.rms_dbfs, 'dBFS')}}`);
      return parts.length ? parts.join(' / ') : '未知';
    }}
    function voiceChainStateSummary(chain) {{
      if (!chain || Object.keys(chain).length === 0) return '未知';
      const parts = [];
      parts.push(text(chain.state_label || chain.state));
      if (chain.wake_state) parts.push(`wake=${{chain.wake_state}}`);
      if (chain.phase) parts.push(`phase=${{chain.phase}}`);
      return parts.join(' / ');
    }}
    function voiceChainStepsSummary(chain) {{
      const steps = (chain || {{}}).steps || [];
      const parts = steps
        .filter((step) => step && step.key)
        .map((step) => `${{text(step.label || step.key)}} ${{metric(step.latency_ms, 'ms')}}`);
      return parts.length ? parts.join(' / ') : '未知';
    }}
    function openclawWsSummary(ws) {{
      if (!ws || Object.keys(ws).length === 0) return '未配置';
      const parts = [];
      parts.push(ws.connected ? 'connected' : 'disconnected');
      if (ws.session_state) parts.push(`session=${{ws.session_state}}`);
      if (ws.url) parts.push(ws.url);
      return parts.join(' / ');
    }}
    Promise.allSettled([
      loadJson('/health'),
      loadJson('/api/vision/realtime'),
      loadJson('/api/neck/status'),
      loadJson('/api/voice/realtime'),
    ]).then((results) => {{
      const values = results.map((item) => item.status === 'fulfilled' ? item.value : {{ status: 'timeout' }});
      const health = values[0];
      const vision = values[1];
      const neck = values[2];
      const voice = values[3];
      setPill('health-pill', '健康', health.status);
      setPill('vision-pill', '视觉', vision.status);
      setPill('neck-pill', '脖子', neck.status);
      setPill('voice-pill', '语音', voice.status);
      setText('health', `${{text(health.status)}} · ${{text(health.runtime)}}`);
      setText('vision', `${{text(vision.status)}} · ${{metric(vision.fps, ' fps')}}`);
      setText('neck', `${{text(neck.status)}} · ${{metric(neck.current_angle, 'deg')}} -> ${{metric(neck.target_angle, 'deg')}}`);
      setText('voice', `${{text(voice.status)}} · audio=${{text((voice.realtime_audio || {{}}).running)}}`);
      setText('health-hint', `providers: eye ${{providerSummary(health, 'eye')}}`);
      setText('vision-hint', `${{text(vision.detections_summary, '无检测摘要')}}`);
      setText('neck-hint', `${{text(neck.readiness_message, '无 readiness 信息')}}`);
      setText('voice-hint', `${{voiceReadiness(voice)}}`);
      const voiceChain = voice.voice_chain || {{}};
      const voiceChainLatency = voiceChain.latency_ms || {{}};
      setText('voice-chain-state', voiceChainStateSummary(voiceChain));
      setText('voice-chain-steps', voiceChainStepsSummary(voiceChain));
      setRows('vision-evidence', [
        ['状态', vision.status],
        ['画面帧', first(vision.frame_id, (vision.overlay || {{}}).frame?.frame_id)],
        ['FPS', metric(vision.fps)],
        ['帧年龄', metric(first(vision.last_frame_age_s, vision.last_frame_age), 's')],
        ['检测数量', first(vision.detection_count, (vision.detections || []).length)],
        ['最高目标', text((vision.top_detection || {{}}).label, '无')],
        ['数据新鲜度', sourceFreshness(vision)],
        ['图像', text(((vision.overlay || {{}}).frame || {{}}).image_message, '无图像说明')],
      ]);
      setRows('neck-evidence', [
        ['状态', neck.status],
        ['当前角度', metric(neck.current_angle, 'deg')],
        ['目标角度', metric(neck.target_angle, 'deg')],
        ['运动验证', neckMotionEvidence(neck)],
        ['本次指令会动', currentActionWillMove(neck)],
        ['是否抑制', neck.suppressed],
        ['抑制原因', neck.suppression_reason],
        ['Servo', `${{text((neck.servo || {{}}).status)}} / ${{text((neck.servo || {{}}).reason)}}`],
        ['Axis', `pan=${{text(((neck.axis_support || {{}}).pan || {{}}).status)}} / tilt=${{text(((neck.axis_support || {{}}).tilt || {{}}).status)}}`],
      ]);
      setRows('voice-evidence', [
        ['状态', voice.status],
        ['实时音频', `enabled=${{text((voice.realtime_audio || {{}}).enabled)}} / running=${{text((voice.realtime_audio || {{}}).running)}}`],
        ['OpenClaw WS', openclawWsSummary(voice.openclaw_ws || {{}})],
        ['OpenClaw 错误', text((voice.openclaw_ws || {{}}).last_error, '无')],
        ['回声门控', playbackGateSummary(voice)],
        ['Round', `${{text((voice.round || {{}}).phase)}} / active=${{text((voice.round || {{}}).active)}}`],
        ['Scheduler', text((voice.scheduler || {{}}).state)],
        ['事件数', metric(voice.event_count)],
        ['首 token', metric(((voice.latency || {{}}).stage_latency_ms || {{}}).first_reply_token, 'ms')],
        ['首语音', metric(((voice.latency || {{}}).stage_latency_ms || {{}}).first_speech, 'ms')],
        ['性能优化', optimizationSummary(voice.optimization || {{}})],
        ['Readiness', voiceReadiness(voice)],
      ]);
      setRows('voice-chain-evidence', [
        ['链路状态', voiceChainStateSummary(voiceChain)],
        ['ASR 识别', metric(voiceChainLatency.listen_asr, 'ms')],
        ['脑端回复', metric(voiceChainLatency.dialogue, 'ms')],
        ['TTS 播放', metric(voiceChainLatency.speak, 'ms')],
        ['总耗时', metric(voiceChainLatency.total, 'ms')],
        ['ASR 到首文本', metric(voiceChainLatency.asr_to_first_text, 'ms')],
        ['ASR 到首音频', metric(voiceChainLatency.asr_to_first_audio, 'ms')],
        ['首文本到首音频', metric(voiceChainLatency.first_text_to_first_audio, 'ms')],
        ['音频接收持续', metric(voiceChainLatency.audio_receive_span, 'ms')],
        ['最大音频间隔', metric(voiceChainLatency.audio_gap_max, 'ms')],
        ['音频块数量', metric(voiceChainLatency.audio_chunks)],
        ['最后 ASR', voiceChain.last_asr_text],
        ['最后 TTS', voiceChain.last_tts_text],
      ]);
      setRows('health-evidence', [
        ['状态', health.status],
        ['Runtime', health.runtime],
        ['节点', first(health.node_id, health.node_role)],
        ['Eye provider', providerSummary(health, 'eye')],
        ['Ear provider', providerSummary(health, 'ear')],
        ['Mouth provider', providerSummary(health, 'mouth')],
        ['Neck provider', providerSummary(health, 'neck')],
      ]);
    }});
  </script>
</body>
</html>
"""


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _safe_payload(factory: Callable[[], JsonObject]) -> JsonObject:
    try:
        return factory()
    except Exception as exc:
        return {
            "ok": False,
            "status": "error",
            "error": {
                "code": "render_failed",
                "message": str(exc),
                "exception": exc.__class__.__name__,
            },
        }


def _json_for_html(payload: Mapping[str, Any]) -> str:
    return html.escape(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True))


def _display_value(value: Any) -> str:
    return html.escape(str(value))


def _first_mapping_value(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if isinstance(value, Mapping):
        return value
    return {}


def _metric_value(value: Any, *, suffix: str = "") -> str:
    if value in (None, ""):
        return "unknown"
    return f"{value}{suffix}"


def _top_detection_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "none"
    label = value.get("label", "unknown")
    score = value.get("score", value.get("confidence"))
    if score in (None, ""):
        return str(label)
    return f"{label} ({score})"


def _overlay_frame_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    width = value.get("width")
    height = value.get("height")
    if width not in (None, "") and height not in (None, ""):
        return f"{width}x{height}"
    return "unknown"


def _overlay_image_message(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "no live frame image yet"
    return str(value.get("image_message") or "no live frame image yet")


def _overlay_boxes_summary(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        return "none"
    if not value:
        return "none"
    return ", ".join(
        _overlay_box_summary(item)
        for item in value
        if isinstance(item, Mapping)
    ) or "none"


def _overlay_box_summary(value: Mapping[str, Any]) -> str:
    label = value.get("label") or "target"
    score_label = value.get("score_label") or label
    return (
        f"{score_label} "
        f"[{value.get('x_min')}, {value.get('y_min')}, {value.get('x_max')}, {value.get('y_max')}]"
    )


def _overlay_scores_summary(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        return "none"
    if not value:
        return "none"
    return ", ".join(str(item) for item in value)


def _overlay_top_target_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "none"
    score_label = value.get("score_label") or value.get("label") or "target"
    center = value.get("center")
    error = value.get("error")
    if isinstance(center, Mapping) and isinstance(error, Mapping):
        return (
            f"{score_label} "
            f"center=({center.get('x')}, {center.get('y')}) "
            f"error=({error.get('x')}, {error.get('y')})"
        )
    return str(score_label)


def _point_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "none"
    if value.get("x") in (None, "") or value.get("y") in (None, ""):
        return "none"
    return f"({value.get('x')}, {value.get('y')})"


def _hooks_used_summary(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return ", ".join(_metric_value(item) for item in value)
    return _metric_value(value)


def _pipeline_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    backend = value.get("backend") or value.get("transport") or value.get("source")
    sink = value.get("sink")
    if backend and sink:
        return f"{backend} -> {sink}"
    if backend:
        return _metric_value(backend)
    return _metric_value(value)


def _devices_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    camera = value.get("camera") or value.get("camera_device")
    hailo = value.get("hailo") or value.get("hailo_device")
    if camera and hailo:
        return f"{camera}, {hailo}"
    return _metric_value(value)


def _neck_servo_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    status = value.get("status") or "unknown"
    available = value.get("available")
    reason = value.get("reason")
    parts = [str(status)]
    if available is not None:
        parts.append("available" if available is True else "unavailable")
    if reason and reason != "unknown":
        parts.append(str(reason))
    return " / ".join(parts)


def _neck_motion_evidence_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    verified = value.get("verified")
    if verified is True:
        servo_id = value.get("servo_id")
        axis = "pan" if value.get("axis") in {None, "", "pan"} else str(value.get("axis"))
        servo = f"S{servo_id}" if servo_id is not None else "servo"
        return f"verified / {servo} / {axis}"
    if verified is False:
        return "unverified"
    return "unknown"


def _neck_axis_support_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    parts: list[str] = []
    for axis in ("pan", "tilt"):
        axis_payload = value.get(axis)
        if not isinstance(axis_payload, Mapping):
            parts.append(f"{axis}=unknown")
            continue
        status = axis_payload.get("status")
        supported = axis_payload.get("supported")
        reason = axis_payload.get("reason")
        if status:
            rendered = str(status)
        elif supported is True:
            rendered = "supported"
        elif supported is False:
            rendered = "unsupported"
        else:
            rendered = "unknown"
        if reason:
            rendered = f"{rendered} ({reason})"
        parts.append(f"{axis}={rendered}")
    return " / ".join(parts)


def _voice_component_summary(value: Any, *, kind: str) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    state = value.get("state") or value.get("status") or "unknown"
    if kind == "ear":
        provider = value.get("provider")
        if provider:
            return f"{state} ({provider})"
        return str(state)
    backend = value.get("backend")
    model = value.get("model")
    if backend and model:
        return f"{state} ({backend}/{model})"
    if backend:
        return f"{state} ({backend})"
    return str(state)


def _voice_dialogue_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    phase = value.get("phase") or value.get("last_status")
    transcript = value.get("last_transcript")
    if phase and transcript:
        return f"{phase}: {transcript}"
    if phase:
        return str(phase)
    return "unknown"


def _voice_heard_text(value: Any) -> str:
    dialogue, last_turn = _voice_dialogue_and_turn(value)
    return str(
        dialogue.get("last_transcript")
        or last_turn.get("transcript")
        or last_turn.get("text")
        or "unknown"
    )


def _voice_reply_text(value: Any) -> str:
    dialogue, last_turn = _voice_dialogue_and_turn(value)
    return str(
        dialogue.get("last_reply")
        or last_turn.get("reply")
        or last_turn.get("response")
        or "unknown"
    )


def _voice_dialogue_engine_summary(value: Any) -> str:
    dialogue, _ = _voice_dialogue_and_turn(value)
    engine = dialogue.get("dialogue")
    if not isinstance(engine, Mapping):
        return "unknown"
    parts: list[str] = []
    provider = engine.get("provider")
    if provider:
        parts.append(str(provider))
    returncode = engine.get("returncode")
    if returncode not in (None, ""):
        parts.append(f"returncode={returncode}")
    elapsed_ms = engine.get("elapsed_ms")
    if elapsed_ms not in (None, ""):
        parts.append(f"{elapsed_ms}ms")
    return " / ".join(parts) if parts else "unknown"


def _voice_protocol_event_summary(value: Any) -> str:
    dialogue, _ = _voice_dialogue_and_turn(value)
    engine = dialogue.get("dialogue")
    if not isinstance(engine, Mapping):
        return "unknown"
    event = engine.get("event_name")
    round_id = engine.get("round_id") or dialogue.get("current_round_id")
    event_id = engine.get("event_id")
    parts = [str(item) for item in (event, round_id, _short_identifier(event_id)) if item]
    return " / ".join(parts) if parts else "unknown"


def _voice_latency_breakdown_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    latency = value.get("latency")
    stage_latency = latency.get("stage_latency_ms") if isinstance(latency, Mapping) else None
    if not isinstance(stage_latency, Mapping):
        dialogue, _ = _voice_dialogue_and_turn(value)
        stage_latency = dialogue.get("last_stage_latency_ms")
    if not isinstance(stage_latency, Mapping):
        return "unknown"
    labels = (
        ("listen_asr", "ASR"),
        ("dialogue", "eibrain"),
        ("speak", "TTS"),
        ("total", "total"),
    )
    parts = [
        f"{label} {stage_latency[key]}ms"
        for key, label in labels
        if stage_latency.get(key) not in (None, "")
    ]
    return " / ".join(parts) if parts else "unknown"


def _voice_chain_state_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    parts = [str(value.get("state_label") or value.get("state") or "unknown")]
    wake_state = value.get("wake_state")
    phase = value.get("phase")
    if wake_state:
        parts.append(f"wake={wake_state}")
    if phase:
        parts.append(f"phase={phase}")
    return " / ".join(parts)


def _voice_chain_steps_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    steps = value.get("steps")
    if isinstance(steps, (list, tuple)):
        parts = [
            f"{step.get('label') or step.get('key')} {_metric_value(step.get('latency_ms'), suffix='ms')}"
            for step in steps
            if isinstance(step, Mapping) and (step.get("label") or step.get("key"))
        ]
        if parts:
            return " / ".join(parts)
    latency = value.get("latency_ms")
    if not isinstance(latency, Mapping):
        return "unknown"
    parts = [
        f"{label} {_metric_value(latency.get(key), suffix='ms')}"
        for key, label in (
            ("listen_asr", "ASR 识别"),
            ("dialogue", "脑端回复"),
            ("speak", "TTS 播放"),
            ("total", "总耗时"),
        )
        if latency.get(key) not in (None, "")
    ]
    return " / ".join(parts) if parts else "unknown"


def _voice_chain_text(value: Any, key: str) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    text = value.get(key)
    if text in (None, ""):
        return "unknown"
    return str(text)


def _openclaw_ws_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "not configured"
    parts = ["connected" if value.get("connected") is True else "disconnected"]
    session_state = value.get("session_state") or value.get("sessionState")
    if session_state:
        parts.append(f"session={session_state}")
    url = value.get("url")
    if url:
        parts.append(str(url))
    return " / ".join(parts)


def _openclaw_ws_error_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "none"
    error = value.get("last_error") or value.get("lastError")
    if error in (None, ""):
        return "none"
    return str(error)


def _voice_playback_gate_summary(value: Any) -> str:
    gate = _voice_playback_gate(value)
    if not gate:
        return "not reported"
    muted = "压制中" if gate.get("muted") is True else "未压制"
    auto_barge = "自动打断开" if gate.get("barge_in_enabled") is True or gate.get("bargeInEnabled") is True else "自动打断关"
    output_active = "输出中" if gate.get("output_active") is True or gate.get("outputActive") is True else "输出空闲"
    suppressed = gate.get("suppressed_frames") or gate.get("suppressedFrames") or 0
    barge_in = gate.get("barge_in_count") or gate.get("bargeInCount") or 0
    rms = gate.get("last_rms") or gate.get("lastRms")
    peak = gate.get("last_peak") or gate.get("lastPeak")
    return (
        f"{muted} / {output_active} / {auto_barge} / 回声帧 {suppressed} / 打断 {barge_in} / "
        f"rms {_metric_value(rms)} / peak {_metric_value(peak)}"
    )


def _voice_playback_gate(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    observation = value.get("observation")
    if not isinstance(observation, Mapping):
        return {}
    runtime = observation.get("eivoice_runtime") or observation.get("eivoiceRuntime")
    if not isinstance(runtime, Mapping):
        return {}
    frontend = runtime.get("audio_frontend") or runtime.get("audioFrontend")
    if not isinstance(frontend, Mapping):
        return {}
    gate = frontend.get("playback_gate") or frontend.get("playbackGate")
    if isinstance(gate, Mapping):
        return gate
    return {}


def _voice_optimization_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    latency = value.get("latency_ms")
    if not isinstance(latency, Mapping):
        latency = {}
    bottleneck = value.get("bottleneck")
    if not isinstance(bottleneck, Mapping):
        bottleneck = {}
    wakeword = value.get("wakeword")
    if not isinstance(wakeword, Mapping):
        wakeword = {}
    realtime_audio = value.get("realtime_audio")
    if not isinstance(realtime_audio, Mapping):
        realtime_audio = {}
    parts: list[str] = []
    stage = bottleneck.get("stage")
    bottleneck_ms = bottleneck.get("latency_ms")
    if stage:
        parts.append(f"瓶颈 {stage} {_metric_value(bottleneck_ms, suffix='ms')}")
    for key, label in (
        ("listen_asr", "ASR"),
        ("dialogue", "eibrain"),
        ("speak", "TTS"),
        ("total", "总"),
    ):
        if latency.get(key) not in (None, ""):
            parts.append(f"{label} {_metric_value(latency.get(key), suffix='ms')}")
    wake_state = wakeword.get("state") or wakeword.get("last_gate_reason")
    if wake_state:
        parts.append(f"唤醒 {wake_state}")
    if realtime_audio.get("audio_level") not in (None, ""):
        parts.append(f"level {realtime_audio.get('audio_level')}")
    if realtime_audio.get("rms_dbfs") not in (None, ""):
        parts.append(f"rms {realtime_audio.get('rms_dbfs')}dBFS")
    return " / ".join(parts) if parts else "unknown"


def _voice_tts_playback_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    mouth = value.get("mouth")
    if not isinstance(mouth, Mapping):
        return "unknown"
    playback = mouth.get("tts_playback")
    details = playback.get("details") if isinstance(playback, Mapping) else None
    if not isinstance(details, Mapping):
        details = {}
    status = mouth.get("status") or (playback.get("status") if isinstance(playback, Mapping) else None)
    backend = mouth.get("backend") or details.get("provider")
    model = mouth.get("model") or details.get("model")
    voice_id = mouth.get("voice_id") or details.get("voice_id")
    device = details.get("device")
    parts = [str(item) for item in (status, backend, model, voice_id, device) if item]
    return " / ".join(parts) if parts else "unknown"


def _voice_mic_vad_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    ear = value.get("ear")
    realtime_audio = value.get("realtime_audio")
    if not isinstance(ear, Mapping):
        ear = {}
    if not isinstance(realtime_audio, Mapping):
        realtime_audio = {}
    capture = ear.get("capture")
    details = capture.get("details") if isinstance(capture, Mapping) else None
    if not isinstance(details, Mapping):
        details = {}
    device = details.get("device")
    audio_level = ear.get("audio_level", realtime_audio.get("audio_level"))
    rms_dbfs = ear.get("rms_dbfs", realtime_audio.get("rms_dbfs"))
    vad = ear.get("vad_triggered", realtime_audio.get("vad_triggered"))
    parts: list[str] = []
    if device:
        parts.append(str(device))
    if audio_level not in (None, ""):
        parts.append(f"level={audio_level}")
    if rms_dbfs not in (None, ""):
        parts.append(f"rms={rms_dbfs}dBFS")
    if vad not in (None, ""):
        parts.append(f"vad={vad}")
    return " / ".join(parts) if parts else "unknown"


def _eivoice_local_vad_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    frontend = value.get("audioFrontend") or value.get("audio_frontend")
    if not isinstance(frontend, Mapping):
        return "unknown"
    local_vad = frontend.get("localVad") or frontend.get("local_vad")
    if not isinstance(local_vad, Mapping) or not local_vad:
        return "未上报"
    state = "语音段" if local_vad.get("active") is True else "静音/待机"
    enabled = "开启" if local_vad.get("enabled") is True else "关闭"
    passed = _metric_value(local_vad.get("passedFrames") or local_vad.get("passed_frames"))
    dropped = _metric_value(local_vad.get("droppedFrames") or local_vad.get("dropped_frames"))
    segment = _metric_value(local_vad.get("segmentFrames") or local_vad.get("segment_frames"))
    max_frames = _metric_value(local_vad.get("maxFrames") or local_vad.get("max_frames"))
    rms = _metric_value(local_vad.get("rmsThreshold") or local_vad.get("rms_threshold"))
    peak = _metric_value(local_vad.get("peakThreshold") or local_vad.get("peak_threshold"))
    return (
        f"{enabled} / {state} / 放行={passed} / 丢弃={dropped} / "
        f"段={segment}/{max_frames} / rms阈值={rms} / peak阈值={peak}"
    )


def _eivoice_local_wake_gate_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    frontend = value.get("audioFrontend") or value.get("audio_frontend")
    if not isinstance(frontend, Mapping):
        return "unknown"
    gate = frontend.get("localWakeGate") or frontend.get("local_wake_gate")
    if not isinstance(gate, Mapping) or not gate:
        return "未上报"
    enabled = "开启" if gate.get("enabled") is True else "关闭"
    active = "唤醒中" if gate.get("conversationActive") is True or gate.get("conversation_active") is True else "休眠"
    state = gate.get("state") or "unknown"
    reason = gate.get("lastGateReason") or gate.get("last_gate_reason") or "none"
    transcript = gate.get("lastTranscript") or gate.get("last_transcript") or ""
    asr_ms = gate.get("lastAsrMs") or gate.get("last_asr_ms")
    dropped = gate.get("droppedSegments") or gate.get("dropped_segments") or 0
    wake_hits = gate.get("wakeDetections") or gate.get("wake_detections") or 0
    end_hits = gate.get("endDetections") or gate.get("end_detections") or 0
    transcriber = gate.get("transcriber") if isinstance(gate.get("transcriber"), Mapping) else {}
    provider = transcriber.get("provider") or "unknown"
    provider_state = transcriber.get("state") or "unknown"
    transcript_part = f" / ASR={transcript}" if transcript else ""
    return (
        f"{enabled} / {active} / 状态={state} / 原因={reason}{transcript_part} / "
        f"识别={_metric_value(asr_ms, suffix='ms')} / 丢弃段={_metric_value(dropped)} / "
        f"唤醒={_metric_value(wake_hits)} / 结束={_metric_value(end_hits)} / {provider}:{provider_state}"
    )


def _voice_asr_detail_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    ear = value.get("ear")
    if not isinstance(ear, Mapping):
        return "unknown"
    asr = ear.get("asr")
    if not isinstance(asr, Mapping):
        asr = {}
    diagnostics = asr.get("provider_diagnostics")
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    provider = ear.get("provider") or asr.get("provider") or diagnostics.get("provider")
    state = asr.get("provider_state") or diagnostics.get("state") or asr.get("status")
    model_type = diagnostics.get("model_type")
    final_count = asr.get("final_count")
    parts = [str(item) for item in (provider, state, model_type) if item]
    if final_count not in (None, ""):
        parts.append(f"final={final_count}")
    return " / ".join(parts) if parts else "unknown"


def _voice_dialogue_and_turn(value: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return {}, {}
    dialogue = value.get("dialogue")
    last_turn = value.get("last_turn")
    return (
        dialogue if isinstance(dialogue, Mapping) else {},
        last_turn if isinstance(last_turn, Mapping) else {},
    )


def _short_identifier(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    if len(text) <= 14:
        return text
    return f"{text[:10]}..."


def _voice_latency_total_ms(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    return value.get("total_ms")


def _voice_latency_stage_ms(value: Any, key: str) -> Any:
    if not isinstance(value, Mapping):
        return None
    stage_latency = value.get("stage_latency_ms")
    if isinstance(stage_latency, Mapping):
        return stage_latency.get(key)
    return None


def _voice_bottleneck_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    stage = value.get("stage")
    latency_ms = value.get("latency_ms")
    if stage and latency_ms not in (None, ""):
        return f"{stage} ({latency_ms}ms)"
    if stage:
        return str(stage)
    return "unknown"


def _voice_last_turn_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    transcript = value.get("transcript") or value.get("text")
    reply = value.get("reply") or value.get("response")
    if transcript and reply:
        return f"{transcript} -> {reply}"
    if transcript:
        return str(transcript)
    if reply:
        return str(reply)
    return "unknown"


def _voice_round_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    round_id = value.get("current_round_id") or value.get("round_id")
    token = value.get("current_cancellation_token") or value.get("cancellation_token")
    if not round_id:
        return "unknown"
    if isinstance(token, Mapping) and token.get("cancelled") is True:
        return f"{round_id} (cancelled)"
    if token:
        return f"{round_id} (cancel token)"
    return str(round_id)


def _voice_scheduler_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    state = value.get("state") or value.get("status") or value.get("component_state") or "unknown"
    active_round = value.get("active_round_id") or value.get("round_id")
    stale = value.get("stale") is True
    parts = [str(state)]
    if active_round:
        parts.append(str(active_round))
    if stale and "stale" not in {part.lower() for part in parts}:
        parts.append("stale")
    return " / ".join(parts)


def _voice_realtime_component_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    summary = value.get("summary")
    if summary not in (None, ""):
        return str(summary)
    state = value.get("state") or value.get("status") or value.get("component_state") or "unknown"
    return str(state)


def _voice_interruption_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    state = value.get("state") or "unknown"
    parts = [str(state)]
    interrupt_count = value.get("interrupt_count")
    interrupted_round_count = value.get("interrupted_round_count")
    if interrupt_count is not None:
        parts.append(f"{interrupt_count} interrupts")
    if interrupted_round_count is not None:
        parts.append(f"{interrupted_round_count} rounds")
    last_interrupt = value.get("last_interrupt")
    if isinstance(last_interrupt, Mapping):
        reason = last_interrupt.get("reason") or last_interrupt.get("type")
        if reason:
            parts.append(str(reason))
    if value.get("stale") is True and "stale" not in {part.lower() for part in parts}:
        parts.append("stale")
    return " / ".join(parts)


def _voice_microfeedback_summary(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, Mapping):
        label = value.get("text") or value.get("last") or value.get("label") or value.get("status") or value.get("message")
        score = value.get("score")
        if label and score not in (None, ""):
            return f"{label} ({score})"
        if label:
            return str(label)
    return _metric_value(value)


def _voice_closed_loop_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    parts: list[str] = []
    for key in ("final_asr", "first_reply_delta", "first_speech", "complete"):
        if key in value:
            parts.append(f"{key}={'yes' if value.get(key) is True else 'no'}")
    return " / ".join(parts) if parts else "unknown"


def _voice_realtime_audio_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "not wired"
    enabled = value.get("enabled") is True
    running = value.get("running") is True
    parts = ["running" if running else "enabled" if enabled else "disabled"]
    buffer_ms = value.get("buffer_ms")
    if buffer_ms not in (None, ""):
        parts.append(f"buffer {buffer_ms}ms")
    detector = value.get("wake_detector")
    if isinstance(detector, Mapping):
        emitted = detector.get("emitted_count")
        polls = detector.get("poll_count")
        if emitted not in (None, ""):
            parts.append(f"wake {emitted}")
        if polls not in (None, ""):
            parts.append(f"poll {polls}")
        last_text = detector.get("last_text")
        if last_text:
            parts.append(str(last_text))
    last_error = value.get("last_error")
    if last_error:
        parts.append(f"error: {last_error}")
    return " / ".join(parts)


def _voice_cancellation_chain_summary(value: Any) -> str:
    if value is None:
        return "none"
    if not isinstance(value, (list, tuple)):
        return _metric_value(value)
    if not value:
        return "none"
    targets: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            target = item.get("target") or item.get("event_type") or item.get("reason")
            if target:
                targets.append(str(target))
        elif item not in (None, ""):
            targets.append(str(item))
    if not targets:
        return "none"
    return " -> ".join(targets)


def _voice_chain_readiness_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    summary = value.get("summary")
    if summary not in (None, ""):
        return str(summary)
    turn_count = value.get("turnCount")
    if value.get("honjiaReady") is True:
        return f"ready: {turn_count} live turns"
    source = value.get("source") or "unknown"
    return str(source)


def _voice_chain_bottleneck_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    bottleneck = value.get("bottleneck")
    if not isinstance(bottleneck, Mapping):
        failed = value.get("failedMetrics")
        if isinstance(failed, list) and failed:
            return ", ".join(str(item) for item in failed)
        return "none"
    field = bottleneck.get("field") or bottleneck.get("label") or "unknown"
    p95 = bottleneck.get("p95")
    threshold = bottleneck.get("threshold")
    if p95 not in (None, "") and threshold not in (None, ""):
        return f"{field} ({p95}ms / {threshold}ms)"
    if p95 not in (None, ""):
        return f"{field} ({p95}ms)"
    return str(field)


def _is_healthy(payload: Mapping[str, Any]) -> bool:
    state = str(payload.get("status", "ok")).lower()
    if payload.get("ok") is False:
        return False
    return state not in UNHEALTHY_STATES


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
    if status_code >= 500:
        return "internal_error"
    return "http_error"


__all__ = [
    "ACTION_LOG_ATTRS",
    "EiheadMonitorError",
    "EiheadMonitorServer",
    "create_handler",
    "create_server",
    "serve",
]
