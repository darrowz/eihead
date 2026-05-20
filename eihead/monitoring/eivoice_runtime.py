"""EIVoice runtime monitoring normalization."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


QUEUE_NAMES = (
    "opus_encode_queue",
    "ws_send_queue",
    "opus_decode_queue",
    "audio_playback_queue",
)


def build_eivoice_runtime_panel(status: dict[str, Any]) -> dict[str, Any]:
    """Build a Web-friendly EIVoice runtime diagnostics panel."""

    runtime = _mapping(status)
    state = _text(runtime.get("state") or runtime.get("runtime_state") or runtime.get("status"))
    conversation_state = _text(
        runtime.get("conversationState")
        or runtime.get("conversation_state")
        or runtime.get("dialogue_state")
        or runtime.get("mode")
        or state,
        default="unknown",
    )
    queues = {name: _normalize_queue(name, runtime) for name in QUEUE_NAMES}
    dropped_total = sum(queue["droppedOldest"] + queue["droppedNewest"] for queue in queues.values())
    has_audio_frontend = any(key in runtime for key in ("audio_frontend", "acousticFrontend"))
    audio_frontend = _normalize_audio_frontend(runtime)
    wakeword = dict(_mapping(runtime.get("wakeword") or runtime.get("wake_word") or runtime.get("wakeword_buffer")))
    transport = _normalize_transport(runtime.get("transport") or runtime.get("voiceTransport"))
    openclaw_ws = _normalize_openclaw_ws(runtime, transport)

    warnings: list[str] = []
    if not state:
        warnings.append("runtime state is missing")
    if dropped_total > 0:
        warnings.append(f"queue drops detected: {dropped_total}")
    if state and not has_audio_frontend:
        warnings.append("audio frontend readiness is missing")
    warnings.extend(str(item) for item in audio_frontend.get("warnings", []) if item)
    if _component_unavailable(audio_frontend.get("aec")):
        warnings.append("AEC unavailable")
    if _component_unavailable(audio_frontend.get("ns")):
        warnings.append("NS unavailable")
    if _component_unavailable(audio_frontend.get("vad")):
        warnings.append("VAD unavailable")
    if _component_unavailable(audio_frontend.get("loopback")):
        warnings.append("loopback unavailable")
    if _transport_degraded(transport):
        warnings.append(f"transport {transport['state']}")
    if _openclaw_ws_degraded(transport, openclaw_ws):
        warnings.append(f"openclaw_ws {openclaw_ws['sessionState']}")
    transport_error = _mapping(transport.get("lastError"))
    if transport_error:
        warnings.append(
            "transport error: "
            f"{_text(transport_error.get('kind'), default='Error')} "
            f"{_text(transport_error.get('context'), default='unknown')}"
        )
    if openclaw_ws.get("lastError"):
        warnings.append(f"openclaw_ws error: {_text(openclaw_ws.get('lastError'))}")
    warnings = list(dict.fromkeys(warnings))

    health = "healthy"
    if (
        dropped_total > 0
        or (state and not has_audio_frontend)
        or _component_unavailable(audio_frontend.get("aec"))
        or _component_unavailable(audio_frontend.get("ns"))
        or _component_unavailable(audio_frontend.get("vad"))
        or _component_unavailable(audio_frontend.get("loopback"))
        or _transport_degraded(transport)
        or _openclaw_ws_degraded(transport, openclaw_ws)
        or bool(transport_error)
    ):
        health = "degraded"
    elif not state:
        health = "waiting"

    return {
        "state": state or "waiting",
        "conversationState": conversation_state,
        "queueSummary": _queue_summary(queues),
        "queues": queues,
        "droppedTotal": dropped_total,
        "audioFrontend": audio_frontend,
        "transport": transport,
        "openclawWs": openclaw_ws,
        "wakeword": wakeword,
        "health": health,
        "warnings": warnings,
    }


def eivoice_runtime_status_from_app(app: Any) -> dict[str, Any]:
    """Read EIVoice runtime status from app hooks or nested status payloads."""

    for attr_name in (
        "eivoice_runtime_status",
        "eivoice_runtime",
        "latest_eivoice_runtime_status",
        "latest_eivoice_runtime",
    ):
        if not hasattr(app, attr_name):
            continue
        source = getattr(app, attr_name)
        payload = source() if callable(source) else source
        if isinstance(payload, Mapping):
            return dict(payload)

    status_fn = getattr(app, "status", None)
    if callable(status_fn):
        try:
            status = status_fn()
        except Exception:
            status = {}
        if isinstance(status, Mapping):
            for key in ("eivoice_runtime", "eivoiceRuntime", "voice_runtime", "runtime_status"):
                payload = status.get(key)
                if isinstance(payload, Mapping):
                    return dict(payload)
            body_runtime = status.get("body_runtime")
            if isinstance(body_runtime, Mapping):
                for key in ("eivoice_runtime", "eivoiceRuntime", "voice_runtime", "runtime_status"):
                    payload = body_runtime.get(key)
                    if isinstance(payload, Mapping):
                        return dict(payload)

    body_runtime = getattr(app, "body_runtime", None)
    for attr_name in (
        "eivoice_runtime_status",
        "eivoice_runtime",
        "latest_eivoice_runtime_status",
        "latest_eivoice_runtime",
    ):
        if not hasattr(body_runtime, attr_name):
            continue
        source = getattr(body_runtime, attr_name)
        payload = source() if callable(source) else source
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _normalize_queue(name: str, runtime: Mapping[str, Any]) -> dict[str, Any]:
    queue = _queue_source(name, runtime)
    depth = _number(
        queue.get("depth")
        or queue.get("size")
        or queue.get("qsize")
        or queue.get("current_depth"),
        default=0,
    )
    capacity = _number(
        queue.get("capacity")
        or queue.get("maxsize")
        or queue.get("max_size")
        or queue.get("limit"),
        default=0,
    )
    dropped_oldest = _number(
        queue.get("droppedOldest")
        or queue.get("dropped_oldest")
        or queue.get("drop_oldest")
        or _mapping(queue.get("dropped")).get("oldest"),
        default=0,
    )
    dropped_newest = _number(
        queue.get("droppedNewest")
        or queue.get("dropped_newest")
        or queue.get("drop_newest")
        or _mapping(queue.get("dropped")).get("newest"),
        default=0,
    )
    return {
        "depth": depth,
        "capacity": capacity,
        "fillRatio": _fill_ratio(depth, capacity),
        "policy": _text(
            queue.get("policy")
            or queue.get("full_policy")
            or queue.get("drop_policy")
            or queue.get("overflow_policy"),
            default="unknown",
        ),
        "droppedOldest": dropped_oldest,
        "droppedNewest": dropped_newest,
    }


def _queue_source(name: str, runtime: Mapping[str, Any]) -> Mapping[str, Any]:
    queues = _mapping(runtime.get("queues") or runtime.get("queue_status") or runtime.get("queueStatus"))
    return _mapping(queues.get(name) or runtime.get(name))


def _queue_summary(queues: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    total_depth = sum(_number(queue.get("depth"), default=0) for queue in queues.values())
    total_capacity = sum(_number(queue.get("capacity"), default=0) for queue in queues.values())
    max_fill = max((_float(queue.get("fillRatio"), default=0.0) for queue in queues.values()), default=0.0)
    return {
        "count": len(queues),
        "totalDepth": total_depth,
        "totalCapacity": total_capacity,
        "maxFillRatio": max_fill,
    }


def _normalize_audio_frontend(runtime: Mapping[str, Any]) -> dict[str, Any]:
    frontend = _mapping(runtime.get("audio_frontend") or runtime.get("acousticFrontend"))
    audio_format = _mapping(frontend.get("audio_format") or frontend.get("audioFormat"))
    return {
        "mode": _text(frontend.get("mode"), default=""),
        "aec": _normalize_component(frontend.get("aec")),
        "ns": _normalize_component(frontend.get("ns") or frontend.get("noise_suppression")),
        "vad": _normalize_component(frontend.get("vad")),
        "loopback": _normalize_component(frontend.get("loopback")),
        "playbackGate": _normalize_playback_gate(frontend.get("playback_gate") or frontend.get("playbackGate")),
        "devices": dict(_mapping(frontend.get("devices"))),
        "audioFormat": {
            "sampleRate": _number(audio_format.get("sample_rate") or audio_format.get("sampleRate"), default=0),
            "frameMs": _number(audio_format.get("frame_ms") or audio_format.get("frameMs"), default=0),
            "channels": _number(audio_format.get("channels"), default=0),
        },
        "aecBackend": _text(frontend.get("aec_backend") or frontend.get("aecBackend"), default=""),
        "aecStatus": _text(frontend.get("aec_status") or frontend.get("aecStatus"), default=""),
        "lastCapture": _normalize_last_capture(frontend.get("last_capture") or frontend.get("lastCapture")),
        "warnings": _list(frontend.get("warnings")),
    }


def _normalize_playback_gate(value: Any) -> dict[str, Any]:
    gate = _mapping(value)
    if not gate:
        return {}
    last_barge_in = _mapping(gate.get("last_barge_in") or gate.get("lastBargeIn"))
    return {
        "muted": bool(gate.get("muted")),
        "suppressedFrames": _number(gate.get("suppressed_frames") or gate.get("suppressedFrames"), default=0),
        "bargeInCount": _number(gate.get("barge_in_count") or gate.get("bargeInCount"), default=0),
        "speechFrames": _number(gate.get("speech_frames") or gate.get("speechFrames"), default=0),
        "consecutiveFrames": _number(gate.get("consecutive_frames") or gate.get("consecutiveFrames"), default=0),
        "rmsThreshold": _float(gate.get("rms_threshold") or gate.get("rmsThreshold"), default=0.0),
        "peakThreshold": _float(gate.get("peak_threshold") or gate.get("peakThreshold"), default=0.0),
        "lastRms": _float(gate.get("last_rms") or gate.get("lastRms"), default=0.0),
        "lastPeak": _float(gate.get("last_peak") or gate.get("lastPeak"), default=0.0),
        "lastBargeIn": _normalize_barge_in(last_barge_in),
    }


def _normalize_barge_in(value: Mapping[str, Any]) -> dict[str, Any]:
    if not value:
        return {}
    return {
        "reason": _text(value.get("reason"), default=""),
        "rms": _float(value.get("rms"), default=0.0),
        "peak": _float(value.get("peak"), default=0.0),
        "cleared": bool(value.get("cleared")),
        "cancelledRemoteOutput": bool(
            value.get("cancelled_remote_output") or value.get("cancelledRemoteOutput")
        ),
    }


def _normalize_last_capture(value: Any) -> dict[str, Any]:
    capture = _mapping(value)
    if not capture:
        return {}
    loopback_reference = _mapping(capture.get("loopback_reference") or capture.get("loopbackReference"))
    return {
        "playbackReferenceAvailable": bool(
            capture.get("playback_reference_available") or capture.get("playbackReferenceAvailable")
        ),
        "referenceAgeMs": _float(capture.get("reference_age_ms") or capture.get("referenceAgeMs"), default=0.0),
        "referenceMatchedBy": _text(capture.get("reference_matched_by") or capture.get("referenceMatchedBy")),
        "referenceSequence": _number(capture.get("reference_sequence") or capture.get("referenceSequence"), default=0),
        "fallbackReason": _text(capture.get("fallback_reason") or capture.get("fallbackReason")),
        "loopbackReference": _normalize_loopback_reference(loopback_reference),
    }


def _normalize_loopback_reference(value: Mapping[str, Any]) -> dict[str, Any]:
    if not value:
        return {}
    return {
        "ready": bool(value.get("ready")),
        "state": _text(value.get("state")),
        "reason": _text(value.get("reason")),
        "referenceAgeMs": _float(value.get("reference_age_ms") or value.get("referenceAgeMs"), default=0.0),
        "matchedBy": _text(value.get("matched_by") or value.get("matchedBy")),
        "maxAgeMs": _number(value.get("max_age_ms") or value.get("maxAgeMs"), default=0),
        "device": _text(value.get("device")),
        "aecStatus": _text(value.get("aec_status") or value.get("aecStatus")),
    }


def _normalize_transport(value: Any) -> dict[str, Any]:
    transport = _mapping(value)
    heartbeat = _mapping(transport.get("heartbeat"))
    reconnect = _mapping(transport.get("reconnect"))
    last_error = _mapping(transport.get("last_error") or transport.get("lastError"))
    return {
        "name": _text(transport.get("transport") or transport.get("name"), default="unknown"),
        "state": _text(transport.get("state") or _mapping(transport.get("connection")).get("state"), default="unknown"),
        "url": _text(transport.get("url") or transport.get("ws_url") or transport.get("wsUrl"), default=""),
        "heartbeat": {
            "awaiting_pong": bool(heartbeat.get("awaiting_pong") or heartbeat.get("awaitingPong")),
            "timed_out": bool(heartbeat.get("timed_out") or heartbeat.get("timedOut")),
            "latency_ms": _number(heartbeat.get("latency_ms") or heartbeat.get("latencyMs"), default=0),
        },
        "reconnect": {
            "attempt": _number(reconnect.get("attempt"), default=0),
            "backoff_s": _float(reconnect.get("backoff_s") or reconnect.get("backoffS"), default=0.0),
            "ready": bool(reconnect.get("ready")),
            "reason": _text(reconnect.get("reason"), default=""),
        },
        "lastError": dict(last_error),
    }


def _transport_degraded(transport: Mapping[str, Any]) -> bool:
    state = _text(transport.get("state")).lower()
    heartbeat = _mapping(transport.get("heartbeat"))
    return state in {"reconnect_wait", "closed", "error", "failed"} or heartbeat.get("timed_out") is True


def _normalize_openclaw_ws(runtime: Mapping[str, Any], transport: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(runtime.get("openclaw_ws") or runtime.get("openclawWs"))
    connected = payload.get("connected")
    if connected is None:
        connected = _text(transport.get("name")).lower() == "openclaw_realtime" and _text(transport.get("state")).lower() == "connected"
    return {
        "connected": bool(connected),
        "url": _text(payload.get("url") or transport.get("url"), default=""),
        "lastError": _text(payload.get("last_error") or payload.get("lastError"), default=""),
        "lastRxMs": _number(payload.get("last_rx_ms") or payload.get("lastRxMs"), default=0)
        if payload.get("last_rx_ms") is not None or payload.get("lastRxMs") is not None
        else None,
        "lastTxMs": _number(payload.get("last_tx_ms") or payload.get("lastTxMs"), default=0)
        if payload.get("last_tx_ms") is not None or payload.get("lastTxMs") is not None
        else None,
        "sessionState": _text(payload.get("session_state") or payload.get("sessionState"), default="unknown"),
    }


def _openclaw_ws_degraded(transport: Mapping[str, Any], openclaw_ws: Mapping[str, Any]) -> bool:
    if _text(transport.get("name")).lower() != "openclaw_realtime" and not openclaw_ws.get("url"):
        return False
    if openclaw_ws.get("connected") is True:
        return False
    session_state = _text(openclaw_ws.get("sessionState")).lower()
    return session_state in {"unknown", "missing_url", "pending_transport_api", "auth_failed", "error", "failed"} or bool(
        openclaw_ws.get("lastError")
    )


def _normalize_component(value: Any) -> dict[str, Any]:
    component = dict(_mapping(value))
    if not component and value is not None:
        component["enabled"] = bool(value)
    return component


def _component_unavailable(value: Any) -> bool:
    component = _mapping(value)
    if component.get("enabled") is False:
        return True
    available = component.get("available")
    if available is False:
        return True
    state = _text(component.get("state") or component.get("status"))
    return state in {"unavailable", "missing", "disabled_by_platform"}


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _fill_ratio(depth: int, capacity: int) -> float:
    if capacity <= 0:
        return 0.0
    return depth / capacity


def _mapping(value: Any) -> Mapping[str, Any]:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return value
    return {}


def _text(value: Any, *, default: str = "") -> str:
    if value in (None, ""):
        return default
    return str(value)


def _number(value: Any, *, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, *, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
