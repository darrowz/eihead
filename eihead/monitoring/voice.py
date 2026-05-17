"""Voice diagnostics helpers for the eihead native monitor.

This module aggregates status for wired/unwired diagnostics, but it does not own
business policy. It reads diagnostics from runtime and mouth components and
normalizes them for monitoring output.

Mouth diagnostics here are playback-focused only; session and policy decisions
must be sourced from `eihead.eivoice_runtime` and not reconstructed from
mouth planning payloads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from eihead.monitoring.eivoice_runtime import build_eivoice_runtime_panel, eivoice_runtime_status_from_app
from eihead.monitoring.voice_readiness import build_voice_chain_readiness
from eihead.protocol import serialize_message


VOICE_REALTIME_SCHEMA = "eihead.monitor.voice_realtime.v1"
VOICE_RUNTIME_ATTRS = (
    "voice_realtime",
    "voice_status",
    "latest_voice_realtime",
    "latest_voice_status",
)


def build_voice_diagnostics_from_app(app: Any, timestamp: float) -> dict[str, Any]:
    """Read voice diagnostics from runtime hooks or a body snapshot."""

    candidates: list[dict[str, Any]] = []
    for attr_name in VOICE_RUNTIME_ATTRS:
        if not hasattr(app, attr_name):
            continue
        source = getattr(app, attr_name)
        raw_payload = _resolve_voice_candidate(source() if callable(source) else source)
        candidates.append(
            _build_voice_payload(
                raw_payload,
                timestamp=timestamp,
                source=attr_name,
                wired=raw_payload is not None,
            )
        )

    snapshot_payload = _voice_payload_from_snapshot(app)
    if snapshot_payload is not None:
        candidates.append(
            _build_voice_payload(
                snapshot_payload,
                timestamp=timestamp,
                source="snapshot",
                wired=True,
            )
        )

    eivoice_payload = _voice_payload_from_eivoice_runtime(eivoice_runtime_status_from_app(app))
    if eivoice_payload is not None:
        candidates.append(
            _build_voice_payload(
                eivoice_payload,
                timestamp=timestamp,
                source="eivoice_runtime",
                wired=True,
            )
        )

    if candidates:
        return max(candidates, key=_voice_payload_rank)

    return _build_voice_payload(None, timestamp=timestamp, source=None, wired=False)


def _build_voice_payload(
    observation: Any,
    *,
    timestamp: float,
    source: str | None,
    wired: bool | None,
) -> dict[str, Any]:
    resolved = _resolve_voice_candidate(observation)
    data = _mapping_payload(resolved) if resolved is not None else None
    ear = _normalize_ear(_mapping_from_keys(data, "ear"))
    mouth = _normalize_mouth(_mapping_from_keys(data, "mouth"))
    dialogue_source = _dialogue_mapping(data)
    dialogue = _normalize_dialogue(dialogue_source, root=data)
    realtime_audio = _realtime_audio_payload(data, dialogue_source)
    realtime_session = _realtime_session_payload(data, dialogue_source)
    round_info = _round_payload(data, dialogue_source, realtime_session)
    scheduler = _scheduler_payload(data, dialogue_source)
    cognition = _realtime_cognition_payloads(data, dialogue_source, realtime_session, scheduler)
    interruption = _interruption_payload(data, dialogue_source)
    microfeedback = _microfeedback_payload(data, dialogue_source)
    latency = _latency_payload(data, dialogue_source)
    _merge_component_latency(latency, ear=ear, mouth=mouth)
    streaming = _streaming_payload(data, dialogue_source, realtime_session)
    realtime_events = _realtime_events_payload(data, dialogue_source, realtime_session)
    event_count = _event_count_payload(data, dialogue_source, realtime_session, realtime_events)
    closed_loop_state = _closed_loop_state_payload(data, dialogue_source, realtime_session)
    last_reply_delta = _last_reply_delta_payload(data, dialogue_source, realtime_session, realtime_events)
    cancellation_chain = _cancellation_chain_payload(data, dialogue_source, realtime_session)
    bottleneck = _bottleneck_payload(data, dialogue_source)
    last_turn = _last_turn_payload(data, dialogue_source)
    voice_chain_readiness = _voice_chain_readiness_payload(data, dialogue_source)
    status, derived_wired, not_wired = _voice_overall_status(
        ear=ear,
        mouth=mouth,
        dialogue=dialogue,
        scheduler=scheduler,
        interruption=interruption,
        streaming=streaming,
        last_turn=last_turn,
        latency=latency,
    )
    _merge_realtime_latency(latency, data=data, dialogue=dialogue_source, realtime_session=realtime_session)
    optimization = _optimization_payload(
        data,
        dialogue_source,
        ear=ear,
        mouth=mouth,
        realtime_audio=realtime_audio,
        latency=latency,
        bottleneck=bottleneck,
    )
    is_wired = derived_wired if wired is None else bool(wired and derived_wired)
    readiness_message = _voice_readiness_message(
        data,
        ear=ear,
        mouth=mouth,
        dialogue=dialogue,
        scheduler=scheduler,
        interruption=interruption,
        streaming=streaming,
        status=status,
    )
    payload: dict[str, Any] = {
        "schema": VOICE_REALTIME_SCHEMA,
        "runtime": "eihead",
        "status": status,
        "wired": is_wired,
        "source": source,
        "channel": "voice.realtime",
        "aliases": ["audio.realtime"],
        "captured_at_ts": float(timestamp),
        "observation": data,
        "ear": ear,
        "mouth": mouth,
        "dialogue": dialogue,
        "realtime_audio": realtime_audio,
        "round": round_info,
        "scheduler": scheduler,
        "lanes": cognition["lanes"],
        "fast_think": cognition["fast_think"],
        "slow_reasoner": cognition["slow_reasoner"],
        "arbiter": cognition["arbiter"],
        "speech_action_plan": cognition["speech_action_plan"],
        "proactive_activity": cognition["proactive_activity"],
        "interruption": interruption,
        "streaming": streaming,
        "microfeedback": microfeedback,
        "latency": latency,
        "optimization": optimization,
        "realtime_session": realtime_session,
        "realtime_events": realtime_events,
        "event_count": event_count,
        "closed_loop_state": closed_loop_state,
        "last_reply_delta": last_reply_delta,
        "cancellation_chain": cancellation_chain,
        "bottleneck": bottleneck,
        "last_turn": last_turn,
        "voice_chain_readiness": voice_chain_readiness,
        "not_wired": bool(not_wired),
        "readiness_message": readiness_message,
    }
    if not is_wired and status == "not_wired" and not readiness_message:
        payload["readiness_message"] = "runtime app does not expose voice diagnostics"
    return payload


def _voice_payload_from_snapshot(app: Any) -> dict[str, Any] | None:
    snapshot_fn = getattr(app, "snapshot", None)
    if not callable(snapshot_fn):
        return None
    snapshot = snapshot_fn()
    if not isinstance(snapshot, Mapping):
        return None

    payload: dict[str, Any] = {}
    voice_dialogue = _mapping_from_keys(snapshot, "voice_dialogue", "dialogue")
    if voice_dialogue is not None:
        payload["voice_dialogue"] = voice_dialogue
    for key in (
        "current_round_id",
        "current_cancellation_token",
        "scheduler_state",
        "interrupt_count",
        "interrupted_round_count",
        "interrupt_active",
        "interruption",
        "last_interrupt",
        "microfeedback",
        "last_stage_latency_ms",
        "latency_ms",
        "realtime_session",
        "realtime_events",
        "events",
        "event_count",
        "closed_loop_state",
        "last_reply_delta",
        "cancellation_chain",
        "lanes",
        "fast_think",
        "slow_reasoner",
        "arbiter",
        "speech_action_plan",
        "proactive_activity",
    ):
        if key in snapshot:
            payload[key] = snapshot[key]
    organs = snapshot.get("organs")
    if isinstance(organs, Mapping):
        ear = _mapping_from_keys(organs, "ear")
        mouth = _mapping_from_keys(organs, "mouth")
        if ear is not None:
            payload["ear"] = ear
        if mouth is not None:
            payload["mouth"] = mouth
    return payload or None


def _voice_payload_from_eivoice_runtime(status: Mapping[str, Any]) -> dict[str, Any] | None:
    runtime = _json_mapping(status)
    if not runtime:
        return None
    panel = build_eivoice_runtime_panel(dict(runtime))
    audio_frontend = _json_mapping(panel.get("audioFrontend")) or {}
    transport = _json_mapping(panel.get("transport")) or {}
    warnings = [str(item) for item in panel.get("warnings", []) if item]
    asr_raw = runtime.get("asr") or runtime.get("asr_status") or runtime.get("recognition")
    asr = _json_mapping(asr_raw) if isinstance(asr_raw, Mapping) else {}
    mouth = _mouth_payload_from_eivoice_runtime(runtime)
    mouth_missing = mouth is None
    if mouth is None:
        mouth = {
            "status": "not_wired",
            "backend": "",
            "readiness_message": "mouth playback diagnostics are missing from eivoice runtime",
            "tts_playback": {
                "status": "not_wired",
                "reason": "mouth_playback_diagnostics_missing",
            },
        }
    capture_status = "degraded" if panel.get("health") == "degraded" else "ready"
    ear_status = _first_text(asr.get("status"), "ready")
    if panel.get("health") == "degraded":
        ear_status = "degraded"
    readiness_messages = warnings or ["eivoice runtime state is present"]
    if mouth_missing:
        readiness_messages.append("mouth playback diagnostics are missing")
    runtime_running = bool(runtime.get("running")) or panel.get("state") in {"running", "conversation"}
    transport_state = _first_text(transport.get("state"))
    realtime_audio_running = (
        transport_state == "connected"
        or (runtime_running and transport_state in {"", "unknown"})
    )
    return {
        "eivoice_runtime": {
            "state": panel.get("state"),
            "conversation_state": panel.get("conversationState"),
            "health": panel.get("health"),
            "warnings": warnings,
            "queues": panel.get("queues"),
            "transport": transport,
            "audio_frontend": audio_frontend,
        },
        "ear": {
            "status": ear_status,
            "provider": _first_text(asr.get("provider"), asr.get("backend")),
            "readiness_message": "; ".join(readiness_messages),
            "capture": {
                "status": capture_status,
                "details": {
                    "devices": audio_frontend.get("devices"),
                    "audio_format": audio_frontend.get("audioFormat"),
                    "aec": audio_frontend.get("aec"),
                    "ns": audio_frontend.get("ns"),
                    "vad": audio_frontend.get("vad"),
                    "loopback": audio_frontend.get("loopback"),
                },
            },
            "asr": {
                "status": ear_status,
                "details": asr,
            },
        },
        "mouth": mouth,
        "voice_dialogue": {
            "enabled": True,
            "running": panel.get("state") in {"running", "conversation"},
            "phase": panel.get("conversationState"),
            "last_status": panel.get("state"),
            "readiness_message": "; ".join(readiness_messages),
        },
        "realtime_audio": {
            "enabled": True,
            "running": realtime_audio_running,
            "transport": transport,
        },
        "streaming": {
            "state": "running" if transport.get("state") == "connected" else transport.get("state"),
            "transport": transport,
        },
        "readiness_message": "; ".join(readiness_messages),
    }


def _mouth_payload_from_eivoice_runtime(runtime: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("mouth", "mouth_status", "tts_playback", "playback"):
        raw_payload = runtime.get(key)
        payload = _json_mapping(raw_payload) if isinstance(raw_payload, Mapping) else None
        if payload:
            if key in {"tts_playback", "playback"}:
                return {
                    "status": _first_text(payload.get("status"), payload.get("state")),
                    "backend": _first_text(payload.get("backend"), payload.get("provider")),
                    "model": _first_text(payload.get("model")),
                    "voice_id": _first_text(payload.get("voice_id"), payload.get("voiceId")),
                    "tts_playback": payload,
                }
            return payload
    return None


def _voice_payload_rank(payload: Mapping[str, Any]) -> tuple[int, int]:
    status = _normalized_text(payload.get("status"))
    if payload.get("wired") is True and status == "wired":
        return (400, 0)
    if payload.get("not_wired") is False and status in {"degraded", "stale"}:
        return (300, 0)
    if payload.get("wired") is True:
        return (250, 0)
    if payload.get("not_wired") is False:
        return (200, 0)
    if payload.get("observation") is not None:
        return (100, 0)
    return (0, 0)


def _normalize_ear(raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    capture = _subfunction(raw, "capture")
    asr = _subfunction(raw, "asr")
    provider = _first_text(
        raw.get("provider"),
        _details_value(asr, "provider"),
        _details_value(asr, "backend"),
    )
    status = _status_text(raw, fallback=capture or asr)
    readiness = _first_text(
        raw.get("readiness_message"),
        raw.get("message"),
        _details_value(asr, "reason"),
        _details_value(capture, "reason"),
        _details_value(asr, "status"),
        _details_value(capture, "status"),
    )
    return {
        "status": status,
        "health": _text_or_none(raw.get("health")),
        "state": _classify_component_state(raw, fallback_status=status, role="ear"),
        "provider": provider,
        "live_probe_skipped": _truthy(raw.get("live_probe_skipped")) or _details_truthy(asr, "live_probe_skipped"),
        "readiness_message": readiness,
        "stage_latency_ms": _ear_stage_latency_ms(raw, capture=capture, asr=asr),
        "capture": capture,
        "asr": asr,
    }


def _normalize_mouth(raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    playback = _subfunction(raw, "tts_playback")
    # Mouth status is playback-oriented only. Planning/policy details in
    # tts_plan are intentionally treated as non-authoritative diagnostic cargo.
    plan = _subfunction(raw, "tts_plan")
    backend = _first_text(
        raw.get("backend"),
        raw.get("provider"),
        _details_value(playback, "backend"),
        _details_value(playback, "provider"),
    )
    status = _status_text(raw, fallback=playback)
    readiness = _first_text(
        raw.get("readiness_message"),
        raw.get("message"),
        _details_value(playback, "reason"),
        _details_value(playback, "status"),
    )
    return {
        "status": status,
        "health": _text_or_none(raw.get("health")),
        "state": _classify_component_state(raw, fallback_status=status, role="mouth"),
        "backend": backend,
        "model": _first_text(raw.get("model"), _details_value(playback, "model")),
        "voice_id": _first_text(
            raw.get("voice_id"),
            _details_value(playback, "voice_id"),
        ),
        "text_preview": _first_text(raw.get("text_preview"), _details_value(playback, "text_preview")),
        "readiness_message": readiness,
        "busy": _truthy(_first_value(raw, "busy")) or _details_truthy(playback, "busy"),
        "playback_state": _playback_state(raw, playback=playback, status=status),
        "stage_latency_ms": _mouth_stage_latency_ms(raw, playback=playback),
        "stop": _stop_payload(raw, playback=playback),
        "tts_playback": playback,
        "tts_plan": plan,
    }


def _normalize_dialogue(raw: Mapping[str, Any] | None, *, root: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if raw is None and root is None:
        return None
    mapping = raw or {}
    phase = _first_text(mapping.get("phase"), root.get("phase") if root else None)
    last_status = _first_text(mapping.get("last_status"), root.get("last_status") if root else None)
    status = last_status or phase or _status_text(mapping)
    readiness = _first_text(
        mapping.get("readiness_message"),
        mapping.get("message"),
        root.get("readiness_message") if root else None,
        root.get("message") if root else None,
    )
    payload: dict[str, Any] = {
        "phase": phase,
        "last_status": last_status,
        "state": _classify_component_state(mapping, fallback_status=status, role="dialogue"),
        "enabled": _truthy(mapping.get("enabled")),
        "running": _truthy(mapping.get("running")),
        "last_transcript": _first_text(mapping.get("last_transcript"), root.get("last_transcript") if root else None),
        "last_reply": _first_text(mapping.get("last_reply"), root.get("last_reply") if root else None),
        "readiness_message": readiness,
    }
    last_error = _first_text(mapping.get("last_error"), root.get("last_error") if root else None)
    if last_error or "last_error" in mapping:
        payload["last_error"] = last_error
    turn_count = _first_value(mapping, "turn_count", "utterance_count")
    if turn_count is None:
        turn_count = _first_value(root, "turn_count", "utterance_count")
    if turn_count is not None:
        payload["turn_count"] = _json_ready(turn_count)
    current_round_id = _first_value(mapping, "current_round_id", "round_id")
    if current_round_id is None:
        current_round_id = _first_value(root, "current_round_id", "round_id")
    if current_round_id is not None:
        payload["current_round_id"] = _json_ready(current_round_id)
    stage_latency = _mapping_from_keys(mapping, "last_stage_latency_ms")
    if stage_latency is None:
        stage_latency = _mapping_from_keys(root, "last_stage_latency_ms")
    if stage_latency is not None:
        payload["last_stage_latency_ms"] = stage_latency
    engine = _mapping_from_keys(mapping, "dialogue", "dialogue_engine", "engine")
    if engine is not None:
        payload["dialogue"] = engine
    for key in ("conversation_active", "wake_word_required", "wake_words", "end_phrases", "last_gate_reason"):
        value = _first_value(mapping, key)
        if value is None:
            value = _first_value(root, key) if root else None
        if value is not None:
            payload[key] = _json_ready(value)
    return payload


def _realtime_audio_payload(
    data: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw = _first_value(dialogue, "realtime_audio", "realtimeAudio")
    if raw is None:
        raw = _first_value(data, "realtime_audio", "realtimeAudio")
    if isinstance(raw, Mapping):
        payload = _json_mapping(raw)
        payload.setdefault("enabled", False)
        payload.setdefault("running", False)
        return payload
    return {"enabled": False, "running": False}


def _round_payload(
    data: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    realtime_session: Mapping[str, Any] | None,
) -> dict[str, Any]:
    round_id = _first_value(dialogue, "current_round_id", "round_id")
    if round_id is None:
        round_id = _first_value(data, "current_round_id", "round_id")
    if round_id is None:
        round_id = _first_value(realtime_session, "current_round_id", "round_id", "roundId")
    cancellation_token = _first_value(dialogue, "current_cancellation_token", "cancellation_token")
    if cancellation_token is None:
        cancellation_token = _first_value(data, "current_cancellation_token", "cancellation_token")
    if cancellation_token is None:
        cancellation_token = _first_value(realtime_session, "current_cancellation_token", "cancellation_token", "cancellationToken")
    phase = _first_text(
        _first_value(dialogue, "phase"),
        _first_value(data, "phase"),
        _first_value(realtime_session, "phase"),
    )
    last_status = _first_text(
        _first_value(dialogue, "last_status", "status"),
        _first_value(data, "last_status", "status"),
        _first_value(realtime_session, "last_status", "status"),
    )
    has_round = round_id not in (None, "")
    has_token = cancellation_token not in (None, "")
    normalized_status = _normalized_text(last_status)
    interrupted = normalized_status in {"interrupted", "interrupt", "cancelled", "canceled"} or _truthy(
        _first_value(dialogue, "interrupted")
    )
    complete = _truthy(_first_value(realtime_session, "complete")) or normalized_status in {
        "completed",
        "complete",
        "done",
        "finished",
    }
    lifecycle = (
        "interrupted"
        if interrupted
        else "completed"
        if complete
        else "active"
        if has_round
        else "unknown"
    )
    return {
        "current_round_id": _json_ready(round_id) if has_round else None,
        "current_cancellation_token": _json_ready(cancellation_token) if has_token else None,
        "has_cancellation_token": bool(has_token),
        "phase": phase,
        "last_status": last_status,
        "active": lifecycle == "active",
        "complete": bool(complete),
        "interrupted": bool(interrupted),
        "lifecycle": lifecycle,
        "state": lifecycle,
    }


def _scheduler_payload(data: Mapping[str, Any] | None, dialogue: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = _first_value(dialogue, "scheduler_state", "scheduler")
    if raw is None:
        raw = _first_value(data, "scheduler_state", "scheduler")
    details = _json_ready(raw) if raw is not None else None
    payload: dict[str, Any] = {}
    if isinstance(details, Mapping):
        payload.update(_json_mapping(details))
        state = _first_text(
            payload.get("state"),
            payload.get("status"),
            payload.get("phase"),
        )
    else:
        state = _first_text(details)
        if details is not None:
            payload["value"] = details
    stale = _truthy(payload.get("stale")) or _normalized_text(state) == "stale"
    not_wired = _truthy(payload.get("not_wired")) or _normalized_text(state) in {
        "not_wired",
        "missing",
        "unavailable",
        "disabled",
        "offline",
    }
    component_state = _scheduler_component_state(state=state, stale=stale, not_wired=not_wired)
    payload["state"] = state or ("not_wired" if not_wired else "unknown")
    payload["component_state"] = component_state
    payload["wired"] = component_state == "wired"
    payload["not_wired"] = component_state == "not_wired"
    payload["stale"] = bool(stale)
    return payload


def _scheduler_component_state(*, state: str, stale: bool, not_wired: bool) -> str:
    normalized = _normalized_text(state)
    if not_wired:
        return "not_wired"
    if stale or normalized in {"stale", "blocked", "error", "failed", "unhealthy"}:
        return "degraded"
    if normalized in {"ok", "healthy", "ready", "running", "active", "scheduled", "idle"}:
        return "wired"
    return "unknown"


def _realtime_cognition_payloads(
    data: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    realtime_session: Mapping[str, Any] | None,
    scheduler: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    lanes_source = _first_mapping(data, dialogue, realtime_session, scheduler, keys=("lanes", "lane_states", "scheduler_lanes"))
    fast_think = _lane_payload(
        _first_mapping(
            lanes_source,
            scheduler,
            dialogue,
            data,
            realtime_session,
            keys=("fast_think", "fastThink", "fast_lane", "fast"),
        ),
        default_state="unknown",
    )
    slow_reasoner = _lane_payload(
        _first_mapping(
            lanes_source,
            scheduler,
            dialogue,
            data,
            realtime_session,
            keys=("slow_reasoner", "slowReasoner", "slow_lane", "slow", "slow_thinking"),
        ),
        default_state="unknown",
    )
    arbiter = _lane_payload(
        _first_mapping(
            lanes_source,
            scheduler,
            dialogue,
            data,
            realtime_session,
            keys=("arbiter", "response_arbiter"),
        ),
        default_state="unknown",
    )
    lanes = _lanes_payload(fast_think=fast_think, slow_reasoner=slow_reasoner, arbiter=arbiter)
    speech_action_plan = _speech_action_plan_payload(
        _first_mapping(
            scheduler,
            dialogue,
            data,
            realtime_session,
            keys=("speech_action_plan", "speechActionPlan", "speech_plan"),
        )
    )
    proactive_activity = _proactive_activity_payload(
        _first_mapping(
            scheduler,
            dialogue,
            data,
            realtime_session,
            keys=("proactive_activity", "proactiveActivity", "activity_proposal", "activity"),
        )
    )
    return {
        "lanes": lanes,
        "fast_think": fast_think,
        "slow_reasoner": slow_reasoner,
        "arbiter": arbiter,
        "speech_action_plan": speech_action_plan,
        "proactive_activity": proactive_activity,
    }


def _lane_payload(raw: Mapping[str, Any] | None, *, default_state: str) -> dict[str, Any]:
    if raw is None:
        return _missing_realtime_component(default_state)
    payload = _json_mapping(raw)
    state = _first_text(payload.get("state"), payload.get("status"), payload.get("phase"), default_state)
    normalized = _normalized_text(state)
    if normalized in {"not_wired", "missing", "unavailable", "disabled", "offline"}:
        component_state = "not_wired"
    elif normalized in {"stale", "blocked", "error", "failed", "unhealthy"}:
        component_state = "degraded"
    elif normalized in {"unknown", ""}:
        component_state = "unknown"
    else:
        component_state = "wired"
    latency_ms = _float_or_none(_first_value(payload, "latency_ms", "latencyMs", "elapsed_ms", "elapsedMs"))
    summary = state if latency_ms is None else f"{state} ({latency_ms}ms)"
    payload.update(
        {
            "state": state,
            "status": state,
            "component_state": component_state,
            "wired": component_state == "wired",
            "not_wired": component_state == "not_wired",
            "summary": summary,
        }
    )
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    return payload


def _lanes_payload(
    *,
    fast_think: Mapping[str, Any],
    slow_reasoner: Mapping[str, Any],
    arbiter: Mapping[str, Any],
) -> dict[str, Any]:
    states = [
        str(component.get("component_state") or "unknown")
        for component in (fast_think, slow_reasoner, arbiter)
    ]
    if any(state == "wired" for state in states):
        component_state = "wired"
    elif any(state == "degraded" for state in states):
        component_state = "degraded"
    elif any(state == "not_wired" for state in states):
        component_state = "not_wired"
    else:
        component_state = "unknown"
    return {
        "fast_think": dict(fast_think),
        "slow_reasoner": dict(slow_reasoner),
        "arbiter": dict(arbiter),
        "component_state": component_state,
        "wired": component_state == "wired",
        "not_wired": component_state == "not_wired",
        "summary": " / ".join(
            f"{name}={component.get('state', 'unknown')}"
            for name, component in (
                ("fast_think", fast_think),
                ("slow_reasoner", slow_reasoner),
                ("arbiter", arbiter),
            )
        ),
    }


def _speech_action_plan_payload(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return _missing_realtime_component("not_wired")
    payload = _json_mapping(raw)
    plan_id = _first_text(payload.get("planId"), payload.get("plan_id"), payload.get("id"), "unknown")
    speech_segments = _first_list(payload, "speechSegments", "speech_segments", "speech")
    action_segments = _first_list(payload, "actionSegments", "action_segments", "actions", "action_plan")
    state = _first_text(payload.get("state"), payload.get("status"), "ready")
    payload.update(
        {
            "plan_id": plan_id,
            "state": state,
            "status": state,
            "component_state": "wired",
            "wired": True,
            "not_wired": False,
            "speech_count": len(speech_segments),
            "action_count": len(action_segments),
            "summary": f"{plan_id}: {len(speech_segments)} speech, {len(action_segments)} {_plural('action', len(action_segments))}",
        }
    )
    return payload


def _proactive_activity_payload(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return _missing_realtime_component("not_wired")
    payload = _json_mapping(raw)
    proposal_id = _first_text(payload.get("proposalId"), payload.get("proposal_id"), payload.get("id"), "unknown")
    channel = _first_text(payload.get("channel"), "unknown")
    should_emit = _truthy(_first_value(payload, "shouldEmit", "should_emit"))
    state = _first_text(payload.get("state"), payload.get("status"), "proposed")
    payload.update(
        {
            "proposal_id": proposal_id,
            "channel": channel,
            "should_emit": should_emit,
            "state": state,
            "status": state,
            "component_state": "wired",
            "wired": True,
            "not_wired": False,
            "summary": f"{proposal_id}: {channel} / {'emit' if should_emit else 'hold'}",
        }
    )
    return payload


def _missing_realtime_component(state: str) -> dict[str, Any]:
    return {
        "state": state,
        "status": state,
        "component_state": state,
        "wired": False,
        "not_wired": state == "not_wired",
        "summary": state,
    }


def _plural(word: str, count: int) -> str:
    return word if count == 1 else f"{word}s"


def _interruption_payload(data: Mapping[str, Any] | None, dialogue: Mapping[str, Any] | None) -> dict[str, Any]:
    interrupt_count = _int_or_none(
        _first_value(dialogue, "interrupt_count", "interruption_count", "interrupts")
    )
    if interrupt_count is None:
        interrupt_count = _int_or_none(_first_value(data, "interrupt_count", "interruption_count", "interrupts"))
    interrupted_round_count = _int_or_none(_first_value(dialogue, "interrupted_round_count"))
    if interrupted_round_count is None:
        interrupted_round_count = _int_or_none(_first_value(data, "interrupted_round_count"))
    last_interrupt_raw = _first_value(dialogue, "last_interrupt", "interruption", "interrupt")
    if last_interrupt_raw is None:
        last_interrupt_raw = _first_value(data, "last_interrupt", "interruption", "interrupt")
    last_interrupt = _json_ready(last_interrupt_raw) if last_interrupt_raw is not None else None
    last_status = _normalized_text(_first_text(_first_value(dialogue, "last_status"), _first_value(data, "last_status")))
    interrupt_active_raw = _first_value(dialogue, "interrupt_active")
    if interrupt_active_raw is None:
        interrupt_active_raw = _first_value(data, "interrupt_active")
    interrupt_active = _truthy(interrupt_active_raw)
    stale = _truthy(_mapping_value(last_interrupt, "stale")) or _truthy(_first_value(dialogue, "interrupt_stale"))
    if not stale:
        stale = _truthy(_first_value(data, "interrupt_stale"))
    interrupted = (
        interrupt_active
        or _truthy(_first_value(dialogue, "interrupted"))
        or _truthy(_first_value(data, "interrupted"))
        or last_status in {"interrupted", "interrupt", "cancelled", "canceled"}
    )
    has_history = bool(
        last_interrupt is not None
        or (interrupt_count is not None and interrupt_count > 0)
        or (interrupted_round_count is not None and interrupted_round_count > 0)
    )
    has_interrupt_signal = bool(
        interrupt_active_raw is not None
        or interrupt_count is not None
        or interrupted_round_count is not None
        or last_interrupt is not None
        or _first_value(dialogue, "interrupted") is not None
        or _first_value(data, "interrupted") is not None
    )
    no_interrupts_seen = (
        has_interrupt_signal
        and
        interrupt_active is False
        and not interrupted
        and last_interrupt is None
        and (interrupt_count == 0 or interrupt_count is None)
        and (interrupted_round_count == 0 or interrupted_round_count is None)
    )
    state = (
        "stale"
        if stale
        else "interrupted"
        if interrupted
        else "history"
        if has_history
        else "clear"
        if no_interrupts_seen
        else "unknown"
    )
    component_state = "degraded" if stale or interrupted else "wired" if state == "clear" else "unknown"
    return {
        "state": state,
        "status": state,
        "component_state": component_state,
        "active": bool(interrupt_active),
        "interrupted": bool(interrupted),
        "stale": bool(stale),
        "clear": state == "clear",
        "has_history": bool(has_history),
        "interrupt_count": interrupt_count,
        "interrupted_round_count": interrupted_round_count,
        "last_interrupt": last_interrupt,
    }


def _microfeedback_payload(data: Mapping[str, Any] | None, dialogue: Mapping[str, Any] | None) -> Any:
    raw = _first_value(dialogue, "microfeedback", "micro_feedback")
    if raw is None:
        raw = _first_value(data, "microfeedback", "micro_feedback")
    return _json_ready(raw) if raw is not None else None


def _latency_payload(data: Mapping[str, Any] | None, dialogue: Mapping[str, Any] | None) -> dict[str, Any]:
    stage_latency = _mapping_from_keys(dialogue, "last_stage_latency_ms") or _mapping_from_keys(data, "last_stage_latency_ms")
    stage_latency_ms: dict[str, float] = {}
    if stage_latency is not None:
        for key, value in stage_latency.items():
            number = _float_or_none(value)
            if number is not None:
                stage_latency_ms[str(key)] = number
    latency_seconds = _mapping_from_keys(dialogue, "last_latency_s") or _mapping_from_keys(data, "last_latency_s")
    latency_s: dict[str, float] = {}
    if latency_seconds is not None:
        for key, value in latency_seconds.items():
            number = _float_or_none(value)
            if number is not None:
                latency_s[str(key)] = number
                stage_latency_ms.setdefault(str(key), round(number * 1000.0, 3))
    total_ms = _float_or_none(_first_value(data, "last_total_latency_ms", "total_latency_ms"))
    if total_ms is None:
        total_ms = stage_latency_ms.get("total")
    if total_ms is None and stage_latency_ms:
        total_ms = round(
            sum(
                value
                for key, value in stage_latency_ms.items()
                if key not in {"total", "overhead"}
            ),
            3,
        )
    return {
        "total_ms": total_ms,
        "stage_latency_ms": stage_latency_ms,
        "stage_latency_s": latency_s,
    }


def _merge_component_latency(
    latency: dict[str, Any],
    *,
    ear: Mapping[str, Any] | None,
    mouth: Mapping[str, Any] | None,
) -> None:
    stage_latency_ms = latency.setdefault("stage_latency_ms", {})
    if not isinstance(stage_latency_ms, dict):
        return
    for component in (ear, mouth):
        component_latency = _mapping_from_keys(component, "stage_latency_ms")
        if component_latency is None:
            continue
        for key, value in component_latency.items():
            number = _float_or_none(value)
            if number is not None:
                stage_latency_ms.setdefault(str(key), number)
    if latency.get("total_ms") is None and stage_latency_ms:
        total_ms = _float_or_none(stage_latency_ms.get("total"))
        if total_ms is None:
            total_ms = round(
                sum(
                    value
                    for key, value in stage_latency_ms.items()
                    if key not in {"total", "overhead", "tts_total"}
                ),
                3,
            )
        latency["total_ms"] = total_ms


def _ear_stage_latency_ms(
    raw: Mapping[str, Any],
    *,
    capture: Mapping[str, Any] | None,
    asr: Mapping[str, Any] | None,
) -> dict[str, float]:
    latencies = _stage_latency_ms_from(raw)
    _set_latency(latencies, "vad", _first_float(raw, "vad_elapsed_ms", "vad_latency_ms"))
    _set_latency(latencies, "capture", _first_float(raw, "capture_elapsed_ms"))
    _set_latency(latencies, "asr", _first_float(raw, "decode_elapsed_ms", "asr_decode_elapsed_ms", "asr_elapsed_ms"))
    _set_latency(latencies, "vad", _first_details_float(capture, "vad_elapsed_ms", "vad_latency_ms"))
    _set_latency(latencies, "capture", _first_details_float(capture, "capture_elapsed_ms", "elapsed_ms"))
    _set_latency(latencies, "asr", _first_details_float(asr, "decode_elapsed_ms", "asr_decode_elapsed_ms", "asr_elapsed_ms", "elapsed_ms"))
    return latencies


def _mouth_stage_latency_ms(raw: Mapping[str, Any], *, playback: Mapping[str, Any] | None) -> dict[str, float]:
    latencies = _stage_latency_ms_from(raw)
    _set_latency(latencies, "tts_synthesis", _first_float(raw, "synthesis_elapsed_ms"))
    _set_latency(latencies, "tts_playback", _first_float(raw, "playback_elapsed_ms"))
    _set_latency(latencies, "tts_total", _first_float(raw, "total_elapsed_ms"))
    _set_latency(latencies, "tts_synthesis", _first_details_float(playback, "synthesis_elapsed_ms"))
    _set_latency(latencies, "tts_playback", _first_details_float(playback, "playback_elapsed_ms"))
    _set_latency(latencies, "tts_total", _first_details_float(playback, "total_elapsed_ms"))
    return latencies


def _stage_latency_ms_from(raw: Mapping[str, Any]) -> dict[str, float]:
    latencies: dict[str, float] = {}
    explicit = _mapping_from_keys(raw, "stage_latency_ms")
    if explicit is not None:
        for key, value in explicit.items():
            number = _float_or_none(value)
            if number is not None:
                latencies[str(key)] = number
    latency = _mapping_from_keys(raw, "latency")
    nested = _mapping_from_keys(latency, "stage_latency_ms") if latency is not None else None
    if nested is not None:
        for key, value in nested.items():
            number = _float_or_none(value)
            if number is not None:
                latencies.setdefault(str(key), number)
    return latencies


def _set_latency(latencies: dict[str, float], key: str, value: float | None) -> None:
    if value is not None:
        latencies.setdefault(key, value)


def _realtime_session_payload(
    data: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    raw = _first_value(data, "realtime_session", "realtime_voice_session", "voice_session")
    if raw is None:
        raw = _first_value(dialogue, "realtime_session", "realtime_voice_session", "voice_session")
    session = _mapping_payload(_resolve_voice_candidate(raw)) if raw is not None else None
    if session is not None:
        return session
    for candidate in (data, dialogue):
        if isinstance(candidate, Mapping) and _looks_like_realtime_session(candidate):
            return _json_mapping(candidate)
    return None


def _looks_like_realtime_session(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in ("session_id", "actor_id", "closed_loop_state", "cancellation_chain")) and any(
        key in value for key in ("round_id", "roundId", "cancellation_token", "cancellationToken", "events")
    )


def _realtime_events_payload(
    data: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    realtime_session: Mapping[str, Any] | None,
) -> list[Any]:
    raw = _first_value(data, "realtime_events")
    if raw is None:
        raw = _first_value(dialogue, "realtime_events")
    if raw is None:
        raw = _first_value(realtime_session, "events", "realtime_events")
    if raw is None and _looks_like_realtime_session(data or {}):
        raw = _first_value(data, "events")
    return _json_list(raw)


def _event_count_payload(
    data: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    realtime_session: Mapping[str, Any] | None,
    realtime_events: list[Any],
) -> int:
    raw = _first_value(data, "event_count", "realtime_event_count")
    if raw is None:
        raw = _first_value(dialogue, "event_count", "realtime_event_count")
    if raw is None:
        raw = _first_value(realtime_session, "event_count", "realtime_event_count")
    count = _int_or_none(raw)
    if count is not None:
        return count
    return len(realtime_events)


def _streaming_payload(
    data: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    realtime_session: Mapping[str, Any] | None,
) -> dict[str, Any]:
    llm = _normalize_streaming_component(
        _first_mapping(data, dialogue, realtime_session, keys=("streaming_llm", "llm_stream", "llm")),
        role="llm",
    )
    tts = _normalize_streaming_component(
        _first_mapping(data, dialogue, realtime_session, keys=("streaming_tts", "tts_stream", "tts")),
        role="tts",
    )
    states = [llm["component_state"], tts["component_state"]]
    if any(state == "not_wired" for state in states):
        component_state = "not_wired"
    elif any(state == "degraded" for state in states):
        component_state = "degraded"
    elif states and all(state == "wired" for state in states):
        component_state = "wired"
    else:
        component_state = "unknown"
    return {
        "llm": llm,
        "tts": tts,
        "component_state": component_state,
        "wired": component_state == "wired",
        "not_wired": component_state == "not_wired",
    }


def _normalize_streaming_component(raw: Mapping[str, Any] | None, *, role: str) -> dict[str, Any]:
    if raw is None:
        return {
            "status": "unknown",
            "state": "unknown",
            "component_state": "unknown",
            "wired": False,
            "not_wired": False,
            "readiness_message": f"streaming {role} status is unknown",
        }
    status = _first_text(raw.get("status"), raw.get("state"), raw.get("phase"), "unknown")
    component_state = _classify_component_state(raw, fallback_status=status, role=f"streaming_{role}")
    return {
        **_json_mapping(raw),
        "status": status,
        "state": component_state,
        "component_state": component_state,
        "wired": component_state == "wired",
        "not_wired": component_state == "not_wired",
        "readiness_message": _first_text(
            raw.get("readiness_message"),
            raw.get("message"),
            f"streaming {role} status is {component_state}",
        ),
    }


def _first_mapping(*sources: Mapping[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any] | None:
    for source in sources:
        mapping = _mapping_from_keys(source, *keys)
        if mapping is not None:
            return mapping
    return None


def _closed_loop_state_payload(
    data: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    realtime_session: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    raw = _mapping_from_keys(data, "closed_loop_state")
    if raw is None:
        raw = _mapping_from_keys(dialogue, "closed_loop_state")
    if raw is None:
        raw = _mapping_from_keys(realtime_session, "closed_loop_state")
    return raw


def _last_reply_delta_payload(
    data: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    realtime_session: Mapping[str, Any] | None,
    realtime_events: list[Any],
) -> str:
    explicit = _first_text(
        _first_value(data, "last_reply_delta", "reply_delta"),
        _first_value(dialogue, "last_reply_delta", "reply_delta"),
        _first_value(realtime_session, "last_reply_delta", "reply_delta"),
    )
    if explicit:
        return explicit
    for event in reversed(realtime_events):
        if not isinstance(event, Mapping):
            continue
        delta = _first_text(event.get("reply_delta"), event.get("delta"))
        if delta:
            return delta
    return ""


def _cancellation_chain_payload(
    data: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    realtime_session: Mapping[str, Any] | None,
) -> list[Any]:
    raw = _first_value(data, "cancellation_chain")
    if raw is None:
        raw = _first_value(dialogue, "cancellation_chain")
    if raw is None:
        raw = _first_value(realtime_session, "cancellation_chain")
    return _json_list(raw)


def _merge_realtime_latency(
    latency: dict[str, Any],
    *,
    data: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    realtime_session: Mapping[str, Any] | None,
) -> None:
    stage_latency_ms = latency.setdefault("stage_latency_ms", {})
    if not isinstance(stage_latency_ms, dict):
        return
    for source in (data, dialogue, realtime_session):
        realtime_latency = _mapping_from_keys(source, "latency_ms", "realtime_latency_ms")
        if realtime_latency is None:
            continue
        for key, value in realtime_latency.items():
            number = _float_or_none(value)
            if number is not None:
                stage_latency_ms.setdefault(str(key), number)
    if latency.get("total_ms") is None:
        total_ms = _float_or_none(stage_latency_ms.get("total"))
        if total_ms is not None:
            latency["total_ms"] = total_ms


def _bottleneck_payload(data: Mapping[str, Any] | None, dialogue: Mapping[str, Any] | None) -> dict[str, Any] | None:
    stage = _first_text(
        _first_value(dialogue, "last_bottleneck_stage"),
        _first_value(data, "last_bottleneck_stage"),
    )
    latency_ms = _float_or_none(_first_value(dialogue, "last_bottleneck_ms"))
    if latency_ms is None:
        latency_ms = _float_or_none(_first_value(data, "last_bottleneck_ms"))
    if not stage and latency_ms is None:
        return None
    return {
        "stage": stage or None,
        "latency_ms": latency_ms,
    }


def _optimization_payload(
    data: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    *,
    ear: Mapping[str, Any] | None,
    mouth: Mapping[str, Any] | None,
    realtime_audio: Mapping[str, Any] | None,
    latency: Mapping[str, Any] | None,
    bottleneck: Mapping[str, Any] | None,
) -> dict[str, Any]:
    latency_ms = _latency_ms_payload(latency)
    return {
        "latency_ms": latency_ms,
        "bottleneck": _optimization_bottleneck(latency_ms, bottleneck),
        "wakeword": _optimization_wakeword(data, dialogue),
        "dialogue": _optimization_dialogue(dialogue),
        "dialogue_engine": _optimization_dialogue_engine(dialogue),
        "asr": _optimization_asr(ear),
        "tts": _optimization_tts(mouth),
        "realtime_audio": _optimization_realtime_audio(realtime_audio),
    }


def _latency_ms_payload(latency: Mapping[str, Any] | None) -> dict[str, float]:
    stage_latency = _mapping_from_keys(latency, "stage_latency_ms")
    if stage_latency is None:
        return {}
    payload: dict[str, float] = {}
    for key, value in stage_latency.items():
        number = _float_or_none(value)
        if number is not None:
            payload[str(key)] = number
    return payload


def _optimization_bottleneck(
    latency_ms: Mapping[str, float],
    explicit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stage = _first_text(_first_value(explicit, "stage"))
    latency_value = _float_or_none(_first_value(explicit, "latency_ms", "latencyMs"))
    if stage or latency_value is not None:
        return {"stage": stage or None, "latency_ms": latency_value}
    candidates = {
        key: value
        for key, value in latency_ms.items()
        if key not in {"total", "overhead"} and value is not None
    }
    if not candidates:
        return {"stage": None, "latency_ms": None}
    stage, latency_value = max(candidates.items(), key=lambda item: item[1])
    return {"stage": stage, "latency_ms": latency_value}


def _optimization_wakeword(data: Mapping[str, Any] | None, dialogue: Mapping[str, Any] | None) -> dict[str, Any]:
    wakeword = _mapping_from_keys(data, "wakeword", "wake_word")
    if wakeword is None:
        runtime = _mapping_from_keys(data, "eivoice_runtime")
        wakeword = _mapping_from_keys(runtime, "wakeword", "wake_word")
    return {
        "enabled": _first_value(wakeword, "enabled") if wakeword is not None else _first_value(dialogue, "wake_word_required"),
        "state": _first_text(
            _first_value(wakeword, "state"),
            "active" if _truthy(_first_value(dialogue, "conversation_active")) else "",
        ),
        "conversation_active": _truthy(_first_value(dialogue, "conversation_active")),
        "last_gate_reason": _first_text(
            _first_value(wakeword, "last_gate_reason"),
            _first_value(dialogue, "last_gate_reason"),
        ),
    }


def _optimization_dialogue(dialogue: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "phase": _first_text(_first_value(dialogue, "phase")),
        "last_status": _first_text(_first_value(dialogue, "last_status")),
        "turn_count": _json_ready(_first_value(dialogue, "turn_count")),
        "current_round_id": _json_ready(_first_value(dialogue, "current_round_id", "round_id")),
        "last_transcript": _first_text(_first_value(dialogue, "last_transcript")),
        "last_reply": _first_text(_first_value(dialogue, "last_reply")),
    }


def _optimization_dialogue_engine(dialogue: Mapping[str, Any] | None) -> dict[str, Any]:
    engine = _mapping_from_keys(dialogue, "dialogue", "dialogue_engine", "engine") or {}
    return {
        "provider": _first_text(_first_value(engine, "provider")),
        "event_name": _first_text(_first_value(engine, "event_name", "eventName")),
        "round_id": _json_ready(_first_value(engine, "round_id", "roundId")),
        "returncode": _json_ready(_first_value(engine, "returncode")),
        "elapsed_ms": _float_or_none(_first_value(engine, "elapsed_ms", "elapsedMs", "latency_ms", "latencyMs")),
    }


def _optimization_asr(ear: Mapping[str, Any] | None) -> dict[str, Any]:
    asr = _mapping_from_keys(ear, "asr") or {}
    diagnostics = _mapping_from_keys(asr, "provider_diagnostics", "details") or {}
    return {
        "provider": _first_text(_first_value(ear, "provider"), _first_value(asr, "provider"), _first_value(diagnostics, "provider")),
        "state": _first_text(_first_value(asr, "provider_state", "state", "status"), _first_value(diagnostics, "state")),
        "final_count": _json_ready(_first_value(asr, "final_count", "finalCount")),
        "last_decode_ms": _float_or_none(_first_value(diagnostics, "last_decode_ms", "lastDecodeMs", "decode_elapsed_ms")),
        "last_error": _first_text(_first_value(diagnostics, "last_error", "lastError")),
    }


def _optimization_tts(mouth: Mapping[str, Any] | None) -> dict[str, Any]:
    playback = _mapping_from_keys(mouth, "tts_playback") or {}
    details = _mapping_from_keys(playback, "details") or {}
    stage_latency = _mapping_from_keys(mouth, "stage_latency_ms") or {}
    return {
        "backend": _first_text(_first_value(mouth, "backend"), _first_value(details, "provider")),
        "model": _first_text(_first_value(mouth, "model"), _first_value(details, "model")),
        "voice_id": _first_text(_first_value(mouth, "voice_id"), _first_value(details, "voice_id", "voiceId")),
        "playback_state": _first_text(_first_value(mouth, "playback_state"), _first_value(playback, "status")),
        "speak_ms": _float_or_none(_first_value(stage_latency, "speak")),
        "playback_elapsed_ms": _float_or_none(_first_value(details, "playback_elapsed_ms", "playbackElapsedMs")),
    }


def _optimization_realtime_audio(realtime_audio: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "enabled": _json_ready(_first_value(realtime_audio, "enabled")),
        "running": _json_ready(_first_value(realtime_audio, "running")),
        "audio_level": _float_or_none(_first_value(realtime_audio, "audio_level", "rms")),
        "rms_dbfs": _float_or_none(_first_value(realtime_audio, "rms_dbfs", "rmsDbfs")),
        "vad_triggered": _json_ready(_first_value(realtime_audio, "vad_triggered", "vadTriggered")),
        "captured_ms": _float_or_none(_first_value(realtime_audio, "captured_ms", "capturedMs")),
        "voice_ms": _float_or_none(_first_value(realtime_audio, "voice_ms", "voiceMs")),
        "silence_after_voice_ms": _float_or_none(
            _first_value(realtime_audio, "silence_after_voice_ms", "silenceAfterVoiceMs")
        ),
    }


def _last_turn_payload(data: Mapping[str, Any] | None, dialogue: Mapping[str, Any] | None) -> dict[str, Any] | None:
    explicit = (
        _mapping_from_keys(data, "last_turn")
        or _mapping_from_keys(dialogue, "last_completed_turn")
        or _mapping_from_keys(dialogue, "last_turn")
    )
    if explicit is not None:
        return explicit
    transcript = _first_text(
        _first_value(dialogue, "last_transcript"),
        _first_value(data, "last_transcript"),
    )
    reply = _first_text(
        _first_value(dialogue, "last_reply"),
        _first_value(data, "last_reply"),
    )
    if not transcript and not reply:
        return None
    payload: dict[str, Any] = {}
    if transcript:
        payload["transcript"] = transcript
    if reply:
        payload["reply"] = reply
    status = _first_text(
        _first_value(dialogue, "last_status", "phase"),
        _first_value(data, "last_status", "phase"),
    )
    if status:
        payload["status"] = status
    return payload


def _voice_chain_readiness_payload(
    data: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
) -> dict[str, Any]:
    explicit = _first_mapping(
        data,
        dialogue,
        keys=("voice_chain_readiness", "voiceChainReadiness"),
    )
    benchmark = _first_mapping(
        data,
        dialogue,
        keys=("voice_chain_benchmark", "voiceChainBenchmark"),
    )
    scenario_targets = _first_mapping(
        data,
        dialogue,
        keys=("scenarioTargets", "voice_chain_scenarios", "voiceChainScenarios"),
    )
    return build_voice_chain_readiness(
        {
            "explicit": explicit,
            "benchmark": benchmark,
            "scenario_targets": scenario_targets,
        }
    )


def _voice_overall_status(
    *,
    ear: Mapping[str, Any] | None,
    mouth: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    scheduler: Mapping[str, Any] | None,
    interruption: Mapping[str, Any] | None,
    streaming: Mapping[str, Any] | None,
    last_turn: Mapping[str, Any] | None,
    latency: Mapping[str, Any] | None,
) -> tuple[str, bool, bool]:
    states = [
        str(component.get("component_state") or component.get("state", "") or "")
        for component in (ear, mouth, dialogue, scheduler, interruption, streaming)
        if isinstance(component, Mapping)
    ]
    has_signal = bool(
        any(state and state != "unknown" for state in states)
        or last_turn
        or (latency and latency.get("stage_latency_ms"))
    )
    if not has_signal:
        return "not_wired", False, True
    if any(state == "wired" for state in states):
        if any(state in {"degraded", "not_wired"} for state in states):
            return "degraded", False, False
        return "wired", True, False
    if any(state == "degraded" for state in states):
        return "degraded", False, False
    if any(state == "not_wired" for state in states):
        return "not_wired", False, True
    return "unknown", False, False


def _voice_readiness_message(
    data: Mapping[str, Any] | None,
    *,
    ear: Mapping[str, Any] | None,
    mouth: Mapping[str, Any] | None,
    dialogue: Mapping[str, Any] | None,
    scheduler: Mapping[str, Any] | None,
    interruption: Mapping[str, Any] | None,
    streaming: Mapping[str, Any] | None,
    status: str,
) -> str:
    explicit = _first_text(
        _first_value(data, "readiness_message"),
        _first_value(data, "message"),
    )
    messages = [explicit] if explicit else []
    for name, component in (
        ("ear", ear),
        ("mouth", mouth),
        ("dialogue", dialogue),
        ("scheduler", scheduler),
        ("interruption", interruption),
        ("streaming", streaming),
    ):
        if not isinstance(component, Mapping):
            continue
        component_status = component.get("status")
        if _normalized_text(component_status) == "unknown":
            component_status = None
        text = _first_text(
            component.get("readiness_message"),
            component.get("message"),
            _state_readiness_text(component),
            component_status,
        )
        if text and text not in messages:
            messages.append(f"{name}: {text}")
    if messages:
        return "; ".join(messages)
    if status == "not_wired":
        return "voice diagnostics are not wired"
    if status == "degraded":
        return "voice diagnostics are degraded"
    if status == "unknown":
        return "voice diagnostics are present but incomplete"
    return ""


def _state_readiness_text(component: Mapping[str, Any]) -> str:
    state = _normalized_text(component.get("state"))
    component_state = _normalized_text(component.get("component_state"))
    labels: list[str] = []
    if _truthy(component.get("interrupted")) or state == "interrupted":
        labels.append("interrupted")
    if _truthy(component.get("stale")) or state == "stale":
        labels.append("stale")
    if _truthy(component.get("not_wired")) or component_state == "not_wired" or state == "not_wired":
        labels.append("not_wired")
    if labels:
        return "/".join(labels)
    if component_state == "degraded":
        return state or "degraded"
    return ""


def _resolve_voice_candidate(payload: Any, *, seen: set[int] | None = None) -> Any:
    if payload is None:
        return None
    seen = seen or set()
    candidate_id = id(payload)
    if candidate_id in seen:
        return payload
    seen.add(candidate_id)

    latest_status = getattr(payload, "latest_status", None)
    if latest_status is not None:
        resolved = _resolve_voice_candidate(latest_status, seen=seen)
        if resolved is not None:
            return resolved

    for method_name in ("status", "poll"):
        method = getattr(payload, method_name, None)
        if not callable(method):
            continue
        try:
            resolved = _resolve_voice_candidate(method(), seen=seen)
        except TypeError:
            continue
        if resolved is not None:
            return resolved
    return payload


def _mapping_payload(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        return _json_mapping(payload)
    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        data = payload.to_dict()
        if isinstance(data, Mapping):
            return _json_mapping(data)
    if is_dataclass(payload):
        return _json_mapping(asdict(payload))
    try:
        serialized = serialize_message(payload)
    except TypeError:
        return None
    return _json_mapping(serialized) if isinstance(serialized, Mapping) else None


def _json_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_ready(value) for key, value in mapping.items()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return _json_mapping(payload)
    return str(value)


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        return [_json_ready(value)]
    try:
        iterator = iter(value)
    except TypeError:
        return [_json_ready(value)]
    return [_json_ready(item) for item in iterator]


def _first_list(mapping: Mapping[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, list):
            return list(value)
    return []


def _mapping_from_keys(mapping: Mapping[str, Any] | None, *keys: str) -> dict[str, Any] | None:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Mapping):
            return _json_mapping(value)
    return None


def _dialogue_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(mapping, Mapping):
        return None
    if isinstance(mapping.get("dialogue"), Mapping):
        return _json_mapping(mapping["dialogue"])
    if isinstance(mapping.get("voice_dialogue"), Mapping):
        return _json_mapping(mapping["voice_dialogue"])
    dialogue_keys = {
        "enabled",
        "running",
        "phase",
        "last_status",
        "last_transcript",
        "last_reply",
        "last_error",
        "turn_count",
        "utterance_count",
        "last_completed_turn",
        "last_stage_latency_ms",
        "last_latency_s",
        "dialogue",
        "last_bottleneck_stage",
        "last_bottleneck_ms",
        "current_round_id",
        "current_cancellation_token",
        "scheduler_state",
        "interrupt_count",
        "interrupted_round_count",
        "interrupt_active",
        "interruption",
        "last_interrupt",
        "microfeedback",
        "lanes",
        "fast_think",
        "slow_reasoner",
        "arbiter",
        "speech_action_plan",
        "proactive_activity",
    }
    if any(key in mapping for key in dialogue_keys):
        return _json_mapping({key: mapping[key] for key in dialogue_keys if key in mapping})
    return None


def _subfunction(raw: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    subfunctions = raw.get("subfunctions")
    if isinstance(subfunctions, Mapping) and isinstance(subfunctions.get(name), Mapping):
        return _json_mapping(subfunctions[name])
    if isinstance(raw.get(name), Mapping):
        return _json_mapping(raw[name])
    return None


def _details_value(subfunction: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(subfunction, Mapping):
        return None
    details = subfunction.get("details")
    if isinstance(details, Mapping):
        return details.get(key)
    return None


def _details_truthy(subfunction: Mapping[str, Any] | None, key: str) -> bool:
    value = _details_value(subfunction, key)
    return _truthy(value)


def _playback_state(raw: Mapping[str, Any], *, playback: Mapping[str, Any] | None, status: str) -> str:
    explicit = _first_text(raw.get("playback_state"), _details_value(playback, "playback_state"))
    if explicit:
        return explicit
    busy = _truthy(_first_value(raw, "busy")) or _details_truthy(playback, "busy")
    if busy:
        return "busy"
    stop = _stop_payload(raw, playback=playback)
    if isinstance(stop, Mapping) and _truthy(stop.get("success")):
        return "stopped"
    normalized = _normalized_text(status)
    if normalized in {"stopped", "cancelled", "canceled"}:
        return "stopped"
    if normalized in {"completed", "done", "finished"}:
        return "completed"
    if normalized in {"error", "failed"}:
        return "error"
    return "idle"


def _stop_payload(raw: Mapping[str, Any], *, playback: Mapping[str, Any] | None) -> dict[str, Any] | None:
    for source in (raw, _details_mapping(playback)):
        for key in ("stop", "last_stop", "stop_speech"):
            value = source.get(key)
            if isinstance(value, Mapping):
                return _json_mapping(value)
    return None


def _details_mapping(subfunction: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(subfunction, Mapping):
        return {}
    details = subfunction.get("details")
    return _json_mapping(details) if isinstance(details, Mapping) else {}


def _status_text(raw: Mapping[str, Any], *, fallback: Mapping[str, Any] | None = None) -> str:
    return _first_text(
        raw.get("status"),
        raw.get("health"),
        fallback.get("status") if fallback else None,
        fallback.get("health") if fallback else None,
    )


def _classify_component_state(raw: Mapping[str, Any], *, fallback_status: str, role: str) -> str:
    status = _normalized_text(fallback_status)
    if not status:
        status = _normalized_text(raw.get("status") or raw.get("health"))
    health = _normalized_text(raw.get("health"))
    data_status = _normalized_text(raw.get("data_status"))
    backend = _normalized_text(raw.get("backend") or raw.get("provider"))
    if _truthy(raw.get("not_wired")) or status in {"not_wired", "offline", "missing", "unavailable", "disabled"}:
        return "not_wired"
    if status == "noop" or backend == "noop":
        return "not_wired"
    if health in {"not_wired", "offline", "missing", "unavailable", "disabled"}:
        return "not_wired"
    if health in {"degraded", "error", "failed", "unhealthy"} or data_status in {"compat", "fallback"}:
        return "degraded"
    if health in {"online", "live"} or data_status == "live":
        return "wired"
    if _truthy(raw.get("live_probe_skipped")) or status in {
        "waiting",
        "waiting_for_data",
        "warming_up",
        "pending",
        "no_data",
    }:
        return "degraded"
    if status in {"degraded", "error", "failed", "unhealthy"}:
        return "degraded"
    if role == "dialogue" and status in {"idle", "waiting_for_voice", "sleeping", "dormant"}:
        return "unknown"
    if role == "mouth" and status in {"idle", "playing", "synthesizing", "completed", "stopped"} and backend:
        return "wired"
    if status in {"ok", "healthy", "ready", "running", "active", "listening", "thinking", "speaking", "completed"}:
        return "wired"
    return "unknown"


def _first_value(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _mapping_value(mapping: Any, key: str) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return None


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text_or_none(value)
        if text:
            return text
    return ""


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_text(value: Any) -> str:
    text = _text_or_none(value)
    return text.lower() if text is not None else ""


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(mapping: Mapping[str, Any] | None, *keys: str) -> float | None:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        number = _float_or_none(mapping.get(key))
        if number is not None:
            return number
    return None


def _first_details_float(subfunction: Mapping[str, Any] | None, *keys: str) -> float | None:
    details = _details_mapping(subfunction)
    return _first_float(details, *keys)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


__all__ = [
    "VOICE_REALTIME_SCHEMA",
    "build_voice_diagnostics_from_app",
]
