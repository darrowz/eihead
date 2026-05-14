"""Adapters from legacy eibrain protocol objects to eiprotocol envelopes.

This bridge owns protocol adaptation from legacy, mixed-shape payloads into the
typed eiprotocol envelope model. Upstream runtime code should prefer these
helpers and keep event construction policy in one place.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from eiprotocol import (
    EventEnvelope,
    PolicyState,
    SourceRef,
    TargetRef,
    build_dialogue_cancellation_applied_event,
    build_dialogue_fast_hypothesis_event,
    build_dialogue_stable_decision_event,
    build_emotion_context_event,
    build_event,
    build_head_status_report_event,
    build_memory_prefetch_requested_event,
    build_proactive_activity_proposed_event,
    build_speech_action_plan_event,
    build_vision_frame_event,
    EventIdFactory,
)

from .capabilities import (
    CapabilityManifest as LegacyCapabilityManifest,
    HeadBackend,
    HeadDevice,
    HeadHealth,
    HeadLimit,
    build_modality_inventory,
)
from .head import (
    AudioTurn as LegacyAudioTurn,
    ExecutionOutcome as LegacyExecutionOutcome,
    HeadAction as LegacyHeadAction,
    VisionObservation as LegacyVisionObservation,
)


DEFAULT_EVENT_TIME = "1970-01-01T00:00:00.000Z"


def to_eiprotocol_event(
    message: object,
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Convert a supported legacy protocol message into an eiprotocol event."""

    if isinstance(message, Mapping):
        return payload_to_eiprotocol_event(
            message,
            event_id=event_id,
            request_id=request_id,
            sequence=sequence,
            time=time,
        )
    if isinstance(message, LegacyCapabilityManifest):
        return capability_manifest_to_eiprotocol_event(
            message,
            event_id=event_id,
            request_id=request_id,
            sequence=sequence,
            time=time,
        )
    if isinstance(message, LegacyAudioTurn):
        return audio_turn_to_eiprotocol_event(
            message,
            event_id=event_id,
            request_id=request_id,
            sequence=sequence,
            time=time,
        )
    if isinstance(message, LegacyVisionObservation):
        return vision_observation_to_eiprotocol_event(
            message,
            event_id=event_id,
            request_id=request_id,
            sequence=sequence,
            time=time,
        )
    if isinstance(message, LegacyHeadAction):
        return head_action_to_eiprotocol_event(
            message,
            event_id=event_id,
            request_id=request_id,
            sequence=sequence,
            time=time,
        )
    if isinstance(message, LegacyExecutionOutcome):
        return execution_outcome_to_eiprotocol_event(
            message,
            event_id=event_id,
            request_id=request_id,
            sequence=sequence,
            time=time,
        )

    raise TypeError(f"Unsupported eiprotocol bridge message: {type(message).__name__}")


def payload_to_eiprotocol_event(
    payload: Mapping[str, Any],
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Convert an explicit internal payload kind/name into an eiprotocol event."""
    kind = _first_text(payload.get("name"), payload.get("event_name"), payload.get("kind"), payload.get("type"))
    normalized_kind = str(kind).strip().lower()
    handler = _EVENT_KIND_ROUTING.get(normalized_kind)
    if handler is None:
        raise TypeError(f"Unsupported eiprotocol bridge payload kind: {kind or 'unknown'}")
    return handler(
        payload,
        event_id=event_id,
        request_id=request_id,
        sequence=sequence,
        time=time,
    )


def head_status_report_to_eiprotocol_event(
    payload: Mapping[str, Any],
    *,
    source: str | SourceRef | Mapping[str, Any] | None = None,
    target: str | TargetRef | Mapping[str, Any] | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Normalize an eihead /status-style payload into a typed head status event."""

    raw = _coerce_mapping(payload, label="head status payload")
    node_id = _first_text(raw.get("node_id"), raw.get("head_id"), raw.get("id"), fallback="honjia")
    trace_id = _first_text(raw.get("trace_id"), raw.get("traceId"))
    reported_at = _reported_at(raw, fallback=time)
    resolved_event_id = _resolve_event_id(event_id, "head_status_report", node_id, trace_id, reported_at)
    return build_head_status_report_event(
        source=_source_like(source, fallback=_first_text(raw.get("source"), f"eihead.{node_id}"), device_id=node_id),
        target=_target_like(target, fallback=_first_text(raw.get("target"))),
        status=_status_value(raw),
        components=_status_components(raw),
        reported_at=reported_at,
        summary=_status_summary(raw),
        metadata=_status_metadata(raw),
        event_id=resolved_event_id,
        request_id=_first_text(request_id, trace_id, resolved_event_id),
        sequence=_positive_sequence(raw, sequence),
        time=time or reported_at,
        trace_id=trace_id,
        ttl_ms=_optional_int(raw.get("ttlMs", raw.get("ttl_ms")), fallback=2000),
        mode=_dict_from(raw.get("mode")),
    )


def dialogue_fast_hypothesis_to_eiprotocol_event(
    payload: Mapping[str, Any],
    *,
    source: str | SourceRef | Mapping[str, Any] | None = None,
    target: str | TargetRef | Mapping[str, Any] | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Normalize an eibrain fast dialogue hypothesis payload into an event."""

    raw = _coerce_mapping(payload, label="dialogue fast hypothesis payload")
    trace_id = _first_text(raw.get("trace_id"), raw.get("traceId"))
    hypothesis_id = _first_text(raw.get("hypothesisId"), raw.get("hypothesis_id"))
    resolved_event_id = _resolve_event_id(event_id, "dialogue_fast_hypothesis", hypothesis_id, trace_id, raw.get("text"))
    basis_event_id = _first_text(raw.get("basisEventId"), raw.get("basis_event_id"))
    return build_dialogue_fast_hypothesis_event(
        source=_source_like(source, fallback=_first_text(raw.get("source"), fallback="eibrain.honxin"), device_id=""),
        target=_target_like(target, fallback=_first_text(raw.get("target"))),
        hypothesis_id=hypothesis_id,
        text=_first_text(raw.get("text")),
        confidence=_optional_float(raw.get("confidence")),
        basis_event_id=basis_event_id,
        latency_ms=_optional_float(raw.get("latencyMs", raw.get("latency_ms"))),
        metadata=_dict_from(raw.get("metadata")),
        event_id=resolved_event_id,
        request_id=_first_text(request_id, trace_id, resolved_event_id),
        sequence=_positive_sequence(raw, sequence),
        time=time or DEFAULT_EVENT_TIME,
        session_id=_first_text(raw.get("sessionId"), raw.get("session_id")),
        round_id=_first_text(raw.get("roundId"), raw.get("round_id")),
        correlation_id=_first_text(raw.get("correlationId"), raw.get("correlation_id"), basis_event_id),
        causation_id=_first_text(raw.get("causationId"), raw.get("causation_id"), basis_event_id),
        trace_id=trace_id,
        ttl_ms=_optional_int(raw.get("ttlMs", raw.get("ttl_ms")), fallback=800),
        mode=_dict_from(raw.get("mode")),
    )


def dialogue_stable_decision_to_eiprotocol_event(
    payload: Mapping[str, Any],
    *,
    source: str | SourceRef | Mapping[str, Any] | None = None,
    target: str | TargetRef | Mapping[str, Any] | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Normalize an eibrain stable dialogue decision payload into an event."""

    raw = _coerce_mapping(payload, label="dialogue stable decision payload")
    trace_id = _first_text(raw.get("trace_id"), raw.get("traceId"))
    decision_id = _first_text(raw.get("decisionId"), raw.get("decision_id"))
    resolved_event_id = _resolve_event_id(event_id, "dialogue_stable_decision", decision_id, trace_id)
    return build_dialogue_stable_decision_event(
        source=_source_like(source, fallback=_first_text(raw.get("source"), fallback="eibrain.honxin"), device_id=""),
        target=_target_like(target, fallback=_first_text(raw.get("target"))),
        decision_id=decision_id,
        decision_value=_first_text(raw.get("decision")),
        confidence=_optional_float(raw.get("confidence")),
        text=_first_text(raw.get("text")),
        actions=_list_of_dicts(raw.get("actions")),
        stable_since_ms=_optional_float(raw.get("stableSinceMs", raw.get("stable_since_ms"))),
        metadata=_dict_from(raw.get("metadata")),
        event_id=resolved_event_id,
        request_id=_first_text(request_id, trace_id, resolved_event_id),
        sequence=_positive_sequence(raw, sequence),
        time=time or DEFAULT_EVENT_TIME,
        session_id=_first_text(raw.get("sessionId"), raw.get("session_id")),
        round_id=_first_text(raw.get("roundId"), raw.get("round_id")),
        correlation_id=_first_text(raw.get("correlationId"), raw.get("correlation_id")),
        causation_id=_first_text(raw.get("causationId"), raw.get("causation_id")),
        trace_id=trace_id,
        ttl_ms=_optional_int(raw.get("ttlMs", raw.get("ttl_ms")), fallback=3000),
        mode=_dict_from(raw.get("mode")),
    )


def scheduler_snapshot_to_eiprotocol_events(
    snapshot: object,
    *,
    source: str | SourceRef | Mapping[str, Any] | None = None,
    target: str | TargetRef | Mapping[str, Any] | None = None,
    session_id: str = "",
    ids: EventIdFactory | None = None,
    sequence_start: int = 1,
    time: str | None = None,
) -> list[EventEnvelope]:
    """Convert a scheduler snapshot into stable realtime cognition events.

    The converter consumes only JSON-like snapshot keys so it can accept current
    and future scheduler implementations without importing cognition classes.
    """

    raw = _snapshot_mapping(snapshot)
    if not raw:
        return []
    current = _mapping_from_any(raw.get("current")) or _mapping_from_any(raw.get("turn")) or {}
    scheduler = _mapping_from_any(raw.get("scheduler")) or _mapping_from_any(raw.get("scheduler_state")) or {}
    round_id = _first_text(
        current.get("round_id"),
        current.get("roundId"),
        raw.get("current_round_id"),
        raw.get("round_id"),
        raw.get("roundId"),
    )
    cancellation_token = _first_text(
        current.get("cancellation_token"),
        current.get("cancellationToken"),
        raw.get("current_cancellation_token"),
        raw.get("cancellation_token"),
        raw.get("cancellationToken"),
    )
    brain_source = _source_like(source, fallback="eibrain.honxin")
    brain_target = _target_from_source_like(source, fallback="eibrain.honxin")
    head_source = _source_from_target_like(target, fallback="eihead.honjia")
    head_target = _target_like(target, fallback="eihead.honjia")
    memory_source = _source_like("eimemory.memoria", fallback="eimemory.memoria")
    memory_target = _target_like("eimemory.memoria", fallback="eimemory.memoria")
    common = {
        "session_id": session_id or _first_text(raw.get("session_id"), raw.get("sessionId")),
        "round_id": round_id,
        "ids": ids,
        "time": time or DEFAULT_EVENT_TIME,
    }

    events: list[EventEnvelope] = []
    sequence = max(1, int(sequence_start))

    emotion = _first_mapping(
        current,
        raw,
        scheduler,
        keys=("emotion_state", "emotion_context", "emotion"),
    )
    if emotion is not None:
        events.append(
            build_emotion_context_event(
                source=head_source,
                target=brain_target,
                context=emotion,
                context_id=_first_text(
                    emotion.get("contextId"),
                    emotion.get("context_id"),
                    f"{round_id}:emotion" if round_id else "emotion",
                ),
                mood=_first_text(emotion.get("mood"), emotion.get("state"), "unknown"),
                confidence=_optional_float(emotion.get("confidence")) or 0.0,
                sequence=sequence,
                **common,
            )
        )
        sequence += 1

    for index, prefetch in enumerate(_prefetch_items(raw=raw, current=current, scheduler=scheduler)):
        query = _first_text(prefetch.get("query"), prefetch.get("text"), prefetch.get("summary"))
        if not query:
            continue
        events.append(
            build_memory_prefetch_requested_event(
                source=brain_source,
                target=memory_target,
                prefetch=prefetch,
                prefetch_id=_first_text(
                    prefetch.get("prefetchId"),
                    prefetch.get("prefetch_id"),
                    prefetch.get("id"),
                    f"{round_id}:prefetch:{index}" if round_id else f"prefetch:{index}",
                ),
                query=query,
                reason=_first_text(prefetch.get("reason"), prefetch.get("source"), "scheduler_prefetch"),
                sequence=sequence,
                **common,
            )
        )
        sequence += 1

    for trace_index, trace in enumerate(_memory_trace_items(raw=raw, current=current, scheduler=scheduler)):
        trace_round_id = _first_text(trace.get("roundId"), trace.get("round_id"), round_id)
        trace_session_id = _first_text(trace.get("sessionId"), trace.get("session_id"), common["session_id"])
        trace_schema = _first_text(trace.get("schema"), trace.get("trace_schema"))
        trace_id = _first_text(trace.get("traceId"), trace.get("trace_id"), f"{trace_round_id}:memory:{trace_index}" if trace_round_id else "")
        recall = _mapping_from_any(trace.get("recall")) or {}
        recall_items = _mapping_items(recall.get("items"))
        for recall_index, recall_item in enumerate(recall_items):
            query = _first_text(recall_item.get("query"), recall_item.get("text"), recall_item.get("summary"))
            results = _memory_trace_selected_records(recall_item)
            if not query and not results:
                continue
            events.append(
                build_event(
                    ids=ids,
                    name="ei.memory.recall.result",
                    event_type="memory",
                    source=memory_source,
                    target=brain_target,
                    content={
                        "query": query,
                        "resultCount": _optional_int(recall_item.get("selectedCount", recall_item.get("selected_count")), fallback=len(results)),
                        "results": results,
                        "metadata": {
                            "traceSchema": trace_schema,
                            "traceRoundId": trace_round_id,
                            "summary": _first_text(recall_item.get("summary")),
                            "selectedCount": _optional_int(recall_item.get("selectedCount", recall_item.get("selected_count")), fallback=len(results)),
                            "sourceComposition": _dict_from(recall_item.get("sourceComposition", recall_item.get("source_composition"))),
                            "errors": _mapping_items(trace.get("errors")),
                            "recallItem": recall_item,
                        },
                    },
                    sequence=sequence,
                    session_id=trace_session_id,
                    round_id=trace_round_id,
                    trace_id=trace_id,
                    correlation_id=_first_text(recall_item.get("correlationId"), recall_item.get("correlation_id"), trace_id),
                    causation_id=_first_text(recall_item.get("causationId"), recall_item.get("causation_id")),
                    time=time or DEFAULT_EVENT_TIME,
                    priority="normal",
                    ttl_ms=5000,
                )
            )
            sequence += 1

        writeback = _mapping_from_any(trace.get("writeback")) or {}
        writeback_items = _mapping_items(writeback.get("items"))
        for write_index, write_item in enumerate(writeback_items):
            if _first_text(write_item.get("status")) == "skipped":
                continue
            memory_id = _memory_trace_writeback_id(write_item)
            if not memory_id:
                continue
            events.append(
                build_event(
                    ids=ids,
                    name="ei.memory.write.committed",
                    event_type="memory",
                    source=memory_source,
                    target=brain_target,
                    content={
                        "memoryId": memory_id,
                        "traceSchema": trace_schema,
                        "traceRoundId": trace_round_id,
                        "status": _first_text(write_item.get("status"), "ok"),
                        "summary": _first_text(write_item.get("summary")),
                        "source": _first_text(write_item.get("source")),
                        "memoryType": _first_text(write_item.get("memoryType"), write_item.get("memory_type"), write_item.get("type")),
                        "writebackIndex": write_index,
                        "writeback": write_item,
                    },
                    sequence=sequence,
                    session_id=trace_session_id,
                    round_id=trace_round_id,
                    trace_id=trace_id,
                    correlation_id=_first_text(write_item.get("correlationId"), write_item.get("correlation_id"), trace_id),
                    causation_id=_first_text(write_item.get("causationId"), write_item.get("causation_id")),
                    time=time or DEFAULT_EVENT_TIME,
                    priority="normal",
                    ttl_ms=5000,
                )
            )
            sequence += 1

    plan = _first_mapping(
        current,
        raw,
        scheduler,
        keys=("speech_action_plan", "speechActionPlan", "speech_plan", "plan"),
    )
    if plan is not None:
        if not any(plan.get(key) for key in ("action_segments", "actionSegments", "action_plan", "actionPlan", "actions")):
            action_plan = current.get("action_plan") or raw.get("action_plan") or scheduler.get("action_plan")
            if isinstance(action_plan, list):
                plan = {**dict(plan), "action_segments": action_plan, "action_plan": action_plan}
        events.append(
            build_speech_action_plan_event(
                source=brain_source,
                target=head_target,
                plan=plan,
                plan_id=_first_text(
                    plan.get("planId"),
                    plan.get("plan_id"),
                    plan.get("id"),
                    f"{round_id}:speech_action_plan" if round_id else "speech_action_plan",
                ),
                stable=bool(plan.get("stable", False)),
                sequence=sequence,
                **common,
            )
        )
        sequence += 1

    activity = _first_mapping(
        raw,
        current,
        scheduler,
        keys=("proactive_activity", "proactiveActivity", "activity", "activity_proposal"),
    )
    if activity is not None and _activity_should_emit(activity):
        events.append(
            build_proactive_activity_proposed_event(
                source=brain_source,
                target=head_target,
                proposal=activity,
                proposal_id=_first_text(
                    activity.get("proposalId"),
                    activity.get("proposal_id"),
                    activity.get("id"),
                    f"{round_id}:activity" if round_id else "activity",
                ),
                channel=_first_text(activity.get("channel"), "silent"),
                reason=_first_text(activity.get("reason"), "unspecified"),
                should_emit=_truthy(activity.get("shouldEmit", activity.get("should_emit"))),
                sequence=sequence,
                **common,
            )
        )
        sequence += 1

    cancellation = _first_mapping(
        raw,
        current,
        scheduler,
        keys=("cancellation", "cancellation_applied", "last_interrupt", "interrupt"),
    )
    if cancellation is not None and _cancellation_is_applied(cancellation):
        cancelled_round_id = _first_text(
            cancellation.get("cancelledRoundId"),
            cancellation.get("cancelled_round_id"),
            cancellation.get("round_id"),
            cancellation.get("roundId"),
        )
        token = _first_text(
            cancellation.get("cancellationToken"),
            cancellation.get("cancellation_token"),
            cancellation_token,
        )
        reason = _first_text(cancellation.get("reason"), cancellation.get("type"), "cancelled")
        if cancelled_round_id and token and reason:
            events.append(
                build_dialogue_cancellation_applied_event(
                    source=brain_source,
                    target=head_target,
                    cancellation=cancellation,
                    cancellation_id=_first_text(
                        cancellation.get("cancellationId"),
                        cancellation.get("cancellation_id"),
                        cancellation.get("id"),
                        f"{cancelled_round_id}:cancelled",
                    ),
                    cancelled_round_id=cancelled_round_id,
                    cancellation_token=token,
                    reason=reason,
                    sequence=sequence,
                    **common,
                )
            )

    return events


def capability_manifest_to_eiprotocol_event(
    manifest: LegacyCapabilityManifest,
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Wrap a legacy capability manifest in an eiprotocol manifest event."""

    modalities = build_modality_inventory(manifest.devices, manifest.backends)
    content = {
        "manifestId": _first_text(manifest.node_id, manifest.trace_id, manifest.source, fallback="manifest"),
        "manifestVersion": manifest.protocol_version or "head.v1",
        "device": {
            "nodeId": manifest.node_id,
            "nodeRole": manifest.node_role,
            "source": manifest.source,
            "target": manifest.target,
            "timestampMs": manifest.timestamp_ms,
            "devices": [_device_to_capability_metadata(device) for device in manifest.devices],
        },
        "runtime": {
            "nodeRole": manifest.node_role,
            "protocolVersion": manifest.protocol_version or "head.v1",
            "timestampMs": manifest.timestamp_ms,
        },
        "transports": _manifest_transports(manifest),
        "modalities": modalities,
        "capabilities": [_device_to_capability(device) for device in manifest.devices],
        "backends": [_backend_to_capability(backend) for backend in manifest.backends],
        "health": _health_to_dict(manifest.health),
        "limits": {},
        "metadata": {
            **dict(manifest.metadata),
            "legacyCapabilities": list(manifest.capabilities),
            "modalitySummary": {name: {"available": value["available"], "count": _modality_count(value)} for name, value in modalities.items()},
        },
    }
    return _event(
        manifest,
        event_type="capability",
        name="ei.capability.manifest.report",
        content=content,
        priority="normal",
        event_id=_resolve_event_id(event_id, "capability_manifest", manifest.node_id, manifest.trace_id),
        request_id=request_id,
        sequence=sequence,
        time=time,
        source_device_id=manifest.node_id,
        round_scoped=False,
    )


def _manifest_transports(manifest: LegacyCapabilityManifest) -> dict[str, Any]:
    metadata: Mapping[str, Any] | None
    if isinstance(manifest.metadata, Mapping):
        metadata = manifest.metadata
        metadata_transports = manifest.metadata.get("transports")
        if isinstance(metadata_transports, Mapping):
            return _copy_jsonish(dict(metadata_transports))
    else:
        metadata = None
    host = _first_text(manifest.node_id, manifest.source, fallback="honjia")
    port = 18081
    endpoint = None
    if isinstance(metadata, Mapping):
        endpoint = metadata.get("runtime")
        if not isinstance(endpoint, Mapping):
            endpoint = metadata.get("monitoring", metadata.get("monitor"))
    if isinstance(endpoint, Mapping):
        try:
            port = int(endpoint.get("port", port))
        except (TypeError, ValueError):
            port = 18081
    return {
        "http": {"baseUrl": f"http://{host}.local:{port}"},
        "websocket": {"path": "/events"},
    }


def audio_turn_to_eiprotocol_event(
    turn: LegacyAudioTurn,
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Wrap a legacy audio ASR turn in an eiprotocol dialogue event."""

    legacy_payload = dict(turn.payload)
    content = {
        "text": turn.text,
        "language": turn.language,
        "final": bool(turn.is_final),
        "confidence": turn.confidence,
        "startMs": turn.start_ms,
        "endMs": turn.end_ms,
        "audioLevel": turn.audio_level,
        "wakeWord": turn.wake_word,
        "asrBackend": _first_text(legacy_payload.get("asrBackend"), legacy_payload.get("asr_backend")),
        "timingsMs": _dict_from(legacy_payload.get("timingsMs") or legacy_payload.get("timings_ms")),
        "metadata": {
            "legacyPayload": legacy_payload,
            "observationType": turn.observation_type,
            "status": turn.status,
        },
    }
    return _event(
        turn,
        event_type="dialogue",
        name="ei.dialogue.asr.final" if turn.is_final else "ei.dialogue.asr.partial",
        content=content,
        priority="realtime",
        event_id=_resolve_event_id(event_id, "audio_turn", turn.trace_id, turn.text),
        request_id=request_id,
        sequence=sequence,
        time=time,
        source_device_id=turn.device_id,
        round_scoped=True,
    )


def realtime_vision_payload_to_eiprotocol_event(
    payload: Mapping[str, Any],
    *,
    source: str | SourceRef | Mapping[str, Any] | None = None,
    target: str | TargetRef | Mapping[str, Any] | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Normalize JSON-like realtime vision output into a typed vision-frame event."""

    raw = _coerce_mapping(payload, label="realtime vision payload")
    trace_id = _first_text(raw.get("trace_id"), raw.get("traceId"))
    frame_id = _first_text(raw.get("frameId"), raw.get("frame_id"), raw.get("frame"), raw.get("id"), fallback="frame")
    resolved_event_id = _resolve_event_id(event_id, "realtime_vision_frame", frame_id, trace_id)
    detections = _vision_detections(raw)
    return build_vision_frame_event(
        source=_source_like(
            source,
            fallback=_first_text(raw.get("source"), fallback="eihead.honjia"),
            device_id=_first_text(raw.get("deviceId"), raw.get("device_id")),
        ),
        target=_target_like(target, fallback=_first_text(raw.get("target"), fallback="eibrain.honxin")),
        frame_id=frame_id,
        width=_optional_int(raw.get("width"), fallback=None),
        height=_optional_int(raw.get("height"), fallback=None),
        frame_age_ms=_optional_float(raw.get("frameAgeMs", raw.get("frame_age_ms"))),
        backend=_first_text(raw.get("backend"), raw.get("vision_backend")),
        detections=detections,
        boxes=_vision_boxes(raw, detections),
        scores=_vision_scores(raw, detections),
        tracked_target=_vision_tracked_target(raw),
        latency_ms=_vision_latency_ms(raw),
        tracking_diagnostics=_dict_from(raw.get("trackingDiagnostics", raw.get("tracking_diagnostics"))),
        pose=_dict_from(raw.get("pose")),
        clip_labels=_mapping_items(raw.get("clipLabels", raw.get("clip_labels"))),
        semantic_labels=_mapping_items(raw.get("semanticLabels", raw.get("semantic_labels"))),
        depth=_dict_from(raw.get("depth")),
        distance=_dict_from(raw.get("distance")),
        image_url=_first_text(raw.get("imageUrl"), raw.get("image_url")),
        status=_first_text(raw.get("status"), fallback="ok"),
        metadata=_dict_from(raw.get("metadata")),
        event_id=resolved_event_id,
        request_id=_first_text(request_id, trace_id, resolved_event_id),
        sequence=_positive_sequence(raw, sequence),
        time=time or _reported_at(raw),
        session_id=_first_text(raw.get("sessionId"), raw.get("session_id")),
        round_id=_first_text(raw.get("roundId"), raw.get("round_id")),
        correlation_id=_first_text(raw.get("correlationId"), raw.get("correlation_id")),
        causation_id=_first_text(raw.get("causationId"), raw.get("causation_id")),
        trace_id=trace_id,
        ttl_ms=_optional_int(raw.get("ttlMs", raw.get("ttl_ms")), fallback=500),
        mode=_dict_from(raw.get("mode")),
    )


def generic_vision_scene_payload_to_eiprotocol_event(
    payload: Mapping[str, Any],
    *,
    source: str | SourceRef | Mapping[str, Any] | None = None,
    target: str | TargetRef | Mapping[str, Any] | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Normalize a v0.1.1 generic vision-scene payload into an event envelope."""

    raw = _coerce_mapping(payload, label="vision scene payload")
    scene_id = _first_text(raw.get("sceneId"), raw.get("scene_id"), raw.get("id"), fallback="scene")
    observed_at = _first_text(raw.get("observedAt"), raw.get("observed_at"), _reported_at(raw, fallback=time))
    content = _generic_payload_content(raw, exclude=_GENERIC_VISION_SCENE_CONTENT_ALIASES)
    content["sceneId"] = scene_id
    content["observedAt"] = observed_at
    if _first_text(raw.get("summary")):
        content["summary"] = _first_text(raw.get("summary"))
    objects = _list_of_dicts(raw.get("objects")) or _list_of_dicts(raw.get("detections"))
    if objects:
        content["objects"] = objects
    relationships = _list_of_dicts(raw.get("relationships"))
    if relationships:
        content["relationships"] = relationships
    return _generic_payload_to_event(
        raw,
        name="ei.observation.vision.scene",
        event_type="observation",
        content=content,
        event_id=event_id,
        request_id=request_id,
        sequence=sequence,
        time=time,
        source=source,
        target=target,
        source_fallback="eihead.honjia",
        target_fallback="eibrain.honxin",
        event_id_prefix="vision_scene",
        event_id_token=scene_id,
        priority="realtime",
        round_scoped=False,
    )


def generic_vision_event_payload_to_eiprotocol_event(
    payload: Mapping[str, Any],
    *,
    source: str | SourceRef | Mapping[str, Any] | None = None,
    target: str | TargetRef | Mapping[str, Any] | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Normalize a v0.1.1 generic vision-event payload into an event envelope."""

    raw = _coerce_mapping(payload, label="vision event payload")
    vision_event_id = _first_text(
        raw.get("visionEventId"),
        raw.get("vision_event_id"),
        raw.get("eventId"),
        raw.get("event_id"),
        raw.get("id"),
        fallback="vision_event",
    )
    observed_at = _first_text(raw.get("observedAt"), raw.get("observed_at"), _reported_at(raw, fallback=time))
    content = _generic_payload_content(raw, exclude=_GENERIC_VISION_EVENT_CONTENT_ALIASES)
    content["eventId"] = vision_event_id
    event_type_value = _first_text(raw.get("eventType"), raw.get("event_type"), raw.get("event"), fallback="vision_event")
    content["eventType"] = event_type_value
    content["observedAt"] = observed_at
    subject = raw.get("subject")
    if isinstance(subject, Mapping):
        content["subject"] = _copy_jsonish(dict(subject))
    return _generic_payload_to_event(
        raw,
        name="ei.observation.vision.event",
        event_type="observation",
        content=content,
        event_id=event_id,
        request_id=request_id,
        sequence=sequence,
        time=time,
        source=source,
        target=target,
        source_fallback="eihead.honjia",
        target_fallback="eibrain.honxin",
        event_id_prefix="vision_event",
        event_id_token=vision_event_id,
        priority="realtime",
        round_scoped=False,
    )


def generic_memory_policy_report_payload_to_eiprotocol_event(
    payload: Mapping[str, Any],
    *,
    source: str | SourceRef | Mapping[str, Any] | None = None,
    target: str | TargetRef | Mapping[str, Any] | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Normalize a v0.1.1 generic memory-policy report payload into an event envelope."""

    raw = _coerce_mapping(payload, label="memory policy report payload")
    policy_id = _first_text(
        raw.get("policyId"),
        raw.get("policy_id"),
        raw.get("reportId"),
        raw.get("report_id"),
        raw.get("id"),
        fallback="memory_policy_report",
    )
    content = _generic_payload_content(raw, exclude=_GENERIC_MEMORY_POLICY_REPORT_CONTENT_ALIASES)
    content["policyId"] = policy_id
    scope = raw.get("scope")
    content["scope"] = _copy_jsonish(dict(scope)) if isinstance(scope, Mapping) else {}
    if _first_text(raw.get("decision")):
        content["decision"] = _first_text(raw.get("decision"))
    if _first_text(raw.get("reason")):
        content["reason"] = _first_text(raw.get("reason"))
    writes = _list_of_dicts(raw.get("writes")) or _list_of_dicts(raw.get("writebacks"))
    if writes:
        content["writes"] = writes
    return _generic_payload_to_event(
        raw,
        name="ei.memory.policy.report",
        event_type="memory",
        content=content,
        event_id=event_id,
        request_id=request_id,
        sequence=sequence,
        time=time,
        source=source,
        target=target,
        source_fallback="eibrain.honxin",
        target_fallback="eimemory.memoria",
        event_id_prefix="memory_policy_report",
        event_id_token=policy_id,
        priority="normal",
        round_scoped=True,
    )


def vision_observation_to_eiprotocol_event(
    observation: LegacyVisionObservation,
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Wrap a legacy vision observation in an eiprotocol vision-frame event."""

    legacy_payload = dict(observation.payload)
    content = {
        "frameId": observation.frame_id,
        "width": observation.width,
        "height": observation.height,
        "frameAgeMs": legacy_payload.get("frameAgeMs", legacy_payload.get("frame_age_ms")),
        "backend": _first_text(legacy_payload.get("backend"), legacy_payload.get("vision_backend")),
        "detections": [dict(item) for item in observation.detections],
        "boxes": _vision_boxes({"boxes": legacy_payload.get("boxes")}, observation.detections),
        "scores": _vision_scores({"scores": legacy_payload.get("scores")}, observation.detections),
        "latencyMs": _dict_from(legacy_payload.get("latencyMs") or legacy_payload.get("latency_ms")),
        "imageUrl": observation.image_url,
        "status": observation.status,
        "trackedTarget": dict(observation.tracked_target),
        "metadata": {
            "legacyPayload": legacy_payload,
            "observationType": observation.observation_type,
        },
    }
    return _event(
        observation,
        event_type="observation",
        name="ei.observation.vision.frame",
        content=content,
        priority="realtime",
        event_id=_resolve_event_id(event_id, "vision_observation", observation.frame_id, observation.trace_id),
        request_id=request_id,
        sequence=sequence,
        time=time,
        source_device_id=observation.device_id,
        round_scoped=False,
    )


def head_action_to_eiprotocol_event(
    action: LegacyHeadAction,
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Wrap a legacy head action in an eiprotocol side-effecting action event."""

    legacy_payload = dict(action.payload)
    params = dict(action.params)
    resolved_event_id = _resolve_event_id(event_id, "head_action", action.action_id, action.trace_id)
    risk_level = _first_text(
        params.get("riskLevel"),
        params.get("risk_level"),
        legacy_payload.get("riskLevel"),
        legacy_payload.get("risk_level"),
        fallback="L1",
    )
    idempotency_key = _first_text(
        params.get("idempotencyKey"),
        params.get("idempotency_key"),
        legacy_payload.get("idempotencyKey"),
        legacy_payload.get("idempotency_key"),
        action.action_id,
        fallback=resolved_event_id,
    )
    content = {
        "actionId": action.action_id,
        "actionType": action.action_type,
        "target": _first_text(action.device_id, action.target),
        "params": params,
        "riskLevel": risk_level,
        "timeline": _list_of_dicts(legacy_payload.get("timeline")),
        "requiresPolicy": bool(
            params.get("requiresPolicy")
            or params.get("requires_policy")
            or legacy_payload.get("requiresPolicy")
            or legacy_payload.get("requires_policy")
            or False
        ),
        "metadata": {
            "legacyPayload": legacy_payload,
            "priority": action.priority,
        },
        "idempotencyKey": idempotency_key,
    }
    event = _event(
        action,
        event_type="action",
        name="ei.action.request",
        content=content,
        priority="high",
        event_id=resolved_event_id,
        request_id=request_id,
        sequence=sequence,
        time=time,
        source_device_id="",
        round_scoped=True,
    )
    event.policy = PolicyState(decision="not_required", risk_level=risk_level)
    return event


def execution_outcome_to_eiprotocol_event(
    outcome: LegacyExecutionOutcome,
    *,
    event_id: str | None = None,
    request_id: str | None = None,
    sequence: int | None = None,
    time: str | None = None,
) -> EventEnvelope:
    """Wrap a legacy execution outcome in an eiprotocol outcome event."""

    details = dict(outcome.details)
    content = {
        "outcomeId": _first_text(
            details.get("outcomeId"),
            details.get("outcome_id"),
            f"outcome-{outcome.action_id}" if outcome.action_id else "",
            outcome.trace_id,
            fallback="outcome",
        ),
        "actionId": outcome.action_id,
        "actionType": outcome.action_type,
        "success": bool(outcome.success),
        "status": outcome.status,
        "latencyMs": outcome.latency_ms,
        "didWhat": _list_of_text(details.get("didWhat") or details.get("did_what")),
        "errors": _list_of_dicts(details.get("errors")),
        "details": details,
        "deviceId": outcome.device_id,
    }
    return _event(
        outcome,
        event_type="outcome",
        name="ei.outcome.execution",
        content=content,
        priority="normal",
        event_id=_resolve_event_id(event_id, "execution_outcome", outcome.action_id, outcome.trace_id),
        request_id=request_id,
        sequence=sequence,
        time=time,
        source_device_id=outcome.device_id,
        round_scoped=True,
    )


def _event(
    message: object,
    *,
    event_type: str,
    name: str,
    content: dict[str, Any],
    priority: str,
    event_id: str,
    request_id: str | None,
    sequence: int | None,
    time: str | None,
    source_device_id: str,
    round_scoped: bool,
) -> EventEnvelope:
    resolved_request_id = _first_text(request_id, getattr(message, "trace_id", ""), event_id)
    session_id = _first_text(getattr(message, "session_id", ""))
    return EventEnvelope(
        event_id=event_id,
        event_type=event_type,
        name=name,
        time=_resolve_time(message, time),
        sequence=_resolve_sequence(message, sequence),
        request_id=resolved_request_id,
        session_id=session_id,
        round_id=_resolve_round_id(message, event_id) if round_scoped else "",
        trace_id=_first_text(getattr(message, "trace_id", "")),
        source=_source_ref(_first_text(getattr(message, "source", "")), source_device_id),
        target=_target_ref(_first_text(getattr(message, "target", ""))),
        priority=priority,
        content=content,
        policy=PolicyState(),
    )


def _resolve_event_id(explicit: str | None, prefix: str, *candidates: object) -> str:
    if explicit:
        return explicit
    token = _first_text(*candidates, fallback=prefix)
    return f"evt_{prefix}_{_stable_token(token)}"


def _resolve_sequence(message: object, explicit: int | None) -> int:
    if explicit is not None:
        value = int(explicit)
        return value if value > 0 else 1
    sequence = getattr(message, "sequence", None)
    if sequence is None:
        return 1
    try:
        value = int(sequence)
    except (TypeError, ValueError):
        return 1
    return value if value > 0 else 1


def _resolve_time(message: object, explicit: str | None) -> str:
    if explicit:
        return explicit
    timestamp_ms = getattr(message, "timestamp_ms", None)
    if timestamp_ms is None:
        return DEFAULT_EVENT_TIME
    instant = datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=UTC)
    return instant.isoformat(timespec="milliseconds").replace("+00:00", "Z")


_EVENT_KIND_ROUTING: dict[str, Any] = {
    "ei.observation.head.status.report": head_status_report_to_eiprotocol_event,
    "head_status_report": head_status_report_to_eiprotocol_event,
    "head_status": head_status_report_to_eiprotocol_event,
    "ei.observation.vision.frame": realtime_vision_payload_to_eiprotocol_event,
    "realtime_vision_frame": realtime_vision_payload_to_eiprotocol_event,
    "vision_frame": realtime_vision_payload_to_eiprotocol_event,
    "vision_observation": realtime_vision_payload_to_eiprotocol_event,
    "ei.dialogue.fast_hypothesis": dialogue_fast_hypothesis_to_eiprotocol_event,
    "dialogue_fast_hypothesis": dialogue_fast_hypothesis_to_eiprotocol_event,
    "fast_hypothesis": dialogue_fast_hypothesis_to_eiprotocol_event,
    "ei.dialogue.decision.stable": dialogue_stable_decision_to_eiprotocol_event,
    "dialogue_decision_stable": dialogue_stable_decision_to_eiprotocol_event,
    "stable_decision": dialogue_stable_decision_to_eiprotocol_event,
    "ei.observation.vision.scene": generic_vision_scene_payload_to_eiprotocol_event,
    "vision_scene": generic_vision_scene_payload_to_eiprotocol_event,
    "ei.observation.vision.event": generic_vision_event_payload_to_eiprotocol_event,
    "vision_event": generic_vision_event_payload_to_eiprotocol_event,
    "ei.memory.policy.report": generic_memory_policy_report_payload_to_eiprotocol_event,
    "memory_policy_report": generic_memory_policy_report_payload_to_eiprotocol_event,
}


def _resolve_round_id(message: object, event_id: str) -> str:
    return _first_text(
        getattr(message, "round_id", ""),
        getattr(message, "session_id", ""),
        getattr(message, "trace_id", ""),
        event_id,
    )


def _source_ref(source: str, device_id: str = "") -> SourceRef:
    domain, instance_id, source_device_id = _split_ref(source)
    return SourceRef(
        domain=domain or "unknown",
        instance_id=instance_id,
        device_id=_first_text(device_id, source_device_id),
    )


def _target_ref(target: str) -> TargetRef | None:
    if not target:
        return None
    domain, instance_id, _ = _split_ref(target)
    if not domain:
        return None
    return TargetRef(domain=domain, instance_id=instance_id)


def _source_like(
    source: str | SourceRef | Mapping[str, Any] | None,
    *,
    fallback: str,
    device_id: str = "",
) -> SourceRef:
    if isinstance(source, SourceRef):
        return source
    if isinstance(source, Mapping):
        return SourceRef.from_dict(source)
    return _source_ref(_first_text(source, fallback), device_id)


def _target_like(target: str | TargetRef | Mapping[str, Any] | None, *, fallback: str = "") -> TargetRef | None:
    if isinstance(target, TargetRef):
        return target
    if isinstance(target, Mapping):
        return TargetRef.from_dict(target)
    return _target_ref(_first_text(target, fallback))


def _source_from_target_like(
    target: str | TargetRef | Mapping[str, Any] | None,
    *,
    fallback: str,
) -> SourceRef:
    if isinstance(target, TargetRef):
        return SourceRef(domain=target.domain, instance_id=target.instance_id, metadata=dict(target.metadata))
    if isinstance(target, Mapping):
        return SourceRef(
            domain=str(target.get("domain", "") or ""),
            instance_id=str(target.get("instanceId", target.get("instance_id", "")) or ""),
            metadata=_dict_from(target.get("metadata")),
        )
    return _source_ref(_first_text(target, fallback))


def _target_from_source_like(
    source: str | SourceRef | Mapping[str, Any] | None,
    *,
    fallback: str,
) -> TargetRef | None:
    if isinstance(source, SourceRef):
        return TargetRef(domain=source.domain, instance_id=source.instance_id, metadata=dict(source.metadata))
    if isinstance(source, Mapping):
        return TargetRef(
            domain=str(source.get("domain", "") or ""),
            instance_id=str(source.get("instanceId", source.get("instance_id", "")) or ""),
            metadata=_dict_from(source.get("metadata")),
        )
    return _target_ref(_first_text(source, fallback))


def _split_ref(value: str) -> tuple[str, str, str]:
    parts = [part for part in str(value).split(".") if part]
    domain = parts[0] if parts else ""
    instance_id = parts[1] if len(parts) > 1 else ""
    device_id = ".".join(parts[2:]) if len(parts) > 2 else ""
    return domain, instance_id, device_id


def _device_to_capability(device: HeadDevice) -> dict[str, Any]:
    return {
        "capabilityId": _first_text(device.device_id, device.kind, fallback="device"),
        "kind": device.kind,
        "provider": "",
        "model": "",
        "version": "",
        "devicePath": device.path,
        "actions": list(device.capabilities),
        "status": device.health.status,
        "limits": _limits_by_name(device.limits),
        "metadata": {
            **dict(device.metadata),
            "name": device.name,
            "enabled": device.enabled,
            "health": _health_to_dict(device.health),
        },
    }


def _backend_to_capability(backend: HeadBackend) -> dict[str, Any]:
    return {
        "capabilityId": _first_text(backend.backend_id, backend.kind, fallback="backend"),
        "kind": backend.kind,
        "provider": backend.provider,
        "model": backend.model,
        "version": backend.version,
        "devicePath": "",
        "actions": list(backend.capabilities),
        "status": backend.health.status,
        "limits": _limits_by_name(backend.limits),
        "metadata": {
            **dict(backend.metadata),
            "enabled": backend.enabled,
            "health": _health_to_dict(backend.health),
        },
    }


def _modality_count(modality: Mapping[str, Any]) -> int:
    count = 0
    for key, value in modality.items():
        if key == "available":
            continue
        if isinstance(value, list):
            count += len(value)
    return count


def _device_to_capability_metadata(device: HeadDevice) -> dict[str, Any]:
    return {
        "deviceId": device.device_id,
        "kind": device.kind,
        "name": device.name,
        "path": device.path,
        "enabled": device.enabled,
        "capabilities": list(device.capabilities),
        "limits": [limit.to_dict() for limit in device.limits],
        "health": _health_to_dict(device.health),
        "metadata": dict(device.metadata),
    }


def _limits_by_name(limits: list[HeadLimit]) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for index, limit in enumerate(limits):
        key = _first_text(limit.name, f"limit_{index}")
        indexed[key] = limit.to_dict()
    return indexed


def _health_to_dict(health: HeadHealth) -> dict[str, Any]:
    return {
        "status": health.status,
        "message": health.message,
        "checkedAtMs": health.checked_at_ms,
        "metrics": dict(health.metrics),
    }


def _dict_from(value: object) -> dict[str, Any]:
    return _copy_jsonish(dict(value)) if isinstance(value, Mapping) else {}


def _snapshot_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _copy_jsonish(dict(value))
    for method_name in ("snapshot", "status_payload", "to_dict", "status"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            payload = method()
        except TypeError:
            continue
        if isinstance(payload, Mapping):
            return _copy_jsonish(dict(payload))
    return {}


def _mapping_from_any(value: object) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return _copy_jsonish(dict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        if isinstance(payload, Mapping):
            return _copy_jsonish(dict(payload))
    return None


def _first_mapping(*sources: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any] | None:
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            payload = _mapping_from_any(source.get(key))
            if payload is not None and payload:
                return payload
    return None


def _prefetch_items(
    *,
    raw: Mapping[str, Any],
    current: Mapping[str, Any],
    scheduler: Mapping[str, Any],
) -> list[dict[str, Any]]:
    for source in (raw, current, scheduler):
        if not isinstance(source, Mapping):
            continue
        for key in ("memory_prefetch", "memory_prefetch_requests", "prefetch", "prefetch_requests"):
            value = source.get(key)
            items = _mapping_items(value)
            if items:
                return items
    candidates = _mapping_items(current.get("memory_candidates"))
    return [
        item
        for item in candidates
        if _first_text(item.get("query"), item.get("text")) and _first_text(item.get("source"), item.get("kind"))
    ]


def _memory_trace_items(
    *,
    raw: Mapping[str, Any],
    current: Mapping[str, Any],
    scheduler: Mapping[str, Any],
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for source in (current, raw, scheduler):
        if not isinstance(source, Mapping):
            continue
        for key in ("memory_traces", "closed_loop_traces", "memory_trace_history"):
            traces.extend(_mapping_items(source.get(key)))
    return traces


def _memory_trace_selected_records(recall_item: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = _mapping_items(
        recall_item.get("selectedRecords")
        or recall_item.get("selected_records")
        or recall_item.get("records")
        or recall_item.get("results")
    )
    if records:
        return records
    memories = recall_item.get("relevant_memories")
    if isinstance(memories, list):
        return [
            {"record_id": f"memory_{index + 1}", "summary": str(memory), "source": "unknown"}
            for index, memory in enumerate(memories)
            if memory not in (None, "")
        ]
    return []


def _memory_trace_writeback_id(write_item: Mapping[str, Any]) -> str:
    diagnostics = _mapping_from_any(write_item.get("diagnostics")) or {}
    return _first_text(
        write_item.get("memoryId"),
        write_item.get("memory_id"),
        write_item.get("recordId"),
        write_item.get("record_id"),
        diagnostics.get("memoryId"),
        diagnostics.get("memory_id"),
        diagnostics.get("recordId"),
        diagnostics.get("record_id"),
    )


def _mapping_items(value: object) -> list[dict[str, Any]]:
    payload = _mapping_from_any(value)
    if payload is not None:
        return [payload]
    if not isinstance(value, list):
        return []
    return [_copy_jsonish(dict(item)) for item in value if isinstance(item, Mapping)]


_GENERIC_EVENT_CONTROL_KEYS = {
    "causationId",
    "causation_id",
    "correlationId",
    "correlation_id",
    "deviceId",
    "device_id",
    "event_name",
    "extensions",
    "kind",
    "mode",
    "name",
    "policy",
    "priority",
    "requestId",
    "request_id",
    "roundId",
    "round_id",
    "sequence",
    "sessionId",
    "session_id",
    "source",
    "target",
    "time",
    "timestampMs",
    "timestamp_ms",
    "traceId",
    "trace_id",
    "ttlMs",
    "ttl_ms",
    "type",
}
_GENERIC_VISION_SCENE_CONTENT_ALIASES = _GENERIC_EVENT_CONTROL_KEYS | {
    "detections",
    "id",
    "objects",
    "observedAt",
    "observed_at",
    "relationships",
    "sceneId",
    "scene_id",
    "summary",
}
_GENERIC_VISION_EVENT_CONTENT_ALIASES = _GENERIC_EVENT_CONTROL_KEYS | {
    "event",
    "eventId",
    "eventType",
    "event_id",
    "event_type",
    "id",
    "observedAt",
    "observed_at",
    "subject",
    "visionEventId",
    "vision_event_id",
}
_GENERIC_MEMORY_POLICY_REPORT_CONTENT_ALIASES = _GENERIC_EVENT_CONTROL_KEYS | {
    "decision",
    "id",
    "policyId",
    "policy_id",
    "reason",
    "reportId",
    "report_id",
    "scope",
    "writebacks",
    "writes",
}


def _generic_payload_to_event(
    payload: Mapping[str, Any],
    *,
    name: str,
    event_type: str,
    content: Mapping[str, Any],
    event_id: str | None,
    request_id: str | None,
    sequence: int | None,
    time: str | None,
    source: str | SourceRef | Mapping[str, Any] | None,
    target: str | TargetRef | Mapping[str, Any] | None,
    source_fallback: str,
    target_fallback: str,
    event_id_prefix: str,
    event_id_token: str,
    priority: str,
    round_scoped: bool,
) -> EventEnvelope:
    trace_id = _first_text(payload.get("trace_id"), payload.get("traceId"))
    resolved_event_id = _resolve_event_id(event_id, event_id_prefix, event_id_token, trace_id)
    resolved_round_id = _first_text(payload.get("roundId"), payload.get("round_id")) if round_scoped else ""
    return EventEnvelope(
        event_id=resolved_event_id,
        event_type=event_type,
        name=name,
        time=time or _reported_at(payload),
        sequence=_positive_sequence(payload, sequence),
        request_id=_first_text(request_id, payload.get("requestId"), payload.get("request_id"), trace_id, resolved_event_id),
        session_id=_first_text(payload.get("sessionId"), payload.get("session_id")),
        round_id=resolved_round_id,
        correlation_id=_first_text(payload.get("correlationId"), payload.get("correlation_id")),
        causation_id=_first_text(payload.get("causationId"), payload.get("causation_id")),
        trace_id=trace_id,
        source=_source_like(
            source,
            fallback=_first_text(payload.get("source"), fallback=source_fallback),
            device_id=_first_text(payload.get("deviceId"), payload.get("device_id")),
        ),
        target=_target_like(target, fallback=_first_text(payload.get("target"), fallback=target_fallback)),
        priority=_first_text(payload.get("priority"), fallback=priority),
        ttl_ms=_optional_int(payload.get("ttlMs", payload.get("ttl_ms")), fallback=None),
        mode=_dict_from(payload.get("mode")),
        content=_copy_jsonish(dict(content)),
        policy=PolicyState.from_dict(payload.get("policy") if isinstance(payload.get("policy"), Mapping) else None),
        extensions=_dict_from(payload.get("extensions")),
    )


def _generic_payload_content(payload: Mapping[str, Any], *, exclude: set[str]) -> dict[str, Any]:
    return {
        str(key): _copy_jsonish(value)
        for key, value in payload.items()
        if str(key) not in exclude and value is not None
    }


def _coerce_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return _copy_jsonish(dict(value))


def _positive_sequence(payload: Mapping[str, Any], explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    sequence = payload.get("sequence")
    if sequence is None:
        return 1
    try:
        value = int(sequence)
    except (TypeError, ValueError):
        return 1
    return value if value > 0 else 1


def _status_value(payload: Mapping[str, Any]) -> str:
    for key in ("status", "overall_status", "state"):
        value = payload.get(key)
        if value:
            return str(value)
    health = payload.get("health")
    if isinstance(health, Mapping) and health.get("status"):
        return str(health["status"])
    return "unknown"


def _status_components(payload: Mapping[str, Any]) -> dict[str, Any]:
    components = payload.get("components")
    if isinstance(components, Mapping):
        return _copy_jsonish(dict(components))

    capabilities = payload.get("capabilities")
    if isinstance(capabilities, Mapping):
        return {str(key): _status_component(value) for key, value in capabilities.items()}

    collected: dict[str, Any] = {}
    for field_name, id_field in (("devices", "device_id"), ("backends", "backend_id")):
        for item in _component_items(payload.get(field_name), id_field=id_field):
            key = _first_text(item.get(id_field), item.get("id"), item.get("name"), item.get("kind"))
            if key:
                collected[key] = item
    return collected


def _status_component(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _copy_jsonish(dict(value))
    return {"status": str(value)}


def _component_items(value: object, *, id_field: str) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                payload = _copy_jsonish(dict(item))
            else:
                payload = {"status": str(item)}
            payload.setdefault(id_field, str(key))
            items.append(payload)
        return items
    if isinstance(value, list):
        return [_copy_jsonish(dict(item)) for item in value if isinstance(item, Mapping)]
    return []


def _status_summary(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        parts = [f"{key}={summary[key]}" for key in sorted(summary)]
        return ", ".join(parts)
    return _first_text(summary, payload.get("message"))


def _status_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _dict_from(payload.get("metadata"))
    for key in ("schema", "command", "runtime", "node_role", "manifest_schema"):
        if payload.get(key) not in (None, ""):
            metadata.setdefault(key, payload[key])
    return metadata


def _reported_at(payload: Mapping[str, Any], *, fallback: str | None = None) -> str:
    for key in ("reportedAt", "reported_at", "capturedAt", "captured_at", "generatedAt", "generated_at"):
        value = payload.get(key)
        if value:
            return str(value)
    captured_at_ts = payload.get("captured_at_ts")
    if captured_at_ts is not None:
        return _timestamp_seconds_to_rfc3339(captured_at_ts)
    timestamp_ms = payload.get("timestamp_ms", payload.get("timestampMs"))
    if timestamp_ms is not None:
        return _timestamp_ms_to_rfc3339(timestamp_ms)
    return fallback or DEFAULT_EVENT_TIME


def _timestamp_seconds_to_rfc3339(value: object) -> str:
    instant = datetime.fromtimestamp(float(value), tz=UTC)
    return instant.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _timestamp_ms_to_rfc3339(value: object) -> str:
    instant = datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    return instant.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: object, *, fallback: int | None) -> int | None:
    if value is None or value == "":
        return fallback
    return int(value)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "emit", "active"}


def _activity_should_emit(activity: Mapping[str, Any]) -> bool:
    if _truthy(activity.get("shouldEmit", activity.get("should_emit"))):
        return True
    channel = _first_text(activity.get("channel")).strip().lower()
    return channel not in {"", "silent", "none", "off"}


def _cancellation_is_applied(cancellation: Mapping[str, Any]) -> bool:
    if _truthy(cancellation.get("cancelled")) or _truthy(cancellation.get("canceled")):
        return True
    if _truthy(cancellation.get("interrupted")) or _truthy(cancellation.get("applied")):
        return True
    if cancellation.get("cancelled_at_ts") or cancellation.get("cancelledAt"):
        return True
    if cancellation.get("appliedTo") or cancellation.get("applied_to"):
        return True
    state = _first_text(cancellation.get("state"), cancellation.get("status")).strip().lower()
    return state in {"cancelled", "canceled", "interrupted", "applied"}


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_copy_jsonish(dict(item)) for item in value if isinstance(item, Mapping)]


def _list_of_text(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _vision_detections(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    detections = payload.get("detections", payload.get("objects"))
    return _list_of_dicts(detections)


def _vision_boxes(payload: Mapping[str, Any], detections: list[Mapping[str, Any]]) -> list[Any]:
    boxes = payload.get("boxes")
    if isinstance(boxes, list):
        return [_vision_bbox(item) for item in boxes]
    return [_vision_bbox(item.get("bbox")) for item in detections if isinstance(item, Mapping) and item.get("bbox") is not None]


def _vision_scores(payload: Mapping[str, Any], detections: list[Mapping[str, Any]]) -> list[float]:
    scores = payload.get("scores")
    if isinstance(scores, list):
        return [float(item) for item in scores]
    values: list[float] = []
    for item in detections:
        if not isinstance(item, Mapping):
            continue
        score = item.get("score", item.get("confidence"))
        if score is not None:
            values.append(float(score))
    return values


def _vision_tracked_target(payload: Mapping[str, Any]) -> dict[str, Any]:
    tracked_target = payload.get("trackedTarget", payload.get("tracked_target"))
    if isinstance(tracked_target, Mapping):
        return _copy_jsonish(dict(tracked_target))
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        tracked_target = metadata.get("trackedTarget", metadata.get("tracked_target"))
        if isinstance(tracked_target, Mapping):
            return _copy_jsonish(dict(tracked_target))
    return {}


def _vision_latency_ms(payload: Mapping[str, Any]) -> dict[str, Any]:
    latency_ms = payload.get("latencyMs", payload.get("latency_ms"))
    if isinstance(latency_ms, Mapping):
        return _copy_jsonish(dict(latency_ms))
    if latency_ms not in (None, ""):
        return {"total": float(latency_ms)}
    return {}


def _vision_bbox(value: object) -> Any:
    if isinstance(value, Mapping):
        ordered: dict[str, Any] = {}
        preferred_keys = (
            "x",
            "y",
            "w",
            "h",
            "x1",
            "y1",
            "x2",
            "y2",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "left",
            "top",
            "right",
            "bottom",
            "width",
            "height",
        )
        for key in preferred_keys:
            if key in value:
                ordered[key] = value[key]
        for key in sorted(str(key) for key in value):
            if key not in ordered:
                ordered[key] = value[key]
        return ordered
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _first_text(*values: object, fallback: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return fallback


def _stable_token(value: object) -> str:
    token = str(value).strip().replace(" ", "_")
    return token or "event"


def _copy_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_jsonish(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_jsonish(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_jsonish(item) for item in value]
    return value


__all__ = [
    "DEFAULT_EVENT_TIME",
    "audio_turn_to_eiprotocol_event",
    "capability_manifest_to_eiprotocol_event",
    "dialogue_fast_hypothesis_to_eiprotocol_event",
    "dialogue_stable_decision_to_eiprotocol_event",
    "execution_outcome_to_eiprotocol_event",
    "generic_memory_policy_report_payload_to_eiprotocol_event",
    "generic_vision_event_payload_to_eiprotocol_event",
    "generic_vision_scene_payload_to_eiprotocol_event",
    "head_action_to_eiprotocol_event",
    "head_status_report_to_eiprotocol_event",
    "payload_to_eiprotocol_event",
    "realtime_vision_payload_to_eiprotocol_event",
    "scheduler_snapshot_to_eiprotocol_events",
    "to_eiprotocol_event",
    "vision_observation_to_eiprotocol_event",
]
