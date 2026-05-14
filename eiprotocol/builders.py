"""Factory helpers for constructing valid eiprotocol event envelopes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Any
import uuid

from .models import (
    AudioTurn,
    Detection,
    DialogueCancellationApplied,
    DialogueFastHypothesis,
    DialogueStableDecision,
    EmotionContext,
    EventEnvelope,
    ExecutionOutcome,
    HeadStatusReport,
    MemoryPolicyReport,
    MemoryPrefetchRequest,
    PolicyState,
    ProactiveActivityProposal,
    RealtimeVisionObservation,
    SourceRef,
    SpeechActionPlan,
    TargetRef,
    VisionEventObservation,
    VisionSceneObservation,
)
from .validation import ValidationIssue, validate_event_strict

try:
    from .catalog import get_event_definition
except ImportError:  # pragma: no cover - supports partial protocol checkouts.
    get_event_definition = None  # type: ignore[assignment]


Clock = Callable[[], datetime | str]
IdSuffixFactory = Callable[[], str]
SourceLike = SourceRef | Mapping[str, Any]
TargetLike = TargetRef | Mapping[str, Any] | None

_ROUND_SCOPED_EVENT_TYPES = {"dialogue", "action", "memory", "outcome", "training"}
_EVENT_DEFAULTS: dict[str, tuple[str, bool]] = {
    "ei.control.hello": ("control", False),
    "ei.control.ping": ("control", False),
    "ei.control.pong": ("control", False),
    "ei.control.resume": ("control", False),
    "ei.control.ack": ("control", False),
    "ei.control.error": ("control", False),
    "ei.capability.manifest.report": ("capability", False),
    "ei.observation.audio.chunk": ("observation", False),
    "ei.voice.audio.frame": ("observation", False),
    "ei.observation.vision.frame": ("observation", False),
    "ei.observation.vision.scene": ("observation", False),
    "ei.observation.vision.event": ("observation", False),
    "ei.observation.head.status.report": ("observation", False),
    "ei.observation.emotion.context": ("observation", True),
    "ei.dialogue.asr.partial": ("dialogue", True),
    "ei.dialogue.asr.final": ("dialogue", True),
    "ei.voice.asr.partial": ("dialogue", True),
    "ei.voice.asr.final": ("dialogue", True),
    "ei.dialogue.fast_hypothesis": ("dialogue", True),
    "ei.dialogue.decision.stable": ("dialogue", True),
    "ei.dialogue.speech_action.plan": ("dialogue", True),
    "ei.dialogue.cancellation.applied": ("dialogue", True),
    "ei.dialogue.agent.delta": ("dialogue", True),
    "ei.dialogue.agent.final": ("dialogue", True),
    "ei.dialogue.tts.delta": ("dialogue", True),
    "ei.dialogue.tts.final": ("dialogue", True),
    "ei.voice.tts.sentence_start": ("dialogue", True),
    "ei.voice.tts.chunk": ("dialogue", True),
    "ei.voice.playback.started": ("dialogue", True),
    "ei.voice.playback.stopped": ("dialogue", True),
    "ei.voice.barge_in.detected": ("dialogue", True),
    "ei.dialogue.interrupt.requested": ("dialogue", True),
    "ei.voice.session.heartbeat": ("control", False),
    "ei.action.request": ("action", True),
    "ei.action.dispatch": ("action", True),
    "ei.action.progress": ("action", True),
    "ei.action.complete": ("action", True),
    "ei.action.emergency.stop": ("action", True),
    "ei.policy.decision": ("policy", True),
    "ei.memory.recall.request": ("memory", True),
    "ei.memory.prefetch.requested": ("memory", True),
    "ei.memory.policy.report": ("memory", True),
    "ei.memory.recall.result": ("memory", True),
    "ei.memory.write.proposed": ("memory", True),
    "ei.memory.write.committed": ("memory", True),
    "ei.outcome.execution": ("outcome", True),
    "ei.outcome.user.feedback": ("outcome", True),
    "ei.activity.proactive.proposed": ("dialogue", True),
    "ei.training.signal": ("training", True),
}


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


def _default_id_suffix() -> str:
    return uuid.uuid4().hex


def _event_defaults(name: str, definition: Any) -> tuple[str | None, bool | None]:
    if definition is not None:
        return str(definition.event_type), bool(definition.round_scoped)
    defaults = _EVENT_DEFAULTS.get(name)
    if defaults is None:
        return None, None
    return defaults


class EventIdFactory:
    """Generate protocol-prefixed IDs and timestamps for event builders."""

    def __init__(self, *, clock: Clock | None = None, id_factory: IdSuffixFactory | None = None) -> None:
        self._clock = clock or _default_clock
        self._id_factory = id_factory or _default_id_suffix

    def id(self, prefix: str) -> str:
        return f"{prefix}_{self._id_factory()}"

    def event_id(self) -> str:
        return self.id("evt")

    def request_id(self) -> str:
        return self.id("req")

    def round_id(self) -> str:
        return self.id("rnd")

    def trace_id(self) -> str:
        return self.id("trc")

    def evt(self) -> str:
        return self.event_id()

    def req(self) -> str:
        return self.request_id()

    def rnd(self) -> str:
        return self.round_id()

    def trc(self) -> str:
        return self.trace_id()

    def time(self) -> str:
        value = self._clock()
        if isinstance(value, str):
            return value
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat(timespec="milliseconds")


EventIds = EventIdFactory


def build_event(
    *,
    name: str,
    source: SourceLike,
    content: Mapping[str, Any] | None = None,
    event_type: str | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    priority: str = "normal",
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
    round_scoped: bool | None = None,
) -> EventEnvelope:
    """Build and validate an EventEnvelope with safe protocol defaults."""

    id_factory = ids or EventIdFactory()
    definition = get_event_definition(name) if get_event_definition is not None else None
    default_event_type, default_round_scoped = _event_defaults(name, definition)
    resolved_event_type = event_type or default_event_type
    if not resolved_event_type:
        raise ValueError("event_type is required for unknown eiprotocol event names")
    if int(sequence) < 1:
        raise ValueError("sequence must be >= 1")

    is_round_scoped = round_scoped
    if is_round_scoped is None:
        is_round_scoped = default_round_scoped if default_round_scoped is not None else resolved_event_type in _ROUND_SCOPED_EVENT_TYPES

    resolved_event_id = event_id or id_factory.event_id()
    resolved_request_id = request_id or id_factory.request_id()
    resolved_time = time or id_factory.time()
    resolved_round_id = round_id or ""
    if is_round_scoped and not resolved_round_id:
        resolved_round_id = id_factory.round_id()

    event = EventEnvelope(
        event_id=resolved_event_id,
        event_type=resolved_event_type,
        name=name,
        time=resolved_time,
        sequence=int(sequence),
        request_id=resolved_request_id,
        session_id=session_id,
        round_id=resolved_round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        source=_source_ref(source),
        target=_target_ref(target),
        priority=priority,
        ttl_ms=ttl_ms,
        mode=dict(mode or {}),
        content=dict(content or {}),
        policy=_policy_state(policy),
        extensions=dict(extensions or {}),
    )
    _raise_if_invalid(event)
    return event


def build_action_request_event(
    *,
    source: SourceLike,
    action_id: str,
    action_type: str,
    target: str,
    params: Mapping[str, Any] | None = None,
    risk_level: str = "L1",
    idempotency_key: str | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target_ref: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = {
        "actionId": action_id,
        "actionType": action_type,
        "target": target,
        "params": dict(params or {}),
        "riskLevel": risk_level,
        "idempotencyKey": idempotency_key or action_id,
    }
    action_policy = policy if policy is not None else PolicyState(decision="not_required", risk_level=risk_level)
    return build_event(
        ids=ids,
        name="ei.action.request",
        event_type="action",
        source=source,
        target=target_ref,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="high",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=action_policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_asr_event(
    *,
    source: SourceLike,
    text: str,
    final: bool,
    language: str = "und",
    confidence: float | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    audio_level: float | None = None,
    wake_word: str = "",
    asr_backend: str = "",
    timings_ms: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    audio = AudioTurn(
        text=text,
        language=language,
        final=final,
        confidence=confidence,
        start_ms=start_ms,
        end_ms=end_ms,
        audio_level=audio_level,
        wake_word=wake_word,
        asr_backend=asr_backend,
        timings_ms=dict(timings_ms or {}),
        metadata=dict(metadata or {}),
    )
    return build_event(
        ids=ids,
        name="ei.dialogue.asr.final" if final else "ei.dialogue.asr.partial",
        event_type="dialogue",
        source=source,
        target=target,
        content=audio.to_content(),
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_voice_audio_frame_event(
    *,
    source: SourceLike,
    stream_id: str,
    chunk_index: int,
    audio_base64: str,
    sample_rate_hz: int | None = None,
    channels: int | None = None,
    audio_format: str = "",
    duration_ms: float | None = None,
    latency_ms: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = 1000,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = {
        "streamId": stream_id,
        "chunkIndex": int(chunk_index),
        "audioBase64": audio_base64,
        "sampleRateHz": sample_rate_hz,
        "channels": channels,
        "format": audio_format,
        "durationMs": duration_ms,
        "latencyMs": latency_ms,
        "metadata": dict(metadata or {}),
    }
    return build_event(
        ids=ids,
        name="ei.voice.audio.frame",
        event_type="observation",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=False,
    )


def build_voice_asr_event(
    *,
    source: SourceLike,
    text: str,
    final: bool,
    language: str = "und",
    confidence: float | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    latency_ms: float | None = None,
    asr_backend: str = "",
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = {
        "text": text,
        "final": bool(final),
        "language": language,
        "confidence": confidence,
        "startMs": start_ms,
        "endMs": end_ms,
        "latencyMs": latency_ms,
        "asrBackend": asr_backend,
        "metadata": dict(metadata or {}),
    }
    return build_event(
        ids=ids,
        name="ei.voice.asr.final" if final else "ei.voice.asr.partial",
        event_type="dialogue",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_voice_tts_sentence_start_event(
    *,
    source: SourceLike,
    text: str,
    sentence_id: str = "",
    stream_id: str = "",
    chunk_index: int | None = None,
    latency_ms: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = {
        "text": text,
        "sentenceId": sentence_id,
        "streamId": stream_id,
        "chunkIndex": chunk_index,
        "latencyMs": latency_ms,
        "metadata": dict(metadata or {}),
    }
    return build_event(
        ids=ids,
        name="ei.voice.tts.sentence_start",
        event_type="dialogue",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_voice_tts_chunk_event(
    *,
    source: SourceLike,
    stream_id: str,
    chunk_index: int,
    audio_base64: str,
    sentence_id: str = "",
    text: str = "",
    final: bool = False,
    latency_ms: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = {
        "streamId": stream_id,
        "chunkIndex": int(chunk_index),
        "audioBase64": audio_base64,
        "sentenceId": sentence_id,
        "text": text,
        "final": bool(final),
        "latencyMs": latency_ms,
        "metadata": dict(metadata or {}),
    }
    return build_event(
        ids=ids,
        name="ei.voice.tts.chunk",
        event_type="dialogue",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_voice_playback_state_event(
    *,
    source: SourceLike,
    started: bool,
    reason: str = "",
    stream_id: str = "",
    playback_id: str = "",
    chunk_index: int | None = None,
    latency_ms: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    state = "started" if started else "stopped"
    content = {
        "state": state,
        "started": bool(started),
        "reason": reason if started else reason or "completed",
        "streamId": stream_id,
        "playbackId": playback_id,
        "chunkIndex": chunk_index,
        "latencyMs": latency_ms,
        "metadata": dict(metadata or {}),
    }
    return build_event(
        ids=ids,
        name="ei.voice.playback.started" if started else "ei.voice.playback.stopped",
        event_type="dialogue",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_voice_barge_in_detected_event(
    *,
    source: SourceLike,
    reason: str,
    confidence: float | None = None,
    audio_level: float | None = None,
    latency_ms: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = {
        "reason": reason,
        "confidence": confidence,
        "audioLevel": audio_level,
        "latencyMs": latency_ms,
        "metadata": dict(metadata or {}),
    }
    return build_event(
        ids=ids,
        name="ei.voice.barge_in.detected",
        event_type="dialogue",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_voice_session_heartbeat_event(
    *,
    source: SourceLike,
    state: str,
    health: Mapping[str, Any] | None = None,
    queue_lengths: Mapping[str, Any] | None = None,
    capture: Mapping[str, Any] | None = None,
    playback: Mapping[str, Any] | None = None,
    latency_ms: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = 2000,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = {
        "state": state,
        "health": dict(health or {}),
        "queueLengths": dict(queue_lengths or {}),
        "capture": dict(capture or {}),
        "playback": dict(playback or {}),
        "latencyMs": latency_ms,
        "metadata": dict(metadata or {}),
    }
    return build_event(
        ids=ids,
        name="ei.voice.session.heartbeat",
        event_type="control",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=False,
    )


def build_head_status_report_event(
    *,
    source: SourceLike,
    report: HeadStatusReport | Mapping[str, Any] | None = None,
    status: str = "",
    components: Mapping[str, Any] | None = None,
    reported_at: str = "",
    summary: str = "",
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = 2000,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    status_report = _head_status_report(
        report,
        status=status,
        components=components,
        reported_at=reported_at,
        summary=summary,
        metadata=metadata,
    )
    return build_event(
        ids=ids,
        name="ei.observation.head.status.report",
        event_type="observation",
        source=source,
        target=target,
        content=status_report.to_content(),
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=False,
    )


def build_dialogue_fast_hypothesis_event(
    *,
    source: SourceLike,
    hypothesis: DialogueFastHypothesis | Mapping[str, Any] | None = None,
    hypothesis_id: str = "",
    text: str = "",
    confidence: float | None = None,
    basis_event_id: str = "",
    latency_ms: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = 800,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    if hypothesis is None and confidence is None:
        content = {
            "hypothesisId": hypothesis_id,
            "text": text,
            "confidence": None,
            "basisEventId": basis_event_id,
            "latencyMs": latency_ms,
            "metadata": dict(metadata or {}),
        }
    else:
        content = _dialogue_fast_hypothesis(
            hypothesis,
            hypothesis_id=hypothesis_id,
            text=text,
            confidence=confidence,
            basis_event_id=basis_event_id,
            latency_ms=latency_ms,
            metadata=metadata,
        ).to_content()
    return build_event(
        ids=ids,
        name="ei.dialogue.fast_hypothesis",
        event_type="dialogue",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_dialogue_stable_decision_event(
    *,
    source: SourceLike,
    decision: DialogueStableDecision | Mapping[str, Any] | None = None,
    decision_id: str = "",
    decision_value: str = "",
    confidence: float | None = None,
    text: str = "",
    actions: Iterable[Mapping[str, Any]] | None = None,
    stable_since_ms: float | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = 3000,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    if decision is None and confidence is None:
        content = {
            "decisionId": decision_id,
            "decision": decision_value,
            "confidence": None,
            "text": text,
            "actions": [dict(item) for item in actions or ()],
            "stableSinceMs": stable_since_ms,
            "metadata": dict(metadata or {}),
        }
    else:
        content = _dialogue_stable_decision(
            decision,
            decision_id=decision_id,
            decision_value=decision_value,
            confidence=confidence,
            text=text,
            actions=actions,
            stable_since_ms=stable_since_ms,
            metadata=metadata,
        ).to_content()
    return build_event(
        ids=ids,
        name="ei.dialogue.decision.stable",
        event_type="dialogue",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="high",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_emotion_context_event(
    *,
    source: SourceLike,
    context: EmotionContext | Mapping[str, Any] | None = None,
    context_id: str = "",
    mood: str = "",
    confidence: float | None = None,
    signals: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    context_source: str = "",
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = 1000,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = _emotion_context(
        context,
        context_id=context_id,
        mood=mood,
        confidence=confidence,
        signals=signals,
        environment=environment,
        context_source=context_source,
        metadata=metadata,
    ).to_content()
    return build_event(
        ids=ids,
        name="ei.observation.emotion.context",
        event_type="observation",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_memory_prefetch_requested_event(
    *,
    source: SourceLike,
    prefetch: MemoryPrefetchRequest | Mapping[str, Any] | None = None,
    prefetch_id: str = "",
    query: str = "",
    reason: str = "",
    candidates: Iterable[Mapping[str, Any]] | None = None,
    scope: Iterable[str] | None = None,
    prefetch_source: str = "",
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = 1500,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = _memory_prefetch_request(
        prefetch,
        prefetch_id=prefetch_id,
        query=query,
        reason=reason,
        candidates=candidates,
        scope=scope,
        prefetch_source=prefetch_source,
        metadata=metadata,
    ).to_content()
    return build_event(
        ids=ids,
        name="ei.memory.prefetch.requested",
        event_type="memory",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_memory_policy_report_event(
    *,
    source: SourceLike,
    report: MemoryPolicyReport | Mapping[str, Any] | None = None,
    policy_id: str = "",
    scope: Mapping[str, Any] | None = None,
    decision: str = "",
    reason: str = "",
    evidence: Iterable[Mapping[str, Any]] | None = None,
    writes: Iterable[Mapping[str, Any]] | None = None,
    filters: Iterable[Mapping[str, Any]] | None = None,
    conflict_resolution: Mapping[str, Any] | None = None,
    persona_consistency_signals: Iterable[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = _memory_policy_report(
        report,
        policy_id=policy_id,
        scope=scope,
        decision=decision,
        reason=reason,
        evidence=evidence,
        writes=writes,
        filters=filters,
        conflict_resolution=conflict_resolution,
        persona_consistency_signals=persona_consistency_signals,
        metadata=metadata,
    ).to_content()
    return build_event(
        ids=ids,
        name="ei.memory.policy.report",
        event_type="memory",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="normal",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_speech_action_plan_event(
    *,
    source: SourceLike,
    plan: SpeechActionPlan | Mapping[str, Any] | None = None,
    plan_id: str = "",
    stable: bool = False,
    speech_segments: Iterable[Mapping[str, Any]] | None = None,
    action_segments: Iterable[Mapping[str, Any]] | None = None,
    language: str = "zh-CN",
    fallback_text: str = "",
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = 3000,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = _speech_action_plan(
        plan,
        plan_id=plan_id,
        stable=stable,
        speech_segments=speech_segments,
        action_segments=action_segments,
        language=language,
        fallback_text=fallback_text,
        metadata=metadata,
    ).to_content()
    return build_event(
        ids=ids,
        name="ei.dialogue.speech_action.plan",
        event_type="dialogue",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="high",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_proactive_activity_proposed_event(
    *,
    source: SourceLike,
    proposal: ProactiveActivityProposal | Mapping[str, Any] | None = None,
    proposal_id: str = "",
    channel: str = "",
    reason: str = "",
    should_emit: bool = False,
    urgency: float | None = None,
    disturbance: str = "low",
    requires_user_attention: bool = False,
    text: str = "",
    memory_refs: Iterable[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = 1500,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = _proactive_activity_proposal(
        proposal,
        proposal_id=proposal_id,
        channel=channel,
        reason=reason,
        should_emit=should_emit,
        urgency=urgency,
        disturbance=disturbance,
        requires_user_attention=requires_user_attention,
        text=text,
        memory_refs=memory_refs,
        metadata=metadata,
    ).to_content()
    return build_event(
        ids=ids,
        name="ei.activity.proactive.proposed",
        event_type="dialogue",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_dialogue_cancellation_applied_event(
    *,
    source: SourceLike,
    cancellation: DialogueCancellationApplied | Mapping[str, Any] | None = None,
    cancellation_id: str = "",
    cancelled_round_id: str = "",
    cancellation_token: str = "",
    reason: str = "",
    applied_to: Iterable[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = 3000,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = _dialogue_cancellation_applied(
        cancellation,
        cancellation_id=cancellation_id,
        cancelled_round_id=cancelled_round_id,
        cancellation_token=cancellation_token,
        reason=reason,
        applied_to=applied_to,
        metadata=metadata,
    ).to_content()
    return build_event(
        ids=ids,
        name="ei.dialogue.cancellation.applied",
        event_type="dialogue",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="high",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def build_vision_frame_event(
    *,
    source: SourceLike,
    frame_id: str,
    width: int | None = None,
    height: int | None = None,
    frame_age_ms: float | None = None,
    backend: str = "",
    detections: Iterable[Detection | Mapping[str, Any]] | None = None,
    boxes: Iterable[Any] | None = None,
    scores: Iterable[float] | None = None,
    tracked_target: Mapping[str, Any] | None = None,
    latency_ms: Mapping[str, Any] | None = None,
    tracking_diagnostics: Mapping[str, Any] | None = None,
    pose: Mapping[str, Any] | None = None,
    clip_labels: Iterable[Mapping[str, Any]] | None = None,
    semantic_labels: Iterable[Mapping[str, Any]] | None = None,
    depth: Mapping[str, Any] | None = None,
    distance: Mapping[str, Any] | None = None,
    image_url: str = "",
    status: str = "ok",
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    observation = RealtimeVisionObservation(
        frame_id=frame_id,
        width=width,
        height=height,
        frame_age_ms=frame_age_ms,
        backend=backend,
        detections=[_detection(item) for item in detections or ()],
        boxes=[_bbox(item) for item in boxes or ()],
        scores=[float(item) for item in scores or ()],
        tracked_target=dict(tracked_target or {}),
        latency_ms=dict(latency_ms or {}),
        tracking_diagnostics=dict(tracking_diagnostics or {}),
        pose=dict(pose or {}),
        clip_labels=[dict(item) for item in clip_labels or ()],
        semantic_labels=[dict(item) for item in semantic_labels or ()],
        depth=dict(depth or {}),
        distance=dict(distance or {}),
        image_url=image_url,
        status=status,
        metadata=dict(metadata or {}),
    )
    return build_event(
        ids=ids,
        name="ei.observation.vision.frame",
        event_type="observation",
        source=source,
        target=target,
        content=observation.to_content(),
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=False,
    )


def build_vision_scene_event(
    *,
    source: SourceLike,
    scene: VisionSceneObservation | Mapping[str, Any] | None = None,
    scene_id: str = "",
    observed_at: str = "",
    summary: str = "",
    objects: Iterable[Mapping[str, Any]] | None = None,
    relationships: Iterable[Mapping[str, Any]] | None = None,
    environment: Mapping[str, Any] | None = None,
    clip_labels: Iterable[Mapping[str, Any]] | None = None,
    semantic_labels: Iterable[Mapping[str, Any]] | None = None,
    depth: Mapping[str, Any] | None = None,
    distance: Mapping[str, Any] | None = None,
    scene_graph: Mapping[str, Any] | None = None,
    scene_graph_provenance: Mapping[str, Any] | None = None,
    image_url: str = "",
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = 1000,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = _vision_scene_observation(
        scene,
        scene_id=scene_id,
        observed_at=observed_at,
        summary=summary,
        objects=objects,
        relationships=relationships,
        environment=environment,
        clip_labels=clip_labels,
        semantic_labels=semantic_labels,
        depth=depth,
        distance=distance,
        scene_graph=scene_graph,
        scene_graph_provenance=scene_graph_provenance,
        image_url=image_url,
        metadata=metadata,
    ).to_content()
    return build_event(
        ids=ids,
        name="ei.observation.vision.scene",
        event_type="observation",
        source=source,
        target=target,
        content=content,
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=False,
    )


def build_vision_event_event(
    *,
    source: SourceLike,
    event_id: str,
    event_type: str,
    observed_at: str,
    scene_id: str = "",
    event: VisionEventObservation | Mapping[str, Any] | None = None,
    subject: Mapping[str, Any] | None = None,
    confidence: float | None = None,
    pose: Mapping[str, Any] | None = None,
    clip_labels: Iterable[Mapping[str, Any]] | None = None,
    semantic_labels: Iterable[Mapping[str, Any]] | None = None,
    depth: Mapping[str, Any] | None = None,
    distance: Mapping[str, Any] | None = None,
    scene_graph_provenance: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    protocol_event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = 1000,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    content = _vision_event_observation(
        event,
        event_id=event_id,
        event_type=event_type,
        observed_at=observed_at,
        scene_id=scene_id,
        subject=subject,
        confidence=confidence,
        pose=pose,
        clip_labels=clip_labels,
        semantic_labels=semantic_labels,
        depth=depth,
        distance=distance,
        scene_graph_provenance=scene_graph_provenance,
        details=details,
        metadata=metadata,
    ).to_content()
    return build_event(
        ids=ids,
        name="ei.observation.vision.event",
        event_type="observation",
        source=source,
        target=target,
        content=content,
        event_id=protocol_event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="realtime",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=False,
    )


def build_execution_outcome_event(
    *,
    source: SourceLike,
    outcome_id: str,
    action_id: str = "",
    action_type: str = "",
    success: bool = True,
    status: str = "completed",
    latency_ms: float | None = None,
    did_what: Iterable[str] | None = None,
    errors: Iterable[Mapping[str, Any]] | None = None,
    details: Mapping[str, Any] | None = None,
    ids: EventIdFactory | None = None,
    event_id: str | None = None,
    request_id: str | None = None,
    time: str | None = None,
    sequence: int = 1,
    session_id: str = "",
    round_id: str | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    target: TargetLike = None,
    ttl_ms: int | None = None,
    mode: Mapping[str, Any] | None = None,
    policy: PolicyState | Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> EventEnvelope:
    outcome = ExecutionOutcome(
        outcome_id=outcome_id,
        action_id=action_id,
        action_type=action_type,
        success=success,
        status=status,
        latency_ms=latency_ms,
        did_what=list(did_what or []),
        errors=[dict(item) for item in errors or ()],
        details=dict(details or {}),
    )
    return build_event(
        ids=ids,
        name="ei.outcome.execution",
        event_type="outcome",
        source=source,
        target=target,
        content=outcome.to_content(),
        event_id=event_id,
        request_id=request_id,
        time=time,
        sequence=sequence,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        priority="normal",
        ttl_ms=ttl_ms,
        mode=mode,
        policy=policy,
        extensions=extensions,
        round_scoped=True,
    )


def _source_ref(source: SourceLike) -> SourceRef:
    if isinstance(source, SourceRef):
        return source
    if isinstance(source, Mapping):
        return SourceRef.from_dict(source)
    raise TypeError("source must be a SourceRef or mapping")


def _target_ref(target: TargetLike) -> TargetRef | None:
    if target is None:
        return None
    if isinstance(target, TargetRef):
        return target
    if isinstance(target, Mapping):
        return TargetRef.from_dict(target)
    raise TypeError("target must be a TargetRef, mapping, or None")


def _policy_state(policy: PolicyState | Mapping[str, Any] | None) -> PolicyState:
    if policy is None:
        return PolicyState()
    if isinstance(policy, PolicyState):
        return policy
    if isinstance(policy, Mapping):
        return PolicyState.from_dict(policy)
    raise TypeError("policy must be a PolicyState, mapping, or None")


def _detection(value: Detection | Mapping[str, Any]) -> Detection:
    if isinstance(value, Detection):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("detections must contain Detection objects or mappings")
    return Detection(
        label=str(value.get("label", "") or ""),
        score=float(value.get("score", value.get("confidence", 0.0)) or 0.0),
        bbox=_bbox(value.get("bbox")),
        track_id=str(value.get("trackId", value.get("track_id", "")) or ""),
        pose=_dict_or_empty(value.get("pose")),
        clip_labels=_dict_item_list(value.get("clipLabels", value.get("clip_labels"))),
        semantic_labels=_dict_item_list(value.get("semanticLabels", value.get("semantic_labels"))),
        depth=_dict_or_empty(value.get("depth")),
        distance=_dict_or_empty(value.get("distance")),
        tracking_diagnostics=_dict_or_empty(
            value.get("trackingDiagnostics", value.get("tracking_diagnostics")),
        ),
        metadata=dict(value.get("metadata", {}) or {}),
    )


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dict_item_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            items.append(dict(item))
        elif isinstance(item, str) and item.strip():
            items.append({"label": item.strip()})
    return items


def _bbox(value: Any) -> list[Any] | dict[str, Any]:
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


def _head_status_report(
    report: HeadStatusReport | Mapping[str, Any] | None,
    *,
    status: str,
    components: Mapping[str, Any] | None,
    reported_at: str,
    summary: str,
    metadata: Mapping[str, Any] | None,
) -> HeadStatusReport:
    if isinstance(report, HeadStatusReport):
        return report
    if isinstance(report, Mapping):
        return HeadStatusReport.from_content(report)
    return HeadStatusReport(
        status=status,
        components=dict(components or {}),
        reported_at=reported_at,
        summary=summary,
        metadata=dict(metadata or {}),
    )


def _dialogue_fast_hypothesis(
    hypothesis: DialogueFastHypothesis | Mapping[str, Any] | None,
    *,
    hypothesis_id: str,
    text: str,
    confidence: float | None,
    basis_event_id: str,
    latency_ms: float | None,
    metadata: Mapping[str, Any] | None,
) -> DialogueFastHypothesis:
    if isinstance(hypothesis, DialogueFastHypothesis):
        return hypothesis
    if isinstance(hypothesis, Mapping):
        return DialogueFastHypothesis.from_content(hypothesis)
    return DialogueFastHypothesis(
        hypothesis_id=hypothesis_id,
        text=text,
        confidence=float(confidence) if confidence is not None else 0.0,
        basis_event_id=basis_event_id,
        latency_ms=latency_ms,
        metadata=dict(metadata or {}),
    )


def _dialogue_stable_decision(
    decision: DialogueStableDecision | Mapping[str, Any] | None,
    *,
    decision_id: str,
    decision_value: str,
    confidence: float | None,
    text: str,
    actions: Iterable[Mapping[str, Any]] | None,
    stable_since_ms: float | None,
    metadata: Mapping[str, Any] | None,
) -> DialogueStableDecision:
    if isinstance(decision, DialogueStableDecision):
        return decision
    if isinstance(decision, Mapping):
        return DialogueStableDecision.from_content(decision)
    return DialogueStableDecision(
        decision_id=decision_id,
        decision=decision_value,
        confidence=float(confidence) if confidence is not None else 0.0,
        text=text,
        actions=[dict(item) for item in actions or ()],
        stable_since_ms=stable_since_ms,
        metadata=dict(metadata or {}),
    )


def _emotion_context(
    context: EmotionContext | Mapping[str, Any] | None,
    *,
    context_id: str,
    mood: str,
    confidence: float | None,
    signals: Mapping[str, Any] | None,
    environment: Mapping[str, Any] | None,
    context_source: str,
    metadata: Mapping[str, Any] | None,
) -> EmotionContext:
    if isinstance(context, EmotionContext):
        return context
    if isinstance(context, Mapping):
        merged = dict(context)
        if context_id:
            merged["contextId"] = context_id
        if mood:
            merged["mood"] = mood
        if confidence is not None:
            merged["confidence"] = confidence
        if signals is not None:
            merged["signals"] = dict(signals)
        if environment is not None:
            merged["environment"] = dict(environment)
        if context_source:
            merged["source"] = context_source
        if metadata is not None:
            merged["metadata"] = dict(metadata)
        return EmotionContext.from_content(merged)
    return EmotionContext(
        context_id=context_id,
        mood=mood,
        confidence=float(confidence) if confidence is not None else 0.0,
        signals=dict(signals or {}),
        environment=dict(environment or {}),
        source=context_source,
        metadata=dict(metadata or {}),
    )


def _memory_prefetch_request(
    prefetch: MemoryPrefetchRequest | Mapping[str, Any] | None,
    *,
    prefetch_id: str,
    query: str,
    reason: str,
    candidates: Iterable[Mapping[str, Any]] | None,
    scope: Iterable[str] | None,
    prefetch_source: str,
    metadata: Mapping[str, Any] | None,
) -> MemoryPrefetchRequest:
    if isinstance(prefetch, MemoryPrefetchRequest):
        return prefetch
    if isinstance(prefetch, Mapping):
        merged = dict(prefetch)
        if prefetch_id:
            merged["prefetchId"] = prefetch_id
        if query:
            merged["query"] = query
        if reason:
            merged["reason"] = reason
        if candidates is not None:
            merged["candidates"] = [dict(item) for item in candidates]
        if scope is not None:
            merged["scope"] = [str(item) for item in scope]
        if prefetch_source:
            merged["source"] = prefetch_source
        if metadata is not None:
            merged["metadata"] = dict(metadata)
        return MemoryPrefetchRequest.from_content(merged)
    return MemoryPrefetchRequest(
        prefetch_id=prefetch_id,
        query=query,
        reason=reason,
        candidates=[dict(item) for item in candidates or ()],
        scope=[str(item) for item in scope or ()],
        source=prefetch_source,
        metadata=dict(metadata or {}),
    )


def _memory_policy_report(
    report: MemoryPolicyReport | Mapping[str, Any] | None,
    *,
    policy_id: str,
    scope: Mapping[str, Any] | None,
    decision: str,
    reason: str,
    evidence: Iterable[Mapping[str, Any]] | None,
    writes: Iterable[Mapping[str, Any]] | None,
    filters: Iterable[Mapping[str, Any]] | None,
    conflict_resolution: Mapping[str, Any] | None,
    persona_consistency_signals: Iterable[Mapping[str, Any]] | None,
    metadata: Mapping[str, Any] | None,
) -> MemoryPolicyReport:
    if isinstance(report, MemoryPolicyReport):
        return report
    if isinstance(report, Mapping):
        merged = dict(report)
        if policy_id:
            merged["policyId"] = policy_id
        if scope is not None:
            merged["scope"] = dict(scope)
        if decision:
            merged["decision"] = decision
        if reason:
            merged["reason"] = reason
        if evidence is not None:
            merged["evidence"] = [dict(item) for item in evidence]
        if writes is not None:
            merged["writes"] = [dict(item) for item in writes]
        if filters is not None:
            merged["filters"] = [dict(item) for item in filters]
        if conflict_resolution is not None:
            merged["conflictResolution"] = dict(conflict_resolution)
        if persona_consistency_signals is not None:
            merged["personaConsistencySignals"] = [dict(item) for item in persona_consistency_signals]
        if metadata is not None:
            merged["metadata"] = dict(metadata)
        return MemoryPolicyReport.from_content(merged)
    return MemoryPolicyReport(
        policy_id=policy_id,
        scope=dict(scope or {}),
        decision=decision,
        reason=reason,
        evidence=[dict(item) for item in evidence or ()],
        writes=[dict(item) for item in writes or ()],
        filters=[dict(item) for item in filters or ()],
        conflict_resolution=dict(conflict_resolution or {}),
        persona_consistency_signals=[dict(item) for item in persona_consistency_signals or ()],
        metadata=dict(metadata or {}),
    )


def _speech_action_plan(
    plan: SpeechActionPlan | Mapping[str, Any] | None,
    *,
    plan_id: str,
    stable: bool,
    speech_segments: Iterable[Mapping[str, Any]] | None,
    action_segments: Iterable[Mapping[str, Any]] | None,
    language: str,
    fallback_text: str,
    metadata: Mapping[str, Any] | None,
) -> SpeechActionPlan:
    if isinstance(plan, SpeechActionPlan):
        return plan
    if isinstance(plan, Mapping):
        merged = dict(plan)
        if plan_id:
            merged["planId"] = plan_id
        if stable:
            merged["stable"] = stable
        if speech_segments is not None:
            merged["speechSegments"] = [dict(item) for item in speech_segments]
        if action_segments is not None:
            merged["actionSegments"] = [dict(item) for item in action_segments]
        if language:
            merged["language"] = language
        if fallback_text:
            merged["fallbackText"] = fallback_text
        if metadata is not None:
            merged["metadata"] = dict(metadata)
        return SpeechActionPlan.from_content(merged)
    return SpeechActionPlan(
        plan_id=plan_id,
        stable=bool(stable),
        speech_segments=[dict(item) for item in speech_segments or ()],
        action_segments=[dict(item) for item in action_segments or ()],
        language=language,
        fallback_text=fallback_text,
        metadata=dict(metadata or {}),
    )


def _proactive_activity_proposal(
    proposal: ProactiveActivityProposal | Mapping[str, Any] | None,
    *,
    proposal_id: str,
    channel: str,
    reason: str,
    should_emit: bool,
    urgency: float | None,
    disturbance: str,
    requires_user_attention: bool,
    text: str,
    memory_refs: Iterable[Mapping[str, Any]] | None,
    metadata: Mapping[str, Any] | None,
) -> ProactiveActivityProposal:
    if isinstance(proposal, ProactiveActivityProposal):
        return proposal
    if isinstance(proposal, Mapping):
        merged = dict(proposal)
        if proposal_id:
            merged["proposalId"] = proposal_id
        if channel:
            merged["channel"] = channel
        if reason:
            merged["reason"] = reason
        if should_emit:
            merged["shouldEmit"] = should_emit
        if urgency is not None:
            merged["urgency"] = urgency
        if disturbance:
            merged["disturbance"] = disturbance
        if requires_user_attention:
            merged["requiresUserAttention"] = requires_user_attention
        if text:
            merged["text"] = text
        if memory_refs is not None:
            merged["memoryRefs"] = [dict(item) for item in memory_refs]
        if metadata is not None:
            merged["metadata"] = dict(metadata)
        return ProactiveActivityProposal.from_content(merged)
    return ProactiveActivityProposal(
        proposal_id=proposal_id,
        channel=channel,
        reason=reason,
        should_emit=bool(should_emit),
        urgency=urgency,
        disturbance=disturbance,
        requires_user_attention=requires_user_attention,
        text=text,
        memory_refs=[dict(item) for item in memory_refs or ()],
        metadata=dict(metadata or {}),
    )


def _dialogue_cancellation_applied(
    cancellation: DialogueCancellationApplied | Mapping[str, Any] | None,
    *,
    cancellation_id: str,
    cancelled_round_id: str,
    cancellation_token: str,
    reason: str,
    applied_to: Iterable[str] | None,
    metadata: Mapping[str, Any] | None,
) -> DialogueCancellationApplied:
    if isinstance(cancellation, DialogueCancellationApplied):
        return cancellation
    if isinstance(cancellation, Mapping):
        merged = dict(cancellation)
        if cancellation_id:
            merged["cancellationId"] = cancellation_id
        if cancelled_round_id:
            merged["cancelledRoundId"] = cancelled_round_id
        if cancellation_token:
            merged["cancellationToken"] = cancellation_token
        if reason:
            merged["reason"] = reason
        if applied_to is not None:
            merged["appliedTo"] = [str(item) for item in applied_to]
        if metadata is not None:
            merged["metadata"] = dict(metadata)
        return DialogueCancellationApplied.from_content(merged)
    return DialogueCancellationApplied(
        cancellation_id=cancellation_id,
        cancelled_round_id=cancelled_round_id,
        cancellation_token=cancellation_token,
        reason=reason,
        applied_to=[str(item) for item in applied_to or ()],
        metadata=dict(metadata or {}),
    )


def _vision_scene_observation(
    scene: VisionSceneObservation | Mapping[str, Any] | None,
    *,
    scene_id: str,
    observed_at: str,
    summary: str,
    objects: Iterable[Mapping[str, Any]] | None,
    relationships: Iterable[Mapping[str, Any]] | None,
    environment: Mapping[str, Any] | None,
    clip_labels: Iterable[Mapping[str, Any]] | None,
    semantic_labels: Iterable[Mapping[str, Any]] | None,
    depth: Mapping[str, Any] | None,
    distance: Mapping[str, Any] | None,
    scene_graph: Mapping[str, Any] | None,
    scene_graph_provenance: Mapping[str, Any] | None,
    image_url: str,
    metadata: Mapping[str, Any] | None,
) -> VisionSceneObservation:
    if isinstance(scene, VisionSceneObservation):
        return scene
    if isinstance(scene, Mapping):
        merged = dict(scene)
        if scene_id:
            merged["sceneId"] = scene_id
        if observed_at:
            merged["observedAt"] = observed_at
        if summary:
            merged["summary"] = summary
        if objects is not None:
            merged["objects"] = [dict(item) for item in objects]
        if relationships is not None:
            merged["relationships"] = [dict(item) for item in relationships]
        if environment is not None:
            merged["environment"] = dict(environment)
        if clip_labels is not None:
            merged["clipLabels"] = _dict_item_list(list(clip_labels))
        if semantic_labels is not None:
            merged["semanticLabels"] = _dict_item_list(list(semantic_labels))
        if depth is not None:
            merged["depth"] = dict(depth)
        if distance is not None:
            merged["distance"] = dict(distance)
        if scene_graph is not None:
            merged["sceneGraph"] = dict(scene_graph)
        if scene_graph_provenance is not None:
            merged["sceneGraphProvenance"] = dict(scene_graph_provenance)
        if image_url:
            merged["imageUrl"] = image_url
        if metadata is not None:
            merged["metadata"] = dict(metadata)
        return VisionSceneObservation.from_content(merged)
    return VisionSceneObservation(
        scene_id=scene_id,
        observed_at=observed_at,
        summary=summary,
        objects=[dict(item) for item in objects or ()],
        relationships=[dict(item) for item in relationships or ()],
        environment=dict(environment or {}),
        clip_labels=_dict_item_list(list(clip_labels or ())),
        semantic_labels=_dict_item_list(list(semantic_labels or ())),
        depth=dict(depth or {}),
        distance=dict(distance or {}),
        scene_graph=dict(scene_graph or {}),
        scene_graph_provenance=dict(scene_graph_provenance or {}),
        image_url=image_url,
        metadata=dict(metadata or {}),
    )


def _vision_event_observation(
    event: VisionEventObservation | Mapping[str, Any] | None,
    *,
    event_id: str,
    event_type: str,
    observed_at: str,
    scene_id: str,
    subject: Mapping[str, Any] | None,
    confidence: float | None,
    pose: Mapping[str, Any] | None,
    clip_labels: Iterable[Mapping[str, Any]] | None,
    semantic_labels: Iterable[Mapping[str, Any]] | None,
    depth: Mapping[str, Any] | None,
    distance: Mapping[str, Any] | None,
    scene_graph_provenance: Mapping[str, Any] | None,
    details: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> VisionEventObservation:
    if isinstance(event, VisionEventObservation):
        return event
    if isinstance(event, Mapping):
        merged = dict(event)
        if event_id:
            merged["eventId"] = event_id
        if event_type:
            merged["eventType"] = event_type
        if observed_at:
            merged["observedAt"] = observed_at
        if scene_id:
            merged["sceneId"] = scene_id
        if subject is not None:
            merged["subject"] = dict(subject)
        if confidence is not None:
            merged["confidence"] = confidence
        if pose is not None:
            merged["pose"] = dict(pose)
        if clip_labels is not None:
            merged["clipLabels"] = _dict_item_list(list(clip_labels))
        if semantic_labels is not None:
            merged["semanticLabels"] = _dict_item_list(list(semantic_labels))
        if depth is not None:
            merged["depth"] = dict(depth)
        if distance is not None:
            merged["distance"] = dict(distance)
        if scene_graph_provenance is not None:
            merged["sceneGraphProvenance"] = dict(scene_graph_provenance)
        if details is not None:
            merged["details"] = dict(details)
        if metadata is not None:
            merged["metadata"] = dict(metadata)
        return VisionEventObservation.from_content(merged)
    return VisionEventObservation(
        event_id=event_id,
        event_type=event_type,
        observed_at=observed_at,
        scene_id=scene_id,
        subject=dict(subject or {}),
        confidence=confidence,
        pose=dict(pose or {}),
        clip_labels=_dict_item_list(list(clip_labels or ())),
        semantic_labels=_dict_item_list(list(semantic_labels or ())),
        depth=dict(depth or {}),
        distance=dict(distance or {}),
        scene_graph_provenance=dict(scene_graph_provenance or {}),
        details=dict(details or {}),
        metadata=dict(metadata or {}),
    )


def _raise_if_invalid(event: EventEnvelope) -> None:
    errors = [_issue_to_error(issue) for issue in validate_event_strict(event, known_event_required=True)]
    if errors:
        raise ValueError("invalid eiprotocol event: " + "; ".join(errors))


def _issue_to_error(issue: ValidationIssue) -> str:
    if issue.code in {"required", "invalid_spec_version", "invalid_content", "missing_idempotency_key"}:
        return issue.message
    return f"{issue.code} at {issue.path}: {issue.message}"


__all__ = [
    "EventIds",
    "EventIdFactory",
    "build_action_request_event",
    "build_asr_event",
    "build_dialogue_cancellation_applied_event",
    "build_dialogue_fast_hypothesis_event",
    "build_dialogue_stable_decision_event",
    "build_emotion_context_event",
    "build_event",
    "build_execution_outcome_event",
    "build_head_status_report_event",
    "build_memory_policy_report_event",
    "build_memory_prefetch_requested_event",
    "build_proactive_activity_proposed_event",
    "build_speech_action_plan_event",
    "build_voice_asr_event",
    "build_voice_audio_frame_event",
    "build_voice_barge_in_detected_event",
    "build_voice_playback_state_event",
    "build_voice_session_heartbeat_event",
    "build_voice_tts_chunk_event",
    "build_voice_tts_sentence_start_event",
    "build_vision_event_event",
    "build_vision_frame_event",
    "build_vision_scene_event",
]
