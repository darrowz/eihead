"""eiprotocol v0.1 MVP data contracts.

The package intentionally models only the shared wire shapes needed by the
eihead/eibrain split. It is transport-agnostic and keeps policy as metadata so
the first MVP does not force a safety-gate runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping


SPEC_VERSION = "eiprotocol/0.1"


def _dict(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _list(value: list[Any] | tuple[Any, ...] | None) -> list[Any]:
    return list(value or [])


@dataclass(slots=True)
class SourceRef:
    domain: str
    instance_id: str = ""
    device_id: str = ""
    bot_id: str = ""
    uid: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "instanceId": self.instance_id,
            "deviceId": self.device_id,
            "botId": self.bot_id,
            "uid": self.uid,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceRef":
        return cls(
            domain=str(data.get("domain", "")),
            instance_id=str(data.get("instanceId", data.get("instance_id", "")) or ""),
            device_id=str(data.get("deviceId", data.get("device_id", "")) or ""),
            bot_id=str(data.get("botId", data.get("bot_id", "")) or ""),
            uid=str(data.get("uid", "") or ""),
            metadata=_dict(data.get("metadata") if isinstance(data.get("metadata"), Mapping) else None),
        )


@dataclass(slots=True)
class TargetRef:
    domain: str
    instance_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "instanceId": self.instance_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetRef":
        return cls(
            domain=str(data.get("domain", "")),
            instance_id=str(data.get("instanceId", data.get("instance_id", "")) or ""),
            metadata=_dict(data.get("metadata") if isinstance(data.get("metadata"), Mapping) else None),
        )


@dataclass(slots=True)
class PolicyState:
    decision: str = "not_required"
    risk_level: str = "L0"
    decision_id: str = ""
    required_ack: bool = False
    reason: str = ""
    expires_at: str = ""
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "riskLevel": self.risk_level,
            "decisionId": self.decision_id,
            "requiredAck": self.required_ack,
            "reason": self.reason,
            "expiresAt": self.expires_at,
            "extensions": dict(self.extensions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PolicyState":
        payload = _dict(data)
        return cls(
            decision=str(payload.get("decision", "not_required") or "not_required"),
            risk_level=str(payload.get("riskLevel", payload.get("risk_level", "L0")) or "L0"),
            decision_id=str(payload.get("decisionId", payload.get("decision_id", "")) or ""),
            required_ack=bool(payload.get("requiredAck", payload.get("required_ack", False))),
            reason=str(payload.get("reason", "") or ""),
            expires_at=str(payload.get("expiresAt", payload.get("expires_at", "")) or ""),
            extensions=_dict(payload.get("extensions") if isinstance(payload.get("extensions"), Mapping) else None),
        )


@dataclass(slots=True)
class EventEnvelope:
    event_id: str
    event_type: str
    name: str
    time: str
    sequence: int
    request_id: str
    source: SourceRef
    content: dict[str, Any]
    session_id: str = ""
    round_id: str = ""
    correlation_id: str = ""
    causation_id: str = ""
    trace_id: str = ""
    target: TargetRef | None = None
    priority: str = "normal"
    ttl_ms: int | None = None
    mode: dict[str, Any] = field(default_factory=dict)
    policy: PolicyState = field(default_factory=PolicyState)
    extensions: dict[str, Any] = field(default_factory=dict)
    spec_version: str = SPEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "specVersion": self.spec_version,
            "id": self.event_id,
            "type": self.event_type,
            "name": self.name,
            "time": self.time,
            "sequence": int(self.sequence),
            "requestId": self.request_id,
            "sessionId": self.session_id,
            "roundId": self.round_id,
            "correlationId": self.correlation_id,
            "causationId": self.causation_id,
            "traceId": self.trace_id,
            "source": self.source.to_dict(),
            "priority": self.priority,
            "ttlMs": self.ttl_ms,
            "mode": dict(self.mode),
            "content": dict(self.content),
            "policy": self.policy.to_dict(),
            "extensions": dict(self.extensions),
        }
        if self.target is not None:
            payload["target"] = self.target.to_dict()
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "EventEnvelope":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventEnvelope":
        source = data.get("source")
        target = data.get("target")
        policy = data.get("policy")
        ttl_ms = data.get("ttlMs")
        if ttl_ms is None:
            ttl_ms = data.get("ttl_ms")
        return cls(
            spec_version=str(data.get("specVersion", data.get("spec_version", SPEC_VERSION)) or SPEC_VERSION),
            event_id=str(data.get("id", data.get("event_id", "")) or ""),
            event_type=str(data.get("type", data.get("event_type", "")) or ""),
            name=str(data.get("name", "") or ""),
            time=str(data.get("time", "") or ""),
            sequence=int(data.get("sequence", 0) or 0),
            request_id=str(data.get("requestId", data.get("request_id", "")) or ""),
            session_id=str(data.get("sessionId", data.get("session_id", "")) or ""),
            round_id=str(data.get("roundId", data.get("round_id", "")) or ""),
            correlation_id=str(data.get("correlationId", data.get("correlation_id", "")) or ""),
            causation_id=str(data.get("causationId", data.get("causation_id", "")) or ""),
            trace_id=str(data.get("traceId", data.get("trace_id", "")) or ""),
            source=SourceRef.from_dict(source if isinstance(source, Mapping) else {}),
            target=TargetRef.from_dict(target) if isinstance(target, Mapping) else None,
            priority=str(data.get("priority", "normal") or "normal"),
            ttl_ms=int(ttl_ms) if ttl_ms is not None else None,
            mode=_dict(data.get("mode") if isinstance(data.get("mode"), Mapping) else None),
            content=_dict(data.get("content") if isinstance(data.get("content"), Mapping) else None),
            policy=PolicyState.from_dict(policy if isinstance(policy, Mapping) else None),
            extensions=_dict(data.get("extensions") if isinstance(data.get("extensions"), Mapping) else None),
        )


@dataclass(slots=True)
class DeviceStatus:
    status: str = "unknown"
    message: str = ""
    checked_at_ms: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "checkedAtMs": self.checked_at_ms,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "DeviceStatus":
        payload = _dict(data)
        return cls(
            status=str(payload.get("status", "unknown") or "unknown"),
            message=str(payload.get("message", "") or ""),
            checked_at_ms=int(payload["checkedAtMs"]) if payload.get("checkedAtMs") is not None else None,
            metrics=_dict(payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else None),
        )


@dataclass(slots=True)
class Capability:
    capability_id: str
    kind: str
    provider: str = ""
    model: str = ""
    version: str = ""
    device_path: str = ""
    actions: list[str] = field(default_factory=list)
    status: str = "unknown"
    limits: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capabilityId": self.capability_id,
            "kind": self.kind,
            "provider": self.provider,
            "model": self.model,
            "version": self.version,
            "devicePath": self.device_path,
            "actions": list(self.actions),
            "status": self.status,
            "limits": dict(self.limits),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Capability":
        return cls(
            capability_id=str(data.get("capabilityId", data.get("capability_id", "")) or ""),
            kind=str(data.get("kind", "") or ""),
            provider=str(data.get("provider", "") or ""),
            model=str(data.get("model", "") or ""),
            version=str(data.get("version", "") or ""),
            device_path=str(data.get("devicePath", data.get("device_path", "")) or ""),
            actions=[str(item) for item in _list(data.get("actions") if isinstance(data.get("actions"), list) else None)],
            status=str(data.get("status", "unknown") or "unknown"),
            limits=_dict(data.get("limits") if isinstance(data.get("limits"), Mapping) else None),
            metadata=_dict(data.get("metadata") if isinstance(data.get("metadata"), Mapping) else None),
        )


@dataclass(slots=True)
class CapabilityManifest:
    manifest_id: str
    manifest_version: str = "0.1.0"
    device: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    transports: dict[str, Any] = field(default_factory=dict)
    modalities: dict[str, Any] = field(default_factory=dict)
    capabilities: list[Capability] = field(default_factory=list)
    backends: list[Capability] = field(default_factory=list)
    health: DeviceStatus = field(default_factory=DeviceStatus)
    limits: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "manifestId": self.manifest_id,
            "manifestVersion": self.manifest_version,
            "device": dict(self.device),
            "runtime": dict(self.runtime),
            "transports": dict(self.transports),
            "modalities": dict(self.modalities),
            "capabilities": [item.to_dict() for item in self.capabilities],
            "backends": [item.to_dict() for item in self.backends],
            "health": self.health.to_dict(),
            "limits": dict(self.limits),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "CapabilityManifest":
        capabilities = data.get("capabilities")
        backends = data.get("backends")
        health = data.get("health")
        return cls(
            manifest_id=str(data.get("manifestId", data.get("manifest_id", "")) or ""),
            manifest_version=str(data.get("manifestVersion", data.get("manifest_version", "0.1.0")) or "0.1.0"),
            device=_dict(data.get("device") if isinstance(data.get("device"), Mapping) else None),
            runtime=_dict(data.get("runtime") if isinstance(data.get("runtime"), Mapping) else None),
            transports=_dict(data.get("transports") if isinstance(data.get("transports"), Mapping) else None),
            modalities=_dict(data.get("modalities") if isinstance(data.get("modalities"), Mapping) else None),
            capabilities=[
                Capability.from_dict(item)
                for item in _list(capabilities if isinstance(capabilities, (list, tuple)) else None)
                if isinstance(item, Mapping)
            ],
            backends=[
                Capability.from_dict(item)
                for item in _list(backends if isinstance(backends, (list, tuple)) else None)
                if isinstance(item, Mapping)
            ],
            health=DeviceStatus.from_dict(health if isinstance(health, Mapping) else None),
            limits=_dict(data.get("limits") if isinstance(data.get("limits"), Mapping) else None),
            metadata=_dict(data.get("metadata") if isinstance(data.get("metadata"), Mapping) else None),
        )

    def to_event(
        self,
        *,
        event_id: str,
        request_id: str,
        sequence: int,
        source: SourceRef,
        time: str,
        target: TargetRef | None = None,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_id=event_id,
            event_type="capability",
            name="ei.capability.manifest.report",
            time=time,
            sequence=sequence,
            request_id=request_id,
            source=source,
            target=target,
            content=self.to_content(),
            priority="normal",
        )


@dataclass(slots=True)
class AudioTurn:
    text: str
    language: str = "und"
    final: bool = True
    confidence: float | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    audio_level: float | None = None
    wake_word: str = ""
    asr_backend: str = ""
    timings_ms: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "final": self.final,
            "confidence": self.confidence,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "audioLevel": self.audio_level,
            "wakeWord": self.wake_word,
            "asrBackend": self.asr_backend,
            "timingsMs": dict(self.timings_ms),
            "metadata": dict(self.metadata),
        }

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="dialogue",
            name="ei.dialogue.asr.final" if self.final else "ei.dialogue.asr.partial",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


@dataclass(slots=True)
class HeadStatusReport:
    status: str
    components: dict[str, Any] = field(default_factory=dict)
    reported_at: str = ""
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "components": dict(self.components),
            "reportedAt": self.reported_at,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "HeadStatusReport":
        components = data.get("components")
        metadata = data.get("metadata")
        return cls(
            status=str(data.get("status", "") or ""),
            components=_dict(components if isinstance(components, Mapping) else None),
            reported_at=str(data.get("reportedAt", data.get("reported_at", "")) or ""),
            summary=str(data.get("summary", "") or ""),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )


@dataclass(slots=True)
class DialogueFastHypothesis:
    hypothesis_id: str
    text: str
    confidence: float
    basis_event_id: str = ""
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "hypothesisId": self.hypothesis_id,
            "text": self.text,
            "confidence": float(self.confidence),
            "basisEventId": self.basis_event_id,
            "latencyMs": self.latency_ms,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "DialogueFastHypothesis":
        metadata = data.get("metadata")
        return cls(
            hypothesis_id=str(data.get("hypothesisId", data.get("hypothesis_id", "")) or ""),
            text=str(data.get("text", "") or ""),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            basis_event_id=str(data.get("basisEventId", data.get("basis_event_id", "")) or ""),
            latency_ms=_optional_float(data.get("latencyMs", data.get("latency_ms"))),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="dialogue",
            name="ei.dialogue.fast_hypothesis",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


@dataclass(slots=True)
class DialogueStableDecision:
    decision_id: str
    decision: str
    confidence: float
    text: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    stable_since_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "decisionId": self.decision_id,
            "decision": self.decision,
            "confidence": float(self.confidence),
            "text": self.text,
            "actions": [dict(item) for item in self.actions],
            "stableSinceMs": self.stable_since_ms,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "DialogueStableDecision":
        actions = data.get("actions")
        metadata = data.get("metadata")
        return cls(
            decision_id=str(data.get("decisionId", data.get("decision_id", "")) or ""),
            decision=str(data.get("decision", "") or ""),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            text=str(data.get("text", "") or ""),
            actions=[dict(item) for item in actions] if isinstance(actions, list) else [],
            stable_since_ms=_optional_float(data.get("stableSinceMs", data.get("stable_since_ms"))),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="dialogue",
            name="ei.dialogue.decision.stable",
            content=self.to_content(),
            priority="high",
            **kwargs,
        )


@dataclass(slots=True)
class EmotionContext:
    context_id: str
    mood: str
    confidence: float
    signals: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "contextId": self.context_id,
            "mood": self.mood,
            "confidence": float(self.confidence),
            "signals": dict(self.signals),
            "environment": dict(self.environment),
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "EmotionContext":
        signals = data.get("signals")
        if not isinstance(signals, Mapping) and isinstance(data.get("prosody"), Mapping):
            signals = {"prosody": dict(data.get("prosody") or {})}
        environment = data.get("environment")
        metadata = data.get("metadata")
        return cls(
            context_id=str(data.get("contextId", data.get("context_id", "")) or ""),
            mood=str(data.get("mood", data.get("state", "")) or ""),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            signals=_dict(signals if isinstance(signals, Mapping) else None),
            environment=_dict(environment if isinstance(environment, Mapping) else None),
            source=str(data.get("source", "") or ""),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="observation",
            name="ei.observation.emotion.context",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


@dataclass(slots=True)
class MemoryPrefetchRequest:
    prefetch_id: str
    query: str
    reason: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    scope: list[str] = field(default_factory=list)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "prefetchId": self.prefetch_id,
            "query": self.query,
            "reason": self.reason,
            "candidates": [dict(item) for item in self.candidates],
            "scope": list(self.scope),
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "MemoryPrefetchRequest":
        candidates = data.get("candidates")
        scope = data.get("scope")
        metadata = data.get("metadata")
        return cls(
            prefetch_id=str(data.get("prefetchId", data.get("prefetch_id", "")) or ""),
            query=str(data.get("query", "") or ""),
            reason=str(data.get("reason", "") or ""),
            candidates=_dict_items(candidates),
            scope=[str(item) for item in _list(scope if isinstance(scope, (list, tuple)) else None)],
            source=str(data.get("source", "") or ""),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="memory",
            name="ei.memory.prefetch.requested",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


@dataclass(slots=True)
class MemoryPolicyReport:
    policy_id: str
    scope: dict[str, Any]
    decision: str
    reason: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    writes: list[dict[str, Any]] = field(default_factory=list)
    filters: list[dict[str, Any]] = field(default_factory=list)
    conflict_resolution: dict[str, Any] = field(default_factory=dict)
    persona_consistency_signals: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        content = {
            "policyId": self.policy_id,
            "scope": dict(self.scope),
            "decision": self.decision,
            "reason": self.reason,
            "evidence": [dict(item) for item in self.evidence],
            "metadata": dict(self.metadata),
        }
        if self.writes:
            content["writes"] = [dict(item) for item in self.writes]
        if self.filters:
            content["filters"] = [dict(item) for item in self.filters]
        if self.conflict_resolution:
            content["conflictResolution"] = dict(self.conflict_resolution)
        if self.persona_consistency_signals:
            content["personaConsistencySignals"] = [dict(item) for item in self.persona_consistency_signals]
        return content

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "MemoryPolicyReport":
        scope = data.get("scope")
        evidence = data.get("evidence")
        writes = data.get("writes")
        filters = data.get("filters")
        conflict_resolution = data.get("conflictResolution", data.get("conflict_resolution"))
        persona_consistency_signals = data.get(
            "personaConsistencySignals",
            data.get("persona_consistency_signals"),
        )
        metadata = data.get("metadata")
        return cls(
            policy_id=str(data.get("policyId", data.get("policy_id", "")) or ""),
            scope=_dict(scope if isinstance(scope, Mapping) else None),
            decision=str(data.get("decision", "") or ""),
            reason=str(data.get("reason", "") or ""),
            evidence=_dict_items(evidence),
            writes=_dict_items(writes),
            filters=_dict_items(filters),
            conflict_resolution=_dict(conflict_resolution if isinstance(conflict_resolution, Mapping) else None),
            persona_consistency_signals=_dict_items(persona_consistency_signals),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="memory",
            name="ei.memory.policy.report",
            content=self.to_content(),
            priority="normal",
            **kwargs,
        )


@dataclass(slots=True)
class SpeechActionPlan:
    plan_id: str
    stable: bool
    speech_segments: list[dict[str, Any]] = field(default_factory=list)
    action_segments: list[dict[str, Any]] = field(default_factory=list)
    language: str = "zh-CN"
    fallback_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "planId": self.plan_id,
            "stable": bool(self.stable),
            "speechSegments": [dict(item) for item in self.speech_segments],
            "actionSegments": [dict(item) for item in self.action_segments],
            "language": self.language,
            "fallbackText": self.fallback_text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "SpeechActionPlan":
        speech_segments = data.get("speechSegments", data.get("speech_segments", data.get("speech")))
        action_segments = data.get("actionSegments", data.get("action_segments", data.get("actions")))
        metadata = data.get("metadata")
        return cls(
            plan_id=str(data.get("planId", data.get("plan_id", "")) or ""),
            stable=bool(data.get("stable", False)),
            speech_segments=_dict_items(speech_segments),
            action_segments=_dict_items(action_segments),
            language=str(data.get("language", "zh-CN") or "zh-CN"),
            fallback_text=str(data.get("fallbackText", data.get("fallback_text", "")) or ""),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="dialogue",
            name="ei.dialogue.speech_action.plan",
            content=self.to_content(),
            priority="high",
            **kwargs,
        )


@dataclass(slots=True)
class ProactiveActivityProposal:
    proposal_id: str
    channel: str
    reason: str
    should_emit: bool
    urgency: float | None = None
    disturbance: str = "low"
    requires_user_attention: bool = False
    text: str = ""
    memory_refs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "proposalId": self.proposal_id,
            "channel": self.channel,
            "reason": self.reason,
            "shouldEmit": bool(self.should_emit),
            "urgency": self.urgency,
            "disturbance": self.disturbance,
            "requiresUserAttention": self.requires_user_attention,
            "text": self.text,
            "memoryRefs": [dict(item) for item in self.memory_refs],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "ProactiveActivityProposal":
        memory_refs = data.get("memoryRefs", data.get("memory_refs"))
        metadata = data.get("metadata")
        return cls(
            proposal_id=str(data.get("proposalId", data.get("proposal_id", "")) or ""),
            channel=str(data.get("channel", "") or ""),
            reason=str(data.get("reason", "") or ""),
            should_emit=bool(data.get("shouldEmit", data.get("should_emit", False))),
            urgency=_optional_float(data.get("urgency")),
            disturbance=str(data.get("disturbance", "low") or "low"),
            requires_user_attention=bool(
                data.get("requiresUserAttention", data.get("requires_user_attention", False))
            ),
            text=str(data.get("text", "") or ""),
            memory_refs=_dict_items(memory_refs),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="dialogue",
            name="ei.activity.proactive.proposed",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


@dataclass(slots=True)
class DialogueCancellationApplied:
    cancellation_id: str
    cancelled_round_id: str
    cancellation_token: str
    reason: str
    applied_to: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "cancellationId": self.cancellation_id,
            "cancelledRoundId": self.cancelled_round_id,
            "cancellationToken": self.cancellation_token,
            "reason": self.reason,
            "appliedTo": list(self.applied_to),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "DialogueCancellationApplied":
        applied_to = data.get("appliedTo", data.get("applied_to"))
        metadata = data.get("metadata")
        return cls(
            cancellation_id=str(data.get("cancellationId", data.get("cancellation_id", "")) or ""),
            cancelled_round_id=str(data.get("cancelledRoundId", data.get("cancelled_round_id", "")) or ""),
            cancellation_token=str(data.get("cancellationToken", data.get("cancellation_token", "")) or ""),
            reason=str(data.get("reason", "") or ""),
            applied_to=[str(item) for item in _list(applied_to if isinstance(applied_to, (list, tuple)) else None)],
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="dialogue",
            name="ei.dialogue.cancellation.applied",
            content=self.to_content(),
            priority="high",
            **kwargs,
        )


@dataclass(slots=True)
class Detection:
    label: str
    score: float
    bbox: list[Any] | dict[str, Any]
    track_id: str = ""
    pose: dict[str, Any] = field(default_factory=dict)
    clip_labels: list[dict[str, Any]] = field(default_factory=list)
    semantic_labels: list[dict[str, Any]] = field(default_factory=list)
    depth: dict[str, Any] = field(default_factory=dict)
    distance: dict[str, Any] = field(default_factory=dict)
    tracking_diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "label": self.label,
            "score": float(self.score),
            "bbox": _bbox(self.bbox),
            "trackId": self.track_id,
            "metadata": dict(self.metadata),
        }
        if self.pose:
            payload["pose"] = dict(self.pose)
        if self.clip_labels:
            payload["clipLabels"] = [dict(item) for item in self.clip_labels]
        if self.semantic_labels:
            payload["semanticLabels"] = [dict(item) for item in self.semantic_labels]
        if self.depth:
            payload["depth"] = dict(self.depth)
        if self.distance:
            payload["distance"] = dict(self.distance)
        if self.tracking_diagnostics:
            payload["trackingDiagnostics"] = dict(self.tracking_diagnostics)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Detection":
        pose = data.get("pose")
        clip_labels = data.get("clipLabels", data.get("clip_labels"))
        semantic_labels = data.get("semanticLabels", data.get("semantic_labels"))
        depth = data.get("depth")
        distance = data.get("distance")
        tracking_diagnostics = data.get("trackingDiagnostics", data.get("tracking_diagnostics"))
        return cls(
            label=str(data.get("label", "") or ""),
            score=_score_from_mapping(data),
            bbox=_bbox(data.get("bbox")),
            track_id=str(data.get("trackId", data.get("track_id", "")) or ""),
            pose=_dict(pose if isinstance(pose, Mapping) else None),
            clip_labels=_dict_items(clip_labels),
            semantic_labels=_dict_items(semantic_labels),
            depth=_dict(depth if isinstance(depth, Mapping) else None),
            distance=_dict(distance if isinstance(distance, Mapping) else None),
            tracking_diagnostics=_dict(tracking_diagnostics if isinstance(tracking_diagnostics, Mapping) else None),
            metadata=_dict(data.get("metadata") if isinstance(data.get("metadata"), Mapping) else None),
        )


@dataclass(slots=True)
class RealtimeVisionObservation:
    frame_id: str
    width: int | None = None
    height: int | None = None
    frame_age_ms: float | None = None
    backend: str = ""
    detections: list[Detection] = field(default_factory=list)
    boxes: list[Any] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    tracked_target: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, Any] = field(default_factory=dict)
    tracking_diagnostics: dict[str, Any] = field(default_factory=dict)
    pose: dict[str, Any] = field(default_factory=dict)
    clip_labels: list[dict[str, Any]] = field(default_factory=list)
    semantic_labels: list[dict[str, Any]] = field(default_factory=list)
    depth: dict[str, Any] = field(default_factory=dict)
    distance: dict[str, Any] = field(default_factory=dict)
    image_url: str = ""
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        content = {
            "frameId": self.frame_id,
            "width": self.width,
            "height": self.height,
            "frameAgeMs": self.frame_age_ms,
            "backend": self.backend,
            "detections": [item.to_dict() for item in self.detections],
            "boxes": [_bbox(item) for item in self.boxes],
            "scores": [float(item) for item in self.scores],
            "latencyMs": dict(self.latency_ms),
            "imageUrl": self.image_url,
            "status": self.status,
            "trackedTarget": dict(self.tracked_target),
            "metadata": dict(self.metadata),
        }
        if self.tracking_diagnostics:
            content["trackingDiagnostics"] = dict(self.tracking_diagnostics)
        if self.pose:
            content["pose"] = dict(self.pose)
        if self.clip_labels:
            content["clipLabels"] = [dict(item) for item in self.clip_labels]
        if self.semantic_labels:
            content["semanticLabels"] = [dict(item) for item in self.semantic_labels]
        if self.depth:
            content["depth"] = dict(self.depth)
        if self.distance:
            content["distance"] = dict(self.distance)
        return content

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "RealtimeVisionObservation":
        detections = data.get("detections")
        boxes = data.get("boxes")
        scores = data.get("scores")
        tracked_target = data.get("trackedTarget", data.get("tracked_target"))
        latency_ms = data.get("latencyMs", data.get("latency_ms"))
        tracking_diagnostics = data.get("trackingDiagnostics", data.get("tracking_diagnostics"))
        pose = data.get("pose")
        clip_labels = data.get("clipLabels", data.get("clip_labels"))
        semantic_labels = data.get("semanticLabels", data.get("semantic_labels"))
        depth = data.get("depth")
        distance = data.get("distance")
        metadata = data.get("metadata")
        return cls(
            frame_id=str(data.get("frameId", data.get("frame_id", "")) or ""),
            width=int(data["width"]) if data.get("width") is not None else None,
            height=int(data["height"]) if data.get("height") is not None else None,
            frame_age_ms=float(data["frameAgeMs"]) if data.get("frameAgeMs") is not None else None,
            backend=str(data.get("backend", "") or ""),
            detections=[
                Detection.from_dict(item)
                for item in _list(detections if isinstance(detections, (list, tuple)) else None)
                if isinstance(item, Mapping)
            ],
            boxes=[_bbox(item) for item in _list(boxes if isinstance(boxes, (list, tuple)) else None)],
            scores=[float(item) for item in _list(scores if isinstance(scores, (list, tuple)) else None)],
            tracked_target=_dict(tracked_target if isinstance(tracked_target, Mapping) else None),
            latency_ms=_dict(latency_ms if isinstance(latency_ms, Mapping) else None),
            tracking_diagnostics=_dict(tracking_diagnostics if isinstance(tracking_diagnostics, Mapping) else None),
            pose=_dict(pose if isinstance(pose, Mapping) else None),
            clip_labels=_dict_items(clip_labels),
            semantic_labels=_dict_items(semantic_labels),
            depth=_dict(depth if isinstance(depth, Mapping) else None),
            distance=_dict(distance if isinstance(distance, Mapping) else None),
            image_url=str(data.get("imageUrl", data.get("image_url", "")) or ""),
            status=str(data.get("status", "ok") or "ok"),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="observation",
            name="ei.observation.vision.frame",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


@dataclass(slots=True)
class VisionSceneObservation:
    scene_id: str
    observed_at: str
    summary: str = ""
    objects: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    clip_labels: list[dict[str, Any]] = field(default_factory=list)
    semantic_labels: list[dict[str, Any]] = field(default_factory=list)
    depth: dict[str, Any] = field(default_factory=dict)
    distance: dict[str, Any] = field(default_factory=dict)
    scene_graph: dict[str, Any] = field(default_factory=dict)
    scene_graph_provenance: dict[str, Any] = field(default_factory=dict)
    attention: dict[str, Any] = field(default_factory=dict)
    stable_target: dict[str, Any] = field(default_factory=dict)
    event_summary: str = ""
    tracking_diagnostics: dict[str, Any] = field(default_factory=dict)
    temporal: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    image_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        content = {
            "sceneId": self.scene_id,
            "observedAt": self.observed_at,
            "summary": self.summary,
            "objects": [dict(item) for item in self.objects],
            "relationships": [dict(item) for item in self.relationships],
            "environment": dict(self.environment),
            "imageUrl": self.image_url,
            "metadata": dict(self.metadata),
        }
        if self.clip_labels:
            content["clipLabels"] = [dict(item) for item in self.clip_labels]
        if self.semantic_labels:
            content["semanticLabels"] = [dict(item) for item in self.semantic_labels]
        if self.depth:
            content["depth"] = dict(self.depth)
        if self.distance:
            content["distance"] = dict(self.distance)
        if self.scene_graph:
            content["sceneGraph"] = dict(self.scene_graph)
        if self.scene_graph_provenance:
            content["sceneGraphProvenance"] = dict(self.scene_graph_provenance)
        if self.attention:
            content["attention"] = dict(self.attention)
        if self.stable_target:
            content["stableTarget"] = dict(self.stable_target)
        if self.event_summary:
            content["eventSummary"] = self.event_summary
        if self.tracking_diagnostics:
            content["trackingDiagnostics"] = dict(self.tracking_diagnostics)
        if self.temporal:
            content["temporal"] = dict(self.temporal)
        if self.events:
            content["events"] = [dict(item) for item in self.events]
        return content

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "VisionSceneObservation":
        objects = data.get("objects")
        relationships = data.get("relationships")
        environment = data.get("environment")
        clip_labels = data.get("clipLabels", data.get("clip_labels"))
        semantic_labels = data.get("semanticLabels", data.get("semantic_labels"))
        depth = data.get("depth")
        distance = data.get("distance")
        scene_graph = data.get("sceneGraph", data.get("scene_graph"))
        scene_graph_provenance = data.get("sceneGraphProvenance", data.get("scene_graph_provenance"))
        attention = data.get("attention")
        stable_target = data.get("stableTarget", data.get("stable_target"))
        tracking_diagnostics = data.get("trackingDiagnostics", data.get("tracking_diagnostics"))
        temporal = data.get("temporal")
        events = data.get("events")
        metadata = data.get("metadata")
        return cls(
            scene_id=str(data.get("sceneId", data.get("scene_id", "")) or ""),
            observed_at=str(data.get("observedAt", data.get("observed_at", "")) or ""),
            summary=str(data.get("summary", "") or ""),
            objects=_dict_items(objects),
            relationships=_dict_items(relationships),
            environment=_dict(environment if isinstance(environment, Mapping) else None),
            clip_labels=_dict_items(clip_labels),
            semantic_labels=_dict_items(semantic_labels),
            depth=_dict(depth if isinstance(depth, Mapping) else None),
            distance=_dict(distance if isinstance(distance, Mapping) else None),
            scene_graph=_dict(scene_graph if isinstance(scene_graph, Mapping) else None),
            scene_graph_provenance=_dict(scene_graph_provenance if isinstance(scene_graph_provenance, Mapping) else None),
            attention=_dict(attention if isinstance(attention, Mapping) else None),
            stable_target=_dict(stable_target if isinstance(stable_target, Mapping) else None),
            event_summary=str(data.get("eventSummary", data.get("event_summary", "")) or ""),
            tracking_diagnostics=_dict(tracking_diagnostics if isinstance(tracking_diagnostics, Mapping) else None),
            temporal=_dict(temporal if isinstance(temporal, Mapping) else None),
            events=_dict_items(events),
            image_url=str(data.get("imageUrl", data.get("image_url", "")) or ""),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        kwargs.setdefault("session_id", "")
        kwargs.setdefault("round_id", "")
        return _round_event(
            event_type="observation",
            name="ei.observation.vision.scene",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


@dataclass(slots=True)
class VisionEventObservation:
    event_id: str
    event_type: str
    observed_at: str
    scene_id: str = ""
    subject: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    pose: dict[str, Any] = field(default_factory=dict)
    clip_labels: list[dict[str, Any]] = field(default_factory=list)
    semantic_labels: list[dict[str, Any]] = field(default_factory=list)
    depth: dict[str, Any] = field(default_factory=dict)
    distance: dict[str, Any] = field(default_factory=dict)
    scene_graph_provenance: dict[str, Any] = field(default_factory=dict)
    tracking_diagnostics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        content = {
            "eventId": self.event_id,
            "eventType": self.event_type,
            "observedAt": self.observed_at,
            "sceneId": self.scene_id,
            "subject": dict(self.subject),
            "confidence": self.confidence,
            "details": dict(self.details),
            "metadata": dict(self.metadata),
        }
        if self.pose:
            content["pose"] = dict(self.pose)
        if self.clip_labels:
            content["clipLabels"] = [dict(item) for item in self.clip_labels]
        if self.semantic_labels:
            content["semanticLabels"] = [dict(item) for item in self.semantic_labels]
        if self.depth:
            content["depth"] = dict(self.depth)
        if self.distance:
            content["distance"] = dict(self.distance)
        if self.scene_graph_provenance:
            content["sceneGraphProvenance"] = dict(self.scene_graph_provenance)
        if self.tracking_diagnostics:
            content["trackingDiagnostics"] = dict(self.tracking_diagnostics)
        return content

    @classmethod
    def from_content(cls, data: Mapping[str, Any]) -> "VisionEventObservation":
        subject = data.get("subject")
        pose = data.get("pose")
        clip_labels = data.get("clipLabels", data.get("clip_labels"))
        semantic_labels = data.get("semanticLabels", data.get("semantic_labels"))
        depth = data.get("depth")
        distance = data.get("distance")
        scene_graph_provenance = data.get("sceneGraphProvenance", data.get("scene_graph_provenance"))
        tracking_diagnostics = data.get("trackingDiagnostics", data.get("tracking_diagnostics"))
        details = data.get("details")
        metadata = data.get("metadata")
        return cls(
            event_id=str(data.get("eventId", data.get("event_id", "")) or ""),
            event_type=str(data.get("eventType", data.get("event_type", "")) or ""),
            observed_at=str(data.get("observedAt", data.get("observed_at", "")) or ""),
            scene_id=str(data.get("sceneId", data.get("scene_id", "")) or ""),
            subject=_dict(subject if isinstance(subject, Mapping) else None),
            confidence=_optional_float(data.get("confidence")),
            pose=_dict(pose if isinstance(pose, Mapping) else None),
            clip_labels=_dict_items(clip_labels),
            semantic_labels=_dict_items(semantic_labels),
            depth=_dict(depth if isinstance(depth, Mapping) else None),
            distance=_dict(distance if isinstance(distance, Mapping) else None),
            scene_graph_provenance=_dict(scene_graph_provenance if isinstance(scene_graph_provenance, Mapping) else None),
            tracking_diagnostics=_dict(tracking_diagnostics if isinstance(tracking_diagnostics, Mapping) else None),
            details=_dict(details if isinstance(details, Mapping) else None),
            metadata=_dict(metadata if isinstance(metadata, Mapping) else None),
        )

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        kwargs.setdefault("session_id", "")
        kwargs.setdefault("round_id", "")
        return _round_event(
            event_type="observation",
            name="ei.observation.vision.event",
            content=self.to_content(),
            priority="realtime",
            **kwargs,
        )


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


@dataclass(slots=True)
class HeadAction:
    action_id: str
    action_type: str
    target: str
    params: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "L1"
    idempotency_key: str = ""
    timeline: list[dict[str, Any]] = field(default_factory=list)
    requires_policy: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        content = {
            "actionId": self.action_id,
            "actionType": self.action_type,
            "target": self.target,
            "params": dict(self.params),
            "riskLevel": self.risk_level,
            "timeline": [dict(item) for item in self.timeline],
            "requiresPolicy": self.requires_policy,
            "metadata": dict(self.metadata),
        }
        if self.idempotency_key:
            content["idempotencyKey"] = self.idempotency_key
        return content

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        event = _round_event(
            event_type="action",
            name="ei.action.request",
            content=self.to_content(),
            priority="high",
            **kwargs,
        )
        event.policy = PolicyState(decision="not_required", risk_level=self.risk_level)
        return event


@dataclass(slots=True)
class ExecutionOutcome:
    outcome_id: str
    action_id: str = ""
    action_type: str = ""
    success: bool = True
    status: str = "completed"
    latency_ms: float | None = None
    did_what: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "outcomeId": self.outcome_id,
            "actionId": self.action_id,
            "actionType": self.action_type,
            "success": self.success,
            "status": self.status,
            "latencyMs": self.latency_ms,
            "didWhat": list(self.did_what),
            "errors": [dict(item) for item in self.errors],
            "details": dict(self.details),
        }

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="outcome",
            name="ei.outcome.execution",
            content=self.to_content(),
            priority="normal",
            **kwargs,
        )


@dataclass(slots=True)
class UserFeedback:
    feedback_id: str
    satisfied: bool | None = None
    rating: int | None = None
    text: str = ""
    next_time_change: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> dict[str, Any]:
        return {
            "feedbackId": self.feedback_id,
            "satisfied": self.satisfied,
            "rating": self.rating,
            "text": self.text,
            "nextTimeChange": self.next_time_change,
            "metadata": dict(self.metadata),
        }

    def to_event(self, **kwargs: Any) -> EventEnvelope:
        return _round_event(
            event_type="outcome",
            name="ei.outcome.user.feedback",
            content=self.to_content(),
            priority="normal",
            **kwargs,
        )


def _round_event(
    *,
    event_type: str,
    name: str,
    content: dict[str, Any],
    priority: str,
    event_id: str,
    request_id: str,
    session_id: str,
    round_id: str,
    sequence: int,
    source: SourceRef,
    time: str,
    target: TargetRef | None = None,
    correlation_id: str = "",
    causation_id: str = "",
    trace_id: str = "",
    ttl_ms: int | None = None,
    mode: dict[str, Any] | None = None,
    extensions: dict[str, Any] | None = None,
) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id,
        event_type=event_type,
        name=name,
        time=time,
        sequence=sequence,
        request_id=request_id,
        session_id=session_id,
        round_id=round_id,
        correlation_id=correlation_id,
        causation_id=causation_id,
        trace_id=trace_id,
        source=source,
        target=target,
        priority=priority,
        ttl_ms=ttl_ms,
        mode=dict(mode or {}),
        content=content,
        policy=PolicyState(),
        extensions=dict(extensions or {}),
    )


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            items.append(dict(item))
        elif isinstance(item, str) and item.strip():
            items.append({"label": item.strip()})
    return items


def _optional_float(value: Any) -> float | None:
    return float(value) if value not in (None, "") else None


def _score_from_mapping(data: Mapping[str, Any]) -> float:
    value = data.get("score", data.get("confidence", 0.0))
    return float(value or 0.0)


def validate_event(event: EventEnvelope | Mapping[str, Any]) -> list[str]:
    payload = event.to_dict() if isinstance(event, EventEnvelope) else dict(event)
    errors: list[str] = []
    required = ("specVersion", "id", "type", "name", "time", "sequence", "requestId", "source", "priority", "content", "policy")
    for key in required:
        if key not in payload or payload.get(key) in (None, ""):
            errors.append(f"{key} is required")
    if payload.get("specVersion") != SPEC_VERSION:
        errors.append("specVersion must be eiprotocol/0.1")

    name = str(payload.get("name", ""))
    event_type = str(payload.get("type", ""))
    if event_type in {"dialogue", "action", "memory", "outcome", "training"} and not payload.get("roundId"):
        errors.append("roundId is required for turn-scoped events")

    content = payload.get("content")
    if not isinstance(content, Mapping):
        errors.append("content must be an object")
        return errors
    if name in {"ei.action.request", "ei.action.dispatch", "ei.action.emergency.stop"} and not content.get("idempotencyKey"):
        errors.append("content.idempotencyKey is required for side-effecting action events")
    return errors
