"""Pure eiprotocol event envelope routing helpers."""

from __future__ import annotations

from typing import Any, Mapping

from .catalog import get_event_definition
from .validation import ValidationIssue, validate_event_strict


_ROUTE_NAMES = {
    "ei.control.hello": "control_hello",
    "ei.control.ping": "control_ping",
    "ei.control.pong": "control_pong",
    "ei.control.resume": "control_resume",
    "ei.control.ack": "control_ack",
    "ei.control.error": "control_error",
    "ei.error.event": "error_event",
    "ei.capability.manifest.report": "capability_manifest",
    "ei.observation.audio.chunk": "audio_chunk",
    "ei.voice.audio.frame": "voice_audio_frame",
    "ei.observation.vision.frame": "realtime_vision_frame",
    "ei.observation.vision.scene": "vision_scene",
    "ei.observation.vision.event": "vision_event",
    "ei.observation.head.status.report": "head_status_report",
    "ei.observation.emotion.context": "emotion_context",
    "ei.dialogue.asr.partial": "asr_partial",
    "ei.dialogue.asr.final": "asr_final",
    "ei.voice.asr.partial": "voice_asr_partial",
    "ei.voice.asr.final": "voice_asr_final",
    "ei.dialogue.fast_hypothesis": "dialogue_fast_hypothesis",
    "ei.dialogue.decision.stable": "dialogue_decision_stable",
    "ei.dialogue.speech_action.plan": "speech_action_plan",
    "ei.dialogue.cancellation.applied": "dialogue_cancellation_applied",
    "ei.dialogue.agent.delta": "agent_delta",
    "ei.dialogue.agent.final": "agent_final",
    "ei.dialogue.tts.delta": "tts_delta",
    "ei.dialogue.tts.final": "tts_final",
    "ei.voice.tts.sentence_start": "voice_tts_sentence_start",
    "ei.voice.tts.chunk": "voice_tts_chunk",
    "ei.voice.playback.started": "voice_playback_started",
    "ei.voice.playback.stopped": "voice_playback_stopped",
    "ei.voice.barge_in.detected": "voice_barge_in_detected",
    "ei.dialogue.interrupt.requested": "interrupt_requested",
    "ei.voice.session.heartbeat": "voice_session_heartbeat",
    "ei.action.request": "action_request",
    "ei.action.dispatch": "action_dispatch",
    "ei.action.progress": "action_progress",
    "ei.action.complete": "action_complete",
    "ei.action.emergency.stop": "action_emergency_stop",
    "ei.policy.decision": "policy_decision",
    "ei.memory.recall.request": "memory_recall_request",
    "ei.memory.prefetch.requested": "memory_prefetch_requested",
    "ei.memory.policy.report": "memory_policy_report",
    "ei.memory.recall.result": "memory_recall_result",
    "ei.memory.write.proposed": "memory_write_proposed",
    "ei.memory.write.committed": "memory_write_committed",
    "ei.outcome.execution": "execution_outcome",
    "ei.outcome.user.feedback": "user_feedback",
    "ei.activity.proactive.proposed": "proactive_activity_proposed",
    "ei.training.signal": "training_signal",
}

_ACTION_CONTENT_FIELDS = (
    "actionId",
    "actionType",
    "target",
    "params",
    "riskLevel",
    "idempotencyKey",
)


def classify_event(event: Any) -> dict[str, Any]:
    """Classify an eiprotocol envelope into a JSON-friendly route description."""
    payload, coercion_errors = _coerce_event(event)
    if coercion_errors:
        return _invalid_route(payload, coercion_errors)

    validation_errors = [_issue_to_error(issue) for issue in validate_event_strict(payload)]
    if validation_errors:
        return _invalid_route(payload, validation_errors)

    event_name = _text(payload.get("name"))
    event_type = _text(payload.get("type"))
    definition = get_event_definition(event_name)
    route_name = _ROUTE_NAMES.get(event_name)
    if definition is None or route_name is None:
        return {
            "status": "not_processed",
            "reason": "unsupported_event_name",
            "eventName": event_name,
            "eventType": event_type,
        }

    route_description: dict[str, Any] = {
        "status": "routed",
        "eventName": event_name,
        "eventType": event_type,
        "route": route_name,
        "plane": definition.plane,
        "sideEffecting": definition.side_effecting,
        "roundScoped": definition.round_scoped,
        "realtime": definition.realtime,
        "knownEvent": True,
    }
    if definition.event_type == "action":
        route_description.update(_action_fields(payload))
    return route_description


def _coerce_event(event: Any) -> tuple[dict[str, Any], list[str]]:
    if isinstance(event, Mapping):
        return dict(event), []

    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception as exc:  # pragma: no cover - defensive for EventEnvelope-like inputs.
            return {}, [f"to_dict failed: {exc}"]
        if isinstance(payload, Mapping):
            return dict(payload), []
        return {}, ["to_dict must return a mapping"]

    return {}, ["event must be a mapping or provide to_dict()"]


def _invalid_route(payload: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    return {
        "status": "invalid",
        "reason": "invalid_event",
        "eventName": _text(payload.get("name")),
        "eventType": _text(payload.get("type")),
        "errors": list(errors),
    }


def _action_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, Mapping):
        return {field: "" for field in _ACTION_CONTENT_FIELDS}

    params = content.get("params")
    risk_level = content.get("riskLevel")
    if not risk_level:
        policy = payload.get("policy")
        if isinstance(policy, Mapping):
            risk_level = policy.get("riskLevel")

    return {
        "actionId": _text(content.get("actionId")),
        "actionType": _text(content.get("actionType")),
        "target": _text(content.get("target")),
        "params": dict(params) if isinstance(params, Mapping) else {},
        "riskLevel": _text(risk_level),
        "idempotencyKey": _text(content.get("idempotencyKey")),
    }


def _text(value: Any) -> str:
    return str(value or "")


def _issue_to_error(issue: ValidationIssue) -> str:
    if issue.code in {"required", "invalid_spec_version", "invalid_content", "missing_idempotency_key"}:
        return issue.message
    return f"{issue.code} at {issue.path}: {issue.message}"


__all__ = ["classify_event"]
