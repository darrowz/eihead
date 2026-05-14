"""Strict validation helpers for eiprotocol v0.1 envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .catalog import get_event_definition
from .models import validate_event


VALID_EVENT_TYPES = {
    "action",
    "capability",
    "control",
    "dialogue",
    "error",
    "memory",
    "observation",
    "outcome",
    "policy",
    "training",
}
VALID_PRIORITIES = {"low", "normal", "high", "realtime", "emergency"}
VALID_DOMAINS = {
    "eihead",
    "eibrain",
    "eiprotocol",
    "eimemory",
    "eibody",
    "eiskills",
    "eidocs",
    "eitraining",
    "safety_gate",
    "user",
    "external",
}
VALID_POLICY_DECISIONS = {
    "not_required",
    "allow",
    "confirm",
    "deny",
    "pause",
    "pending",
    "emergency_stop",
}
VALID_POLICY_RISK_LEVELS = {"L0", "L1", "L2", "L3", "L4"}
SIDE_EFFECTING_ACTION_NAMES = {
    "ei.action.request",
    "ei.action.dispatch",
    "ei.action.emergency.stop",
}
FALLBACK_KNOWN_EVENT_NAMES = {
    "ei.control.hello",
    "ei.control.ping",
    "ei.control.pong",
    "ei.control.resume",
    "ei.control.ack",
    "ei.control.error",
    "ei.error.event",
    "ei.capability.manifest.report",
    "ei.observation.audio.chunk",
    "ei.observation.vision.frame",
    "ei.observation.head.status.report",
    "ei.dialogue.asr.partial",
    "ei.dialogue.asr.final",
    "ei.dialogue.fast_hypothesis",
    "ei.dialogue.decision.stable",
    "ei.dialogue.agent.delta",
    "ei.dialogue.agent.final",
    "ei.dialogue.tts.delta",
    "ei.dialogue.tts.final",
    "ei.dialogue.interrupt.requested",
    "ei.action.request",
    "ei.action.dispatch",
    "ei.action.progress",
    "ei.action.complete",
    "ei.action.emergency.stop",
    "ei.policy.decision",
    "ei.memory.recall.request",
    "ei.memory.recall.result",
    "ei.memory.write.proposed",
    "ei.memory.write.committed",
    "ei.outcome.execution",
    "ei.outcome.user.feedback",
    "ei.training.signal",
}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: str
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


def validate_event_strict(event: Any, *, known_event_required: bool = False) -> list[ValidationIssue]:
    """Return structured v0.1 validation issues without mutating the event."""
    payload, issues = _coerce_event(event)
    if issues:
        return issues

    issues.extend(_base_issues(validate_event(payload)))
    _validate_enum(payload, issues, "type", VALID_EVENT_TYPES, "invalid_type")
    _validate_enum(payload, issues, "priority", VALID_PRIORITIES, "invalid_priority")
    _validate_policy(payload, issues)
    _validate_ref_domain(payload, issues, "source")
    _validate_ref_domain(payload, issues, "target")
    _validate_sequence(payload, issues)
    _validate_ttl_ms(payload, issues)
    _validate_time(payload, issues)
    _validate_content(payload, issues)
    _validate_side_effect_idempotency(payload, issues)
    _validate_catalog_contract(payload, issues)

    if known_event_required:
        _validate_known_event(payload, issues)

    return issues


def assert_event_valid(event: Any) -> None:
    issues = validate_event_strict(event)
    if not issues:
        return

    details = "; ".join(f"{issue.code} at {issue.path}: {issue.message}" for issue in issues)
    raise ValueError(f"Invalid eiprotocol event: {details}")


def _coerce_event(event: Any) -> tuple[dict[str, Any], list[ValidationIssue]]:
    if isinstance(event, Mapping):
        return dict(event), []

    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception as exc:  # pragma: no cover - defensive for EventEnvelope-like inputs.
            return {}, [_issue("$", "invalid_event", f"to_dict() failed: {exc}")]
        if isinstance(payload, Mapping):
            return dict(payload), []
        return {}, [_issue("$", "invalid_event", "to_dict() must return a mapping")]

    return {}, [_issue("$", "invalid_event", "event must be a mapping or provide to_dict()")]


def _base_issues(errors: list[str]) -> list[ValidationIssue]:
    return [_base_issue(error) for error in errors]


def _base_issue(error: str) -> ValidationIssue:
    if error.endswith(" is required"):
        path = error.removesuffix(" is required")
        return _issue(path, "required", error)
    if error == "specVersion must be eiprotocol/0.1":
        return _issue("specVersion", "invalid_spec_version", error)
    if error == "roundId is required for turn-scoped events":
        return _issue("roundId", "required", error)
    if error == "content must be an object":
        return _issue("content", "invalid_content", error)
    if error == "content.idempotencyKey is required for side-effecting action events":
        return _issue("content.idempotencyKey", "missing_idempotency_key", error)
    return _issue("$", "base_validation", error)


def _validate_enum(
    payload: Mapping[str, Any],
    issues: list[ValidationIssue],
    path: str,
    allowed: set[str],
    code: str,
) -> None:
    value = payload.get(path)
    if value in (None, ""):
        return
    if not isinstance(value, str) or value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        issues.append(_issue(path, code, f"{path} must be one of {allowed_text}"))


def _validate_policy(payload: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    policy = payload.get("policy")
    if policy in (None, ""):
        return
    if not isinstance(policy, Mapping):
        issues.append(_issue("policy", "invalid_object", "policy must be an object"))
        return

    _validate_optional_policy_enum(policy, issues, "decision", VALID_POLICY_DECISIONS, "invalid_policy_decision")
    _validate_optional_policy_enum(policy, issues, "riskLevel", VALID_POLICY_RISK_LEVELS, "invalid_policy_risk_level")


def _validate_optional_policy_enum(
    policy: Mapping[str, Any],
    issues: list[ValidationIssue],
    key: str,
    allowed: set[str],
    code: str,
) -> None:
    path = f"policy.{key}"
    value = policy.get(key)
    if value in (None, ""):
        return
    if not isinstance(value, str) or value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        issues.append(_issue(path, code, f"{path} must be one of {allowed_text}"))


def _validate_ref_domain(payload: Mapping[str, Any], issues: list[ValidationIssue], key: str) -> None:
    ref = payload.get(key)
    if ref in (None, ""):
        return
    if not isinstance(ref, Mapping):
        issues.append(_issue(key, "invalid_object", f"{key} must be an object"))
        return

    domain = ref.get("domain")
    if domain in (None, ""):
        return
    if not isinstance(domain, str) or domain not in VALID_DOMAINS:
        issues.append(_issue(f"{key}.domain", "invalid_domain", f"{key}.domain must be one of {_domains_text()}"))


def _validate_sequence(payload: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    if "sequence" not in payload or payload.get("sequence") in (None, ""):
        return
    sequence = payload.get("sequence")
    if not _is_int(sequence) or sequence <= 0:
        issues.append(_issue("sequence", "invalid_sequence", "sequence must be a positive integer"))


def _validate_ttl_ms(payload: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    if "ttlMs" not in payload or payload.get("ttlMs") is None:
        return
    ttl_ms = payload.get("ttlMs")
    if not _is_int(ttl_ms) or ttl_ms < 0:
        issues.append(_issue("ttlMs", "invalid_ttl_ms", "ttlMs must be a non-negative integer"))


def _validate_time(payload: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    value = payload.get("time")
    if value in (None, ""):
        return
    if not isinstance(value, str):
        issues.append(_issue("time", "invalid_datetime", "time must be an ISO 8601 datetime string"))
        return

    candidate = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        issues.append(_issue("time", "invalid_datetime", "time must be accepted by datetime.fromisoformat"))
        return
    if "T" not in value or parsed.tzinfo is None or parsed.utcoffset() is None:
        issues.append(_issue("time", "invalid_datetime", "time must be an RFC3339 datetime with timezone"))


def _validate_content(payload: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    content = payload.get("content")
    if content in (None, ""):
        return
    if not isinstance(content, Mapping) and not _has_issue(issues, "content", "invalid_content"):
        issues.append(_issue("content", "invalid_content", "content must be an object"))


def _validate_side_effect_idempotency(payload: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    if payload.get("name") not in SIDE_EFFECTING_ACTION_NAMES:
        return
    content = payload.get("content")
    if not isinstance(content, Mapping):
        return
    if not content.get("idempotencyKey") and not _has_issue(issues, "content.idempotencyKey", "missing_idempotency_key"):
        issues.append(
            _issue(
                "content.idempotencyKey",
                "missing_idempotency_key",
                "content.idempotencyKey is required for side-effecting action events",
            )
        )


def _validate_known_event(payload: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    name = payload.get("name")
    if name in (None, ""):
        return
    if not isinstance(name, str) or not _is_known_event(name):
        issues.append(_issue("name", "unknown_event", f"unknown eiprotocol event name: {name}"))


def _validate_catalog_contract(payload: Mapping[str, Any], issues: list[ValidationIssue]) -> None:
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        return
    definition = get_event_definition(name)
    if definition is None:
        return

    event_type = payload.get("type")
    if (
        event_type not in (None, "")
        and event_type != definition.event_type
        and not _has_issue(issues, "type", "invalid_type")
    ):
        issues.append(
            _issue(
                "type",
                "event_type_mismatch",
                f"type must be {definition.event_type} for event name {name}",
            )
        )

    if definition.round_scoped and not payload.get("roundId"):
        if not _has_issue(issues, "roundId", "required"):
            issues.append(_issue("roundId", "required", "roundId is required for this event name"))

    content = payload.get("content")
    if not isinstance(content, Mapping):
        return
    for field_name in definition.required_content_fields:
        if content.get(field_name) in (None, ""):
            issues.append(
                _issue(
                    f"content.{field_name}",
                    "missing_content_field",
                    f"content.{field_name} is required for event name {name}",
                )
            )


def _is_known_event(name: str) -> bool:
    return get_event_definition(name) is not None or name in FALLBACK_KNOWN_EVENT_NAMES


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _has_issue(issues: list[ValidationIssue], path: str, code: str) -> bool:
    return any(issue.path == path and issue.code == code for issue in issues)


def _domains_text() -> str:
    return ", ".join(sorted(VALID_DOMAINS))


def _issue(path: str, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(path=path, code=code, message=message, severity="error")


__all__ = ["ValidationIssue", "assert_event_valid", "validate_event_strict"]
