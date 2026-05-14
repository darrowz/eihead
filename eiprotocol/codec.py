"""Safe JSON codec helpers for eiprotocol event envelopes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .models import EventEnvelope
from .validation import ValidationIssue, validate_event_strict


class EventDecodeError(ValueError):
    """Structured error raised when incoming event JSON cannot be decoded."""

    def __init__(self, kind: str, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "details": dict(self.details),
        }


def event_to_dict(event: EventEnvelope | Mapping[str, Any] | Any) -> dict[str, Any]:
    """Return an event envelope as a plain dict."""

    if isinstance(event, EventEnvelope):
        return event.to_dict()
    if isinstance(event, Mapping):
        return dict(event)

    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
        raise TypeError("event.to_dict() must return a mapping")

    raise TypeError("event must be a mapping, EventEnvelope, or provide to_dict()")


def dumps_event(event: EventEnvelope | Mapping[str, Any] | Any, *, canonical: bool = False) -> str:
    """Serialize an event envelope to JSON."""

    if canonical:
        return canonical_event_json(event)
    return json.dumps(event_to_dict(event), ensure_ascii=False, separators=(",", ":"))


def loads_event(text_or_bytes: str | bytes | bytearray | memoryview, *, validate: bool = True) -> EventEnvelope:
    """Decode event JSON into an EventEnvelope."""

    text = _decode_json_text(text_or_bytes)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EventDecodeError(
            "invalid_json",
            "event JSON could not be parsed",
            {"line": exc.lineno, "column": exc.colno, "position": exc.pos},
        ) from exc

    if not isinstance(payload, Mapping):
        raise EventDecodeError(
            "invalid_json_object",
            "event JSON must be an object",
            {"actualType": type(payload).__name__},
        )

    event_payload = dict(payload)
    if validate:
        errors = [_issue_to_error(issue) for issue in validate_event_strict(event_payload, known_event_required=True)]
        if errors:
            raise EventDecodeError(
                "invalid_event",
                "event failed validation",
                {"errors": errors},
            )

    try:
        return EventEnvelope.from_dict(event_payload)
    except (TypeError, ValueError) as exc:
        raise EventDecodeError(
            "invalid_event",
            "event could not be decoded",
            {"error": str(exc)},
        ) from exc


def canonical_event_json(event: EventEnvelope | Mapping[str, Any] | Any) -> str:
    """Serialize an event envelope using stable canonical JSON settings."""

    return json.dumps(
        event_to_dict(event),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _decode_json_text(text_or_bytes: str | bytes | bytearray | memoryview) -> str:
    if isinstance(text_or_bytes, str):
        return text_or_bytes
    if isinstance(text_or_bytes, memoryview):
        text_or_bytes = text_or_bytes.tobytes()
    if isinstance(text_or_bytes, bytearray):
        text_or_bytes = bytes(text_or_bytes)
    if isinstance(text_or_bytes, bytes):
        try:
            return text_or_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EventDecodeError(
                "invalid_encoding",
                "event JSON bytes must be valid UTF-8",
                {"encoding": "utf-8", "position": exc.start},
            ) from exc
    raise TypeError("loads_event() expects str, bytes, bytearray, or memoryview")


def _issue_to_error(issue: ValidationIssue) -> str:
    if issue.code in {"required", "invalid_spec_version", "invalid_content", "missing_idempotency_key"}:
        return issue.message
    return f"{issue.code} at {issue.path}: {issue.message}"


__all__ = [
    "EventDecodeError",
    "canonical_event_json",
    "dumps_event",
    "event_to_dict",
    "loads_event",
]
