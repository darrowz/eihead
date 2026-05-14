"""Helpers for posting eiprotocol events to eihead."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from eiprotocol.event_routing import classify_event


JsonObject = dict[str, Any]


class HeadEventClient(Protocol):
    """Small protocol shared by HeadClient and test fakes."""

    def post_event(self, event: Any, *, trace_id: str | None = None) -> JsonObject:
        """Post one eiprotocol event mapping."""


def event_idempotency_key(event: Any) -> str | None:
    """Return the best idempotency key available for an eiprotocol event."""

    payload = _event_to_dict(event)
    content = payload.get("content")
    candidates: list[Any] = []
    if isinstance(content, Mapping):
        candidates.extend(
            (
                content.get("idempotencyKey"),
                content.get("idempotency_key"),
            )
        )
        if _is_action_event(payload):
            candidates.extend(
                (
                    content.get("actionId"),
                    content.get("action_id"),
                )
            )

    candidates.extend(
        (
            payload.get("idempotencyKey"),
            payload.get("idempotency_key"),
        )
    )
    if _is_action_event(payload):
        candidates.extend(
            (
                payload.get("id"),
                payload.get("event_id"),
            )
        )

    return _first_text(*candidates)


def post_head_event(client: HeadEventClient, event: Any, *, trace_id: str | None = None) -> JsonObject:
    """Post a JSON-friendly eiprotocol event through a HeadClient-like object."""

    payload = _event_to_dict(event)
    return client.post_event(payload, trace_id=_resolve_trace_id(payload, trace_id))


def post_head_action_event(client: HeadEventClient, event: Any, *, trace_id: str | None = None) -> JsonObject:
    """Post an eiprotocol action event to eihead."""

    payload = _event_to_dict(event)
    route = classify_event(payload)
    if route.get("status") != "routed" or route.get("route") != "action_request":
        raise ValueError(
            "expected valid ei.action.request eiprotocol event; "
            f"got status={route.get('status')!r} route={route.get('route')!r} "
            f"reason={route.get('reason')!r} errors={route.get('errors', [])!r}"
        )
    return client.post_event(payload, trace_id=_resolve_trace_id(payload, trace_id))


def _event_to_dict(event: Any) -> JsonObject:
    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        event = to_dict()
        if not isinstance(event, Mapping):
            raise TypeError("event.to_dict() must return a mapping")
    elif not isinstance(event, Mapping):
        raise TypeError("event must be a mapping")
    return dict(event)


def _is_action_event(event: Mapping[str, Any]) -> bool:
    event_type = _first_text(event.get("type"), event.get("event_type")) or ""
    name = _first_text(event.get("name")) or ""
    return event_type == "action" or name.startswith("ei.action.")


def _resolve_trace_id(event: Mapping[str, Any], explicit_trace_id: str | None) -> str | None:
    if explicit_trace_id is not None:
        return explicit_trace_id
    return _first_text(event.get("traceId"), event.get("trace_id"))


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return None


__all__ = [
    "HeadEventClient",
    "event_idempotency_key",
    "post_head_action_event",
    "post_head_event",
]
