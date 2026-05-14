"""Event projection helpers for eihead runtime."""

from __future__ import annotations

from typing import Any, Mapping


def event_outcome_common(route: Mapping[str, Any], *, trace_id: str | None) -> dict[str, Any]:
    """Build stable event outcome metadata consumed by all head runtime handlers."""

    return {
        "runtime": "eihead",
        "node_role": "head",
        "trace_id": trace_id or "",
        "event_name": _string_or_default(route.get("eventName"), ""),
        "event_type": _string_or_default(route.get("eventType"), ""),
    }


def _string_or_default(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


__all__ = ["event_outcome_common"]
