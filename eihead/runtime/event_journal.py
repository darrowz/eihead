"""Pure local journal for recent eihead eiprotocol event handling records."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from threading import RLock
import time
from typing import Any


JsonObject = dict[str, Any]
Clock = Callable[[], float]
_MISSING = object()
_SUMMARY_KEYS = frozenset({"status", "accepted", "processed", "reason", "trace_id", "traceId"})
_TRUE_TEXT = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_TEXT = frozenset({"0", "false", "no", "n", "off", ""})


class EventJournal:
    """Bounded in-memory journal for monitor/debug event visibility."""

    def __init__(self, max_items: int = 100, *, clock: Clock | None = None) -> None:
        resolved_max_items = int(max_items)
        if resolved_max_items <= 0:
            raise ValueError("max_items must be positive")
        self.max_items = resolved_max_items
        self._clock = clock or time.time
        self._records: deque[JsonObject] = deque(maxlen=self.max_items)
        self._lock = RLock()

    def append(
        self,
        event: Any,
        outcome: Mapping[str, Any] | Any | None = None,
        *,
        status: Any = None,
        processed: Any = None,
        accepted: Any = None,
        reason: Any = None,
        trace_id: Any = None,
        metadata: Mapping[str, Any] | None = None,
        **extra_metadata: Any,
    ) -> JsonObject:
        """Capture a JSON-safe event handling record and return that record."""

        event_id = _string_field(event, ("id", "event_id", "eventId"))
        event_type = _string_field(event, ("type", "event_type", "eventType"))
        event_name = _string_field(event, ("name", "event_name", "eventName"))

        resolved_status = _first_available(status, _safe_get(outcome, "status"), default="unknown")
        resolved_processed = _first_available(processed, _safe_get(outcome, "processed"), default=_MISSING)
        resolved_accepted = _first_available(accepted, _safe_get(outcome, "accepted"), default=_MISSING)
        resolved_reason = _first_available(
            reason,
            _safe_get(outcome, "reason"),
            _safe_get(_safe_get(outcome, "details"), "reason"),
            default=_MISSING,
        )
        resolved_trace_id = _first_available(
            trace_id,
            _safe_get(outcome, "trace_id"),
            _safe_get(outcome, "traceId"),
            _safe_get(event, "trace_id"),
            _safe_get(event, "traceId"),
            default="",
        )

        record: JsonObject = {
            "event_id": event_id,
            "event_name": event_name,
            "event_type": event_type,
            "status": _text(resolved_status),
            "captured_at_ts": float(self._clock()),
        }
        if resolved_processed is not _MISSING:
            record["processed"] = _coerce_bool(resolved_processed)
        if resolved_accepted is not _MISSING:
            record["accepted"] = _coerce_bool(resolved_accepted)
        if resolved_reason is not _MISSING:
            record["reason"] = _text(resolved_reason)
        record["trace_id"] = _text(resolved_trace_id)

        record_metadata = _metadata_from_outcome(outcome)
        if metadata is not None:
            record_metadata.update(_json_safe_mapping(metadata))
        if extra_metadata:
            record_metadata.update(_json_safe_mapping(extra_metadata))
        if record_metadata:
            record["metadata"] = record_metadata

        safe_record = _json_safe_mapping(record)
        with self._lock:
            self._records.append(safe_record)
        return _copy_json_object(safe_record)

    def recent(self, limit: int | None = None) -> list[JsonObject]:
        """Return retained records in chronological order."""

        with self._lock:
            records = list(self._records)

        if limit is not None:
            resolved_limit = max(0, int(limit))
            records = records[-resolved_limit:] if resolved_limit else []
        return [_copy_json_object(record) for record in records]

    def summary(self) -> JsonObject:
        """Return a JSON-safe summary of retained journal records."""

        records = self.recent()
        status_counts: dict[str, int] = {}
        accepted_count = 0
        processed_count = 0
        latest_captured_at_ts: float | None = None

        for record in records:
            status = _text(record.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            if record.get("accepted") is True:
                accepted_count += 1
            if record.get("processed") is True:
                processed_count += 1
            captured_at_ts = record.get("captured_at_ts")
            if isinstance(captured_at_ts, int | float):
                latest_captured_at_ts = float(captured_at_ts)

        return {
            "count": len(records),
            "max_items": self.max_items,
            "status_counts": status_counts,
            "accepted_count": accepted_count,
            "processed_count": processed_count,
            "latest_captured_at_ts": latest_captured_at_ts,
        }


def _string_field(source: Any, names: tuple[str, ...]) -> str:
    return _text(_first_available(*(_safe_get(source, name) for name in names), default=""))


def _safe_get(source: Any, key: str) -> Any:
    if source is None:
        return _MISSING
    try:
        if isinstance(source, Mapping):
            return source.get(key, _MISSING)
        return getattr(source, key, _MISSING)
    except Exception:
        return _MISSING


def _first_available(*values: Any, default: Any = _MISSING) -> Any:
    for value in values:
        if value is not _MISSING and value is not None:
            return value
    return default


def _metadata_from_outcome(outcome: Any) -> JsonObject:
    if not isinstance(outcome, Mapping):
        return {}
    metadata: JsonObject = {}
    try:
        items = list(outcome.items())
    except Exception:
        return metadata
    for key, value in items:
        text_key = _text(key)
        if text_key not in _SUMMARY_KEYS:
            metadata[text_key] = _json_safe(value)
    return metadata


def _json_safe_mapping(payload: Mapping[str, Any]) -> JsonObject:
    safe: JsonObject = {}
    try:
        items = list(payload.items())
    except Exception:
        return safe
    for key, value in items:
        safe[_text(key)] = _json_safe(value)
    return safe


def _copy_json_object(payload: Mapping[str, Any]) -> JsonObject:
    return _json_safe_mapping(payload)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _FALSE_TEXT:
            return False
        if normalized in _TRUE_TEXT:
            return True
    return bool(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, list | tuple | set | frozenset):
        return [_json_safe(item) for item in value]
    return _text(value)


def _text(value: Any) -> str:
    if value is None or value is _MISSING:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return repr(value)


__all__ = ["EventJournal"]
