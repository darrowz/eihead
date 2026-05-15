from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Iterator

import pytest

from eihead.runtime.event_journal import EventJournal


class FixedClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


class BrokenMapping(Mapping[str, Any]):
    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("cannot iterate event")

    def __len__(self) -> int:
        raise RuntimeError("cannot measure event")

    def __getitem__(self, key: str) -> Any:
        raise RuntimeError(f"cannot read {key}")


def test_append_extracts_event_outcome_fields_and_returns_json_safe_record() -> None:
    journal = EventJournal(max_items=3, clock=FixedClock())

    record = journal.append(
        {
            "id": "evt-1",
            "type": "wakeword.detected",
            "name": "ei.wakeword.detected",
            "traceId": "trace-from-event",
        },
        outcome={
            "status": "accepted",
            "accepted": True,
            "processed": True,
            "reason": "matched_hotword",
            "trace_id": "trace-from-outcome",
            "details": {"confidence": 0.91},
        },
    )

    assert record == {
        "event_id": "evt-1",
        "event_name": "ei.wakeword.detected",
        "event_type": "wakeword.detected",
        "status": "accepted",
        "processed": True,
        "accepted": True,
        "reason": "matched_hotword",
        "trace_id": "trace-from-outcome",
        "captured_at_ts": 1001.0,
        "metadata": {"details": {"confidence": 0.91}},
    }
    assert json.loads(json.dumps(record)) == record
    assert journal.recent() == [record]


def test_journal_bounds_records_and_recent_limit_keeps_chronological_order() -> None:
    journal = EventJournal(max_items=2, clock=FixedClock())

    journal.append({"id": "evt-1", "type": "one"}, status="accepted", accepted=True)
    second = journal.append({"id": "evt-2", "type": "two"}, status="skipped", accepted=False)
    third = journal.append({"id": "evt-3", "type": "three"}, status="error", processed=False)

    assert journal.recent() == [second, third]
    assert journal.recent(1) == [third]


def test_summary_reports_retained_status_and_boolean_counts() -> None:
    journal = EventJournal(max_items=5, clock=FixedClock())

    journal.append({"id": "evt-1"}, status="accepted", accepted=True, processed=True)
    journal.append({"id": "evt-2"}, status="accepted", accepted=True, processed=False)
    journal.append({"id": "evt-3"}, status="skipped", accepted=False)

    assert journal.summary() == {
        "count": 3,
        "max_items": 5,
        "status_counts": {"accepted": 2, "skipped": 1},
        "accepted_count": 2,
        "processed_count": 1,
        "latest_captured_at_ts": 1003.0,
    }


def test_malformed_event_mapping_and_non_json_values_do_not_break_serialization() -> None:
    journal = EventJournal(max_items=1, clock=FixedClock())

    record = journal.append(
        BrokenMapping(),
        outcome={
            "status": object(),
            "accepted": "yes",
            "processed": None,
            "reason": ValueError("bad event"),
            "extra": {"bad": object()},
        },
        metadata={"source": object()},
    )

    assert record["event_id"] == ""
    assert record["event_name"] == ""
    assert record["event_type"] == ""
    assert record["status"].startswith("<object")
    assert record["accepted"] is True
    assert record["trace_id"] == ""
    assert "bad event" in record["reason"]
    assert json.loads(json.dumps(journal.recent())) == journal.recent()


def test_false_like_string_booleans_are_recorded_as_false() -> None:
    journal = EventJournal(max_items=1, clock=FixedClock())

    record = journal.append({"id": "evt-bool"}, accepted="false", processed="0")

    assert record["accepted"] is False
    assert record["processed"] is False


def test_reason_can_be_extracted_from_outcome_details() -> None:
    journal = EventJournal(max_items=1, clock=FixedClock())

    record = journal.append({"id": "evt-detail"}, outcome={"status": "skipped", "details": {"reason": "muted"}})

    assert record["reason"] == "muted"
    assert record["metadata"] == {"details": {"reason": "muted"}}


def test_rejects_non_positive_max_items() -> None:
    with pytest.raises(ValueError, match="max_items must be positive"):
        EventJournal(max_items=0)
