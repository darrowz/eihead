from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eihead.protocol import ActionExecuted, MoveHeadAction
from eihead.runtime.app import HeadRuntimeApp


FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "eiprotocol" / "tests" / "fixtures" / "eiprotocol").resolve()


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


class _RecordingBodyRuntime:
    def __init__(self) -> None:
        self.dispatched: list[object] = []

    def snapshot(self) -> dict[str, object]:
        return {"node_id": "honjia-test"}

    def dispatch_actions(self, actions: list[object]) -> list[object]:
        self.dispatched.extend(actions)
        action = actions[0]
        if isinstance(action, MoveHeadAction):
            return [
                ActionExecuted(
                    ts=action.ts,
                    source="neck.motor",
                    status="ok",
                    action_kind=action.kind,
                    details={"target_angle": action.target_angle},
                )
            ]
        return []


def test_handle_event_invalid_envelope_is_not_processed_and_journaled() -> None:
    runtime = HeadRuntimeApp(body_runtime=_RecordingBodyRuntime())
    event = _fixture("asr_final.json")
    del event["source"]

    outcome = runtime.handle_event(event, trace_id="trace-invalid")

    assert outcome["ok"] is False
    assert outcome["accepted"] is False
    assert outcome["processed"] is False
    assert outcome["status"] == "not_processed"
    assert outcome["reason"] == "invalid_event"
    assert "source is required" in outcome["errors"]
    assert json.loads(json.dumps(outcome)) == outcome
    assert runtime.recent_events()[-1]["reason"] == "invalid_event"
    assert runtime.recent_events()[-1]["trace_id"] == "trace-invalid"


def test_handle_event_unknown_valid_route_is_not_processed_and_journaled() -> None:
    body_runtime = _RecordingBodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)
    event = _fixture("asr_final.json")
    event["name"] = "ei.dialogue.agent.hidden"
    event["content"] = {"delta": "hello"}

    outcome = runtime.handle_event(event)

    assert outcome["ok"] is False
    assert outcome["accepted"] is False
    assert outcome["processed"] is False
    assert outcome["status"] == "not_processed"
    assert outcome["reason"] == "unsupported_event_name"
    assert outcome["event_name"] == "ei.dialogue.agent.hidden"
    assert body_runtime.dispatched == []
    assert runtime.recent_events()[-1]["reason"] == "unsupported_event_name"


def test_handle_event_action_request_maps_eiprotocol_content_to_handle_action() -> None:
    body_runtime = _RecordingBodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)

    outcome = runtime.handle_event(_fixture("head_action_request.json"), trace_id="trace-override")

    assert outcome["ok"] is True
    assert outcome["accepted"] is True
    assert outcome["processed"] is True
    assert outcome["status"] == "accepted"
    assert outcome["route"] == "action_request"
    assert outcome["trace_id"] == "trace-override"
    assert outcome["action_outcome"]["action_id"] == "act_move_head_001"
    assert outcome["action_outcome"]["action_type"] == "move_head"
    assert outcome["action_outcome"]["trace_id"] == "trace-override"
    assert isinstance(body_runtime.dispatched[0], MoveHeadAction)
    assert body_runtime.dispatched[0].target_angle == 92
    assert body_runtime.dispatched[0].target_name == "neck.pan"
    assert runtime.recent_events()[-1]["metadata"]["action_outcome"]["status"] == "accepted"


def test_handle_event_records_diagnostics_routes_without_downstream_processing() -> None:
    body_runtime = _RecordingBodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)
    fixture_names = [
        "capability_manifest.json",
        "asr_partial.json",
        "asr_final.json",
        "realtime_vision_frame.json",
        "execution_outcome.json",
        "user_feedback.json",
    ]

    outcomes = [runtime.handle_event(_fixture(name)) for name in fixture_names]

    assert body_runtime.dispatched == []
    for outcome in outcomes:
        assert outcome["ok"] is True
        assert outcome["accepted"] is True
        assert outcome["processed"] is False
        assert outcome["status"] == "recorded"
        assert outcome["reason"] == "recorded_for_diagnostics"
    assert [record["status"] for record in runtime.recent_events()] == ["recorded"] * len(fixture_names)


def test_handle_event_records_new_known_protocol_routes_as_diagnostics() -> None:
    body_runtime = _RecordingBodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)
    event = _fixture("asr_final.json")
    event["name"] = "ei.dialogue.agent.delta"
    event["content"] = {"delta": "继续", "index": 1}

    outcome = runtime.handle_event(event)

    assert outcome["ok"] is True
    assert outcome["accepted"] is True
    assert outcome["processed"] is False
    assert outcome["status"] == "recorded"
    assert outcome["reason"] == "recorded_for_diagnostics"
    assert outcome["route"] == "agent_delta"
    assert body_runtime.dispatched == []


def test_handle_event_records_side_effecting_non_request_routes_without_dispatch() -> None:
    body_runtime = _RecordingBodyRuntime()
    runtime = HeadRuntimeApp(body_runtime=body_runtime)

    for fixture_name, route_name in [
        ("action_dispatch.json", "action_dispatch"),
        ("emergency_stop.json", "action_emergency_stop"),
        ("memory_write_committed.json", "memory_write_committed"),
    ]:
        outcome = runtime.handle_event(_fixture(fixture_name))

        assert outcome["ok"] is True
        assert outcome["accepted"] is True
        assert outcome["processed"] is False
        assert outcome["status"] == "recorded"
        assert outcome["reason"] == "recorded_for_diagnostics"
        assert outcome["route"] == route_name

    assert body_runtime.dispatched == []


def test_event_journal_accessors_are_json_friendly_and_limitable() -> None:
    runtime = HeadRuntimeApp(body_runtime=_RecordingBodyRuntime())

    runtime.handle_event(_fixture("asr_partial.json"))
    runtime.handle_event(_fixture("asr_final.json"))

    recent = runtime.recent_events(limit=1)
    summary = runtime.event_summary()

    assert len(recent) == 1
    assert recent[0]["event_name"] == "ei.dialogue.asr.final"
    assert summary["count"] == 2
    assert summary["status_counts"] == {"recorded": 2}
    assert json.loads(json.dumps({"recent": recent, "summary": summary})) == {
        "recent": recent,
        "summary": summary,
    }
