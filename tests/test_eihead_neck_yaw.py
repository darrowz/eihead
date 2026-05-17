from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from eihead.runtime.app import HeadRuntimeApp


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Keep these tests independent from the transitional eihead runtime package import.
yaw = _load_module("eihead_neck_yaw_under_test", "eihead/organs/neck/yaw.py")
neck_servo = _load_module("eihead_neck_servo_under_test", "eihead/devices/neck_servo.py")

NeckServoCommandAdapter = neck_servo.NeckServoCommandAdapter
RaspbotServoDriver = neck_servo.RaspbotServoDriver
YawControlConfig = yaw.YawControlConfig
YawControlState = yaw.YawControlState
YawController = yaw.YawController
YawIntent = yaw.YawIntent


def test_yaw_maps_normalized_target_x_to_bounded_pan_steps() -> None:
    controller = YawController(
        YawControlConfig(deadband=0.10, step_gain=20.0, max_step=8, smoothing_alpha=1.0)
    )
    state = YawControlState(last_angle=90)

    right = controller.decide(
        intent=YawIntent(source="eye.tracking", target_name="face", target_x=1.0),
        state=state,
        now_ts=10.0,
    )

    assert right.should_command is True
    assert right.angle == 98
    assert state.last_angle == 98

    left = controller.decide(
        intent=YawIntent(source="eye.tracking", target_name="face", target_x=0.0),
        state=state,
        now_ts=11.0,
    )

    assert left.should_command is True
    assert left.angle == 90


def test_yaw_suppresses_targets_inside_deadband_without_jittering() -> None:
    controller = YawController(YawControlConfig(deadband=0.16, min_command_interval_s=0.0))
    state = YawControlState(last_angle=92, last_commanded_angle=92)

    decision = controller.decide(
        intent=YawIntent(source="eye.tracking", target_name="face", target_x=0.55),
        state=state,
        now_ts=20.0,
    )

    assert decision.should_command is False
    assert decision.reason == "deadband"
    assert decision.angle == 92
    assert state.last_commanded_angle == 92
    assert state.last_angle == 92


def test_yaw_smoothing_reduces_large_step_changes() -> None:
    controller = YawController(
        YawControlConfig(
            deadband=0.05,
            step_gain=18.0,
            max_step=6,
            smoothing_alpha=0.25,
            min_command_interval_s=0.0,
        )
    )
    state = YawControlState(last_angle=90, last_target_x=0.70, last_commanded_angle=90)

    first = controller.decide(
        intent=YawIntent(source="eye.tracking", target_name="face", target_x=0.80),
        state=state,
        now_ts=30.0,
    )

    assert first.should_command is True
    assert first.angle == 91

    assert state.last_angle == 91


def test_yaw_suppresses_repeated_direct_angle_commands() -> None:
    controller = YawController(YawControlConfig(min_command_interval_s=0.0))
    state = YawControlState(last_angle=95, last_commanded_angle=95)

    repeated = controller.decide(
        intent=YawIntent(source="manual_override", target_angle=95),
        state=state,
        now_ts=31.0,
    )

    assert repeated.should_command is False
    assert repeated.reason == "same_angle"
    assert repeated.angle == 95


def test_yaw_suppresses_commands_inside_min_interval() -> None:
    controller = YawController(YawControlConfig(deadband=0.05, min_command_interval_s=0.75))
    state = YawControlState(last_angle=90, last_commanded_angle=90, last_command_at_ts=100.0)

    decision = controller.decide(
        intent=YawIntent(source="eye.tracking", target_name="face", target_x=1.0),
        state=state,
        now_ts=100.2,
    )

    assert decision.should_command is False
    assert decision.reason == "min_interval"
    assert decision.angle > 90
    assert state.last_angle == 90
    assert state.last_commanded_angle == 90


def test_servo_command_adapter_invokes_injected_driver_only_for_commanded_decisions() -> None:
    driver = RecordingServoDriver()
    adapter = NeckServoCommandAdapter(driver, servo_id=1)

    skipped = adapter.apply_decision(_Decision(False, 91, "deadband"))

    assert skipped == {"status": "suppressed", "reason": "deadband", "angle": 91}
    assert driver.calls == []

    sent = adapter.apply_decision(_Decision(True, 97, ""))

    assert sent == {"status": "ok", "servo_id": 1, "angle": 97, "payload": [1, 97]}
    assert driver.calls == [(97, 1)]


def test_servo_command_adapter_applies_native_pan_plans_without_jittering() -> None:
    driver = RecordingServoDriver()
    adapter = NeckServoCommandAdapter(driver, servo_id=1)

    sent = adapter.apply_plan(_pan_plan(status="planned", will_move=True, angle=97))
    skipped = adapter.apply_plan(_pan_plan(status="suppressed", will_move=False, angle=98, reason="deadband"))

    assert sent == {"status": "ok", "servo_id": 1, "angle": 97, "payload": [1, 97]}
    assert skipped == {"status": "suppressed", "reason": "deadband", "angle": 98}
    assert driver.calls == [(97, 1)]
    assert json.loads(json.dumps({"sent": sent, "skipped": skipped}, allow_nan=False)) == {
        "sent": sent,
        "skipped": skipped,
    }


def test_build_neck_servo_adapter_is_safely_unavailable_off_honjia() -> None:
    assert hasattr(neck_servo, "build_neck_servo_adapter")
    adapter = neck_servo.build_neck_servo_adapter(node_id="developer-laptop")

    outcome = adapter.apply_plan(_pan_plan(status="planned", will_move=True, angle=97))

    assert outcome == {
        "status": "unavailable",
        "success": False,
        "reason": "neck_servo_unavailable_off_honjia",
        "node_id": "developer-laptop",
        "angle": 97,
    }
    assert json.loads(json.dumps(outcome, allow_nan=False)) == outcome


def test_honjia_pan_servo_status_can_report_physical_verification() -> None:
    driver = RaspbotServoDriver(mock=True, servo_id=1, hardware_verified=True)
    adapter = NeckServoCommandAdapter(driver, servo_id=1)

    status = adapter.status()

    assert status["status"] == "ready"
    assert status["servo_id"] == 1
    assert status["hardware_verified"] is True
    assert status["driver"]["hardware_verified"] is True


def test_runtime_routes_move_head_through_native_pan_servo_boundary_and_reports_suppression() -> None:
    body_runtime = RecordingBodyRuntime()
    driver = RecordingServoDriver()
    runtime = HeadRuntimeApp(
        body_runtime=body_runtime,
        neck_servo_adapter=NeckServoCommandAdapter(driver, servo_id=1),
    )

    first = runtime.handle_action(
        {
            "type": "move_head",
            "axis": "pan",
            "angle": 112,
            "metadata": {"unsafe": float("inf")},
        },
        trace_id="trace-pan-1",
    )
    second = runtime.handle_action(
        {"type": "move_head", "axis": "pan", "angle": 113},
        trace_id="trace-pan-2",
    )

    assert first["status"] == "accepted"
    assert first["success"] is True
    assert first["delegated"] is True
    assert first["details"]["axis"] == "pan"
    assert first["details"]["neck_plan"]["status"] == "planned"
    assert first["details"]["neck_plan"]["action"]["metadata"] == {"unsafe": None}
    assert first["details"]["neck_servo"]["status"] == "ok"

    assert second["status"] == "skipped"
    assert second["success"] is True
    assert second["details"]["neck_plan"]["status"] == "suppressed"
    assert second["details"]["neck_plan"]["state"]["suppression_reason"] == "deadband"

    assert driver.calls == [(112, 1)]
    assert body_runtime.dispatched == []
    assert json.loads(json.dumps({"first": first, "second": second}, allow_nan=False)) == {
        "first": first,
        "second": second,
    }


def test_runtime_reports_tilt_unsupported_through_native_pan_planner_without_servo_call() -> None:
    body_runtime = RecordingBodyRuntime()
    driver = RecordingServoDriver()
    runtime = HeadRuntimeApp(
        body_runtime=body_runtime,
        neck_servo_adapter=NeckServoCommandAdapter(driver, servo_id=1),
    )

    outcome = runtime.handle_action({"type": "move_head", "axis": "tilt", "angle": 30})

    assert outcome["status"] == "unsupported"
    assert outcome["success"] is False
    assert outcome["details"]["reason"] == "tilt_not_supported"
    assert outcome["details"]["neck_plan"]["status"] == "unsupported"
    assert driver.calls == []
    assert body_runtime.dispatched == []
    assert json.loads(json.dumps(outcome, allow_nan=False)) == outcome


class RecordingServoDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None]] = []

    def ctrl_servo(self, angle: int, servo_id: int | None = None) -> list[int]:
        self.calls.append((angle, servo_id))
        return [servo_id or 0, angle]


class RecordingBodyRuntime:
    def __init__(self) -> None:
        self.dispatched: list[object] = []

    def snapshot(self) -> dict[str, object]:
        return {"node_id": "honjia-test"}

    def dispatch_actions(self, actions: list[object]) -> list[object]:
        self.dispatched.extend(actions)
        return []


class _Decision:
    def __init__(self, should_command: bool, angle: int, reason: str) -> None:
        self.should_command = should_command
        self.angle = angle
        self.reason = reason


def _pan_plan(*, status: str, will_move: bool, angle: int, reason: str = "") -> dict[str, object]:
    return {
        "status": status,
        "will_move": will_move,
        "reason": reason,
        "action": {"target_angle": angle},
    }
