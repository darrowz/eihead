from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Keep these tests independent from the transitional top-level eihead import.
pan = _load_module("eihead_neck_pan_under_test", "eihead/neck/pan.py")

PanMoveCommand = pan.PanMoveCommand
PanNeckState = pan.PanNeckState
plan_pan_move = pan.plan_pan_move


def test_pan_target_angle_is_clamped_to_state_limits() -> None:
    state = PanNeckState(current_angle=90, target_angle=90, min_angle=40, max_angle=140)

    result = plan_pan_move(PanMoveCommand(target_angle=200), state)

    assert result["status"] == "planned"
    assert result["action"]["axis"] == "pan"
    assert result["action"]["target_angle"] == 140
    assert result["state"]["target_angle"] == 140
    assert result["state"]["current_angle"] == 90
    assert result["state"]["last_command_status"] == "planned"


def test_pan_deadband_suppresses_small_angle_changes_without_mutating_input() -> None:
    state = PanNeckState(current_angle=90, target_angle=90, deadband=2.0)

    result = plan_pan_move(PanMoveCommand(target_angle=91), state)

    assert result["status"] == "suppressed"
    assert result["will_move"] is False
    assert result["outcome"]["success"] is True
    assert result["state"]["last_command_status"] == "suppressed"
    assert result["state"]["suppression_reason"] == "deadband"
    assert state.last_command_status == "idle"
    assert state.suppression_reason == ""


def test_pan_target_x_maps_left_and_right_to_pan_direction() -> None:
    state = PanNeckState(current_angle=90, target_angle=90, min_angle=40, max_angle=140)

    right = plan_pan_move(PanMoveCommand(target_x=1.0), state)
    left = plan_pan_move(PanMoveCommand(target_x=0.0), state)

    assert right["status"] == "planned"
    assert right["action"]["direction"] == "right"
    assert right["action"]["target_angle"] > state.current_angle
    assert right["outcome"]["details"]["normalized_target_x"] == 1.0

    assert left["status"] == "planned"
    assert left["action"]["direction"] == "left"
    assert left["action"]["target_angle"] < state.current_angle
    assert left["outcome"]["details"]["normalized_target_x"] == 0.0


def test_tilt_request_returns_unsupported_without_fake_success() -> None:
    state = PanNeckState(current_angle=90, target_angle=90)

    result = plan_pan_move(PanMoveCommand(axis="tilt", target_angle=30), state)

    assert result["status"] == "unsupported"
    assert result["will_move"] is False
    assert result["outcome"]["success"] is False
    assert result["outcome"]["status"] == "unsupported"
    assert result["outcome"]["details"]["reason"] == "tilt_not_supported"
    assert result["state"]["last_command_status"] == "unsupported"
    assert result["state"]["suppression_reason"] == "tilt_not_supported"


def test_pan_plan_dict_is_json_serializable() -> None:
    state = PanNeckState(current_angle=90, target_angle=90)

    result = plan_pan_move(PanMoveCommand(action_id="manual-1", trace_id="trace-1", target_x=0.75), state)

    assert json.loads(json.dumps(result)) == result


def test_nonfinite_pan_targets_are_invalid_and_metadata_stays_strict_json() -> None:
    state = PanNeckState(current_angle=90, target_angle=90)

    result = plan_pan_move(
        PanMoveCommand(
            action_id="manual-nan",
            target_angle=float("nan"),
            metadata={"unsafe": float("inf"), "nested": [float("-inf")]},
        ),
        state,
    )

    assert result["status"] == "invalid"
    assert result["reason"] == "invalid_target_angle"
    assert result["action"]["metadata"] == {"unsafe": None, "nested": [None]}
    assert json.loads(json.dumps(result, allow_nan=False)) == result
