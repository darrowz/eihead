from __future__ import annotations

from eihead.neck.reframe import ReframeConfig, ReframeState, VisualTarget, plan_reframe_action


def test_reframe_holds_when_known_target_is_clear() -> None:
    state = ReframeState(current_pan_deg=90.0, last_commanded_pan_deg=90.0)
    target = VisualTarget(label="face", target_x=0.42, known=True, confidence=0.8, crop_width=140, crop_height=180)

    action = plan_reframe_action(target, state=state, now_ts=10.0)

    assert action.mode == "hold"
    assert action.will_move is False
    assert action.reason == "target_clear"
    assert action.pan_deg == 90.0


def test_reframe_moves_once_for_unclear_edge_target_then_observes() -> None:
    state = ReframeState(current_pan_deg=90.0, last_commanded_pan_deg=90.0)
    config = ReframeConfig(confirm_frames=2, reframe_step_deg=5.0, min_command_interval_s=0.0)
    target = VisualTarget(label="face", target_x=0.9, known=False, confidence=0.8, crop_width=50, crop_height=170)

    first = plan_reframe_action(target, state=state, config=config, now_ts=10.0)
    second = plan_reframe_action(target, state=state, config=config, now_ts=10.5)
    third = plan_reframe_action(target, state=state, config=config, now_ts=11.0)

    assert first.mode == "hold"
    assert first.reason == "needs_confirmation"
    assert second.mode == "reframe"
    assert second.will_move is True
    assert second.pan_deg == 95.0
    assert third.mode == "observe"
    assert third.will_move is False


def test_reframe_returns_home_after_observe_hold() -> None:
    state = ReframeState(
        current_pan_deg=96.0,
        last_commanded_pan_deg=96.0,
        phase="observe",
        phase_started_at_ts=10.0,
        last_command_at_ts=9.0,
    )
    config = ReframeConfig(home_pan_deg=90.0, observe_hold_s=1.5, return_step_deg=4.0, min_command_interval_s=0.0)

    observe = plan_reframe_action(None, state=state, config=config, now_ts=11.0)
    returning = plan_reframe_action(None, state=state, config=config, now_ts=12.0)

    assert observe.mode == "observe"
    assert observe.will_move is False
    assert returning.mode == "return_home"
    assert returning.will_move is True
    assert returning.pan_deg == 92.0


def test_reframe_enters_cooldown_when_home_reached() -> None:
    state = ReframeState(
        current_pan_deg=91.0,
        last_commanded_pan_deg=91.0,
        phase="return_home",
        phase_started_at_ts=10.0,
        last_command_at_ts=9.0,
    )
    config = ReframeConfig(home_pan_deg=90.0, return_step_deg=4.0, cooldown_s=3.0, min_command_interval_s=0.0)

    action = plan_reframe_action(None, state=state, config=config, now_ts=12.0)
    cooldown = plan_reframe_action(
        VisualTarget(label="face", target_x=0.95, known=False, confidence=0.9, crop_width=50, crop_height=160),
        state=state,
        config=config,
        now_ts=13.0,
    )

    assert action.mode == "return_home"
    assert action.pan_deg == 90.0
    assert state.phase == "cooldown"
    assert cooldown.mode == "hold"
    assert cooldown.reason == "cooldown"


def test_reframe_suppresses_direction_flip_oscillation() -> None:
    state = ReframeState(
        current_pan_deg=95.0,
        last_commanded_pan_deg=95.0,
        last_direction=1,
        phase="idle",
        last_command_at_ts=5.0,
    )
    config = ReframeConfig(confirm_frames=1, anti_oscillation_cooldown_s=3.0, min_command_interval_s=0.0)
    target = VisualTarget(label="face", target_x=0.05, known=False, confidence=0.9, crop_width=60, crop_height=170)

    action = plan_reframe_action(target, state=state, config=config, now_ts=6.0)

    assert action.mode == "hold"
    assert action.will_move is False
    assert action.reason == "direction_flip_cooldown"
