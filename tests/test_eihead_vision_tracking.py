from __future__ import annotations

import pytest


def test_select_tracking_target_prefers_trackable_labels_and_reports_error() -> None:
    from eihead.eye.tracking import select_tracking_target

    detections = [
        {"label": "dog", "score": 0.99, "bbox": [10, 20, 620, 460], "track_id": "dog-1"},
        {"label": "person", "score": 0.72, "bbox": [260, 100, 420, 340], "track_id": "person-1"},
        {"label": "face", "score": 0.71, "bbox": [284, 132, 364, 244], "track_id": "face-1"},
    ]

    target = select_tracking_target(detections, frame_width=640, frame_height=480, frame_id="frame-7")

    assert target is not None
    assert target.label == "person"
    assert target.track_id == "person-1"
    assert target.frame_id == "frame-7"
    assert target.bbox == (260.0, 100.0, 420.0, 340.0)
    assert target.center_x == pytest.approx(340.0)
    assert target.center_y == pytest.approx(220.0)
    assert target.horizontal_error == pytest.approx((340.0 - 320.0) / 320.0)
    assert target.score == pytest.approx(0.72)


def test_select_tracking_target_accepts_realtime_bbox_variants() -> None:
    from eihead.eye.tracking import select_tracking_target

    detections = [
        {
            "label": "person",
            "score": 0.90,
            "bbox": {"x_min": 0.25, "y_min": 0.25, "x_max": 0.75, "y_max": 0.75},
            "trackId": "normalized-person",
        },
        {
            "label": "face",
            "score": 0.89,
            "bbox": {"x": 300, "y": 160, "w": 80, "h": 80},
            "trackId": "xywh-face",
        },
    ]

    target = select_tracking_target(detections, frame_width=640, frame_height=480)

    assert target is not None
    assert target.track_id == "normalized-person"
    assert target.lock_id == "normalized-person"
    assert target.bbox == (160.0, 120.0, 480.0, 360.0)
    assert target.horizontal_error == pytest.approx(0.0)


def test_select_tracking_target_accepts_protocol_normalized_list_xywh_bbox() -> None:
    from eihead.eye.tracking import select_tracking_target

    target = select_tracking_target(
        [{"label": "person", "confidence": 0.91, "bbox": [0.62, 0.35, 0.12, 0.18], "trackId": "person-list"}],
        frame_width=640,
        frame_height=480,
    )

    assert target is not None
    assert target.track_id == "person-list"
    assert target.score == pytest.approx(0.91)
    assert target.bbox == pytest.approx((396.8, 168.0, 473.6, 254.4))
    assert target.center_x == pytest.approx(435.2)


def test_select_tracking_target_accepts_protocol_pixel_list_xywh_bbox() -> None:
    from eihead.eye.tracking import select_tracking_target

    detections = [
        {"label": "person", "score": 0.91, "bbox": [320, 120, 160, 120], "track_id": "person-pixel-list"}
    ]

    target = select_tracking_target(detections, frame_width=640, frame_height=480, frame_id="frame-pixel-list")

    assert target is not None
    assert target.track_id == "person-pixel-list"
    assert target.bbox == pytest.approx((320.0, 120.0, 480.0, 240.0))
    assert target.center_x == pytest.approx(400.0)


def test_select_tracking_target_honors_explicit_normalized_list_xyxy_bbox() -> None:
    from eihead.eye.tracking import select_tracking_target

    target = select_tracking_target(
        [
            {
                "label": "person",
                "score": 0.91,
                "bbox": [0.62, 0.35, 0.74, 0.53],
                "bboxFormat": "xyxy",
                "trackId": "person-normalized-xyxy",
            }
        ],
        frame_width=640,
        frame_height=480,
    )

    assert target is not None
    assert target.track_id == "person-normalized-xyxy"
    assert target.bbox == pytest.approx((396.8, 168.0, 473.6, 254.4))
    assert target.center_x == pytest.approx(435.2)


def test_select_tracking_target_uses_score_area_and_center_for_stable_ranking() -> None:
    from eihead.eye.tracking import select_tracking_target

    detections = [
        {"label": "person", "score": 0.80, "bbox": [500, 120, 620, 420], "track_id": "edge"},
        {"label": "person", "score": 0.82, "bbox": [240, 130, 400, 430], "track_id": "center"},
        {"label": "face", "score": 0.60, "bbox": [300, 180, 340, 220], "track_id": "tiny-face"},
    ]

    target = select_tracking_target(detections, frame_width=640, frame_height=480)

    assert target is not None
    assert target.track_id == "center"
    assert target.horizontal_error == pytest.approx(0.0)


def test_select_tracking_target_returns_none_for_empty_or_invalid_detections() -> None:
    from eihead.eye.tracking import select_tracking_target

    assert select_tracking_target([], frame_width=640, frame_height=480) is None
    assert select_tracking_target(
        [{"label": "person", "score": 0.9, "bbox": [10, 10, 0, 40]}],
        frame_width=640,
        frame_height=480,
    ) is None


def test_long_term_tracker_keeps_active_id_through_short_score_spikes() -> None:
    from eihead.eye.tracking import LongTermVisualTracker

    tracker = LongTermVisualTracker(switch_score_margin=0.15, switch_hold_frames=2)

    first = tracker.update(
        [{"label": "person", "score": 0.80, "bbox": [100, 80, 220, 360], "track_id": "stable"}],
        frame_width=640,
        frame_height=480,
        frame_id="f1",
    )
    spike = tracker.update(
        [
            {"label": "person", "score": 0.81, "bbox": [100, 80, 220, 360], "track_id": "stable"},
            {"label": "person", "score": 0.93, "bbox": [420, 80, 540, 360], "track_id": "spike"},
        ],
        frame_width=640,
        frame_height=480,
        frame_id="f2",
    )

    assert first is not None
    assert spike is not None
    assert first.track_id == "stable"
    assert spike.track_id == "stable"
    assert spike.age == 2
    assert spike.frame_count == 2
    assert spike.last_seen == "f2"
    assert spike.miss_count == 0
    assert spike.lost is False
    assert spike.reacquired is False
    assert tracker.diagnostics() == {
        "track_count": 2,
        "active_track_id": "stable",
        "switch_count": 0,
        "reacquired_count": 0,
        "lost_count": 0,
        "stability_ratio": pytest.approx(1.0),
        "suppressed_reason": "switch_margin",
    }


def test_long_term_tracker_switches_after_margin_and_hold_frames() -> None:
    from eihead.eye.tracking import LongTermVisualTracker

    tracker = LongTermVisualTracker(switch_score_margin=0.10, switch_hold_frames=2)

    tracker.update(
        [{"label": "person", "score": 0.70, "bbox": [100, 80, 220, 360], "track_id": "a"}],
        frame_width=640,
        frame_height=480,
        frame_id="f1",
    )
    held = tracker.update(
        [
            {"label": "person", "score": 0.70, "bbox": [100, 80, 220, 360], "track_id": "a"},
            {"label": "person", "score": 0.86, "bbox": [420, 80, 540, 360], "track_id": "b"},
        ],
        frame_width=640,
        frame_height=480,
        frame_id="f2",
    )
    switched = tracker.update(
        [
            {"label": "person", "score": 0.70, "bbox": [100, 80, 220, 360], "track_id": "a"},
            {"label": "person", "score": 0.87, "bbox": [420, 80, 540, 360], "track_id": "b"},
        ],
        frame_width=640,
        frame_height=480,
        frame_id="f3",
    )

    assert held is not None
    assert switched is not None
    assert held.track_id == "a"
    assert switched.track_id == "b"
    assert switched.frame_count == 2
    assert tracker.diagnostics()["switch_count"] == 1
    assert tracker.diagnostics()["suppressed_reason"] is None


def test_long_term_tracker_marks_lost_and_reacquired_tracks() -> None:
    from eihead.eye.tracking import LongTermVisualTracker

    tracker = LongTermVisualTracker(max_misses=2)

    tracker.update(
        [{"label": "face", "score": 0.90, "bbox": [250, 120, 390, 300], "track_id": "face-1"}],
        frame_width=640,
        frame_height=480,
        frame_id="f1",
    )
    missing = tracker.update([], frame_width=640, frame_height=480, frame_id="f2")
    reacquired = tracker.update(
        [{"label": "face", "score": 0.91, "bbox": [252, 121, 392, 301], "track_id": "face-1"}],
        frame_width=640,
        frame_height=480,
        frame_id="f3",
    )

    assert missing is None
    assert reacquired is not None
    assert reacquired.track_id == "face-1"
    assert reacquired.age == 2
    assert reacquired.frame_count == 2
    assert reacquired.last_seen == "f3"
    assert reacquired.miss_count == 0
    assert reacquired.lost is False
    assert reacquired.reacquired is True
    assert tracker.diagnostics()["lost_count"] == 1
    assert tracker.diagnostics()["reacquired_count"] == 1
    assert tracker.diagnostics()["active_track_id"] == "face-1"


def test_long_term_tracker_keeps_synthetic_id_when_detector_omits_track_id() -> None:
    from eihead.eye.tracking import LongTermVisualTracker

    tracker = LongTermVisualTracker(max_misses=2)

    first = tracker.update(
        [{"label": "person", "score": 0.90, "bbox": [250, 120, 140, 180]}],
        frame_width=640,
        frame_height=480,
        frame_id="f1",
    )
    second = tracker.update(
        [{"label": "person", "score": 0.91, "bbox": [258, 122, 140, 180]}],
        frame_width=640,
        frame_height=480,
        frame_id="f2",
    )
    third = tracker.update(
        [{"label": "person", "score": 0.92, "bbox": [266, 124, 140, 180]}],
        frame_width=640,
        frame_height=480,
        frame_id="f3",
    )

    assert first is not None
    assert second is not None
    assert third is not None
    assert second.track_id == first.track_id
    assert third.track_id == first.track_id
    assert third.frame_count == 3
    assert tracker.diagnostics()["lost_count"] == 0


def test_plan_pan_follow_action_outputs_smoothed_pan_only_command() -> None:
    from eihead.eye.tracking import TrackingTarget
    from eihead.neck.vision_follow import VisionFollowState, plan_pan_follow_action

    state = VisionFollowState(current_pan_deg=90.0, last_commanded_pan_deg=90.0)
    target = TrackingTarget(
        bbox=(400.0, 100.0, 560.0, 340.0),
        center_x=480.0,
        center_y=220.0,
        horizontal_error=0.5,
        score=0.92,
        label="person",
        track_id="person-1",
        frame_id="frame-8",
    )

    action = plan_pan_follow_action(target, state=state)

    assert action.mode == "hold"
    assert action.pan_delta_deg == pytest.approx(0.0)
    assert action.pan_deg == pytest.approx(90.0)
    assert action.target_angle == pytest.approx(90.0)
    assert action.delta == pytest.approx(0.0)
    assert action.tilt_deg is None
    assert action.reason == "bias_not_confirmed"
    assert action.suppressed_reason == "bias_not_confirmed"
    assert action.deadband_applied is False
    assert action.lock_id == "person-1"
    assert state.smoothed_error == pytest.approx(0.25)
    assert state.last_commanded_pan_deg == pytest.approx(90.0)
    assert state.current_pan_deg == pytest.approx(90.0)
    assert state.lost_frames == 0


def test_plan_pan_follow_action_suppresses_deadband_and_tiny_angle_changes() -> None:
    from eihead.eye.tracking import TrackingTarget
    from eihead.neck.vision_follow import VisionFollowConfig, VisionFollowState, plan_pan_follow_action

    state = VisionFollowState(current_pan_deg=90.0, last_commanded_pan_deg=90.0)
    tiny_target = TrackingTarget(
        bbox=(326.4, 100.0, 390.4, 240.0),
        center_x=358.4,
        center_y=170.0,
        horizontal_error=0.12,
        score=0.8,
        label="face",
        track_id=None,
        frame_id=None,
    )

    deadband = plan_pan_follow_action(tiny_target, state=state)
    assert deadband.mode == "hold"
    assert deadband.pan_delta_deg == 0.0
    assert deadband.reason == "deadband"
    assert deadband.deadband_applied is True
    assert deadband.suppressed_reason == "inside_deadband"
    assert deadband.target_angle == pytest.approx(90.0)
    assert deadband.delta == pytest.approx(0.0)

    small_step = plan_pan_follow_action(
        tiny_target,
        state=state,
        config=VisionFollowConfig(deadband=0.02, min_angle_delta_deg=2.0, hold_frames=1),
    )
    assert small_step.mode == "hold"
    assert small_step.pan_delta_deg == 0.0
    assert small_step.reason == "min_angle_delta"
    assert small_step.deadband_applied is False
    assert small_step.suppressed_reason == "min_angle_delta"


def test_plan_pan_follow_action_tracks_after_bias_is_confirmed() -> None:
    from eihead.eye.tracking import TrackingTarget
    from eihead.neck.vision_follow import VisionFollowConfig, VisionFollowState, plan_pan_follow_action

    config = VisionFollowConfig(min_action_interval_s=0.1)
    state = VisionFollowState(current_pan_deg=90.0, last_commanded_pan_deg=90.0)
    target = TrackingTarget(
        bbox=(400.0, 100.0, 560.0, 340.0),
        center_x=480.0,
        center_y=220.0,
        horizontal_error=0.5,
        score=0.92,
        label="person",
        track_id="person-1",
        lock_id="lock-person-1",
        frame_id="frame-8",
    )

    first = plan_pan_follow_action(target, state=state, config=config, now_ts=1.0)
    second = plan_pan_follow_action(target, state=state, config=config, now_ts=1.2)

    assert first.mode == "hold"
    assert second.mode == "track"
    assert second.pan_delta_deg == pytest.approx(7.5)
    assert second.pan_deg == pytest.approx(97.5)
    assert second.target_angle == pytest.approx(97.5)
    assert second.delta == pytest.approx(7.5)
    assert second.reason == "tracking"
    assert second.suppressed_reason is None
    assert second.deadband_applied is False
    assert second.lock_id == "lock-person-1"


def test_plan_pan_follow_action_enforces_minimum_action_interval() -> None:
    from eihead.eye.tracking import TrackingTarget
    from eihead.neck.vision_follow import VisionFollowConfig, VisionFollowState, plan_pan_follow_action

    config = VisionFollowConfig(min_action_interval_s=0.3, hold_frames=1)
    state = VisionFollowState(current_pan_deg=90.0, last_commanded_pan_deg=90.0)
    target = TrackingTarget(
        bbox=(400.0, 100.0, 560.0, 340.0),
        center_x=480.0,
        center_y=220.0,
        horizontal_error=0.4,
        score=0.92,
        label="person",
        track_id="person-1",
        lock_id="lock-person-1",
        frame_id="frame-9",
    )

    first = plan_pan_follow_action(target, state=state, config=config, now_ts=1.0)
    second = plan_pan_follow_action(target, state=state, config=config, now_ts=1.1)

    assert first.mode == "track"
    assert first.pan_deg == pytest.approx(94.0)
    assert second.mode == "hold"
    assert second.reason == "min_interval"
    assert second.pan_deg == pytest.approx(94.0)
    assert second.pan_delta_deg == pytest.approx(0.0)
    assert second.target_angle == pytest.approx(94.0)
    assert second.delta == pytest.approx(0.0)
    assert second.deadband_applied is False
    assert second.suppressed_reason == "min_interval"
    assert second.lock_id == "lock-person-1"


def test_plan_pan_follow_action_holds_then_decays_when_target_is_lost() -> None:
    from eihead.neck.vision_follow import VisionFollowState, plan_pan_follow_action

    state = VisionFollowState(current_pan_deg=100.0, last_commanded_pan_deg=100.0, smoothed_error=0.4)

    first_lost = plan_pan_follow_action(None, state=state)
    second_lost = plan_pan_follow_action(None, state=state)
    third_lost = plan_pan_follow_action(None, state=state)
    fourth_lost = plan_pan_follow_action(None, state=state)

    assert first_lost.mode == "hold"
    assert first_lost.reason == "target_lost_hold"
    assert first_lost.pan_deg == pytest.approx(100.0)
    assert second_lost.mode == "hold"
    assert third_lost.mode == "decay"
    assert third_lost.reason == "target_lost_decay"
    assert third_lost.pan_delta_deg == pytest.approx(-2.0)
    assert third_lost.pan_deg == pytest.approx(98.0)
    assert fourth_lost.mode == "decay"
    assert fourth_lost.pan_deg == pytest.approx(96.0)
    assert state.smoothed_error == pytest.approx(0.0)
