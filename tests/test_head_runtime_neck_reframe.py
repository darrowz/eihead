from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import threading
from typing import Any, Mapping
from urllib import request

from eihead.neck import PanNeckState, ReframeConfig, ReframeState
from eihead.runtime import cli
from eihead.runtime.app import HeadRuntimeApp
from eihead.runtime.http_api import create_server


class FakeVisionRuntime:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)

    def vision_realtime(self) -> dict[str, Any]:
        return dict(self.payload)


class FakeVoiceRuntime:
    def __init__(self, payload: Mapping[str, Any] | None) -> None:
        self.payload = dict(payload) if isinstance(payload, Mapping) else None

    def voice_status(self) -> dict[str, Any] | None:
        return dict(self.payload) if self.payload is not None else None


class RecordingNeckServoAdapter:
    def __init__(self, *, status: str = "ok", success: bool = True) -> None:
        self.plans: list[dict[str, Any]] = []
        self.status_value = status
        self.success = success

    def apply_plan(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(plan)
        self.plans.append(payload)
        action = payload.get("action") if isinstance(payload.get("action"), Mapping) else {}
        return {
            "status": self.status_value,
            "success": self.success,
            "target_angle": action.get("target_angle"),
        }

    def status(self) -> dict[str, Any]:
        return {"status": "ready", "available": True}


class RecordingLoopApp:
    def __init__(self) -> None:
        self.started = 0

    def start_neck_reframe_loop(self) -> dict[str, Any]:
        self.started += 1
        return {"status": "running", "started": True}


def test_tick_neck_reframe_drives_pan_for_unclear_edge_face() -> None:
    adapter = RecordingNeckServoAdapter()
    runtime = HeadRuntimeApp(
        body_runtime=FakeVisionRuntime(
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "frame_id": "frame-edge",
                "frame_width": 640,
                "frame_height": 480,
                "detections": [
                    {
                        "label": "face",
                        "score": 0.88,
                        "bbox": [560, 140, 620, 280],
                        "track_id": "face-edge",
                    }
                ],
                "evidence": {"face_crops": [{"width": 60, "height": 140}]},
            }
        ),
        neck_servo_adapter=adapter,
        neck_reframe_config=ReframeConfig(confirm_frames=1, min_command_interval_s=0.0),
    )

    result = runtime.tick_neck_reframe(now_ts=10.0, live=True)

    assert result["status"] == "accepted"
    assert result["target"]["target_x"] == 0.921875
    assert result["action"]["mode"] == "reframe"
    assert result["action"]["will_move"] is True
    assert result["dispatch"]["status"] == "accepted"
    assert len(adapter.plans) == 1
    assert adapter.plans[0]["action"]["target_angle"] == 95
    assert runtime.neck_status()["neck_reframe"]["action"]["mode"] == "reframe"


def test_tick_neck_reframe_holds_for_clear_known_face_without_servo_call() -> None:
    adapter = RecordingNeckServoAdapter()
    runtime = HeadRuntimeApp(
        body_runtime=FakeVisionRuntime(
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "frame_id": "frame-clear",
                "detections": [
                    {
                        "label": "face",
                        "confidence": 0.91,
                        "center": {"x": 0.52, "y": 0.44},
                        "bbox": {"x_min": 0.42, "y_min": 0.2, "x_max": 0.62, "y_max": 0.7},
                    }
                ],
                "identity_observations": [{"known": True, "display_name": "Darrow", "confidence": 0.97}],
                "evidence": {"face_crops": [{"width": 128, "height": 180}]},
            }
        ),
        neck_servo_adapter=adapter,
        neck_reframe_config=ReframeConfig(confirm_frames=1, min_command_interval_s=0.0),
    )

    result = runtime.tick_neck_reframe(now_ts=20.0, live=True)

    assert result["status"] == "hold"
    assert result["reason"] == "target_clear"
    assert result["target"]["known"] is True
    assert result["action"]["will_move"] is False
    assert adapter.plans == []
    assert runtime.neck_status()["neck_reframe"]["reason"] == "target_clear"


def test_tick_neck_reframe_uses_identity_observation_bbox_when_detections_are_empty() -> None:
    adapter = RecordingNeckServoAdapter()
    runtime = HeadRuntimeApp(
        body_runtime=FakeVisionRuntime(
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "frame_id": "frame-identity-edge",
                "detections": [],
                "evidence": {
                    "frame": {"width": 640, "height": 640},
                    "face_crops": [{"width": 90, "height": 170}],
                },
                "identity_observations": [
                    {
                        "known": False,
                        "confidence": 0.63,
                        "evidence": {
                            "bbox": {"x_min": 0.82, "y_min": 0.24, "x_max": 0.98, "y_max": 0.66},
                            "track_id": "identity-edge",
                        },
                    }
                ],
            }
        ),
        neck_servo_adapter=adapter,
        neck_reframe_config=ReframeConfig(confirm_frames=1, min_command_interval_s=0.0),
    )

    result = runtime.tick_neck_reframe(now_ts=30.0, live=True)

    assert result["status"] == "accepted"
    assert round(result["target"]["target_x"], 4) == 0.9
    assert result["target"]["reason"] == "identity_observation"
    assert result["action"]["mode"] == "reframe"
    assert len(adapter.plans) == 1
    assert adapter.plans[0]["action"]["target_angle"] == 95


def test_tick_neck_reframe_does_not_move_while_voice_is_sleeping() -> None:
    adapter = RecordingNeckServoAdapter()
    runtime = HeadRuntimeApp(
        body_runtime=FakeVisionRuntime(
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "frame_id": "frame-sleeping-edge",
                "frame_width": 640,
                "frame_height": 480,
                "detections": [
                    {
                        "label": "face",
                        "score": 0.88,
                        "bbox": [560, 140, 620, 280],
                    }
                ],
                "evidence": {"face_crops": [{"width": 60, "height": 140}]},
            }
        ),
        voice_runtime=FakeVoiceRuntime(
            {
                "voice_dialogue": {
                    "phase": "sleeping",
                    "conversation_active": False,
                    "last_gate_status": "waiting_for_wake_word",
                    "local_gate": {"state": "armed", "conversationActive": False},
                }
            }
        ),
        neck_servo_adapter=adapter,
        neck_reframe_config=ReframeConfig(confirm_frames=1, min_command_interval_s=0.0),
        neck_reframe_require_voice_awake=True,
    )

    result = runtime.tick_neck_reframe(now_ts=34.0, live=True)

    assert result["status"] == "hold"
    assert result["reason"] == "voice_sleeping"
    assert result["target"] is None
    assert result["action"]["mode"] == "hold"
    assert result["action"]["will_move"] is False
    assert result["voice_gate"]["allowed"] is False
    assert adapter.plans == []


def test_tick_neck_reframe_moves_after_voice_wakes() -> None:
    adapter = RecordingNeckServoAdapter()
    runtime = HeadRuntimeApp(
        body_runtime=FakeVisionRuntime(
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "frame_id": "frame-awake-edge",
                "frame_width": 640,
                "frame_height": 480,
                "detections": [
                    {
                        "label": "face",
                        "score": 0.88,
                        "bbox": [560, 140, 620, 280],
                    }
                ],
                "evidence": {"face_crops": [{"width": 60, "height": 140}]},
            }
        ),
        voice_runtime=FakeVoiceRuntime(
            {
                "voice_dialogue": {
                    "phase": "listening",
                    "conversation_active": True,
                    "local_gate": {"state": "active", "conversationActive": True},
                }
            }
        ),
        neck_servo_adapter=adapter,
        neck_reframe_config=ReframeConfig(confirm_frames=1, min_command_interval_s=0.0),
        neck_reframe_require_voice_awake=True,
    )

    result = runtime.tick_neck_reframe(now_ts=35.0, live=True)

    assert result["status"] == "accepted"
    assert result["reason"] == "target_at_edge"
    assert result["action"]["mode"] == "reframe"
    assert result["voice_gate"]["allowed"] is True
    assert len(adapter.plans) == 1


def test_tick_neck_reframe_holds_for_zero_confidence_identity_bbox() -> None:
    adapter = RecordingNeckServoAdapter()
    runtime = HeadRuntimeApp(
        body_runtime=FakeVisionRuntime(
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "frame_id": "frame-identity-zero",
                "detections": [],
                "evidence": {
                    "frame": {"width": 640, "height": 640},
                    "face_crops": [{"width": 90, "height": 170}],
                },
                "identity_observations": [
                    {
                        "known": False,
                        "confidence": 0.0,
                        "evidence": {
                            "bbox": {"x_min": 0.82, "y_min": 0.24, "x_max": 0.98, "y_max": 0.66},
                        },
                    }
                ],
            }
        ),
        neck_servo_adapter=adapter,
        neck_reframe_config=ReframeConfig(confirm_frames=1, min_command_interval_s=0.0, min_confidence=0.45),
    )

    result = runtime.tick_neck_reframe(now_ts=31.0, live=True)

    assert result["status"] == "hold"
    assert result["reason"] == "low_confidence"
    assert result["target"]["confidence"] == 0.0
    assert adapter.plans == []


def test_tick_neck_reframe_restores_state_when_servo_dispatch_fails() -> None:
    adapter = RecordingNeckServoAdapter(status="unavailable", success=False)
    runtime = HeadRuntimeApp(
        body_runtime=FakeVisionRuntime(
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "frame_id": "frame-edge-fail",
                "frame_width": 640,
                "frame_height": 480,
                "detections": [
                    {
                        "label": "face",
                        "score": 0.88,
                        "bbox": [560, 140, 620, 280],
                    }
                ],
                "evidence": {"face_crops": [{"width": 60, "height": 140}]},
            }
        ),
        neck_servo_adapter=adapter,
        neck_reframe_config=ReframeConfig(confirm_frames=1, min_command_interval_s=0.0),
    )
    before = replace(runtime.neck_reframe_state)

    result = runtime.tick_neck_reframe(now_ts=32.0, live=True)

    assert result["status"] == "skipped"
    assert result["reason"] == "neck_servo_adapter_unavailable"
    assert result["action"]["will_move"] is False
    assert result["action"]["pan_deg"] == 90.0
    assert result["action"]["state"]["phase"] == "idle"
    assert adapter.plans
    assert runtime.neck_reframe_state.as_dict() == before.as_dict()


def test_tick_neck_reframe_matches_known_identity_by_track_id() -> None:
    adapter = RecordingNeckServoAdapter()
    runtime = HeadRuntimeApp(
        body_runtime=FakeVisionRuntime(
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "frame_id": "frame-two-faces",
                "detections": [
                    {
                        "label": "face",
                        "score": 0.90,
                        "bbox": {"x_min": 0.40, "y_min": 0.2, "x_max": 0.60, "y_max": 0.7},
                        "track_id": "known-center",
                    },
                    {
                        "label": "face",
                        "score": 0.92,
                        "bbox": {"x_min": 0.82, "y_min": 0.2, "x_max": 0.98, "y_max": 0.7},
                        "track_id": "unknown-edge",
                    },
                ],
                "identity_observations": [
                    {
                        "known": True,
                        "display_name": "Darrow",
                        "confidence": 0.96,
                        "evidence": {"track_id": "known-center"},
                    }
                ],
                "evidence": {"face_crops": [{"width": 128, "height": 180}]},
            }
        ),
        neck_servo_adapter=adapter,
        neck_reframe_config=ReframeConfig(confirm_frames=1, min_command_interval_s=0.0),
    )

    result = runtime.tick_neck_reframe(now_ts=33.0, live=True)

    assert result["target"]["track_id"] == "unknown-edge"
    assert result["target"]["known"] is False
    assert result["status"] == "accepted"
    assert result["action"]["mode"] == "reframe"


def test_tick_neck_reframe_returns_home_when_target_disappears_after_observe_hold() -> None:
    adapter = RecordingNeckServoAdapter()
    runtime = HeadRuntimeApp(
        body_runtime=FakeVisionRuntime(
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "frame_id": "frame-empty",
                "detections": [],
            }
        ),
        neck_servo_adapter=adapter,
        neck_pan_state=PanNeckState(current_angle=96.0, target_angle=96.0),
        neck_reframe_state=ReframeState(
            current_pan_deg=96.0,
            last_commanded_pan_deg=96.0,
            phase="observe",
            phase_started_at_ts=10.0,
            last_command_at_ts=9.0,
        ),
        neck_reframe_config=ReframeConfig(observe_hold_s=1.0, return_step_deg=4.0, min_command_interval_s=0.0),
    )

    result = runtime.tick_neck_reframe(now_ts=12.0, live=True)

    assert result["status"] == "accepted"
    assert result["target"] is None
    assert result["action"]["mode"] == "return_home"
    assert result["action"]["pan_deg"] == 92.0
    assert len(adapter.plans) == 1
    assert adapter.plans[0]["action"]["target_angle"] == 92


def test_runtime_http_exposes_live_neck_reframe_state() -> None:
    runtime = HeadRuntimeApp(
        body_runtime=FakeVisionRuntime(
            {
                "kind": "realtime_vision_observation",
                "mode": "realtime",
                "status": "tracking",
                "frame_id": "frame-http",
                "detections": [
                    {
                        "label": "face",
                        "score": 0.88,
                        "bbox": [560, 140, 620, 280],
                    }
                ],
                "evidence": {"face_crops": [{"width": 60, "height": 140}]},
            }
        ),
        neck_servo_adapter=RecordingNeckServoAdapter(),
        neck_reframe_config=ReframeConfig(confirm_frames=1, min_command_interval_s=0.0),
    )
    runtime.tick_neck_reframe(now_ts=40.0, live=True)

    with _running_server(runtime) as base_url:
        status_code, payload = _read_json(f"{base_url}/api/neck/status")

    assert status_code == 200
    assert payload["neck_reframe"]["schema"] == "eihead.neck.reframe_tick.v1"
    assert payload["neck_reframe"]["action"]["mode"] == "reframe"


def test_cli_http_starts_neck_reframe_loop_but_monitor_does_not() -> None:
    apps: list[RecordingLoopApp] = []

    def factory(config_path: str) -> RecordingLoopApp:
        _ = config_path
        app = RecordingLoopApp()
        apps.append(app)
        return app

    http_payload = cli.dispatch(
        cli.build_parser().parse_args(["--config", "config/test.yaml", "http"]),
        app_factory=factory,  # type: ignore[arg-type]
        http_server=lambda **kwargs: {"command": "http", "status": "stopped"},
    )
    monitor_payload = cli.dispatch(
        cli.build_parser().parse_args(["--config", "config/test.yaml", "monitor"]),
        app_factory=factory,  # type: ignore[arg-type]
        monitor_server=lambda **kwargs: {"command": "monitor", "status": "stopped"},
    )

    assert http_payload["command"] == "http"
    assert monitor_payload["command"] == "monitor"
    assert apps[0].started == 1
    assert apps[1].started == 0


@contextmanager
def _running_server(app: Any) -> Any:
    server = create_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def _read_json(url: str) -> tuple[int, dict[str, Any]]:
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=2.0) as response:
        return response.status, json.loads(response.read().decode("utf-8"))
