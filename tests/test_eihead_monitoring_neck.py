from __future__ import annotations

from contextlib import contextmanager
import json
import threading
from typing import Any, Iterator
from urllib import request
from urllib.error import HTTPError

from eihead.monitoring.web import create_server


class BaseMonitorApp:
    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "ok",
            "runtime": "eihead",
            "node_id": "honjia-test",
            "overall_status": "online",
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "schema": "eihead.status_snapshot.v1",
            "node_id": "honjia-test",
            "capabilities": {
                "neck": {"status": "online", "limits": {"yaw_deg": [40, 140], "tilt_deg": None}},
            },
        }


class SnapshotNeckApp(BaseMonitorApp):
    def snapshot(self) -> dict[str, Any]:
        return {
            "runtime": "eihead",
            "node_id": "honjia-test",
            "neck": {
                "pan": {
                    "current_angle": 87,
                    "target_angle": 92,
                    "will_move": True,
                    "suppressed": False,
                },
                "servo": {
                    "status": "unavailable",
                    "available": False,
                    "reason": "neck_servo_unavailable_off_honjia",
                },
                "axis_support": {
                    "pan": {"supported": True, "status": "supported"},
                    "tilt": {
                        "supported": False,
                        "status": "unsupported",
                        "reason": "tilt_not_supported",
                    },
                },
            },
        }


class RecentActionNeckApp(BaseMonitorApp):
    def recent_actions(self) -> list[dict[str, Any]]:
        return [
            {
                "action_id": "neck-2",
                "action_type": "move_head",
                "status": "skipped",
                "success": True,
                "details": {
                    "axis": "pan",
                    "reason": "deadband",
                    "neck_plan": _pan_plan(
                        status="suppressed",
                        will_move=False,
                        current_angle=88,
                        target_angle=89,
                        reason="deadband",
                    ),
                    "neck_servo": {
                        "status": "suppressed",
                        "reason": "deadband",
                        "angle": 89,
                    },
                },
            }
        ]


class DirectTiltPlanApp(BaseMonitorApp):
    neck_plan = {
        "schema": "eihead.neck.pan_plan.v1",
        "status": "unsupported",
        "success": False,
        "will_move": False,
        "reason": "tilt_not_supported",
        "action": {
            "axis": "pan",
            "target": "neck.pan",
            "target_angle": 90,
            "params": {"axis": "pan", "target_angle": 90},
        },
        "state": {
            "current_angle": 90,
            "target_angle": 90,
            "last_command_status": "unsupported",
            "suppression_reason": "tilt_not_supported",
            "min_angle": 40,
            "max_angle": 140,
        },
        "outcome": {
            "status": "unsupported",
            "success": False,
            "details": {"axis": "tilt", "reason": "tilt_not_supported"},
        },
    }


class BodyRuntimeOrganNeckApp(BaseMonitorApp):
    def snapshot(self) -> dict[str, Any]:
        return {
            "runtime": "eihead",
            "node_id": "honjia-test",
            "body_runtime": {
                "organs": {
                    "neck": {
                        "organ": "neck",
                        "health": "healthy",
                        "subfunctions": {
                            "motor": {
                                "health": "healthy",
                                "details": {
                                    "status": "healthy",
                                    "details": {
                                        "device": "/dev/i2c-1",
                                        "device_exists": True,
                                    },
                                },
                            },
                            "tracking": {
                                "health": "healthy",
                                "details": {
                                    "status": "tracking_ready",
                                    "neck_control": {
                                        "state": "idle",
                                        "last_angle": 90,
                                        "desired_angle": 92,
                                        "last_commanded_angle": None,
                                        "last_suppression_reason": "none",
                                    },
                                },
                            },
                        },
                    }
                }
            },
        }


@contextmanager
def running_server(app: Any, **kwargs: Any) -> Iterator[tuple[str, object, threading.Thread]]:
    server = create_server(app, host="127.0.0.1", port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", server, thread
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def test_neck_api_extracts_recent_action_plan_and_servo_truthfully() -> None:
    with running_server(RecentActionNeckApp(), clock=lambda: 1001.0) as (base_url, _server, _thread):
        status_code, payload = read_json_or_error(f"{base_url}/api/neck/status")

    assert status_code == 200
    assert payload["schema"] == "eihead.monitor.neck.v1"
    assert payload["source"] == "recent_actions.details.neck_plan"
    assert payload["captured_at_ts"] == 1001.0
    assert payload["status"] == "suppressed"
    assert payload["wired"] is True
    assert payload["current_angle"] == 88
    assert payload["target_angle"] == 89
    assert payload["will_move"] is False
    assert payload["suppressed"] is True
    assert payload["suppression_reason"] == "deadband"
    assert payload["servo_status"] == "suppressed"
    assert payload["servo"]["status"] == "suppressed"
    assert payload["servo"]["available"] is True
    assert payload["axis_support"]["pan"]["supported"] is True
    assert payload["axis_support"]["tilt"]["supported"] is False
    assert payload["axis_support"]["tilt"]["reason"] == "tilt_not_supported"


def test_neck_api_uses_snapshot_without_promoting_unavailable_servo_to_ok() -> None:
    with running_server(SnapshotNeckApp(), clock=lambda: 1002.0) as (base_url, _server, _thread):
        status_code, payload = read_json_or_error(f"{base_url}/api/neck/realtime")

    assert status_code == 200
    assert payload["source"] == "snapshot.neck"
    assert payload["status"] == "degraded"
    assert payload["wired"] is False
    assert payload["not_wired"] is False
    assert payload["current_angle"] == 87
    assert payload["target_angle"] == 92
    assert payload["will_move"] is True
    assert payload["suppressed"] is False
    assert payload["suppression_reason"] == "none"
    assert payload["servo_status"] == "unavailable"
    assert payload["servo"] == {
        "status": "unavailable",
        "available": False,
        "reason": "neck_servo_unavailable_off_honjia",
    }
    assert payload["status"] != "ok"
    assert payload.get("ok") is not True


def test_neck_api_reports_tilt_unsupported_from_direct_neck_plan() -> None:
    with running_server(DirectTiltPlanApp(), clock=lambda: 1003.0) as (base_url, _server, _thread):
        status_code, payload = read_json_or_error(f"{base_url}/api/neck/status")

    assert status_code == 200
    assert payload["source"] == "neck_plan"
    assert payload["status"] == "unsupported"
    assert payload["will_move"] is False
    assert payload["suppressed"] is False
    assert payload["suppression_reason"] == "tilt_not_supported"
    assert payload["servo_status"] == "unknown"
    assert payload["axis_support"]["pan"]["supported"] is True
    assert payload["axis_support"]["tilt"] == {
        "supported": False,
        "status": "unsupported",
        "reason": "tilt_not_supported",
    }
    assert payload["tilt"]["supported"] is False
    assert payload["tilt"]["reason"] == "tilt_not_supported"


def test_neck_api_extracts_body_runtime_organ_health_and_angles() -> None:
    with running_server(BodyRuntimeOrganNeckApp(), clock=lambda: 1003.5) as (
        base_url,
        _server,
        _thread,
    ):
        status_code, payload = read_json_or_error(f"{base_url}/api/neck/status")

    assert status_code == 200
    assert payload["source"] == "snapshot.body_runtime.organs.neck"
    assert payload["status"] == "wired"
    assert payload["wired"] is True
    assert payload["not_wired"] is False
    assert payload["current_angle"] == 90
    assert payload["target_angle"] == 92
    assert payload["will_move"] is False
    assert payload["suppressed"] is False
    assert payload["suppression_reason"] == "none"
    assert payload["servo_status"] == "healthy"
    assert payload["servo"]["available"] is True
    assert payload["servo"]["device"] == "/dev/i2c-1"
    assert payload["axis_support"]["pan"]["supported"] is True
    assert payload["axis_support"]["tilt"]["reason"] == "tilt_not_supported"


def test_neck_api_reports_not_wired_without_native_neck_data() -> None:
    with running_server(BaseMonitorApp(), clock=lambda: 1004.0) as (base_url, _server, _thread):
        status_code, payload = read_json_or_error(f"{base_url}/api/neck/status")

    assert status_code == 200
    assert payload["status"] == "not_wired"
    assert payload["wired"] is False
    assert payload["not_wired"] is True
    assert payload["source"] is None
    assert payload["current_angle"] is None
    assert payload["target_angle"] is None
    assert payload["angle_state"] == "unknown"
    assert payload["servo_status"] == "unknown"
    assert payload["servo"]["status"] == "unknown"
    assert payload["axis_support"]["pan"]["status"] == "unknown"
    assert payload["axis_support"]["tilt"]["status"] == "unsupported"
    assert payload["status"] != "ok"
    assert payload.get("ok") is not True


def test_neck_html_renders_angles_suppression_servo_and_axis_support() -> None:
    with running_server(RecentActionNeckApp(), clock=lambda: 1005.0) as (base_url, _server, _thread):
        status_code, headers, body = read_text(f"{base_url}/")

    assert status_code == 200
    assert headers["Content-Type"].startswith("text/html")
    assert "Neck Diagnostics" in body
    assert "/api/neck/status" in body
    assert "Current angle" in body
    assert "Target angle" in body
    assert "Suppressed" in body
    assert "Suppression reason" in body
    assert "Servo" in body
    assert "Axis support" in body
    assert "88" in body
    assert "89" in body
    assert "deadband" in body
    assert "suppressed" in body
    assert "tilt_not_supported" in body


def test_monitor_exposes_status_json_and_healthz_aliases() -> None:
    with running_server(BaseMonitorApp(), clock=lambda: 1006.0) as (base_url, _server, _thread):
        status_code, status_payload = read_json_or_error(f"{base_url}/status.json")
        health_code, health_payload = read_json_or_error(f"{base_url}/healthz")

    assert status_code == 200
    assert status_payload["runtime"] == "eihead"
    assert health_code == 200
    assert health_payload["status"] == "ok"


def test_monitor_lightweight_root_avoids_synchronous_diagnostics(monkeypatch) -> None:
    monkeypatch.setenv("EIHEAD_MONITOR_LIGHTWEIGHT_ROOT", "1")
    with running_server(BaseMonitorApp(), clock=lambda: 1007.0) as (base_url, _server, _thread):
        status_code, headers, body = read_text(f"{base_url}/")

    assert status_code == 200
    assert headers["Content-Type"].startswith("text/html")
    assert "lightweight shell" in body
    assert "/api/vision/realtime" in body
    assert "Promise.allSettled" in body


def test_monitor_lightweight_root_renders_human_diagnostics_instead_of_raw_json(monkeypatch) -> None:
    monkeypatch.setenv("EIHEAD_MONITOR_LIGHTWEIGHT_ROOT", "1")
    with running_server(BaseMonitorApp(), clock=lambda: 1008.0) as (base_url, _server, _thread):
        status_code, _headers, body = read_text(f"{base_url}/")

    assert status_code == 200
    assert "阻塞点" in body
    assert "证据" in body
    assert "下一步" in body
    assert "具体数据" in body
    assert "Latest JSON" not in body
    assert "JSON.stringify" not in body


def _pan_plan(
    *,
    status: str,
    will_move: bool,
    current_angle: int,
    target_angle: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": "eihead.neck.pan_plan.v1",
        "status": status,
        "success": status not in {"invalid", "unsupported"},
        "will_move": will_move,
        "reason": reason,
        "action": {
            "axis": "pan",
            "target": "neck.pan",
            "target_angle": target_angle,
            "params": {"axis": "pan", "target_angle": target_angle},
        },
        "state": {
            "current_angle": current_angle,
            "target_angle": target_angle,
            "last_command_status": status,
            "suppression_reason": reason,
            "min_angle": 40,
            "max_angle": 140,
            "deadband": 2,
        },
        "outcome": {
            "status": status,
            "success": status not in {"invalid", "unsupported"},
            "details": {"reason": reason},
        },
    }


def read_json_or_error(url: str) -> tuple[int, Any]:
    req = request.Request(url, headers={"Accept": "application/json"})
    try:
        with request.urlopen(req, timeout=2.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def read_text(url: str) -> tuple[int, Any, str]:
    req = request.Request(url, headers={"Accept": "text/html"})
    with request.urlopen(req, timeout=2.0) as response:
        return response.status, response.headers, response.read().decode("utf-8")
