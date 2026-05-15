"""Persistent native vision loop for the honjia eye pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable, Mapping

from eihead.eye import GStreamerHailoRealtimeAdapter, GStreamerHailoRealtimeConfig
from eihead.runtime.config import load_eihead_config
from eihead.runtime.native_services import gstreamer_hailo_config_from_eihead_config


AdapterFactory = Callable[[GStreamerHailoRealtimeConfig], Any]


def build_vision_state_payload(
    status: Mapping[str, Any] | Any,
    *,
    config: GStreamerHailoRealtimeConfig,
    config_path: str,
    state_path: str | Path,
    interval_s: float,
    updated_at_ts: float,
    pid: int | None = None,
) -> dict[str, Any]:
    status_payload = _status_to_dict(status)
    status_payload.setdefault("schema", "eihead.eye.realtime_status.v1")
    status_payload.setdefault("kind", "realtime_vision_observation")
    status_payload.setdefault("mode", config.mode)
    status_payload.setdefault("backend", config.backend)
    status_payload.setdefault("source", "eihead.eye.vision_loop")
    status_payload.setdefault("placeholder", False)
    status_payload.setdefault("not_wired", False)
    status_payload.setdefault("stream_ready", False)
    status_payload.setdefault("pipeline", config.pipeline_fields())
    status_payload.setdefault(
        "devices",
        {
            "camera": config.camera_device,
            "hailo": config.hailo_device,
        },
    )
    frame_id = status_payload.get("frame_id") or status_payload.get("last_frame_id")
    captured_at_ts = (
        status_payload.get("captured_at_ts")
        or status_payload.get("timestamp")
        or status_payload.get("frame_ts")
        or updated_at_ts
    )
    return {
        "schema": "eihead.vision_state.v1",
        "source": "eihead.eye.vision_loop",
        "driver": "vision_state",
        "status": status_payload.get("status", "unknown"),
        "status_reason": status_payload.get("status_reason") or status_payload.get("status"),
        "message": status_payload.get("message") or status_payload.get("readiness_message") or "",
        "stream_ready": bool(status_payload.get("stream_ready", False)),
        "not_wired": bool(status_payload.get("not_wired", False)),
        "degraded": bool(status_payload.get("degraded", status_payload.get("status") == "degraded")),
        "frame_id": frame_id,
        "captured_at_ts": captured_at_ts,
        "updated_at_ts": updated_at_ts,
        "state_path": str(state_path),
        "config_path": str(config_path),
        "service": {
            "pid": int(pid if pid is not None else os.getpid()),
            "interval_s": float(interval_s),
        },
        "pipeline": status_payload.get("pipeline", config.pipeline_fields()),
        "devices": status_payload.get(
            "devices",
            {
                "camera": config.camera_device,
                "hailo": config.hailo_device,
            },
        ),
        "detections": status_payload.get("detections", []),
        "detection_count": _detection_count(status_payload),
        "status_payload": status_payload,
    }


def write_vision_state(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    tmp_path.replace(target)


def run_vision_loop(
    *,
    config_path: str,
    state_path: str | Path,
    interval_s: float = 0.1,
    once: bool = False,
    adapter_factory: AdapterFactory | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    config = load_eihead_config(config_path)
    realtime_config = gstreamer_hailo_config_from_eihead_config(config, config_path=config_path)
    adapter = (
        adapter_factory(realtime_config)
        if adapter_factory is not None
        else GStreamerHailoRealtimeAdapter.from_native_gstreamer(realtime_config)
    )
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    try:
        while not stop_requested:
            updated_at_ts = clock()
            try:
                status = adapter.poll()
            except Exception as exc:  # pragma: no cover - native backend defensive guard.
                status = {
                    "schema": "eihead.eye.realtime_status.v1",
                    "kind": "realtime_vision_observation",
                    "mode": realtime_config.mode,
                    "status": "degraded",
                    "backend": realtime_config.backend,
                    "source": "eihead.eye.vision_loop",
                    "placeholder": False,
                    "not_wired": False,
                    "stream_ready": False,
                    "degraded": True,
                    "status_reason": "vision_loop_poll_failed",
                    "degraded_reason": f"{exc.__class__.__name__}: {exc}",
                    "message": f"persistent vision poll failed: {exc.__class__.__name__}: {exc}",
                    "detections": [],
                    "detection_boxes": [],
                    "detection_scores": [],
                    "pipeline": realtime_config.pipeline_fields(),
                    "devices": {
                        "camera": realtime_config.camera_device,
                        "hailo": realtime_config.hailo_device,
                    },
                }
            payload = build_vision_state_payload(
                status,
                config=realtime_config,
                config_path=config_path,
                state_path=state_path,
                interval_s=interval_s,
                updated_at_ts=updated_at_ts,
            )
            write_vision_state(state_path, payload)
            if once:
                break
            sleep(max(float(interval_s), 0.01))
    finally:
        _stop_native_adapter(adapter)
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the persistent eihead Hailo vision loop.")
    parser.add_argument("--config", default="/etc/eihead/eihead.honjia.yaml")
    parser.add_argument("--state-path", default="/tmp/eibrain-vision/state.json")
    parser.add_argument("--interval-s", type=float, default=0.1)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    return run_vision_loop(
        config_path=args.config,
        state_path=args.state_path,
        interval_s=args.interval_s,
        once=args.once,
    )


def _status_to_dict(status: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(status, Mapping):
        return dict(status)
    to_dict = getattr(status, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        return dict(payload) if isinstance(payload, Mapping) else {"value": payload}
    if is_dataclass(status):
        return asdict(status)
    return {"value": status}


def _detection_count(status_payload: Mapping[str, Any]) -> int:
    detections = status_payload.get("detections")
    if isinstance(detections, list):
        return len(detections)
    value = status_payload.get("detection_count")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _stop_native_adapter(adapter: Any) -> None:
    native_reader = getattr(adapter, "_native_frame_reader", None)
    stop = getattr(native_reader, "stop", None)
    if callable(stop):
        stop()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
