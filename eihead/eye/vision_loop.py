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

from eihead.eye import (
    FaceEvidence,
    FaceIdentityMatcher,
    GStreamerHailoRealtimeAdapter,
    GStreamerHailoRealtimeConfig,
    JsonIdentityRegistry,
    OnnxFaceEmbeddingProvider,
    UnavailableFaceEmbeddingProvider,
)
from eihead.eye.identity_memory import EimemoryIdentityConfig, IdentityMemoryAdapter
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
    evidence: Mapping[str, Any] | None = None,
    identity_observations: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    status_payload = _status_to_dict(status)
    evidence_payload = dict(evidence) if isinstance(evidence, Mapping) and evidence else {}
    identity_payload = [dict(item) for item in (identity_observations or []) if isinstance(item, Mapping)]
    if evidence_payload:
        status_payload.setdefault("evidence", evidence_payload)
    if identity_payload:
        status_payload.setdefault("identity_observations", identity_payload)
        status_payload.setdefault("identity_count", len(identity_payload))
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
    payload = {
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
    if evidence_payload:
        payload["evidence"] = evidence_payload
    if identity_payload:
        payload["identity_observations"] = identity_payload
        payload["identity_count"] = len(identity_payload)
    return payload


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
    identity_matcher, memory_adapter = _identity_runtime_from_config(config)
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
            evidence = _adapter_evidence(adapter)
            identity_observations = _identity_observations_from_evidence(
                status,
                evidence=evidence,
                matcher=identity_matcher,
                memory_adapter=memory_adapter,
            )
            payload = build_vision_state_payload(
                status,
                config=realtime_config,
                config_path=config_path,
                state_path=state_path,
                interval_s=interval_s,
                updated_at_ts=updated_at_ts,
                evidence=evidence,
                identity_observations=identity_observations,
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


def _adapter_evidence(adapter: Any) -> dict[str, Any]:
    getter = getattr(adapter, "last_evidence", None)
    if callable(getter):
        try:
            evidence = getter()
        except Exception:
            return {}
        return dict(evidence) if isinstance(evidence, Mapping) else {}
    evidence = getattr(adapter, "evidence", None)
    return dict(evidence) if isinstance(evidence, Mapping) else {}


def _identity_runtime_from_config(config: Any) -> tuple[FaceIdentityMatcher | None, IdentityMemoryAdapter | None]:
    identity_config = _visual_identity_config(config)
    if not _truthy(identity_config.get("enabled"), False):
        return None, None
    registry_path = _text(identity_config.get("registry_path") or identity_config.get("registryPath"), "")
    if not registry_path:
        registry_path = "/var/lib/eihead/identity/people.json"
    model_path = _text(
        identity_config.get("model_path")
        or identity_config.get("modelPath")
        or identity_config.get("embedding_model_path")
        or identity_config.get("embeddingModelPath")
        or identity_config.get("model"),
        "",
    )
    provider = (
        OnnxFaceEmbeddingProvider(model_path=model_path)
        if model_path
        else UnavailableFaceEmbeddingProvider()
    )
    matcher = FaceIdentityMatcher(
        registry=JsonIdentityRegistry(registry_path),
        embedding_provider=provider,
        threshold=_float(identity_config.get("match_threshold") or identity_config.get("matchThreshold"), 0.85),
    )
    memory_config = _identity_memory_config(identity_config)
    memory_adapter = IdentityMemoryAdapter(memory_config) if memory_config.enabled else None
    return matcher, memory_adapter


def _identity_observations_from_evidence(
    status: Mapping[str, Any] | Any,
    *,
    evidence: Mapping[str, Any],
    matcher: FaceIdentityMatcher | None,
    memory_adapter: IdentityMemoryAdapter | None = None,
) -> list[dict[str, Any]]:
    if matcher is None or not isinstance(evidence, Mapping):
        return []
    status_payload = _status_to_dict(status)
    frame_id = _text(status_payload.get("frame_id") or status_payload.get("last_frame_id"), "")
    observed_at = _text(
        status_payload.get("observed_at")
        or status_payload.get("captured_at")
        or status_payload.get("last_frame_captured_at_ts")
        or status_payload.get("captured_at_ts"),
        "",
    )
    face_crops = evidence.get("face_crops")
    if not isinstance(face_crops, list):
        return []
    observations: list[dict[str, Any]] = []
    for index, crop in enumerate(face_crops):
        if not isinstance(crop, Mapping):
            continue
        crop_path = _text(crop.get("path") or crop.get("uri"), "")
        crop_frame_id = _text(crop.get("frame_id") or frame_id, "")
        face_id = _text(crop.get("face_id") or crop.get("faceId"), "")
        if not face_id:
            face_id = f"{crop_frame_id or frame_id}:face:{index}"
        observation = matcher.match(
            FaceEvidence(
                face_id=face_id,
                track_id=crop.get("track_id", crop.get("trackId")),
                crop_path=crop_path,
                bbox=crop.get("bbox", ()),
                embedding=crop.get("embedding"),
            )
        ).as_dict()
        observation.update(
            {
                "frame_id": crop_frame_id or frame_id,
                "observed_at": observed_at,
                "source": "eihead.eye.vision_loop",
                "crop": {
                    key: value
                    for key, value in {
                        "path": crop_path,
                        "media_type": crop.get("mime_type") or crop.get("media_type"),
                    }.items()
                    if value not in (None, "")
                },
            }
        )
        if memory_adapter is not None:
            observation["memory"] = memory_adapter.ingest_identity_observation(observation)
        observations.append(observation)
    return observations


def _visual_identity_config(config: Any) -> dict[str, Any]:
    raw = _mapping(getattr(config, "raw", None))
    capabilities = _mapping(raw.get("capabilities"))
    software = _mapping(capabilities.get("software"))
    payload = software.get("visual_identity") or software.get("identity")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _identity_memory_config(identity_config: Mapping[str, Any]) -> EimemoryIdentityConfig:
    raw_memory = identity_config.get("memory")
    memory = dict(raw_memory) if isinstance(raw_memory, Mapping) else {}
    memory.setdefault(
        "scope",
        {
            "agent_id": "honxin",
            "workspace_id": "honjia",
            "user_id": "darrow",
            "hardware_node": "honjia",
        },
    )
    return EimemoryIdentityConfig.from_mapping(memory)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any, default: str = "") -> str:
    if value in (None, ""):
        return default
    return str(value).strip()


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _stop_native_adapter(adapter: Any) -> None:
    native_reader = getattr(adapter, "_native_frame_reader", None)
    stop = getattr(native_reader, "stop", None)
    if callable(stop):
        stop()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
