"""Vision pipeline soak-test summaries.

This module is intentionally hardware-agnostic: callers can feed samples from
the live Hailo service, web monitor snapshots, or fixture data.
"""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_STATUS_URL = "http://127.0.0.1:18081/status.json"


def make_http_status_source(status_url: str, *, timeout_s: float = 5.0) -> Callable[[], dict[str, Any]]:
    """Build a small JSON status fetcher for honjia's local monitor endpoint."""

    def _source() -> dict[str, Any]:
        request = Request(status_url, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=max(0.1, float(timeout_s))) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} while reading {status_url}") from exc
        except URLError as exc:
            raise RuntimeError(f"could not read {status_url}: {exc.reason}") from exc
        if not isinstance(payload, dict):
            raise ValueError("vision soak status payload must be a JSON object")
        return payload

    return _source


def collect_vision_soak(
    status_source: Callable[[], Mapping[str, Any]],
    *,
    duration_s: float,
    interval_s: float,
    target_fps: float | None = None,
    thresholds: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Sample a status source for ``duration_s`` and return a JSON-ready summary."""

    requested_duration_s = max(0.0, _to_float(duration_s))
    effective_interval_s = max(0.001, _to_float(interval_s))
    start = clock()
    deadline = start + requested_duration_s
    now = start
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    while now < deadline:
        elapsed_s = max(0.0, now - start)
        try:
            payload = status_source()
            samples.append(normalize_vision_status_sample(payload, elapsed_s=elapsed_s, target_fps=target_fps))
        except Exception as exc:  # noqa: BLE001 - soak collection should survive transient endpoint failures.
            error = {"elapsed_s": _round(elapsed_s), "type": type(exc).__name__, "message": str(exc)}
            errors.append(error)
            samples.append(
                {
                    "elapsed_s": elapsed_s,
                    "fps": 0.0,
                    "target_fps": target_fps,
                    "frame_age_ms": None,
                    "dropped_frames": 0,
                    "service_state": "error",
                    "collection_error": dict(error),
                }
            )
        remaining_s = deadline - clock()
        if remaining_s <= 0.0:
            break
        sleeper(min(effective_interval_s, remaining_s))
        now = clock()

    summary = summarize_vision_soak(samples, target_fps=target_fps, **dict(thresholds or {}))
    end = clock()
    summary["collection"] = {
        "source": "status_source",
        "requested_duration_s": _round(requested_duration_s),
        "duration_s": _round(max(0.0, end - start)),
        "interval_s": _round(effective_interval_s),
        "error_count": len(errors),
        "errors": errors,
    }
    return summary


def run_vision_soak(
    *,
    duration_s: float,
    interval_s: float,
    status_url: str | None = None,
    status_source: Callable[[], Mapping[str, Any]] | None = None,
    output_path: str | Path | None = None,
    target_fps: float | None = None,
    thresholds: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Collect a long-running vision soak and optionally write its JSON summary."""

    source = status_source or make_http_status_source(status_url or DEFAULT_STATUS_URL)
    summary = collect_vision_soak(
        source,
        duration_s=duration_s,
        interval_s=interval_s,
        target_fps=target_fps,
        thresholds=thresholds,
        clock=clock,
        sleeper=sleeper,
    )
    if status_url:
        summary["collection"]["status_url"] = status_url
    if output_path is not None:
        write_vision_soak_summary(summary, output_path)
    return summary


def write_vision_soak_summary(summary: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_vision_status_sample(
    payload: Mapping[str, Any],
    *,
    elapsed_s: float | None = None,
    target_fps: float | None = None,
) -> dict[str, Any]:
    """Flatten honjia monitor status or eye diagnostics into a soak sample."""

    root = dict(payload)
    eye = _mapping_or_empty(root.get("eye"))
    visual = _mapping_or_empty(root.get("visual_diagnostics"))
    body_camera = _body_eye_camera_details(root)
    diagnostics = _first_mapping(
        root.get("diagnostics"),
        body_camera or None,
        eye.get("diagnostics"),
        visual.get("diagnostics"),
        visual,
        eye,
        root,
    )
    soak_summary_candidates = (
        root.get("soak_summary"),
        diagnostics.get("soak_summary"),
        visual.get("soak_summary"),
        eye.get("soak_summary"),
    )
    soak_summary = _first_mapping(*soak_summary_candidates)
    monitor = _monitor_snapshot(root, visual=visual, eye=eye, diagnostics=diagnostics)
    restart_evidence = _service_restart_evidence(root, diagnostics=diagnostics, visual=visual, eye=eye)
    body_frame_age_ms = _frame_age_ms(body_camera)
    summary_frame_age_ms = _first_frame_age_ms(*soak_summary_candidates)
    diagnostic_frame_age_ms = _frame_age_ms(diagnostics)
    frame_age_ms = (
        body_frame_age_ms
        if body_frame_age_ms is not None
        else summary_frame_age_ms
        if summary_frame_age_ms is not None
        else diagnostic_frame_age_ms
    )
    sample: dict[str, Any] = {
        "elapsed_s": elapsed_s,
        "fps": _first_present(
            diagnostics,
            "fps",
            "vision_fps",
            "current_fps",
            "currentFps",
        ),
        "target_fps": target_fps
        if target_fps is not None
        else _first_present(diagnostics, "target_fps", "vision_target_fps", "configured_target_fps"),
        "frame_age_ms": frame_age_ms,
        "loop_elapsed_ms": _first_present(diagnostics, "loop_elapsed_ms", "vision_loop_elapsed_ms"),
        "dropped_frames": _first_present(diagnostics, "dropped_frames", "frame_drops", default=0),
        "dropout": _truthy(_first_present(diagnostics, "dropout", "frame_dropout", default=False)),
        "service_state": _first_present(
            diagnostics,
            "service_state",
            "vision_service_status",
            "status",
            "state",
            default=eye.get("status")
            or visual.get("vision_service_status")
            or visual.get("data_status")
            or visual.get("tracking_status")
            or visual.get("status")
            or root.get("system_health")
            or root.get("status"),
        ),
        "event_count": _first_present(diagnostics, "event_count", "vision_event_count"),
        "events": _first_list(root.get("events"), eye.get("events"), diagnostics.get("events")),
        "tracks": _first_list(
            root.get("tracks"),
            eye.get("tracks"),
            diagnostics.get("tracks"),
            diagnostics.get("objects"),
            diagnostics.get("tracking_target"),
        ),
        "stable_target": _first_mapping(
            diagnostics.get("stable_target"),
            diagnostics.get("target"),
            diagnostics.get("tracking_target"),
        ),
        "hailo_metadata": _first_mapping(
            root.get("hailo_metadata"),
            diagnostics.get("hailo_metadata"),
            visual.get("hailo_metadata"),
            eye.get("hailo_metadata"),
        ),
        "monitor_active": monitor["active"],
        "monitor_status": monitor["status"],
        "service_restart_count": restart_evidence["restart_count"],
        "service_restart_evidence": restart_evidence,
    }
    if soak_summary:
        sample.update(
            {
                "track_id_switch_count": soak_summary.get("track_id_switch_count"),
                "target_stability_ratio": soak_summary.get("target_stability_ratio"),
                "event_rate_hz": soak_summary.get("event_rate_hz"),
                "frame_drop_tolerance": soak_summary.get("frame_drop_tolerance"),
            }
        )
    return sample


def summarize_vision_soak(
    samples: Iterable[dict[str, Any]],
    *,
    target_fps: float | None = None,
    min_fps_ratio: float = 0.8,
    max_p95_frame_age_ms: float = 500.0,
    max_drop_rate: float = 0.1,
    min_service_ok_ratio: float = 0.95,
    min_target_stability_ratio: float | None = None,
) -> dict[str, Any]:
    """Summarize health from vision telemetry samples.

    ``drop_rate`` is intentionally reported as dropped frames per sample because
    current runtime telemetry exposes cumulative-ish sample counters rather than
    total produced frames.
    """

    rows = [dict(sample) for sample in samples]
    if not rows:
        empty_tracking = _tracking_diagnostics([], target_fps=0.0)
        empty_restart = _aggregate_restart_evidence([])
        return {
            "pass": False,
            "sample_count": 0,
            "fps": _stats([]),
            "frame_age_ms": _stats([]),
            "p95_frame_age_ms": 0.0,
            "loop_elapsed_ms": _stats([]),
            "fps_ratio": 0.0,
            "drop_rate": 0.0,
            "stale_ratio": 0.0,
            "service_ok_ratio": 0.0,
            "monitor_active_ratio": 0.0,
            "bottleneck_reason": "no_samples",
            "pass_fail_reason": "no_samples",
            "fail_reason": "no_samples",
            **empty_tracking,
            "service_restart_evidence": empty_restart,
            "readiness": _vision_readiness_summary(
                fps_ratio=0.0,
                min_fps_ratio=min_fps_ratio,
                p95_frame_age_ms=0.0,
                max_p95_frame_age_ms=max_p95_frame_age_ms,
                drop_rate=0.0,
                max_drop_rate=max_drop_rate,
                monitor_active_ratio=0.0,
                monitor_checked=False,
            ),
            "metadata": _summary_metadata(empty_tracking, hailo_metadata={}),
        }

    fps_values = [_to_float(row.get("fps")) for row in rows if row.get("fps") is not None]
    frame_age_values = [
        _to_float(row.get("frame_age_ms"))
        for row in rows
        if row.get("frame_age_ms") is not None
    ]
    loop_values = [
        _to_float(row.get("loop_elapsed_ms"))
        for row in rows
        if row.get("loop_elapsed_ms") is not None
    ]
    effective_target_fps = _resolve_target_fps(rows, target_fps)
    fps_avg = _average(fps_values)
    fps_ratio = fps_avg / effective_target_fps if effective_target_fps > 0 else 0.0
    total_drops = _drop_count(rows)
    drop_rate = total_drops / len(rows)
    stale_count = sum(1 for value in frame_age_values if value > max_p95_frame_age_ms)
    stale_ratio = stale_count / len(rows)
    service_ok_count = sum(1 for row in rows if _service_is_ok(row.get("service_state")))
    service_ok_ratio = service_ok_count / len(rows)
    monitor_values = [bool(row.get("monitor_active")) for row in rows if row.get("monitor_active") is not None]
    monitor_active_ratio = sum(1 for value in monitor_values if value) / len(rows)
    restart_evidence = _aggregate_restart_evidence(rows)

    fps_summary = _stats(fps_values)
    frame_age_summary = _stats(frame_age_values)
    service_unstable = service_ok_ratio < min_service_ok_ratio
    low_fps = fps_ratio < min_fps_ratio
    stale_frames = frame_age_summary["p95"] > max_p95_frame_age_ms
    frame_drops = drop_rate > max_drop_rate
    tracking = _tracking_diagnostics(rows, target_fps=effective_target_fps)
    target_unstable = (
        min_target_stability_ratio is not None
        and tracking["target_observed"]
        and tracking["target_stability_ratio"] < _to_float(min_target_stability_ratio)
    )
    reason = _bottleneck_reason(
        service_unstable=service_unstable,
        low_fps=low_fps,
        stale_frames=stale_frames,
        frame_drops=frame_drops,
        target_unstable=target_unstable,
    )
    metadata = _summary_metadata(tracking, hailo_metadata=_hailo_metadata(rows))
    readiness = _vision_readiness_summary(
        fps_ratio=fps_ratio,
        min_fps_ratio=min_fps_ratio,
        p95_frame_age_ms=frame_age_summary["p95"],
        max_p95_frame_age_ms=max_p95_frame_age_ms,
        drop_rate=drop_rate,
        max_drop_rate=max_drop_rate,
        monitor_active_ratio=monitor_active_ratio,
        monitor_checked=bool(monitor_values),
    )

    return {
        "pass": reason == "healthy",
        "sample_count": len(rows),
        "fps": fps_summary,
        "frame_age_ms": frame_age_summary,
        "p95_frame_age_ms": frame_age_summary["p95"],
        "loop_elapsed_ms": _stats(loop_values),
        "target_fps": _round(effective_target_fps),
        "fps_ratio": fps_ratio,
        "drop_rate": drop_rate,
        "stale_ratio": stale_ratio,
        "service_ok_ratio": service_ok_ratio,
        "monitor_active_ratio": monitor_active_ratio,
        "bottleneck_reason": reason,
        "pass_fail_reason": reason,
        "fail_reason": None if reason == "healthy" else reason,
        **tracking,
        "service_restart_evidence": restart_evidence,
        "readiness": readiness,
        "metadata": metadata,
    }


def run_synthetic_vision_soak(
    *,
    frame_count: int = 90,
    target_fps: float = 10.0,
) -> dict[str, Any]:
    """Run a deterministic Hailo-like tracking soak without hardware.

    The scenario injects small jitter, a one-frame detection dropout, a forced
    track-id switch, a persistent target swap, and a short loss/recovery window.
    """

    samples = list(_synthetic_long_tracking_samples(frame_count=max(1, int(frame_count)), target_fps=target_fps))
    summary = summarize_vision_soak(samples, target_fps=target_fps)
    coverage = {
        "jitter": any("jitter" in row.get("scenario_flags", []) for row in samples),
        "dropout": any("dropout" in row.get("scenario_flags", []) for row in samples),
        "track_id_switch": summary["track_id_switch_count"] > 0,
        "target_swap": summary["target_switch_count"] > 0,
        "short_loss_recovery": summary["frame_drop_tolerance"] > 0,
    }
    summary["pass"] = bool(summary["pass"] and all(coverage.values()) and summary["target_stability_ratio"] >= 0.75)
    summary["scenario_coverage"] = coverage
    summary["metadata"]["trace"]["name"] = "vision_soak.synthetic_long_tracking"
    summary["metadata"]["web"]["scenario_coverage"] = dict(coverage)
    return summary


def _resolve_target_fps(samples: list[dict[str, Any]], explicit: float | None) -> float:
    if explicit is not None:
        return max(0.0, _to_float(explicit))
    for row in samples:
        value = row.get("target_fps")
        if value is not None:
            return max(0.0, _to_float(value))
    return 10.0


def _tracking_diagnostics(rows: list[dict[str, Any]], *, target_fps: float) -> dict[str, Any]:
    explicit_track_switches = _explicit_numbers(rows, "track_id_switch_count")
    track_id_switch_count = (
        int(max(explicit_track_switches))
        if explicit_track_switches
        else _track_id_switch_count(rows)
    )
    target_ids = [_stable_target_track_id(row) for row in rows]
    target_ids = [track_id for track_id in target_ids if track_id]
    target_switch_count = sum(
        1
        for previous, current in zip(target_ids, target_ids[1:], strict=False)
        if previous != current
    )
    comparable_targets = max(0, len(target_ids) - 1)
    target_stability_ratio = (
        (comparable_targets - target_switch_count) / comparable_targets
        if comparable_targets > 0
        else (1.0 if target_ids else 0.0)
    )
    explicit_stability = _explicit_numbers(rows, "target_stability_ratio")
    if not target_ids and explicit_stability:
        positive = [value for value in explicit_stability if value > 0.0]
        target_stability_ratio = _average(positive) if positive else 1.0
    target_observed = bool(target_ids or any(_sample_tracks(row) or row.get("stable_target") for row in rows))
    total_events = sum(_event_count(row) for row in rows)
    duration_s = _duration_s(rows, target_fps=target_fps)
    explicit_event_rates = _explicit_numbers(rows, "event_rate_hz")
    explicit_drop_tolerance = _explicit_numbers(rows, "frame_drop_tolerance")
    return {
        "track_id_switch_count": int(track_id_switch_count),
        "target_switch_count": int(target_switch_count),
        "target_stability_ratio": target_stability_ratio,
        "target_observed": target_observed,
        "event_rate_hz": _round(_average(explicit_event_rates))
        if explicit_event_rates and total_events == 0
        else (_round(total_events / duration_s) if duration_s > 0.0 else 0.0),
        "frame_drop_tolerance": int(max(explicit_drop_tolerance))
        if explicit_drop_tolerance
        else int(_frame_drop_tolerance(rows)),
    }


def _track_id_switch_count(rows: list[dict[str, Any]]) -> int:
    last_track_by_subject: dict[str, str] = {}
    switches = 0
    for row in rows:
        for track in _sample_tracks(row):
            subject_id = str(track.get("synthetic_subject_id") or track.get("subject_id") or "")
            track_id = str(track.get("track_id") or track.get("trackId") or "")
            if not subject_id or not track_id:
                continue
            previous = last_track_by_subject.get(subject_id)
            if previous and previous != track_id:
                switches += 1
            last_track_by_subject[subject_id] = track_id
    return switches


def _sample_tracks(row: dict[str, Any]) -> list[dict[str, Any]]:
    tracks = row.get("tracks", [])
    if isinstance(tracks, list):
        return [dict(item) for item in tracks if isinstance(item, dict)]
    objects = row.get("objects", [])
    if isinstance(objects, list):
        return [dict(item) for item in objects if isinstance(item, dict)]
    return []


def _stable_target_track_id(row: dict[str, Any]) -> str:
    for key in ("stable_target_track_id", "stableTargetTrackId"):
        value = row.get(key)
        if value:
            return str(value)
    for key in ("stable_target", "stableTarget", "attention"):
        value = row.get(key)
        if isinstance(value, dict):
            track_id = value.get("track_id") or value.get("trackId")
            if track_id:
                return str(track_id)
    return ""


def _event_count(row: dict[str, Any]) -> int:
    if row.get("event_count") is not None:
        return max(0, int(_to_float(row.get("event_count"))))
    events = row.get("events")
    return len(events) if isinstance(events, list) else 0


def _duration_s(rows: list[dict[str, Any]], *, target_fps: float) -> float:
    explicit = [_to_float(row.get("elapsed_s")) for row in rows if row.get("elapsed_s") is not None]
    if explicit:
        return max(0.001, max(explicit) - min(explicit))
    fps = _average([_to_float(row.get("fps")) for row in rows if row.get("fps") is not None])
    effective_fps = fps if fps > 0.0 else target_fps
    if effective_fps <= 0.0:
        return 0.0
    return max(1.0 / effective_fps, len(rows) / effective_fps)


def _frame_drop_tolerance(rows: list[dict[str, Any]]) -> int:
    max_run = 0
    current_run = 0
    for row in rows:
        is_dropout = bool(row.get("dropout")) or _to_float(row.get("dropped_frames", 0.0)) > 0.0
        if is_dropout:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _drop_count(rows: list[dict[str, Any]]) -> float:
    values = [max(0.0, _to_float(row.get("dropped_frames", 0.0))) for row in rows]
    if not values:
        return 0.0
    monotonic = all(current >= previous for previous, current in zip(values, values[1:], strict=False))
    if monotonic:
        return values[-1] - values[0] if values[0] == 0.0 else values[-1]
    return sum(values)


def _hailo_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        value = row.get("hailo_metadata")
        if isinstance(value, dict):
            return dict(value)
    return {}


def _summary_metadata(metrics: dict[str, Any], *, hailo_metadata: dict[str, Any]) -> dict[str, Any]:
    metric_subset = {
        "track_id_switch_count": metrics.get("track_id_switch_count", 0),
        "target_stability_ratio": metrics.get("target_stability_ratio", 0.0),
        "event_rate_hz": metrics.get("event_rate_hz", 0.0),
        "frame_drop_tolerance": metrics.get("frame_drop_tolerance", 0),
    }
    return {
        "hailo": dict(hailo_metadata),
        "trace": {
            "kind": "vision_tracking_soak",
            "name": "vision_soak.summary",
            "metrics": metric_subset,
        },
        "web": {
            "kind": "vision_soak_summary",
            "metrics": metric_subset,
        },
    }


def _synthetic_long_tracking_samples(*, frame_count: int, target_fps: float) -> Iterable[dict[str, Any]]:
    from eibrain.cognition.vision_realtime import RealtimeVisionSimulator

    simulator = RealtimeVisionSimulator(
        match_distance=0.12,
        move_threshold=0.08,
        max_missing_frames=2,
        attention_switch_margin=0.10,
        attention_switch_cooldown_frames=2,
    )
    last_primary_track_id = ""
    primary_was_missing = False
    for index in range(frame_count):
        frame_number = index + 1
        flags: list[str] = []
        detections: list[dict[str, Any]] = []
        subjects: dict[str, dict[str, float]] = {}

        primary_bbox = _primary_bbox(index)
        if frame_number in {8, 26}:
            flags.append("dropout")
        else:
            if frame_number == 16:
                primary_bbox = {"x_min": 0.62, "y_min": 0.18, "x_max": 0.82, "y_max": 0.78}
                flags.append("track_id_switch")
            if index % 2 == 1:
                flags.append("jitter")
            subjects["primary"] = primary_bbox
            detections.append(_synthetic_detection("person", primary_bbox, 0.92))

        if frame_number >= 21:
            flags.append("target_swap")
            secondary_bbox = {"x_min": 0.12, "y_min": 0.12, "x_max": 0.42, "y_max": 0.88}
            subjects["swap"] = secondary_bbox
            detections.append(_synthetic_detection("person", secondary_bbox, 0.97))

        snapshot = simulator.update(
            frame_id=f"synthetic-{frame_number:04d}",
            observed_at=f"2026-05-05T10:00:{index / target_fps:06.3f}+08:00",
            detections=detections,
        )
        scene = snapshot["sceneSnapshot"]
        tracks = _assign_synthetic_subjects(scene["objects"], subjects)
        primary_track = next((track["track_id"] for track in tracks if track["synthetic_subject_id"] == "primary"), "")
        if primary_was_missing and primary_track and primary_track == last_primary_track_id:
            flags.append("short_loss_recovery")
        primary_was_missing = "primary" not in subjects
        if primary_track:
            last_primary_track_id = primary_track

        yield {
            "fps": target_fps,
            "target_fps": target_fps,
            "frame_age_ms": 85.0 + float(index % 5) * 4.0,
            "loop_elapsed_ms": 70.0 + float(index % 4) * 3.0,
            "dropped_frames": 1 if "dropout" in flags else 0,
            "dropout": "dropout" in flags,
            "service_state": "ok",
            "event_count": len(snapshot["events"]),
            "events": snapshot["events"],
            "stable_target_track_id": str(scene.get("stableTarget", {}).get("trackId", "")),
            "tracks": tracks,
            "scenario_flags": flags,
            "hailo_metadata": {
                "backend": "synthetic_hailo",
                "model": "synthetic-long-tracking",
                "device": "software",
                "frame_id": scene["frameId"],
            },
        }


def _primary_bbox(index: int) -> dict[str, float]:
    jitter = 0.006 if index % 2 else -0.004
    return {
        "x_min": 0.38 + jitter,
        "y_min": 0.20,
        "x_max": 0.58 + jitter,
        "y_max": 0.80,
    }


def _synthetic_detection(label: str, bbox: dict[str, float], confidence: float) -> dict[str, Any]:
    return {"label": label, "confidence": confidence, "bbox": dict(bbox)}


def _assign_synthetic_subjects(
    objects: list[dict[str, Any]],
    subjects: dict[str, dict[str, float]],
) -> list[dict[str, str]]:
    assignments: list[dict[str, str]] = []
    used_track_ids: set[str] = set()
    for subject_id, bbox in subjects.items():
        subject_center = _bbox_center(bbox)
        best: tuple[float, dict[str, Any]] | None = None
        for obj in objects:
            track_id = str(obj.get("trackId", ""))
            if not track_id or track_id in used_track_ids:
                continue
            obj_center = obj.get("center")
            if isinstance(obj_center, dict):
                center = (_to_float(obj_center.get("x")), _to_float(obj_center.get("y")))
            else:
                center = _bbox_center(obj.get("bbox", {}))
            distance = ((subject_center[0] - center[0]) ** 2 + (subject_center[1] - center[1]) ** 2) ** 0.5
            if best is None or distance < best[0]:
                best = (distance, obj)
        if best is None:
            continue
        track_id = str(best[1].get("trackId", ""))
        used_track_ids.add(track_id)
        assignments.append({"synthetic_subject_id": subject_id, "track_id": track_id})
    return assignments


def _bbox_center(bbox: Any) -> tuple[float, float]:
    if not isinstance(bbox, dict):
        return (0.0, 0.0)
    return (
        (_to_float(bbox.get("x_min")) + _to_float(bbox.get("x_max"))) / 2.0,
        (_to_float(bbox.get("y_min")) + _to_float(bbox.get("y_max"))) / 2.0,
    )


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _first_list(*values: Any) -> list[dict[str, Any]]:
    for value in values:
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
        if isinstance(value, Mapping):
            return [dict(value)]
    return []


def _body_eye_camera_details(root: Mapping[str, Any]) -> dict[str, Any]:
    body = _mapping_or_empty(root.get("body"))
    body_state = _first_mapping(root.get("body_state"), body.get("body_state"))
    organs = _mapping_or_empty(body_state.get("organs"))
    eye = _mapping_or_empty(organs.get("eye"))
    subfunctions = _mapping_or_empty(eye.get("subfunctions"))
    camera = _mapping_or_empty(subfunctions.get("camera"))
    return _first_mapping(camera.get("details"), camera)


def _first_frame_age_ms(*values: Any) -> float | None:
    for value in values:
        if not isinstance(value, Mapping):
            continue
        frame_age_ms = _frame_age_ms(value)
        if frame_age_ms is not None:
            return frame_age_ms
    return None


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _frame_age_ms(diagnostics: Mapping[str, Any]) -> float | None:
    value = _first_present(diagnostics, "frame_age_ms", "vision_frame_age_ms", "p95_frame_age_ms")
    if value is not None:
        return _try_float(value)
    seconds = _first_present(
        diagnostics,
        "frame_age_s",
        "vision_frame_age_s",
        "last_frame_age_s",
        "last_frame_age",
        "state_age_s",
        "frame_state_age_s",
        "frame_age",
    )
    if seconds is None:
        return None
    parsed = _try_float(seconds)
    if parsed is None:
        return None
    return parsed * 1000.0


def _explicit_numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [_to_float(row.get(key)) for row in rows if row.get(key) is not None]


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _monitor_snapshot(
    root: Mapping[str, Any],
    *,
    visual: Mapping[str, Any],
    eye: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    monitor = _first_mapping(root.get("monitor"), root.get("web_monitor"), root.get("monitoring"))
    active_value = monitor.get("active")
    if active_value is None:
        active_value = _first_present(diagnostics, "monitor_active", "tracking_running", "frame_available")
    if active_value is None:
        status_value = (
            visual.get("data_status")
            or visual.get("tracking_status")
            or visual.get("vision_service_status")
            or eye.get("status")
            or root.get("status")
        )
        active_value = _service_is_ok(status_value)
    status = str(
        monitor.get("status")
        or visual.get("data_status")
        or visual.get("tracking_status")
        or visual.get("vision_service_status")
        or root.get("status")
        or ("active" if _truthy(active_value) else "inactive")
    )
    return {"active": _truthy(active_value), "status": status}


def _service_restart_evidence(
    root: Mapping[str, Any],
    *,
    diagnostics: Mapping[str, Any],
    visual: Mapping[str, Any],
    eye: Mapping[str, Any],
) -> dict[str, Any]:
    services = _mapping_or_empty(root.get("services"))
    candidates = [
        services.get("vision"),
        services.get("monitor"),
        services.get("runtime"),
        root.get("service"),
        diagnostics.get("service"),
        visual.get("service"),
        eye.get("service"),
    ]
    count = 0
    last_restart_ts = None
    active_since_ts = None
    active = None
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        count = max(count, _int_from_mapping(candidate, "restart_count", "restartCount", "n_restarts", "NRestarts", "restarts"))
        if last_restart_ts is None:
            last_restart_ts = _first_present(candidate, "last_restart_ts", "lastRestartTs", "ExecMainStartTimestamp")
        if active_since_ts is None:
            active_since_ts = _first_present(candidate, "active_since_ts", "activeSinceTs", "ActiveEnterTimestamp")
        if active is None:
            active = _first_present(candidate, "active", "running")
    return {
        "observed": count > 0 or bool(last_restart_ts),
        "restart_count": count,
        "last_restart_ts": last_restart_ts,
        "active_since_ts": active_since_ts,
        "active": _truthy(active) if active is not None else None,
    }


def _aggregate_restart_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = 0
    last_restart_ts = None
    active_since_ts = None
    for row in rows:
        count = max(count, int(_to_float(row.get("service_restart_count", 0.0))))
        evidence = row.get("service_restart_evidence")
        if isinstance(evidence, Mapping):
            if last_restart_ts is None:
                last_restart_ts = evidence.get("last_restart_ts")
            if active_since_ts is None:
                active_since_ts = evidence.get("active_since_ts")
    return {
        "observed": count > 0 or bool(last_restart_ts),
        "restart_count": count,
        "last_restart_ts": last_restart_ts,
        "active_since_ts": active_since_ts,
    }


def _vision_readiness_summary(
    *,
    fps_ratio: float,
    min_fps_ratio: float,
    p95_frame_age_ms: float,
    max_p95_frame_age_ms: float,
    drop_rate: float,
    max_drop_rate: float,
    monitor_active_ratio: float,
    monitor_checked: bool,
) -> dict[str, Any]:
    monitor_ok = monitor_active_ratio > 0.0 if monitor_checked else None
    return {
        "hailo_fps": {
            "ok": fps_ratio >= min_fps_ratio,
            "observed_ratio": _round(fps_ratio),
            "threshold_ratio": _round(min_fps_ratio),
        },
        "hailo_drop_rate": {
            "ok": drop_rate <= max_drop_rate,
            "observed": _round(drop_rate),
            "threshold": _round(max_drop_rate),
        },
        "hailo_frame_age": {
            "ok": p95_frame_age_ms <= max_p95_frame_age_ms,
            "observed_p95_ms": _round(p95_frame_age_ms),
            "threshold_ms": _round(max_p95_frame_age_ms),
        },
        "monitor_active": {
            "ok": monitor_ok,
            "status": "checked" if monitor_checked else "not_checked",
            "observed_ratio": _round(monitor_active_ratio),
        },
    }


def _service_is_ok(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"ok", "active", "running", "healthy", "ready", "live", "tracking"}


def _bottleneck_reason(
    *,
    service_unstable: bool,
    low_fps: bool,
    stale_frames: bool,
    frame_drops: bool,
    target_unstable: bool = False,
) -> str:
    if service_unstable:
        return "service_unstable"
    if low_fps:
        return "low_fps"
    if stale_frames:
        return "stale_frames"
    if frame_drops:
        return "frame_drops"
    if target_unstable:
        return "target_unstable"
    return "healthy"


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    return {
        "avg": _round(_average(sorted_values)),
        "p50": _round(_percentile(sorted_values, 0.50)),
        "p95": _round(_percentile(sorted_values, 0.95)),
        "max": _round(sorted_values[-1]),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, ceil(percentile * len(sorted_values)) - 1))
    return sorted_values[index]


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _try_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_from_mapping(mapping: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return int(max(0.0, _to_float(value)))
    return 0


def _round(value: float) -> float:
    return round(float(value), 3)
