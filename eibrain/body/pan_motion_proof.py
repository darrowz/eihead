"""Pan motion proof helpers for honjia camera-on-servo verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def summarize_pan_motion_pairs(
    pair_metrics: Mapping[str, Mapping[str, Any]],
    *,
    min_shift_px: float = 20.0,
    max_return_shift_px: float = 5.0,
) -> dict[str, Any]:
    """Summarize whether frame shifts prove horizontal camera motion.

    The expected proof pattern is intentionally simple: left and right commands
    must produce large opposite horizontal shifts, and returning to center must
    produce a near-zero shift from the initial center frame.
    """

    left_dx = _round_float(_metric(pair_metrics, "center_to_left", "phase_dx_px_320"))
    right_dx = _round_float(_metric(pair_metrics, "center_to_right", "phase_dx_px_320"))
    center_return_dx = _round_float(_metric(pair_metrics, "center_return", "phase_dx_px_320"))
    left_to_right_dx = _round_float(_metric(pair_metrics, "left_to_right", "phase_dx_px_320"))
    opposite_shift = abs(left_dx) >= min_shift_px and abs(right_dx) >= min_shift_px and (left_dx * right_dx) < 0
    center_returned = abs(center_return_dx) <= max_return_shift_px
    motion_score = round(((1.0 if opposite_shift else 0.0) + (1.0 if center_returned else 0.0)) / 2.0, 2)
    verified = bool(opposite_shift and center_returned)
    return {
        "status": "verified" if verified else "not_verified",
        "verified": verified,
        "left_dx_px": left_dx,
        "right_dx_px": right_dx,
        "center_return_dx_px": center_return_dx,
        "left_to_right_dx_px": left_to_right_dx,
        "motion_score": motion_score,
        "min_shift_px": min_shift_px,
        "max_return_shift_px": max_return_shift_px,
        "pairs": {name: dict(metrics) for name, metrics in pair_metrics.items()},
    }


def compare_frame_paths(left: str | Path, right: str | Path, *, resize_px: int = 320) -> dict[str, Any]:
    """Compare two image files and estimate horizontal shift.

    OpenCV is optional so the main runtime can still start on hosts without it.
    The caller should surface the error as a degraded diagnostic if unavailable.
    """

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - host dependency
        raise RuntimeError(f"opencv unavailable for pan motion proof: {exc}") from exc

    left_gray = cv2.imread(str(left), cv2.IMREAD_GRAYSCALE)
    right_gray = cv2.imread(str(right), cv2.IMREAD_GRAYSCALE)
    if left_gray is None or right_gray is None:
        raise RuntimeError("failed to read pan proof image")
    left_resized = cv2.resize(left_gray, (resize_px, resize_px), interpolation=cv2.INTER_AREA).astype("float32")
    right_resized = cv2.resize(right_gray, (resize_px, resize_px), interpolation=cv2.INTER_AREA).astype("float32")
    diff = left_resized - right_resized
    mad = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    corr = float(np.corrcoef(left_resized.flatten(), right_resized.flatten())[0, 1])
    window = cv2.createHanningWindow((resize_px, resize_px), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(left_resized * window, right_resized * window)
    return {
        "mad_0_255": round(mad, 2),
        "rmse_0_255": round(rmse, 2),
        "corr": round(corr, 4),
        "phase_dx_px_320": round(float(shift[0]), 2),
        "phase_dy_px_320": round(float(shift[1]), 2),
        "phase_response": round(float(response), 4),
    }


def write_pan_motion_summary(path: str | Path, summary: Mapping[str, Any]) -> None:
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(dict(summary), ensure_ascii=False, indent=2), encoding="utf-8")


def _metric(pair_metrics: Mapping[str, Mapping[str, Any]], pair_name: str, metric_name: str) -> float:
    try:
        return float(pair_metrics.get(pair_name, {}).get(metric_name, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _round_float(value: float) -> float:
    return round(float(value), 2)
