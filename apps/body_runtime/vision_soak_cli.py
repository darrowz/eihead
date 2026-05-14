"""CLI runner for long Hailo vision soak collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from apps.body_runtime.vision_soak import DEFAULT_STATUS_URL
from apps.body_runtime.vision_soak import run_vision_soak


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect a long-running Hailo vision soak summary")
    parser.add_argument("--duration", type=float, default=3600.0, help="collection duration in seconds")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between status samples")
    parser.add_argument("--status-url", default=DEFAULT_STATUS_URL, help="honjia monitor JSON status URL")
    parser.add_argument("--output-path", type=Path, default=None, help="optional JSON summary output path")
    parser.add_argument("--target-fps", type=float, default=None, help="expected vision FPS")
    parser.add_argument("--min-fps-ratio", type=float, default=0.8)
    parser.add_argument("--max-p95-frame-age-ms", type=float, default=500.0)
    parser.add_argument("--max-drop-rate", type=float, default=0.1)
    parser.add_argument("--min-service-ok-ratio", type=float, default=0.95)
    parser.add_argument("--min-target-stability-ratio", type=float, default=None)
    args = parser.parse_args(argv)

    thresholds = {
        "min_fps_ratio": args.min_fps_ratio,
        "max_p95_frame_age_ms": args.max_p95_frame_age_ms,
        "max_drop_rate": args.max_drop_rate,
        "min_service_ok_ratio": args.min_service_ok_ratio,
    }
    if args.min_target_stability_ratio is not None:
        thresholds["min_target_stability_ratio"] = args.min_target_stability_ratio

    summary = run_vision_soak(
        duration_s=args.duration,
        interval_s=args.interval,
        status_url=args.status_url,
        output_path=args.output_path,
        target_fps=args.target_fps,
        thresholds=thresholds,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("pass") is True else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
