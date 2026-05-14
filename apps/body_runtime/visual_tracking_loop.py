"""Continuous visual tracking loop for honjia."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

from apps.body_runtime.app import BodyRuntimeApp


class VisualTrackingLoop:
    def __init__(
        self,
        *,
        body_runtime: BodyRuntimeApp,
        interval_s: float = 0.5,
        recenter_after_misses: int = 3,
        sleeping_interval_s: float = 0.5,
        engagement_reader: callable | None = None,
        source: str = "visual_tracking",
    ) -> None:
        self.body_runtime = body_runtime
        self.interval_s = max(0.2, float(interval_s))
        self.recenter_after_misses = max(1, int(recenter_after_misses))
        self.sleeping_interval_s = max(0.2, float(sleeping_interval_s))
        self.engagement_reader = engagement_reader
        self.source = source
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sleeping_paused = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="eibrain-visual-tracking", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                if self.engagement_reader is not None:
                    should_track = (
                        self.engagement_reader.should_run_vision()
                        if hasattr(self.engagement_reader, "should_run_vision")
                        else bool(self.engagement_reader())
                    )
                else:
                    should_track = True
                if not should_track:
                    if hasattr(self.body_runtime, "update_voice_dialogue_state"):
                        self.body_runtime.update_voice_dialogue_state(
                            running=False,
                            phase="waiting",
                        )
                    if hasattr(self.body_runtime, "_update_visual_tracking_state"):
                        self.body_runtime._update_visual_tracking_state(
                            running=False,
                            status="sleeping",
                        )
                    if not self._sleeping_paused and hasattr(self.body_runtime, "pause_visual_tracking"):
                        self.body_runtime.pause_visual_tracking(reason="engagement_inactive")
                        self._sleeping_paused = True
                    self._stop.wait(max(0.0, self.sleeping_interval_s))
                    continue
                self._sleeping_paused = False
                if hasattr(self.body_runtime, "update_voice_dialogue_state"):
                    self.body_runtime.update_voice_dialogue_state(running=True)
                tracking_payload: object | None = None
                for kwargs in (
                    {
                        "recenter_after_misses": self.recenter_after_misses,
                        "source": self.source,
                        "session_id": "tracking-session",
                        "actor_id": "vision-runtime",
                    },
                    {
                        "source": self.source,
                        "session_id": "tracking-session",
                        "actor_id": "vision-runtime",
                    },
                    {
                        "recenter_after_misses": self.recenter_after_misses,
                        "session_id": "tracking-session",
                        "actor_id": "vision-runtime",
                    },
                    {
                        "session_id": "tracking-session",
                        "actor_id": "vision-runtime",
                    },
                ):
                    try:
                        tracking_payload = self.body_runtime.track_visual_target_once(**kwargs)
                        break
                    except TypeError:
                        continue
                if hasattr(self.body_runtime, "_update_visual_tracking_state"):
                    diagnostics = self._tracking_diagnostic_updates(tracking_payload)
                    self.body_runtime._update_visual_tracking_state(
                        source=self.source,
                        running=True,
                        **diagnostics,
                    )
            except Exception as exc:  # pragma: no cover - hardware boundary
                if hasattr(self.body_runtime, "record_runtime_event"):
                    self.body_runtime.record_runtime_event(
                        kind="visual_tracking_error",
                        source="eye.tracking",
                        status="error",
                        details={"error": str(exc)},
                    )
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.interval_s - elapsed))

    @staticmethod
    def _tracking_diagnostic_updates(payload: object | None) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return {}
        raw_diagnostics = payload.get("tracking_diagnostics")
        if not isinstance(raw_diagnostics, Mapping):
            return {}
        allowed_keys = {
            "track_count",
            "active_track_id",
            "switch_count",
            "reacquired_count",
            "lost_count",
            "stability_ratio",
            "suppressed_reason",
        }
        diagnostics = {key: raw_diagnostics[key] for key in allowed_keys if key in raw_diagnostics}
        if not diagnostics:
            return {}
        return {
            "tracking_diagnostics": dict(diagnostics),
            **diagnostics,
        }
