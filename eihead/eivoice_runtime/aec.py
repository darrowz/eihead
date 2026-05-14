from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import time
from typing import Any

from .core import AudioFrame


DEFAULT_LOOPBACK_REFERENCE_MAX_AGE_MS = 240


@dataclass(frozen=True)
class LoopbackReferenceMatch:
    frame: AudioFrame
    age_ms: float
    matched_by: str


class LoopbackReferenceBuffer:
    """Small in-memory playback reference buffer for hardware-free AEC tests."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_ms: int = 60,
        max_age_ms: int = 240,
        capacity_ms: int = 2_000,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if max_age_ms < 0:
            raise ValueError("max_age_ms must be non-negative")
        if capacity_ms <= 0:
            raise ValueError("capacity_ms must be positive")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.max_age_ms = max_age_ms
        self.capacity_ms = capacity_ms
        self._max_frames = max(1, (capacity_ms + frame_ms - 1) // frame_ms)
        self._frames: deque[AudioFrame] = deque()

    @property
    def depth(self) -> int:
        return len(self._frames)

    def write_playback(
        self,
        reference: AudioFrame | bytes,
        *,
        sequence: int = 0,
        created_at_ts: float | None = None,
        duration_ms: int | None = None,
        sample_rate: int | None = None,
        channels: int = 1,
    ) -> AudioFrame:
        frame = (
            reference
            if isinstance(reference, AudioFrame)
            else AudioFrame(
                pcm=bytes(reference),
                duration_ms=duration_ms or self.frame_ms,
                sample_rate_hz=sample_rate or self.sample_rate,
                channels=channels,
                sequence=sequence,
                created_at_ts=time() if created_at_ts is None else created_at_ts,
            )
        )
        self._frames.append(frame)
        while len(self._frames) > self._max_frames:
            self._frames.popleft()
        return frame

    def reference_for_capture(self, capture: AudioFrame) -> LoopbackReferenceMatch | None:
        sequence_match = self._match_by_sequence(capture)
        if sequence_match is not None:
            return sequence_match
        return self._match_by_time(capture)

    def _match_by_sequence(self, capture: AudioFrame) -> LoopbackReferenceMatch | None:
        for reference in reversed(self._frames):
            if reference.sequence != capture.sequence:
                continue
            match = self._match(capture, reference, matched_by="sequence")
            if match is not None:
                return match
        return None

    def _match_by_time(self, capture: AudioFrame) -> LoopbackReferenceMatch | None:
        candidates = [
            match
            for reference in self._frames
            if (match := self._match(capture, reference, matched_by="time")) is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda match: match.age_ms)

    def _match(
        self,
        capture: AudioFrame,
        reference: AudioFrame,
        *,
        matched_by: str,
    ) -> LoopbackReferenceMatch | None:
        age_ms = _reference_age_ms(capture, reference)
        if age_ms < 0 or age_ms > self.max_age_ms:
            return None
        return LoopbackReferenceMatch(frame=reference, age_ms=age_ms, matched_by=matched_by)


@dataclass(frozen=True)
class AcousticFrontendConfig:
    """Configurable acoustic front-end contract with explicit fallback status."""

    aec_enabled: bool = False
    aec_available: bool = False
    ns_enabled: bool = False
    ns_available: bool = False
    vad_enabled: bool = False
    vad_available: bool = False
    loopback_enabled: bool = False
    loopback_available: bool = False
    capture_enabled: bool = True
    capture_available: bool = True
    mode: str = "noop"
    warnings: tuple[str, ...] = ()
    capture_device: str | None = None
    playback_device: str | None = None
    loopback_device: str | None = None
    sample_rate: int = 16_000
    frame_ms: int = 60
    channels: int = 1
    aec_backend: str = "none"

    def diagnostics(
        self,
        *,
        processed_frames: int = 0,
        dropped_frames: int = 0,
        last_frame_duration_ms: int | None = None,
    ) -> dict[str, Any]:
        warnings = list(self.warnings)
        warnings.extend(_frontend_component_warnings(self))
        return {
            "mode": self.mode,
            "capture": _frontend_component(
                enabled=self.capture_enabled,
                available=self.capture_available,
            ),
            "aec": _frontend_component(
                enabled=self.aec_enabled,
                available=self.aec_available,
            ),
            "ns": _frontend_component(
                enabled=self.ns_enabled,
                available=self.ns_available,
            ),
            "vad": _frontend_component(
                enabled=self.vad_enabled,
                available=self.vad_available,
            ),
            "loopback": _frontend_component(
                enabled=self.loopback_enabled,
                available=self.loopback_available,
            ),
            "devices": {
                "capture": self.capture_device,
                "playback": self.playback_device,
                "loopback": self.loopback_device,
            },
            "audio_format": {
                "sample_rate": self.sample_rate,
                "frame_ms": self.frame_ms,
                "channels": self.channels,
            },
            "aec_backend": self.aec_backend,
            "aec_status": _aec_status(self),
            "healthy": self._healthy,
            "processed_frames": processed_frames,
            "dropped_frames": dropped_frames,
            "last_frame_duration_ms": last_frame_duration_ms,
            "warnings": list(dict.fromkeys(warnings)),
        }

    @property
    def _healthy(self) -> bool:
        return all(
            (
                self.capture_available if self.capture_enabled else True,
                self.aec_available if self.aec_enabled else True,
                self.ns_available if self.ns_enabled else True,
                self.vad_available if self.vad_enabled else True,
                self.loopback_available if self.loopback_enabled else True,
            )
        )


@dataclass(frozen=True)
class ProcessedCaptureFrame:
    frame: AudioFrame
    diagnostics: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.frame, name)


@dataclass(frozen=True)
class LoopbackReferenceHealth:
    ready: bool
    state: str
    reason: str
    reference_age_ms: float | None
    matched_by: str | None
    max_age_ms: int
    device: str | None
    device_ready: bool
    device_state: str
    dry_run: bool
    aec_status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "state": self.state,
            "reason": self.reason,
            "reference_age_ms": self.reference_age_ms,
            "matched_by": self.matched_by,
            "max_age_ms": self.max_age_ms,
            "device": self.device,
            "device_ready": self.device_ready,
            "device_state": self.device_state,
            "dry_run": self.dry_run,
            "aec_status": self.aec_status,
        }


def build_loopback_reference_diagnostics(
    config: AcousticFrontendConfig,
    reference_match: LoopbackReferenceMatch | None,
    *,
    max_age_ms: int = DEFAULT_LOOPBACK_REFERENCE_MAX_AGE_MS,
) -> dict[str, Any]:
    age_ms = reference_match.age_ms if reference_match is not None else None
    matched_by = reference_match.matched_by if reference_match is not None else None
    aec_status = _aec_status(config)
    dry_run = _is_dry_run_config(config, aec_status=aec_status)
    device_ready = _loopback_device_ready(config)

    if config.aec_enabled and not config.aec_available:
        state = "aec_unavailable"
        reason = "aec_unavailable"
    elif not config.aec_enabled or not config.loopback_enabled:
        state = "passthrough"
        reason = "passthrough_frontend"
    elif not config.loopback_available:
        state = "missing"
        reason = "loopback_unavailable"
    elif reference_match is None:
        state = "missing"
        reason = "missing_playback_reference"
    elif age_ms is not None and age_ms > max_age_ms:
        state = "stale"
        reason = "stale_playback_reference"
    else:
        state = "ready"
        reason = "loopback_reference_ready"

    return LoopbackReferenceHealth(
        ready=state == "ready",
        state=state,
        reason=reason,
        reference_age_ms=age_ms,
        matched_by=matched_by,
        max_age_ms=max_age_ms,
        device=config.loopback_device,
        device_ready=device_ready,
        device_state=_loopback_device_state(config, device_ready=device_ready),
        dry_run=dry_run,
        aec_status=aec_status,
    ).as_dict()


class NoOpAcousticFrontend:
    """Pass-through acoustic front-end that never claims real AEC was applied."""

    def __init__(
        self,
        config: AcousticFrontendConfig | None = None,
        *,
        loopback_buffer: LoopbackReferenceBuffer | None = None,
    ) -> None:
        self.config = config or AcousticFrontendConfig()
        self.loopback_buffer = loopback_buffer
        self.processed_frames = 0
        self.dropped_frames = 0
        self.last_frame_duration_ms: int | None = None
        self.last_capture_diagnostics: dict[str, Any] | None = None

    def process_capture(
        self,
        frame: AudioFrame,
        *,
        playback_reference: AudioFrame | LoopbackReferenceMatch | None = None,
    ) -> ProcessedCaptureFrame:
        self.processed_frames += 1
        self.last_frame_duration_ms = frame.duration_ms
        reference_match = self._reference_match(frame, playback_reference)
        diagnostics = self._capture_diagnostics(frame, reference_match)
        self.last_capture_diagnostics = diagnostics
        return ProcessedCaptureFrame(frame=frame, diagnostics=diagnostics)

    def readiness(self) -> dict[str, Any]:
        diagnostics = self.config.diagnostics(
            processed_frames=self.processed_frames,
            dropped_frames=self.dropped_frames,
            last_frame_duration_ms=self.last_frame_duration_ms,
        )
        diagnostics["last_capture"] = (
            dict(self.last_capture_diagnostics) if self.last_capture_diagnostics is not None else None
        )
        return diagnostics

    def _reference_match(
        self,
        frame: AudioFrame,
        playback_reference: AudioFrame | LoopbackReferenceMatch | None,
    ) -> LoopbackReferenceMatch | None:
        if isinstance(playback_reference, LoopbackReferenceMatch):
            return playback_reference
        if isinstance(playback_reference, AudioFrame):
            return LoopbackReferenceMatch(
                frame=playback_reference,
                age_ms=_reference_age_ms(frame, playback_reference),
                matched_by="explicit",
            )
        if self.loopback_buffer is not None:
            return self.loopback_buffer.reference_for_capture(frame)
        return None

    def _capture_diagnostics(
        self,
        frame: AudioFrame,
        reference_match: LoopbackReferenceMatch | None,
    ) -> dict[str, Any]:
        reference = reference_match.frame if reference_match is not None else None
        max_age_ms = (
            self.loopback_buffer.max_age_ms
            if self.loopback_buffer is not None
            else DEFAULT_LOOPBACK_REFERENCE_MAX_AGE_MS
        )
        loopback_reference = build_loopback_reference_diagnostics(
            self.config,
            reference_match,
            max_age_ms=max_age_ms,
        )
        return {
            "mode": self.config.mode,
            "aec_backend": self.config.aec_backend,
            "aec_status": _aec_status(self.config),
            "aec_applied": False,
            "fallback_reason": _fallback_reason(self.config, reference, loopback_reference),
            "loopback_reference": loopback_reference,
            "loopback_reference_ready": loopback_reference["ready"],
            "loopback_reference_state": loopback_reference["state"],
            "loopback_reference_reason": loopback_reference["reason"],
            "loopback_reference_max_age_ms": loopback_reference["max_age_ms"],
            "reference_age_ms": reference_match.age_ms if reference_match is not None else None,
            "reference_sequence": reference.sequence if reference is not None else None,
            "reference_matched_by": reference_match.matched_by if reference_match is not None else None,
            "playback_reference_available": reference is not None,
            "capture_sequence": frame.sequence,
            "capture_duration_ms": frame.duration_ms,
        }


def _aec_status(config: AcousticFrontendConfig) -> str:
    if not config.aec_enabled:
        return "disabled"
    if not config.aec_available:
        return "unavailable"
    return "passthrough"


def _is_dry_run_config(
    config: AcousticFrontendConfig,
    *,
    aec_status: str | None = None,
) -> bool:
    status = aec_status or _aec_status(config)
    return str(config.mode or "").lower() in {"noop", "passthrough"} or status == "passthrough"


def _loopback_device_ready(config: AcousticFrontendConfig) -> bool:
    if not config.loopback_enabled:
        return True
    return bool(config.loopback_device)


def _loopback_device_state(
    config: AcousticFrontendConfig,
    *,
    device_ready: bool | None = None,
) -> str:
    if not config.loopback_enabled:
        return "not_required"
    if device_ready if device_ready is not None else _loopback_device_ready(config):
        return "ready"
    return "missing"


def _fallback_reason(
    config: AcousticFrontendConfig,
    reference: AudioFrame | None,
    loopback_reference: dict[str, Any] | None = None,
) -> str | None:
    if not config.aec_enabled:
        return None
    if not config.aec_available:
        return "aec_unavailable"
    if config.loopback_enabled and reference is None:
        return "missing_playback_reference"
    if loopback_reference is not None and loopback_reference["state"] == "stale":
        return "stale_playback_reference"
    return "passthrough_frontend"


def _reference_age_ms(capture: AudioFrame, reference: AudioFrame) -> float:
    return (capture.created_at_ts - reference.created_at_ts) * 1_000.0


def _frontend_component(*, enabled: bool, available: bool) -> dict[str, bool | str]:
    enabled_b = bool(enabled)
    available_b = bool(available)
    if enabled_b and available_b:
        state = "ready"
    elif enabled_b:
        state = "unavailable"
    else:
        state = "disabled"
    return {
        "enabled": enabled_b,
        "available": available_b,
        "state": state,
    }


def _frontend_component_warnings(config: AcousticFrontendConfig) -> list[str]:
    warnings: list[str] = []
    for label, enabled, available in (
        ("AEC", config.aec_enabled, config.aec_available),
        ("NS", config.ns_enabled, config.ns_available),
        ("VAD", config.vad_enabled, config.vad_available),
        ("loopback", config.loopback_enabled, config.loopback_available),
        ("capture", config.capture_enabled, config.capture_available),
    ):
        if enabled and not available:
            warnings.append(f"{label} configured but unavailable")
    return warnings
