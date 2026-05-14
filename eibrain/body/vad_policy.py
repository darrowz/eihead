"""VAD endpoint policy for realtime and quasi-streaming voice capture."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VadFrame:
    rms_level: float
    speech_probability: float | None = None


@dataclass(frozen=True, slots=True)
class VadEndpointDecision:
    is_voice: bool
    should_start: bool
    should_stop: bool
    should_force_decode: bool
    reason: str


@dataclass(slots=True)
class VadEndpointPolicy:
    """Pure endpointing policy that can accept RMS or Silero speech probability."""

    rms_threshold: float = 0.085
    speech_probability_threshold: float = 0.55
    frame_ms: int = 80
    min_voice_ms: int = 160
    end_silence_ms: int = 520
    min_capture_ms: int = 900
    max_capture_ms: int = 4200
    fallback_rms_threshold: float = 0.075

    voice_ms: int = 0
    silence_after_voice_ms: int = 0
    captured_ms: int = 0
    triggered: bool = False
    peak_rms: float = 0.0

    def observe(self, frame: VadFrame) -> VadEndpointDecision:
        frame_ms = max(1, int(self.frame_ms))
        self.captured_ms += frame_ms
        self.peak_rms = max(self.peak_rms, float(frame.rms_level))
        is_voice = self._is_voice(frame)

        if is_voice:
            self.voice_ms += frame_ms
            self.silence_after_voice_ms = 0
        elif self.triggered:
            self.silence_after_voice_ms += frame_ms

        should_start = not self.triggered and self.voice_ms >= self.min_voice_ms
        if should_start:
            self.triggered = True

        should_stop = (
            self.triggered
            and self.captured_ms >= self.min_capture_ms
            and self.silence_after_voice_ms >= self.end_silence_ms
        )
        maxed = self.captured_ms >= self.max_capture_ms
        should_force_decode = maxed and (self.triggered or self.peak_rms >= self.fallback_rms_threshold)

        if should_stop:
            reason = "endpoint_silence"
        elif should_force_decode:
            reason = "max_capture_force_decode"
        elif should_start:
            reason = "speech_started"
        elif is_voice:
            reason = "speech_candidate"
        else:
            reason = "silence"

        return VadEndpointDecision(
            is_voice=is_voice,
            should_start=should_start,
            should_stop=should_stop,
            should_force_decode=should_force_decode,
            reason=reason,
        )

    def reset(self) -> None:
        self.voice_ms = 0
        self.silence_after_voice_ms = 0
        self.captured_ms = 0
        self.triggered = False
        self.peak_rms = 0.0

    def _is_voice(self, frame: VadFrame) -> bool:
        if frame.speech_probability is not None:
            return float(frame.speech_probability) >= self.speech_probability_threshold
        return float(frame.rms_level) >= self.rms_threshold
