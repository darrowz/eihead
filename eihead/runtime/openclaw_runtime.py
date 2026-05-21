"""OpenClaw realtime runtime for eihead voice capture and playback."""

from __future__ import annotations

from array import array
from collections import deque
from collections.abc import Callable, Mapping
import math
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any

from eihead.eivoice_runtime import (
    AudioFrame,
    EiVoiceRuntimeRunner,
    OpenClawRealtimeTransport,
)
from eihead.eivoice_runtime.native_loop import (
    ArecordAudioFrameSource,
    NativeVoiceLoopConfig,
    SherpaOnnxWindowTranscriber,
    _contains_spoken_phrase,
    _normalize_spoken_phrase,
    _strip_wake_word_prefix,
)


TransportFactory = Callable[[NativeVoiceLoopConfig], OpenClawRealtimeTransport]
TERMINAL_OPENCLAW_SESSION_STATES = {"ended", "closed", "error"}


class AplayPcmPlaybackSink:
    """Persistent raw PCM playback sink for OpenClaw realtime audio frames."""

    def __init__(
        self,
        *,
        device: str = "default",
        active_grace_s: float = 0.4,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.device = str(device or "default")
        self.active_grace_s = float(active_grace_s)
        self._clock = clock or time.monotonic
        self._process: subprocess.Popen[bytes] | None = None
        self._format: tuple[int, int] | None = None
        self._last_error = ""
        self._last_play_at: float | None = None
        self._last_frame_duration_ms: int | None = None
        self._active_until = 0.0

    def play(self, frame: AudioFrame) -> None:
        pcm = frame.pcm or frame.payload
        if not pcm:
            return
        self._record_playback_window(frame)
        self._ensure_process(sample_rate=frame.sample_rate_hz, channels=frame.channels)
        if self._process is None or self._process.stdin is None:
            return
        try:
            self._process.stdin.write(pcm)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._last_error = str(exc)
            self.stop()

    def stop(self) -> None:
        process = self._process
        self._process = None
        self._format = None
        self._active_until = 0.0
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)
        if process.stderr is not None:
            error_text = process.stderr.read().decode("utf-8", errors="replace").strip()
            if error_text:
                self._last_error = error_text

    def status(self) -> dict[str, Any]:
        now = self._clock()
        return {
            "device": self.device,
            "running": self._process is not None and self._process.poll() is None,
            "active": now <= self._active_until,
            "active_until_s": self._active_until,
            "last_play_at_s": self._last_play_at,
            "last_frame_duration_ms": self._last_frame_duration_ms,
            "sample_rate": self._format[0] if self._format else None,
            "channels": self._format[1] if self._format else None,
            "last_error": self._last_error,
        }

    def _record_playback_window(self, frame: AudioFrame) -> None:
        now = self._clock()
        duration_s = max(0.0, float(frame.duration_ms) / 1000.0)
        self._last_play_at = now
        self._last_frame_duration_ms = int(frame.duration_ms)
        self._active_until = max(self._active_until, now + duration_s + max(0.0, self.active_grace_s))

    def _ensure_process(self, *, sample_rate: int, channels: int) -> None:
        audio_format = (int(sample_rate), int(channels))
        if self._process is not None and self._process.poll() is None and self._format == audio_format:
            return
        self.stop()
        self._format = audio_format
        self._process = subprocess.Popen(
            [
                "aplay",
                "-q",
                "-D",
                self.device,
                "-f",
                "S16_LE",
                "-r",
                str(audio_format[0]),
                "-c",
                str(audio_format[1]),
                "-t",
                "raw",
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


class OpenClawPlaybackEchoGate:
    """Mute upstream echo during playback, optionally allowing local barge-in."""

    def __init__(
        self,
        *,
        playback_sink: Any,
        on_barge_in: Callable[[Mapping[str, Any]], None],
        is_output_active: Callable[[], bool] | None = None,
        barge_in_enabled: bool = False,
        rms_threshold: float = 0.13,
        peak_threshold: float = 0.2,
        consecutive_frames: int = 2,
        local_vad_enabled: bool = False,
        local_vad_rms_threshold: float = 0.14,
        local_vad_peak_threshold: float = 0.28,
        local_vad_hangover_frames: int = 5,
        local_vad_max_frames: int = 35,
        local_transcriber: Any | None = None,
        wake_word_required: bool = False,
        wake_words: tuple[str, ...] = ("你好鸿途",),
        end_phrases: tuple[str, ...] = ("结束对话",),
        wake_ack_text: str = "我在。",
        end_ack_text: str = "好的，结束对话。",
        on_gate_event: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.playback_sink = playback_sink
        self.on_barge_in = on_barge_in
        self.is_output_active = is_output_active
        self.barge_in_enabled = bool(barge_in_enabled)
        self.rms_threshold = float(rms_threshold)
        self.peak_threshold = float(peak_threshold)
        self.consecutive_frames = max(1, int(consecutive_frames))
        self.local_vad_enabled = bool(local_vad_enabled)
        self.local_vad_rms_threshold = max(0.0, float(local_vad_rms_threshold))
        self.local_vad_peak_threshold = max(0.0, float(local_vad_peak_threshold))
        self.local_vad_hangover_frames = max(0, int(local_vad_hangover_frames))
        self.local_vad_max_frames = max(1, int(local_vad_max_frames))
        self.local_transcriber = local_transcriber
        self.wake_word_required = bool(wake_word_required)
        self.wake_words = tuple(str(item) for item in wake_words if str(item))
        self.end_phrases = tuple(str(item) for item in end_phrases if str(item))
        self.wake_ack_text = str(wake_ack_text or "")
        self.end_ack_text = str(end_ack_text or "")
        self.on_gate_event = on_gate_event
        self._speech_frames = 0
        self._suppressed_frames = 0
        self._barge_in_count = 0
        self._conversation_active = not self.wake_word_required
        self._local_vad_active = False
        self._local_vad_voice_frames = 0
        self._local_vad_silence_frames = 0
        self._local_vad_passed_frames = 0
        self._local_vad_dropped_frames = 0
        self._local_vad_segment_frames = 0
        self._local_gate_segment_frames: list[AudioFrame] = []
        self._local_gate_replay_frames: deque[AudioFrame] = deque()
        self._local_gate_last_transcript = ""
        self._local_gate_last_reason = ""
        self._local_gate_last_status = "disabled" if not self.wake_word_required else "armed"
        self._local_gate_last_error = ""
        self._local_gate_last_asr_ms: float | None = None
        self._local_gate_dropped_segments = 0
        self._local_gate_wake_detections = 0
        self._local_gate_end_detections = 0
        self._last_rms = 0.0
        self._last_peak = 0.0
        self._last_muted = False
        self._last_output_active = False
        self._last_barge_in: dict[str, Any] | None = None

    def process_capture(
        self,
        frame: AudioFrame,
        *,
        playback_reference: AudioFrame | None = None,
    ) -> AudioFrame | None:
        _ = playback_reference
        rms, peak = _pcm_rms_peak(frame.pcm or frame.payload, channels=frame.channels)
        playback = _mapping(self.playback_sink.status() if hasattr(self.playback_sink, "status") else {})
        remote_output_active = bool(self.is_output_active()) if self.is_output_active is not None else False
        playback_active = bool(playback.get("active") or remote_output_active)
        self._last_output_active = playback_active
        self._last_rms = round(rms, 5)
        self._last_peak = round(peak, 5)
        if not playback_active:
            self._speech_frames = 0
            if self.wake_word_required:
                replay = self._pop_local_gate_replay_frame()
                if replay is not None:
                    return replay
                return self._process_local_wake_gate(frame, rms=rms, peak=peak)
            if self.local_vad_enabled:
                return self._process_local_vad(frame, rms=rms, peak=peak)
            self._last_muted = False
            return frame

        if not self.barge_in_enabled:
            self._speech_frames = 0
            self._reset_local_vad()
            self._local_gate_segment_frames.clear()
            self._local_gate_replay_frames.clear()
            self._suppressed_frames += 1
            self._last_muted = True
            return None

        if rms >= self.rms_threshold and peak >= self.peak_threshold:
            self._speech_frames += 1
        else:
            self._speech_frames = 0

        if self._speech_frames >= self.consecutive_frames:
            self._speech_frames = 0
            self._barge_in_count += 1
            self._last_muted = False
            payload = {
                "reason": "barge-in",
                "rms": self._last_rms,
                "peak": self._last_peak,
                "threshold": self.rms_threshold,
                "peak_threshold": self.peak_threshold,
                "suppressed_frames": self._suppressed_frames,
            }
            self._last_barge_in = dict(payload)
            self.on_barge_in(payload)
            return frame

        self._suppressed_frames += 1
        self._last_muted = True
        return None

    def _process_local_wake_gate(self, frame: AudioFrame, *, rms: float, peak: float) -> AudioFrame | None:
        if self.local_transcriber is None:
            self._local_gate_last_status = "gate_unavailable"
            self._local_gate_last_reason = "missing_local_transcriber"
            self._local_vad_dropped_frames += 1
            self._last_muted = True
            return None
        if self._local_vad_active and self._local_vad_segment_frames >= self.local_vad_max_frames:
            return self._finalize_local_gate_segment(reason="max_segment_frames")

        speech_like = rms >= self.local_vad_rms_threshold and peak >= self.local_vad_peak_threshold
        if speech_like:
            self._local_vad_active = True
            self._local_vad_voice_frames += 1
            self._local_vad_silence_frames = 0
            self._local_vad_segment_frames += 1
            self._local_gate_segment_frames.append(frame)
            if self._conversation_active:
                self._last_muted = True
                return None
            self._local_vad_dropped_frames += 1
            self._last_muted = True
            return None

        if self._local_vad_active and self._local_vad_silence_frames < self.local_vad_hangover_frames:
            self._local_vad_silence_frames += 1
            self._local_vad_segment_frames += 1
            self._local_gate_segment_frames.append(frame)
            if self._conversation_active:
                self._last_muted = True
                return None
            self._local_vad_dropped_frames += 1
            self._last_muted = True
            return None

        if self._local_vad_active:
            return self._finalize_local_gate_segment(reason="end_silence")

        self._local_vad_dropped_frames += 1
        self._last_muted = True
        return None

    def _finalize_local_gate_segment(self, *, reason: str) -> None:
        frames = list(self._local_gate_segment_frames)
        self._local_gate_segment_frames.clear()
        self._reset_local_vad()
        self._last_muted = True
        if not frames:
            return None

        started = time.perf_counter()
        try:
            text = str(self.local_transcriber.transcribe(frames) if self.local_transcriber is not None else "").strip()
            self._local_gate_last_error = ""
        except Exception as exc:  # pragma: no cover - host ASR dependency safeguard
            text = ""
            self._local_gate_last_error = str(exc)
            self._local_gate_last_reason = "local_asr_error"
            self._local_gate_last_status = "local_asr_error"
            self._local_gate_dropped_segments += 1
        self._local_gate_last_asr_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if not text:
            self._local_gate_last_transcript = ""
            if not self._local_gate_last_error:
                self._local_gate_last_reason = "empty_transcript"
                self._local_gate_last_status = "empty_transcript"
                self._local_gate_dropped_segments += 1
            return None

        self._local_gate_last_transcript = text
        if self._conversation_active:
            if _contains_spoken_phrase(text, self.end_phrases):
                self._conversation_active = False
                self._local_gate_replay_frames.clear()
                self._local_gate_end_detections += 1
                self._local_gate_last_reason = "end_phrase"
                self._local_gate_last_status = "conversation_ended"
                self._emit_gate_event(
                    {
                        "type": "end_phrase_detected",
                        "text": text,
                        "reply_text": self.end_ack_text,
                        "asr_ms": self._local_gate_last_asr_ms,
                        "reason": "end_phrase",
                    }
                )
            else:
                if _is_meaningful_active_transcript(text):
                    self._local_gate_replay_frames.extend(frames)
                    self._local_gate_last_reason = "active_utterance_replayed"
                    self._local_gate_last_status = "conversation_active"
                    return self._pop_local_gate_replay_frame()
                self._local_gate_dropped_segments += 1
                self._local_gate_last_reason = "active_transcript_rejected"
                self._local_gate_last_status = "conversation_active_filtered"
            return None

        remainder = _strip_wake_word_prefix(text, self.wake_words)
        if remainder is None:
            self._local_gate_dropped_segments += 1
            self._local_gate_last_reason = "wake_word_required"
            self._local_gate_last_status = "waiting_for_wake_word"
            return None

        self._conversation_active = True
        self._local_gate_wake_detections += 1
        self._local_gate_last_reason = "wake_word_detected"
        self._local_gate_last_status = "wake_word_detected"
        self._emit_gate_event(
            {
                "type": "wake_detected",
                "text": text,
                "remainder": remainder,
                "reply_text": self.wake_ack_text,
                "asr_ms": self._local_gate_last_asr_ms,
                "reason": "wake_word_detected",
            }
        )
        return None

    def _emit_gate_event(self, payload: Mapping[str, Any]) -> None:
        if self.on_gate_event is not None:
            self.on_gate_event(payload)

    def _process_local_vad(self, frame: AudioFrame, *, rms: float, peak: float) -> AudioFrame | None:
        if self._local_vad_active and self._local_vad_segment_frames >= self.local_vad_max_frames:
            self._reset_local_vad()
            self._local_vad_dropped_frames += 1
            self._last_muted = True
            return None
        speech_like = rms >= self.local_vad_rms_threshold and peak >= self.local_vad_peak_threshold
        if speech_like:
            self._local_vad_active = True
            self._local_vad_voice_frames += 1
            self._local_vad_silence_frames = 0
            self._local_vad_segment_frames += 1
            self._local_vad_passed_frames += 1
            self._last_muted = False
            return frame
        if self._local_vad_active and self._local_vad_silence_frames < self.local_vad_hangover_frames:
            self._local_vad_silence_frames += 1
            self._local_vad_segment_frames += 1
            self._local_vad_passed_frames += 1
            self._last_muted = False
            return frame
        self._reset_local_vad()
        self._local_vad_dropped_frames += 1
        self._last_muted = True
        return None

    def _reset_local_vad(self) -> None:
        self._local_vad_active = False
        self._local_vad_voice_frames = 0
        self._local_vad_silence_frames = 0
        self._local_vad_segment_frames = 0

    def _pop_local_gate_replay_frame(self) -> AudioFrame | None:
        if not self._local_gate_replay_frames:
            return None
        frame = self._local_gate_replay_frames.popleft()
        self._local_vad_passed_frames += 1
        self._last_muted = False
        return frame

    def reset_conversation(self, *, reason: str = "reset") -> None:
        self._conversation_active = not self.wake_word_required
        self._local_gate_segment_frames.clear()
        self._local_gate_replay_frames.clear()
        self._reset_local_vad()
        self._last_muted = False
        self._local_gate_last_reason = str(reason or "reset")
        self._local_gate_last_status = "disabled" if not self.wake_word_required else "armed"

    def readiness(self) -> dict[str, Any]:
        vad_state = "disabled"
        if self.local_vad_enabled:
            vad_state = "local_vad_ready"
        elif self.barge_in_enabled:
            vad_state = "barge_in_ready"
        elif self._last_output_active or self._suppressed_frames:
            vad_state = "echo_suppression_only"
        local_gate_state = "disabled"
        if self.wake_word_required:
            local_gate_state = "active" if self._conversation_active else "armed"
            if self.local_transcriber is None:
                local_gate_state = "unavailable"
        return {
            "mode": "openclaw_playback_echo_gate",
            "healthy": True,
            "aec": {"enabled": True, "available": True, "state": "playback_gate"},
            "ns": {"enabled": False, "available": False, "state": "disabled"},
            "vad": {
                "enabled": self.local_vad_enabled or self.barge_in_enabled,
                "available": self.local_vad_enabled or self.barge_in_enabled,
                "state": vad_state,
            },
            "loopback": {"enabled": False, "available": False, "state": "disabled"},
            "warnings": [],
            "localVad": {
                "enabled": self.local_vad_enabled,
                "active": self._local_vad_active,
                "passedFrames": self._local_vad_passed_frames,
                "droppedFrames": self._local_vad_dropped_frames,
                "voiceFrames": self._local_vad_voice_frames,
                "silenceFrames": self._local_vad_silence_frames,
                "segmentFrames": self._local_vad_segment_frames,
                "hangoverFrames": self.local_vad_hangover_frames,
                "maxFrames": self.local_vad_max_frames,
                "rmsThreshold": self.local_vad_rms_threshold,
                "peakThreshold": self.local_vad_peak_threshold,
            },
            "localWakeGate": {
                "enabled": self.wake_word_required,
                "state": local_gate_state,
                "conversationActive": self._conversation_active,
                "wakeWords": list(self.wake_words),
                "endPhrases": list(self.end_phrases),
                "lastTranscript": self._local_gate_last_transcript,
                "lastGateReason": self._local_gate_last_reason,
                "lastStatus": self._local_gate_last_status,
                "lastAsrMs": self._local_gate_last_asr_ms,
                "lastError": self._local_gate_last_error,
                "droppedSegments": self._local_gate_dropped_segments,
                "wakeDetections": self._local_gate_wake_detections,
                "endDetections": self._local_gate_end_detections,
                "segmentFrames": len(self._local_gate_segment_frames),
                "replayFrames": len(self._local_gate_replay_frames),
                "transcriber": self._local_transcriber_status(),
            },
            "playbackGate": {
                "bargeInEnabled": self.barge_in_enabled,
                "outputActive": self._last_output_active,
                "muted": self._last_muted,
                "suppressedFrames": self._suppressed_frames,
                "bargeInCount": self._barge_in_count,
                "speechFrames": self._speech_frames,
                "consecutiveFrames": self.consecutive_frames,
                "rmsThreshold": self.rms_threshold,
                "peakThreshold": self.peak_threshold,
                "lastRms": self._last_rms,
                "lastPeak": self._last_peak,
                "lastBargeIn": dict(self._last_barge_in) if self._last_barge_in else None,
            },
        }

    def _local_transcriber_status(self) -> dict[str, Any]:
        if self.local_transcriber is None:
            return {"provider": None, "state": "missing"}
        status = getattr(self.local_transcriber, "status", None)
        if not callable(status):
            return {"provider": type(self.local_transcriber).__name__, "state": "attached"}
        payload = status()
        return dict(payload) if isinstance(payload, Mapping) else {"state": "unknown"}


class OpenClawRealtimeRuntime:
    """Runtime adapter that bridges honjia microphone/playback to VoiceClaw."""

    OUTPUT_AUDIO_GRACE_S = 4.0

    def __init__(
        self,
        config: NativeVoiceLoopConfig,
        *,
        transport_factory: TransportFactory | None = None,
        capture_source: ArecordAudioFrameSource | None = None,
        playback_sink: AplayPcmPlaybackSink | None = None,
    ) -> None:
        self.config = config
        self.capture_source = capture_source or ArecordAudioFrameSource(config)
        self.playback_sink = playback_sink or AplayPcmPlaybackSink(
            device=config.speaker_device,
            active_grace_s=_playback_grace_s(config),
        )
        self.transport = (transport_factory or _default_transport_factory)(config) if config.openclaw_ws_url else None
        self.audio_frontend = OpenClawPlaybackEchoGate(
            playback_sink=self.playback_sink,
            on_barge_in=self._handle_barge_in,
            is_output_active=self._is_remote_output_active,
            barge_in_enabled=config.openclaw_barge_in_enabled,
            rms_threshold=max(0.08, float(config.vad_rms_threshold)),
            peak_threshold=max(0.18, float(config.vad_rms_threshold) * 1.5),
            consecutive_frames=_barge_in_consecutive_frames(config),
            local_vad_enabled=True,
            local_vad_rms_threshold=max(0.14, float(config.vad_rms_threshold)),
            local_vad_peak_threshold=max(0.28, float(config.vad_rms_threshold) * 1.8),
            local_vad_hangover_frames=_local_vad_hangover_frames(config),
            local_vad_max_frames=_local_vad_max_frames(config),
            local_transcriber=_build_local_wake_transcriber(config),
            wake_word_required=config.wake_word_required,
            wake_words=config.wake_words,
            end_phrases=config.end_phrases,
            wake_ack_text=config.wake_ack_text,
            end_ack_text=config.end_ack_text,
            on_gate_event=self._handle_local_gate_event,
        )
        self.runner = (
            EiVoiceRuntimeRunner(
                capture_source=self.capture_source,
                audio_frontend=self.audio_frontend,
                playback_sink=self.playback_sink,
                transport=self.transport,
                uid=config.dialogue_actor_id,
                mid=config.dialogue_session_id,
            )
            if self.transport is not None
            else None
        )
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._last_error = ""
        self._last_barge_in: dict[str, Any] | None = None
        self._last_local_gate_event: dict[str, Any] | None = None
        self._local_output_active_until = 0.0

    def start(self) -> None:
        if not self.config.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            self._started = True
            self._last_error = ""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="eihead-openclaw-realtime", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.capture_source.stop()
        self.playback_sink.stop()
        if self.transport is not None:
            self.transport.close("runtime_stop")
        with self._lock:
            self._started = False

    def speak(self, text: str) -> dict[str, Any]:
        return {
            "status": "not_supported",
            "success": False,
            "details": {
                "provider": "openclaw_realtime",
                "reason": "manual_speak_is_owned_by_openclaw_live_audio",
                "text_preview": str(text or "")[:60],
            },
        }

    def stop_speech(self) -> dict[str, Any]:
        cleared = self.runner.interrupt_playback(reason="user_interrupt", source="openclaw_runtime") if self.runner else 0
        cancelled = self.transport.cancel_output("user_interrupt") if self.transport is not None else False
        return {
            "status": "stopped",
            "success": True,
            "details": {
                "provider": "openclaw_realtime",
                "cleared": cleared,
                "cancelled_remote_output": cancelled,
            },
        }

    def status(self) -> dict[str, Any]:
        runtime = self.runner.status() if self.runner is not None else _empty_runtime_status()
        openclaw_ws = self._openclaw_ws_status(runtime)
        transport = _mapping(runtime.get("transport")) or self._transport_payload(openclaw_ws)
        running = bool(self._started and openclaw_ws.get("connected") is True)
        runtime.update(
            {
                "schema": "eihead.eivoice_runtime.diagnostics.v1",
                "state": "running" if running else ("error" if self._last_error else "degraded"),
                "conversation_state": openclaw_ws.get("session_state") or transport.get("state") or "unknown",
                "health": "healthy" if running else "degraded",
                "running": running,
                "transport": transport,
                "openclaw_ws": openclaw_ws,
                "mouth": self._mouth_status(running=running),
                "voice_dialogue": self._dialogue_status(running=running, openclaw_ws=openclaw_ws),
                "wakeword": self._wakeword_status(running=running, openclaw_ws=openclaw_ws),
                "realtime_audio": self._realtime_audio_status(running=running, openclaw_ws=openclaw_ws, transport=transport),
            }
        )
        return runtime

    def voice_status(self) -> dict[str, Any]:
        runtime = self.status()
        openclaw_ws = dict(runtime["openclaw_ws"])
        transport = dict(_mapping(runtime["transport"]))
        running = bool(runtime["running"])
        return {
            "status": "ready" if running else "degraded",
            "ear": {
                "status": "listening" if running else "ready",
                "provider": "openclaw_realtime",
                "readiness_message": self._readiness_message(openclaw_ws),
                "capture": {
                    "status": "running" if running else "ready",
                    "details": self.capture_source.status(),
                },
                "asr": {
                    "status": openclaw_ws.get("session_state") or "unknown",
                    "details": {
                        "provider": "openclaw_realtime",
                        "last_user_transcript": openclaw_ws.get("last_user_transcript", ""),
                    },
                },
            },
            "mouth": self._mouth_status(running=running),
            "voice_dialogue": dict(runtime["voice_dialogue"]),
            "realtime_audio": self._realtime_audio_status(running=running, openclaw_ws=openclaw_ws, transport=transport),
            "streaming": {
                "state": transport.get("state"),
                "transport": transport,
                "openclaw_ws": dict(openclaw_ws),
            },
            "openclaw_ws": dict(openclaw_ws),
            "eivoice_runtime": runtime,
            "barge_in": dict(self._last_barge_in) if self._last_barge_in else None,
            "readiness_message": self._readiness_message(openclaw_ws),
        }

    def _run(self) -> None:
        if self.transport is None or self.runner is None:
            with self._lock:
                self._last_error = "OpenClaw realtime ws_url is missing"
            return
        while not self._stop_event.is_set():
            if not self._ensure_connected():
                time.sleep(0.1)
                continue
            try:
                result = self._step_runtime_once()
                if not any(result.values()):
                    time.sleep(0.005)
            except BaseException as exc:
                self._record_runtime_error(exc)
        self.capture_source.stop()

    def _step_runtime_once(self) -> dict[str, bool]:
        if self.runner is None:
            return {}
        if self._is_output_phase():
            return self._step_output_once()
        return self.runner.step_once()

    def _step_output_once(self) -> dict[str, bool]:
        if self.runner is None:
            return {}
        result = {
            "playback": self.runner.step_playback(),
            "decode": self.runner.step_decode(),
            "receive": self.runner.step_receive(),
        }
        if result["receive"]:
            result["decode"] = self.runner.step_decode() or result["decode"]
            result["playback"] = self.runner.step_playback() or result["playback"]
        return result

    def _is_output_phase(self) -> bool:
        playback = _mapping(self.playback_sink.status() if hasattr(self.playback_sink, "status") else {})
        if playback.get("active"):
            return True
        return self._is_remote_output_active()

    def _ensure_connected(self) -> bool:
        if self.transport is None:
            return False
        status = self.transport.status()
        state = str(_mapping(status.get("connection")).get("state") or status.get("state") or "")
        if state == "connected":
            openclaw_ws = _mapping(status.get("openclaw_ws"))
            session_state = str(openclaw_ws.get("session_state") or "").strip().lower()
            if _is_terminal_openclaw_session_state(session_state):
                self._handle_terminal_openclaw_session(session_state)
                return False
            return True
        if state == "reconnect_wait" and not self.transport.ready_to_reconnect():
            return False
        try:
            self.transport.reconnect() if state == "reconnect_wait" else self.transport.connect()
            with self._lock:
                self._last_error = ""
            return True
        except BaseException as exc:
            with self._lock:
                self._last_error = str(exc)
            return False

    def _handle_terminal_openclaw_session(self, session_state: str) -> None:
        reason = f"relay_session_{str(session_state or 'ended').strip().lower() or 'ended'}"
        self.audio_frontend.reset_conversation(reason="openclaw_session_ended")
        if self.transport is None:
            return
        self.transport.close(reason)
        self.transport.schedule_reconnect(reason)

    def _record_runtime_error(self, exc: BaseException) -> None:
        with self._lock:
            self._last_error = str(exc)
        if self.transport is None:
            return
        status = self.transport.status()
        state = str(_mapping(status.get("connection")).get("state") or status.get("state") or "")
        if state != "reconnect_wait":
            self.transport.record_error(exc, context="runtime_loop")
            self.transport.close("runtime_error")
            self.transport.schedule_reconnect("runtime_error")

    def _handle_barge_in(self, payload: Mapping[str, Any]) -> None:
        cancelled = self.transport.cancel_output("barge-in") if self.transport is not None else False
        cleared = self.runner.interrupt_playback(reason="barge-in", source="openclaw_echo_gate") if self.runner else 0
        with self._lock:
            self._last_barge_in = {
                **dict(payload),
                "cancelled_remote_output": cancelled,
                "cleared": cleared,
                "ts": time.time(),
            }

    def _handle_local_gate_event(self, payload: Mapping[str, Any]) -> None:
        event = dict(payload)
        event["ts"] = time.time()
        reply_text = str(event.get("reply_text") or "")
        if reply_text:
            event["playback"] = self._play_local_gate_reply(reply_text)
        with self._lock:
            self._last_local_gate_event = event
        event_type = str(event.get("type") or "")
        if event_type == "end_phrase_detected":
            if self.transport is not None:
                self.transport.cancel_output("local_end_phrase")
            if self.runner is not None:
                self.runner.interrupt_playback(reason="local_end_phrase", source="local_wake_gate")

    def _play_local_gate_reply(self, text: str) -> dict[str, Any]:
        text_value = str(text or "").strip()
        if not text_value:
            return {"status": "skipped", "success": False, "details": {"reason": "empty_text"}}
        model_path = self.config.piper_model_path.strip()
        if not model_path:
            return {"status": "skipped", "success": False, "details": {"reason": "missing_piper_model_path"}}
        started = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)
        command = [self.config.piper_command, "--model", model_path]
        config_path = self.config.piper_config_path.strip()
        if config_path:
            command.extend(["--config", config_path])
        command.extend(["--output_file", str(wav_path)])
        try:
            synth = subprocess.run(
                command,
                input=text_value,
                capture_output=True,
                text=True,
                check=False,
            )
            if synth.returncode != 0:
                return {
                    "status": "error",
                    "success": False,
                    "details": {
                        "backend": "piper",
                        "reason": "synthesis_failed",
                        "returncode": synth.returncode,
                        "stderr": (synth.stderr or "").strip(),
                    },
                }
            playback = subprocess.run(
                ["aplay", "-q", "-D", self.config.speaker_device, str(wav_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            cooldown_s = max(0.0, float(self.config.playback_echo_cooldown_ms) / 1000.0)
            self._local_output_active_until = max(self._local_output_active_until, time.monotonic() + cooldown_s)
            return {
                "status": "ok" if playback.returncode == 0 else "error",
                "success": playback.returncode == 0,
                "details": {
                    "backend": "piper",
                    "command": self.config.piper_command,
                    "model_path": model_path,
                    "config_path": config_path,
                    "device": self.config.speaker_device,
                    "playback_elapsed_ms": elapsed_ms,
                    "returncode": playback.returncode,
                    "stderr": (playback.stderr or "").strip(),
                },
            }
        finally:
            wav_path.unlink(missing_ok=True)

    def _is_remote_output_active(self) -> bool:
        if time.monotonic() <= self._local_output_active_until:
            return True
        if self.transport is None:
            return False
        payload = _mapping(self.transport.status().get("openclaw_ws"))
        if str(payload.get("session_state") or "").lower() != "speaking":
            return False
        return _recent_audio_active(payload, grace_s=self.OUTPUT_AUDIO_GRACE_S)

    def _openclaw_ws_status(self, runtime: Mapping[str, Any]) -> dict[str, Any]:
        transport = _mapping(runtime.get("transport"))
        payload = _mapping(runtime.get("openclaw_ws")) or _mapping(transport.get("openclaw_ws"))
        if payload:
            raw_session_state = str(payload.get("session_state") or "unknown")
            playback = _mapping(self.playback_sink.status() if hasattr(self.playback_sink, "status") else {})
            session_state = raw_session_state
            if (
                raw_session_state.lower() == "speaking"
                and not playback.get("active")
                and not _recent_audio_active(payload, grace_s=self.OUTPUT_AUDIO_GRACE_S)
            ):
                session_state = "ready"
            result = {
                "connected": bool(payload.get("connected")),
                "url": str(payload.get("url") or self.config.openclaw_ws_url),
                "last_error": str(payload.get("last_error") or self._last_error),
                "last_rx_ms": payload.get("last_rx_ms"),
                "last_audio_rx_ms": payload.get("last_audio_rx_ms"),
                "last_tx_ms": payload.get("last_tx_ms"),
                "latency_ms": dict(_mapping(payload.get("latency_ms"))),
                "session_state": session_state,
                "reported_session_state": raw_session_state,
                "session_id": str(payload.get("session_id") or ""),
                "last_user_transcript": str(payload.get("last_user_transcript") or ""),
                "last_assistant_transcript": str(payload.get("last_assistant_transcript") or ""),
            }
            if self._last_error and not result["last_error"]:
                result["last_error"] = self._last_error
            return result
        if not self.config.openclaw_ws_url:
            session_state = "missing_url"
        elif self._last_error:
            session_state = "error"
        elif self._started:
            session_state = "connecting"
        else:
            session_state = "idle"
        return {
            "connected": False,
            "url": self.config.openclaw_ws_url,
            "last_error": self._last_error,
            "last_rx_ms": None,
            "last_audio_rx_ms": None,
            "last_tx_ms": None,
            "latency_ms": {},
            "session_state": session_state,
            "reported_session_state": session_state,
            "session_id": "",
            "last_user_transcript": "",
            "last_assistant_transcript": "",
        }

    def _transport_payload(self, openclaw_ws: Mapping[str, Any]) -> dict[str, Any]:
        state = "connected" if openclaw_ws.get("connected") else ("error" if openclaw_ws.get("last_error") else "disconnected")
        return {
            "transport": "openclaw_realtime",
            "name": "openclaw_realtime",
            "provider": self.config.openclaw_provider,
            "state": state,
            "url": self.config.openclaw_ws_url,
            "connection": {"state": state},
            "heartbeat": {"awaiting_pong": False, "timed_out": False, "latency_ms": 0},
            "reconnect": {"attempt": 0, "backoff_s": 0.0, "ready": self._started, "reason": ""},
            "last_error": {"message": openclaw_ws.get("last_error")} if openclaw_ws.get("last_error") else {},
        }

    def _mouth_status(self, *, running: bool) -> dict[str, Any]:
        return {
            "status": "ready" if running else "idle",
            "backend": "openclaw_realtime",
            "provider": self.config.openclaw_provider,
            "model": self.config.openclaw_model or "openclaw",
            "voice_id": self.config.openclaw_voice,
            "text_preview": "",
            "readiness_message": "OpenClaw realtime audio output is attached",
            "busy": False,
            "playback_state": "idle",
            "tts_playback": {
                "status": "ready" if self.config.speaker_device else "not_wired",
                "details": {
                    "provider": "openclaw_realtime",
                    "device": self.config.speaker_device,
                    "running": running,
                    "playback_sink": self.playback_sink.status(),
                },
            },
        }

    def _dialogue_status(self, *, running: bool, openclaw_ws: Mapping[str, Any]) -> dict[str, Any]:
        stage_latency = _openclaw_stage_latency(openclaw_ws)
        return {
            "enabled": True,
            "running": running,
            "phase": str(openclaw_ws.get("session_state") or "unknown"),
            "last_status": str(openclaw_ws.get("session_state") or "unknown"),
            "last_transcript": str(openclaw_ws.get("last_user_transcript") or ""),
            "last_reply": str(openclaw_ws.get("last_assistant_transcript") or ""),
            "last_stage_latency_ms": stage_latency,
            "last_error": str(openclaw_ws.get("last_error") or ""),
            "conversation_active": running,
            "wake_word_required": self.config.wake_word_required,
            "wake_words": list(self.config.wake_words),
            "end_phrases": list(self.config.end_phrases),
            "turn_count": 0,
            "current_round_id": self.config.dialogue_session_id,
            "dialogue": {
                "provider": "openclaw_realtime",
                "session_id": openclaw_ws.get("session_id") or "",
            },
            "readiness_message": self._readiness_message(openclaw_ws),
        }

    def _wakeword_status(self, *, running: bool, openclaw_ws: Mapping[str, Any]) -> dict[str, Any]:
        if not self.config.wake_word_required:
            state = "disabled"
        elif running:
            state = "active"
        else:
            state = "armed"
        return {
            "enabled": self.config.wake_word_required,
            "state": state,
            "wake_words": list(self.config.wake_words),
            "end_phrases": list(self.config.end_phrases),
            "readiness_message": self._readiness_message(openclaw_ws),
        }

    def _realtime_audio_status(
        self,
        *,
        running: bool,
        openclaw_ws: Mapping[str, Any],
        transport: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "enabled": True,
            "running": running,
            "transport": dict(transport),
            "openclaw_ws": dict(openclaw_ws),
            "latency_ms": dict(_mapping(openclaw_ws.get("latency_ms"))),
            "capture": self.capture_source.status(),
            "barge_in": dict(self._last_barge_in) if self._last_barge_in else None,
        }

    def _readiness_message(self, openclaw_ws: Mapping[str, Any]) -> str:
        if not self.config.openclaw_ws_url:
            return "openclaw realtime provider selected; configure ws_url"
        if openclaw_ws.get("connected") is True:
            return "openclaw realtime provider is connected"
        if openclaw_ws.get("last_error"):
            return f"openclaw realtime provider error: {openclaw_ws['last_error']}"
        return "openclaw realtime provider selected; waiting for connection"


def _default_transport_factory(config: NativeVoiceLoopConfig) -> OpenClawRealtimeTransport:
    return OpenClawRealtimeTransport(
        url=config.openclaw_ws_url,
        token_env_var=config.openclaw_token_env_var,
        protocol=config.openclaw_protocol,
        connect_timeout=config.openclaw_connect_timeout_s,
        receive_timeout=config.openclaw_receive_timeout_s,
        session_ready_timeout=config.openclaw_session_ready_timeout_s,
        session_config=_session_config(config),
    )


def _build_local_wake_transcriber(config: NativeVoiceLoopConfig) -> SherpaOnnxWindowTranscriber | None:
    if not config.wake_word_required or not config.asr_model_dir:
        return None
    return SherpaOnnxWindowTranscriber(
        model_dir=config.asr_model_dir,
        model_type=config.asr_model_type,
        sample_rate=config.sample_rate,
    )


def _session_config(config: NativeVoiceLoopConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sessionKey": config.dialogue_session_id,
        "provider": config.openclaw_provider,
        "voice": config.openclaw_voice,
        "brainAgent": config.openclaw_brain_agent,
        "watchdog": "enabled",
        "audio": {
            "input": {
                "encoding": "pcm16",
                "sampleRateHz": config.sample_rate,
                "channels": config.channels,
            },
            "output": {
                "encoding": "pcm16",
            },
        },
    }
    if config.openclaw_model:
        payload["model"] = config.openclaw_model
    return payload


def _empty_runtime_status() -> dict[str, Any]:
    return {
        "queues": {
            "opus_encode_queue": {"depth": 0, "capacity": 1},
            "ws_send_queue": {"depth": 0, "capacity": 1},
            "opus_decode_queue": {"depth": 0, "capacity": 1},
            "audio_playback_queue": {"depth": 0, "capacity": 1},
        },
        "audio_frontend": {},
        "worker_metrics": {},
    }


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _is_terminal_openclaw_session_state(session_state: str) -> bool:
    return str(session_state or "").strip().lower() in TERMINAL_OPENCLAW_SESSION_STATES


def _is_meaningful_active_transcript(text: str) -> bool:
    normalized = _normalize_spoken_phrase(text)
    if not normalized:
        return False
    if normalized in {"我", "嗯", "啊", "哦", "好", "好的"}:
        return False
    cjk_count = sum(1 for char in normalized if "\u4e00" <= char <= "\u9fff")
    if cjk_count:
        return cjk_count >= 3
    return len(normalized) >= 6


def _recent_audio_active(payload: Mapping[str, Any], *, grace_s: float) -> bool:
    last_audio_rx_ms = _float_or_none(payload.get("last_audio_rx_ms") or payload.get("lastAudioRxMs"))
    if last_audio_rx_ms is None:
        return False
    now_ms = time.monotonic() * 1000.0
    return (now_ms - last_audio_rx_ms) <= max(0.0, float(grace_s)) * 1000.0


def _playback_grace_s(config: NativeVoiceLoopConfig) -> float:
    configured = max(0.0, float(config.playback_echo_cooldown_ms) / 1000.0)
    return max(4.0, configured)


def _openclaw_stage_latency(openclaw_ws: Mapping[str, Any]) -> dict[str, Any]:
    latency = _mapping(openclaw_ws.get("latency_ms"))
    return {
        "asr_to_first_text": latency.get("asr_to_first_text_ms"),
        "asr_to_first_audio": latency.get("asr_to_first_audio_ms"),
        "first_text_to_first_audio": latency.get("first_text_to_first_audio_ms"),
        "audio_receive_span": latency.get("audio_receive_span_ms"),
        "audio_gap_max": latency.get("audio_gap_max_ms"),
        "audio_chunks": latency.get("audio_chunk_count"),
    }


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _barge_in_consecutive_frames(config: NativeVoiceLoopConfig) -> int:
    frame_ms = max(1, int(config.frame_ms))
    min_voice_ms = max(frame_ms * 2, int(config.vad_min_voice_ms))
    return max(2, min(5, math.ceil(min_voice_ms / frame_ms)))


def _local_vad_hangover_frames(config: NativeVoiceLoopConfig) -> int:
    frame_ms = max(1, int(config.frame_ms))
    end_silence_ms = max(frame_ms, int(config.vad_end_silence_ms))
    return max(1, min(10, math.ceil(end_silence_ms / frame_ms)))


def _local_vad_max_frames(config: NativeVoiceLoopConfig) -> int:
    frame_ms = max(1, int(config.frame_ms))
    max_utterance_ms = max(frame_ms, int(config.max_utterance_ms))
    return max(5, min(80, math.ceil(max_utterance_ms / frame_ms)))


def _pcm_rms_peak(pcm: bytes, *, channels: int) -> tuple[float, float]:
    if not pcm:
        return 0.0, 0.0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0, 0.0
    if channels > 1:
        mono = []
        for idx in range(0, len(samples) - (len(samples) % channels), channels):
            mono.append(max(abs(samples[idx + channel]) for channel in range(channels)))
        values = [sample / 32768.0 for sample in mono]
    else:
        values = [sample / 32768.0 for sample in samples]
    if not values:
        return 0.0, 0.0
    peak = max(abs(sample) for sample in values)
    rms = math.sqrt(sum(sample * sample for sample in values) / len(values))
    return rms, peak


__all__ = ["AplayPcmPlaybackSink", "OpenClawPlaybackEchoGate", "OpenClawRealtimeRuntime"]
