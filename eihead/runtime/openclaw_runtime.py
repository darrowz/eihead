"""OpenClaw realtime runtime for eihead voice capture and playback."""

from __future__ import annotations

from array import array
from collections.abc import Callable, Mapping
import math
import subprocess
import threading
import time
from typing import Any

from eihead.eivoice_runtime import (
    AudioFrame,
    EiVoiceRuntimeRunner,
    OpenClawRealtimeTransport,
)
from eihead.eivoice_runtime.native_loop import ArecordAudioFrameSource, NativeVoiceLoopConfig


TransportFactory = Callable[[NativeVoiceLoopConfig], OpenClawRealtimeTransport]


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
    ) -> None:
        self.playback_sink = playback_sink
        self.on_barge_in = on_barge_in
        self.is_output_active = is_output_active
        self.barge_in_enabled = bool(barge_in_enabled)
        self.rms_threshold = float(rms_threshold)
        self.peak_threshold = float(peak_threshold)
        self.consecutive_frames = max(1, int(consecutive_frames))
        self._speech_frames = 0
        self._suppressed_frames = 0
        self._barge_in_count = 0
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
            self._last_muted = False
            return frame

        if not self.barge_in_enabled:
            self._speech_frames = 0
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

    def readiness(self) -> dict[str, Any]:
        return {
            "mode": "openclaw_playback_echo_gate",
            "healthy": True,
            "aec": {"enabled": True, "available": True, "state": "playback_gate"},
            "ns": {"enabled": False, "available": False, "state": "disabled"},
            "vad": {
                "enabled": self.barge_in_enabled,
                "available": self.barge_in_enabled,
                "state": "barge_in_ready" if self.barge_in_enabled else "echo_suppression_only",
            },
            "loopback": {"enabled": False, "available": False, "state": "disabled"},
            "warnings": [],
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

    def _is_remote_output_active(self) -> bool:
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
