"""OpenClaw realtime runtime for eihead voice capture and playback."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import subprocess
import threading
import time
from typing import Any

from eihead.eivoice_runtime import (
    AudioFrame,
    EiVoiceRuntimeRunner,
    NoOpAcousticFrontend,
    OpenClawRealtimeTransport,
)
from eihead.eivoice_runtime.native_loop import ArecordAudioFrameSource, NativeVoiceLoopConfig


TransportFactory = Callable[[NativeVoiceLoopConfig], OpenClawRealtimeTransport]


class AplayPcmPlaybackSink:
    """Persistent raw PCM playback sink for OpenClaw realtime audio frames."""

    def __init__(self, *, device: str = "default") -> None:
        self.device = str(device or "default")
        self._process: subprocess.Popen[bytes] | None = None
        self._format: tuple[int, int] | None = None
        self._last_error = ""

    def play(self, frame: AudioFrame) -> None:
        pcm = frame.pcm or frame.payload
        if not pcm:
            return
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
        return {
            "device": self.device,
            "running": self._process is not None and self._process.poll() is None,
            "sample_rate": self._format[0] if self._format else None,
            "channels": self._format[1] if self._format else None,
            "last_error": self._last_error,
        }

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


class OpenClawRealtimeRuntime:
    """Runtime adapter that bridges honjia microphone/playback to VoiceClaw."""

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
        self.playback_sink = playback_sink or AplayPcmPlaybackSink(device=config.speaker_device)
        self.transport = (transport_factory or _default_transport_factory)(config) if config.openclaw_ws_url else None
        self.runner = (
            EiVoiceRuntimeRunner(
                capture_source=self.capture_source,
                audio_frontend=NoOpAcousticFrontend(),
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
        return {
            "status": "stopped",
            "success": True,
            "details": {
                "provider": "openclaw_realtime",
                "cleared": cleared,
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
                result = self.runner.step_once()
                if not any(result.values()):
                    time.sleep(0.005)
            except BaseException as exc:
                self._record_runtime_error(exc)
        self.capture_source.stop()

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

    def _openclaw_ws_status(self, runtime: Mapping[str, Any]) -> dict[str, Any]:
        transport = _mapping(runtime.get("transport"))
        payload = _mapping(runtime.get("openclaw_ws")) or _mapping(transport.get("openclaw_ws"))
        if payload:
            result = {
                "connected": bool(payload.get("connected")),
                "url": str(payload.get("url") or self.config.openclaw_ws_url),
                "last_error": str(payload.get("last_error") or self._last_error),
                "last_rx_ms": payload.get("last_rx_ms"),
                "last_tx_ms": payload.get("last_tx_ms"),
                "session_state": str(payload.get("session_state") or "unknown"),
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
            "last_tx_ms": None,
            "session_state": session_state,
            "session_id": "",
            "last_user_transcript": "",
            "last_assistant_transcript": "",
        }

    def _transport_payload(self, openclaw_ws: Mapping[str, Any]) -> dict[str, Any]:
        state = "connected" if openclaw_ws.get("connected") else ("error" if openclaw_ws.get("last_error") else "disconnected")
        return {
            "transport": "openclaw_realtime",
            "name": "openclaw_realtime",
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
        return {
            "enabled": True,
            "running": running,
            "phase": str(openclaw_ws.get("session_state") or "unknown"),
            "last_status": str(openclaw_ws.get("session_state") or "unknown"),
            "last_transcript": str(openclaw_ws.get("last_user_transcript") or ""),
            "last_reply": str(openclaw_ws.get("last_assistant_transcript") or ""),
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
            "capture": self.capture_source.status(),
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


__all__ = ["AplayPcmPlaybackSink", "OpenClawRealtimeRuntime"]
