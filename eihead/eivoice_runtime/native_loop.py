"""Native honjia voice interaction loop.

This loop is intentionally self-contained in eihead. It captures PCM from ALSA,
uses sherpa-onnx for utterance transcription, and plays a short audible reply.
It reports explicit diagnostics instead of presenting the loop as full cloud
LLM/TTS streaming.
"""

from __future__ import annotations

from array import array
from collections import deque
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Mapping
from urllib import error as urlerror, request as urlrequest

from .core import AudioFrame


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class NativeVoiceLoopConfig:
    enabled: bool = True
    transport_provider: str = "legacy_native"
    fallback_transport_provider: str = "legacy_native"
    openclaw_ws_url: str = ""
    openclaw_token_env_var: str = "OPENCLAW_REALTIME_TOKEN"
    openclaw_provider: str = "openai"
    openclaw_model: str = ""
    openclaw_voice: str = "Zephyr"
    openclaw_brain_agent: str = "enabled"
    openclaw_protocol: str = ""
    openclaw_connect_timeout_s: float = 10.0
    openclaw_receive_timeout_s: float = 0.02
    openclaw_session_ready_timeout_s: float = 15.0
    microphone_device: str = "default"
    speaker_device: str = "default"
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 120
    vad_rms_threshold: float = 0.075
    vad_min_voice_ms: int = 240
    vad_end_silence_ms: int = 600
    max_utterance_ms: int = 4200
    asr_model_dir: str = ""
    asr_model_type: str = "lstm"
    reply_template: str = "我听到了：{text}"
    wake_word_required: bool = False
    wake_words: tuple[str, ...] = ("你好鸿途",)
    end_phrases: tuple[str, ...] = ("结束对话",)
    wake_ack_text: str = "我在。"
    end_ack_text: str = "好的，结束对话。"
    playback_backend: str = "aplay"
    tts_backend: str = "piper"
    tts_fallback_provider: str = ""
    piper_command: str = "piper"
    piper_model_path: str = ""
    piper_config_path: str = ""
    playback_echo_cooldown_ms: int = 350
    minimax_api_key: str = ""
    minimax_api_base_url: str = "https://api.minimaxi.com"
    minimax_model: str = "speech-2.8-hd"
    minimax_voice_id: str = "female-shaonv"
    minimax_audio_format: str = "wav"
    minimax_sample_rate: int = 32000
    minimax_bitrate: int = 128000
    minimax_channel: int = 1
    minimax_speed: float = 1.0
    minimax_volume: float = 1.0
    minimax_pitch: float = 0.0
    minimax_language_boost: str = "auto"
    minimax_timeout_s: float = 30.0
    dialogue_backend: str = "template"
    dialogue_command: str = ""
    dialogue_module: str = "apps.cognitive_runtime"
    dialogue_cwd: str = ""
    dialogue_config_path: str = ""
    dialogue_pythonpath: str = ""
    dialogue_timeout_s: float = 12.0
    dialogue_session_id: str = "honjia-voice"
    dialogue_actor_id: str = "darrow"
    dialogue_head_instance_id: str = "honjia"
    dialogue_brain_instance_id: str = "honxin"


@dataclass(slots=True)
class _LoopState:
    state: str = "created"
    phase: str = "not_started"
    health: str = "waiting"
    running: bool = False
    started_at_ts: float | None = None
    updated_at_ts: float | None = None
    frame_count: int = 0
    utterance_count: int = 0
    turn_count: int = 0
    audio_level: float = 0.0
    rms_dbfs: float = -120.0
    vad_triggered: bool = False
    voice_ms: int = 0
    silence_after_voice_ms: int = 0
    captured_ms: int = 0
    last_transcript: str = ""
    last_reply: str = ""
    last_error: str = ""
    last_status: str = "not_started"
    conversation_active: bool = False
    last_gate_reason: str = ""
    last_stage_latency_ms: dict[str, float] = field(default_factory=dict)
    last_dialogue_details: dict[str, Any] = field(default_factory=dict)
    current_round_id: str = ""


class ArecordAudioFrameSource:
    def __init__(self, config: NativeVoiceLoopConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._sequence = 0
        self._last_error = ""

    @property
    def frame_bytes(self) -> int:
        bytes_per_sample = 2
        alignment = max(1, self.config.channels * bytes_per_sample)
        size = int(self.config.sample_rate * self.config.channels * bytes_per_sample * self.config.frame_ms / 1000)
        return max(alignment, size - (size % alignment))

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            [
                "arecord",
                "-q",
                "-D",
                self.config.microphone_device,
                "-f",
                "S16_LE",
                "-r",
                str(self.config.sample_rate),
                "-c",
                str(self.config.channels),
                "-t",
                "raw",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)
        if process.stderr is not None:
            self._last_error = process.stderr.read().decode("utf-8", errors="replace").strip()

    def read_frame(self) -> AudioFrame | None:
        self.start()
        process = self._process
        if process is None or process.stdout is None:
            return None
        payload = process.stdout.read(self.frame_bytes)
        if not payload:
            if process.poll() is not None and process.stderr is not None:
                self._last_error = process.stderr.read().decode("utf-8", errors="replace").strip()
                self._process = None
            return None
        self._sequence += 1
        return AudioFrame(
            pcm=payload,
            duration_ms=self.config.frame_ms,
            sample_rate_hz=self.config.sample_rate,
            channels=self.config.channels,
            sequence=self._sequence,
            created_at_ts=time.time(),
        )

    def status(self) -> dict[str, Any]:
        return {
            "device": self.config.microphone_device,
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
            "frame_ms": self.config.frame_ms,
            "frame_bytes": self.frame_bytes,
            "running": self._process is not None and self._process.poll() is None,
            "last_error": self._last_error,
        }


class SherpaOnnxWindowTranscriber:
    def __init__(self, *, model_dir: str, model_type: str = "lstm", sample_rate: int = 16000) -> None:
        self.model_dir = str(model_dir)
        self.model_type = str(model_type or "lstm")
        self.sample_rate = int(sample_rate)
        self._recognizer: Any | None = None
        self._load_error = ""
        self._last_decode_ms: float | None = None

    def transcribe(self, frames: list[AudioFrame]) -> str:
        started = time.perf_counter()
        recognizer = self._get_recognizer()
        stream = recognizer.create_stream()
        for frame in frames:
            samples = _pcm_to_float_samples(frame.pcm, channels=frame.channels)
            if not samples:
                continue
            stream.accept_waveform(sample_rate=self.sample_rate, waveform=_waveform_buffer(samples))
            while hasattr(recognizer, "is_ready") and recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
        tail_padding = [0.0] * int(self.sample_rate * 0.6)
        stream.accept_waveform(sample_rate=self.sample_rate, waveform=_waveform_buffer(tail_padding))
        if hasattr(stream, "input_finished"):
            stream.input_finished()
        while hasattr(recognizer, "is_ready") and recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        result = recognizer.get_result(stream) if hasattr(recognizer, "get_result") else getattr(stream, "result", None)
        self._last_decode_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if isinstance(result, str):
            return result.strip()
        return str(getattr(result, "text", "") or "").strip()

    def status(self) -> dict[str, Any]:
        return {
            "provider": "sherpa_onnx",
            "state": "ready" if self._recognizer is not None else ("error" if self._load_error else "not_loaded"),
            "model_dir": self.model_dir,
            "model_type": self.model_type,
            "last_decode_ms": self._last_decode_ms,
            "last_error": self._load_error,
        }

    def _get_recognizer(self) -> Any:
        if self._recognizer is not None:
            return self._recognizer
        try:
            import sherpa_onnx

            model_dir = Path(self.model_dir).expanduser()
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(model_dir / "tokens.txt"),
                encoder=str(model_dir / "encoder.onnx"),
                decoder=str(model_dir / "decoder.onnx"),
                joiner=str(model_dir / "joiner.onnx"),
                sample_rate=self.sample_rate,
                model_type=self.model_type,
            )
            self._load_error = ""
            return self._recognizer
        except Exception as exc:
            self._load_error = str(exc)
            raise


class MiniMaxRestTtsSynthesizer:
    def __init__(self, config: NativeVoiceLoopConfig, *, urlopen: Any | None = None) -> None:
        self.config = config
        self.urlopen = urlopen or urlrequest.urlopen

    def synthesize(self, text: str) -> dict[str, Any]:
        text_value = str(text or "").strip()
        if not text_value:
            return {"status": "error", "details": {"backend": "minimax", "reason": "empty_text"}}
        if not self.config.minimax_api_key:
            return {"status": "error", "details": {"backend": "minimax", "reason": "missing_minimax_api_key"}}
        payload = {
            "model": self.config.minimax_model,
            "text": text_value,
            "stream": False,
            "voice_setting": {
                "voice_id": self.config.minimax_voice_id,
                "speed": _normalize_minimax_number(self.config.minimax_speed),
                "vol": _normalize_minimax_number(self.config.minimax_volume),
                "pitch": _normalize_minimax_number(self.config.minimax_pitch),
            },
            "audio_setting": {
                "sample_rate": int(self.config.minimax_sample_rate),
                "bitrate": int(self.config.minimax_bitrate),
                "format": self.config.minimax_audio_format.lower(),
                "channel": int(self.config.minimax_channel),
            },
            "language_boost": self.config.minimax_language_boost,
            "subtitle_enable": False,
        }
        endpoint = _minimax_t2a_url(self.config.minimax_api_base_url)
        req = urlrequest.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.minimax_api_key}",
            },
        )
        try:
            with self.urlopen(req, timeout=float(self.config.minimax_timeout_s)) as response:
                body = response.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            error_body = exc.read().decode("utf-8", "replace")
            return {
                "status": "error",
                "details": {
                    "backend": "minimax",
                    "reason": "http_error",
                    "endpoint": endpoint,
                    "status_code": exc.code,
                    "body": _redact_text(error_body, self.config.minimax_api_key),
                },
            }
        except (urlerror.URLError, OSError) as exc:
            return {
                "status": "error",
                "details": {
                    "backend": "minimax",
                    "reason": "request_failed",
                    "endpoint": endpoint,
                    "error": _redact_text(str(exc), self.config.minimax_api_key),
                },
            }

        try:
            parsed = json.loads(body)
            base_resp = dict(parsed.get("base_resp", {}))
            status_code = int(base_resp.get("status_code", -1))
            if status_code != 0:
                return {
                    "status": "error",
                    "details": {
                        "backend": "minimax",
                        "reason": "api_error",
                        "endpoint": endpoint,
                        "status_code": status_code,
                        "status_msg": base_resp.get("status_msg", ""),
                        "body": _redact_text(body, self.config.minimax_api_key),
                    },
                }
            data = dict(parsed.get("data", {}))
            audio_hex = str(data.get("audio", "") or "")
            if not audio_hex:
                return {
                    "status": "error",
                    "details": {
                        "backend": "minimax",
                        "reason": "missing_audio",
                        "endpoint": endpoint,
                    },
                }
            audio_bytes = bytes.fromhex(audio_hex)
        except (ValueError, TypeError, KeyError) as exc:
            return {
                "status": "error",
                "details": {
                    "backend": "minimax",
                    "reason": "invalid_response",
                    "endpoint": endpoint,
                    "error": _redact_text(str(exc), self.config.minimax_api_key),
                },
            }
        extra_info = parsed.get("extra_info", {})
        if not isinstance(extra_info, Mapping):
            extra_info = {}
        return {
            "status": "ok",
            "audio_bytes": audio_bytes,
            "details": {
                "backend": "minimax",
                "endpoint": endpoint,
                "model": self.config.minimax_model,
                "voice_id": self.config.minimax_voice_id,
                "trace_id": parsed.get("trace_id"),
                "audio_size": extra_info.get("audio_size", len(audio_bytes)),
                "audio_length": extra_info.get("audio_length"),
                "audio_sample_rate": extra_info.get("audio_sample_rate", self.config.minimax_sample_rate),
                "audio_format": extra_info.get("audio_format", self.config.minimax_audio_format.lower()),
                "audio_channel": extra_info.get("audio_channel", self.config.minimax_channel),
                "usage_characters": extra_info.get("usage_characters"),
            },
        }


class TemplateDialogueClient:
    def __init__(self, config: NativeVoiceLoopConfig) -> None:
        self.config = config

    def reply_to_transcript(self, text: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "reply_text": self.config.reply_template.format(text=text),
            "details": {
                "provider": "template",
                "round_id": str(kwargs.get("round_id") or ""),
            },
        }


class EIBrainSubprocessDialogueClient:
    def __init__(
        self,
        config: NativeVoiceLoopConfig,
        *,
        runner: Runner | None = None,
    ) -> None:
        self.config = config
        self.runner = runner or subprocess.run

    def reply_to_transcript(self, text: str, **kwargs: Any) -> dict[str, Any]:
        text_value = str(text or "").strip()
        if not text_value:
            return {"status": "skipped", "reply_text": "", "details": {"provider": "eibrain_subprocess", "reason": "empty_text"}}
        event = _voice_asr_final_event_payload(
            self.config,
            text_value,
            round_id=str(kwargs.get("round_id") or ""),
            trace_id=str(kwargs.get("trace_id") or ""),
            latency_ms=_safe_float(kwargs.get("asr_latency_ms")),
        )
        command = _eibrain_dialogue_command(self.config, text_value)
        started = time.perf_counter()
        try:
            completed = self.runner(
                command,
                cwd=self.config.dialogue_cwd or None,
                env=_dialogue_env(self.config),
                capture_output=True,
                text=True,
                timeout=float(self.config.dialogue_timeout_s),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "reply_text": "",
                "details": {
                    "provider": "eibrain_subprocess",
                    "reason": "timeout",
                    "timeout_s": self.config.dialogue_timeout_s,
                    "event_name": event.get("name", ""),
                    "round_id": event.get("roundId", ""),
                },
            }
        except OSError as exc:
            return {
                "status": "error",
                "reply_text": "",
                "details": {
                    "provider": "eibrain_subprocess",
                    "reason": "subprocess_failed",
                    "error": str(exc),
                    "event_name": event.get("name", ""),
                    "round_id": event.get("roundId", ""),
                },
            }
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        details: dict[str, Any] = {
            "provider": "eibrain_subprocess",
            "command": _redact_command(command),
            "returncode": completed.returncode,
            "elapsed_ms": elapsed_ms,
            "event_name": event.get("name", ""),
            "event_id": event.get("id", ""),
            "round_id": event.get("roundId", ""),
            "stderr": (completed.stderr or "").strip()[-300:],
        }
        if completed.returncode != 0:
            details["reason"] = "nonzero_exit"
            return {"status": "error", "reply_text": "", "details": details}
        parsed = _parse_json(completed.stdout)
        reply = _extract_reply_text(parsed)
        if not reply:
            details["reason"] = "missing_reply"
            details["stdout_preview"] = (completed.stdout or "").strip()[:300]
            return {"status": "error", "reply_text": "", "details": details}
        return {"status": "ok", "reply_text": reply, "details": details}


def _build_dialogue_client(config: NativeVoiceLoopConfig, *, runner: Runner | None = None) -> Any:
    if config.dialogue_backend in {"eibrain_subprocess", "subprocess"}:
        return EIBrainSubprocessDialogueClient(config, runner=runner)
    return TemplateDialogueClient(config)


class NativeVoiceInteractionLoop:
    def __init__(
        self,
        config: NativeVoiceLoopConfig,
        *,
        capture_source: ArecordAudioFrameSource | None = None,
        transcriber: SherpaOnnxWindowTranscriber | None = None,
        tts_synthesizer: Any | None = None,
        dialogue_client: Any | None = None,
        runner: Runner | None = None,
    ) -> None:
        self.config = config
        self.capture_source = capture_source or ArecordAudioFrameSource(config)
        self.transcriber = transcriber or SherpaOnnxWindowTranscriber(
            model_dir=config.asr_model_dir,
            model_type=config.asr_model_type,
            sample_rate=config.sample_rate,
        )
        self.tts_synthesizer = (
            tts_synthesizer
            if tts_synthesizer is not None
            else (MiniMaxRestTtsSynthesizer(config) if config.tts_backend == "minimax" else None)
        )
        self.runner = runner or subprocess.run
        self.dialogue_client = dialogue_client if dialogue_client is not None else _build_dialogue_client(config, runner=self.runner)
        self._state = _LoopState()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames: deque[AudioFrame] = deque()

    def start(self) -> None:
        if not self.config.enabled:
            with self._lock:
                self._state.state = "disabled"
                self._state.phase = "disabled"
                self._state.health = "disabled"
                self._state.last_status = "disabled"
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="eihead-native-voice-loop", daemon=True)
        self._thread.start()
        with self._lock:
            self._state.state = "running"
            self._state.phase = self._idle_phase(self._state)
            self._state.health = "healthy"
            self._state.running = True
            self._state.started_at_ts = time.time()
            self._state.updated_at_ts = self._state.started_at_ts
            self._state.last_status = self._state.phase

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.capture_source.stop()
        with self._lock:
            self._state.running = False
            self._state.state = "stopped"
            self._state.phase = "stopped"
            self._state.last_status = "stopped"
            self._state.updated_at_ts = time.time()

    def stop_speech(self) -> dict[str, Any]:
        return {"status": "stopped", "success": True, "details": {"reason": "no_async_playback_process"}}

    def speak(self, text: str) -> dict[str, Any]:
        text_value = str(text or "").strip()
        if not text_value:
            return {"status": "skipped", "success": False, "details": {"reason": "missing_text"}}
        return self._play_text(text_value)

    def voice_status(self) -> dict[str, Any]:
        return self._voice_status_payload()

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._copy_state()
        capture_status = self.capture_source.status()
        asr_status = self.transcriber.status()
        return {
            "schema": "eihead.eivoice_runtime.diagnostics.v1",
            "state": state.state,
            "conversation_state": state.phase,
            "health": state.health,
            "running": state.running,
            "worker_metrics": {
                "capture_frames": state.frame_count,
                "utterance_count": state.utterance_count,
                "turn_count": state.turn_count,
                "last_capture_frame_duration_ms": self.config.frame_ms if state.frame_count else None,
                "last_playback_frame_duration_ms": state.last_stage_latency_ms.get("speak"),
            },
            "queues": {
                "opus_encode_queue": {"depth": 0, "capacity": 1},
                "ws_send_queue": {"depth": 0, "capacity": 1},
                "opus_decode_queue": {"depth": 0, "capacity": 1},
                "audio_playback_queue": {"depth": 0, "capacity": 1},
            },
            "audio_frontend": {
                "mode": "alsa_vad",
                "healthy": state.health != "error",
                "vad": {
                    "enabled": True,
                    "state": "triggered" if state.vad_triggered else "listening",
                    "rms_threshold": self.config.vad_rms_threshold,
                    "audio_level": state.audio_level,
                    "rms_dbfs": state.rms_dbfs,
                },
                "capture": capture_status,
            },
            "asr": {
                "enabled": bool(self.config.asr_model_dir),
                "provider": "sherpa_onnx",
                "provider_state": asr_status.get("state", "unknown"),
                "final_count": state.turn_count,
                "latest_voice": {
                    "duration_ms": state.captured_ms,
                    "audio_level": state.audio_level,
                    "rms_dbfs": state.rms_dbfs,
                },
                "provider_diagnostics": asr_status,
            },
            "mouth": self._mouth_status(state),
            "wakeword": self._wakeword_status(state),
            "voice_dialogue": self._dialogue_status(state),
            "realtime_audio": self._realtime_audio_status(state),
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self.capture_source.read_frame()
            except Exception as exc:
                self._record_error(exc)
                self._stop_event.wait(0.2)
                continue
            if frame is None:
                self._stop_event.wait(0.02)
                continue
            self._observe_frame(frame)
        self.capture_source.stop()

    def _observe_frame(self, frame: AudioFrame) -> None:
        audio_level = _rms_level(frame.pcm, channels=frame.channels)
        is_voice = audio_level >= self.config.vad_rms_threshold
        finalize = False
        with self._lock:
            self._state.frame_count += 1
            self._state.audio_level = round(audio_level, 5)
            self._state.rms_dbfs = round(_dbfs(audio_level), 2)
            self._state.updated_at_ts = time.time()
            if is_voice:
                if self._state.voice_ms == 0 and not self._state.vad_triggered:
                    self._frames.clear()
                self._frames.append(frame)
                self._state.voice_ms += frame.duration_ms
                self._state.silence_after_voice_ms = 0
                self._state.captured_ms += frame.duration_ms
                if self._state.voice_ms >= self.config.vad_min_voice_ms:
                    if not self._state.vad_triggered:
                        self._state.current_round_id = f"voice-{int(time.time() * 1000)}"
                    self._state.phase = "listening_voice"
                    self._state.vad_triggered = True
            elif self._state.vad_triggered:
                self._frames.append(frame)
                self._state.silence_after_voice_ms += frame.duration_ms
                self._state.captured_ms += frame.duration_ms
            elif self._state.voice_ms > 0:
                self._frames.clear()
                self._state.voice_ms = 0
                self._state.silence_after_voice_ms = 0
                self._state.captured_ms = 0
                self._state.current_round_id = ""
                if self._state.phase != "speaking":
                    self._state.phase = self._idle_phase(self._state)

            if self._state.vad_triggered:
                if (
                    self._state.silence_after_voice_ms >= self.config.vad_end_silence_ms
                    or self._state.captured_ms >= self.config.max_utterance_ms
                ):
                    finalize = True
            self._state.last_status = self._state.phase
        if finalize:
            self._finalize_utterance()

    def _finalize_utterance(self) -> None:
        frames = list(self._frames)
        self._frames.clear()
        if not frames:
            self._reset_vad("silence")
            return
        started = time.perf_counter()
        with self._lock:
            self._state.phase = "asr_decoding"
            self._state.utterance_count += 1
            self._state.updated_at_ts = time.time()
        try:
            text = self.transcriber.transcribe(frames)
        except Exception as exc:
            self._record_error(exc)
            self._reset_vad("asr_error")
            return
        asr_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if not text:
            with self._lock:
                self._state.last_stage_latency_ms = {"listen_asr": asr_ms, "total": asr_ms}
                self._state.last_status = "empty_transcript"
            self._reset_vad("empty_transcript")
            return
        gated_text = self._apply_wake_gate(text, asr_ms=asr_ms)
        if gated_text is None:
            return
        text = gated_text
        with self._lock:
            self._state.phase = "thinking"
            self._state.last_status = "asr_final"
            round_id = self._state.current_round_id
        dialogue_started = time.perf_counter()
        dialogue = self.dialogue_client.reply_to_transcript(
            text,
            round_id=round_id,
            session_id=self.config.dialogue_session_id,
            actor_id=self.config.dialogue_actor_id,
            asr_latency_ms=asr_ms,
        )
        dialogue_ms = round((time.perf_counter() - dialogue_started) * 1000.0, 2)
        dialogue_payload = _mapping(dialogue)
        dialogue_details = _mapping(dialogue_payload.get("details"))
        reply = _dialogue_reply_text(dialogue)
        if not reply:
            reply = self.config.reply_template.format(text=text)
            dialogue_details["fallback_reason"] = "dialogue_reply_unavailable"
        with self._lock:
            self._state.phase = "speaking"
            self._state.last_transcript = text
            self._state.last_reply = reply
            self._state.turn_count += 1
            self._state.last_stage_latency_ms = {"listen_asr": asr_ms, "dialogue": dialogue_ms}
            self._state.last_dialogue_details = dialogue_details
            self._state.last_status = "reply_ready" if dialogue_payload.get("status") == "ok" else "reply_degraded"
        speech = self._play_text(reply)
        speak_ms = _safe_float(_mapping(speech.get("details")).get("playback_elapsed_ms"))
        with self._lock:
            self._state.last_stage_latency_ms["speak"] = speak_ms
            self._state.last_stage_latency_ms["total"] = round(asr_ms + dialogue_ms + speak_ms, 2)
            self._state.phase = self._idle_phase(self._state)
            self._state.last_status = "turn_complete" if speech.get("success") else "speech_error"
            self._state.updated_at_ts = time.time()
        self._reset_vad("turn_complete")

    def _apply_wake_gate(self, text: str, *, asr_ms: float) -> str | None:
        if not self.config.wake_word_required:
            return text
        with self._lock:
            conversation_active = self._state.conversation_active

        if conversation_active and _contains_spoken_phrase(text, self.config.end_phrases):
            with self._lock:
                self._state.conversation_active = False
                self._state.last_gate_reason = "end_phrase"
            self._finish_local_voice_turn(
                text=text,
                reply=self.config.end_ack_text,
                asr_ms=asr_ms,
                status="conversation_ended",
                gate_reason="end_phrase",
            )
            return None

        if conversation_active:
            return text

        remainder = _strip_wake_word_prefix(text, self.config.wake_words)
        if remainder is None:
            with self._lock:
                self._state.last_transcript = text
                self._state.last_reply = ""
                self._state.last_stage_latency_ms = {"listen_asr": asr_ms, "total": asr_ms}
                self._state.last_dialogue_details = {"provider": "wake_gate", "reason": "waiting_for_wake_word"}
                self._state.last_gate_reason = "wake_word_required"
                self._state.phase = self._idle_phase(self._state)
                self._state.last_status = "waiting_for_wake_word"
                self._state.updated_at_ts = time.time()
            self._reset_vad("waiting_for_wake_word")
            return None

        with self._lock:
            self._state.conversation_active = True
            self._state.last_gate_reason = "wake_word_detected"

        if not remainder:
            self._finish_local_voice_turn(
                text=text,
                reply=self.config.wake_ack_text,
                asr_ms=asr_ms,
                status="wake_word_detected",
                gate_reason="wake_word_detected",
            )
            return None
        return remainder

    def _finish_local_voice_turn(
        self,
        *,
        text: str,
        reply: str,
        asr_ms: float,
        status: str,
        gate_reason: str,
    ) -> None:
        with self._lock:
            self._state.phase = "speaking"
            self._state.last_transcript = text
            self._state.last_reply = reply
            self._state.turn_count += 1
            self._state.last_stage_latency_ms = {"listen_asr": asr_ms, "dialogue": 0.0}
            self._state.last_dialogue_details = {"provider": "wake_gate", "reason": gate_reason}
            self._state.last_gate_reason = gate_reason
            self._state.last_status = "reply_ready"
        speech = self._play_text(reply)
        speak_ms = _safe_float(_mapping(speech.get("details")).get("playback_elapsed_ms"))
        final_status = status if speech.get("success") else "speech_error"
        with self._lock:
            self._state.last_stage_latency_ms["speak"] = speak_ms
            self._state.last_stage_latency_ms["total"] = round(asr_ms + speak_ms, 2)
            self._state.phase = self._idle_phase(self._state)
            self._state.last_status = final_status
            self._state.updated_at_ts = time.time()
        self._reset_vad(final_status)

    def _play_text(self, text: str) -> dict[str, Any]:
        started = time.perf_counter()
        self.capture_source.stop()
        try:
            if self.config.tts_backend == "minimax":
                minimax = self._play_minimax_text(text, started=started)
                if minimax.get("success"):
                    return minimax
                fallback = self._play_configured_tts_fallback(text, started=time.perf_counter())
                if fallback is None:
                    primary_error = _mapping(minimax.get("details"))
                    return {
                        "status": "error",
                        "success": False,
                        "details": {
                            **primary_error,
                            "fallback_from": "minimax",
                            "fallback_provider": self.config.tts_fallback_provider,
                            "fallback_reason": "tts_fallback_not_configured",
                        },
                    }
                fallback_details = _mapping(fallback.get("details"))
                fallback_details["fallback_from"] = "minimax"
                fallback_details["primary_error"] = _mapping(minimax.get("details"))
                fallback["details"] = fallback_details
                return fallback
            if self.config.tts_backend == "piper":
                return self._play_piper_text(text, started=started)
            return {
                "status": "error",
                "success": False,
                "details": {
                    "backend": self.config.tts_backend,
                    "reason": "unsupported_tts_backend",
                },
            }
        finally:
            cooldown_s = max(0.0, float(self.config.playback_echo_cooldown_ms) / 1000.0)
            if cooldown_s:
                self._stop_event.wait(cooldown_s)
            with self._lock:
                self._frames.clear()
                self._state.vad_triggered = False
                self._state.voice_ms = 0
                self._state.silence_after_voice_ms = 0
                self._state.captured_ms = 0
                self._state.current_round_id = ""

    def _play_configured_tts_fallback(self, text: str, *, started: float) -> dict[str, Any] | None:
        fallback_provider = self.config.tts_fallback_provider.strip().lower()
        if fallback_provider == "piper" or (not fallback_provider and self.config.piper_model_path):
            return self._play_piper_text(text, started=started)
        if fallback_provider:
            return {
                "status": "error",
                "success": False,
                "details": {
                    "backend": fallback_provider,
                    "reason": "unsupported_tts_fallback_provider",
                },
            }
        return None

    def _play_minimax_text(self, text: str, *, started: float) -> dict[str, Any]:
        synthesizer = self.tts_synthesizer
        if synthesizer is None:
            return {
                "status": "error",
                "success": False,
                "details": {"backend": "minimax", "reason": "minimax_synthesizer_unavailable"},
            }
        synth = synthesizer.synthesize(text)
        if not isinstance(synth, Mapping) or synth.get("status") != "ok":
            return {
                "status": "error",
                "success": False,
                "details": dict(_mapping(synth.get("details") if isinstance(synth, Mapping) else None)),
            }
        audio = synth.get("audio_bytes")
        if not isinstance(audio, (bytes, bytearray)) or not audio:
            return {
                "status": "error",
                "success": False,
                "details": {"backend": "minimax", "reason": "missing_audio_bytes"},
            }
        suffix = ".wav" if self.config.minimax_audio_format.lower() == "wav" else f".{self.config.minimax_audio_format.lower()}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            wav_path = Path(handle.name)
            handle.write(bytes(audio))
        try:
            playback = self.runner(
                ["aplay", "-q", "-D", self.config.speaker_device, str(wav_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            details = dict(_mapping(synth.get("details")))
            return {
                "status": "ok" if playback.returncode == 0 else "error",
                "success": playback.returncode == 0,
                "details": {
                    **details,
                    "backend": "minimax",
                    "playback_backend": self.config.playback_backend,
                    "device": self.config.speaker_device,
                    "text_preview": text[:60],
                    "playback_elapsed_ms": elapsed_ms,
                    "returncode": playback.returncode,
                    "stderr": (playback.stderr or "").strip(),
                },
            }
        finally:
            wav_path.unlink(missing_ok=True)

    def _play_piper_text(self, text: str, *, started: float) -> dict[str, Any]:
        model_path = self.config.piper_model_path.strip()
        if not model_path:
            return {
                "status": "error",
                "success": False,
                "details": {
                    "backend": "piper",
                    "reason": "missing_piper_model_path",
                },
            }
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = Path(handle.name)
        command = [
            self.config.piper_command,
            "--model",
            model_path,
        ]
        config_path = self.config.piper_config_path.strip()
        if config_path:
            command.extend(["--config", config_path])
        command.extend(["--output_file", str(wav_path)])
        try:
            synth = self.runner(
                command,
                input=text,
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
                        "command": self.config.piper_command,
                        "model_path": model_path,
                        "config_path": config_path,
                        "returncode": synth.returncode,
                        "stderr": (synth.stderr or "").strip(),
                    },
                }
            playback = self.runner(
                ["aplay", "-q", "-D", self.config.speaker_device, str(wav_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return {
                "status": "ok" if playback.returncode == 0 else "error",
                "success": playback.returncode == 0,
                "details": {
                    "backend": "piper",
                    "command": self.config.piper_command,
                    "model_path": model_path,
                    "config_path": config_path,
                    "playback_backend": self.config.playback_backend,
                    "device": self.config.speaker_device,
                    "text_preview": text[:60],
                    "playback_elapsed_ms": elapsed_ms,
                    "returncode": playback.returncode,
                    "stderr": (playback.stderr or "").strip(),
                },
            }
        finally:
            wav_path.unlink(missing_ok=True)

    def _voice_status_payload(self) -> dict[str, Any]:
        runtime = self.status()
        state = self._copy_state()
        return {
            "status": "ready" if state.running and state.health != "error" else state.health,
            "ear": {
                "status": "listening" if state.running else state.state,
                "provider": "sherpa_onnx",
                "readiness_message": "native realtime voice loop is attached",
                "capture": {
                    "status": "running" if state.running else state.state,
                    "details": runtime["audio_frontend"]["capture"],
                },
                "asr": runtime["asr"],
                "audio_level": state.audio_level,
                "rms": state.audio_level,
                "vad_triggered": state.vad_triggered,
                "transcript": state.last_transcript,
                "stage_latency_ms": {"listen_asr": state.last_stage_latency_ms.get("listen_asr")},
            },
            "mouth": self._mouth_status(state),
            "wakeword": self._wakeword_status(state),
            "voice_dialogue": self._dialogue_status(state),
            "realtime_audio": self._realtime_audio_status(state),
            "current_round_id": state.current_round_id or None,
            "scheduler_state": "listening" if state.running else state.state,
            "last_stage_latency_ms": dict(state.last_stage_latency_ms),
            "last_turn": {
                "transcript": state.last_transcript,
                "reply": state.last_reply,
                "status": state.last_status,
            }
            if state.last_transcript or state.last_reply
            else None,
            "eivoice_runtime": runtime,
            "readiness_message": "native realtime voice loop is attached",
        }

    def _dialogue_status(self, state: _LoopState) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "running": state.running,
            "phase": state.phase,
            "last_status": state.last_status,
            "last_transcript": state.last_transcript,
            "last_reply": state.last_reply,
            "last_error": state.last_error,
            "conversation_active": state.conversation_active,
            "wake_word_required": self.config.wake_word_required,
            "wake_words": list(self.config.wake_words),
            "end_phrases": list(self.config.end_phrases),
            "last_gate_reason": state.last_gate_reason,
            "turn_count": state.turn_count,
            "current_round_id": state.current_round_id,
            "last_stage_latency_ms": dict(state.last_stage_latency_ms),
            "dialogue": dict(state.last_dialogue_details),
            "readiness_message": "native realtime voice loop is attached",
        }

    def _wakeword_status(self, state: _LoopState) -> dict[str, Any]:
        if not self.config.wake_word_required:
            wake_state = "disabled"
        else:
            wake_state = "active" if state.conversation_active else "armed"
        return {
            "enabled": self.config.wake_word_required,
            "state": wake_state,
            "wake_words": list(self.config.wake_words),
            "end_phrases": list(self.config.end_phrases),
            "last_gate_reason": state.last_gate_reason,
            "readiness_message": "wake gate is attached" if self.config.wake_word_required else "wake gate is disabled",
        }

    def _realtime_audio_status(self, state: _LoopState) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "running": state.running,
            "audio_level": state.audio_level,
            "rms_dbfs": state.rms_dbfs,
            "vad_triggered": state.vad_triggered,
            "captured_ms": state.captured_ms,
            "voice_ms": state.voice_ms,
            "silence_after_voice_ms": state.silence_after_voice_ms,
        }

    def _mouth_status(self, state: _LoopState) -> dict[str, Any]:
        if self.config.tts_backend == "minimax":
            model = self.config.minimax_model
            voice_id = self.config.minimax_voice_id
        elif self.config.tts_backend == "piper":
            model = Path(self.config.piper_model_path).name if self.config.piper_model_path else "piper"
            voice_id = ""
        else:
            model = self.config.tts_backend
            voice_id = ""
        return {
            "status": "idle" if state.phase != "speaking" else "playing",
            "backend": self.config.tts_backend,
            "model": model,
            "voice_id": voice_id,
            "text_preview": state.last_reply[:60],
            "readiness_message": "native playback path is attached",
            "busy": state.phase == "speaking",
            "playback_state": "playing" if state.phase == "speaking" else "idle",
            "stage_latency_ms": {"speak": state.last_stage_latency_ms.get("speak")},
            "tts_playback": {
                "status": "ready",
                "details": {
                    "provider": self.config.tts_backend,
                    "model": model,
                    "voice_id": voice_id,
                    "fallback_provider": self.config.tts_fallback_provider,
                    "piper_model_path": self.config.piper_model_path,
                    "device": self.config.speaker_device,
                },
            },
        }

    def _copy_state(self) -> _LoopState:
        return _LoopState(
            **{
                name: dict(value) if isinstance(value, dict) else value
                for name, value in {
                    field_name: getattr(self._state, field_name)
                    for field_name in _LoopState.__dataclass_fields__
                }.items()
            }
        )

    def _reset_vad(self, reason: str) -> None:
        with self._lock:
            self._state.vad_triggered = False
            self._state.voice_ms = 0
            self._state.silence_after_voice_ms = 0
            self._state.captured_ms = 0
            self._state.current_round_id = ""
            if self._state.phase not in {"speaking", "stopped", "disabled"}:
                self._state.phase = self._idle_phase(self._state)
            self._state.updated_at_ts = time.time()
            if reason:
                self._state.last_status = reason

    def _idle_phase(self, state: _LoopState) -> str:
        if self.config.wake_word_required and not state.conversation_active:
            return "awaiting_wake"
        return "listening"

    def _record_error(self, exc: BaseException) -> None:
        with self._lock:
            self._state.state = "error"
            self._state.phase = "error"
            self._state.health = "error"
            self._state.last_error = str(exc)
            self._state.last_status = "error"
            self._state.updated_at_ts = time.time()


_SPOKEN_SEPARATORS = set(" \t\r\n,，.。!！?？;；:：、\"“”'‘’（）()[]【】{}<>《》-—_…")


def _spoken_char_map(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    indexes: list[int] = []
    for index, char in enumerate(str(text or "")):
        if char in _SPOKEN_SEPARATORS:
            continue
        normalized_chars.append(char.lower())
        indexes.append(index)
    return "".join(normalized_chars), indexes


def _normalize_spoken_phrase(text: str) -> str:
    normalized, _ = _spoken_char_map(text)
    return normalized


def _contains_spoken_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    normalized, _ = _spoken_char_map(text)
    for phrase in phrases:
        normalized_phrase = _normalize_spoken_phrase(phrase)
        if normalized_phrase and normalized_phrase in normalized:
            return True
    return False


def _strip_wake_word_prefix(text: str, wake_words: tuple[str, ...]) -> str | None:
    normalized, indexes = _spoken_char_map(text)
    if not normalized:
        return None
    for wake_word in wake_words:
        for wake in _wake_word_candidates(wake_word):
            if not normalized.startswith(wake):
                continue
            original_end = indexes[len(wake) - 1] + 1
            return str(text or "")[original_end:].strip("".join(_SPOKEN_SEPARATORS))
    return None


def _wake_word_candidates(wake_word: str) -> tuple[str, ...]:
    wake = _normalize_spoken_phrase(wake_word)
    if not wake:
        return ()
    candidates = [wake]
    for greeting in ("你好", "您好"):
        prefix = _normalize_spoken_phrase(greeting)
        if wake.startswith(prefix) and len(wake) > len(prefix):
            candidates.append(wake[len(prefix) :])
    return tuple(dict.fromkeys(candidates))


def _pcm_to_float_samples(pcm_bytes: bytes, *, channels: int) -> list[float]:
    samples = array("h")
    samples.frombytes(pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % 2)])
    if channels <= 1:
        mono = samples
    else:
        channel_samples = [array("h") for _ in range(channels)]
        for idx in range(0, len(samples) - (len(samples) % channels), channels):
            for channel_index in range(channels):
                channel_samples[channel_index].append(samples[idx + channel_index])
        mono = max(channel_samples, key=lambda values: sum(sample * sample for sample in values), default=array("h"))
    return [sample / 32768.0 for sample in mono]


def _waveform_buffer(samples: list[float]) -> Any:
    try:
        import numpy as np

        return np.asarray(samples, dtype=np.float32)
    except Exception:
        return array("f", samples)


def _rms_level(pcm_bytes: bytes, *, channels: int) -> float:
    samples = _pcm_to_float_samples(pcm_bytes, channels=channels)
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _dbfs(rms_level: float) -> float:
    if rms_level <= 0.0:
        return -120.0
    return 20.0 * math.log10(rms_level)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _minimax_t2a_url(api_base_url: str) -> str:
    endpoint = str(api_base_url or "https://api.minimaxi.com").rstrip("/")
    if endpoint.endswith("/v1/t2a_v2"):
        return endpoint
    return f"{endpoint}/v1/t2a_v2"


def _normalize_minimax_number(value: float | int) -> float | int:
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _redact_secret(value: str) -> str:
    text = str(value)
    if len(text) <= 8:
        return "***"
    return f"{text[:4]}...{text[-4:]}"


def _redact_text(text: str, *secrets: str) -> str:
    redacted = str(text)
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, _redact_secret(secret))
    return redacted


def _eibrain_dialogue_command(config: NativeVoiceLoopConfig, text: str) -> list[str]:
    command = shlex.split(config.dialogue_command) if config.dialogue_command else [sys.executable]
    command.extend(["-m", config.dialogue_module])
    if config.dialogue_config_path:
        command.extend(["--config", config.dialogue_config_path])
    command.extend(
        [
            "--text",
            text,
            "--session-id",
            config.dialogue_session_id,
            "--actor-id",
            config.dialogue_actor_id,
        ]
    )
    return command


def _dialogue_env(config: NativeVoiceLoopConfig) -> dict[str, str] | None:
    if not config.dialogue_pythonpath:
        return None
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    paths = [config.dialogue_pythonpath]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _voice_asr_final_event_payload(
    config: NativeVoiceLoopConfig,
    text: str,
    *,
    round_id: str,
    trace_id: str,
    latency_ms: float,
) -> dict[str, Any]:
    source = {"domain": "eihead", "instanceId": config.dialogue_head_instance_id}
    target = {"domain": "eibrain", "instanceId": config.dialogue_brain_instance_id}
    try:
        from eiprotocol import build_voice_asr_event

        return build_voice_asr_event(
            source=source,
            target=target,
            text=text,
            final=True,
            language="zh-CN",
            latency_ms=latency_ms,
            asr_backend="sherpa_onnx",
            session_id=config.dialogue_session_id,
            round_id=round_id or None,
            trace_id=trace_id,
        ).to_dict()
    except Exception:
        return {
            "name": "ei.voice.asr.final",
            "type": "dialogue",
            "source": source,
            "target": target,
            "sessionId": config.dialogue_session_id,
            "roundId": round_id,
            "traceId": trace_id,
            "content": {
                "text": text,
                "final": True,
                "language": "zh-CN",
                "latencyMs": latency_ms,
                "asrBackend": "sherpa_onnx",
            },
        }


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _extract_reply_text(payload: Any) -> str:
    if isinstance(payload, list):
        for item in payload:
            reply = _extract_reply_text(item)
            if reply:
                return reply
        return ""
    if not isinstance(payload, Mapping):
        return ""
    for key in ("reply_text", "replyText", "last_reply", "lastReply", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    content = payload.get("content")
    if isinstance(content, Mapping):
        reply = _extract_reply_text(content)
        if reply:
            return reply
    for key in ("actions", "speechSegments", "speech_segments", "speech"):
        value = payload.get(key)
        if isinstance(value, list):
            reply = _extract_reply_text(value)
            if reply:
                return reply
    kind = str(payload.get("kind") or payload.get("type") or payload.get("actionType") or "")
    if kind in {"play_speech_action", "speak", "speech", "play_speech"}:
        value = payload.get("text")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _dialogue_reply_text(payload: Any) -> str:
    if isinstance(payload, Mapping):
        for key in ("reply_text", "replyText", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return _extract_reply_text(payload)


def _redact_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for part in command:
        if redact_next:
            redacted.append("***")
            redact_next = False
            continue
        redacted.append(part)
        if part.lower() in {"--api-key", "--token", "--password"}:
            redact_next = True
    return redacted


__all__ = [
    "ArecordAudioFrameSource",
    "EIBrainSubprocessDialogueClient",
    "MiniMaxRestTtsSynthesizer",
    "NativeVoiceInteractionLoop",
    "NativeVoiceLoopConfig",
    "SherpaOnnxWindowTranscriber",
]
