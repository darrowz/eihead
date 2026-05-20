"""Native service composition for the standalone eihead runtime."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

from eihead.devices.neck_servo import build_neck_servo_adapter
from eihead.eivoice_runtime.native_loop import NativeVoiceInteractionLoop, NativeVoiceLoopConfig
from eihead.eye import GStreamerHailoRealtimeConfig, RealtimeEyeService
from .openclaw_runtime import OpenClawRealtimeRuntime


EyeAdapterFactory = Callable[[GStreamerHailoRealtimeConfig], Any]


def build_native_provider_services(
    config: Any | None,
    *,
    config_path: str,
    eye_adapter_factory: EyeAdapterFactory | None = None,
) -> dict[str, Any]:
    services: dict[str, Any] = {}
    eye_service = build_realtime_eye_service(
        config,
        config_path=config_path,
        adapter_factory=eye_adapter_factory,
    )
    if eye_service is not None:
        services["eye"] = eye_service
    return services


def build_native_voice_runtime(config: Any | None) -> NativeVoiceInteractionLoop | None:
    if _bool(
        os.environ.get("EIHEAD_NATIVE_VOICE_RUNTIME_DISABLED")
        or os.environ.get("EIHEAD_DISABLE_NATIVE_VOICE_RUNTIME"),
        False,
    ):
        return None
    if not _is_honjia_config(config):
        return None
    loop_config = native_voice_loop_config_from_eihead_config(config)
    if loop_config.transport_provider == "openclaw_realtime":
        return OpenClawRealtimeRuntime(loop_config)
    if not _voice_realtime_enabled(config):
        return None
    return NativeVoiceInteractionLoop(loop_config)


def native_voice_loop_config_from_eihead_config(config: Any) -> NativeVoiceLoopConfig:
    raw = _mapping(getattr(config, "raw", None))
    devices = _mapping(raw.get("devices"))
    microphone = _mapping(devices.get("microphone"))
    speaker = _mapping(devices.get("speaker"))
    asr = _software_capability(config, "asr")
    tts = _software_capability(config, "tts")
    dialogue = _software_capability(config, "dialogue")
    asr_limits = _mapping(getattr(asr, "limits", None))
    asr_extra = _mapping(getattr(asr, "extra", None))
    tts_extra = _mapping(getattr(tts, "extra", None))
    dialogue_extra = _mapping(getattr(dialogue, "extra", None))
    dialogue_limits = _mapping(getattr(dialogue, "limits", None))
    microphone_limits = _mapping(microphone.get("limits"))
    tts_provider = _text(getattr(tts, "provider", ""), "")
    minimax_backend = tts_provider.lower() == "minimax"
    piper_backend = tts_provider.lower() == "piper"
    dialogue_provider = _text(getattr(dialogue, "provider", ""), "template")
    transport_provider = _normalize_transport_provider(
        dialogue_extra.get("transport_provider")
        or dialogue_extra.get("transportProvider")
        or dialogue_limits.get("transport_provider")
        or dialogue_limits.get("transportProvider")
        or (
            dialogue_provider
            if dialogue_provider == "openclaw_realtime"
            else dialogue_limits.get("transport")
        )
        or os.environ.get("EIHEAD_VOICE_TRANSPORT_PROVIDER")
    )
    fallback_transport_provider_raw = (
        dialogue_extra.get("fallback_transport_provider")
        or dialogue_extra.get("fallbackTransportProvider")
        or os.environ.get("EIHEAD_VOICE_FALLBACK_PROVIDER")
        or ""
    )
    fallback_transport_provider = (
        _normalize_transport_provider(fallback_transport_provider_raw)
        if _text(fallback_transport_provider_raw, "").strip()
        else ""
    )

    return NativeVoiceLoopConfig(
        enabled=True,
        transport_provider=transport_provider,
        fallback_transport_provider=fallback_transport_provider,
        openclaw_ws_url=_text(
            dialogue_extra.get("ws_url")
            or dialogue_extra.get("wsUrl")
            or dialogue_extra.get("openclaw_ws_url")
            or dialogue_extra.get("openclawWsUrl")
            or os.environ.get("EIHEAD_OPENCLAW_WS_URL"),
            "",
        ),
        openclaw_token_env_var=_text(
            dialogue_extra.get("token_env_var")
            or dialogue_extra.get("tokenEnvVar")
            or os.environ.get("EIHEAD_OPENCLAW_TOKEN_ENV_VAR"),
            "OPENCLAW_REALTIME_TOKEN",
        ),
        openclaw_provider=_text(
            dialogue_extra.get("realtime_provider")
            or dialogue_extra.get("realtimeProvider")
            or dialogue_extra.get("openclaw_provider")
            or dialogue_extra.get("openclawProvider")
            or os.environ.get("EIHEAD_OPENCLAW_PROVIDER"),
            "openai",
        ),
        openclaw_model=_text(
            dialogue_extra.get("model")
            or dialogue_extra.get("openclaw_model")
            or dialogue_extra.get("openclawModel")
            or os.environ.get("EIHEAD_OPENCLAW_MODEL"),
            "",
        ),
        openclaw_voice=_text(
            dialogue_extra.get("voice")
            or dialogue_extra.get("openclaw_voice")
            or dialogue_extra.get("openclawVoice")
            or os.environ.get("EIHEAD_OPENCLAW_VOICE"),
            "Zephyr",
        ),
        openclaw_brain_agent=_text(
            dialogue_extra.get("brain_agent")
            or dialogue_extra.get("brainAgent")
            or os.environ.get("EIHEAD_OPENCLAW_BRAIN_AGENT"),
            "enabled",
        ),
        openclaw_protocol=_text(
            dialogue_extra.get("protocol")
            or dialogue_extra.get("ws_protocol")
            or dialogue_extra.get("wsProtocol")
            or os.environ.get("EIHEAD_OPENCLAW_WS_PROTOCOL"),
            "",
        ),
        openclaw_connect_timeout_s=_float(
            dialogue_extra.get("connect_timeout_s")
            or dialogue_extra.get("connectTimeoutS")
            or os.environ.get("EIHEAD_OPENCLAW_CONNECT_TIMEOUT_S"),
            10.0,
        ),
        openclaw_receive_timeout_s=_float(
            dialogue_extra.get("receive_timeout_s")
            or dialogue_extra.get("receiveTimeoutS")
            or os.environ.get("EIHEAD_OPENCLAW_RECEIVE_TIMEOUT_S"),
            0.02,
        ),
        openclaw_session_ready_timeout_s=_float(
            dialogue_extra.get("session_ready_timeout_s")
            or dialogue_extra.get("sessionReadyTimeoutS")
            or os.environ.get("EIHEAD_OPENCLAW_SESSION_READY_TIMEOUT_S"),
            15.0,
        ),
        microphone_device=_text(microphone.get("device") or microphone.get("path"), "default"),
        speaker_device=_text(speaker.get("device") or speaker.get("path"), "default"),
        sample_rate=_int(microphone.get("sample_rate") or microphone.get("sampleRate"), 16000),
        channels=_int(microphone.get("channels"), 1),
        frame_ms=_int(asr_extra.get("frame_ms") or asr_extra.get("frameMs") or microphone_limits.get("frame_ms"), 120),
        vad_rms_threshold=_float(
            asr_extra.get("vad_rms_threshold")
            or asr_extra.get("vadRmsThreshold")
            or microphone.get("vad_rms_threshold")
            or microphone.get("vadRmsThreshold"),
            0.075,
        ),
        vad_min_voice_ms=_int(asr_extra.get("vad_min_voice_ms") or asr_extra.get("vadMinVoiceMs"), 240),
        vad_end_silence_ms=_int(
            asr_extra.get("vad_end_silence_ms") or asr_extra.get("vadEndSilenceMs"),
            600,
        ),
        max_utterance_ms=_int(
            asr_extra.get("max_utterance_ms")
            or asr_extra.get("maxUtteranceMs")
            or asr_limits.get("max_utterance_ms"),
            4200,
        ),
        asr_model_dir=_text(getattr(asr, "model_dir", ""), ""),
        asr_model_type=_text(asr_extra.get("model_type") or asr_extra.get("modelType"), "lstm"),
        wake_word_required=_bool(
            dialogue_extra.get("wake_word_required") or dialogue_extra.get("wakeWordRequired"),
            False,
        ),
        wake_words=_tuple_text(dialogue_extra.get("wake_words") or dialogue_extra.get("wakeWords")) or ("你好鸿途",),
        end_phrases=_tuple_text(dialogue_extra.get("end_phrases") or dialogue_extra.get("endPhrases")) or ("结束对话",),
        wake_ack_text=_text(dialogue_extra.get("wake_ack_text") or dialogue_extra.get("wakeAckText"), "我在。"),
        end_ack_text=_text(dialogue_extra.get("end_ack_text") or dialogue_extra.get("endAckText"), "好的，结束对话。"),
        tts_backend="minimax" if minimax_backend else ("piper" if piper_backend else tts_provider),
        tts_fallback_provider=_text(
            tts_extra.get("fallback_provider")
            or tts_extra.get("fallbackProvider")
            or os.environ.get("EIHEAD_TTS_FALLBACK_PROVIDER"),
            "",
        ),
        piper_command=_text(
            tts_extra.get("piper_command")
            or tts_extra.get("piperCommand")
            or os.environ.get("EIHEAD_PIPER_COMMAND"),
            "piper",
        ),
        piper_model_path=_text(
            tts_extra.get("piper_model_path")
            or tts_extra.get("piperModelPath")
            or os.environ.get("EIHEAD_PIPER_MODEL_PATH")
            or os.environ.get("PIPER_MODEL_PATH"),
            "",
        ),
        piper_config_path=_text(
            tts_extra.get("piper_config_path")
            or tts_extra.get("piperConfigPath")
            or os.environ.get("EIHEAD_PIPER_CONFIG_PATH")
            or os.environ.get("PIPER_CONFIG_PATH"),
            "",
        ),
        playback_backend="aplay",
        playback_echo_cooldown_ms=_int(
            tts_extra.get("playback_echo_cooldown_ms")
            or tts_extra.get("playbackEchoCooldownMs")
            or asr_extra.get("playback_echo_cooldown_ms")
            or asr_extra.get("playbackEchoCooldownMs"),
            350,
        ),
        minimax_api_key=_text(
            tts_extra.get("api_key")
            or tts_extra.get("apiKey")
            or os.environ.get("EIVOICE_MINIMAX_API_KEY")
            or os.environ.get("MINIMAX_API_KEY"),
            "",
        ),
        minimax_api_base_url=_text(
            tts_extra.get("api_base_url")
            or tts_extra.get("apiBaseUrl")
            or os.environ.get("EIVOICE_MINIMAX_API_BASE_URL")
            or os.environ.get("MINIMAX_API_HOST")
            or os.environ.get("MINIMAX_API_BASE_URL"),
            "https://api.minimaxi.com",
        ),
        minimax_model=_text(getattr(tts, "model", "") or os.environ.get("EIVOICE_MINIMAX_MODEL") or os.environ.get("MINIMAX_MODEL"), "speech-2.8-hd"),
        minimax_voice_id=_text(
            tts_extra.get("voice_id")
            or tts_extra.get("voiceId")
            or tts_extra.get("minimax_voice_id")
            or tts_extra.get("minimaxVoiceId")
            or os.environ.get("EIVOICE_MINIMAX_VOICE_ID")
            or os.environ.get("MINIMAX_VOICE_ID"),
            "female-shaonv",
        ),
        minimax_audio_format=_text(tts_extra.get("audio_format") or tts_extra.get("audioFormat"), "wav"),
        minimax_sample_rate=_int(tts_extra.get("sample_rate") or tts_extra.get("sampleRate"), 32000),
        minimax_bitrate=_int(tts_extra.get("bitrate"), 128000),
        minimax_channel=_int(tts_extra.get("channel") or tts_extra.get("channels"), 1),
        minimax_speed=_float(tts_extra.get("speed"), 1.0),
        minimax_volume=_float(tts_extra.get("volume") or tts_extra.get("vol"), 1.0),
        minimax_pitch=_float(tts_extra.get("pitch"), 0.0),
        minimax_language_boost=_text(tts_extra.get("language_boost") or tts_extra.get("languageBoost"), "auto"),
        minimax_timeout_s=_float(tts_extra.get("timeout_s") or tts_extra.get("timeoutS"), 30.0),
        dialogue_backend=dialogue_provider,
        dialogue_command=_text(dialogue_extra.get("command"), ""),
        dialogue_module=_text(dialogue_extra.get("module"), "apps.cognitive_runtime"),
        dialogue_cwd=_text(dialogue_extra.get("cwd") or dialogue_extra.get("working_dir") or dialogue_extra.get("workingDir"), ""),
        dialogue_config_path=_text(dialogue_extra.get("config_path") or dialogue_extra.get("configPath"), ""),
        dialogue_pythonpath=_text(dialogue_extra.get("pythonpath") or dialogue_extra.get("python_path") or dialogue_extra.get("pythonPath"), ""),
        dialogue_timeout_s=_float(dialogue_extra.get("timeout_s") or dialogue_extra.get("timeoutS"), 12.0),
        dialogue_session_id=_text(dialogue_extra.get("session_id") or dialogue_extra.get("sessionId"), "honjia-voice"),
        dialogue_actor_id=_text(dialogue_extra.get("actor_id") or dialogue_extra.get("actorId"), "darrow"),
        dialogue_head_instance_id=_text(dialogue_extra.get("head_instance_id") or dialogue_extra.get("headInstanceId"), "honjia"),
        dialogue_brain_instance_id=_text(dialogue_extra.get("brain_instance_id") or dialogue_extra.get("brainInstanceId"), "honxin"),
    )


def build_realtime_eye_service(
    config: Any | None,
    *,
    config_path: str,
    adapter_factory: EyeAdapterFactory | None = None,
) -> RealtimeEyeService | None:
    if not _is_honjia_config(config) or not _vision_realtime_enabled(config):
        return None
    gstreamer_config = gstreamer_hailo_config_from_eihead_config(config, config_path=config_path)
    if adapter_factory is not None:
        adapter = adapter_factory(gstreamer_config)
    else:
        state_path = _camera_state_path(config)
        adapter = (
            StateFileEyeAdapter(gstreamer_config, state_path=state_path)
            if state_path
            else SafeSubprocessEyeAdapter(gstreamer_config)
        )
    return RealtimeEyeService(adapter=adapter)


class StateFileEyeAdapter:
    """Read realtime eye status from the persistent vision loop state file."""

    def __init__(
        self,
        config: GStreamerHailoRealtimeConfig,
        *,
        state_path: str | Path,
        max_age_s: float = 3.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.state_path = Path(state_path)
        self.max_age_s = float(max_age_s)
        self._clock = clock

    def status(self) -> dict[str, Any]:
        return self._read()

    def poll(self) -> dict[str, Any]:
        return self._read()

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._degraded(
                "persistent vision state file is not available yet",
                status_reason="vision_state_missing",
            )
        except Exception as exc:
            return self._degraded(
                f"persistent vision state file is unreadable: {exc.__class__.__name__}: {exc}",
                status_reason="vision_state_unreadable",
            )
        if not isinstance(payload, Mapping):
            return self._degraded(
                "persistent vision state file did not contain a JSON object",
                status_reason="vision_state_invalid",
            )

        updated_at = _float(payload.get("updated_at_ts"), 0.0)
        age_s = self._clock() - updated_at if updated_at > 0 else None
        if age_s is None:
            return self._degraded(
                "persistent vision state file is missing updated_at_ts",
                status_reason="vision_state_missing_timestamp",
            )
        if self.max_age_s > 0 and age_s > self.max_age_s:
            return self._degraded(
                f"persistent vision state is stale by {age_s:.1f}s",
                status_reason="vision_state_stale",
                age_s=age_s,
            )

        status_payload = payload.get("status_payload")
        status = dict(status_payload) if isinstance(status_payload, Mapping) else dict(payload)
        status.setdefault("schema", "eihead.eye.realtime_status.v1")
        status.setdefault("kind", "realtime_vision_observation")
        status.setdefault("mode", self.config.mode)
        status.setdefault("backend", self.config.backend)
        status.setdefault("source", "eihead.eye.vision_loop")
        status.setdefault("placeholder", False)
        status.setdefault("not_wired", False)
        status.setdefault("stream_ready", bool(payload.get("stream_ready", False)))
        status.setdefault("degraded", status.get("status") == "degraded")
        status.setdefault("status_reason", payload.get("status_reason") or status.get("status"))
        status.setdefault("message", payload.get("message") or "persistent vision state is live")
        status["state_file"] = {
            "path": str(self.state_path),
            "updated_at_ts": updated_at,
            "age_s": age_s,
            "source": payload.get("source", "eihead.eye.vision_loop"),
        }
        status.setdefault("pipeline", self.config.pipeline_fields())
        status.setdefault(
            "devices",
            {
                "camera": self.config.camera_device,
                "hailo": self.config.hailo_device,
            },
        )
        return status

    def _degraded(self, message: str, *, status_reason: str, **details: Any) -> dict[str, Any]:
        return {
            "schema": "eihead.eye.realtime_status.v1",
            "kind": "realtime_vision_observation",
            "mode": self.config.mode,
            "status": "degraded",
            "backend": self.config.backend,
            "source": "eihead.runtime.native_services.state_file",
            "placeholder": False,
            "not_wired": False,
            "stream_ready": False,
            "degraded": True,
            "status_reason": status_reason,
            "degraded_reason": message,
            "message": message,
            "detections": [],
            "detection_boxes": [],
            "detection_scores": [],
            "parse_error_count": 0,
            "parse_errors": [],
            "state_file": {
                "path": str(self.state_path),
                **{str(key): value for key, value in details.items() if value not in (None, "")},
            },
            "pipeline": self.config.pipeline_fields(),
            "devices": {
                "camera": self.config.camera_device,
                "hailo": self.config.hailo_device,
            },
        }


class SafeSubprocessEyeAdapter:
    """Poll native GStreamer/Hailo in a child process so plugin crashes do not kill the monitor."""

    def __init__(
        self,
        config: GStreamerHailoRealtimeConfig,
        *,
        timeout_s: float = 10.0,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.config = config
        self.timeout_s = float(timeout_s)
        self._runner = runner or subprocess.run

    def status(self) -> dict[str, Any]:
        return {
            "schema": "eihead.eye.realtime_status.v1",
            "mode": self.config.mode,
            "status": "waiting_for_frame",
            "backend": self.config.backend,
            "source": "eihead.runtime.native_services",
            "placeholder": False,
            "not_wired": False,
            "stream_ready": False,
            "degraded": False,
            "readiness_message": "native GStreamer/Hailo poll is isolated in a subprocess",
            **self._diagnostics(),
        }

    def poll(self) -> dict[str, Any]:
        config_payload = json.dumps(asdict(self.config), ensure_ascii=True, allow_nan=False)
        try:
            completed = self._runner(
                [sys.executable, "-c", _SUBPROCESS_EYE_POLL_SCRIPT],
                input=config_payload,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return self._degraded(f"native eye subprocess timed out after {self.timeout_s:.1f}s", error=str(exc))
        except Exception as exc:  # pragma: no cover - host/process dependent.
            return self._degraded(
                f"native eye subprocess failed before polling: {exc.__class__.__name__}: {exc}",
                error=str(exc),
            )
        if completed.returncode != 0:
            reason = _subprocess_error_message(completed)
            return self._degraded(
                f"native eye subprocess exited with {completed.returncode}: {reason}",
                returncode=completed.returncode,
                stderr=completed.stderr,
                stdout=completed.stdout,
            )
        try:
            return json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception as exc:
            return self._degraded(
                f"native eye subprocess returned invalid JSON: {exc.__class__.__name__}: {exc}",
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

    def _degraded(self, message: str, **details: Any) -> dict[str, Any]:
        return {
            "schema": "eihead.eye.realtime_status.v1",
            "kind": "realtime_vision_observation",
            "mode": self.config.mode,
            "status": "degraded",
            "backend": self.config.backend,
            "source": "eihead.runtime.native_services",
            "placeholder": False,
            "not_wired": False,
            "stream_ready": False,
            "degraded": True,
            "status_reason": "native_eye_subprocess_failed",
            "degraded_reason": message,
            "message": message,
            "detections": [],
            "detection_boxes": [],
            "detection_scores": [],
            "parse_error_count": 0,
            "parse_errors": [],
            "subprocess": {str(key): value for key, value in details.items() if value not in (None, "")},
            **self._diagnostics(),
        }

    def _diagnostics(self) -> dict[str, Any]:
        return {
            "pipeline": self.config.pipeline_fields(),
            "devices": {
                "camera": self.config.camera_device,
                "hailo": self.config.hailo_device,
            },
        }


def gstreamer_hailo_config_from_eihead_config(
    config: Any,
    *,
    config_path: str,
) -> GStreamerHailoRealtimeConfig:
    raw = _mapping(getattr(config, "raw", None))
    devices = _mapping(raw.get("devices"))
    camera = _mapping(devices.get("camera"))
    hailo = _mapping(devices.get("hailo"))
    vision_backend = _software_capability(config, "vision_backend")
    limits = _mapping(getattr(vision_backend, "limits", None))

    legacy_detection = _legacy_detection_config(config, config_path=config_path)
    merged_hailo = {**legacy_detection, **hailo}
    labels = _tuple_text(merged_hailo.get("labels")) or ("person", "face")
    width, height = _frame_size(camera.get("video_size") or camera.get("size"))
    return GStreamerHailoRealtimeConfig(
        camera_device=_text(camera.get("path") or getattr(getattr(config, "devices", None), "camera", ""), "/dev/video0"),
        hailo_device=_text(merged_hailo.get("path") or merged_hailo.get("device") or getattr(getattr(config, "devices", None), "hailo", ""), "/dev/hailo0"),
        hailo_device_id=_text(merged_hailo.get("device_id") or merged_hailo.get("deviceId"), ""),
        width=_int(camera.get("width"), width),
        height=_int(camera.get("height"), height),
        framerate=_int(camera.get("framerate") or camera.get("fps") or limits.get("max_fps"), 30),
        inference_width=_int(merged_hailo.get("inference_width"), 640),
        inference_height=_int(merged_hailo.get("inference_height"), 640),
        inference_format=_text(merged_hailo.get("inference_format"), "RGB"),
        sample_timeout_s=_float(merged_hailo.get("sample_timeout_s") or merged_hailo.get("sampleTimeoutS"), 5.0),
        hef_path=_text(merged_hailo.get("hef_path") or merged_hailo.get("hefPath"), ""),
        postprocess_so_path=_text(merged_hailo.get("postprocess_so_path") or merged_hailo.get("postprocessSoPath"), ""),
        postprocess_config_path=_text(
            merged_hailo.get("postprocess_config_path") or merged_hailo.get("postprocessConfigPath"),
            "",
        ),
        postprocess_function=_text(
            merged_hailo.get("postprocess_function") or merged_hailo.get("postprocessFunction"),
            "filter",
        ),
        score_threshold=_float(merged_hailo.get("score_threshold") or merged_hailo.get("scoreThreshold"), 0.3),
        labels=labels,
        model_id=_text(getattr(vision_backend, "model", ""), "hailo"),
    )


def build_native_neck_servo_adapter(config: Any | None) -> Any | None:
    if not _is_honjia_config(config):
        return None
    raw = _mapping(getattr(config, "raw", None))
    devices = _mapping(raw.get("devices"))
    neck = _mapping(devices.get("neck"))
    return build_neck_servo_adapter(
        node_id=_text(getattr(config, "node_id", ""), ""),
        bus=_int(neck.get("bus"), 1),
        addr=_int(neck.get("i2c_addr") or neck.get("addr"), 0x2B),
        servo_id=_int(neck.get("servo_id") or neck.get("servoId"), 1),
        enabled=_bool(neck.get("enabled"), True),
        mock=_bool(neck.get("mock"), False),
        hardware_verified=_bool(neck.get("hardware_verified") or neck.get("hardwareVerified"), False),
        motion_verified=_bool(neck.get("motion_verified") or neck.get("motionVerified"), False),
        motion_evidence=_text(neck.get("motion_evidence") or neck.get("motionEvidence"), ""),
    )


def build_native_voice_status(config: Any | None) -> dict[str, Any] | None:
    if not _is_honjia_config(config):
        return None
    asr = _software_capability(config, "asr")
    tts = _software_capability(config, "tts")
    if asr is None and tts is None:
        return None
    raw = _mapping(getattr(config, "raw", None))
    devices = _mapping(raw.get("devices"))
    microphone = _mapping(devices.get("microphone"))
    speaker = _mapping(devices.get("speaker"))

    return {
        "status": "degraded",
        "ear": {
            "status": "ready" if _enabled(asr) else "not_wired",
            "provider": _text(getattr(asr, "provider", ""), ""),
            "live_probe_skipped": True,
            "readiness_message": "native ASR config is present; eivoice runtime is not attached",
            "capture": {
                "status": "ready",
                "details": {
                    "device": microphone.get("device") or microphone.get("path") or "/dev/snd",
                    "sample_rate": microphone.get("sample_rate"),
                    "channels": microphone.get("channels"),
                },
            },
            "asr": {
                "status": "ready" if _enabled(asr) else "not_wired",
                "details": {
                    "provider": _text(getattr(asr, "provider", ""), ""),
                    "model": _text(getattr(asr, "model", ""), ""),
                    "model_dir": _text(getattr(asr, "model_dir", ""), ""),
                    "live_probe_skipped": True,
                },
            },
        },
        "mouth": {
            "status": "ready" if _enabled(tts) else "not_wired",
            "backend": _text(getattr(tts, "provider", ""), ""),
            "model": _text(getattr(tts, "model", ""), ""),
            "live_probe_skipped": True,
            "readiness_message": "native TTS config is present; playback runtime is not attached",
            "tts_playback": {
                "status": "ready" if _enabled(tts) else "not_wired",
                "details": {
                    "provider": _text(getattr(tts, "provider", ""), ""),
                    "model": _text(getattr(tts, "model", ""), ""),
                    "device": speaker.get("device") or speaker.get("path") or "default",
                    "live_probe_skipped": True,
                },
            },
        },
        "voice_dialogue": {
            "enabled": False,
            "running": False,
            "phase": "not_started",
            "last_status": "not_started",
            "readiness_message": "eivoice runtime loop is not attached to HeadRuntimeApp",
        },
        "realtime_audio": {"enabled": False, "running": False},
        "readiness_message": "native voice config is present; realtime eivoice loop is not attached",
    }


def _is_honjia_config(config: Any | None) -> bool:
    return _text(getattr(config, "node_id", ""), "") == "honjia"


def _vision_realtime_enabled(config: Any | None) -> bool:
    vision_backend = _software_capability(config, "vision_backend")
    if vision_backend is None or not _enabled(vision_backend):
        return False
    backend = _text(getattr(vision_backend, "backend", "") or getattr(vision_backend, "provider", ""), "")
    if backend and backend != "hailo":
        return False
    limits = _mapping(getattr(vision_backend, "limits", None))
    return _bool(limits.get("realtime"), True)


def _voice_realtime_enabled(config: Any | None) -> bool:
    asr = _software_capability(config, "asr")
    if asr is None or not _enabled(asr):
        return False
    if not _text(getattr(asr, "model_dir", ""), ""):
        return False
    provider = _text(getattr(asr, "provider", ""), "")
    if provider and provider != "sherpa_onnx":
        return False
    limits = _mapping(getattr(asr, "limits", None))
    return _bool(limits.get("streaming"), True)


def _camera_state_path(config: Any | None) -> str:
    raw = _mapping(getattr(config, "raw", None))
    devices = _mapping(raw.get("devices"))
    camera = _mapping(devices.get("camera"))
    return _text(camera.get("state_path") or camera.get("statePath"), "")


def _normalize_transport_provider(value: Any) -> str:
    normalized = _text(value, "").strip().lower()
    if normalized == "openclaw_realtime":
        return "openclaw_realtime"
    if normalized in {"", "legacy_native", "native", "subprocess", "eibrain_subprocess"}:
        return "legacy_native"
    return normalized


def _legacy_detection_config(config: Any, *, config_path: str) -> dict[str, Any]:
    legacy = getattr(config, "legacy", None)
    legacy_path = _text(getattr(legacy, "eibrain_config_path", ""), "")
    if not legacy_path:
        return {}
    path = Path(legacy_path)
    if not path.is_absolute():
        config_dir = Path(config_path).resolve().parent
        path = config_dir / path
        if not path.exists() and legacy_path.replace("\\", "/").startswith("config/"):
            path = config_dir.parent / legacy_path
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    body = _mapping(_mapping(payload).get("body"))
    organs = _mapping(body.get("organs"))
    eye = _mapping(organs.get("eye"))
    detection = _mapping(eye.get("detection"))
    driver = _mapping(detection.get("driver"))
    return {**detection, **driver}


def _software_capability(config: Any | None, name: str) -> Any | None:
    capabilities = getattr(config, "capabilities", None)
    software = getattr(capabilities, "software", None)
    if not isinstance(software, Mapping):
        return None
    return software.get(name)


def _enabled(capability: Any | None) -> bool:
    return bool(getattr(capability, "enabled", False)) if capability is not None else False


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _tuple_text(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return ()


def _frame_size(value: Any) -> tuple[int, int]:
    if isinstance(value, str) and "x" in value:
        width, height = value.lower().split("x", 1)
        return (_int(width, 640), _int(height, 480))
    return (640, 480)


def _text(value: Any, default: str) -> str:
    if value in (None, ""):
        return default
    return str(value)


def _int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, str) and value.strip().lower().startswith("0x"):
        try:
            return int(value, 16)
        except ValueError:
            return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _subprocess_error_message(completed: subprocess.CompletedProcess[str]) -> str:
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    if stderr:
        return stderr.splitlines()[-1]
    if stdout:
        return stdout.splitlines()[-1]
    return "no subprocess output"


_SUBPROCESS_EYE_POLL_SCRIPT = r"""
import json
import sys

from eihead.eye import GStreamerHailoRealtimeAdapter, GStreamerHailoRealtimeConfig

payload = json.loads(sys.stdin.read())
config = GStreamerHailoRealtimeConfig(**payload)
adapter = GStreamerHailoRealtimeAdapter.from_native_gstreamer(config)
status = adapter.poll()
print(json.dumps(status.to_dict(), ensure_ascii=False, allow_nan=False))
"""


__all__ = [
    "EyeAdapterFactory",
    "SafeSubprocessEyeAdapter",
    "StateFileEyeAdapter",
    "build_native_neck_servo_adapter",
    "build_native_provider_services",
    "build_native_voice_runtime",
    "build_native_voice_status",
    "build_realtime_eye_service",
    "gstreamer_hailo_config_from_eihead_config",
    "native_voice_loop_config_from_eihead_config",
]
