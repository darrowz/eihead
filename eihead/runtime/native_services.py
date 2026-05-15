"""Native service composition for the standalone eihead runtime."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping

from eihead.devices.neck_servo import build_neck_servo_adapter
from eihead.eye import GStreamerHailoRealtimeConfig, RealtimeEyeService


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


def build_realtime_eye_service(
    config: Any | None,
    *,
    config_path: str,
    adapter_factory: EyeAdapterFactory | None = None,
) -> RealtimeEyeService | None:
    if not _is_honjia_config(config) or not _vision_realtime_enabled(config):
        return None
    gstreamer_config = gstreamer_hailo_config_from_eihead_config(config, config_path=config_path)
    adapter = adapter_factory(gstreamer_config) if adapter_factory is not None else SafeSubprocessEyeAdapter(gstreamer_config)
    return RealtimeEyeService(adapter=adapter)


class SafeSubprocessEyeAdapter:
    """Poll native GStreamer/Hailo in a child process so plugin crashes do not kill the monitor."""

    def __init__(
        self,
        config: GStreamerHailoRealtimeConfig,
        *,
        timeout_s: float = 6.0,
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
    "build_native_neck_servo_adapter",
    "build_native_provider_services",
    "build_native_voice_status",
    "build_realtime_eye_service",
    "gstreamer_hailo_config_from_eihead_config",
]
