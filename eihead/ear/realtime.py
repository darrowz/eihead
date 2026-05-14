"""Native ear realtime status contracts.

This module is intentionally standard-library-only and does not import any legacy
`eibrain.body` runtime modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


NOT_WIRED_STATUS_REASONS = {"missing", "unavailable", "not_wired", "live_probe_skipped", "noop", "disabled"}


def _coerce_str(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes", "y"}
    return bool(value)


def _coerce_status(value: Any, *, default: str = "unknown") -> str:
    return _coerce_str(value, default=default).lower()


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _truthy_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _extract_details(obj: Any) -> dict[str, Any]:
    return _as_mapping(_as_mapping(obj).get("details", {}))


@dataclass(frozen=True, slots=True)
class EarDeviceConfig:
    device: str = "default"
    sample_rate: int = 16000
    channels: int = 1
    provider: str = "sherpa_onnx"
    model: str = ""
    streaming_vad: bool = False
    vad_frame_ms: int = 80
    vad_rms_threshold: float = 0.028
    vad_min_voice_ms: int = 160
    vad_end_silence_ms: int = 360
    vad_pre_roll_ms: int = 240
    vad_min_capture_ms: int = 0
    vad_miss_rms_threshold: float = 0.0
    transcribe_vad_miss: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "provider": self.provider,
            "model": self.model,
            "streaming_vad": self.streaming_vad,
            "vad_frame_ms": self.vad_frame_ms,
            "vad_rms_threshold": self.vad_rms_threshold,
            "vad_min_voice_ms": self.vad_min_voice_ms,
            "vad_end_silence_ms": self.vad_end_silence_ms,
            "vad_pre_roll_ms": self.vad_pre_roll_ms,
            "vad_min_capture_ms": self.vad_min_capture_ms,
            "vad_miss_rms_threshold": self.vad_miss_rms_threshold,
            "transcribe_vad_miss": self.transcribe_vad_miss,
        }


@dataclass(frozen=True, slots=True)
class EarCaptureStatus:
    status: str = "not_wired"
    phase: str = "idle"
    audio_level: float | None = None
    rms: float | None = None
    vad_triggered: bool = False
    vad_elapsed_ms: float | None = None
    capture_elapsed_ms: float | None = None
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    last_error: str | None = None
    not_wired: bool = True
    readiness_message: str = "capture path is not wired"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "phase": self.phase,
            "audio_level": self.audio_level,
            "rms": self.rms,
            "vad_triggered": self.vad_triggered,
            "vad_elapsed_ms": self.vad_elapsed_ms,
            "capture_elapsed_ms": self.capture_elapsed_ms,
            "stage_latency_ms": dict(self.stage_latency_ms),
            "last_error": self.last_error,
            "not_wired": self.not_wired,
            "readiness_message": self.readiness_message,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class AsrStatus:
    status: str = "not_wired"
    phase: str = "idle"
    transcript: str = ""
    partial: bool = False
    final: bool = False
    decode_elapsed_ms: float | None = None
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    last_error: str | None = None
    not_wired: bool = True
    readiness_message: str = "asr path is not wired"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "phase": self.phase,
            "transcript": self.transcript,
            "partial": self.partial,
            "final": self.final,
            "decode_elapsed_ms": self.decode_elapsed_ms,
            "stage_latency_ms": dict(self.stage_latency_ms),
            "last_error": self.last_error,
            "not_wired": self.not_wired,
            "readiness_message": self.readiness_message,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class EarRealtimeStatus:
    schema: str = "eihead.ear.realtime_status.v1"
    backend: str = "native_ear"
    status: str = "degraded"
    phase: str = "initializing"
    transcript: str = ""
    partial: bool = False
    final: bool = False
    audio_level: float | None = None
    rms: float | None = None
    vad_triggered: bool = False
    capture_elapsed_ms: float | None = None
    decode_elapsed_ms: float | None = None
    total_elapsed_ms: float | None = None
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    last_error: str | None = None
    not_wired: bool = True
    readiness_message: str = "ear realtime status is not wired"
    config: EarDeviceConfig = field(default_factory=EarDeviceConfig)
    capture: EarCaptureStatus | None = None
    asr: AsrStatus | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "backend": self.backend,
            "status": self.status,
            "phase": self.phase,
            "transcript": self.transcript,
            "partial": self.partial,
            "final": self.final,
            "audio_level": self.audio_level,
            "rms": self.rms,
            "vad_triggered": self.vad_triggered,
            "capture_elapsed_ms": self.capture_elapsed_ms,
            "decode_elapsed_ms": self.decode_elapsed_ms,
            "total_elapsed_ms": self.total_elapsed_ms,
            "stage_latency_ms": dict(self.stage_latency_ms),
            "last_error": self.last_error,
            "not_wired": self.not_wired,
            "readiness_message": self.readiness_message,
            "config": self.config.to_dict(),
            "capture": self.capture.to_dict() if isinstance(self.capture, EarCaptureStatus) else None,
            "asr": self.asr.to_dict() if isinstance(self.asr, AsrStatus) else None,
        }


def read_ear_config_from_legacy_details(
    *,
    capture_details: Mapping[str, Any] | None = None,
    asr_details: Mapping[str, Any] | None = None,
) -> EarDeviceConfig:
    capture = _as_mapping(capture_details)
    asr = _as_mapping(asr_details)

    if "capture_device" in capture:
        device = _coerce_str(capture.get("capture_device"), default="")
    elif "device" in capture:
        device = _coerce_str(capture.get("device"), default="")
    else:
        device = "default"
    provider = _coerce_str(asr.get("provider"), default=_coerce_str(asr.get("model_provider"), default="sherpa_onnx"))
    model = _coerce_str(
        asr.get("model"),
        default=_coerce_str(asr.get("model_dir"), default=_coerce_str(asr.get("model_name"), default="")),
    )

    return EarDeviceConfig(
        device=device,
        sample_rate=_coerce_int(capture.get("sample_rate"), default=16000),
        channels=_coerce_int(capture.get("channels"), default=1),
        provider=provider or "sherpa_onnx",
        model=model,
        streaming_vad=_coerce_bool(capture.get("streaming_vad"), default=False),
        vad_frame_ms=_coerce_int(capture.get("vad_frame_ms"), default=80),
        vad_rms_threshold=_coerce_float(capture.get("vad_rms_threshold"), default=0.028) or 0.028,
        vad_min_voice_ms=_coerce_int(capture.get("vad_min_voice_ms"), default=160),
        vad_end_silence_ms=_coerce_int(capture.get("vad_end_silence_ms"), default=360),
        vad_pre_roll_ms=_coerce_int(capture.get("vad_pre_roll_ms"), default=240),
        vad_min_capture_ms=_coerce_int(capture.get("vad_min_capture_ms"), default=0),
        vad_miss_rms_threshold=_coerce_float(capture.get("vad_miss_rms_threshold"), default=0.0) or 0.0,
        transcribe_vad_miss=_coerce_bool(capture.get("transcribe_vad_miss"), default=False),
    )


def build_ear_realtime_status(
    *,
    capture_details: Mapping[str, Any] | None = None,
    asr_details: Mapping[str, Any] | None = None,
    config: EarDeviceConfig | Mapping[str, Any] | None = None,
) -> EarRealtimeStatus:
    capture_details_m = _as_mapping(capture_details)
    asr_details_m = _as_mapping(asr_details)
    config_obj = _coerce_ear_config(config, capture_details_m, asr_details_m)

    capture_status = _build_capture_status(capture_details_m)
    asr_status = _build_asr_status(asr_details_m)

    not_wired, reasons = _build_not_wired_reason(config_obj, capture_status, asr_status)
    readiness_message = _readiness_message(not_wired, reasons, capture_status, asr_status)
    overall_status = _overall_status(not_wired, capture_status, asr_status)
    overall_phase = _overall_phase(capture_status, asr_status)

    stage_latency_ms = _ear_stage_latency_ms(capture_status, asr_status)
    total_elapsed_ms = stage_latency_ms.get(
        "total",
        _total_elapsed_ms(capture_status.capture_elapsed_ms, asr_status.decode_elapsed_ms),
    )
    transcript = asr_status.transcript
    audio_level = capture_status.audio_level
    rms = capture_status.rms
    vad_triggered = capture_status.vad_triggered

    return EarRealtimeStatus(
        status=overall_status,
        phase=overall_phase,
        transcript=transcript,
        partial=asr_status.partial,
        final=asr_status.final,
        audio_level=audio_level,
        rms=rms,
        vad_triggered=vad_triggered,
        capture_elapsed_ms=capture_status.capture_elapsed_ms,
        decode_elapsed_ms=asr_status.decode_elapsed_ms,
        total_elapsed_ms=total_elapsed_ms,
        stage_latency_ms=stage_latency_ms,
        last_error=_first_error(capture_status.last_error, asr_status.last_error),
        not_wired=not_wired,
        readiness_message=readiness_message,
        config=config_obj,
        capture=capture_status,
        asr=asr_status,
    )


def legacy_ear_details_to_status(
    snapshot: Mapping[str, Any] | None,
    *,
    config: EarDeviceConfig | Mapping[str, Any] | None = None,
) -> EarRealtimeStatus:
    snapshot_m = _as_mapping(snapshot)
    organs = _as_mapping(snapshot_m.get("organs"))
    ear = _as_mapping(organs.get("ear"))
    subfunctions = _as_mapping(ear.get("subfunctions"))
    capture = _extract_details(subfunctions.get("capture"))
    asr = _extract_details(subfunctions.get("asr"))
    return build_ear_realtime_status(
        capture_details=capture,
        asr_details=asr,
        config=_coerce_ear_config(config, capture, asr),
    )


def _coerce_ear_config(
    config: EarDeviceConfig | Mapping[str, Any] | None,
    capture_details: dict[str, Any],
    asr_details: dict[str, Any],
) -> EarDeviceConfig:
    if isinstance(config, EarDeviceConfig):
        return config
    if config is not None:
        return read_ear_config_from_legacy_details(
            capture_details=_merge_capture_defaults(capture_details, mapping=_as_mapping(config)),
            asr_details=_merge_asr_defaults(asr_details, mapping=_as_mapping(config)),
        )
    return read_ear_config_from_legacy_details(capture_details=capture_details, asr_details=asr_details)


def _merge_capture_defaults(details: dict[str, Any], *, mapping: dict[str, Any]) -> dict[str, Any]:
    merged = dict(details)
    merged.update(
        {
            key: mapping.get(key)
            for key in [
                "device",
                "capture_device",
                "sample_rate",
                "channels",
                "streaming_vad",
                "vad_frame_ms",
                "vad_rms_threshold",
                "vad_min_voice_ms",
                "vad_end_silence_ms",
                "vad_pre_roll_ms",
                "vad_min_capture_ms",
                "vad_miss_rms_threshold",
                "transcribe_vad_miss",
            ]
            if key in mapping
        }
    )
    return merged


def _merge_asr_defaults(details: dict[str, Any], *, mapping: dict[str, Any]) -> dict[str, Any]:
    merged = dict(details)
    merged.update(
        {
            key: mapping.get(key)
            for key in ["provider", "model", "model_name", "model_dir"]
            if key in mapping
        }
    )
    return merged


def _build_capture_status(details: dict[str, Any]) -> EarCaptureStatus:
    status_text = _coerce_status(details.get("status"), default="unavailable")
    phase = _capture_phase(status_text)
    audio_level = _coerce_float(details.get("dbfs"), default=None)
    rms = _coerce_float(details.get("rms_level"), default=None)
    vad_triggered = _coerce_bool(
        details.get("vad_triggered"),
        default=_coerce_bool(details.get("last_vad_triggered"), default=False),
    )
    elapsed_ms = _coerce_float(details.get("elapsed_ms"), default=None)
    if elapsed_ms is None:
        elapsed_ms = _coerce_float(details.get("capture_elapsed_ms"), default=None)
    vad_elapsed_ms = _first_float(details, "vad_elapsed_ms", "vad_latency_ms", "vad_ms")
    stage_latency_ms = _stage_latency_ms_mapping(details.get("stage_latency_ms"))
    _set_stage_latency(stage_latency_ms, "vad", vad_elapsed_ms)
    _set_stage_latency(stage_latency_ms, "capture", elapsed_ms)

    error = _truthy_text(details.get("error"))
    if not error:
        capture_stderr = _truthy_text(details.get("capture_stderr"))
        error = capture_stderr

    readiness = "capture is wired"
    if status_text in NOT_WIRED_STATUS_REASONS or status_text in {"not_wired", "unavailable"}:
        readiness = f"capture is not wired: status={status_text}"
    elif not details:
        readiness = "capture details are not available"

    return EarCaptureStatus(
        status=status_text,
        phase=phase,
        audio_level=audio_level,
        rms=rms,
        vad_triggered=vad_triggered,
        vad_elapsed_ms=vad_elapsed_ms,
        capture_elapsed_ms=elapsed_ms,
        stage_latency_ms=stage_latency_ms,
        last_error=error or None,
        readiness_message=readiness,
        details=details,
    )


def _build_asr_status(details: dict[str, Any]) -> AsrStatus:
    status_text = _coerce_status(details.get("status"), default="unavailable")
    phase = _asr_phase(status_text, transcript=_truthy_text(details.get("transcript")))
    transcript = _truthy_text(details.get("transcript"))
    decode_elapsed_ms = _coerce_float(details.get("elapsed_ms"), default=None)
    if decode_elapsed_ms is None:
        decode_elapsed_ms = _coerce_float(details.get("decode_elapsed_ms"), default=None)
    if decode_elapsed_ms is None:
        decode_elapsed_ms = _first_float(details, "asr_decode_elapsed_ms", "asr_elapsed_ms", "latency_ms")
    stage_latency_ms = _stage_latency_ms_mapping(details.get("stage_latency_ms"))
    _set_stage_latency(stage_latency_ms, "asr", decode_elapsed_ms)
    partial = _coerce_bool(details.get("partial"), default=False)
    final = _coerce_bool(details.get("final"), default=(bool(transcript) and status_text in {"transcribed", "final", "ok"}))

    error = _truthy_text(details.get("error"))
    if not error:
        asr_stderr = _truthy_text(details.get("asr_stderr"))
        error = asr_stderr

    readiness = "asr is wired"
    if status_text in NOT_WIRED_STATUS_REASONS or status_text in {"not_wired", "unavailable"}:
        readiness = f"asr is not wired: status={status_text}"
    elif not details:
        readiness = "asr details are not available"

    return AsrStatus(
        status=status_text,
        phase=phase,
        transcript=transcript,
        partial=partial,
        final=final,
        decode_elapsed_ms=decode_elapsed_ms,
        stage_latency_ms=stage_latency_ms,
        last_error=error or None,
        readiness_message=readiness,
        details=details,
    )


def _build_not_wired_reason(
    config: EarDeviceConfig,
    capture_status: EarCaptureStatus,
    asr_status: AsrStatus,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if _coerce_status(capture_status.status, default="unavailable") in {"", "unavailable", "not_wired", "live_probe_skipped"}:
        reasons.append("capture status unavailable")
    if _coerce_status(asr_status.status, default="unavailable") in {"", "unavailable", "not_wired", "live_probe_skipped"}:
        reasons.append("asr status unavailable")
    driver = _coerce_str(capture_status.details.get("driver"), default="")
    if driver == "noop":
        reasons.append("capture driver is noop")
    if not _coerce_str(config.device, default=""):
        reasons.append("no microphone device configured")
    if _coerce_status(config.provider, default="disabled") in {"", "noop", "disabled"}:
        reasons.append("no asr provider configured")
    if not _coerce_str(config.model, default=""):
        reasons.append("no asr model configured")
    if capture_status.status in {"live_probe_skipped", "skipped"}:
        reasons.append("capture probe skipped")
    if asr_status.status in {"live_probe_skipped", "skipped"}:
        reasons.append("asr probe skipped")

    return bool(reasons), reasons


def _overall_status(
    not_wired: bool,
    capture_status: EarCaptureStatus,
    asr_status: AsrStatus,
) -> str:
    if not_wired:
        return "degraded"
    if _coerce_str(capture_status.last_error, default="") or _coerce_str(asr_status.last_error, default=""):
        return "degraded"
    if asr_status.transcript:
        return "ok"
    if _coerce_str(capture_status.status, default="") in {"capture_failed", "unavailable", "capture_unavailable", "error"}:
        return "degraded"
    return "ok" if asr_status.status not in {"error", "failed", ""} else "degraded"


def _overall_phase(capture_status: EarCaptureStatus, asr_status: AsrStatus) -> str:
    if asr_status.status == "transcribed":
        return "decode"
    if capture_status.vad_triggered:
        return "capture"
    if _coerce_str(capture_status.status, default="") in {"warming_up", "warmup", "initializing"}:
        return "init"
    return "capture"


def _first_error(capture_error: str | None, asr_error: str | None) -> str | None:
    return capture_error or asr_error


def _readiness_message(
    not_wired: bool,
    reasons: list[str],
    capture_status: EarCaptureStatus,
    asr_status: AsrStatus,
) -> str:
    if not_wired:
        if reasons:
            return "ear is not wired: " + ", ".join(reasons)
        return "ear is not wired"
    if asr_status.transcript:
        return "realtime ear is ready and transcript present"
    if _coerce_str(capture_status.status, default="") in {"capture_failed", "capture_unavailable", "unavailable"}:
        return "capture path unavailable"
    if _coerce_str(asr_status.status, default="") in {"warming_up", "below_asr_threshold"}:
        return "asr is waking up"
    return "realtime ear is wired"


def _capture_phase(status: str) -> str:
    if status in {"capture_failed", "capture_unavailable", "unavailable", "error"}:
        return "capture_error"
    if status in {"warming_up", "warmup", "initializing"}:
        return "init"
    if status in {"silence", "below_asr_threshold", "observed", "waiting"}:
        return "waiting"
    return "capture"


def _asr_phase(status: str, *, transcript: str) -> str:
    if status in {"warming_up", "warming"}:
        return "decode_init"
    if status in {"transcribed", "final", "ok"} or transcript:
        return "decode"
    return "idle"


def _total_elapsed_ms(capture_elapsed_ms: float | None, decode_elapsed_ms: float | None) -> float | None:
    if capture_elapsed_ms is None and decode_elapsed_ms is None:
        return None
    capture_value = capture_elapsed_ms or 0.0
    decode_value = decode_elapsed_ms or 0.0
    return round(capture_value + decode_value, 3)


def _ear_stage_latency_ms(capture_status: EarCaptureStatus, asr_status: AsrStatus) -> dict[str, float]:
    stage_latency_ms: dict[str, float] = {}
    stage_latency_ms.update(capture_status.stage_latency_ms)
    stage_latency_ms.update(asr_status.stage_latency_ms)
    if stage_latency_ms and "total" not in stage_latency_ms:
        stage_latency_ms["total"] = round(
            sum(value for key, value in stage_latency_ms.items() if key not in {"total", "overhead"}),
            3,
        )
    return stage_latency_ms


def _stage_latency_ms_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    latencies: dict[str, float] = {}
    for key, raw in value.items():
        number = _coerce_float(raw, default=None)
        if number is not None:
            latencies[str(key)] = number
    return latencies


def _set_stage_latency(latencies: dict[str, float], key: str, value: float | None) -> None:
    if value is not None:
        latencies.setdefault(key, value)


def _first_float(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        number = _coerce_float(mapping.get(key), default=None)
        if number is not None:
            return number
    return None


__all__ = [
    "AsrStatus",
    "EarCaptureStatus",
    "EarDeviceConfig",
    "EarRealtimeStatus",
    "build_ear_realtime_status",
    "legacy_ear_details_to_status",
    "read_ear_config_from_legacy_details",
]
