"""Manual voice IO tests for the eihead monitor."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any, Callable, Mapping
import wave

import yaml


VOICE_MANUAL_TEST_SCHEMA = "eihead.monitor.voice_manual_test.v1"
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class VoiceIoConfig:
    microphone_device: str = "default"
    speaker_device: str = "default"
    sample_rate: int = 16000
    channels: int = 1


def run_voice_manual_test(
    app: Any,
    request_payload: Mapping[str, Any] | None = None,
    *,
    output_dir: str | os.PathLike[str] | None = None,
    runner: Runner | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Run a direct microphone or speaker check against configured audio devices."""

    payload = dict(request_payload or {})
    kind = _normalize_kind(payload.get("kind"))
    config = _load_voice_io_config(app)
    root = _output_dir(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run = runner or subprocess.run

    if kind == "speaker":
        return _run_speaker_test(config, payload, root, run, timestamp=timestamp)
    if kind == "microphone":
        return _run_microphone_test(config, payload, root, run, timestamp=timestamp)
    return _base_payload(kind=kind, status="error", timestamp=timestamp) | {
        "readiness_message": f"不支持的语音自测类型：{kind}",
    }


def _run_speaker_test(
    config: VoiceIoConfig,
    payload: Mapping[str, Any],
    output_dir: Path,
    runner: Runner,
    *,
    timestamp: float | None,
) -> dict[str, Any]:
    duration_s = _float_in_range(payload.get("duration_s"), default=0.7, lower=0.1, upper=5.0)
    frequency_hz = int(_float_in_range(payload.get("frequency_hz"), default=660.0, lower=120.0, upper=2000.0))
    wav_path = output_dir / "speaker-test.wav"
    _write_tone_wav(wav_path, sample_rate=config.sample_rate, duration_s=duration_s, frequency_hz=frequency_hz)
    command = ["aplay", "-q", "-D", config.speaker_device, str(wav_path)]

    result = _run_command(command, runner, timeout_s=max(3.0, duration_s + 2.0))
    base = _base_payload(kind="speaker", status="ok" if result.returncode == 0 else "error", timestamp=timestamp)
    response = base | {
        "device": config.speaker_device,
        "file": str(wav_path),
        "duration_s": duration_s,
        "frequency_hz": frequency_hz,
        "command": command[:4],
        "returncode": result.returncode,
        "stderr": (result.stderr or "").strip(),
    }
    if result.returncode == 0:
        response["readiness_message"] = f"扬声器已播放 {frequency_hz}Hz 测试音 {duration_s:.2f}s"
    else:
        response["readiness_message"] = _command_failure_message("扬声器测试失败", result)
    return response


def _run_microphone_test(
    config: VoiceIoConfig,
    payload: Mapping[str, Any],
    output_dir: Path,
    runner: Runner,
    *,
    timestamp: float | None,
) -> dict[str, Any]:
    requested_duration_s = _float_in_range(payload.get("duration_s"), default=2.0, lower=0.1, upper=10.0)
    record_duration_s = max(1, math.ceil(requested_duration_s))
    wav_path = output_dir / "microphone-test.wav"
    command = [
        "arecord",
        "-q",
        "-D",
        config.microphone_device,
        "-f",
        "S16_LE",
        "-r",
        str(config.sample_rate),
        "-c",
        str(config.channels),
        "-d",
        str(record_duration_s),
        str(wav_path),
    ]

    result = _run_command(command, runner, timeout_s=record_duration_s + 3.0)
    base = _base_payload(
        kind="microphone",
        status="ok" if result.returncode == 0 and wav_path.exists() else "error",
        timestamp=timestamp,
    )
    response = base | {
        "device": config.microphone_device,
        "file": str(wav_path),
        "requested_duration_s": requested_duration_s,
        "record_duration_s": record_duration_s,
        "command": command[:12],
        "returncode": result.returncode,
        "stderr": (result.stderr or "").strip(),
    }
    if result.returncode != 0 or not wav_path.exists():
        response["readiness_message"] = _command_failure_message("麦克风录制失败", result)
        return response

    levels = _analyze_wav(wav_path)
    response.update(levels)
    response["readiness_message"] = (
        f"麦克风已录制 {levels['duration_s']:.2f}s，"
        f"RMS {levels['rms_dbfs']:.1f} dBFS，峰值 {levels['peak_dbfs']:.1f} dBFS"
    )
    return response


def _load_voice_io_config(app: Any) -> VoiceIoConfig:
    path_value = getattr(app, "config_path", None)
    if callable(path_value):
        path_value = path_value()
    config_path = Path(str(path_value)) if path_value else None
    data: Mapping[str, Any] = {}
    if config_path and config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, Mapping):
            data = loaded

    devices = data.get("devices") if isinstance(data.get("devices"), Mapping) else {}
    microphone = devices.get("microphone") if isinstance(devices.get("microphone"), Mapping) else {}
    speaker = devices.get("speaker") if isinstance(devices.get("speaker"), Mapping) else {}
    return VoiceIoConfig(
        microphone_device=_text(microphone.get("device"), default="default"),
        speaker_device=_text(speaker.get("device"), default="default"),
        sample_rate=_int(microphone.get("sample_rate"), default=16000),
        channels=_int(microphone.get("channels"), default=1),
    )


def _output_dir(path: str | os.PathLike[str] | None) -> Path:
    if path is not None:
        return Path(path)
    configured = os.environ.get("EIHEAD_VOICE_TEST_DIR")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "eihead-voice-test"


def _normalize_kind(value: Any) -> str:
    kind = str(value or "microphone").strip().lower().replace("-", "_")
    aliases = {
        "mic": "microphone",
        "input": "microphone",
        "record": "microphone",
        "speaker": "speaker",
        "output": "speaker",
        "playback": "speaker",
        "tone": "speaker",
    }
    return aliases.get(kind, kind)


def _run_command(command: list[str], runner: Runner, *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    try:
        return runner(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(command, 127, stdout="", stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(command, 124, stdout=str(exc.stdout or ""), stderr=str(exc.stderr or exc))


def _write_tone_wav(
    path: Path,
    *,
    sample_rate: int,
    duration_s: float,
    frequency_hz: int,
) -> None:
    total_samples = max(1, int(sample_rate * duration_s))
    frames = bytearray()
    amplitude = 9000
    for index in range(total_samples):
        sample = int(amplitude * math.sin(2 * math.pi * frequency_hz * (index / sample_rate)))
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))


def _analyze_wav(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        frames = wav.readframes(frame_count)

    if sample_width != 2 or not frames:
        return {
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_s": 0.0 if not sample_rate else round(frame_count / sample_rate, 3),
            "bytes": path.stat().st_size,
            "rms_dbfs": -120.0,
            "peak_dbfs": -120.0,
        }

    sample_count = len(frames) // 2
    unpacked = struct.unpack(f"<{sample_count}h", frames)
    if not unpacked:
        rms = 0.0
        peak = 0
    else:
        rms = math.sqrt(sum(sample * sample for sample in unpacked) / len(unpacked))
        peak = max(abs(sample) for sample in unpacked)
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_s": round(frame_count / sample_rate, 3) if sample_rate else 0.0,
        "bytes": path.stat().st_size,
        "rms_dbfs": round(_dbfs(rms), 2),
        "peak_dbfs": round(_dbfs(float(peak)), 2),
    }


def _dbfs(value: float) -> float:
    if value <= 0:
        return -120.0
    return 20.0 * math.log10(value / 32768.0)


def _base_payload(*, kind: str, status: str, timestamp: float | None) -> dict[str, Any]:
    return {
        "schema": VOICE_MANUAL_TEST_SCHEMA,
        "runtime": "eihead",
        "kind": kind,
        "status": status,
        "captured_at_ts": timestamp,
    }


def _command_failure_message(prefix: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout or "").strip()
    if detail:
        return f"{prefix}：{detail}"
    return f"{prefix}：returncode={result.returncode}"


def _text(value: Any, *, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_in_range(value: Any, *, default: float, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, lower), upper)
