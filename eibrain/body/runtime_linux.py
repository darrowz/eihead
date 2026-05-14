"""Linux helpers for honjia device probing and local actuation."""

from __future__ import annotations

import hashlib
import json
import os
from urllib.error import HTTPError, URLError
from urllib import request
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def probe_sherpa_model_dir(model_dir: str) -> dict[str, object]:
    path = Path(model_dir).expanduser()
    required = ("tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx")
    missing = [name for name in required if not (path / name).exists()]
    return {
        "status": "healthy" if not missing else "degraded",
        "details": {
            "driver": "sherpa_onnx",
            "model_dir": str(path),
            "missing_files": missing,
        },
    }


def probe_faster_whisper_model(
    *,
    model_name: str,
    python_executable: str = "/usr/bin/python3",
) -> dict[str, object]:
    model_path = resolve_faster_whisper_model_path(model_name)
    exists = Path(model_path).exists()
    completed = subprocess.run(
        [python_executable, "-c", "from faster_whisper import WhisperModel; print('ok')"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = "healthy" if exists and completed.returncode == 0 else "degraded"
    return {
        "status": status,
        "details": {
            "driver": "faster_whisper",
            "model_name": model_name,
            "model_path": model_path,
            "model_exists": exists,
            "python_executable": python_executable,
            "returncode": completed.returncode,
            "stdout": (completed.stdout or "").strip(),
            "stderr": (completed.stderr or "").strip(),
        },
    }


def probe_binary_device(*, binary_name: str, device_path: str, label: str) -> dict[str, object]:
    binary = shutil.which(binary_name)
    exists = Path(device_path).exists()
    status = "healthy" if binary and exists else "degraded"
    if not binary and not exists:
        status = "unavailable"
    return {
        "status": status,
        "details": {
            "label": label,
            "binary": binary or "",
            "device": device_path,
            "device_exists": exists,
        },
    }


def map_target_x_to_angle(*, target_x: float, pan_min: int, pan_max: int) -> int:
    clipped = min(max(target_x, 0.0), 1.0)
    return int(round(pan_min + (pan_max - pan_min) * clipped))


def compute_tracking_pan_angle(
    *,
    current_angle: int,
    target_x: float,
    pan_min: int,
    pan_max: int,
    deadband: float = 0.08,
    step_gain: float = 30.0,
    max_step: int = 12,
    invert: bool = False,
) -> int:
    clipped = min(max(target_x, 0.0), 1.0)
    error = clipped - 0.5
    if invert:
        error = -error
    if abs(error) <= deadband:
        return int(max(pan_min, min(pan_max, current_angle)))
    delta = int(round(error * step_gain))
    if delta == 0:
        delta = 1 if error > 0 else -1
    delta = max(-max_step, min(max_step, delta))
    return int(max(pan_min, min(pan_max, current_angle + delta)))


def speak_text(
    *,
    text: str,
    output_device: str,
    backend: str = "espeak",
    playback_backend: str = "aplay",
    api_key: str = "",
    api_base_url: str = "https://api.minimaxi.com",
    model: str = "speech-2.8-hd",
    voice_id: str = "female-shaonv",
    audio_format: str = "wav",
    sample_rate: int = 32000,
    bitrate: int = 128000,
    channel: int = 1,
    speed: float = 1.0,
    volume: float = 1.0,
    pitch: float = 0.0,
    emotion: str = "",
    language_boost: str = "auto",
    timeout_s: int = 30,
    runner=subprocess.run,
    urlopen=request.urlopen,
    cache_dir: str | Path | None = None,
    temp_dir: str | Path | None = None,
) -> dict[str, object]:
    if backend == "minimax":
        return _speak_text_with_minimax(
            text=text,
            output_device=output_device,
            playback_backend=playback_backend,
            api_key=api_key,
            api_base_url=api_base_url,
            model=model,
            voice_id=voice_id,
            audio_format=audio_format,
            sample_rate=sample_rate,
            bitrate=bitrate,
            channel=channel,
            speed=speed,
            volume=volume,
            pitch=pitch,
            emotion=emotion,
            language_boost=language_boost,
            timeout_s=timeout_s,
            runner=runner,
            urlopen=urlopen,
            cache_dir=cache_dir,
            temp_dir=temp_dir,
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as handle:
        wav_path = Path(handle.name)
    try:
        synth = runner(
            ["espeak", "-w", str(wav_path), "--", text],
            capture_output=True,
            text=True,
            check=False,
        )
        if synth.returncode != 0:
            return {"status": "error", "details": {"stderr": synth.stderr.strip(), "stdout": synth.stdout.strip()}}
        return _play_audio_file(
            audio_path=wav_path,
            output_device=output_device,
            playback_backend=playback_backend,
            runner=runner,
            details={"output_device": output_device, "backend": backend},
        )
    finally:
        wav_path.unlink(missing_ok=True)


def probe_tts_playback(
    *,
    output_device: str,
    backend: str = "espeak",
    playback_backend: str = "aplay",
    api_key: str = "",
    api_base_url: str = "https://api.minimaxi.com",
    model: str = "speech-2.8-hd",
    voice_id: str = "female-shaonv",
) -> dict[str, object]:
    binary_name = _normalize_playback_backend(playback_backend)
    playback_probe = probe_binary_device(binary_name=binary_name, device_path="/dev/snd", label=f"speaker:{output_device}")
    details = dict(playback_probe.get("details", {}))
    details.update(
        {
            "backend": backend,
            "playback_backend": binary_name,
            "output_device": output_device,
            "api_base_url": api_base_url,
            "model": model,
            "voice_id": voice_id,
            "api_key_present": bool(api_key),
        }
    )
    if backend != "minimax":
        return {
            "status": playback_probe["status"],
            "details": details,
        }
    if not api_key:
        status = "degraded" if playback_probe["status"] == "healthy" else playback_probe["status"]
        details["reason"] = "missing_minimax_api_key"
        return {"status": status, "details": details}
    return {
        "status": playback_probe["status"],
        "details": details,
    }


def _speak_text_with_minimax(
    *,
    text: str,
    output_device: str,
    playback_backend: str,
    api_key: str,
    api_base_url: str,
    model: str,
    voice_id: str,
    audio_format: str,
    sample_rate: int,
    bitrate: int,
    channel: int,
    speed: float,
    volume: float,
    pitch: float,
    emotion: str,
    language_boost: str,
    timeout_s: int,
    runner,
    urlopen,
    cache_dir: str | Path | None,
    temp_dir: str | Path | None,
) -> dict[str, object]:
    if not api_key:
        return {
            "status": "error",
            "details": {
                "backend": "minimax",
                "reason": "missing_minimax_api_key",
                "output_device": output_device,
                "playback_backend": playback_backend,
            },
        }
    if audio_format.lower() != "wav":
        return {
            "status": "error",
            "details": {
                "backend": "minimax",
                "reason": "unsupported_playback_format",
                "audio_format": audio_format,
                "output_device": output_device,
                "playback_backend": playback_backend,
            },
        }

    cache_path = _minimax_audio_cache_path(
        cache_dir=cache_dir,
        text=text,
        model=model,
        voice_id=voice_id,
        audio_format=audio_format,
        sample_rate=sample_rate,
        bitrate=bitrate,
        channel=channel,
        speed=speed,
        volume=volume,
        pitch=pitch,
        emotion=emotion,
        language_boost=language_boost,
    ) if cache_dir else None
    if cache_path is not None and cache_path.exists() and cache_path.stat().st_size > 0:
        return _play_minimax_audio_file(
            audio_path=cache_path,
            output_device=output_device,
            playback_backend=playback_backend,
            runner=runner,
            details={
                "backend": "minimax",
                "playback_backend": playback_backend,
                "model": model,
                "voice_id": voice_id,
                "audio_format": audio_format.lower(),
                "cache_hit": True,
                "cache_key": cache_path.stem,
                "cache_path": str(cache_path),
            },
        )

    audio_path: Path | None = None
    delete_audio_path = False
    cache_error = ""
    try:
        synthesized = synthesize_minimax_speech(
            text=text,
            api_key=api_key,
            api_base_url=api_base_url,
            model=model,
            voice_id=voice_id,
            audio_format=audio_format,
            sample_rate=sample_rate,
            bitrate=bitrate,
            channel=channel,
            speed=speed,
            volume=volume,
            pitch=pitch,
            emotion=emotion,
            language_boost=language_boost,
            timeout_s=timeout_s,
            urlopen=urlopen,
        )
        if synthesized.get("status") != "ok":
            details = dict(synthesized.get("details", {}))
            details["output_device"] = output_device
            return {"status": "error", "details": details}

        audio_bytes = synthesized["audio_bytes"]
        if cache_path is not None:
            try:
                _write_audio_cache(cache_path=cache_path, audio_bytes=audio_bytes)
                audio_path = cache_path
            except OSError as exc:
                cache_error = str(exc)
        if audio_path is None:
            audio_path = _write_temp_audio(audio_bytes=audio_bytes, audio_format=audio_format, temp_dir=temp_dir)
            delete_audio_path = True

        synth_details = dict(synthesized.get("details", {}))
        synth_details.update(
            {
                "backend": "minimax",
                "output_device": output_device,
                "playback_backend": playback_backend,
                "cache_hit": False,
                "cache_key": cache_path.stem if cache_path is not None else "",
                "cache_path": str(cache_path) if cache_path is not None else "",
            }
        )
        if cache_error:
            synth_details["cache_error"] = cache_error
        return _play_minimax_audio_file(
            audio_path=audio_path,
            output_device=output_device,
            playback_backend=playback_backend,
            runner=runner,
            details=synth_details,
        )
    finally:
        if delete_audio_path and audio_path is not None:
            audio_path.unlink(missing_ok=True)



def _minimax_audio_cache_path(
    *,
    cache_dir: str | Path | None,
    text: str,
    model: str,
    voice_id: str,
    audio_format: str,
    sample_rate: int,
    bitrate: int,
    channel: int,
    speed: float,
    volume: float,
    pitch: float,
    emotion: str,
    language_boost: str,
) -> Path | None:
    if not cache_dir:
        return None
    cache_key_payload = {
        "text": text,
        "model": model,
        "voice_id": voice_id,
        "audio_format": audio_format.lower(),
        "sample_rate": sample_rate,
        "bitrate": bitrate,
        "channel": channel,
        "speed": _normalize_minimax_number(speed),
        "volume": _normalize_minimax_number(volume),
        "pitch": _normalize_minimax_number(pitch),
        "emotion": emotion,
        "language_boost": language_boost,
    }
    encoded = json.dumps(cache_key_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cache_key = hashlib.sha256(encoded).hexdigest()
    return Path(cache_dir).expanduser() / f"{cache_key}.{audio_format.lower()}"


def _write_audio_cache(*, cache_path: Path, audio_bytes: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=cache_path.suffix, delete=False, dir=cache_path.parent) as handle:
            handle.write(audio_bytes)
            temp_path = Path(handle.name)
        temp_path.replace(cache_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _write_temp_audio(*, audio_bytes: bytes, audio_format: str, temp_dir: str | Path | None) -> Path:
    with tempfile.NamedTemporaryFile(suffix=f".{audio_format.lower()}", delete=False, dir=temp_dir) as handle:
        handle.write(audio_bytes)
        return Path(handle.name)


def _play_minimax_audio_file(
    *,
    audio_path: Path,
    output_device: str,
    playback_backend: str,
    runner,
    details: dict[str, object],
) -> dict[str, object]:
    return _play_audio_file(
        audio_path=audio_path,
        output_device=output_device,
        playback_backend=playback_backend,
        runner=runner,
        details=details,
    )


def _play_audio_file(
    *,
    audio_path: Path,
    output_device: str,
    playback_backend: str,
    runner,
    details: dict[str, object],
) -> dict[str, object]:
    command = build_audio_playback_command(
        audio_path=audio_path,
        output_device=output_device,
        playback_backend=playback_backend,
    )
    playback_kwargs = {
        "capture_output": True,
        "text": True,
        "check": False,
    }
    playback_env = _playback_env(playback_backend)
    if playback_env is not None:
        playback_kwargs["env"] = playback_env
    playback = runner(command, **playback_kwargs)
    playback_details = dict(details)
    playback_details.update(
        {
            "audio_path": str(audio_path),
            "playback_backend": _normalize_playback_backend(playback_backend),
            "playback_command": command[0],
            "pipewire_runtime_dir": (playback_env or {}).get("XDG_RUNTIME_DIR", ""),
            "returncode": playback.returncode,
            "stdout": playback.stdout.strip(),
            "stderr": playback.stderr.strip(),
        }
    )
    return {
        "status": "ok" if playback.returncode == 0 else "error",
        "details": playback_details,
    }


def build_audio_playback_command(*, audio_path: Path, output_device: str, playback_backend: str = "aplay") -> list[str]:
    backend = _normalize_playback_backend(playback_backend)
    if backend == "pw-play":
        command = ["pw-play"]
        if output_device:
            command.extend(["--target", output_device])
        command.append(str(audio_path))
        return command
    return ["aplay", "-D", output_device, str(audio_path)]


def _normalize_playback_backend(value: str) -> str:
    normalized = str(value or "aplay").strip().lower()
    if normalized in {"pipewire", "pwplay", "pw-play"}:
        return "pw-play"
    return "aplay"


def _playback_env(playback_backend: str) -> dict[str, str] | None:
    if _normalize_playback_backend(playback_backend) != "pw-play":
        return None
    env = dict(os.environ)
    if env.get("XDG_RUNTIME_DIR"):
        return env
    getuid = getattr(os, "getuid", None)
    if callable(getuid):
        runtime_dir = f"/run/user/{getuid()}"
        if Path(runtime_dir).exists():
            env["XDG_RUNTIME_DIR"] = runtime_dir
    return env


def synthesize_minimax_speech(
    *,
    text: str,
    api_key: str,
    api_base_url: str = "https://api.minimaxi.com",
    model: str = "speech-2.8-hd",
    voice_id: str = "female-shaonv",
    audio_format: str = "wav",
    sample_rate: int = 32000,
    bitrate: int = 128000,
    channel: int = 1,
    speed: float = 1.0,
    volume: float = 1.0,
    pitch: float = 0.0,
    emotion: str = "",
    language_boost: str = "auto",
    timeout_s: int = 30,
    urlopen=request.urlopen,
) -> dict[str, object]:
    if not text.strip():
        return {
            "status": "error",
            "details": {
                "backend": "minimax",
                "reason": "empty_text",
            },
        }
    voice_setting: dict[str, object] = {
        "voice_id": voice_id,
        "speed": _normalize_minimax_number(speed),
        "vol": _normalize_minimax_number(volume),
        "pitch": _normalize_minimax_number(pitch),
    }
    if emotion:
        voice_setting["emotion"] = emotion
    payload = {
        "model": model,
        "text": text,
        "stream": False,
        "voice_setting": voice_setting,
        "audio_setting": {
            "sample_rate": sample_rate,
            "bitrate": bitrate,
            "format": audio_format.lower(),
            "channel": channel,
        },
        "language_boost": language_boost,
        "subtitle_enable": False,
    }
    endpoint = _minimax_t2a_url(api_base_url)
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urlopen(req, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", "replace")
        return {
            "status": "error",
            "details": {
                "backend": "minimax",
                "reason": "http_error",
                "endpoint": endpoint,
                "status_code": exc.code,
                "body": error_body,
            },
        }
    except (URLError, OSError) as exc:
        return {
            "status": "error",
            "details": {
                "backend": "minimax",
                "reason": "request_failed",
                "endpoint": endpoint,
                "error": str(exc),
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
                    "body": body,
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
                    "body": body,
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
                "error": str(exc),
                "body": body,
            },
        }

    extra_info = parsed.get("extra_info", {})
    if not isinstance(extra_info, dict):
        extra_info = {}
    return {
        "status": "ok",
        "audio_bytes": audio_bytes,
        "details": {
            "backend": "minimax",
            "endpoint": endpoint,
            "model": model,
            "voice_id": voice_id,
            "trace_id": parsed.get("trace_id"),
            "audio_size": extra_info.get("audio_size", len(audio_bytes)),
            "audio_length": extra_info.get("audio_length"),
            "audio_sample_rate": extra_info.get("audio_sample_rate", sample_rate),
            "audio_format": extra_info.get("audio_format", audio_format.lower()),
            "audio_channel": extra_info.get("audio_channel", channel),
            "usage_characters": extra_info.get("usage_characters"),
        },
    }


def _minimax_t2a_url(api_base_url: str) -> str:
    endpoint = api_base_url.rstrip("/")
    if endpoint.endswith("/v1/t2a_v2"):
        return endpoint
    return f"{endpoint}/v1/t2a_v2"


def _normalize_minimax_number(value: float | int) -> float | int:
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def move_gimbal(
    *,
    target_name: str,
    servo_id: int,
    home_angle: int = 90,
    target_angle: int | None = None,
    target_x: float | None = None,
    pan_min: int = 40,
    pan_max: int = 140,
    driver=None,
) -> dict[str, object]:
    if driver is None:
        raise RuntimeError("gimbal driver is required")
    if target_angle is not None:
        angle = int(max(pan_min, min(pan_max, target_angle)))
    elif target_x is None:
        angle = home_angle
    else:
        angle = map_target_x_to_angle(target_x=target_x, pan_min=pan_min, pan_max=pan_max)
    payload = driver.ctrl_servo(angle, servo_id=servo_id)
    return {
        "status": "ok",
        "details": {
            "target_name": target_name,
            "servo_id": servo_id,
            "angle": angle,
            "payload": payload,
        },
    }


def capture_frame(
    *,
    device: str,
    output_path: str | Path,
    input_format: str = "",
    video_size: str = "",
    timeout_s: float = 5.0,
    runner=subprocess.run,
) -> dict[str, object]:
    frame_path = Path(output_path)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "v4l2",
    ]
    if input_format:
        command.extend(["-input_format", input_format])
    if video_size:
        command.extend(["-video_size", video_size])
    command.extend([
        "-i",
        device,
        "-frames:v",
        "1",
        "-y",
        str(frame_path),
    ])
    try:
        completed = runner(command, capture_output=True, text=True, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "details": {
                "device": device,
                "output_path": str(frame_path),
                "reason": "capture_timeout",
                "timeout_s": timeout_s,
            },
        }
    return {
        "status": "ok" if completed.returncode == 0 and frame_path.exists() else "error",
        "details": {
            "device": device,
            "output_path": str(frame_path),
            "returncode": completed.returncode,
            "stdout": (completed.stdout or "").strip(),
            "stderr": (completed.stderr or "").strip(),
        },
    }


def compare_frame_hashes(left_path: str | Path, right_path: str | Path) -> dict[str, object]:
    left = Path(left_path)
    right = Path(right_path)
    left_hash = hashlib.sha256(left.read_bytes()).hexdigest()
    right_hash = hashlib.sha256(right.read_bytes()).hexdigest()
    same_hash = left_hash == right_hash
    return {
        "status": "unchanged" if same_hash else "changed",
        "details": {
            "left_path": str(left),
            "right_path": str(right),
            "left_hash": left_hash,
            "right_hash": right_hash,
            "same_hash": same_hash,
            "left_size": left.stat().st_size,
            "right_size": right.stat().st_size,
        },
    }


def run_hailo_detection(
    *,
    post_process_file: str,
    timeout_s: int = 8,
    runner=subprocess.run,
) -> dict[str, object]:
    command = [
        "timeout",
        f"{timeout_s}s",
        "rpicam-hello",
        "--nopreview",
        "--post-process-file",
        post_process_file,
        "--verbose",
        "2",
    ]
    completed = runner(command, capture_output=True, text=True, check=False)
    combined_output = "\n".join(
        part.strip()
        for part in (completed.stdout or "", completed.stderr or "")
        if part.strip()
    )
    lowered = combined_output.lower()
    status = "ok" if completed.returncode == 0 else "error"
    reason = ""
    if "adding camera" in lowered and "no cameras available" in lowered:
        status = "degraded"
        reason = "uvc_camera_not_usable_by_rpicam"
    elif "no cameras available" in lowered:
        status = "degraded"
        reason = "rpicam_no_cameras_available"
    elif completed.returncode == 124:
        status = "ok"
        reason = "timed_out_after_start"
    return {
        "status": status,
        "details": {
            "post_process_file": post_process_file,
            "returncode": completed.returncode,
            "reason": reason,
            "stdout": (completed.stdout or "").strip(),
            "stderr": (completed.stderr or "").strip(),
            "combined_output": combined_output,
        },
    }


def parse_hailo_nms_output(
    raw_output,
    *,
    class_labels: list[str] | None = None,
    score_threshold: float = 0.0,
) -> list[dict[str, object]]:
    labels = class_labels or []
    detections: list[dict[str, object]] = []
    for batch_index, batch in enumerate(raw_output or []):
        if not isinstance(batch, list):
            continue
        for class_id, class_detections in enumerate(batch):
            if class_detections is None:
                continue
            for row in class_detections:
                values = row.tolist() if hasattr(row, "tolist") else list(row)
                if len(values) < 5:
                    continue
                y_min, x_min, y_max, x_max, score = (float(value) for value in values[:5])
                if score < score_threshold:
                    continue
                detections.append(
                    {
                        "batch_index": batch_index,
                        "class_id": class_id,
                        "label": labels[class_id] if class_id < len(labels) else f"class_{class_id}",
                        "score": round(score, 6),
                        "bbox": {
                            "x_min": round(x_min, 6),
                            "y_min": round(y_min, 6),
                            "x_max": round(x_max, 6),
                            "y_max": round(y_max, 6),
                        },
                    }
                )
    detections.sort(key=lambda item: float(item["score"]), reverse=True)
    return detections


def run_hailo_frame_inference(
    *,
    image_path: str | Path,
    hef_path: str,
    labels: list[str] | None = None,
    score_threshold: float = 0.3,
) -> dict[str, object]:
    try:
        import numpy as np  # type: ignore
        from hailo_platform import (  # type: ignore
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            HEF,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )
    except Exception as exc:
        return _run_hailo_frame_inference_with_system_python(
            image_path=image_path,
            hef_path=hef_path,
            labels=labels,
            score_threshold=score_threshold,
            import_error=str(exc),
        )

    frame_path = Path(image_path)
    if not frame_path.exists():
        return {
            "status": "error",
            "details": {
                "reason": "image_load_failed",
                "image_path": str(frame_path),
                "hef_path": hef_path,
            },
        }

    hef = HEF(hef_path)
    input_info = hef.get_input_vstream_infos()[0]
    height, width = tuple(input_info.shape[:2])
    decode = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(frame_path),
            "-vf",
            f"scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    expected_size = width * height * 3
    if decode.returncode != 0 or len(decode.stdout) != expected_size:
        return {
            "status": "error",
            "details": {
                "reason": "image_decode_failed",
                "image_path": str(frame_path),
                "hef_path": hef_path,
                "returncode": decode.returncode,
                "stderr": decode.stderr.decode("utf-8", "replace").strip(),
                "stdout_size": len(decode.stdout),
                "expected_size": expected_size,
            },
        }
    resized = np.frombuffer(decode.stdout, dtype=np.uint8).reshape((height, width, 3)).astype(np.float32)

    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        input_params = InputVStreamParams.make_from_network_group(
            network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )
        output_params = OutputVStreamParams.make_from_network_group(
            network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )
        with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                result = infer_pipeline.infer({input_info.name: np.expand_dims(resized, axis=0)})

    output_name, raw_output = next(iter(result.items()))
    detections = parse_hailo_nms_output(
        raw_output,
        class_labels=labels or ["person", "face"],
        score_threshold=score_threshold,
    )
    return {
        "status": "ok",
        "details": {
            "image_path": str(frame_path),
            "hef_path": hef_path,
            "input_name": input_info.name,
            "output_name": output_name,
            "model_shape": [int(height), int(width), int(input_info.shape[2])],
            "detection_count": len(detections),
            "detections": detections,
        },
    }


def _run_hailo_frame_inference_with_system_python(
    *,
    image_path: str | Path,
    hef_path: str,
    labels: list[str] | None,
    score_threshold: float,
    import_error: str,
) -> dict[str, object]:
    payload = {
        "image_path": str(image_path),
        "hef_path": hef_path,
        "labels": labels or ["person", "face"],
        "score_threshold": score_threshold,
    }
    script = r"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from hailo_platform import (
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    HEF,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)


def parse_hailo_nms_output(raw_output, class_labels, score_threshold):
    detections = []
    for batch_index, batch in enumerate(raw_output or []):
        if not isinstance(batch, list):
            continue
        for class_id, class_detections in enumerate(batch):
            if class_detections is None:
                continue
            for row in class_detections:
                values = row.tolist() if hasattr(row, "tolist") else list(row)
                if len(values) < 5:
                    continue
                y_min, x_min, y_max, x_max, score = (float(value) for value in values[:5])
                if score < score_threshold:
                    continue
                detections.append(
                    {
                        "batch_index": batch_index,
                        "class_id": class_id,
                        "label": class_labels[class_id] if class_id < len(class_labels) else f"class_{class_id}",
                        "score": round(score, 6),
                        "bbox": {
                            "x_min": round(x_min, 6),
                            "y_min": round(y_min, 6),
                            "x_max": round(x_max, 6),
                            "y_max": round(y_max, 6),
                        },
                    }
                )
    detections.sort(key=lambda item: float(item["score"]), reverse=True)
    return detections


payload = json.loads(sys.stdin.read())
frame_path = Path(payload["image_path"])
hef_path = payload["hef_path"]
labels = payload["labels"]
score_threshold = float(payload["score_threshold"])

hef = HEF(hef_path)
input_info = hef.get_input_vstream_infos()[0]
height, width = tuple(input_info.shape[:2])
decode = subprocess.run(
    [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(frame_path),
        "-vf",
        f"scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ],
    capture_output=True,
    check=False,
)
expected_size = width * height * 3
if decode.returncode != 0 or len(decode.stdout) != expected_size:
    print(
        json.dumps(
            {
                "status": "error",
                "details": {
                    "reason": "image_decode_failed",
                    "image_path": str(frame_path),
                    "hef_path": hef_path,
                    "returncode": decode.returncode,
                    "stderr": decode.stderr.decode("utf-8", "replace").strip(),
                    "stdout_size": len(decode.stdout),
                    "expected_size": expected_size,
                },
            }
        )
    )
    raise SystemExit(0)
resized = np.frombuffer(decode.stdout, dtype=np.uint8).reshape((height, width, 3)).astype(np.float32)
with VDevice() as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
    input_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    with InferVStreams(network_group, input_params, output_params) as infer_pipeline:
        with network_group.activate(network_group_params):
            result = infer_pipeline.infer({input_info.name: np.expand_dims(resized, axis=0)})
output_name, raw_output = next(iter(result.items()))
print(
    json.dumps(
        {
            "status": "ok",
            "details": {
                "image_path": str(frame_path),
                "hef_path": hef_path,
                "input_name": input_info.name,
                "output_name": output_name,
                "model_shape": [int(height), int(width), int(input_info.shape[2])],
                "detection_count": len(parse_hailo_nms_output(raw_output, labels, score_threshold)),
                "detections": parse_hailo_nms_output(raw_output, labels, score_threshold),
            },
        }
    )
)
"""
    completed = subprocess.run(
        ["/usr/bin/python3", "-c", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        try:
            return json.loads(completed.stdout)
        except Exception:
            pass
    return {
        "status": "error",
        "details": {
            "reason": "hailo_runtime_unavailable",
            "error": import_error,
            "fallback_returncode": completed.returncode,
            "fallback_stdout": completed.stdout.strip(),
            "fallback_stderr": completed.stderr.strip(),
            "image_path": str(image_path),
            "hef_path": hef_path,
        },
    }


def transcribe_pcm_with_sherpa_subprocess(
    *,
    pcm_bytes: bytes,
    model_dir: str,
    sample_rate: int,
    channels: int,
    model_type: str | None = None,
    chunk_bytes: int = 4096,
    python_executable: str | None = None,
    timeout_s: int = 20,
) -> dict[str, object]:
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as handle:
        raw_path = Path(handle.name)
        handle.write(pcm_bytes)
    try:
        script = r"""
import json
from pathlib import Path
from eibrain.body.sherpa_streaming import SherpaOnnxStreamingRecognizer

payload = json.loads(Path("__PAYLOAD_PATH__").read_text(encoding="utf-8"))
pcm_bytes = Path(payload["pcm_path"]).read_bytes()
recognizer = SherpaOnnxStreamingRecognizer(
    model_dir=payload["model_dir"],
    model_type=payload.get("model_type"),
)
chunk_bytes = int(payload.get("chunk_bytes", 4096))
pcm_chunks = [
    pcm_bytes[index : index + chunk_bytes]
    for index in range(0, len(pcm_bytes), chunk_bytes)
    if pcm_bytes[index : index + chunk_bytes]
]
text = recognizer.transcribe(
    pcm_chunks,
    sample_rate=int(payload["sample_rate"]),
    channels=int(payload["channels"]),
)
print(json.dumps({"status": "ok", "details": {"text": text}}, ensure_ascii=False))
"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as payload_handle:
            payload_path = Path(payload_handle.name)
            json.dump(
                {
                    "pcm_path": str(raw_path),
                    "model_dir": model_dir,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "model_type": model_type,
                    "chunk_bytes": chunk_bytes,
                },
                payload_handle,
                ensure_ascii=False,
            )
        script = script.replace("__PAYLOAD_PATH__", str(payload_path))
        completed = subprocess.run(
            [python_executable or sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        if completed.returncode == 0:
            try:
                return json.loads(completed.stdout)
            except json.JSONDecodeError:
                pass
        return {
            "status": "error",
            "details": {
                "reason": "sherpa_subprocess_failed",
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            },
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "details": {
                "reason": "sherpa_subprocess_timeout",
                "timeout_s": timeout_s,
            },
        }
    finally:
        raw_path.unlink(missing_ok=True)
        if "payload_path" in locals():
            payload_path.unlink(missing_ok=True)


def resolve_faster_whisper_model_path(model_name: str) -> str:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate)
    repo_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
        if snapshots:
            return str(snapshots[0])
    return model_name


def transcribe_pcm_with_faster_whisper_subprocess(
    *,
    pcm_bytes: bytes,
    model_name: str,
    sample_rate: int,
    channels: int,
    language: str = "zh",
    compute_type: str = "int8",
    beam_size: int = 1,
    vad_filter: bool = False,
    python_executable: str | None = None,
    timeout_s: int = 30,
) -> dict[str, object]:
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as handle:
        raw_path = Path(handle.name)
        handle.write(pcm_bytes)
    try:
        script = r"""
import json
import wave
from pathlib import Path

from faster_whisper import WhisperModel

payload = json.loads(Path("__PAYLOAD_PATH__").read_text(encoding="utf-8"))
pcm_path = Path(payload["pcm_path"])
wav_path = pcm_path.with_suffix(".wav")
with wave.open(str(wav_path), "wb") as wav_file:
    wav_file.setnchannels(int(payload["channels"]))
    wav_file.setsampwidth(2)
    wav_file.setframerate(int(payload["sample_rate"]))
    wav_file.writeframes(pcm_path.read_bytes())

model = WhisperModel(
    payload["model_path"],
    device="cpu",
    compute_type=payload.get("compute_type", "int8"),
)
segments, info = model.transcribe(
    str(wav_path),
    language=payload.get("language") or None,
    beam_size=int(payload.get("beam_size", 1)),
    vad_filter=bool(payload.get("vad_filter", False)),
)
text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
print(
    json.dumps(
        {
            "status": "ok",
            "details": {
                "text": text,
                "language": getattr(info, "language", payload.get("language", "")),
                "duration": getattr(info, "duration", 0.0),
                "duration_after_vad": getattr(info, "duration_after_vad", 0.0),
                "model_path": payload["model_path"],
            },
        },
        ensure_ascii=False,
    )
)
"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as payload_handle:
            payload_path = Path(payload_handle.name)
            json.dump(
                {
                    "pcm_path": str(raw_path),
                    "model_path": resolve_faster_whisper_model_path(model_name),
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "language": language,
                    "compute_type": compute_type,
                "beam_size": beam_size,
                "vad_filter": vad_filter,
            },
                payload_handle,
                ensure_ascii=False,
            )
        script = script.replace("__PAYLOAD_PATH__", str(payload_path))
        completed = subprocess.run(
            [python_executable or sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        if completed.returncode == 0:
            try:
                return json.loads(completed.stdout)
            except json.JSONDecodeError:
                pass
        return {
            "status": "error",
            "details": {
                "stdout": (completed.stdout or "").strip(),
                "stderr": (completed.stderr or "").strip(),
                "returncode": completed.returncode,
                "model_path": resolve_faster_whisper_model_path(model_name),
            },
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "details": {
                "reason": "faster_whisper_timeout",
                "timeout_s": timeout_s,
                "model_path": resolve_faster_whisper_model_path(model_name),
            },
        }
    finally:
        raw_path.unlink(missing_ok=True)
        if "payload_path" in locals():
            payload_path.unlink(missing_ok=True)
        raw_path.with_suffix(".wav").unlink(missing_ok=True)
