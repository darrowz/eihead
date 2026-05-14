"""Command-driver entrypoints for honjia local device operations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from urllib import request

from eibrain.body.raspbot_driver import RaspbotDriver
from eibrain.body.runtime_linux import capture_frame
from eibrain.body.runtime_linux import compare_frame_hashes
from eibrain.body.runtime_linux import move_gimbal
from eibrain.body.runtime_linux import probe_binary_device
from eibrain.body.runtime_linux import probe_faster_whisper_model
from eibrain.body.runtime_linux import probe_tts_playback
from eibrain.body.runtime_linux import run_hailo_frame_inference
from eibrain.body.runtime_linux import run_hailo_detection
from eibrain.body.runtime_linux import probe_sherpa_model_dir
from eibrain.body.runtime_linux import speak_text
from eibrain.body.pan_motion_proof import compare_frame_paths
from eibrain.body.pan_motion_proof import summarize_pan_motion_pairs
from eibrain.body.pan_motion_proof import write_pan_motion_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="honjia local device operations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    binary_probe = subparsers.add_parser("probe-binary-device")
    binary_probe.add_argument("--binary", required=True)
    binary_probe.add_argument("--device", required=True)
    binary_probe.add_argument("--label", required=True)

    sherpa_probe = subparsers.add_parser("probe-sherpa-model")
    sherpa_probe.add_argument("--model-dir", required=True)

    faster_whisper_probe = subparsers.add_parser("probe-faster-whisper-model")
    faster_whisper_probe.add_argument("--model-name", required=True)
    faster_whisper_probe.add_argument("--python-executable", default="/usr/bin/python3")

    speaker_probe = subparsers.add_parser("probe-speaker")
    speaker_probe.add_argument("--output-device", required=True)
    speaker_probe.add_argument("--backend", default="espeak")
    speaker_probe.add_argument("--playback-backend", default="aplay")
    speaker_probe.add_argument("--api-key-env", default="MINIMAX_API_KEY")
    speaker_probe.add_argument("--api-base-url", default="https://api.minimaxi.com")
    speaker_probe.add_argument("--model", default="speech-2.8-hd")
    speaker_probe.add_argument("--voice-id", default="female-shaonv")

    speak = subparsers.add_parser("speak")
    speak.add_argument("--output-device", required=True)
    speak.add_argument("--backend", default="espeak")
    speak.add_argument("--playback-backend", default="aplay")
    speak.add_argument("--api-key-env", default="MINIMAX_API_KEY")
    speak.add_argument("--api-base-url", default="https://api.minimaxi.com")
    speak.add_argument("--model", default="speech-2.8-hd")
    speak.add_argument("--voice-id", default="female-shaonv")
    speak.add_argument("--audio-format", default="wav")
    speak.add_argument("--sample-rate", type=int, default=32000)
    speak.add_argument("--bitrate", type=int, default=128000)
    speak.add_argument("--channel", type=int, default=1)
    speak.add_argument("--speed", type=float, default=1.0)
    speak.add_argument("--volume", type=float, default=1.0)
    speak.add_argument("--pitch", type=float, default=0.0)
    speak.add_argument("--emotion", default="")
    speak.add_argument("--language-boost", default="auto")
    speak.add_argument("--timeout-s", type=int, default=30)
    speak.add_argument("--cache-dir", default="")

    gimbal = subparsers.add_parser("move-gimbal")
    gimbal.add_argument("--servo-id", type=int, default=1)
    gimbal.add_argument("--home-angle", type=int, default=90)

    pan_motion = subparsers.add_parser("verify-pan-motion")
    pan_motion.add_argument("--servo-id", type=int, default=1)
    pan_motion.add_argument("--center-angle", type=int, default=90)
    pan_motion.add_argument("--left-angle", type=int, default=75)
    pan_motion.add_argument("--right-angle", type=int, default=105)
    pan_motion.add_argument("--settle-s", type=float, default=1.2)
    pan_motion.add_argument("--frame-url", default="http://127.0.0.1:18080/vision/latest.jpg")
    pan_motion.add_argument("--output-dir", default="/tmp/eibrain-pan-proof")
    pan_motion.add_argument("--min-shift-px", type=float, default=20.0)
    pan_motion.add_argument("--max-return-shift-px", type=float, default=5.0)

    capture = subparsers.add_parser("capture-frame")
    capture.add_argument("--device", required=True)
    capture.add_argument("--output-path", required=True)

    compare = subparsers.add_parser("compare-frames")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)

    hailo = subparsers.add_parser("hailo-camera-detect")
    hailo.add_argument(
        "--post-process-file",
        default="/usr/share/rpi-camera-assets/hailo_yolov5_personface.json",
    )
    hailo.add_argument("--timeout-s", type=int, default=8)

    hailo_frame = subparsers.add_parser("hailo-frame-infer")
    hailo_frame.add_argument("--image-path", required=True)
    hailo_frame.add_argument("--hef-path", default="/usr/share/hailo-models/yolov5s_personface_h8l.hef")
    hailo_frame.add_argument("--score-threshold", type=float, default=0.3)

    args = parser.parse_args()
    if args.command == "probe-binary-device":
        result = probe_binary_device(binary_name=args.binary, device_path=args.device, label=args.label)
    elif args.command == "probe-sherpa-model":
        result = probe_sherpa_model_dir(args.model_dir)
    elif args.command == "probe-faster-whisper-model":
        result = probe_faster_whisper_model(model_name=args.model_name, python_executable=args.python_executable)
    elif args.command == "probe-speaker":
        result = probe_tts_playback(
            output_device=args.output_device,
            backend=args.backend,
            playback_backend=args.playback_backend,
            api_key=os.environ.get(args.api_key_env, ""),
            api_base_url=args.api_base_url,
            model=args.model,
            voice_id=args.voice_id,
        )
    elif args.command == "speak":
        payload = json.loads(sys.stdin.read() or "{}")
        result = speak_text(
            text=str(payload.get("payload", {}).get("text", "")),
            output_device=args.output_device,
            backend=args.backend,
            playback_backend=args.playback_backend,
            api_key=os.environ.get(args.api_key_env, ""),
            api_base_url=args.api_base_url,
            model=args.model,
            voice_id=args.voice_id,
            audio_format=args.audio_format,
            sample_rate=args.sample_rate,
            bitrate=args.bitrate,
            channel=args.channel,
            speed=args.speed,
            volume=args.volume,
            pitch=args.pitch,
            emotion=args.emotion,
            language_boost=args.language_boost,
            timeout_s=args.timeout_s,
            cache_dir=args.cache_dir or None,
        )
    elif args.command == "move-gimbal":
        payload = json.loads(sys.stdin.read() or "{}")
        try:
            body_payload = payload.get("payload", {})
            driver = RaspbotDriver(bus=1, addr=0x2B, servo_id=args.servo_id, enabled=True, mock=False)
            result = move_gimbal(
                target_name=str(body_payload.get("target_name", "")),
                servo_id=args.servo_id,
                home_angle=args.home_angle,
                target_angle=body_payload.get("target_angle"),
                target_x=body_payload.get("target_x"),
                pan_min=int(body_payload.get("pan_min", 40)),
                pan_max=int(body_payload.get("pan_max", 140)),
                driver=driver,
            )
        except Exception as exc:  # pragma: no cover - only on honjia
            result = {"status": "error", "details": {"error": str(exc), "driver": "raspbot"}}
    elif args.command == "verify-pan-motion":
        result = _verify_pan_motion(args)
    elif args.command == "capture-frame":
        result = capture_frame(device=args.device, output_path=args.output_path)
    elif args.command == "compare-frames":
        result = compare_frame_hashes(args.left, args.right)
    elif args.command == "hailo-camera-detect":
        result = run_hailo_detection(
            post_process_file=args.post_process_file,
            timeout_s=args.timeout_s,
        )
    elif args.command == "hailo-frame-infer":
        result = run_hailo_frame_inference(
            image_path=args.image_path,
            hef_path=args.hef_path,
            score_threshold=args.score_threshold,
        )
    else:  # pragma: no cover - argparse enforces
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False))


def _verify_pan_motion(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    driver = RaspbotDriver(bus=1, addr=0x2B, servo_id=args.servo_id, enabled=True, mock=False)
    samples = [
        ("center_a", int(args.center_angle)),
        ("left", int(args.left_angle)),
        ("right", int(args.right_angle)),
        ("center_b", int(args.center_angle)),
    ]
    command_results: dict[str, object] = {}
    for name, angle in samples:
        command_results[name] = move_gimbal(
            target_name=f"pan_motion_proof_{name}",
            servo_id=args.servo_id,
            home_angle=args.center_angle,
            target_angle=angle,
            pan_min=0,
            pan_max=180,
            driver=driver,
        )
        (output_dir / f"{name}.cmd.json").write_text(
            json.dumps(command_results[name], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        time.sleep(max(0.0, float(args.settle_s)))
        frame_bytes = request.urlopen(args.frame_url, timeout=5).read()
        (output_dir / f"{name}.jpg").write_bytes(frame_bytes)

    try:
        pair_metrics = {
            "center_to_left": compare_frame_paths(output_dir / "center_a.jpg", output_dir / "left.jpg"),
            "center_to_right": compare_frame_paths(output_dir / "center_a.jpg", output_dir / "right.jpg"),
            "center_return": compare_frame_paths(output_dir / "center_a.jpg", output_dir / "center_b.jpg"),
            "left_to_right": compare_frame_paths(output_dir / "left.jpg", output_dir / "right.jpg"),
        }
        summary = summarize_pan_motion_pairs(
            pair_metrics,
            min_shift_px=float(args.min_shift_px),
            max_return_shift_px=float(args.max_return_shift_px),
        )
    except Exception as exc:  # pragma: no cover - host dependency
        summary = {
            "status": "error",
            "verified": False,
            "error": str(exc),
            "pairs": {},
        }
    summary.update(
        {
            "servo_id": int(args.servo_id),
            "center_angle": int(args.center_angle),
            "left_angle": int(args.left_angle),
            "right_angle": int(args.right_angle),
            "frame_url": str(args.frame_url),
            "output_dir": str(output_dir),
            "command_results": command_results,
            "generated_at_ts": time.time(),
        }
    )
    write_pan_motion_summary(output_dir / "summary.json", summary)
    return {"status": "ok" if summary.get("verified") else "degraded", "details": summary}


if __name__ == "__main__":
    main()
