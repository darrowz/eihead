"""Hardware verification CLI for honjia/honxin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from apps.body_runtime.app import BodyRuntimeApp
from eibrain.body.raspbot_driver import RaspbotDriver
from eibrain.body.runtime_linux import capture_frame
from eibrain.body.runtime_linux import compare_frame_hashes
from eibrain.body.runtime_linux import move_gimbal
from eibrain.body.runtime_linux import run_hailo_frame_inference
from eibrain.body.runtime_linux import run_hailo_detection
from eibrain.infra.config import load_config
from eibrain.verification import (
    run_ear_stream_check,
    run_gimbal_frame_check,
    run_hailo_camera_check,
    run_hailo_frame_check,
    run_vision_frame_check,
    run_voice_dialogue_check,
)

if TYPE_CHECKING:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify honjia hardware and vision chains")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gimbal = subparsers.add_parser("gimbal-frame-check")
    gimbal.add_argument("--config", default="config/eibrain.yaml")
    gimbal.add_argument("--device", default="")
    gimbal.add_argument("--output-dir", required=True)
    gimbal.add_argument("--angles", nargs="+", type=int, default=[40, 90, 140])
    gimbal.add_argument("--servo-id", type=int)

    ear = subparsers.add_parser("ear-stream-check")
    ear.add_argument("--config", default="config/eibrain.yaml")
    ear.add_argument("--chunk-count", type=int, default=3)
    ear.add_argument("--session-id", default="verify-ear")
    ear.add_argument("--actor-id", default="verify-user")

    dialogue = subparsers.add_parser("voice-dialogue-check")
    dialogue.add_argument("--config", default="config/eibrain.yaml")
    dialogue.add_argument("--chunk-count", type=int, default=6)
    dialogue.add_argument("--session-id", default="verify-dialogue")
    dialogue.add_argument("--actor-id", default="verify-user")

    vision = subparsers.add_parser("vision-frame-check")
    vision.add_argument("--config", default="config/eibrain.yaml")
    vision.add_argument("--images", nargs="+", required=True)

    hailo = subparsers.add_parser("hailo-camera-check")
    hailo.add_argument("--config", default="config/eibrain.yaml")
    hailo.add_argument(
        "--post-process-file",
        default="/usr/share/rpi-camera-assets/hailo_yolov5_personface.json",
    )
    hailo.add_argument("--timeout-s", type=int, default=8)

    hailo_frame = subparsers.add_parser("hailo-frame-check")
    hailo_frame.add_argument("--config", default="config/eibrain.yaml")
    hailo_frame.add_argument("--device", default="")
    hailo_frame.add_argument("--output-path", required=True)
    hailo_frame.add_argument("--hef-path", default="/usr/share/hailo-models/yolov5s_personface_h8l.hef")
    hailo_frame.add_argument("--score-threshold", type=float, default=0.3)

    args = parser.parse_args()
    if args.command == "gimbal-frame-check":
        config = load_config(args.config)
        eye_cfg = config.body.organs.get("eye")
        neck_cfg = config.body.organs.get("neck")
        camera_cfg = eye_cfg.subfunctions.get("camera") if eye_cfg is not None else None
        motor_cfg = neck_cfg.subfunctions.get("motor") if neck_cfg is not None else None
        device = args.device or str(camera_cfg.driver.extra.get("device", "/dev/video0")) if camera_cfg is not None else "/dev/video0"
        servo_id = args.servo_id or _extract_servo_id(motor_cfg.driver.command if motor_cfg is not None else [])
        pan_min = int(motor_cfg.driver.extra.get("pan_min", 40)) if motor_cfg is not None else 40
        pan_max = int(motor_cfg.driver.extra.get("pan_max", 140)) if motor_cfg is not None else 140
        driver = RaspbotDriver(bus=1, addr=0x2B, servo_id=servo_id, enabled=True, mock=False)
        result = run_gimbal_frame_check(
            angles=list(args.angles),
            output_dir=args.output_dir,
            move_fn=lambda angle: move_gimbal(
                target_name=f"angle-{angle}",
                servo_id=servo_id,
                home_angle=angle,
                pan_min=pan_min,
                pan_max=pan_max,
                driver=driver,
            ),
            capture_fn=lambda angle, frame_path: capture_frame(device=device, output_path=frame_path),
            compare_fn=compare_frame_hashes,
        )
    elif args.command == "ear-stream-check":
        runtime = BodyRuntimeApp.from_config_path(args.config)
        result = run_ear_stream_check(
            chunk_count=args.chunk_count,
            transcribe_fn=lambda chunk_count: runtime.transcribe_audio_window(
                chunk_count=chunk_count,
                session_id=args.session_id,
                actor_id=args.actor_id,
            ).to_dict(),
        )
    elif args.command == "voice-dialogue-check":
        from apps.cognitive_runtime.app import CognitiveRuntimeApp

        body_runtime = BodyRuntimeApp.from_config_path(args.config)
        cognitive_runtime = CognitiveRuntimeApp.from_config_path(args.config)
        last_observation = {}

        def _listen(chunk_count: int) -> dict[str, object]:
            nonlocal last_observation
            observation = body_runtime.transcribe_audio_window(
                chunk_count=chunk_count,
                session_id=args.session_id,
                actor_id=args.actor_id,
            )
            last_observation = observation.to_dict()
            return last_observation

        def _plan(transcript: dict[str, object]) -> list[dict[str, object]]:
            from eibrain.protocol.observations import AudioTranscriptFinal

            observation = AudioTranscriptFinal(
                ts=float(transcript.get("ts", 1.0) or 1.0),
                source=str(transcript.get("source", "ear.asr")),
                session_id=str(transcript.get("session_id", args.session_id)),
                actor_id=str(transcript.get("actor_id", args.actor_id)),
                text=str(transcript.get("text", "")),
            )
            return [
                action.to_dict() if hasattr(action, "to_dict") else dict(action)
                for action in cognitive_runtime.handle_observation(observation)
            ]

        def _dispatch(actions: list[dict[str, object]]) -> list[dict[str, object]]:
            from eibrain.protocol.actions import PlaySpeechAction

            action_objects = [
                PlaySpeechAction(
                    ts=float(action.get("ts", 1.0) or 1.0),
                    source=str(action.get("source", "voice-dialogue-check")),
                    session_id=str(action.get("session_id", args.session_id)),
                    actor_id=str(action.get("actor_id", args.actor_id)),
                    target_id=action.get("target_id"),
                    text=str(action.get("text", "")),
                )
                for action in actions
                if str(action.get("kind", "")) == "play_speech_action"
            ]
            return [
                outcome.to_dict() if hasattr(outcome, "to_dict") else dict(outcome)
                for outcome in body_runtime.dispatch_actions(action_objects)
            ]

        result = run_voice_dialogue_check(
            chunk_count=args.chunk_count,
            listen_fn=_listen,
            plan_fn=_plan,
            dispatch_fn=_dispatch,
        )
        result["cognition"] = cognitive_runtime.snapshot()
    elif args.command == "vision-frame-check":
        from apps.cognitive_runtime.app import CognitiveRuntimeApp

        runtime = CognitiveRuntimeApp.from_config_path(args.config)
        result = run_vision_frame_check(
            image_paths=list(args.images),
            describe_fn=lambda image_path: _describe_frame(runtime, image_path),
        )
    elif args.command == "hailo-frame-check":
        config = load_config(args.config)
        eye_cfg = config.body.organs.get("eye")
        camera_cfg = eye_cfg.subfunctions.get("camera") if eye_cfg is not None else None
        device = args.device or str(camera_cfg.driver.extra.get("device", "/dev/video0")) if camera_cfg is not None else "/dev/video0"
        result = run_hailo_frame_check(
            capture_fn=lambda: capture_frame(device=device, output_path=args.output_path),
            infer_fn=lambda: run_hailo_frame_inference(
                image_path=args.output_path,
                hef_path=args.hef_path,
                score_threshold=args.score_threshold,
            ),
        )
    else:
        result = run_hailo_camera_check(
            detect_fn=lambda: run_hailo_detection(
                post_process_file=args.post_process_file,
                timeout_s=args.timeout_s,
            )
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _describe_frame(runtime: CognitiveRuntimeApp, image_path: str) -> dict[str, object]:
    understanding = runtime.describe_visual_frame(image_url=image_path)
    if understanding is None:
        return {"summary": "", "primary_subject": "", "confidence": 0.0}
    return {
        "summary": understanding.summary,
        "primary_subject": understanding.primary_subject,
        "confidence": understanding.confidence,
    }


def _extract_servo_id(command: list[str]) -> int:
    for index, token in enumerate(command):
        if token == "--servo-id" and index + 1 < len(command):
            try:
                return int(command[index + 1])
            except ValueError:
                break
    return 1


if __name__ == "__main__":
    main()
