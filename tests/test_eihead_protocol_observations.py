from __future__ import annotations

import json
from dataclasses import dataclass

from eihead.protocol import (
    AudioTranscriptFinal,
    HeadObservation,
    ProtocolMessage,
    RealtimeVisionObservation,
    VisionObservation,
)
from eihead.protocol.base import serialize_message


def _json_round_trip(model: ProtocolMessage) -> tuple[ProtocolMessage, dict]:
    payload = json.loads(json.dumps(model.to_dict()))
    return type(model).from_dict(payload), payload


def test_audio_transcript_final_is_local_and_round_trips() -> None:
    observation = AudioTranscriptFinal(
        ts=1.0,
        source="eihead.ear.asr",
        text="你好鸿途",
        language="zh",
        session_id="s1",
        actor_id="user-1",
    )

    restored, payload = _json_round_trip(observation)

    assert observation.__module__ == "eihead.protocol.observations"
    assert observation.kind == "audio_transcript_final"
    assert payload == {
        "ts": 1.0,
        "source": "eihead.ear.asr",
        "session_id": "s1",
        "actor_id": "user-1",
        "target_id": None,
        "kind": "audio_transcript_final",
        "text": "你好鸿途",
        "language": "zh",
    }
    assert restored.to_dict() == payload
    assert isinstance(restored, ProtocolMessage)


def test_vision_observation_is_local_and_round_trips() -> None:
    observation = VisionObservation(
        ts=3.0,
        source="eihead.honjia.eye",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_000_200,
        trace_id="trace-vision",
        frame_id="frame-42",
        width=1280,
        height=720,
        payload={"camera": "front"},
        detections=[
            {
                "label": "person",
                "score": 0.91,
                "bbox": [0.3, 0.2, 0.4, 0.5],
            }
        ],
        tracked_target={"label": "person", "center_x": 0.5},
    )

    restored, payload = _json_round_trip(observation)

    assert observation.__module__ == "eihead.protocol.observations"
    assert isinstance(observation, HeadObservation)
    assert restored.to_dict() == payload
    assert restored.kind == "vision_observation"
    assert restored.observation_type == "vision_observation"
    assert restored.modality == "vision"
    assert restored.detections[0]["score"] == 0.91
    assert restored.tracked_target["center_x"] == 0.5
    assert restored.payload == {"camera": "front"}
    assert restored.mode == "compat/static"
    assert restored.primary_mode is False


def test_realtime_vision_observation_is_primary_realtime_and_round_trips() -> None:
    observation = RealtimeVisionObservation(
        ts=4.0,
        source="eihead.honjia.eye.realtime",
        target="eibrain.honxin",
        timestamp_ms=1_714_800_000_260,
        trace_id="trace-realtime",
        stream_id="front-main",
        camera_id="front",
        status="tracking",
        frame_id="frame-99",
        width=1280,
        height=720,
        fps=29.8,
        latency_ms=42.5,
        payload={"simulated": True},
        detections=[{"label": "person", "score": 0.93}],
        tracked_target={"label": "person", "center_x": 0.52},
        stream={"transport": "in-process", "connected": True},
        health={"dropped_frames": 0},
    )

    restored, payload = _json_round_trip(observation)

    assert observation.__module__ == "eihead.protocol.observations"
    assert isinstance(observation, HeadObservation)
    assert payload["kind"] == "realtime_vision_observation"
    assert payload["channel"] == "eye.realtime"
    assert payload["aliases"] == ["vision.realtime"]
    assert payload["mode"] == "realtime"
    assert payload["primary_mode"] is True
    assert restored.to_dict() == payload
    assert restored.observation_type == "realtime_vision_observation"
    assert restored.modality == "vision.realtime"
    assert restored.stream_id == "front-main"
    assert restored.detections[0]["score"] == 0.93
    assert restored.tracked_target["center_x"] == 0.52
    assert restored.stream["connected"] is True
    assert restored.health["dropped_frames"] == 0


def test_serialize_message_handles_protocol_messages_mappings_and_dataclasses() -> None:
    @dataclass(slots=True)
    class ExternalMessage:
        kind: str
        value: int

    observation = AudioTranscriptFinal(ts=1.0, source="eihead.ear", text="hello")

    assert serialize_message(observation)["kind"] == "audio_transcript_final"
    assert serialize_message({"kind": "raw", "value": 1}) == {"kind": "raw", "value": 1}
    assert serialize_message(ExternalMessage(kind="external", value=2)) == {
        "kind": "external",
        "value": 2,
    }
