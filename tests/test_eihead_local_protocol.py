from __future__ import annotations

from eihead.protocol import (
    ActionExecuted,
    MoveHeadAction,
    PlaySpeechAction,
    ProtocolMessage,
    SpeechPlaybackCompleted,
    StopSpeechAction,
)
import eihead.runtime.app as runtime_app
from eihead.runtime.app import HeadRuntimeApp


def test_local_actions_keep_existing_protocol_shape() -> None:
    speech = PlaySpeechAction(
        ts=1.25,
        source="eihead.runtime",
        text="你好鸿途",
        session_id="s1",
        actor_id="darrow",
        target_id="mouth",
    )
    move = MoveHeadAction(
        ts=2.0,
        source="eihead.runtime",
        target_name="speaker",
        target_x=0.42,
        target_angle=112,
    )
    stop = StopSpeechAction(ts=3.0, source="eihead.runtime")

    assert speech.kind == "play_speech_action"
    assert isinstance(speech, ProtocolMessage)
    assert speech.to_dict() == {
        "ts": 1.25,
        "source": "eihead.runtime",
        "session_id": "s1",
        "actor_id": "darrow",
        "target_id": "mouth",
        "kind": "play_speech_action",
        "text": "你好鸿途",
    }
    assert move.to_dict()["kind"] == "move_head_action"
    assert move.to_dict()["target_angle"] == 112
    assert move.to_dict()["target_x"] == 0.42
    assert stop.to_dict()["kind"] == "stop_speech_action"


def test_local_outcomes_keep_existing_protocol_shape() -> None:
    playback = SpeechPlaybackCompleted(
        ts=4.0,
        source="mouth.tts_playback",
        status="ok",
        session_id="s2",
    )
    executed = ActionExecuted(
        ts=5.0,
        source="neck.motor",
        status="ok",
        action_kind="move_head_action",
        details={"target_angle": 90},
    )

    assert playback.to_dict() == {
        "ts": 4.0,
        "source": "mouth.tts_playback",
        "session_id": "s2",
        "actor_id": None,
        "target_id": None,
        "status": "ok",
        "kind": "speech_playback_completed",
    }
    assert isinstance(executed, ProtocolMessage)
    assert executed.to_dict()["kind"] == "action_executed"
    assert executed.to_dict()["action_kind"] == "move_head_action"
    assert executed.to_dict()["details"] == {"target_angle": 90}


def test_runtime_uses_eihead_local_protocol_classes() -> None:
    assert HeadRuntimeApp.__module__ == "eihead.runtime.app"
    assert runtime_app.PlaySpeechAction is PlaySpeechAction
    assert runtime_app.MoveHeadAction is MoveHeadAction
    assert runtime_app.StopSpeechAction is StopSpeechAction
    assert PlaySpeechAction.__module__ == "eihead.protocol.actions"
    assert MoveHeadAction.__module__ == "eihead.protocol.actions"
    assert StopSpeechAction.__module__ == "eihead.protocol.actions"


def test_local_streaming_and_tracking_protocol_shapes_are_stable() -> None:
    from eihead.protocol.actions import StreamSpeechAction
    from eihead.protocol.observations import VoiceAudioFrameObservation, VisionTrackingObservation

    stream = StreamSpeechAction(
        ts=6.0,
        source="eihead.runtime",
        session_id="s3",
        stream_id="tts-stream-1",
        chunk_index=4,
        audio_base64="AAEC",
        text="你好",
        final=False,
    )
    voice_frame = VoiceAudioFrameObservation(
        ts=7.0,
        source="eihead.ear",
        session_id="s3",
        stream_id="mic-stream-1",
        chunk_index=5,
        audio_base64="UklGRg==",
        sample_rate_hz=16000,
        channels=1,
        latency_ms=14.5,
    )
    tracking = VisionTrackingObservation(
        ts=8.0,
        source="eihead.eye",
        session_id="s3",
        frame_id="frame-88",
        tracked_target={"track_id": "person-1", "label": "person"},
        detections=[{"label": "person", "score": 0.91}],
        latency_ms=23.0,
        status="tracking",
    )

    assert stream.to_dict()["kind"] == "stream_speech_action"
    assert stream.to_dict()["chunk_index"] == 4
    assert voice_frame.to_dict()["kind"] == "voice_audio_frame_observation"
    assert voice_frame.observation_type == "voice_audio_frame_observation"
    assert voice_frame.modality == "audio"
    assert tracking.to_dict()["kind"] == "vision_tracking_observation"
    assert tracking.modality == "vision"
    assert tracking.to_dict()["tracked_target"] == {"track_id": "person-1", "label": "person"}
