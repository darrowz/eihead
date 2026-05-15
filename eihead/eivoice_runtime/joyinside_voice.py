"""JoyInside WebSocket voice event mapping helpers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


UPSTREAM = "upstream"
DOWNSTREAM = "downstream"

TEXT_TYPE_FINAL = "IS_FINAL"

UPSTREAM_NAMES = {
    "CLIENT_VOICE_CHAT_UPDATE": "ei.voice.config.update.requested",
    "AUDIO": "ei.voice.audio.frame",
    "TEXT": "ei.dialogue.user.text",
    "CLIENT_AUDIO_FINISH": "ei.voice.audio.finish.requested",
    "CLIENT_INTERRUPT": "ei.dialogue.interrupt.requested",
    "CLIENT_INPUT_TEXT_TO_SPEECH": "ei.voice.tts.requested",
    "PING": "ei.voice.session.heartbeat",
}

DOWNSTREAM_NAMES = {
    "CFG_BOT_EVENT": "ei.voice.config.update",
    "SERVER_VOICE_CHAT_UPDATED": "ei.voice.config.updated",
    "CALL_AGENT_START_EVENT": "ei.voice.agent.started",
    "EMPTY_CONTENT": "ei.voice.empty",
    "AGENT": "ei.voice.agent.delta",
    "ACTIVITY": "ei.voice.activity.delta",
    "TTS_SENTENCE_START": "ei.voice.tts.sentence_start",
    "TTS": "ei.voice.tts.chunk",
    "TTS_COMPLETE": "ei.voice.tts.complete",
    "COMPLETE": "ei.dialogue.complete",
    "CALL_AGENT_INTERRUPTED": "ei.dialogue.interrupt.applied",
    "QUEUE_HEALTH": "ei.voice.queue.health",
    "PONG": "ei.voice.session.heartbeat",
}


@dataclass(slots=True)
class JoyInsideVoiceEvent:
    """Normalized JoyInside voice event wrapper."""

    direction: str
    content_type: str
    event_type: str
    uid: str | None = None
    mid: str | None = None
    content: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dict with JoyInside field names preserved."""

        return {
            "direction": self.direction,
            "contentType": self.content_type,
            "eventType": self.event_type,
            "uid": self.uid,
            "mid": self.mid,
            "content": deepcopy(self.content),
            "raw": deepcopy(self.raw),
        }


def voice_chat_update(
    *,
    uid: str | None = None,
    mid: str | None = None,
    audio_input: Mapping[str, Any] | None = None,
    audio_output: Mapping[str, Any] | None = None,
    timbre: Mapping[str, Any] | None = None,
    chat: Mapping[str, Any] | None = None,
) -> JoyInsideVoiceEvent:
    """Build an upstream CLIENT_VOICE_CHAT_UPDATE event."""

    content: dict[str, Any] = {"eventType": "CLIENT_VOICE_CHAT_UPDATE"}
    audio: dict[str, Any] = {}
    if audio_input is not None:
        audio["input"] = dict(audio_input)
    if audio_output is not None:
        audio["output"] = dict(audio_output)
    if audio:
        content["audio"] = audio
    if timbre is not None:
        content["timbre"] = dict(timbre)
    if chat is not None:
        content["chat"] = dict(chat)
    return _upstream_event("EVENT", uid=uid, mid=mid, content=content)


def audio_chunk(
    *,
    uid: str | None = None,
    mid: str | None = None,
    index: int | None = None,
    audio_base64: str | None = None,
    audioBase64: str | None = None,
) -> JoyInsideVoiceEvent:
    """Build an upstream AUDIO frame event."""

    content = {"eventType": "AUDIO"}
    if index is not None:
        content["index"] = index
    encoded = audio_base64 if audio_base64 is not None else audioBase64
    if encoded is not None:
        content["audioBase64"] = encoded
    return _upstream_event("AUDIO", uid=uid, mid=mid, content=content)


def text_message(*, uid: str | None = None, mid: str | None = None, text: str = "") -> JoyInsideVoiceEvent:
    """Build an upstream TEXT event."""

    return _upstream_event("TEXT", uid=uid, mid=mid, content={"eventType": "TEXT", "text": text})


def audio_finish(*, uid: str | None = None, mid: str | None = None) -> JoyInsideVoiceEvent:
    """Build an upstream CLIENT_AUDIO_FINISH event."""

    return _upstream_event(
        "EVENT",
        uid=uid,
        mid=mid,
        content={"eventType": "CLIENT_AUDIO_FINISH"},
    )


def interrupt(
    *,
    uid: str | None = None,
    mid: str | None = None,
    reason: str | None = None,
) -> JoyInsideVoiceEvent:
    """Build an upstream CLIENT_INTERRUPT event."""

    content = {"eventType": "CLIENT_INTERRUPT"}
    if reason is not None:
        content["reason"] = reason
    return _upstream_event("EVENT", uid=uid, mid=mid, content=content)


def input_text_to_speech(
    *,
    uid: str | None = None,
    mid: str | None = None,
    text: str = "",
) -> JoyInsideVoiceEvent:
    """Build an upstream CLIENT_INPUT_TEXT_TO_SPEECH event."""

    return _upstream_event(
        "EVENT",
        uid=uid,
        mid=mid,
        content={"eventType": "CLIENT_INPUT_TEXT_TO_SPEECH", "text": text},
    )


def ping(
    *,
    uid: str | None = None,
    mid: str | None = None,
    timestamp: int | float | None = None,
) -> JoyInsideVoiceEvent:
    """Build an upstream PING event."""

    content: dict[str, Any] = {"eventType": "PING"}
    if timestamp is not None:
        content["timestamp"] = timestamp
    return _upstream_event("PING", uid=uid, mid=mid, content=content)


def parse_downstream_event(raw: Mapping[str, Any]) -> JoyInsideVoiceEvent:
    """Parse a downstream JoyInside payload into a JoyInsideVoiceEvent."""

    payload = _copy_mapping(raw)
    content = _copy_mapping(payload.get("content"))
    content_type = _text(payload.get("contentType"), payload.get("content_type"), content.get("contentType"))
    event_type = _text(content.get("eventType"), content.get("event_type"), content_type)
    return JoyInsideVoiceEvent(
        direction=DOWNSTREAM,
        content_type=content_type,
        event_type=event_type,
        uid=_text(payload.get("uid")),
        mid=_text(payload.get("mid")),
        content=content,
        raw=payload,
    )


def to_eiprotocol_name(event: JoyInsideVoiceEvent | Mapping[str, Any]) -> str | None:
    """Map a JoyInside event to a stable eiprotocol-friendly event name."""

    parsed = _event_from(event)
    if parsed.content_type == "ASR" or parsed.event_type == "ASR":
        return "ei.voice.asr.final" if _is_final_asr(parsed.content) else "ei.voice.asr.partial"
    if parsed.direction == UPSTREAM:
        return UPSTREAM_NAMES.get(parsed.event_type) or UPSTREAM_NAMES.get(parsed.content_type)
    return DOWNSTREAM_NAMES.get(parsed.event_type) or DOWNSTREAM_NAMES.get(parsed.content_type)


def normalize_voice_event(raw: JoyInsideVoiceEvent | Mapping[str, Any], direction: str) -> dict[str, Any]:
    """Return a flat eiprotocol-friendly JoyInside voice event dict."""

    event = _event_from(raw, direction=direction)
    content = event.content
    normalized: dict[str, Any] = {
        "direction": event.direction,
        "eiprotocolName": to_eiprotocol_name(event),
        "uid": event.uid,
        "mid": event.mid,
        "contentType": event.content_type,
        "eventType": event.event_type,
        "payload": deepcopy(content),
        "joyinside": {
            "contentType": event.content_type,
            "eventType": event.event_type,
            "uid": event.uid,
            "mid": event.mid,
        },
    }

    text = _text(content.get("text"), content.get("sentence"), content.get("message"))
    if text:
        normalized["text"] = text

    audio_base64 = _text(content.get("audioBase64"), content.get("audio_base64"), content.get("audio"))
    if audio_base64:
        normalized["audioBase64"] = audio_base64

    chunk_index = _optional_int(content.get("index"), content.get("chunkIndex"), content.get("chunk_index"))
    if chunk_index is not None:
        normalized["chunkIndex"] = chunk_index

    reason = _text(content.get("reason"), content.get("interruptReason"))
    if reason:
        normalized["reason"] = reason

    if event.content_type == "ASR" or event.event_type == "ASR":
        normalized["final"] = _is_final_asr(content)
    elif event.event_type in {"TTS_COMPLETE", "COMPLETE"} or event.content_type in {"TTS_COMPLETE", "COMPLETE"}:
        normalized["final"] = True

    if "textType" in content:
        normalized["textType"] = content["textType"]
    if "timestamp" in content:
        normalized["timestamp"] = content["timestamp"]
    if event.raw:
        normalized["joyinside"]["raw"] = deepcopy(event.raw)
    return normalized


def _upstream_event(
    content_type: str,
    *,
    uid: str | None,
    mid: str | None,
    content: dict[str, Any],
) -> JoyInsideVoiceEvent:
    event_type = _text(content.get("eventType"), content_type)
    raw = {"contentType": content_type, "uid": uid, "mid": mid, "content": deepcopy(content)}
    return JoyInsideVoiceEvent(
        direction=UPSTREAM,
        content_type=content_type,
        event_type=event_type,
        uid=uid,
        mid=mid,
        content=content,
        raw=raw,
    )


def _event_from(
    raw: JoyInsideVoiceEvent | Mapping[str, Any],
    *,
    direction: str | None = None,
) -> JoyInsideVoiceEvent:
    if isinstance(raw, JoyInsideVoiceEvent):
        if direction is None or raw.direction == direction:
            return raw
        return JoyInsideVoiceEvent(
            direction=direction,
            content_type=raw.content_type,
            event_type=raw.event_type,
            uid=raw.uid,
            mid=raw.mid,
            content=deepcopy(raw.content),
            raw=deepcopy(raw.raw),
        )

    payload = _copy_mapping(raw)
    content = _copy_mapping(payload.get("content"))
    resolved_direction = _text(direction, payload.get("direction"), fallback=DOWNSTREAM)
    content_type = _text(payload.get("contentType"), payload.get("content_type"), content.get("contentType"))
    event_type = _text(payload.get("eventType"), content.get("eventType"), content.get("event_type"), content_type)
    return JoyInsideVoiceEvent(
        direction=resolved_direction,
        content_type=content_type,
        event_type=event_type,
        uid=_text(payload.get("uid")),
        mid=_text(payload.get("mid")),
        content=content,
        raw=deepcopy(payload.get("raw")) if isinstance(payload.get("raw"), Mapping) else payload,
    )


def _is_final_asr(content: Mapping[str, Any]) -> bool:
    return _text(content.get("textType"), content.get("text_type")).upper() == TEXT_TYPE_FINAL


def _copy_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _text(*values: Any, fallback: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return fallback


def _optional_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


__all__ = [
    "JoyInsideVoiceEvent",
    "audio_chunk",
    "ping",
]
