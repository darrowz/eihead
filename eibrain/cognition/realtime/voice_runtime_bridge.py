"""Bridge voice runtime events into realtime cognition scheduling."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from .scheduler import RealtimeCognitiveScheduler


class VoiceRuntimeBridge:
    """Translate normalized voice runtime events into cognition lane updates.

    The bridge accepts loose normalized dictionaries and eiprotocol-like
    dictionaries. It returns a stable JSON-ready envelope for the realtime
    cognition side without making final decisions from unstable voice signals.
    """

    def __init__(
        self,
        *,
        scheduler: RealtimeCognitiveScheduler | None = None,
        clock: Any | None = None,
    ) -> None:
        self._clock = clock or time.time
        self.scheduler = scheduler or RealtimeCognitiveScheduler(clock=self._clock)

    def handle_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        normalized = _normalize_event(event)
        event_type = normalized["type"]

        if event_type == "ASR":
            return self._handle_asr_final(normalized) if _truthy(_first_value(normalized, "final")) else self._handle_asr_partial(normalized)
        if event_type in {"ASR_PARTIAL", "ASR_PARTIAL_RESULT", "VOICE_ASR_PARTIAL", "EI_VOICE_ASR_PARTIAL"}:
            return self._handle_asr_partial(normalized)
        if event_type in {"ASR_FINAL", "ASR_FINAL_RESULT", "VOICE_ASR_FINAL", "EI_VOICE_ASR_FINAL"}:
            return self._handle_asr_final(normalized)
        if event_type in {"TTS_SENTENCE_START", "TTS_SENTENCE", "VOICE_TTS_SENTENCE_START", "EI_VOICE_TTS_SENTENCE_START"}:
            return self._handle_tts(normalized, phase="sentence_start")
        if event_type in {"TTS", "TTS_CHUNK", "TTS_AUDIO_CHUNK", "VOICE_TTS_CHUNK", "EI_VOICE_TTS_CHUNK"}:
            return self._handle_tts(normalized, phase="chunk")
        if event_type in {"CALL_AGENT_INTERRUPTED", "CLIENT_INTERRUPT", "USER_INTERRUPT", "EI_DIALOGUE_INTERRUPT_REQUESTED", "EI_DIALOGUE_INTERRUPT_APPLIED"}:
            return self._handle_interrupt(normalized)
        if event_type in {"ACTIVITY", "VOICE_ACTIVITY", "PROACTIVE_ACTIVITY", "EI_VOICE_ACTIVITY_DELTA"}:
            return self._handle_activity(normalized)
        if event_type in {"QUEUE_HEALTH", "QUEUE_HEALTH_DEGRADED", "VOICE_QUEUE_HEALTH", "EI_VOICE_QUEUE_HEALTH"}:
            return self._handle_queue_health(normalized)

        return self._envelope(
            round_id=self._current_round_id(normalized),
            conversation_state="observing",
            lane="runtime_event",
            blackboard_patch={
                "runtimeEvent": {
                    "type": event_type,
                    "payload": dict(normalized["payload"]),
                }
            },
        )

    def _handle_asr_partial(self, event: dict[str, Any]) -> dict[str, Any]:
        text = _event_text(event, "partialText", "text", "transcript")
        observed = self.scheduler.observe_partial(
            text,
            persona_context=_mapping_from(event, "persona", "personaContext"),
            emotion_context=_mapping_from(event, "emotion", "emotionContext"),
            environment_context=_mapping_from(event, "environment", "environmentContext"),
            memory_candidates=_sequence_from(event, "memoryCandidates", "memory_candidates"),
        )
        fast = dict(observed.get("fast") or {})
        stored = dict(fast.get("stored_hypothesis") or {})
        memory_hints = [
            {
                "kind": "prefetch",
                "query": item.get("query") or item.get("text") or text,
                "source": item.get("source", "voice_runtime_bridge"),
                "stable": False,
                "candidate": dict(item),
            }
            for item in observed.get("memory_prefetch", [])
            if isinstance(item, Mapping)
        ]
        if not memory_hints and text:
            memory_hints.append(
                {
                    "kind": "prefetch",
                    "query": text,
                    "source": "voice_runtime_bridge",
                    "stable": False,
                    "risk": "low",
                }
            )

        return self._envelope(
            round_id=str(observed["round_id"]),
            conversation_state="listening",
            lane="fast_think",
            blackboard_patch={
                "asrPartial": text,
                "voiceRoundId": event.get("roundId"),
                "fastHypothesis": {
                    "partialText": text,
                    "stable": False,
                    "risk": "low",
                    "source": "voice_runtime_bridge",
                    "schedulerHypothesis": stored,
                },
                "microFeedback": fast.get("microfeedback")
                or _mapping_text(fast.get("micro_feedback"), "text"),
            },
            memory_hints=memory_hints,
        )

    def _handle_asr_final(self, event: dict[str, Any]) -> dict[str, Any]:
        text = _event_text(event, "finalText", "text", "transcript")
        observed = self.scheduler.observe_final(
            text,
            memory_candidates=_sequence_from(event, "memoryCandidates", "memory_candidates"),
            persona_context=_mapping_from(event, "persona", "personaContext"),
            emotion_context=_mapping_from(event, "emotion", "emotionContext"),
            environment_context=_mapping_from(event, "environment", "environmentContext"),
        )
        round_id = str(observed["round_id"])
        token = str(observed["cancellation_token"])
        return self._envelope(
            round_id=round_id,
            conversation_state="reasoning",
            lane="slow_reasoning",
            blackboard_patch={
                "asrFinal": observed.get("final_text") or text,
                "voiceRoundId": event.get("roundId"),
                "slowReasonerInput": {
                    "roundId": round_id,
                    "cancellationToken": token,
                    "finalText": observed.get("final_text") or text,
                    "memoryCandidates": list(observed.get("memory_candidates") or []),
                    "source": "voice_runtime_bridge",
                },
            },
            trace=self._trace(
                event,
                lane="slow_reasoning",
                round_id=round_id,
                cancellation_token=token,
            ),
        )

    def _handle_tts(self, event: dict[str, Any], *, phase: str) -> dict[str, Any]:
        payload = event["payload"]
        current = self.scheduler.current_turn()
        event_round_id = str(event.get("roundId") or "")
        event_token = str(event.get("cancellationToken") or "")
        stale_reason = self._stale_tts_reason(event, current=current)
        stale = stale_reason is not None
        round_id = event_round_id or (current.round_id if current is not None else self._current_round_id(event))
        cancellation_token = event_token or (
            current.cancellation_token if current is not None else self._current_cancellation_token(event)
        )
        metadata = dict(payload)
        text = _event_text(event, "text", "sentence", "transcript")
        plan = {
            "type": "speech_playback",
            "phase": phase,
            "roundId": round_id,
            "cancellationToken": cancellation_token,
            "sentenceId": _first_value(event, "sentenceId", "sentence_id"),
            "chunkIndex": _first_value(event, "chunkIndex", "chunk_index"),
            "text": text,
            "metadata": metadata,
            "source": "voice_runtime_bridge",
        }
        if stale:
            plan["stale"] = True
            plan["staleReason"] = stale_reason
            return self._envelope(
                round_id=round_id,
                conversation_state="speaking",
                lane="speaking",
                blackboard_patch={"ttsPlayback": plan},
                summary=self._summary(
                    round_id=round_id,
                    lane="speaking",
                    conversation_state="speaking",
                    action_count=0,
                    has_speech_plan=False,
                    stale=True,
                ),
                trace=self._trace(
                    event,
                    lane="speaking",
                    round_id=round_id,
                    cancellation_token=cancellation_token,
                ),
            )
        return self._envelope(
            round_id=round_id,
            conversation_state="speaking",
            lane="speaking",
            actions=[{"type": "speech_playback", "phase": phase, "roundId": round_id}],
            blackboard_patch={"ttsPlayback": plan},
            speech_plan=plan,
            summary=self._summary(
                round_id=round_id,
                lane="speaking",
                conversation_state="speaking",
                action_count=1,
                has_speech_plan=True,
                stale=False,
            ),
            trace=self._trace(
                event,
                lane="speaking",
                round_id=round_id,
                cancellation_token=cancellation_token,
            ),
        )

    def _handle_interrupt(self, event: dict[str, Any]) -> dict[str, Any]:
        reason = str(_first_value(event, "reason") or "user_interrupt")
        summary = self.scheduler.interrupt(reason=reason)
        old = dict(summary.get("mark_interrupted") or {})
        new = dict(summary.get("start_new_round") or {})
        interrupt = {
            "applied": True,
            "cancelOldRound": True,
            "reason": reason,
            "oldRound": {
                "roundId": old.get("round_id"),
                "cancellationToken": old.get("cancellation_token"),
                "state": old.get("state"),
                "stale": True,
                "cancelled": bool(old.get("cancelled")),
            },
            "newRound": {
                "roundId": new.get("round_id"),
                "cancellationToken": new.get("cancellation_token"),
            },
            "cancellationChain": [
                "tts_playback",
                "llm_generation",
                "action_plan",
                "memory_prefetch",
            ],
            "controller": summary,
        }
        return self._envelope(
            round_id=str(new.get("round_id") or ""),
            conversation_state="interrupted",
            lane="interrupt",
            actions=[
                {"type": "stop_tts", "applied": bool(summary.get("stop_tts"))},
                {"type": "cancel_old_round", "roundId": old.get("round_id"), "stale": True},
            ],
            blackboard_patch={
                "staleRound": interrupt["oldRound"],
                "activeRound": interrupt["newRound"],
            },
            interrupt=interrupt,
            trace=self._trace(
                event,
                lane="interrupt",
                round_id=str(new.get("round_id") or ""),
                cancellation_token=str(new.get("cancellation_token") or ""),
            ),
        )

    def _handle_activity(self, event: dict[str, Any]) -> dict[str, Any]:
        turn = self.scheduler.current_turn()
        if turn is None:
            turn = self.scheduler.turn_manager.start_round(reason="proactive_activity")

        payload = event["payload"]
        idle_seconds = _as_float(
            _first_value(event, "idleSeconds", "idle_seconds", default=payload.get("idle", 0.0))
        )
        memory_candidates = _sequence_from(event, "memoryCandidates", "memory_candidates", "memories")
        proposal = self.scheduler.activity_manager.propose(
            idle_seconds=idle_seconds,
            emotion_context=_mapping_from(event, "emotion", "emotionContext", "emotion_state"),
            memory_candidates=memory_candidates,
            execution_result=_mapping_from(event, "executionResult", "execution_result"),
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
            allow_speech=bool(_first_value(event, "allowSpeech", "allow_speech", default=True)),
        )
        action = {
            "type": "proactive_activity",
            "interruptible": True,
            "proposal": proposal,
        }
        return self._envelope(
            round_id=turn.round_id,
            conversation_state="proactive",
            lane="proactive_activity",
            actions=[action],
            blackboard_patch={
                "proactiveActivity": {
                    "proposal": proposal,
                    "interruptible": True,
                    "voiceRoundId": event.get("roundId"),
                }
            },
            memory_hints=[
                {"kind": "activity_context", "stable": False, "candidate": dict(item)}
                for item in memory_candidates
                if isinstance(item, Mapping)
            ],
        )

    def _handle_queue_health(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = event["payload"]
        status = str(_first_value(event, "status", "state", "health", default="ok")).lower()
        degraded = status in {"degraded", "backpressure", "unhealthy", "warning"} or bool(payload.get("degraded"))
        round_id = self._current_round_id(event)
        warnings: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        if degraded:
            warnings.append(
                {
                    "type": "queue_health_degraded",
                    "queue": _first_value(event, "queue", "name", default="voice_runtime"),
                    "status": status,
                    "latencyMs": _first_value(event, "latencyMs", "latency_ms"),
                    "recommendation": "降低 verbal density 或暂停主动交互",
                    "source": "voice_runtime_bridge",
                }
            )
            actions.append(
                {
                    "type": "runtime_backpressure",
                    "lowerVerbalDensity": True,
                    "pauseProactiveInteraction": True,
                    "reason": "queue_health_degraded",
                }
            )

        return self._envelope(
            round_id=round_id,
            conversation_state="degraded" if degraded else "monitoring",
            lane="runtime_health",
            actions=actions,
            blackboard_patch={"queueHealth": dict(payload), "warnings": warnings},
            trace=self._trace(
                event,
                lane="runtime_health",
                round_id=round_id,
                cancellation_token=self._current_cancellation_token(event),
            ),
        )

    def _current_round_id(self, event: Mapping[str, Any]) -> str:
        event_round = event.get("roundId")
        if event_round:
            return str(event_round)
        turn = self.scheduler.current_turn()
        if turn is not None:
            return turn.round_id
        return ""

    def _current_cancellation_token(self, event: Mapping[str, Any]) -> str:
        event_token = event.get("cancellationToken")
        if event_token:
            return str(event_token)
        turn = self.scheduler.current_turn()
        if turn is not None:
            return turn.cancellation_token
        return ""

    def _stale_tts_reason(
        self,
        event: Mapping[str, Any],
        *,
        current: Any,
    ) -> str | None:
        if current is None:
            return "stable_content_unavailable"
        event_round = event.get("roundId")
        event_token = event.get("cancellationToken")
        if event_round and str(event_round) != current.round_id:
            return "round_or_token_mismatch"
        if event_token and str(event_token) != current.cancellation_token:
            return "round_or_token_mismatch"
        if not self._has_stable_speaking_content(current):
            if not event_round and not event_token and not current.asr_final:
                return "untagged_tts_without_active_final"
            if not event_token and current.asr_final:
                return None
            return "stable_content_unavailable"
        return None

    def _has_stable_speaking_content(self, turn: Any) -> bool:
        if turn is None:
            return False
        if getattr(turn, "state", None) != "active":
            return False
        cancellation = getattr(turn, "cancellation", None)
        if cancellation is not None and bool(getattr(cancellation, "cancelled", False)):
            return False
        stable_segments = getattr(turn, "stable_speech_segments", None)
        if not isinstance(stable_segments, list) or not stable_segments:
            return False
        if not all(isinstance(item, Mapping) and item.get("stable") is True for item in stable_segments):
            return False
        safety_state = getattr(turn, "safety_state", {})
        if not isinstance(safety_state, Mapping):
            return False
        verdict = safety_state.get("arbiter_verdict")
        if not isinstance(verdict, Mapping):
            return False
        return str(verdict.get("state") or verdict.get("status") or "") == "approved"

    def _trace(
        self,
        event: Mapping[str, Any],
        *,
        lane: str,
        round_id: str,
        cancellation_token: str,
    ) -> dict[str, Any]:
        return {
            "eventName": str(event.get("eventName") or event.get("type") or "UNKNOWN"),
            "roundId": round_id,
            "lane": lane,
            "cancellationToken": cancellation_token,
            "source": "voice_runtime_bridge",
            "timestamp": _trace_timestamp(event, clock=self._clock),
        }

    def _envelope(
        self,
        *,
        round_id: str,
        conversation_state: str,
        lane: str,
        actions: list[dict[str, Any]] | None = None,
        blackboard_patch: dict[str, Any] | None = None,
        memory_hints: list[dict[str, Any]] | None = None,
        speech_plan: dict[str, Any] | None = None,
        interrupt: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
        trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "roundId": round_id,
            "conversationState": conversation_state,
            "lane": lane,
            "actions": actions or [],
            "blackboardPatch": blackboard_patch or {},
            "memoryHints": memory_hints or [],
            "speechPlan": speech_plan,
            "interrupt": interrupt,
            "summary": summary
            or self._summary(
                round_id=round_id,
                lane=lane,
                conversation_state=conversation_state,
                action_count=len(actions or []),
                has_speech_plan=speech_plan is not None,
                stale=False,
            ),
            "trace": trace,
        }

    def _summary(
        self,
        *,
        round_id: str,
        lane: str,
        conversation_state: str,
        action_count: int,
        has_speech_plan: bool,
        stale: bool,
    ) -> dict[str, Any]:
        return {
            "roundId": round_id,
            "lane": lane,
            "conversationState": conversation_state,
            "actionCount": action_count,
            "hasSpeechPlan": has_speech_plan,
            "stale": stale,
        }


def _normalize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = _payload(event)
    event_type = _event_type(event, payload)
    round_id = _first_of(event, payload, "roundId", "round_id", "conversationRoundId", "conversation_round_id")
    cancellation_token = _first_of(event, payload, "cancellationToken", "cancellation_token")
    event_name = _first_of(event, payload, "eiprotocolName", "eventName", default=event_type)
    return {
        "type": event_type,
        "eventName": str(event_name or event_type),
        "roundId": str(round_id) if round_id is not None else None,
        "cancellationToken": str(cancellation_token) if cancellation_token is not None else None,
        "payload": payload,
        "raw": dict(event),
    }


def _payload(event: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("payload", "data", "body", "message"):
        value = event.get(key)
        if isinstance(value, Mapping):
            payload = dict(value)
            for derived_key in (
                "text",
                "partialText",
                "finalText",
                "transcript",
                "final",
                "textType",
                "audioBase64",
                "chunkIndex",
                "sentenceId",
                "reason",
                "roundId",
                "cancellationToken",
                "idleSeconds",
                "emotion",
                "environment",
                "memoryCandidates",
                "executionResult",
                "allowSpeech",
                "status",
                "state",
                "health",
                "queue",
                "latencyMs",
            ):
                if derived_key in event and derived_key not in payload:
                    payload[derived_key] = event[derived_key]
            return payload
    return {
        key: value
        for key, value in event.items()
        if key not in {"type", "eventType", "event_type", "kind", "name", "event"}
    }


def _event_type(event: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    value = _first_of(event, payload, "type", "eventType", "event_type", "kind", "name", "event", "eiprotocolName")
    text = str(value or "UNKNOWN")
    return text.replace(".", "_").replace("-", "_").upper()


def _event_text(event: Mapping[str, Any], *keys: str) -> str:
    value = _first_value(event, *keys)
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _first_value(event: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        payload = {}
    return _first_of(event, payload, *keys, default=default)


def _first_of(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    for key in keys:
        if key in first and first[key] is not None:
            return first[key]
        if key in second and second[key] is not None:
            return second[key]
    return default


def _mapping_from(event: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    value = _first_value(event, *keys)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _sequence_from(event: Mapping[str, Any], *keys: str) -> list[Mapping[str, Any]]:
    value = _first_value(event, *keys)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if isinstance(value, tuple):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _mapping_text(value: Any, key: str) -> str:
    if isinstance(value, Mapping):
        return str(value.get(key) or "")
    return ""


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "final"}


def _trace_timestamp(event: Mapping[str, Any], *, clock: Any) -> float:
    value = _first_value(event, "timestamp")
    if value is not None:
        return _as_float(value)
    return _as_float(clock())


__all__ = ["VoiceRuntimeBridge"]
