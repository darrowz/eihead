"""Speech and action planning for realtime cognition rounds."""

from __future__ import annotations

from typing import Any, Mapping

from .persona import resolve_voice_style_policy
from .turn import TurnBlackboard


class SpeechActionPlanner:
    """Build JSON-ready speech/action segments from stable round state."""

    _DEFAULT_FALLBACK = "我现在不能直接执行这个动作，但我可以先用语言说明。"

    def plan(
        self,
        turn: TurnBlackboard,
        *,
        speech_text: str | None = None,
        emotion: str = "warm",
        action_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        action_results = action_results or []
        action_segments = self._build_action_segments(turn.tool_candidates, action_results)
        fallback = self._fallback(turn.tool_candidates, action_results)
        if fallback:
            action_segments = self._suppress_ready_actions(action_segments, fallback)
        speech_segments = self._build_speech_segments(
            turn=turn,
            speech_text=speech_text,
            emotion=emotion,
            fallback=fallback,
        )
        stable = bool(speech_segments) and all(segment.get("stable") is True for segment in speech_segments)
        alignment = self._alignment(
            action_segments=action_segments,
            fallback=fallback,
            speech_text=self._speech_text_for_alignment(speech_text=speech_text, fallback=fallback),
        )

        plan = {
            "round_id": turn.round_id,
            "cancellation_token": turn.cancellation_token,
            "language": "zh-CN",
            "stable": stable,
            "speech_segments": speech_segments,
            "action_segments": action_segments,
            "speech": speech_segments,
            "actions": action_segments,
            "speechSegments": speech_segments,
            "actionSegments": action_segments,
            "action_plan": action_segments,
            "speech_action_alignment": alignment,
            "startOffsetMs": 0,
        }
        turn.speech_plan = plan
        turn.action_plan = action_segments
        turn.stable_speech_segments = [segment for segment in speech_segments if segment.get("stable") is True]
        return plan

    def _build_action_segments(
        self,
        tool_candidates: list[dict[str, Any]],
        action_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        segments = [
            self._candidate_to_action_segment(candidate, idx)
            for idx, candidate in enumerate(tool_candidates)
        ]
        for result in action_results:
            if result.get("ok", True):
                continue
            segments.append(self._failed_result_to_action_segment(result, len(segments)))
        return segments

    def _candidate_to_action_segment(
        self,
        candidate: Mapping[str, Any],
        idx: int,
    ) -> dict[str, Any]:
        capability_id = (
            candidate.get("capabilityId")
            or candidate.get("capability_id")
            or candidate.get("name")
            or candidate.get("type")
            or f"action.{idx}"
        )
        available = candidate.get("available", True) is not False
        return {
            "capabilityId": str(capability_id),
            "startOffsetMs": int(candidate.get("startOffsetMs", 120 + idx * 120)),
            "durationMs": int(candidate.get("durationMs", 0)),
            "style": str(candidate.get("style", "default")),
            "payload": dict(candidate.get("payload") or {}),
            "status": "ready" if available else "unavailable",
        }

    def _failed_result_to_action_segment(
        self,
        result: Mapping[str, Any],
        idx: int,
    ) -> dict[str, Any]:
        capability_id = (
            result.get("capabilityId")
            or result.get("capability_id")
            or result.get("id")
            or f"action.failure.{idx}"
        )
        return {
            "capabilityId": str(capability_id),
            "startOffsetMs": int(result.get("startOffsetMs", 120 + idx * 120)),
            "durationMs": int(result.get("durationMs", 0)),
            "style": str(result.get("style", "fallback")),
            "payload": dict(result.get("payload") or {}),
            "status": "retry",
            "reason": result.get("reason"),
        }

    def _fallback(
        self,
        tool_candidates: list[dict[str, Any]],
        action_results: list[dict[str, Any]],
    ) -> dict[str, str] | None:
        for candidate in tool_candidates:
            if candidate.get("available", True) is False:
                return {
                    "text": str(candidate.get("fallbackText") or candidate.get("fallback_text") or self._DEFAULT_FALLBACK),
                    "source": "action_unavailable_fallback",
                }
        for result in action_results:
            if not result.get("ok", True):
                return {
                    "text": str(result.get("fallbackText") or result.get("fallback_text") or self._DEFAULT_FALLBACK),
                    "source": "action_fallback",
                }
        return None

    def _suppress_ready_actions(
        self,
        action_segments: list[dict[str, Any]],
        fallback: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        suppressed: list[dict[str, Any]] = []
        for segment in action_segments:
            next_segment = dict(segment)
            if next_segment.get("status") == "ready":
                next_segment["status"] = "cancelled"
                next_segment["reason"] = str(fallback.get("source") or "fallback_active")
            suppressed.append(next_segment)
        return suppressed

    def _alignment(
        self,
        *,
        action_segments: list[dict[str, Any]],
        fallback: Mapping[str, str] | None,
        speech_text: str,
    ) -> dict[str, Any]:
        ready_count = sum(1 for segment in action_segments if segment.get("status") == "ready")
        capability_ids = [str(segment.get("capabilityId") or "") for segment in action_segments if segment.get("status") == "ready"]
        semantic_matches = [
            capability_id
            for capability_id in capability_ids
            if self._speech_mentions_capability(speech_text, capability_id)
        ]
        semantic_checked = bool(capability_ids and speech_text.strip())
        return {
            "consistent": ready_count == 0 if fallback else not semantic_checked or len(semantic_matches) == len(capability_ids),
            "ready_action_count": ready_count,
            "fallback_active": fallback is not None,
            "semantic_checked": semantic_checked,
            "matched_capability_count": len(semantic_matches),
            "capability_ids": capability_ids,
        }

    def _speech_text_for_alignment(
        self,
        *,
        speech_text: str | None,
        fallback: Mapping[str, str] | None,
    ) -> str:
        if fallback:
            return str(fallback.get("text") or "")
        return str(speech_text or "")

    def _speech_mentions_capability(self, speech_text: str, capability_id: str) -> bool:
        text = speech_text.lower()
        capability = capability_id.lower()
        if not text or not capability:
            return False
        keyword_groups = {
            "light": ("灯", "light", "照明", "打开", "关闭"),
            "speech": ("说", "讲", "播放", "语音", "speech"),
            "neck": ("转", "看", "点头", "云台", "头", "neck", "pan"),
            "gimbal": ("转", "看", "云台", "gimbal", "pan"),
            "memory": ("记", "提醒", "remember", "memory"),
            "action": ("执行", "处理", "指令", "action"),
        }
        for marker, keywords in keyword_groups.items():
            if marker in capability and any(keyword in text for keyword in keywords):
                return True
        tail = capability.rsplit(".", 1)[-1].replace("_", " ")
        return bool(tail and tail in text)

    def _build_speech_segments(
        self,
        *,
        turn: TurnBlackboard,
        speech_text: str | None,
        emotion: str,
        fallback: dict[str, str] | None,
    ) -> list[dict[str, Any]]:
        voice_style = self._voice_style(turn, emotion=emotion)
        if fallback:
            return [
                self._with_voice_style(
                    {
                        "text": fallback["text"],
                        "emotion": emotion,
                        "startOffsetMs": 0,
                        "stable": True,
                        "source": fallback["source"],
                    },
                    voice_style,
                )
            ]

        text = (speech_text or "").strip()
        if not text and turn.asr_final:
            text = "我听到了，我正在处理。"
        if not text:
            return []
        return [
            self._with_voice_style(
                {
                    "text": text,
                    "emotion": emotion,
                    "startOffsetMs": 0,
                    "stable": True,
                    "source": "slow_reasoner" if speech_text else "safe_ack",
                },
                voice_style,
            )
        ]

    def _voice_style(self, turn: TurnBlackboard, *, emotion: str) -> dict[str, Any] | None:
        if not turn.persona_state and not turn.emotion_state:
            return None
        return resolve_voice_style_policy(
            turn.persona_state,
            turn.emotion_state,
            fallback_emotion=emotion,
        )

    def _with_voice_style(
        self,
        segment: dict[str, Any],
        voice_style: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if not voice_style:
            return segment
        segment["emotion"] = str(voice_style.get("emotion") or segment.get("emotion") or "warm")
        segment["voice_style"] = str(voice_style.get("voice_style") or "warm")
        segment["voice_code"] = str(voice_style.get("voice_code") or "")
        segment["speed"] = float(voice_style.get("speed") or 1.0)
        segment["volume"] = float(voice_style.get("volume") or 0.8)
        return segment
