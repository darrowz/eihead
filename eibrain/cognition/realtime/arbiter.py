"""Stable response arbitration for realtime cognition rounds."""

from __future__ import annotations

from typing import Any, Mapping

from .turn import RealtimeTurnManager, TurnBlackboard


class ResponseArbiter:
    """Prevent unstable, stale, or cancelled output from reaching speech/action lanes."""

    def allow_speaking(
        self,
        turn_manager: RealtimeTurnManager,
        turn: TurnBlackboard,
        plan: Mapping[str, Any],
    ) -> bool:
        if not isinstance(plan, Mapping):
            return False
        if plan.get("hypothesis") is True or plan.get("is_hypothesis") is True:
            return False
        if plan.get("stable") is not True:
            return False

        round_id = str(plan.get("round_id") or turn.round_id)
        cancellation_token = str(plan.get("cancellation_token") or turn.cancellation_token)
        if not turn_manager.is_current(round_id=round_id, cancellation_token=cancellation_token):
            return False
        if turn.round_id != round_id or turn.cancellation_token != cancellation_token:
            return False
        if turn.state != "active":
            return False
        if turn.cancellation is not None and turn.cancellation.cancelled:
            return False

        speech_segments = self._speech_segments(plan)
        action_segments = self._action_segments(plan)
        if speech_segments is None or action_segments is None or not speech_segments:
            return False
        if not all(self._is_stable_speech_segment(segment) for segment in speech_segments):
            return False
        return all(self._is_structured_action_segment(segment) for segment in action_segments)

    def _speech_segments(self, plan: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
        return self._segments_from(plan, ("speech_segments", "speechSegments", "speech"))

    def _action_segments(self, plan: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
        return self._segments_from(
            plan,
            ("action_segments", "actionSegments", "action_plan", "actionPlan", "actions"),
        )

    def _segments_from(
        self,
        plan: Mapping[str, Any],
        keys: tuple[str, ...],
    ) -> list[Mapping[str, Any]] | None:
        segments: Any = []
        for key in keys:
            if key in plan:
                segments = plan.get(key) or []
                break
        if not isinstance(segments, (list, tuple)):
            return None
        if not all(isinstance(segment, Mapping) for segment in segments):
            return None
        return list(segments)

    def _is_stable_speech_segment(self, segment: Mapping[str, Any]) -> bool:
        return (
            segment.get("stable") is True
            and isinstance(segment.get("text"), str)
            and segment.get("text", "").strip() != ""
            and isinstance(segment.get("startOffsetMs"), int)
            and int(segment.get("startOffsetMs", -1)) >= 0
        )

    def _is_structured_action_segment(self, segment: Mapping[str, Any]) -> bool:
        return (
            isinstance(segment.get("capabilityId"), str)
            and segment.get("capabilityId", "").strip() != ""
            and isinstance(segment.get("startOffsetMs"), int)
            and isinstance(segment.get("durationMs"), int)
            and int(segment.get("startOffsetMs", -1)) >= 0
            and int(segment.get("durationMs", -1)) >= 0
            and isinstance(segment.get("style"), str)
        )
