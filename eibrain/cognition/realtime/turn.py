"""Standard-library realtime cognition turn orchestration primitives."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping


Clock = Callable[[], float]


@dataclass
class CancellationToken:
    """Serializable cancellation guard for a single realtime round."""

    round_id: str
    token_id: str
    cancelled: bool = False
    reason: str | None = None
    cancelled_at_ts: float | None = None

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled

    def cancel(self, *, reason: str | None = None, at_ts: float | None = None) -> "CancellationToken":
        if not self.cancelled:
            self.cancelled = True
            self.reason = reason
            self.cancelled_at_ts = at_ts
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnBlackboard:
    round_id: str
    cancellation_token: str
    cancellation: CancellationToken | None = None
    state: str = "active"
    asr_partial: list[str] = field(default_factory=list)
    asr_final: str | None = None
    fast_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    stable_decisions: list[dict[str, Any]] = field(default_factory=list)
    intent_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    emotion_state: dict[str, Any] = field(default_factory=dict)
    memory_candidates: list[dict[str, Any]] = field(default_factory=list)
    memory_traces: list[dict[str, Any]] = field(default_factory=list)
    tool_candidates: list[dict[str, Any]] = field(default_factory=list)
    persona_state: dict[str, Any] = field(default_factory=dict)
    safety_state: dict[str, Any] = field(default_factory=dict)
    speech_plan: dict[str, Any] | None = None
    action_plan: list[dict[str, Any]] = field(default_factory=list)
    stable_speech_segments: list[dict[str, Any]] = field(default_factory=list)
    created_at_ts: float = 0.0
    updated_at_ts: float = 0.0
    interrupted_at_ts: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def status_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["summary"] = self.operator_summary()
        payload["trace"] = self.operator_trace()
        return payload

    def operator_summary(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "cancellation_token": self.cancellation_token,
            "state": self.state,
            "fast_hypothesis_count": len(self.fast_hypotheses),
            "stable_decision_count": len(self.stable_decisions),
            "stable_speech_count": len(self.stable_speech_segments),
            "action_count": len(self.action_plan),
            "cancelled": bool(self.cancellation and self.cancellation.cancelled),
        }

    def operator_trace(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "cancellation_token": self.cancellation_token,
            "state": self.state,
            "created_at_ts": self.created_at_ts,
            "updated_at_ts": self.updated_at_ts,
            "interrupted_at_ts": self.interrupted_at_ts,
            "cancelled": bool(self.cancellation and self.cancellation.cancelled),
            "cancellation_reason": self.cancellation.reason if self.cancellation is not None else None,
        }


@dataclass(frozen=True)
class FastThinkResult:
    round_id: str
    cancellation_token: str
    deadline_ms: int
    microfeedback: str
    intent_hypotheses: list[dict[str, Any]]
    stable: bool = False
    source: str = "fast_think"


class RealtimeTurnManager:
    """Manage turn lifecycle and cancellation semantics."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock = clock or time.time
        self._round_index = 0
        self._current_turn: TurnBlackboard | None = None
        self._history: dict[str, TurnBlackboard] = {}

    def start_round(self, *, reason: str | None = None) -> TurnBlackboard:
        now = self._clock()
        self._round_index += 1
        previous_turn = self._current_turn
        if previous_turn is not None and previous_turn.state not in {"cancelled", "interrupted"}:
            previous_turn.state = "completed"
            previous_turn.updated_at_ts = now
            if previous_turn.cancellation is not None:
                previous_turn.cancellation.cancel(reason="superseded", at_ts=now)

        round_id = f"round-{self._round_index}"
        token = uuid.uuid4().hex
        turn = TurnBlackboard(
            round_id=round_id,
            cancellation_token=token,
            cancellation=CancellationToken(round_id=round_id, token_id=token),
            state="active",
            created_at_ts=now,
            updated_at_ts=now,
        )
        if reason is not None:
            turn.safety_state["round_reason"] = reason

        self._history[round_id] = turn
        self._current_turn = turn
        return turn

    def current_turn(self) -> TurnBlackboard | None:
        return self._current_turn

    def is_current(self, *, round_id: str, cancellation_token: str) -> bool:
        current = self._current_turn
        if current is None:
            return False
        return (
            current.round_id == round_id
            and current.cancellation_token == cancellation_token
            and current.state == "active"
            and (current.cancellation is None or not current.cancellation.cancelled)
        )

    def reject_if_cancelled(self, *, round_id: str, cancellation_token: str) -> None:
        if not self.is_current(round_id=round_id, cancellation_token=cancellation_token):
            raise RuntimeError("round/token is not current or already cancelled")

    def observe_partial(
        self,
        *,
        round_id: str,
        cancellation_token: str,
        asr_text: str,
    ) -> TurnBlackboard:
        self.reject_if_cancelled(round_id=round_id, cancellation_token=cancellation_token)
        if self._current_turn is None:
            raise RuntimeError("no active turn")
        now = self._clock()
        text = asr_text.strip()
        if text:
            self._current_turn.asr_partial.append(text)
        self._current_turn.updated_at_ts = now
        self._current_turn.state = "active"
        return self._current_turn

    def finalize_asr(
        self,
        *,
        round_id: str,
        cancellation_token: str,
        asr_text: str,
    ) -> TurnBlackboard:
        self.reject_if_cancelled(round_id=round_id, cancellation_token=cancellation_token)
        if self._current_turn is None:
            raise RuntimeError("no active turn")
        now = self._clock()
        self._current_turn.asr_final = asr_text.strip() or None
        self._current_turn.updated_at_ts = now
        return self._current_turn

    def write_fast_hypothesis(
        self,
        *,
        round_id: str,
        cancellation_token: str,
        hypothesis: Mapping[str, Any],
        source: str = "fast_lane",
    ) -> dict[str, Any]:
        self.reject_if_cancelled(round_id=round_id, cancellation_token=cancellation_token)
        if self._current_turn is None:
            raise RuntimeError("no active turn")
        if hypothesis.get("stable") is True or "decision" in hypothesis:
            raise ValueError("fast lane may only record non-stable hypotheses")
        now = self._clock()
        payload = dict(hypothesis)
        payload.update(
            {
                "round_id": round_id,
                "cancellation_token": cancellation_token,
                "stable": False,
                "source": source,
                "created_at_ts": now,
            }
        )
        self._current_turn.fast_hypotheses.append(payload)
        self._current_turn.updated_at_ts = now
        return payload

    def commit_stable_decision(
        self,
        *,
        round_id: str,
        cancellation_token: str,
        decision: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.reject_if_cancelled(round_id=round_id, cancellation_token=cancellation_token)
        if self._current_turn is None:
            raise RuntimeError("no active turn")
        if decision.get("stable") is not True:
            raise ValueError("stable decisions must set stable=True")

        speech_segments = list(decision.get("speech_segments") or [])
        if any(segment.get("stable") is not True for segment in speech_segments):
            raise ValueError("stable decision speech_segments must be stable")
        action_segments = list(
            decision.get("action_segments")
            or decision.get("action_plan")
            or decision.get("actions")
            or []
        )
        if any(segment.get("stable") is not True for segment in action_segments):
            raise ValueError("stable decision action segments must be stable")

        now = self._clock()
        payload = dict(decision)
        payload.update(
            {
                "round_id": round_id,
                "cancellation_token": cancellation_token,
                "stable": True,
                "created_at_ts": now,
            }
        )
        self._current_turn.stable_decisions.append(payload)
        self._current_turn.stable_speech_segments = speech_segments
        self._current_turn.action_plan = [dict(segment) for segment in action_segments]
        self._current_turn.updated_at_ts = now
        return payload

    def interrupt(self, *, reason: str | None = None) -> TurnBlackboard:
        current = self._current_turn
        now = self._clock()
        if current is not None and current.state != "cancelled":
            current.state = "interrupted"
            current.interrupted_at_ts = now
            current.updated_at_ts = now
            if current.cancellation is not None:
                current.cancellation.cancel(reason=reason, at_ts=now)
            if reason is not None:
                current.safety_state["interrupt_reason"] = reason

        return self.start_round(reason=reason or "interrupted")

    def status_payload(self) -> dict[str, Any]:
        current = self._current_turn
        return {
            "active": current is not None and current.state == "active",
            "current_round_id": current.round_id if current is not None else None,
            "current": current.status_payload() if current is not None else None,
            "history": {round_id: turn.status_payload() for round_id, turn in self._history.items()},
            "round_count": self._round_index,
        }

    def status(self) -> dict[str, Any]:
        return self.status_payload()


class FastThinkEngine:
    """Generate low-risk microfeedback and intent hypotheses from partial ASR."""

    def process_partial(
        self,
        turn: TurnBlackboard,
        asr_text: str,
        *,
        deadline_ms: int = 500,
    ) -> FastThinkResult:
        text = (asr_text or "").strip()
        ms = min(500, max(50, deadline_ms))
        microfeedback = "我先听你说完，再给你更完整的回应。"
        hypotheses: list[dict[str, Any]] = []

        if not text:
            return FastThinkResult(
                round_id=turn.round_id,
                cancellation_token=turn.cancellation_token,
                deadline_ms=ms,
                microfeedback="我还在等待你的后续内容，请继续说。",
                intent_hypotheses=[],
                stable=False,
            )

        if any(marker in text for marker in ("吗", "？", "?", "什么", "如何", "谁", "多少")):
            microfeedback = "我先确认你的问题意图再回应。"
            hypotheses.append({"intent": "clarify_question", "confidence": 0.61, "label": "question"})
        if any(marker in text for marker in ("开", "播放", "打开", "启动")):
            microfeedback = "我先准备一条可执行的尝试性回应。"
            hypotheses.append({"intent": "action_oriented", "confidence": 0.73, "label": "action_intent"})
        if not hypotheses:
            hypotheses.append({"intent": "listen", "confidence": 0.45, "label": "ambient"})
        # do not perform commitments in partial mode
        turn.intent_hypotheses = hypotheses
        return FastThinkResult(
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
            deadline_ms=ms,
            microfeedback=microfeedback,
            intent_hypotheses=list(hypotheses),
            stable=False,
        )


class SpeechActionPlanner:
    """Build structured speech+action plan with fallback text for action failure."""

    _DEFAULT_FALLBACK = "我刚才理解有点歧义，我先用语言说明一下。"

    def plan(
        self,
        turn: TurnBlackboard,
        *,
        speech_text: str | None = None,
        emotion: str = "warm",
        action_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        action_results = action_results or []
        action_failures = [result for result in action_results if not bool(result.get("ok"))]
        action_plan = self._build_actions(turn, action_failures)
        fallback_mode = bool(action_failures)

        speech_segments = self._build_speech_segments(
            turn=turn,
            action_failed=fallback_mode,
            speech_text=speech_text,
            emotion=emotion,
            fallback_text=self._failure_fallback(action_failures, turn),
        )
        plan = {
            "round_id": turn.round_id,
            "cancellation_token": turn.cancellation_token,
            "language": "zh-CN",
            "stable": bool(speech_segments and all(segment.get("stable", False) for segment in speech_segments)),
            "speech": speech_segments,
            "actions": action_plan,
            "speech_segments": speech_segments,
            "action_plan": action_plan,
            "startOffsetMs": 0,
        }
        turn.speech_plan = {
            "language": plan["language"],
            "speech": speech_segments,
            "speech_segments": speech_segments,
            "stable": plan["stable"],
        }
        turn.action_plan = action_plan
        turn.stable_speech_segments = [segment for segment in speech_segments if segment.get("stable", False)]
        return plan

    def _build_actions(
        self,
        turn: TurnBlackboard,
        action_failures: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        base_start_ms = 120
        plans: list[dict[str, Any]] = []
        for idx, item in enumerate(turn.tool_candidates):
            capability_id = item.get("capabilityId") or item.get("capability_id") or item.get("name") or item.get("type")
            plans.append(
                {
                    "type": item.get("type", "tool_action"),
                    "name": item.get("name", f"tool_action_{idx}"),
                    "capabilityId": capability_id,
                    "payload": dict(item),
                    "startOffsetMs": int(item.get("startOffsetMs", base_start_ms + idx * 120)),
                    "durationMs": item.get("durationMs"),
                    "style": item.get("style"),
                    "status": "ready",
                }
            )
        for failure in action_failures:
            if "id" in failure:
                plans.append(
                    {
                        "type": "retry_action",
                        "action_id": failure.get("id"),
                        "capabilityId": failure.get("capabilityId") or failure.get("capability_id"),
                        "startOffsetMs": base_start_ms + len(plans) * 120,
                        "status": "retry",
                        "reason": failure.get("reason"),
                    }
                )
        return plans

    def _failure_fallback(
        self,
        action_failures: list[dict[str, Any]],
        turn: TurnBlackboard,
    ) -> str:
        if not action_failures:
            return ""
        if turn.asr_final:
            return self._DEFAULT_FALLBACK
        return "我先确认一下你的指令再执行。"

    def _build_speech_segments(
        self,
        *,
        turn: TurnBlackboard,
        action_failed: bool,
        speech_text: str | None,
        emotion: str,
        fallback_text: str,
    ) -> list[dict[str, Any]]:
        if action_failed:
            return [
                {
                    "text": fallback_text,
                    "emotion": emotion,
                    "startOffsetMs": 0,
                    "stable": True,
                    "source": "action_fallback",
                }
            ]
        if not turn.asr_final:
            return []
        text = (speech_text or "").strip() or "我听到了，我正在处理。"
        return [
            {
                "text": text,
                "emotion": emotion,
                "startOffsetMs": 0,
                "stable": True,
                "source": "slow_reasoner" if speech_text else "safe_ack",
            }
        ]


class ResponseArbiter:
    """Guard stable outputs before speaking; prevents hypothesis or stale turn leakage."""

    def allow_speaking(
        self,
        turn_manager: RealtimeTurnManager,
        turn: TurnBlackboard,
        plan: Mapping[str, Any],
    ) -> bool:
        if not bool(plan.get("hypothesis", False)) and not bool(plan.get("is_hypothesis", False)):
            if not turn_manager.is_current(
                round_id=plan.get("round_id", turn.round_id),
                cancellation_token=plan.get("cancellation_token", turn.cancellation_token),
            ):
                return False
            if turn.state not in {"active", "completed"}:
                return False
            if plan.get("stable") is not True:
                return False
            if not plan.get("speech_segments"):
                return False
            return True
        return False


class InterruptionController:
    """Create cancellation action summaries for interruption events."""

    def summarize(
        self,
        *,
        old_turn: TurnBlackboard,
        new_turn: TurnBlackboard,
        reason: str = "user_interrupt",
    ) -> dict[str, Any]:
        return {
            "reason": reason,
            "stop_tts": True,
            "cancel_generation": True,
            "cancel_actions": True,
            "mark_interrupted": {
                "round_id": old_turn.round_id,
                "cancellation_token": old_turn.cancellation_token,
                "at_ts": old_turn.interrupted_at_ts,
                "state": old_turn.state,
            },
            "start_new_round": {
                "round_id": new_turn.round_id,
                "cancellation_token": new_turn.cancellation_token,
            },
        }

    def interrupt_and_start_new_round(
        self,
        manager: RealtimeTurnManager,
        *,
        reason: str = "user_interrupt",
    ) -> dict[str, Any]:
        old_turn = manager.current_turn()
        if old_turn is None:
            new_turn = manager.start_round(reason=reason)
            return self.summarize(
                old_turn=TurnBlackboard(
                    round_id="none",
                    cancellation_token="none",
                    state="unknown",
                    created_at_ts=0.0,
                    updated_at_ts=0.0,
                ),
                new_turn=new_turn,
                reason=reason,
            )
        new_turn = manager.interrupt(reason=reason)
        return self.summarize(old_turn=old_turn, new_turn=new_turn, reason=reason)


# Compatibility: historical imports from ``realtime.turn`` now delegate to the
# focused Task 3 modules while keeping RealtimeTurnManager and TurnBlackboard in
# this file.
from .arbiter import ResponseArbiter as ResponseArbiter
from .interruption import InterruptionController as InterruptionController
from .planner import SpeechActionPlanner as SpeechActionPlanner
