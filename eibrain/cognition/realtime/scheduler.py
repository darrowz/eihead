"""High-level facade for realtime cognitive scheduling."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from .activity import ProactiveActivityManager
from .arbiter import ResponseArbiter
from .fast import FastThinkEngine
from .interruption import InterruptionController
from .memory import MemoryOrchestrator
from .persona import PersonaRuntime
from .slow import SlowReasoner
from .turn import (
    RealtimeTurnManager,
    TurnBlackboard,
)


Clock = Callable[[], float]


class RealtimeCognitiveScheduler:
    """Coordinate partial observation, final decisions, interruption, and snapshots."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        turn_manager: RealtimeTurnManager | None = None,
        fast_engine: FastThinkEngine | None = None,
        slow_reasoner: SlowReasoner | None = None,
        activity_manager: ProactiveActivityManager | None = None,
        arbiter: ResponseArbiter | None = None,
        interruption_controller: InterruptionController | None = None,
        memory_orchestrator: MemoryOrchestrator | None = None,
    ) -> None:
        self._clock = clock or time.time
        self.turn_manager = turn_manager or RealtimeTurnManager(clock=self._clock)
        self.fast_engine = fast_engine or FastThinkEngine()
        self.slow_reasoner = slow_reasoner or SlowReasoner()
        self.activity_manager = activity_manager or ProactiveActivityManager()
        self.arbiter = arbiter or ResponseArbiter()
        self.interruption_controller = interruption_controller or InterruptionController()
        self.memory_orchestrator = memory_orchestrator or MemoryOrchestrator()

    def observe_partial(
        self,
        asr_text: str | None = None,
        *,
        text: str | None = None,
        partial_text: str | None = None,
        round_id: str | None = None,
        cancellation_token: str | None = None,
        persona_context: Mapping[str, Any] | None = None,
        emotion_context: Mapping[str, Any] | None = None,
        environment_context: Mapping[str, Any] | None = None,
        memory_candidates: Sequence[Mapping[str, Any]] | None = None,
        session_id: str | None = None,
        actor_id: str | None = None,
        task_context: Mapping[str, Any] | None = None,
        deadline_ms: int = 500,
    ) -> dict[str, Any]:
        observed_text = _first_text(asr_text, text, partial_text)
        turn = self._active_turn(reason="partial_observation")
        self._guard_if_explicit(turn=turn, round_id=round_id, cancellation_token=cancellation_token)
        self._merge_context(
            turn,
            persona_context=persona_context,
            emotion_context=emotion_context,
            environment_context=environment_context,
        )

        self.turn_manager.observe_partial(
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
            asr_text=observed_text,
        )
        fast_started = time.perf_counter()
        fast_result = self.fast_engine.process_partial(turn, observed_text, deadline_ms=deadline_ms)
        fast_latency_ms = round(max(0.0, time.perf_counter() - fast_started) * 1000, 3)
        fast_payload = _to_dict(fast_result)
        context_summary = fast_payload.get("context_summary")
        if not isinstance(context_summary, Mapping):
            context_summary = _context_summary(turn)
            fast_payload["context_summary"] = context_summary
        microfeedback = _first_text(
            fast_payload.get("microfeedback"),
            _mapping_text(fast_payload.get("micro_feedback"), "text"),
        )
        intent_hypotheses = _first_list(
            fast_payload.get("intent_hypotheses"),
            fast_payload.get("intent_hints"),
        )
        stored_hypothesis = self.turn_manager.write_fast_hypothesis(
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
            hypothesis={
                "partial_text": observed_text,
                "microfeedback": microfeedback,
                "intent_hypotheses": intent_hypotheses,
                "deadline_ms": fast_payload.get("deadline_ms", deadline_ms),
                "context_summary": context_summary,
                "stable": False,
            },
            source="scheduler_fast_lane",
        )
        prefetch = self._prefetch_memory(
            turn=turn,
            text=observed_text,
            session_id=session_id,
            actor_id=actor_id,
            task_context=task_context,
        )
        if memory_candidates:
            prefetch.extend(_normalize_memory_candidates(memory_candidates, source="caller_memory"))
        turn.memory_candidates = _merge_memory_candidates(turn.memory_candidates, prefetch)
        self._set_lane_metric(turn, "fast", latency_ms=fast_latency_ms)

        return {
            "round_id": turn.round_id,
            "cancellation_token": turn.cancellation_token,
            "fast": {
                **fast_payload,
                "microfeedback": microfeedback,
                "intent_hypotheses": intent_hypotheses,
                "stored_hypothesis": stored_hypothesis,
            },
            "memory_prefetch": prefetch,
            "summary": self._observe_summary(turn, lane="fast"),
            "trace": self._operator_trace(turn, source="realtime_cognitive_scheduler", lane="fast"),
            "turn": turn.to_dict(),
        }

    def observe_final(
        self,
        final_text: str | None = None,
        *,
        text: str | None = None,
        round_id: str | None = None,
        cancellation_token: str | None = None,
        memory_candidates: Sequence[Mapping[str, Any]] | None = None,
        persona_context: Mapping[str, Any] | None = None,
        emotion_context: Mapping[str, Any] | None = None,
        environment_context: Mapping[str, Any] | None = None,
        session_id: str | None = None,
        actor_id: str | None = None,
        task_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        observed_text = _first_text(final_text, text)
        turn = self._active_turn(reason="final_observation")
        self._guard_if_explicit(turn=turn, round_id=round_id, cancellation_token=cancellation_token)
        self._merge_context(
            turn,
            persona_context=persona_context,
            emotion_context=emotion_context,
            environment_context=environment_context,
        )
        if memory_candidates:
            incoming = _normalize_memory_candidates(memory_candidates, source="caller_memory")
            turn.memory_candidates = _merge_memory_candidates(incoming, turn.memory_candidates)

        self.turn_manager.finalize_asr(
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
            asr_text=observed_text,
        )
        self._set_lane_metric(turn, "slow", latency_ms=self._lane_latency_ms(turn, "slow"))
        return {
            "round_id": turn.round_id,
            "cancellation_token": turn.cancellation_token,
            "final_text": turn.asr_final,
            "memory_candidates": [dict(item) for item in turn.memory_candidates],
            "summary": self._observe_summary(turn, lane="slow"),
            "trace": self._operator_trace(turn, source="realtime_cognitive_scheduler", lane="slow"),
            "turn": turn.to_dict(),
        }

    def decide(
        self,
        *,
        round_id: str | None = None,
        cancellation_token: str | None = None,
        final_text: str | None = None,
        fast_hypotheses: Sequence[Mapping[str, Any]] | None = None,
        memory_candidates: Sequence[Mapping[str, Any]] | None = None,
        persona_context: Mapping[str, Any] | None = None,
        emotion_context: Mapping[str, Any] | None = None,
        execution_result: Mapping[str, Any] | None = None,
        idle_seconds: float = 0.0,
        session_id: str | None = None,
        actor_id: str | None = None,
        task_context: Mapping[str, Any] | None = None,
        auto_commit_memory: bool = True,
    ) -> dict[str, Any]:
        turn = self._active_turn(reason="decide")
        requested_round_id = round_id or turn.round_id
        requested_token = cancellation_token or turn.cancellation_token
        self.turn_manager.reject_if_cancelled(
            round_id=requested_round_id,
            cancellation_token=requested_token,
        )
        self._merge_context(turn, persona_context=persona_context, emotion_context=emotion_context)
        if memory_candidates:
            incoming = _normalize_memory_candidates(memory_candidates, source="caller_memory")
            turn.memory_candidates = _merge_memory_candidates(incoming, turn.memory_candidates)
        self._apply_persona_memory_guardrails(turn)

        slow_started = time.perf_counter()
        decision = self.slow_reasoner.decide(
            turn=turn,
            round_id=requested_round_id,
            cancellation_token=requested_token,
            final_text=final_text,
            fast_hypotheses=fast_hypotheses,
            memory_candidates=turn.memory_candidates,
            persona_context=turn.persona_state,
            emotion_context=turn.emotion_state,
            execution_result=execution_result,
        )
        committed = self.turn_manager.commit_stable_decision(
            round_id=requested_round_id,
            cancellation_token=requested_token,
            decision=decision,
        )
        slow_latency_ms = round(max(0.0, time.perf_counter() - slow_started) * 1000, 3)
        turn.speech_plan = {
            "stable": True,
            "speech_segments": list(committed.get("speech_segments", [])),
            "action_segments": list(committed.get("action_segments", committed.get("action_plan", []))),
            "action_plan": list(committed.get("action_plan", [])),
            "actions": list(committed.get("actions", committed.get("action_plan", []))),
            "language": committed.get("persona", {}).get("language", "zh-CN"),
            "source": "realtime_cognitive_scheduler",
        }
        turn.action_plan = list(committed.get("action_plan", []))
        arbiter_started = time.perf_counter()
        can_speak = self.arbiter.allow_speaking(self.turn_manager, turn, committed)
        arbiter_latency_ms = round(max(0.0, time.perf_counter() - arbiter_started) * 1000, 3)
        turn.safety_state["arbiter_verdict"] = {
            "state": "approved" if can_speak else "blocked",
            "status": "approved" if can_speak else "blocked",
            "can_speak": can_speak,
            "round_id": turn.round_id,
            "cancellation_token": turn.cancellation_token,
        }
        self._set_lane_metric(turn, "slow", latency_ms=slow_latency_ms)
        self._set_lane_metric(turn, "arbiter", latency_ms=arbiter_latency_ms)
        self._set_lane_metric(turn, "speaking", latency_ms=round(slow_latency_ms + arbiter_latency_ms, 3))
        activity = self.activity_manager.propose(
            idle_seconds=idle_seconds,
            emotion_context=turn.emotion_state,
            memory_candidates=turn.memory_candidates,
            execution_result=execution_result,
            round_id=turn.round_id,
            cancellation_token=turn.cancellation_token,
        )
        turn.safety_state["proactive_activity"] = activity
        memory_trace = (
            self._auto_commit_decision_memory(
                turn,
                committed=committed,
                session_id=session_id,
                actor_id=actor_id,
                task_context=task_context,
            )
            if auto_commit_memory
            else {}
        )
        reply_memory_trace = self._record_reply_memory_usage(
            turn,
            committed=committed,
            session_id=session_id,
            actor_id=actor_id,
        )
        if reply_memory_trace:
            memory_trace = reply_memory_trace
        result = {
            "round_id": turn.round_id,
            "cancellation_token": turn.cancellation_token,
            "decision": committed,
            "can_speak": can_speak,
            "speech_segments": list(committed.get("speech_segments", [])),
            "action_plan": list(committed.get("action_plan", [])),
            "proactive_activity": activity,
            "summary": {
                "round_id": turn.round_id,
                "cancellation_token": turn.cancellation_token,
                "lane": "speaking",
                "decision": committed.get("decision"),
                "can_speak": can_speak,
                "speech_count": len(list(committed.get("speech_segments", []))),
                "action_count": len(list(committed.get("action_plan", []))),
            },
            "trace": {
                "round_id": turn.round_id,
                "cancellation_token": turn.cancellation_token,
                "fast_hypothesis_count": self._logical_fast_hypothesis_count(turn),
                "stable_decision_count": len(turn.stable_decisions),
                "speaking_state": "approved" if can_speak else "blocked",
                "source": "realtime_cognitive_scheduler",
            },
        }
        if memory_trace:
            result["memory_trace"] = memory_trace
        return result

    def interrupt(self, *, reason: str = "user_interrupt") -> dict[str, Any]:
        summary = self.interruption_controller.interrupt_and_start_new_round(
            self.turn_manager,
            reason=reason,
        )
        old = dict(summary.get("mark_interrupted") or {})
        new = dict(summary.get("start_new_round") or {})
        summary["summary"] = {
            "old_round_id": old.get("round_id"),
            "new_round_id": new.get("round_id"),
            "reason": reason,
            "cancelled": bool(old.get("cancelled")),
        }
        summary["trace"] = {
            "round_id": new.get("round_id"),
            "cancellation_token": new.get("cancellation_token"),
            "interrupted_round_id": old.get("round_id"),
            "source": "realtime_cognitive_scheduler",
        }
        return summary

    def commit_memory_candidates(
        self,
        *,
        session_id: str | None = None,
        actor_id: str | None = None,
        task_context: Mapping[str, Any] | None = None,
        default_modality: str = "audio_text",
        default_organ: str = "ear",
    ) -> dict[str, Any]:
        turn = self._active_turn(reason="memory_commit")
        return self.memory_orchestrator.commit_candidates(
            turn,
            session_id=session_id,
            actor_id=actor_id,
            task_context=task_context,
            default_modality=default_modality,
            default_organ=default_organ,
        )

    def snapshot(self) -> dict[str, Any]:
        payload = self.turn_manager.status_payload()
        current = payload.get("current") or {}
        turn = self.turn_manager.current_turn()
        payload["lanes"] = self._lane_snapshots(self.turn_manager.current_turn())
        payload["scheduler"] = {
            "lane": "realtime_cognitive_scheduler",
            "current_round_id": payload.get("current_round_id"),
            "fast_hypothesis_count": len(current.get("fast_hypotheses") or []),
            "stable_decision_count": len(current.get("stable_decisions") or []),
            "memory_candidate_count": len(current.get("memory_candidates") or []),
            "memory_trace_count": len(current.get("memory_traces") or []),
            "persona": _persona_summary(turn.persona_state if turn is not None else {}),
            "emotion": _emotion_summary(turn.emotion_state if turn is not None else {}),
            "proactive_activity": _proactive_summary(
                turn.safety_state.get("proactive_activity") if turn is not None else None
            ),
        }
        return payload

    def current_turn(self) -> TurnBlackboard | None:
        return self.turn_manager.current_turn()

    def _operator_trace(
        self,
        turn: TurnBlackboard,
        *,
        source: str,
        lane: str,
    ) -> dict[str, Any]:
        return {
            "round_id": turn.round_id,
            "cancellation_token": turn.cancellation_token,
            "source": source,
            "lane": lane,
        }

    def _observe_summary(self, turn: TurnBlackboard, *, lane: str) -> dict[str, Any]:
        return {
            "round_id": turn.round_id,
            "cancellation_token": turn.cancellation_token,
            "lane": lane,
            "state": turn.state,
            "fast_hypothesis_count": self._logical_fast_hypothesis_count(turn),
            "stable_decision_count": len(turn.stable_decisions),
        }

    def _logical_fast_hypothesis_count(self, turn: TurnBlackboard) -> int:
        return sum(
            1
            for item in turn.fast_hypotheses
            if isinstance(item, Mapping) and item.get("source") == "scheduler_fast_lane"
        )

    def _active_turn(self, *, reason: str) -> TurnBlackboard:
        turn = self.turn_manager.current_turn()
        if turn is None or turn.state != "active" or (turn.cancellation is not None and turn.cancellation.cancelled):
            turn = self.turn_manager.start_round(reason=reason)
        return turn

    def _guard_if_explicit(
        self,
        *,
        turn: TurnBlackboard,
        round_id: str | None,
        cancellation_token: str | None,
    ) -> None:
        if round_id is None and cancellation_token is None:
            return
        self.turn_manager.reject_if_cancelled(
            round_id=round_id or turn.round_id,
            cancellation_token=cancellation_token or turn.cancellation_token,
        )

    def _merge_context(
        self,
        turn: TurnBlackboard,
        *,
        persona_context: Mapping[str, Any] | None = None,
        emotion_context: Mapping[str, Any] | None = None,
        environment_context: Mapping[str, Any] | None = None,
    ) -> None:
        if persona_context:
            turn.persona_state.update(dict(persona_context))
        if emotion_context:
            turn.emotion_state.update(_expand_emotion_context(emotion_context))
        if environment_context:
            environment = dict(turn.emotion_state.get("environment", {}))
            environment.update(dict(environment_context))
            turn.emotion_state["environment"] = environment
            nested = turn.emotion_state.get("emotion_state")
            if isinstance(nested, dict):
                nested_environment = dict(nested.get("environment", {}))
                nested_environment.update(dict(environment_context))
                nested["environment"] = nested_environment

    def _set_lane_metric(self, turn: TurnBlackboard, lane: str, *, latency_ms: float) -> None:
        metrics = turn.safety_state.setdefault("lane_metrics", {})
        if isinstance(metrics, dict):
            metrics[str(lane)] = {"latency_ms": round(max(0.0, float(latency_ms)), 3)}

    def _lane_latency_ms(self, turn: TurnBlackboard | None, lane: str) -> float:
        if turn is None:
            return 0.0
        metrics = turn.safety_state.get("lane_metrics")
        if not isinstance(metrics, Mapping):
            return 0.0
        lane_payload = metrics.get(lane)
        if not isinstance(lane_payload, Mapping):
            return 0.0
        latency = lane_payload.get("latency_ms")
        try:
            return round(max(0.0, float(latency)), 3)
        except (TypeError, ValueError):
            return 0.0

    def _lane_snapshots(self, turn: TurnBlackboard | None) -> dict[str, dict[str, Any]]:
        round_id = turn.round_id if turn is not None else ""
        cancellation_token = turn.cancellation_token if turn is not None else ""
        cancellable = bool(
            turn is not None
            and turn.state == "active"
            and (turn.cancellation is None or not turn.cancellation.cancelled)
        )
        fast_hypotheses = list(turn.fast_hypotheses) if turn is not None else []
        stable_decisions = list(turn.stable_decisions) if turn is not None else []
        stable_speech_segments = list(turn.stable_speech_segments) if turn is not None else []
        final_text = str(turn.asr_final or "") if turn is not None else ""
        verdict = turn.safety_state.get("arbiter_verdict") if turn is not None else None
        verdict_state = str(verdict.get("state") or "") if isinstance(verdict, Mapping) else ""
        fast_status = "hypothesis_pending" if fast_hypotheses else "idle"
        slow_status = "stable_committed" if stable_decisions else "decision_pending" if final_text else "idle"
        arbiter_status = verdict_state or ("pending" if stable_decisions else "idle")
        speaking_status = (
            "ready"
            if stable_speech_segments and verdict_state == "approved"
            else "blocked"
            if stable_speech_segments
            else "idle"
        )

        return {
            "fast": self._lane_payload(
                lane="fast",
                status=fast_status,
                latency_ms=self._lane_latency_ms(turn, "fast"),
                pending_count=len(fast_hypotheses),
                stable_count=0,
                round_id=round_id,
                cancellation_token=cancellation_token,
                cancellable=cancellable,
            ),
            "slow": self._lane_payload(
                lane="slow",
                status=slow_status,
                latency_ms=self._lane_latency_ms(turn, "slow"),
                pending_count=1 if final_text and not stable_decisions else 0,
                stable_count=len(stable_decisions),
                round_id=round_id,
                cancellation_token=cancellation_token,
                cancellable=cancellable,
            ),
            "arbiter": self._lane_payload(
                lane="arbiter",
                status=arbiter_status,
                latency_ms=self._lane_latency_ms(turn, "arbiter"),
                pending_count=1 if stable_decisions and not verdict_state else 0,
                stable_count=1 if verdict_state else 0,
                round_id=round_id,
                cancellation_token=cancellation_token,
                cancellable=cancellable,
            ),
            "speaking": self._lane_payload(
                lane="speaking",
                status=speaking_status,
                latency_ms=self._lane_latency_ms(turn, "speaking"),
                pending_count=len(stable_speech_segments) if stable_speech_segments and verdict_state != "approved" else 0,
                stable_count=len(stable_speech_segments),
                round_id=round_id,
                cancellation_token=cancellation_token,
                cancellable=cancellable,
            ),
        }

    def _lane_payload(
        self,
        *,
        lane: str,
        status: str,
        latency_ms: float,
        pending_count: int,
        stable_count: int,
        round_id: str,
        cancellation_token: str,
        cancellable: bool,
    ) -> dict[str, Any]:
        return {
            "lane": lane,
            "status": status,
            "latency_ms": round(max(0.0, float(latency_ms)), 3),
            "pending_count": max(0, int(pending_count)),
            "stable_count": max(0, int(stable_count)),
            "round_id": round_id,
            "cancellation_token": cancellation_token,
            "cancellable": cancellable,
        }

    def _auto_commit_decision_memory(
        self,
        turn: TurnBlackboard,
        *,
        committed: Mapping[str, Any],
        session_id: str | None,
        actor_id: str | None,
        task_context: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if getattr(self.memory_orchestrator, "memory_service", None) is None:
            return {}
        if turn.safety_state.get("memory_closed_loop_committed"):
            return {}

        pending_writeback = [
            item
            for item in turn.memory_candidates
            if isinstance(item, Mapping)
            and item.get("kind") == "writeback_proposal"
            and item.get("requires_commit") is not False
            and item.get("committed") is not True
        ]
        final_text = str(turn.asr_final or committed.get("final_text") or "").strip()
        if not pending_writeback and not _should_writeback_dialogue(final_text):
            return {}
        if not pending_writeback:
            reply = str(committed.get("speech_text") or "").strip() or _speech_segment_text(committed)
            summary = f"user:{final_text} | reply:{reply}".strip()
            self.memory_orchestrator.build_writeback_proposal(
                turn,
                query=final_text,
                channels=("voice",),
                priority="normal",
                reason="realtime_decision_dialogue_writeback",
                summary=summary,
                metadata={
                    "title": "Realtime dialogue turn",
                    "memory_type": "conversation",
                    "source": "eibrain.audio_dialogue",
                    "modality": "audio_text",
                    "organ": "ear",
                    "content": {
                        "event_type": "dialogue_turn",
                        "user_text": final_text,
                        "reply_text": reply,
                        "decision": committed.get("decision", ""),
                        "round_id": turn.round_id,
                    },
                    "meta": {
                        "source_system": "eibrain",
                        "round_id": turn.round_id,
                        "cancellation_token": turn.cancellation_token,
                        "trace_id": turn.round_id,
                        "source_event_id": f"{turn.round_id}:dialogue",
                        "dedupe_key": f"realtime-dialogue:{session_id or turn.round_id}:{turn.round_id}",
                        "memory_kind": "episodic",
                        "retention": "episode",
                        "promotion_status": "candidate" if _explicit_memory_request(final_text) else "not_promoted",
                        "identity_memory": False,
                        "persona_memory": False,
                        "privacy": {
                            "scope": "subject_conversation",
                            "sensitivity": "personal",
                            "allowed_use": "embodied_response",
                        },
                    },
                    "outcome": {
                        "success": bool(committed.get("speech_segments")),
                        "status": "planned",
                        "action_count": len(list(committed.get("action_plan") or [])),
                    },
                    "tags": _unique_values(
                        [
                            "dialogue",
                            "audio_text",
                            "ear",
                            str(committed.get("decision") or ""),
                            "explicit_memory_request" if _explicit_memory_request(final_text) else "",
                        ]
                    ),
                },
            )

        trace = self.memory_orchestrator.commit_candidates(
            turn,
            session_id=session_id,
            actor_id=actor_id,
            task_context={
                "phase": "decision_writeback",
                "modality": "audio_text",
                "organ": "ear",
                **dict(task_context or {}),
            },
            default_modality="audio_text",
            default_organ="ear",
        )
        turn.safety_state["memory_closed_loop_committed"] = True
        turn.safety_state["memory_closed_loop_trace"] = trace
        return trace

    def _record_reply_memory_usage(
        self,
        turn: TurnBlackboard,
        *,
        committed: Mapping[str, Any],
        session_id: str | None,
        actor_id: str | None,
    ) -> dict[str, Any]:
        if not any(isinstance(item, Mapping) and item.get("kind") == "recall" for item in turn.memory_candidates):
            return {}
        reply_text = str(committed.get("speech_text") or "").strip() or _speech_segment_text(committed)
        trace = self.memory_orchestrator.record_reply_memory_usage(
            turn,
            reply_text=reply_text,
            used_items=[dict(item) for item in list(committed.get("memory_refs") or []) if isinstance(item, Mapping)],
            session_id=session_id,
            actor_id=actor_id,
        )
        turn.safety_state["memory_closed_loop_trace"] = trace
        return trace

    def _apply_persona_memory_guardrails(self, turn: TurnBlackboard) -> dict[str, Any]:
        if not turn.memory_candidates:
            return {}
        persona_runtime = PersonaRuntime.from_persona_code(_persona_code_from_state(turn.persona_state))
        constraints = persona_runtime.stable_style_constraints()
        protected_keys = {str(item) for item in constraints.get("protected_keys", [])}
        filtered: list[dict[str, Any]] = []
        for candidate in turn.memory_candidates:
            if not isinstance(candidate, dict):
                continue
            key_path = _memory_candidate_key_path(candidate)
            guardrail = persona_runtime.apply_memory_guardrails(_memory_candidate_context(candidate))
            should_filter = bool(guardrail.get("persona_guardrail_applied")) or bool(key_path and key_path in protected_keys)
            if not should_filter:
                continue
            reason = "persona_guardrail_applied"
            candidate["reply_context_status"] = "filtered"
            candidate["reply_context_filter_reason"] = reason
            candidate["persona_guardrail_applied"] = True
            candidate["persona_guardrail_reason"] = reason
            if isinstance(candidate.get("policy_decision"), Mapping):
                candidate["original_policy_decision"] = dict(candidate["policy_decision"])
            candidate["policy_decision"] = {"decision": "filter", "reason": reason}
            filtered.append(
                _without_empty(
                    {
                        "id": candidate.get("id"),
                        "record_id": candidate.get("record_id"),
                        "key": key_path,
                        "reason": reason,
                        "guardrail": guardrail,
                    }
                )
            )
        if not filtered:
            return {}
        payload = {
            "status": "blocked",
            "summary": f"{len(filtered)} persona-drifting memory candidate(s) filtered",
            "filtered_count": len(filtered),
            "filtered": filtered,
            "constraints": constraints,
        }
        turn.safety_state["persona_memory_guardrail"] = payload
        return payload

    def _prefetch_memory(
        self,
        *,
        turn: TurnBlackboard,
        text: str,
        session_id: str | None = None,
        actor_id: str | None = None,
        task_context: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        stripped = text.strip()
        if not stripped:
            return []
        memory_service_configured = getattr(self.memory_orchestrator, "memory_service", None) is not None
        if memory_service_configured and stripped in {"记住", "记一下", "帮我记", "帮我记一下", "提醒我"}:
            return []
        if not any(marker in stripped for marker in ("记", "提醒", "上次", "以前", "喜欢", "妈妈", "爸爸")):
            return []
        if memory_service_configured:
            return self.memory_orchestrator.prefetch_recall(
                turn,
                query=stripped,
                channels=("voice",),
                priority="realtime",
                reason="prefetch_context_for_fast_lane",
                session_id=session_id,
                actor_id=actor_id,
                task_context={
                    "goal": "prefetch memory for realtime user turn",
                    "query_source": "asr_partial",
                    **dict(task_context or {}),
                },
                metadata={
                    "task_type": "brain.respond",
                    "goal": "prefetch memory for realtime user turn",
                    "phase": "fast_prefetch",
                    "modality": "audio_text",
                    "organ": "ear",
                    "recall_profile": "subject_dialogue",
                },
            )
        return [
            {
                "id": f"{turn.round_id}:prefetch:{len(turn.memory_candidates)}",
                "query": stripped,
                "text": stripped,
                "kind": "recall",
                "score": 0.5,
                "source": "scheduler_prefetch",
            }
        ]


def _first_text(*values: str | None) -> str:
    for value in values:
        if value is not None:
            return value
    return ""


def _explicit_memory_request(text: str) -> bool:
    return any(marker in text for marker in ("记住", "记一下", "记得", "帮我记", "以后", "下次", "偏好", "喜欢", "不喜欢"))


def _should_writeback_dialogue(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return _explicit_memory_request(stripped)


def _speech_segment_text(payload: Mapping[str, Any]) -> str:
    segments = payload.get("speech_segments")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        return ""
    for segment in segments:
        if isinstance(segment, Mapping) and str(segment.get("text") or "").strip():
            return str(segment["text"])
    return ""


def _unique_values(values: Sequence[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in unique:
            unique.append(text)
    return unique


def _to_dict(value: Any) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return dict(getattr(value, "__dict__", {}))


def _mapping_text(value: Any, key: str) -> str:
    if isinstance(value, Mapping):
        return str(value.get(key) or "")
    return ""


def _first_list(*values: Any) -> list[Any]:
    for value in values:
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
    return []


def _normalize_memory_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        item.setdefault("id", f"{source}:{index}")
        item.setdefault("source", source)
        normalized.append(item)
    return normalized


def _merge_memory_candidates(
    primary: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in list(primary or []) + list(secondary or []):
        item = dict(candidate)
        key = str(item.get("id") or item.get("query") or item.get("text") or len(merged))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _persona_code_from_state(persona_state: Mapping[str, Any] | None) -> str | None:
    if not isinstance(persona_state, Mapping):
        return None
    return str(
        persona_state.get("personaCode")
        or persona_state.get("persona_code")
        or persona_state.get("persona_id")
        or ""
    ) or None


def _memory_candidate_key_path(candidate: Mapping[str, Any]) -> str:
    for key in ("key", "preference_key", "memory_key"):
        value = candidate.get(key)
        if value not in (None, ""):
            return str(value)
    selected_record = candidate.get("selected_record")
    if isinstance(selected_record, Mapping):
        for key in ("key", "preference_key", "memory_key"):
            value = selected_record.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def _memory_candidate_context(candidate: Mapping[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key in ("memory_context", "context", "content"):
        value = candidate.get(key)
        if isinstance(value, Mapping):
            _deep_merge(context, value)
    selected_record = candidate.get("selected_record")
    if isinstance(selected_record, Mapping):
        for key in ("memory_context", "context", "content"):
            value = selected_record.get(key)
            if isinstance(value, Mapping):
                _deep_merge(context, value)
    key_path = _memory_candidate_key_path(candidate)
    if key_path:
        _nested_set(context, key_path, candidate.get("value") or candidate.get("text") or True)
    return context


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(str(key)), dict):
            _deep_merge(target[str(key)], value)
        elif isinstance(value, Mapping):
            target[str(key)] = dict(value)
        else:
            target[str(key)] = value


def _nested_set(payload: dict[str, Any], key_path: str, value: Any) -> None:
    current = payload
    parts = [part for part in key_path.split(".") if part]
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    if parts:
        current[parts[-1]] = value


def _expand_emotion_context(emotion_context: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(emotion_context)
    nested = payload.get("emotion_state")
    if isinstance(nested, Mapping):
        for key in ("mood", "state", "energy", "arousal", "valence", "environment", "confidence", "stability"):
            if key in nested:
                payload[key] = nested[key]
    return payload


def _context_summary(turn: TurnBlackboard) -> dict[str, Any]:
    return {
        "persona": _persona_summary(turn.persona_state),
        "emotion": _emotion_summary(turn.emotion_state),
    }


def _persona_summary(persona_state: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(persona_state, Mapping):
        return {}
    speaking_style = persona_state.get("speaking_style")
    speaking_style = speaking_style if isinstance(speaking_style, Mapping) else {}
    response_policy = persona_state.get("response_policy")
    response_policy = response_policy if isinstance(response_policy, Mapping) else {}
    return _without_empty(
        {
            "personaCode": persona_state.get("personaCode")
            or persona_state.get("persona_code")
            or persona_state.get("persona_id"),
            "voice_code": persona_state.get("voice_code") or persona_state.get("voiceCode"),
            "tone": speaking_style.get("tone") or persona_state.get("tone") or persona_state.get("style"),
            "max_chars": response_policy.get("max_chars"),
        }
    )


def _emotion_summary(emotion_state: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(emotion_state, Mapping):
        return {}
    nested = emotion_state.get("emotion_state")
    state = nested if isinstance(nested, Mapping) else emotion_state
    environment = state.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    strategy = emotion_state.get("response_strategy")
    strategy = strategy if isinstance(strategy, Mapping) else {}
    return _without_empty(
        {
            "mood": state.get("mood") or state.get("state") or _emotion_hint_label(emotion_state),
            "energy": state.get("energy"),
            "noise": environment.get("noise"),
            "time": environment.get("time"),
            "proximity": environment.get("proximity"),
            "tone": strategy.get("tone") or emotion_state.get("tone"),
        }
    )


def _emotion_hint_label(emotion_state: Mapping[str, Any]) -> str:
    hint = emotion_state.get("emotion_hint")
    if isinstance(hint, Mapping):
        return str(hint.get("label") or "")
    return ""


def _proactive_summary(activity: Any) -> dict[str, Any]:
    if not isinstance(activity, Mapping):
        return {}
    summary = activity.get("summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    return _without_empty(
        {
            "channel": activity.get("channel"),
            "reason": activity.get("reason"),
            "should_emit": activity.get("should_emit"),
            "disturbance": activity.get("disturbance"),
            "urgency": activity.get("urgency"),
        }
    )


def _without_empty(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in payload.items() if item not in (None, "", [], {})}
