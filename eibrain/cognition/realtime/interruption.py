"""Interruption and cancellation summaries for realtime cognition rounds."""

from __future__ import annotations

from typing import Any

from .turn import RealtimeTurnManager, TurnBlackboard


class InterruptionController:
    """Cancel old work lanes and start a fresh realtime round."""

    _CANCELLATION_CHAIN = [
        "stop_tts",
        "cancel_generation",
        "cancel_memory_prefetch",
        "cancel_action_plan",
        "mark_interrupted",
        "start_new_round",
    ]

    def summarize(
        self,
        *,
        old_turn: TurnBlackboard,
        new_turn: TurnBlackboard,
        reason: str = "user_interrupt",
    ) -> dict[str, Any]:
        self._mark_action_plan_cancelled(old_turn)
        return {
            "reason": reason,
            "cancellation_chain": list(self._CANCELLATION_CHAIN),
            "stop_tts": True,
            "cancel_generation": True,
            "cancel_memory_prefetch": True,
            "cancel_action_plan": True,
            "cancel_actions": True,
            "mark_interrupted": {
                "round_id": old_turn.round_id,
                "cancellation_token": old_turn.cancellation_token,
                "at_ts": old_turn.interrupted_at_ts,
                "state": old_turn.state,
                "cancelled": bool(old_turn.cancellation and old_turn.cancellation.cancelled),
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
            old_turn = TurnBlackboard(
                round_id="none",
                cancellation_token="none",
                state="unknown",
                created_at_ts=0.0,
                updated_at_ts=0.0,
            )
            return self.summarize(old_turn=old_turn, new_turn=new_turn, reason=reason)

        new_turn = manager.interrupt(reason=reason)
        return self.summarize(old_turn=old_turn, new_turn=new_turn, reason=reason)

    def _mark_action_plan_cancelled(self, turn: TurnBlackboard) -> None:
        for action in turn.action_plan:
            if action.get("status") not in {"completed", "failed", "unavailable"}:
                action["status"] = "cancelled"
