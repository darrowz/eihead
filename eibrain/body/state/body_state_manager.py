"""Aggregate embodied runtime state for brain and monitoring consumers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import time

from eibrain.body.health import DegradationManager, FallbackPolicy, OrganHealth


class BodyStateManager:
    """Build a JSON-safe snapshot from organ health and runtime sections."""

    def __init__(
        self,
        *,
        node_id: str,
        degradation_manager: DegradationManager | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.node_id = node_id
        self.degradation_manager = degradation_manager or DegradationManager()
        self._clock = clock or time.time

    def snapshot(
        self,
        organ_states: Sequence[OrganHealth],
        *,
        recent_events: Sequence[Mapping[str, object]] | None = None,
        runtime: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        states = list(organ_states)
        degradation = self.degradation_manager.evaluate(states)
        fallback = FallbackPolicy.from_capabilities(
            degradation.capabilities,
            degradation_mode=degradation.degradation_mode,
        )
        event_list = [dict(event) for event in (recent_events or ())]

        return {
            "schema": "eibrain.body_state.v1",
            "node_id": self.node_id,
            "updated_at_ts": self._clock(),
            "organ_count": len(states),
            "degradation_mode": degradation.degradation_mode,
            "capabilities": degradation.capabilities.to_dict(),
            "fallback_policy": fallback.to_dict(),
            "organs": {state.organ: state.to_dict() for state in states},
            "recent_event_count": len(event_list),
            "recent_events": event_list,
            "runtime": dict(runtime or {}),
        }
