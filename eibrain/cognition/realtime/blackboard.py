"""Round-scoped realtime context blackboard."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from .events import to_json_ready


def _payload(item: Mapping[str, Any]) -> dict[str, Any]:
    return to_json_ready(dict(item))


@dataclass
class TurnBlackboard:
    """Append-only per-round state shared by realtime cognition lanes."""

    round_id: str
    cancellation_token: str | None = None
    state: str = "active"
    observations: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    memory: list[dict[str, Any]] = field(default_factory=list)
    persona: list[dict[str, Any]] = field(default_factory=list)
    emotion: list[dict[str, Any]] = field(default_factory=list)
    speech: list[dict[str, Any]] = field(default_factory=list)
    action: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at_ts: float | None = None
    updated_at_ts: float | None = None

    def append_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return self._append("observations", observation)

    def append_hypothesis(self, hypothesis: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(hypothesis)
        payload.setdefault("stable", False)
        return self._append("hypotheses", payload)

    def append_memory(self, memory: Mapping[str, Any]) -> dict[str, Any]:
        return self._append("memory", memory)

    def append_persona(self, persona: Mapping[str, Any]) -> dict[str, Any]:
        return self._append("persona", persona)

    def append_emotion(self, emotion: Mapping[str, Any]) -> dict[str, Any]:
        return self._append("emotion", emotion)

    def append_speech(self, speech: Mapping[str, Any]) -> dict[str, Any]:
        return self._append("speech", speech)

    def append_action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        return self._append("action", action)

    def snapshot(self) -> dict[str, Any]:
        """Return a deep-copied JSON-ready snapshot for protocol/monitoring use."""

        return deepcopy(
            to_json_ready(
                {
                    "round_id": self.round_id,
                    "cancellation_token": self.cancellation_token,
                    "state": self.state,
                    "observations": self.observations,
                    "hypotheses": self.hypotheses,
                    "memory": self.memory,
                    "persona": self.persona,
                    "emotion": self.emotion,
                    "speech": self.speech,
                    "action": self.action,
                    "metadata": self.metadata,
                    "created_at_ts": self.created_at_ts,
                    "updated_at_ts": self.updated_at_ts,
                }
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return self.snapshot()

    def status_payload(self) -> dict[str, Any]:
        return self.snapshot()

    def _append(self, attr: str, item: Mapping[str, Any]) -> dict[str, Any]:
        payload = _payload(item)
        getattr(self, attr).append(payload)
        return payload


__all__ = ["TurnBlackboard"]
