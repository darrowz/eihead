"""Simple runtime tracing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TraceRecorder:
    entries: list[dict[str, object]] = field(default_factory=list)

    def record(self, *, trace_id: str, kind: str, payload: dict[str, object]) -> None:
        self.entries.append({"trace_id": trace_id, "kind": kind, "payload": payload})

    def snapshot(self) -> list[dict[str, object]]:
        return list(self.entries)
