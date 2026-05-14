"""Memory contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MemoryQuery:
    query: str
    session_id: str | None = None
    actor_id: str | None = None
    task_context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryResult:
    summary: str = ""
    relevant_memories: list[str] = field(default_factory=list)
    actor_profile: dict[str, str] = field(default_factory=dict)
    session_summary: str = ""
    recall_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryTraceSummary:
    trace_id: str = ""
    status: str = "idle"
    prefetch_requested: int = 0
    prefetch_result: int = 0
    write_proposed: int = 0
    write_committed: int = 0
    reply_used: int = 0
    queries: list[str] = field(default_factory=list)
    written_memory_types: list[str] = field(default_factory=list)
    used_memory_ids: list[str] = field(default_factory=list)
