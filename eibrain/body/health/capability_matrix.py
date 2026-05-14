"""Capability matrix derived from organ health.

This value-object captures cross-organ capabilities that higher layers consume
as a read-only capability contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict


@dataclass(slots=True)
class CapabilityMatrix:
    """Observed embodied capabilities for snapshot and routing decisions."""

    can_hear_voice: bool = False
    can_transcribe_speech: bool = False
    can_see_people: bool = False
    can_identify_person: bool = False
    can_speak: bool = False
    can_orient_head: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)
