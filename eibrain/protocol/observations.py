"""Observation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ProtocolMessage


@dataclass(slots=True)
class AudioTranscriptFinal(ProtocolMessage):
    text: str = ""
    language: str = "und"
    kind: str = field(init=False, default="audio_transcript_final")
