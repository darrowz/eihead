"""Fallback policy derived from embodied capability status.

This module translates capability availability into a policy boundary:
which actions remain safe, which must be disabled, and whether automatic
execution requires operator confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .capability_matrix import CapabilityMatrix


_NORMAL_SAFE_ACTIONS = (
    "dialogue.listen",
    "dialogue.respond",
    "speech.play",
    "speech.stop",
    "head.move",
    "vision.track",
    "identity.recognize",
)
_SPEECH_INTERACTION_ACTIONS = ("dialogue.respond",)
_LOW_CONFIDENT_ACTION = "dialogue.finalize"
_CRITICAL_AUTORUN_ACTIONS = {"dialogue.listen", "dialogue.respond", "speech.play", "dialogue.finalize"}

_OPERATOR_MESSAGES = {
    "normal": "All core embodied capabilities are available.",
    "mute_companion": "Speech playback is unavailable; keep listening and avoid spoken replies.",
    "low_confidence_body": "Speech capture is present but ASR confidence is low; avoid final decisions without confirmation.",
    "fixed_gaze": "Head orientation is unavailable; keep voice interaction active without moving the neck.",
}


@dataclass(slots=True)
class FallbackPolicy:
    mode: str = "normal"
    reason: str = "all_core_capabilities_online"
    can_autorun: bool = True
    requires_confirmation: bool = False
    safe_actions: tuple[str, ...] = field(default_factory=lambda: _NORMAL_SAFE_ACTIONS)
    disabled_actions: tuple[str, ...] = field(default_factory=tuple)
    operator_message: str = _OPERATOR_MESSAGES["normal"]

    @classmethod
    def from_capabilities(
        cls,
        capabilities: CapabilityMatrix,
        *,
        degradation_mode: str | None = None,
    ) -> "FallbackPolicy":
        # Boundary: policy derivation keeps routing control separate from organ
        # runtime state collection.
        mode = degradation_mode or cls._infer_mode(capabilities)
        safe_actions: list[str] = []
        disabled_actions: list[str] = []
        cls._collect_voice_actions(capabilities, safe_actions=safe_actions, disabled_actions=disabled_actions)
        cls._collect_speech_actions(capabilities, safe_actions=safe_actions, disabled_actions=disabled_actions)
        cls._collect_head_actions(capabilities, safe_actions=safe_actions, disabled_actions=disabled_actions)
        cls._collect_visual_actions(capabilities, safe_actions=safe_actions, disabled_actions=disabled_actions)

        disabled = cls._dedupe(disabled_actions)
        safe = cls._dedupe(action for action in safe_actions if action not in disabled)
        requires_confirmation = mode != "normal" or _LOW_CONFIDENT_ACTION in disabled
        can_autorun = not requires_confirmation and _CRITICAL_AUTORUN_ACTIONS.isdisjoint(disabled)
        reason = cls._reason_for(mode, capabilities)

        return cls(
            mode=mode,
            reason=reason,
            can_autorun=can_autorun,
            requires_confirmation=requires_confirmation,
            safe_actions=safe,
            disabled_actions=disabled,
            operator_message=cls._operator_message_for(mode, reason),
        )

    @classmethod
    def _collect_voice_actions(
        cls,
        capabilities: CapabilityMatrix,
        *,
        safe_actions: list[str],
        disabled_actions: list[str],
    ) -> None:
        if capabilities.can_hear_voice:
            safe_actions.append("dialogue.listen")
        else:
            disabled_actions.extend(("dialogue.listen", _LOW_CONFIDENT_ACTION))

    @classmethod
    def _collect_speech_actions(
        cls,
        capabilities: CapabilityMatrix,
        *,
        safe_actions: list[str],
        disabled_actions: list[str],
    ) -> None:
        if (
            capabilities.can_hear_voice
            and capabilities.can_transcribe_speech
            and capabilities.can_speak
        ):
            safe_actions.extend(_SPEECH_INTERACTION_ACTIONS)
        else:
            disabled_actions.extend(_SPEECH_INTERACTION_ACTIONS)
        if capabilities.can_hear_voice and not capabilities.can_transcribe_speech:
            disabled_actions.append(_LOW_CONFIDENT_ACTION)
        if capabilities.can_speak:
            safe_actions.append("speech.play")
        else:
            disabled_actions.append("speech.play")
        safe_actions.append("speech.stop")

    @classmethod
    def _collect_head_actions(
        cls,
        capabilities: CapabilityMatrix,
        *,
        safe_actions: list[str],
        disabled_actions: list[str],
    ) -> None:
        if capabilities.can_orient_head:
            safe_actions.append("head.move")
        else:
            disabled_actions.append("head.move")

    @classmethod
    def _collect_visual_actions(
        cls,
        capabilities: CapabilityMatrix,
        *,
        safe_actions: list[str],
        disabled_actions: list[str],
    ) -> None:
        if capabilities.can_see_people:
            safe_actions.append("vision.track")
        else:
            disabled_actions.append("vision.track")
        if capabilities.can_identify_person:
            safe_actions.append("identity.recognize")
        else:
            disabled_actions.append("identity.recognize")

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "can_autorun": self.can_autorun,
            "requires_confirmation": self.requires_confirmation,
            "safe_actions": list(self.safe_actions),
            "disabled_actions": list(self.disabled_actions),
            "operator_message": self.operator_message,
        }

    @staticmethod
    def _infer_mode(capabilities: CapabilityMatrix) -> str:
        if capabilities.can_hear_voice and not capabilities.can_transcribe_speech:
            return "low_confidence_body"
        if not capabilities.can_speak:
            return "mute_companion"
        if not capabilities.can_orient_head:
            return "fixed_gaze"
        return "normal"

    @staticmethod
    def _reason_for(mode: str, capabilities: CapabilityMatrix) -> str:
        if mode == "mute_companion":
            return "speech_playback_unavailable"
        if mode == "low_confidence_body":
            return "asr_unavailable_or_low_confidence"
        if mode == "fixed_gaze":
            return "head_orientation_unavailable"
        missing_visual = not capabilities.can_see_people or not capabilities.can_identify_person
        if mode == "normal" and missing_visual:
            return "visual_capabilities_partial"
        return "all_core_capabilities_online"

    @staticmethod
    def _operator_message_for(mode: str, reason: str) -> str:
        if reason == "visual_capabilities_partial":
            return "Voice interaction can continue; visual tracking and identity actions are disabled."
        return _OPERATOR_MESSAGES.get(mode, "Body runtime is degraded; restrict automatic actions.")

    @staticmethod
    def _dedupe(actions) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        for action in actions:
            if action in seen:
                continue
            seen.add(action)
            result.append(action)
        return tuple(result)
