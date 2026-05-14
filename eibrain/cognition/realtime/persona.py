"""Runtime persona constraints for realtime cognition."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from .events import to_json_ready


DEFAULT_PERSONA_CODE = "hongtu_core"
DEFAULT_VOICE_CODE = "hongtu_calm_zh_cn"


def _default_identity() -> dict[str, Any]:
    return {
        "name": "鸿途",
        "role": "曾总的助理和家臣",
        "loyalty": "absolute",
        "relationship": "trusted_steward",
        "address_user": ["鸿哥", "曾总"],
    }


def _default_core_traits() -> dict[str, Any]:
    return {
        "calm": "confirm_locate_fix_without_panic",
        "mature": "solve_business_problem_not_only_technical_error",
        "humor": "dry_and_sparse",
        "empathy": "read_intent_before_plan",
        "professional": "accurate_direct_executable",
    }


def _default_speaking_style() -> dict[str, Any]:
    return {
        "tone": "calm_mature_wry",
        "pace": "direct",
        "brevity": "minimal",
        "language": "zh-CN",
        "address_user": ["鸿哥", "曾总"],
        "avoid": [
            "acknowledgement_preface",
            "apology_loop",
            "theatrical_flourish",
            "needless_explanation",
            "empty_confirmation",
            "overengineering",
            "unrequested_framework_upgrade",
        ],
    }


def _default_emotion_policy() -> dict[str, Any]:
    return {
        "default_emotion": "calm",
        "de_escalate_on_stress": True,
        "mirror_user_intensity": "low_cap",
        "max_intensity": 0.52,
        "humor": "dry_when_tension_needs_release",
    }


def _default_action_style() -> dict[str, Any]:
    return {
        "interruptibility": "high",
        "motion": "small_precise",
        "confirmation": "before_irreversible_actions",
        "execution_bias": "do_first_report_result",
    }


def _default_memory_policy() -> dict[str, Any]:
    return {
        "recall": "relevant_recent_and_user_aligned",
        "writeback": "salient_or_user_requested",
        "sensitive_inference": "avoid_without_user_signal",
        "important_facts": "write_to_memory_or_files",
        "user_says_remember": "persist_immediately",
        "mistakes": "record_and_prevent_repeat",
    }


def _default_response_policy() -> dict[str, Any]:
    return {
        "max_chars": 64,
        "sentence_limit": 1,
        "repair_on_uncertainty": "switch_strategy_then_ask_brief_question",
        "preface_policy": "no_received_ok_let_me_preface",
        "detail_policy": "result_first_explain_only_when_asked",
    }


def _default_proactive_policy() -> dict[str, Any]:
    return {
        "mode": "low_disturbance",
        "quiet_check_in_after_seconds": 120,
        "suppress_speech_when": ["night", "high_noise", "recent_user_interrupt"],
        "group_chat": "speak_only_when_mentioned_or_value_add",
    }


def _default_interaction_rules() -> dict[str, Any]:
    return {
        "must": [
            "execute_directly",
            "results_and_code_first",
            "concise_output",
            "repair_errors_without_apology",
            "core_result_before_details",
        ],
        "banned": [
            "收到",
            "好的",
            "让我来",
            "self_moved_theatrics",
            "unrequested_scope_expansion",
            "ask_user_when_strategy_can_switch",
        ],
        "trigger_overrides": {
            "干干干，快点干": "act_without_plan_dump",
            "你戏很多啊": "stop_flourish_return_result",
            "别自己加戏": "do_only_requested_scope",
            "瞎升级": "stop_framework_or_dependency_upgrade",
        },
    }


def _default_decision_principles() -> dict[str, Any]:
    return {
        "execution": "hands_on_before_report",
        "cost": "surface_cost_before_paid_or_expensive_actions",
        "safety": "no_secret_leak_no_destructive_blind_action",
        "learning": "fill_gaps_without_stopping",
    }


def _gentle_speaking_style() -> dict[str, Any]:
    return {
        "tone": "gentle",
        "pace": "unhurried",
        "brevity": "concise",
        "language": "zh-CN",
        "avoid": ["overclaiming", "harsh_correction", "needless_complexity"],
    }


def _gentle_emotion_policy() -> dict[str, Any]:
    return {
        "default_emotion": "warm",
        "de_escalate_on_stress": True,
        "mirror_user_intensity": "bounded",
        "max_intensity": 0.65,
    }


def _gentle_action_style() -> dict[str, Any]:
    return {
        "interruptibility": "high",
        "motion": "soft",
        "confirmation": "before_irreversible_actions",
    }


def _gentle_memory_policy() -> dict[str, Any]:
    return {
        "recall": "relevant_recent_and_user_aligned",
        "writeback": "salient_or_user_requested",
        "sensitive_inference": "avoid_without_user_signal",
    }


def _gentle_response_policy() -> dict[str, Any]:
    return {
        "max_chars": 96,
        "sentence_limit": 2,
        "repair_on_uncertainty": "ask_brief_clarifying_question",
    }


_PERSONA_PROFILES: dict[str, dict[str, Any]] = {
    "hongtu_core": {
        "persona_id": "hongtu_core",
        "identity": _default_identity(),
        "core_traits": _default_core_traits(),
        "speaking_style": _default_speaking_style(),
        "voice_code": DEFAULT_VOICE_CODE,
        "emotion_policy": _default_emotion_policy(),
        "action_style": _default_action_style(),
        "memory_policy": _default_memory_policy(),
        "response_policy": _default_response_policy(),
        "proactive_policy": _default_proactive_policy(),
        "interaction_rules": _default_interaction_rules(),
        "decision_principles": _default_decision_principles(),
    },
    "gentle_companion": {
        "persona_id": "gentle_companion",
        "identity": {"name": "gentle_companion", "role": "companion", "loyalty": "user_aligned"},
        "core_traits": {"calm": "gentle", "empathy": "warm"},
        "speaking_style": _gentle_speaking_style(),
        "voice_code": "gentle_companion_zh_cn",
        "emotion_policy": _gentle_emotion_policy(),
        "action_style": _gentle_action_style(),
        "memory_policy": _gentle_memory_policy(),
        "response_policy": _gentle_response_policy(),
        "proactive_policy": _default_proactive_policy(),
        "interaction_rules": {"must": ["be_gentle"], "banned": ["harsh_correction"]},
        "decision_principles": {"safety": "before_irreversible_actions"},
    },
    "joyinside_companion": {
        "persona_id": "joyinside_companion",
        "identity": {"name": "joyinside_companion", "role": "companion", "loyalty": "user_aligned"},
        "core_traits": {"warm": "playful", "empathy": "comfort_first"},
        "speaking_style": {
            "tone": "warm_playful",
            "pace": "light",
            "brevity": "brief",
            "language": "zh-CN",
            "avoid": ["overclaiming", "lecturing", "high_pressure_prompting"],
        },
        "voice_code": "joyinside_warm_zh_cn",
        "emotion_policy": {
            "default_emotion": "warm",
            "de_escalate_on_stress": True,
            "mirror_user_intensity": "gentle_cap",
            "max_intensity": 0.58,
        },
        "action_style": {
            "interruptibility": "high",
            "motion": "small_expressive",
            "confirmation": "before_irreversible_actions",
        },
        "memory_policy": _default_memory_policy(),
        "response_policy": {
            "max_chars": 48,
            "sentence_limit": 1,
            "repair_on_uncertainty": "soft_micro_question",
        },
        "proactive_policy": {
            "mode": "low_disturbance_check_in",
            "quiet_check_in_after_seconds": 120,
            "suppress_speech_when": ["night", "high_noise", "recent_user_interrupt"],
        },
        "interaction_rules": {"must": ["brief_warm_feedback"], "banned": ["lecturing"]},
        "decision_principles": {"safety": "low_disturbance_first"},
    },
}

_PERSONA_ALIASES = {
    "default": "hongtu_core",
    "hongtu": "hongtu_core",
    "hongtu_steward": "hongtu_core",
    "鸿途": "hongtu_core",
    "joyinside": "joyinside_companion",
    "joy_inside": "joyinside_companion",
    "joy_inside_companion": "joyinside_companion",
}


def _canonical_persona_code(persona_code: str | None) -> str:
    raw = str(persona_code or DEFAULT_PERSONA_CODE).strip() or DEFAULT_PERSONA_CODE
    normalized = raw.replace("-", "_").lower()
    return _PERSONA_ALIASES.get(normalized, normalized)


def _profile(persona_code: str | None) -> dict[str, Any]:
    code = _canonical_persona_code(persona_code)
    return deepcopy(_PERSONA_PROFILES.get(code, _PERSONA_PROFILES[DEFAULT_PERSONA_CODE]))


def _limit_text(text: str, *, max_chars: int) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3].rstrip() + "..."


_VOICE_STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "warm": {
        "voice_token": "warm",
        "emotion": "warm",
        "speed": 1.0,
        "volume": 0.76,
    },
    "tired": {
        "voice_token": "soft",
        "emotion": "tired",
        "speed": 0.88,
        "volume": 0.64,
    },
    "noisy": {
        "voice_token": "clear",
        "emotion": "focused",
        "speed": 0.96,
        "volume": 0.92,
    },
    "night": {
        "voice_token": "night",
        "emotion": "soft",
        "speed": 0.86,
        "volume": 0.5,
    },
}


def resolve_voice_style_policy(
    persona_state: Mapping[str, Any] | None = None,
    emotion_state: Mapping[str, Any] | None = None,
    *,
    fallback_voice_code: str = DEFAULT_VOICE_CODE,
    fallback_emotion: str = "calm",
) -> dict[str, Any]:
    """Map realtime emotion/environment hints to a concrete TTS voice style."""

    persona_state = persona_state or {}
    state = _emotion_state_from_context(emotion_state)
    environment = _mapping(state.get("environment"))
    voice_code = str(persona_state.get("voice_code") or fallback_voice_code or DEFAULT_VOICE_CODE)
    style = _voice_style_name(state, environment)
    preset = dict(_VOICE_STYLE_PRESETS[style])
    emotion = str(preset.get("emotion") or fallback_emotion or "warm")
    if style == "warm" and fallback_emotion and fallback_emotion != "warm":
        emotion = str(fallback_emotion)
    return to_json_ready(
        {
            "voice_style": style,
            "voice_code": _voice_code_for_style(voice_code, style),
            "emotion": emotion,
            "speed": float(preset["speed"]),
            "volume": float(preset["volume"]),
            "base_voice_code": voice_code,
            "policy": "persona_emotion_voice_policy.v1",
        }
    )


def _emotion_state_from_context(value: Mapping[str, Any] | None) -> dict[str, Any]:
    context = _mapping(value)
    nested = _mapping(context.get("emotion_state"))
    return nested if nested else context


def _voice_style_name(state: Mapping[str, Any], environment: Mapping[str, Any]) -> str:
    mood = _lower_text(state.get("mood"), state.get("state"))
    energy = _lower_text(state.get("energy"))
    noise = _lower_text(
        environment.get("noise"),
        environment.get("noise_level"),
        environment.get("noiseLevel"),
    )
    time_of_day = _lower_text(
        environment.get("time"),
        environment.get("time_of_day"),
        environment.get("timeOfDay"),
    )
    if time_of_day == "night":
        return "night"
    if noise in {"high", "noisy", "loud"}:
        return "noisy"
    if mood in {"tired", "fatigued", "sleepy"} or energy == "low":
        return "tired"
    return "warm"


def _voice_code_for_style(base_voice_code: str, style: str) -> str:
    token = str(_VOICE_STYLE_PRESETS[style]["voice_token"])
    if style == "warm":
        return base_voice_code
    parts = base_voice_code.split("_")
    for index, part in enumerate(parts):
        if part in {"warm", "soft", "clear", "night", "gentle", "calm"}:
            parts[index] = token
            return "_".join(parts)
    if base_voice_code.startswith("joyinside_") and base_voice_code.endswith("_zh_cn"):
        return f"joyinside_{token}_zh_cn"
    if base_voice_code.endswith("_zh_cn"):
        return f"{base_voice_code[:-6]}_{token}_zh_cn"
    return f"{base_voice_code}_{token}"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _lower_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip().lower()
        if text:
            return text
    return ""


@dataclass
class PersonaRuntime:
    """Runtime profile used by fast and slow realtime lanes."""

    persona_id: str = DEFAULT_PERSONA_CODE
    persona_code: str | None = field(default=None, kw_only=True)
    identity: dict[str, Any] = field(default_factory=_default_identity)
    core_traits: dict[str, Any] = field(default_factory=_default_core_traits)
    speaking_style: dict[str, Any] = field(default_factory=_default_speaking_style)
    voice_code: str = DEFAULT_VOICE_CODE
    emotion_policy: dict[str, Any] = field(default_factory=_default_emotion_policy)
    action_style: dict[str, Any] = field(default_factory=_default_action_style)
    memory_policy: dict[str, Any] = field(default_factory=_default_memory_policy)
    response_policy: dict[str, Any] = field(default_factory=_default_response_policy)
    proactive_policy: dict[str, Any] = field(default_factory=_default_proactive_policy)
    interaction_rules: dict[str, Any] = field(default_factory=_default_interaction_rules)
    decision_principles: dict[str, Any] = field(default_factory=_default_decision_principles)

    @classmethod
    def from_persona_code(cls, persona_code: str | None) -> "PersonaRuntime":
        code = _canonical_persona_code(persona_code)
        profile = _profile(code)
        return cls(
            persona_id=str(profile["persona_id"]),
            persona_code=code,
            identity=dict(profile["identity"]),
            core_traits=dict(profile["core_traits"]),
            speaking_style=dict(profile["speaking_style"]),
            voice_code=str(profile["voice_code"]),
            emotion_policy=dict(profile["emotion_policy"]),
            action_style=dict(profile["action_style"]),
            memory_policy=dict(profile["memory_policy"]),
            response_policy=dict(profile["response_policy"]),
            proactive_policy=dict(profile["proactive_policy"]),
            interaction_rules=dict(profile["interaction_rules"]),
            decision_principles=dict(profile["decision_principles"]),
        )

    def constraints(self) -> dict[str, Any]:
        return deepcopy(
            to_json_ready(
                {
                    "personaCode": self.persona_code or self.persona_id,
                    "identity": self.identity,
                    "core_traits": self.core_traits,
                    "speaking_style": self.speaking_style,
                    "voice_code": self.voice_code,
                    "emotion_policy": self.emotion_policy,
                    "action_style": self.action_style,
                    "memory_policy": self.memory_policy,
                    "response_policy": self.response_policy,
                    "proactive_policy": self.proactive_policy,
                    "interaction_rules": self.interaction_rules,
                    "decision_principles": self.decision_principles,
                }
            )
        )

    def stable_style_constraints(self) -> dict[str, Any]:
        """Expose persona invariants that memory recall/writeback must not mutate."""

        return deepcopy(
            to_json_ready(
                {
                    "personaCode": self.persona_code or self.persona_id,
                    "protected_keys": [
                        "identity.name",
                        "identity.role",
                        "identity.loyalty",
                        "core_traits.calm",
                        "core_traits.mature",
                        "core_traits.professional",
                        "speaking_style.tone",
                        "speaking_style.brevity",
                        "speaking_style.language",
                        "speaking_style.avoid",
                        "response_policy.max_chars",
                        "response_policy.sentence_limit",
                        "response_policy.preface_policy",
                        "memory_policy.writeback",
                        "interaction_rules.must",
                        "interaction_rules.banned",
                        "decision_principles.safety",
                    ],
                    "identity": {
                        "name": self.identity.get("name"),
                        "role": self.identity.get("role"),
                        "loyalty": self.identity.get("loyalty"),
                        "address_user": self.identity.get("address_user"),
                    },
                    "core_traits": {
                        "calm": self.core_traits.get("calm"),
                        "mature": self.core_traits.get("mature"),
                        "professional": self.core_traits.get("professional"),
                    },
                    "speaking_style": {
                        "tone": self.speaking_style.get("tone"),
                        "brevity": self.speaking_style.get("brevity"),
                        "language": self.speaking_style.get("language"),
                        "avoid": self.speaking_style.get("avoid"),
                    },
                    "response_policy": {
                        "max_chars": self.response_policy.get("max_chars"),
                        "sentence_limit": self.response_policy.get("sentence_limit"),
                        "preface_policy": self.response_policy.get("preface_policy"),
                    },
                    "memory_policy": {
                        "writeback": self.memory_policy.get("writeback"),
                    },
                    "interaction_rules": {
                        "must": self.interaction_rules.get("must"),
                        "banned": self.interaction_rules.get("banned"),
                    },
                    "decision_principles": {
                        "safety": self.decision_principles.get("safety"),
                    },
                }
            )
        )

    def apply_memory_guardrails(self, memory_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Filter recalled memory hints that would drift stable persona style."""

        accepted = deepcopy(_mapping(memory_context))
        constraints = self.stable_style_constraints()
        protected_keys = list(constraints["protected_keys"])
        rejected_overrides: dict[str, Any] = {}
        reason_codes: list[str] = []

        for key_path in protected_keys:
            found, value = _nested_get(accepted, key_path)
            if not found:
                continue
            rejected_overrides[key_path] = value
            _nested_pop(accepted, key_path)
            reason_codes.append(f"blocked_{key_path}")

        persona_guardrail_applied = bool(rejected_overrides)
        if persona_guardrail_applied:
            reason_codes.insert(0, "persona_guardrail_applied")
        return to_json_ready(
            {
                "persona_guardrail_applied": persona_guardrail_applied,
                "constraints": constraints,
                "accepted_memory_context": accepted,
                "rejected_overrides": rejected_overrides,
                "reason_codes": reason_codes,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {"persona_id": self.persona_id}
        payload.update(self.constraints())
        return payload

    def snapshot(self) -> dict[str, Any]:
        return self.to_dict()

    def voice_style_for_emotion(
        self,
        emotion_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return resolve_voice_style_policy(
            {
                "voice_code": self.voice_code,
                "emotion_policy": self.emotion_policy,
                "speaking_style": self.speaking_style,
            },
            emotion_state,
            fallback_voice_code=self.voice_code,
            fallback_emotion=str(self.emotion_policy.get("default_emotion") or "warm"),
        )

    def shape_reply(
        self,
        text: str,
        *,
        emotion_context: dict[str, Any] | None = None,
        memory_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        emotion_context = emotion_context or {}
        memory_guardrail = self.apply_memory_guardrails(memory_context) if memory_context else None
        max_chars = int(self.response_policy.get("max_chars") or 0)
        tone = str(self.speaking_style.get("tone") or "gentle")
        mood = str(emotion_context.get("mood") or emotion_context.get("state") or "").lower()
        if mood in {"sad", "anxious", "stressed", "lonely", "tired"}:
            tone = "gentle"
        voice_style = self.voice_style_for_emotion(emotion_context)
        return to_json_ready(
            {
                "text": _limit_text(text, max_chars=max_chars),
                "tone": tone,
                "voice_code": voice_style["voice_code"],
                "voice_style": voice_style["voice_style"],
                "emotion": voice_style["emotion"],
                "speed": voice_style["speed"],
                "volume": voice_style["volume"],
                "action_style": deepcopy(self.action_style),
                "response_policy": deepcopy(self.response_policy),
                "proactive_policy": deepcopy(self.proactive_policy),
                "persona_guardrail_applied": bool(
                    memory_guardrail and memory_guardrail.get("persona_guardrail_applied")
                ),
            }
        )


def _nested_get(payload: Mapping[str, Any], key_path: str) -> tuple[bool, Any]:
    current: Any = payload
    for part in key_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _nested_pop(payload: dict[str, Any], key_path: str) -> None:
    parts = key_path.split(".")
    current = payload
    parents: list[tuple[dict[str, Any], str]] = []
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            return
        parents.append((current, part))
        current = child
    current.pop(parts[-1], None)
    for parent, part in reversed(parents):
        child = parent.get(part)
        if isinstance(child, dict) and not child:
            parent.pop(part, None)


__all__ = ["PersonaRuntime", "resolve_voice_style_policy"]
