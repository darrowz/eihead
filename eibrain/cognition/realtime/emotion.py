"""Emotion and environment context normalization for realtime lanes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .events import to_json_ready


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _merge(target: dict[str, Any], source: Mapping[str, Any] | None) -> None:
    if source:
        target.update(dict(source))


@dataclass
class EmotionContextBuilder:
    """Merge prosody, environment, and vision hints into response guidance."""

    high_noise_db: float = 70.0
    medium_noise_db: float = 60.0

    def build(
        self,
        *,
        observations: Iterable[Mapping[str, Any]] | None = None,
        prosody: Mapping[str, Any] | None = None,
        environment: Mapping[str, Any] | None = None,
        vision: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        prosody_hints: dict[str, Any] = {}
        environment_hints: dict[str, Any] = {}
        vision_hints: dict[str, Any] = {}

        for item in observations or ():
            kind = item.get("kind")
            payload = item.get("payload", {})
            if kind == "prosody" and isinstance(payload, Mapping):
                _merge(prosody_hints, payload)
            elif kind == "environment" and isinstance(payload, Mapping):
                _merge(environment_hints, payload)
            elif kind == "vision" and isinstance(payload, Mapping):
                _merge(vision_hints, payload)

        _merge(prosody_hints, prosody)
        _merge(environment_hints, environment)
        _merge(vision_hints, vision)

        emotion_hint = self._emotion_hint(prosody_hints, vision_hints)
        noise_policy = self._noise_policy(environment_hints)
        emotion_state = self._emotion_state(
            prosody_hints=prosody_hints,
            environment_hints=environment_hints,
            vision_hints=vision_hints,
            emotion_hint=emotion_hint,
            noise_policy=noise_policy,
        )
        response_strategy = self._response_strategy(
            emotion_state=emotion_state,
            noise_policy=noise_policy,
        )
        response_style = self._response_style(emotion_hint=emotion_hint, noise_policy=noise_policy)

        return to_json_ready(
            {
                "emotion_hint": emotion_hint,
                "emotion_state": emotion_state,
                "noise_policy": noise_policy,
                "response_strategy": response_strategy,
                "response_style": response_style,
                "inputs": {
                    "prosody": prosody_hints,
                    "environment": environment_hints,
                    "vision": vision_hints,
                },
            }
        )

    def _emotion_hint(
        self,
        prosody_hints: Mapping[str, Any],
        vision_hints: Mapping[str, Any],
    ) -> dict[str, Any]:
        arousal = _as_float(prosody_hints.get("arousal"))
        valence = _as_float(prosody_hints.get("valence"))
        stress = _as_float(prosody_hints.get("stress"))
        expression = str(vision_hints.get("face_expression", "")).lower()
        attention = str(vision_hints.get("attention", "")).lower()
        sources: list[str] = []

        if stress >= 0.7 or (arousal >= 0.75 and valence < 0.0):
            label = "stressed"
            confidence = max(stress, arousal)
            sources.append("prosody")
        elif valence >= 0.25 or attention == "present":
            label = "engaged"
            confidence = max(0.55, valence)
            if prosody_hints:
                sources.append("prosody")
        else:
            label = "calm"
            confidence = 0.5
            if prosody_hints:
                sources.append("prosody")

        if expression in {"tired", "sad", "concerned", "angry", "stressed"}:
            if label == "calm":
                label = "concerned"
                confidence = max(confidence, 0.6)
            if "vision" not in sources:
                sources.append("vision")
        elif attention and "vision" not in sources and label != "stressed":
            sources.append("vision")

        return {
            "label": label,
            "confidence": round(min(max(confidence, 0.0), 1.0), 2),
            "sources": sources,
        }

    def _noise_policy(self, environment_hints: Mapping[str, Any]) -> dict[str, Any]:
        noise_db = _as_float(environment_hints.get("noise_db"), default=-1.0)
        noise_level = str(environment_hints.get("noise_level", "")).lower()

        if noise_db >= self.high_noise_db or noise_level in {"high", "noisy", "loud"}:
            return {
                "mode": "reduce_verbal_density",
                "reason": "high_environment_noise",
                "noise_db": noise_db if noise_db >= 0 else None,
            }
        if noise_db >= self.medium_noise_db or noise_level == "medium":
            return {
                "mode": "confirm_hearing",
                "reason": "moderate_environment_noise",
                "noise_db": noise_db if noise_db >= 0 else None,
            }
        return {
            "mode": "normal",
            "reason": "clear_environment",
            "noise_db": noise_db if noise_db >= 0 else None,
        }

    def _emotion_state(
        self,
        *,
        prosody_hints: Mapping[str, Any],
        environment_hints: Mapping[str, Any],
        vision_hints: Mapping[str, Any],
        emotion_hint: Mapping[str, Any],
        noise_policy: Mapping[str, Any],
    ) -> dict[str, Any]:
        arousal_value = _as_float(prosody_hints.get("arousal"), default=0.5)
        valence_value = _as_float(prosody_hints.get("valence"), default=0.0)
        stress = _as_float(prosody_hints.get("stress"))
        fatigue = _as_float(prosody_hints.get("fatigue"))
        energy_hint = str(prosody_hints.get("energy") or "").lower()
        expression = str(vision_hints.get("face_expression", "")).lower()
        attention = str(vision_hints.get("attention", "")).lower()

        if stress >= 0.7 or emotion_hint.get("label") == "stressed":
            mood = "stressed"
        elif fatigue >= 0.65 or energy_hint == "low" or expression == "tired":
            mood = "tired"
        elif valence_value <= -0.25 or expression in {"sad", "concerned"}:
            mood = "sad"
        elif valence_value >= 0.25 or attention == "present":
            mood = "engaged"
        else:
            mood = "calm"

        if energy_hint in {"low", "medium", "high"}:
            energy = energy_hint
        elif fatigue >= 0.65 or arousal_value <= 0.35:
            energy = "low"
        elif arousal_value >= 0.7:
            energy = "high"
        else:
            energy = "medium"

        environment = {
            "noise": _noise_label(noise_policy),
            "time": _time_label(environment_hints),
            "proximity": _proximity_label(vision_hints),
        }
        proactive = _proactive_policy(mood=mood, environment=environment)

        return {
            "mood": mood,
            "energy": energy,
            "arousal": _band(arousal_value, low=0.35, high=0.7),
            "valence": _valence_label(valence_value),
            "environment": environment,
            "proactive": proactive,
            "confidence": emotion_hint.get("confidence", 0.5),
            "sources": list(emotion_hint.get("sources") or []),
            "stability": "stable_hint",
        }

    def _response_strategy(
        self,
        *,
        emotion_state: Mapping[str, Any],
        noise_policy: Mapping[str, Any],
    ) -> dict[str, Any]:
        environment = emotion_state.get("environment")
        environment = environment if isinstance(environment, Mapping) else {}
        mood = str(emotion_state.get("mood") or "calm")
        noise = str(environment.get("noise") or _noise_label(noise_policy))
        time_of_day = str(environment.get("time") or "day")
        proximity = str(environment.get("proximity") or "near")
        vulnerable = mood in {"sad", "anxious", "stressed", "lonely", "tired"}
        constrained_channel = noise == "high" or time_of_day == "night" or proximity == "far"
        proactive = emotion_state.get("proactive")
        proactive = proactive if isinstance(proactive, Mapping) else _proactive_policy(
            mood=mood,
            environment={"noise": noise, "time": time_of_day, "proximity": proximity},
        )

        if noise == "high" and (time_of_day == "night" or proximity == "far"):
            brevity = "very_concise"
        elif noise in {"high", "medium"} or constrained_channel:
            brevity = "concise"
        else:
            brevity = "normal"

        return {
            "tone": "gentle" if vulnerable else "warm",
            "pace": "slow" if vulnerable else "normal",
            "brevity": brevity,
            "micro_ack": vulnerable or constrained_channel,
            "nonverbal_preferred": constrained_channel,
            "speech_risk": "cautious" if vulnerable or constrained_channel else "normal",
            "proactive_priority": proactive["priority"],
            "proactive_disturbance": proactive["disturbance"],
            "proactive_suppress_speech": proactive["suppress_speech"],
            "proactive_suppression_reasons": list(proactive["suppression_reasons"]),
        }

    def _response_style(
        self,
        *,
        emotion_hint: Mapping[str, Any],
        noise_policy: Mapping[str, Any],
    ) -> dict[str, Any]:
        stressed = emotion_hint.get("label") in {"stressed", "concerned"}
        noisy = noise_policy.get("mode") in {"reduce_verbal_density", "confirm_hearing"}
        return {
            "tone": "gentle" if stressed else "warm",
            "pace": "slow" if stressed else "normal",
            "brevity": "concise" if noisy else "normal",
            "micro_ack": stressed or noisy,
        }


def _band(value: float, *, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "medium"


def _valence_label(value: float) -> str:
    if value <= -0.25:
        return "negative"
    if value >= 0.25:
        return "positive"
    return "neutral"


def _noise_label(noise_policy: Mapping[str, Any]) -> str:
    mode = str(noise_policy.get("mode") or "")
    if mode == "reduce_verbal_density":
        return "high"
    if mode == "confirm_hearing":
        return "medium"
    return "low"


def _time_label(environment_hints: Mapping[str, Any]) -> str:
    value = str(environment_hints.get("time_of_day") or environment_hints.get("time") or "").lower()
    if value in {"night", "late_night", "sleeping_hours"}:
        return "night"
    if value in {"morning", "afternoon", "evening", "day"}:
        return value
    return "day"


def _proximity_label(vision_hints: Mapping[str, Any]) -> str:
    distance = _as_float(vision_hints.get("distance_m"), default=-1.0)
    if distance >= 2.0:
        return "far"
    if 0 <= distance <= 1.2:
        return "near"
    return "mid"


def _proactive_policy(*, mood: str, environment: Mapping[str, Any]) -> dict[str, Any]:
    noise = str(environment.get("noise") or "low")
    time_of_day = str(environment.get("time") or "day")
    proximity = str(environment.get("proximity") or "near")
    vulnerable = mood in {"sad", "anxious", "stressed", "lonely", "tired"}
    suppression_reasons: list[str] = []

    if time_of_day == "night":
        suppression_reasons.append("night")
    if noise == "high":
        suppression_reasons.append("high_noise")
    if proximity == "far":
        suppression_reasons.append("far_distance")

    suppress_speech = bool(suppression_reasons)
    if suppress_speech and (time_of_day == "night" or (noise == "high" and proximity == "far")):
        priority = "defer"
    elif vulnerable:
        priority = "supportive"
    else:
        priority = "low"

    return {
        "priority": priority,
        "disturbance": "nonverbal" if suppress_speech else "low",
        "suppress_speech": suppress_speech,
        "suppression_reasons": suppression_reasons,
    }


__all__ = ["EmotionContextBuilder"]
