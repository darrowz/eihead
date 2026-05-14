"""Fast, non-committal realtime hypothesis generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _without_empty(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in payload.items() if item not in (None, "", [], {})}


def _context_summary(persona_state: Any, emotion_state: Any) -> dict[str, Any]:
    return _json_ready(
        {
            "persona": _persona_summary(persona_state),
            "emotion": _emotion_summary(emotion_state),
        }
    )


def _persona_summary(persona_state: Any) -> dict[str, Any]:
    if not isinstance(persona_state, Mapping):
        return {}
    response_policy = persona_state.get("response_policy")
    response_policy = response_policy if isinstance(response_policy, Mapping) else {}
    speaking_style = persona_state.get("speaking_style")
    speaking_style = speaking_style if isinstance(speaking_style, Mapping) else {}
    return _without_empty(
        {
            "personaCode": persona_state.get("personaCode")
            or persona_state.get("persona_code")
            or persona_state.get("persona_id"),
            "voice_code": persona_state.get("voice_code") or persona_state.get("voiceCode"),
            "tone": speaking_style.get("tone") or persona_state.get("tone") or persona_state.get("style"),
            "max_chars": response_policy.get("max_chars"),
        }
    )


def _emotion_summary(emotion_state: Any) -> dict[str, Any]:
    if not isinstance(emotion_state, Mapping):
        return {}
    nested = emotion_state.get("emotion_state")
    state = nested if isinstance(nested, Mapping) else emotion_state
    environment = state.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    strategy = emotion_state.get("response_strategy")
    strategy = strategy if isinstance(strategy, Mapping) else {}
    return _without_empty(
        {
            "mood": state.get("mood") or state.get("state") or _emotion_hint_label(emotion_state),
            "energy": state.get("energy"),
            "noise": environment.get("noise"),
            "time": environment.get("time"),
            "proximity": environment.get("proximity"),
            "tone": strategy.get("tone") or emotion_state.get("tone"),
        }
    )


def _emotion_hint_label(emotion_state: Mapping[str, Any]) -> str:
    hint = emotion_state.get("emotion_hint")
    if isinstance(hint, Mapping):
        return str(hint.get("label") or "")
    return ""


@dataclass(slots=True)
class FastThinkOutput:
    """JSON-ready output from the fast lane.

    The fast lane is intentionally unstable: it may hint, but must not decide.
    """

    round_id: str
    cancellation_token: str | None
    deadline_ms: int
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    micro_feedback: dict[str, Any] = field(default_factory=dict)
    intent_hints: list[dict[str, Any]] = field(default_factory=list)
    context_summary: dict[str, Any] = field(default_factory=dict)
    stable: bool = False
    source: str = "fast_think"

    @property
    def microfeedback(self) -> str:
        return str(self.micro_feedback.get("text", ""))

    @property
    def intent_hypotheses(self) -> list[dict[str, Any]]:
        return list(self.intent_hints)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(
            {
                "round_id": self.round_id,
                "cancellation_token": self.cancellation_token,
                "deadline_ms": self.deadline_ms,
                "hypotheses": self.hypotheses,
                "micro_feedback": self.micro_feedback,
                "intent_hints": self.intent_hints,
                "context_summary": self.context_summary,
                "stable": False,
                "source": self.source,
            }
        )


class FastThinkEngine:
    """Generate low-risk hints from partial ASR, vision, and emotion signals."""

    SAFE_MICRO_FEEDBACK: tuple[str, ...] = (
        "我在听，你可以继续说。",
        "我先陪你梳理一下。",
        "我听到了，先不急着判断。",
        "可以慢慢说，我先保持确认。",
        "我先听完整，再一起确认。",
    )

    _QUESTION_MARKERS = ("吗", "？", "?", "什么", "如何", "怎么", "谁", "多少", "为什么", "会不会")
    _ACTION_MARKERS = ("打开", "启动", "播放", "关掉", "关闭", "调高", "调低", "帮我", "能不能", "可以")
    _MEMORY_MARKERS = ("记住", "记得", "以后", "下次", "偏好", "喜欢", "不喜欢", "prefer", "remember")
    _STRESS_LABELS = {"stress", "stressed", "anxious", "uncertain", "tired", "frustrated", "紧张", "不确定", "累"}

    def process(
        self,
        turn: Any,
        *,
        asr_partial_text: str | None = None,
        vision_hint: Mapping[str, Any] | None = None,
        emotion_hint: Mapping[str, Any] | None = None,
        observations: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | None = None,
        deadline_ms: int = 500,
    ) -> FastThinkOutput:
        round_id = str(_value(turn, "round_id", ""))
        cancellation_token = _value(turn, "cancellation_token")
        extracted = self._extract_observations(observations or ())
        text = _clean_text(asr_partial_text if asr_partial_text is not None else extracted["asr_text"])
        if not text:
            text = self._latest_partial_from_turn(turn)

        merged_vision = self._merge_hint(extracted["vision_hint"], vision_hint)
        merged_emotion = self._merge_hint(extracted["emotion_hint"], emotion_hint or _value(turn, "emotion_state", {}))
        ms = min(500, max(50, int(deadline_ms or 500)))

        hypotheses = [self.sanitize_hypothesis(item) for item in self._build_hypotheses(text, merged_vision, merged_emotion)]
        intent_hints = [self.sanitize_hypothesis(item) for item in self._build_intent_hints(hypotheses)]
        micro_feedback = self._micro_feedback(text=text, emotion_hint=merged_emotion, hypotheses=hypotheses)
        context_summary = _context_summary(_value(turn, "persona_state", {}), merged_emotion)

        self._record_hypotheses(turn, hypotheses)
        self._record_intents(turn, intent_hints)

        return FastThinkOutput(
            round_id=round_id,
            cancellation_token=str(cancellation_token) if cancellation_token is not None else None,
            deadline_ms=ms,
            hypotheses=hypotheses,
            micro_feedback=micro_feedback,
            intent_hints=intent_hints,
            context_summary=context_summary,
            stable=False,
        )

    def process_partial(self, turn: Any, asr_text: str, *, deadline_ms: int = 500) -> FastThinkOutput:
        return self.process(turn, asr_partial_text=asr_text, deadline_ms=deadline_ms)

    def sanitize_hypothesis(self, hypothesis: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(hypothesis)
        if payload.get("stable") is True:
            raise ValueError("fast hypotheses must remain non-stable")
        if "decision" in payload or "final_decision" in payload:
            raise ValueError("fast lane cannot emit final decisions")
        payload["stable"] = False
        payload.setdefault("source", "fast_think")
        payload.setdefault("confidence", 0.4)
        payload["confidence"] = max(0.0, min(0.89, float(payload.get("confidence") or 0.0)))
        return _json_ready(payload)

    def _build_hypotheses(
        self,
        text: str,
        vision_hint: dict[str, Any],
        emotion_hint: dict[str, Any],
    ) -> list[dict[str, Any]]:
        hypotheses: list[dict[str, Any]] = []
        if text:
            lowered = text.lower()
            if any(marker in text or marker in lowered for marker in self._QUESTION_MARKERS):
                hypotheses.append(
                    {
                        "kind": "intent_hypothesis",
                        "intent": "possible_question",
                        "confidence": 0.58,
                        "source": "asr_partial",
                        "evidence": text,
                    }
                )
            if any(marker in text or marker in lowered for marker in self._ACTION_MARKERS):
                hypotheses.append(
                    {
                        "kind": "intent_hypothesis",
                        "intent": "possible_action_request",
                        "confidence": 0.66,
                        "source": "asr_partial",
                        "evidence": text,
                    }
                )
            if any(marker in text or marker in lowered for marker in self._MEMORY_MARKERS):
                hypotheses.append(
                    {
                        "kind": "intent_hypothesis",
                        "intent": "possible_memory_writeback",
                        "confidence": 0.62,
                        "source": "asr_partial",
                        "evidence": text,
                    }
                )
            if not hypotheses:
                hypotheses.append(
                    {
                        "kind": "intent_hypothesis",
                        "intent": "continue_listening",
                        "confidence": 0.41,
                        "source": "asr_partial",
                        "evidence": text,
                    }
                )

        if vision_hint:
            hypotheses.append(
                {
                    "kind": "context_hypothesis",
                    "intent": "possible_environment_context",
                    "confidence": 0.48,
                    "source": "vision_hint",
                    "evidence": vision_hint,
                }
            )

        if emotion_hint:
            hypotheses.append(
                {
                    "kind": "affect_hypothesis",
                    "intent": "supportive_tone_hint",
                    "confidence": self._emotion_confidence(emotion_hint),
                    "source": "emotion_hint",
                    "evidence": emotion_hint,
                }
            )

        if not hypotheses:
            hypotheses.append(
                {
                    "kind": "intent_hypothesis",
                    "intent": "waiting_for_signal",
                    "confidence": 0.3,
                    "source": "fast_think",
                }
            )
        return hypotheses

    def _build_intent_hints(self, hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hints: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in hypotheses:
            intent = str(item.get("intent") or "")
            if not intent or intent in seen:
                continue
            seen.add(intent)
            hints.append(
                {
                    "intent": intent,
                    "confidence": item.get("confidence", 0.4),
                    "source": item.get("source", "fast_think"),
                    "reason": "fast_lane_hint_only",
                    "stable": False,
                }
            )
        return hints

    def _micro_feedback(
        self,
        *,
        text: str,
        emotion_hint: dict[str, Any],
        hypotheses: list[dict[str, Any]],
    ) -> dict[str, Any]:
        intents = {str(item.get("intent") or "") for item in hypotheses}
        if self._emotion_needs_softening(emotion_hint):
            phrase = "我在听，你可以继续说。"
        elif "possible_action_request" in intents:
            phrase = "我听到了，先不急着判断。"
        elif "possible_question" in intents or text:
            phrase = "我先听完整，再一起确认。"
        else:
            phrase = "我先陪你梳理一下。"
        return {
            "kind": "companionship",
            "risk": "low",
            "text": phrase,
            "stable": False,
            "policy": "no_facts_no_final_decision",
        }

    def _extract_observations(self, observations: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
        asr_text = ""
        vision_hint: dict[str, Any] = {}
        emotion_hint: dict[str, Any] = {}
        for observation in observations:
            kind = str(observation.get("kind", "") or "")
            payload = observation.get("payload")
            payload_dict = dict(payload) if isinstance(payload, Mapping) else {}
            if kind == "asr_partial":
                asr_text = _clean_text(payload_dict.get("text", asr_text))
            elif kind == "vision":
                vision_hint.update(payload_dict)
            elif kind in {"prosody", "environment"}:
                emotion_hint.update(payload_dict)
        return {"asr_text": asr_text, "vision_hint": vision_hint, "emotion_hint": emotion_hint}

    def _latest_partial_from_turn(self, turn: Any) -> str:
        partials = _value(turn, "asr_partial", [])
        if isinstance(partials, (list, tuple)) and partials:
            return _clean_text(partials[-1])
        if isinstance(partials, str):
            return _clean_text(partials)
        return ""

    def _record_hypotheses(self, turn: Any, hypotheses: list[dict[str, Any]]) -> None:
        if hasattr(turn, "append_hypothesis"):
            for item in hypotheses:
                turn.append_hypothesis(item)
            return
        current = _value(turn, "fast_hypotheses")
        if isinstance(current, list):
            current.extend(hypotheses)
        elif isinstance(turn, dict):
            turn.setdefault("fast_hypotheses", []).extend(hypotheses)

    def _record_intents(self, turn: Any, intent_hints: list[dict[str, Any]]) -> None:
        current = _value(turn, "intent_hypotheses")
        if isinstance(current, list):
            current.extend(intent_hints)
        elif isinstance(turn, dict):
            turn.setdefault("intent_hypotheses", []).extend(intent_hints)

    def _merge_hint(self, first: Mapping[str, Any] | None, second: Mapping[str, Any] | None) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if first:
            merged.update(dict(first))
        if second:
            merged.update(dict(second))
        return _json_ready(merged)

    def _emotion_confidence(self, hint: Mapping[str, Any]) -> float:
        label = self._emotion_label(hint)
        stress = self._signal_float(hint.get("stress", 0.0))
        arousal = self._signal_float(hint.get("arousal", 0.0))
        if label in self._STRESS_LABELS or stress >= 0.6 or arousal >= 0.7:
            return 0.64
        return 0.45

    def _emotion_needs_softening(self, hint: Mapping[str, Any]) -> bool:
        label = self._emotion_label(hint)
        stress = self._signal_float(hint.get("stress", 0.0))
        arousal = self._signal_float(hint.get("arousal", 0.0))
        return label in self._STRESS_LABELS or stress >= 0.6 or arousal >= 0.7

    def _emotion_label(self, hint: Mapping[str, Any]) -> str:
        label = str(hint.get("label") or hint.get("emotion") or "").lower()
        if label:
            return label
        nested = hint.get("emotion_hint")
        if isinstance(nested, Mapping):
            return str(nested.get("label") or "").lower()
        return ""

    def _signal_float(self, value: Any) -> float:
        if isinstance(value, str):
            band = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(value.lower())
            if band is not None:
                return band
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0


__all__ = ["FastThinkEngine", "FastThinkOutput"]
