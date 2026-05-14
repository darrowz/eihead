"""Deterministic slow-lane reasoning for realtime cognition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class SlowReasoner:
    """Assemble stable decisions without calling a remote LLM."""

    def decide(
        self,
        *,
        turn: Any | None = None,
        round_id: str | None = None,
        cancellation_token: Any | None = None,
        final_text: str | None = None,
        fast_hypotheses: Sequence[Mapping[str, Any]] | None = None,
        memory_candidates: Sequence[Mapping[str, Any]] | None = None,
        persona_context: Mapping[str, Any] | None = None,
        emotion_context: Mapping[str, Any] | None = None,
        execution_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        round_id = round_id or getattr(turn, "round_id", None)
        token_source = cancellation_token if cancellation_token is not None else getattr(turn, "cancellation_token", None)
        token_id = _token_id(token_source)
        cancellation = cancellation_token if cancellation_token is not None else getattr(turn, "cancellation", None)
        if _is_cancelled(cancellation):
            raise RuntimeError("cannot build stable decision for cancelled round/token")
        if not round_id or not token_id:
            raise ValueError("round_id and cancellation_token are required")

        # A committed ASR final on the turn is the source of truth. External
        # callers may pass final_text for direct use, but must not override a
        # finalized user utterance already stored on the blackboard.
        turn_final_text = getattr(turn, "asr_final", None)
        text = (turn_final_text or final_text or "").strip()
        hypotheses = _coerce_sequence(
            fast_hypotheses
            if fast_hypotheses is not None
            else (
                list(getattr(turn, "fast_hypotheses", []) or [])
                + list(getattr(turn, "intent_hypotheses", []) or [])
            )
        )
        memories = _memory_refs(
            memory_candidates
            if memory_candidates is not None
            else getattr(turn, "memory_candidates", []) or []
        )
        persona = _persona(persona_context if persona_context is not None else getattr(turn, "persona_state", {}) or {})
        emotion = _emotion(emotion_context if emotion_context is not None else getattr(turn, "emotion_state", {}) or {})
        intent = _pick_intent(text, hypotheses)
        action_plan = _action_plan(intent=intent, final_text=text, execution_result=execution_result)
        speech_text = _speech_text(final_text=text, intent=intent, memory_refs=memories, emotion=emotion)

        return {
            "round_id": round_id,
            "cancellation_token": token_id,
            "stable": True,
            "decision": intent,
            "final_text": text,
            "confidence": _confidence(hypotheses),
            "speech_text": speech_text,
            "speech_segments": [
                {
                    "text": speech_text,
                    "emotion": emotion.get("tone", "warm"),
                    "startOffsetMs": 0,
                    "stable": True,
                    "source": "slow_reasoner",
                }
            ],
            "action_plan": action_plan,
            "action_segments": action_plan,
            "actions": action_plan,
            "memory_refs": memories,
            "persona": persona,
            "emotion": emotion,
            "hypotheses_used": hypotheses,
            "model": "deterministic_policy",
            "requires_llm": False,
            "source": "slow_reasoner",
        }


def _token_id(token: Any) -> str | None:
    if token is None:
        return None
    token_id = getattr(token, "token_id", None)
    if token_id is not None:
        return str(token_id)
    return str(token)


def _is_cancelled(token: Any) -> bool:
    if token is None:
        return False
    if bool(getattr(token, "is_cancelled", False)):
        return True
    return bool(getattr(token, "cancelled", False))


def _coerce_sequence(items: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    return [dict(item) for item in items or []]


def _persona(persona_context: Mapping[str, Any]) -> dict[str, Any]:
    persona = dict(persona_context)
    persona.setdefault("style", "warm")
    persona.setdefault("language", "zh-CN")
    return persona


def _emotion(emotion_context: Mapping[str, Any]) -> dict[str, Any]:
    emotion = dict(emotion_context)
    mood = str(emotion.get("mood") or emotion.get("state") or "neutral").lower()
    tone = emotion.get("tone")
    if tone is None:
        if mood in {"sad", "anxious", "stressed", "lonely", "tired"}:
            tone = "gentle"
        elif mood in {"focused", "busy"}:
            tone = "concise"
        else:
            tone = "warm"
    emotion["mood"] = mood
    emotion.setdefault("arousal", "medium")
    emotion["tone"] = tone
    return emotion


def _memory_refs(memory_candidates: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    ranked: list[tuple[int, float, Mapping[str, Any]]] = []
    for index, candidate in enumerate(memory_candidates or []):
        if _memory_candidate_filtered(candidate):
            continue
        score = _as_float(candidate.get("score", candidate.get("importance", 0.0)))
        ranked.append((index, score, candidate))

    refs: list[dict[str, Any]] = []
    for _, _, candidate in sorted(ranked, key=lambda item: (-item[1], item[0]))[:3]:
        ref: dict[str, Any] = {}
        if "id" in candidate:
            ref["id"] = candidate["id"]
        text = candidate.get("text") or candidate.get("summary") or candidate.get("content") or candidate.get("query")
        if text is not None:
            ref["text"] = str(text)
        if "score" in candidate:
            ref["score"] = candidate["score"]
        elif "importance" in candidate:
            ref["score"] = candidate["importance"]
        if ref:
            refs.append(ref)
    return refs


def _memory_candidate_filtered(candidate: Mapping[str, Any]) -> bool:
    if str(candidate.get("reply_context_status") or "").strip().lower() == "filtered":
        return True
    if bool(candidate.get("persona_guardrail_applied")):
        return True
    policy = candidate.get("policy_decision")
    if isinstance(policy, Mapping):
        return str(policy.get("decision") or "").strip().lower() in {"filter", "reject", "blocked"}
    return False


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pick_intent(final_text: str, hypotheses: Sequence[Mapping[str, Any]]) -> str:
    heuristic = _heuristic_intent(final_text)
    if heuristic not in {"respond", "no_final_text"}:
        return heuristic
    best: Mapping[str, Any] | None = None
    for hypothesis in hypotheses:
        if not hypothesis.get("intent"):
            continue
        if best is None or _as_float(hypothesis.get("confidence", 0.0)) > _as_float(best.get("confidence", 0.0)):
            best = hypothesis
    if best is not None:
        intent = str(best["intent"])
        if intent in {"listen", "continue_listening", "waiting_for_signal"} and heuristic != "respond":
            return heuristic
        if intent == "possible_memory_writeback":
            return "create_reminder" if heuristic == "create_reminder" else "remember"
        if intent == "possible_action_request":
            return "action_oriented"
        if intent == "possible_question":
            return "answer_question"
        return intent

    return heuristic


def _heuristic_intent(final_text: str) -> str:
    if any(marker in final_text for marker in ("提醒", "记一下", "记得", "帮我记")):
        return "create_reminder"
    if any(marker in final_text for marker in ("吗", "？", "?", "什么", "如何", "为什么", "多少")):
        return "answer_question"
    if any(marker in final_text for marker in ("打开", "启动", "播放", "关闭", "关掉")):
        return "action_oriented"
    if final_text:
        return "respond"
    return "no_final_text"


def _confidence(hypotheses: Sequence[Mapping[str, Any]]) -> float:
    if not hypotheses:
        return 0.62
    return round(max(_as_float(item.get("confidence", 0.0)) for item in hypotheses), 3)


def _action_plan(
    *,
    intent: str,
    final_text: str,
    execution_result: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if intent not in {"action_oriented", "create_reminder", "remember"}:
        return []
    status = "ready"
    if execution_result is not None and execution_result.get("ok") is False:
        status = "needs_retry"
    capability_id = {
        "action_oriented": "action.request",
        "create_reminder": "memory.reminder.create",
        "remember": "memory.write.proposed",
    }.get(intent, intent)
    return [
        {
            "type": intent,
            "capabilityId": capability_id,
            "payload": {"text": final_text},
            "startOffsetMs": 120,
            "durationMs": 0,
            "style": "default",
            "status": status,
            "stable": True,
        }
    ]


def _speech_text(
    *,
    final_text: str,
    intent: str,
    memory_refs: Sequence[Mapping[str, Any]],
    emotion: Mapping[str, Any],
) -> str:
    if not final_text:
        return "我还没有收到完整内容，先不做最终判断。"
    if intent in {"create_reminder", "remember"}:
        text = f"我记下了：{final_text}。"
    elif intent == "action_oriented":
        text = f"我先确认可以执行，再处理：{final_text}。"
    elif intent == "answer_question":
        text = f"我听到你的问题：{final_text}。我会结合上下文给出稳定回应。"
    else:
        text = f"我听到了：{final_text}。"

    if memory_refs:
        text += "我会参考相关记忆。"
    if emotion.get("mood") in {"sad", "anxious", "stressed", "lonely", "tired"}:
        text += "我会尽量温和一点。"
    return text
