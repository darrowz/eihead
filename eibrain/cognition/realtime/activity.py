"""Low-disturbance proactive activity proposals."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable


Cap = tuple[int, float]


class ProactiveActivityManager:
    """Choose silent, visual-only, or spoken proactive activity."""

    def __init__(
        self,
        *,
        min_idle_seconds: float = 30.0,
        visual_idle_seconds: float = 60.0,
        speak_idle_seconds: float = 120.0,
        channel_cooldowns: Mapping[str, float] | None = None,
        channel_caps: Mapping[str, Cap] | None = None,
        event_type_caps: Mapping[str, Cap] | None = None,
        feedback_cooldown_seconds: float = 900.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.min_idle_seconds = min_idle_seconds
        self.visual_idle_seconds = visual_idle_seconds
        self.speak_idle_seconds = speak_idle_seconds
        self.channel_cooldowns = {
            "speak": 600.0,
            "visual_only": 0.0,
            **dict(channel_cooldowns or {}),
        }
        self.channel_caps = {
            "speak": (1, 900.0),
            "visual_only": (4, 900.0),
            **dict(channel_caps or {}),
        }
        self.event_type_caps = {
            "emotion_check_in": (1, 900.0),
            "emotion_and_memory_nudge": (3, 900.0),
            "memory_nudge": (2, 1200.0),
            "long_quiet_check_in": (1, 1200.0),
            "execution_follow_up": (2, 600.0),
            "execution_result_needs_attention": (2, 600.0),
            **dict(event_type_caps or {}),
        }
        self.feedback_cooldown_seconds = feedback_cooldown_seconds
        self.clock = clock or time.monotonic
        self._last_emit_at: dict[str, float] = {}
        self._emissions: list[dict[str, Any]] = []
        self._quiet_until = 0.0
        self._metrics: dict[str, Any] = {
            "proposed": 0,
            "suppressed": 0,
            "emitted": 0,
            "reason_counts": {},
            "next_allowed_at": None,
        }

    def propose(
        self,
        *,
        idle_seconds: float,
        emotion_context: Mapping[str, Any] | None = None,
        memory_candidates: Sequence[Mapping[str, Any]] | None = None,
        execution_result: Mapping[str, Any] | None = None,
        round_id: str | None = None,
        cancellation_token: str | None = None,
        allow_speech: bool = True,
        **aliases: Any,
    ) -> dict[str, Any]:
        now = _now_seconds(aliases=aliases, clock=self.clock)
        emotion_context = emotion_context or aliases.get("emotion") or aliases.get("emotion_state") or {}
        memory_candidates = memory_candidates or aliases.get("memories") or aliases.get("memory") or []
        memory_refs = _memory_refs(memory_candidates)
        emotion = _emotion_state(emotion_context)
        mood = str(emotion.get("mood") or emotion.get("state") or "neutral").lower()
        proactive_policy = _proactive_policy(emotion)
        execution_failed = execution_result is not None and execution_result.get("ok") is False
        needs_followup = _needs_followup(execution_result)
        long_quiet = idle_seconds >= self.speak_idle_seconds
        urgency = _urgency(
            mood=mood,
            memory_refs=memory_refs,
            execution_failed=execution_failed,
            needs_followup=needs_followup,
            long_quiet=long_quiet,
        )
        recent_user_interrupt = bool(
            aliases.get("recent_user_interrupt")
            or aliases.get("user_recently_interrupted")
            or emotion.get("recent_user_interrupt")
        )
        if recent_user_interrupt or _has_negative_feedback(emotion=emotion, aliases=aliases):
            self._quiet_until = max(self._quiet_until, now + self.feedback_cooldown_seconds)

        suppression_reason = _suppression_reason(
            emotion=emotion,
            allow_speech=allow_speech,
            recent_user_interrupt=recent_user_interrupt,
            proactive_policy=proactive_policy,
        )
        disturbance = str(proactive_policy.get("disturbance") or "low")

        if recent_user_interrupt:
            return self._proposal(
                channel="silent",
                reason="recent_user_interrupt",
                text="",
                urgency=0.0,
                emotion=emotion,
                memory_refs=memory_refs,
                round_id=round_id,
                cancellation_token=cancellation_token,
                disturbance=disturbance,
                speech_suppressed=True,
                suppression_reason="recent_user_interrupt",
                now_seconds=now,
                event_type="recent_user_interrupt",
                next_allowed_at=self._quiet_until,
            )

        if idle_seconds < self.min_idle_seconds and not needs_followup:
            return self._proposal(
                channel="silent",
                reason="recent_activity",
                text="",
                urgency=0.0,
                emotion=emotion,
                memory_refs=memory_refs,
                round_id=round_id,
                cancellation_token=cancellation_token,
                disturbance=disturbance,
                now_seconds=now,
                event_type="recent_activity",
            )

        reason = _reason(
            mood=mood,
            memory_refs=memory_refs,
            execution_failed=execution_failed,
            needs_followup=needs_followup,
            long_quiet=long_quiet,
        )
        would_speak = idle_seconds >= self.speak_idle_seconds and urgency >= 0.65 and not execution_failed
        if would_speak and suppression_reason == "":
            channel = "speak"
        elif (idle_seconds >= self.visual_idle_seconds or needs_followup or would_speak) and urgency > 0.0:
            channel = "visual_only"
        else:
            channel = "silent"

        if channel != "silent":
            blocked_reason, next_allowed_at = self._rate_limit(
                channel=channel,
                event_type=reason,
                now_seconds=now,
            )
            if blocked_reason:
                return self._proposal(
                    channel="silent",
                    reason="rate_limited",
                    text="",
                    urgency=0.0,
                    emotion=emotion,
                    memory_refs=memory_refs,
                    round_id=round_id,
                    cancellation_token=cancellation_token,
                    disturbance=disturbance,
                    speech_suppressed=channel == "speak",
                    suppression_reason=blocked_reason,
                    now_seconds=now,
                    event_type=reason,
                    next_allowed_at=next_allowed_at,
                )

        return self._proposal(
            channel=channel,
            reason=reason,
            text=_text(
                channel=channel,
                mood=mood,
                memory_refs=memory_refs,
                execution_failed=execution_failed,
                execution_result=execution_result,
                needs_followup=needs_followup,
                long_quiet=long_quiet,
            ),
            urgency=urgency,
            emotion=emotion,
            memory_refs=memory_refs,
            round_id=round_id,
            cancellation_token=cancellation_token,
            disturbance=disturbance,
            speech_suppressed=channel == "visual_only" and would_speak and suppression_reason != "",
            suppression_reason=suppression_reason if channel == "visual_only" and would_speak else "",
            now_seconds=now,
            event_type=reason,
        )

    def _proposal(
        self,
        *,
        channel: str,
        reason: str,
        text: str,
        urgency: float,
        emotion: Mapping[str, Any],
        memory_refs: Sequence[Mapping[str, Any]],
        round_id: str | None,
        cancellation_token: str | None,
        disturbance: str,
        speech_suppressed: bool = False,
        suppression_reason: str = "",
        now_seconds: float,
        event_type: str,
        next_allowed_at: float | None = None,
    ) -> dict[str, Any]:
        should_emit = channel != "silent"
        if should_emit:
            self._remember_emission(channel=channel, event_type=event_type, now_seconds=now_seconds)
            next_allowed_at = self._next_allowed_at(channel=channel, event_type=event_type, now_seconds=now_seconds)
        self._record_metrics(
            should_emit=should_emit,
            reason=suppression_reason or reason,
            next_allowed_at=next_allowed_at,
        )
        payload = {
            "type": "proactive_activity",
            "round_id": round_id,
            "cancellation_token": cancellation_token,
            "channel": channel,
            "disturbance": disturbance,
            "should_emit": should_emit,
            "requires_user_attention": False,
            "reason": reason,
            "text": text,
            "urgency": round(max(0.0, min(1.0, urgency)), 3),
            "emotion": dict(emotion),
            "memory_refs": [dict(item) for item in memory_refs],
            "source": "proactive_activity_manager",
            "next_allowed_at": next_allowed_at,
            "metrics": self.metrics(),
        }
        payload["speech_suppressed"] = bool(speech_suppressed)
        payload["suppression_reason"] = suppression_reason
        payload["summary"] = {
            "channel": channel,
            "reason": reason,
            "should_emit": should_emit,
            "disturbance": disturbance,
            "urgency": payload["urgency"],
        }
        return payload

    def metrics(self) -> dict[str, Any]:
        return {
            "proposed": int(self._metrics["proposed"]),
            "suppressed": int(self._metrics["suppressed"]),
            "emitted": int(self._metrics["emitted"]),
            "reason_counts": dict(self._metrics["reason_counts"]),
            "next_allowed_at": self._metrics["next_allowed_at"],
        }

    def _record_metrics(
        self,
        *,
        should_emit: bool,
        reason: str,
        next_allowed_at: float | None,
    ) -> None:
        self._metrics["proposed"] += 1
        if should_emit:
            self._metrics["emitted"] += 1
        else:
            self._metrics["suppressed"] += 1
        reason_counts = self._metrics["reason_counts"]
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        self._metrics["next_allowed_at"] = next_allowed_at

    def _remember_emission(self, *, channel: str, event_type: str, now_seconds: float) -> None:
        self._last_emit_at[channel] = now_seconds
        self._emissions.append({"at": now_seconds, "channel": channel, "event_type": event_type})
        oldest_relevant = now_seconds - self._max_window_seconds()
        self._emissions = [item for item in self._emissions if item["at"] >= oldest_relevant]

    def _rate_limit(self, *, channel: str, event_type: str, now_seconds: float) -> tuple[str, float | None]:
        if self._quiet_until > now_seconds:
            return "negative_feedback_cooldown", self._quiet_until

        cooldown_until = self._channel_cooldown_until(channel=channel)
        if cooldown_until is not None and cooldown_until > now_seconds:
            return "channel_cooldown", cooldown_until

        channel_cap_until = self._cap_until(
            key="channel",
            value=channel,
            cap=self.channel_caps.get(channel),
            now_seconds=now_seconds,
        )
        if channel_cap_until is not None and channel_cap_until > now_seconds:
            return "channel_cap", channel_cap_until

        event_cap_until = self._cap_until(
            key="event_type",
            value=event_type,
            cap=self.event_type_caps.get(event_type),
            now_seconds=now_seconds,
        )
        if event_cap_until is not None and event_cap_until > now_seconds:
            return "event_type_cap", event_cap_until

        return "", None

    def _channel_cooldown_until(self, *, channel: str) -> float | None:
        last_emit = self._last_emit_at.get(channel)
        if last_emit is None:
            return None
        cooldown = self.channel_cooldowns.get(channel, 0.0)
        return last_emit + cooldown

    def _cap_until(
        self,
        *,
        key: str,
        value: str,
        cap: Cap | None,
        now_seconds: float,
    ) -> float | None:
        if cap is None:
            return None
        limit, window_seconds = cap
        if limit <= 0:
            return now_seconds + window_seconds
        window_start = now_seconds - window_seconds
        matching = sorted(item["at"] for item in self._emissions if item.get(key) == value and item["at"] > window_start)
        if len(matching) < limit:
            return None
        return matching[0] + window_seconds

    def _next_allowed_at(self, *, channel: str, event_type: str, now_seconds: float) -> float | None:
        candidates = [
            value
            for value in (
                self._quiet_until if self._quiet_until > now_seconds else None,
                self._channel_cooldown_until(channel=channel),
                self._cap_until(
                    key="channel",
                    value=channel,
                    cap=self.channel_caps.get(channel),
                    now_seconds=now_seconds,
                ),
                self._cap_until(
                    key="event_type",
                    value=event_type,
                    cap=self.event_type_caps.get(event_type),
                    now_seconds=now_seconds,
                ),
            )
            if value is not None and value > now_seconds
        ]
        return max(candidates) if candidates else None

    def _max_window_seconds(self) -> float:
        caps = [*self.channel_caps.values(), *self.event_type_caps.values()]
        if not caps:
            return 0.0
        return max(window_seconds for _, window_seconds in caps)


def _emotion_state(emotion_context: Mapping[str, Any]) -> dict[str, Any]:
    emotion = dict(emotion_context)
    nested = emotion.get("emotion_state")
    if isinstance(nested, Mapping):
        merged = dict(nested)
        for key, value in emotion.items():
            if key != "emotion_state" and key not in merged:
                merged[key] = value
        return merged
    return emotion


def _now_seconds(*, aliases: Mapping[str, Any], clock: Callable[[], float]) -> float:
    if "now_seconds" in aliases:
        return _as_float(aliases["now_seconds"])
    if "now" in aliases:
        return _as_float(aliases["now"])
    return float(clock())


def _proactive_policy(emotion: Mapping[str, Any]) -> dict[str, Any]:
    proactive = emotion.get("proactive")
    if isinstance(proactive, Mapping):
        return dict(proactive)
    strategy = emotion.get("response_strategy")
    if isinstance(strategy, Mapping):
        policy: dict[str, Any] = {}
        if "proactive_priority" in strategy:
            policy["priority"] = strategy["proactive_priority"]
        if "proactive_disturbance" in strategy:
            policy["disturbance"] = strategy["proactive_disturbance"]
        if "proactive_suppression_reasons" in strategy:
            policy["suppression_reasons"] = strategy["proactive_suppression_reasons"]
        if "proactive_suppress_speech" in strategy:
            policy["suppress_speech"] = strategy["proactive_suppress_speech"]
        return policy
    return {}


def _has_negative_feedback(*, emotion: Mapping[str, Any], aliases: Mapping[str, Any]) -> bool:
    if aliases.get("recent_user_rejection") or aliases.get("low_satisfaction_feedback"):
        return True
    feedback = aliases.get("user_feedback") or emotion.get("user_feedback") or emotion.get("feedback")
    if isinstance(feedback, Mapping):
        if feedback.get("rejected") or feedback.get("refused") or feedback.get("interrupted"):
            return True
        satisfaction = feedback.get("satisfaction")
        if satisfaction is not None and _as_float(satisfaction) <= 0.35:
            return True
        rating = feedback.get("rating")
        if rating is not None and _as_float(rating) <= 2.0:
            return True
    return False


def _memory_refs(memory_candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for candidate in memory_candidates[:3]:
        ref: dict[str, Any] = {}
        if "id" in candidate:
            ref["id"] = candidate["id"]
        text = candidate.get("text") or candidate.get("summary") or candidate.get("content") or candidate.get("query")
        if text is not None:
            ref["text"] = str(text)
        score = candidate.get("importance", candidate.get("score"))
        if score is not None:
            ref["importance"] = _as_float(score)
        if ref:
            refs.append(ref)
    return refs


def _urgency(
    *,
    mood: str,
    memory_refs: Sequence[Mapping[str, Any]],
    execution_failed: bool,
    needs_followup: bool,
    long_quiet: bool,
) -> float:
    urgency = 0.0
    if mood in {"sad", "anxious", "stressed", "lonely", "tired"}:
        urgency += 0.45
    if execution_failed:
        urgency += 0.45
    if needs_followup:
        urgency += 0.5
    if long_quiet:
        urgency += 0.25
    if memory_refs:
        urgency += max(_as_float(item.get("importance", 0.0)) for item in memory_refs) * 0.45
    return min(1.0, urgency)


def _reason(
    *,
    mood: str,
    memory_refs: Sequence[Mapping[str, Any]],
    execution_failed: bool,
    needs_followup: bool,
    long_quiet: bool,
) -> str:
    if execution_failed:
        return "execution_result_needs_attention"
    if needs_followup:
        return "execution_follow_up"
    if memory_refs and mood in {"sad", "anxious", "stressed", "lonely", "tired"}:
        return "emotion_and_memory_nudge"
    if memory_refs:
        return "memory_nudge"
    if mood in {"sad", "anxious", "stressed", "lonely", "tired"}:
        return "emotion_check_in"
    if long_quiet:
        return "long_quiet_check_in"
    return "no_low_disturbance_opportunity"


def _text(
    *,
    channel: str,
    mood: str,
    memory_refs: Sequence[Mapping[str, Any]],
    execution_failed: bool,
    execution_result: Mapping[str, Any] | None,
    needs_followup: bool,
    long_quiet: bool,
) -> str:
    if channel == "silent":
        return ""
    if execution_failed:
        return "我刚才的执行没有成功，先在这里留一个低打扰提示。"
    if needs_followup:
        summary = str((execution_result or {}).get("summary") or "我可以稍后继续确认刚才的执行结果")
        return f"低打扰回访：{summary}。"
    memory_text = memory_refs[0].get("text") if memory_refs else None
    if channel == "speak":
        if memory_text:
            return f"我注意到你可能需要一点提醒：{memory_text}。"
        if mood in {"sad", "anxious", "stressed", "lonely", "tired"}:
            return "我在这里，需要我轻轻陪你一下吗？"
        return "我注意到你可能需要一点帮助，要我继续吗？"
    if memory_text:
        return f"低打扰提示：{memory_text}。"
    if mood in {"sad", "anxious", "stressed", "lonely", "tired"}:
        return "低打扰提示：我可以在你需要时继续陪你。"
    if long_quiet:
        return "低打扰提示：我在这里，等你需要时再开口。"
    return "低打扰提示已准备。"


def _needs_followup(execution_result: Mapping[str, Any] | None) -> bool:
    if not execution_result:
        return False
    return bool(
        execution_result.get("needs_followup")
        or execution_result.get("follow_up")
        or execution_result.get("followup")
    )


def _suppression_reason(
    *,
    emotion: Mapping[str, Any],
    allow_speech: bool,
    recent_user_interrupt: bool,
    proactive_policy: Mapping[str, Any],
) -> str:
    if recent_user_interrupt:
        return "recent_user_interrupt"
    if not allow_speech:
        return "speech_not_allowed"
    if proactive_policy.get("suppress_speech") is True:
        reasons = proactive_policy.get("suppression_reasons")
        if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)) and reasons:
            return str(reasons[0])
        return "proactive_nonverbal_preferred"
    environment = emotion.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    if str(environment.get("time") or "").lower() == "night":
        return "night"
    if str(environment.get("noise") or "").lower() == "high":
        return "high_noise"
    if str(environment.get("proximity") or "").lower() == "far":
        return "far_distance"
    return ""


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
