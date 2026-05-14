"""Compatibility helpers for eimemory scoring metadata."""

from __future__ import annotations

from typing import Any, Mapping


_TIERS = ("rejected", "candidate", "confirmed", "core")
_KNOWN_NAMESPACES = {
    "relevance",
    "confidence",
    "salience",
    "freshness",
    "provenance",
    "reuse",
    "risk",
    "lifecycle",
}


def normalize_memory_metadata(meta: Mapping[str, object] | None) -> dict[str, object]:
    """Return metadata with v1 scoring and legacy quality kept in sync."""

    merged = merge_memory_metadata(meta)
    quality = _mapping(merged.get("quality"))
    scoring = _mapping(merged.get("scoring"))
    score = _normalize_memory_score(scoring.get("memory_score_v1"))
    if not score and quality:
        score = _score_from_legacy_quality(quality)
    if score:
        scoring["memory_score_v1"] = score
        merged["scoring"] = scoring
    if quality:
        merged["quality"] = _normalize_legacy_quality(quality, score=score)
    elif score:
        merged["quality"] = _legacy_quality_from_score(score)
    return merged


def merge_memory_metadata(*metas: Mapping[str, object] | None) -> dict[str, object]:
    """Merge metadata payloads without dropping nested scoring fields."""

    merged: dict[str, object] = {}
    scoring_payload: dict[str, object] = {}
    quality_payload: dict[str, object] = {}
    for meta in metas:
        payload = _mapping(meta)
        if not payload:
            continue
        for key, value in payload.items():
            if key == "scoring":
                scoring_payload = _merge_nested(scoring_payload, _mapping(value))
                continue
            if key == "quality":
                quality_payload = _merge_nested(quality_payload, _mapping(value))
                continue
            if key == "memory_score_v1":
                scoring_payload = _merge_nested(scoring_payload, {"memory_score_v1": _mapping(value)})
                continue
            merged[key] = value
    if quality_payload:
        merged["quality"] = quality_payload
    if scoring_payload:
        merged["scoring"] = scoring_payload
    return merged


def score_meta_from_recall_entry(entry: Mapping[str, object] | None) -> dict[str, object]:
    """Build metadata-compatible scoring payload from recall explanation rows."""

    payload = _mapping(entry)
    if not payload:
        return {}
    direct_meta = merge_memory_metadata(
        _mapping(payload.get("meta")),
        _mapping(payload.get("scoring")),
        {"memory_score_v1": _mapping(payload.get("memory_score_v1"))},
        {"quality": _mapping(payload.get("quality"))},
    )
    if direct_meta:
        normalized = normalize_memory_metadata(direct_meta)
        scoring = _mapping(normalized.get("scoring"))
        if _mapping(scoring.get("memory_score_v1")) or _mapping(normalized.get("quality")):
            return normalized
    final_score = _clamp_float(payload.get("final_score"))
    quality_score = _clamp_float(payload.get("quality_score"))
    if final_score is None and quality_score is None:
        return {}
    tier = _normalize_tier(payload.get("quality_tier"), final_score if final_score is not None else quality_score)
    salience = quality_score if quality_score is not None else final_score
    score = {
        "schema_version": "memory_score.v1",
        "final_score": round(final_score if final_score is not None else float(quality_score or 0.0), 3),
        "tier": tier,
        "components": {
            "salience": {
                "name": "salience",
                "value": round(float(salience or 0.0), 3),
                "weight": 0.0,
                "evidence": {"source": "eimemory.recall.scoring"},
            }
        }
        if salience is not None
        else {},
        "labels": normalize_scoring_labels(payload.get("labels"), tier=tier, salience=salience),
        "explanation": {
            "compatibility_source": "eimemory.recall.scoring",
            "quality_tier": tier,
            "quality_score": quality_score,
        },
        "provenance": {
            "agent": "eibrain.memory.compat",
            "activity": "memory.recall_score",
            "source": "eimemory.recall",
        },
    }
    return normalize_memory_metadata({"scoring": {"memory_score_v1": score}})


def normalize_scoring_labels(
    labels: object,
    *,
    tier: str = "",
    confidence: float | None = None,
    salience: float | None = None,
    freshness: float | None = None,
    reuse: float | None = None,
) -> list[str]:
    normalized: list[str] = []
    for raw in labels if isinstance(labels, list) else []:
        label = _canonical_label(raw)
        if label and label not in normalized:
            normalized.append(label)
    for derived in (
        f"lifecycle.{tier}" if tier else "",
        _level_label("confidence", confidence),
        _level_label("salience", salience),
        _freshness_label(freshness),
        _level_label("reuse", reuse),
    ):
        if derived and derived not in normalized:
            normalized.append(derived)
    return normalized


def _normalize_memory_score(value: object) -> dict[str, object]:
    payload = _mapping(value)
    if not payload:
        return {}
    final_score = _clamp_float(payload.get("final_score", payload.get("score")))
    tier = _normalize_tier(payload.get("tier"), final_score)
    components = _normalize_components(payload.get("components"))
    confidence = _component_value(components, "confidence")
    salience = _component_value(components, "salience")
    freshness = _component_value(components, "freshness")
    reuse = _component_value(components, "reuse")
    if salience is None:
        salience = _clamp_float(payload.get("quality_score"))
    score: dict[str, object] = {
        "schema_version": "memory_score.v1",
        "final_score": round(final_score if final_score is not None else 0.0, 3),
        "tier": tier,
        "components": components,
        "labels": normalize_scoring_labels(
            payload.get("labels"),
            tier=tier,
            confidence=confidence,
            salience=salience,
            freshness=freshness,
            reuse=reuse,
        ),
    }
    explanation = _mapping(payload.get("explanation"))
    provenance = _mapping(payload.get("provenance"))
    if explanation:
        score["explanation"] = explanation
    if provenance:
        score["provenance"] = provenance
    return score


def _normalize_legacy_quality(quality: Mapping[str, object], *, score: Mapping[str, object] | None) -> dict[str, object]:
    normalized = _legacy_quality_from_score(score or {})
    incoming = dict(quality)
    salience = _clamp_float(incoming.get("salience_score", incoming.get("importance")))
    confidence = _clamp_float(incoming.get("confidence"))
    freshness = _clamp_float(incoming.get("freshness"))
    reuse = _clamp_float(incoming.get("reuse_potential"))
    if salience is not None:
        normalized["importance"] = salience
        normalized["salience_score"] = salience
    if confidence is not None:
        normalized["confidence"] = confidence
    if freshness is not None:
        normalized["freshness"] = freshness
    if reuse is not None:
        normalized["reuse_potential"] = reuse
    tier = _normalize_tier(incoming.get("quality_tier"), _clamp_float(incoming.get("salience_score")))
    if tier:
        normalized["quality_tier"] = tier
        normalized["capture_decision"] = "reject" if tier == "rejected" else "accept"
    capture = str(incoming.get("capture_decision") or "").strip().lower()
    if capture in {"accept", "reject"}:
        normalized["capture_decision"] = capture
    return normalized


def _score_from_legacy_quality(quality: Mapping[str, object]) -> dict[str, object]:
    salience = _clamp_float(quality.get("salience_score", quality.get("importance")))
    confidence = _clamp_float(quality.get("confidence"))
    freshness = _clamp_float(quality.get("freshness"))
    reuse = _clamp_float(quality.get("reuse_potential"))
    values = [value for value in (salience, confidence, freshness, reuse) if value is not None]
    final_score = salience if salience is not None else (sum(values) / len(values) if values else 0.0)
    tier = _normalize_tier(quality.get("quality_tier"), final_score)
    components: dict[str, object] = {}
    for name, value in (
        ("salience", salience),
        ("confidence", confidence),
        ("freshness", freshness),
        ("reuse", reuse),
    ):
        if value is None:
            continue
        components[name] = {
            "name": name,
            "value": round(value, 3),
            "weight": 0.0,
            "evidence": {"source": "legacy.meta.quality"},
        }
    return {
        "schema_version": "memory_score.v1",
        "final_score": round(final_score, 3),
        "tier": tier,
        "components": components,
        "labels": normalize_scoring_labels(
            quality.get("labels"),
            tier=tier,
            confidence=confidence,
            salience=salience,
            freshness=freshness,
            reuse=reuse,
        ),
        "explanation": {"compatibility_source": "legacy.meta.quality"},
        "provenance": {
            "agent": "eibrain.memory.compat",
            "activity": "memory.compat_backfill",
            "source": "legacy.meta.quality",
        },
    }


def _legacy_quality_from_score(score: Mapping[str, object]) -> dict[str, object]:
    payload = _mapping(score)
    tier = _normalize_tier(payload.get("tier"), _clamp_float(payload.get("final_score")))
    components = _normalize_components(payload.get("components"))
    salience = _component_value(components, "salience")
    confidence = _component_value(components, "confidence")
    freshness = _component_value(components, "freshness")
    reuse = _component_value(components, "reuse")
    if salience is None:
        salience = _clamp_float(payload.get("final_score"))
    quality: dict[str, object] = {
        "quality_tier": tier,
        "capture_decision": "reject" if tier == "rejected" else "accept",
    }
    if salience is not None:
        quality["importance"] = salience
        quality["salience_score"] = salience
    if confidence is not None:
        quality["confidence"] = confidence
    if freshness is not None:
        quality["freshness"] = freshness
    if reuse is not None:
        quality["reuse_potential"] = reuse
    return quality


def _normalize_components(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, dict[str, object]] = {}
    for name, component in value.items():
        payload = _mapping(component)
        if not payload:
            continue
        component_name = str(payload.get("name") or name or "").strip().lower()
        if not component_name:
            continue
        numeric = _clamp_float(payload.get("value"))
        weight = _clamp_float(payload.get("weight"))
        normalized[component_name] = {
            "name": component_name,
            "value": round(numeric if numeric is not None else 0.0, 3),
            "weight": round(weight if weight is not None else 0.0, 3),
            "evidence": _mapping(payload.get("evidence")),
        }
    return normalized


def _component_value(components: Mapping[str, Mapping[str, object]], name: str) -> float | None:
    payload = _mapping(components.get(name))
    if payload:
        return _clamp_float(payload.get("value"))
    fallback = _mapping(components.get(f"{name}_score"))
    if fallback:
        return _clamp_float(fallback.get("value"))
    return None


def _normalize_tier(value: object, final_score: float | None) -> str:
    tier = str(value or "").strip().lower()
    if tier in _TIERS:
        return tier
    if final_score is None:
        return "candidate"
    if final_score < 0.25:
        return "rejected"
    if final_score < 0.5:
        return "candidate"
    if final_score < 0.75:
        return "confirmed"
    return "core"


def _canonical_label(value: object) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        return ""
    if token in _TIERS:
        return f"lifecycle.{token}"
    if "." not in token:
        for namespace in _KNOWN_NAMESPACES:
            prefix = f"{namespace}_"
            if token.startswith(prefix):
                suffix = token[len(prefix) :]
                return f"{namespace}.{suffix}" if suffix else ""
        return ""
    namespace, _, suffix = token.partition(".")
    if namespace not in _KNOWN_NAMESPACES or not suffix:
        return ""
    return f"{namespace}.{suffix}"


def _level_label(namespace: str, value: float | None) -> str:
    if value is None:
        return ""
    if value < 0.34:
        level = "low"
    elif value < 0.67:
        level = "medium"
    else:
        level = "high"
    return f"{namespace}.{level}"


def _freshness_label(value: float | None) -> str:
    if value is None:
        return ""
    if value < 0.34:
        return "freshness.stale"
    if value < 0.67:
        return "freshness.stable"
    return "freshness.recent"


def _merge_nested(left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, object]:
    merged = dict(left)
    for key, value in right.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_nested(_mapping(merged[key]), _mapping(value))
        else:
            merged[key] = value
    return merged


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clamp_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return round(number, 3)
