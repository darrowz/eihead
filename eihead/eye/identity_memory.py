"""Eimemory RPC adapter for visual identity sightings."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable, Mapping
from urllib import error as urlerror, request as urlrequest


@dataclass(frozen=True, slots=True)
class EimemoryIdentityConfig:
    enabled: bool = False
    endpoint_url: str = ""
    timeout_s: float = 5.0
    min_interval_s: float = 60.0
    scope: Mapping[str, str] = field(default_factory=dict)
    source: str = "eihead.eye.visual_identity"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "EimemoryIdentityConfig":
        payload = dict(raw or {})
        endpoint = payload.get("endpoint_url", payload.get("endpoint", ""))
        timeout = payload.get("timeout_s", payload.get("timeoutS", 5.0))
        min_interval = payload.get(
            "min_interval_s",
            payload.get("minIntervalS", payload.get("sighting_min_interval_s", payload.get("sightingMinIntervalS", 60.0))),
        )
        return cls(
            enabled=_truthy(payload.get("enabled", False)),
            endpoint_url=str(endpoint or ""),
            timeout_s=_safe_float(timeout, 5.0),
            min_interval_s=max(0.0, _safe_float(min_interval, 60.0)),
            scope=_string_mapping(payload.get("scope")),
            source=str(payload.get("source") or "eihead.eye.visual_identity"),
        )


class IdentityMemoryAdapter:
    """Format known-person identity observations for eimemory ``memory.ingest``."""

    def __init__(
        self,
        config: EimemoryIdentityConfig,
        *,
        urlopen: Any | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.urlopen = urlopen or urlrequest.urlopen
        self._clock = clock
        self._last_sent_by_person: dict[str, float] = {}

    def ingest_identity_observation(self, observation: Any) -> dict[str, Any]:
        observation_payload = _observation_mapping(observation)
        if not self.config.enabled:
            return {"status": "skipped", "reason": "disabled"}
        if not self.config.endpoint_url.strip():
            return {"status": "skipped", "reason": "endpoint_unconfigured"}
        if not _known_person(observation_payload):
            return {"status": "skipped", "reason": "unknown_person"}
        throttle_key = _throttle_key(observation_payload)
        if throttle_key and self.config.min_interval_s > 0:
            now_ts = float(self._clock())
            last_sent_at = self._last_sent_by_person.get(throttle_key)
            if last_sent_at is not None and now_ts - last_sent_at < self.config.min_interval_s:
                return {
                    "status": "skipped",
                    "reason": "recently_sent",
                    "last_sent_at_ts": round(last_sent_at, 6),
                    "next_allowed_at_ts": round(last_sent_at + self.config.min_interval_s, 6),
                }

        payload = {"method": "memory.ingest", "params": self.build_memory_params(observation_payload)}
        try:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            return {"status": "unavailable", "reason": "invalid_payload"}

        req = urlrequest.Request(
            self.config.endpoint_url.strip(),
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with self.urlopen(req, timeout=float(self.config.timeout_s)) as response:
                response_body = response.read()
        except urlerror.URLError as exc:
            return {"status": "unavailable", "reason": "url_error", "detail": str(exc)}
        except TimeoutError as exc:
            return {"status": "unavailable", "reason": "timeout", "detail": str(exc)}
        except OSError as exc:
            return {"status": "unavailable", "reason": "connection_error", "detail": str(exc)}

        try:
            decoded = json.loads(response_body.decode("utf-8") if isinstance(response_body, bytes) else response_body)
        except (AttributeError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
            return {"status": "unavailable", "reason": "invalid_response"}
        if not isinstance(decoded, Mapping):
            return {"status": "unavailable", "reason": "invalid_response"}
        if "error" in decoded:
            return {"status": "unavailable", "reason": "rpc_error", "detail": _rpc_error_message(decoded["error"])}
        if decoded.get("ok") is False:
            return {"status": "unavailable", "reason": "rpc_error", "detail": str(decoded.get("error") or "ok_false")}

        result = decoded.get("result")
        memory_id = (
            result.get("memory_id") or result.get("record_id")
            if isinstance(result, Mapping)
            else decoded.get("memory_id") or decoded.get("record_id")
        )
        sent: dict[str, Any] = {"status": "sent"}
        if memory_id:
            sent["memory_id"] = str(memory_id)
        if throttle_key and self.config.min_interval_s > 0:
            self._last_sent_by_person[throttle_key] = float(self._clock())
        return sent

    def build_memory_params(self, observation: Any) -> dict[str, Any]:
        observation = _observation_mapping(observation)
        person_id = _first_text(observation, "person_id", "personId", "identity_id", "identityId")
        display_name = _first_text(observation, "display_name", "displayName", "name", "person_name", "personName")
        label = display_name or person_id or "known person"
        confidence = _safe_float(observation.get("confidence", observation.get("score")), 0.0)
        frame_id = _first_text(observation, "frame_id", "frameId", "last_frame_id")
        observed_at = _first_text(observation, "observed_at", "observedAt", "captured_at", "capturedAt")
        source = _first_text(observation, "source") or self.config.source
        bbox = _bbox(observation.get("bbox"))
        crop = _crop(observation)

        content: dict[str, Any] = {
            "event_type": "known_person_sighting",
            "known": True,
            "person_id": person_id,
            "display_name": display_name,
            "confidence": round(confidence, 6),
            "frame_id": frame_id,
            "observed_at": observed_at,
            "track_id": _first_text(observation, "track_id", "trackId"),
        }
        if bbox:
            content["bbox"] = bbox
        if crop:
            content["crop"] = dict(crop)
        content = {key: value for key, value in content.items() if value not in ("", None, {}, [])}

        evidence = _evidence(frame_id=frame_id, bbox=bbox, crop=crop)
        tags = ["visual_identity", "known_person"]
        if person_id:
            tags.append(f"person:{person_id}")

        return {
            "text": _sighting_text(label=label, person_id=person_id, confidence=confidence, frame_id=frame_id),
            "title": f"Known person sighting: {label}",
            "memory_type": "visual_identity_event",
            "source": source,
            "scope": dict(self.config.scope),
            "organ": "eye",
            "modality": "vision",
            "content": content,
            "meta": {
                "schema": "eihead.eye.visual_identity.memory_ingest.v1",
                "adapter": "eihead.eye.identity_memory",
            },
            "tags": tags,
            "evidence": evidence,
            "links": _links(crop),
        }


def _known_person(observation: Mapping[str, Any]) -> bool:
    known = observation.get("known", observation.get("is_known"))
    if known is not None:
        return _truthy(known)
    label = _first_text(observation, "label", "identity_status", "identityStatus").lower()
    if "unknown" in label:
        return False
    return bool(_first_text(observation, "person_id", "personId", "display_name", "displayName", "name"))


def _throttle_key(observation: Mapping[str, Any]) -> str:
    return _first_text(observation, "person_id", "personId", "display_name", "displayName", "name")


def _observation_mapping(observation: Any) -> dict[str, Any]:
    if isinstance(observation, Mapping):
        return dict(observation)
    for method_name in ("to_dict", "as_dict"):
        converter = getattr(observation, method_name, None)
        if not callable(converter):
            continue
        payload = converter()
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _sighting_text(*, label: str, person_id: str, confidence: float, frame_id: str) -> str:
    parts = [f"eihead saw {label}"]
    if person_id and person_id != label:
        parts.append(f"(person_id {person_id})")
    if confidence:
        parts.append(f"with confidence {round(confidence, 4):g}")
    if frame_id:
        parts.append(f"in frame {frame_id}")
    return " ".join(parts) + "."


def _evidence(*, frame_id: str, bbox: dict[str, float], crop: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    if crop:
        crop_evidence = {"type": "crop", **crop}
        if frame_id:
            crop_evidence["frame_id"] = frame_id
        evidence.append(crop_evidence)
    if bbox:
        bbox_evidence: dict[str, Any] = {"type": "bbox", "bbox": bbox}
        if frame_id:
            bbox_evidence["frame_id"] = frame_id
        evidence.append(bbox_evidence)
    return evidence


def _links(crop: Mapping[str, Any]) -> list[dict[str, str]]:
    uri = str(crop.get("uri") or "").strip() if isinstance(crop, Mapping) else ""
    if uri.startswith("http://") or uri.startswith("https://"):
        return [{"rel": "crop", "href": uri}]
    return []


def _crop(observation: Mapping[str, Any]) -> dict[str, Any]:
    raw = observation.get("crop", observation.get("crop_evidence"))
    if isinstance(raw, Mapping):
        allowed = {}
        for key in ("uri", "path", "sha256", "media_type", "width", "height"):
            value = raw.get(key)
            if value not in (None, "", {}, []):
                allowed[key] = value
        return allowed
    crop_uri = _first_text(observation, "crop_uri", "cropUri", "crop_path", "cropPath")
    return {"uri": crop_uri} if crop_uri else {}


def _bbox(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    keys = ("x_min", "y_min", "x_max", "y_max")
    aliases = {
        "x_min": ("x_min", "xmin", "left"),
        "y_min": ("y_min", "ymin", "top"),
        "x_max": ("x_max", "xmax", "right"),
        "y_max": ("y_max", "ymax", "bottom"),
    }
    bbox: dict[str, float] = {}
    for key in keys:
        for alias in aliases[key]:
            if alias not in raw:
                continue
            bbox[key] = round(_safe_float(raw.get(alias), 0.0), 6)
            break
    return bbox if len(bbox) == 4 else {}


def _rpc_error_message(error: Any) -> str:
    if isinstance(error, Mapping):
        return str(error.get("message") or error.get("code") or error)
    return str(error)


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _string_mapping(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items() if value not in (None, "")}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "known"}
    return bool(value)


__all__ = ["EimemoryIdentityConfig", "IdentityMemoryAdapter"]
