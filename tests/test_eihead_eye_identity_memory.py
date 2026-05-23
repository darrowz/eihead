from __future__ import annotations

import json
from urllib.error import URLError

from eihead.eye.identity_memory import EimemoryIdentityConfig, IdentityMemoryAdapter


class FakeResponse:
    def __init__(self, payload: object, *, status: int = 200) -> None:
        self.status = status
        self._payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode("utf-8")


class CapturingUrlopen:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.requests: list[object] = []
        self.timeouts: list[float] = []

    def __call__(self, request: object, *, timeout: float) -> FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class DictBackedObservation:
    def to_dict(self) -> dict[str, object]:
        return {
            "known": True,
            "person_id": "person-bob",
            "display_name": "Bob",
            "confidence": 0.88,
        }


class AsDictBackedObservation:
    def as_dict(self) -> dict[str, object]:
        return {
            "known": True,
            "person_id": "person-chen",
            "display_name": "Chen",
            "confidence": 0.91,
        }


def test_identity_memory_adapter_posts_known_person_sighting_payload() -> None:
    urlopen = CapturingUrlopen(FakeResponse({"ok": True, "result": {"record_id": "mem-1"}}))
    adapter = IdentityMemoryAdapter(
        EimemoryIdentityConfig(
            enabled=True,
            endpoint_url="http://honxin:8091/",
            timeout_s=2.5,
            scope={"agent_id": "honxin", "workspace_id": "honjia", "user_id": "darrow"},
        ),
        urlopen=urlopen,
    )

    result = adapter.ingest_identity_observation(
        {
            "known": True,
            "person_id": "person-alice",
            "display_name": "Alice",
            "confidence": 0.9234567,
            "frame_id": "frame-001",
            "observed_at": "2026-05-22T10:00:00.000+08:00",
            "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.8},
            "crop": {"uri": "file:///tmp/crops/alice.jpg", "sha256": "abc123"},
            "track_id": "track-7",
            "source": "eihead.eye.identity",
        }
    )

    assert result == {"status": "sent", "memory_id": "mem-1"}
    assert urlopen.timeouts == [2.5]

    request = urlopen.requests[0]
    assert request.full_url == "http://honxin:8091/"
    assert request.headers["Content-type"] == "application/json"
    assert "Authorization" not in request.headers

    body = json.loads(request.data.decode("utf-8"))
    assert body["method"] == "memory.ingest"
    params = body["params"]
    assert params["title"] == "Known person sighting: Alice"
    assert "Alice" in params["text"]
    assert params["memory_type"] == "visual_identity_event"
    assert params["source"] == "eihead.eye.identity"
    assert params["scope"] == {"agent_id": "honxin", "workspace_id": "honjia", "user_id": "darrow"}
    assert params["organ"] == "eye"
    assert params["modality"] == "vision"
    assert params["content"]["person_id"] == "person-alice"
    assert params["content"]["known"] is True
    assert params["content"]["bbox"] == {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.8}
    assert params["evidence"] == [
        {
            "type": "crop",
            "uri": "file:///tmp/crops/alice.jpg",
            "sha256": "abc123",
            "frame_id": "frame-001",
        },
        {
            "type": "bbox",
            "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.8},
            "frame_id": "frame-001",
        },
    ]
    assert params["tags"] == ["visual_identity", "known_person", "person:person-alice"]


def test_identity_memory_adapter_accepts_observation_objects_with_to_dict() -> None:
    urlopen = CapturingUrlopen(FakeResponse({"result": {"memory_id": "mem-2"}}))
    adapter = IdentityMemoryAdapter(
        EimemoryIdentityConfig(enabled=True, endpoint_url="http://honxin:8091/"),
        urlopen=urlopen,
    )

    result = adapter.ingest_identity_observation(DictBackedObservation())

    assert result == {"status": "sent", "memory_id": "mem-2"}
    body = json.loads(urlopen.requests[0].data.decode("utf-8"))
    assert body["params"]["content"]["person_id"] == "person-bob"


def test_identity_memory_adapter_accepts_identity_observations_with_as_dict() -> None:
    urlopen = CapturingUrlopen(FakeResponse({"ok": True, "result": {"record_id": "mem-3"}}))
    adapter = IdentityMemoryAdapter(
        EimemoryIdentityConfig(enabled=True, endpoint_url="http://honxin:8091/"),
        urlopen=urlopen,
    )

    result = adapter.ingest_identity_observation(AsDictBackedObservation())

    assert result == {"status": "sent", "memory_id": "mem-3"}
    body = json.loads(urlopen.requests[0].data.decode("utf-8"))
    assert body["params"]["content"]["person_id"] == "person-chen"


def test_identity_memory_adapter_throttles_repeated_person_sightings() -> None:
    clock_value = 100.0

    def clock() -> float:
        return clock_value

    urlopen = CapturingUrlopen(FakeResponse({"ok": True, "result": {"record_id": "mem-1"}}))
    adapter = IdentityMemoryAdapter(
        EimemoryIdentityConfig(enabled=True, endpoint_url="http://honxin:8091/", min_interval_s=60.0),
        urlopen=urlopen,
        clock=clock,
    )

    first = adapter.ingest_identity_observation({"known": True, "person_id": "person-darrow", "display_name": "Darrow"})
    second = adapter.ingest_identity_observation({"known": True, "person_id": "person-darrow", "display_name": "Darrow"})
    clock_value = 161.0
    third = adapter.ingest_identity_observation({"known": True, "person_id": "person-darrow", "display_name": "Darrow"})

    assert first == {"status": "sent", "memory_id": "mem-1"}
    assert second["status"] == "skipped"
    assert second["reason"] == "recently_sent"
    assert third == {"status": "sent", "memory_id": "mem-1"}
    assert len(urlopen.requests) == 2


def test_identity_memory_adapter_skips_unknown_person_without_http() -> None:
    urlopen = CapturingUrlopen(FakeResponse({"result": {"memory_id": "mem-1"}}))
    adapter = IdentityMemoryAdapter(
        EimemoryIdentityConfig(enabled=True, endpoint_url="http://honxin:8091/"),
        urlopen=urlopen,
    )

    result = adapter.ingest_identity_observation({"known": False, "label": "unknown_person", "confidence": 0.66})

    assert result == {"status": "skipped", "reason": "unknown_person"}
    assert urlopen.requests == []


def test_identity_memory_adapter_disabled_and_unconfigured_do_not_call_http() -> None:
    urlopen = CapturingUrlopen(FakeResponse({"result": {"memory_id": "mem-1"}}))
    disabled = IdentityMemoryAdapter(EimemoryIdentityConfig(enabled=False, endpoint_url="http://honxin:8091/"), urlopen=urlopen)
    unconfigured = IdentityMemoryAdapter(EimemoryIdentityConfig(enabled=True, endpoint_url=""), urlopen=urlopen)

    assert disabled.ingest_identity_observation({"known": True, "person_id": "p1"}) == {
        "status": "skipped",
        "reason": "disabled",
    }
    assert unconfigured.ingest_identity_observation({"known": True, "person_id": "p1"}) == {
        "status": "skipped",
        "reason": "endpoint_unconfigured",
    }
    assert urlopen.requests == []


def test_eimemory_identity_config_from_mapping_keeps_endpoint_explicit() -> None:
    config = EimemoryIdentityConfig.from_mapping(
        {
            "enabled": "true",
            "endpoint": "http://honxin:8091/",
            "timeoutS": "3.25",
            "scope": {"agent_id": "honxin", "workspace_id": "honjia", "empty": ""},
        }
    )
    default_config = EimemoryIdentityConfig.from_mapping({})

    assert config.enabled is True
    assert config.endpoint_url == "http://honxin:8091/"
    assert config.timeout_s == 3.25
    assert config.scope == {"agent_id": "honxin", "workspace_id": "honjia"}
    assert default_config.enabled is False
    assert default_config.endpoint_url == ""


def test_identity_memory_adapter_reports_unavailable_and_invalid_responses() -> None:
    unavailable = IdentityMemoryAdapter(
        EimemoryIdentityConfig(enabled=True, endpoint_url="http://honxin:8091/"),
        urlopen=CapturingUrlopen(URLError("connection refused")),
    )
    invalid_json = IdentityMemoryAdapter(
        EimemoryIdentityConfig(enabled=True, endpoint_url="http://honxin:8091/"),
        urlopen=CapturingUrlopen(FakeResponse(b"not-json")),
    )
    rpc_error = IdentityMemoryAdapter(
        EimemoryIdentityConfig(enabled=True, endpoint_url="http://honxin:8091/"),
        urlopen=CapturingUrlopen(FakeResponse({"error": {"message": "bad request"}})),
    )
    ok_false = IdentityMemoryAdapter(
        EimemoryIdentityConfig(enabled=True, endpoint_url="http://honxin:8091/"),
        urlopen=CapturingUrlopen(FakeResponse({"ok": False})),
    )

    assert unavailable.ingest_identity_observation({"known": True, "person_id": "p1"}) == {
        "status": "unavailable",
        "reason": "url_error",
        "detail": "<urlopen error connection refused>",
    }
    assert invalid_json.ingest_identity_observation({"known": True, "person_id": "p1"}) == {
        "status": "unavailable",
        "reason": "invalid_response",
    }
    assert rpc_error.ingest_identity_observation({"known": True, "person_id": "p1"}) == {
        "status": "unavailable",
        "reason": "rpc_error",
        "detail": "bad request",
    }
    assert ok_false.ingest_identity_observation({"known": True, "person_id": "p1"}) == {
        "status": "unavailable",
        "reason": "rpc_error",
        "detail": "ok_false",
    }
