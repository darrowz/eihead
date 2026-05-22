from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eihead.eye.identity import (
    FaceEvidence,
    FaceIdentityMatcher,
    JsonIdentityRegistry,
    OnnxFaceEmbeddingProvider,
    StaticFaceEmbeddingProvider,
    UnavailableFaceEmbeddingProvider,
)


def test_json_identity_registry_enrolls_and_updates_named_people(tmp_path) -> None:
    registry_path = tmp_path / "identity-registry.json"
    registry = JsonIdentityRegistry(registry_path)

    person = registry.enroll_or_update(
        person_id="person-honjia",
        display_name="honjia",
        embeddings=[[1.0, 0.0, 0.0], [0.98, 0.02, 0.0]],
    )
    updated = registry.enroll_or_update(
        person_id="person-honjia",
        display_name="Honjia",
        embeddings=[[0.99, 0.01, 0.0]],
    )

    reloaded = JsonIdentityRegistry(registry_path)

    assert person.person_id == "person-honjia"
    assert updated.display_name == "Honjia"
    assert len(updated.embeddings) == 3
    assert reloaded.get("person-honjia") == updated
    assert json.loads(registry_path.read_text(encoding="utf-8")) == {
        "version": 1,
        "people": [
            {
                "person_id": "person-honjia",
                "display_name": "Honjia",
                "embeddings": [
                    {"embedding_id": "person-honjia:0001", "vector": [1.0, 0.0, 0.0]},
                    {"embedding_id": "person-honjia:0002", "vector": [0.98, 0.02, 0.0]},
                    {"embedding_id": "person-honjia:0003", "vector": [0.99, 0.01, 0.0]},
                ],
            }
        ],
    }


def test_matcher_reports_unavailable_when_no_embedding_or_provider(tmp_path) -> None:
    registry = JsonIdentityRegistry(tmp_path / "identity-registry.json")
    matcher = FaceIdentityMatcher(registry=registry, embedding_provider=UnavailableFaceEmbeddingProvider())

    observation = matcher.match(
        FaceEvidence(
            face_id="face-1",
            track_id="track-1",
            crop_path="crops/face-1.jpg",
            bbox=[10, 20, 80, 120],
        )
    )

    assert observation.known is False
    assert observation.person_id is None
    assert observation.display_name is None
    assert observation.confidence == 0.0
    assert observation.match_source == "embedding_unavailable"
    assert observation.evidence == {
        "face_id": "face-1",
        "track_id": "track-1",
        "crop_path": "crops/face-1.jpg",
        "bbox": [10.0, 20.0, 80.0, 120.0],
        "embedding_present": False,
        "provider": "unavailable",
        "reason": "face_embedding_provider_unavailable",
    }


def test_matcher_identifies_known_person_from_supplied_face_embedding(tmp_path) -> None:
    registry = JsonIdentityRegistry(tmp_path / "identity-registry.json")
    registry.enroll_or_update(
        person_id="person-honjia",
        display_name="honjia",
        embeddings=[[1.0, 0.0, 0.0], [0.96, 0.04, 0.0]],
    )
    registry.enroll_or_update(
        person_id="person-guest",
        display_name="guest",
        embeddings=[[0.0, 1.0, 0.0]],
    )
    matcher = FaceIdentityMatcher(registry=registry, threshold=0.85)

    observation = matcher.match(
        {
            "face_id": "face-2",
            "track_id": "track-2",
            "crop_path": "crops/face-2.jpg",
            "bbox": [12, 18, 82, 118],
            "embedding": [0.99, 0.01, 0.0],
        }
    )

    assert observation.as_dict() == {
        "known": True,
        "person_id": "person-honjia",
        "display_name": "honjia",
        "confidence": pytest.approx(0.9999, abs=0.0001),
        "match_source": "provided_embedding",
        "evidence": {
            "face_id": "face-2",
            "track_id": "track-2",
            "crop_path": "crops/face-2.jpg",
            "bbox": [12.0, 18.0, 82.0, 118.0],
            "embedding_present": True,
            "provider": None,
            "matched_embedding_id": "person-honjia:0001",
            "similarity": pytest.approx(0.9999, abs=0.0001),
            "threshold": 0.85,
        },
    }


def test_matcher_emits_unknown_when_embedding_is_below_threshold(tmp_path) -> None:
    registry = JsonIdentityRegistry(tmp_path / "identity-registry.json")
    registry.enroll_or_update(person_id="person-honjia", display_name="honjia", embeddings=[[1.0, 0.0, 0.0]])
    matcher = FaceIdentityMatcher(registry=registry, threshold=0.90)

    observation = matcher.match(
        FaceEvidence(
            face_id="face-3",
            track_id="track-3",
            crop_path="crops/face-3.jpg",
            bbox=[0, 0, 10, 10],
            embedding=[0.0, 1.0, 0.0],
        )
    )

    assert observation.known is False
    assert observation.person_id is None
    assert observation.display_name is None
    assert observation.confidence == 0.0
    assert observation.match_source == "provided_embedding"
    assert observation.evidence["matched_embedding_id"] == "person-honjia:0001"
    assert observation.evidence["similarity"] == pytest.approx(0.0)
    assert observation.evidence["threshold"] == 0.90


def test_matcher_uses_explicit_embedding_provider_when_evidence_has_no_embedding(tmp_path) -> None:
    registry = JsonIdentityRegistry(tmp_path / "identity-registry.json")
    registry.enroll_or_update(person_id="person-honjia", display_name="honjia", embeddings=[[0.0, 1.0, 0.0]])
    provider = StaticFaceEmbeddingProvider({"face-4": [0.0, 0.99, 0.01]}, provider_id="test-face-model")
    matcher = FaceIdentityMatcher(registry=registry, embedding_provider=provider, threshold=0.85)

    observation = matcher.match(
        FaceEvidence(
            face_id="face-4",
            track_id="track-4",
            crop_path="crops/face-4.jpg",
            bbox=[2, 4, 20, 40],
        )
    )

    assert observation.known is True
    assert observation.person_id == "person-honjia"
    assert observation.display_name == "honjia"
    assert observation.match_source == "provider:test-face-model"
    assert observation.evidence["embedding_present"] is True
    assert observation.evidence["provider"] == "test-face-model"


def test_onnx_face_embedding_provider_reports_unavailable_without_model(tmp_path) -> None:
    provider = OnnxFaceEmbeddingProvider(model_path=tmp_path / "missing.onnx")

    assert provider.is_available() is False
    assert provider.embed(FaceEvidence(face_id="face-no-model", crop_path=str(tmp_path / "face.jpg"))) is None


def test_onnx_face_embedding_provider_returns_l2_normalized_embedding(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    model_path = tmp_path / "face.onnx"
    crop_path = tmp_path / "face.jpg"
    model_path.write_bytes(b"onnx")
    crop_path.write_bytes(b"jpeg")
    session = _FakeOnnxSession(np.array([[3.0, 4.0]], dtype=np.float32))
    provider = OnnxFaceEmbeddingProvider(
        model_path=model_path,
        session_factory=lambda _path: session,
        image_loader=lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )

    vector = provider.embed(FaceEvidence(face_id="face-onnx", crop_path=str(crop_path)))

    assert provider.is_available() is True
    assert vector == pytest.approx((0.6, 0.8))
    assert session.requests[0]["input"] == "input"
    assert session.requests[0]["shape"] == (1, 3, 112, 112)


class _FakeOnnxSession:
    def __init__(self, output: object) -> None:
        self.output = output
        self.requests: list[dict[str, object]] = []

    def get_inputs(self) -> list[object]:
        return [SimpleNamespace(name="input")]

    def run(self, _outputs: object, inputs: dict[str, object]) -> list[object]:
        tensor = inputs["input"]
        self.requests.append({"input": "input", "shape": getattr(tensor, "shape", None)})
        return [self.output]
