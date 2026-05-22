"""Face evidence to identity observation matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .registry import JsonIdentityRegistry


@dataclass(frozen=True, slots=True)
class FaceEvidence:
    face_id: str
    track_id: Any | None = None
    crop_path: str = ""
    bbox: Sequence[float] = ()
    embedding: Sequence[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_bbox(self) -> dict[str, float] | list[float]:
        if isinstance(self.bbox, Mapping):
            return {str(key): float(value) for key, value in self.bbox.items()}
        return [float(value) for value in self.bbox]


@dataclass(frozen=True, slots=True)
class IdentityObservation:
    known: bool
    person_id: str | None
    display_name: str | None
    confidence: float
    match_source: str
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "known": self.known,
            "person_id": self.person_id,
            "display_name": self.display_name,
            "confidence": self.confidence,
            "match_source": self.match_source,
            "evidence": dict(self.evidence),
        }


class FaceEmbeddingProvider(Protocol):
    provider_id: str

    def is_available(self) -> bool:
        """Return whether a real face embedding model is configured."""

    def embed(self, evidence: FaceEvidence) -> Sequence[float] | None:
        """Return a face embedding for the evidence, or None if unavailable."""


class UnavailableFaceEmbeddingProvider:
    provider_id = "unavailable"

    def is_available(self) -> bool:
        return False

    def embed(self, evidence: FaceEvidence) -> Sequence[float] | None:
        return None


class StaticFaceEmbeddingProvider:
    """Deterministic provider for tests and explicit offline fixtures."""

    def __init__(self, embeddings_by_face_id: Mapping[str, Sequence[float]], *, provider_id: str) -> None:
        self.provider_id = str(provider_id)
        self._embeddings_by_face_id = {
            str(face_id): tuple(float(value) for value in embedding)
            for face_id, embedding in embeddings_by_face_id.items()
        }

    def is_available(self) -> bool:
        return True

    def embed(self, evidence: FaceEvidence) -> Sequence[float] | None:
        return self._embeddings_by_face_id.get(evidence.face_id)


class FaceIdentityMatcher:
    """Match face evidence against the local visual identity registry."""

    def __init__(
        self,
        *,
        registry: JsonIdentityRegistry,
        embedding_provider: FaceEmbeddingProvider | None = None,
        threshold: float = 0.85,
    ) -> None:
        self.registry = registry
        self.embedding_provider = embedding_provider or UnavailableFaceEmbeddingProvider()
        self.threshold = float(threshold)

    def match(self, evidence: FaceEvidence | Mapping[str, Any]) -> IdentityObservation:
        face_evidence = _coerce_evidence(evidence)
        vector = face_evidence.embedding
        provider_id: str | None = None
        match_source = "provided_embedding"

        if vector is None:
            provider_id = self.embedding_provider.provider_id
            if not self.embedding_provider.is_available():
                return IdentityObservation(
                    known=False,
                    person_id=None,
                    display_name=None,
                    confidence=0.0,
                    match_source="embedding_unavailable",
                    evidence={
                        **_base_evidence(face_evidence),
                        "embedding_present": False,
                        "provider": provider_id,
                        "reason": "face_embedding_provider_unavailable",
                    },
                )
            vector = self.embedding_provider.embed(face_evidence)
            match_source = f"provider:{provider_id}"

        if vector is None:
            return IdentityObservation(
                known=False,
                person_id=None,
                display_name=None,
                confidence=0.0,
                match_source="embedding_unavailable",
                evidence={
                    **_base_evidence(face_evidence),
                    "embedding_present": False,
                    "provider": provider_id,
                    "reason": "face_embedding_missing",
                },
            )

        best = self.registry.best_match(vector)
        if best is None:
            return IdentityObservation(
                known=False,
                person_id=None,
                display_name=None,
                confidence=0.0,
                match_source=match_source,
                evidence={
                    **_base_evidence(face_evidence),
                    "embedding_present": True,
                    "provider": provider_id,
                    "threshold": self.threshold,
                    "reason": "identity_registry_empty",
                },
            )

        known = best.similarity >= self.threshold
        return IdentityObservation(
            known=known,
            person_id=best.person.person_id if known else None,
            display_name=best.person.display_name if known else None,
            confidence=best.similarity if known else 0.0,
            match_source=match_source,
            evidence={
                **_base_evidence(face_evidence),
                "embedding_present": True,
                "provider": provider_id,
                "matched_embedding_id": best.embedding.embedding_id,
                "similarity": best.similarity,
                "threshold": self.threshold,
            },
        )


def _coerce_evidence(evidence: FaceEvidence | Mapping[str, Any]) -> FaceEvidence:
    if isinstance(evidence, FaceEvidence):
        return evidence
    return FaceEvidence(
        face_id=str(evidence.get("face_id", "")),
        track_id=evidence.get("track_id", evidence.get("trackId")),
        crop_path=str(evidence.get("crop_path", evidence.get("cropPath", ""))),
        bbox=evidence.get("bbox", ()),
        embedding=evidence.get("embedding"),
        metadata=dict(evidence.get("metadata", {})),
    )


def _base_evidence(evidence: FaceEvidence) -> dict[str, Any]:
    return {
        "face_id": evidence.face_id,
        "track_id": evidence.track_id,
        "crop_path": evidence.crop_path,
        "bbox": evidence.normalized_bbox(),
    }
