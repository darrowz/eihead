"""Durable local registry for visual identity embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REGISTRY_VERSION = 1


@dataclass(frozen=True, slots=True)
class FaceEmbeddingRecord:
    embedding_id: str
    vector: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "embedding_id": self.embedding_id,
            "vector": list(self.vector),
        }


@dataclass(frozen=True, slots=True)
class IdentityPerson:
    person_id: str
    display_name: str
    embeddings: tuple[FaceEmbeddingRecord, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "display_name": self.display_name,
            "embeddings": [embedding.as_dict() for embedding in self.embeddings],
        }


@dataclass(frozen=True, slots=True)
class IdentityMatch:
    person: IdentityPerson
    embedding: FaceEmbeddingRecord
    similarity: float


class JsonIdentityRegistry:
    """Small JSON-backed identity registry for local face embeddings."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._people: dict[str, IdentityPerson] = {}
        self._load()

    def get(self, person_id: str) -> IdentityPerson | None:
        return self._people.get(str(person_id))

    def list_people(self) -> list[IdentityPerson]:
        return [self._people[key] for key in sorted(self._people)]

    def enroll_or_update(
        self,
        *,
        person_id: str,
        display_name: str,
        embeddings: Iterable[Sequence[float]],
    ) -> IdentityPerson:
        person_id = _require_text(person_id, field_name="person_id")
        display_name = _require_text(display_name, field_name="display_name")
        existing = self._people.get(person_id)
        current_embeddings = list(existing.embeddings) if existing is not None else []
        next_index = len(current_embeddings) + 1
        for vector in embeddings:
            normalized = _coerce_vector(vector)
            current_embeddings.append(
                FaceEmbeddingRecord(
                    embedding_id=f"{person_id}:{next_index:04d}",
                    vector=normalized,
                )
            )
            next_index += 1
        person = IdentityPerson(
            person_id=person_id,
            display_name=display_name,
            embeddings=tuple(current_embeddings),
        )
        self._people[person_id] = person
        self.save()
        return person

    def best_match(self, vector: Sequence[float]) -> IdentityMatch | None:
        query = _coerce_vector(vector)
        best: IdentityMatch | None = None
        for person in self.list_people():
            for embedding in person.embeddings:
                similarity = cosine_similarity(query, embedding.vector)
                if best is None or similarity > best.similarity:
                    best = IdentityMatch(person=person, embedding=embedding, similarity=similarity)
        return best

    def save(self) -> None:
        payload = {
            "version": REGISTRY_VERSION,
            "people": [person.as_dict() for person in self.list_people()],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    def _load(self) -> None:
        if not self.path.exists():
            self._people = {}
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("version") != REGISTRY_VERSION:
            raise ValueError(f"unsupported identity registry version: {payload.get('version')!r}")
        people: dict[str, IdentityPerson] = {}
        for raw_person in payload.get("people", []):
            person = _person_from_dict(raw_person)
            people[person.person_id] = person
        self._people = people


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    first_vector = _coerce_vector(first)
    second_vector = _coerce_vector(second)
    if len(first_vector) != len(second_vector):
        raise ValueError("embedding vectors must have the same dimension")
    first_norm = sum(value * value for value in first_vector) ** 0.5
    second_norm = sum(value * value for value in second_vector) ** 0.5
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return sum(left * right for left, right in zip(first_vector, second_vector)) / (first_norm * second_norm)


def _person_from_dict(payload: Mapping[str, Any]) -> IdentityPerson:
    person_id = _require_text(payload.get("person_id"), field_name="person_id")
    display_name = _require_text(payload.get("display_name"), field_name="display_name")
    embeddings = tuple(
        FaceEmbeddingRecord(
            embedding_id=_require_text(item.get("embedding_id"), field_name="embedding_id"),
            vector=_coerce_vector(item.get("vector", ())),
        )
        for item in payload.get("embeddings", [])
        if isinstance(item, Mapping)
    )
    return IdentityPerson(person_id=person_id, display_name=display_name, embeddings=embeddings)


def _require_text(value: object, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _coerce_vector(vector: Sequence[float] | object) -> tuple[float, ...]:
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise ValueError("embedding vector must be a sequence of numbers")
    coerced = tuple(float(value) for value in vector)
    if not coerced:
        raise ValueError("embedding vector must not be empty")
    return coerced
