"""Local visual identity registry and matching contracts."""

from .embedding import OnnxFaceEmbeddingProvider
from .matcher import (
    FaceEmbeddingProvider,
    FaceEvidence,
    FaceIdentityMatcher,
    IdentityObservation,
    StaticFaceEmbeddingProvider,
    UnavailableFaceEmbeddingProvider,
)
from .registry import FaceEmbeddingRecord, IdentityPerson, JsonIdentityRegistry

__all__ = [
    "FaceEmbeddingProvider",
    "FaceEmbeddingRecord",
    "FaceEvidence",
    "FaceIdentityMatcher",
    "IdentityObservation",
    "IdentityPerson",
    "JsonIdentityRegistry",
    "OnnxFaceEmbeddingProvider",
    "StaticFaceEmbeddingProvider",
    "UnavailableFaceEmbeddingProvider",
]
