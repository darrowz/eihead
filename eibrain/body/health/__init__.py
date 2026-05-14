"""Body health and degradation helpers."""

from .capability_matrix import CapabilityMatrix
from .degradation_manager import DegradationManager, DegradationResult
from .fallback_policy import FallbackPolicy
from .organ_health import OrganHealth, SubfunctionHealth

__all__ = [
    "CapabilityMatrix",
    "DegradationManager",
    "DegradationResult",
    "FallbackPolicy",
    "OrganHealth",
    "SubfunctionHealth",
]
