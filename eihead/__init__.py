"""eihead compatibility package.

The package starts as a thin wrapper around the existing body runtime so the
honjia head node can be extracted without interrupting the current deployment.
"""

from .runtime.app import HeadRuntimeApp

__all__ = ["HeadRuntimeApp"]
