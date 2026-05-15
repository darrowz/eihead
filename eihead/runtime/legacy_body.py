"""Deprecated legacy body runtime shim."""

from __future__ import annotations

from typing import Any


DEFAULT_BODY_RUNTIME_DELEGATE = "eihead.native_runtime"


class LegacyBodyRuntimeAdapter:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)
        raise RuntimeError(
            "LegacyBodyRuntimeAdapter has been removed; use eihead native providers"
        )


def run_body_hardware_verifier() -> None:
    raise RuntimeError(
        "Legacy body hardware verifier has been removed; use eihead native providers"
    )
