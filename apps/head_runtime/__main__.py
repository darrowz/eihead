"""Module entrypoint for ``python -m apps.head_runtime``."""

from __future__ import annotations

from eihead.runtime.cli import main, verify_hardware_main

__all__ = ["main", "verify_hardware_main"]


if __name__ == "__main__":
    raise SystemExit(main())
