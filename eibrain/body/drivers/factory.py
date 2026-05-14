"""Driver factory."""

from __future__ import annotations

from eibrain.infra.config import DriverConfig

from .command import CommandDriver
from .http import HttpDriver
from .noop import NoopDriver


def build_driver(config: DriverConfig):
    if config.kind == "command":
        return CommandDriver(config)
    if config.kind == "http":
        return HttpDriver(config)
    return NoopDriver()
