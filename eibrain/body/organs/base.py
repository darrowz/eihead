"""Base organ contract at the embodied execution boundary.

This module defines the minimal contract for an ``Organ`` from the Body layer.
It intentionally stays close to runtime primitives: build drivers, probe
subfunctions, and report organ health only.
"""

from __future__ import annotations

from collections.abc import Iterable

from eibrain.body.drivers import build_driver
from eibrain.body.drivers.base import DriverResult
from eibrain.body.health.organ_health import OrganHealth
from eibrain.body.health.organ_health import SubfunctionHealth
from eibrain.infra.config import OrganConfig, SubfunctionConfig


class BaseOrgan:
    """Embodied contract shared by all concrete organs.

    Responsibility boundaries:
    - Own organ-level configuration and driver wiring.
    - Expose common heartbeat entry points for this organ's subfunctions.
    - Normalize raw driver health into the shared health vocabulary.

    Domain-specific behavior, action parsing, and provider tuning belong to
    concrete organ implementations, not this base class.
    """

    name = "organ"
    subfunction_names: tuple[str, ...] = ()

    def __init__(self, *, config: OrganConfig | None = None) -> None:
        self.config = config or self.default_config()
        self.drivers: dict[str, object] = {
            name: build_driver(subfunction.driver)
            for name, subfunction in self.config.subfunctions.items()
        }

    @classmethod
    def default_config(cls) -> OrganConfig:
        return OrganConfig(
            enabled=True,
            subfunctions={name: SubfunctionConfig() for name in cls.subfunction_names},
        )

    def heartbeat(self) -> OrganHealth:
        # Boundary: Heartbeat orchestration for this organ (execution layer).
        subfunctions = self._collect_subfunction_health()
        health = self._derive_health([state.health for state in subfunctions.values()])
        return OrganHealth(organ=self.name, health=health, subfunctions=subfunctions)

    def supports_action(self, action) -> bool:
        return False

    def handle_action(self, action):
        return None

    def _collect_subfunction_health(self) -> dict[str, SubfunctionHealth]:
        return {
            name: self._subfunction_health(name)
            for name in self.subfunction_names
        }

    def _driver_kind(self, name: str) -> str:
        config = self.config.subfunctions.get(name)
        if config is None:
            return "noop"
        return str(config.driver.kind)

    @staticmethod
    def _derive_health(statuses: Iterable[str]) -> str:
        status_list = list(statuses)
        if not status_list:
            return "unavailable"
        if all(status == "healthy" for status in status_list):
            return "healthy"
        if any(status == "healthy" for status in status_list):
            return "degraded"
        if any(status == "degraded" for status in status_list):
            return "degraded"
        return "unavailable"

    def _subfunction_health(self, name: str) -> SubfunctionHealth:
        driver = self.drivers.get(name)
        if driver is None:
            return SubfunctionHealth(name=name, health="unavailable")
        heartbeat = driver.heartbeat()
        if isinstance(heartbeat, DriverResult):
            return SubfunctionHealth(
                name=name,
                health=self._normalize_status(heartbeat.status),
                details=dict(heartbeat.details),
            )
        return SubfunctionHealth(name=name, health=self._normalize_status(str(heartbeat)))

    @staticmethod
    def _normalize_status(status: str) -> str:
        normalized = str(status or "").strip().lower()
        if not normalized:
            return "unavailable"
        if normalized in {"ok", "healthy"}:
            return "healthy"
        if normalized == "unavailable" or normalized.startswith("missing_") or normalized.endswith("_unavailable"):
            return "unavailable"
        if any(token in normalized for token in ("error", "fail", "timeout", "degraded")):
            return "degraded"
        if normalized.startswith("waiting_") or normalized.endswith("_skipped"):
            return "healthy"
        return "healthy"
