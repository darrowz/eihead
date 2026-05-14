"""Adapter for the transitional legacy body runtime.

This module is the only eihead runtime layer that should know how to load the
old ``apps.body_runtime`` package or translate local eihead actions back to
``eibrain.protocol`` action classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from eihead.protocol import MoveHeadAction, PlaySpeechAction, StopSpeechAction

DEFAULT_BODY_RUNTIME_DELEGATE = "apps.body_runtime.BodyRuntimeApp"
BodyRuntimeFactory = Callable[[str], Any]


def default_body_runtime_factory(config_path: str) -> Any:
    from apps.body_runtime.app import BodyRuntimeApp

    return BodyRuntimeApp.from_config_path(config_path)


@dataclass(frozen=True, slots=True)
class LegacyBodyRuntimeAdapter:
    """Boundary around the old body runtime used during the eihead split."""

    body_runtime_factory: BodyRuntimeFactory = default_body_runtime_factory
    delegate_name: str = DEFAULT_BODY_RUNTIME_DELEGATE

    def load_runtime(self, config_path: str) -> Any:
        body_config_path = self.body_runtime_config_path(config_path)
        return self.body_runtime_factory(body_config_path)

    def body_runtime_config_path(self, config_path: str) -> str:
        """Resolve the legacy body config when the user passes an eihead config."""

        if not Path(config_path).name.startswith("eihead"):
            return config_path
        try:
            from eihead.runtime.config import EiheadConfigError, load_eihead_config

            config = load_eihead_config(config_path)
        except (EiheadConfigError, OSError):
            return config_path
        return config.legacy.body_runtime_config_path or config_path

    def compat_action(self, action: Any) -> Any | None:
        return to_legacy_eibrain_action(action)


def to_legacy_eibrain_action(action: Any) -> Any | None:
    """Convert local eihead actions for the old in-repo body runtime.

    The standalone eihead package must not require eibrain. During the
    transitional split, however, the existing body organs still check action
    classes with ``isinstance(eibrain.protocol.*)``. Keep this import optional
    so standalone exports run without eibrain while the current monorepo
    delegate remains compatible.
    """

    try:
        from eibrain.protocol.actions import MoveHeadAction as LegacyMoveHeadAction
        from eibrain.protocol.actions import PlaySpeechAction as LegacyPlaySpeechAction
        from eibrain.protocol.actions import StopSpeechAction as LegacyStopSpeechAction
    except ImportError:
        return None

    if isinstance(action, PlaySpeechAction):
        return LegacyPlaySpeechAction(
            ts=action.ts,
            source=action.source,
            text=action.text,
            session_id=action.session_id,
            actor_id=action.actor_id,
            target_id=action.target_id,
        )
    if isinstance(action, MoveHeadAction):
        return LegacyMoveHeadAction(
            ts=action.ts,
            source=action.source,
            session_id=action.session_id,
            actor_id=action.actor_id,
            target_id=action.target_id,
            target_name=action.target_name,
            target_x=action.target_x,
            target_angle=action.target_angle,
        )
    if isinstance(action, StopSpeechAction):
        return LegacyStopSpeechAction(
            ts=action.ts,
            source=action.source,
            session_id=action.session_id,
            actor_id=action.actor_id,
            target_id=action.target_id,
        )
    return None


def run_body_hardware_verifier() -> None:
    from apps.body_runtime.verify_hardware import main as body_verify_main

    body_verify_main()
