"""Vision model capability registry.

This module is intentionally declarative: it records model/profile capabilities
for runtime selection, but it does not import or load any model runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SUPPORTED_CAPABILITIES = frozenset(
    {
        "detection",
        "face",
        "pose",
        "segmentation",
        "depth",
        "clip_scene",
        "tracking",
    }
)
SUPPORTED_BACKENDS = frozenset({"hailo", "rpicam", "gstreamer", "opencv"})
SUPPORTED_DEVICES = frozenset({"hailo8", "hailo8l", "cpu"})
DEFAULT_PROFILE_ID = "yolov8s_h8"


@dataclass(frozen=True, slots=True)
class VisionModelProfile:
    """A selectable vision profile without any loaded model state."""

    id: str
    model_id: str
    backend: str
    device: str
    capabilities: frozenset[str]
    max_fps: float
    priority: int = 0
    degraded_from: str | None = None
    notes: str = ""
    loadable: bool = False

    def supports(self, required_capabilities: Iterable[str]) -> bool:
        return _normalize_set(required_capabilities) <= self.capabilities


@dataclass(frozen=True, slots=True)
class VisionProfileSelection:
    """Result of a profile selection attempt."""

    profile: VisionModelProfile | None
    missing_capabilities: frozenset[str]
    diagnostics: dict[str, object]
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.profile is not None and not self.missing_capabilities


_DEFAULT_PROFILES: tuple[VisionModelProfile, ...] = (
    VisionModelProfile(
        id="yolov8s_h8",
        model_id="yolov8s_h8",
        backend="gstreamer",
        device="hailo8",
        capabilities=frozenset({"detection", "tracking"}),
        max_fps=30.0,
        priority=100,
        notes="Default Hailo-8 GStreamer object detection/tracking placeholder.",
    ),
    VisionModelProfile(
        id="yolov8n_h8l",
        model_id="yolov8n_h8l",
        backend="gstreamer",
        device="hailo8l",
        capabilities=frozenset({"detection", "tracking"}),
        max_fps=15.0,
        priority=90,
        degraded_from="yolov8s_h8",
        notes="Hailo-8L downgrade for the default detection/tracking task.",
    ),
    VisionModelProfile(
        id="personface_h8l",
        model_id="personface_h8l",
        backend="hailo",
        device="hailo8l",
        capabilities=frozenset({"detection", "face", "tracking"}),
        max_fps=12.0,
        priority=80,
        degraded_from="yolov8s_h8",
        notes="Person/face detection placeholder for identity-oriented tasks.",
    ),
    VisionModelProfile(
        id="opencv_cpu_detector",
        model_id="opencv_cpu_detector",
        backend="opencv",
        device="cpu",
        capabilities=frozenset({"detection"}),
        max_fps=6.0,
        priority=30,
        notes="CPU fallback detector placeholder.",
    ),
    VisionModelProfile(
        id="opencv_pose_cpu",
        model_id="opencv_pose_cpu",
        backend="opencv",
        device="cpu",
        capabilities=frozenset({"pose"}),
        max_fps=8.0,
        priority=25,
        notes="CPU pose estimation placeholder.",
    ),
    VisionModelProfile(
        id="opencv_segmentation_cpu",
        model_id="opencv_segmentation_cpu",
        backend="opencv",
        device="cpu",
        capabilities=frozenset({"segmentation"}),
        max_fps=2.0,
        priority=20,
        notes="CPU segmentation placeholder.",
    ),
    VisionModelProfile(
        id="rpicam_depth_placeholder",
        model_id="rpicam_depth_placeholder",
        backend="rpicam",
        device="cpu",
        capabilities=frozenset({"depth"}),
        max_fps=10.0,
        priority=20,
        notes="Camera depth placeholder for later sensor-specific expansion.",
    ),
    VisionModelProfile(
        id="opencv_clip_scene_cpu",
        model_id="opencv_clip_scene_cpu",
        backend="opencv",
        device="cpu",
        capabilities=frozenset({"clip_scene"}),
        max_fps=1.0,
        priority=10,
        notes="Low-cadence CLIP scene classification placeholder.",
    ),
)


def available_profiles(profiles: Iterable[VisionModelProfile] | None = None) -> tuple[VisionModelProfile, ...]:
    """Return the registered profiles as immutable data."""

    return tuple(profiles) if profiles is not None else _DEFAULT_PROFILES


def get_profile(profile_id: str, profiles: Iterable[VisionModelProfile] | None = None) -> VisionModelProfile | None:
    """Look up a profile by id or model_id."""

    wanted = str(profile_id)
    for profile in available_profiles(profiles):
        if profile.id == wanted or profile.model_id == wanted:
            return profile
    return None


def select_profile(
    *,
    device_capabilities: Iterable[str] | str | None = None,
    target_fps: float = 10.0,
    required_capabilities: Iterable[str] | str | None = None,
    allowed_backends: Iterable[str] | str | None = None,
    profiles: Iterable[VisionModelProfile] | None = None,
) -> VisionProfileSelection:
    """Select the best profile for devices, FPS, task capabilities, and backend policy."""

    devices = _normalize_devices(device_capabilities)
    required = _normalize_set(required_capabilities, default={"detection"})
    backends = _normalize_set(allowed_backends, default=SUPPORTED_BACKENDS)
    target = max(0.0, float(target_fps))
    candidates = [
        profile
        for profile in available_profiles(profiles)
        if profile.device in devices and profile.backend in backends
    ]
    matching = [profile for profile in candidates if required <= profile.capabilities]

    if matching:
        selected = max(matching, key=lambda profile: _selection_key(profile, target))
        status = "ok" if selected.max_fps >= target else "fps_degraded"
        diagnostics = _success_diagnostics(
            selected,
            status=status,
            target_fps=target,
            required_capabilities=required,
            devices=devices,
            backends=backends,
        )
        reason = (
            f"selected {selected.model_id} for {sorted(required)} "
            f"on {selected.device}/{selected.backend}"
        )
        if status == "fps_degraded":
            reason += f"; target_fps {target:g} exceeds profile max_fps {selected.max_fps:g}"
        return VisionProfileSelection(
            profile=selected,
            missing_capabilities=frozenset(),
            diagnostics=diagnostics,
            reason=reason,
        )

    return _missing_selection(
        required_capabilities=required,
        candidates=candidates,
        devices=devices,
        backends=backends,
        target_fps=target,
    )


def _selection_key(profile: VisionModelProfile, target_fps: float) -> tuple[bool, int, float, int]:
    return (
        profile.max_fps >= target_fps,
        profile.priority,
        profile.max_fps,
        -len(profile.capabilities),
    )


def _success_diagnostics(
    profile: VisionModelProfile,
    *,
    status: str,
    target_fps: float,
    required_capabilities: frozenset[str],
    devices: frozenset[str],
    backends: frozenset[str],
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "status": status,
        "profile_id": profile.id,
        "model_id": profile.model_id,
        "backend": profile.backend,
        "device": profile.device,
        "target_fps": target_fps,
        "max_fps": profile.max_fps,
        "required_capabilities": sorted(required_capabilities),
        "capabilities": sorted(profile.capabilities),
        "device_capabilities": sorted(devices),
        "allowed_backends": sorted(backends),
        "missing_capabilities": [],
        "loads_model": False,
    }
    if profile.degraded_from is not None:
        diagnostics["degraded_from"] = profile.degraded_from
    return diagnostics


def _missing_selection(
    *,
    required_capabilities: frozenset[str],
    candidates: list[VisionModelProfile],
    devices: frozenset[str],
    backends: frozenset[str],
    target_fps: float,
) -> VisionProfileSelection:
    available_capabilities = frozenset(
        capability for profile in candidates for capability in profile.capabilities
    )
    missing = required_capabilities - available_capabilities
    status = "missing_capabilities" if missing else "no_combined_profile"
    diagnostics: dict[str, object] = {
        "status": status,
        "required_capabilities": sorted(required_capabilities),
        "available_capabilities": sorted(available_capabilities),
        "missing_capabilities": sorted(missing),
        "device_capabilities": sorted(devices),
        "allowed_backends": sorted(backends),
        "candidate_profiles": [profile.model_id for profile in candidates],
        "target_fps": target_fps,
        "loads_model": False,
    }
    reason = (
        f"no profile supports required capabilities {sorted(required_capabilities)} "
        f"on devices {sorted(devices)} with backends {sorted(backends)}"
    )
    if missing:
        reason += f"; missing capabilities: {sorted(missing)}"
    else:
        reason += "; capabilities exist only across separate profiles"
    return VisionProfileSelection(
        profile=None,
        missing_capabilities=missing,
        diagnostics=diagnostics,
        reason=reason,
    )


def _normalize_devices(value: Iterable[str] | str | None) -> frozenset[str]:
    devices = _normalize_set(value, default={"cpu"})
    if "hailo8" in devices or "hailo8l" in devices:
        devices = frozenset(set(devices) | {"cpu"})
    return devices


def _normalize_set(
    value: Iterable[str] | str | None,
    *,
    default: Iterable[str] | None = None,
) -> frozenset[str]:
    source: Iterable[str] | str | None = default if value is None else value
    if source is None:
        return frozenset()
    if isinstance(source, str):
        source = {source}
    return frozenset(str(item).strip().lower() for item in source if str(item).strip())


__all__ = [
    "DEFAULT_PROFILE_ID",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_CAPABILITIES",
    "SUPPORTED_DEVICES",
    "VisionModelProfile",
    "VisionProfileSelection",
    "available_profiles",
    "get_profile",
    "select_profile",
]
