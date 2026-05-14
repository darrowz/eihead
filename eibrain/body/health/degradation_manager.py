"""Body degradation manager at the health boundary.

It converts low-level organ snapshots into capability-level semantics and a
single degradation mode used by higher orchestration layers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .capability_matrix import CapabilityMatrix
from .organ_health import OrganHealth

_EAR_ORGAN = "ear"
_MOUTH_ORGAN = "mouth"
_EYE_ORGAN = "eye"
_NECK_ORGAN = "neck"

_CAPTURE_SUBFUNCTION = "capture"
_ASR_SUBFUNCTION = "asr"
_DETECTION_SUBFUNCTION = "detection"
_IDENTITY_SUBFUNCTION = "identity"
_TTS_PLAYBACK_SUBFUNCTION = "tts_playback"
_MOTOR_SUBFUNCTION = "motor"


@dataclass(slots=True)
class DegradationResult:
    capabilities: CapabilityMatrix
    degradation_mode: str


class DegradationManager:
    def evaluate(self, organ_states: list[OrganHealth]) -> DegradationResult:
        by_name = {state.organ: state for state in organ_states}
        # Boundary: evaluation remains read-only over health snapshots and does not
        # invoke execution paths.
        ear = by_name.get(_EAR_ORGAN)
        mouth = by_name.get(_MOUTH_ORGAN)
        eye = by_name.get(_EYE_ORGAN)
        neck = by_name.get(_NECK_ORGAN)

        capabilities = CapabilityMatrix(
            can_hear_voice=self._is_real_capability(
                ear,
                _CAPTURE_SUBFUNCTION,
                allow_degraded=True,
            ),
            can_transcribe_speech=self._has_transcribe_capability(ear),
            can_see_people=self._is_real_capability(
                eye,
                _DETECTION_SUBFUNCTION,
                allow_degraded=True,
            ),
            can_identify_person=self._is_real_capability(eye, _IDENTITY_SUBFUNCTION),
            can_speak=self._is_real_capability(
                mouth,
                _TTS_PLAYBACK_SUBFUNCTION,
                allow_degraded=True,
            ),
            can_orient_head=self._is_real_capability(
                neck,
                _MOTOR_SUBFUNCTION,
                allow_degraded=True,
            ),
        )

        degradation_mode = "normal"
        if ear is not None and capabilities.can_hear_voice and not capabilities.can_transcribe_speech:
            degradation_mode = "low_confidence_body"
        elif mouth is not None and not capabilities.can_speak:
            degradation_mode = "mute_companion"
        elif neck is not None and not capabilities.can_orient_head:
            degradation_mode = "fixed_gaze"

        return DegradationResult(capabilities=capabilities, degradation_mode=degradation_mode)

    @staticmethod
    def _has_transcribe_capability(organ: OrganHealth | None) -> bool:
        if not DegradationManager._is_real_capability(organ, "asr", allow_degraded=True):
            return False
        if organ is None:
            return False
        subfunction = organ.subfunctions.get(_ASR_SUBFUNCTION)
        if subfunction is None:
            return False
        if subfunction.health == "healthy":
            return True
        status = str(subfunction.details.get("status", "")).strip().lower()
        return status in {
            "silence",
            "below_asr_threshold",
            "live_probe_skipped",
            "waiting_for_data",
            "warming_up",
            "recent_trace",
            "transcribed",
        }

    @staticmethod
    def _is_real_capability(
        organ: OrganHealth | None,
        subfunction_name: str,
        *,
        allow_degraded: bool = False,
    ) -> bool:
        if organ is None:
            return False
        subfunction = organ.subfunctions.get(subfunction_name)
        if subfunction is None:
            return False
        if subfunction.details.get("driver") == "noop":
            return False
        if subfunction.health == "healthy":
            return True
        if allow_degraded and subfunction.health == "degraded":
            return True
        return False
