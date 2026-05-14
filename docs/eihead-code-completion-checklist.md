# eihead Code Completion Checklist

Status date: 2026-05-05

Code-level completion is not honjia cutover completion. This checklist records
what the export may claim after Wave 3 software closure and what remains blocked
until honjia hardware evidence exists.

## Manifest Fields

- `code_completion.software_closure`: `complete` means the P0/P1 software
  readiness fields, docs, and focused export tests exist.
- `code_completion.honjia_cutover`: `blocked_by_hardware_validation` until
  honjia Phase 0 parity and cutover checks are recorded.
- `software_closure.completed`: code-level P0/P1 gates that are complete.
- `software_closure.blocked_by_hardware_validation`: real-device checks that
  still block cutover.
- `software_closure.blocked_by_legacy_detachment`: transitional shim removals
  that still block any fully detached claim.

## Code-Level Complete

- [x] P0 export manifest readiness names `code_completion`,
  `software_closure`, `cutover_readiness`, and `legacy_shim_policy`.
- [x] P0 runtime and monitor truthfulness gates require real data or explicit
  `not_wired`, `unknown`, `degraded`, or `blocked` states.
- [x] P0 realtime eye native boundary files are exported and static-image
  detection remains compatibility/test-only.
- [x] P1 ear/mouth diagnostic boundaries are exported as native code-level
  surfaces while real streaming remains unverified.
- [x] P1 legacy shim policy names each copied legacy package and keeps
  `full_detachment_claim_allowed` false.

## Still Blocked

- [ ] P0 honjia Phase 0 parity baseline has not been recorded for the exported
  eihead repo.
- [ ] P0 realtime `/dev/video0` plus `/dev/hailo0` validation has not been
  recorded on honjia.
- [ ] P0 `/dev/i2c-1` pan/yaw movement and unsupported tilt truthfulness have
  not been recorded on honjia.
- [ ] P1 U4K microphone, VAD/ASR, audible TTS playback, busy suppression, and
  `stop_speech` have not been recorded on honjia.
- [ ] P1 service cutover, reboot persistence, and rollback have not been
  recorded on honjia.

## Claim Rules

- Do not say the repo is fully detached while legacy shims are copied or needed
  at runtime.
- Do not convert `blocked_by_hardware_validation` to complete based on local
  tests, static fixtures, or offline/quasi-streaming diagnostics.
- Do not enable permanent honjia eihead services until the hardware and rollback
  checks above have recorded evidence.
