# eihead

Standalone head-node export for the ei-hongtu project.

This repository is generated from the eibrain monorepo by
`scripts/export-eihead-repo.py`. It contains the honjia-facing head runtime,
the `apps.head_runtime` compatibility entrypoint, a transitional copy of the
old body runtime, eihead systemd templates, and eihead migration/deployment
docs.

## Expected sync target

- Source of truth on honxin: `/dev-project/eihead`
- Runtime deployment path on honjia: `/opt/eihead/current`
- Runtime API: `eihead-runtime http --host 0.0.0.0 --port 18081`
- Native Web monitor: `eihead-runtime monitor --host 0.0.0.0 --port 18080`

## Eye direction

The production eye target for `/dev-project/eihead` is realtime stream detection:
continuous `/dev/video0` camera frames and `/dev/hailo0` detections feeding live
RealtimeVisionObservation payloads, runtime status, and the operator monitor.

The native voice boundary is under `eihead/ear` and `eihead/mouth`:
`eihead/ear/realtime.py`, `eihead/ear/__init__.py`,
`eihead/mouth/playback.py`, and `eihead/mouth/__init__.py`.
Its monitor adapter is `eihead/monitoring/voice.py`.
The monitor endpoint bridge is exported in `eihead/monitoring/web.py`, with
runtime facade support in `eihead/runtime/app.py`.

Native runtime and monitor surface includes:
- `GET /api/voice/realtime`
- `GET /api/audio/realtime`

Voice chain is now in a scheduler-backed functional stage using Realtime
Cognitive Scheduler for round lifecycle, scheduler status, and interrupt
visibility. Realtime Cognitive Scheduler compatibility is transitional. It
provides functional offline/quasi-streaming diagnostics for the closed-loop
voice diagnostics surface, but it is not hardware-verified real streaming.
The closed-loop voice diagnostics are functional offline/quasi-streaming diagnostics,
not hardware-verified real streaming or real streaming LLM/TTS.
It is still functional-not-complete: the loop has not been wired to real
streaming LLM/TTS, and the Web monitor should make round/scheduler/interrupt
state visible without presenting missing streaming stages as complete.

## Code completion vs cutover

Code-level completion is not honjia cutover completion. In
`EXPORT_MANIFEST.json`, `code_completion.software_closure` is `complete`, but
`code_completion.honjia_cutover` is `blocked_by_hardware_validation`.

The `software_closure` field lists which Wave 3 P0/P1 software gates are
complete at code level, which P0/P1 checks still require honjia hardware
validation, and which legacy shim removals still block any fully detached claim.
Do not describe this export as fully detached while
`legacy_body_runtime_detached` or `full_detachment_claim_allowed` is `false`.
Real cutover still requires recorded honjia parity for realtime eye, pan-only
neck, ear/mouth audio, services, reboot persistence, and rollback.

The standalone export intentionally includes the native realtime eye adapter and
monitor payload files:

- `eihead/eye/adapters.py`
- `eihead/eye/gstreamer.py`
- `eihead/eye/hailo_metadata.py`
- `eihead/eye/realtime.py`
- `eihead/monitoring/realtime_vision.py`

Native voice boundaries are exported as:

- `eihead/ear/__init__.py`
- `eihead/ear/realtime.py`
- `eihead/mouth/__init__.py`
- `eihead/mouth/playback.py`
- `eihead/monitoring/voice.py`
- `eihead/runtime/http_api.py`
- `eihead/monitoring/web.py`

The monitor truthfulness rule is strict: missing live wiring must be shown as
`not wired`, `not_wired`, `unknown`, or explicit offline/degraded data. Do not
show blank or fake-normal realtime vision status.

Static image detection is compatibility/test-only. Keep it only for old callers,
fixtures, and non-hardware tests; do not treat it as the deployment direction.

## Local commands

```bash
python -m pip install -e .
eihead-runtime status
eihead-runtime http --host 0.0.0.0 --port 18081
eihead-runtime monitor --host 0.0.0.0 --port 18080
```

The current runtime still carries a small `eibrain.protocol` compatibility
subset, transitional `eibrain.body` hardware code, and the minimal
`eibrain.cognition.realtime` scheduler primitives needed by the exported
`apps.body_runtime` voice chain, shared `eibrain.voice` readiness helpers,
plus transitional hardware verification helpers.

`eihead` consumes `eiprotocol` as a standalone dependency. Install both from
the parent workspace during development:

```bash
python -m pip install -e D:/github/ei-workspace/repos/eiprotocol
python -m pip install -e D:/github/ei-workspace/repos/eihead
```

`EXPORT_MANIFEST.json` also contains `native_completion_gates`. Treat those
gates as the source of truth for whether eye, neck, ear, mouth, runtime,
export, and deploy are complete. A module remains transitional or blocked until
its gate is verified on honjia; status and monitor payloads must say
`not_wired`, `unknown`, `degraded`, or `blocked` rather than implying fake
completion.

## Cutover readiness and fake completion

`EXPORT_MANIFEST.json` contains `cutover_readiness`, a machine-readable summary
for cutover review. It lists `native_provider_modules`, `monitor_endpoints`, and
`legacy_shim_policy` so reviewers can tell native boundaries from transitional
compatibility.

How to judge fake completion:
- If `cutover_readiness.hardware_verified` is `false`, the hardware has not been verified on honjia and the export remains blocked/transitional even if local tests or static fixtures pass.
- If `legacy_shim_policy.legacy_body_runtime_detached` is `false`, the export
  still carries legacy body runtime shims. Those paths must stay explicitly
  marked as transitional shims and must not be described as fully detached.
- Monitor endpoints are readiness probes, not proof of completion. A response
  is only acceptable when it shows real data or explicit `not_wired`, `unknown`,
  `degraded`, or `blocked` state for missing hardware or unwired stages.
