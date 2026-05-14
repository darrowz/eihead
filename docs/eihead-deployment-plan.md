# eihead Deployment Plan

This plan defines the deployment shape for moving honjia head hardware from
the monolithic eibrain body runtime into an eihead service pair. It is a plan
and template set only; it does not perform deployment.

## Repository And Runtime Paths

- honxin `/dev-project` is the code source of truth. It is not a runtime path.
- honxin `/dev-project/eibrain` keeps the current source tree while eihead is
  being extracted.
- honxin `/dev-project/eihead` is the target source repository once the split
  becomes independent.
- honjia `/opt/eihead/current` is the target runtime path for the deployed
  release.
- honjia `/etc/eihead/eihead.honjia.yaml` is the default runtime config.
- honjia `/etc/eihead/eihead.env` is the optional environment override file.
- honjia `18081` is reserved for the eihead runtime HTTP API.
- honjia `18080` remains the operator Web monitoring URL through the
  eihead-native monitor.

## Eye Runtime Direction

- The formal Eye runtime target is realtime stream detection from the live
  `/dev/video0` camera plus `/dev/hailo0` Hailo feed into
  `RealtimeVisionObservation` payloads, runtime status, and the `18080`
  monitor.
- The exported realtime Eye files are `eihead/eye/realtime.py` for contracts,
  `eihead/eye/adapters.py` for the runtime adapter boundary,
  `eihead/eye/gstreamer.py` for the native `/dev/video0` appsink reader,
  `eihead/eye/hailo_metadata.py` for `/dev/hailo0` detection metadata parsing,
  and `eihead/monitoring/realtime_vision.py` for monitor payload truthfulness.
- Static-image detection is compatibility/test-only. It can support fixtures,
  old callers, and no-hardware checks, but it is not the deployment direction
  and must not be used as the primary acceptance signal.
- Voice runtime is native through `eihead/ear` and `eihead/mouth`. Exported native
  voice files are `eihead/ear/realtime.py`, `eihead/ear/__init__.py`,
  `eihead/mouth/playback.py`, `eihead/mouth/__init__.py`, and
  `eihead/monitoring/voice.py`, with Web monitor endpoint wiring in
  `eihead/monitoring/web.py` and runtime facade support in `eihead/runtime/app.py`.
- Web runtime endpoints for native voice status are:
  - `GET /api/voice/realtime`
  - `GET /api/audio/realtime`
- The voice chain has entered a scheduler-backed functional stage for
  round lifecycle, scheduler status, and interrupt visibility. It is still not
  wired to real streaming LLM/TTS. Treat the current closed-loop voice
  diagnostics as functional offline/quasi-streaming diagnostics, not
  hardware-verified real streaming, so missing flow stages must remain explicit
  as `not_wired/unknown/degraded`.
- While `apps.body_runtime` is still exported, `/dev-project/eihead` also
  carries the minimal `eibrain/cognition/realtime` transitional package for
  temporary realtime scheduler compatibility until the eibrain/eihead protocol
  split is complete.
- If the camera, Hailo device, GStreamer runtime, or Hailo metadata parser is
  missing, the monitor must show `not_wired`, `unknown`, or explicit degraded
  state. A blank or fake-normal Eye panel is a deployment failure.
- `/dev-project/eihead` exports should make this direction visible in README,
  manifest metadata, and migration docs while transitional body/runtime code is
  still present.

## Service Templates

- `deploy/systemd/eihead-runtime.service` starts the runtime API with
  `eihead-runtime --config /etc/eihead/eihead.honjia.yaml http --host 0.0.0.0 --port 18081`.
- `deploy/systemd/eihead-monitor.service` starts the eihead-native Web monitor
  after `eihead-runtime.service` and keeps the user-facing Web port on `18080`.
- The templates run as user `darrow` from `/opt/eihead/current`.
- The templates do not edit, remove, or override existing eibrain service
  files.
- The native monitor starts with
  `eihead-runtime --config /etc/eihead/eihead.honjia.yaml monitor --host 0.0.0.0 --port 18080`;
  keep `monitoring.port: 18080` in config for operators and future generated
  service templates.
- No safety or permission gating is introduced in this phase.

## Cutover Strategy

The current honjia node has no production business load, so the preferred
strategy is a short downtime cutover rather than a side-by-side migration. This
reduces port conflicts and makes acceptance easier.

During the downtime window, it is acceptable to stop the old eibrain head-side
services before starting eihead:

```bash
sudo systemctl stop eibrain-monitor.service eibrain-vision-hailo.service
systemctl --user stop eibrain-monitor.service eibrain-vision-hailo.service brain-runtime.service
```

Some units may not exist on a given honjia image. Treat "unit not found" as
non-fatal, then confirm ports and devices are free before starting eihead.

Start the new services only after the old monitor/body/vision ownership is
released:

```bash
sudo systemctl daemon-reload
sudo systemctl start eihead-runtime.service
sudo systemctl start eihead-monitor.service
```

Enable boot persistence only after acceptance passes:

```bash
sudo systemctl enable eihead-runtime.service eihead-monitor.service
```

## Acceptance Checks

- `systemctl status eihead-runtime.service` is active.
- `systemctl status eihead-monitor.service` is active.
- `curl http://127.0.0.1:18081/status` returns eihead runtime status.
- `curl http://127.0.0.1:18081/capabilities` returns the honjia capability
  manifest when the runtime API supports it.
- `curl http://127.0.0.1:18080` opens the Web monitor.
- `/dev/video0`, `/dev/hailo0`, `/dev/i2c-1`, microphone, and speaker state
  appear in the Web monitor as real data, degraded data, explicit offline data,
  `unknown`, or `not wired`; blank "normal" placeholders are not acceptable.
- Voice status appears via `/api/voice/realtime` and `/api/audio/realtime`, and
  Web can show round/scheduler/interrupt state; until real streaming LLM/TTS is
  connected and hardware-verified, monitor values must remain explicit about
  incomplete flow stages. Offline/quasi-streaming closed-loop diagnostics are
  useful for deployment triage, but they do not satisfy real streaming voice
  acceptance by themselves.
- Realtime eye stream detection from `/dev/video0` and `/dev/hailo0` publishes
  frame age, FPS, detection boxes, scores, backend, and stale/error state to
  the monitor.
- Static-image detection may validate compatibility/test fixtures, but it does
  not satisfy deployment acceptance by itself.
- Voice, realtime eye stream detection, Hailo-backed boxes, and horizontal neck
  movement are manually verified after service start.

## Rollback

Rollback is service-level only. Stop `eihead-monitor.service` and
`eihead-runtime.service`, then start the previous eibrain services from the same
downtime window. Do not overwrite honxin source repositories during rollback.
