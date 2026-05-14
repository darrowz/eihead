# eihead Cutover Checklist

This checklist freezes the current honjia baseline before splitting the head
runtime out of eibrain. It intentionally focuses on read-only verification and
human acceptance. The security/permission layer is out of scope for this pass.

## Fixed ownership model

- honxin source of truth: `/dev-project/eibrain`
- future honxin eihead source of truth: `/dev-project/eihead`
- shared protocol source of truth, once created: `/dev-project/eiprotocol`
- honjia current runtime path from config: `/home/darrow/eibrain`
- honjia target runtime path to confirm before cutover: `/opt/eihead/current`
- honjia mutable runtime data path to confirm before cutover: `/var/lib/eihead`
- honjia local config path to confirm before cutover: `/etc/eihead`

Do not treat `/dev-project` as a runtime deployment path. It is the honxin code
workspace. honjia should run from a versioned deployment path with explicit
service files.

## Phase 0: baseline freeze

- Confirm honxin `/dev-project/eibrain` is clean or the dirty files are named and owned.
- Confirm honjia currently runs from the expected eibrain deployment path.
- Run `bash scripts/capture-eihead-baseline.sh` on honjia and save the output.
- Capture hostname, date, kernel, git commit, active eibrain services, listen ports, device nodes, monitor health/status, and recent journal tails.
- Confirm `/dev/video0`, `/dev/hailo0`, `/dev/i2c-1`, and `/dev/snd` are visible or the missing devices are explicitly recorded.
- Confirm honjia Web monitor on port `18080` shows real values for ear, eye, mouth, neck, cadence, and latest runtime errors.
- Run one manual voice wake test and record ASR text, LLM latency, TTS playback, and whether the answer is conversational.
- Run one manual realtime eye stream detection test and record whether the
  camera/Hailo stream produces face/person boxes on the monitor.
- Treat static-image detection as a compatibility/test placeholder only; do not
  use it as cutover evidence unless realtime hardware is unavailable and the
  gap is explicitly recorded.
- Run one manual pan-only neck test and record whether the pan angle moves without oscillation.

## Phase 1: eiprotocol MVP

- Define versioned observations, actions, capabilities, and outcomes before moving code.
- Keep the protocol backward compatible with current eibrain body runtime payloads.
- Verify sample payloads exist for audio turn, vision frame, detection result, device status, move head action, speak action, and execution outcome.
- Verify eibrain can parse unknown future fields without failing.
- Verify eimemory/eitraining feedback fields can be attached without coupling eihead to storage internals.

## Phase 2: eihead scaffold

- Create honxin `/dev-project/eihead` as a separate source repository.
- Keep eibrain source unchanged except for an explicit bridge or adapter boundary.
- Copy, do not move, the current working honjia runtime pieces first.
- Add packaging, config, service template, and a minimal local health endpoint.
- Confirm tests can run without honjia hardware by using static checks and fake adapters.
- Confirm the scaffold has no dependency on honxin-only secrets or absolute local development paths.

## Phase 3: capability manifest

- On eihead startup, publish a `CapabilityManifest` to eibrain.
- Include cameras, microphones, speakers, neck axis support, ASR, TTS, vision, embedding, and health status.
- Mark honjia neck as pan-only unless tilt hardware is actually present.
- Include concrete device paths such as `/dev/video0`, `/dev/hailo0`, `/dev/i2c-1`, and audio card names.
- Confirm eibrain stores the latest manifest and the Web monitor renders it from real data.
- Confirm a device replacement changes only the manifest/config, not eibrain core logic.

## Phase 4: eibrain to eihead runtime bridge

- Treat this batch as HTTP JSON event routing after the `POST /events`
  scaffold: one eiprotocol envelope per request. Realtime streaming remains
  future work.
- Route audio observations from eihead into the current eibrain dialogue loop.
- Route realtime eye stream observations from eihead into detection/identity diagnostics.
- Route action request events through the `handle_action` bridge before they
  reach mouth, neck, attention, or diagnostic capture handlers.
- Route speak actions from eibrain back to eihead mouth playback.
- Route move head actions from eibrain back to eihead neck control.
- Record observation, outcome, and feedback envelopes in the recent event
  journal/diagnostics for the monitor/API.
- Return clear JSON error responses and `not_processed` outcomes for invalid
  envelopes; return explicit `not_wired`/`not_processed` for unwired handlers.
- Confirm the monitor or runtime API can inspect recent routed events.
- Preserve the old eibrain body runtime as fallback until parity is confirmed.
- Verify bridge logs include request id, turn id, action id, latency, and outcome.

## Phase 5: Web monitor split

- Confirm honjia port `18080` reads from the new eihead runtime or a clear proxy.
- Remove placeholder "healthy" values that are not backed by runtime data.
- Show real ear state, ASR text, VAD/recording state, LLM timing, TTS state,
  realtime eye frame age, FPS, detections, detection scores, neck target angle,
  actual last command, and error tail.
- If static-image detection appears anywhere in the monitor or diagnostics,
  label it compatibility/test-only; the accepted Eye direction is realtime
  stream detection.
- Keep layout stable while changing data sources.
- Confirm refresh cadence and average latency values are populated from measured timestamps.
- Compare the Phase 5 monitor with the Phase 0 baseline before accepting.

## Phase 6: memory and training feedback

- Send execution outcomes back to eibrain after speech, vision, and neck actions.
- Store useful outcomes in eimemory without mixing raw telemetry with durable identity memory.
- Send failed or low-confidence cases to eitraining as candidates, not automatic truth.
- Record what was done, whether it worked, user feedback if present, and next improvement hints.
- Verify a recent dialogue can be retrieved from eimemory with the correct source and timestamp.

## Phase 7: service cutover

- Confirm honxin `/dev-project/eibrain` and `/dev-project/eihead` contain the exact revisions being deployed.
- Confirm honjia deployment path is the target runtime path, not the honxin source workspace.
- Install eihead service files side by side with current eibrain services first.
- Start eihead services manually and compare against the Phase 0 baseline.
- Switch Web monitor to eihead only after ear, eye, mouth, and neck have real data.
- Disable old honjia eibrain body services only after eihead passes parity checks.
- Keep eibrain cognitive services on honxin and verify honjia can reach honxin over Tailscale.
- Confirm restart persistence after honjia reboot for eihead runtime and monitor services.
- Record the final service names, deployment path, config path, data path, git revisions, and owner.

## Acceptance summary

Cutover is accepted only when the Phase 0 baseline can be repeated after Phase 7
with equal or better results for voice wake, conversational response, realtime
eye stream detections, pan-only neck behavior, and Web monitor truthfulness.
