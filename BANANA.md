# The banana demo: tabletop environment + hand-eye calibration IN the loop

**Status (2026-08-07): calibration and perception chains PASS and are wired
together; the final approach is in active tuning.** 16 of 17 gates pass; the
open item is finger-vs-surface clearance on the last centimetres of descent.
Code: `scripts/b601_banana_demo.py` · Record: `artifacts/banana/b601_banana.json`

```bash
TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N \
  ~/isaaclab-venv/bin/python scripts/b601_banana_demo.py \
    --repair-nested-xforms --out artifacts/banana/b601_banana.json
```

## What this is

The Seeed demo's own loop, in one Isaac Sim session, on the demo's own numbers
(`config/default.yaml`: ready (0.30, 0, 0.30) pitch 0.7, pregrasp −0.080 m,
insertion +0.015 m, depth quantile 0.5):

```
tabletop scene -> ArUco hand-eye calibration (vendor solver, vendor classes)
   -> hide marker, drop banana -> wrist RGB-D -> yellow mask -> min-area-rect
   grasp estimate (camera frame) -> SOLVED hand-eye X -> base-frame TCP target
   -> ready -> pregrasp -> insert -> grasp -> lift -> release
```

Honesty rules enforced in code: the grasp target is computed **only** from the
camera, the depth image, the measured gripper pose and the **solved** X — the
banana's true pose is used solely as a post-hoc error diagnostic. The hand-eye
method is selected **without ground truth** (static marker → the spread of its
recovered base position scores each candidate X — a criterion that transfers to
hardware); truth comparison is reported as a sim-only diagnostic.

## What passes, with numbers (run 31/32, 16/17 gates)

| Stage | Result |
|---|---|
| Environment | arm-on-table (base plane = tabletop, as the vendor mounts it), banana proxy lying flat, wrist camera, geometry-tile ArUco |
| Ready pose | vendor (0.30, 0, 0.30) pitch 0.7 reached ≤6 mm |
| fx self-check | effective fx 1.8–2.1 % below assumed, corrected |
| Hand-eye | ≥6 detections, rotation diversity 71°/24° gated, 4 solvers compared, deployed by marker-spread criterion |
| **Perception → solved X → base frame** | **grasp target lands 5.7–14.6 mm lateral / ≤7 mm z from the banana's true position** |
| Drives after every render | 0.0002 rad probe error (was 0.25 before the timeline fix) |

The perception number is the headline: a camera detection becomes a base-frame
target through the *solved* calibration at ~real-demo accuracy.

## The bug that ate a day: `orchestrator.step` pauses the sim clock

The single nastiest find, proven by bisection probes and the Isaac source:

- `rep.orchestrator.step()` defaults **`pause_timeline=True`** — it stops the
  Kit timeline after rendering (verbatim in
  `omni/replicator/core/scripts/orchestrator.py:1502`).
- `world.step(render=False)` — what every drive ramp uses — **skips physics
  entirely when the timeline is not playing** (`simulation_context.py`, the
  `if self.is_playing()` guard).
- Net effect: after any wrist-camera capture, every commanded motion becomes a
  silent no-op. Drives test perfectly healthy in isolation (0.0001 rad), then
  the next commanded approach produces **exactly zero displacement**, which
  masquerades as a collision, dead handles, IK branch flips — we chased all
  three first.
- Probe proof: drive error 0.000125 rad before a capture, **0.25 rad (= the
  full commanded delta)** immediately after one (`run28.json`).
- Fix: `pause_timeline=False` at the capture site, plus a `world.play()` guard
  before motion (`world.play()` ticks the app so the play state takes effect —
  a raw `timeline.play()` does not, which is why the first guard changed
  nothing).

Also fixed en route, each with a measured signature:

1. **Spawning a physics prim mid-run kills the articulation's control handle**
   — register the object in `world.scene` *before* one `world.reset()`, rebuild
   the monitor, re-apply gains (the b601_pick pattern). A standalone
   `initialize()` after the reset re-kills it, nondeterministically.
2. **Random-restart IK branch-flips** — the vendor's `solve_ik_with_retry`
   found shoulder-flipped solutions (exact FK, joint path through the table;
   4.000 rad tracking error signature). Replaced for path-following by a
   step-clamped, joint-limit-projected DLS (same Pinocchio), local by
   construction; Cartesian-linear waypoints with slerped orientation;
   rotate-in-place before translating (wrist rolls are safe, shoulder jumps
   are guarded).
3. **Textured-quad ArUco undetectable in this scene** — replaced with a
   36-tile `displayColor` geometry marker built straight from
   `generateImageMarker(dict, id, 6)`: what you author is what the camera sees.
4. **Tool occludes a marker at the grasp spot** — the calibration marker lies
   to the side, decoupled from the spot (tying them moved it out of view:
   4/18 detections).
5. **Washed-out yellow** — displayColor desaturates under the dome light;
   thresholds probed on the actual render (banana H≈24 S≈43 vs table S≈10).
6. **j4 gravity droop at deep reach** — endpoint error-feedback servo cycles,
   guarded to not run while in contact (unguarded, they jam the wrist into the
   surface — measured j4 driven onto its −1.87 hard stop).

## The open item, precisely

Two layers, one solved. The finger pads are ~39 mm thick: around a **24 mm**
object on a surface they cannot close without grazing it — contact fired at the
final waypoint every time (the P2 pedestal lesson in new clothes). A
banana-realistic **38 mm** proxy (real bananas run 35–45 mm) pinched on its
upper half **ended the contact aborts** and sharpened perception to **4.8 mm**.

What remains (run 33): the descent completes but stops **93 mm short of the
pregrasp with j2 0.233 rad under target** — either residual light contact the
0.25 rad abort threshold doesn't catch, or shoulder drive authority at deep
reach (x≈0.26, TCP z≈0.08). Next probes: per-joint applied-effort telemetry at
the stall, fingertip height measurement during descent, and a shallower target
spot. The grasp/lift/release gates after this point are untested in this scene
— every prior stage of them is proven in P2/P3.

## Files

- `scripts/b601_banana_demo.py` — the whole loop, all gates
- `artifacts/banana/handeye_view.png`, `wrist_rgb.png`, `wrist_mask.png` —
  what the wrist camera actually saw (regenerated each run)
- Superseded run JSONs are kept locally, gitignored; the canonical record slot
  is `artifacts/banana/b601_banana.json` (committed once the full loop passes).
