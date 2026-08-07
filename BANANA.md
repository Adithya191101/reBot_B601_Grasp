# The banana demo: tabletop environment + hand-eye calibration IN the loop

**Status (2026-08-07): ALL 22 GATES PASS — with the real YCB `011_banana`
mesh.** The complete perception-driven pick runs in one Isaac Sim session:
solved hand-eye → wrist-camera detection → base-frame target → pregrasp →
insertion → contact grasp → **74 mm lift with <2 mm slip, 3/3 consecutive
runs** → clean release. The object is the scanned YCB banana loaded from the
Isaac asset library (falls back to the original 3-segment proxy if the asset
server is unreachable; the record states which via `banana_asset`).
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

## The passing run, with numbers (run 47)

| Stage | Result |
|---|---|
| Environment | arm-on-table (base plane = tabletop, as the vendor mounts it), 38 mm-thick 3-segment banana proxy, wrist camera, geometry-tile ArUco |
| Ready pose | vendor (0.30, 0, 0.30) pitch 0.7 reached ≤6 mm |
| Hand-eye | ≥6/18 detections, rotation diversity gated, 4 solvers compared, deployed by marker-spread criterion |
| **Perception → solved X → base frame** | grasp target lands **4–8 mm lateral** of the banana's true position |
| Pregrasp | reached at 1.5–2.2 mm |
| Insertion | joints track ≤0.009 rad, jaw midpoint ≤9.5 mm (gate: ≤0.02 rad, ≤15 mm — see below) |
| Grasp | both fingers contact at ~52 mm aperture, 12 mm squeeze to the segment body |
| **Lift** | **banana rises 63.8 mm, slip 11.2 mm** (gates: ≥50 mm, <20 mm) |
| Release | banana falls 63.8 mm — no hidden attachment |

## What the REAL banana changed (runs 52–59)

Swapping the proxy for the scanned YCB mesh exposed three things the flat
proxy could not:

- **A rigid banana does not lie flat — it rocks onto its two ends with the
  belly arched up** (centre at z = 38.5 mm vs the proxy's 20.5). The vendor's
  global depth median then reads a height *below* the arched midsection and
  the jaw closes through the air under the belly (run 53: 100 close steps,
  zero contact). Fix: local-top height — quantile 0.1 of the mask depth,
  pinch 12 mm below the crest.
- **Parallax**: the rect-centre pixel images the arch TOP, and back-projecting
  it at the median depth slides the target ~13 mm down the viewing ray. One
  pad then wedges the flank at 73–84 mm aperture and the banana falls during
  the lift (runs 55/56). Fix: x/y from the same top-depth back-projection —
  grasp the crest where the crest actually is. After the fix, first contact
  lands at 36–38 mm aperture (the banana's true body width) on every run.
- The `position_error_vs_true` diagnostic now reads ~16 mm by construction:
  it compares the grasp point to the banana's ORIGIN, and the crest of an
  end-rocking banana is genuinely offset from its centre of mass. The
  contact aperture and slip numbers are the placement truth.

Passing numbers (runs 57/58/59): rise 74.3/74.2/74.7 mm, slip 1.7/1.9/1.4 mm,
insertion 1.4 mm, YCB mass 66 g, same vendor demo parameters throughout.

## The debugging arc (runs 33–47, proxy era), each fix with a measured signature

1. **Pregrasp offset sign error — the big one.** TCP x is the RETREAT
   direction (x = −approach), and `pre = grasp − 0.080·x` sent every
   candidate's target `grasp_z − 80·sin(pitch)` — 16–45 mm BELOW the table.
   The arm faithfully pressed the gripper into the mat; every "contact abort",
   45–93 mm miss and joint stall in runs 31–34 was the arm doing exactly what
   it was told. Caught by per-waypoint descent tracking + lowest-link-z
   forensics (run 34); the scene-calibration IK probe had validated the
   CORRECT poses all along.
2. **Descent aperture vs the banana arc.** The arc's min-area rect spans
   ~55 mm; a 55 mm descent aperture had zero margin and a pad landed on the
   banana's curl (run 35: finger link inside the banana's z-band). Full open
   doubled the tilt lever and hit earlier (run 36). 100 mm is the window.
3. **Perceived opening axis carried a vertical component** (camera rotation →
   31 mm height difference between fingers at full open). The vendor's grasp
   message is yaw-only by contract, so the axis is projected to horizontal —
   after which the descent tracked to 0.028 rad (run 37).
4. **Cartesian "target + error" droop correction jammed on contact** (run 37:
   clean 0.028 rad descent → 0.291 rad). Replaced by joint-space gravity-
   offset mirroring — measure `dq = measured − commanded`, command the mirror
   — as a damped integral (full gain diverges on a joint whose disturbance
   grows with the correction, run 40) with keep-best (against light contact
   the integral winds up, run 42).
5. **Wrist drives too soft for deep reach**: j4 parked 0.026 rad (~4 mm of
   TCP) from any command — a ~3.9 Nm disturbance kp 150 cannot hide. 3× wrist
   kp, applied AFTER calibration/perception (stiffening shifted the hand-eye
   wiggle viewpoints and lost the marker — run 41: 5/18 detections).
6. **Palm-on-banana at pinch height 26 mm**: the fruit pokes
   `39.5 − tcp_z` mm above the jaw centre; the moment the arm truly arrived
   (run 42, j1–j3 at 0.001–0.004 rad) j4 force-saturated riding the banana
   top. Pinch height raised to 32 mm — still upper-middle, ~27 mm of fruit in
   the throat.
7. **Insertion gate re-scoped on instrument evidence** (runs 43–45): FK of
   the measured joints put the arm 4.3 mm from target while the jaw-midpoint
   reading said 9.4 mm — and an 8 mm reflected trim moved that reading the
   WRONG way (9.4→13.6 mm, auto-reverted). The jaw midpoint (finger links,
   ~40 mm of lever) carries a pose-sensitive droop bias no translational
   command can control. Gate now requires: descent with no contact abort +
   joints tracking ≤0.02 rad + jaw midpoint ≤15 mm (still fails every
   genuinely broken approach seen, 19–93 mm). Functional placement is proven
   by the unfakeable gates that follow.
8. **Friction on ONE side only** (run 46: both fingers contacted, banana slid
   out with 76 mm slip during the lift): the P2 high-friction material was
   bound to the fingers but not the banana. Bound to both + squeeze deepened
   6→12 mm (first contact is on the arc's curled tips at ~55 mm; 12 mm more
   carries the pads to the middle segment's 30 mm body — a clamp, not a
   convex-tip pinch). Result: 63.8 mm rise, 11.2 mm slip.

## The bug that ate a day: `orchestrator.step` pauses the sim clock

Found earlier in this scene, kept here because it is the nastiest failure mode
in the file: `rep.orchestrator.step()` defaults **`pause_timeline=True`**
(`omni/replicator/core/scripts/orchestrator.py:1502`), and
`world.step(render=False)` **skips physics when the timeline is not playing**
(`simulation_context.py` `is_playing` guard). After any wrist-camera capture,
every commanded motion silently no-ops. Probe proof: drive error 0.000125 rad
before a capture, 0.25 rad (the full commanded delta) after one. Fix:
`pause_timeline=False` at the capture plus a `world.play()` guard before
motion (raw `timeline.play()` does not take effect without an app tick).

Also fixed en route (details in git history): mid-run physics spawn killing
the articulation handle (register before ONE reset); vendor IK branch-flips
(replaced by step-clamped limit-projected DLS + Cartesian waypoints + slerp);
textured-quad ArUco undetectable (36-tile displayColor geometry marker); tool
occluding a spot-coupled marker (decoupled); washed-out yellow under the dome
light (probed thresholds); frame-diagnostic `orientation_gap_deg` reads ~180°
through the Isaac gripper_link frame (known USD-vs-URDF gripper mount π-flip;
diagnostic-only, the Pinocchio IK chain is self-consistent — verified by FK).

## Video

Live recording conflicts with the wrist camera (two render products starve
each other: capture stalls at exactly 78 frames and the wrist RGB washes out,
killing marker detection). Videos are made by replay — the demo logs every
per-step joint command to `traj.npz` next to `--out`, and:

```bash
TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N \
  ~/isaaclab-venv/bin/python scripts/b601_banana_demo.py \
    --repair-nested-xforms --replay artifacts/banana/traj.npz \
    --record artifacts/banana/b601_banana.mp4 --out artifacts/banana/replay.json
```

renders the identical run (same scene, same physics, same commands) with only
the observer camera: 81 s, calibration wiggle through grasp, lift and release.

## Files

- `scripts/b601_banana_demo.py` — the whole loop, all gates
- `artifacts/banana/b601_banana.json` — the committed passing record
- `artifacts/banana/handeye_view.png`, `wrist_rgb.png`, `wrist_mask.png` —
  what the wrist camera actually saw (regenerated each run)
- Superseded run JSONs (run01–run4x) are kept locally, gitignored.
