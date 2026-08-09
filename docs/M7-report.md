# M7 — Static gantry collision avoidance (visible re-routing)

**Date:** 2026-08-08 · **Verdict:** **PASS** (10/10 scored + 3/3 contrast)
· **Artifact:** `artifacts/m7/gantry_avoidance.json`
· **Design doc gate:** sec. 21 "M7 | static gantry collision avoidance |
visible re-routing works"

## What was demonstrated

10 pose pairs (pick-zone ↔ place-zone transits) were chosen so that the
**straight-line joint path qA→qB provably sweeps the gripper/wrist through
the gantry crossbar and touches nothing else** — proven with
rebot_planner's mesh-based `collision_core` (canonical URDF collision
meshes, vendor-SRDF active pairs, `cell_geometry.yaml` table at true z=0
+ gantry). cuMotion (Isaac ROS 4.5, standalone `cumotion/motion_plan`
action server — the passing M6 stack unchanged) planned all 10 with the
gantry in its world: **every plan is mesh-collision-free, executed
through the M5 sim bridge (FJT via adapter + sim-JTC shim), converged
(≤ 0.02 rad), landed exactly on the goal configuration (goal error
0.000 rad), and visibly re-routed around the bar.** The same 3 first
pairs re-planned with the gantry **removed** produce near-straight paths
that sweep straight through the gantry volume — the re-routing is
attributable to the obstacle, not to planner habit.

## Results (run of 2026-08-08, `artifacts/m7/`)

### With gantry — 10/10 PASS

| pair | direction | max TCP dev (m) | max joint dev (rad) | min gantry clearance (m) | chord hits (ref/actual) |
|---:|---|---:|---:|---:|---:|
| 1 | pick→place | 0.193 | 0.614 | 0.033 | 32/33 |
| 2 | place→pick | 0.191 | 0.345 | 0.036 | 18/18 |
| 3 | pick→place | 0.112 | 0.419 | 0.031 | 89/89 |
| 4 | place→pick | 0.223 | 0.422 | 0.034 | 38/38 |
| 5 | pick→place | 0.121 | 0.262 | 0.029 | 47/47 |
| 6 | place→pick | 0.239 | 0.779 | 0.029 | 25/25 |
| 7 | pick→place | 0.204 | 0.445 | 0.036 | 23/21 |
| 8 | place→pick | 0.133 | 0.279 | 0.033 | 29/29 |
| 9 | pick→place | 0.083 | 0.193 | 0.028 | 21/21 |
| 10 | place→pick | 0.120 | 0.297 | 0.030 | 17/17 |

- *max TCP dev* = max distance of the planned TCP path from the TCP
  trace of the straight joint chord between the trajectory's own
  endpoints (gate ≥ 0.02 m; measured min **0.083 m**).
- *min gantry clearance* = min mesh distance (pinocchio/hppfcl) between
  any robot collision mesh and the gantry box over the ≤ 0.05 rad
  densified planned path: **27.6–36.3 mm**, consistent with the XRDF
  sphere pads + 5–10 mm buffers.
- *chord hits* = colliding configs on the straight chord (≤ 0.02 rad
  steps), proven **twice** per trial: for the sampled pair AND for the
  actually-executed endpoints; every colliding pair involves only
  `world/gantry` and the chord is clean in the gantry-removed model.
- Planning 0.155–0.164 s; execution 5.1–11.1 s (time_dilation 0.5);
  tracking ≤ 0.017 rad; 0 pairs rejected by cuMotion.

### Gantry removed — 3/3 PASS (contrast / doc-gate evidence)

| pair | max TCP dev (m) | dev ratio with/without | direct path crosses gantry volume |
|---:|---:|---:|---|
| 1 | 0.006 | 32.2 | yes (20 configs inside) |
| 2 | 0.015 | 12.8 | yes (10 configs inside) |
| 3 | 0.009 | 12.5 | yes (58 configs inside) |

Without the gantry the planner goes essentially straight (≤ 15 mm TCP
deviation vs 112–193 mm for the same pairs with the gantry — a **12–32×**
contrast) and its path passes straight through where the bar was. That
is the "visible re-routing" evidence.

### Resources

Whole-GPU VRAM: 20 MiB baseline → **2557 MiB peak** (stack delta
2537 MiB: Isaac Sim bridge + adapter container + cuMotion container),
same envelope as the M6 run (2561 MiB).

## Method (M6 infrastructure reused unchanged)

Same stack, images, bridge and configs as the passing M6 gate:
`rebot-jazzy-baseline` + `rebot-m6-cumotion` images, M5
`b601_sim_bridge.py`, `config/cell_geometry.yaml`,
`config/rebot_b601dm.xrdf`, FastDDS UDPv4 profile, ROS_DOMAIN_ID 42.
`scripts/m7_gantry_avoidance.sh` mirrors the `m6_cumotion_trials.sh`
flow; `scripts/m7_trial_runner.py` subclasses the M6 `TrialRunner`.

One deliberate difference from M6: goals are **`plan_cspace` joint
goals**, not `plan_pose`. The M7 claim is about the straight joint path
between two known configurations; a pose goal would let cuMotion's IK
pick a different branch and detach the executed motion from the chord
proof. (Measured side effect: goal error is exactly 0.0 rad on all 30
moves.) The gantry-removed world for contrast trials is
`artifacts/m7/cell_geometry_no_gantry.yaml` — the canonical cell file
minus the gantry, nothing else.

## Lesson recorded (first run failed 9/10)

The first gate run failed on one trial with a **phantom 1.4 mm
fingertip "collision"**: the mesh verifier inherited `ik_core.full_q`,
which pins the gripper jaws at 0 (shut), while the sim bridge and
cuMotion's XRDF lock the jaws **open at 0.0715 m** for every trial. The
shut-jaw finger volume never physically existed; at the actual jaw
state the path cleared the bar by > 5 mm, and one sampled chord "hit"
the gantry only with the phantom shut fingers. All M7 mesh checks
(endpoint validity, chord proofs, path recheck, clearance) now run at
the executed jaw state via `OpenJawKin` (`scripts/m7_sample_pairs.py`),
and the full gate was re-run end-to-end. M6 was unaffected (its paths
never approached obstacles within the jaw stroke), but any future
milestone that verifies meshes near obstacles must check at the jaw
state that actually executes.

## Reproduce

```bash
./scripts/m7_gantry_avoidance.sh   # exit 0 iff the M7 gate passes
```

Pipeline: `m7_sample_pairs.py` (host, seed 70100, deterministic) →
stack bring-up (M6 flow) → `m7_trial_runner.py` (in the cuMotion
container) → `m7_verify_trials.py` (host mesh recheck + metrics →
`artifacts/m7/gantry_avoidance.json`).
