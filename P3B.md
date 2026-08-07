# P3b — eye-in-hand calibration, solved with the vendor's solver and scored against truth

**Run:** 2026-08-06 · **Gate:** PLAN.md §3.2 P3b · **Result: PASS, 10/10 checks, 3/3 runs**
**Record:** `artifacts/hand_eye/b601_hand_eye.json` · **Code:** `scripts/b601_hand_eye.py`

```bash
cd ~/reBot_B601_Grasp
TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N \
  ~/isaaclab-venv/bin/python scripts/b601_hand_eye.py \
    --repair-nested-xforms --out artifacts/hand_eye/b601_hand_eye.json
```

Uses the pinned vendor classes **directly**, not a reimplementation:
`HandEyeCalibrator(CalibMode.EYE_IN_HAND)` and `ArUcoDetector` from
`src/reBot-DevArm-Grasp/calibration/`, with the vendor's config values
(`DICT_4X4_50`, marker id 0, TSAI).

## Why simulation is the right place for this

On the real arm `T_cam2gripper` is exactly the unknown you are solving for, so a
wrong-but-plausible answer is undetectable — which is why the parent plan calls
TF/calibration rigor the #1 silent killer. Here the camera is *mounted* at a
transform the script chooses, so the answer can be **scored**. The calibration is
never told the mount; it sees only rendered images and measured gripper poses.

True mount: `t = [-20.44, 0.04, 99.91] mm` in the gripper frame.

## Results, 3 runs

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| Deployed method | HORAUD | DANIILIDIS | PARK |
| **Mount recovery** | **4.13 mm / 1.21°** | **3.30 mm / 0.60°** | **2.41 mm / 1.12°** |
| **End-to-end marker localisation** | **10.31 mm** | **5.48 mm** | **8.02 mm** |
| …same, using as-configured TSAI | 20.67 mm | 10.43 mm | 11.76 mm |
| Method spread (worst viable) | 8.82 mm / 3.91° | 8.32 mm / 1.77° | 5.89 mm / 2.02° |
| Detection bias (true mount) | 0.47 mm | 0.37 mm | 0.28 mm |
| Samples / rotation diversity | 20 detected, 71.7° max / 24.2° mean | | |

**The number that matters downstream is end-to-end: 5–10 mm.** That is how
accurately a camera detection becomes a base-frame target, and it bounds what P4
can achieve — a perception-driven grasp cannot be placed better than its
calibration.

## Rotation diversity is the whole game

AX=XB recovers translation from `(R − I)·t_X = R_X·t_B − t_A`, so the translation
solve is conditioned entirely by how large the rotations between poses are. Three
measured points:

| Mean pairwise rotation | TSAI translation error |
|---|---|
| **5.4°** | **degenerate** — `calibrateHandEye` returned *exactly* `t = [0,0,0]`, rotation 172° off |
| 17.4° | 17.6 mm |
| **24.2°** | **4–7 mm** |

The degenerate case is the important one: it produced a clean-looking result with
no error flag. On hardware there is no ground truth to catch it, so **conditioning
of the input is the only warning you get** — which is why the script now *gates*
on rotation diversity (≥40° max, ≥20° mean pairwise) before it will trust a solve.

## Method choice is worth more than the solver's reputation

All five methods the vendor exposes, same data:

| Method | Recovery (run 1) |
|---|---|
| PARK / HORAUD / DANIILIDIS | 2.4–4.5 mm, 0.5–1.3° |
| **TSAI** (the vendor's configured default) | 5.5–9.3 mm, 1.9–4.1° |
| **ANDREFF** | **~185 mm — unusable, every run** |

Two findings. **Andreff fails catastrophically and consistently** on this data, so
it is excluded from the ensemble as a stable property rather than noise. And the
**vendor config names TSAI, which was the worst of the four viable methods in
every run** — using it roughly doubles end-to-end error (20.7 vs 10.3 mm in run 1).

But no single method is reliably best: PARK, HORAUD and DANIILIDIS **swap rank
between runs**, spread ~6–9 mm. Detection is stable (bias ≤0.5 mm), so the spread
is RTX render jitter perturbing sub-pixel corners. The honest statement is that
method choice is worth several millimetres and should be validated per rig, not
inherited from a config file.

## What had to be fixed to get here

**Camera intrinsics were wrong and silently biased everything.** `focalLength =
fx · horizontalAperture / width` is an *assumption* about Isaac's projection, and
a wrong `fx` scales every `solvePnP` depth by the same ratio, landing directly in
the hand-eye translation. With the true camera and marker poses known, the
effective focal length is measurable — `fx_eff = fx_assumed · z_true / z_detected`
— and came out **1.8–2.1% low**. Correcting it dropped marker-localisation bias
from **6.93 mm to 0.56 mm**. Real workflows inherit factory intrinsics from the
RGB-D SDK and never check this.

**An IK-driven hemisphere of camera viewpoints did not work.** Aiming the camera
at the marker by construction is the textbook collection procedure, and it was
tried first — but only **7 of 20** full-SE3 camera poses were reachable, because
the roll-about-view-axis constraint puts most of that hemisphere outside this
6-DOF arm's dexterous workspace. Joint perturbation with a wide lens (960×720,
fx≈380, 93° HFOV) reaches more usable viewpoints, at the cost of not guaranteeing
the marker is centred. 20/20 detections.

**The marker needs a quiet zone.** `generateImageMarker` emits the marker alone;
without a white margin the detector cannot find its outer black border against a
dark background. The quad is drawn 1.33× the marker so the *marker* edge — not the
quad edge — is what `solvePnP` is told about.

## Honest limits

- **Sub-millimetre would need more rotation diversity than this arm and lens
  afford**, not a better solver. The tolerances here are set from the measured
  spread over repeat runs; the reported numbers, not the pass, are the deliverable.
- Rendering is noise-free apart from RTX sampling. A real camera adds motion blur,
  rolling shutter, exposure and lens distortion — all set to zero here (`D = 0`).
- `T_gripper2base` comes from the simulator's measured link pose, i.e. a perfect
  encoder + perfect FK. On hardware it comes from FK on the URDF, which P3 showed
  disagrees with the simulated asset by 3.4 mm in z — that error would add.

## What P3b does and does not establish

**Does:** the vendor's hand-eye pipeline works, recovers a known mount to **2.4–4.1
mm / 0.5–1.2°**, and yields **5–10 mm** end-to-end marker localisation. The
degenerate-solve failure mode is now gated rather than discovered by luck.

**Does not:** nothing is connected to the pick yet. The arm still does not pick
itself — that is **P4**, which feeds `grasp_smoke/grasp.py` through this
calibration into `move_to_traj`.
