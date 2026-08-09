# M8 — Robot segmentation + nvblox ESDF mapping feeding cuMotion

**Date:** 2026-08-09 · **Verdict:** **PASS** (5/5 map configs + all four
mapping/planning phases)
· **Artifact:** `artifacts/m8/mapping_acceptance.json`
· **Design doc gate:** sec. 15.3/15.4 mapping flow + mapping acceptance
test, on the M5/M6/M7 stack
· **Camera decision:** `docs/DDR-002-camera-architecture.md`
(PROVISIONAL — fixed overhead first, wrist D405 deferred)

## What was demonstrated

The full doc-15.3 chain runs in simulation on the passing M5/M6/M7
stack: the Isaac Sim bridge grew a **fixed overhead RGB-D camera**
(RGB + 32FC1 depth + two camera_info topics + `/tf_static`, ~10 Hz,
`config/sim_topics.yaml`), the **cuMotion robot segmenter**
(`isaac_ros_cumotion_robot_segmenter`, apt, 4.5) masks the arm out of
that depth using `/joint_states` + the XRDF spheres, **nvblox**
(`ros-jazzy-isaac-ros-nvblox`) builds a workspace-bounded TSDF/3D-ESDF
from the robot-masked depth only, and the **cuMotion planner** consumes
the ESDF as its collision world (`read_esdf_world:=true` + per-goal
`update_esdf`). All doc-15.4 criteria passed in one gated run:

1. **Robot body absent** — the arm was driven through 5 distinct
   configurations (cuMotion plans, executed via the M5 FJT chain,
   tracking ≤ 0.011 rad); at every held config the ESDF contained **zero
   occupied voxels inside the robot's XRDF sphere volume (+1 cm)**.
2. **Gantry present** — the crossbar showed up as **273 occupied voxels**
   in its (1-voxel-dilated) box at every single check, unchanged while
   the arm moved.
3. **Stale map clears / rebuilds** — hiding the sim gantry dropped its
   voxels **273 → 0 in 3.1 s**; restoring it rebuilt **0 → 273 in
   3.0 s** (scene-command file → bridge visibility toggle → depth →
   TSDF carving/decay → ESDF).
4. **cuMotion routes around a MAPPED (never statically declared)
   obstacle** — a fresh M7-style pose pair whose straight joint chord
   provably hits ONLY the gantry (19 colliding chord configs, mesh
   proof at the executed endpoints) was planned with a **table-only**
   static world: the planner re-routed with **max TCP deviation
   0.270 m**, mesh-verified collision-free with **35 mm** min clearance
   to the TRUE bar box, executed and converged (goal error 0, tracking
   0.010 rad). After the map cleared, the SAME goal with the SAME
   static world went **near-straight (0.010 m deviation, 26× contrast)**
   straight through the old gantry volume (10 densified configs inside
   it) — the re-route is attributable to the map and nothing else.

## Results (run of 2026-08-09, `artifacts/m8/`)

### Map phase — 5/5 PASS

| cfg | TCP (m) | robot voxels in map | gantry voxels | robot px masked | tracking (rad) |
|---:|---|---:|---:|---:|---:|
| 1 | (0.51, −0.19, 0.39) | 0 | 273 | 19 793 | 0.010 |
| 2 | (0.47, −0.19, 0.27) | 0 | 273 | 19 033 | 0.007 |
| 3 | (0.23, 0.25, 0.24) | 0 | 273 | 20 419 | 0.006 |
| 4 | (0.16, 0.08, 0.40) | 0 | 273 | 16 830 | 0.010 |
| 5 | (0.55, −0.27, 0.16) | 0 | 273 | 20 614 | 0.011 |

Map totals per query: ~7 610 occupied / ~55 250 unknown / 383 116 voxels
(76×71×71 @ 1 cm over the doc-15.5 workspace). The "display" evidence
(doc 15.4) is archived per config as raw-depth / robot-mask /
masked-depth arrays + rendered PNGs under `artifacts/m8/evidence/`
(mask semantics: 32FC1 binary, **0 = robot pixel**; the masked-depth
invalidates exactly the mask-0 pixels — 0 px mismatch).

### Mapped-obstacle avoidance (D1) vs cleared-map contrast (D2)

| phase | static world | gantry in map | planning (s) | max TCP dev (m) | outcome |
|---|---|---:|---:|---:|---|
| D1 | table only | 273 voxels | 0.203 | **0.270** | re-routes; 35 mm mesh clearance to true bar; collision-free (full cell) |
| D2 | table only | 0 voxels | 0.230 | **0.010** | near-straight THROUGH old bar volume; collision-free (no-gantry cell); dev ratio **26.0** |

### Resources — DDR-001 stage-2 checkpoint

Whole-GPU VRAM: 20 MiB baseline → **3 524 MiB peak** (stack delta
3 504 MiB). Stage 1 (M6/M7) peaked at ~2 557 MiB; segmentation + nvblox
+ the overhead RTX camera add ≈ **950 MiB**. The first relief valve
(voxel 0.01 → 0.02, reduced camera resolution) was **not needed**;
voxel_size stays at the doc value 0.01 with ~12.8 GiB of headroom on
the 16 GiB gate.

## Stack

* **Host bridge:** `scripts/b601_sim_bridge.py --m8-scene` — M5 bridge
  + visible cell geometry (true-z table, gantry bar, floor; VISUAL only,
  no colliders) + overhead camera per DDR-002 (640×480, ≈67° HFOV, eye
  (0.275, 0, 1.10), straight down, optical-frame prim construction from
  the rearm wrist camera; depth validated pixel-exact at bring-up) +
  `ROS2PublishRawTransformTree` static TF + a polled scene-command file
  for gantry removal. M5/M6/M7 behavior without the flag is untouched.
* **Container 1:** `rebot-jazzy-baseline` — adapters + sim-JTC shims,
  unchanged.
* **Container 2:** `rebot-m8-nvblox` = `rebot-m6-cumotion` + apt
  `ros-jazzy-isaac-ros-cumotion-robot-segmenter` +
  `ros-jazzy-isaac-ros-nvblox` (committed locally, same pattern as the
  M6 image). One `component_container_mt` runs RobotSegmenter →
  NvbloxNode → StaticPlanningSceneServer + CumotionPlanner.
* nvblox: `global_frame base_link`, `voxel_size 0.01`, `esdf_mode 3d`,
  depth-only, `workspace_bounds_type bounding_box` min
  (−0.10, −0.35, −0.05) max (0.65, 0.35, 0.65) — verified applied via
  `ros2 param get` (the startup "kUnbounded not recognized" warning is
  the unused *dynamic* mapper defaulting, not the static mapper).
* FastDDS UDPv4 profile + `ROS_DOMAIN_ID=42` on every side; Isaac Sim
  owns `/clock`; all container nodes `use_sim_time:=true` (M5 rules).

## Lessons recorded (first gate run failed, root-caused, no workarounds)

1. **The 4.5 planner refreshes its ESDF only when the MotionPlan GOAL
   asks** (`bool update_esdf` in `MotionPlan.action`). With only the
   node parameter `update_esdf_on_request:=true`, the planner called the
   nvblox service exactly once at startup ("Initialized grid from
   nvblox" appears a single time in the log) and planned every later
   request against that stale first grid — measured: after nvblox
   cleared the gantry, the contrast plan still re-routed with dev ratio
   1.01. The frozen M6/M7 runners never set the field (harmless there:
   `read_esdf_world` was false). The M8 runner sets `update_esdf=True`
   on every goal; with it, D2 went near-straight (ratio 26.0). Upstream
   context: cuMotion uses the ESDF at planning time only
   ([isaac_ros_cumotion docs](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion/index.html),
   [isaac_ros_cumotion#45](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion/issues/45)).
2. **`robot_mask` is 32FC1, not mono8** — a binary float image with
   0 = robot; the first runner attempt parsed it as uint8 and crashed.
   The runner now decodes by the `encoding` field.
3. The overhead camera sees only a **shell** of the gantry (top +
   visible faces); its occluded interior stays unknown. The gates
   therefore verify clearance against the TRUE bar box on the canonical
   meshes (M7 `Metrics`, jaws at the executed open state), not against
   the mapped shell — recorded in DDR-002 consequences.
4. ESDF service semantics pinned by probe: `Float32MultiArray` dims
   (x, y, z), unknown voxels encoded as **−1000.0** (excluded from the
   occupancy analysis; cuRobo tolerates them — plans succeed with the
   arm inside unknown space).

## Reproduce

```bash
./scripts/m8_mapping_acceptance.sh   # exit 0 iff the M8 gate passes
```

Pipeline: `m8_select_configs.py` (host, seed 80100: 8 mesh+sphere-valid
map configs, 3 chord-proven gantry-blocked pairs, world-frame sphere
sets) → stack bring-up (M6/M7 flow + `--m8-scene` + M8 container) →
`m8_acceptance_runner.py` (in-container phases A/B, D1, C, D2, C2) →
`m8_verify_mapping.py` (host: doc-15.4 gates + mesh recheck of both
D-phase trajectories + VRAM → `artifacts/m8/mapping_acceptance.json`).
