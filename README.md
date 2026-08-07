# reBot Arm B601-DM — One Simulated Grasp

Learning project that recreates one complete grasp from Seeed's vendor demo in
Isaac Sim:
<https://wiki.seeedstudio.com/rebot_arm_b601_dm_grasping_demo/>

> **Current objective:** make the shipped B601-DM asset physically open, approach,
> close around one object, and lift it once. [PLAN.md](PLAN.md) puts that result on
> the critical path. The existing perception smoke test remains useful regression
> support; the 300-scene benchmark, ROS/BYOR integration, and publication work are
> deferred until the pick succeeds.

## Critical path

| Stage | Outcome |
|---|---|
| **P0** ✅ | Load the shipped DM USD; verify its 8 driven joints, limits, gains, tracking, and measured finger separation. |
| **P1** ✅ | Close both physical finger joints on one object and validate contacts — first-contact aperture measured at 41.0 mm for a 40.0 mm object. |
| **P2** ✅ | `GRASP → CLOSE → LIFT → HOLD → RELEASE` with fixed joint targets. 59.8 mm rise, 1.0 s hold, falls 321.6 mm on release. |
| **P3** ✅ | The demo's own interface — `move_to_traj` / `open_gripper` / `grasp` / `release_gripper` / `get_tcp_pose` — on the pinned `reBotArm_control_py` Pinocchio IK. Cartesian accuracy 0.13–0.33 mm; pick succeeds under Cartesian command (54.5 mm rise, 6.4 mm slip). See [P3.md](P3.md). |
| **P3b** ✅ | Eye-in-hand `AX=XB` with the vendor's own solver, scored against the true mount: **2.4–4.1 mm / 0.5–1.2°**, end-to-end marker localisation **5–10 mm**. See [P3B.md](P3B.md). |
| **P4** | Feed the same executor an oracle grasp, then a YOLOE-derived grasp. |

A P2 pass requires the object to rise at least 50 mm and remain held for at
least 1 second. Joint motion must use physics drive targets and measured
feedback. Teleporting, parenting, or attaching the object does not count.

**Status: P0, P1 and P2 pass.** The shipped B601-DM asset grasps and lifts an
object through physics — **59.8 mm rise, 1.0 s hold, 3/3 repetitions, 20/20
checks** — see **[PICK.md](PICK.md)** and `artifacts/b601_pick/b601_pick.json`.
Getting there found a fourth asset defect: the finger colliders are convex hulls
that leave only ~24 mm of usable jaw against 143 mm of authored travel. **P3 and P3b also pass** — the arm is commanded by Cartesian TCP poses through the
demo's own Pinocchio IK ([P3.md](P3.md)), and eye-in-hand calibration recovers a
known camera mount to 2.4–4.1 mm ([P3B.md](P3B.md)). **P4 (perception) is the last
step**; it does not pick itself yet.

**Upstream sources are cloned and read.** `upstream.repos` pins `reBot-DevArm-Grasp`,
`reBotArmController_ROS2`, `reBot-Isaacsim`, and **`reBotArm_control_py`** — the demo's own
Pinocchio IK / trajectory / gripper controller, which is the entire robot interface the demo
uses and which had been missing from the pin set. `src/` is gitignored — restore with:

```bash
vcs import src < upstream.repos
```

**BYOR / XRDF / cuMotion / MoveIt are cut, not deferred** (PLAN.md §6). The demo moves the
arm with `move_to_traj` on Pinocchio — no ROS, no MoveIt, no cuMotion anywhere in its control
path — so that work was never on the route to this goal.

## Root-cause analysis: frozen render, joint4 sag, grasp SOTA, prior art

**[ANALYSIS.md](ANALYSIS.md)** — both standing mysteries solved with measured
fixes: the arm rendered frozen because the session repair's `resetXformStack`
disables PhysX→USD writeback on the repaired links (fix: capture-time tensor→USD
sync, `scripts/b601_usd_sync.py`); the imported URDF's joint4 sag was the
importer's `acceleration` drive type starving the low-inertia wrist (fix: flip
drives to `force`, 0.807→0.004 rad). Plus a 14-agent worldwide prior-art sweep:
no one else has published autonomous contact grasping of this arm in Isaac Sim.

## Inspect the arm yourself — keyboard teleop

```bash
TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N DISPLAY=:1 \
  ~/isaaclab-venv/bin/python scripts/b601_teleop.py            # imported URDF
  ~/isaaclab-venv/bin/python scripts/b601_teleop.py --source usd   # shipped USD
```

`1`-`6` pick an arm joint, `7` the gripper, `UP`/`DOWN` move it, `[`/`]` change
step, `O`/`C` open/close, `I` toggles Cartesian IK (`W A S D Q E`), `H` home,
`R` reset, `ESC` quit. Isaac swallows stdout, so watch
`tail -f artifacts/teleop/status.txt`.

A headless self-test drives a scripted sweep instead:
`--headless --selftest 3`.

**Resolved:** the former joint4 sag (~0.81 rad) was the importer authoring
`drive:type="acceleration"`; the teleop now flips imported drives to `force`
and joint4 tracks to 0.03 rad. Full story in [ANALYSIS.md](ANALYSIS.md).

## Regression support: the ten-scene smoke test

Implemented and passing on the Isaac Sim backend. **[SMOKE_TEST.md](SMOKE_TEST.md) has the
measured results**, including the three bugs it caught. It validates perception geometry,
depth handling, dataset replay, and scoring; it does not move the B601, exercise contacts,
or prove a pick.

```bash
./run_smoke.sh            # Isaac Sim backend; installs nothing
./run_smoke.sh analytic   # force the dependency-free backend
```

Needs only Isaac Sim's interpreter (`~/isaaclab-venv/bin/python`) — no ROS 2, no sudo, no
downloads. The chain runs end to end and A1 oracle recovers the grasp to
**0.0004 mm / 0.008°**. Keep this as a regression check while building P0–P4;
do not expand it to 300 scenes before the physical pick works.

| Path | What |
|---|---|
| `grasp_smoke/` | pure library — geometry, grasp, dataset, detect, scorer, overlay. No ROS, no Isaac. |
| `capture/isaac_capture.py` | Isaac Sim 5.1 Replicator capture backend |
| `ros2_iface/` | Jazzy dataset publisher + grasp node — **written, not run** (no ROS 2 here) |
| `tests/` | 104 tests: geometry, pipeline, mocked YOLOE, gate, and ROS message contract |
| `run_smoke.py` / `run_smoke.sh` | the chain, and the one command that reproduces it |

## What the demo is

Seeed's real-hardware grasping pipeline for the reBot Arm B601: RGB-D camera → YOLO
object detection → grasp pose estimation (heuristic, or GraspNet optionally) → eye-in-hand
hand-eye calibration → arm executes the grasp. Configuration lives in `config/default.yaml`
(camera type, detection thresholds, robot params, grasp pipeline settings).

## How it relates to the pick-place project

This is the **vendor reference demo on real hardware** — separate deliverable from the
sim2real portfolio build in `~/docs/sota-gap-plan-aug2026.md`, but the same arm. Useful as:

- a working real-hardware grasping baseline to compare the classical stack against,
- a second implementation of eye-in-hand calibration (the plan calls TF/calibration rigor
  the #1 silent killer), and
- a sanity check that the arm + camera + USB2CAN chain works before any sim2real claims.

## Hardware the page assumes

- reBot Arm B601 (DM or RS)
- RGB-D camera: Orbbec Gemini 2, Intel RealSense D435i, or D405
- USB2CAN serial bridge; 24 V for DM (48 V for RS)
- Ubuntu 22.04+, Python 3.10, x86_64

## Setup, as the wiki states it

```bash
# main repo — note it clones into ./rebot_grasp
git clone https://github.com/Seeed-Projects/reBot-DevArm-Grasp.git rebot_grasp
cd rebot_grasp

# env
conda env create -f environment.yml
conda activate rebotarm

# arm SDK
git clone https://github.com/vectorBH6/reBotArm_control_py.git sdk/reBotArm_control_py
cd sdk/reBotArm_control_py && pip install -e . && cd ../..

# camera SDK — one of:
pip install pyorbbecsdk2      # Orbbec Gemini 2
pip install pyrealsense2      # RealSense D435i / D405

# optional: GraspNet
cd sdk && git clone https://github.com/graspnet/graspnet-baseline.git
```

The page also lists `https://github.com/EclipseaHime017/reBot-DevArm-Grasp.git` as an
alternative development mirror.

## Running it

```bash
python scripts/collect_handeye_eih.py            # hand-eye calibration (required first)
python scripts/collect_handeye_eih.py --manual   # manual calibration mode
python scripts/main.py                           # main grasping pipeline, live preview
python scripts/set.py                            # grasp-and-place
python scripts/object_detection.py               # YOLO detection only
python scripts/ordinary_grasp_pipeline.py        # simplified grasp test, no arm required
python scripts/graspnet_camera_demo.py           # GraspNet camera demo
python scripts/grasp.py --dry-run                # GraspNet + arm, no motion
python scripts/grasp.py --target-class "light blue coffee cup"
```

`ordinary_grasp_pipeline.py` and `grasp.py --dry-run` are the ones that run without the
arm powered — start there.

## Open items on this machine

- **No conda.** `conda`/`mamba`/`micromamba` are all absent, and there's no
  miniconda/anaconda/miniforge install. `environment.yml` won't work as written — either
  install miniforge (user-local, no sudo needed) or translate the env to a venv.
- **Python 3.10** isn't in apt on this box and there's no passwordless sudo — conda or uv
  is the practical route to 3.10.
- `pip install pyorbbecsdk2` but the page's smoke test is `import pyorbbecsdk` — verify the
  actual module name once a camera is on hand. Orbbec also ships a build-from-source SDK.
- Which camera is actually available hasn't been decided — that picks the SDK branch.

**Status of the claims above:** the wiki recipe in this file is Seeed's, reproduced as written
and *not* re-verified end to end. The code itself has been cloned and read, and several details
differ from the wiki summary — see PLAN.md's verification notes. Confirmed directly in the pinned
tree: the grasp geometry in `utils/ordinary_grasp.py` (the approach vector is the camera viewing
ray; the mask/OBB short edge sets the opening axis), `depth_quantile` defaulting to `0.75` in
code but `0.5` in `config/default.yaml:61`, mask **and** OBB use in `utils/yolo_utils.py`, and
the three shipped `.pt` models. The wiki's real-hardware install steps remain unverified —
nothing here has been installed or run.
