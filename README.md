# reBot Arm B601-DM — Visual Grasping Demo

Working folder for Seeed's vendor reference demo:
<https://wiki.seeedstudio.com/rebot_arm_b601_dm_grasping_demo/>

> **→ [PLAN.md](PLAN.md) is the working document** (rev 2.1 audited): a staged Isaac Sim +
> Isaac ROS, simulation-only build of this demo on the B601-DM, with milestones through Oct.
> The notes below are the wiki's real-hardware recipe, kept for reference.

**Upstream sources are cloned and read.** `upstream.repos` pins `reBot-DevArm-Grasp`,
`reBotArmController_ROS2`, and `reBot-Isaacsim` at the exact commits every claim in PLAN.md was
verified against. `src/` is gitignored — restore with:

```bash
vcs import src < upstream.repos
```

⚠️ **Pinning is incomplete.** `isaac_ros_manipulation` and `topic_based_ros2_control` arrive with
the Isaac ROS container and are **not yet pinned** — record their exact SHAs into
`upstream.repos` when the container workspace is first created. Nothing is *installed* yet (no
ROS 2, no Isaac ROS); that's S0.

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
