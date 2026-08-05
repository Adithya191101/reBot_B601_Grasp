# reBot B601-DM Visual Grasping — Isaac Sim + Isaac ROS, simulation only

**Written:** 2026-08-04 · **Rev 2.1 (audited)** — evaluation stratified, capture/replay contract
fixed, release-candidate window reserved
**Parent plan:** `~/docs/sota-gap-plan-aug2026.md` (this is an expanded, learning-first P1)
**Reference demo:** <https://wiki.seeedstudio.com/rebot_arm_b601_dm_grasping_demo/>

> **Revision history — what each pass got wrong, so it isn't relearned:**
> **rev 1** treated the B601 swap as URDF→USD + XRDF and put it on the Aug 29 critical path.
> **rev 2** corrected the scope (BYOR = two authored packages), moved B601 to September, and
> dropped the wrong "swap RT-DETR for YOLO" one-liner.
> **rev 2.1** corrected the *measurement* — rev 2 scored the approach vector, which in Seeed's
> code is just the camera viewing ray — plus: a **DM USD already exists**, the MJCF parity model
> is the **RS** arm, and BYOR needs a **framework extension** because `RobotType` is closed.
> **rev 2.1 audited** splits oracle scoring into A0/A1/A2 (a single oracle bar conflated a code
> bug with legitimate quantile/perspective error), fixes the capture→replay contract, decouples
> M2 from the unvalidated DM USD, reserves Aug 21–28 for release, and puts M3 fully after Aug 29.

---

## 0. The idea in one line

Seeed's demo is a hand-rolled Python grasping pipeline on real hardware. NVIDIA's *Isaac for
Manipulation* is the same pipeline shape, GPU-accelerated, in ROS 2, in Isaac Sim.
**Start from what already runs, substitute one piece at a time.**

| Seeed demo (real, Python) | Your version (sim, ROS 2) | Difficulty |
|---|---|---|
| Orbbec / RealSense RGB-D | Isaac Sim camera → `isaacsim.ros2.bridge` | easy |
| YOLO **seg + OBB** (`yoloe-26s-seg.pt`, `yolov8s-world.pt`) | see §8 — **not** a drop-in | medium |
| min-area-rect short edge → **opening axis** | same maths, ROS node | medium |
| `calibration/hand_eye.py` + `aruco_pose.py` | known extrinsics in sim → solve `AX=XB` anyway | easy |
| `reBotArm_control_py` (Pinocchio IK) | `isaac_ros_cumotion` + MoveIt 2 + `ros2_control` | **hard** (§6) |
| `scripts/main.py` orchestration | `isaac_ros_manipulation_orchestration` (BT) | medium |

## 1. What sim-only does and doesn't buy you

Sim-only fully closes **G2** (perception → grasp) and **G3** (C++). It does **not** close **G1**
— a sim2real delta needs hardware.

⚠️ Going to hardware does *not* swap "exactly one layer." It changes the `ros2_control` hardware
interface **and** the camera source (real sensor noise, exposure, rolling shutter), hand-eye
calibration (measured, not known), frame definitions, the session zeroing procedure, and
possibly the controller interface. The sim work removes most of the risk — but budget the
transfer honestly rather than as a config swap.

---

## 2. Verified environment (checked 2026-08-04)

| Thing | Status |
|---|---|
| GPU | RTX 5000 Ada Laptop, **16 GB VRAM** — binding constraint, §10 |
| Driver | **580.173.02** ✅ meets Isaac ROS 4.5's 580+/CUDA 13.0+; avoids the 595 `rtx.scenedb` crash. **`apt-mark hold` today.** |
| OS | Ubuntu 24.04.4 ✅ |
| Isaac Sim | **5.1.0.0**, pip in `~/isaaclab-venv` (Python **3.11.15**) ✅ matches the cuMotion tutorial |
| ROS 2 bridge | present, both `humble/` and `jazzy/` lib dirs ✅ |
| ROS 2 | **not installed** — S0 |
| Docker | 29.5.0, user in `docker` group ✅ — the install path (§4) |
| Upstream repos | cloned into `src/`, pinned in **`upstream.repos`** |

**The distro-split landmine is gone.** Isaac ROS **release-4.5** (2026-07-06): *"All Isaac ROS
packages are designed and tested to be compatible with ROS 2 Jazzy."* Parent plan §9.3 is stale.

### ⚠️ Pinning is incomplete — two SHAs still to record

`upstream.repos` pins the three Seeed repos at the commits every fact in this document was read
from. **`isaac_ros_manipulation` and `topic_based_ros2_control` are not pinned** — they arrive
with the container in S0. **Record their exact commit SHAs into `upstream.repos` when the
container workspace is first created**, and re-verify the `RobotType`/`GripperType` quotes in
§6.1 against the SHA you actually get. Do **not** invent or guess SHAs; leave the entries absent
until they are read from the real workspace.

**Two process rules.** (1) Never `import rclpy` inside Isaac Sim's interpreter — Isaac Sim is
Python 3.11, Jazzy is 3.12; the bridge is OmniGraph nodes, your ROS nodes are separate
processes. (2) Cyclone DDS + matching `ROS_DOMAIN_ID` everywhere, host and container — Isaac Sim
defaults to FastRTPS and a mismatch looks exactly like a broken pipeline.

---

## 3. Milestones

**Hard rule: something ships Aug 29, and it is M2.**

| # | By | Deliverable | Status |
|---|---|---|---|
| **GO/NO-GO** | **Aug 8** | End-to-end 10-scene chain passes (§5.2.1) | **gate** |
| **M1** | Aug 10 | Vendor code read (§5.1); vendor MoveIt demo runs (mock hw); Isaac ROS container up; S0 gate passed; **Branch B model frozen (§5.2.5)** | required |
| **M2** | **Aug 20** | Free-camera scene (§5.2.6) + 300-scene dataset + offline `RGB-D → grasp PoseStamped`, all strata evaluated (§5.2) | **required — load-bearing** |
| **RC** | Aug 21–28 | Release-candidate window (§5.4) — clean-container repro, locked-test run, checksums, CI, licensing, docs, video, buffer | required |
| **M4** | **Aug 29** | **Publication: demo video + public repo. Start applying.** | required |
| **M3** | post-Aug 29 | Stock reference pick-and-place in Isaac Sim (soup can) | **may not start until the Aug 29 RC reproduces cleanly** |
| **M5a** | Sep 5 | DM USD validated + **ROS 2 Action Graph** + raw joint round-trip + `TopicBasedSystem` | |
| **M5b** | Sep 12 | B601 **framework extension** + two packages + MoveIt/OMPL execution | |
| **M5c** | Sep 19 | Flattened URDF + XRDF + **cuMotion** | |
| **M6** | Sep 26 | Active two-finger gripper, contact, attachment, authored grasps | |
| **M7** | Oct 10 | Honest depth, Replicator-trained perception, eval protocol, C++ node, writeup | |

**M3 is out of August entirely.** It is not a fallback and not a stretch task — it may begin
**only after the Aug 29 release candidate has been reproduced cleanly from a fresh clone and
container**. When it does start it competes directly with M5a; take it only if it doesn't move
Sep 5.

**Why M2 moved to Aug 20:** rev 2.1 added an Action Graph to S0 and stratified evaluation, a
300-scene capture pipeline, a debug overlay, and confidence intervals to M2 — on a box with no
ROS installed, in the week the parent plan already called the riskiest. The dataset does **not**
shrink: scene count is compute, not your time, and cutting a preregistered test set under
deadline pressure is what would actually damage the result.

M5 is split into three because "cuMotion by Sep 12" was not credible — the Flexiv template
clarifies the sequence but doesn't compress it.

---

## 4. S0 — Environment (Aug 4–6)

1. `sudo apt-mark hold` the NVIDIA driver packages.
2. ROS 2 Jazzy + `ros-jazzy-rmw-cyclonedds-cpp` on the host.
3. **Isaac ROS 4.5 via Docker** — NVIDIA's recommendation, your runtime already works, and §6
   means building against the full Isaac ROS tree. Host Isaac Sim ↔ container over DDS:
   Cyclone + same `ROS_DOMAIN_ID`, host networking. **Record the workspace SHAs (§2).**

### The S0 gate — `ros2 topic list` is not sufficient

An arbitrary Isaac Sim scene publishes **nothing**. Topics only exist if the scene has an
explicit ROS 2 Action Graph. Build one (clock, camera helper, TF, joint states), then verify
**from inside the container**:

- [ ] `/clock` publishing, and `use_sim_time` respected by a test node
- [ ] **actual RGB and depth messages arriving** (`ros2 topic hz`, not just listed)
- [ ] `CameraInfo` present, with K matching the render resolution
- [ ] **TF resolvable at the image timestamp** — not just "TF exists"
- [ ] rates and **QoS profiles** compatible (sensor-data QoS vs default is a classic silent drop)

---

## 5. S1–S3 — the August path

### 5.1 · S1 — Read the vendor code, run the vendor demo (Aug 4–10) → **M1**

Cloned in `src/`, pinned in `upstream.repos`. **Read in this order**, producing a short
**frame / unit / interface note** as you go (this note is the spec for M2):

1. `utils/ordinary_grasp.py` — the grasp geometry, the important one
2. `utils/transforms.py` — frame conventions; compare against ROS TF2
3. `scripts/main.py` — the orchestration you'll replace with a behavior tree
4. `utils/camera_utils.py` — intrinsics, alignment, depth units
5. `calibration/hand_eye.py` (+ `aruco_pose.py`) — the `AX=XB` solve

**Defer `scripts/grasp.py`** — the separate GraspNet route, not the baseline pipeline.

Then run the vendor MoveIt demo:
```bash
ros2 launch rebotarm_moveit_config demo.launch.py
```
⚠️ Verified: it drives `mock_components/GenericSystem`
(`rebotarm_moveit_config/config/rebotarm.ros2_control.xacro:8`) — virtual hardware only, it will
**not** drive Isaac Sim. Run it anyway to see the planning group, SRDF, and joint limits.

### 5.2 · S2 — The offline perception slice (**Aug 10–20**) → **M2** ← *the deliverable*

> **recorded RGB-D → `geometry_msgs/PoseStamped` grasp pose**, validated offline.

No arm, no planner, no orchestration, no live simulator. **Freeze the evaluator before writing
the node** — object, grasp GT, output frame, depth units, timestamp policy, dataset seeds,
model, and pass/fail accounting all fixed in writing first.

#### 5.2.1 · The Aug 8 go/no-go — an end-to-end chain, not a capture test

Ten scenes must traverse the **entire** chain before any of the rest is committed to:

```
randomize → capture → serialize → replay → oracle mask → predicted mask
          → PoseStamped → evaluator
```

A capture script that writes files but whose output can't be replayed, or whose masks can't be
scored, has not passed. **Record and check:**

| Measure | Why |
|---|---|
| wall-clock **time per scene** | ×300 must fit inside Aug 10–20 with room for reruns |
| **disk per scene** | ×300 against free space (660 GB today) |
| **peak VRAM** | 16 GB is binding (§10); capture must coexist with whatever else runs |
| explicit **extrapolation to 300** | write the arithmetic down; if it doesn't fit, cut scope on **Aug 8**, not Aug 19 |

⚠️ Your note that `distance_to_image_plane` has hung in headless capture scripts applies to
exactly this pipeline. The hedge was that it was an Isaac Lab observation, but a standalone
Isaac Sim script using the same annotator can stall the same way. **Find that out on Aug 8 with
10 scenes, not on Aug 19 with 300.**

**Depth fallback ladder — in order, do not skip a rung:**
1. **Prefer `distance_to_image_plane`** — it is already optical-axis Z, which is what the
   pinhole back-projection in `ordinary_grasp.py` assumes.
2. If it stalls, **test headful capture** before changing annotators. The stall is a headless
   symptom; keeping the annotator and losing headless is the cheaper trade.
3. Only then fall back to **`distance_to_camera`**, and only with an **explicitly tested
   radial-range → optical-Z conversion**:
   `Z = r / sqrt(1 + ((u−cx)/fx)² + ((v−cy)/fy)²)`.
   Treat this conversion as code under test — an A0 fixture (§5.2.3) must pass with it in the
   loop. Silently feeding radial range into a pinhole back-projection produces a smooth,
   plausible, entirely wrong depth field that grows worse toward the image edges.

#### 5.2.2 · Capture / replay contract

**The dataset is immutable and file-native.** Not a bag as the source of truth.

- **Per frame on disk:** RGB, **optical-axis Z depth**, `CameraInfo`, and the TF snapshot needed
  to resolve the camera pose at that frame's timestamp.
- **Ground truth lives in scorer-only sidecars, keyed by timestamp.** The GT sidecar is never
  published on a topic the perception node can subscribe to. This makes GT leakage structurally
  impossible rather than a matter of discipline.
- **A deterministic Jazzy dataset publisher** replays the dataset: RGB, optical-axis Z depth,
  `CameraInfo`, TF. Same input → same output, every run, with `use_sim_time` and the recorded
  stamps.
- **MCAP is a derived artifact**, generated from that replay for the demo video and for anyone
  who wants to `ros2 bag play` it. It is downstream of the dataset, never upstream.

Why this way: a bag-native dataset makes the scorer and the perception node share a bus, which
is how GT leaks; and bag replay timing is not deterministic enough to be a regression test.

#### 5.2.3 · What to measure — the opening axis, not the approach

Verified in `utils/ordinary_grasp.py`: `approach = _normalize(-position)` — that is just the
**camera viewing ray**, carrying almost no information from perception. The mask/OBB result
determines `open_axis`, from the min-area-rect short edge, orthogonalized against the approach.
**So score the opening axis:**

$$\theta_{\text{open}} = \cos^{-1}\!\left(\left|\hat{o}\cdot o^{*}\right|\right)$$

The absolute value is required — a parallel-jaw gripper is symmetric under 180°, so ô and −ô are
the same grasp. Approach-vs-object-surface-normal is a **secondary diagnostic**, not the
headline metric.

**Ground truth must be a *grasp*, not an object pose.** The object origin is not the intended
contact point. **Author a `grasp_gt` child transform** on the object and compare the predicted
contact pose to it **at the image timestamp**.

**Object choice:** one **flat-topped, non-square, textured** object. A cube, cylinder, or soup
can is orientation-degenerate — the opening axis is unidentifiable and the angular metric becomes
noise.

#### 5.2.4 · Strata — A0, A1, A2, B

Rev 2.1 had a single "oracle" branch with a sub-millimetre bar. That conflated two different
things, and would have flagged correct code as broken.

| Stratum | Input | What it actually tests | Bar |
|---|---|---|---|
| **A0 · analytic red tests** | **fronto-parallel, constant-depth synthetic fixtures** — no renderer | pure geometry: units, frame conventions, back-projection, min-area-rect maths | **strict ≤1 mm / ≤1°** |
| **A1 · oracle sim baseline** | GT mask + clean optical-Z depth, nominal camera | the real pipeline with perception removed | ≤3 mm / ≤3° (see caveat) |
| **A2 · oblique stress strata** | as A1, at **predefined camera tilts** | degradation with viewing angle | **reported per tilt, not pooled** |
| **B · predicted** | frozen model mask + the same depth | the actual RGB-D perception slice | median ≤5 mm p90 ≤10 mm; median ≤5° p90 ≤15°; recall ≥95 % |

⚠️ **An A1 or A2 failure is not automatically an implementation bug.** Two legitimate sources of
error exist even with a perfect mask and perfect depth:

- **The depth quantile.** `estimate_grasp()` takes `z_m = quantile(depth_values, q)` over the
  mask — a *single scalar depth for the whole grasp*. (Code default `depth_quantile=0.75`;
  `config/default.yaml:61` overrides it to `0.5`. **Record which you use** — it changes the
  numbers.) On any object with extent along the viewing axis, that scalar is a deliberate
  approximation, not an error.
- **Perspective projection.** The opening axis is recovered by back-projecting a 2-D short-edge
  direction at a single depth. Under perspective, the true 3-D opening axis of a tilted object
  is not the back-projection of its image-plane short edge, and the discrepancy grows with tilt.
  That is exactly what A2 is designed to expose.

So: **A0 failing means your code is wrong.** A1/A2 failing means either your code is wrong *or*
the vendor algorithm's approximations are showing — and distinguishing those two is a genuine
result worth a paragraph in the writeup, not a bug to paper over.

**Report** conditional pose errors (given a true positive) *and* end-to-end within-tolerance
yield *and* false-positive rate. **Print confidence intervals.**

⚠️ These are **engineering targets, not error budgets derived from hardware.** Do *not* justify
them with the arm's ±0.2 mm repeatability — that's joint repeatability and has nothing to do
with visual pose error. If a target is missed, **the August artifact still ships**, with an
honest failure analysis. A missed preregistered target you can diagnose beats a target invented
afterwards.

#### 5.2.5 · Branch B must be frozen by Aug 10 — blocking M1 decision

"Model mask" is not a specification. **By Aug 10, commit in writing to:**

- [ ] exact **segmentation model and checkpoint** (file, version, hash) — e.g. one of the shipped
      `yoloe-26s-seg.pt` / `yolov8s-world.pt`, or an Isaac ROS route from §8
- [ ] exact **prompt / class list** (open-vocabulary models make this part of the model)
- [ ] **confidence threshold**
- [ ] **IoU matching rule** — mask IoU **≥ 0.5** = true positive, unless changed here before
      any measurement

**If this cannot be decided by Aug 10, it is a blocking M1 item** — escalate it rather than
starting M2 with it vague. Every number in the artifact is conditioned on these four choices; a
threshold picked after seeing results is the exact failure preregistration exists to prevent.

#### 5.2.6 · The camera is free-floating — not on the arm

⚠️ **The M2 camera must be a free simulated camera with a synthetic `base → camera` transform.**
It must **not** be mounted on, or depend on, the B601-DM USD — that asset is unvalidated until
M5a (§6.0), and it carries an unresolved π-rotation question (§6.4). Coupling M2 to it would put
the Aug 29 artifact behind a September validation task.

Publish the synthetic extrinsic as a normal TF so the pipeline is frame-correct and wrist-mount
migration is a transform change later, not a rewrite.

#### 5.2.7 · Build order

```
pure tested library  →  dataset publisher (replay)  →  PoseStamped + debug overlay  →  evaluator
```

The **debug overlay** renders mask, min-area rectangle, opening axis, predicted grasp, and GT
grasp. It is the highest-value debugging artifact in the project and it is also your demo video.

### 5.3 · S3 — Stock reference pick-and-place → **M3, post-Aug 29 only**

Not part of the August path (§3). When it starts:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch isaac_ros_manipulation_bringup workflows.launch.py \
   manipulator_workflow_config:=${ISAAC_ROS_MANIPULATION_WORKFLOW_CONFIG_DIR}/sim_launch_params.yaml
```
⚠️ **Start with the soup can.** Isaac ROS 4.5 known issue: *"the pick-and-place workflow may
fail to pick the mac-and-cheese object, `object_class: 22`, with the Robotiq 2F-140 gripper."*
⚠️ Stated test platform is Jetson AGX Thor; issue **#22** (open, Thor, 4.1–4.2) reports it
failing after working in 4.0. Fallbacks: Pose to Pose Planning, Object Following.

### 5.4 · RC window — Aug 21–28, reserved

Not slack. This week has its own deliverables, and M2 work does not spill into it:

- [ ] **Clean-container reproduction** — fresh clone, fresh container, `vcs import`, full run
- [ ] **Locked-test execution** — the 200-scene set, touched once, with **checksums** recorded
      for dataset and results
- [ ] **CI + results artifact** — the A0 fixtures run automatically; results committed
- [ ] **Licensing decisions** — this repo's license, and whether model weights and the dataset
      can be redistributed (check each model's license individually before publishing weights)
- [ ] **Data-distribution decision** — full 300-scene dataset, a sample, or generation script only
- [ ] **Documentation** — README, reproduction steps, metric definitions
- [ ] **Video** — from the debug overlay
- [ ] **Buffer** — deliberately unallocated

---

## 6. S5 — Bring the B601-DM in (Sep) → **M5a/b/c**

### 6.0 Assets — select a canonical source, then validate. Do *not* start by converting.

⚠️ **A DM USD already exists** — `src/reBot-Isaacsim/usd/reBot_B601_DM/reBot_B601_DM.usda`, with
`payloads/` for base, robot, geometries, materials, and physics (`physx`, `physics`, `mujoco`).
The "RS-variant only" claim is **false** for the current tree. The task is
**select → validate → patch or regenerate only if required**, not convert.

⚠️ **The DM USD has no ROS 2 Action Graph.** Verified — no OmniGraph/ROS2Publish prims. Author
it: **clock, joint-state publisher, joint-command subscriber, articulation controller.** Without
it `TopicBasedSystem` has nothing to talk to. This is **M5a**, prerequisite to everything else.

⚠️ **The MJCF parity model is the RS arm, not DM.** `mjcf/rebot_devarm/rebot_devarm.xml` uses
joint classes `rs06`/`rs00`, gripper joints `joint_left`/`joint_right`, joint4 range
`-1.79 1.69` (DM URDF: −1.87). **Adapt the technique; it is not DM ground truth.**

Also verify actuator limits (URDF claims 50 rad/s joints 1–3, 200 rad/s joints 4–6 — 200 rad/s ≈
1,900 RPM at the joint) against Damiao DM4310/DM4340P datasheets, override, and document the
override. `joint4` lower limit −1.87 rad.

### 6.1 The framework extension (M5b)

⚠️ Isaac ROS 4.5 **hard-codes** supported robots and grippers —
`isaac_ros_manipulation_ros_python_utils/manipulator_types.py` @ `release-4.5`:

```python
class RobotType(ManipulatorEnum):
    UR = 'UR'
    FLEXIV = 'FLEXIV'

class GripperType(enum.Enum):
    ROBOTIQ_2F_140 = 'robotiq_2f_140'
    ROBOTIQ_2F_85  = 'robotiq_2f_85'
    GRAV           = 'grav'
```

Closed enums, no dynamic registration. The B601 needs a **third piece of work beyond the two
packages**: extend `RobotType`/`GripperType` and the `DriverConfig` dispatch in `config.py`
(including `get_gripper_type()`). Decide early: **fork** or **patch overlay** — and say which in
the writeup. **Re-verify this quote against the SHA the container actually ships (§2).**

### 6.2 The two packages

**`isaac_ros_manipulation_b601_robot_description/`** (config) — `urdf/b601.xacro` with the
**`TopicBasedSystem`** ros2_control plugin mapping `joint_commands_topic`/`joint_states_topic`
to Isaac Sim; `srdf/`; `config/` for `initial_positions`, `joint_limits`, `kinematics_sim`,
`moveit_sim_controllers`, `ros2_control_controllers_sim`.

**`isaac_ros_manipulation_b601_driver_utils/`** (Python) — `config.py` subclassing
`DriverConfig`; `robot_description.py`; `b601_driver_utils.py` subclassing `RobotControllerBase`
(`get_robot_state_publisher()`, `get_moveit_group_node()`, `get_robot_control_nodes()`);
`launch/b601_driver.launch.py`; `params/b601.yaml`; `src/isaac_sim_joint_parser_node.py`.

Plus routing (`robot_launch_file_path` in the workflow YAML, `package.xml` dep in
`isaac_ros_manipulation_bringup`) and a launch test.

✅ **Your template**: `isaac_ros_manipulation_robots/` ships
`isaac_ros_manipulation_flexiv_driver_utils` + `isaac_ros_manipulation_flexiv_robot_description`
— a two-package third-party integration on an arm you know from `~/Flexiv_RL` (and `GRAV` is the
Flexiv gripper, so the gripper path is worked too). `isaac_ros_manipulation_ur_isaac_sim_utils`
is worth reading for Isaac Sim joint-state filtering. **Read both Flexiv packages end to end
before writing B601 code.** No Franka package exists here.

### 6.3 XRDF + cuMotion (M5c)

Flattened URDF + `b601_gripper.xrdf`: c-space joints with acceleration and jerk limits, tool
frames, per-link collision spheres, self-collision ignore rules. Use Isaac Sim's **Robot
Description Editor**. **Only after §6.4 is settled.**

### 6.4 ⚠️ The π gripper-mount question — verified, unresolved

| Source | `gripper_joint` origin |
|---|---|
| `src/reBot-Isaacsim/urdf/reBot_B601_DM/urdf/reBot_B601_DM.urdf:251` | `xyz="0 0 0.15971" rpy="3.1416 -1.5708 0"` |
| `src/reBotArmController_ROS2/.../reBot_B601_DM_with_gripper.urdf:435` | `xyz="0 0 0.15971" rpy="0 -1.5708 0"` |

**Roll differs by π. Everything below the mount is byte-identical** — verified: both trees have
`gripper_joint1 → gripper_left` at `xyz="-0.042091 2.7531E-05 -1.3031E-05" rpy="0 0 -1.5708"`
and `gripper_joint2 → gripper_right` at `xyz="-0.042091 -2.7531E-05 1.3031E-05" rpy="0 0 1.5708"`,
both `axis="1 0 0"`, both `limit upper="0.0715"`. **The mesh sets differ entirely** — vendor uses
`meshes_b601_gripper/gripper_left.STL` / `gripper_right.STL`; the Isaac-sim tree uses
`meshes/pla7_black.STL`, `pla7_green.STL`, `Rcnc.STL`, `Rpla.STL`.

**How to investigate — joint and link origins are not sufficient.** Identical joint origins under
a π-rotated mount is precisely the case where origin comparison tells you nothing.

1. Compute **physical finger-pad landmark positions in world coordinates** — actual contact-face
   points, taken from the meshes, not link frames.
2. Do it across **several arm configurations**, not one — a single pose can be accidentally
   symmetric.
3. Do it at **closed, mid, and open apertures** — a mount rotation and a finger-direction
   convention can cancel at one aperture and not others.
4. **Compare the pad set as an unordered pair.** If the unordered geometry matches and only the
   left/right *labels* are exchanged, this is a **convention difference, not a defect** — pick
   one convention, document it, and move on. Only a genuine mismatch in unordered pad geometry
   is a bug to fix.

Settle this before defining TCP frames or authoring the XRDF. Either outcome is a writeup
paragraph.

---

## 7. The gripper (M6)

Isaac orchestration expects a **`control_msgs/GripperCommand` action**. Verified in the vendor
URDF: `gripper_joint` (:436) is **`type="fixed"`** — a mount, not actuated, **not a mimic joint**
(no `mimic` tags anywhere in the file). Only **`gripper_joint1`/`gripper_joint2`** (prismatic,
:489/:547, `upper="0.0715"`) are actuated, and being un-mimicked they must **both** be commanded.

Needed: a gripper action controller/adapter mapping one commanded width → two prismatic joints;
contact and friction setup in the USD (defaults will drop objects and look like a planning bug);
attach/detach handling during transport; grasp authoring via `Isaac Utils → Grasp Editor` →
`isaac_grasp` YAML. Documented examples are parallel-jaw grippers like the 2F-140.

## 8. Perception — why YOLO is not a drop-in

`isaac_ros_yolov8` publishes `vision_msgs/Detection2DArray` — **axis-aligned boxes only, no
masks, no oriented boxes** — and isn't among the workflow's standard detector configs (which use
`RTDETR`), so it needs a custom adapter and launch graph. Seeed's `utils/yolo_utils.py` uses
`result.masks` **and** `result.obb`, shipping `yoloe-26s-seg.pt` and `yolov8s-world.pt`. **The
mask/OBB short edge is where the opening axis comes from** (§5.2.3) — an axis-aligned box cannot
produce it.

| Want | Isaac ROS route |
|---|---|
| mask → opening axis (closest to Seeed) | detector box → **`isaac_ros_segment_anything2`** (box-prompted) → mask → min-area rect |
| full 6D pose (stronger) | detector + mask → **`isaac_ros_foundationpose`** |
| open-vocab (`--target-class "coffee cup"`) | **`isaac_ros_grounding_dino`** |

**Sequencing rule:** close the loop with a trivially-correct pose first — ArUco (Seeed ships
`aruco_pose.py` and printable PDFs) or GT pose published straight onto the topic — *then* swap in
the estimator.

## 9. S8–S9 — Honest depth, trained perception, writeup (Sep–Oct) → **M7**

1. **Replace clean depth with a simulated RGB-D sensor** (noise, dropouts at edges and on
   speculars). **Measure yield before and after.**
   ⚠️ Isaac ROS 4.5 known issue: *"Workflows that involve pose estimation may generate incorrect
   pose estimates when using ESS or FoundationStereo depth. As a workaround, use
   RealSense/camera sensor depth instead."*
2. **Replicator SDG → train a component**, evaluate in sim, report pose error in mm. State
   plainly: *trained in sim, evaluated in sim* — **not** sim2real until it runs on the arm.
3. **Eval protocol**: ≥20 episodes, varied start poses, continuous execution, no restart.
4. **Characterize a hard case** — deformable object; pose, grasp validity, or slip?
5. **Port one node to C++** (`rclcpp`) — the perception→grasp bridge. Closes G3.
6. **Writeup** in the style of your SmolVLA page.

## 10. Cross-cutting rules

1. **16 GB VRAM is binding.** Isaac Sim alone ~8 GB. **Develop against the recorded dataset** —
   S2 is built around this deliberately.
2. **One substitution at a time.**
3. **Git-tag every working state** — `m1-vendor-moveit`, `m2-offline-grasp`, `rc-aug29`, …
4. **Isaac Lab is not this project.** *Caveat:* if you author Replicator/SDG through Isaac Lab in
   §9, your own notes apply — `CameraCfg` needs `--enable_cameras`, `distance_to_image_plane`
   has hung headless. Different entry point via Isaac Sim standalone.
5. **Don't let Ubuntu upgrade the driver.** 595 crashes Isaac Sim 5.1.

## 11. Hardware bolt-on (later)

Swap `TopicBasedSystem` for the real hardware interface — **and** re-do camera sourcing, hand-eye
calibration, frame definitions, and session zeroing (§1). The B601 stores no persistent
calibration; motors re-zero against whatever pose the arm holds at connect, so **measure
session-to-session zero variance** — the noise floor the sim2real delta must clear.

## 12. Open questions and unresolved decisions

**Blocking, dated:**
- **Branch B model/checkpoint, prompt, threshold, IoU rule** — must be frozen by **Aug 10**
  (§5.2.5), else it blocks M1.
- **Does the capture chain extrapolate to 300 scenes?** — answered **Aug 8** (§5.2.1).

**Unresolved, not yet blocking:**
- **Which convention wins** the π gripper-mount question, and is it a defect at all (§6.4)?
- Is the shipped **DM USD** physically complete — inertias, joint drives, collision meshes?
- **`isaac_ros_manipulation` and `topic_based_ros2_control` SHAs** — record at container
  creation (§2); the §6.1 enum quote is unverified against whatever SHA ships.
- Does the reference pick-and-place run on **x86 + Isaac Sim 5.1**? Test platform is Jetson Thor.
- How much of the **Flexiv** `RobotControllerBase` subclass is Flexiv-specific vs boilerplate?
- **FoundationPose VRAM** alongside a live Isaac Sim on 16 GB — unmeasured.
- **Licensing** — repo license, and whether model weights and the 300-scene dataset may be
  redistributed. Decided in the RC window (§5.4).

---

### Verification notes (2026-08-04)

Read directly from the pinned trees: `approach = _normalize(-position)` and the `open_axis`
derivation in `utils/ordinary_grasp.py`; `depth_quantile` default `0.75` in code (:56, :103) vs
`0.5` in `config/default.yaml:61`; `mock_components/GenericSystem` in
`rebotarm.ros2_control.xacro:8`; `gripper_joint` `type="fixed"` at :436 and no `mimic` tags in
the vendor URDF; `gripper_joint1/2` prismatic at :489/:547 with `upper="0.0715"`; **identical
finger joint origins/axes/limits and identical `gripper_left`/`gripper_right` child labels in
both trees, differing mount rpy, and disjoint mesh sets**; `usd/reBot_B601_DM/reBot_B601_DM.usda`
plus payloads; MJCF classes `rs06`/`rs00`; no OmniGraph prims in the DM USD; `result.masks` +
`result.obb` and the three shipped `.pt` models.
From NVIDIA sources: Isaac ROS 4.5 (2026-07-06) Jazzy/24.04/driver 580+; the BYOR package list;
`RobotType`/`GripperType` quoted from `manipulator_types.py` @ `release-4.5` (**re-verify against
the container's SHA**); the mac-and-cheese `object_class: 22` and ESS/FoundationStereo known
issues; `isaac_ros_yolov8` publishing axis-aligned `Detection2DArray`; the
`isaac_ros_manipulation_robots/` listing; XRDF fields; Grasp Editor location. Issue #22 read
directly (open, Thor, 4.1–4.2). B601 actuator-limit values carried from the parent plan —
**re-verify against the cloned URDF.**
