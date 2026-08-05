# reBot B601-DM Visual Grasping — Isaac Sim + Isaac ROS, simulation only

**Written:** 2026-08-04 · **Rev 2.1** — evaluation design corrected, asset story corrected,
BYOR scope corrected again
**Parent plan:** `~/docs/sota-gap-plan-aug2026.md` (this is an expanded, learning-first P1)
**Reference demo:** <https://wiki.seeedstudio.com/rebot_arm_b601_dm_grasping_demo/>

> **Revision history — what each pass got wrong, so it isn't relearned:**
> **rev 1** treated the B601 swap as URDF→USD + XRDF and put it on the Aug 29 critical path.
> **rev 2** corrected the scope (BYOR = two authored packages), moved B601 to September, and
> dropped the wrong "swap RT-DETR for YOLO" one-liner.
> **rev 2.1** corrects the *measurement* — rev 2 scored the approach vector, which in Seeed's
> code is just the camera viewing ray and therefore measures nothing — plus: a **DM USD already
> exists** (rev 2 said convert from URDF), the MJCF parity model is the **RS** arm not DM, and
> BYOR additionally needs a **framework extension** because `RobotType` is a closed enum.

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

⚠️ **Correction to rev 2:** going to hardware does *not* swap "exactly one layer." It changes
the `ros2_control` hardware interface **and** the camera source (real sensor noise, exposure,
rolling shutter), hand-eye calibration (measured, not known), frame definitions, the session
zeroing procedure, and possibly the controller interface itself. The sim work still removes
most of the risk — but budget the transfer honestly rather than as a config swap.

---

## 2. Verified environment (checked 2026-08-04)

| Thing | Status |
|---|---|
| GPU | RTX 5000 Ada Laptop, **16 GB VRAM** — binding constraint, §10 |
| Driver | **580.173.02** ✅ meets Isaac ROS 4.5's 580+/CUDA 13.0+; avoids the 595 `rtx.scenedb` crash. **`apt-mark hold` today.** |
| OS | Ubuntu 24.04.4 ✅ |
| Isaac Sim | **5.1.0.0**, pip in `~/isaaclab-venv` (Python **3.11.15**) ✅ matches the cuMotion tutorial |
| ROS 2 bridge | present, both `humble/` and `jazzy/` lib dirs ✅ |
| ROS 2 | **not installed** — first task |
| Docker | 29.5.0, user in `docker` group ✅ — the install path (§4) |
| Upstream repos | cloned into `src/`, pinned in **`upstream.repos`** |

**The distro-split landmine is gone.** Isaac ROS **release-4.5** (2026-07-06): *"All Isaac ROS
packages are designed and tested to be compatible with ROS 2 Jazzy."* Parent plan §9.3 is stale.

**Two process rules.** (1) Never `import rclpy` inside Isaac Sim's interpreter — Isaac Sim is
Python 3.11, Jazzy is 3.12; the bridge is OmniGraph nodes, your ROS nodes are separate
processes. (2) Cyclone DDS + matching `ROS_DOMAIN_ID` everywhere, host and container — Isaac Sim
defaults to FastRTPS and a mismatch looks exactly like a broken pipeline.

---

## 3. Milestones

**Hard rule: something ships Aug 29, and it is M2.** M3 is explicitly **optional and
timeboxed** — it must not delay the video or the repo.

| # | By | Deliverable | Status |
|---|---|---|---|
| **M1** | Aug 10 | Vendor code read (§5.1); vendor MoveIt demo runs (mock hw); Isaac ROS container up; S0 gate passed | required |
| **M2** | Aug 17 | Wrist-cam scene + recorded dataset + **offline `RGB-D → grasp PoseStamped`**, both branches evaluated (§5.2) | **required — load-bearing** |
| **M3** | Aug 24 | Stock reference pick-and-place in Isaac Sim (soup can) | **optional, 3-day timebox** |
| **M4** | **Aug 29** | **Demo video + public repo. Start applying.** | required |
| **M5a** | Sep 5 | DM USD validated + **ROS 2 Action Graph** + raw joint round-trip + `TopicBasedSystem` | |
| **M5b** | Sep 12 | B601 **framework extension** + two packages + MoveIt/OMPL execution | |
| **M5c** | Sep 19 | Flattened URDF + XRDF + **cuMotion** | |
| **M6** | Sep 26 | Active two-finger gripper, contact, attachment, authored grasps | |
| **M7** | Oct 10 | Honest depth, Replicator-trained perception, eval protocol, C++ node, writeup | |

M5 is split into four because "cuMotion by Sep 12" was not credible — the Flexiv template
clarifies the sequence but doesn't compress it.

---

## 4. S0 — Environment (Aug 4–6)

1. `sudo apt-mark hold` the NVIDIA driver packages.
2. ROS 2 Jazzy + `ros-jazzy-rmw-cyclonedds-cpp` on the host.
3. **Isaac ROS 4.5 via Docker** — NVIDIA's recommendation, your runtime already works, and §6
   means building against the full Isaac ROS tree. Host Isaac Sim ↔ container over DDS:
   Cyclone + same `ROS_DOMAIN_ID`, host networking.

### The S0 gate — `ros2 topic list` is not sufficient

An arbitrary Isaac Sim scene publishes **nothing**. Topics only exist if the scene has an
explicit ROS 2 Action Graph. Build one (clock, camera helper, TF, joint states), then verify
**from inside the container**:

- [ ] `/clock` publishing, and `use_sim_time` respected by a test node
- [ ] **actual RGB and depth messages arriving** (`ros2 topic hz`, not just listed)
- [ ] `CameraInfo` present, with K matching the render resolution
- [ ] **TF resolvable at the image timestamp** — not just "TF exists"
- [ ] rates and **QoS profiles** compatible (sensor-data QoS vs default is a classic silent drop)

Only then is S0 done.

---

## 5. S1–S3 — Aug 29 without the B601

### 5.1 · S1 — Read the vendor code, run the vendor demo (Aug 4–10) → **M1**

Cloned in `src/`, pinned in `upstream.repos`. **Read in this order**, producing a short
**frame / unit / interface note** as you go (this note is the spec for M2):

1. `utils/ordinary_grasp.py` — the grasp geometry, the important one
2. `utils/transforms.py` — frame conventions; compare against ROS TF2
3. `scripts/main.py` — the orchestration you'll replace with a behavior tree
4. `utils/camera_utils.py` — intrinsics, alignment, depth units
5. `calibration/hand_eye.py` (+ `aruco_pose.py`) — the `AX=XB` solve

**Defer `scripts/grasp.py`** — that's the separate GraspNet route, not the baseline pipeline.

Then run the vendor MoveIt demo:
```bash
ros2 launch rebotarm_moveit_config demo.launch.py
```
⚠️ Verified: it drives `mock_components/GenericSystem`
(`rebotarm_moveit_config/config/rebotarm.ros2_control.xacro:8`) — virtual hardware only, it
will **not** drive Isaac Sim. Run it anyway to see the planning group, SRDF, and joint limits.

### 5.2 · S2 — The offline perception slice (Aug 10–17) → **M2** ← *the deliverable*

> **recorded RGB-D → `geometry_msgs/PoseStamped` grasp pose**, validated offline.

No arm, no planner, no orchestration, no live simulator. **Freeze the evaluator before writing
the node** — object, grasp GT, output frame, depth units, timestamp policy, dataset seeds,
model, and pass/fail accounting all fixed in writing first.

#### What to measure — the opening axis, not the approach

Verified in `utils/ordinary_grasp.py`: `approach = _normalize(-position)` — that is just the
**camera viewing ray**, and it carries almost no information from perception. The mask/OBB
result actually determines `open_axis`, derived from the min-area-rect short edge and then
orthogonalized against the approach. **So score the opening axis:**

$$\theta_{\text{open}} = \cos^{-1}\!\left(\left|\hat{o}\cdot o^{*}\right|\right)$$

The absolute value is required — a parallel-jaw gripper is symmetric under 180°, so ô and −ô
are the same grasp. Approach-vs-object-surface-normal is a useful **secondary diagnostic**,
not the headline metric.

#### Ground truth must be a *grasp*, not an object pose

The object origin is not the intended contact point. **Add a `grasp_gt` child transform** to the
object in the scene and compare the predicted contact pose against that transform **at the image
timestamp**.

**Object choice matters:** use one **flat-topped, non-square, textured** object. A cube,
cylinder, or soup can is orientation-degenerate — the opening axis is unidentifiable and the
angular metric becomes noise. (Soup can is fine for M3's stock workflow; it is wrong here.)

#### Two branches, reported separately

| Branch | Input | What it validates |
|---|---|---|
| **A · Oracle** | ground-truth mask + clean depth | geometry, depth handling, TF, `PoseStamped` plumbing |
| **B · Predicted** | model mask + the same depth | the actual RGB-D perception slice |

⚠️ **Branch A's recall is 100% by construction and must never be reported as detector
performance.** Freeze model, prompt, threshold, and matching rule; **mask IoU ≥ 0.5 = true
positive.**

#### Dataset

| Split | Scenes | Use |
|---|---|---|
| Development | 50 | iterate freely |
| **Locked test** | **200 positive** | touch once, at the end |
| Target-absent | 50 | false-positive rate |

One capture per randomized scene.

#### Preregistered targets (set now, before measuring)

| Metric | Target |
|---|---|
| Grasp-point position error ‖p̂ − p*‖ | median ≤ 5 mm, p90 ≤ 10 mm |
| **Opening-axis error** θ_open | median ≤ 5°, p90 ≤ 15° |
| Detection recall (branch B) | ≥ 95 % |

Report **conditional pose errors** (given a true positive) *and* **end-to-end within-tolerance
yield** *and* **false-positive rate** — the conditional numbers alone hide a detector that only
succeeds on easy scenes. **Print confidence intervals.**

⚠️ These are **engineering targets, not error budgets derived from hardware.** Do *not* justify
them with the arm's ±0.2 mm repeatability — that's a joint-repeatability figure and has nothing
to do with visual pose error. If a target is missed, **the August artifact still ships**, with
an honest failure analysis. A missed preregistered target that you diagnose is a better
portfolio item than a target invented after the fact.

#### Build it in this order

```
pure tested library  →  rosbag replay wrapper  →  PoseStamped + debug overlay  →  evaluator
```

The **debug overlay** should render: mask, min-area rectangle, opening axis, predicted grasp,
and GT grasp. It is the single highest-value debugging artifact in the project and it is also
your demo video.

### 5.3 · S3 — Stock reference pick-and-place (Aug 17–24) → **M3, optional**

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch isaac_ros_manipulation_bringup workflows.launch.py \
   manipulator_workflow_config:=${ISAAC_ROS_MANIPULATION_WORKFLOW_CONFIG_DIR}/sim_launch_params.yaml
```
⚠️ **Start with the soup can.** Isaac ROS 4.5 known issue: *"the pick-and-place workflow may
fail to pick the mac-and-cheese object, `object_class: 22`, with the Robotiq 2F-140 gripper."*

⚠️ Stated test platform is Jetson AGX Thor; issue **#22** (open, Thor, 4.1–4.2) reports it
failing after working in 4.0. **Three-day stretch task. If it doesn't come up, drop it and
ship M2.** Fallback tutorials: Pose to Pose Planning, Object Following.

---

## 6. S5 — Bring the B601-DM in (Sep) → **M5a/b/c**

### 6.0 Assets — select a canonical source, then validate. Do *not* start by converting.

⚠️ **Correction to rev 2 (and to my stored notes): a DM USD already exists** —
`src/reBot-Isaacsim/usd/reBot_B601_DM/reBot_B601_DM.usda`, with `payloads/` for base, robot,
geometries, materials, and physics (`physx`, `physics`, `mujoco`). The "RS-variant only" claim
is **false** for the current tree.

So the task is **select → validate → patch or regenerate only if required**, not convert.

⚠️ **Unresolved 180° gripper-mount discrepancy — verified, resolve before anything else:**

| Source | `gripper_joint` origin rpy |
|---|---|
| `src/reBot-Isaacsim/urdf/reBot_B601_DM/urdf/reBot_B601_DM.urdf:251` | `rpy="3.1416 -1.5708 0"` |
| `src/reBotArmController_ROS2/.../reBot_B601_DM_with_gripper.urdf:435` | `rpy="0 -1.5708 0"` |

Same translation (`xyz="0 0 0.15971"`), **roll differs by π**. This is exactly the Flexiv
flange-offset bug class that cost you a week — except it's visible now, before you build on it.
**Resolve it before defining TCP frames or authoring the XRDF**, and record which source won.

⚠️ **The MJCF parity model is the RS arm, not DM.** `mjcf/rebot_devarm/rebot_devarm.xml` uses
joint classes `rs06`/`rs00`, gripper joints named `joint_left`/`joint_right`, and joint4 range
`-1.79 1.69` (the DM URDF has −1.87). **Adapt the cross-validation *technique*; do not treat it
as DM ground truth.**

⚠️ **The DM USD has no ROS 2 Action Graph.** Verified — no OmniGraph/ROS2Publish prims. You must
author it: **clock, joint-state publisher, joint-command subscriber, articulation controller.**
Without it `TopicBasedSystem` has nothing to talk to. This is **M5a**, and it is prerequisite to
everything else.

Also still true: verify actuator limits (URDF claims 50 rad/s joints 1–3, 200 rad/s joints 4–6 —
200 rad/s ≈ 1,900 RPM at the joint) against Damiao DM4310/DM4340P datasheets, override, and
document the override. `joint4` lower limit −1.87 rad.

### 6.1 The framework extension (M5b) — *new in rev 2.1*

⚠️ Isaac ROS 4.5 **hard-codes** the supported robots and grippers. Verified in
`isaac_ros_manipulation_ros_python_utils/manipulator_types.py`:

```python
class RobotType(ManipulatorEnum):
    UR = 'UR'
    FLEXIV = 'FLEXIV'

class GripperType(enum.Enum):
    ROBOTIQ_2F_140 = 'robotiq_2f_140'
    ROBOTIQ_2F_85  = 'robotiq_2f_85'
    GRAV           = 'grav'
```

Closed enums, no dynamic registration. So the B601 needs a **third piece of work beyond the two
packages**: extend `RobotType`/`GripperType` and the `DriverConfig` dispatch in `config.py`
(the `get_gripper_type()` classmethod included). Decide early whether you **fork** Isaac ROS or
carry a **patch overlay** — and say which in the writeup, because "I had to extend the framework
enum" is a more interesting sentence than "I wrote a config file."

### 6.2 The two packages

**`isaac_ros_manipulation_b601_robot_description/`** (config) — `urdf/b601.xacro` with the
**`TopicBasedSystem`** ros2_control plugin mapping `joint_commands_topic`/`joint_states_topic`
to Isaac Sim; `srdf/`; `config/` for `initial_positions`, `joint_limits`, `kinematics_sim`,
`moveit_sim_controllers`, `ros2_control_controllers_sim`.

**`isaac_ros_manipulation_b601_driver_utils/`** (Python) — `config.py` subclassing
`DriverConfig`; `robot_description.py`; `b601_driver_utils.py` subclassing `RobotControllerBase`
(implementing `get_robot_state_publisher()`, `get_moveit_group_node()`,
`get_robot_control_nodes()`); `launch/b601_driver.launch.py`; `params/b601.yaml`;
`src/isaac_sim_joint_parser_node.py`.

Plus routing (`robot_launch_file_path` in the workflow YAML, `package.xml` dep in
`isaac_ros_manipulation_bringup`) and a launch test.

✅ **Your template**: `isaac_ros_manipulation_robots/` ships
`isaac_ros_manipulation_flexiv_driver_utils` + `isaac_ros_manipulation_flexiv_robot_description`
— a two-package third-party integration on an arm you already know from `~/Flexiv_RL` (and
`GRAV` is the Flexiv gripper, so the gripper path is worked too). `isaac_ros_manipulation_ur_isaac_sim_utils`
is worth reading for Isaac Sim joint-state filtering. **Read both Flexiv packages end to end
before writing B601 code.** No Franka package exists here.

### 6.3 XRDF + cuMotion (M5c)

Flattened URDF + `b601_gripper.xrdf`: c-space joints with acceleration and jerk limits, tool
frames, per-link collision spheres, self-collision ignore rules. Use Isaac Sim's visual **Robot
Description Editor**. **Only after the 180° question is settled.**

---

## 7. The gripper (M6)

Isaac orchestration expects a **`control_msgs/GripperCommand` action**. Verified in the vendor
URDF: `gripper_joint` (:436) is **`type="fixed"`** — a mount, not actuated, **not a mimic joint**
(there are no `mimic` tags in the file at all). Only **`gripper_joint1` and `gripper_joint2`**
(both prismatic, :489/:547) are actuated, and being un-mimicked they must **both** be commanded.

Needed: a gripper action controller/adapter mapping one commanded width → two prismatic joints;
contact and friction setup in the USD (defaults will drop objects and look like a planning bug);
attach/detach handling during transport; and grasp authoring via `Isaac Utils → Grasp Editor` →
`isaac_grasp` YAML. Documented examples are parallel-jaw grippers like the 2F-140.

## 8. Perception — why YOLO is not a drop-in

`isaac_ros_yolov8` publishes `vision_msgs/Detection2DArray` — **axis-aligned boxes only, no
masks, no oriented boxes** — and isn't among the workflow's standard detector configs (which use
`RTDETR`), so it needs a custom adapter and launch graph. Seeed's `utils/yolo_utils.py` uses
`result.masks` **and** `result.obb`, shipping `yoloe-26s-seg.pt` and `yolov8s-world.pt`. **The
mask/OBB short edge is where the opening axis comes from** (§5.2) — an axis-aligned box cannot
produce it.

| Want | Isaac ROS route |
|---|---|
| mask → opening axis (closest to Seeed) | detector box → **`isaac_ros_segment_anything2`** (box-prompted) → mask → min-area rect |
| full 6D pose (stronger) | detector + mask → **`isaac_ros_foundationpose`** |
| open-vocab (`--target-class "coffee cup"`) | **`isaac_ros_grounding_dino`** |

**Sequencing rule:** close the loop with a trivially-correct pose first — ArUco (Seeed ships
`aruco_pose.py` and printable PDFs) or ground-truth pose published straight onto the topic —
*then* swap in the estimator.

## 9. S8–S9 — Honest depth, trained perception, writeup (Sep–Oct) → **M7**

1. **Replace ground-truth depth with a simulated RGB-D sensor** (noise, dropouts at edges and on
   speculars). **Measure yield before and after.**
   ⚠️ Isaac ROS 4.5 known issue: *"Workflows that involve pose estimation may generate incorrect
   pose estimates when using ESS or FoundationStereo depth. As a workaround, use
   RealSense/camera sensor depth instead."*
2. **Replicator SDG → train a component**, evaluate in sim, report pose error in mm. State
   plainly that this is *trained in sim, evaluated in sim* — **not** sim2real until it runs on
   the arm.
3. **Eval protocol**: ≥20 episodes, varied start poses, continuous execution, no restart.
4. **Characterize a hard case** — deformable object; is it pose, grasp validity, or slip?
5. **Port one node to C++** (`rclcpp`) — the perception→grasp bridge. Closes G3.
6. **Writeup** in the style of your SmolVLA page.

## 10. Cross-cutting rules

1. **16 GB VRAM is binding.** Isaac Sim alone ~8 GB. **Develop against rosbags** — S2 is built
   around this deliberately.
2. **One substitution at a time.**
3. **Git-tag every working state** — `m1-vendor-moveit`, `m2-offline-grasp`, …
4. **Isaac Lab is not this project.** *Caveat:* if you author Replicator/SDG through Isaac Lab in
   §9, your own notes apply — `CameraCfg` needs `--enable_cameras`, `distance_to_image_plane`
   has hung headless. Different entry point via Isaac Sim standalone.
5. **Don't let Ubuntu upgrade the driver.** 595 crashes Isaac Sim 5.1.

## 11. Hardware bolt-on (later)

Swap `TopicBasedSystem` for the real hardware interface — **and** re-do camera sourcing,
hand-eye calibration, frame definitions, and session zeroing (§1). The B601 stores no persistent
calibration; motors re-zero against whatever pose the arm holds at connect, so **measure
session-to-session zero variance** — that's the noise floor the sim2real delta must clear.

## 12. Open questions

- Does the reference pick-and-place run on **x86 + Isaac Sim 5.1**? Test platform is Jetson Thor.
- **Which URDF wins** the 180° gripper-mount conflict (§6.0)?
- Is the shipped **DM USD** physically complete — inertias, joint drives, collision meshes — or
  does it need regeneration after validation?
- How much of the **Flexiv** `RobotControllerBase` subclass is Flexiv-specific vs boilerplate?
- **FoundationPose VRAM** alongside a live Isaac Sim on 16 GB — unmeasured.

---

### Verification notes (2026-08-04)

Read directly from the pinned trees: `approach = _normalize(-position)` and the `open_axis`
derivation in `utils/ordinary_grasp.py`; `mock_components/GenericSystem` in
`rebotarm.ros2_control.xacro:8`; `gripper_joint` `type="fixed"` at :436 and no `mimic` tags
anywhere in the vendor URDF; `gripper_joint1/2` prismatic at :489/:547; the two conflicting
`gripper_joint` rpy values; `usd/reBot_B601_DM/reBot_B601_DM.usda` plus payloads; MJCF classes
`rs06`/`rs00` and `joint_left`/`joint_right`; no OmniGraph prims in the DM USD; `result.masks`
+ `result.obb` and the three shipped `.pt` models in `yolo_utils.py`/`models/`.
From NVIDIA sources: Isaac ROS 4.5 (2026-07-06) Jazzy/24.04/driver 580+; the BYOR package list;
`RobotType`/`GripperType` enums quoted from `manipulator_types.py` @ `release-4.5`; the
mac-and-cheese `object_class: 22` and ESS/FoundationStereo known issues; `isaac_ros_yolov8`
publishing axis-aligned `Detection2DArray`; the `isaac_ros_manipulation_robots/` directory
listing; XRDF fields; Grasp Editor location. Issue #22 read directly (open, Thor, 4.1–4.2).
B601 actuator-limit values carried from the parent plan — **re-verify against the cloned URDF.**
