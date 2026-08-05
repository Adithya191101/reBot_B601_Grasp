# reBot B601-DM Visual Grasping — Isaac Sim + Isaac ROS, simulation only

**Written:** 2026-08-04 · **Rev 2** (after review — scope corrected, milestones re-dated)
**Parent plan:** `~/docs/sota-gap-plan-aug2026.md` (this is an expanded, learning-first P1)
**Reference demo:** <https://wiki.seeedstudio.com/rebot_arm_b601_dm_grasping_demo/>

> **What changed in rev 2, and why it matters:** rev 1 treated the B601-DM swap as a
> URDF→USD + XRDF job and put it on the critical path to Aug 29. NVIDIA's *Bring Your Own
> Robot* guide (read directly) shows it's **two new ROS packages plus Python subclasses plus a
> launch test plus XRDF** — multi-week, not multi-day. Rev 2 moves the B601 to September and
> makes the Aug 29 artifact something that doesn't depend on it. It also drops the "swap
> RT-DETR for YOLO" one-liner, which was wrong (§7).

---

## 0. The idea in one line

Seeed's demo is a hand-rolled Python grasping pipeline on real hardware. NVIDIA's *Isaac for
Manipulation* is the same pipeline shape, GPU-accelerated, in ROS 2, running in Isaac Sim.
**Start from what already runs, substitute one piece at a time.** Never a big-bang integration.

| Seeed demo (real, Python) | Your version (sim, ROS 2) | Honest difficulty |
|---|---|---|
| Orbbec / RealSense RGB-D | Isaac Sim camera → `isaacsim.ros2.bridge` | easy |
| YOLO **seg + OBB** (`yoloe-26s-seg.pt`, `yolov8s-world.pt`) | see §7 — **not** a drop-in | medium |
| heuristic grasp from mask axis / GraspNet | `isaac_grasp` YAML + Grasp Editor, or FoundationPose | medium |
| `calibration/hand_eye.py` + `aruco_pose.py` | known extrinsics in sim → solve `AX=XB` anyway | easy |
| `reBotArm_control_py` (Pinocchio IK) | `isaac_ros_cumotion` + MoveIt 2 + `ros2_control` | **hard** (§6) |
| `scripts/main.py` orchestration | `isaac_ros_manipulation_orchestration` (behavior tree) | medium |

## 1. What sim-only does and doesn't buy you

Sim-only fully closes **G2** (perception → grasp) and **G3** (C++). It does **not** close **G1**
— a sim2real delta needs hardware. The design consequence: this is one ROS 2 graph, and going
to hardware swaps exactly one layer — the `ros2_control` hardware interface — leaving
perception, grasp, planning, and orchestration untouched.

---

## 2. Verified environment (checked 2026-08-04)

| Thing | Status |
|---|---|
| GPU | RTX 5000 Ada Laptop, **16 GB VRAM** — binding constraint, see §10 |
| Driver | **580.173.02** ✅ meets Isaac ROS 4.5's "Driver 580+ / CUDA 13.0+"; also avoids the 595 `rtx.scenedb` crash. **`apt-mark hold` it today.** |
| OS | Ubuntu 24.04.4 ✅ |
| Isaac Sim | **5.1.0.0**, pip in `~/isaaclab-venv` (Python **3.11.15**) ✅ the version the cuMotion tutorial targets |
| ROS 2 bridge | present, ships both `humble/` and `jazzy/` lib dirs ✅ |
| ROS 2 | **not installed** — first task |
| Docker | 29.5.0, user in `docker` group ✅ — **this is the install path** (§4) |
| Repos | cloned + pinned in `.repos`, see `src/` |

### The distro-split landmine is gone

Parent plan §9.3 warns Isaac ROS is Humble-pinned in Docker and suggests building the arm stack
from source to avoid a bridge. **Stale.** Isaac ROS **release-4.5** (released 2026-07-06):
*"All Isaac ROS packages are designed and tested to be compatible with ROS 2 Jazzy."* Isaac Sim
5.1, Isaac ROS 4.5, `reBotArmController_ROS2`, and your OS are all Jazzy.

### 🎁 Isaac ROS 4.5 added **Flexiv Rizon** support

You have a Flexiv Rizon 4 project (`~/Flexiv_RL`). Release 4.5's manipulation notes list
*"Added Flexiv Rizon support"* alongside the new BYOR guide. **Read the Flexiv driver-utils
package as your template for §6** — it's a worked example of exactly the integration you need,
for an arm whose kinematics you already know. That's the single biggest de-risker available.

### Two process rules

1. **Never `import rclpy` inside Isaac Sim's interpreter** — Isaac Sim is Python 3.11, Jazzy is
   3.12. Isaac Sim talks ROS via OmniGraph bridge nodes; your ROS nodes are separate processes.
2. **Cyclone DDS everywhere**: `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and a matching
   `ROS_DOMAIN_ID` in every terminal *and* inside the container. Isaac Sim defaults to FastRTPS;
   a mismatch looks exactly like a broken pipeline — topics silently never connect.

---

## 3. Milestones — re-dated in rev 2

**Hard rule from the parent plan: something ships Aug 29.** In rev 2 the Aug 29 artifact is
**M2 + M3**, neither of which needs the B601-DM. The B601 integration (M5) is September work.
That's not a retreat — it's what the BYOR scope actually costs, and finding it out now is worth
more than discovering it on Aug 26.

| # | By | Deliverable | Needs B601? |
|---|---|---|---|
| **M1** | Aug 10 | Vendor MoveIt demo runs (mock hw); vendor grasp code read; Isaac ROS container up | no |
| **M2** | Aug 17 | Wrist-cam scene + recorded dataset + **offline `RGB-D → grasp PoseStamped`, validated against sim ground truth** | no |
| **M3** | Aug 24 | Stock reference pick-and-place running in Isaac Sim (**soup can**, §5) | no |
| **M4** | **Aug 29** | **Demo video + public repo. Start applying.** | no |
| **M5** | Sep 12 | B601-DM driven from Isaac Sim through `TopicBasedSystem` + cuMotion (BYOR packages built) | yes |
| **M6** | Sep 26 | Gripper working + your objects → full B601 pick-place in sim | yes |
| **M7** | Oct 10 | Honest depth, Replicator-trained perception, eval protocol, C++ node, writeup | yes |

**M2 is the load-bearing one.** It's arm-independent, it's the part that's genuinely *yours*
rather than NVIDIA's, and it can't be blocked by the reference workflow failing to launch.

---

## 4. S0 — Environment (Aug 4–6)

1. `sudo apt-mark hold` the NVIDIA driver packages. 30 seconds, protects the project.
2. Install **ROS 2 Jazzy** + `ros-jazzy-rmw-cyclonedds-cpp` on the host.
3. **Isaac ROS 4.5 via Docker.** NVIDIA recommends it, your Docker + NVIDIA runtime already
   work, and the BYOR work in §6 means building packages against the full Isaac ROS tree —
   which the dev container is built for. Host-side Isaac Sim talks to the container over DDS:
   **Cyclone DDS + same `ROS_DOMAIN_ID`**, host networking.
4. Launch Isaac Sim with `ROS_DISTRO=jazzy`, open any scene.
5. **Done when:** `ros2 topic list` *inside the container* shows topics published by host-side
   Isaac Sim. That crossing is the foundation — don't move on without it.

---

## 5. S1–S3 — Get to Aug 29 without the B601

### S1 · Read the vendor code, run the vendor demo (Aug 4–10) → **M1**

Already cloned into `src/` and pinned in `.repos`. Read these in order — they are a compact,
complete, working implementation of the exact pipeline you're rebuilding:

| File | Why |
|---|---|
| `reBot-DevArm-Grasp/scripts/main.py` | the orchestration you'll replace with a behavior tree |
| `utils/yolo_utils.py` | **the important one** — see §7 |
| `utils/ordinary_grasp.py` | how a grasp pose is derived from mask/OBB + depth |
| `utils/transforms.py` | frame conventions; compare against ROS TF2 |
| `calibration/hand_eye.py` + `aruco_pose.py` | the `AX=XB` solve you'll reimplement |
| `drivers/camera/*.py` | what the RGB-D interface actually needs to provide |

Then run the vendor MoveIt demo, before touching cuMotion:
```bash
ros2 launch rebotarm_moveit_config demo.launch.py
```
⚠️ **Verified: this drives `mock_components/GenericSystem`** (`rebotarm_moveit_config/config/
rebotarm.ros2_control.xacro:8`) — virtual hardware only. It will *not* drive Isaac Sim, and
that's precisely the gap §6 closes. Run it anyway: it's the fastest way to see the B601's
planning group, SRDF, and joint limits behave.

### S2 · The offline perception slice (Aug 10–17) → **M2** ← *most important stage*

Build the wrist-camera scene in Isaac Sim and **record**, don't stream: RGB, aligned depth,
`camera_info`, TF, `/joint_states`, and **ground-truth object pose**. Then write one node that
does exactly one thing:

> **recorded RGB-D → `geometry_msgs/PoseStamped` grasp pose**

Validate it offline against the recorded ground truth. Nothing else. No arm, no planner, no
orchestration, no live simulator.

**Commit to the metric and the bar now, before you measure — otherwise the number is post-hoc:**

| Metric | Pass bar |
|---|---|
| Grasp-point **position error** (‖p̂ − p*‖, mm) | median ≤ 5 mm, 90th pct ≤ 10 mm |
| Grasp-axis **angular error** (∠ between approach vectors, deg) | median ≤ 5°, 90th pct ≤ 15° |
| Detection **recall** on the recorded set | ≥ 95 % |

Report the three separately — a pipeline can nail position and be useless on orientation, and
collapsing them into one score hides exactly the failure the gripper will find. Set the bars
against the B601's ±0.2 mm repeatability and its finger stroke: a grasp-axis error large enough
to miss the object's short side is a failure regardless of what the position error says.

Why this is the right first deliverable: it's the piece the Seeed demo actually is, it's
arm-independent, it runs on a laptop against a bag file, and it has a **number** attached. It
also sidesteps the 16 GB VRAM problem entirely (§10) and gives you a regression test that
every later change can be checked against.

### S3 · Stock reference pick-and-place (Aug 17–24) → **M3**

Run the reference workflow unmodified (UR + Robotiq 2F-140, RT-DETR, ground-truth depth):
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch isaac_ros_manipulation_bringup workflows.launch.py \
   manipulator_workflow_config:=${ISAAC_ROS_MANIPULATION_WORKFLOW_CONFIG_DIR}/sim_launch_params.yaml
```
⚠️ **Start with the soup can.** Isaac ROS 4.5 known issues: *"In Isaac Sim, the pick-and-place
workflow may fail to pick the mac-and-cheese object, `object_class: 22`, with the Robotiq 2F-140
gripper."* Don't debug a documented bug.

⚠️ Also expect friction generally: this workflow's stated test platform is Jetson AGX Thor, and
`NVIDIA-ISAAC-ROS/isaac_ros_manipulation` **issue #22** (open, Thor, 4.1–4.2) reports it failing
after working in 4.0. **Timebox to 3 days.** Fallback: the simpler **Pose to Pose Planning** and
**Object Following** sim tutorials. M2 already secures the Aug 29 artifact either way.

### S4 · Ship (Aug 29) → **M4**

Demo video: the offline grasp-pose result with error numbers, plus whatever of S3 came up.
Public repo, site updated, start applying.

---

## 6. S5 — Bring the B601-DM in (Sep) → **M5** · *the real scope*

Per NVIDIA's **Bring Your Own Robot** guide, this is **two packages you author**, not a config
tweak.

✅ **Your template exists and is exactly the right shape.** `NVIDIA-ISAAC-ROS/isaac_ros_manipulation`
→ `isaac_ros_manipulation_robots/` contains:

```
isaac_ros_manipulation_flexiv_driver_utils/       ← copy this structure
isaac_ros_manipulation_flexiv_robot_description/  ← and this
isaac_ros_manipulation_robot_utils/               ← RobotControllerBase lives here
isaac_ros_manipulation_ur_driver_utils/
isaac_ros_manipulation_ur_isaac_sim_utils/        ← likely the Isaac Sim joint parser
isaac_ros_manipulation_ur_robot_description/
```

The **Flexiv pair is a two-package integration matching the BYOR guide exactly** — a
third-party arm added by following the same path you're about to walk, on an arm whose
kinematics you already know from `~/Flexiv_RL`. **Read both Flexiv packages end to end before
writing a line of B601 code.** (UR has a third, sim-specific package worth reading too, for how
Isaac Sim joint states get filtered.) There is no Franka package here — Franka support is via
`isaac_ros_cumotion_examples` only.

**Package 1 — `isaac_ros_manipulation_b601_robot_description/`** (config)
- `urdf/b601.xacro` — **with the `TopicBasedSystem` ros2_control plugin**, mapping
  `joint_commands_topic` / `joint_states_topic` to Isaac Sim topics. This is the specific thing
  the vendor's `mock_components/GenericSystem` cannot do.
- `srdf/`, `config/initial_positions.yaml`, `joint_limits.yaml`, `kinematics_sim.yaml`,
  `moveit_sim_controllers.yaml`, `ros2_control_controllers_sim.yaml`

**Package 2 — `isaac_ros_manipulation_b601_driver_utils/`** (Python you write)
- `config.py` — subclass `DriverConfig` from `isaac_ros_manipulation_ros_python_utils.config`
- `robot_description.py` — xacro → URDF XML string
- `b601_driver_utils.py` — subclass `RobotControllerBase` from
  `isaac_ros_manipulation_robot_utils.robot_controller_base`, implementing
  `get_robot_state_publisher()`, `get_moveit_group_node()`, `get_robot_control_nodes()`
  (ros2_control node + JointTrajectoryController + JointStateBroadcaster + gripper spawners)
- `launch/b601_driver.launch.py`, `params/b601.yaml`
- `src/isaac_sim_joint_parser_node.py` — filters Isaac Sim joint states down to arm joints

**Plus:** routing (`robot_launch_file_path` in the workflow YAML + a `package.xml` dep in
`isaac_ros_manipulation_bringup`), a launch test (`test/test_b601_driver_launch.py`), and the
**XRDF**.

**XRDF** (`b601_gripper.xrdf`): c-space joints with acceleration and jerk limits, tool frames,
per-link collision spheres, self-collision ignore rules. Use Isaac Sim's visual **Robot
Description Editor** (4.0+) for the spheres.

**Before any of it — USD and kinematics sanity (do this first, it's cheap):**
1. URDF → USD with **`joint_drive=None`** or explicit gains (the `UrdfConverterCfg` default has
   `MISSING` stiffness).
2. **Verify actuator limits.** The URDF claims 50 rad/s (joints 1–3) and 200 rad/s (joints 4–6);
   200 rad/s ≈ 1,900 RPM at the joint — motor-side or placeholder. Check Damiao DM4310 /
   DM4340P datasheets, override, **document the override**. Note `joint4` lower limit −1.87 rad.
3. **Cross-validate FK** against `reBot-Isaacsim/mjcf/rebot_devarm` in MuJoCo. One afternoon;
   catches the silent-frame-offset bug class that cost you a week on the Flexiv.

---

## 7. The gripper — its own stage, not a bullet (Sep) → **M6**

Isaac's orchestration expects a **`control_msgs/GripperCommand` action**. The B601 exposes
`gripper_joint1` and `gripper_joint2`, both **prismatic** (verified in
`reBot_B601_DM_with_gripper.urdf:489,547`; there's also a `gripper_joint` at :436 — check
whether it's a mimic/driver joint before wiring anything). You need:

1. A **gripper action controller** (or adapter node) exposing `GripperCommand` and mapping the
   single commanded width onto two prismatic joints.
2. **Contact/friction setup** in the USD so grasps hold — default material params will drop
   objects and it will look like a planning bug.
3. **Attach/detach handling** during transport.
4. **Grasp authoring** for this gripper: Isaac Sim `Isaac Utils → Grasp Editor` → `isaac_grasp`
   YAML. The documented examples are parallel-jaw grippers like the 2F-140; budget real time.

## 8. Perception — why YOLO is not a drop-in

Rev 1 said "replace RT-DETR with `isaac_ros_yolov8`." **That was wrong**, and the reason is
worth understanding because it's the whole design of the Seeed pipeline:

- `isaac_ros_yolov8` publishes `vision_msgs/Detection2DArray` on `detections_output` —
  **axis-aligned boxes only, no masks, no oriented boxes.** It also isn't among the workflow's
  standard detector configs (which use `RTDETR`), so it needs a custom adapter and launch graph.
- Seeed's `utils/yolo_utils.py` uses `result.masks` **and** `result.obb`, and ships
  `yoloe-26s-seg.pt` (segmentation) and `yolov8s-world.pt` (open-vocabulary). **The mask/OBB
  principal axis is where their grasp orientation comes from.** An axis-aligned box cannot
  produce it.

Honest mappings, pick one:

| Want | Isaac ROS route |
|---|---|
| mask → grasp axis (closest to Seeed) | detector box → **`isaac_ros_segment_anything2`** (box-prompted) → mask → principal axis |
| full 6D pose (stronger result) | detector + mask → **`isaac_ros_foundationpose`** |
| open-vocab, like `--target-class "coffee cup"` | **`isaac_ros_grounding_dino`** |

**Sequencing rule:** close the loop with a **trivially-correct pose first** — an ArUco marker
(Seeed ships `aruco_pose.py` and the printable PDFs) or sim ground-truth pose published straight
onto the pose topic. *Then* swap in the real estimator. Debugging perception and integration at
the same time is how weeks disappear.

## 9. S8–S9 — Honest depth, trained perception, writeup (Sep–Oct) → **M7**

1. **Replace ground-truth depth with a simulated RGB-D sensor** — noise, missing returns at
   edges and on specular surfaces. **Measure pick success before and after.** The reference
   workflow's perfect depth is what makes its perception claim hollow; this step is what makes
   your number mean something.
   ⚠️ Isaac ROS 4.5 known issue: *"Workflows that involve pose estimation may generate incorrect
   pose estimates when using ESS or FoundationStereo depth. As a workaround, use
   RealSense/camera sensor depth instead."* Use sensor depth.
2. **Replicator SDG → train a component.** Render RGB-D + 6D pose labels with randomized
   lighting, textures, distractors, extrinsics (there's a *Grasping Synthetic Data Generation*
   tutorial). Fine-tune **on synthetic only**, evaluate in sim, report pose error in mm.
   Be precise in the writeup: this is *a component trained in simulation, evaluated in
   simulation*. It is **not** sim2real until it runs on the arm.
3. **Eval protocol** (same as your VLA work): ≥20 episodes, varied start poses, continuous
   execution, no restart between episodes. Report success with a failure breakdown.
4. **Characterize a hard case** — deformable object through the classical pipeline; pin down
   whether pose, grasp validity, or slip is what fails.
5. **Port one node to C++** (`rclcpp`) — the perception→grasp bridge. Closes G3.
6. **Writeup** in the style of your SmolVLA page.

## 10. Cross-cutting rules

1. **16 GB VRAM is the binding constraint.** Isaac Sim alone is ~8 GB. **Develop against
   rosbags** — record once, run perception offline, go live only when each stage works alone.
   S2 is built around this deliberately.
2. **One substitution at a time.** Every stage changes exactly one thing and ends working.
3. **Git-tag every working state** — `m1-vendor-moveit`, `m2-offline-grasp`, `m3-refworkflow`, …
4. **Isaac Lab is not this project.** It's for RL env authoring; this is Isaac Sim + Isaac ROS.
   *Caveat:* if you author Replicator/SDG through Isaac Lab in §9, your own notes apply there —
   `CameraCfg` scenes need `--enable_cameras`, and `distance_to_image_plane` has hung in
   headless scripts. Through Isaac Sim standalone it's a different entry point.
5. **Don't let Ubuntu upgrade the driver.** 595 crashes Isaac Sim 5.1.

## 11. Hardware bolt-on (later)

Nothing above changes. Swap `TopicBasedSystem` for the real `ros2_control` hardware interface
from `reBotArmController_ROS2` → real hand-eye calibration → **session-to-session zero variance
measurement** (the B601 stores no persistent calibration; motors re-zero against whatever pose
the arm holds at connect — that variance is your noise floor) → rerun the same eval → the
sim2real delta quoted against the noise floor. That's G1, ~2 weeks instead of ~8.

## 12. Open questions

- Does the full reference pick-and-place run on **x86 + Isaac Sim 5.1**? Stated test platform is
  Jetson Thor. S3's timebox exists for this.
- **FoundationPose VRAM** alongside a live Isaac Sim on 16 GB — unmeasured. Rule 1 is the hedge.
- Is `gripper_joint` (URDF :436) a mimic/driver joint for the two prismatic fingers, or separate?
- How close is the **Flexiv Rizon** driver-utils package to what the B601 needs? The structure
  matches the BYOR guide exactly (§6); what's unknown is how much Flexiv-specific logic sits in
  `RobotControllerBase` subclass vs. boilerplate. Reading it is the first task of §6 and could
  pull M5 earlier than Sep 12.

---

### Verification notes (2026-08-04)

Checked on this machine: GPU/driver/OS/disk; Isaac Sim 5.1.0.0 + both bridge distro lib dirs;
`/opt/ros` absent. Cloned and grepped directly: `mock_components/GenericSystem` in
`rebotarm_moveit_config`; `gripper_joint1/2` prismatic in `reBot_B601_DM_with_gripper.urdf`;
`result.masks` + `result.obb` in Seeed's `yolo_utils.py`; models `yoloe-26s-seg.pt`,
`yolov8s-world.pt`, `yolo11n.pt`. From NVIDIA docs: Isaac ROS 4.5 (2026-07-06) Jazzy/24.04/driver
580+; BYOR package and file list; the mac-and-cheese `object_class: 22` and ESS/FoundationStereo
known issues; `isaac_ros_yolov8` publishing `Detection2DArray` axis-aligned only; Flexiv Rizon
support added in 4.5; XRDF fields; Grasp Editor location. Issue #22 read directly (open, Thor,
4.1–4.2 — a caution, not a blocker on x86). B601 actuator-limit values carried from the parent
plan; re-verify against the cloned URDF before trusting them in sim.
