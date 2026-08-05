# reBot B601-DM Visual Grasping — first physics-driven pick in Isaac Sim

**Written:** 2026-08-04 · **Rev 3 (outcome-first), 2026-08-05** — one
contact-driven B601-DM grasp-and-lift is the primary goal; evaluation and full
ROS integration follow only after that loop closes.
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
> **rev 3** keeps the completed smoke work as regression evidence but reverses the critical
> path: validate the shipped articulation → validate two-finger contact → make one fixed
> physics-driven pick → replace fixed waypoints with GT IK → connect the existing perception.

---

## 0. The goal in one line

Make the shipped B601-DM asset grasp and lift one known object in Isaac Sim using
physics-driven arm and finger motion. The first pick deliberately uses known object placement
and scripted joint targets. It does **not** require a perception benchmark, ROS 2, BYOR,
MoveIt, cuMotion, or publication. Once that physical loop works, substitute one input at a
time: fixed joints → GT-pose IK → oracle perception pose → YOLOE pose.

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

## 3. Critical path — one B601-DM pick

**P0, P1 and P2 are complete** (2026-08-05): 59.8 mm rise, 1.0 s hold, 3/3
repetitions, 20/20 checks — see `PICK.md` and `artifacts/b601_pick/b601_pick.json`.
**The next deliverable is P3**, the demo's own `move_to_traj` interface — not a
benchmark or framework integration.

### 3.1 What counts as the first pick

- Use the shipped B601-DM USD and an object initially supported by the scene.
- Command the arm and both fingers with articulation position targets while physics steps.
  `set_joint_positions` may be used for an explicit reset only; it is never scored motion.
- Both `gripper_joint1` and `gripper_joint2` must close on the object through collision/contact.
- The object must rise at least **50 mm** and remain held for at least **1.0 s**, with finite
  joint and object state. Save a machine-readable JSON result and, optionally, a video.
- Teleporting, kinematic parenting, or attaching the object does not count. An assisted
  attachment experiment must be labelled diagnostic and scored separately.
- First make one successful run, then require 3/3 repetitions with the same seed before moving on.

Fixed joint waypoints and ground-truth placement are intentionally allowed for P2. They isolate
asset, drive, collision, and contact problems before IK or perception can obscure them.

### 3.2 Executable sequence

| Stage | Implement | Pass gate |
|---|---|---|
| **P0 · asset/control** | `scripts/b601_asset_probe.py`: load the existing DM USD; assert `joint1`…`joint6` plus the two prismatic finger DOFs; apply runtime gains; command safe targets through physics; measure link poses and aperture. | Finite/stable articulation, safe target tracked without teleport, and finger separation changes monotonically at 0/mid/max. |
| **P1 · contact** | Put a lightweight ~40 mm object at the measured jaw location; tune only explicit collider/friction/contact parameters; close both fingers. | Object settles, both fingers make contact, and no tunnelling or numerical explosion occurs. |
| **P2 · fixed pick** | `HOME → OPEN → GRASP_POSE → CLOSE → LIFT → HOLD`, using smooth joint targets and measured feedback. Place the fixture from a known reachable `q_grasp` so IK is not yet a dependency. | Physical ≥50 mm lift and ≥1 s hold, without teleport or attachment; then 3/3 same-seed runs. |
| **P3 · `move_to_traj`** | Implement the demo's *own* robot interface against the Isaac articulation: `move_to_traj(x,y,z,rx,ry,rz,duration)`, `open_gripper(distance_m)`, `grasp(force)`, `release_gripper()`, `get_tcp_pose()`. IK via the pinned `reBotArm_control_py` (Pinocchio), the same library the demo uses. Reuse the P2 executor and scorer unchanged. | The same contact-based pick succeeds when commanded by Cartesian TCP pose instead of joint waypoints. |
| **P3b · hand-eye** | Implement the demo's `collect_handeye_eih.py` equivalent: solve `AX=XB` (Tsai–Lenz) for the eye-in-hand extrinsic, using the ArUco path the vendor ships. | Solved extrinsic matches the sim's known camera mount to a stated tolerance — the one validation hardware cannot give you. |
| **P4 · perception** | Feed the executor the existing grasp estimate in-process: oracle first, then the pinned YOLOE segmentation checkpoint. | One perception-derived physical pick; Ultralytics approval gates YOLOE only, not P0–P3b. |

P0 must report the exact DOF order and limits, runtime gains, measured position error, base
stability, and unordered finger-link separation at 0/mid/max. Measuring the composed asset
sidesteps the unresolved π mount comparison: the local simulated pads, not cross-tree joint
origins, define the aperture used by P1–P4.

P1 begins with a simple light cuboid and high friction. Infer whether zero or 0.0715 m is open
from measured separation; do not assume it from the joint label. P2 writes its result under
`artifacts/b601_pick/` (gitignored). P3 may use the already available Pinocchio stack with the
DM URDF; XRDF and cuMotion are unnecessary here. P4 preserves the known perspective/quantile
limitations and changes only the source of the target pose.

**Why P3 is named after `move_to_traj` and not "IK".** The demo's entire robot
interface is four calls — `move_to_traj`, `open_gripper`/`grasp`/`release_gripper`,
`get_tcp_pose` — implemented on Pinocchio inside `reBotArm_control_py`
(`drivers/robot/grasp_driver.py:136`, `scripts/main.py:54,92,98`). Naming the stage
after that interface keeps it anchored: reproducing the demo means implementing
*those functions* against the Isaac articulation, not adopting a different motion
architecture. See §6 for what was cut on those grounds.

**Why hand-eye calibration is a stage and not a footnote.** It is a whole script in
the demo (`collect_handeye_eih.py`, `calibration/hand_eye.py`, `aruco_pose.py`) and
the parent plan calls TF/calibration rigor the #1 silent killer. It was missing from
the ladder entirely. In simulation it is *cheaper and stronger* than on hardware:
the true extrinsic is known, so the `AX=XB` solver can be validated against ground
truth — a check that is impossible on the real arm, where the answer is what you are
trying to find.

### 3.3 Existing smoke suite = regression support

The 104 ROS-free tests and completed ten-scene Isaac chain remain valuable for grasp geometry,
depth conversion, serialization/replay, scoring, and overlays. They do not move the B601,
exercise contacts, or prove a pick. Run them after relevant changes, but do not collect or tune
a 300-scene benchmark before P2.

### 3.4 Explicitly deferred until after P2

- the 300-scene evaluation, confidence intervals, and broad model tuning;
- ROS 2 Jazzy, the Action Graph, `TopicBasedSystem`, BYOR, and enum/framework extensions;
- MoveIt, cuMotion, flattened URDF/XRDF, and the stock soup-can reference workflow;
- release-candidate/publication work, licensing/data distribution, domain randomization, and
  larger episode counts.

The audited material below is preserved as a reference for those later phases. It is no longer
the current execution order.

---

## 4. Deferred infrastructure — ROS 2 Jazzy + Isaac ROS

This section resumes after P2. Its S0 gate is required for the later ROS/TopicBasedSystem path,
not for the direct Isaac P0–P4 learning loop.

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

## 6. Cut: BYOR, XRDF, cuMotion, MoveIt — not on the path to this goal

**Removed 2026-08-05.** Earlier revisions planned two authored Isaac ROS packages, a
`RobotType`/`GripperType` framework patch, a flattened URDF + XRDF, and cuMotion
planning. **None of that appears in the demo being reproduced.** Verified in the
pinned tree: `scripts/main.py` moves the arm with
`controller.move_to_traj(x, y, z, rx, ry, rz, duration)` (`:54,92,98`) and
`drivers/robot/grasp_driver.py:136` imports `reBotArm_control_py.kinematics` —
Pinocchio. There is no ROS, MoveIt, cuMotion or XRDF anywhere in the demo's control
path.

So this work is **cut, not deferred**. Deferring it left a large, plausible-looking
backlog that was never going to serve the goal, and kept pulling the critical path
toward a different architecture. If the ROS/Isaac-ROS integration is ever wanted it
is a *separate* project with its own justification, not a later phase of this one.

The full removed text is recoverable from git history (`git show d7a7fa5:PLAN.md`).

### 6.1 The two asset facts worth keeping

These were embedded in the cut section but are properties of the **asset**, and P0–P4
depend on them.

⚠️ **A DM USD ships** — `src/reBot-Isaacsim/usd/reBot_B601_DM/reBot_B601_DM.usda`,
with `payloads/` for base, robot, geometries, materials and physics. The
"RS-variant only" claim is false for the current tree, so the task is
**select → validate → patch only if required**, not convert. P0 validated it.

⚠️ **The MJCF parity model is the RS arm, not DM.** `mjcf/rebot_devarm/rebot_devarm.xml`
uses joint classes `rs06`/`rs00`, gripper joints `joint_left`/`joint_right`, and
joint4 range `-1.79 1.69` (the DM URDF says −1.87). Adapt the cross-validation
*technique*; it is not DM ground truth.

### 6.2 The π gripper-mount question — superseded by measurement

The two DM URDFs disagree by π in the `gripper_joint` mount roll
(`reBot-Isaacsim/.../reBot_B601_DM.urdf:251` has `rpy="3.1416 -1.5708 0"`;
`reBotArmController_ROS2/.../reBot_B601_DM_with_gripper.urdf:435` has
`rpy="0 -1.5708 0"`), while everything below the mount is byte-identical and the
mesh sets are disjoint.

**P0 makes this moot for P1–P4.** The probe measures the *composed simulated asset*:
finger-link separation of 0.059 / 71.445 / 142.943 mm about a jaw midpoint that is
invariant to 5 decimals across the sweep. Local measured geometry, not a
cross-tree origin comparison, defines the aperture the pick uses. The question only
returns if the vendor URDF is ever used to drive the shipped USD.

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
