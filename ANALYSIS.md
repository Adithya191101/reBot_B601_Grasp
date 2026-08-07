# Why the robot rendered frozen, why the gripper works, and who else has done this

**Date:** 2026-08-07 · **Status: both mysteries solved, fixes measured and shipped**

Evidence labels used throughout — nothing here is asserted without one:

- **[measured]** — produced by a script in this repo, artifact on disk
- **[fetched]** — a research agent WebFetched and read the cited page
- **[snippet]** — search-result-level evidence only, not independently read
- **[hypothesis]** — explicitly not established

A 14-agent research sweep fed the prior-art and SOTA sections
(`wf_915bd70d-e4f`; six finders, all completed). Its adversarial verification
stage failed on a plumbing bug (the verify agents received empty URLs), so
**per-finding re-verification did not run**; finder-level fetches are labeled
[fetched] on the finders' own testimony.

---

## 1. The robot not moving in the video — SOLVED

### The symptom, quantified [measured]

In the first recorded pick video, over the 17-second calibration-sweep window:
arm centroid moved **0.23 px** while the tensor API recorded **~100 mm ≈ 107 px**
of jaw travel. The rigid-body block rendered its motion correctly in the *same
frames* (appearing, falling 332.8 mm).

### The three-way instrument [measured]

`scripts/b601_render_experiment.py` samples, at every step of a joint2 sweep:
the PhysX tensor API (truth), the composed USD transform (what the renderer
uses), and the rendered pixels. Results (`artifacts/render_exp/`):

| Scene | tensor jaw-z | USD jaw-z | changed pixels |
|---|---|---|---|
| DM USD + session repair | **77.0 mm** | **0.00 mm** | 129 (noise floor) |
| URDF import, no repair | moves | **tracks tensor** | **58,563** |
| DM USD + repair + per-capture sync | 77.0 mm | **77.03 mm** | **109,014** |

### Root cause [measured]

1. The DM USD's nine nested rigid bodies require the P0 session repair —
   `resetXformStack` + an explicit transform op — to be a valid PhysX
   articulation at all (without it the articulation has **0 DOFs**).
2. **`resetXformStack` disables PhysX's physics→USD writeback on those prims.**
   Forensics show PhysX still *writing* live `xformOp:translate/orient` values,
   but `xformOpOrder = [!resetXformStack!, xformOp:transform:b601PhysxRepair]`
   composes only the static repair op, so the live values never take effect.
   Re-authoring the repair as translate/orient/scale ops does **not** help
   (USD range 0.00 mm — the writeback simply does not land once reset is set).
3. The renderer draws the composed result → arm frozen at spawn pose. The block
   has no repair op → its writeback works → it renders.
4. `/physics/updateToUsd` was **already `true`** the whole time [measured,
   settings dump] — the setting was never the problem.

This also retroactively explains **bug #1 from PICK.md** (stale
`get_world_pose` on arm links): same writeback gap, seen through a different API.

### Corroborating official architecture [fetched]

- omni.physx writes sim output to USD (`/physics/updateToUsd`) or to Fabric;
  USD writeback is called an "expert setting" and Fabric is the recommended
  path — [omni_physics settings](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/settings.html)
- Stale `get_world_pose` during simulation is a known, staff-answered pattern —
  [forum 245372](https://forums.developer.nvidia.com/t/world-position-not-updating-during-simulation-run/245372),
  [forum 270965](https://forums.developer.nvidia.com/t/bug-isaac-sim-does-not-update-world-poses-which-makes-imitation-policies-nearly-impossible-to-implement/270965)
- OmniHydra transform precedence and Fabric-vs-USD reads —
  [usdrt omnihydra xforms](https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/latest/omnihydra_xforms.html);
  FSD gained `resetXformStack` support only in Kit 106.2 —
  [release notes](https://docs.omniverse.nvidia.com/dev-guide/latest/release-notes/106_2.html) [snippet]
- No forum thread was found with this *exact* symptom resolved — the specific
  interaction (session-layer resetXformStack repair × articulation writeback ×
  render) appears **undocumented**; our measurement is the primary source.

### The fix, shipped [measured]

`scripts/b601_usd_sync.py` — pushes tensor link transforms into the repair ops,
throttled to captured frames, batched in an `Sdf.ChangeBlock` (per-step writes
destabilised Kit). Wired into the pick's `Recorder`; resilient to the tensor-view
invalidation inside `world.reset()`.

**Re-recorded result:** run passes unchanged (59.6 mm rise, 332.8 mm fall);
arm centroid travel in the sweep window **25.4 px vs 0.44 px** — the arm, the
calibration sweep, the grasp, the lift and the release are all now visible.

### For the record: two wrong turns, kept honest

- Three "segfaults" were actually a JSON-serialization crash *inside `finally`*
  (a raw `Gf.Matrix4d`), which skipped `sim_app.close()` and produced a messy
  Kit atexit. The crash-reporter's relative timestamps mimicked a startup crash.
- The vendor's own Isaac visualizer cannot hit this bug: it drives the arm with
  `set_joint_positions` (kinematic teleport) on the RS asset, no physics drives,
  no repair [measured — `reBotArm_Isaacsim/isaacsim_joint_receiver.py:369`].

---

## 2. Bonus root cause: the URDF-import joint4 sag — SOLVED

Standing mystery from the import probe: joint4 settling **0.807 rad** off target
with a constant ~0.15 rad/s creep, unaffected by solver iterations.

**Root cause [measured]:** the importer authors every drive as
`drive:type = "acceleration"` with the URDF effort as `maxForce`
(joint4: 7.0). The hand-authored USD uses `type = "force"` (P0 dump), where
7 N·m dwarfs the ~0.64 N·m gravity load. Under acceleration-drive semantics the
same cap starves the low-inertia wrist. Eliminated en route [measured]:
`maxJointVelocity` = 11,459 °/s (no clamp), `jointFriction` = 0, `armature` = 0,
no URDF `<dynamics>` tags.

**Fix [measured]:** flip the imported drives to `"force"` —
`artifacts/render_exp/urdf_force.json`: joint4 error **0.807 → 0.004–0.013 rad**,
all eight joints track. Shipped into the teleop, whose self-test now reports
max|err| 0.030 rad (was 0.8075).

Background that pointed there [fetched]: USD angular drives are per-degree
([UsdPhysics.DriveAPI](https://openusd.org/release/api/class_usd_physics_drive_a_p_i.html),
[forum 284971](https://forums.developer.nvidia.com/t/units-of-angular-velocity-in-e-g-revolute-joints-in-degree-per-second/284971));
Isaac Lab's converter compensates gains ×π/180 for revolute joints
([urdf_converter](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/sim/converters/urdf_converter.html));
acceleration drives normalize inertia — failure lands on the lowest-inertia
joint. Runtime `set_gains` never helped because it sets stiffness/damping, not
drive type or maxForce.

---

## 3. Gripper open/close and contact grasping — ours vs SOTA

### What this repo measured before the sweep [measured]

- Shipped finger colliders are `convexHull`: the two hulls overlap closed
  (the real reason P0 had to disable self-collision) and leave **~24 mm of
  usable throat against 143 mm of authored travel**; a 40 mm object is ejected.
- `convexDecomposition` fixes it: first-contact aperture **41.0 mm for a
  40.0 mm object** (0.5 mm inset/finger, vs 40 mm with hulls).
- Close-to-contact via drive-tracking error (0.5 mm steps, both fingers
  resisted, two consecutive), squeeze 6.0 mm past contact, friction 1.2/1.1,
  position drives with URDF effort caps (100), solver iterations 32/4.
  Result: 59.6 mm lift, 6.4 mm slip under Cartesian transport, clean release.

### The Aug-2026 sanctioned practice agrees [fetched]

- **Collision:** never leave mesh colliders on dynamic fingers — PhysX silently
  falls back to convexHull whose bridged volume prevents contact; fix is
  convexDecomposition — [IsaacLab discussion #2651](https://github.com/isaac-sim/IsaacLab/discussions/2651)
  (the *same defect and same fix* we found independently). NVIDIA's
  [GraspDataGen](https://github.com/NVlabs/GraspDataGen) defaults objects to
  convexDecomposition; NVIDIA physics staff recommended convex decomposition
  over SDF for grasp stability — [forum 253107](https://forums.developer.nvidia.com/t/objects-clipping-through-gripper/253107).
- **Friction:** GraspDataGen uses 1.0/1.0; community-confirmed grasps ~1.5.
  Ours (1.2/1.1) sits in the band.
- **Finger control, two sanctioned patterns:** (a) position-drive to closed
  with effort cap (Isaac Lab Franka: stiffness 2e3 / damping 1e2 / effort 200)
  — ours is this pattern; (b) effort-limited velocity-style closing (stiffness
  0, damping 5000, tuned Max Force — the [Robotiq tutorial](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html)
  drops it from 180 to 5) — the closest sim analogue of the vendor's
  force-control `grasp(force)`, worth adopting when P4 needs calibrated squeeze.
- **Offsets/solver:** contactOffset 0.005–0.01 m for earlier contact;
  GraspDataGen validates at position-iteration counts up to 64.

**Assessment:** our contact pipeline independently converged on the sanctioned
approach; the one SOTA technique we don't use yet is effort-mode closing for
calibrated grip force.

---

## 4. Has anyone done this for the reBot, anywhere? [fetched unless noted]

Six finders, English + Chinese web, GitHub, Gitee, Bilibili, Hugging Face.

**Seeed official:**
- [reBot-Isaacsim](https://github.com/Seeed-Projects/reBot-Isaacsim) — UDP
  joint mirroring; `set_joint_positions` teleport; control/protocol testbed.
  No contact grasping [also verified locally in the pinned tree].
- [LeRobot integration blog](https://www.seeedstudio.com/blog/2026/07/08/seeed-rebot-arm-successfully-integrates-with-lerobot-v0-6-0-completing-the-robot-learning-loop-in-nvidia-isaac-simulation/)
  + LeRobot PRs [#3624](https://github.com/huggingface/lerobot/pull/3624),
  [#3955](https://github.com/huggingface/lerobot/pull/3955) and docs
  ([rebot_b601](https://huggingface.co/docs/lerobot/main/en/rebot_b601),
  [isaac_teleop](https://huggingface.co/docs/lerobot/main/en/isaac_teleop)) —
  teleop → data → GR00T fine-tune → sim validation. Data loop, not a
  contact-grasp loop.
- Chinese sources ([CN wiki](https://wiki.seeedstudio.com/cn/rebot_arm_b601_rs_isaacsim/),
  [Gitee mirror](https://gitee.com/seeed-projects/reBot-DevArm), two official
  Bilibili videos ~137k plays each) — all Seeed's own material; the launch
  video *markets* Isaac/RL as directions without demonstrating them; the only
  grasping demo anywhere is the real-hardware YOLO one.

**Third-party (all mid-2026, all sparse):**
- [XIAOHU7771/my-reBot-DevArm](https://github.com/XIAOHU7771/my-reBot-DevArm) —
  **PyBullet** force-servo grasping (virtual force sensors, adaptive PID, rigid
  block + soft sponge, slip compensation). The only other contact-grasp work
  found for this arm, in a different simulator.
- [isaiahbjork/rebot-teleop](https://github.com/isaiahbjork/rebot-teleop) —
  **MuJoCo** iPhone/ARKit teleop with a pick-and-lift demo (human in the loop).
- [johnnynunez/sim2real-rebot-devarm](https://github.com/johnnynunez/sim2real-rebot-devarm)
  — Isaac Sim + Newton real-to-sim mirroring + experimental Quest 3 teleop;
  explicitly no pick-place or RL.
- [HJX-exoskeleton/reBotArm_develop_hjx](https://github.com/HJX-exoskeleton/reBotArm_develop_hjx),
  [LAN-GER/reBot-B601-RS-for-mujoco_sim](https://github.com/LAN-GER/reBot-B601-RS-for-mujoco_sim)
  — MuJoCo digital twins (gravity comp, impedance, CAN sync); grasp objects are
  props.
- RL training for this arm in any simulator: **nothing found.** Autonomous
  (non-teleop) pick-and-place in any simulator: **nothing found.**
  [negative results; GitHub code search requires login, a stated coverage gap]

**Positioning:** as of Aug 2026, on both the English and Chinese web, an
**autonomous, physics-contact grasp of the B601 in Isaac Sim appears to be
unpublished by anyone else** — the nearest neighbours are force-control grasping
in PyBullet and human-teleoperated picks in MuJoCo. Stated with the standard
caveat: absence of evidence across six search angles, not proof of absence.

---

## 5. Open items

- The render-freeze fix is capture-time sync; the **interactive viewport for
  the USD asset still freezes** between syncs (URDF-source teleop is unaffected
  and now tracks). A per-frame sync hook in the teleop would close that.
- The A/B silhouette change seen in one early diagnostic (8% dark-pixel drift)
  remains attributed to denoiser convergence [hypothesis] — superseded by the
  three-way instrument rather than resolved.
- The workflow's verify stage needs its URL-plumbing bug fixed before its
  verdicts are usable; finder-level fetches stand on their own testimony.
- With the drive-type fix, the imported URDF now tracks *and* renders natively
  — it is a live candidate to replace the repaired USD entirely, which would
  delete the whole repair/sync apparatus. One P2-gate rerun on the imported
  asset would settle it.
