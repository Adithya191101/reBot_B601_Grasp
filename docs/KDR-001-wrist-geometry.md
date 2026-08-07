# KDR-001 — Kinematics Decision Record: wrist geometry

Per design doc §0.2. Recorded 2026-08-07 against pinned vendor tree
`reBotArmController_ROS2 @ 39fbea5`.

## Decision

**Canonical model: `reBot_B601_DM_with_gripper.urdf`** (SHA-256 prefix
`f808f6f0d33274b2`). It is the model validated end-to-end in Isaac Sim by the
sim-only grasping track: URDF import probe, teleop, and the full
perception-driven pick (74 mm lifts, <2 mm slip, 3/3).

**Canonical TCP:** `gripper_tcp` = `gripper_link` ⊕ **[-41.763, 0.008,
3.427] mm** (P3-calibrated jaw midpoint, EE frame). Must be re-measured on
real hardware before hardware grasping (M12).

## The doc's 4.316 mm alarm — resolved by measurement

The two DM models differ at `joint6` origin x (with_gripper 0.023692 vs
fixend 0.028008 m) and at the link6→terminal z (0.15971 vs 0.15539 m). These
are an internal **re-parenting that cancels exactly**:

```
0.023692 + 0.004316 = 0.028008
0.15971  − 0.00432  = 0.15539
```

FK proof (Pinocchio, 1000 uniform-random in-limit configurations, seed 7):

| Quantity | mean | max | min |
|---|---|---|---|
| ‖ TCP(with_gripper ⊕ calib) − end_link(fixend) ‖ | 41.899 mm | 41.899 mm | 41.899 mm |

A configuration-independent norm means the chains agree at the terminal
point; the residual 41.899 mm **is exactly** ‖[-41.763, 0.008, 3.427]‖ — the
jaw offset itself. Therefore `end_link`(fixend) coincides with
`gripper_link`(with_gripper), and the driver-model transform to the canonical
TCP is the constant `T_end_link_gripper_tcp` = translation [-41.763, 0.008,
3.427] mm.

**Open sub-check (in-container, M3′):** orientation equivalence of
`end_link` vs `gripper_link` frames (the norm test proves position only).

## Recorded transforms (canonical model, q = 0)

| Transform | Value |
|---|---|
| T_link5_link6 (joint6 origin) | t = [0.023692, 0, 0.04] m |
| T_link6_gripper_link | t = [0, 0, 0.15971] m |
| T_gripper_link_gripper_tcp | t = [-0.041763, 0.000008, 0.003427] m (measured, sim) |
| T_link6_camera_mount | **not yet authored** — required before M8 perception (camera-architecture decision pending) |

## File hashes (pinned vendor tree 39fbea5)

| File | SHA-256 (prefix) |
|---|---|
| reBot_B601_DM_with_gripper.urdf | f808f6f0d33274b2 |
| reBot-DevArm_fixend.urdf | 4171bdf8d12102b1 |

Any future wrist change invalidates these hashes and forces grasp-library
regeneration (doc rule).

## Generated URDF product

`urdf/rebot_b601dm_canonical.urdf` (SHA-256 prefix **dae842f4f4fa89d4**):
byte-identical to the vendor with_gripper URDF except arm velocity limits
replaced with **5,5,5,3,3,3 rad/s** (the vendor's 50/200/15 are unphysical;
cuMotion reads the URDF, not MoveIt YAML). Verified via Pinocchio:
velocityLimit[:6] = [5,5,5,3,3,3]. This is the file every downstream
consumer (MoveIt, XRDF, USD import, driver-URDF generation) must reference.

## Outstanding M3′ items

1. `check_urdf` + FK parity + mimic +1.0 + frame-contract tests re-run inside
   the Jazzy container once M0 completes.
2. Orientation-equivalence sub-check for end_link vs gripper_link.
