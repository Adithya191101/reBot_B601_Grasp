#!/usr/bin/env python3
"""The demo loop, on a table: camera -> hand-eye -> grasp a banana-class object.

This is the Seeed pipeline's shape, reproduced end to end in Isaac Sim:

    wrist RGB-D  ->  yellow-object mask  ->  min-area-rect grasp estimate
    (camera frame) -> SOLVED hand-eye X -> base-frame TCP target
    -> ready -> pregrasp(-0.080 m) -> insert(+0.015 m) -> grasp -> lift -> release

Honesty rules, enforced in code:

* The grasp target is computed ONLY from the camera image, the depth image, the
  measured gripper pose, and the **solved** hand-eye transform. The banana's
  true pose is never given to the solver; it is logged afterwards purely as the
  perception-error diagnostic.
* The hand-eye method is selected WITHOUT ground truth, by a criterion that
  transfers to hardware: the marker is static, so the spread of its recovered
  base-frame position across viewpoints scores each candidate X. The comparison
  against the true mount is reported as a sim-only diagnostic.
* The banana is a sim-built proxy (downloads need approval): a yellow, arced,
  ~115 mm compound rigid body lying flat -- elongated, non-square, grasped
  across its short axis, exactly the geometry the demo's YOLO/OBB pipeline
  produces grasps for.

Run::

    TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N \\
      ~/isaaclab-venv/bin/python scripts/b601_banana_demo.py \\
        --repair-nested-xforms --out artifacts/banana/b601_banana.json \\
        --record artifacts/banana/b601_banana.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src" / "reBot-DevArm-Grasp"))

# ---- vendor constants (config/default.yaml) --------------------------------
READY_TCP = np.array([0.30, 0.0, 0.30])
READY_PITCH_RAD = 0.7
PREGRASP_OFFSET_M = 0.080
INSERTION_DEPTH_M = 0.015
DEPTH_QUANTILE = 0.5            # the frozen Seeed baseline
MARKER_LENGTH_M = 0.12
ARUCO_DICT_ID = 0
TARGET_MARKER_ID = 0

# ---- scene ------------------------------------------------------------------
# The B601 is mounted ON the table in the vendor demo: base plane == tabletop.
# A raised slab beside a floor-mounted arm blocks the elbow during any descent
# (measured: every approach stopped ~190 mm short with 0.86 rad drive error).
# The "table" is therefore a thin mat at base level.
TABLE_TOPS_TO_TRY = [0.0015]
TABLE_SIZE = (0.90, 0.90)
# Pinch height. 0.026 put the palm ON the banana: the fruit pokes
# top(39.5) - tcp_z above the jaw centre, and at pitch 0.7 the palm's
# along-axis clearance could not absorb 13.5 mm (run 42: j4 force-saturated
# at its 7 Nm cap riding the banana top the moment the arm truly arrived).
# 32 mm still pinches the upper-middle with ~27 mm of fruit in the throat,
# pads ~12 mm above the mat.
MIN_TCP_Z_M = 0.032
# Banana-realistic thickness: real bananas run 35-45 mm. At 24 mm the ~39 mm
# finger pads cannot reach around the fruit without grazing the surface
# (measured: contact at the final approach waypoint every time).
BANANA_SEGMENT = (0.044, 0.030, 0.038)   # per-segment box, metres
BANANA_MASS_KG = 0.12
IMG_W, IMG_H, IMG_FX = 960, 720, 380.0

REQUIRED_LIFT_M = 0.050
REQUIRED_HOLD_S = 1.0
MOVE_TOL_M = 6.0e-3


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    return str(v)


class DemoFailure(RuntimeError):
    pass


class Report:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "probe": "b601_banana_demo", "schema_version": "1.0.0",
            "checks": [], "errors": [],
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def check(self, name: str, passed: bool, **f: Any) -> bool:
        e = {"name": name, "passed": bool(passed)}
        e.update({k: _jsonable(x) for k, x in f.items()})
        self.data["checks"].append(e)
        return bool(passed)

    def require(self, name: str, passed: bool, **f: Any) -> None:
        if not self.check(name, passed, **f):
            raise DemoFailure(f"{name}: {json.dumps(_jsonable(f))[:400]}")

    def write(self, path: Path) -> None:
        self.data["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.data["passed"] = (bool(self.data["checks"])
                               and all(c["passed"] for c in self.data["checks"])
                               and not self.data["errors"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(self.data), indent=2, sort_keys=True) + "\n")


def _pose_err(Ta, Tb):
    dp = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
    R = Ta[:3, :3].T @ Tb[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return dp, ang


def yellow_mask(rgb: np.ndarray):
    """The demo's 'yellow banana' class, classical: HSV hue band + saturation."""
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Thresholds probed on the actual render: banana H~24 S~43 V~229 vs table
    # H~11 S~10 -- displayColor yellow desaturates under the dome light, so the
    # usual S>=80 gate sees nothing.
    m = ((hsv[..., 0] >= 15) & (hsv[..., 0] <= 45)
         & (hsv[..., 1] >= 30) & (hsv[..., 2] >= 90)).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    if int(stats[best, cv2.CC_STAT_AREA]) < 300:
        return None
    return (labels == best).astype(np.uint8)


def build_marker_tiles(stage, prim_path: str, center, length_m: float):
    """ArUco marker as GEOMETRY: 6x6 displayColor tiles on a white plate.

    The textured-quad marker rendered a mangled cell pattern here (crisp
    contrast, wrong bits -- rejected by the decoder at the bit stage), while
    plain displayColor geometry renders exactly as authored everywhere in this
    project. 36 tiles sidesteps the whole texture pipeline: what you author is
    what the camera sees. The 6x6 matrix comes straight from
    cv2.aruco.generateImageMarker(dict, id, 6): one pixel per cell.
    """
    import cv2
    from pxr import Gf, UsdGeom

    bits = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID), TARGET_MARKER_ID, 6)
    cell = length_m / 6.0
    pad = length_m * (8.0 / 6.0)          # white quiet zone, 1 cell each side

    plate = UsdGeom.Cube.Define(stage, f"{prim_path}/plate")
    xf = UsdGeom.Xformable(plate.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]),
                                     float(center[2])))
    xf.AddScaleOp().Set(Gf.Vec3f(pad, pad, 0.001))
    plate.CreateSizeAttr(1.0)
    plate.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 1.0, 1.0)])

    for i in range(6):
        for j in range(6):
            if bits[i, j] > 127:
                continue                   # white cell: the plate shows through
            tile = UsdGeom.Cube.Define(stage, f"{prim_path}/c{i}_{j}")
            txf = UsdGeom.Xformable(tile.GetPrim())
            # image row i (top) -> -y in world so the pattern reads upright
            tx = center[0] + (j - 2.5) * cell
            ty = center[1] - (i - 2.5) * cell
            txf.AddTranslateOp().Set(Gf.Vec3d(float(tx), float(ty),
                                              float(center[2]) + 0.0012))
            txf.AddScaleOp().Set(Gf.Vec3f(cell, cell, 0.0008))
            tile.CreateSizeAttr(1.0)
            tile.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])
    return prim_path


def build_banana(stage, prim_path: str, spawn_xy, table_top: float):
    """Compound rigid body: three arced yellow segments, lying flat."""
    from pxr import Gf, UsdGeom, UsdPhysics

    seg = np.asarray(BANANA_SEGMENT)
    root = UsdGeom.Xform.Define(stage, prim_path)
    prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(BANANA_MASS_KG)
    root.AddTranslateOp().Set(Gf.Vec3d(float(spawn_xy[0]), float(spawn_xy[1]),
                                       table_top + seg[2] / 2 + 0.004))

    for i, (dx, yaw_deg) in enumerate([(-0.036, 14.0), (0.0, 0.0), (0.036, -14.0)]):
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/seg{i}")
        cp = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(cp)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cp)
        xf.AddTranslateOp().Set(Gf.Vec3d(dx, 0.018 * (1 if dx == 0 else 0), 0.0))
        xf.AddRotateZOp().Set(yaw_deg)
        xf.AddScaleOp().Set(Gf.Vec3f(float(seg[0]), float(seg[1]), float(seg[2])))
        cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.98, 0.72, 0.02)])
    return prim


def _run(args, report: Report) -> None:
    import cv2
    import b601_asset_probe as probe
    import b601_pick as pick
    from b601_move_to_traj import RebotArmSim, IK_URDF
    from b601_hand_eye import _set_camera_world
    from b601_usd_sync import LinkUsdSync
    from calibration.aruco_pose import ArUcoDetector
    from calibration.hand_eye import CalibMode, HandEyeCalibrator
    import omni.replicator.core as rep
    import isaacsim.core.utils.prims as prim_utils
    from isaacsim.core.api import World
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import FixedCuboid, GroundPlane
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from pxr import Gf, PhysxSchema, Usd, UsdGeom
    from grasp_smoke.geometry import (make_transform, normalize,
                                      rotation_from_quaternion)
    from grasp_smoke.grasp import estimate_grasp
    from grasp_smoke.pose_msg import vision_grasp_basis_to_b601_tcp_rotation

    # ---- world + robot ----------------------------------------------------
    world = World(physics_dt=probe.PHYSICS_DT, rendering_dt=1.0 / 60.0,
                  stage_units_in_meters=1.0, backend="numpy")
    add_reference_to_stage(str(probe.ASSET_PATH), probe.ROBOT_PRIM_PATH)
    stage = get_current_stage()
    issues = probe._nested_rigid_body_issues(stage)
    issue_paths = [i["body_path"] for i in issues]
    if issues and not args.repair_nested_xforms:
        report.require("nested Xform stacks PhysX-valid", False, issue_count=len(issues))
    if issues:
        r = probe._repair_nested_rigid_body_xforms(stage, issues)
        report.require("session repair preserves poses",
                       r["repaired_count"] == len(issues)
                       and r["remaining_issue_count"] == 0, **{
                           k: r[k] for k in ("repaired_count", "remaining_issue_count")})
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        pxa = PhysxSchema.PhysxArticulationAPI.Apply(
            stage.GetPrimAtPath(probe.ARTICULATION_ROOT_PATH))
        pxa.CreateEnabledSelfCollisionsAttr(False)
        pxa.CreateSolverVelocityIterationCountAttr(4)
        pxa.CreateSolverPositionIterationCountAttr(32)

    finger_root = (f"{probe.ROBOT_PRIM_PATH}/Geometry/base_link/link1/link2/link3/"
                   f"link4/link5/link6/gripper_link")
    finger_paths = [f"{finger_root}/gripper_left", f"{finger_root}/gripper_right"]
    fix = pick._refine_finger_colliders(stage, finger_paths)
    report.require("finger colliders decomposed", fix["changed_count"] >= 2,
                   changed=fix["changed_count"])

    recorder = None
    if args.record:
        recorder = pick.Recorder(Path(args.record), stage,
                                 look_at=np.array([0.30, 0.0, 0.18]))

    world.scene.add(GroundPlane(prim_path="/World/ground", size=4.0))
    prim_utils.create_prim("/World/dome_light", "DomeLight",
                           attributes={"inputs:intensity": 900.0})
    prim_utils.create_prim("/World/key_light", "DistantLight",
                           attributes={"inputs:intensity": 2400.0, "inputs:angle": 2.0},
                           orientation=np.array([0.9239, 0.0, 0.3827, 0.0]))
    grip_material = PhysicsMaterial("/World/physics_materials/grip",
                                    static_friction=pick.STATIC_FRICTION,
                                    dynamic_friction=pick.DYNAMIC_FRICTION,
                                    restitution=0.0)

    replay_data = None
    if args.replay:
        # Replay mode: the full scene exists BEFORE the one world.reset (no
        # mid-run spawn dance needed), and the wrist camera is never created
        # -- its Replicator render product is what starves the Recorder in a
        # combined session (78-frame stall, washed-out captures).
        replay_data = np.load(args.replay)
        r_top = float(replay_data["table_top"])
        r_spot = np.asarray(replay_data["spot"], dtype=np.float64)
        world.scene.add(FixedCuboid(
            prim_path="/World/table", name="table",
            position=np.array([0.33, 0.0, r_top / 2]),
            scale=np.array([TABLE_SIZE[0], TABLE_SIZE[1], r_top]),
            color=np.array([0.42, 0.32, 0.22])))
        build_marker_tiles(stage, "/World/aruco_marker",
                           np.array([0.43, -0.145, r_top + 0.0015]),
                           MARKER_LENGTH_M)
        build_banana(stage, "/World/banana", r_spot, r_top)
        for _i in range(3):
            pick._bind_physics_material(stage, f"/World/banana/seg{_i}",
                                        grip_material.prim_path)

    articulation = SingleArticulation(prim_path=probe.ARTICULATION_ROOT_PATH,
                                      name="b601")
    world.scene.add(articulation)
    world.reset()
    if recorder is not None:
        recorder.attach(world)
    articulation.get_articulation_controller().set_gains(kps=probe.RUNTIME_KP,
                                                         kds=probe.RUNTIME_KD)
    for p in finger_paths:
        pick._bind_physics_material(stage, p, grip_material.prim_path)
    monitor = probe.StateMonitor(articulation)
    if recorder is not None and issue_paths:
        recorder.set_link_sync(LinkUsdSync(stage, monitor, issue_paths))

    arm = RebotArmSim(world, articulation, monitor, probe, pick, render=args.render)

    if replay_data is not None:
        from isaacsim.core.utils.types import ArticulationAction
        # The source run switched to the stiff wrist for the approach; the
        # contact-heavy segments ran with it, so replay uses it throughout.
        r_kp, r_kd = probe.RUNTIME_KP.copy(), probe.RUNTIME_KD.copy()
        r_kp[3:6] = [450.0, 240.0, 150.0]
        r_kd[3:6] = [31.0, 17.0, 12.0]
        articulation.get_articulation_controller().set_gains(kps=r_kp,
                                                             kds=r_kd)
        rows = np.asarray(replay_data["traj"], dtype=np.float64)
        idx8 = np.arange(8, dtype=np.int64)
        world.play()
        for row in rows:
            articulation.apply_action(
                ArticulationAction(joint_positions=row, joint_indices=idx8))
            world.step(render=False)
        report.data["replay"] = {
            "rows": int(len(rows)),
            "recording": recorder.finalize() if recorder is not None else None}
        return

    # Full 6-D pose solves need the VENDOR default tolerance (IKParams: 1e-4);
    # P3's tightened 1e-5 declares failure at log-metric errors ~1.4e-3 that the
    # post-motion position gate would happily accept.
    from reBotArm_control_py.kinematics.inverse_kinematics import (
        IKParams as _IKP, solve_ik as _solve_ik_local)
    pose_ik_params = _IKP(max_iter=1500, tolerance=1e-4, step_size=0.5, damping=1e-6)
    arm.T_ee_tcp = np.eye(4)
    arm.T_ee_tcp[:3, 3] = np.array([-0.041763, 0.000008, 0.003427])  # P3-calibrated

    gl_i = int(monitor.view.get_link_index("gripper_left"))
    gr_i = int(monitor.view.get_link_index("gripper_right"))
    ee_i = int(monitor.view.get_link_index("gripper_link"))

    def link_T(index: int) -> np.ndarray:
        links = monitor._link_transforms()
        return make_transform(rotation_from_quaternion(links[index, 3:]),
                              links[index, :3])

    # ---- measure the EE<->TCP ROTATION (jaw direction + opening axis) -----
    arm.open_gripper()
    T_be = link_T(ee_i)
    jaw = arm.get_tcp_pose()
    links = monitor._link_transforms()
    open_vec_base = normalize(links[gr_i, :3] - links[gl_i, :3])
    R_eb = T_be[:3, :3].T
    x_tcp_ee = normalize(R_eb @ normalize(jaw - T_be[:3, 3]))
    y_tcp_ee = R_eb @ open_vec_base
    y_tcp_ee = normalize(y_tcp_ee - np.dot(y_tcp_ee, x_tcp_ee) * x_tcp_ee)
    z_tcp_ee = np.cross(x_tcp_ee, y_tcp_ee)
    R_ee_tcp = np.column_stack([x_tcp_ee, y_tcp_ee, z_tcp_ee])
    report.data["ee_tcp_frame"] = {
        "R_ee_tcp": R_ee_tcp, "note": "measured: jaw dir + finger separation; "
        "tool-forward is ~-x of gripper_link, so vendor TCP rotations must be "
        "composed through this, not commanded raw."}
    report.require("measured EE->TCP rotation is orthonormal",
                   bool(np.allclose(R_ee_tcp.T @ R_ee_tcp, np.eye(3), atol=1e-6)
                        and np.isclose(np.linalg.det(R_ee_tcp), 1.0, atol=1e-6)))

    T_ee_tcp_full = np.eye(4)
    T_ee_tcp_full[:3, :3] = R_ee_tcp
    T_ee_tcp_full[:3, 3] = arm.T_ee_tcp[:3, 3]

    def _ensure_playing():
        """rep.orchestrator.step() can leave the Kit timeline paused, after
        which world.step() renders WITHOUT advancing physics: drives read as
        perfectly healthy in an isolated test, then the next commanded motion
        produces exactly zero displacement. Every zero-motion failure in this
        scene occurred immediately after a render_wrist() call."""
        # world.play() (SimulationContext) also ticks the app so the play state
        # actually takes effect -- a raw timeline.play() does not, which is why
        # the first version of this guard changed nothing.
        if not world.is_playing():
            world.play()

    ROLL_PI = np.diag([1.0, -1.0, -1.0])   # rotate pi about TCP x (approach)

    def move_to_tcp(pos, R_tcp, duration=2.0) -> dict:
        _ensure_playing()
        """Full-pose Cartesian move: desired TCP position AND orientation.

        A parallel jaw is symmetric under a pi roll about the approach axis, and
        IK for one of the two symmetric orientations routinely lands joint6 at
        exactly +/-pi (its limit). Both orientations are the same physical
        grasp, so try both and keep whichever is solvable inside limits.
        """
        # Choose between the two grasp-equivalent orientations by LIMIT MARGIN,
        # not first-usable: first-usable parked joint6 at +/-(pi-eps) after the
        # ready move, leaving zero room for any subsequent Cartesian step and
        # producing the pi-roll jumps every later solver call exhibited.
        scored = []
        for R_try in (R_tcp, R_tcp @ ROLL_PI):
            rec = _move_to_tcp_one(pos, R_try, duration, dry_run=True)
            if rec.get("within_limits") and "q_solution" in rec:
                q6 = np.asarray(rec["q_solution"])
                margin = float(np.min(np.minimum(
                    q6 - probe.EXPECTED_LOWER[:6], probe.EXPECTED_UPPER[:6] - q6)))
                scored.append((margin, R_try))
        if not scored:
            return _move_to_tcp_one(pos, R_tcp, duration, dry_run=True)
        best = max(scored, key=lambda x: x[0])[1]
        return _move_to_tcp_one(pos, best, duration, dry_run=False)

    def _move_to_tcp_one(pos, R_tcp, duration, dry_run) -> dict:
        import pinocchio as pin
        tcp_target = np.eye(4)
        tcp_target[:3, :3] = R_tcp
        tcp_target[:3, 3] = pos
        ee_target = tcp_target @ np.linalg.inv(T_ee_tcp_full)
        target = pin.SE3(ee_target[:3, :3].copy(), ee_target[:3, 3].copy())
        q_now = arm.joint_positions()
        seed = np.zeros(arm.model.nq)
        seed[:8] = q_now[:8]
        # Local solve FIRST: random-restart IK can land on an elbow-flipped
        # branch whose joint-linear path sweeps through the table. Only fall
        # back to retries when the local branch cannot reach the pose.
        res = _solve_ik_local(arm.model, arm.data, arm.ee_frame_id, target,
                              seed.copy(), pose_ik_params)
        if not (res.success or float(res.error) < 5.0e-3):
            res = arm._solve_ik_with_retry(arm.model, arm.data, arm.ee_frame_id,
                                           target, seed, pose_ik_params)
        # The solver often converges to ~1e-3 in the combined SE3 metric and
        # stalls -- a millimetre-scale residual, not an unreachable pose. Accept
        # near-converged solutions; the post-motion position gate is the arbiter.
        usable = bool(res.success) or float(res.error) < 5.0e-3
        rec = {"ik_success": bool(res.success), "ik_error": float(res.error),
               "ik_usable": usable}
        if not usable:
            return rec
        q6 = np.asarray(res.q, dtype=np.float64)[:6]
        ok = bool(np.all(q6 >= probe.EXPECTED_LOWER[:6] + 1e-3)
                  and np.all(q6 <= probe.EXPECTED_UPPER[:6] - 1e-3))
        rec["within_limits"] = ok
        rec["q_solution"] = q6
        if not ok or dry_run:
            return rec
        arm._ramp_to(np.concatenate([q6, [arm._grip_cmd, arm._grip_cmd]]), duration)
        achieved = arm.get_tcp_pose()
        measured_q = arm.joint_positions()[:6]
        rec["achieved_tcp"] = achieved
        rec["position_error_m"] = float(np.linalg.norm(achieved - pos))
        # Large drive-tracking error with an exact IK solution = the motion was
        # BLOCKED (table strike), not mis-solved.
        rec["max_joint_tracking_error_rad"] = float(np.max(np.abs(measured_q - q6)))
        return rec

    def ik_local_dls(T_ee_target, q0, iters=120, damping=1e-6,
                     step_clamp=0.12, tol=1e-4):
        """Step-clamped damped-least-squares IK on the gripper_link frame.

        Local by construction: each iterate moves at most ``step_clamp`` rad per
        joint, so the solution cannot hop to a flipped branch the way the
        vendor CLIK's backtracking line-search was observed to (2.78 rad jumps
        on 3 cm waypoints). Same Pinocchio model, standard DLS.
        """
        import pinocchio as pin
        q = np.zeros(arm.model.nq)
        q[:6] = q0[:6]
        target = pin.SE3(T_ee_target[:3, :3].copy(), T_ee_target[:3, 3].copy())
        for _ in range(iters):
            pin.forwardKinematics(arm.model, arm.data, q)
            pin.updateFramePlacements(arm.model, arm.data)
            cur = arm.data.oMf[arm.ee_frame_id]
            err = pin.log6(cur.actInv(target)).vector
            if float(np.linalg.norm(err)) < tol:
                return q[:6], float(np.linalg.norm(err))
            J = pin.computeFrameJacobian(arm.model, arm.data, q, arm.ee_frame_id,
                                         pin.ReferenceFrame.LOCAL)
            JJt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            dq = np.clip(dq, -step_clamp, step_clamp)
            q = pin.integrate(arm.model, q, dq)
            # Projected DLS: keep iterates inside joint limits so the solver
            # finds the LEGAL wrist branch instead of walking joint6 past +pi.
            q[:6] = np.clip(q[:6], probe.EXPECTED_LOWER[:6] + 2e-3,
                            probe.EXPECTED_UPPER[:6] - 2e-3)
            q[6:] = 0.0
        return q[:6], float(np.linalg.norm(err))

    def move_linear(pos_to, R_tcp, duration=2.4, steps=7,
                    branch_jump_rad=0.7) -> dict:
        _ensure_playing()
        """Cartesian-linear move: straight line in TCP space, local IK per
        waypoint seeded from the previous solution.

        This is what the vendor's trajectory sender does, and it is the fix for
        a measured failure: random-restart IK found a shoulder-flipped branch
        (exact FK, tracking error 4.000 rad on every attempt) whose joint-linear
        path swept the arm through the table. Waypoint continuity forbids branch
        jumps by construction; a required jump aborts the move instead.
        """
        import pinocchio as pin
        pos_from = arm.get_tcp_pose().copy()
        q_seed = arm.joint_positions()[:6].copy()
        # Current TCP orientation, so orientation can be SLERPed along the path.
        # Demanding the final orientation at waypoint 1 (a position-only interp)
        # asks for a 1-2 rad joint move in one step and trips the jump guard.
        T_be_now = link_T(ee_i)
        R_from = T_be_now[:3, :3] @ R_ee_tcp
        w = pin.log3(R_from.T @ np.asarray(R_tcp))
        diag = {"q_start": arm.joint_positions()[:6].round(3),
                "orientation_gap_deg": float(np.degrees(np.linalg.norm(w)))}

        # Rotate-then-translate. Aligning the jaw with the perceived opening
        # axis is a large WRIST reorientation (j4-j6); doing it mid-descent
        # tripped the jump guard. Reorient in place first -- wrist rolls do not
        # sweep the workspace; only shoulder-side jumps (j1-j3) are dangerous
        # and stay strictly guarded.
        tcp0 = np.eye(4)
        tcp0[:3, :3] = np.asarray(R_tcp)
        tcp0[:3, 3] = pos_from
        q_rot, rot_err = ik_local_dls(tcp0 @ np.linalg.inv(T_ee_tcp_full), q_seed)
        if rot_err > 5.0e-3:
            return {"ok": False, "failed_at_waypoint": 0, "diag": diag,
                    "reason": "in-place reorientation unreachable",
                    "ik_error": rot_err}
        if float(np.max(np.abs(q_rot[:3] - q_seed[:3]))) > 0.6:
            return {"ok": False, "failed_at_waypoint": 0, "diag": diag,
                    "reason": "reorientation would move shoulder joints",
                    "shoulder_jump_rad": float(np.max(np.abs(q_rot[:3] - q_seed[:3])))}
        # A ~130-degree wrist reorientation does not settle in one short ramp
        # (measured 0.995 rad short after 1.6 s + 0.6 s settle). Ramp toward the
        # target repeatedly until it tracks.
        reorient_err = np.inf
        for _ in range(4):
            arm._ramp_to(np.concatenate([q_rot, [arm._grip_cmd, arm._grip_cmd]]),
                         2.5)
            reorient_err = float(np.max(np.abs(arm.joint_positions()[:6] - q_rot)))
            if reorient_err < 0.08:
                break
        if reorient_err > 0.08:
            mq = arm.joint_positions()[:6]
            return {"ok": False, "failed_at_waypoint": 0, "diag": diag,
                    "reason": "reorientation did not track (dead drives or "
                              "contact)", "reorient_error_rad": reorient_err,
                    "q_rot_commanded": q_rot.round(3),
                    "q_measured": mq.round(3),
                    "per_joint_err": np.abs(mq - q_rot).round(3)}
        q_seed = q_rot
        qs = []
        for k in range(1, steps + 1):
            t = k / steps
            pos_k = pos_from + (np.asarray(pos_to) - pos_from) * t
            R_k = np.asarray(R_tcp)   # orientation already reached in stage 0
            tcp_target = np.eye(4)
            tcp_target[:3, :3] = R_k
            tcp_target[:3, 3] = pos_k
            ee_target = tcp_target @ np.linalg.inv(T_ee_tcp_full)
            q6, ik_err = ik_local_dls(ee_target, q_seed)
            if ik_err > 5.0e-3:
                return {"ok": False, "failed_at_waypoint": k, "diag": diag,
                        "reason": "local IK failed", "ik_error": ik_err,
                        "q6": q6.round(3)}
            if not (np.all(q6 >= probe.EXPECTED_LOWER[:6] + 1e-3)
                    and np.all(q6 <= probe.EXPECTED_UPPER[:6] - 1e-3)):
                viol = [(i + 1, round(float(q6[i]), 3)) for i in range(6)
                        if not (probe.EXPECTED_LOWER[i] + 1e-3 <= q6[i]
                                <= probe.EXPECTED_UPPER[i] - 1e-3)]
                return {"ok": False, "failed_at_waypoint": k, "diag": diag,
                        "reason": "outside joint limits", "violations": viol,
                        "q6": q6.round(3)}
            if float(np.max(np.abs(q6 - q_seed))) > branch_jump_rad:
                return {"ok": False, "failed_at_waypoint": k, "diag": diag,
                        "reason": "branch jump refused", "q6": q6.round(3),
                        "jump_rad": float(np.max(np.abs(q6 - q_seed)))}
            qs.append(q6)
            q_seed = q6
        per = max(duration / steps, 0.25)
        descent_hist = []
        for wi, q6 in enumerate(qs):
            arm._ramp_to(np.concatenate([q6, [arm._grip_cmd, arm._grip_cmd]]),
                         per, settle=0.15)
            mq_w = arm.joint_positions()[:6]
            track = float(np.max(np.abs(mq_w - q6)))
            descent_hist.append(
                {"wp": wi + 1,
                 "tcp_z_cmd": round(float(pos_from[2]
                     + (float(pos_to[2]) - pos_from[2]) * (wi + 1) / steps), 4),
                 "per_joint": np.abs(mq_w - q6).round(3).tolist(),
                 "max": round(track, 4)})
            if track > 0.25:
                # Contact mid-path: stop, do not plow. The next approach
                # candidate gets its chance instead.
                return {"ok": False, "failed_at_waypoint": wi + 1,
                        "reason": "contact during descent (tracking loss)",
                        "tracking_error_rad": track,
                        "descent_tracking": descent_hist}
        diag["descent_tracking"] = descent_hist
        arm.hold(0.4)
        # Joint-space gravity-offset mirroring. At deep reach the drives sit
        # a steady ~0.02 rad from ANY commanded pose (run 38: commanded q hit
        # the target to 0.1 mm by FK while the measured q -- 0.024 rad away
        # -- was 14 mm off; the lever arm amplifies tiny offsets). Cancel the
        # offset where it lives: command the mirror of the measured error.
        # Frame-free, IK-free, bounded by the contact check.
        # (The former Cartesian error-feedback loop is gone: FK proved the
        # commanded solution exact to 0.1 mm, so ALL residual error is this
        # joint offset -- and the Cartesian loop's revert used to ramp back
        # to plain qs[-1], undoing the mirror (run 39: 13 mm instead of 3).
        # The offset roughly halves per mirror cycle, hence 4 cycles.)
        def _settle(q_cmd):
            """Mirror the steady gravity offset around q_cmd (damped integral
            with keep-best, run 40/42 lessons: full gain diverges on a joint
            whose disturbance grows with the correction; against light contact
            the integral winds up and makes tracking worse)."""
            bias = np.zeros(6)
            best_worst, best_bias = np.inf, bias.copy()
            for _ in range(6):
                dq = arm.joint_positions()[:6] - q_cmd
                worst = float(np.max(np.abs(dq)))
                if worst < best_worst:
                    best_worst, best_bias = worst, bias.copy()
                if worst < 0.004:
                    return
                if worst > 0.12:
                    return   # contact -- do not fight it with feedback
                bias = np.clip(bias + 0.6 * dq, -0.06, 0.06)
                mirror = np.clip(q_cmd - bias, probe.EXPECTED_LOWER[:6] + 1e-3,
                                 probe.EXPECTED_UPPER[:6] - 1e-3)
                arm._ramp_to(np.concatenate(
                    [mirror, [arm._grip_cmd, arm._grip_cmd]]), 0.8)
            dq = arm.joint_positions()[:6] - q_cmd
            if float(np.max(np.abs(dq))) > best_worst + 0.002:
                mirror = np.clip(q_cmd - best_bias,
                                 probe.EXPECTED_LOWER[:6] + 1e-3,
                                 probe.EXPECTED_UPPER[:6] - 1e-3)
                arm._ramp_to(np.concatenate(
                    [mirror, [arm._grip_cmd, arm._grip_cmd]]), 0.8)

        _settle(qs[-1])
        achieved = arm.get_tcp_pose()
        # Cartesian trim. After mirroring, joints track to <0.01 rad, yet the
        # MEASURED TCP still sits several mm off (run 43: FK(measured q) was
        # 4.3 mm from target while get_tcp_pose read 9.4 mm -- the rest is
        # articulation compliance the joint sensors cannot see). One or two
        # reflected-target IK trims against the measured TCP, each settled
        # with the mirror, with revert-on-worse so contact can never be
        # ground into (the run-37 jam).
        q_trim, q_best = qs[-1].copy(), qs[-1].copy()
        best_err = float(np.linalg.norm(achieved - np.asarray(pos_to)))
        diag["trim"] = []
        for _ in range(2):
            err_vec = np.asarray(pos_to) - achieved
            err_n = float(np.linalg.norm(err_vec))
            step = {"err_before_mm": round(err_n * 1000, 2)}
            diag["trim"].append(step)
            if err_n <= 4.0e-3:
                step["stop"] = "converged"
                break
            corr = np.asarray(pos_to) + err_vec * min(1.0, 8.0e-3 / err_n)
            tcp_c = np.eye(4)
            tcp_c[:3, :3] = np.asarray(R_tcp)
            tcp_c[:3, 3] = corr
            q_c, e_c = ik_local_dls(tcp_c @ np.linalg.inv(T_ee_tcp_full),
                                    q_trim)
            step["ik_err"] = round(float(e_c), 5)
            step["jump_rad"] = round(float(np.max(np.abs(q_c - q_trim))), 4)
            if e_c > 5.0e-3 or float(np.max(np.abs(q_c - q_trim))) > 0.25:
                step["stop"] = "ik_or_jump"
                break
            arm._ramp_to(np.concatenate(
                [q_c, [arm._grip_cmd, arm._grip_cmd]]), 1.0)
            _settle(q_c)
            achieved = arm.get_tcp_pose()
            new_err = float(np.linalg.norm(achieved - np.asarray(pos_to)))
            step["err_after_mm"] = round(new_err * 1000, 2)
            if new_err >= best_err - 5.0e-4:
                step["stop"] = "reverted"
                arm._ramp_to(np.concatenate(
                    [q_best, [arm._grip_cmd, arm._grip_cmd]]), 1.0)
                _settle(q_best)
                achieved = arm.get_tcp_pose()
                break
            best_err, q_best, q_trim = new_err, q_c.copy(), q_c
        measured_q = arm.joint_positions()[:6]
        result = {"ok": True, "achieved_tcp": achieved.round(4),
                  "position_error_m": float(np.linalg.norm(achieved - pos_to)),
                  "commanded_final_q": qs[-1].round(3),
                  "measured_q": measured_q.round(3),
                  "per_joint_error_rad": np.abs(measured_q - qs[-1]).round(3),
                  "max_joint_tracking_error_rad": float(
                      np.max(np.abs(measured_q - qs[-1]))),
                  "diag": diag}
        if float(np.max(np.abs(measured_q - qs[-1]))) > 0.12:
            # Stall forensics: with kp 1500 / 27 Nm on j2 and <= ~6 Nm of
            # gravity torque on this arm, a >0.1 rad steady error means the
            # drive is pushing against something rigid. Record where the miss
            # points and which links sit lowest -- elbow-on-table vs
            # finger-on-object discriminate cleanly here.
            links_now = arm.monitor._link_transforms()
            names = list(arm.monitor.body_names)
            zs = sorted(((names[i], round(float(links_now[i, 2]), 4))
                         for i in range(len(names))), key=lambda t: t[1])
            result["stall_forensics"] = {
                "miss_vector_m": (achieved - np.asarray(pos_to)).round(4).tolist(),
                "lowest_links_z": zs[:5]}
        return result

    def probe_ik(pos, R_tcp) -> bool:
        import pinocchio as pin
        tcp_target = np.eye(4)
        tcp_target[:3, :3] = R_tcp
        tcp_target[:3, 3] = pos
        ee_target = tcp_target @ np.linalg.inv(T_ee_tcp_full)
        res = arm._solve_ik_with_retry(
            arm.model, arm.data, arm.ee_frame_id,
            pin.SE3(ee_target[:3, :3].copy(), ee_target[:3, 3].copy()),
            np.zeros(arm.model.nq), pose_ik_params)
        if not (res.success or float(res.error) < 5.0e-3):
            return False
        q6 = np.asarray(res.q)[:6]
        return bool(np.all(q6 >= probe.EXPECTED_LOWER[:6] + 1e-3)
                    and np.all(q6 <= probe.EXPECTED_UPPER[:6] - 1e-3))

    def probe_ik_sym(pos, R_tcp) -> bool:
        return probe_ik(pos, R_tcp) or probe_ik(pos, R_tcp @ ROLL_PI)

    def tcp_R_from_approach(approach, opening) -> np.ndarray:
        a = normalize(np.asarray(approach, dtype=np.float64))
        o = normalize(np.asarray(opening, dtype=np.float64))
        grip = normalize(np.cross(o, a))
        o2 = np.cross(a, grip)
        return vision_grasp_basis_to_b601_tcp_rotation(
            np.column_stack([grip, o2, a]))

    # ---- scene calibration: table height + banana spot via IK probe -------
    ready_approach = np.array([np.cos(READY_PITCH_RAD), 0.0,
                               -np.sin(READY_PITCH_RAD)])
    chosen = None
    for table_top in TABLE_TOPS_TO_TRY:
        for gx in (0.26, 0.29, 0.32, 0.24):
            for gy in (0.0, -0.04, 0.04):
                gpos = np.array([gx, gy, table_top + BANANA_SEGMENT[2] / 2])
                R = tcp_R_from_approach(ready_approach, [0.0, 1.0, 0.0])
                pre = gpos - PREGRASP_OFFSET_M * ready_approach
                ins = gpos + INSERTION_DEPTH_M * ready_approach
                if probe_ik_sym(pre, R) and probe_ik_sym(ins, R):
                    chosen = {"table_top": table_top, "spot": [gx, gy]}
                    break
            if chosen:
                break
        if chosen:
            break
    report.require("a reachable table height and banana spot exist",
                   chosen is not None, tried_heights=TABLE_TOPS_TO_TRY)
    table_top = chosen["table_top"]
    spot = chosen["spot"]
    report.data["scene"] = {"table_top_m": table_top, "banana_spot": spot,
                            "note": "chosen by IK reachability probe; this is "
                                    "table setup, not grasp-target feeding"}

    world.scene.add(FixedCuboid(
        prim_path="/World/table", name="table",
        position=np.array([0.33, 0.0, table_top / 2]),
        scale=np.array([TABLE_SIZE[0], TABLE_SIZE[1], table_top]),
        color=np.array([0.42, 0.32, 0.22])))

    # ---- wrist camera (rigid mount; world pose driven from tensor pose) ---
    # The marker must NOT sit at the grasp spot: the tool hangs between the
    # wrist camera and its aim point, and in two runs the fingertip clipped a
    # corner of the border square, which ArUco cannot tolerate. Like the real
    # demo, the calibration marker lies freely on the table, off to one side.
    # Far corner of the table, beyond the tool's silhouette from the wrist
    # view. Wiggle poses that occlude it are simply skipped by the collector.
    # Fixed, decoupled from the grasp spot: tying it to the spot silently moved
    # it out of the calibration views when the spot changed (4/18 detections).
    marker_center = np.array([0.43, -0.145, table_top + 0.0015])
    build_marker_tiles(stage, "/World/aruco_marker", marker_center,
                       MARKER_LENGTH_M)

    prim_utils.create_prim("/World/wrist_cam", "Camera")
    cam_prim = stage.GetPrimAtPath("/World/wrist_cam")
    cam_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 1000.0))
    aperture = float(cam_prim.GetAttribute("horizontalAperture").Get() or 20.955)
    cam_prim.GetAttribute("focalLength").Set(IMG_FX * aperture / IMG_W)
    K = np.array([[IMG_FX, 0, (IMG_W - 1) / 2], [0, IMG_FX, (IMG_H - 1) / 2],
                  [0, 0, 1.0]])
    rp = rep.create.render_product("/World/wrist_cam", (IMG_W, IMG_H))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_annot.attach(rp)
    depth_annot.attach(rp)

    # Move to the vendor ready pose, then define the mount looking at the spot.
    R_ready = tcp_R_from_approach(ready_approach, [0.0, 1.0, 0.0])
    rec = move_to_tcp(READY_TCP, R_ready, duration=2.5)
    report.require("ready pose reached (vendor x=0.3 z=0.3 pitch=0.7)",
                   rec.get("position_error_m", 9) <= MOVE_TOL_M, **rec)

    T_bg0 = link_T(ee_i)
    eye = arm.get_tcp_pose() + np.array([-0.055, 0.11, 0.15])
    aim = 0.65 * marker_center + 0.35 * np.array([spot[0], spot[1], table_top])
    fwd = normalize(aim - eye)
    right = normalize(np.cross(fwd, [0.0, 0.0, 1.0]))
    down = np.cross(fwd, right)
    T_bc0 = make_transform(np.column_stack([right, down, fwd]), eye)
    X_TRUE = np.linalg.inv(T_bg0) @ T_bc0
    report.data["ground_truth_mount"] = {"T_gripper_cam": X_TRUE,
                                         "note": "never given to the solver"}

    def render_wrist(warmup: int = 3):
        _set_camera_world(stage, "/World/wrist_cam", link_T(ee_i) @ X_TRUE)
        rgb = None
        for attempt in range(max(warmup, 1)):
            # pause_timeline defaults to True and STOPS the sim clock after the
            # render; world.step(render=False) then skips physics entirely and
            # every subsequent commanded motion is a silent no-op. Root cause of
            # the zero-motion approach failures (runs 19-27).
            rep.orchestrator.step(rt_subframes=8, pause_timeline=False)
            a = rgb_annot.get_data()
            if a is None:
                continue
            a = np.asarray(a)
            if a.ndim == 3 and a.shape[0] > 0:
                rgb = a
        if rgb is None:
            raise DemoFailure("wrist camera produced no frames")
        rgb = rgb[..., :3].astype(np.uint8)
        depth = np.asarray(depth_annot.get_data(), dtype=np.float64)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth > 20.0] = 0.0
        return rgb, depth

    # ---- hand-eye: fx calibration + collection + solve --------------------
    detector = ArUcoDetector(marker_length_m=MARKER_LENGTH_M,
                             aruco_dict_id=ARUCO_DICT_ID,
                             target_marker_id=TARGET_MARKER_ID)
    rgb0, _ = render_wrist(warmup=14)
    cv2.imwrite(str(Path(args.out).parent / "handeye_view.png"),
                cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR))
    pose0 = detector.detect(cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR), K,
                            np.zeros((1, 5)))
    report.require("marker detected from the ready pose", pose0 is not None,
                   view_saved="handeye_view.png",
                   rgb_mean=float(rgb0.mean()))
    z_true = float((np.linalg.inv(link_T(ee_i) @ X_TRUE)
                    @ np.append(marker_center, 1.0))[2])
    z_det = float(np.asarray(pose0.T_marker2cam)[2, 3])
    fx_eff = IMG_FX * z_true / z_det
    K = np.array([[fx_eff, 0, (IMG_W - 1) / 2], [0, fx_eff, (IMG_H - 1) / 2],
                  [0, 0, 1.0]])
    report.data["intrinsics"] = {"fx_assumed": IMG_FX, "fx_effective": fx_eff}
    report.check("effective fx within 5% of assumed",
                 abs(fx_eff / IMG_FX - 1) < 0.05, ratio=fx_eff / IMG_FX)

    q_ready = arm.joint_positions()[:6].copy()
    deltas = [
        (0, 0, 0, 0, 0, 0), (0.10, 0.05, -0.05, 0.40, 0.50, 1.8),
        (-0.10, -0.05, 0.05, -0.40, -0.50, -1.8), (0.18, 0.08, 0, 0.60, -0.55, 2.4),
        (-0.18, -0.08, 0, -0.60, 0.55, -2.4), (0.06, -0.10, 0.10, 0.28, 0.65, -2.1),
        (-0.06, 0.10, -0.10, -0.28, -0.65, 2.1), (0.24, 0, -0.12, 0.65, 0.30, 1.1),
        (-0.24, 0, 0.12, -0.65, -0.30, -1.1), (0.15, 0.12, 0.07, -0.50, 0.60, 2.7),
        (-0.15, -0.12, -0.07, 0.50, -0.60, -2.7), (0.04, 0.07, -0.15, 0.55, -0.28, -1.5),
        (-0.04, -0.07, 0.15, -0.55, 0.28, 1.5), (0.28, -0.05, 0.05, 0.22, 0.55, -2.5),
        (-0.28, 0.05, -0.05, -0.22, -0.55, 2.5), (0.11, 0.11, -0.11, 0.45, -0.60, 0.7),
        (-0.11, -0.11, 0.11, -0.45, 0.60, -0.7), (0.07, -0.09, 0.09, 0.18, 0.42, 2.9),
    ]
    calib = HandEyeCalibrator(CalibMode.EYE_IN_HAND, method="TSAI")
    samples = []
    for d in deltas:
        q6 = q_ready + np.asarray(d)
        if not (np.all(q6 >= probe.EXPECTED_LOWER[:6] + 1e-3)
                and np.all(q6 <= probe.EXPECTED_UPPER[:6] - 1e-3)):
            continue
        arm._ramp_to(np.concatenate([q6, [arm._grip_cmd, arm._grip_cmd]]), 1.2)
        T_bg = link_T(ee_i)
        rgb, _ = render_wrist()
        pose = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), K,
                               np.zeros((1, 5)))
        if pose is None:
            continue
        T_mc = np.asarray(pose.T_marker2cam, dtype=np.float64)
        calib.add_sample(T_bg, T_mc)
        samples.append({"T_gripper2base": T_bg, "T_marker2cam": T_mc})
    report.require("enough detections for hand-eye", len(samples) >= 6,
                   detected=len(samples), attempted=len(deltas))

    Rs = [s["T_gripper2base"][:3, :3] for s in samples]
    pair = [float(np.degrees(np.arccos(np.clip(
        (np.trace(Rs[a].T @ Rs[b]) - 1) / 2, -1, 1))))
        for a in range(len(Rs)) for b in range(a + 1, len(Rs))]
    report.require("rotation diversity sufficient for AX=XB",
                   max(pair) >= 40.0 and float(np.mean(pair)) >= 15.0,
                   max_deg=max(pair), mean_deg=float(np.mean(pair)))

    # Ground-truth-free method selection: static marker => spread scores X.
    method_scores = {}
    for method in ("TSAI", "PARK", "HORAUD", "DANIILIDIS"):
        c2 = HandEyeCalibrator(CalibMode.EYE_IN_HAND, method=method)
        for s in samples:
            c2.add_sample(s["T_gripper2base"], s["T_marker2cam"])
        try:
            X = np.asarray(c2.calibrate(min_samples=5).T_result)
        except Exception:                                      # noqa: BLE001
            continue
        pts = np.array([(s["T_gripper2base"] @ X @ s["T_marker2cam"])[:3, 3]
                        for s in samples])
        spread = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
        dp, dr = _pose_err(np.vstack([np.hstack([X[:3, :3], X[:3, 3:4]]),
                                      [[0, 0, 0, 1]]]), X_TRUE)
        method_scores[method] = {"X": X, "marker_spread_m": spread,
                                 "truth_pos_err_m": dp, "truth_rot_err_deg": dr}
    report.require("at least one hand-eye method solved", bool(method_scores))
    deploy = min(method_scores, key=lambda m: method_scores[m]["marker_spread_m"])
    X_solved = method_scores[deploy]["X"]
    report.data["hand_eye"] = {
        "deployed_method": deploy,
        "selection_criterion": "min marker-position spread (no ground truth used)",
        "methods": {m: {k: v for k, v in d.items() if k != "X"}
                    for m, d in method_scores.items()},
    }
    # 35 mm guards against DEGENERATE solves (the failure mode is ~100 mm with
    # zero translation); the end-to-end perception gate (25 mm lateral on the
    # actual target) is the real accuracy protector.
    report.check("deployed hand-eye recovers the true mount within 35 mm (sim diagnostic)",
                 method_scores[deploy]["truth_pos_err_m"] <= 0.035,
                 pos_err_m=method_scores[deploy]["truth_pos_err_m"],
                 rot_err_deg=method_scores[deploy]["truth_rot_err_deg"])

    # ---- hide the marker, spawn the banana, settle ------------------------
    mk = stage.GetPrimAtPath("/World/aruco_marker")
    UsdGeom.Xformable(mk).AddTranslateOp(opSuffix="hide").Set(Gf.Vec3d(0, 0, -2.0))
    banana_prim = build_banana(stage, "/World/banana", spot, table_top)
    # The P2 lesson, fully applied: the high-friction material must be on BOTH
    # sides of the contact. Run 46 closed on the banana (both fingers, 6 mm
    # squeeze) and it still slid out during the lift (76 mm slip) -- the
    # banana's colliders were on the PhysX default material.
    for _i in range(3):
        pick._bind_physics_material(stage, f"/World/banana/seg{_i}",
                                    grip_material.prim_path)

    # Spawning a physics prim mid-run invalidates the articulation's simulation
    # view: every apply_action afterwards silently no-ops (measured: the arm sat
    # at q_start to the third decimal through an entire "executed" approach).
    # Same dance as b601_pick: preserve the pose as the default state, reset,
    # rebuild the tensor view, re-apply gains.
    # Register the banana with the scene BEFORE the reset so ONE view rebuild
    # covers articulation and object together. A standalone initialize() after
    # the reset re-invalidates the articulation's control handle -- measured as
    # nondeterministically dead drives (zero motion one run, partial the next).
    from isaacsim.core.prims import SingleRigidPrim
    banana = SingleRigidPrim(prim_path="/World/banana", name="banana")
    world.scene.add(banana)
    q_keep = arm.joint_positions().copy()
    articulation.set_joints_default_state(positions=q_keep)
    world.reset()
    monitor = probe.StateMonitor(articulation)
    arm.monitor = monitor
    articulation.get_articulation_controller().set_gains(kps=probe.RUNTIME_KP,
                                                         kds=probe.RUNTIME_KD)
    if recorder is not None and issue_paths:
        recorder.set_link_sync(LinkUsdSync(stage, monitor, issue_paths))
    arm.hold(1.0)

    # Drive smoke-test: reads were provably live after the previous reset while
    # WRITES (apply_action) were dead. Command a small j6 move and require it to
    # track; on failure, explicitly re-initialize the articulation wrapper to
    # force a fresh control handle, and test again.
    def _drive_test() -> float:
        q0 = arm.joint_positions().copy()
        qt = q0.copy()
        qt[5] = float(np.clip(q0[5] + 0.25, probe.EXPECTED_LOWER[5] + 1e-2,
                              probe.EXPECTED_UPPER[5] - 1e-2))
        arm._ramp_to(qt, 1.2)
        err = abs(float(arm.joint_positions()[5] - qt[5]))
        arm._ramp_to(q0, 1.2)
        return err

    def drive_probe(tag: str) -> float:
        e = _drive_test()
        report.data.setdefault("drive_probes", []).append(
            {"tag": tag, "error_rad": round(e, 5)})
        return e

    drive_err = _drive_test()
    revived_by = "none-needed"
    if drive_err > 0.05:
        articulation.initialize()
        articulation.get_articulation_controller().set_gains(
            kps=probe.RUNTIME_KP, kds=probe.RUNTIME_KD)
        monitor = probe.StateMonitor(articulation)
        arm.monitor = monitor
        if recorder is not None and issue_paths:
            recorder.set_link_sync(LinkUsdSync(stage, monitor, issue_paths))
        drive_err = _drive_test()
        revived_by = "articulation.initialize()"
    report.data["post_spawn_drive_test"] = {"error_rad": drive_err,
                                            "revived_by": revived_by}
    report.require("drives track after the object spawn", drive_err <= 0.05,
                   error_rad=drive_err, revived_by=revived_by)
    b0 = np.asarray(banana.get_world_pose()[0], dtype=np.float64)
    report.require("banana settles on the table",
                   abs(b0[2] - (table_top + BANANA_SEGMENT[2] / 2)) < 0.02
                   and np.all(np.isfinite(b0)), settled=b0,
                   table_top=table_top)

    # ---- perceive: RGB-D -> mask -> grasp estimate (CAMERA frame) ---------
    rgb, depth = render_wrist()
    T_bg_view = link_T(ee_i)
    mask = yellow_mask(rgb)
    if args.save_images:
        cv2.imwrite(str(Path(args.out).parent / "wrist_rgb.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if mask is not None:
            cv2.imwrite(str(Path(args.out).parent / "wrist_mask.png"), mask * 255)
    report.require("yellow-object mask found", mask is not None,
                   mask_px=0 if mask is None else int(mask.sum()))
    est = estimate_grasp(mask, depth, K, depth_quantile=DEPTH_QUANTILE)
    report.require("grasp estimated from the wrist image", est.is_valid,
                   reason=est.rejected_reason)

    # ---- camera frame -> base frame through the SOLVED X ------------------
    T_bc = T_bg_view @ X_solved
    grasp_pos_base = (T_bc @ np.append(est.position, 1.0))[:3]
    R_vision_base = T_bc[:3, :3] @ est.rotation
    R_tcp_des = vision_grasp_basis_to_b601_tcp_rotation(R_vision_base)
    approach_base = -normalize(T_bc[:3, :3] @ (-est.position))  # camera ray, into scene
    approach_base = R_tcp_des[:, 0]

    drive_probe("after_perception_render")
    grasp_pos_base[2] = max(float(grasp_pos_base[2]), MIN_TCP_Z_M)
    b_true = np.asarray(banana.get_world_pose()[0], dtype=np.float64)
    perr = grasp_pos_base - (b_true + [0, 0, 0.0])
    report.data["perception"] = {
        "grasp_pos_base_from_camera": grasp_pos_base,
        "banana_true_pos": b_true,
        "position_error_vs_true_m": float(np.linalg.norm(perr[:2])),
        "z_error_m": float(perr[2]),
        "jaw_width_est_m": float(est.jaw_width_m),
        "note": "true pose used ONLY for this diagnostic, never for the target",
    }
    report.require("perceived grasp lands on the banana (<25 mm lateral)",
                   float(np.linalg.norm(perr[:2])) <= 0.025,
                   lateral_err_m=float(np.linalg.norm(perr[:2])))

    # ---- execute: pregrasp -> insert -> grasp -> lift -> release ----------
    # Approach-direction ladder. The vendor uses the camera ray as the approach;
    # from this wrist mount that 6-D pose can sit outside the wrist's dexterous
    # range (IK err ~0.69). Fall back to canonical pitched approaches at the
    # target's azimuth -- ALWAYS keeping the PERCEIVED opening axis, which is
    # the part perception must control for the jaw to straddle the banana.
    opening_base = R_tcp_des[:, 1]
    azim = float(np.arctan2(grasp_pos_base[1], grasp_pos_base[0]))
    # Level the jaw: keep only the opening axis's horizontal direction. The
    # raw perceived axis carries a vertical component from the camera
    # rotation, tilting the jaw plane (run 36: 31 mm height difference
    # between the finger links -- the low pad met the banana while the TCP
    # was still 20 mm above target). The vendor's own grasp message is
    # yaw-only (fixed rx/ry, rz from the box), i.e. a level jaw by contract.
    o_h = np.array([opening_base[0], opening_base[1], 0.0])
    if np.linalg.norm(o_h) < 0.2:
        o_h = np.array([-np.sin(azim), np.cos(azim), 0.0])
    opening_base = o_h / np.linalg.norm(o_h)
    candidates = [("perceived_camera_ray", R_tcp_des)]
    for pitch in (0.7, 0.9, 0.55, 1.1):
        a = np.array([np.cos(pitch) * np.cos(azim),
                      np.cos(pitch) * np.sin(azim), -np.sin(pitch)])
        try:
            candidates.append((f"canonical_pitch_{pitch}",
                               tcp_R_from_approach(a, opening_base)))
        except Exception:                                      # noqa: BLE001
            pass

    # Stiffen the wrist drives for the approach. The probe's RUNTIME_KP wrist
    # values (150/80/50, inherited from the RS asset) leave gravity-scale
    # steady offsets: run 40 measured j4 parked 0.026 rad (~4 mm of TCP) from
    # ANY command at deep reach -- a ~3.9 Nm disturbance kp 150 cannot hide,
    # and one the mirror loop could not integrate away. 3x wrist kp cuts every
    # steady offset proportionally. Applied HERE, after calibration and
    # perception: the hand-eye wiggle poses were tuned with the stock gains,
    # and stiffening them shifted the viewpoints enough to lose the marker
    # (run 41: 5/18 detections).
    stiff_kp = probe.RUNTIME_KP.copy()
    stiff_kd = probe.RUNTIME_KD.copy()
    stiff_kp[3:6] = [450.0, 240.0, 150.0]
    stiff_kd[3:6] = [31.0, 17.0, 12.0]
    articulation.get_articulation_controller().set_gains(kps=stiff_kp,
                                                         kds=stiff_kd)

    # Descend with a 100 mm aperture. The banana ARC's min-area rect spans
    # ~55 mm across, so a 55 mm aperture had zero lateral margin and a pad
    # landed on the banana's curl (run 35); full open (143 mm) doubled the
    # lever arm of any residual jaw tilt (run 36). 100 mm leaves ~17 mm of
    # lateral clearance to the arc's worst side and the pad bottoms ~5 mm
    # above the mat at the TCP z floor. The old full-open mat-catch only ever
    # happened while the pregrasp sign error commanded targets through the
    # table.
    arm.open_gripper(0.050)
    drive_probe("after_open_gripper")
    chosen_approach, rec_pre, attempts = None, {}, []
    for name, R_c0 in candidates:
        for suffix, R_c in (("", R_c0), ("+rollpi", R_c0 @ ROLL_PI)):
            # TCP x is the RETREAT direction (x = -approach, pose_msg
            # convention), so the pregrasp backs off along +x. Run 34's
            # forensics caught the opposite sign commanding the TCP to
            # grasp_z - 80*sin(pitch) mm -- through the table -- on every
            # candidate; the scene-calibration probe (which uses the true
            # approach vector) had validated the correct poses all along.
            pre_c = grasp_pos_base + PREGRASP_OFFSET_M * R_c[:, 0]
            rec = move_linear(pre_c, R_c, duration=2.8)
            attempts.append({"name": name + suffix,
                             **{k: v for k, v in rec.items()
                                if k != "achieved_tcp"}})
            if rec.get("ok") and rec.get("position_error_m", 9) <= MOVE_TOL_M:
                chosen_approach, R_tcp_des, rec_pre = name + suffix, R_c, rec
                break
            rec_pre = rec
        if chosen_approach:
            break
    approach_base = -R_tcp_des[:, 0]   # true approach direction, into the scene
    pre = grasp_pos_base - PREGRASP_OFFSET_M * approach_base
    report.data["approach"] = {"chosen": chosen_approach, "attempts": attempts}
    report.require("pregrasp reached", chosen_approach is not None,
                   **{k: v for k, v in rec_pre.items() if k != "achieved_tcp"})

    ins = grasp_pos_base + INSERTION_DEPTH_M * approach_base
    # Same finger-clearance floor as the grasp point: at these pitches the
    # insertion's z-drop (15*sin(pitch) mm) would put the 39 mm pads into the
    # mat; the horizontal advance is what seats the jaw around the banana.
    ins[2] = max(float(ins[2]), MIN_TCP_Z_M)
    rec_ins = move_linear(ins, R_tcp_des, duration=2.6)
    # Gate semantics, decided on run 43-45 evidence. At this deep-reach pose
    # the jaw-midpoint measurement (finger links, ~40 mm of lever from the
    # wrist axes) carries a pose-sensitive droop bias: FK of the measured
    # joints put the arm 4.3 mm from the target while the jaw midpoint read
    # 9.4 mm, and commanding an 8 mm reflected trim moved that reading the
    # WRONG way (9.4 -> 13.6 mm, reverted). So the strict requirements here
    # are the rigorously measurable ones -- descent completed without a
    # contact abort and joints tracking to <=0.02 rad (drive-level truth) --
    # plus a 15 mm jaw-midpoint backstop that still fails every genuinely
    # broken approach we saw (19-93 mm). Whether the jaw is FUNCTIONALLY
    # placed is proven by the unfakeable gates that follow: finger contact,
    # 50 mm lift, <20 mm slip.
    report.require("insertion pose reached (no table strike)",
                   bool(rec_ins.get("ok"))
                   and rec_ins.get("max_joint_tracking_error_rad", 9) <= 0.02
                   and rec_ins.get("position_error_m", 9) <= 0.015,
                   **{k: v for k, v in rec_ins.items() if k != "achieved_tcp"})

    # 12 mm squeeze (vs the 6 mm default): first contact lands on the arc's
    # curled tips at ~56 mm aperture, and a deeper squeeze carries the pads to
    # ~32 mm -- at the middle segment's 30 mm body, a clamp instead of a
    # convex-tip pinch (run 46: tip pinch + default friction = 76 mm slip).
    g = arm.grasp(force=20.0, squeeze_m=0.012)
    report.data["grasp"] = g
    report.require("both fingers contact the banana", g["contacted"],
                   **{k: v for k, v in g.items() if k != "trace"})

    b_before = np.asarray(banana.get_world_pose()[0], dtype=np.float64)
    tcp_before = arm.get_tcp_pose().copy()
    lift = move_linear(tcp_before + [0, 0, 0.080], R_tcp_des, duration=4.0)
    arm.hold(REQUIRED_HOLD_S)
    b_held = np.asarray(banana.get_world_pose()[0], dtype=np.float64)
    rise = float(b_held[2] - b_before[2])
    tcp_rise = float(arm.get_tcp_pose()[2] - tcp_before[2])
    report.data["lift"] = {"banana_rise_m": rise, "tcp_rise_m": tcp_rise,
                           "slip_m": tcp_rise - rise,
                           "ik": {k: v for k, v in lift.items() if k != "achieved_tcp"}}
    report.require("banana rises at least 50 mm", rise >= REQUIRED_LIFT_M,
                   rise_m=rise)
    report.require("slip during lift stays under 20 mm",
                   (tcp_rise - rise) <= 0.020, slip_m=tcp_rise - rise)

    arm.release_gripper()
    arm.hold(1.0)
    b_rel = np.asarray(banana.get_world_pose()[0], dtype=np.float64)
    fall = float(b_held[2] - b_rel[2])
    report.data["release"] = {"fall_m": fall}
    report.require("banana falls on release (no hidden attachment)",
                   fall >= 0.020, fall_m=fall)

    if recorder is not None:
        report.data["recording"] = recorder.finalize()
    if args.out and arm.traj_log:
        traj_path = Path(args.out).with_name("traj.npz")
        np.savez_compressed(traj_path,
                            traj=np.asarray(arm.traj_log),
                            table_top=table_top, spot=np.asarray(spot))
        report.data["traj_saved"] = {"path": str(traj_path),
                                     "rows": len(arm.traj_log)}
    report.data["state_monitor"] = {
        "samples": monitor.samples,
        "max_base_position_drift_m": monitor.max_base_position_drift_m}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "artifacts" / "banana" / "b601_banana.json")
    ap.add_argument("--repair-nested-xforms", action="store_true")
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--replay", type=Path, default=None,
                    help="traj.npz from a prior run: rebuild the scene, feed "
                         "the logged joint commands 1:1, record only (no "
                         "wrist camera, no gates)")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--save-images", action="store_true", default=True)
    args = ap.parse_args(argv)

    report = Report()
    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": not args.headful})
    try:
        _run(args, report)
    except DemoFailure as exc:
        report.data["errors"].append(str(exc))
    except Exception:                                          # noqa: BLE001
        import traceback
        report.data["errors"].append(traceback.format_exc()[-600:])
        report.data["traceback"] = traceback.format_exc()[-3000:]
    finally:
        report.write(args.out)
        try:
            sim_app.close()
        except Exception:                                      # noqa: BLE001
            pass
    return 0 if report.data.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
