#!/usr/bin/env python3
"""P3: the demo's own robot interface, implemented against the Isaac articulation.

The Seeed demo drives the arm through exactly five calls
(``drivers/robot/grasp_driver.py``, ``scripts/main.py:54,92,98``)::

    move_to_traj(x, y, z, rx, ry, rz, duration)
    open_gripper(distance_m)
    grasp(force)
    release_gripper()
    get_tcp_pose()

:class:`RebotArmSim` implements those five against the simulated B601-DM, with IK
from the **pinned** ``reBotArm_control_py`` (Pinocchio) -- the same library the
demo uses. P2's pick is then re-run commanded entirely by **Cartesian TCP poses**
instead of joint waypoints, and scored on the same gates.

⚠️ **The URDF and the shipped USD disagree, so the TCP is calibrated, not trusted.**
Pinocchio FK on either DM URDF puts the jaw midpoint 3.5 mm below where the
simulated asset actually has it at the same joint values (x/y agree to 0.3 mm).
Both URDFs give byte-identical FK, so this is a URDF-vs-USD discrepancy, not the
unresolved pi gripper-mount question. This script measures the offset across
several poses, checks whether it is a **constant frame offset** (correctable) or
**pose-dependent** (a genuine kinematic disagreement), and applies it.

Run::

    TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N \\
      ~/isaaclab-venv/bin/python scripts/b601_move_to_traj.py \\
        --repair-nested-xforms --out artifacts/b601_pick/p3_move_to_traj.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src" / "reBotArm_control_py"))

#: Pinned explicitly. The same filename exists three times in this workspace with
#: three different hashes (reBotArm_control_py, reBot-Isaacsim/third_party, and
#: reBotArmController_ROS2), and the SDK's YAML chain would silently pick one.
IK_URDF = (REPO_ROOT / "src" / "reBotArmController_ROS2" / "src" /
           "rebotarm_bringup" / "description" / "urdf" /
           "reBot_B601_DM_with_gripper.urdf")
IK_EE_FRAME = "gripper_link"

#: Poses used to calibrate the URDF->USD TCP offset. Spread so a pose-dependent
#: error cannot hide behind a single sample.
CALIBRATION_POSES = [
    np.array([0.125, -0.497, -0.407, -0.095, 0.027, -0.019]),
    np.array([0.000, -0.560, -0.360, -0.050, 0.000, 0.000]),
    np.array([0.300, -0.430, -0.480, -0.150, 0.100, 0.150]),
    np.array([-0.200, -0.520, -0.330, 0.050, -0.080, -0.120]),
]
TCP_OFFSET_CONSTANT_TOL_M = 1.0e-3   # spread above this means pose-dependent

CARTESIAN_LIFT_M = 0.065             # commanded as a Cartesian +Z instruction
MOVE_TOL_POSITION_M = 5.0e-3


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)


class P3Failure(RuntimeError):
    pass


class Report:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "probe": "b601_move_to_traj", "schema_version": "1.0.0",
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
            raise P3Failure(f"{name}: {json.dumps(_jsonable(f))[:400]}")

    def write(self, path: Path) -> None:
        self.data["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.data["passed"] = (bool(self.data["checks"])
                               and all(c["passed"] for c in self.data["checks"])
                               and not self.data["errors"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(self.data), indent=2, sort_keys=True) + "\n")


class RebotArmSim:
    """The demo's five-call robot interface, backed by the Isaac articulation."""

    def __init__(self, world, articulation, monitor, probe, pick,
                 render: bool = False):
        import pinocchio as pin

        # The SDK's top-level __init__ imports `actuator`, which imports
        # `motorbridge` -- the CAN hardware driver. It is not installable or
        # meaningful in simulation, and it is not needed: the kinematics
        # subpackage depends only on numpy and pinocchio. Register a stub parent
        # package so the real __init__ never executes, then import the
        # kinematics module normally. Nothing about the IK code is bypassed.
        sdk_root = REPO_ROOT / "src" / "reBotArm_control_py"
        if "reBotArm_control_py" not in sys.modules:
            import types
            stub = types.ModuleType("reBotArm_control_py")
            stub.__path__ = [str(sdk_root / "reBotArm_control_py")]
            sys.modules["reBotArm_control_py"] = stub
        from reBotArm_control_py.kinematics.inverse_kinematics import (
            IKParams, solve_ik_with_retry,
        )

        self._pin = pin
        self._solve_ik_with_retry = solve_ik_with_retry
        self._ik_params = IKParams(max_iter=1000, tolerance=1e-5,
                                   step_size=0.5, damping=1e-6)
        self.world = world
        self.articulation = articulation
        self.monitor = monitor
        self.probe = probe
        self.pick = pick
        self.render = render

        self.model = pin.buildModelFromUrdf(str(IK_URDF))
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(IK_EE_FRAME)
        #: base_link <- gripper_link, corrected so FK agrees with the simulator.
        self.T_ee_tcp = np.eye(4)
        self._grip_cmd = 0.0
        self._last_contact_grip: float | None = None

        # Log every per-step joint command so a run can be replayed 1:1 in a
        # render-only session (the wrist-camera Replicator captures conflict
        # with the observer Recorder inside one session, so videos are made by
        # replay). Wrapping apply_action catches every call site at once.
        self.traj_log: list = []
        _orig_apply = self.articulation.apply_action

        def _logged_apply(action):
            jp = getattr(action, "joint_positions", None)
            if jp is not None and len(jp) == 8:
                self.traj_log.append(np.asarray(jp, dtype=np.float64).copy())
            return _orig_apply(action)

        self.articulation.apply_action = _logged_apply

    # -- helpers ---------------------------------------------------------
    def joint_positions(self) -> np.ndarray:
        return np.asarray(self.articulation.get_joint_positions(), dtype=np.float64)

    def jaw_midpoint(self, label: str = "jaw") -> np.ndarray:
        s = self.monitor.sample(label)
        return (np.asarray(s["left_link_position_m"], dtype=np.float64)
                + np.asarray(s["right_link_position_m"], dtype=np.float64)) / 2.0

    def fk_ee(self, q_arm: np.ndarray):
        q = np.zeros(self.model.nq)
        q[:6] = np.asarray(q_arm, dtype=np.float64)[:6]
        self._pin.forwardKinematics(self.model, self.data, q)
        self._pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_frame_id]

    # -- the demo's interface -------------------------------------------
    def get_tcp_pose(self) -> np.ndarray:
        """Measured TCP (jaw midpoint) position in the base frame."""
        return self.jaw_midpoint("get_tcp_pose")

    def move_to_traj(self, x: float, y: float, z: float,
                     rx: float = 0.0, ry: float = 0.0, rz: float = 0.0,
                     duration: float = 2.0, keep_orientation: bool = True) -> dict:
        """Cartesian TCP target -> IK -> ramped joint drive targets.

        Signature mirrors the vendor controller. ``keep_orientation`` reuses the
        current EE orientation, which is what a top-down pick wants and what
        avoids asking a 6-DOF arm for an unreachable full pose.
        """
        q_now = self.joint_positions()
        target_ee_R = (self.fk_ee(q_now[:6]).rotation if keep_orientation
                       else self._pin.rpy.rpyToMatrix(rx, ry, rz))

        # The commanded pose is a TCP pose; IK solves for the EE frame.
        tcp_target = np.eye(4)
        tcp_target[:3, :3] = target_ee_R
        tcp_target[:3, 3] = [x, y, z]
        ee_target = tcp_target @ np.linalg.inv(self.T_ee_tcp)

        target_se3 = self._pin.SE3(ee_target[:3, :3].copy(),
                                   ee_target[:3, 3].copy())
        # Seed with the FULL model width. `solve_ik_with_retry` takes no
        # `controlled_joints` argument (unlike `solve_ik`) and does
        # `q_seed[:] = best.q` with an nq-sized result, so a 6-vector seed raises
        # "could not broadcast (8,) into (6,)". Seeding all 8 avoids it, and the
        # two finger joints are downstream of the gripper_link EE frame, so their
        # Jacobian columns are zero and IK cannot move them to reach the target.
        q_seed = np.zeros(self.model.nq)
        q_seed[:6] = q_now[:6]
        q_seed[6:8] = q_now[6:8]
        result = self._solve_ik_with_retry(
            self.model, self.data, self.ee_frame_id, target_se3,
            q_seed, self._ik_params,
        )
        record = {"commanded_tcp_m": [x, y, z], "ik_success": bool(result.success),
                  "ik_error": float(result.error), "ik_iterations": int(result.iterations),
                  "q_solution_rad": np.asarray(result.q)[:6]}
        if not result.success:
            record["achieved_tcp_m"] = None
            return record

        q_target = np.asarray(result.q, dtype=np.float64)[:6]
        lower = self.probe.EXPECTED_LOWER[:6] + 1e-3
        upper = self.probe.EXPECTED_UPPER[:6] - 1e-3
        record["within_joint_limits"] = bool(np.all(q_target >= lower)
                                             and np.all(q_target <= upper))
        if not record["within_joint_limits"]:
            record["achieved_tcp_m"] = None
            return record

        full = np.concatenate([q_target, [self._grip_cmd, self._grip_cmd]])
        self._ramp_to(full, duration)
        achieved = self.get_tcp_pose()
        record["achieved_tcp_m"] = achieved
        record["position_error_m"] = float(np.linalg.norm(achieved - np.array([x, y, z])))
        return record

    def _ramp_to(self, target_full: np.ndarray, duration: float,
                 settle: float = 0.6) -> None:
        """Smoothstep ramp to a joint target over ``duration``, then settle.

        ``probe._step_target`` ramps over a fixed RAMP_SECONDS, so routing
        move_to_traj through it silently ignored the ``duration`` argument the
        vendor signature promises. A slower ramp also matters physically: the
        first Cartesian lift slipped the object 25 mm because it was too brisk.
        """
        from isaacsim.core.utils.types import ArticulationAction
        idx = np.arange(8, dtype=np.int64)
        start = self.joint_positions()
        steps = max(1, int(round(max(duration, 1e-3) / self.probe.PHYSICS_DT)))
        for i in range(steps):
            f = float(i + 1) / steps
            smooth = f * f * (3.0 - 2.0 * f)
            cmd = start + smooth * (target_full - start)
            self.articulation.apply_action(
                ArticulationAction(joint_positions=cmd, joint_indices=idx))
            self.world.step(render=self.render)
            self.monitor.sample("ramp")
        for _ in range(max(1, int(round(settle / self.probe.PHYSICS_DT)))):
            self.articulation.apply_action(
                ArticulationAction(joint_positions=target_full, joint_indices=idx))
            self.world.step(render=self.render)
            self.monitor.sample("settle")

    def _drive_gripper(self, distance_m: float, seconds: float = 0.8) -> None:
        from isaacsim.core.utils.types import ArticulationAction
        self._grip_cmd = float(np.clip(distance_m, 0.0,
                                       float(self.probe.EXPECTED_UPPER[6])))
        q = self.joint_positions()
        target = np.concatenate([q[:6], [self._grip_cmd, self._grip_cmd]])
        idx = np.arange(8, dtype=np.int64)
        for _ in range(max(1, int(round(seconds / self.probe.PHYSICS_DT)))):
            self.articulation.apply_action(
                ArticulationAction(joint_positions=target, joint_indices=idx))
            self.world.step(render=self.render)
            self.monitor.sample("gripper")

    def open_gripper(self, distance_m: float | None = None) -> float:
        """Open each finger to ``distance_m`` (default: fully open)."""
        if distance_m is None:
            distance_m = float(self.probe.EXPECTED_UPPER[6])
        self._drive_gripper(distance_m)
        return self._grip_cmd

    def grasp(self, force: float | None = None, squeeze_m: float | None = None) -> dict:
        """Close until both fingers are resisted, then squeeze past contact.

        The vendor SDK closes under a force-control state machine; this closes
        under position control and detects contact from drive tracking error,
        which is the measurable equivalent in a position-driven articulation.
        ``squeeze_m`` is the commanded overshoot past first contact.
        """
        from isaacsim.core.utils.types import ArticulationAction
        # 3.5 mm held under P2's joint-space lift but slipped 25 mm under the
        # Cartesian lift. Slip is a normal-force problem, so squeeze harder rather
        # than commanding a taller lift to clear the gate.
        squeeze = 0.0060 if squeeze_m is None else float(squeeze_m)
        idx = np.arange(8, dtype=np.int64)
        q_arm = self.joint_positions()[:6]
        grip = self._grip_cmd
        consecutive, contact, trace = 0, None, []

        while grip > 0.0:
            grip = max(0.0, grip - self.pick.CLOSE_STEP_M)
            target = np.concatenate([q_arm, [grip, grip]])
            for _ in range(max(1, int(round(self.pick.CLOSE_STEP_SETTLE_S
                                            / self.probe.PHYSICS_DT)))):
                self.articulation.apply_action(
                    ArticulationAction(joint_positions=target, joint_indices=idx))
                self.world.step(render=self.render)
                self.monitor.sample("grasp")
            measured = self.joint_positions()
            per_finger = np.abs(measured[6:8] - grip)
            trace.append({"grip_cmd_m": grip, "per_finger_error_m": per_finger})
            if (float(np.max(per_finger)) > self.pick.CONTACT_ERROR_TOL_M
                    and float(np.min(per_finger)) > self.pick.CONTACT_ERROR_TOL_M / 3.0):
                consecutive += 1
                if consecutive >= 2:
                    contact = grip
                    break
            else:
                consecutive = 0

        if contact is None:
            self._grip_cmd = 0.0
            return {"contacted": False, "first_contact_grip_cmd_m": None,
                    "n_steps": len(trace)}

        self._last_contact_grip = contact
        self._drive_gripper(max(0.0, contact - squeeze), seconds=0.6)
        return {"contacted": True, "first_contact_grip_cmd_m": contact,
                "squeeze_cmd_m": self._grip_cmd, "squeeze_depth_m": contact - self._grip_cmd,
                "n_steps": len(trace), "requested_force": force}

    def release_gripper(self, extra_m: float | None = None) -> float:
        base = (self._last_contact_grip if self._last_contact_grip is not None
                else self._grip_cmd)
        extra = self.pick.RELEASE_EXTRA_M if extra_m is None else float(extra_m)
        self._drive_gripper(min(float(self.probe.EXPECTED_UPPER[6]), base + extra),
                            seconds=1.2)
        return self._grip_cmd

    def hold(self, seconds: float) -> None:
        from isaacsim.core.utils.types import ArticulationAction
        q = self.joint_positions()
        target = np.concatenate([q[:6], [self._grip_cmd, self._grip_cmd]])
        idx = np.arange(8, dtype=np.int64)
        for _ in range(max(1, int(round(seconds / self.probe.PHYSICS_DT)))):
            self.articulation.apply_action(
                ArticulationAction(joint_positions=target, joint_indices=idx))
            self.world.step(render=self.render)
            self.monitor.sample("hold")


def _run(args: argparse.Namespace, report: Report) -> None:
    import b601_asset_probe as probe
    import b601_pick as pick
    from isaacsim.core.api import World
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, GroundPlane
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from pxr import PhysxSchema, Usd

    report.require("pinned IK URDF exists", IK_URDF.is_file(), path=str(IK_URDF))
    report.data["ik_model"] = {
        "urdf": str(IK_URDF),
        "sha256": hashlib.sha256(IK_URDF.read_bytes()).hexdigest(),
        "end_effector_frame": IK_EE_FRAME,
        "note": "pinned explicitly; the SDK's YAML chain would pick "
                "reBot-DevArm_fixend.urdf, which has no gripper at all",
    }

    # ---- world + robot (same setup P2 validated) -------------------------
    world = World(physics_dt=probe.PHYSICS_DT, rendering_dt=1.0 / 60.0,
                  stage_units_in_meters=1.0, backend="numpy")
    add_reference_to_stage(str(probe.ASSET_PATH), probe.ROBOT_PRIM_PATH)
    stage = get_current_stage()
    root = stage.GetPrimAtPath(probe.ARTICULATION_ROOT_PATH)
    nested = probe._nested_rigid_body_issues(stage)
    if nested and not args.repair_nested_xforms:
        report.require("nested Xform stacks PhysX-valid", False, issue_count=len(nested))
    if nested:
        rep = probe._repair_nested_rigid_body_xforms(stage, nested)
        report.require("session repair preserves poses",
                       rep["repaired_count"] == len(nested)
                       and rep["remaining_issue_count"] == 0,
                       **{k: rep[k] for k in ("repaired_count", "remaining_issue_count")})

    finger_root = (f"{probe.ROBOT_PRIM_PATH}/Geometry/base_link/link1/link2/link3/"
                   f"link4/link5/link6/gripper_link")
    finger_paths = [f"{finger_root}/gripper_left", f"{finger_root}/gripper_right"]
    fix = pick._refine_finger_colliders(stage, finger_paths)
    report.require("finger colliders decomposed", fix["changed_count"] >= 2, **fix)

    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        pxa = PhysxSchema.PhysxArticulationAPI.Apply(root)
        pxa.CreateEnabledSelfCollisionsAttr(False)
        pxa.CreateSolverVelocityIterationCountAttr(4)
        pxa.CreateSolverPositionIterationCountAttr(32)

    world.scene.add(GroundPlane(prim_path=pick.GROUND_PRIM_PATH, size=4.0))
    grip_material = PhysicsMaterial(
        prim_path="/World/physics_materials/grip",
        static_friction=pick.STATIC_FRICTION,
        dynamic_friction=pick.DYNAMIC_FRICTION, restitution=pick.RESTITUTION)
    articulation = SingleArticulation(prim_path=probe.ARTICULATION_ROOT_PATH,
                                      name="b601_dm")
    world.scene.add(articulation)
    world.reset()
    report.require("exact eight-DOF name order",
                   list(articulation.dof_names) == probe.EXPECTED_DOF_NAMES,
                   actual=list(articulation.dof_names))
    articulation.get_articulation_controller().set_gains(kps=probe.RUNTIME_KP,
                                                         kds=probe.RUNTIME_KD)
    for p in finger_paths:
        pick._bind_physics_material(stage, p, grip_material.prim_path)

    monitor = probe.StateMonitor(articulation)
    arm = RebotArmSim(world, articulation, monitor, probe, pick, render=args.render)
    arm.open_gripper()

    # ---- 1. calibrate the URDF -> USD TCP offset -------------------------
    samples = []
    for i, q in enumerate(CALIBRATION_POSES):
        full = np.concatenate([q, [arm._grip_cmd, arm._grip_cmd]])
        probe._step_target(world, articulation, monitor, arm.joint_positions(),
                           full, label=f"calib_{i}", render=args.render)
        sim_jaw = arm.jaw_midpoint(f"calib_{i}")
        ee = arm.fk_ee(q)
        # Offset expressed in the EE frame, so it is a genuine frame correction
        # rather than a world-space fudge that only holds at one orientation.
        local = ee.rotation.T @ (sim_jaw - ee.translation)
        samples.append({"q_rad": q, "sim_jaw_m": sim_jaw,
                        "fk_ee_m": ee.translation, "offset_ee_frame_m": local})

    offsets = np.array([s["offset_ee_frame_m"] for s in samples])
    mean_offset = offsets.mean(axis=0)
    spread = float(np.max(np.linalg.norm(offsets - mean_offset, axis=1)))
    report.data["tcp_calibration"] = {
        "samples": samples, "mean_offset_ee_frame_m": mean_offset,
        "max_deviation_from_mean_m": spread,
        "constant_offset_tolerance_m": TCP_OFFSET_CONSTANT_TOL_M,
    }
    report.require(
        "URDF->USD TCP offset is a constant frame offset, not pose-dependent",
        spread <= TCP_OFFSET_CONSTANT_TOL_M,
        max_deviation_from_mean_m=spread, mean_offset_ee_frame_m=mean_offset,
        note="a pose-dependent residual would mean the URDF and the shipped USD "
             "are genuinely different kinematics, not a fixed frame difference",
    )
    arm.T_ee_tcp = np.eye(4)
    arm.T_ee_tcp[:3, 3] = mean_offset

    # ---- 2. move_to_traj accuracy ---------------------------------------
    home = CALIBRATION_POSES[0]
    probe._step_target(world, articulation, monitor, arm.joint_positions(),
                       np.concatenate([home, [arm._grip_cmd, arm._grip_cmd]]),
                       label="home", render=args.render)
    grasp_tcp = arm.get_tcp_pose().copy()

    moves = []
    for name, delta in (("up", [0.0, 0.0, 0.040]), ("side", [0.0, 0.030, 0.0]),
                        ("back", [-0.025, 0.0, 0.020]), ("return", [0.0, 0.0, 0.0])):
        tgt = grasp_tcp + np.asarray(delta)
        rec = arm.move_to_traj(*tgt, duration=1.5)
        rec["name"] = name
        moves.append(rec)
    report.data["move_to_traj_accuracy"] = moves
    errs = [m["position_error_m"] for m in moves if m.get("position_error_m") is not None]
    report.require("every move_to_traj target is solved and reached",
                   len(errs) == len(moves) and max(errs) <= MOVE_TOL_POSITION_M,
                   solved=len(errs), commanded=len(moves),
                   max_position_error_m=(max(errs) if errs else None),
                   tolerance_m=MOVE_TOL_POSITION_M)

    # ---- 3. the pick, commanded in Cartesian ----------------------------
    arm.move_to_traj(*grasp_tcp, duration=1.5)
    arm.open_gripper()
    jaw = arm.get_tcp_pose()

    half_h = pick.OBJECT_HEIGHT_M / 2.0
    support_top = float(jaw[2]) - pick.GRASP_CLEARANCE_M
    world.scene.add(FixedCuboid(
        prim_path=pick.PEDESTAL_PRIM_PATH, name="pedestal",
        position=np.array([jaw[0], jaw[1], support_top / 2.0]),
        scale=np.array([pick.PEDESTAL_XY_M[0], pick.PEDESTAL_XY_M[1], support_top]),
        color=np.array([0.30, 0.30, 0.34])))
    obj = world.scene.add(DynamicCuboid(
        prim_path=pick.OBJECT_PRIM_PATH, name="pick_object",
        position=np.array([jaw[0], jaw[1], jaw[2] + 0.50]),
        scale=np.array([pick.OBJECT_SIZE_M, pick.OBJECT_SIZE_M, pick.OBJECT_HEIGHT_M]),
        color=np.array([0.90, 0.35, 0.20]), mass=pick.OBJECT_MASS_KG))
    obj.apply_physics_material(grip_material)

    q_now = arm.joint_positions()
    articulation.set_joints_default_state(positions=q_now)
    world.reset()
    monitor = probe.StateMonitor(articulation)
    arm.monitor = monitor
    articulation.get_articulation_controller().set_gains(kps=probe.RUNTIME_KP,
                                                         kds=probe.RUNTIME_KD)
    arm._grip_cmd = float(q_now[6])
    arm.hold(pick.SETTLE_SECONDS)

    jaw = arm.get_tcp_pose()
    obj.set_world_pose(position=np.array([jaw[0], jaw[1], support_top + half_h]))
    obj.set_linear_velocity(np.zeros(3))
    obj.set_angular_velocity(np.zeros(3))
    arm.hold(pick.SETTLE_SECONDS)
    settled = np.asarray(obj.get_world_pose()[0], dtype=np.float64)
    report.require("object settles on its support",
                   abs(settled[2] - (support_top + half_h)) < 0.01,
                   settled_m=settled, expected_z_m=support_top + half_h)

    grasp_info = arm.grasp(force=20.0)
    report.data["grasp"] = grasp_info
    report.require("grasp() closes both fingers onto the object",
                   grasp_info["contacted"], **grasp_info)

    # The lift is now a Cartesian instruction, not a swept joint delta.
    tcp_before = arm.get_tcp_pose().copy()
    lift_rec = arm.move_to_traj(tcp_before[0], tcp_before[1],
                                tcp_before[2] + CARTESIAN_LIFT_M, duration=4.0)
    report.data["cartesian_lift"] = lift_rec
    report.require("move_to_traj solves the Cartesian lift",
                   lift_rec.get("achieved_tcp_m") is not None
                   and lift_rec.get("position_error_m", 9.9) <= MOVE_TOL_POSITION_M,
                   **{k: v for k, v in lift_rec.items() if k != "q_solution_rad"})

    arm.hold(pick.REQUIRED_HOLD_S)
    held = np.asarray(obj.get_world_pose()[0], dtype=np.float64)
    rise = float(held[2] - settled[2])
    tcp_after = arm.get_tcp_pose().copy()
    tcp_rise = float(tcp_after[2] - tcp_before[2])
    slip = float(tcp_rise - rise)
    report.data["p3_pick"] = {
        "commanded_cartesian_lift_m": CARTESIAN_LIFT_M,
        "object_settled_m": settled, "object_held_m": held, "rise_m": rise,
        "tcp_before_lift_m": tcp_before, "tcp_after_lift_m": tcp_after,
        "tcp_rise_m": tcp_rise, "slip_m": slip,
    }
    report.check("object slip in the jaw stays small during the Cartesian lift",
                 slip <= 0.010, slip_m=slip, tcp_rise_m=tcp_rise, object_rise_m=rise,
                 note="reported explicitly so a taller commanded lift cannot hide "
                      "a poor grasp")
    report.require("P3: object rises at least 50 mm under Cartesian command",
                   rise >= pick.REQUIRED_LIFT_M, rise_m=rise,
                   required_m=pick.REQUIRED_LIFT_M)

    arm.release_gripper()
    released = np.asarray(obj.get_world_pose()[0], dtype=np.float64)
    fall = float(held[2] - released[2])
    report.data["release"] = {"object_released_m": released, "fall_m": fall}
    report.require("release_gripper() drops the object (no hidden attachment)",
                   fall >= pick.RELEASE_FALL_TOL_M, fall_m=fall,
                   required_m=pick.RELEASE_FALL_TOL_M)

    report.data["state_monitor"] = {
        "samples": monitor.samples,
        "max_base_position_drift_m": monitor.max_base_position_drift_m,
        "max_base_angle_drift_rad": monitor.max_base_angle_drift_rad,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "artifacts" / "b601_pick" / "p3_move_to_traj.json")
    ap.add_argument("--repair-nested-xforms", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--hold-open", type=float, default=0.0, metavar="SECONDS")
    args = ap.parse_args(argv)

    report = Report()
    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": not args.headful})
    try:
        _run(args, report)
    except P3Failure as exc:
        report.data["errors"].append(str(exc))
    except Exception as exc:                                   # noqa: BLE001
        import traceback
        report.data["errors"].append(f"{type(exc).__name__}: {exc}")
        report.data["traceback"] = traceback.format_exc()
    finally:
        report.write(args.out)
        if args.hold_open > 0 and args.headful:
            try:
                deadline = time.time() + args.hold_open
                while time.time() < deadline and sim_app.is_running():
                    sim_app.update()
            except KeyboardInterrupt:
                pass
        try:
            sim_app.close()
        except Exception:                                      # noqa: BLE001
            pass
    return 0 if report.data.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
