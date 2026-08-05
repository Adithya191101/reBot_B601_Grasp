#!/usr/bin/env python3
"""P1 + P2: close the B601-DM fingers on a cuboid and physically lift it.

Reproduces the pick half of the Seeed demo in Isaac Sim. Nothing here teleports,
parents, or attaches the object: the arm and both fingers are driven by
articulation position targets while physics steps, and the object moves only
because it is squeezed between two colliders.

Run::

    TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N \\
      ~/isaaclab-venv/bin/python scripts/b601_pick.py \\
        --repair-nested-xforms --out artifacts/b601_pick/b601_pick.json

Stages, in one Isaac session:

1. **calibrate-grasp** -- move to the grasp arm pose with the jaw open, and
   measure the jaw midpoint in world. Nothing is guessed from the URDF; the
   composed simulated asset defines the geometry (PLAN.md 6.2).
2. **calibrate-lift** -- sweep candidate lift poses and measure the jaw rise for
   each, then pick one that clears the required lift with margin. The lift pose
   is *measured*, not assumed reachable.
3. **place** -- spawn the pedestal and cuboid under the measured jaw midpoint and
   let them settle under gravity. Placement is scene setup, not scored motion.
4. **P1 contact** -- close the fingers in small increments and detect first
   contact from the position-target tracking error. This is the number the P0
   probe could not give: it measured finger *link-origin* separation, and the
   colliders are convex hulls, so the true aperture has to be measured against
   the object.
5. **P2 pick** -- squeeze past first contact, raise the arm to the lift pose,
   hold, and measure the object's rise and stability.
6. **release** -- reopen the fingers and confirm the object falls. A held object
   that does not fall when the jaw opens was never really held; this is the
   check that rules out a hidden attachment.
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

# ---------------------------------------------------------------------------
# Scene / task constants
# ---------------------------------------------------------------------------

OBJECT_PRIM_PATH = "/World/pick_object"
PEDESTAL_PRIM_PATH = "/World/pedestal"
GROUND_PRIM_PATH = "/World/ground"

#: 40 mm across the closing direction -- the dimension the P1 definition fixes --
#: but 100 mm tall. The finger colliders are convex hulls roughly 89 x 93 x 39 mm
#: (measured, see report["finger_geometry"]), so a 40 mm cube sitting on a support
#: puts that support INSIDE the jaw: the fingers clamped the pedestal instead of
#: the cube and the lift left the cube behind. A taller object moves the grasp
#: 50 mm clear of whatever holds it up, which is also how the vendor demo grasps
#: bottles and cups.
OBJECT_SIZE_M = 0.040          # x and y: the grasped width
OBJECT_HEIGHT_M = 0.100
GRASP_CLEARANCE_M = 0.050      # jaw height above the support
OBJECT_MASS_KG = 0.050
#: Narrower than the object in the closing direction so the fingers reach the
#: cuboid before they reach the support.
PEDESTAL_XY_M = (0.030, 0.026)

STATIC_FRICTION = 1.2
DYNAMIC_FRICTION = 1.1
RESTITUTION = 0.0

REQUIRED_LIFT_M = 0.050        # PLAN.md 3.1
REQUIRED_HOLD_S = 1.0
LIFT_MARGIN_M = 0.015          # aim above the bar so a marginal pose is not scored

SETTLE_SECONDS = 1.0
CLOSE_STEP_M = 0.0005          # 0.5 mm per probe increment while seeking contact
CLOSE_STEP_SETTLE_S = 0.08
CONTACT_ERROR_TOL_M = 3.0e-4   # tracking error that means "something is in the way"
SQUEEZE_M = 0.0035             # commanded overshoot past first contact
RELEASE_EXTRA_M = 0.020
RELEASE_FALL_TOL_M = 0.010     # object must drop at least this far when released


def _lift_candidates() -> list[np.ndarray]:
    """Arm poses to evaluate for the lift, as deltas on the grasp pose.

    Only the shoulder/elbow/wrist-pitch joints move. Which combination actually
    raises the tool is a property of the composed asset, so every candidate is
    measured rather than reasoned about.
    """
    # Sign and scale are measured, not assumed. A first sweep with negative
    # shoulder deltas drove the tool DOWN monotonically (-24 mm to -216 mm), so
    # the lift direction is +joint2, at roughly 0.24 m/rad near this pose.
    # joint2 headroom from the grasp pose is 0.497 rad (its upper limit is 0),
    # which is ample for the 65 mm target.
    out = []
    for d2, d3, d4 in [
        (0.10, 0.00, 0.00),
        (0.15, 0.00, 0.00),
        (0.20, 0.00, 0.00),
        (0.25, 0.00, 0.00),
        (0.28, -0.05, 0.00),
        (0.32, -0.05, 0.00),
        (0.36, -0.10, 0.00),
        (0.42, -0.12, 0.05),
    ]:
        out.append(np.array([0.0, d2, d3, d4, 0.0, 0.0], dtype=np.float64))
    return out


# ---------------------------------------------------------------------------


class PickFailure(RuntimeError):
    """Expected failure with a concise message for the report."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


class Report:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "probe": "b601_pick",
            "schema_version": "1.0.0",
            "checks": [],
            "errors": [],
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def check(self, name: str, passed: bool, **fields: Any) -> bool:
        entry = {"name": name, "passed": bool(passed)}
        entry.update({k: _jsonable(v) for k, v in fields.items()})
        self.data["checks"].append(entry)
        return bool(passed)

    def require(self, name: str, passed: bool, **fields: Any) -> None:
        if not self.check(name, passed, **fields):
            raise PickFailure(f"{name}: {json.dumps(_jsonable(fields))[:400]}")

    @property
    def passed(self) -> bool:
        return bool(self.data["checks"]) and all(
            c["passed"] for c in self.data["checks"]
        ) and not self.data["errors"]

    def write(self, path: Path) -> None:
        self.data["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.data["passed"] = self.passed
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(self.data), indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------


def _refine_finger_colliders(stage: Any, finger_paths: list[str]) -> dict:
    """Session-only: replace the fingers' convexHull colliders with a decomposition.

    **Asset defect #4, found by trying to grasp with it.** The finger colliders are
    authored ``physics:approximation = "convexHull"``. A convex hull fills the
    concave inner face of a jaw, so each finger becomes a solid blob spanning the
    throat: measured, the two hulls overlap when closed (which is why PhysX
    self-collision had to be disabled at all) and leave only about 24 mm of free
    gap at FULL open, against an authored travel of 143 mm. A 40 mm object cannot
    enter the jaw and is ejected on contact.

    ``convexDecomposition`` recovers the concave shape as a union of convex parts,
    which is the standard approximation for gripper fingers. The opinion is
    written to the session layer only; the referenced USD on disk is untouched.
    """
    from pxr import Usd, UsdPhysics

    changed = []
    deinstanced = []
    inspected = 0
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        # The finger geometry is authored `instanceable = true`, so it lives in a
        # prototype: a plain PrimRange never reaches the collider, and an instance
        # proxy cannot carry an opinion. De-instance the two finger subtrees first
        # -- it costs nothing for two prims and makes the colliders editable.
        for finger_path in finger_paths:
            root = stage.GetPrimAtPath(finger_path)
            if not root or not root.IsValid():
                continue
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if prim.IsInstance():
                    target = stage.GetPrimAtPath(prim.GetPath())
                    if target and target.IsValid() and target.SetInstanceable(False):
                        deinstanced.append(str(prim.GetPath()))

        for finger_path in finger_paths:
            root = stage.GetPrimAtPath(finger_path)
            if not root or not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                inspected += 1
                if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    continue
                api = UsdPhysics.MeshCollisionAPI(prim)
                attr = api.GetApproximationAttr()
                before = attr.Get() if attr else None
                api.CreateApproximationAttr().Set("convexDecomposition")
                changed.append({"prim": str(prim.GetPath()),
                                "before": str(before), "after": "convexDecomposition"})
    return {"changed_count": len(changed), "changed": changed,
            "deinstanced_count": len(deinstanced), "deinstanced": deinstanced,
            "prims_inspected": inspected, "persisted_to_disk": False}


def _bind_physics_material(stage: Any, prim_path: str, material_path: str) -> bool:
    """Bind a physics material to an existing prim (including mesh colliders)."""
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim or not material_prim.IsValid():
        return False
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(
        UsdShade.Material(material_prim),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )
    return True


def _run(args: argparse.Namespace, report: Report, sim_app: Any) -> None:
    import b601_asset_probe as probe
    from isaacsim.core.api import World
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, GroundPlane
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import PhysxSchema, Usd

    # ---- world + robot, reusing the P0-validated setup --------------------
    report.require("shipped DM USD exists", probe.ASSET_PATH.is_file(),
                   asset_path=str(probe.ASSET_PATH))
    import hashlib
    report.data["asset"] = {
        "path": str(probe.ASSET_PATH),
        "sha256_root_layer": hashlib.sha256(probe.ASSET_PATH.read_bytes()).hexdigest(),
    }

    world = World(physics_dt=probe.PHYSICS_DT, rendering_dt=1.0 / 60.0,
                  stage_units_in_meters=1.0, backend="numpy")
    add_reference_to_stage(str(probe.ASSET_PATH), probe.ROBOT_PRIM_PATH)
    stage = get_current_stage()
    articulation_root_prim = stage.GetPrimAtPath(probe.ARTICULATION_ROOT_PATH)
    report.require("referenced USD prims resolve",
                   bool(articulation_root_prim.IsValid()))

    nested_issues = probe._nested_rigid_body_issues(stage)
    if nested_issues and not args.repair_nested_xforms:
        report.require("nested rigid-body Xform stacks are PhysX-valid", False,
                       issue_count=len(nested_issues),
                       hint="rerun with --repair-nested-xforms")
    if nested_issues:
        repair = probe._repair_nested_rigid_body_xforms(stage, nested_issues)
        report.require(
            "session repair preserves poses",
            repair["repaired_count"] == len(nested_issues)
            and repair["remaining_issue_count"] == 0
            and repair["max_initial_world_matrix_error"] <= 1.0e-10,
            **{k: repair[k] for k in
               ("repaired_count", "remaining_issue_count", "max_initial_world_matrix_error")},
        )

    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(articulation_root_prim)
        physx_articulation.CreateEnabledSelfCollisionsAttr(False)
        physx_articulation.CreateSolverVelocityIterationCountAttr(4)
        physx_articulation.CreateSolverPositionIterationCountAttr(32)
    finger_root = (f"{probe.ROBOT_PRIM_PATH}/Geometry/base_link/link1/link2/link3/"
                   f"link4/link5/link6/gripper_link")
    finger_paths = [f"{finger_root}/gripper_left", f"{finger_root}/gripper_right"]
    collider_fix = _refine_finger_colliders(stage, finger_paths)
    report.data["asset_compatibility"] = {"finger_collider_refinement": collider_fix}
    report.require("finger colliders refined from convexHull to convexDecomposition",
                   collider_fix["changed_count"] >= 2, **collider_fix)

    report.check("PhysX articulation configured for contact",
                 True, self_collisions=False, position_iterations=32,
                 note="self-collision off (coarse convex hulls); external object "
                      "contacts remain enabled")

    # ---- scene ------------------------------------------------------------
    world.scene.add(GroundPlane(prim_path=GROUND_PRIM_PATH, size=4.0))
    grip_material = PhysicsMaterial(
        prim_path="/World/physics_materials/grip",
        static_friction=STATIC_FRICTION,
        dynamic_friction=DYNAMIC_FRICTION,
        restitution=RESTITUTION,
    )

    articulation = SingleArticulation(prim_path=probe.ARTICULATION_ROOT_PATH,
                                      name="b601_dm")
    world.scene.add(articulation)
    world.reset()

    dof_names = list(articulation.dof_names)
    report.require("exact eight-DOF name order",
                   dof_names == probe.EXPECTED_DOF_NAMES, actual=dof_names)
    n_dof = len(dof_names)
    articulation.get_articulation_controller().set_gains(
        kps=probe.RUNTIME_KP, kds=probe.RUNTIME_KD
    )

    # Friction on the finger colliders, so a squeeze can actually hold.
    bound = [_bind_physics_material(stage, path, grip_material.prim_path)
             for path in finger_paths]
    report.check("friction material bound to both fingers", all(bound),
                 bound=bound, static_friction=STATIC_FRICTION,
                 dynamic_friction=DYNAMIC_FRICTION)

    def finger_local_bbox(finger: str) -> dict:
        """Local-frame collider bounds of one finger.

        The P0 probe measured finger *link origins*; the colliders are convex
        hulls whose surfaces sit somewhere else entirely. Without these bounds
        the true aperture is unknown, and 'origin separation' silently
        overstates how open the jaw is.
        """
        from pxr import Gf, Usd, UsdGeom
        path = (f"{probe.ROBOT_PRIM_PATH}/Geometry/base_link/link1/link2/link3/"
                f"link4/link5/link6/gripper_link/{finger}")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                  [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return {"path": path, "valid": False}
        bound = cache.ComputeLocalBound(prim)
        rng = bound.ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        return {"path": path, "valid": True,
                "local_min_m": [mn[0], mn[1], mn[2]],
                "local_max_m": [mx[0], mx[1], mx[2]],
                "local_size_m": [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]]}

    report.data["finger_geometry"] = {
        f: finger_local_bbox(f) for f in ("gripper_left", "gripper_right")
    }

    monitor = probe.StateMonitor(articulation)

    def jaw_midpoint(label: str = "jaw") -> np.ndarray:
        """Live jaw midpoint from the PhysX tensor API.

        Must NOT come from USD xforms: with PhysX simulating, the stage is not
        written back, so ``get_world_pose`` returns the spawn pose and every
        measured lift is exactly zero. This is the same measurement path the P0
        probe validated.
        """
        state = monitor.sample(label)
        return (np.asarray(state["left_link_position_m"], dtype=np.float64)
                + np.asarray(state["right_link_position_m"], dtype=np.float64)) / 2.0

    def goto(arm_q: np.ndarray, grip_m: float, label: str) -> dict:
        target = np.concatenate([arm_q, np.array([grip_m, grip_m])])
        start = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
        return probe._step_target(world, articulation, monitor, start, target,
                                  label=label, render=args.render)

    def hold_steps(target: np.ndarray, seconds: float) -> None:
        indices = np.arange(n_dof, dtype=np.int64)
        for _ in range(max(1, int(round(seconds / probe.PHYSICS_DT)))):
            articulation.apply_action(
                ArticulationAction(joint_positions=target, joint_indices=indices))
            world.step(render=args.render)
            monitor.sample("step")

    # ---- 1. calibrate the grasp pose -------------------------------------
    grasp_arm_q = probe.SAFE_ARM_TARGET.copy()
    open_grip = float(probe.EXPECTED_UPPER[6])
    goto(grasp_arm_q, open_grip, "grasp_pose_open")
    jaw_open = jaw_midpoint("grasp_open")
    report.data["grasp_pose"] = {
        "arm_q_rad": grasp_arm_q, "gripper_open_m": open_grip,
        "jaw_midpoint_world_m": jaw_open,
    }
    report.check("jaw midpoint measured at grasp pose",
                 bool(np.all(np.isfinite(jaw_open))), jaw_midpoint_world_m=jaw_open)

    # ---- 2. calibrate the lift pose --------------------------------------
    lift_trials = []
    chosen_lift = None
    for i, delta in enumerate(_lift_candidates()):
        cand = grasp_arm_q + delta
        inside = bool(np.all(cand >= probe.EXPECTED_LOWER[:6] + 1e-3)
                      and np.all(cand <= probe.EXPECTED_UPPER[:6] - 1e-3))
        if not inside:
            lift_trials.append({"index": i, "delta": delta, "within_limits": False})
            continue
        goto(cand, open_grip, f"lift_candidate_{i}")
        jaw = jaw_midpoint(f"lift_candidate_{i}")
        rise = float(jaw[2] - jaw_open[2])
        lateral = float(np.linalg.norm((jaw - jaw_open)[:2]))
        lift_trials.append({"index": i, "delta": delta, "within_limits": True,
                            "jaw_world_m": jaw, "rise_m": rise, "lateral_m": lateral})
        if chosen_lift is None and rise >= REQUIRED_LIFT_M + LIFT_MARGIN_M:
            chosen_lift = {"index": i, "arm_q_rad": cand, "rise_m": rise,
                           "lateral_m": lateral}
    report.data["lift_calibration"] = {"trials": lift_trials, "chosen": chosen_lift}
    report.require(
        "a lift pose clears the required rise with margin",
        chosen_lift is not None,
        required_m=REQUIRED_LIFT_M, margin_m=LIFT_MARGIN_M,
        best_rise_m=max((t.get("rise_m", -9.9) for t in lift_trials), default=None),
    )
    lift_arm_q = np.asarray(chosen_lift["arm_q_rad"], dtype=np.float64)

    # Return to the grasp pose before anything is placed.
    goto(grasp_arm_q, open_grip, "return_to_grasp_pose")
    jaw_open = jaw_midpoint("grasp_open_recheck")

    # ---- 3. place the object ---------------------------------------------
    half = OBJECT_SIZE_M / 2.0
    pedestal_top = float(jaw_open[2]) - GRASP_CLEARANCE_M
    report.require("pedestal fits under the jaw", pedestal_top > 0.02,
                   pedestal_top_m=pedestal_top, jaw_z_m=float(jaw_open[2]))
    world.scene.add(FixedCuboid(
        prim_path=PEDESTAL_PRIM_PATH, name="pedestal",
        position=np.array([jaw_open[0], jaw_open[1], pedestal_top / 2.0]),
        scale=np.array([PEDESTAL_XY_M[0], PEDESTAL_XY_M[1], pedestal_top]),
        color=np.array([0.30, 0.30, 0.34]),
    ))
    # Spawn the cuboid PARKED well above the workspace. Spawning it at the jaw
    # put it directly in the arm's approach path: world.reset() returns the arm
    # to its default pose, and the sweep back to the grasp pose drove a finger
    # into the cube, which pinned gripper_joint2 at 16.5 mm and registered as an
    # instant false contact. The object is placed only once the jaw is already
    # open and in position -- placement is scene setup, not scored motion.
    obj = world.scene.add(DynamicCuboid(
        prim_path=OBJECT_PRIM_PATH, name="pick_object",
        position=np.array([jaw_open[0], jaw_open[1], jaw_open[2] + 0.50]),
        scale=np.array([OBJECT_SIZE_M, OBJECT_SIZE_M, OBJECT_HEIGHT_M]),
        color=np.array([0.90, 0.35, 0.20]),
        mass=OBJECT_MASS_KG,
    ))
    obj.apply_physics_material(grip_material)

    # Land the arm directly on the grasp pose instead of sweeping to it. The
    # pedestal is a 273 mm column standing where the jaw works, and the default
    # pose -> grasp pose sweep collided with it: the jaw ended 86 mm off, so the
    # cube was placed beside its support and fell to the floor. Seeding the
    # default state is an explicit reset, which PLAN.md 3.1 permits; the scored
    # motion is the close and lift that follow.
    grasp_full = np.concatenate([grasp_arm_q, [open_grip, open_grip]])
    articulation.set_joints_default_state(positions=grasp_full)
    world.reset()
    # world.reset() rebuilds the PhysX simulation view, which invalidates the
    # tensor handle StateMonitor cached at construction ("Failed to get link
    # transforms from backend"). Rebuild the monitor against the new view.
    def finger_local_bbox(finger: str) -> dict:
        """Local-frame collider bounds of one finger.

        The P0 probe measured finger *link origins*; the colliders are convex
        hulls whose surfaces sit somewhere else entirely. Without these bounds
        the true aperture is unknown, and 'origin separation' silently
        overstates how open the jaw is.
        """
        from pxr import Gf, Usd, UsdGeom
        path = (f"{probe.ROBOT_PRIM_PATH}/Geometry/base_link/link1/link2/link3/"
                f"link4/link5/link6/gripper_link/{finger}")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                  [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return {"path": path, "valid": False}
        bound = cache.ComputeLocalBound(prim)
        rng = bound.ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        return {"path": path, "valid": True,
                "local_min_m": [mn[0], mn[1], mn[2]],
                "local_max_m": [mx[0], mx[1], mx[2]],
                "local_size_m": [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]]}

    report.data["finger_geometry"] = {
        f: finger_local_bbox(f) for f in ("gripper_left", "gripper_right")
    }

    monitor = probe.StateMonitor(articulation)
    articulation.get_articulation_controller().set_gains(
        kps=probe.RUNTIME_KP, kds=probe.RUNTIME_KD)
    # The arm is already at the grasp pose; hold it there under drive control so
    # the solver settles, then drop the object into the already-open jaw.
    hold_steps(grasp_full, SETTLE_SECONDS)
    jaw_ready = jaw_midpoint("jaw_ready")
    jaw_shift = float(np.linalg.norm(jaw_ready - jaw_open))
    report.require(
        "jaw returns to the calibrated grasp pose after the scene is built",
        jaw_shift < 5.0e-3,
        jaw_open_m=jaw_open, jaw_ready_m=jaw_ready, shift_m=jaw_shift,
        note="a large shift means the arm was deflected by the scene, which "
             "would place the object off its support",
    )
    obj.set_world_pose(position=np.array([jaw_ready[0], jaw_ready[1],
                                          pedestal_top + OBJECT_HEIGHT_M / 2.0]))
    obj.set_linear_velocity(np.zeros(3))
    obj.set_angular_velocity(np.zeros(3))
    hold_steps(np.concatenate([grasp_arm_q, [open_grip, open_grip]]), SETTLE_SECONDS)

    settled_pos, _ = obj.get_world_pose()
    settled_pos = np.asarray(settled_pos, dtype=np.float64)
    report.data["object"] = {
        "size_m": OBJECT_SIZE_M, "mass_kg": OBJECT_MASS_KG,
        "parked_world_m": [jaw_open[0], jaw_open[1], jaw_open[2] + 0.50],
        "placed_world_m": [jaw_ready[0], jaw_ready[1],
                           pedestal_top + OBJECT_HEIGHT_M / 2.0],
        "height_m": OBJECT_HEIGHT_M,
        "grasp_clearance_above_support_m": GRASP_CLEARANCE_M,
        "settled_world_m": settled_pos,
        "pedestal_top_m": pedestal_top,
    }
    report.require("object settles on the pedestal without exploding",
                   bool(np.all(np.isfinite(settled_pos))
                        and abs(settled_pos[2] - (pedestal_top + OBJECT_HEIGHT_M / 2.0)) < 0.01),
                   settled_world_m=settled_pos,
                   expected_z_m=pedestal_top + OBJECT_HEIGHT_M / 2.0)

    # ---- 4. P1: seek first contact ---------------------------------------
    contact_grip = None
    close_trace = []
    grip_cmd = open_grip
    consecutive = 0
    indices = np.arange(n_dof, dtype=np.int64)
    while grip_cmd > 0.0:
        grip_cmd = max(0.0, grip_cmd - CLOSE_STEP_M)
        target = np.concatenate([grasp_arm_q, [grip_cmd, grip_cmd]])
        for _ in range(max(1, int(round(CLOSE_STEP_SETTLE_S / probe.PHYSICS_DT)))):
            articulation.apply_action(
                ArticulationAction(joint_positions=target, joint_indices=indices))
            world.step(render=args.render)
            monitor.sample("step")
        measured = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
        per_finger = np.abs(measured[6:8] - grip_cmd)
        err = float(np.max(per_finger))
        obj_pos = np.asarray(obj.get_world_pose()[0], dtype=np.float64)
        close_trace.append({"grip_cmd_m": grip_cmd,
                            "measured_m": measured[6:8],
                            "per_finger_error_m": per_finger,
                            "tracking_error_m": err,
                            "object_z_m": float(obj_pos[2])})
        # Both fingers must be resisted, and it must persist for two consecutive
        # increments -- a single lagging sample is not contact.
        both = bool(np.min(per_finger) > CONTACT_ERROR_TOL_M / 3.0)
        if err > CONTACT_ERROR_TOL_M and both:
            consecutive += 1
            if consecutive >= 2:
                contact_grip = grip_cmd
                break
        else:
            consecutive = 0

    report.data["p1_contact"] = {
        "first_contact_grip_cmd_m": contact_grip,
        "implied_aperture_m": None if contact_grip is None else 2.0 * contact_grip,
        "object_size_m": OBJECT_SIZE_M,
        "tracking_error_tol_m": CONTACT_ERROR_TOL_M,
        "both_fingers_resisted": True if contact_grip is not None else False,
        "trace_tail": close_trace[-8:],
        "n_steps": len(close_trace),
    }
    report.require("P1: both fingers make contact with the object before closing",
                   contact_grip is not None and contact_grip > 0.0,
                   first_contact_grip_cmd_m=contact_grip)
    # The gap between contact and the object width is the convex-hull inset per
    # finger -- an asset property, measured here for the first time. It is
    # reported, not scored: P0 could only measure link-origin separation, which
    # overstates how open the jaw really is by exactly this much.
    finger_inset = (2.0 * contact_grip - OBJECT_SIZE_M) / 2.0
    report.data["p1_contact"]["implied_finger_inset_m"] = finger_inset
    report.check(
        "P1: contact occurs strictly between fully open and fully closed",
        0.0 < contact_grip < open_grip,
        first_contact_grip_cmd_m=contact_grip, open_grip_m=open_grip,
        implied_finger_inset_m=finger_inset,
        note="inset is how far each convex-hull finger surface sits inside its "
             "link origin; link-origin separation is not the aperture",
    )

    # ---- 5. P2: squeeze, lift, hold --------------------------------------
    squeeze_cmd = max(0.0, contact_grip - SQUEEZE_M)
    grasp_target = np.concatenate([grasp_arm_q, [squeeze_cmd, squeeze_cmd]])
    hold_steps(grasp_target, 0.6)
    after_squeeze = np.asarray(obj.get_world_pose()[0], dtype=np.float64)

    lift_target = np.concatenate([lift_arm_q, [squeeze_cmd, squeeze_cmd]])
    start = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
    probe._step_target(world, articulation, monitor, start, lift_target,
                       label="lift", render=args.render)

    hold_samples = []
    hold_steps_n = max(1, int(round(REQUIRED_HOLD_S / probe.PHYSICS_DT)))
    for i in range(hold_steps_n):
        articulation.apply_action(
            ArticulationAction(joint_positions=lift_target, joint_indices=indices))
        world.step(render=args.render)
        monitor.sample("hold")
        if i % 12 == 0:
            p = np.asarray(obj.get_world_pose()[0], dtype=np.float64)
            hold_samples.append({"t_s": i * probe.PHYSICS_DT, "object_world_m": p})
    held_pos = np.asarray(obj.get_world_pose()[0], dtype=np.float64)
    measured_after_lift = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
    jaw_after_lift = jaw_midpoint("jaw_after_lift")

    rise = float(held_pos[2] - settled_pos[2])
    hold_z = [s["object_world_m"][2] for s in hold_samples]
    hold_drop = float(max(hold_z) - min(hold_z)) if hold_z else 0.0

    report.data["p2_pick"] = {
        "squeeze_cmd_m": squeeze_cmd,
        "squeeze_depth_m": contact_grip - squeeze_cmd,
        "object_after_squeeze_m": after_squeeze,
        "object_after_lift_m": held_pos,
        "rise_m": rise,
        "hold_seconds": REQUIRED_HOLD_S,
        "hold_z_range_m": hold_drop,
        "hold_samples": hold_samples,
        "lift_arm_q_rad": lift_arm_q,
        "measured_arm_after_lift_rad": measured_after_lift[:6],
        "arm_tracking_error_rad": measured_after_lift[:6] - lift_arm_q,
        "measured_grip_after_lift_m": measured_after_lift[6:8],
        "jaw_after_lift_m": jaw_after_lift,
        "jaw_rise_m": float(jaw_after_lift[2] - jaw_ready[2]),
    }
    report.check("P2 diagnostic: the arm itself reached the lift pose",
                 float(np.max(np.abs(measured_after_lift[:6] - lift_arm_q))) < 5.0e-2,
                 arm_tracking_error_rad=measured_after_lift[:6] - lift_arm_q,
                 jaw_rise_m=float(jaw_after_lift[2] - jaw_ready[2]))
    report.require("P2: object rises at least 50 mm", rise >= REQUIRED_LIFT_M,
                   rise_m=rise, required_m=REQUIRED_LIFT_M)
    report.require("P2: object is still held after the hold interval",
                   hold_drop < 0.010, hold_z_range_m=hold_drop,
                   hold_seconds=REQUIRED_HOLD_S)
    report.require("P2: object state stays finite",
                   bool(np.all(np.isfinite(held_pos))), object_world_m=held_pos)

    # ---- 6. release: prove there is no hidden attachment ------------------
    release_cmd = min(float(probe.EXPECTED_UPPER[6]), contact_grip + RELEASE_EXTRA_M)
    release_target = np.concatenate([lift_arm_q, [release_cmd, release_cmd]])
    hold_steps(release_target, 1.2)
    released_pos = np.asarray(obj.get_world_pose()[0], dtype=np.float64)
    fall = float(held_pos[2] - released_pos[2])
    report.data["release"] = {
        "release_cmd_m": release_cmd,
        "object_after_release_m": released_pos,
        "fall_m": fall,
    }
    report.require(
        "release: the object falls when the jaw opens (no hidden attachment)",
        fall >= RELEASE_FALL_TOL_M,
        fall_m=fall, required_m=RELEASE_FALL_TOL_M,
        note="an object that does not fall when released was never held by contact",
    )

    report.data["state_monitor"] = {
        "samples": monitor.samples,
        "max_base_position_drift_m": monitor.max_base_position_drift_m,
        "max_base_angle_drift_rad": monitor.max_base_angle_drift_rad,
    }
    report.check("fixed base stays stable through the pick",
                 monitor.max_base_position_drift_m <= probe.BASE_POSITION_DRIFT_TOL_M
                 and monitor.max_base_angle_drift_rad <= probe.BASE_ANGLE_DRIFT_TOL_RAD,
                 max_base_position_drift_m=monitor.max_base_position_drift_m,
                 max_base_angle_drift_rad=monitor.max_base_angle_drift_rad)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "artifacts" / "b601_pick" / "b601_pick.json")
    ap.add_argument("--repair-nested-xforms", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--headful", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = Report()

    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": not args.headful})
    try:
        _run(args, report, sim_app)
    except PickFailure as exc:
        report.data["errors"].append(str(exc))
    except Exception as exc:                                   # noqa: BLE001
        import traceback
        report.data["errors"].append(f"{type(exc).__name__}: {exc}")
        report.data["traceback"] = traceback.format_exc()
    finally:
        report.write(args.out)
        try:
            sim_app.close()
        except Exception:                                      # noqa: BLE001
            pass
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
