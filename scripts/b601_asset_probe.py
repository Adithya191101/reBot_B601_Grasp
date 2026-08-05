#!/usr/bin/env python3
"""Dynamic articulation probe for the shipped reBot B601 DM USD.

Run this file with the Isaac Sim 5.1 Python environment, for example::

    ~/isaaclab-venv/bin/python scripts/b601_asset_probe.py \
        --out /tmp/b601_asset_probe.json

The probe deliberately distinguishes state teleportation from control.  It never
calls ``set_joint_positions``: all motion is produced by runtime PD gains and
``ArticulationAction`` position targets followed by physics steps.

The converted DM asset applies ``PhysicsDriveAPI`` and authors force limits, but
does not author drive stiffness or damping.  Runtime gains are therefore part of
the probe rather than an optional tuning step.  They are not saved back to USD.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_PATH = (
    REPO_ROOT
    / "src"
    / "reBot-Isaacsim"
    / "usd"
    / "reBot_B601_DM"
    / "reBot_B601_DM.usda"
)
ROBOT_PRIM_PATH = "/World/reBot_B601_DM"
ARTICULATION_ROOT_PATH = f"{ROBOT_PRIM_PATH}/Geometry/base_link"

EXPECTED_DOF_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_joint1",
    "gripper_joint2",
]
EXPECTED_LOWER = np.array(
    [-2.8, -3.14, -3.14, -1.87, -1.57, -3.14, 0.0, 0.0],
    dtype=np.float64,
)
EXPECTED_UPPER = np.array(
    [2.8, 0.0, 0.0, 1.57, 1.57, 3.14, 0.0715, 0.0715],
    dtype=np.float64,
)
EXPECTED_MAX_EFFORT = np.array(
    [27.0, 27.0, 27.0, 7.0, 7.0, 7.0, 100.0, 100.0],
    dtype=np.float64,
)

# A conservative pose copied from the repository's no-hardware joint sender.
# It stays comfortably inside every DM joint limit.
SAFE_ARM_TARGET = np.array(
    [0.125, -0.497, -0.407, -0.095, 0.027, -0.019],
    dtype=np.float64,
)

# These are runtime-only starting gains.  The arm values are the neighboring RS
# asset's PhysX-validated gains; the DM probe still verifies actual tracking and
# reports the values explicitly rather than claiming they are DM-calibrated.
RUNTIME_KP = np.array(
    [500.0, 1500.0, 1000.0, 150.0, 80.0, 50.0, 5000.0, 5000.0],
    dtype=np.float64,
)
RUNTIME_KD = np.array(
    [60.0, 96.0, 76.0, 18.0, 10.0, 7.0, 41.28, 41.28],
    dtype=np.float64,
)

PHYSICS_DT = 1.0 / 120.0
RAMP_SECONDS = 1.5
SETTLE_SECONDS = 2.0
SETTLE_TAIL_SECONDS = 0.5

LIMIT_ATOL = 2.0e-4
EFFORT_ATOL = 1.0e-4
ARM_TRACKING_TOL_RAD = 2.0e-2
GRIPPER_TRACKING_TOL_M = 1.0e-3
SETTLED_ARM_POSITION_TOL_RAD = 1.0e-4
SETTLED_GRIPPER_POSITION_TOL_M = 1.0e-5
BASE_POSITION_DRIFT_TOL_M = 1.0e-4
BASE_ANGLE_DRIFT_TOL_RAD = 1.0e-3
FINGER_SYMMETRY_TOL_M = 5.0e-4
SEPARATION_MODEL_TOL_M = 2.0e-3


class ProbeFailure(RuntimeError):
    """Expected validation failure with a concise report message."""


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy, Torch, or Warp-backed Isaac tensors to NumPy."""

    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _quat_angle_rad(first_xyzw: np.ndarray, second_xyzw: np.ndarray) -> float:
    first = np.asarray(first_xyzw, dtype=np.float64)
    second = np.asarray(second_xyzw, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return 2.0 * math.acos(cosine)


@dataclass
class Report:
    output_path: Path
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data.update(
            {
                "schema_version": 1,
                "probe": "b601_asset_probe",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "passed": False,
                "checks": [],
                "errors": [],
            }
        )

    def check(self, name: str, passed: bool, **details: Any) -> None:
        entry = {"name": name, "passed": bool(passed)}
        entry.update(details)
        self.data["checks"].append(_jsonable(entry))

    def require(self, name: str, condition: bool, **details: Any) -> None:
        self.check(name, condition, **details)
        if not condition:
            raise ProbeFailure(name)

    def error(self, message: str, trace: str | None = None) -> None:
        entry: dict[str, Any] = {"message": message}
        if trace:
            entry["traceback"] = trace
        self.data["errors"].append(entry)

    def finish(self) -> None:
        checks = self.data["checks"]
        self.data["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self.data["passed"] = bool(
            checks
            and all(check["passed"] for check in checks)
            and not self.data["errors"]
        )

    def write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(_jsonable(self.data), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class StateMonitor:
    """Sample live articulation tensors and retain worst-case base drift."""

    def __init__(self, articulation: Any) -> None:
        self.articulation = articulation
        self.view = articulation._articulation_view
        self.physics_view = self.view._physics_view
        self.body_names = list(self.view.body_names)
        self.base_index = int(self.view.get_link_index("base_link"))
        self.left_index = int(self.view.get_link_index("gripper_left"))
        self.right_index = int(self.view.get_link_index("gripper_right"))
        self.base_reference: np.ndarray | None = None
        self.max_base_position_drift_m = 0.0
        self.max_base_angle_drift_rad = 0.0
        self.samples = 0

    def _link_transforms(self) -> np.ndarray:
        transforms = _as_numpy(self.physics_view.get_link_transforms())
        return transforms.reshape(1, len(self.body_names), 7)[0].astype(
            np.float64, copy=True
        )

    def sample(self, label: str) -> dict[str, Any]:
        positions = np.asarray(
            self.articulation.get_joint_positions(), dtype=np.float64
        )
        velocities = np.asarray(
            self.articulation.get_joint_velocities(), dtype=np.float64
        )
        links = self._link_transforms()
        arrays = {
            "joint_positions": positions,
            "joint_velocities": velocities,
            "link_transforms": links,
        }
        nonfinite = {
            name: np.argwhere(~np.isfinite(array)).tolist()
            for name, array in arrays.items()
            if not np.all(np.isfinite(array))
        }
        if nonfinite:
            raise ProbeFailure(f"non-finite state at {label}: {nonfinite}")

        base = links[self.base_index]
        if self.base_reference is None:
            self.base_reference = base.copy()
        position_drift = float(
            np.linalg.norm(base[:3] - self.base_reference[:3])
        )
        angle_drift = _quat_angle_rad(base[3:], self.base_reference[3:])
        self.max_base_position_drift_m = max(
            self.max_base_position_drift_m, position_drift
        )
        self.max_base_angle_drift_rad = max(
            self.max_base_angle_drift_rad, angle_drift
        )
        self.samples += 1

        left_position = links[self.left_index, :3]
        right_position = links[self.right_index, :3]
        return {
            "label": label,
            "joint_positions": positions,
            "joint_velocities": velocities,
            "base_transform_xyz_xyzw": base,
            "base_position_drift_m": position_drift,
            "base_angle_drift_rad": angle_drift,
            "left_link_position_m": left_position,
            "right_link_position_m": right_position,
            "link_origin_separation_m": float(
                np.linalg.norm(left_position - right_position)
            ),
        }


def _step_target(
    world: Any,
    articulation: Any,
    monitor: StateMonitor,
    start: np.ndarray,
    target: np.ndarray,
    *,
    label: str,
    render: bool,
) -> dict[str, Any]:
    """Ramp to and settle at a physics-drive target; never set joint state."""

    from isaacsim.core.utils.types import ArticulationAction

    indices = np.arange(len(EXPECTED_DOF_NAMES), dtype=np.int64)
    ramp_steps = max(1, int(round(RAMP_SECONDS / PHYSICS_DT)))
    settle_steps = max(1, int(round(SETTLE_SECONDS / PHYSICS_DT)))
    tail_steps = min(
        settle_steps,
        max(2, int(round(SETTLE_TAIL_SECONDS / PHYSICS_DT))),
    )

    for step in range(ramp_steps):
        fraction = float(step + 1) / float(ramp_steps)
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        command = start + smooth * (target - start)
        articulation.apply_action(
            ArticulationAction(
                joint_positions=command,
                joint_indices=indices,
            )
        )
        world.step(render=render)
        monitor.sample(f"{label}:ramp:{step + 1}")

    action = ArticulationAction(
        joint_positions=target,
        joint_indices=indices,
    )
    settle_tail: list[dict[str, Any]] = []
    for step in range(settle_steps):
        articulation.apply_action(action)
        world.step(render=render)
        state = monitor.sample(f"{label}:settle:{step + 1}")
        if step >= settle_steps - tail_steps:
            settle_tail.append(state)

    final = monitor.sample(f"{label}:final")
    tail_positions = np.stack(
        [np.asarray(state["joint_positions"]) for state in settle_tail], axis=0
    )
    tail_velocities = np.stack(
        [np.asarray(state["joint_velocities"]) for state in settle_tail], axis=0
    )
    final["settle_tail"] = {
        "duration_s": SETTLE_TAIL_SECONDS,
        "samples": len(settle_tail),
        "position_span": np.ptp(tail_positions, axis=0),
        "position_net_change": tail_positions[-1] - tail_positions[0],
        "median_abs_reported_velocity": np.median(
            np.abs(tail_velocities), axis=0
        ),
        "max_abs_reported_velocity": np.max(
            np.abs(tail_velocities), axis=0
        ),
    }
    return final


def _authored_drive_state(stage: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, name in enumerate(EXPECTED_DOF_NAMES):
        kind = "angular" if index < 6 else "linear"
        prim = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/Physics/{name}")
        attributes: dict[str, Any] = {}
        for field_name in (
            "type",
            "stiffness",
            "damping",
            "targetPosition",
            "maxForce",
        ):
            attribute = prim.GetAttribute(
                f"drive:{kind}:physics:{field_name}"
            )
            attributes[field_name] = {
                "valid": bool(attribute.IsValid()),
                "authored": bool(
                    attribute.IsValid() and attribute.HasAuthoredValueOpinion()
                ),
                "value": attribute.Get() if attribute.IsValid() else None,
            }
        result[name] = {"kind": kind, "attributes": attributes}
    return result


def _matrix_numpy(matrix: Any) -> np.ndarray:
    return np.array(
        [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def _nested_rigid_body_issues(stage: Any) -> list[dict[str, str]]:
    """Find nested rigid bodies that inherit another rigid body's Xform stack.

    PhysX permits nested USD prims, but each child rigid body must reset its
    Xform stack.  The shipped DM conversion omits those resets on every body
    below ``base_link``; PhysX then discards the bodies and their joints.
    """

    from pxr import UsdGeom, UsdPhysics

    robot_prefix = ROBOT_PRIM_PATH + "/"
    rigid_prims = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(robot_prefix)
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    rigid_paths = {str(prim.GetPath()) for prim in rigid_prims}
    issues: list[dict[str, str]] = []
    for prim in rigid_prims:
        ancestor = prim.GetParent()
        rigid_ancestor = None
        while ancestor and ancestor.IsValid():
            if str(ancestor.GetPath()) in rigid_paths:
                rigid_ancestor = str(ancestor.GetPath())
                break
            ancestor = ancestor.GetParent()
        if rigid_ancestor and not UsdGeom.Xformable(prim).GetResetXformStack():
            issues.append(
                {
                    "body_path": str(prim.GetPath()),
                    "rigid_ancestor_path": rigid_ancestor,
                    "problem": "missing resetXformStack",
                }
            )
    return issues


def _repair_nested_rigid_body_xforms(
    stage: Any, issues: list[dict[str, str]]
) -> dict[str, Any]:
    """Apply a non-persistent, world-pose-preserving Xform-stack repair.

    Every world matrix is cached before any edit.  Each affected rigid body is
    then given that same matrix as an independent transform and an explicit
    reset marker.  Joint local frames remain untouched.  The referenced USD on
    disk is never modified; opinions live only in the current stage layer.
    """

    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    cached = {
        item["body_path"]: cache.GetLocalToWorldTransform(
            stage.GetPrimAtPath(item["body_path"])
        )
        for item in issues
    }
    before = {path: _matrix_numpy(matrix) for path, matrix in cached.items()}

    session_layer = stage.GetSessionLayer()
    with Usd.EditContext(stage, Usd.EditTarget(session_layer)):
        # Do not wrap these high-level Usd edits in Sdf.ChangeBlock: the next
        # AddXformOp call must see the attribute just composed by the prior
        # call, which a raw Sdf change block deliberately defers.
        for path, world_matrix in cached.items():
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            repair_op = xform.AddTransformOp(
                precision=UsdGeom.XformOp.PrecisionDouble,
                opSuffix="b601PhysxRepair",
            )
            repair_op.Set(world_matrix)
            # One exact session-layer order masks the referenced
            # translate/orient/scale order and prefixes !resetXformStack!.
            xform.SetXformOpOrder([repair_op], resetXformStack=True)

    verified_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    after = {
        path: _matrix_numpy(
            verified_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        )
        for path in cached
    }
    max_world_matrix_error = max(
        (float(np.max(np.abs(after[path] - before[path]))) for path in cached),
        default=0.0,
    )
    remaining = _nested_rigid_body_issues(stage)
    return {
        "applied": True,
        "persistent": False,
        "edit_layer": session_layer.identifier,
        "repaired_body_paths": sorted(cached),
        "repaired_count": len(cached),
        "remaining_issue_count": len(remaining),
        "remaining_issues": remaining,
        "max_initial_world_matrix_error": max_world_matrix_error,
        "method": (
            "cache initial local-to-world matrix; author independent matrix "
            "with resetXformStack in current stage layer"
        ),
    }


def _run_probe(args: argparse.Namespace, report: Report, sim_app: Any) -> None:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from pxr import PhysxSchema, Usd

    report.require(
        "shipped DM USD exists",
        ASSET_PATH.is_file(),
        asset_path=str(ASSET_PATH),
    )
    report.data["asset"] = {
        "path": str(ASSET_PATH),
        "sha256_root_layer": hashlib.sha256(ASSET_PATH.read_bytes()).hexdigest(),
        "reference_prim_path": ROBOT_PRIM_PATH,
        "authored_articulation_root_path": ARTICULATION_ROOT_PATH,
    }

    world = World(
        physics_dt=PHYSICS_DT,
        rendering_dt=1.0 / 60.0,
        stage_units_in_meters=1.0,
        backend="numpy",
    )
    add_reference_to_stage(str(ASSET_PATH), ROBOT_PRIM_PATH)
    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    articulation_root_prim = stage.GetPrimAtPath(ARTICULATION_ROOT_PATH)
    report.require(
        "referenced USD prims resolve",
        bool(root_prim.IsValid() and articulation_root_prim.IsValid()),
        root_valid=bool(root_prim.IsValid()),
        articulation_root_valid=bool(articulation_root_prim.IsValid()),
    )

    report.data["authored_drives"] = _authored_drive_state(stage)

    nested_issues = _nested_rigid_body_issues(stage)
    report.data["asset_compatibility"] = {
        "shipped_usd_usable_unmodified": not nested_issues,
        "nested_rigid_body_issue_count": len(nested_issues),
        "nested_rigid_body_issues": nested_issues,
        "session_repair_requested": bool(args.repair_nested_xforms),
    }
    if nested_issues and not args.repair_nested_xforms:
        report.require(
            "nested rigid-body Xform stacks are PhysX-valid",
            False,
            issue_count=len(nested_issues),
            hint="rerun with --repair-nested-xforms for a non-persistent session repair",
        )
    if nested_issues:
        repair = _repair_nested_rigid_body_xforms(stage, nested_issues)
        report.data["asset_compatibility"]["session_repair"] = repair
        report.require(
            "session repair preserves poses and resolves nested Xform stacks",
            repair["repaired_count"] == len(nested_issues)
            and repair["remaining_issue_count"] == 0
            and repair["max_initial_world_matrix_error"] <= 1.0e-10,
            repaired_count=repair["repaired_count"],
            expected_count=len(nested_issues),
            remaining_issue_count=repair["remaining_issue_count"],
            max_initial_world_matrix_error=repair[
                "max_initial_world_matrix_error"
            ],
        )

    # The asset disables self-collision for Newton only. PhysX's schema default
    # is true, and these imported whole-link convex hulls overlap/fight at valid
    # arm and gripper configurations. Disable articulation self-collision in the
    # session layer; collisions with external scene objects remain enabled.
    had_physx_articulation_api = articulation_root_prim.HasAPI(
        PhysxSchema.PhysxArticulationAPI
    )
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(
            articulation_root_prim
        )
        initial_self_collision = (
            physx_articulation.GetEnabledSelfCollisionsAttr().Get()
        )
        initial_position_iterations = (
            physx_articulation.GetSolverPositionIterationCountAttr().Get()
        )
        initial_velocity_iterations = (
            physx_articulation.GetSolverVelocityIterationCountAttr().Get()
        )
        physx_articulation.CreateEnabledSelfCollisionsAttr(False)
        # The schema's single velocity iteration leaves a small non-decaying
        # constraint velocity under gravity. Four is a modest robotics setting;
        # keep the already-strong default of 32 position iterations.
        physx_articulation.CreateSolverVelocityIterationCountAttr(4)
    effective_self_collision = (
        PhysxSchema.PhysxArticulationAPI(articulation_root_prim)
        .GetEnabledSelfCollisionsAttr()
        .Get()
    )
    effective_velocity_iterations = (
        PhysxSchema.PhysxArticulationAPI(articulation_root_prim)
        .GetSolverVelocityIterationCountAttr()
        .Get()
    )
    report.data["physx_articulation_config"] = {
        "api_authored_by_asset": bool(had_physx_articulation_api),
        "initial_enabled_self_collisions": bool(initial_self_collision),
        "effective_enabled_self_collisions": bool(effective_self_collision),
        "solver_position_iterations": int(initial_position_iterations),
        "initial_solver_velocity_iterations": int(initial_velocity_iterations),
        "effective_solver_velocity_iterations": int(
            effective_velocity_iterations
        ),
        "session_override": True,
        "reason": (
            "asset authors newton:selfCollisionEnabled=0 but no PhysX "
            "equivalent; coarse imported convex hulls overlap"
        ),
        "external_object_contacts_remain_enabled": True,
    }
    report.require(
        "PhysX articulation self-collision is disabled",
        effective_self_collision is False,
        initial_enabled_self_collisions=initial_self_collision,
        effective_enabled_self_collisions=effective_self_collision,
    )
    report.require(
        "PhysX articulation velocity solver uses four iterations",
        int(effective_velocity_iterations) == 4,
        initial=int(initial_velocity_iterations),
        effective=int(effective_velocity_iterations),
    )

    articulation = world.scene.add(
        SingleArticulation(
            prim_path=ROBOT_PRIM_PATH,
            name="b601_dm_probe",
        )
    )
    world.reset()
    if not articulation.handles_initialized:
        articulation.initialize()

    physics_scene_path = SimulationManager.get_default_physics_scene()
    physics_scene_prim = stage.GetPrimAtPath(physics_scene_path)
    has_physx_scene_api = bool(
        physics_scene_prim.IsValid()
        and physics_scene_prim.HasAPI(PhysxSchema.PhysxSceneAPI)
    )
    report.data["physics"] = {
        "engine": "PhysX" if has_physx_scene_api else "unknown",
        "scene_prim_path": physics_scene_path,
        "physics_dt_s": PHYSICS_DT,
        "ramp_seconds": RAMP_SECONDS,
        "settle_seconds": SETTLE_SECONDS,
        "control": "runtime PD gains + ArticulationAction position targets",
        "state_teleport_used": False,
    }
    report.require(
        "PhysX is active",
        has_physx_scene_api,
        physics_scene_path=physics_scene_path,
        has_physx_scene_api=has_physx_scene_api,
    )

    dof_names = list(articulation.dof_names)
    properties = articulation.dof_properties
    lower = np.asarray(properties["lower"], dtype=np.float64)
    upper = np.asarray(properties["upper"], dtype=np.float64)
    # Isaac Sim 5.1 SingleArticulation.dof_properties broadcasts the first
    # type over every DOF (its implementation indexes get_dof_types()[0]).
    # Query the articulation view directly so the prismatic fingers remain
    # distinguishable from the six rotational joints.
    legacy_property_types = np.asarray(properties["type"], dtype=np.int64)
    dof_types = np.asarray(
        [int(value) for value in articulation._articulation_view.get_dof_types()],
        dtype=np.int64,
    )
    max_effort = np.asarray(properties["maxEffort"], dtype=np.float64)
    initial_kp = np.asarray(properties["stiffness"], dtype=np.float64)
    initial_kd = np.asarray(properties["damping"], dtype=np.float64)
    report.data["dofs"] = {
        "names": dof_names,
        "count": articulation.num_dof,
        "types": dof_types,
        "type_encoding": {"rotation": 0, "translation": 1},
        "legacy_dof_properties_types": legacy_property_types,
        "legacy_type_note": (
            "Isaac Sim 5.1 SingleArticulation.dof_properties broadcasts the "
            "first type; validation uses Articulation.get_dof_types directly"
        ),
        "lower_rad_or_m": lower,
        "upper_rad_or_m": upper,
        "max_effort_nm_or_n": max_effort,
        "initial_runtime_kp": initial_kp,
        "initial_runtime_kd": initial_kd,
    }
    report.require(
        "exact eight-DOF name order",
        articulation.num_dof == 8 and dof_names == EXPECTED_DOF_NAMES,
        expected=EXPECTED_DOF_NAMES,
        actual=dof_names,
    )
    report.require(
        "six revolute then two prismatic DOFs",
        np.array_equal(dof_types, np.array([0] * 6 + [1] * 2)),
        expected=[0] * 6 + [1] * 2,
        actual=dof_types,
    )
    report.require(
        "DOF lower limits match DM URDF",
        np.allclose(lower, EXPECTED_LOWER, atol=LIMIT_ATOL, rtol=0.0),
        tolerance=LIMIT_ATOL,
        expected=EXPECTED_LOWER,
        actual=lower,
        error=lower - EXPECTED_LOWER,
    )
    report.require(
        "DOF upper limits match DM URDF",
        np.allclose(upper, EXPECTED_UPPER, atol=LIMIT_ATOL, rtol=0.0),
        tolerance=LIMIT_ATOL,
        expected=EXPECTED_UPPER,
        actual=upper,
        error=upper - EXPECTED_UPPER,
    )
    report.require(
        "DOF effort caps match DM URDF",
        np.allclose(
            max_effort, EXPECTED_MAX_EFFORT, atol=EFFORT_ATOL, rtol=0.0
        ),
        tolerance=EFFORT_ATOL,
        expected=EXPECTED_MAX_EFFORT,
        actual=max_effort,
    )

    view = articulation._articulation_view
    body_names = list(view.body_names)
    required_bodies = {"base_link", "gripper_left", "gripper_right"}
    report.data["bodies"] = {"names": body_names, "count": len(body_names)}
    report.require(
        "base and finger links are exposed",
        required_bodies.issubset(body_names),
        required=sorted(required_bodies),
        actual=body_names,
    )

    controller = articulation.get_articulation_controller()
    controller.set_gains(kps=RUNTIME_KP, kds=RUNTIME_KD, save_to_usd=False)
    applied_kp, applied_kd = controller.get_gains()
    applied_kp = np.asarray(applied_kp, dtype=np.float64)
    applied_kd = np.asarray(applied_kd, dtype=np.float64)
    report.data["runtime_gains"] = {
        "kp": applied_kp,
        "kd": applied_kd,
        "saved_to_usd": False,
        "source_note": (
            "starting values from neighboring RS asset; DM acceptance is based "
            "on measured tracking in this probe"
        ),
    }
    report.require(
        "positive runtime PD gains applied",
        np.allclose(applied_kp, RUNTIME_KP, atol=1.0e-5, rtol=1.0e-6)
        and np.allclose(applied_kd, RUNTIME_KD, atol=1.0e-5, rtol=1.0e-6)
        and np.all(applied_kp > 0.0)
        and np.all(applied_kd > 0.0),
        requested_kp=RUNTIME_KP,
        requested_kd=RUNTIME_KD,
        applied_kp=applied_kp,
        applied_kd=applied_kd,
    )

    monitor = StateMonitor(articulation)
    initial = monitor.sample("initial")
    initial_positions = np.asarray(initial["joint_positions"], dtype=np.float64)

    from isaacsim.core.utils.types import ArticulationAction

    articulation.apply_action(
        ArticulationAction(
            joint_positions=initial_positions,
            joint_indices=np.arange(8, dtype=np.int64),
        )
    )
    for warmup_step in range(12):
        world.step(render=args.headful)
        monitor.sample(f"warmup:{warmup_step + 1}")

    safe_target = np.concatenate((SAFE_ARM_TARGET, np.array([0.0, 0.0])))
    safe_state = _step_target(
        world,
        articulation,
        monitor,
        np.asarray(articulation.get_joint_positions(), dtype=np.float64),
        safe_target,
        label="safe_arm",
        render=args.headful,
    )
    safe_positions = np.asarray(safe_state["joint_positions"], dtype=np.float64)
    safe_velocities = np.asarray(safe_state["joint_velocities"], dtype=np.float64)
    arm_error = safe_positions[:6] - SAFE_ARM_TARGET
    report.data["safe_arm"] = {
        "target_rad": SAFE_ARM_TARGET,
        "measured_rad": safe_positions[:6],
        "error_rad": arm_error,
        "velocity_rad_s": safe_velocities[:6],
        "settle_tail": safe_state["settle_tail"],
    }
    report.check(
        "safe arm target tracks through physics drives",
        np.max(np.abs(arm_error)) <= ARM_TRACKING_TOL_RAD,
        tolerance_rad=ARM_TRACKING_TOL_RAD,
        max_abs_error_rad=float(np.max(np.abs(arm_error))),
    )
    arm_tail = safe_state["settle_tail"]
    arm_tail_span = float(
        np.max(np.asarray(arm_tail["position_span"], dtype=np.float64)[:6])
    )
    arm_tail_net = float(
        np.max(
            np.abs(
                np.asarray(
                    arm_tail["position_net_change"], dtype=np.float64
                )[:6]
            )
        )
    )
    report.check(
        "safe arm settles by measured position",
        arm_tail_span <= SETTLED_ARM_POSITION_TOL_RAD
        and arm_tail_net <= SETTLED_ARM_POSITION_TOL_RAD,
        tolerance_rad=SETTLED_ARM_POSITION_TOL_RAD,
        tail_duration_s=SETTLE_TAIL_SECONDS,
        max_position_span_rad=arm_tail_span,
        max_net_change_rad=arm_tail_net,
        max_abs_reported_velocity_rad_s=float(
            np.max(np.abs(safe_velocities[:6]))
        ),
        velocity_note=(
            "reported solver velocity is diagnostic; the pass gate uses "
            "direct joint-position motion over the tail window"
        ),
    )

    gripper_levels = [0.0, 0.5 * EXPECTED_UPPER[6], EXPECTED_UPPER[6]]
    gripper_results: list[dict[str, Any]] = []
    for name, level in zip(("zero", "mid", "max"), gripper_levels):
        start = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
        target = safe_target.copy()
        target[6:] = level
        state = _step_target(
            world,
            articulation,
            monitor,
            start,
            target,
            label=f"gripper_{name}",
            render=args.headful,
        )
        measured = np.asarray(state["joint_positions"], dtype=np.float64)
        velocities = np.asarray(state["joint_velocities"], dtype=np.float64)
        gripper_results.append(
            {
                "name": name,
                "target_each_m": level,
                "measured_m": measured[6:],
                "error_m": measured[6:] - level,
                "velocity_m_s": velocities[6:],
                "finger_symmetry_error_m": float(abs(measured[6] - measured[7])),
                "left_link_position_m": state["left_link_position_m"],
                "right_link_position_m": state["right_link_position_m"],
                "link_origin_separation_m": state["link_origin_separation_m"],
                "settle_tail": state["settle_tail"],
            }
        )

    report.data["gripper_sweep"] = gripper_results
    gripper_max_error = max(
        float(np.max(np.abs(np.asarray(item["error_m"]))))
        for item in gripper_results
    )
    gripper_max_position_span = max(
        float(
            np.max(
                np.asarray(item["settle_tail"]["position_span"])[6:]
            )
        )
        for item in gripper_results
    )
    gripper_max_net_change = max(
        float(
            np.max(
                np.abs(
                    np.asarray(
                        item["settle_tail"]["position_net_change"]
                    )[6:]
                )
            )
        )
        for item in gripper_results
    )
    gripper_max_asymmetry = max(
        float(item["finger_symmetry_error_m"]) for item in gripper_results
    )
    report.check(
        "gripper zero/mid/max targets track through physics drives",
        gripper_max_error <= GRIPPER_TRACKING_TOL_M,
        tolerance_m=GRIPPER_TRACKING_TOL_M,
        max_abs_error_m=gripper_max_error,
    )
    report.check(
        "gripper settles by measured position at zero/mid/max",
        gripper_max_position_span <= SETTLED_GRIPPER_POSITION_TOL_M
        and gripper_max_net_change <= SETTLED_GRIPPER_POSITION_TOL_M,
        tolerance_m=SETTLED_GRIPPER_POSITION_TOL_M,
        tail_duration_s=SETTLE_TAIL_SECONDS,
        max_position_span_m=gripper_max_position_span,
        max_net_change_m=gripper_max_net_change,
    )
    report.check(
        "finger joint coordinates remain symmetric",
        gripper_max_asymmetry <= FINGER_SYMMETRY_TOL_M,
        tolerance_m=FINGER_SYMMETRY_TOL_M,
        max_asymmetry_m=gripper_max_asymmetry,
    )

    separations = np.array(
        [item["link_origin_separation_m"] for item in gripper_results],
        dtype=np.float64,
    )
    mean_joint_positions = np.array(
        [np.mean(item["measured_m"]) for item in gripper_results],
        dtype=np.float64,
    )
    separation_deltas = separations - separations[0]
    expected_deltas = 2.0 * (mean_joint_positions - mean_joint_positions[0])
    separation_residual = separation_deltas - expected_deltas
    report.data["gripper_separation_model"] = {
        "separation_m": separations,
        "measured_delta_m": separation_deltas,
        "expected_two_finger_delta_m": expected_deltas,
        "residual_m": separation_residual,
    }
    report.check(
        "finger link separation increases zero to mid to max",
        bool(np.all(np.diff(separations) > 1.0e-3)),
        separation_m=separations,
    )
    report.check(
        "finger link separation matches opposed joint travel",
        float(np.max(np.abs(separation_residual))) <= SEPARATION_MODEL_TOL_M,
        tolerance_m=SEPARATION_MODEL_TOL_M,
        max_abs_residual_m=float(np.max(np.abs(separation_residual))),
    )

    report.data["state_monitor"] = {
        "samples": monitor.samples,
        "all_samples_finite": True,
        "max_base_position_drift_m": monitor.max_base_position_drift_m,
        "max_base_angle_drift_rad": monitor.max_base_angle_drift_rad,
        "measurement_api": (
            "omni.physics.tensors.ArticulationView.get_link_transforms"
        ),
    }
    report.check(
        "all sampled articulation states are finite",
        True,
        samples=monitor.samples,
    )
    report.check(
        "fixed base remains position-stable",
        monitor.max_base_position_drift_m <= BASE_POSITION_DRIFT_TOL_M,
        tolerance_m=BASE_POSITION_DRIFT_TOL_M,
        max_drift_m=monitor.max_base_position_drift_m,
    )
    report.check(
        "fixed base remains orientation-stable",
        monitor.max_base_angle_drift_rad <= BASE_ANGLE_DRIFT_TOL_RAD,
        tolerance_rad=BASE_ANGLE_DRIFT_TOL_RAD,
        max_drift_rad=monitor.max_base_angle_drift_rad,
    )

    # Turn non-fatal metric checks into the probe's nonzero exit status.
    failed = [check["name"] for check in report.data["checks"] if not check["passed"]]
    if failed:
        raise ProbeFailure("failed metric checks: " + ", ".join(failed))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "b601_asset_probe.json",
        help="JSON report path (default: artifacts/b601_asset_probe.json)",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="show the Isaac Sim viewport while running the same probe",
    )
    parser.add_argument(
        "--repair-nested-xforms",
        action="store_true",
        help=(
            "apply a non-persistent, world-pose-preserving resetXformStack "
            "repair required by the shipped DM USD under PhysX 5.1"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = Report(output_path=args.out.expanduser().resolve())
    sim_app = None
    try:
        isaac_version = importlib.metadata.version("isaacsim")
        report.data["isaac_sim_version"] = isaac_version
        report.require(
            "Isaac Sim version is 5.1",
            isaac_version.startswith("5.1."),
            actual=isaac_version,
        )

        from isaacsim import SimulationApp

        sim_app = SimulationApp({"headless": not args.headful})
        _run_probe(args, report, sim_app)
    except Exception as exc:  # report failures even if Kit/physics initialization fails
        report.error(
            f"{type(exc).__name__}: {exc}",
            trace=traceback.format_exc(),
        )
    finally:
        # Isaac Sim 5.1 fast shutdown terminates with status 0, masking a failed
        # probe, while full extension shutdown can hang for minutes. Persist the
        # verdict, flush it, then terminate this standalone process explicitly
        # with the verdict status. The OS releases the isolated Kit resources.
        report.data["process_exit_strategy"] = (
            "persist report then os._exit(verdict); avoids Kit fast-shutdown "
            "status override and slow full-extension teardown"
        )
        report.finish()
        report.write()
        exit_code = 0 if report.data["passed"] else 1
        print(
            f"B601 asset probe {'PASS' if report.data['passed'] else 'FAIL'}: "
            f"{report.output_path}",
            flush=True,
        )
        if sim_app is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
    return 0 if report.data["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
