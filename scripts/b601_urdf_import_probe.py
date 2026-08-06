#!/usr/bin/env python3
"""Does the B601-DM URDF actually import and work in Isaac Sim?

Everything so far has used the *shipped USD*, which needed four session repairs
before it would simulate. This probe answers the separate question: if you import
the URDF yourself with Isaac's own importer, what do you get -- and are those four
defects inherited from the URDF or introduced by Seeed's conversion?

It runs the same audit P0 ran on the USD, so the two are directly comparable:

* imports at all, with default settings a user would get
* exactly the eight expected DOFs, in order, with URDF limits and efforts
* **geometry**: are links real meshes, or did the importer substitute primitives?
* the four known USD defects, checked one by one on the import
* it simulates: command a safe pose through drives and measure tracking
* finger separation at 0 / mid / max, against the USD's measured
  0.059 / 71.445 / 142.943 mm

Run::

    TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N \\
      ~/isaaclab-venv/bin/python scripts/b601_urdf_import_probe.py \\
        --urdf src/reBotArmController_ROS2/src/rebotarm_bringup/description/urdf/reBot_B601_DM_with_gripper.urdf \\
        --out artifacts/urdf_import/with_gripper.json
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

DEFAULT_URDF = (REPO_ROOT / "src" / "reBotArmController_ROS2" / "src" /
                "rebotarm_bringup" / "description" / "urdf" /
                "reBot_B601_DM_with_gripper.urdf")

#: Measured on the shipped USD by scripts/b601_asset_probe.py (P0), for comparison.
USD_REFERENCE_SEPARATION_M = [5.897956590610169e-05, 0.07144530077542822, 0.1429430741057294]


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
            "probe": "b601_urdf_import",
            "schema_version": "1.0.0",
            "checks": [], "errors": [], "findings": [],
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def check(self, name: str, passed: bool, **f: Any) -> bool:
        e = {"name": name, "passed": bool(passed)}
        e.update({k: _jsonable(v) for k, v in f.items()})
        self.data["checks"].append(e)
        return bool(passed)

    def finding(self, text: str) -> None:
        self.data["findings"].append(text)

    def write(self, path: Path) -> None:
        self.data["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.data["passed"] = (bool(self.data["checks"])
                               and all(c["passed"] for c in self.data["checks"])
                               and not self.data["errors"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(self.data), indent=2, sort_keys=True) + "\n")


def _run(args: argparse.Namespace, report: Report) -> None:
    import b601_asset_probe as probe
    import omni.kit.commands
    from isaacsim.asset.importer.urdf import _urdf
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    urdf_path = Path(args.urdf).resolve()
    report.data["urdf"] = {
        "path": str(urdf_path),
        "exists": urdf_path.is_file(),
        "sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest() if urdf_path.is_file() else None,
    }
    if not report.check("URDF file exists", urdf_path.is_file(), path=str(urdf_path)):
        return

    world = World(physics_dt=probe.PHYSICS_DT, rendering_dt=1.0 / 60.0,
                  stage_units_in_meters=1.0, backend="numpy")

    # Default import settings -- what a user actually gets, not a tuned config.
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.distance_scale = 1.0
    if args.convex_decomp:
        import_config.convex_decomp = True
    recorded = {}
    for attr in ("merge_fixed_joints", "fix_base", "self_collision", "density",
                 "default_drive_type", "default_drive_strength",
                 "default_position_drive_damping", "convex_decomp",
                 "import_inertia_tensor", "distance_scale", "collision_from_visuals"):
        try:
            recorded[attr] = getattr(import_config, attr)
        except Exception:                                      # noqa: BLE001
            pass
    report.data["import_config"] = recorded

    t0 = time.perf_counter()
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=str(urdf_path),
        import_config=import_config, get_articulation_root=True,
    )
    import_seconds = time.perf_counter() - t0
    report.data["import"] = {"status": bool(status), "prim_path": prim_path,
                             "seconds": round(import_seconds, 2)}
    if not report.check("URDF imports without error", bool(status) and bool(prim_path),
                        prim_path=prim_path, seconds=round(import_seconds, 2)):
        return

    stage = get_current_stage()

    # ---- geometry: meshes, or substituted primitives? --------------------
    counts: dict[str, int] = {}
    mesh_points, empty_meshes, prim_shapes = 0, [], []
    for p in stage.Traverse():
        t = p.GetTypeName()
        if t:
            counts[str(t)] = counts.get(str(t), 0) + 1
        if t == "Mesh":
            pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
            n = len(pts) if pts else 0
            mesh_points += n
            if n == 0:
                empty_meshes.append(str(p.GetPath()))
        if str(t) in ("Cube", "Sphere", "Cylinder", "Capsule", "Cone"):
            prim_shapes.append(str(p.GetPath()))
    report.data["geometry"] = {
        "prim_type_counts": counts,
        "total_mesh_points": mesh_points,
        "empty_mesh_count": len(empty_meshes),
        "empty_meshes": empty_meshes[:10],
        "primitive_shape_count": len(prim_shapes),
        "primitive_shapes": prim_shapes[:10],
    }
    report.check("imported links are real meshes, not substituted primitives",
                 counts.get("Mesh", 0) > 0 and mesh_points > 0 and not empty_meshes,
                 mesh_prims=counts.get("Mesh", 0), total_points=mesh_points,
                 empty_meshes=len(empty_meshes), primitive_shapes=len(prim_shapes))

    # ---- the four USD defects, checked on the import ---------------------
    nested = probe._nested_rigid_body_issues(stage)
    report.data["defect_1_nested_xform_stacks"] = {
        "issue_count": len(nested), "issues": nested[:12],
    }
    report.check("D1: no nested rigid bodies missing resetXformStack",
                 len(nested) == 0, issue_count=len(nested))

    root_prim = stage.GetPrimAtPath(prim_path)
    self_coll = None
    if root_prim and root_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        self_coll = PhysxSchema.PhysxArticulationAPI(root_prim).GetEnabledSelfCollisionsAttr().Get()
    report.data["defect_2_self_collision"] = {"enabled_self_collisions": self_coll}
    report.check("D2: articulation self-collision is disabled",
                 self_coll is False, enabled_self_collisions=self_coll)

    drives = {}
    for p in stage.Traverse():
        for token in ("angular", "linear"):
            if UsdPhysics.DriveAPI.Get(p, token):
                d = UsdPhysics.DriveAPI(p, token)
                name = p.GetName()
                drives[name] = {
                    "type": token,
                    "stiffness": d.GetStiffnessAttr().Get(),
                    "damping": d.GetDampingAttr().Get(),
                    "maxForce": d.GetMaxForceAttr().Get(),
                    "stiffness_authored": d.GetStiffnessAttr().HasAuthoredValue(),
                    "damping_authored": d.GetDampingAttr().HasAuthoredValue(),
                }
    with_gains = [n for n, v in drives.items()
                  if (v["stiffness"] or 0) > 0 or (v["damping"] or 0) > 0]
    report.data["defect_3_drive_gains"] = {"drive_count": len(drives),
                                           "with_nonzero_gains": len(with_gains),
                                           "drives": drives}
    report.check("D3: importer authors non-zero drive stiffness/damping",
                 len(drives) > 0 and len(with_gains) == len(drives),
                 drive_count=len(drives), with_nonzero_gains=len(with_gains))

    approximations: dict[str, str] = {}
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximations[str(p.GetPath())] = str(
                UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get())
    finger_approx = {k: v for k, v in approximations.items()
                     if "gripper_left" in k or "gripper_right" in k}
    report.data["defect_4_collision_approximation"] = {
        "collider_count": len(approximations),
        "finger_colliders": finger_approx,
        "distinct_values": sorted(set(approximations.values())),
    }
    report.check("D4: finger colliders are not coarse convex hulls",
                 bool(finger_approx) and all(v != "convexHull" for v in finger_approx.values()),
                 finger_colliders=finger_approx)

    # ---- does it simulate? ----------------------------------------------
    # PhysX defaults to a single velocity iteration, which leaves a small
    # non-decaying constraint velocity under gravity -- the same setting the P0
    # probe had to raise on the shipped USD. Without it joints 2-4 creep instead
    # of settling. This is a PhysX default, not a URDF or importer defect.
    if args.solver_iterations:
        with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
            pxa = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
            pxa.CreateSolverVelocityIterationCountAttr(4)
            pxa.CreateSolverPositionIterationCountAttr(32)
        report.data["solver_override"] = {"velocity_iterations": 4,
                                          "position_iterations": 32}

    articulation = SingleArticulation(prim_path=prim_path, name="b601_urdf")
    world.scene.add(articulation)
    world.reset()

    dof_names = list(articulation.dof_names)
    report.data["dofs"] = {"names": dof_names, "count": len(dof_names)}
    ok_names = dof_names == probe.EXPECTED_DOF_NAMES
    report.check("exact eight-DOF name order matches the shipped USD",
                 ok_names, actual=dof_names, expected=probe.EXPECTED_DOF_NAMES)
    if not ok_names:
        return

    lower = np.asarray(articulation.dof_properties["lower"], dtype=np.float64)
    upper = np.asarray(articulation.dof_properties["upper"], dtype=np.float64)
    report.data["limits"] = {"lower": lower, "upper": upper}
    report.check("DOF limits match the DM URDF",
                 bool(np.allclose(lower, probe.EXPECTED_LOWER, atol=2e-4)
                      and np.allclose(upper, probe.EXPECTED_UPPER, atol=2e-4)),
                 lower=lower, upper=upper)

    articulation.get_articulation_controller().set_gains(
        kps=probe.RUNTIME_KP, kds=probe.RUNTIME_KD)
    monitor = probe.StateMonitor(articulation)

    start = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
    target = np.concatenate([probe.SAFE_ARM_TARGET, [0.0, 0.0]])
    safe_state = probe._step_target(world, articulation, monitor, start, target,
                                    label="safe_arm", render=False)
    measured = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
    arm_err = float(np.max(np.abs(measured[:6] - probe.SAFE_ARM_TARGET)))
    report.data["tracking"] = {
        "target": target, "measured": measured,
        "per_joint_error_rad": measured[:6] - probe.SAFE_ARM_TARGET,
        "max_arm_error_rad": arm_err,
        # Distinguishes a joint that is oscillating from one that has settled at
        # the wrong place: a stuck joint has near-zero tail velocity.
        "settle_tail": safe_state.get("settle_tail"),
        "joint_velocities": safe_state.get("joint_velocities"),
    }
    report.check("imported articulation tracks a safe pose through physics drives",
                 arm_err <= probe.ARM_TRACKING_TOL_RAD, max_arm_error_rad=arm_err)

    # ---- finger separation sweep, versus the shipped USD -----------------
    sweep = []
    for name, level in (("zero", 0.0), ("mid", float(probe.EXPECTED_UPPER[6]) / 2.0),
                        ("max", float(probe.EXPECTED_UPPER[6]))):
        s = np.asarray(articulation.get_joint_positions(), dtype=np.float64)
        t = np.concatenate([probe.SAFE_ARM_TARGET, [level, level]])
        state = probe._step_target(world, articulation, monitor, s, t,
                                   label=f"gripper_{name}", render=False)
        sep = float(state["link_origin_separation_m"])
        sweep.append({"name": name, "target_each_m": level, "separation_m": sep})
    seps = [x["separation_m"] for x in sweep]
    deltas = [abs(seps[i] - USD_REFERENCE_SEPARATION_M[i]) for i in range(3)]
    report.data["gripper_sweep"] = {"sweep": sweep,
                                    "usd_reference_m": USD_REFERENCE_SEPARATION_M,
                                    "abs_delta_m": deltas}
    report.check("finger separation increases zero -> mid -> max",
                 seps[0] < seps[1] < seps[2], separations_m=seps)
    report.check("finger separation agrees with the shipped USD within 2 mm",
                 max(deltas) <= 2.0e-3, separations_m=seps,
                 usd_reference_m=USD_REFERENCE_SEPARATION_M, abs_delta_m=deltas)

    report.data["state_monitor"] = {
        "samples": monitor.samples,
        "max_base_position_drift_m": monitor.max_base_position_drift_m,
        "max_base_angle_drift_rad": monitor.max_base_angle_drift_rad,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--headful", action="store_true",
                    help="open the Isaac Sim window so the imported robot is visible")
    ap.add_argument("--hold-open", type=float, default=0.0, metavar="SECONDS",
                    help="keep the window open after the audit finishes")
    ap.add_argument("--solver-iterations", action="store_true",
                    help="raise PhysX solver iterations, as P0 did for the USD")
    ap.add_argument("--convex-decomp", action="store_true",
                    help="import with convex decomposition instead of the "
                         "importer's default convexHull colliders")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "artifacts" / "urdf_import" / "import_probe.json")
    args = ap.parse_args(argv)

    report = Report()
    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": not args.headful})
    try:
        _run(args, report)
    except Exception as exc:                                   # noqa: BLE001
        import traceback
        report.data["errors"].append(f"{type(exc).__name__}: {exc}")
        report.data["traceback"] = traceback.format_exc()
    finally:
        report.write(args.out)
        if args.hold_open > 0 and args.headful:
            print(f"[urdf] holding the viewport open for {args.hold_open:.0f}s "
                  f"(Ctrl-C to exit early)", flush=True)
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
