#!/usr/bin/env python3
"""Decisive experiment: why does the arm render frozen while physics moves it?

Measures the full chain at every sample, so the break is located rather than
guessed:

    tensor API (physics truth)  ->  USD xform (what Hydra may read)  ->  pixels

Scenes (one process each):
    dm-usd   the shipped DM USD with the P0 session repair (the frozen case)
    urdf     the importer's articulation, no repair (control: same drive path,
             no session-layer transform overrides)

Fix trials (on dm-usd):
    --fix update-to-usd   set every physics->USD writeback setting BEFORE World
    --fix writeback       push tensor link transforms into the session-layer
                          repair ops before every captured frame

Run::

    TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N \\
      ~/isaaclab-venv/bin/python scripts/b601_render_experiment.py \\
        --scene dm-usd --fix none --out artifacts/render_exp/dm_none.json
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

SWEEP_RAD = 0.42          # joint2 travel, well inside [-3.14, 0] from -0.497
SWEEP_STEPS = 300         # 2.5 s
SETTLE_STEPS = 360        # 3 s with rendering, lets the RTX denoiser converge
SAMPLE_EVERY = 30
CHANGED_PX_THRESHOLD = 20  # gray levels; pixels changed vs sweep-start frame


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
    # Gf.Matrix4d / Gf.Quatd etc. -- a raw pxr object here previously blew up
    # json.dumps inside `finally`, which skipped sim_app.close() and made Kit's
    # atexit produce what looked like a segfault. Everything unknown becomes str.
    return str(v)


def dump_physics_settings() -> dict:
    """Every omni.physx SETTING_* plus the render-delegate keys, actual values."""
    import carb.settings
    settings = carb.settings.get_settings()
    out = {}
    try:
        import omni.physx.bindings._physx as pxb
        for name in dir(pxb):
            if not name.startswith("SETTING_"):
                continue
            key = getattr(pxb, name)
            if not isinstance(key, str) or not key.startswith("/"):
                continue
            try:
                out[key] = settings.get(key)
            except Exception:                                  # noqa: BLE001
                out[key] = "<unreadable>"
    except Exception as exc:                                   # noqa: BLE001
        out["_physx_bindings_error"] = str(exc)
    for key in ("/app/useFabricSceneDelegate", "/app/renderer/skipWhileMinimized",
                "/physics/fabricEnabled", "/physics/updateToUsd",
                "/physics/updateVelocitiesToUsd", "/physics/fabricUpdateTransformations"):
        try:
            out.setdefault(key, settings.get(key))
        except Exception:                                      # noqa: BLE001
            pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=["dm-usd", "urdf"], required=True)
    ap.add_argument("--fix", choices=["none", "update-to-usd", "writeback",
                                      "native-ops", "force-drives"],
                    default="none")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    frames_dir = args.frames_dir or args.out.with_suffix("")
    frames_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"scene": args.scene, "fix": args.fix, "phases": []}

    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": True})
    try:
        import cv2
        import carb.settings
        import b601_asset_probe as probe
        import omni.replicator.core as rep
        import isaacsim.core.utils.prims as prim_utils
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, GroundPlane
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import Gf, PhysxSchema, Usd, UsdGeom

        # -- fix trial 1: writeback settings BEFORE any physics exists ------
        if args.fix == "update-to-usd":
            s = carb.settings.get_settings()
            applied = {}
            for key in ("/physics/updateToUsd", "/physics/updateVelocitiesToUsd",
                        "/physics/outputVelocitiesLocalSpace",
                        "/physics/fabricUpdateTransformations"):
                try:
                    s.set_bool(key, True)
                    applied[key] = True
                except Exception as exc:                       # noqa: BLE001
                    applied[key] = f"failed: {exc}"
            # And try turning fabric OFF entirely, which forces classic USD flow.
            for key in ("/physics/fabricEnabled",):
                try:
                    s.set_bool(key, False)
                    applied[key] = False
                except Exception as exc:                       # noqa: BLE001
                    applied[key] = f"failed: {exc}"
            result["fix_settings_applied"] = applied

        world = World(physics_dt=probe.PHYSICS_DT, rendering_dt=1.0 / 60.0,
                      stage_units_in_meters=1.0, backend="numpy")
        stage = get_current_stage()

        repair_body_paths: list[str] = []
        if args.scene == "dm-usd":
            add_reference_to_stage(str(probe.ASSET_PATH), probe.ROBOT_PRIM_PATH)
            issues = probe._nested_rigid_body_issues(stage)
            repair_body_paths = [i["body_path"] for i in issues]
            if issues and args.fix != "native-ops":
                probe._repair_nested_rigid_body_xforms(stage, issues)
            elif issues:
                # Fix B: same pose-preserving reset, but authored as the
                # translate/orient/scale ops PhysX's updateToUsd writes into --
                # so the writeback lands INSIDE the composed stack and the
                # viewport stays live with no per-frame help.
                cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                cached = {i["body_path"]: cache.GetLocalToWorldTransform(
                    stage.GetPrimAtPath(i["body_path"])) for i in issues}
                with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
                    for path, mtx in cached.items():
                        prim = stage.GetPrimAtPath(path)
                        xf = UsdGeom.Xformable(prim)
                        t = mtx.ExtractTranslation()
                        q = mtx.ExtractRotationQuat()
                        xf.ClearXformOpOrder()

                        def _prec(attr_name, dbl, flt):
                            a = prim.GetAttribute(attr_name)
                            tn = str(a.GetTypeName()) if a and a.IsValid() else ""
                            return (UsdGeom.XformOp.PrecisionDouble, dbl) \
                                if "double" in tn or not tn else \
                                (UsdGeom.XformOp.PrecisionFloat, flt)

                        prec_t, vec_t = _prec("xformOp:translate", Gf.Vec3d, Gf.Vec3f)
                        prec_o, quat_t = _prec("xformOp:orient", Gf.Quatd, Gf.Quatf)
                        prec_s, vec_s = _prec("xformOp:scale", Gf.Vec3d, Gf.Vec3f)
                        op_t = xf.AddTranslateOp(prec_t)
                        op_o = xf.AddOrientOp(prec_o)
                        op_s = xf.AddScaleOp(prec_s)
                        op_t.Set(vec_t(float(t[0]), float(t[1]), float(t[2])))
                        im = q.GetImaginary()
                        op_o.Set(quat_t(float(q.GetReal()), float(im[0]),
                                        float(im[1]), float(im[2])))
                        op_s.Set(vec_s(1.0, 1.0, 1.0))
                        xf.SetResetXformStack(True)
                result["native_ops_repaired"] = len(cached)
            root_path = probe.ARTICULATION_ROOT_PATH
        else:
            import omni.kit.commands
            _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
            cfg.merge_fixed_joints = False
            cfg.fix_base = True
            cfg.make_default_prim = True
            cfg.distance_scale = 1.0
            cfg.convex_decomp = True
            urdf = (REPO_ROOT / "src" / "reBotArmController_ROS2" / "src" /
                    "rebotarm_bringup" / "description" / "urdf" /
                    "reBot_B601_DM_with_gripper.urdf")
            _, root_path = omni.kit.commands.execute(
                "URDFParseAndImportFile", urdf_path=str(urdf),
                import_config=cfg, get_articulation_root=True)

        with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
            pxa = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root_path))
            pxa.CreateEnabledSelfCollisionsAttr(False)
            pxa.CreateSolverVelocityIterationCountAttr(4)

        if args.fix == "force-drives":
            # The importer authors drive type "acceleration" with the URDF
            # effort as maxForce; the hand-authored USD uses type "force" and
            # tracks. Flip the imported drives to force and compare.
            from pxr import UsdPhysics
            flipped = []
            with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
                for pr in stage.Traverse():
                    for token in ("angular", "linear"):
                        drv = UsdPhysics.DriveAPI.Get(pr, token)
                        if drv and drv.GetTypeAttr().Get() == "acceleration":
                            drv.GetTypeAttr().Set("force")
                            flipped.append(pr.GetName())
            result["force_drives_flipped"] = flipped

        world.scene.add(GroundPlane(prim_path="/World/ground", size=4.0))
        prim_utils.create_prim("/World/dome_light", "DomeLight",
                               attributes={"inputs:intensity": 900.0})
        prim_utils.create_prim("/World/key_light", "DistantLight",
                               attributes={"inputs:intensity": 2500.0})
        # A falling control body: rigid-body render behaviour measured in the
        # SAME frames as the articulation.
        control = world.scene.add(DynamicCuboid(
            prim_path="/World/control_cube", name="control_cube",
            position=np.array([0.45, -0.25, 0.60]),
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([0.9, 0.2, 0.15]), mass=0.05))

        prim_utils.create_prim("/World/cam", "Camera")
        cam = stage.GetPrimAtPath("/World/cam")
        cam.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 1000.0))
        eye = np.array([0.95, -0.95, 0.65])
        tgt = np.array([0.18, 0.02, 0.22])
        fwd = tgt - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.column_stack([right, down, fwd]) @ np.diag([1.0, -1.0, -1.0])
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()
        m = Gf.Matrix4d(); m.SetIdentity()
        m.SetRotateOnly(Gf.Matrix3d(*[float(v) for v in R.T.flatten()]))
        m.SetTranslateOnly(Gf.Vec3d(*eye))
        xf.AddTransformOp().Set(m)
        rp = rep.create.render_product("/World/cam", (960, 540))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach(rp)

        art = SingleArticulation(prim_path=root_path, name="b601")
        world.scene.add(art)
        world.reset()
        art.get_articulation_controller().set_gains(kps=probe.RUNTIME_KP,
                                                     kds=probe.RUNTIME_KD)
        monitor = probe.StateMonitor(art)
        idx = np.arange(8, dtype=np.int64)

        result["settings"] = dump_physics_settings()

        # Locate USD prims for the three-way readback.
        def find_prim_named(name: str):
            for p in stage.Traverse():
                if p.GetName() == name and (p.HasAPI(UsdGeom.Xformable) or p.IsA(UsdGeom.Xformable)):
                    return p
            return None

        gl_prim = find_prim_named("gripper_left")
        cube_prim = stage.GetPrimAtPath("/World/control_cube")

        def usd_world_pos(prim) -> list | None:
            if prim is None or not prim.IsValid():
                return None
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            mtx = cache.GetLocalToWorldTransform(prim)
            t = mtx.ExtractTranslation()
            return [float(t[0]), float(t[1]), float(t[2])]

        # Writeback machinery (fix trial 2): tensor link transforms -> session ops.
        writeback_map = []
        if args.fix == "writeback" and repair_body_paths:
            from grasp_smoke.geometry import rotation_from_quaternion
            for path in repair_body_paths:
                prim = stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    continue
                ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
                top = [o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTransform]
                name = path.rsplit("/", 1)[-1]
                if top and name in monitor.body_names:
                    writeback_map.append((monitor.body_names.index(name), top[0]))
            result["writeback_links"] = len(writeback_map)

            def push_transforms():
                from pxr import Sdf
                links = monitor._link_transforms()
                with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())), \
                        Sdf.ChangeBlock():
                    for li, op in writeback_map:
                        p = links[li, :3]
                        q = links[li, 3:]
                        Rl = rotation_from_quaternion(np.asarray(q, dtype=np.float64))
                        gm = Gf.Matrix4d()
                        gm.SetIdentity()
                        gm.SetRotateOnly(Gf.Matrix3d(*[float(v) for v in Rl.T.flatten()]))
                        gm.SetTranslateOnly(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                        op.Set(gm)
        else:
            def push_transforms():
                return None

        def grab_gray():
            a = annot.get_data()
            if a is None:
                return None
            a = np.asarray(a)
            if a.ndim != 3 or a.shape[0] == 0:
                return None
            return cv2.cvtColor(a[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)

        def run_sweep(tag: str) -> dict:
            """Sweep joint2, sampling tensor vs USD vs pixels."""
            start = np.asarray(art.get_joint_positions(), dtype=np.float64)
            base_q = start.copy()
            # settle + converge the denoiser at the sweep start pose
            for si in range(SETTLE_STEPS):
                art.apply_action(ArticulationAction(joint_positions=base_q,
                                                    joint_indices=idx))
                if si % SAMPLE_EVERY == 0:
                    push_transforms()      # before the render inside world.step
                world.step(render=True)
            baseline = grab_gray()
            samples = []
            frames_saved = 0
            for k in range(SWEEP_STEPS):
                q = base_q.copy()
                # joint2's range is [-3.14, 0]; from the post-reset zero pose the
                # sweep must go NEGATIVE. (A positive sweep clips to nothing --
                # the first run of this experiment measured exactly that null.)
                q[1] = base_q[1] - SWEEP_RAD * (k + 1) / SWEEP_STEPS
                q[1] = float(np.clip(q[1], -3.13, -0.001))
                art.apply_action(ArticulationAction(joint_positions=q,
                                                    joint_indices=idx))
                if (k + 1) % SAMPLE_EVERY == 0:
                    push_transforms()      # sync USD to physics, then render
                world.step(render=True)
                if (k + 1) % SAMPLE_EVERY == 0:
                    state = monitor.sample(f"{tag}:{k}")
                    jaw = (np.asarray(state["left_link_position_m"]) +
                           np.asarray(state["right_link_position_m"])) / 2.0
                    gray = grab_gray()
                    changed = None
                    if gray is not None and baseline is not None:
                        changed = int((np.abs(gray.astype(np.int16)
                                              - baseline.astype(np.int16))
                                       > CHANGED_PX_THRESHOLD).sum())
                        if frames_saved < 4:
                            cv2.imwrite(str(frames_dir / f"{tag}_{k:04d}.png"), gray)
                            frames_saved += 1
                    samples.append({
                        "k": k,
                        "joint2_cmd": q[1],
                        "joint2_measured": float(state["joint_positions"][1]),
                        "tensor_jaw_z": float(jaw[2]),
                        "usd_gripper_left": usd_world_pos(gl_prim),
                        "usd_control_cube": usd_world_pos(cube_prim),
                        "tensor_control_cube": _jsonable(
                            np.asarray(control.get_world_pose()[0])),
                        "changed_px_vs_start": changed,
                    })
            final_state = monitor.sample(f"{tag}:final")
            final_err = np.abs(np.asarray(final_state["joint_positions"])
                               - np.concatenate([base_q[:1], [q[1]], base_q[2:]]))
            tz = [s["tensor_jaw_z"] for s in samples]
            uz = [s["usd_gripper_left"][2] for s in samples
                  if s["usd_gripper_left"] is not None]
            cpx = [s["changed_px_vs_start"] for s in samples
                   if s["changed_px_vs_start"] is not None]
            return {
                "tag": tag,
                "samples": samples,
                "tensor_jaw_z_range_mm": (max(tz) - min(tz)) * 1000 if tz else None,
                "usd_jaw_z_range_mm": (max(uz) - min(uz)) * 1000 if uz else None,
                "changed_px_final": cpx[-1] if cpx else None,
                "changed_px_max": max(cpx) if cpx else None,
                "final_joint_error_rad": _jsonable(final_err),
            }

        result["phases"].append(run_sweep("sweep1"))

        # Mechanism forensics: what xform ops exist on a link after the sweep,
        # which are in xformOpOrder, and what do they hold? If PhysX wrote live
        # translate/orient attrs that are NOT in the op order, the freeze is
        # "writeback lands outside the composed op stack".
        def xform_forensics(path: str) -> dict:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                return {"path": path, "valid": False}
            xf = UsdGeom.Xformable(prim)
            order = [op.GetOpName() for op in xf.GetOrderedXformOps()]
            reset = xf.GetResetXformStack()
            attrs = {}
            for attr in prim.GetAttributes():
                n = attr.GetName()
                if n.startswith("xformOp"):
                    try:
                        attrs[n] = _jsonable(attr.Get())
                    except Exception:                          # noqa: BLE001
                        attrs[n] = "<unreadable>"
            return {"path": path, "valid": True, "resetXformStack": reset,
                    "xformOpOrder": order, "attributes": attrs}

        forensic_paths = []
        if repair_body_paths:
            forensic_paths = [repair_body_paths[2], repair_body_paths[-1]]
        else:
            for want in ("link3", "gripper_left"):
                for pr in stage.Traverse():
                    if pr.GetName() == want:
                        forensic_paths.append(str(pr.GetPath()))
                        break
        result["forensics"] = [xform_forensics(p) for p in forensic_paths]

        # Joint-clamp forensics: maxJointVelocity is authored in DEG/S, and a
        # clamp there presents as exactly the constant-velocity creep joint4
        # shows. Dump every clamp-relevant attr on every joint prim.
        joints = {}
        for pr in stage.Traverse():
            tn = pr.GetTypeName()
            if tn in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
                rec = {"type": str(tn)}
                for attr in pr.GetAttributes():
                    n = attr.GetName()
                    if any(w in n for w in ("maxJointVelocity", "jointFriction",
                                            "armature", "stiffness", "damping",
                                            "maxForce", "targetPosition",
                                            "lower", "upper", "type")):
                        try:
                            rec[n] = _jsonable(attr.Get())
                        except Exception:                      # noqa: BLE001
                            rec[n] = "<unreadable>"
                joints[pr.GetName()] = rec
        result["joint_forensics"] = joints

        # Mimic the pick: spawn scene objects, set default state, reset AGAIN.
        art.set_joints_default_state(
            positions=np.asarray(art.get_joint_positions(), dtype=np.float64))
        world.reset()
        art.get_articulation_controller().set_gains(kps=probe.RUNTIME_KP,
                                                     kds=probe.RUNTIME_KD)
        monitor = probe.StateMonitor(art)
        if args.fix == "writeback":
            writeback_map = [(monitor.body_names.index(
                path.rsplit("/", 1)[-1]), op) for _, op in []] or writeback_map
        result["phases"].append(run_sweep("sweep2_after_reset"))

    except Exception:                                          # noqa: BLE001
        import traceback
        result["error"] = traceback.format_exc()[-2500:]
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
        try:
            sim_app.close()
        except Exception:                                      # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
