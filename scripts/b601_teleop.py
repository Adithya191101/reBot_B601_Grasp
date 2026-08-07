#!/usr/bin/env python3
"""Pop the B601 in Isaac Sim and drive its joints from the keyboard.

Purpose is verification by eye: load the asset, move every DOF yourself, and see
whether the model articulates correctly. It also settles a question the recorded
video could not -- whether the viewport follows physics interactively.

    # the URDF, imported by Isaac's own importer (the default)
    TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N DISPLAY=:1 \\
      ~/isaaclab-venv/bin/python scripts/b601_teleop.py

    # the shipped USD instead, for an A/B against the same controls
    ... scripts/b601_teleop.py --source usd

Keys (focus the Isaac Sim window):

    1..6      select arm joint 1-6
    7         select the gripper (both fingers together)
    UP/DOWN   move the selected joint  (also: EQUAL / MINUS)
    [ / ]     smaller / larger step
    O / C     gripper fully open / fully closed
    I         toggle Cartesian IK mode (Pinocchio, the demo's own solver)
    W/S       IK: TCP +x / -x        A/D: TCP +y / -y        Q/E: TCP +z / -z
    H         return to the home pose
    R         reset the articulation
    P         print full state to the status file
    ESC       quit

Isaac redirects stdout into the Kit logger, so status goes to a file. Watch it
with::

    tail -f artifacts/teleop/status.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src" / "reBotArm_control_py"))

DEFAULT_URDF = (REPO_ROOT / "src" / "reBotArmController_ROS2" / "src" /
                "rebotarm_bringup" / "description" / "urdf" /
                "reBot_B601_DM_with_gripper.urdf")
IK_EE_FRAME = "gripper_link"

ARM_STEP_RAD = 0.05
GRIP_STEP_M = 0.005
IK_STEP_M = 0.010


class Teleop:
    def __init__(self, args, status: Path):
        self.args = args
        self.status = status
        self.selected = 0            # 0..5 arm, 6 = gripper
        self.arm_step = ARM_STEP_RAD
        self.grip_step = GRIP_STEP_M
        self.ik_step = IK_STEP_M
        self.ik_mode = False
        self.quit = False
        self.dirty = True
        self.target = None
        self.home = None
        self.model = None

    def log(self, msg: str) -> None:
        with self.status.open("a") as fh:
            fh.write(msg + "\n")

    # -- keyboard ------------------------------------------------------
    def on_key(self, event, *_) -> bool:
        import carb.input
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        K = carb.input.KeyboardInput
        k = event.input

        digits = {K.KEY_1: 0, K.KEY_2: 1, K.KEY_3: 2, K.KEY_4: 3,
                  K.KEY_5: 4, K.KEY_6: 5, K.KEY_7: 6}
        if k in digits:
            self.selected = digits[k]
            self.log(f"[select] {'gripper' if self.selected == 6 else f'joint{self.selected+1}'}")
            return True

        if k == K.ESCAPE:
            self.quit = True
        elif k == K.I:
            self.ik_mode = not self.ik_mode
            self.log(f"[mode] {'CARTESIAN IK' if self.ik_mode else 'JOINT'}")
        elif k in (K.LEFT_BRACKET, K.RIGHT_BRACKET):
            f = 0.5 if k == K.LEFT_BRACKET else 2.0
            self.arm_step = float(np.clip(self.arm_step * f, 0.002, 0.5))
            self.grip_step = float(np.clip(self.grip_step * f, 0.0005, 0.02))
            self.ik_step = float(np.clip(self.ik_step * f, 0.001, 0.10))
            self.log(f"[step] arm={self.arm_step:.3f} rad  grip={self.grip_step*1000:.1f} mm"
                     f"  ik={self.ik_step*1000:.1f} mm")
        elif k in (K.UP, K.EQUAL, K.NUMPAD_ADD):
            self._nudge(+1)
        elif k in (K.DOWN, K.MINUS, K.NUMPAD_SUBTRACT):
            self._nudge(-1)
        elif k == K.O:
            self.target[6] = self.target[7] = self.upper[6]
            self.dirty = True
        elif k == K.C:
            self.target[6] = self.target[7] = 0.0
            self.dirty = True
        elif k == K.H:
            self.target = self.home.copy()
            self.dirty = True
            self.log("[home]")
        elif k == K.R:
            self.request_reset = True
        elif k == K.P:
            self.dirty = True
        elif self.ik_mode and k in (K.W, K.S, K.A, K.D, K.Q, K.E):
            axis = {K.W: (0, +1), K.S: (0, -1), K.A: (1, +1),
                    K.D: (1, -1), K.Q: (2, +1), K.E: (2, -1)}[k]
            self._ik_nudge(axis[0], axis[1])
        return True

    def _nudge(self, sign: int) -> None:
        if self.selected == 6:
            v = float(np.clip(self.target[6] + sign * self.grip_step,
                              0.0, self.upper[6]))
            self.target[6] = self.target[7] = v
        else:
            i = self.selected
            self.target[i] = float(np.clip(self.target[i] + sign * self.arm_step,
                                           self.lower[i] + 1e-3, self.upper[i] - 1e-3))
        self.dirty = True

    def _ik_nudge(self, axis: int, sign: int) -> None:
        """Cartesian move of the TCP via the demo's Pinocchio solver."""
        if self.model is None:
            self.log("[ik] unavailable (pinocchio/URDF not loaded)")
            return
        import pinocchio as pin
        q = np.zeros(self.model.nq)
        q[:6] = self.target[:6]
        pin.forwardKinematics(self.model, self.mdata, q)
        pin.updateFramePlacements(self.model, self.mdata)
        cur = self.mdata.oMf[self.ee_id]
        goal = np.array(cur.translation, dtype=np.float64)
        goal[axis] += sign * self.ik_step
        target_se3 = pin.SE3(np.array(cur.rotation, dtype=np.float64), goal)
        seed = np.zeros(self.model.nq)
        seed[:6] = self.target[:6]
        res = self.solve_ik(self.model, self.mdata, self.ee_id, target_se3,
                            seed, self.ikp)
        if not res.success:
            self.log(f"[ik] FAILED err={res.error:.2e}")
            return
        q6 = np.asarray(res.q, dtype=np.float64)[:6]
        if not (np.all(q6 >= self.lower[:6] + 1e-3) and np.all(q6 <= self.upper[:6] - 1e-3)):
            self.log("[ik] solution outside joint limits, ignored")
            return
        self.target[:6] = q6
        self.dirty = True
        self.log(f"[ik] TCP -> {np.round(goal, 4).tolist()}")


def _load_urdf(args, report_log) -> str:
    import omni.kit.commands
    status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False
    cfg.fix_base = True
    cfg.make_default_prim = True
    cfg.distance_scale = 1.0
    # The importer defaults to convexHull colliders, which turns each finger into
    # a solid blob spanning the jaw (see PICK.md, asset defect #4).
    cfg.convex_decomp = True
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=str(args.urdf),
        import_config=cfg, get_articulation_root=True)
    # The importer authors drive type "acceleration" with the URDF effort as
    # maxForce; on the low-inertia wrist that cap starves the drive and joint4
    # settles ~0.81 rad from target with a constant creep. Flipping the drives
    # to "force" (what the hand-authored USD uses) drops the error to 0.004 rad
    # -- measured in artifacts/render_exp/urdf_force.json.
    from isaacsim.core.utils.stage import get_current_stage as _gcs
    from pxr import UsdPhysics as _UP
    _stage = _gcs()
    _flipped = 0
    for _pr in _stage.Traverse():
        for _token in ("angular", "linear"):
            _drv = _UP.DriveAPI.Get(_pr, _token)
            if _drv and _drv.GetTypeAttr().Get() == "acceleration":
                _drv.GetTypeAttr().Set("force")
                _flipped += 1
    report_log(f"[load] flipped {_flipped} drives from acceleration to force")
    report_log(f"[load] URDF {args.urdf}")
    report_log(f"[load] articulation root: {prim_path}")
    return prim_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["urdf", "usd"], default="urdf")
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--status", type=Path,
                    default=REPO_ROOT / "artifacts" / "teleop" / "status.txt")
    ap.add_argument("--headless", action="store_true",
                    help="run without a window (self-test only; no keyboard)")
    ap.add_argument("--selftest", type=float, default=0.0, metavar="SECONDS",
                    help="drive a scripted joint sweep instead of reading keys, "
                         "and report whether the render follows it")
    args = ap.parse_args(argv)

    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.status.write_text("")
    tele = Teleop(args, args.status)
    tele.request_reset = False
    tele.log(f"[start] source={args.source}  {time.strftime('%H:%M:%S')}")

    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": bool(args.headless)})

    try:
        import b601_asset_probe as probe
        import isaacsim.core.utils.prims as prim_utils
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import GroundPlane
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import PhysxSchema, Usd

        world = World(physics_dt=probe.PHYSICS_DT, rendering_dt=1.0 / 60.0,
                      stage_units_in_meters=1.0, backend="numpy")
        stage = get_current_stage()

        if args.source == "urdf":
            root_path = _load_urdf(args, tele.log)
        else:
            add_reference_to_stage(str(probe.ASSET_PATH), probe.ROBOT_PRIM_PATH)
            root_path = probe.ARTICULATION_ROOT_PATH
            issues = probe._nested_rigid_body_issues(stage)
            if issues:
                probe._repair_nested_rigid_body_xforms(stage, issues)
                tele.log(f"[load] session-repaired {len(issues)} nested Xform stacks")
            tele.log(f"[load] USD {probe.ASSET_PATH}")

        with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
            pxa = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(root_path))
            pxa.CreateEnabledSelfCollisionsAttr(False)
            pxa.CreateSolverVelocityIterationCountAttr(4)

        world.scene.add(GroundPlane(prim_path="/World/ground", size=4.0))
        prim_utils.create_prim("/World/dome_light", "DomeLight",
                               attributes={"inputs:intensity": 900.0})
        prim_utils.create_prim("/World/key_light", "DistantLight",
                               attributes={"inputs:intensity": 2500.0})

        art = SingleArticulation(prim_path=root_path, name="b601")
        world.scene.add(art)
        world.reset()
        art.get_articulation_controller().set_gains(kps=probe.RUNTIME_KP,
                                                     kds=probe.RUNTIME_KD)

        names = list(art.dof_names)
        tele.lower = np.asarray(art.dof_properties["lower"], dtype=np.float64)
        tele.upper = np.asarray(art.dof_properties["upper"], dtype=np.float64)
        tele.home = np.concatenate([probe.SAFE_ARM_TARGET, [tele.upper[6], tele.upper[6]]])
        tele.target = tele.home.copy()
        tele.log(f"[dofs] {names}")
        tele.log(f"[limits] lower={np.round(tele.lower,3).tolist()}")
        tele.log(f"[limits] upper={np.round(tele.upper,3).tolist()}")

        # Pinocchio IK, same solver the demo uses.
        try:
            import pinocchio as pin
            if "reBotArm_control_py" not in sys.modules:
                import types
                stub = types.ModuleType("reBotArm_control_py")
                stub.__path__ = [str(REPO_ROOT / "src" / "reBotArm_control_py"
                                     / "reBotArm_control_py")]
                sys.modules["reBotArm_control_py"] = stub
            from reBotArm_control_py.kinematics.inverse_kinematics import (
                IKParams, solve_ik_with_retry)
            tele.model = pin.buildModelFromUrdf(str(args.urdf))
            tele.mdata = tele.model.createData()
            tele.ee_id = tele.model.getFrameId(IK_EE_FRAME)
            tele.solve_ik = solve_ik_with_retry
            tele.ikp = IKParams(max_iter=1000, tolerance=1e-5,
                                step_size=0.5, damping=1e-6)
            tele.log(f"[ik] pinocchio ready, EE frame '{IK_EE_FRAME}'")
        except Exception as exc:                               # noqa: BLE001
            tele.log(f"[ik] unavailable: {type(exc).__name__}: {exc}")

        idx = np.arange(len(names), dtype=np.int64)
        sub = None
        if not args.headless and args.selftest <= 0:
            import carb.input
            import omni.appwindow
            appwin = omni.appwindow.get_default_app_window()
            iface = carb.input.acquire_input_interface()
            sub = iface.subscribe_to_keyboard_events(appwin.get_keyboard(), tele.on_key)
            tele.log("[keys] 1-6 joints | 7 gripper | UP/DOWN move | [ ] step | "
                     "O/C open/close | I ik | WASDQE ik-move | H home | R reset | "
                     "P print | ESC quit")

        # --- self-test: does the render follow a scripted sweep? -----------
        sweep = None
        if args.selftest > 0:
            sweep = int(args.selftest / probe.PHYSICS_DT)
            tele.log(f"[selftest] sweeping joint2 over {args.selftest}s")

        step_i = 0
        last_report = 0.0
        while sim_app.is_running() and not tele.quit:
            if tele.request_reset:
                world.reset()
                art.get_articulation_controller().set_gains(kps=probe.RUNTIME_KP,
                                                             kds=probe.RUNTIME_KD)
                tele.target = tele.home.copy()
                tele.request_reset = False
                tele.dirty = True
                tele.log("[reset]")

            if sweep is not None:
                f = (step_i % sweep) / sweep
                tele.target[1] = tele.lower[1] * 0.0 + (
                    probe.SAFE_ARM_TARGET[1] + 0.45 * np.sin(2 * np.pi * f))
                tele.target[1] = float(np.clip(tele.target[1],
                                               tele.lower[1] + 1e-3, tele.upper[1] - 1e-3))
                if step_i >= sweep * 2:
                    break

            art.apply_action(ArticulationAction(joint_positions=tele.target,
                                                joint_indices=idx))
            world.step(render=True)
            step_i += 1

            now = time.time()
            if tele.dirty or (now - last_report) > 2.0:
                meas = np.asarray(art.get_joint_positions(), dtype=np.float64)
                err = np.abs(meas - tele.target)
                sel = "gripper" if tele.selected == 6 else f"joint{tele.selected+1}"
                tele.log(
                    f"[state] mode={'IK' if tele.ik_mode else 'JOINT'} sel={sel}"
                    f"\n        target  ={np.round(tele.target,4).tolist()}"
                    f"\n        measured={np.round(meas,4).tolist()}"
                    f"\n        max|err|={err.max():.4f}")
                tele.dirty = False
                last_report = now

        if sub is not None:
            import carb.input
            carb.input.acquire_input_interface().unsubscribe_to_keyboard_events(
                appwin.get_keyboard(), sub)
        tele.log("[done]")
    except Exception as exc:                                   # noqa: BLE001
        import traceback
        tele.log("[ERROR] " + traceback.format_exc()[-2000:])
    finally:
        try:
            sim_app.close()
        except Exception:                                      # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
