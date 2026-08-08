#!/usr/bin/env python3
"""M5 parity check -- runs INSIDE the rebot-jazzy-baseline container.

Executed by scripts/m5_parity_test.sh after the sim bridge (host) and the
sim-profile launch (container) are up::

    docker exec <stack> bash -lc 'source /opt/ros/jazzy/setup.bash &&
        source /work/ros2_ws/install/setup.bash &&
        python3 /work/scripts/m5_parity_check.py --out /work/artifacts/m5/parity_container.json'

Checks (design doc M5 gate: "state/command parity passes"):

1. /clock is arriving (Isaac Sim is the sole owner; every node here runs
   use_sim_time).
2. /joint_states (canonical, via the sim joint-state adapter) publishes the
   eight canonical joints and MATCHES the raw /isaac_joint_states stream.
3. One MoveToPose goal -- the vendor ready pose (0.30, 0, 0.30) pitch 0.7,
   solved through the vendor's own IK on its driver model (end_link,
   reBot-DevArm_fixend.urdf) and mapped to the canonical gripper_tcp
   contract (KDR-001) -- planned and executed through
   planner -> trajectory adapter -> sim-JTC shim -> Isaac articulation.
4. The sim articulation CONVERGES: measured /joint_states arm joints within
   --tol rad (default 0.02) of the planner's q_final.

Writes a JSON verdict; exit 0 iff every check passed.  The host wrapper adds
peak GPU VRAM (nvidia-smi runs host-side).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path("/work")

CANONICAL_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5",
                    "joint6", "gripper_joint1", "gripper_joint2"]
ARM_JOINTS = CANONICAL_JOINTS[:6]

#: Vendor ready pose: RebotArmEndPose.move_to_traj(x=0.3, y=0.0, z=0.3,
#: pitch=0.7) -- the SDK docstring example, solved on the vendor driver model.
READY_XYZ = (0.30, 0.0, 0.30)
READY_PITCH = 0.7

VENDOR_SDK = REPO / "src" / "reBotArm_control_py"
FIXEND_URDF = (REPO / "src" / "reBotArmController_ROS2" / "src"
               / "rebotarm_bringup" / "description" / "urdf"
               / "reBot-DevArm_fixend.urdf")
CANONICAL_URDF = REPO / "urdf" / "rebot_b601dm_canonical.urdf"


def compute_ready_goal() -> dict:
    """Vendor IK (its own solver + driver model) -> canonical TCP goal.

    The vendor commands RPY on its ``end_link`` frame; KDR-001 proved
    end_link(fixend) coincides with gripper_link(with_gripper), so the joint
    solution transfers, and the canonical-TCP pose at that solution is the
    honest MoveToPose encoding of the vendor ready pose.
    """
    import pinocchio as pin

    # Stub the SDK's parent package: its __init__ imports the CAN hardware
    # driver, meaningless here (same pattern as scripts/b601_move_to_traj.py).
    sys.path.insert(0, str(VENDOR_SDK))
    if "reBotArm_control_py" not in sys.modules:
        stub = types.ModuleType("reBotArm_control_py")
        stub.__path__ = [str(VENDOR_SDK / "reBotArm_control_py")]
        sys.modules["reBotArm_control_py"] = stub
    from reBotArm_control_py.kinematics.inverse_kinematics import (
        IKParams, solve_ik_with_retry)

    model = pin.buildModelFromUrdf(str(FIXEND_URDF))
    data = model.createData()
    frame_id = model.getFrameId("end_link")
    target = pin.SE3(pin.rpy.rpyToMatrix(0.0, READY_PITCH, 0.0),
                     np.array(READY_XYZ))
    result = solve_ik_with_retry(
        model, data, frame_id, target, np.zeros(model.nq),
        IKParams(max_iter=1000, tolerance=1e-5, step_size=0.5, damping=1e-6))
    if not result.success:
        raise RuntimeError(f"vendor IK failed: error={result.error}")
    q_ready = np.asarray(result.q, dtype=float)[:6]

    sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "rebot_planner"))
    from rebot_planner.core import ik_core
    kin = ik_core.KinematicsCore(str(CANONICAL_URDF))
    T_tcp = kin.fk_tcp(q_ready)
    quaternion = pin.Quaternion(T_tcp[:3, :3])
    quaternion.normalize()
    return {
        "vendor_pose": {"xyz": list(READY_XYZ), "rpy": [0.0, READY_PITCH, 0.0],
                        "frame": "end_link (driver model)"},
        "vendor_q_ready_rad": [float(v) for v in q_ready],
        "tcp_position_m": [float(v) for v in T_tcp[:3, 3]],
        "tcp_quat_xyzw": [float(quaternion.x), float(quaternion.y),
                          float(quaternion.z), float(quaternion.w)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts" / "m5"
                        / "parity_container.json")
    parser.add_argument("--tol", type=float, default=0.02,
                        help="arm convergence gate, rad")
    parser.add_argument("--state-match-tol", type=float, default=1.0e-6,
                        help="/joint_states vs raw match gate, rad; pairs "
                             "are STAMP-aligned (the adapter republishes the "
                             "raw sample's stamp), so this is float copy "
                             "tolerance, not motion slack")
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--result-timeout", type=float, default=180.0)
    args = parser.parse_args()

    verdict = {
        "probe": "m5_parity_check",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": [],
        "errors": [],
    }

    def check(name: str, passed: bool, **fields) -> bool:
        entry = {"name": name, "passed": bool(passed)}
        entry.update(fields)
        verdict["checks"].append(entry)
        print(("PASS " if passed else "FAIL ") + name, fields, flush=True)
        return bool(passed)

    def write_and_exit() -> int:
        verdict["finished_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        verdict["passed"] = (bool(verdict["checks"])
                             and all(c["passed"] for c in verdict["checks"])
                             and not verdict["errors"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(verdict, indent=2) + "\n")
        print("m5_parity_check %s -> %s"
              % ("PASS" if verdict["passed"] else "FAIL", args.out), flush=True)
        return 0 if verdict["passed"] else 1

    try:
        goal_info = compute_ready_goal()
        verdict["ready_goal"] = goal_info
        check("vendor ready pose solved and mapped to canonical TCP", True,
              q_ready=[round(v, 4) for v in goal_info["vendor_q_ready_rad"]])
    except Exception as exc:  # noqa: BLE001
        verdict["errors"].append(f"ready-goal computation: {exc}")
        return write_and_exit()

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState
    from rebot_planner_msgs.action import MoveToPose

    rclpy.init()
    node = Node("m5_parity_check")
    node.set_parameters([Parameter("use_sim_time", value=True)])

    latest = {"clock": None, "canon": None, "raw": None}
    raw_by_stamp = {}       # (sec, nanosec) -> {joint: position}
    canon_backlog = []      # (stamp key, {joint: position}) awaiting a match

    def _on_clock(msg):
        latest["clock"] = msg

    def _on_canon(msg):
        latest["canon"] = msg
        canon_backlog.append(
            ((msg.header.stamp.sec, msg.header.stamp.nanosec),
             dict(zip(msg.name, msg.position))))
        del canon_backlog[:-240]

    def _on_raw(msg):
        latest["raw"] = msg
        raw_by_stamp[(msg.header.stamp.sec, msg.header.stamp.nanosec)] = \
            dict(zip(msg.name, msg.position))
        while len(raw_by_stamp) > 480:
            raw_by_stamp.pop(next(iter(raw_by_stamp)))

    node.create_subscription(Clock, "/clock", _on_clock, 10)
    node.create_subscription(JointState, "/joint_states", _on_canon, 50)
    node.create_subscription(JointState, "/isaac_joint_states", _on_raw, 50)

    def spin_until(predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if predicate():
                return True
        return False

    try:
        # -- 1. clock + streams up ---------------------------------------
        ok = spin_until(lambda: latest["clock"] is not None,
                        args.startup_timeout)
        if not check("/clock arriving from Isaac Sim", ok):
            return write_and_exit()
        ok = spin_until(lambda: latest["raw"] is not None
                        and latest["canon"] is not None,
                        args.startup_timeout)
        if not check("/isaac_joint_states and /joint_states publishing", ok,
                     raw_up=latest["raw"] is not None,
                     canonical_up=latest["canon"] is not None):
            return write_and_exit()

        # -- 2. canonical stream shape + parity with raw ------------------
        canon = latest["canon"]
        check("canonical /joint_states carries the eight canonical joints",
              list(canon.name) == CANONICAL_JOINTS, names=list(canon.name))

        # STAMP-ALIGNED parity: the sim adapter stamps its output from the
        # raw sample it merged (joint_state_core.merge_sim), so a canonical
        # message and the raw message with the SAME stamp describe the same
        # physics step -- comparing "latest vs latest" instead would measure
        # arm motion between two sample instants, not adapter fidelity.
        pairs_checked = 0
        state_err = 0.0
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and pairs_checked < 30:
            rclpy.spin_once(node, timeout_sec=0.05)
            while canon_backlog:
                key, canon_map = canon_backlog.pop(0)
                match = raw_by_stamp.get(key)
                if match is None:
                    continue
                if any(j not in match or j not in canon_map
                       for j in ARM_JOINTS):
                    continue
                state_err = max(state_err,
                                max(abs(match[j] - canon_map[j])
                                    for j in ARM_JOINTS))
                pairs_checked += 1
        check("adapter /joint_states matches raw sim state (stamp-aligned)",
              pairs_checked >= 10 and state_err <= args.state_match_tol,
              pairs_checked=pairs_checked,
              max_abs_diff_rad=state_err,
              tolerance_rad=args.state_match_tol)
        canon_map = dict(zip(latest["canon"].name, latest["canon"].position))
        verdict["initial_arm_q_rad"] = [
            round(canon_map[j], 4) for j in ARM_JOINTS if j in canon_map]

        # -- 3. MoveToPose to the vendor ready pose -----------------------
        client = ActionClient(node, MoveToPose, "/rebot_planner/move_to_pose")
        if not check("planner action server available",
                     client.wait_for_server(timeout_sec=args.startup_timeout)):
            return write_and_exit()

        goal = MoveToPose.Goal()
        goal.target.header.frame_id = "base_link"
        px, py, pz = goal_info["tcp_position_m"]
        qx, qy, qz, qw = goal_info["tcp_quat_xyzw"]
        goal.target.pose.position.x = px
        goal.target.pose.position.y = py
        goal.target.pose.position.z = pz
        goal.target.pose.orientation.x = qx
        goal.target.pose.orientation.y = qy
        goal.target.pose.orientation.z = qz
        goal.target.pose.orientation.w = qw

        send_future = client.send_goal_async(goal)
        ok = spin_until(send_future.done, args.startup_timeout)
        handle = send_future.result() if ok else None
        if not check("MoveToPose goal accepted",
                     bool(handle and handle.accepted)):
            return write_and_exit()

        result_future = handle.get_result_async()
        wall_t0 = time.monotonic()
        ok = spin_until(result_future.done, args.result_timeout)
        if not check("MoveToPose result received", ok,
                     waited_wall_sec=round(time.monotonic() - wall_t0, 1)):
            return write_and_exit()
        move_result = result_future.result().result
        verdict["move_result"] = {
            "success": bool(move_result.success),
            "message": str(move_result.message),
            "q_final_rad": [float(v) for v in move_result.q_final],
            "planned_duration_sec": float(move_result.planned_duration_sec),
        }
        if not check("MoveToPose reports success", move_result.success,
                     message=str(move_result.message)):
            return write_and_exit()

        # -- 4. convergence: sim state vs planned q_final -----------------
        q_final = np.asarray(move_result.q_final, dtype=float)
        best = math.inf
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            canon = latest["canon"]
            if canon is None:
                continue
            canon_map = dict(zip(canon.name, canon.position))
            try:
                q_now = np.array([canon_map[j] for j in ARM_JOINTS])
            except KeyError:
                continue
            best = min(best, float(np.max(np.abs(q_now - q_final))))
            if best <= args.tol:
                break
        check("sim articulation converged to planned q_final",
              best <= args.tol, max_abs_error_rad=round(best, 5),
              tolerance_rad=args.tol)
        verdict["convergence"] = {
            "q_final_rad": [round(float(v), 4) for v in q_final],
            "max_abs_error_rad": None if math.isinf(best) else round(best, 5),
            "tolerance_rad": args.tol,
        }

        # against the vendor's own solution, informational (not a gate: two
        # different IKs may legitimately settle on close-but-distinct configs)
        vendor_q = np.asarray(goal_info["vendor_q_ready_rad"], dtype=float)
        verdict["vendor_q_delta_rad"] = round(
            float(np.max(np.abs(q_final - vendor_q))), 5)
    except Exception as exc:  # noqa: BLE001
        import traceback
        verdict["errors"].append(f"{type(exc).__name__}: {exc}")
        verdict["traceback"] = traceback.format_exc()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    return write_and_exit()


if __name__ == "__main__":
    sys.exit(main())
