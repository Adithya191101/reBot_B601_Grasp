#!/usr/bin/env python3
"""M6: drive 100 pose-to-pose cuMotion trials against the M5 sim stack.

Runs INSIDE the Isaac ROS 4.5 cuMotion container (rebot-m6-cumotion image,
repo mounted at /work, --network host, ROS_DOMAIN_ID=42, FastDDS UDP
profile) -- see scripts/m6_cumotion_trials.sh.

Per trial (targets from artifacts/m6/trial_poses.json):

  1. send an isaac_ros_cumotion_interfaces/action/MotionPlan goal to the
     STANDALONE cuMotion action server ``cumotion/motion_plan``
     (plan_pose, use_current_state, use_planning_scene + the static trial
     world boxes embedded as typed moveit CollisionObjects -- recorded M6
     decision: no MoveIt anywhere, the goal-embedded world merges with the
     static-scene objects in cumotion_planner.cpp ExecuteMotionPlan);
  2. forward the returned RobotTrajectory to the canonical
     ``/rebot_controller/follow_joint_trajectory`` (trajectory adapter ->
     sim-JTC shim -> Isaac Sim), DROPPING the leading t=0 point: it
     duplicates the live state and the shim requires strictly positive
     ``time_from_start`` (same convention as rebot_planner);
  3. wait for the FJT result and verify tracking convergence on
     /joint_states (<= CONV_TOL_RAD per joint, M5 gate value).

Candidate rejection (recorded M6 decision): cuMotion's collision-aware
IK occasionally cannot solve a random pose that ik_core's local DLS can
(measured: trajopt INVERSE_KINEMATICS_FAILURE).  A candidate failing at
the PLANNING stage leaves the arm unmoved, so it is recorded under
``rejected_candidates`` and the next spare candidate is used -- exactly
``n_trials_required`` trials are executed, and any EXECUTION or
CONVERGENCE failure still fails the gate outright.  Rejections are
capped by the spare-candidate count (130 sampled for 100 trials);
running out of candidates fails the run.  (Pre-vetting via the
``cumotion/ik`` action was rejected: its handler resets the cuMotion
world to the static-scene set, which does not include the goal-embedded
trial world, so it would vet against an EMPTY world.)

Writes artifacts/m6/trials_raw.json for host-side mesh verification
(scripts/m6_verify_trials.py).  Exit 0 iff every trial planned, executed,
and converged (collision re-verification happens on the host).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseArray
from isaac_ros_cumotion_interfaces.action import MotionPlan
from moveit_msgs.msg import CollisionObject
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

WORK = Path("/work")
POSES = WORK / "artifacts" / "m6" / "trial_poses.json"
OUT = WORK / "artifacts" / "m6" / "trials_raw.json"

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
TIME_DILATION = 0.5           # conservative commissioning speed
CONV_TOL_RAD = 0.02           # M5 tracking-convergence gate
CONV_TIMEOUT_S = 5.0
PLAN_TIMEOUT_FIRST_S = 300.0  # first plan includes CUDA warmup
PLAN_TIMEOUT_S = 60.0
SERVER_WAIT_S = 240.0


def make_collision_objects(world_boxes, stamp):
    objs = []
    for box in world_boxes:
        obj = CollisionObject()
        obj.header.frame_id = "base_link"
        obj.header.stamp = stamp
        obj.id = box["name"]
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(v) for v in box["size"]]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (
            float(v) for v in box["center"])
        pose.orientation.w = 1.0
        obj.primitives.append(prim)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD
        objs.append(obj)
    return objs


class TrialRunner(Node):

    def __init__(self) -> None:
        super().__init__(
            "m6_trial_runner",
            parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.q_now = {}
        self.js_count = 0
        self.create_subscription(JointState, "/joint_states",
                                 self._on_js, 10)
        self.plan_client = ActionClient(self, MotionPlan,
                                        "cumotion/motion_plan")
        self.fjt_client = ActionClient(
            self, FollowJointTrajectory,
            "/rebot_controller/follow_joint_trajectory")

    def _on_js(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self.q_now[name] = float(pos)
        self.js_count += 1

    # -- helpers ----------------------------------------------------------

    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_ready(self) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < SERVER_WAIT_S:
            rclpy.spin_once(self, timeout_sec=0.2)
            ok_plan = self.plan_client.server_is_ready()
            ok_fjt = self.fjt_client.server_is_ready()
            ok_js = all(j in self.q_now for j in ARM_JOINTS)
            if ok_plan and ok_fjt and ok_js:
                self.get_logger().info(
                    f"ready after {time.monotonic() - t0:.1f}s "
                    f"(js_count={self.js_count})")
                return
        raise RuntimeError(
            f"stack not ready after {SERVER_WAIT_S}s: "
            f"motion_plan={self.plan_client.server_is_ready()} "
            f"fjt={self.fjt_client.server_is_ready()} "
            f"joint_states={sorted(self.q_now)}")

    def await_action(self, client, goal, timeout_s: float):
        """send goal -> (result_msg or None, error string)."""
        send_fut = client.send_goal_async(goal)
        t0 = time.monotonic()
        while not send_fut.done():
            if time.monotonic() - t0 > timeout_s:
                return None, "goal-send timeout"
            rclpy.spin_once(self, timeout_sec=0.05)
        handle = send_fut.result()
        if not handle.accepted:
            return None, "goal rejected"
        res_fut = handle.get_result_async()
        while not res_fut.done():
            if time.monotonic() - t0 > timeout_s:
                handle.cancel_goal_async()
                return None, f"result timeout after {timeout_s}s"
            rclpy.spin_once(self, timeout_sec=0.05)
        return res_fut.result().result, ""

    # -- trial ------------------------------------------------------------

    def run_trial(self, index: int, target: dict, world_objs,
                  first: bool) -> dict:
        rec = {"index": index, "target": {
            "position": target["position"],
            "quat_xyzw": target["quat_xyzw"]}}

        goal = MotionPlan.Goal()
        goal.plan_pose = True
        goal.use_current_state = True
        goal.use_planning_scene = True
        goal.world.collision_objects = world_objs
        goal.time_dilation_factor = float(TIME_DILATION)
        pa = PoseArray()
        pa.header.frame_id = "base_link"
        pose = Pose()
        (pose.position.x, pose.position.y,
         pose.position.z) = (float(v) for v in target["position"])
        (pose.orientation.x, pose.orientation.y, pose.orientation.z,
         pose.orientation.w) = (float(v) for v in target["quat_xyzw"])
        pa.poses.append(pose)
        goal.goal_pose = pa

        t_plan0 = time.monotonic()
        result, err = self.await_action(
            self.plan_client, goal,
            PLAN_TIMEOUT_FIRST_S if first else PLAN_TIMEOUT_S)
        rec["plan_wall_s"] = round(time.monotonic() - t_plan0, 3)
        if result is None:
            rec["planned"] = False
            rec["error"] = f"motion_plan: {err}"
            return rec
        rec["planned"] = bool(result.success)
        rec["error_code"] = int(result.error_code.val)
        rec["planning_time_s"] = round(float(result.planning_time), 4)
        if not result.success or not result.planned_trajectory:
            rec["error"] = (f"cumotion planning failed "
                            f"(error_code={result.error_code.val})")
            return rec

        jt = result.planned_trajectory[0].joint_trajectory
        rec["joint_names"] = list(jt.joint_names)
        if list(jt.joint_names) != ARM_JOINTS:
            rec["error"] = f"unexpected joint order {list(jt.joint_names)}"
            return rec
        points = [{
            "t": p.time_from_start.sec + p.time_from_start.nanosec * 1e-9,
            "q": [float(v) for v in p.positions],
        } for p in jt.points]
        rec["trajectory"] = points
        rec["duration_s"] = round(points[-1]["t"], 3) if points else 0.0

        # Drop leading t=0 point(s): they duplicate the live state and the
        # sim-JTC shim requires strictly positive time_from_start.
        fjt = FollowJointTrajectory.Goal()
        fjt.trajectory.joint_names = list(jt.joint_names)
        fjt.trajectory.points = [p for p in jt.points
                                 if (p.time_from_start.sec
                                     + p.time_from_start.nanosec * 1e-9)
                                 > 1e-9]
        if not fjt.trajectory.points:
            rec["error"] = "trajectory empty after dropping t=0 point"
            return rec

        t_exec0 = time.monotonic()
        exec_result, err = self.await_action(
            self.fjt_client, fjt, rec["duration_s"] + 30.0)
        rec["exec_wall_s"] = round(time.monotonic() - t_exec0, 3)
        if exec_result is None:
            rec["executed"] = False
            rec["error"] = f"fjt: {err}"
            return rec
        rec["executed"] = (exec_result.error_code
                           == FollowJointTrajectory.Result.SUCCESSFUL)
        if not rec["executed"]:
            rec["error"] = (f"fjt error_code={exec_result.error_code} "
                            f"{exec_result.error_string!r}")
            return rec

        # Tracking convergence vs the final trajectory point.
        q_final = points[-1]["q"]
        t0 = time.monotonic()
        conv_err = float("inf")
        while time.monotonic() - t0 < CONV_TIMEOUT_S:
            rclpy.spin_once(self, timeout_sec=0.05)
            conv_err = max(abs(self.q_now[j] - q_final[i])
                           for i, j in enumerate(ARM_JOINTS))
            if conv_err <= CONV_TOL_RAD:
                break
        rec["converged"] = conv_err <= CONV_TOL_RAD
        rec["tracking_err_rad"] = round(conv_err, 5)
        if not rec["converged"]:
            rec["error"] = f"no convergence: {conv_err:.4f} rad"
        return rec


def main() -> int:
    doc = json.loads(POSES.read_text())
    n_required = int(doc.get("n_trials_required", 100))
    rclpy.init()
    node = TrialRunner()
    trials = []
    rejected = []
    try:
        node.wait_ready()
        world_objs = make_collision_objects(
            doc["world_boxes"], node.get_clock().now().to_msg())
        first = True
        for target in doc["targets"]:
            if len(trials) >= n_required:
                break
            rec = node.run_trial(len(trials) + 1, target, world_objs,
                                 first=first)
            first = False
            rec["candidate_index"] = target["index"]
            if not rec.get("planned"):
                # cuMotion could not solve this candidate (measured cause:
                # trajopt INVERSE_KINEMATICS_FAILURE on awkward poses that
                # ik_core's local DLS does solve).  The arm never moved, so
                # the chain start state is unchanged: record the candidate
                # as REJECTED and continue with the next one.
                rejected.append(rec)
                node.get_logger().warn(
                    f"candidate {target['index']:3d} REJECTED "
                    f"({rec.get('error', '?')}); "
                    f"{len(rejected)} rejected so far")
                continue
            ok = rec.get("executed") and rec.get("converged")
            node.get_logger().info(
                f"trial {len(trials) + 1:3d}/{n_required}: "
                f"{'OK' if ok else 'FAIL ' + rec.get('error', '?')} "
                f"plan={rec.get('planning_time_s', -1):.3f}s "
                f"dur={rec.get('duration_s', -1):.1f}s "
                f"track={rec.get('tracking_err_rad', -1):.4f}rad")
            trials.append(rec)
    finally:
        n_ok = sum(1 for t in trials
                   if t.get("planned") and t.get("executed")
                   and t.get("converged"))
        OUT.write_text(json.dumps({
            "milestone": "M6",
            "artifact": "trials_raw",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
            "planner": "isaac_ros_cumotion 4.5 standalone action server "
                       "cumotion/motion_plan (no MoveIt)",
            "time_dilation_factor": TIME_DILATION,
            "n_trials_required": n_required,
            "n_trials": len(trials),
            "n_planned_executed_converged": n_ok,
            "n_candidates_rejected": len(rejected),
            "rejected_candidates": rejected,
            "trials": trials,
        }, indent=2) + "\n")
        node.get_logger().info(
            f"wrote {OUT} ({n_ok}/{len(trials)} ok, "
            f"{len(rejected)} candidates rejected)")
        node.destroy_node()
        rclpy.shutdown()
    return 0 if len(trials) == n_required and n_ok == n_required else 1


if __name__ == "__main__":
    sys.exit(main())
