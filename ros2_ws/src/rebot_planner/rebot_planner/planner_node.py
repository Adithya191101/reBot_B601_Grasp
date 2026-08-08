"""rebot_planner_node: thin rclpy wrapper around the planning cores.

MoveIt replacement (design-doc planning slot, pre-cuMotion): offers the
``rebot_planner_msgs/action/MoveToPose`` action (canonical ``gripper_tcp``
goal in ``base_link``), plans a Cartesian-linear waypoint trajectory with
the proven Pinocchio local-DLS IK, collision-checks it against the cell
geometry, times it against the canonical 5/3 rad/s velocity limits, and
executes it through the canonical trajectory adapter::

    Subscribes: /joint_states                     (canonical merged states)
    Action in:  /rebot_planner/move_to_pose       (rebot_planner_msgs)
    Action out: /rebot_controller/follow_joint_trajectory (control_msgs)

All planning logic lives in :mod:`rebot_planner.core` (rclpy-free); this
file only does ROS I/O, goal gating, cancellation propagation, and
result/feedback mapping.

**Scope limit:** colliding paths are REJECTED, not re-routed; obstacle
avoidance arrives with cuMotion (design doc M6/M7).
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Optional

import numpy as np

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rebot_planner_msgs.action import MoveToPose

from .core import collision_core as cc
from .core import ik_core, path_core

ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def _sec_to_duration(sec: float):
    from builtin_interfaces.msg import Duration as DurationMsg

    msg = DurationMsg()
    msg.sec = int(sec)
    msg.nanosec = int(round((sec - int(sec)) * 1e9))
    return msg


class RebotPlannerNode(Node):

    def __init__(self) -> None:
        super().__init__("rebot_planner")

        self.declare_parameter("urdf_path", "")
        self.declare_parameter("package_dirs", [""])
        self.declare_parameter("cell_geometry", "")
        self.declare_parameter("ee_frame", ik_core.DEFAULT_EE_FRAME)
        self.declare_parameter("tcp_offset_m",
                               list(ik_core.DEFAULT_TCP_OFFSET_M))
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter(
            "controller_action", "/rebot_controller/follow_joint_trajectory")
        self.declare_parameter("move_action", "/rebot_planner/move_to_pose")
        self.declare_parameter("velocity_safety_factor",
                               path_core.DEFAULT_SAFETY_FACTOR)
        self.declare_parameter("min_segment_sec",
                               path_core.DEFAULT_MIN_SEGMENT_SEC)
        self.declare_parameter("steps", path_core.DEFAULT_STEPS)
        self.declare_parameter("branch_jump_rad",
                               path_core.DEFAULT_BRANCH_JUMP_RAD)
        self.declare_parameter("shoulder_jump_rad",
                               path_core.DEFAULT_SHOULDER_JUMP_RAD)
        self.declare_parameter("max_rot_step_rad",
                               path_core.DEFAULT_MAX_ROT_STEP_RAD)
        self.declare_parameter("collision_checking", True)
        self.declare_parameter("collision_max_step_rad",
                               cc.DEFAULT_MAX_STEP_RAD)
        self.declare_parameter("joint_state_stale_sec", 1.0)
        self.declare_parameter("goal_connect_timeout_sec", 5.0)
        self.declare_parameter("result_timeout_margin_sec", 3.0)

        urdf_path = str(self.get_parameter("urdf_path").value or "")
        if not urdf_path:
            urdf_path = os.environ.get("REBOT_CANONICAL_URDF", "")
        if not urdf_path or not os.path.isfile(urdf_path):
            raise RuntimeError(
                "urdf_path parameter (or REBOT_CANONICAL_URDF) must point at "
                "the canonical URDF (urdf/rebot_b601dm_canonical.urdf); "
                f"got {urdf_path!r}")

        self._kin = ik_core.KinematicsCore(
            urdf_path,
            ee_frame=str(self.get_parameter("ee_frame").value),
            tcp_offset_m=[float(v) for v in
                          self.get_parameter("tcp_offset_m").value],
        )

        self._collision: Optional[cc.CollisionCore] = None
        if bool(self.get_parameter("collision_checking").value):
            cell_yaml = str(self.get_parameter("cell_geometry").value or "")
            if not cell_yaml:
                from ament_index_python.packages import (
                    get_package_share_directory)
                cell_yaml = os.path.join(
                    get_package_share_directory("rebot_planner"),
                    "config", "cell_geometry.yaml")
            pkg_dirs = [str(d) for d in
                        self.get_parameter("package_dirs").value if d]
            self._collision = cc.CollisionCore(
                self._kin,
                package_dirs=pkg_dirs or None,
                cell_geometry_yaml=cell_yaml,
            )
            self.get_logger().info(
                f"collision checking on: {len(self._collision.world_boxes)} "
                f"world boxes from {cell_yaml}")
        else:
            self.get_logger().warn("collision checking DISABLED by parameter")

        self._safety_factor = float(
            self.get_parameter("velocity_safety_factor").value)
        self._min_segment = float(self.get_parameter("min_segment_sec").value)
        self._steps = int(self.get_parameter("steps").value)
        self._branch_jump = float(self.get_parameter("branch_jump_rad").value)
        self._shoulder_jump = float(
            self.get_parameter("shoulder_jump_rad").value)
        self._max_rot_step = float(
            self.get_parameter("max_rot_step_rad").value)
        self._coll_step = float(
            self.get_parameter("collision_max_step_rad").value)
        self._stale_sec = float(
            self.get_parameter("joint_state_stale_sec").value)
        self._connect_timeout = float(
            self.get_parameter("goal_connect_timeout_sec").value)
        self._result_margin = float(
            self.get_parameter("result_timeout_margin_sec").value)

        # Latest canonical joint state (by name; the merged /joint_states
        # also carries the gripper joints -- ignored here).
        self._js_lock = threading.Lock()
        self._q_by_name: Dict[str, float] = {}
        self._js_stamp: Optional[rclpy.time.Time] = None

        self._busy_lock = threading.Lock()
        self._busy = False

        self._cb_group = ReentrantCallbackGroup()
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._on_joint_state, 10, callback_group=self._cb_group)
        self._fjt_client = ActionClient(
            self, FollowJointTrajectory,
            str(self.get_parameter("controller_action").value),
            callback_group=self._cb_group)
        self._server = ActionServer(
            self, MoveToPose,
            str(self.get_parameter("move_action").value),
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda _gh: CancelResponse.ACCEPT,
            callback_group=self._cb_group)
        self.get_logger().info(
            f"rebot_planner up (urdf={urdf_path}, "
            f"safety_factor={self._safety_factor})")

    # -- joint states -----------------------------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        with self._js_lock:
            for name, pos in zip(msg.name, msg.position):
                self._q_by_name[name] = float(pos)
            self._js_stamp = self.get_clock().now()

    def _current_q6(self) -> Optional[np.ndarray]:
        with self._js_lock:
            if self._js_stamp is None:
                return None
            age = (self.get_clock().now() - self._js_stamp).nanoseconds * 1e-9
            if age > self._stale_sec:
                return None
            try:
                return np.array([self._q_by_name[n] for n in ARM_JOINT_NAMES])
            except KeyError:
                return None

    # -- goal handling ----------------------------------------------------

    def _on_goal(self, goal: MoveToPose.Goal) -> GoalResponse:
        frame = goal.target.header.frame_id
        if frame not in ("", "base_link"):
            self.get_logger().warn(
                f"rejecting MoveToPose: frame_id {frame!r} != base_link "
                "(the planner does not consume TF)")
            return GoalResponse.REJECT
        sf = float(goal.velocity_safety_factor)
        if sf != 0.0 and not 0.0 < sf <= 1.0:
            self.get_logger().warn(
                f"rejecting MoveToPose: velocity_safety_factor {sf} "
                "outside (0, 1]")
            return GoalResponse.REJECT
        with self._busy_lock:
            if self._busy:
                self.get_logger().warn(
                    "rejecting MoveToPose: another goal is active")
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _abort(self, goal_handle, result: MoveToPose.Result,
               message: str) -> MoveToPose.Result:
        result.success = False
        result.message = message
        self.get_logger().warn(f"MoveToPose aborted: {message}")
        goal_handle.abort()
        return result

    async def _execute(self, goal_handle) -> MoveToPose.Result:
        result = MoveToPose.Result()
        with self._busy_lock:
            if self._busy:
                return self._abort(goal_handle, result,
                                   "another goal became active concurrently")
            self._busy = True
        try:
            return await self._plan_and_execute(goal_handle, result)
        finally:
            with self._busy_lock:
                self._busy = False

    async def _plan_and_execute(self, goal_handle,
                                result: MoveToPose.Result) -> MoveToPose.Result:
        goal: MoveToPose.Goal = goal_handle.request
        if goal.target.header.frame_id == "":
            self.get_logger().warn(
                "MoveToPose target has empty frame_id; assuming base_link")

        feedback = MoveToPose.Feedback()
        feedback.stage = "planning"
        goal_handle.publish_feedback(feedback)

        q_now = self._current_q6()
        if q_now is None:
            return self._abort(
                goal_handle, result,
                "no fresh canonical /joint_states (all six arm joints)")

        p = goal.target.pose
        try:
            T_goal = ik_core.pose_to_transform(
                (p.position.x, p.position.y, p.position.z),
                (p.orientation.x, p.orientation.y, p.orientation.z,
                 p.orientation.w))
        except ValueError as exc:
            return self._abort(goal_handle, result, f"bad target pose: {exc}")

        plan = path_core.plan_linear(
            self._kin, q_now, T_goal,
            steps=self._steps,
            branch_jump_rad=self._branch_jump,
            shoulder_jump_rad=self._shoulder_jump,
            max_rot_step_rad=self._max_rot_step)
        if not plan.ok:
            return self._abort(
                goal_handle, result,
                f"planning failed at waypoint {plan.failed_at_waypoint}: "
                f"{plan.reason} {plan.detail}")

        if self._collision is not None:
            feedback.stage = "collision_check"
            goal_handle.publish_feedback(feedback)
            check = self._collision.check_path(
                plan.waypoints, max_step_rad=self._coll_step)
            if not check.ok:
                return self._abort(
                    goal_handle, result,
                    f"path in collision (segment {check.failed_segment}, "
                    f"pairs {list(check.pairs)}); the planner rejects "
                    "colliding paths -- re-pose the goal (obstacle-avoiding "
                    "planning arrives with cuMotion)")

        sf = float(goal.velocity_safety_factor) or self._safety_factor
        times = path_core.time_waypoints(
            plan.waypoints, self._kin.velocity_limits,
            safety_factor=sf, min_segment_sec=self._min_segment)
        duration = times[-1]

        traj = JointTrajectory()
        traj.joint_names = list(ARM_JOINT_NAMES)
        # Drop waypoint 0 (t=0, current config): downstream expects strictly
        # increasing times from a start matching the live state.
        for q6, t in list(zip(plan.waypoints, times))[1:]:
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q6]
            pt.time_from_start = _sec_to_duration(t)
            traj.points.append(pt)

        if not self._fjt_client.wait_for_server(
                timeout_sec=self._connect_timeout):
            return self._abort(goal_handle, result,
                               "controller FJT server unavailable")

        fjt_goal = FollowJointTrajectory.Goal()
        fjt_goal.trajectory = traj

        feedback.stage = "executing"
        goal_handle.publish_feedback(feedback)

        downstream = await self._fjt_client.send_goal_async(fjt_goal)
        if not downstream.accepted:
            return self._abort(goal_handle, result,
                               "controller rejected the trajectory goal")

        result_future = downstream.get_result_async()
        start = self.get_clock().now()
        deadline_sec = duration + self._connect_timeout + self._result_margin
        while not result_future.done():
            elapsed = (self.get_clock().now() - start).nanoseconds * 1e-9
            if goal_handle.is_cancel_requested:
                await downstream.cancel_goal_async()
                await result_future
                goal_handle.canceled()
                result.success = False
                result.message = "goal canceled; controller notified"
                return result
            if elapsed > deadline_sec:
                await downstream.cancel_goal_async()
                return self._abort(goal_handle, result,
                                   "controller result timeout")
            feedback.progress = min(1.0, elapsed / max(duration, 1e-6))
            goal_handle.publish_feedback(feedback)
            await _sleep(self, 0.05)

        fjt_result = result_future.result().result
        if fjt_result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            return self._abort(
                goal_handle, result,
                f"trajectory execution failed: error_code="
                f"{fjt_result.error_code} {fjt_result.error_string!r}")

        result.success = True
        result.message = "reached"
        result.q_final = [float(v) for v in plan.waypoints[-1]]
        result.planned_duration_sec = float(duration)
        goal_handle.succeed()
        return result


async def _sleep(node: Node, seconds: float) -> None:
    """Await-able sleep that keeps the executor spinning."""
    from rclpy.task import Future

    fut = Future()
    timer = node.create_timer(seconds, lambda: fut.set_result(None))
    try:
        await fut
    finally:
        node.destroy_timer(timer)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RebotPlannerNode()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
