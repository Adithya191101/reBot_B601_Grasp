"""sim_jtc_shim: FollowJointTrajectory -> /isaac_joint_commands bridge.

The M5 Isaac bridge (``scripts/b601_sim_bridge.py``, from the rearm
environment) consumes plain ``sensor_msgs/JointState`` position targets each
physics tick; the frozen adapter contract expects FollowJointTrajectory
controllers.  This node closes that gap: it serves ONE FJT action for a
configured joint subset and executes accepted goals by publishing linearly
interpolated JointState commands at the sim rate.

Two instances run in the sim profile (see ``launch/sim_profile.launch.py``):

* arm:      /rebot_sim_arm_controller/follow_joint_trajectory, joint1..joint6
* gripper:  /gripper_controller/follow_joint_trajectory, gripper_joint1/2

Both publish onto the SAME ``/isaac_joint_commands`` topic: the Isaac-side
articulation controller applies only the joints NAMED in each message and
PhysX drive targets persist per-DOF, so the two never clobber each other.

Completion is honest, not time-based: after the final point the node holds
the target and compares the measured state (``/isaac_joint_states``) against
it; SUCCESSFUL only within ``goal_tolerance``, GOAL_TOLERANCE_VIOLATED after
``settle_timeout_sec``.  Runs with use_sim_time so interpolation follows the
sim's /clock.  All pure logic lives in :mod:`rebot_sim_bridge.core.jtc_shim_core`.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState

from .core import jtc_shim_core as core


class SimJtcShim(Node):

    def __init__(self) -> None:
        super().__init__("sim_jtc_shim")

        self.declare_parameter("action_name", "")
        self.declare_parameter("joint_names", [""])
        self.declare_parameter("command_topic", "/isaac_joint_commands")
        self.declare_parameter("state_topic", "/isaac_joint_states")
        self.declare_parameter("command_rate_hz", 60.0)
        self.declare_parameter("goal_tolerance", 0.01)  # rad (arm) / m (jaw)
        self.declare_parameter("settle_timeout_sec", 3.0)

        action_name = str(self.get_parameter("action_name").value)
        if not action_name:
            raise RuntimeError("action_name parameter is required")
        self._joints = tuple(
            str(j) for j in self.get_parameter("joint_names").value if j)
        if not self._joints:
            raise RuntimeError("joint_names parameter is required")
        self._rate = float(self.get_parameter("command_rate_hz").value)
        self._tolerance = float(self.get_parameter("goal_tolerance").value)
        self._settle_timeout = float(
            self.get_parameter("settle_timeout_sec").value)

        self._state_lock = threading.Lock()
        self._measured: Dict[str, float] = {}

        self._busy_lock = threading.Lock()
        self._busy = False

        self._cb_group = ReentrantCallbackGroup()
        self._cmd_pub = self.create_publisher(
            JointState, str(self.get_parameter("command_topic").value), 10)
        self.create_subscription(
            JointState, str(self.get_parameter("state_topic").value),
            self._on_state, 50, callback_group=self._cb_group)
        self._server = ActionServer(
            self, FollowJointTrajectory, action_name,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda _gh: CancelResponse.ACCEPT,
            callback_group=self._cb_group)
        self.get_logger().info(
            "sim_jtc_shim up: %s -> %s for %s"
            % (action_name, self.get_parameter("command_topic").value,
               list(self._joints)))

    # -- state -------------------------------------------------------------

    def _on_state(self, msg: JointState) -> None:
        with self._state_lock:
            for name, pos in zip(msg.name, msg.position):
                self._measured[name] = float(pos)

    def _measured_snapshot(self) -> Dict[str, float]:
        with self._state_lock:
            return dict(self._measured)

    # -- goal handling -----------------------------------------------------

    def _to_core(self, goal: FollowJointTrajectory.Goal) -> core.ShimTrajectory:
        points = [
            core.ShimPoint(
                time_from_start=(p.time_from_start.sec
                                 + p.time_from_start.nanosec * 1e-9),
                positions=tuple(p.positions))
            for p in goal.trajectory.points
        ]
        return core.validate_goal(
            self._joints, list(goal.trajectory.joint_names), points)

    def _on_goal(self, goal: FollowJointTrajectory.Goal) -> GoalResponse:
        try:
            self._to_core(goal)
        except core.TrajectoryGoalError as exc:
            self.get_logger().warn("rejecting FJT goal: %s" % exc)
            return GoalResponse.REJECT
        with self._busy_lock:
            if self._busy:
                self.get_logger().warn(
                    "rejecting FJT goal: another goal is active")
                return GoalResponse.REJECT
        measured = self._measured_snapshot()
        missing = [j for j in self._joints if j not in measured]
        if missing:
            self.get_logger().warn(
                "rejecting FJT goal: no sim state yet for %s" % missing)
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _publish_command(self, positions) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self._joints)
        msg.position = [float(v) for v in positions]
        self._cmd_pub.publish(msg)

    async def _execute(self, goal_handle) -> FollowJointTrajectory.Result:
        result = FollowJointTrajectory.Result()
        with self._busy_lock:
            if self._busy:
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = "another goal became active concurrently"
                goal_handle.abort()
                return result
            self._busy = True
        try:
            return await self._run_trajectory(goal_handle, result)
        finally:
            with self._busy_lock:
                self._busy = False

    async def _run_trajectory(self, goal_handle, result):
        try:
            trajectory = self._to_core(goal_handle.request)
        except core.TrajectoryGoalError as exc:  # params changed since accept
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = str(exc)
            goal_handle.abort()
            return result

        measured = self._measured_snapshot()
        start = tuple(measured[j] for j in self._joints)
        target = trajectory.points[-1].positions
        period = 1.0 / self._rate
        t0 = self.get_clock().now()

        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = list(self._joints)

        # Interpolation phase (sim-time driven).
        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = "goal canceled"
                return result
            elapsed = (self.get_clock().now() - t0).nanoseconds * 1e-9
            command = core.sample(trajectory, start, elapsed)
            self._publish_command(command)
            feedback.desired.positions = list(command)
            measured = self._measured_snapshot()
            feedback.actual.positions = [
                measured.get(j, 0.0) for j in self._joints]
            goal_handle.publish_feedback(feedback)
            if elapsed >= trajectory.duration:
                break
            await _sleep(self, period)

        # Settle phase: hold the target, succeed only on measured convergence.
        settle_t0 = self.get_clock().now()
        while True:
            self._publish_command(target)
            error = core.goal_error(
                target, self._measured_snapshot(), self._joints)
            if error is not None and error <= self._tolerance:
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = (
                    "reached (max error %.4f <= %.4f)"
                    % (error, self._tolerance))
                goal_handle.succeed()
                return result
            settle = (self.get_clock().now() - settle_t0).nanoseconds * 1e-9
            if settle > self._settle_timeout:
                result.error_code = (
                    FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED)
                result.error_string = (
                    "did not settle within %.2fs (max error %s > %.4f)"
                    % (self._settle_timeout, error, self._tolerance))
                goal_handle.abort()
                return result
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = "goal canceled during settle"
                return result
            await _sleep(self, period)


async def _sleep(node: Node, seconds: float) -> None:
    """Await-able sleep that keeps the executor spinning (repo pattern)."""
    from rclpy.task import Future

    fut = Future()
    timer = node.create_timer(seconds, lambda: fut.set_result(None))
    try:
        await fut
    finally:
        node.destroy_timer(timer)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimJtcShim()
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
