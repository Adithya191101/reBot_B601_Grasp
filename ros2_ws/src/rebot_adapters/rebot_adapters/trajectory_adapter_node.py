"""rebot_trajectory_adapter: thin rclpy wrapper around trajectory_core.

Exposes the single MoveIt-facing controller action (design doc sec. 9.2)::

    Public: /rebot_controller/follow_joint_trajectory
    Sim:    /rebot_sim_arm_controller/follow_joint_trajectory
    Real:   /rebotarm/follow_joint_trajectory

All validation gates (canonical names, ordering, finiteness, monotonic
timing, canonical URDF position limits, 5/3 rad/s finite-difference
velocity limits, single-active-goal) live in
:mod:`rebot_adapters.core.trajectory_core`. This file only does ROS I/O,
cancellation propagation, and result/feedback mapping.
"""

from __future__ import annotations

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .core import trajectory_core as tc


def _duration_to_sec(d) -> float:
    return d.sec + d.nanosec * 1e-9


def _sec_to_duration(sec: float):
    from builtin_interfaces.msg import Duration as DurationMsg

    msg = DurationMsg()
    msg.sec = int(sec)
    msg.nanosec = int(round((sec - int(sec)) * 1e9))
    return msg


def _to_core_points(traj: JointTrajectory):
    return [
        tc.TrajectoryPoint(
            time_from_start=_duration_to_sec(p.time_from_start),
            positions=tuple(p.positions),
            velocities=tuple(p.velocities),
            accelerations=tuple(p.accelerations),
        )
        for p in traj.points
    ]


def _to_ros_trajectory(result: tc.ValidationResult) -> JointTrajectory:
    traj = JointTrajectory()
    traj.joint_names = list(result.joint_names)
    for p in result.points:
        rp = JointTrajectoryPoint()
        rp.positions = list(p.positions)
        rp.velocities = list(p.velocities)
        rp.accelerations = list(p.accelerations)
        rp.time_from_start = _sec_to_duration(p.time_from_start)
        traj.points.append(rp)
    return traj


class RebotTrajectoryAdapter(Node):

    def __init__(self) -> None:
        super().__init__("rebot_trajectory_adapter")

        self.declare_parameter("mode", "real")  # sim | real
        self.declare_parameter(
            "public_action", "/rebot_controller/follow_joint_trajectory"
        )
        self.declare_parameter(
            "real_action", "/rebotarm/follow_joint_trajectory"
        )
        self.declare_parameter(
            "sim_action", "/rebot_sim_arm_controller/follow_joint_trajectory"
        )
        self.declare_parameter(
            "joint_names",
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        )
        self.declare_parameter("goal_connect_timeout_sec", 5.0)
        self.declare_parameter("result_timeout_margin_sec", 3.0)
        self.declare_parameter("reorder_joint_names", False)
        self.declare_parameter("validate_segment_velocity", True)

        mode = self.get_parameter("mode").value
        if mode == "real":
            downstream = self.get_parameter("real_action").value
        elif mode == "sim":
            downstream = self.get_parameter("sim_action").value
        else:
            raise RuntimeError(f"unknown mode {mode!r}; use sim|real")

        self._reorder = bool(self.get_parameter("reorder_joint_names").value)
        self._validate_vel = bool(
            self.get_parameter("validate_segment_velocity").value
        )
        self._connect_timeout = float(
            self.get_parameter("goal_connect_timeout_sec").value
        )
        self._result_margin = float(
            self.get_parameter("result_timeout_margin_sec").value
        )

        self._gate = tc.SingleGoalGate()
        self._cb_group = ReentrantCallbackGroup()

        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            downstream,
            callback_group=self._cb_group,
        )
        self._server = ActionServer(
            self,
            FollowJointTrajectory,
            self.get_parameter("public_action").value,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._cb_group,
        )
        self.get_logger().info(
            f"rebot_trajectory_adapter up (mode={mode}, downstream={downstream})"
        )

    # -- goal gating -------------------------------------------------------

    def _validate(self, goal: FollowJointTrajectory.Goal) -> tc.ValidationResult:
        return tc.validate_trajectory(
            list(goal.trajectory.joint_names),
            _to_core_points(goal.trajectory),
            reorder_joint_names=self._reorder,
            validate_segment_velocity=self._validate_vel,
        )

    def _on_goal(self, goal_request: FollowJointTrajectory.Goal) -> GoalResponse:
        check = self._validate(goal_request)
        if not check.ok:
            self.get_logger().warn(f"rejecting FJT goal: {check.reason}")
            return GoalResponse.REJECT
        if self._gate.active_goal is not None:
            self.get_logger().warn(
                "rejecting FJT goal: another goal is already active"
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    # -- execution ---------------------------------------------------------

    async def _execute(self, goal_handle):
        result = FollowJointTrajectory.Result()
        goal_key = bytes(goal_handle.goal_id.uuid)

        # Gate 8: single active downstream goal per adapter instance.
        if not self._gate.try_acquire(goal_key):
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "another goal became active concurrently"
            goal_handle.abort()
            return result

        try:
            # Re-validate: parameters may have changed since goal_callback.
            check = self._validate(goal_handle.request)
            if not check.ok:
                result.error_code = check.error_code
                result.error_string = check.reason
                goal_handle.abort()
                return result

            if not self._client.wait_for_server(
                timeout_sec=self._connect_timeout
            ):
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = "downstream FJT server unavailable"
                goal_handle.abort()
                return result

            downstream_goal = FollowJointTrajectory.Goal()
            downstream_goal.trajectory = _to_ros_trajectory(check)
            downstream_goal.path_tolerance = list(
                goal_handle.request.path_tolerance
            )
            downstream_goal.goal_tolerance = list(
                goal_handle.request.goal_tolerance
            )
            downstream_goal.goal_time_tolerance = (
                goal_handle.request.goal_time_tolerance
            )

            # Gate 10 (feedback half): map downstream feedback to public goal.
            def _feedback_cb(fb_msg) -> None:
                goal_handle.publish_feedback(fb_msg.feedback)

            downstream_handle = await self._client.send_goal_async(
                downstream_goal, feedback_callback=_feedback_cb
            )
            if not downstream_handle.accepted:
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = "downstream server rejected goal"
                goal_handle.abort()
                return result

            result_future = downstream_handle.get_result_async()
            deadline = self.get_clock().now() + Duration(
                seconds=tc.trajectory_duration(check.points)
                + self._connect_timeout
                + self._result_margin
            )

            # Gate 9: propagate cancellation downstream.
            while not result_future.done():
                if goal_handle.is_cancel_requested:
                    await downstream_handle.cancel_goal_async()
                    await result_future
                    goal_handle.canceled()
                    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    result.error_string = "goal canceled; downstream notified"
                    return result
                if self.get_clock().now() > deadline:
                    await downstream_handle.cancel_goal_async()
                    result.error_code = (
                        FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                    )
                    result.error_string = "downstream result timeout"
                    goal_handle.abort()
                    return result
                await _sleep(self, 0.05)

            # Gate 10 (result half): map downstream result to public goal.
            wrapped = result_future.result()
            result = wrapped.result
            if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return result
        finally:
            self._gate.release(goal_key)


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
    node = RebotTrajectoryAdapter()
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
