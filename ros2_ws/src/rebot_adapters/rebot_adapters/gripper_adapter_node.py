"""rebot_gripper_adapter: thin rclpy wrapper around gripper_core.

Public canonical action (design doc sec. 4.4, 8.2)::

    /rebot_controller/gripper_command   control_msgs/action/GripperCommand

``command.position`` is the master prismatic-joint coordinate ``q_jaw_m``
in METRES. Feedback/result ``position`` is the measured/estimated
``q_jaw_m`` in metres. The action name is intentionally not the upstream
``/rebotarm/gripper/command``, whose position is motor radians.

Real profile: clamp -> monotonic piecewise interpolation -> motor_rad goal
on ``/rebotarm/gripper/command``; motor state on ``/rebotarm/gripper/state``
maps back through the inverse table. Real mode refuses to start with an
empty or non-monotonic calibration table (sec. 4.5).

Sim profile: FollowJointTrajectory on
``/gripper_controller/follow_joint_trajectory`` with positions
``[q_jaw_m, q_jaw_m]`` for the two jaw joints.

Both profiles also publish ``/rebot/gripper_joint_states`` with both jaw
joints at equal q (sec. 4.6). All conversion/validation and the goal state
machine live in :mod:`rebot_adapters.core.gripper_core`.
"""

from __future__ import annotations

from typing import Optional

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory, GripperCommand
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .core import gripper_core as gc
from .core.limits import GRIPPER_JOINT_NAMES


def _sec_to_duration(sec: float):
    from builtin_interfaces.msg import Duration as DurationMsg

    msg = DurationMsg()
    msg.sec = int(sec)
    msg.nanosec = int(round((sec - int(sec)) * 1e9))
    return msg


class RebotGripperAdapter(Node):

    def __init__(self) -> None:
        super().__init__("rebot_gripper_adapter")

        self.declare_parameter("mode", "real")  # sim | real
        self.declare_parameter(
            "public_action", "/rebot_controller/gripper_command"
        )
        self.declare_parameter("real_action", "/rebotarm/gripper/command")
        self.declare_parameter("real_state_topic", "/rebotarm/gripper/state")
        self.declare_parameter(
            "sim_action", "/gripper_controller/follow_joint_trajectory"
        )
        self.declare_parameter(
            "jaw_state_topic", "/rebot/gripper_joint_states"
        )
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("sim_motion_duration_sec", 1.0)
        self.declare_parameter("goal_connect_timeout_sec", 5.0)
        self.declare_parameter("result_timeout_sec", 10.0)

        self._mode = self.get_parameter("mode").value
        self._connect_timeout = float(
            self.get_parameter("goal_connect_timeout_sec").value
        )
        self._result_timeout = float(
            self.get_parameter("result_timeout_sec").value
        )
        self._sim_duration = float(
            self.get_parameter("sim_motion_duration_sec").value
        )

        self._table: Optional[gc.CalibrationTable] = None
        self._last_q_jaw: Optional[float] = None
        self._cb_group = ReentrantCallbackGroup()

        if self._mode == "real":
            calib = self.get_parameter("calibration_file").value
            if not calib:
                raise RuntimeError(
                    "real mode requires 'calibration_file'; refusing to start "
                    "without a calibration table (design doc sec. 4.5)"
                )
            # Raises CalibrationError on empty/non-monotonic tables: the
            # mandated refusal to start.
            self._table = gc.CalibrationTable.from_yaml_file(calib)

            from rebotarm_msgs.msg import JointMotorState

            self._real_client = ActionClient(
                self,
                GripperCommand,
                self.get_parameter("real_action").value,
                callback_group=self._cb_group,
            )
            self.create_subscription(
                JointMotorState,
                self.get_parameter("real_state_topic").value,
                self._on_motor_state,
                50,
                callback_group=self._cb_group,
            )
        elif self._mode == "sim":
            self._sim_client = ActionClient(
                self,
                FollowJointTrajectory,
                self.get_parameter("sim_action").value,
                callback_group=self._cb_group,
            )
        else:
            raise RuntimeError(f"unknown mode {self._mode!r}; use sim|real")

        self._jaw_pub = self.create_publisher(
            JointState, self.get_parameter("jaw_state_topic").value, 10
        )
        self._server = ActionServer(
            self,
            GripperCommand,
            self.get_parameter("public_action").value,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=lambda _gh: CancelResponse.ACCEPT,
            callback_group=self._cb_group,
        )
        self.get_logger().info(
            f"rebot_gripper_adapter up (mode={self._mode})"
        )

    # -- state feedback ----------------------------------------------------

    def _on_motor_state(self, msg) -> None:
        try:
            self._last_q_jaw = self._table.motor_to_q(msg.position)
        except gc.RangeError as exc:
            self.get_logger().warn(
                f"motor state outside calibration table: {exc}",
                throttle_duration_sec=1.0,
            )
            return
        self._publish_jaw_state(self._last_q_jaw)

    def _publish_jaw_state(self, q_jaw_m: float) -> None:
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(GRIPPER_JOINT_NAMES)
        out.position = [q_jaw_m, q_jaw_m]  # equal q for both jaws
        self._jaw_pub.publish(out)

    # -- goal handling -----------------------------------------------------

    def _on_goal(self, goal_request: GripperCommand.Goal) -> GoalResponse:
        q = goal_request.command.position
        try:
            if self._mode == "real":
                gc.plan_real_command(q, self._table)
            else:
                gc.sim_jaw_positions(q)
        except gc.RangeError as exc:
            self.get_logger().warn(f"rejecting gripper goal: {exc}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    async def _execute(self, goal_handle):
        machine = gc.GripperStateMachine()
        result = GripperCommand.Result()
        q_request = goal_handle.request.command.position

        machine.to(gc.GripperGoalState.CONVERTING)
        try:
            if self._mode == "real":
                return await self._execute_real(
                    goal_handle, machine, q_request, result
                )
            return await self._execute_sim(
                goal_handle, machine, q_request, result
            )
        except gc.RangeError as exc:
            machine.to(gc.GripperGoalState.ABORTED)
            goal_handle.abort()
            result.stalled = False
            result.reached_goal = False
            result.position = self._last_q_jaw or 0.0
            self.get_logger().warn(f"gripper goal aborted: {exc}")
            return result

    async def _execute_real(self, goal_handle, machine, q_request, result):
        cmd = gc.plan_real_command(q_request, self._table)
        tol = self._table.goal_tolerance_q_m

        downstream = GripperCommand.Goal()
        downstream.command.position = cmd.motor_rad  # motor radians downstream
        downstream.command.max_effort = goal_handle.request.command.max_effort

        if not self._real_client.wait_for_server(
            timeout_sec=self._connect_timeout
        ):
            raise gc.RangeError("downstream gripper server unavailable")

        handle = await self._real_client.send_goal_async(downstream)
        if not handle.accepted:
            raise gc.RangeError("downstream gripper server rejected goal")
        machine.to(gc.GripperGoalState.FORWARDED)

        result_future = handle.get_result_async()
        machine.to(gc.GripperGoalState.TRACKING)
        deadline = self.get_clock().now() + Duration(
            seconds=self._result_timeout
        )
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                await handle.cancel_goal_async()
                machine.to(gc.GripperGoalState.CANCELED)
                goal_handle.canceled()
                result.position = self._last_q_jaw or cmd.q_jaw_m_clamped
                result.reached_goal = False
                return result
            if self.get_clock().now() > deadline:
                await handle.cancel_goal_async()
                machine.to(gc.GripperGoalState.ABORTED)
                goal_handle.abort()
                result.position = self._last_q_jaw or cmd.q_jaw_m_clamped
                result.reached_goal = False
                return result
            q_now = self._last_q_jaw
            if q_now is not None:
                fb = GripperCommand.Feedback()
                fb.position = q_now  # q_jaw_m, metres (inverse map)
                fb.reached_goal = gc.reached_goal(
                    q_now, cmd.q_jaw_m_clamped, tol
                )
                goal_handle.publish_feedback(fb)
            await _sleep(self, 0.05)

        downstream_result = result_future.result().result
        q_final = (
            self._last_q_jaw
            if self._last_q_jaw is not None
            else self._table.motor_to_q(downstream_result.position)
        )
        result.position = q_final
        result.effort = downstream_result.effort
        result.stalled = downstream_result.stalled
        result.reached_goal = gc.reached_goal(q_final, cmd.q_jaw_m_clamped, tol)
        if result.reached_goal or downstream_result.reached_goal:
            machine.to(gc.GripperGoalState.SUCCEEDED)
            goal_handle.succeed()
        else:
            machine.to(gc.GripperGoalState.ABORTED)
            goal_handle.abort()
        return result

    async def _execute_sim(self, goal_handle, machine, q_request, result):
        q0, q1 = gc.sim_jaw_positions(q_request)

        traj = JointTrajectory()
        traj.joint_names = list(GRIPPER_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [q0, q1]  # [q_jaw_m, q_jaw_m]
        point.time_from_start = _sec_to_duration(self._sim_duration)
        traj.points.append(point)
        downstream = FollowJointTrajectory.Goal()
        downstream.trajectory = traj

        if not self._sim_client.wait_for_server(
            timeout_sec=self._connect_timeout
        ):
            raise gc.RangeError("sim gripper controller unavailable")

        handle = await self._sim_client.send_goal_async(downstream)
        if not handle.accepted:
            raise gc.RangeError("sim gripper controller rejected goal")
        machine.to(gc.GripperGoalState.FORWARDED)

        result_future = handle.get_result_async()
        machine.to(gc.GripperGoalState.TRACKING)
        deadline = self.get_clock().now() + Duration(
            seconds=self._sim_duration + self._result_timeout
        )
        while not result_future.done():
            if goal_handle.is_cancel_requested:
                await handle.cancel_goal_async()
                machine.to(gc.GripperGoalState.CANCELED)
                goal_handle.canceled()
                result.position = q0
                result.reached_goal = False
                return result
            if self.get_clock().now() > deadline:
                await handle.cancel_goal_async()
                machine.to(gc.GripperGoalState.ABORTED)
                goal_handle.abort()
                result.position = q0
                result.reached_goal = False
                return result
            await _sleep(self, 0.05)

        fjt_result = result_future.result().result
        ok = (
            fjt_result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        )
        result.position = q0
        result.stalled = False
        result.reached_goal = ok
        self._publish_jaw_state(q0)
        if ok:
            machine.to(gc.GripperGoalState.SUCCEEDED)
            goal_handle.succeed()
        else:
            machine.to(gc.GripperGoalState.ABORTED)
            goal_handle.abort()
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
    node = RebotGripperAdapter()
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
