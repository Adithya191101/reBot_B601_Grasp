"""rebot_joint_state_adapter: thin rclpy wrapper around joint_state_core.

Publishes exactly one canonical ``/joint_states`` stream (design doc
sec. 9.1) in either profile:

* ``mode: sim``  — rename/filter ``/rebot_sim/joint_states_raw``;
* ``mode: real`` — merge ``/rebotarm/joint_states`` with
  ``/rebotarm/gripper/state`` through the inverse calibration map.

All merging/validation logic lives in
:mod:`rebot_adapters.core.joint_state_core`; this file only does ROS I/O.
"""

from __future__ import annotations

from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .core import joint_state_core as jsc
from .core.gripper_core import CalibrationError, CalibrationTable


def _stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _sec_to_stamp(sec: float, stamp_msg) -> None:
    stamp_msg.sec = int(sec)
    stamp_msg.nanosec = int(round((sec - int(sec)) * 1e9))


class RebotJointStateAdapter(Node):

    def __init__(self) -> None:
        super().__init__("rebot_joint_state_adapter")

        self.declare_parameter("mode", "real")  # sim | real
        self.declare_parameter("output_topic", "/joint_states")
        self.declare_parameter("arm_input_topic", "/rebotarm/joint_states")
        self.declare_parameter("gripper_input_topic", "/rebotarm/gripper/state")
        self.declare_parameter("sim_input_topic", "/rebot_sim/joint_states_raw")
        self.declare_parameter("stale_timeout_sec", 0.20)
        self.declare_parameter("publish_rate_hz", 100.0)
        self.declare_parameter("require_all_arm_joints", True)
        self.declare_parameter("calibration_file", "")

        self._mode = self.get_parameter("mode").value
        self._stale_timeout = float(self.get_parameter("stale_timeout_sec").value)
        self._require_all = bool(
            self.get_parameter("require_all_arm_joints").value
        )

        self._arm_sample: Optional[jsc.JointSample] = None
        self._gripper_sample: Optional[jsc.MotorSample] = None
        self._sim_sample: Optional[jsc.JointSample] = None
        self._table: Optional[CalibrationTable] = None

        self._pub = self.create_publisher(
            JointState, self.get_parameter("output_topic").value, 10
        )

        if self._mode == "real":
            calib = self.get_parameter("calibration_file").value
            if not calib:
                raise RuntimeError(
                    "real mode requires 'calibration_file' for the inverse "
                    "motor_rad -> q_jaw_m map (design doc sec. 9.1)"
                )
            # Refuses empty/non-monotonic tables by raising CalibrationError.
            self._table = CalibrationTable.from_yaml_file(calib)

            # rebotarm_msgs is only needed in real mode; import lazily so
            # the sim profile works without the Seeed driver stack.
            from rebotarm_msgs.msg import JointMotorState

            self.create_subscription(
                JointState,
                self.get_parameter("arm_input_topic").value,
                self._on_arm,
                50,
            )
            self.create_subscription(
                JointMotorState,
                self.get_parameter("gripper_input_topic").value,
                self._on_gripper_motor,
                50,
            )
        elif self._mode == "sim":
            self.create_subscription(
                JointState,
                self.get_parameter("sim_input_topic").value,
                self._on_sim,
                50,
            )
        else:
            raise RuntimeError(f"unknown mode {self._mode!r}; use sim|real")

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(1.0 / rate, self._on_timer)
        self.get_logger().info(
            f"rebot_joint_state_adapter up (mode={self._mode})"
        )

    # -- subscriptions -----------------------------------------------------

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _joint_sample(self, msg: JointState) -> jsc.JointSample:
        stamp = _stamp_to_sec(msg.header.stamp)
        if stamp == 0.0:
            stamp = self._now_sec()
        return jsc.JointSample(
            names=tuple(msg.name),
            positions=tuple(msg.position),
            velocities=tuple(msg.velocity),
            efforts=tuple(msg.effort),
            stamp_sec=stamp,
        )

    def _on_arm(self, msg: JointState) -> None:
        self._arm_sample = self._joint_sample(msg)

    def _on_gripper_motor(self, msg) -> None:
        stamp = _stamp_to_sec(msg.header.stamp)
        if stamp == 0.0:
            stamp = self._now_sec()
        self._gripper_sample = jsc.MotorSample(
            position_rad=msg.position,
            velocity_rad_s=msg.velocity,
            torque_nm=msg.torque,
            stamp_sec=stamp,
        )

    def _on_sim(self, msg: JointState) -> None:
        self._sim_sample = self._joint_sample(msg)

    # -- publishing --------------------------------------------------------

    def _on_timer(self) -> None:
        now = self._now_sec()
        try:
            merged = self._merge(now)
        except (jsc.JointStateError, Exception) as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"joint-state merge failed: {exc}", throttle_duration_sec=1.0
            )
            return
        if merged is None:
            return  # diagnostic already emitted; no fresh output (sec. 7.2)

        out = JointState()
        _sec_to_stamp(merged.stamp_sec, out.header.stamp)
        out.name = list(merged.names)
        out.position = list(merged.positions)
        out.velocity = list(merged.velocities)
        out.effort = list(merged.efforts)
        self._pub.publish(out)

    def _merge(self, now: float):
        if self._mode == "sim":
            if self._sim_sample is None or not jsc.is_fresh(
                self._sim_sample.stamp_sec, now, self._stale_timeout
            ):
                self._warn_stale("sim input")
                return None
            return jsc.merge_sim(self._sim_sample)

        if self._arm_sample is None or not jsc.is_fresh(
            self._arm_sample.stamp_sec, now, self._stale_timeout
        ):
            self._warn_stale("arm input")
            return None
        if self._gripper_sample is None or not jsc.is_fresh(
            self._gripper_sample.stamp_sec, now, self._stale_timeout
        ):
            self._warn_stale("gripper input")
            return None
        return jsc.merge_real(
            self._arm_sample,
            self._gripper_sample,
            self._table,
            require_all_arm_joints=self._require_all,
        )

    def _warn_stale(self, what: str) -> None:
        self.get_logger().warn(
            f"{what} missing or stale (> {self._stale_timeout}s); "
            "suppressing /joint_states output",
            throttle_duration_sec=1.0,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RebotJointStateAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
