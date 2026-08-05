#!/usr/bin/env python3
"""Deterministic ROS 2 Jazzy publisher that replays the file-native dataset.

Publishes, all stamped with the **recorded** capture timestamp rather than the
wall clock:

* ``~/rgb``            ``sensor_msgs/Image``       rgb8
* ``~/depth``          ``sensor_msgs/Image``       32FC1, optical-axis Z, metres
* ``~/camera_info``    ``sensor_msgs/CameraInfo``
* ``/tf``              base -> camera optical frame

Determinism is the requirement (PLAN.md 5.2.2): same dataset in, same messages
out, every run. Scenes advance on a fixed-period timer in manifest order, and
nothing is generated -- every field is read from disk.

**Not run in this session: ROS 2 is not installed** (``/opt/ros`` absent), so
this file is written and import-checked but unexecuted. All message *content* is
built and unit-tested ROS-free in :mod:`grasp_smoke.pose_msg` and
``tests/test_pose_msg.py``; this module is the thin shell that copies those
fields into real messages.

Run, once Jazzy is installed::

    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    python3 ros2_iface/dataset_publisher.py --dataset artifacts/smoke/dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import rclpy  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from tf2_ros import TransformBroadcaster  # noqa: E402

from grasp_smoke import dataset as ds  # noqa: E402
from grasp_smoke.geometry import quaternion_from_matrix  # noqa: E402

#: Sensor-data QoS. A mismatch here against a default-QoS subscriber is the
#: classic silent drop the S0 gate checks for (PLAN.md 4).
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)


class DatasetPublisher(Node):
    def __init__(self, dataset_root: Path, period_s: float, loop: bool):
        super().__init__("dataset_publisher")
        self.root = Path(dataset_root)
        self.scene_ids = ds.scene_ids(self.root)
        if not self.scene_ids:
            raise RuntimeError(f"no scenes in {self.root}")
        self.loop = loop
        self.index = 0

        bad = ds.verify_checksums(self.root)
        if bad:
            raise RuntimeError(f"dataset checksum mismatch: {bad}")
        self.get_logger().info(
            f"replaying {len(self.scene_ids)} scenes from {self.root} (checksums OK)"
        )

        self.pub_rgb = self.create_publisher(Image, "~/rgb", SENSOR_QOS)
        self.pub_depth = self.create_publisher(Image, "~/depth", SENSOR_QOS)
        self.pub_info = self.create_publisher(CameraInfo, "~/camera_info", SENSOR_QOS)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(period_s, self.publish_next)

    @staticmethod
    def _stamp(msg, stamp_ns: int) -> None:
        msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
        msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)

    def publish_next(self) -> None:
        if self.index >= len(self.scene_ids):
            if not self.loop:
                self.get_logger().info("dataset exhausted")
                self.timer.cancel()
                return
            self.index = 0

        scene_id = self.scene_ids[self.index]
        self.index += 1
        frame = ds.load_frame(self.root, scene_id)   # sensor side only

        rgb = Image()
        self._stamp(rgb, frame.stamp_ns)
        rgb.header.frame_id = frame.camera_frame_id
        rgb.height, rgb.width = frame.height, frame.width
        rgb.encoding = "rgb8"
        rgb.is_bigendian = 0
        rgb.step = frame.width * 3
        rgb.data = frame.rgb.astype(np.uint8).tobytes()

        depth = Image()
        self._stamp(depth, frame.stamp_ns)
        depth.header.frame_id = frame.camera_frame_id
        depth.height, depth.width = frame.height, frame.width
        depth.encoding = "32FC1"      # metres, optical-axis Z
        depth.is_bigendian = 0
        depth.step = frame.width * 4
        depth.data = frame.depth_m.astype(np.float32).tobytes()

        info = CameraInfo()
        self._stamp(info, frame.stamp_ns)
        info.header.frame_id = frame.camera_frame_id
        info.height, info.width = frame.height, frame.width
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [float(v) for v in np.asarray(frame.K).flatten()]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [
            frame.K[0, 0], 0.0, frame.K[0, 2], 0.0,
            0.0, frame.K[1, 1], frame.K[1, 2], 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        tf = TransformStamped()
        self._stamp(tf, frame.stamp_ns)
        tf.header.frame_id = frame.base_frame_id
        tf.child_frame_id = frame.camera_frame_id
        t = frame.T_base_cam[:3, 3]
        q = quaternion_from_matrix(frame.T_base_cam[:3, :3])
        tf.transform.translation.x = float(t[0])
        tf.transform.translation.y = float(t[1])
        tf.transform.translation.z = float(t[2])
        tf.transform.rotation.x = float(q[0])
        tf.transform.rotation.y = float(q[1])
        tf.transform.rotation.z = float(q[2])
        tf.transform.rotation.w = float(q[3])

        # TF first: a consumer that receives an image before its transform has to
        # buffer or drop it.
        self.tf_broadcaster.sendTransform(tf)
        self.pub_info.publish(info)
        self.pub_rgb.publish(rgb)
        self.pub_depth.publish(depth)
        self.get_logger().info(f"published {scene_id} @ {frame.stamp_ns}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = DatasetPublisher(args.dataset, args.period, args.loop)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
