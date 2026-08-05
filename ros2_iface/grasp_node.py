#!/usr/bin/env python3
"""Grasp node: RGB-D + CameraInfo + TF -> ``geometry_msgs/PoseStamped``.

Subscribes to the dataset publisher (or, later, to a live camera -- the node does
not care which) and republishes a grasp pose in the base frame.

Deliberately thin. All of the geometry lives in :mod:`grasp_smoke.grasp` and all
of the message construction in :mod:`grasp_smoke.pose_msg`, both of which are
unit-tested without ROS. This file only adapts message types and looks up TF, so
there is very little here that can be wrong in a way the tests would not catch.

**Not run in this session: ROS 2 is not installed** (``/opt/ros`` absent). Written
and reviewed, unexecuted.

    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    python3 ros2_iface/grasp_node.py --branch B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import rclpy  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from message_filters import ApproximateTimeSynchronizer, Subscriber  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from tf2_ros import Buffer, TransformListener  # noqa: E402

from grasp_smoke.dataset import Frame  # noqa: E402
from grasp_smoke.detect import build_predictor  # noqa: E402
from grasp_smoke.geometry import make_transform, rotation_from_quaternion  # noqa: E402
from grasp_smoke.grasp import estimate_grasp  # noqa: E402
from grasp_smoke.pose_msg import grasp_to_pose_stamped  # noqa: E402

from .dataset_publisher import SENSOR_QOS  # noqa: E402


class GraspNode(Node):
    def __init__(self, base_frame: str, depth_quantile: float, branch: str):
        super().__init__("grasp_node")
        self.base_frame = base_frame
        self.depth_quantile = depth_quantile
        self.branch = branch
        self.predictor = build_predictor()
        self.get_logger().info(
            f"predictor={self.predictor.config.name} "
            f"(provisional={self.predictor.config.provisional})"
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PoseStamped, "~/grasp_pose", 10)

        self.sync = ApproximateTimeSynchronizer(
            [
                Subscriber(self, Image, "rgb", qos_profile=SENSOR_QOS),
                Subscriber(self, Image, "depth", qos_profile=SENSOR_QOS),
                Subscriber(self, CameraInfo, "camera_info", qos_profile=SENSOR_QOS),
            ],
            queue_size=10,
            slop=0.02,
        )
        self.sync.registerCallback(self.on_frame)

    def on_frame(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        if rgb_msg.encoding != "rgb8":
            self.get_logger().warn(f"expected rgb8, got {rgb_msg.encoding}")
            return
        if depth_msg.encoding != "32FC1":
            # 16UC1 would be millimetres; silently treating it as metres is the
            # classic 1000x error, so refuse rather than guess.
            self.get_logger().warn(
                f"expected 32FC1 metres, got {depth_msg.encoding}; refusing to guess units"
            )
            return

        rgb = np.frombuffer(rgb_msg.data, np.uint8).reshape(rgb_msg.height, rgb_msg.width, 3)
        depth = np.frombuffer(depth_msg.data, np.float32).reshape(
            depth_msg.height, depth_msg.width
        ).astype(np.float64)
        K = np.asarray(info_msg.k, dtype=np.float64).reshape(3, 3)

        # Look TF up AT THE IMAGE STAMP, never at "now".
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, rgb_msg.header.frame_id, rgb_msg.header.stamp
            )
        except Exception as exc:                                # noqa: BLE001
            self.get_logger().warn(f"TF unavailable at image stamp: {exc}")
            return

        t = tf.transform.translation
        q = tf.transform.rotation
        T_base_cam = make_transform(
            rotation_from_quaternion(np.array([q.x, q.y, q.z, q.w])),
            np.array([t.x, t.y, t.z]),
        )

        stamp_ns = int(rgb_msg.header.stamp.sec) * 1_000_000_000 + int(
            rgb_msg.header.stamp.nanosec
        )
        frame = Frame(
            scene_id="live", rgb=rgb, depth_m=depth, K=K,
            width=info_msg.width, height=info_msg.height, stamp_ns=stamp_ns,
            T_base_cam=T_base_cam, base_frame_id=self.base_frame,
            camera_frame_id=rgb_msg.header.frame_id,
        )

        mask = self.predictor.predict(frame)
        if mask is None:
            self.get_logger().info("no detection")
            return

        est = estimate_grasp(mask, frame.depth_m, frame.K, self.depth_quantile)
        if not est.is_valid:
            self.get_logger().info(f"grasp rejected: {est.rejected_reason}")
            return

        payload = grasp_to_pose_stamped(est, frame, branch=self.branch)
        msg = PoseStamped()
        msg.header.stamp = rgb_msg.header.stamp
        msg.header.frame_id = payload["header"]["frame_id"]
        msg.pose.position.x = payload["pose"]["position"]["x"]
        msg.pose.position.y = payload["pose"]["position"]["y"]
        msg.pose.position.z = payload["pose"]["position"]["z"]
        msg.pose.orientation.x = payload["pose"]["orientation"]["x"]
        msg.pose.orientation.y = payload["pose"]["orientation"]["y"]
        msg.pose.orientation.z = payload["pose"]["orientation"]["z"]
        msg.pose.orientation.w = payload["pose"]["orientation"]["w"]
        self.pub.publish(msg)
        self.get_logger().info(
            f"grasp @ ({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, "
            f"{msg.pose.position.z:.3f}) in {msg.header.frame_id}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--depth-quantile", type=float, default=0.75)
    ap.add_argument("--branch", default="B")
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = GraspNode(args.base_frame, args.depth_quantile, args.branch)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
