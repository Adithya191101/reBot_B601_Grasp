#!/usr/bin/env python3
"""Grasp node: RGB-D + CameraInfo + TF -> ``geometry_msgs/PoseStamped``.

Subscribes to the dataset publisher (or, later, to a live camera -- the node does
not care which) and republishes a grasp pose in the base frame.

Deliberately thin. All of the geometry lives in :mod:`grasp_smoke.grasp` and all
of the message construction in :mod:`grasp_smoke.pose_msg`, both of which are
unit-tested without ROS. This file only adapts message types and looks up TF, so
there is very little here that can be wrong in a way the tests would not catch.

Topic names come from :mod:`ros2_iface.topics`, shared with the publisher.
Synchronized triplets are **queued until TF is available at their exact stamp**
rather than dropped: on startup the transform routinely arrives after the first
image, and dropping those frames silently loses the beginning of every run.

**Not run in this session: ROS 2 is not installed** (``/opt/ros`` absent). Written
and reviewed, unexecuted.

    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    python3 -m ros2_iface.grasp_node --predictor saturation --use-sim-time
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

from collections import deque  # noqa: E402

from grasp_smoke.capture import BASELINE_DEPTH_QUANTILE  # noqa: E402
from grasp_smoke.dataset import Frame  # noqa: E402
from grasp_smoke.detect import PREDICTOR_CHOICES, build_predictor  # noqa: E402
from grasp_smoke.geometry import make_transform, rotation_from_quaternion  # noqa: E402
from grasp_smoke.grasp import estimate_grasp  # noqa: E402
from grasp_smoke.pose_msg import grasp_to_pose_stamped  # noqa: E402

from ros2_iface.dataset_publisher import SENSOR_QOS  # noqa: E402
from ros2_iface.topics import (  # noqa: E402
    add_topic_arguments, resolve_topics, validate_triplet,
)


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class GraspNode(Node):
    #: Frames wait this long for their transform before being given up on.
    TF_WAIT_TIMEOUT_S = 5.0
    MAX_PENDING = 30

    def __init__(self, base_frame: str, depth_quantile: float, branch: str,
                 topics, predictor_kind: str = "saturation"):
        super().__init__("grasp_node")
        self.base_frame = base_frame
        self.depth_quantile = depth_quantile
        self.branch = branch
        self.topics = topics
        self.pending = deque()          # triplets awaiting TF at their stamp
        # Explicit, never inferred: build_predictor raises rather than degrading.
        self.predictor = build_predictor(predictor_kind)
        self.get_logger().info(
            f"predictor={self.predictor.config.name} "
            f"(provisional={self.predictor.config.provisional})"
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PoseStamped, topics.grasp_pose, 10)
        self.get_logger().info(
            f"topics: rgb={topics.rgb} depth={topics.depth} "
            f"camera_info={topics.camera_info} -> {topics.grasp_pose}"
        )
        if not self.get_parameter("use_sim_time").value:
            self.get_logger().warn(
                "use_sim_time is false. The dataset publisher stamps messages with "
                "recorded capture time, so TF lookups at the image stamp will fail. "
                "Launch with --use-sim-time (or -p use_sim_time:=true)."
            )

        self.sync = ApproximateTimeSynchronizer(
            [
                Subscriber(self, Image, topics.rgb, qos_profile=SENSOR_QOS),
                Subscriber(self, Image, topics.depth, qos_profile=SENSOR_QOS),
                Subscriber(self, CameraInfo, topics.camera_info, qos_profile=SENSOR_QOS),
            ],
            queue_size=10,
            slop=0.02,
        )
        self.sync.registerCallback(self.on_frame)
        # Retry queued frames independently of arrival, so a transform that shows
        # up late still rescues the frames that were waiting on it.
        self.retry_timer = self.create_timer(0.1, self.drain_pending)

    def on_frame(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        problem = validate_triplet(
            rgb_msg.height, rgb_msg.width, rgb_msg.step, rgb_msg.encoding,
            rgb_msg.is_bigendian, rgb_msg.header.frame_id, _stamp_ns(rgb_msg.header.stamp),
            depth_msg.height, depth_msg.width, depth_msg.step, depth_msg.encoding,
            depth_msg.is_bigendian, depth_msg.header.frame_id,
            _stamp_ns(depth_msg.header.stamp),
            info_msg.height, info_msg.width, info_msg.header.frame_id,
            _stamp_ns(info_msg.header.stamp),
        )
        if problem is not None:
            self.get_logger().warn(f"rejecting triplet: {problem}")
            return

        self.pending.append((rgb_msg, depth_msg, info_msg, self.get_clock().now()))
        if len(self.pending) > self.MAX_PENDING:
            dropped = self.pending.popleft()
            self.get_logger().warn(
                f"pending queue full ({self.MAX_PENDING}); dropped frame at "
                f"{_stamp_ns(dropped[0].header.stamp)}"
            )
        self.drain_pending()

    def drain_pending(self) -> None:
        """Process every queued frame whose transform has now arrived."""
        still_waiting = deque()
        while self.pending:
            rgb_msg, depth_msg, info_msg, queued_at = self.pending.popleft()
            try:
                # Exact stamp. Never "now" -- that silently uses the wrong pose.
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, rgb_msg.header.frame_id, rgb_msg.header.stamp
                )
            except Exception as exc:                            # noqa: BLE001
                waited = (self.get_clock().now() - queued_at).nanoseconds / 1e9
                if waited > self.TF_WAIT_TIMEOUT_S:
                    self.get_logger().warn(
                        f"giving up on frame {_stamp_ns(rgb_msg.header.stamp)} after "
                        f"{waited:.1f}s waiting for TF {self.base_frame} <- "
                        f"{rgb_msg.header.frame_id}: {exc}"
                    )
                else:
                    still_waiting.append((rgb_msg, depth_msg, info_msg, queued_at))
                continue
            self._process(rgb_msg, depth_msg, info_msg, tf)
        self.pending = still_waiting

    def _process(self, rgb_msg, depth_msg, info_msg, tf) -> None:
        rgb = np.frombuffer(rgb_msg.data, np.uint8).reshape(
            rgb_msg.height, rgb_msg.width, 3
        )
        depth = np.frombuffer(depth_msg.data, np.float32).reshape(
            depth_msg.height, depth_msg.width
        ).astype(np.float64)
        K = np.asarray(info_msg.k, dtype=np.float64).reshape(3, 3)

        t = tf.transform.translation
        q = tf.transform.rotation
        T_base_cam = make_transform(
            rotation_from_quaternion(np.array([q.x, q.y, q.z, q.w])),
            np.array([t.x, t.y, t.z]),
        )

        frame = Frame(
            scene_id="live", rgb=rgb, depth_m=depth, K=K,
            width=info_msg.width, height=info_msg.height,
            stamp_ns=_stamp_ns(rgb_msg.header.stamp),
            T_base_cam=T_base_cam, base_frame_id=self.base_frame,
            camera_frame_id=rgb_msg.header.frame_id,
        )

        prediction = self.predictor.predict(frame)
        if prediction is None:
            self.get_logger().info("no detection")
            return

        est = estimate_grasp(prediction.mask, frame.depth_m, frame.K, self.depth_quantile)
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
            f"{msg.pose.position.z:.3f}) in {msg.header.frame_id} "
            f"class={prediction.class_name} conf={prediction.confidence}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--depth-quantile", type=float, default=BASELINE_DEPTH_QUANTILE)
    ap.add_argument("--branch", default="B")
    ap.add_argument("--predictor", choices=PREDICTOR_CHOICES, default="saturation")
    add_topic_arguments(ap)
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = GraspNode(args.base_frame, args.depth_quantile, args.branch,
                     resolve_topics(args), args.predictor)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
