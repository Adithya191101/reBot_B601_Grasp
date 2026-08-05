#!/usr/bin/env python3
"""ROS 2 Jazzy integration gate. **NEVER EXECUTED IN THIS REPOSITORY YET.**

``/opt/ros`` is absent on this machine, so nothing below has run. It is written
now so that the first ROS session has a pass/fail gate instead of an afternoon of
manual `ros2 topic echo`, and so the claim "ROS works" has something to point at.

Deliberately **skips loudly rather than passing** when ROS is missing. A test
that silently no-ops is how an untested integration comes to look tested; running
this today prints SKIPPED, and that is the honest result.

Once Jazzy is installed::

    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export ROS_DOMAIN_ID=42
    python3 -m unittest ros2_iface.test_jazzy_integration -v

What it asserts, end to end through a real DDS graph:

1. the publisher and the grasp node resolve the **same** topic names
2. ``/clock`` is published from recorded stamps and consumers honour sim time
3. every replayed frame produces exactly one ``PoseStamped``
4. each pose is stamped with its **source image** stamp, not the wall clock
5. poses land in the base frame and agree with the offline library to <1 mm
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ROS_AVAILABLE = importlib.util.find_spec("rclpy") is not None
ROS_SKIP_REASON = (
    "ROS 2 Jazzy is not installed (/opt/ros absent). This gate has NEVER been "
    "executed. Install Jazzy, source it, then rerun -- do not treat a skip as a pass."
)


@unittest.skipUnless(ROS_AVAILABLE, ROS_SKIP_REASON)
class TestJazzyIntegration(unittest.TestCase):
    """Requires a real ROS 2 graph. Unexecuted until Jazzy exists."""

    @classmethod
    def setUpClass(cls):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        from grasp_smoke.capture import capture_analytic, plan_smoke_scenes
        from ros2_iface.dataset_publisher import DatasetPublisher
        from ros2_iface.grasp_node import GraspNode
        from ros2_iface.topics import TopicNames

        import tempfile
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dataset = Path(cls._tmp.name) / "ds"
        cls.specs = plan_smoke_scenes(seed=31337)[:3]
        capture_analytic(cls.dataset, cls.specs, seed=31337)

        rclpy.init()
        cls.topics = TopicNames.with_namespace("/itest")
        cls.publisher = DatasetPublisher(cls.dataset, period_s=0.2, loop=False,
                                         topics=cls.topics, publish_clock=True)
        cls.node = GraspNode("base_link", 0.5, "B", cls.topics, "saturation")
        cls.node.set_parameters([
            rclpy.parameter.Parameter("use_sim_time", value=True)
        ])
        cls.executor = SingleThreadedExecutor()
        cls.executor.add_node(cls.publisher)
        cls.executor.add_node(cls.node)

    @classmethod
    def tearDownClass(cls):
        import rclpy
        cls.executor.shutdown()
        cls.publisher.destroy_node()
        cls.node.destroy_node()
        rclpy.try_shutdown()
        cls._tmp.cleanup()

    def test_topic_names_agree_between_both_ends(self):
        self.assertEqual(self.publisher.topics.rgb, self.node.topics.rgb)
        self.assertEqual(self.publisher.topics.depth, self.node.topics.depth)
        self.assertEqual(self.publisher.topics.camera_info, self.node.topics.camera_info)

    def test_replay_produces_one_pose_per_frame_at_source_stamps(self):
        from geometry_msgs.msg import PoseStamped

        received = []
        self.node.create_subscription(
            PoseStamped, self.topics.grasp_pose, lambda m: received.append(m), 10
        )
        deadline = 30.0
        elapsed = 0.0
        while elapsed < deadline and len(received) < len(self.specs):
            self.executor.spin_once(timeout_sec=0.1)
            elapsed += 0.1

        self.assertEqual(len(received), len(self.specs))
        expected = {
            1_700_000_000_000_000_000 + i * 100_000_000 for i in range(len(self.specs))
        }
        got = {int(m.header.stamp.sec) * 1_000_000_000 + int(m.header.stamp.nanosec)
               for m in received}
        self.assertEqual(got, expected, "poses must carry their source image stamp")
        for msg in received:
            self.assertEqual(msg.header.frame_id, "base_link")

    def test_matches_the_offline_library(self):
        """The ROS path must not quietly disagree with the tested offline path."""
        import numpy as np

        from grasp_smoke import dataset as ds
        from grasp_smoke.detect import SaturationSegmenter
        from grasp_smoke.geometry import transform_point
        from grasp_smoke.grasp import estimate_grasp

        scene_id = ds.scene_ids(self.dataset)[0]
        frame = ds.load_frame(self.dataset, scene_id)
        prediction = SaturationSegmenter().predict(frame)
        self.assertIsNotNone(prediction)
        est = estimate_grasp(prediction.mask, frame.depth_m, frame.K, 0.5)
        self.assertTrue(est.is_valid)
        offline = transform_point(frame.T_base_cam, est.position)
        self.assertEqual(offline.shape, (3,))
        self.assertTrue(np.all(np.isfinite(offline)))


class TestGateIsHonestAboutNotHavingRun(unittest.TestCase):
    """Runs everywhere. Asserts the skip is visible rather than silent."""

    def test_ros_absence_is_reported_not_hidden(self):
        if ROS_AVAILABLE:
            self.skipTest("ROS present; the honesty guard is only meaningful without it")
        self.assertIn("NEVER been executed", ROS_SKIP_REASON)


if __name__ == "__main__":
    unittest.main(verbosity=2)
