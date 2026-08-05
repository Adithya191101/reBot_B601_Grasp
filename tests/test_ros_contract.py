"""ROS interface contract, tested without ROS installed.

The nodes themselves cannot be imported here (``rclpy`` is absent), but the two
things most likely to be silently wrong -- topic-name agreement and triplet
validation -- live in :mod:`ros2_iface.topics`, which is deliberately ROS-free.

The remaining ROS surface is covered by ``ros2_iface/test_jazzy_integration.py``,
which **has never been executed** and skips loudly.
"""

from __future__ import annotations

import argparse
import ast
import unittest
from pathlib import Path

from ros2_iface.topics import (
    DEPTH_ENCODING, RGB_ENCODING, TopicNames, add_topic_arguments, resolve_topics,
    validate_triplet,
)

REPO = Path(__file__).resolve().parents[1]


def _triplet(**overrides):
    base = dict(
        rgb_height=480, rgb_width=640, rgb_step=1920, rgb_encoding=RGB_ENCODING,
        rgb_is_bigendian=0, rgb_frame_id="cam", rgb_stamp_ns=1000,
        depth_height=480, depth_width=640, depth_step=2560,
        depth_encoding=DEPTH_ENCODING, depth_is_bigendian=0, depth_frame_id="cam",
        depth_stamp_ns=1000,
        info_height=480, info_width=640, info_frame_id="cam", info_stamp_ns=1000,
    )
    base.update(overrides)
    return base


class TestTopicAgreement(unittest.TestCase):
    def test_defaults_are_absolute_and_shared(self):
        t = TopicNames()
        for name in (t.rgb, t.depth, t.camera_info, t.grasp_pose, t.clock):
            self.assertTrue(name.startswith("/"), f"{name} is not absolute")

    def test_namespace_applies_to_data_topics_but_not_clock(self):
        t = TopicNames.with_namespace("/itest")
        self.assertEqual(t.rgb, "/itest/rgb")
        self.assertEqual(t.depth, "/itest/depth")
        self.assertEqual(t.camera_info, "/itest/camera_info")
        # /clock is global by convention; namespacing it breaks sim time silently.
        self.assertEqual(t.clock, "/clock")

    def test_both_nodes_expose_the_same_topic_cli(self):
        pub = argparse.ArgumentParser()
        sub = argparse.ArgumentParser()
        add_topic_arguments(pub)
        add_topic_arguments(sub)
        pub_args = pub.parse_args(["--namespace", "/x"])
        sub_args = sub.parse_args(["--namespace", "/x"])
        self.assertEqual(resolve_topics(pub_args), resolve_topics(sub_args))

    def test_neither_node_hardcodes_a_private_topic_name(self):
        """Regression: the publisher advertised ~/rgb while the node heard rgb."""
        for filename in ("dataset_publisher.py", "grasp_node.py"):
            source = (REPO / "ros2_iface" / filename).read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertFalse(
                        node.value.startswith("~/"),
                        f"{filename} hardcodes private topic {node.value!r}",
                    )

    def test_documented_invocation_is_module_form(self):
        """``python3 path/to/node.py`` breaks the package-relative imports."""
        for filename in ("dataset_publisher.py", "grasp_node.py"):
            source = (REPO / "ros2_iface" / filename).read_text()
            module = f"ros2_iface.{filename[:-3]}"
            self.assertIn(f"-m {module}", source,
                          f"{filename} must document `python3 -m {module}`")
            self.assertNotIn(f"python3 ros2_iface/{filename}", source)

    def test_nodes_use_absolute_package_imports(self):
        """A relative import fails under ``python3 -m`` from the repo root."""
        source = (REPO / "ros2_iface" / "grasp_node.py").read_text()
        self.assertNotIn("from .dataset_publisher", source)
        self.assertIn("from ros2_iface.dataset_publisher", source)


class TestTripletValidation(unittest.TestCase):
    def test_consistent_triplet_accepted(self):
        self.assertIsNone(validate_triplet(**_triplet()))

    def test_millimetre_depth_encoding_refused(self):
        why = validate_triplet(**_triplet(depth_encoding="16UC1"))
        self.assertIn("refusing to guess units", why)

    def test_wrong_rgb_encoding_refused(self):
        self.assertIn("rgb encoding", validate_triplet(**_triplet(rgb_encoding="bgr8")))

    def test_mismatched_stamps_refused(self):
        why = validate_triplet(**_triplet(depth_stamp_ns=1001))
        self.assertIn("stamps differ", why)

    def test_mismatched_frame_ids_refused(self):
        why = validate_triplet(**_triplet(info_frame_id="other_cam"))
        self.assertIn("frame_ids differ", why)

    def test_empty_frame_id_refused(self):
        why = validate_triplet(**_triplet(rgb_frame_id="", depth_frame_id="",
                                          info_frame_id=""))
        self.assertIn("empty frame_id", why)

    def test_size_mismatch_between_rgb_and_depth_refused(self):
        why = validate_triplet(**_triplet(depth_height=240, depth_step=2560))
        self.assertIn("!= depth", why)

    def test_camera_info_size_mismatch_refused(self):
        why = validate_triplet(**_triplet(info_width=320))
        self.assertIn("intrinsics would be misapplied", why)

    def test_padded_rgb_step_refused(self):
        self.assertIn("step", validate_triplet(**_triplet(rgb_step=1928)))

    def test_wrong_depth_step_refused(self):
        self.assertIn("depth step", validate_triplet(**_triplet(depth_step=1920)))

    def test_big_endian_refused(self):
        why = validate_triplet(**_triplet(rgb_is_bigendian=1))
        self.assertIn("big-endian", why)


class TestIntegrationGateHonesty(unittest.TestCase):
    def test_gate_exists_and_declares_it_has_not_run(self):
        source = (REPO / "ros2_iface" / "test_jazzy_integration.py").read_text()
        self.assertIn("NEVER EXECUTED", source)
        self.assertIn("skipUnless", source)
        # It must skip on absence rather than quietly passing.
        self.assertIn("do not treat a skip as a pass", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
