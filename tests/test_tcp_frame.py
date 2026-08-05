"""Pure tests for the explicit vision-grasp -> B601 TCP control boundary."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from grasp_smoke.geometry import (
    make_transform,
    rotation_about_axis,
    rotation_from_quaternion,
    transform_point,
)
from grasp_smoke.pose_msg import (
    grasp_to_b601_tcp_pose_stamped,
    grasp_to_pose_stamped,
    vision_grasp_basis_to_b601_tcp_rotation,
)


def _orientation_matrix(msg: dict) -> np.ndarray:
    orientation = msg["pose"]["orientation"]
    return rotation_from_quaternion(
        np.array(
            [orientation["x"], orientation["y"], orientation["z"], orientation["w"]],
            dtype=np.float64,
        )
    )


class TestB601TcpBasis(unittest.TestCase):
    def test_axis_signs_match_vendor_tcp_convention(self):
        # Vision basis columns: grip=+X, open=+Y, approach=+Z.
        tcp = vision_grasp_basis_to_b601_tcp_rotation(np.eye(3))

        np.testing.assert_allclose(tcp[:, 0], [0.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(tcp[:, 1], [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(tcp[:, 2], [1.0, 0.0, 0.0], atol=1e-12)

    def test_noisy_basis_is_orthonormal_and_right_handed(self):
        vision = np.column_stack(
            [
                np.array([2.0, 0.1, 0.0]),
                np.array([0.2, 3.0, 0.4]),
                np.array([0.1, 0.0, 4.0]),
            ]
        )
        tcp = vision_grasp_basis_to_b601_tcp_rotation(vision)

        np.testing.assert_allclose(tcp.T @ tcp, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(tcp)), 1.0, places=12)
        np.testing.assert_allclose(
            tcp[:, 0], -vision[:, 2] / np.linalg.norm(vision[:, 2]), atol=1e-12
        )

    def test_degenerate_open_axis_is_rejected(self):
        vision = np.column_stack(
            [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 2.0]),
                np.array([0.0, 0.0, 1.0]),
            ]
        )
        with self.assertRaisesRegex(ValueError, "open axis is parallel"):
            vision_grasp_basis_to_b601_tcp_rotation(vision)


class TestB601TcpPoseStamped(unittest.TestCase):
    def test_tcp_pose_is_transformed_into_base_frame(self):
        R_base_cam = rotation_about_axis(np.array([0.0, 0.0, 1.0]), np.deg2rad(90.0))
        T_base_cam = make_transform(R_base_cam, np.array([0.4, -0.2, 0.7]))
        frame = SimpleNamespace(
            T_base_cam=T_base_cam,
            stamp_ns=1_700_000_000_123_456_789,
            base_frame_id="base_link",
            scene_id="tcp_test",
        )
        estimate = SimpleNamespace(
            is_valid=True,
            position=np.array([0.1, 0.2, 0.6]),
            rotation=np.eye(3),
            open_axis=np.array([0.0, 1.0, 0.0]),
            jaw_width_m=0.05,
            z_m=0.6,
        )

        msg = grasp_to_b601_tcp_pose_stamped(estimate, frame, branch="B")
        tcp_cam = vision_grasp_basis_to_b601_tcp_rotation(estimate.rotation)

        position = msg["pose"]["position"]
        np.testing.assert_allclose(
            [position["x"], position["y"], position["z"]],
            transform_point(T_base_cam, estimate.position),
            atol=1e-12,
        )
        np.testing.assert_allclose(_orientation_matrix(msg), R_base_cam @ tcp_cam, atol=1e-12)
        self.assertEqual(msg["header"]["frame_id"], "base_link")
        self.assertEqual(msg["_meta"]["orientation_convention"], "b601_tcp")

    def test_legacy_pose_keeps_vision_basis_semantics(self):
        frame = SimpleNamespace(
            T_base_cam=np.eye(4),
            stamp_ns=1,
            base_frame_id="base_link",
            scene_id="legacy_test",
        )
        estimate = SimpleNamespace(
            is_valid=True,
            position=np.array([0.0, 0.0, 0.5]),
            rotation=np.eye(3),
            open_axis=np.array([0.0, 1.0, 0.0]),
            jaw_width_m=0.05,
            z_m=0.5,
        )

        legacy = grasp_to_pose_stamped(estimate, frame)
        tcp = grasp_to_b601_tcp_pose_stamped(estimate, frame)

        np.testing.assert_allclose(_orientation_matrix(legacy), np.eye(3), atol=1e-12)
        self.assertFalse(np.allclose(_orientation_matrix(tcp), np.eye(3)))
        self.assertEqual(legacy["_meta"]["orientation_convention"], "vision_grasp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
