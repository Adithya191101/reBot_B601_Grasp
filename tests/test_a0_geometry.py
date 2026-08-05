"""A0 analytic red tests (PLAN.md 5.2.4).

Fronto-parallel, constant-depth fixtures with closed-form answers. Under these
conditions the vendor algorithm is *exact*, so the strict bars apply:

    position error   <= 1 mm
    opening axis     <= 1 deg

A failure here is a bug in this codebase -- not the algorithm approximating, and
not the simulator. That distinction is the entire reason A0 is separate from A1
and A2.

Uses stdlib ``unittest`` so it runs with no additional packages installed.
"""

from __future__ import annotations

import unittest

import numpy as np

from grasp_smoke.geometry import (
    backproject,
    invert_transform,
    make_intrinsics,
    normalize,
    opening_axis_error_rad,
    project,
    quaternion_from_matrix,
    rotation_from_quaternion,
    transform_direction,
    transform_point,
)
from grasp_smoke.grasp import estimate_grasp
from grasp_smoke.render import fronto_parallel_fixture, render

POSITION_TOL_M = 1e-3        # 1 mm
ANGLE_TOL_RAD = np.deg2rad(1.0)


class TestPinholeGeometry(unittest.TestCase):
    def test_backproject_project_roundtrip(self):
        K = make_intrinsics(600.0, 600.0, 319.5, 239.5)
        for u, v, z in [(319.5, 239.5, 0.5), (0.0, 0.0, 1.2), (639.0, 479.0, 0.31)]:
            p = backproject(u, v, z, K)
            self.assertAlmostEqual(float(p[2]), z, places=12)
            uv = project(p, K)
            self.assertAlmostEqual(float(uv[0]), u, places=9)
            self.assertAlmostEqual(float(uv[1]), v, places=9)

    def test_backproject_is_optical_axis_z_not_range(self):
        """A point off-axis at depth z is farther than z from the camera."""
        K = make_intrinsics(600.0, 600.0, 319.5, 239.5)
        p = backproject(0.0, 0.0, 1.0, K)
        self.assertAlmostEqual(float(p[2]), 1.0, places=12)
        self.assertGreater(float(np.linalg.norm(p)), 1.0)

    def test_zero_depth_projection_rejected(self):
        K = make_intrinsics(600.0, 600.0, 319.5, 239.5)
        with self.assertRaises(ValueError):
            project(np.array([0.1, 0.1, 0.0]), K)


class TestTransforms(unittest.TestCase):
    def test_inverse_roundtrip(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            q = normalize(rng.normal(size=4))
            T = np.eye(4)
            T[:3, :3] = rotation_from_quaternion(q)
            T[:3, 3] = rng.normal(size=3)
            p = rng.normal(size=3)
            back = transform_point(invert_transform(T), transform_point(T, p))
            np.testing.assert_allclose(back, p, atol=1e-12)

    def test_direction_ignores_translation(self):
        T = np.eye(4)
        T[:3, 3] = [10.0, -4.0, 3.0]
        v = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(transform_direction(T, v), v, atol=1e-12)

    def test_quaternion_roundtrip(self):
        rng = np.random.default_rng(11)
        for _ in range(20):
            q = normalize(rng.normal(size=4))
            R = rotation_from_quaternion(q)
            R2 = rotation_from_quaternion(quaternion_from_matrix(R))
            np.testing.assert_allclose(R, R2, atol=1e-9)


class TestOpeningAxisMetric(unittest.TestCase):
    def test_antipodal_symmetry_is_zero_error(self):
        o = normalize(np.array([0.3, -0.9, 0.1]))
        self.assertAlmostEqual(opening_axis_error_rad(o, -o), 0.0, places=12)
        self.assertAlmostEqual(opening_axis_error_rad(o, o), 0.0, places=12)

    def test_perpendicular_is_ninety_degrees(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(np.rad2deg(opening_axis_error_rad(a, b)), 90.0, places=9)

    def test_no_nan_from_floating_point_overshoot(self):
        """|dot| can exceed 1 by an ulp; the clamp must absorb it."""
        a = np.array([1.0, 0.0, 0.0])
        self.assertFalse(np.isnan(opening_axis_error_rad(a, a * (1 + 1e-16))))

    def test_degenerate_axis_rejected(self):
        with self.assertRaises(ValueError):
            opening_axis_error_rad(np.zeros(3), np.array([1.0, 0.0, 0.0]))


class TestA0GraspRecovery(unittest.TestCase):
    """The load-bearing A0 test: full pipeline on exact fronto-parallel input."""

    def _run(self, yaw_deg: float, z_m: float = 0.60):
        target, T_base_cam, K, exp_pos, exp_axis = fronto_parallel_fixture(
            z_m=z_m, yaw_deg=yaw_deg
        )
        scene = render(target, T_base_cam, K, 640, 480, np.random.default_rng(0))
        est = estimate_grasp(scene.mask, scene.depth_m, K, depth_quantile=0.75)
        self.assertTrue(est.is_valid, f"rejected: {est.rejected_reason}")

        pos_base = transform_point(T_base_cam, est.position)
        axis_base = transform_direction(T_base_cam, est.open_axis)
        pos_err = float(np.linalg.norm(pos_base - exp_pos))
        ang_err = opening_axis_error_rad(axis_base, exp_axis)
        return pos_err, ang_err

    def test_constant_depth_recovery_across_yaw(self):
        for yaw in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]:
            with self.subTest(yaw=yaw):
                pos_err, ang_err = self._run(yaw)
                self.assertLessEqual(
                    pos_err, POSITION_TOL_M,
                    f"yaw={yaw}: position error {pos_err*1000:.3f} mm > 1 mm",
                )
                self.assertLessEqual(
                    ang_err, ANGLE_TOL_RAD,
                    f"yaw={yaw}: opening-axis error {np.rad2deg(ang_err):.3f} deg > 1 deg",
                )

    def test_constant_depth_recovery_across_range(self):
        for z in [0.45, 0.60, 0.80]:
            with self.subTest(z=z):
                pos_err, ang_err = self._run(37.0, z_m=z)
                self.assertLessEqual(pos_err, POSITION_TOL_M)
                self.assertLessEqual(ang_err, ANGLE_TOL_RAD)

    def test_depth_quantile_is_irrelevant_at_constant_depth(self):
        """All quantiles agree when the visible surface is planar and normal."""
        target, T_base_cam, K, exp_pos, _ = fronto_parallel_fixture(yaw_deg=25.0)
        scene = render(target, T_base_cam, K, 640, 480, np.random.default_rng(0))
        zs = [
            estimate_grasp(scene.mask, scene.depth_m, K, depth_quantile=q).z_m
            for q in (0.5, 0.75, 0.95)
        ]
        self.assertAlmostEqual(max(zs) - min(zs), 0.0, places=9)

    def test_empty_mask_is_rejected_not_crashed(self):
        K = make_intrinsics(600.0, 600.0, 319.5, 239.5)
        est = estimate_grasp(np.zeros((480, 640), np.uint8), np.ones((480, 640)), K)
        self.assertFalse(est.is_valid)
        self.assertEqual(est.rejected_reason, "no_contour")

    def test_mask_without_depth_is_rejected(self):
        K = make_intrinsics(600.0, 600.0, 319.5, 239.5)
        mask = np.zeros((480, 640), np.uint8)
        mask[200:280, 240:400] = 1
        est = estimate_grasp(mask, np.zeros((480, 640)), K)
        self.assertFalse(est.is_valid)
        self.assertEqual(est.rejected_reason, "no_valid_depth_or_rect")

    def test_shape_mismatch_raises(self):
        K = make_intrinsics(600.0, 600.0, 319.5, 239.5)
        with self.assertRaises(ValueError):
            estimate_grasp(np.zeros((10, 10), np.uint8), np.zeros((20, 20)), K)


if __name__ == "__main__":
    unittest.main(verbosity=2)
