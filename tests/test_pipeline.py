"""Dataset contract, GT isolation, scoring, PoseStamped, and depth conversion.

Everything here runs without ROS 2 and without Isaac Sim, which is the point:
the pure library has to stay independently testable (PLAN.md 5.2.7).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_smoke import dataset as ds
from grasp_smoke.capture import capture_analytic, plan_smoke_scenes
from grasp_smoke.detect import SaturationSegmenter
from grasp_smoke.geometry import make_intrinsics, transform_point
from grasp_smoke.grasp import estimate_grasp
from grasp_smoke.pose_msg import grasp_to_pose_stamped, pose_stamped_position, pose_stamped_stamp_ns
from grasp_smoke.scorer import mask_iou, score_scene, summarize


class _TinyDataset(unittest.TestCase):
    """Two analytic scenes, captured once and reused by the tests below."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "ds"
        specs = plan_smoke_scenes(seed=4242)[:2]
        cls.stats = capture_analytic(cls.root, specs, seed=4242)
        cls.specs = specs

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class TestDatasetContract(_TinyDataset):
    def test_manifest_records_provenance(self):
        m = ds.load_manifest(self.root)
        self.assertEqual(m["schema_version"], ds.SCHEMA_VERSION)
        self.assertEqual(m["seed"], 4242)
        self.assertEqual(m["depth_policy"], ds.DEPTH_POLICY_IMAGE_PLANE)
        self.assertEqual(m["capture_backend"], "analytic")
        self.assertIn("depth_quantile", m)

    def test_every_file_is_checksummed_and_verifies(self):
        self.assertEqual(ds.verify_checksums(self.root), [])
        for scene in ds.load_manifest(self.root)["scenes"]:
            self.assertEqual(len(scene["files"]), 6)

    def test_tampering_is_detected(self):
        scene_id = ds.scene_ids(self.root)[0]
        path = self.root / "frames" / scene_id / "camera_info.json"
        original = path.read_text()
        try:
            data = json.loads(original)
            data["width"] = 1
            path.write_text(json.dumps(data))
            self.assertIn(f"frames/{scene_id}/camera_info.json",
                          ds.verify_checksums(self.root))
        finally:
            path.write_text(original)
        self.assertEqual(ds.verify_checksums(self.root), [])

    def test_depth_is_metres_float_and_optical_z(self):
        frame = ds.load_frame(self.root, ds.scene_ids(self.root)[0])
        valid = frame.depth_m[frame.depth_m > 0]
        self.assertGreater(valid.size, 0)
        # A tabletop scene: tens of centimetres, not millimetres and not microns.
        self.assertGreater(float(valid.min()), 0.1)
        self.assertLess(float(valid.max()), 5.0)

    def test_frame_and_labels_share_a_timestamp(self):
        for scene_id in ds.scene_ids(self.root):
            self.assertEqual(
                ds.load_frame(self.root, scene_id).stamp_ns,
                ds.load_labels(self.root, scene_id).stamp_ns,
            )

    def test_capture_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            second = Path(tmp) / "ds2"
            capture_analytic(second, self.specs, seed=4242)
            for scene in ds.load_manifest(self.root)["scenes"]:
                other = next(s for s in ds.load_manifest(second)["scenes"]
                             if s["scene_id"] == scene["scene_id"])
                self.assertEqual(scene["files"], other["files"],
                                 "same seed must reproduce byte-identical scenes")


class TestGroundTruthIsolation(_TinyDataset):
    """Branch B must not be able to reach GT even by accident."""

    def test_frame_exposes_no_ground_truth(self):
        frame = ds.load_frame(self.root, ds.scene_ids(self.root)[0])
        forbidden = {"gt_mask", "grasp_gt_position", "grasp_gt_open_axis", "T_base_object"}
        self.assertEqual(forbidden & set(vars(frame)), set())

    def test_predictor_signature_takes_only_a_frame(self):
        import inspect
        sig = inspect.signature(SaturationSegmenter.predict)
        self.assertEqual([p for p in sig.parameters if p != "self"], ["frame"])

    def test_predicted_mask_differs_from_gt_mask_object(self):
        """A predictor returning the GT array object would be a leak."""
        scene_id = ds.scene_ids(self.root)[0]
        frame = ds.load_frame(self.root, scene_id)
        labels = ds.load_labels(self.root, scene_id)
        predicted = SaturationSegmenter().predict(frame)
        self.assertIsNotNone(predicted)
        self.assertIsNot(predicted, labels.gt_mask)

    def test_labels_live_outside_the_frames_tree(self):
        for scene_id in ds.scene_ids(self.root):
            scene_files = {p.name for p in (self.root / "frames" / scene_id).iterdir()}
            self.assertEqual(scene_files & {"mask.png", "labels.json", "grasp_gt.json"}, set())


class TestScorer(_TinyDataset):
    def test_iou_bounds(self):
        a = np.zeros((10, 10), np.uint8); a[:5, :] = 1
        b = np.zeros((10, 10), np.uint8); b[:5, :] = 1
        self.assertAlmostEqual(mask_iou(a, b), 1.0)
        c = np.zeros((10, 10), np.uint8); c[5:, :] = 1
        self.assertAlmostEqual(mask_iou(a, c), 0.0)
        self.assertAlmostEqual(mask_iou(np.zeros((4, 4)), np.zeros((4, 4))), 0.0)

    def test_oracle_scene_is_a_true_positive_within_tolerance(self):
        scene_id = ds.scene_ids(self.root)[0]
        frame = ds.load_frame(self.root, scene_id)
        labels = ds.load_labels(self.root, scene_id)
        est = estimate_grasp(labels.gt_mask, frame.depth_m, frame.K)
        score = score_scene(scene_id, "A1", est, labels.gt_mask, labels, frame.T_base_cam)
        self.assertTrue(score.true_positive)
        self.assertTrue(score.within_tolerance, f"pos={score.position_error_m}")

    def test_detection_on_absent_target_is_a_false_positive(self):
        scene_id = ds.scene_ids(self.root)[0]
        frame = ds.load_frame(self.root, scene_id)
        labels = ds.load_labels(self.root, scene_id)
        labels.target_present = False
        est = estimate_grasp(labels.gt_mask, frame.depth_m, frame.K)
        score = score_scene(scene_id, "B", est, labels.gt_mask, labels, frame.T_base_cam)
        self.assertTrue(score.false_positive)
        self.assertFalse(score.true_positive)
        self.assertIsNone(score.position_error_m)

    def test_low_iou_is_a_false_positive_not_a_pose_result(self):
        scene_id = ds.scene_ids(self.root)[0]
        frame = ds.load_frame(self.root, scene_id)
        labels = ds.load_labels(self.root, scene_id)
        wrong = np.zeros_like(labels.gt_mask)
        wrong[0:60, 0:60] = 1
        est = estimate_grasp(wrong, np.full_like(frame.depth_m, 0.6), frame.K)
        score = score_scene(scene_id, "B", est, wrong, labels, frame.T_base_cam)
        self.assertTrue(score.false_positive)
        self.assertIsNone(score.position_error_m)

    def test_yield_denominator_is_all_present_scenes(self):
        """A detector that fires once and nails it must not score 100% yield."""
        scene_id = ds.scene_ids(self.root)[0]
        frame = ds.load_frame(self.root, scene_id)
        labels = ds.load_labels(self.root, scene_id)
        est = estimate_grasp(labels.gt_mask, frame.depth_m, frame.K)
        good = score_scene(scene_id, "B", est, labels.gt_mask, labels, frame.T_base_cam)
        missed = score_scene("scene_x", "B", None, None, labels, frame.T_base_cam)
        summary = summarize([good, missed], "B")
        self.assertEqual(summary.n_present, 2)
        self.assertEqual(summary.n_true_positive, 1)
        self.assertAlmostEqual(summary.end_to_end_yield, 0.5)
        self.assertAlmostEqual(summary.recall, 0.5)


class TestPoseStamped(_TinyDataset):
    def test_pose_is_in_base_frame_at_the_image_stamp(self):
        scene_id = ds.scene_ids(self.root)[0]
        frame = ds.load_frame(self.root, scene_id)
        labels = ds.load_labels(self.root, scene_id)
        est = estimate_grasp(labels.gt_mask, frame.depth_m, frame.K)
        msg = grasp_to_pose_stamped(est, frame, branch="A1")

        self.assertEqual(msg["header"]["frame_id"], frame.base_frame_id)
        self.assertEqual(pose_stamped_stamp_ns(msg), frame.stamp_ns)
        np.testing.assert_allclose(
            pose_stamped_position(msg), transform_point(frame.T_base_cam, est.position),
            atol=1e-9,
        )

    def test_orientation_is_a_unit_quaternion(self):
        scene_id = ds.scene_ids(self.root)[0]
        frame = ds.load_frame(self.root, scene_id)
        labels = ds.load_labels(self.root, scene_id)
        est = estimate_grasp(labels.gt_mask, frame.depth_m, frame.K)
        q = grasp_to_pose_stamped(est, frame)["pose"]["orientation"]
        norm = np.linalg.norm([q["x"], q["y"], q["z"], q["w"]])
        self.assertAlmostEqual(float(norm), 1.0, places=9)

    def test_invalid_grasp_refuses_to_publish(self):
        scene_id = ds.scene_ids(self.root)[0]
        frame = ds.load_frame(self.root, scene_id)
        bad = estimate_grasp(np.zeros((frame.height, frame.width), np.uint8),
                             frame.depth_m, frame.K)
        with self.assertRaises(ValueError):
            grasp_to_pose_stamped(bad, frame)


class TestDepthConversion(unittest.TestCase):
    """Fallback ladder rung 3 must be tested before it is ever allowed."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "isaac_capture", Path(__file__).resolve().parents[1] / "capture" / "isaac_capture.py"
        )
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)   # safe: Isaac imports live inside main()

    def test_principal_ray_is_unchanged(self):
        K = make_intrinsics(600.0, 600.0, 319.5, 239.5)
        radial = np.full((480, 640), 2.0)
        z = self.mod.radial_range_to_optical_z(radial, K)
        self.assertAlmostEqual(float(z[240, 320]), 2.0, places=3)

    def test_edges_are_corrected_downward(self):
        """Radial range exceeds optical Z everywhere off-axis, worst at corners."""
        K = make_intrinsics(600.0, 600.0, 319.5, 239.5)
        z = self.mod.radial_range_to_optical_z(np.full((480, 640), 2.0), K)
        self.assertLess(float(z[0, 0]), float(z[240, 320]))
        self.assertLess(float(z[0, 0]), 2.0)

    def test_matches_closed_form(self):
        K = make_intrinsics(600.0, 600.0, 319.5, 239.5)
        z = self.mod.radial_range_to_optical_z(np.full((480, 640), 1.0), K)
        u, v = 100.5, 80.5
        expected = 1.0 / np.sqrt(1 + ((u - 319.5) / 600.0) ** 2 + ((v - 239.5) / 600.0) ** 2)
        self.assertAlmostEqual(float(z[80, 100]), float(expected), places=9)

    def test_uncorrected_radial_would_be_wrong_enough_to_matter(self):
        """Justifies the ladder: the error is millimetres, not rounding."""
        K = make_intrinsics(600.0, 600.0, 319.5, 239.5)
        z = self.mod.radial_range_to_optical_z(np.full((480, 640), 0.6), K)
        self.assertGreater(0.6 - float(z[0, 0]), 0.005)


if __name__ == "__main__":
    unittest.main(verbosity=2)
