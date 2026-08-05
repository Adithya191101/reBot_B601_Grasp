"""The gate must be able to fail.

Runs the **real** ``run_smoke.py`` command as a subprocess rather than a stub of
it, because the failure mode being guarded against is precisely "the command
reports every stage ok while nothing reached the evaluator". A gate exercised
only through a reimplementation of itself is not a tested gate.

Also covers the dataset hardening: locked outputs, roster/seed/schema validation,
timestamp uniqueness, unexpected files, and the inference/scoring split.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp_smoke import dataset as ds
from grasp_smoke import inference as infer
from grasp_smoke.capture import (
    BASELINE_DEPTH_QUANTILE, capture_analytic, expected_scene_ids, plan_smoke_scenes,
    validate_depth_quantile,
)
from grasp_smoke.detect import SaturationSegmenter
from grasp_smoke.validate import validate_dataset

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

EXIT_OK, EXIT_VALIDATION, EXIT_CAPTURE, EXIT_NO_BRANCH_B = 0, 2, 3, 5


def _run_smoke(*args, cwd=REPO):
    return subprocess.run(
        [PYTHON, str(REPO / "run_smoke.py"), *args],
        cwd=cwd, capture_output=True, text=True, timeout=900,
    )


class TestGateFailsLoudly(unittest.TestCase):
    """The acceptance criterion: an empty predictor must not report success."""

    def test_empty_predictor_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = _run_smoke("--backend", "analytic", "--empty-predictor",
                              "--out", str(Path(tmp) / "smoke"))
        self.assertEqual(proc.returncode, EXIT_NO_BRANCH_B,
                         f"expected exit {EXIT_NO_BRANCH_B}\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("SMOKE TEST FAILED", proc.stdout)
        self.assertIn("never reached the evaluator", proc.stdout)
        # And it must not have claimed the branch worked.
        self.assertNotIn("GATE PASSED", proc.stdout)

    def test_empty_predictor_still_records_results_with_gate_passed_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoke"
            _run_smoke("--backend", "analytic", "--empty-predictor", "--out", str(out))
            payload = json.loads((out / "results.json").read_text())
        self.assertIs(payload["gate_passed"], False)
        self.assertEqual(payload["counters"]["B1"]["pose_stamped"], 0)
        self.assertEqual(payload["counters"]["B2"]["pose_stamped"], 0)
        # The oracle branch still ran -- that is what makes the failure diagnosable.
        self.assertGreater(payload["counters"]["A1"]["pose_stamped"], 0)

    def test_working_predictor_passes_and_records_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoke"
            proc = _run_smoke("--backend", "analytic", "--predictor", "saturation",
                              "--out", str(out))
            self.assertEqual(proc.returncode, EXIT_OK, proc.stdout + proc.stderr)
            payload = json.loads((out / "results.json").read_text())
        self.assertIs(payload["gate_passed"], True)
        self.assertGreater(payload["counters"]["B1"]["pose_stamped"], 0)
        for branch in ("A1", "A2", "B1", "B2"):
            c = payload["counters"][branch]
            self.assertGreaterEqual(c["attempts"], c["valid_estimates"])
            self.assertGreaterEqual(c["valid_estimates"], c["pose_stamped"])
        self.assertEqual(len(payload["dataset"]["expected_scene_ids"]), 10)
        self.assertEqual(payload["estimator"]["depth_quantile_used"],
                         BASELINE_DEPTH_QUANTILE)
        self.assertIs(payload["estimator"]["is_ablation"], False)

    def test_yoloe_without_ultralytics_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = _run_smoke("--backend", "analytic", "--predictor", "yoloe",
                              "--out", str(Path(tmp) / "smoke"))
        self.assertEqual(proc.returncode, EXIT_VALIDATION, proc.stdout)
        self.assertIn("ultralytics is not installed", proc.stdout)
        # Critically: it must NOT have silently used the saturation stand-in.
        self.assertNotIn("saturation_largest_blob", proc.stdout)

    def test_off_baseline_quantile_rejected_without_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = _run_smoke("--backend", "analytic", "--depth-quantile", "0.75",
                              "--out", str(Path(tmp) / "smoke"))
        self.assertEqual(proc.returncode, EXIT_VALIDATION, proc.stdout)
        self.assertIn("frozen baseline", proc.stdout)

    def test_declared_ablation_is_permitted_and_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoke"
            proc = _run_smoke("--backend", "analytic", "--depth-quantile", "0.75",
                              "--allow-ablation", "--out", str(out))
            self.assertEqual(proc.returncode, EXIT_OK, proc.stdout + proc.stderr)
            payload = json.loads((out / "results.json").read_text())
        self.assertIs(payload["estimator"]["is_ablation"], True)
        self.assertEqual(payload["estimator"]["depth_quantile_used"], 0.75)


class TestDepthQuantileFreeze(unittest.TestCase):
    def test_baseline_is_the_vendor_config_value_not_the_code_default(self):
        # config/default.yaml:61 sets 0.5; ordinary_grasp.py defaults to 0.75.
        self.assertEqual(BASELINE_DEPTH_QUANTILE, 0.5)
        self.assertEqual(validate_depth_quantile(0.5), 0.5)

    def test_ablation_requires_explicit_permission(self):
        with self.assertRaises(ValueError):
            validate_depth_quantile(0.75)
        self.assertEqual(validate_depth_quantile(0.75, allow_ablation=True), 0.75)

    def test_undeclared_value_rejected_even_as_ablation(self):
        with self.assertRaises(ValueError):
            validate_depth_quantile(0.62, allow_ablation=True)


class TestLockedDataset(unittest.TestCase):
    def test_capture_refuses_existing_nonempty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            specs = plan_smoke_scenes(7)[:2]
            capture_analytic(root, specs, seed=7)
            with self.assertRaises(ds.DatasetExistsError):
                capture_analytic(root, specs, seed=7)

    def test_unlocked_capture_is_possible_but_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            specs = plan_smoke_scenes(7)[:2]
            capture_analytic(root, specs, seed=7)
            capture_analytic(root, specs, seed=7, locked=False)   # no raise

    def test_capture_metadata_written_and_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            capture_analytic(root, plan_smoke_scenes(7)[:2], seed=7)
            meta = json.loads((root / "capture_metadata.json").read_text())
        self.assertEqual(meta["backend"], "analytic")
        self.assertEqual(meta["depth_quantile"], BASELINE_DEPTH_QUANTILE)
        self.assertEqual(len(meta["scene_plan"]), 2)
        self.assertIn("stratum", meta["scene_plan"][0])


class TestDatasetValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "ds"
        cls.specs = plan_smoke_scenes(seed=999)
        capture_analytic(cls.root, cls.specs, seed=999)
        cls.ids = [s.scene_id for s in cls.specs]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_clean_dataset_has_no_problems(self):
        self.assertEqual(
            validate_dataset(self.root, self.ids, expected_seed=999,
                             expected_backend="analytic",
                             expected_depth_policy=ds.DEPTH_POLICY_IMAGE_PLANE),
            [],
        )

    def test_wrong_seed_detected(self):
        problems = validate_dataset(self.root, self.ids, expected_seed=1234)
        self.assertTrue(any("seed" in p for p in problems), problems)

    def test_wrong_backend_detected(self):
        problems = validate_dataset(self.root, self.ids, expected_backend="isaac_sim_5.1")
        self.assertTrue(any("capture_backend" in p for p in problems), problems)

    def test_wrong_depth_policy_detected(self):
        problems = validate_dataset(self.root, self.ids,
                                    expected_depth_policy="something_else")
        self.assertTrue(any("depth_policy" in p for p in problems), problems)

    def test_missing_scene_detected(self):
        problems = validate_dataset(self.root, self.ids + ["scene_9999"])
        self.assertTrue(any("roster" in p for p in problems), problems)

    def test_all_ten_scene_records_required(self):
        self.assertEqual(len(expected_scene_ids(999)), 10)
        problems = validate_dataset(self.root, self.ids[:9])
        self.assertTrue(any("roster" in p for p in problems), problems)

    def test_unexpected_file_detected(self):
        stray = self.root / "frames" / self.ids[0] / "stray.png"
        stray.write_bytes(b"\x89PNG\r\n\x1a\n")
        try:
            self.assertTrue(ds.unexpected_files(self.root))
            problems = validate_dataset(self.root, self.ids)
            self.assertTrue(any("not declared" in p for p in problems), problems)
        finally:
            stray.unlink()
        self.assertEqual(ds.unexpected_files(self.root), [])

    def test_missing_file_detected_as_checksum_failure(self):
        victim = self.root / "frames" / self.ids[0] / "depth.npy"
        data = victim.read_bytes()
        victim.unlink()
        try:
            self.assertTrue(any("MISSING" in b for b in ds.verify_checksums(self.root)))
        finally:
            victim.write_bytes(data)
        self.assertEqual(ds.verify_checksums(self.root), [])

    def test_duplicate_timestamps_detected(self):
        path = self.root / "manifest.json"
        original = path.read_text()
        try:
            manifest = json.loads(original)
            manifest["scenes"][1]["stamp_ns"] = manifest["scenes"][0]["stamp_ns"]
            path.write_text(json.dumps(manifest, indent=2))
            problems = validate_dataset(self.root, self.ids)
            self.assertTrue(any("duplicate capture timestamps" in p for p in problems),
                            problems)
        finally:
            path.write_text(original)

    def test_non_rigid_transform_detected(self):
        scene_id = self.ids[0]
        path = self.root / "frames" / scene_id / "tf.json"
        original = path.read_text()
        try:
            data = json.loads(original)
            m = np.asarray(data["transforms"][0]["matrix"])
            m[:3, :3] *= 2.0                      # scaled rotation: not SE(3)
            data["transforms"][0]["matrix"] = m.tolist()
            path.write_text(json.dumps(data, indent=2))
            problems = validate_dataset(self.root, self.ids)
            self.assertTrue(any("rigid transform" in p for p in problems), problems)
        finally:
            path.write_text(original)

    def test_manifest_sha_is_stable_and_recorded(self):
        first = ds.manifest_sha256(self.root)
        self.assertEqual(first, ds.manifest_sha256(self.root))
        self.assertEqual(len(first), 64)


class TestInferenceScoringSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "ds"
        capture_analytic(cls.root, plan_smoke_scenes(seed=555)[:3], seed=555)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_inference_module_cannot_reach_labels(self):
        """Structural: no *executable* reference to the label loader.

        Parsed rather than grepped -- the module docstring legitimately explains
        that it never calls ``load_labels``, and a substring check would flag its
        own documentation.
        """
        import ast
        tree = ast.parse((REPO / "grasp_smoke" / "inference.py").read_text())
        referenced = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.ImportFrom):
                referenced.update(a.name for a in node.names)
        for forbidden in ("load_labels", "Labels", "gt_mask", "grasp_gt_position"):
            self.assertNotIn(forbidden, referenced,
                             f"inference.py references {forbidden}")

    def test_predictions_are_checksummed_and_verifiable(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pred"
            index = infer.run_inference(self.root, SaturationSegmenter(), out)
            self.assertEqual(infer.verify_predictions(out), [])
            self.assertEqual(index["attempts"], 3)
            self.assertEqual(index["dataset_manifest_sha256"],
                             ds.manifest_sha256(self.root))
            for record in index["records"]:
                if record["detected"]:
                    self.assertEqual(len(record["mask_sha256"]), 64)

    def test_tampered_prediction_mask_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pred"
            index = infer.run_inference(self.root, SaturationSegmenter(), out)
            target = next(r for r in index["records"] if r["detected"])
            path = out / target["mask_file"]
            from PIL import Image
            arr = np.asarray(Image.open(path).convert("L")).copy()
            arr[0, 0] = 255 - arr[0, 0]
            Image.fromarray(arr).save(path)
            self.assertIn(target["mask_file"], infer.verify_predictions(out))

    def test_predictor_config_is_recorded_with_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pred"
            index = infer.run_inference(self.root, SaturationSegmenter(), out)
        self.assertTrue(index["predictor"]["diagnostic_only"])
        self.assertEqual(index["predictor"]["name"], "saturation_largest_blob")


class TestSceneComposition(unittest.TestCase):
    def test_plan_has_nominal_oblique_and_two_kinds_of_negative(self):
        specs = plan_smoke_scenes(1)
        strata = [s.stratum for s in specs]
        self.assertEqual(strata.count("nominal"), 4)
        self.assertEqual(strata.count("oblique"), 4)
        self.assertEqual(strata.count("absent"), 1)
        # A distractor-only negative is the one that can catch "something salient
        # is present" masquerading as detection.
        self.assertEqual(strata.count("distractor_absent"), 1)

    def test_distractors_are_rendered_but_never_in_the_mask(self):
        from grasp_smoke.render import randomize_scene, render
        target, T, K, distractors = randomize_scene(3, tilt_deg=0.0, n_distractors=2)
        self.assertEqual(len(distractors), 2)
        # Render at the size K was built for; a smaller canvas crops to the
        # top-left quadrant and the distractors fall outside the frame.
        with_d = render(target, T, K, 640, 480, np.random.default_rng(0),
                        distractors=distractors)
        without = render(target, T, K, 640, 480, np.random.default_rng(0))
        self.assertFalse(np.array_equal(with_d.rgb, without.rgb),
                         "distractors must be visible in RGB")
        self.assertLessEqual(int(with_d.mask.sum()), int(without.mask.sum()),
                             "distractors must never add mask pixels")

    def test_aim_jitter_moves_the_target_off_centre(self):
        from grasp_smoke.render import randomize_scene, render
        offsets = []
        for seed in range(6):
            target, T, K, _ = randomize_scene(seed, tilt_deg=0.0, aim_jitter=True)
            scene = render(target, T, K, 640, 480, np.random.default_rng(seed))
            ys, xs = np.nonzero(scene.mask)
            if len(xs) == 0:
                continue
            offsets.append(abs(float(xs.mean()) - 320.0) + abs(float(ys.mean()) - 240.0))
        self.assertTrue(offsets)
        self.assertGreater(max(offsets), 5.0,
                           "with jitter the target must not always be centred")


if __name__ == "__main__":
    unittest.main(verbosity=2)
