"""YoloeSegmenter contract tests, against a mocked ultralytics.

**No real inference runs here and ultralytics is not installed.** These tests pin
the *contract* read out of the pinned vendor tree at commit
``547faa08e5161af996892497c0aaa788401454fc``:

* ``YOLO(checkpoint)`` then ``set_classes(class_list)``  -- ``utils/yolo_utils.py:77,79``
* ``predict(bgr, conf=..., iou=...)``                     -- ``utils/yolo_utils.py:212``
* mask resize NEAREST then ``> 0.5``                      -- ``utils/yolo_utils.detection_mask``
* target selection: exact class match, else substring, then max confidence
  **within that candidate set** -- ``utils/graspnet_utils.select_target``

The last one is the substantive one. Picking the globally highest-confidence mask
returns a distractor whenever the detector happens to be surer about it, and on a
scene built to contain target-like distractors that is not a rare case.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np

from grasp_smoke.dataset import Frame
from grasp_smoke.detect import (
    REQUIRED_ULTRALYTICS_VERSION,
    YOLOE_CHECKPOINT,
    YOLOE_CHECKPOINT_SHA256,
    PredictorUnavailable,
    build_predictor,
)
from grasp_smoke.geometry import make_intrinsics, make_transform


def _frame(width: int = 64, height: int = 48) -> Frame:
    return Frame(
        scene_id="mock", rgb=np.zeros((height, width, 3), np.uint8),
        depth_m=np.full((height, width), 0.6), K=make_intrinsics(60.0, 60.0, 31.5, 23.5),
        width=width, height=height, stamp_ns=1,
        T_base_cam=make_transform(np.eye(3), np.zeros(3)),
    )


class _FakeBoxes:
    def __init__(self, confs, clses):
        self.conf = np.asarray(confs, dtype=np.float32)
        self.cls = np.asarray(clses, dtype=np.float32)

    def __len__(self):
        return len(self.conf)


class _FakeMasks:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)


class _FakeResult:
    def __init__(self, masks, confs, clses, names):
        self.masks = _FakeMasks(masks) if masks is not None else None
        self.boxes = _FakeBoxes(confs, clses) if masks is not None else None
        self.names = names


class _FakeYOLO:
    """Records what it was constructed with and what it was told."""

    last_instance = None

    def __init__(self, path):
        self.path = path
        self.set_classes_calls = []
        self.predict_calls = []
        self.results = []
        _FakeYOLO.last_instance = self

    def set_classes(self, classes):
        self.set_classes_calls.append(list(classes))

    def predict(self, image, **kwargs):
        self.predict_calls.append({"image": image, "kwargs": kwargs})
        return self.results


def _install_fake_ultralytics(version=REQUIRED_ULTRALYTICS_VERSION):
    module = types.ModuleType("ultralytics")
    module.YOLO = _FakeYOLO
    module.__version__ = version
    return mock.patch.dict(sys.modules, {"ultralytics": module})


def _build(**kwargs):
    with _install_fake_ultralytics(), \
         mock.patch("grasp_smoke.detect.ultralytics_available", return_value=True):
        return build_predictor("yoloe", **kwargs)


class TestPinnedCheckpoint(unittest.TestCase):
    def test_checkpoint_sha_matches_the_pinned_vendor_tree(self):
        """Guards against the weights changing under us."""
        from grasp_smoke.dataset import sha256_file
        self.assertTrue(YOLOE_CHECKPOINT.exists(), YOLOE_CHECKPOINT)
        self.assertEqual(sha256_file(YOLOE_CHECKPOINT), YOLOE_CHECKPOINT_SHA256)

    def test_sha_mismatch_fails_closed(self):
        """A checkpoint that is not the pinned one must be refused, not loaded."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
            fh.write(b"not the pinned weights")
            impostor = fh.name
        try:
            with self.assertRaises(PredictorUnavailable) as ctx:
                _build(checkpoint=impostor, class_list=["box"], target_class="box")
            self.assertIn("SHA-256 mismatch", str(ctx.exception))
        finally:
            import os
            os.unlink(impostor)

    def test_sha_check_can_be_waived_only_explicitly(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
            fh.write(b"scratch weights")
            impostor = fh.name
        try:
            predictor = _build(checkpoint=impostor, class_list=["box"],
                               target_class="box", verify_sha=False)
            # Waiving the check must still record the hash actually loaded.
            self.assertNotEqual(predictor.config.checkpoint_sha256, YOLOE_CHECKPOINT_SHA256)
        finally:
            os.unlink(impostor)

    def test_wrong_checkpoint_path_fails_closed(self):
        with self.assertRaises(PredictorUnavailable):
            _build(checkpoint="/nonexistent/model.pt")


class TestFailClosed(unittest.TestCase):
    def test_missing_ultralytics_raises_and_never_falls_back(self):
        with mock.patch("grasp_smoke.detect.ultralytics_available", return_value=False):
            with self.assertRaises(PredictorUnavailable) as ctx:
                build_predictor("yoloe")
        self.assertIn("ultralytics is not installed", str(ctx.exception))
        self.assertIn(REQUIRED_ULTRALYTICS_VERSION, str(ctx.exception))

    def test_unknown_predictor_name_rejected(self):
        with self.assertRaises(ValueError):
            build_predictor("definitely-not-a-predictor")

    def test_target_class_must_be_in_class_list(self):
        with self.assertRaises(ValueError):
            _build(class_list=["cup", "banana"], target_class="box")


class TestVendorContract(unittest.TestCase):
    def test_set_classes_is_called_with_the_configured_list(self):
        classes = ["box", "cup", "banana"]
        predictor = _build(class_list=classes, target_class="box")
        self.assertEqual(_FakeYOLO.last_instance.set_classes_calls, [classes])
        self.assertEqual(predictor.config.class_list, classes)

    def test_config_records_full_provenance(self):
        predictor = _build(class_list=["box"], target_class="box",
                           device="cpu", imgsz=512)
        cfg = predictor.config.to_dict()
        self.assertEqual(cfg["checkpoint_sha256"], YOLOE_CHECKPOINT_SHA256)
        self.assertEqual(cfg["package_version"], REQUIRED_ULTRALYTICS_VERSION)
        self.assertEqual(cfg["device"], "cpu")
        self.assertEqual(cfg["imgsz"], 512)
        self.assertEqual(cfg["class_list"], ["box"])
        self.assertEqual(cfg["target_class"], "box")
        self.assertEqual(cfg["inference"]["confidence_threshold"], 0.25)
        self.assertEqual(cfg["inference"]["nms_iou_threshold"], 0.45)
        self.assertEqual(cfg["inference"]["mask_threshold"], 0.5)
        # The scoring rule must be recorded separately from the NMS threshold.
        self.assertEqual(cfg["scoring"]["metric_iou_match_threshold"], 0.5)
        self.assertNotIn("metric_iou_match_threshold", cfg["inference"])
        self.assertFalse(cfg["diagnostic_only"])

    def test_predict_receives_vendor_kwargs(self):
        predictor = _build(class_list=["box"], target_class="box")
        _FakeYOLO.last_instance.results = []
        predictor.predict(_frame())
        kwargs = _FakeYOLO.last_instance.predict_calls[0]["kwargs"]
        self.assertEqual(kwargs["conf"], 0.25)
        self.assertEqual(kwargs["iou"], 0.45)
        self.assertIs(kwargs["verbose"], False)
        self.assertEqual(kwargs["device"], "cpu")


class TestTargetSelection(unittest.TestCase):
    """The substantive behaviour: pick the target class, not the loudest mask."""

    def _run(self, masks, confs, clses, names, target="box", classes=None, size=(48, 64)):
        predictor = _build(class_list=classes or ["box", "cup"], target_class=target)
        _FakeYOLO.last_instance.results = [_FakeResult(masks, confs, clses, names)]
        return predictor.predict(_frame(width=size[1], height=size[0]))

    def test_selects_target_class_over_higher_confidence_distractor(self):
        h, w = 48, 64
        cup = np.zeros((h, w), np.float32); cup[0:10, 0:10] = 1.0
        box = np.zeros((h, w), np.float32); box[20:40, 20:50] = 1.0
        # The cup is the most confident detection by a wide margin.
        pred = self._run([cup, box], [0.95, 0.40], [1, 0], {0: "box", 1: "cup"})
        self.assertIsNotNone(pred)
        self.assertEqual(pred.class_name, "box")
        self.assertAlmostEqual(pred.confidence, 0.40, places=5)
        self.assertEqual(pred.n_candidates, 2)
        self.assertEqual(pred.n_class_matches, 1)
        # And the returned mask is the box's, not the cup's.
        self.assertGreater(int(pred.mask[30, 35]), 0)
        self.assertEqual(int(pred.mask[5, 5]), 0)

    def test_picks_highest_confidence_among_multiple_target_matches(self):
        h, w = 48, 64
        a = np.zeros((h, w), np.float32); a[0:12, 0:12] = 1.0
        b = np.zeros((h, w), np.float32); b[20:40, 20:50] = 1.0
        pred = self._run([a, b], [0.30, 0.80], [0, 0], {0: "box"})
        self.assertAlmostEqual(pred.confidence, 0.80, places=5)
        self.assertEqual(pred.n_class_matches, 2)
        self.assertGreater(int(pred.mask[30, 35]), 0)

    def test_substring_match_used_only_when_no_exact_match(self):
        h, w = 48, 64
        m = np.zeros((h, w), np.float32); m[10:30, 10:40] = 1.0
        pred = self._run([m], [0.5], [0], {0: "cardboard box lid"}, target="box")
        self.assertIsNotNone(pred)
        self.assertEqual(pred.class_name, "cardboard box lid")

    def test_exact_match_beats_substring_even_at_lower_confidence(self):
        h, w = 48, 64
        exact = np.zeros((h, w), np.float32); exact[20:40, 20:50] = 1.0
        sub = np.zeros((h, w), np.float32); sub[0:10, 0:10] = 1.0
        pred = self._run([exact, sub], [0.31, 0.99], [0, 1],
                         {0: "box", 1: "big box of cups"})
        self.assertEqual(pred.class_name, "box")
        self.assertAlmostEqual(pred.confidence, 0.31, places=5)

    def test_no_matching_class_returns_none(self):
        h, w = 48, 64
        m = np.zeros((h, w), np.float32); m[10:30, 10:40] = 1.0
        pred = self._run([m], [0.9], [1], {0: "box", 1: "cup"}, target="box")
        self.assertIsNone(pred)

    def test_empty_results_returns_none(self):
        predictor = _build(class_list=["box"], target_class="box")
        _FakeYOLO.last_instance.results = []
        self.assertIsNone(predictor.predict(_frame()))

    def test_result_without_masks_returns_none(self):
        predictor = _build(class_list=["box"], target_class="box")
        _FakeYOLO.last_instance.results = [_FakeResult(None, [], [], {0: "box"})]
        self.assertIsNone(predictor.predict(_frame()))

    def test_mask_is_resized_to_frame_and_binarised(self):
        """Vendor contract: NEAREST resize to image size, threshold > 0.5."""
        small = np.zeros((12, 16), np.float32)
        small[3:9, 4:12] = 0.9
        small[0:2, 0:2] = 0.4          # below threshold, must be dropped
        pred = self._run([small], [0.7], [0], {0: "box"}, size=(48, 64))
        self.assertEqual(pred.mask.shape, (48, 64))
        self.assertEqual(set(np.unique(pred.mask)).issubset({0, 1}), True)
        self.assertEqual(int(pred.mask[2, 2]), 0)
        self.assertGreater(int(pred.mask[24, 32]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
