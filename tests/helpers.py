"""Test doubles shared by the regression tests.

Kept importable from ``run_smoke.py`` via ``--empty-predictor`` so the gate can be
exercised end to end by the real command, not by a reimplementation of it. A gate
tested only through a stub of itself is not a tested gate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from grasp_smoke.dataset import Frame
from grasp_smoke.detect import MaskPredictor, Prediction, PredictorConfig


class AlwaysEmptyPredictor(MaskPredictor):
    """Never detects anything. The smoke command must exit non-zero on this."""

    def __init__(self):
        self.config = PredictorConfig(
            name="always_empty",
            provisional=True,
            diagnostic_only=True,
            note="regression double: proves the gate fails instead of reporting ok",
        )

    def predict(self, frame: Frame) -> Optional[Prediction]:
        return None


class AlwaysGarbagePredictor(MaskPredictor):
    """Detects a fixed corner blob: a valid mask that can never match the target.

    Distinguishes "no PoseStamped" from "PoseStamped but nothing scored" -- the
    gate has to reject both, and only this double exercises the second path.
    """

    def __init__(self, size: int = 80):
        self.size = int(size)
        self.config = PredictorConfig(
            name="always_garbage",
            provisional=True,
            diagnostic_only=True,
            note="regression double: valid mask, zero true positives",
        )

    def predict(self, frame: Frame) -> Optional[Prediction]:
        mask = np.zeros((frame.height, frame.width), dtype=np.uint8)
        mask[0:self.size, 0:self.size] = 1
        return Prediction(mask=mask, class_name="garbage", confidence=1.0,
                          n_candidates=1, n_class_matches=1)
