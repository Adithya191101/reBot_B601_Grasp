"""Branch B mask prediction. Reads the RGB frame and nothing else.

Every predictor here takes a :class:`~grasp_smoke.dataset.Frame` and returns a
mask. ``Frame`` carries no ground truth, so GT leakage into Branch B is
prevented by the type, not by care (PLAN.md 5.2.2). ``tests/test_branch_isolation.py``
asserts this.

**Provisional, per PLAN.md 5.2.5.** The Aug 8 gate requires only that *a*
predicted-mask path runs end to end. The binding freeze of checkpoint, prompt,
threshold and IoU rule is Aug 10 and has not happened. Whichever predictor runs
records its full configuration into the results, so no number is ever reported
without the config that produced it.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .dataset import Frame

VENDOR_MODEL_DIR = Path(__file__).resolve().parents[1] / "src" / "reBot-DevArm-Grasp" / "models"
YOLOE_CHECKPOINT = VENDOR_MODEL_DIR / "yoloe-26s-seg.pt"


@dataclass
class PredictorConfig:
    """Exactly what produced a mask. Serialised into results alongside metrics."""

    name: str
    provisional: bool
    checkpoint: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    prompt: Optional[str] = None
    confidence_threshold: Optional[float] = None
    iou_match_threshold: float = 0.5
    params: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "provisional": self.provisional,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "prompt": self.prompt,
            "confidence_threshold": self.confidence_threshold,
            "iou_match_threshold": self.iou_match_threshold,
            "params": self.params,
            "note": self.note,
        }


class MaskPredictor:
    config: PredictorConfig

    def predict(self, frame: Frame) -> Optional[np.ndarray]:
        raise NotImplementedError


class SaturationSegmenter(MaskPredictor):
    """Provisional frame-only segmenter: saturation threshold + largest blob.

    Deliberately simple and deterministic. The target is a chromatic box on a
    desaturated ground, so saturation separates them without any learned model.
    Its job is to exercise the Branch B code path -- identical geometry, identical
    scorer, no GT access -- so that swapping in YOLOE on Aug 10 is a one-line
    change rather than a new integration.
    """

    def __init__(self, saturation_threshold: Optional[int] = None, min_area_px: int = 400):
        # Threshold is Otsu-adaptive by default. A fixed threshold tuned on one
        # renderer silently collapses to zero recall on another -- which is
        # exactly what happened when the analytic-tuned value of 60 met Isaac
        # Sim's lighting and materials.
        self.saturation_threshold = saturation_threshold
        self.min_area_px = int(min_area_px)
        self.config = PredictorConfig(
            name="saturation_largest_blob",
            provisional=True,
            confidence_threshold=None,
            params={
                "saturation_threshold": saturation_threshold or "otsu",
                "min_area_px": self.min_area_px,
                "morph_kernel": 5,
            },
            note=(
                "Provisional stand-in for the pinned YOLOE path: ultralytics is "
                "not installed and installing it needs approval. Exercises the "
                "identical Branch B contract. Not a detector result."
            ),
        )

    def predict(self, frame: Frame) -> Optional[np.ndarray]:
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1]
        if self.saturation_threshold is None:
            thr, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thr = float(self.saturation_threshold)
        raw = (sat >= thr).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        if n <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        if int(stats[best, cv2.CC_STAT_AREA]) < self.min_area_px:
            return None
        return (labels == best).astype(np.uint8)


class YoloeSegmenter(MaskPredictor):
    """The pinned vendor YOLOE path. Requires ``ultralytics`` to be importable.

    Weights are already present in the pinned tree -- nothing is downloaded here.
    Only the package is missing.
    """

    def __init__(self, prompt: str = "box", confidence_threshold: float = 0.25):
        from ultralytics import YOLOE  # noqa: F401  (import-time availability check)
        from .dataset import sha256_file

        self.model = YOLOE(str(YOLOE_CHECKPOINT))
        self.prompt = prompt
        self.confidence_threshold = float(confidence_threshold)
        self.config = PredictorConfig(
            name="yoloe-26s-seg",
            provisional=True,
            checkpoint=str(YOLOE_CHECKPOINT),
            checkpoint_sha256=sha256_file(YOLOE_CHECKPOINT),
            prompt=prompt,
            confidence_threshold=self.confidence_threshold,
            note="Pinned vendor checkpoint. Freeze of prompt/threshold is due Aug 10.",
        )

    def predict(self, frame: Frame) -> Optional[np.ndarray]:
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(bgr, conf=self.confidence_threshold, verbose=False)
        for result in results:
            masks = getattr(result, "masks", None)
            if masks is None or masks.data is None or len(masks.data) == 0:
                continue
            # Vendor convention: resize mask to image size, threshold at 0.5
            # (ordinary_grasp.py:_depth_mask).
            data = masks.data.cpu().numpy()
            confs = (
                result.boxes.conf.cpu().numpy()
                if getattr(result, "boxes", None) is not None else np.ones(len(data))
            )
            idx = int(np.argmax(confs))
            mask = cv2.resize(
                data[idx], (frame.width, frame.height), interpolation=cv2.INTER_NEAREST
            )
            return (mask > 0.5).astype(np.uint8)
        return None


def ultralytics_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def build_predictor(prefer_yoloe: bool = True) -> MaskPredictor:
    """Pick the best available Branch B predictor and say which it is."""
    if prefer_yoloe and ultralytics_available() and YOLOE_CHECKPOINT.exists():
        try:
            return YoloeSegmenter()
        except Exception as exc:                     # noqa: BLE001
            print(f"[detect] YOLOE unavailable ({type(exc).__name__}: {exc}); "
                  f"falling back to the provisional segmenter")
    return SaturationSegmenter()
