"""Branch B mask prediction. Reads the RGB frame and nothing else.

Every predictor takes a :class:`~grasp_smoke.dataset.Frame` and returns a
:class:`Prediction`. ``Frame`` carries no ground truth, so GT leakage is
prevented by the type rather than by care.

**Predictor choice is explicit and never inferred from the environment.**
``build_predictor("yoloe")`` either produces the real detector or raises. It does
*not* fall back to the saturation stand-in: a silent fallback means a results
file that says "Branch B" while reporting a colour threshold, which is how a
provisional number gets mistaken for a detector result.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .dataset import Frame, sha256_file

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "reBot-DevArm-Grasp"
YOLOE_CHECKPOINT = VENDOR_ROOT / "models" / "yoloe-26s-seg.pt"

#: Verified 2026-08-04 against the pinned vendor tree at commit
#: 547faa08e5161af996892497c0aaa788401454fc.
YOLOE_CHECKPOINT_SHA256 = "6f62bc7ed9f86056112c383e9b85023291a3929086af26b1a8762335fe39a17d"

#: The version to install, when installation is approved. NOT installed here.
REQUIRED_ULTRALYTICS_VERSION = "8.4.35"

#: Defaults from the vendor config (``config/default.yaml``): detection
#: conf_threshold 0.25, iou_threshold 0.45.
VENDOR_CONF_THRESHOLD = 0.25
VENDOR_NMS_IOU = 0.45
#: Vendor mask binarisation (``yolo_utils.detection_mask``): resize NEAREST, > 0.5.
VENDOR_MASK_THRESHOLD = 0.5

PREDICTOR_CHOICES = ("saturation", "yoloe")


class PredictorUnavailable(RuntimeError):
    """YOLOE was requested and cannot be constructed. Never swallowed."""


@dataclass
class Prediction:
    """A mask plus the metadata needed to interpret it."""

    mask: np.ndarray
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    n_candidates: int = 0
    n_class_matches: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "n_candidates": self.n_candidates,
            "n_class_matches": self.n_class_matches,
            "note": self.note,
        }


@dataclass
class PredictorConfig:
    """Exactly what produced a mask. Serialised alongside every metric."""

    name: str
    provisional: bool
    diagnostic_only: bool = False
    checkpoint: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    package_version: Optional[str] = None
    device: Optional[str] = None
    imgsz: Optional[int] = None
    class_list: list = field(default_factory=list)
    target_class: Optional[str] = None
    confidence_threshold: Optional[float] = None
    nms_iou_threshold: Optional[float] = None
    mask_threshold: Optional[float] = None
    #: Scoring rule, NOT an inference parameter. Kept distinct from
    #: ``nms_iou_threshold`` because conflating the two silently changes what
    #: "IoU" means between the detector and the scorer.
    metric_iou_match_threshold: float = 0.5
    params: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "provisional": self.provisional,
            "diagnostic_only": self.diagnostic_only,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "package_version": self.package_version,
            "device": self.device,
            "imgsz": self.imgsz,
            "class_list": list(self.class_list),
            "target_class": self.target_class,
            "inference": {
                "confidence_threshold": self.confidence_threshold,
                "nms_iou_threshold": self.nms_iou_threshold,
                "mask_threshold": self.mask_threshold,
            },
            "scoring": {"metric_iou_match_threshold": self.metric_iou_match_threshold},
            "params": self.params,
            "note": self.note,
        }


class MaskPredictor:
    config: PredictorConfig

    def predict(self, frame: Frame) -> Optional[Prediction]:
        raise NotImplementedError


class SaturationSegmenter(MaskPredictor):
    """**Diagnostic only.** A frame-only saturation threshold + largest blob.

    Exists to exercise the Branch B contract -- same input type, same geometry,
    same scorer, no GT access -- so the real detector is a swap rather than an
    integration. It is not a detector and its numbers are not detector results;
    ``diagnostic_only`` is set so any consumer can refuse to report them as such.
    """

    def __init__(self, saturation_threshold: Optional[int] = None, min_area_px: int = 400):
        self.saturation_threshold = saturation_threshold
        self.min_area_px = int(min_area_px)
        self.config = PredictorConfig(
            name="saturation_largest_blob",
            provisional=True,
            diagnostic_only=True,
            params={
                "saturation_threshold": saturation_threshold or "otsu",
                "min_area_px": self.min_area_px,
                "morph_kernel": 5,
            },
            note=(
                "Diagnostic stand-in, not a detector. A fixed threshold tuned on one "
                "renderer collapsed to zero recall on another, which is why this is "
                "Otsu-adaptive and why it must never be reported as detector "
                "performance."
            ),
        )

    def predict(self, frame: Frame) -> Optional[Prediction]:
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
        return Prediction(
            mask=(labels == best).astype(np.uint8),
            class_name=None,
            confidence=None,
            n_candidates=int(n - 1),
            n_class_matches=int(n - 1),
            note="saturation stand-in; no class semantics",
        )


class YoloeSegmenter(MaskPredictor):
    """The pinned vendor detector, following ``utils/yolo_utils.py`` exactly.

    Vendor contract, verified against the pinned tree:

    * ``YOLO(checkpoint)`` then ``set_classes(class_list)`` (``yolo_utils.py:77,79``)
    * ``predict(bgr, verbose=False, device=..., conf=..., iou=...)`` -- **BGR** input
      (``yolo_utils.py:212``)
    * mask resize to image size with ``INTER_NEAREST``, threshold ``> 0.5``
      (``yolo_utils.detection_mask``)
    * target selection: exact case-folded class match, else substring match, then
      **max confidence among those candidates** (``graspnet_utils.select_target``)
      -- deliberately *not* the globally highest-confidence mask, which would
      return a distractor whenever the detector is more sure about it.

    Constructing this class raises :class:`PredictorUnavailable` rather than
    degrading. Weights are already in the pinned tree; nothing is downloaded.
    """

    def __init__(
        self,
        class_list: Optional[list] = None,
        target_class: str = "box",
        confidence_threshold: float = VENDOR_CONF_THRESHOLD,
        nms_iou_threshold: float = VENDOR_NMS_IOU,
        device: str = "cpu",
        imgsz: int = 640,
        checkpoint: Path = YOLOE_CHECKPOINT,
        verify_sha: bool = True,
    ):
        if not ultralytics_available():
            raise PredictorUnavailable(
                f"ultralytics is not installed. Install {REQUIRED_ULTRALYTICS_VERSION} "
                f"in an isolated inference environment (requires approval); this code "
                f"downloads nothing."
            )
        if not Path(checkpoint).exists():
            raise PredictorUnavailable(f"checkpoint not found: {checkpoint}")

        actual_sha = sha256_file(Path(checkpoint))
        if verify_sha and actual_sha != YOLOE_CHECKPOINT_SHA256:
            raise PredictorUnavailable(
                f"checkpoint SHA-256 mismatch for {checkpoint}: got {actual_sha}, "
                f"expected {YOLOE_CHECKPOINT_SHA256}"
            )

        import ultralytics
        from ultralytics import YOLO

        version = getattr(ultralytics, "__version__", "unknown")
        if version != REQUIRED_ULTRALYTICS_VERSION:
            # Loud, but not fatal: pinning is the intent, and a mismatch must be
            # visible in the results file rather than silently accepted.
            print(f"[detect] WARNING: ultralytics {version}, "
                  f"expected {REQUIRED_ULTRALYTICS_VERSION}")

        self.class_list = list(class_list or [target_class])
        if target_class not in self.class_list:
            raise ValueError(f"target_class {target_class!r} is not in class_list {self.class_list}")
        self.target_class = target_class
        self.device = device
        self.imgsz = int(imgsz)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)

        self.model = YOLO(str(checkpoint))
        # Open-vocabulary: the class list IS part of the model configuration, so
        # it is frozen and recorded alongside the checkpoint hash.
        self.model.set_classes(self.class_list)

        self.config = PredictorConfig(
            name="yoloe-26s-seg",
            provisional=True,
            diagnostic_only=False,
            checkpoint=str(checkpoint),
            checkpoint_sha256=actual_sha,
            package_version=version,
            device=device,
            imgsz=self.imgsz,
            class_list=self.class_list,
            target_class=target_class,
            confidence_threshold=self.confidence_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            mask_threshold=VENDOR_MASK_THRESHOLD,
            note=(
                "Pinned vendor checkpoint and contract. Prompt/threshold freeze is "
                "due Aug 10; until then this configuration is provisional."
            ),
        )

    def _class_name(self, result, cls_index: int) -> str:
        names = getattr(result, "names", None) or {}
        if isinstance(names, dict):
            return str(names.get(cls_index, cls_index))
        try:
            return str(names[cls_index])
        except Exception:                                      # noqa: BLE001
            return str(cls_index)

    def predict(self, frame: Frame) -> Optional[Prediction]:
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(
            bgr,
            verbose=False,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
        )
        if not results:
            return None

        target_norm = self.target_class.casefold()
        exact, contains, n_candidates = [], [], 0

        for result in results:
            masks = getattr(result, "masks", None)
            boxes = getattr(result, "boxes", None)
            if masks is None or getattr(masks, "data", None) is None or boxes is None:
                continue
            data = np.asarray(_to_numpy(masks.data))
            confs = np.asarray(_to_numpy(boxes.conf)).reshape(-1)
            clses = np.asarray(_to_numpy(boxes.cls)).reshape(-1).astype(int)
            n = min(len(data), len(confs), len(clses))
            n_candidates += n
            for i in range(n):
                name = self._class_name(result, int(clses[i]))
                entry = (float(confs[i]), name, data[i])
                if name.casefold() == target_norm:
                    exact.append(entry)
                elif target_norm in name.casefold():
                    contains.append(entry)

        candidates = exact or contains
        if not candidates:
            return None

        conf, name, raw_mask = max(candidates, key=lambda e: e[0])
        mask = cv2.resize(
            np.asarray(raw_mask, dtype=np.float32),
            (frame.width, frame.height),
            interpolation=cv2.INTER_NEAREST,
        )
        return Prediction(
            mask=(mask > VENDOR_MASK_THRESHOLD).astype(np.uint8),
            class_name=name,
            confidence=conf,
            n_candidates=n_candidates,
            n_class_matches=len(candidates),
            note=f"selected by class match on {self.target_class!r}",
        )


def _to_numpy(value):
    """Torch tensor or array -> numpy, without importing torch."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def ultralytics_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def build_predictor(kind: str, **kwargs) -> MaskPredictor:
    """Explicit predictor selection. No environment sniffing, no fallback."""
    if kind == "saturation":
        return SaturationSegmenter(**kwargs)
    if kind == "yoloe":
        return YoloeSegmenter(**kwargs)          # raises PredictorUnavailable
    raise ValueError(f"unknown predictor {kind!r}; choose from {PREDICTOR_CHOICES}")
