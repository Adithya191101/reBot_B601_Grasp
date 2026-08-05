"""Grasp estimation from a mask + optical-axis depth + intrinsics.

A faithful reimplementation of the vendor pipeline in
``src/reBot-DevArm-Grasp/utils/ordinary_grasp.py``, with two deliberate
divergences, both recorded in PLAN.md:

1. **Input is a mask, not an ultralytics ``Results`` object.** The vendor couples
   detection and geometry through the model's result type; splitting them is what
   lets the oracle branch (A1) and the predicted branch (B) run the *same*
   geometry code with different mask sources.
2. **Depth is metres, not millimetres.** The vendor divides by 1000 at
   ``ordinary_grasp.py:149``; the dataset stores metres, so there is nothing to
   divide.

Everything else -- min-area rect, short-edge selection, the mask cross-section
refinement, the depth quantile, the ``-position`` approach vector, the
orthogonalisation order, the sign convention -- follows the vendor exactly, so
that measured error is attributable to the algorithm rather than to a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .geometry import backproject, normalize, pixel_vec_to_3d

# Vendor code default is 0.75 (ordinary_grasp.py:56,103); the shipped config
# overrides it to 0.5 (config/default.yaml:61). Neither is "the" value -- the
# manifest records which one produced a given dataset. See PLAN.md 5.2.4.
VENDOR_CODE_DEPTH_QUANTILE = 0.75
VENDOR_CONFIG_DEPTH_QUANTILE = 0.5


@dataclass
class GraspEstimate:
    """A grasp in the **optical camera frame**, plus the 2-D evidence for it."""

    position: Optional[np.ndarray]      # (3,) metres, optical frame
    open_axis: Optional[np.ndarray]     # (3,) unit, optical frame
    grip_axis: Optional[np.ndarray]
    approach: Optional[np.ndarray]
    rotation: Optional[np.ndarray]      # (3,3) columns [grip, open, approach]
    center_px: Optional[tuple]
    rect_points: Optional[np.ndarray]   # (4,2) min-area rect
    short_edge_points: Optional[np.ndarray]  # (2,2) the grasp line
    jaw_width_m: float = 0.0
    object_length_m: float = 0.0
    z_m: float = 0.0
    valid_depth_pixels: int = 0
    rejected_reason: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return (
            self.rejected_reason is None
            and self.position is not None
            and self.open_axis is not None
        )


def _reject(reason: str, **kw) -> GraspEstimate:
    base = dict(
        position=None, open_axis=None, grip_axis=None, approach=None,
        rotation=None, center_px=None, rect_points=None, short_edge_points=None,
    )
    base.update(kw)
    return GraspEstimate(rejected_reason=reason, **base)


def rect_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Min-area rectangle of the largest contour. Vendor ``_rect_from_mask``."""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    return cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32))).astype(np.float64)


def short_edge(rect_points: np.ndarray) -> tuple:
    """Shortest edge of the rect as (vector, length). Vendor ``_short_edge``."""
    best_vec = rect_points[1] - rect_points[0]
    best_len = float(np.linalg.norm(best_vec))
    for i in range(4):
        vec = rect_points[(i + 1) % 4] - rect_points[i]
        length = float(np.linalg.norm(vec))
        if length < best_len:
            best_vec, best_len = vec, length
    return np.asarray(best_vec, dtype=np.float64), best_len


def _line_from_center(center: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.stack([center - 0.5 * vec, center + 0.5 * vec], axis=0)


def refine_grasp_line_from_mask(
    mask: np.ndarray,
    center: np.ndarray,
    short_dir_uv: np.ndarray,
    long_len_px: float,
) -> Optional[tuple]:
    """Vendor ``_refine_grasp_line_from_mask``, unchanged in behaviour.

    The short-axis *direction* still comes from the rect; only the grasp centre
    and span are replaced by the mask's actual cross-section at its median
    longitudinal slice.
    """
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 32:
        return None

    points = np.column_stack([xs, ys]).astype(np.float64)
    grip_dir_uv = np.array([-short_dir_uv[1], short_dir_uv[0]], dtype=np.float64)
    rel = points - center.reshape(1, 2)
    grip_coord = rel @ grip_dir_uv
    open_coord = rel @ short_dir_uv

    grip_center = float(np.median(grip_coord))
    band = float(np.clip(long_len_px * 0.04, 2.0, 12.0))
    band_mask = np.abs(grip_coord - grip_center) <= band
    if int(np.count_nonzero(band_mask)) < 24:
        band = float(np.clip(long_len_px * 0.08, 4.0, 18.0))
        band_mask = np.abs(grip_coord - grip_center) <= band
    if int(np.count_nonzero(band_mask)) < 24:
        return None

    band_open = open_coord[band_mask]
    open_min = float(np.percentile(band_open, 5.0))
    open_max = float(np.percentile(band_open, 95.0))
    span = open_max - open_min
    if span < 2.0:
        return None

    open_center = 0.5 * (open_min + open_max)
    refined = center + grip_center * grip_dir_uv + open_center * short_dir_uv
    return refined, _line_from_center(refined, short_dir_uv * span), float(span)


def estimate_grasp(
    mask: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    depth_quantile: float = VENDOR_CODE_DEPTH_QUANTILE,
) -> GraspEstimate:
    """mask + optical-axis Z depth (metres) + K -> grasp in the optical frame.

    ``mask`` is any non-zero/zero array of the image shape. ``depth_m`` uses
    <= 0 to mean "no return", matching the vendor's treatment of 0 in millimetre
    maps.
    """
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    if mask.shape != depth_m.shape:
        raise ValueError(f"mask {mask.shape} and depth {depth_m.shape} disagree")

    rect_points = rect_from_mask(mask)
    if rect_points is None:
        return _reject("no_contour")

    center = rect_points.mean(axis=0)
    short_vec_uv, short_len_px = short_edge(rect_points)
    short_dir_uv = normalize(short_vec_uv)
    edge_lengths = [
        float(np.linalg.norm(rect_points[(i + 1) % 4] - rect_points[i])) for i in range(4)
    ]
    long_len_px = max(edge_lengths)
    grasp_span_px = short_len_px
    short_edge_points = _line_from_center(center, short_vec_uv)

    if short_dir_uv is not None:
        refined = refine_grasp_line_from_mask(mask, center, short_dir_uv, long_len_px)
        if refined is not None:
            center, short_edge_points, grasp_span_px = refined

    center_px = (int(round(float(center[0]))), int(round(float(center[1]))))

    depth_values = depth_m[mask > 0]
    depth_values = depth_values[depth_values > 0]
    if len(depth_values) == 0 or short_dir_uv is None:
        return _reject(
            "no_valid_depth_or_rect",
            rect_points=rect_points,
            short_edge_points=short_edge_points,
            center_px=center_px,
            valid_depth_pixels=0,
        )

    z_m = float(np.quantile(depth_values, float(np.clip(depth_quantile, 0.0, 1.0))))
    position = backproject(float(center[0]), float(center[1]), z_m, K)

    approach = normalize(-position)
    if approach is None:
        approach = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    open_axis = pixel_vec_to_3d(short_dir_uv, z_m, K)
    open_axis = open_axis - float(np.dot(open_axis, approach)) * approach
    open_axis = normalize(open_axis)
    if open_axis is None:
        return _reject(
            "open_axis_failed",
            rect_points=rect_points,
            short_edge_points=short_edge_points,
            center_px=center_px,
            valid_depth_pixels=int(len(depth_values)),
        )

    # Vendor sign convention. Harmless for scoring: theta_open uses |dot|.
    if open_axis[0] < 0:
        open_axis = -open_axis
    grip_axis = normalize(np.cross(open_axis, approach))
    if grip_axis is None:
        return _reject(
            "grasp_axis_failed",
            rect_points=rect_points,
            short_edge_points=short_edge_points,
            center_px=center_px,
            valid_depth_pixels=int(len(depth_values)),
        )
    open_axis = normalize(np.cross(approach, grip_axis))

    return GraspEstimate(
        position=position,
        open_axis=open_axis,
        grip_axis=grip_axis,
        approach=approach,
        rotation=np.column_stack([grip_axis, open_axis, approach]),
        center_px=center_px,
        rect_points=rect_points,
        short_edge_points=short_edge_points,
        jaw_width_m=float(
            np.linalg.norm(pixel_vec_to_3d(short_dir_uv * grasp_span_px, z_m, K))
        ),
        object_length_m=float(
            np.linalg.norm(pixel_vec_to_3d(short_dir_uv * long_len_px, z_m, K))
        ),
        z_m=z_m,
        valid_depth_pixels=int(len(depth_values)),
    )
