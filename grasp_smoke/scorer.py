"""Scoring: pose error, opening-axis error, IoU, and honest yield accounting.

Implements PLAN.md 5.2.4. Three things it refuses to conflate:

* **Conditional pose error** -- computed over true positives only. On its own it
  flatters a detector that only fires on easy scenes.
* **End-to-end yield** -- of *all* present scenes, the fraction that produced a
  detection AND landed inside tolerance. This is the number that cannot be gamed.
* **False positives** -- detections on target-absent scenes, counted separately
  and never folded into the pose statistics.

Reported together, or not at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .geometry import opening_axis_error_rad, transform_direction, transform_point

#: Per-stratum tolerances from PLAN.md 5.2.4. A0 is enforced by unit tests.
TOLERANCES = {
    "A0": {"position_m": 0.001, "angle_deg": 1.0},
    "A1": {"position_m": 0.003, "angle_deg": 3.0},
    "A2": {"position_m": 0.003, "angle_deg": 3.0},   # reported per tilt, not pooled
    "B":  {"position_m": 0.005, "angle_deg": 5.0},
}


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b)) / union


@dataclass
class SceneScore:
    scene_id: str
    branch: str
    target_present: bool
    detected: bool
    iou: float = 0.0
    true_positive: bool = False
    false_positive: bool = False
    position_error_m: Optional[float] = None
    opening_axis_error_deg: Optional[float] = None
    within_tolerance: bool = False
    rejected_reason: Optional[str] = None
    tilt_deg: float = 0.0


def score_scene(
    scene_id: str,
    branch: str,
    estimate,
    predicted_mask: Optional[np.ndarray],
    labels,
    T_base_cam: np.ndarray,
    iou_match_threshold: float = 0.5,
    tilt_deg: float = 0.0,
) -> SceneScore:
    """Score one scene. ``estimate`` is a :class:`~grasp_smoke.grasp.GraspEstimate`."""
    tol = TOLERANCES.get(branch, TOLERANCES["B"])
    detected = predicted_mask is not None and estimate is not None and estimate.is_valid

    score = SceneScore(
        scene_id=scene_id,
        branch=branch,
        target_present=bool(labels.target_present),
        detected=bool(detected),
        tilt_deg=float(tilt_deg),
        rejected_reason=(estimate.rejected_reason if estimate is not None else "no_mask"),
    )

    if not labels.target_present:
        # Any detection on an absent-target scene is a false positive, full stop.
        score.false_positive = bool(detected)
        return score

    if predicted_mask is not None:
        score.iou = mask_iou(predicted_mask, labels.gt_mask)

    if not detected:
        return score

    if score.iou < iou_match_threshold:
        score.false_positive = True
        return score

    score.true_positive = True
    pos_base = transform_point(T_base_cam, estimate.position)
    axis_base = transform_direction(T_base_cam, estimate.open_axis)
    score.position_error_m = float(np.linalg.norm(pos_base - labels.grasp_gt_position))
    score.opening_axis_error_deg = float(
        np.rad2deg(opening_axis_error_rad(axis_base, labels.grasp_gt_open_axis))
    )
    score.within_tolerance = (
        score.position_error_m <= tol["position_m"]
        and score.opening_axis_error_deg <= tol["angle_deg"]
    )
    return score


def _percentile(values: list, q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values else None


def _bootstrap_ci(values: list, q: float, n: int = 2000, seed: int = 0) -> Optional[list]:
    """Percentile bootstrap CI. Ten scenes is far too few for a tight interval --
    printing the interval is how that stays visible instead of implied away."""
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    stats = [
        float(np.percentile(rng.choice(arr, size=arr.size, replace=True), q))
        for _ in range(n)
    ]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


@dataclass
class BranchSummary:
    branch: str
    n_scenes: int = 0
    n_present: int = 0
    n_absent: int = 0
    n_true_positive: int = 0
    n_false_positive: int = 0
    recall: Optional[float] = None
    false_positive_rate_absent: Optional[float] = None
    position_median_mm: Optional[float] = None
    position_p90_mm: Optional[float] = None
    position_median_ci_mm: Optional[list] = None
    angle_median_deg: Optional[float] = None
    angle_p90_deg: Optional[float] = None
    angle_median_ci_deg: Optional[list] = None
    end_to_end_yield: Optional[float] = None
    tolerances: dict = field(default_factory=dict)
    per_tilt: dict = field(default_factory=dict)


def summarize(scores: list, branch: str) -> BranchSummary:
    rows = [s for s in scores if s.branch == branch]
    present = [s for s in rows if s.target_present]
    absent = [s for s in rows if not s.target_present]
    tps = [s for s in present if s.true_positive]

    pos_mm = [s.position_error_m * 1000.0 for s in tps if s.position_error_m is not None]
    ang = [s.opening_axis_error_deg for s in tps if s.opening_axis_error_deg is not None]

    summary = BranchSummary(
        branch=branch,
        n_scenes=len(rows),
        n_present=len(present),
        n_absent=len(absent),
        n_true_positive=len(tps),
        n_false_positive=sum(1 for s in rows if s.false_positive),
        recall=(len(tps) / len(present)) if present else None,
        false_positive_rate_absent=(
            sum(1 for s in absent if s.detected) / len(absent) if absent else None
        ),
        position_median_mm=_percentile(pos_mm, 50),
        position_p90_mm=_percentile(pos_mm, 90),
        position_median_ci_mm=_bootstrap_ci(pos_mm, 50),
        angle_median_deg=_percentile(ang, 50),
        angle_p90_deg=_percentile(ang, 90),
        angle_median_ci_deg=_bootstrap_ci(ang, 50),
        # Denominator is every present scene, not just the detected ones.
        end_to_end_yield=(
            sum(1 for s in present if s.within_tolerance) / len(present) if present else None
        ),
        tolerances=TOLERANCES.get(branch, TOLERANCES["B"]),
    )

    # A2 is reported per tilt; pooling hides the trend that is the whole point.
    tilts = sorted({s.tilt_deg for s in present})
    if len(tilts) > 1:
        for tilt in tilts:
            sub = [s for s in present if s.tilt_deg == tilt and s.true_positive]
            summary.per_tilt[f"{tilt:g}"] = {
                "n_true_positive": len(sub),
                "position_median_mm": _percentile(
                    [s.position_error_m * 1000.0 for s in sub], 50),
                "angle_median_deg": _percentile(
                    [s.opening_axis_error_deg for s in sub], 50),
            }
    return summary


def write_results(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=_json_default) + "\n")
    return path


def _json_default(obj):
    if isinstance(obj, (SceneScore, BranchSummary)):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"cannot serialise {type(obj)}")
