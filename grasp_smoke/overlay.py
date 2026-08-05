"""Debug overlay: RGB + mask + min-area rect + opening axis + predicted vs GT grasp.

PLAN.md 5.2.7 calls this the highest-value debugging artifact in the project and
also the demo video. Everything the grasp decision depends on is drawn, so a bad
result can be diagnosed by looking rather than by instrumenting.

GT is projected back into the image through the same intrinsics and extrinsics the
prediction used -- so if the frames are wrong, the GT marker lands visibly in the
wrong place instead of silently agreeing.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .geometry import invert_transform, project, transform_direction, transform_point

COLOR_MASK = (0, 200, 255)      # BGR amber
COLOR_RECT = (255, 200, 0)      # cyan-ish
COLOR_AXIS = (255, 255, 255)    # white: predicted opening axis
COLOR_PRED = (0, 0, 255)        # red: predicted grasp point
COLOR_GT = (0, 255, 0)          # green: ground-truth grasp
COLOR_TEXT = (0, 255, 255)


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)


def render_overlay(
    frame,
    estimate,
    predicted_mask: Optional[np.ndarray],
    labels=None,
    score=None,
    branch: str = "",
) -> np.ndarray:
    """Return a BGR image ready for ``cv2.imwrite``."""
    canvas = _to_bgr(frame.rgb)

    if predicted_mask is not None and np.any(predicted_mask):
        tint = np.zeros_like(canvas)
        tint[predicted_mask > 0] = COLOR_MASK
        canvas = cv2.addWeighted(canvas, 1.0, tint, 0.30, 0.0)
        contours, _ = cv2.findContours(
            (predicted_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, COLOR_MASK, 1, cv2.LINE_AA)

    if estimate is not None and estimate.rect_points is not None:
        cv2.polylines(
            canvas, [np.round(estimate.rect_points).astype(np.int32)],
            True, COLOR_RECT, 2, cv2.LINE_AA,
        )

    if estimate is not None and estimate.short_edge_points is not None:
        p0, p1 = np.round(estimate.short_edge_points).astype(np.int32)
        cv2.line(canvas, tuple(p0), tuple(p1), COLOR_AXIS, 3, cv2.LINE_AA)
        # Jaw ticks, so the opening direction is unambiguous at a glance.
        for p in (p0, p1):
            cv2.circle(canvas, tuple(p), 4, COLOR_AXIS, -1, cv2.LINE_AA)

    if estimate is not None and estimate.center_px is not None:
        cv2.circle(canvas, tuple(estimate.center_px), 5, COLOR_PRED, -1, cv2.LINE_AA)

    # Ground truth, projected through the same camera model.
    if labels is not None:
        try:
            T_cam_base = invert_transform(frame.T_base_cam)
            gt_cam = transform_point(T_cam_base, labels.grasp_gt_position)
            if float(gt_cam[2]) > 1e-6:
                uv = project(gt_cam, frame.K)
                cv2.drawMarker(
                    canvas, (int(round(uv[0])), int(round(uv[1]))), COLOR_GT,
                    cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA,
                )
                axis_cam = transform_direction(T_cam_base, labels.grasp_gt_open_axis)
                half = 0.5 * max(float(getattr(estimate, "jaw_width_m", 0.0) or 0.0), 0.03)
                for sign in (-1.0, 1.0):
                    end = gt_cam + sign * half * axis_cam
                    if float(end[2]) > 1e-6:
                        uv_end = project(end, frame.K)
                        cv2.line(
                            canvas, (int(round(uv[0])), int(round(uv[1]))),
                            (int(round(uv_end[0])), int(round(uv_end[1]))),
                            COLOR_GT, 2, cv2.LINE_AA,
                        )
        except (ValueError, AttributeError):
            pass  # GT behind the camera is a scene problem, not an overlay problem

    lines = [f"{frame.scene_id}  branch={branch}"]
    if score is not None:
        if score.true_positive and score.position_error_m is not None:
            lines.append(
                f"IoU {score.iou:.3f}  pos {score.position_error_m*1000:.2f} mm  "
                f"axis {score.opening_axis_error_deg:.2f} deg  "
                f"{'PASS' if score.within_tolerance else 'OUT OF TOL'}"
            )
        elif not score.target_present:
            lines.append(f"target absent  detected={score.detected}")
        else:
            lines.append(f"IoU {score.iou:.3f}  no true positive "
                         f"({score.rejected_reason or 'iou below threshold'})")
    if estimate is not None and estimate.is_valid:
        lines.append(f"z {estimate.z_m:.3f} m  jaw {estimate.jaw_width_m*100:.1f} cm")

    y = 22
    for line in lines:
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
        y += 22

    legend = [("pred grasp", COLOR_PRED), ("GT grasp", COLOR_GT),
              ("opening axis", COLOR_AXIS), ("min-area rect", COLOR_RECT)]
    y = canvas.shape[0] - 10 - 18 * len(legend)
    for text, color in legend:
        cv2.line(canvas, (10, y - 4), (34, y - 4), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += 18

    return canvas
