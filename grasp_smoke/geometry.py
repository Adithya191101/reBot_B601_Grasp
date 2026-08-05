"""Pinhole geometry and rigid transforms.

Pure NumPy. No ROS, no Isaac Sim, no OpenCV. This module is the target of the
A0 analytic red tests (PLAN.md 5.2.4): every function here is exercised by
fronto-parallel constant-depth fixtures with a closed-form expected answer, and
a failure means the code is wrong, not that the algorithm is approximating.

Frame conventions, fixed here once:

* **Optical camera frame** -- x right, y down, z forward along the optical axis.
  This is the frame the pinhole model back-projects into, and the frame depth is
  measured along (optical-axis Z, metres). It is *not* REP-103.
* **Base frame** -- the world frame the dataset's ``T_base_cam`` maps into.
  ``T_base_cam`` is stored as base <- optical directly, so no REP-103 rotation
  is implied or applied anywhere in this codebase.

Depth is **metres** throughout. The vendor pipeline carries millimetres and
divides by 1000 at the point of use (``ordinary_grasp.py:149``); we store metres
in the dataset and never convert.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

EPS = 1e-8


def normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    """Unit vector, or None if the input is degenerate.

    Mirrors the vendor's ``_normalize`` including its 1e-8 cutoff.
    """
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < EPS:
        return None
    return vec / norm


def backproject(u: float, v: float, z_m: float, K: np.ndarray) -> np.ndarray:
    """Pixel + optical-axis depth -> 3-D point in the optical camera frame.

    Identical to the vendor's ``_backproject``.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.array(
        [(u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m], dtype=np.float64
    )


def pixel_vec_to_3d(vec_uv: np.ndarray, z_m: float, K: np.ndarray) -> np.ndarray:
    """Image-plane direction -> 3-D vector at constant depth ``z_m``.

    Identical to the vendor's ``_pixel_vec_to_3d``: the z component is zero, so
    the result lies in the plane of constant depth. This is the step that makes
    the recovered opening axis exact only for a fronto-parallel object -- see
    PLAN.md 5.2.4 on why A2 error is legitimate rather than a bug.
    """
    fx = max(float(K[0, 0]), 1e-6)
    fy = max(float(K[1, 1]), 1e-6)
    return np.array(
        [float(vec_uv[0]) * z_m / fx, float(vec_uv[1]) * z_m / fy, 0.0],
        dtype=np.float64,
    )


def project(point_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Optical-frame 3-D point -> pixel. Inverse of :func:`backproject`."""
    x, y, z = (float(c) for c in point_cam)
    if abs(z) < EPS:
        raise ValueError("cannot project a point at zero depth")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.array([fx * x / z + cx, fy * y / z + cy], dtype=np.float64)


def make_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


# --------------------------------------------------------------------------
# Rigid transforms. 4x4 homogeneous, base <- optical.
# --------------------------------------------------------------------------


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(3)
    return (T[:3, :3] @ p) + T[:3, 3]


def transform_direction(T: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a direction. No translation -- directions are not points."""
    return T[:3, :3] @ np.asarray(v, dtype=np.float64).reshape(3)


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build base <- optical for a camera at ``eye`` looking at ``target``.

    Columns of the rotation are the optical axes expressed in base:
    +z forward (eye -> target), +x right, +y down.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    forward = normalize(target - eye)
    if forward is None:
        raise ValueError("camera eye and target coincide")
    right = normalize(np.cross(forward, np.asarray(up, dtype=np.float64)))
    if right is None:
        raise ValueError("up vector is parallel to the view direction")
    down = np.cross(forward, right)
    return make_transform(np.column_stack([right, down, forward]), eye)


def quaternion_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (x, y, z, w), the ROS ordering."""
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def rotation_from_quaternion(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> rotation matrix."""
    x, y, z, w = (float(c) for c in q)
    n = x * x + y * y + z * z + w * w
    if n < EPS:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis."""
    a = normalize(axis)
    if a is None:
        raise ValueError("degenerate rotation axis")
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64)
    return np.eye(3) + s * C + (1.0 - c) * (C @ C)


def opening_axis_error_rad(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """theta_open = acos(clamp(|o_hat . o_gt|, 0, 1)).

    The absolute value folds in the 180-degree symmetry of a parallel-jaw
    gripper: o and -o are the same grasp (PLAN.md 5.2.3). The clamp keeps
    floating-point dot products marginally outside [-1, 1] from producing NaN,
    which is the failure mode this function exists to prevent.
    """
    p = normalize(predicted)
    g = normalize(ground_truth)
    if p is None or g is None:
        raise ValueError("cannot compare degenerate opening axes")
    return float(np.arccos(np.clip(abs(float(np.dot(p, g))), 0.0, 1.0)))
