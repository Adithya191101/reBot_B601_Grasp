"""Cartesian-linear waypoint planning for the reBot B601-DM.

Ports the PROVEN waypoint scheme from ``scripts/b601_banana_demo.py``'s
``move_linear`` (planning half; execution belongs to the node):

* **rotate-then-translate** -- aligning the jaw is a large WRIST
  reorientation (j4-j6); doing it mid-descent tripped the jump guard in the
  demo.  Orientation is corrected first, in place, slerped through
  ``pin.log3``/``pin.exp3`` waypoints; then position is interpolated on a
  straight TCP-space line at constant orientation.
* **local IK per waypoint**, seeded from the previous solution
  (:meth:`KinematicsCore.ik_local_dls`).  Waypoint continuity forbids branch
  jumps by construction; a required jump aborts the plan instead.  This is
  the fix for a measured failure: random-restart IK found a shoulder-flipped
  branch (exact FK, tracking error 4.000 rad on every attempt) whose
  joint-linear path swept the arm through the table.
* **shoulder-branch-jump guard** -- wrist rolls do not sweep the workspace;
  only shoulder-side jumps (j1-j3) are dangerous and stay strictly guarded
  during the reorientation stage.

Timing: :func:`time_waypoints` assigns per-segment durations from the
canonical URDF velocity limits (5,5,5,3,3,3 rad/s) scaled by a safety
factor, so the finite-difference velocity gate of the trajectory adapter
passes by construction.

This module is rclpy-free and must never import any ROS package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np
import pinocchio as pin

from .ik_core import ARM_DOF, IK_ERR_ACCEPT, KinematicsCore

#: demo defaults (move_linear)
DEFAULT_STEPS = 7
DEFAULT_BRANCH_JUMP_RAD = 0.7
DEFAULT_SHOULDER_JUMP_RAD = 0.6

#: max orientation slerp step per rotation-stage waypoint (rad).
DEFAULT_MAX_ROT_STEP_RAD = 0.35

#: shoulder joints guarded during reorientation (j1-j3, 0-based).
SHOULDER_JOINTS = slice(0, 3)


@dataclass(frozen=True)
class PlanResult:
    """Outcome of :func:`plan_linear`.

    ``waypoints`` includes the start configuration as waypoint 0; the
    rotation-stage waypoints follow, then the translation-stage waypoints.
    ``failed_at_waypoint`` counts from 1 in planning order (0 = the
    reorientation pre-checks), mirroring the demo's diagnostics.
    """

    ok: bool
    reason: str = ""
    waypoints: Tuple[Tuple[float, ...], ...] = ()
    n_rotation_waypoints: int = 0
    failed_at_waypoint: int = -1
    detail: Dict[str, object] = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def _fail(reason: str, waypoint: int, **detail) -> PlanResult:
    return PlanResult(ok=False, reason=reason, failed_at_waypoint=waypoint,
                      detail=dict(detail))


def plan_linear(
    kin: KinematicsCore,
    q_start: Sequence[float],
    T_tcp_goal: np.ndarray,
    *,
    steps: int = DEFAULT_STEPS,
    branch_jump_rad: float = DEFAULT_BRANCH_JUMP_RAD,
    shoulder_jump_rad: float = DEFAULT_SHOULDER_JUMP_RAD,
    max_rot_step_rad: float = DEFAULT_MAX_ROT_STEP_RAD,
    ik_err_accept: float = IK_ERR_ACCEPT,
) -> PlanResult:
    """Plan a Cartesian-linear TCP move as a joint waypoint sequence.

    Straight line in TCP space, local IK per waypoint seeded from the
    previous solution; rotate-then-translate with orientation slerp.
    """
    q_start = np.asarray(q_start, dtype=np.float64)[:ARM_DOF].copy()
    if not kin.within_limits(q_start, margin=0.0):
        return _fail("start configuration outside joint limits", 0,
                     violations=kin.limit_violations(q_start, margin=0.0))

    T_goal = np.asarray(T_tcp_goal, dtype=np.float64)
    T_from = kin.fk_tcp(q_start)
    pos_from, R_from = T_from[:3, 3].copy(), T_from[:3, :3].copy()
    pos_to, R_to = T_goal[:3, 3].copy(), T_goal[:3, :3].copy()

    # Orientation gap, slerped along pin.log3/exp3 (demo's diag quantity).
    w = pin.log3(R_from.T @ R_to)
    gap = float(np.linalg.norm(w))
    detail: Dict[str, object] = {
        "q_start": q_start.round(3).tolist(),
        "orientation_gap_deg": float(np.degrees(gap)),
        "translation_m": float(np.linalg.norm(pos_to - pos_from)),
    }

    waypoints = [tuple(q_start)]
    q_seed = q_start.copy()
    wp_index = 0

    # ---- stage 0: rotate in place (slerp) ------------------------------
    n_rot = 0 if gap < 1e-6 else max(1, int(math.ceil(gap / max_rot_step_rad)))
    for k in range(1, n_rot + 1):
        R_k = R_from @ pin.exp3(w * (k / n_rot))
        tcp_k = np.eye(4)
        tcp_k[:3, :3] = R_k
        tcp_k[:3, 3] = pos_from
        q6, ik_err = kin.solve_tcp(tcp_k, q_seed)
        wp_index += 1
        if ik_err > ik_err_accept:
            return _fail("in-place reorientation unreachable", wp_index,
                         ik_error=ik_err, **detail)
        if not kin.within_limits(q6):
            return _fail("outside joint limits", wp_index,
                         violations=kin.limit_violations(q6),
                         q6=q6.round(3).tolist(), **detail)
        # Wrist rolls are safe; SHOULDER motion during reorientation is the
        # dangerous branch change (measured drift against q_start, as in the
        # demo's pre-descent guard).
        shoulder_jump = float(
            np.max(np.abs(q6[SHOULDER_JOINTS] - q_start[SHOULDER_JOINTS])))
        if shoulder_jump > shoulder_jump_rad:
            return _fail("reorientation would move shoulder joints", wp_index,
                         shoulder_jump_rad=shoulder_jump, **detail)
        if float(np.max(np.abs(q6 - q_seed))) > branch_jump_rad:
            return _fail("branch jump refused", wp_index,
                         jump_rad=float(np.max(np.abs(q6 - q_seed))),
                         q6=q6.round(3).tolist(), **detail)
        waypoints.append(tuple(q6))
        q_seed = q6

    # ---- stage 1: translate at constant (goal) orientation -------------
    for k in range(1, steps + 1):
        t = k / steps
        pos_k = pos_from + (pos_to - pos_from) * t
        tcp_k = np.eye(4)
        tcp_k[:3, :3] = R_to  # orientation already reached in stage 0
        tcp_k[:3, 3] = pos_k
        q6, ik_err = kin.solve_tcp(tcp_k, q_seed)
        wp_index += 1
        if ik_err > ik_err_accept:
            return _fail("local IK failed", wp_index, ik_error=ik_err,
                         q6=q6.round(3).tolist(), **detail)
        if not kin.within_limits(q6):
            return _fail("outside joint limits", wp_index,
                         violations=kin.limit_violations(q6),
                         q6=q6.round(3).tolist(), **detail)
        if float(np.max(np.abs(q6 - q_seed))) > branch_jump_rad:
            return _fail("branch jump refused", wp_index,
                         jump_rad=float(np.max(np.abs(q6 - q_seed))),
                         q6=q6.round(3).tolist(), **detail)
        waypoints.append(tuple(q6))
        q_seed = q6

    return PlanResult(ok=True, waypoints=tuple(waypoints),
                      n_rotation_waypoints=n_rot, detail=detail)


# ---- timing -------------------------------------------------------------

DEFAULT_SAFETY_FACTOR = 0.5
DEFAULT_MIN_SEGMENT_SEC = 0.1


def time_waypoints(
    waypoints: Sequence[Sequence[float]],
    velocity_limits: Sequence[float],
    *,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    min_segment_sec: float = DEFAULT_MIN_SEGMENT_SEC,
) -> Tuple[float, ...]:
    """time_from_start for each waypoint (first at 0.0, strictly increasing).

    Each segment lasts long enough that every joint stays below
    ``velocity_limit * safety_factor``, floored at ``min_segment_sec`` --
    so the trajectory adapter's finite-difference velocity gate (canonical
    5/3 rad/s) passes by construction for any ``safety_factor`` <= 1.
    """
    if not 0.0 < safety_factor <= 1.0:
        raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}")
    if min_segment_sec <= 0.0:
        raise ValueError("min_segment_sec must be positive")
    vlim = np.asarray(velocity_limits, dtype=np.float64)[:ARM_DOF]
    if np.any(vlim <= 0.0):
        raise ValueError("velocity limits must be positive")

    times = [0.0]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        dq = np.abs(np.asarray(b, dtype=np.float64)[:ARM_DOF]
                    - np.asarray(a, dtype=np.float64)[:ARM_DOF])
        dt = float(np.max(dq / (vlim * safety_factor)))
        times.append(times[-1] + max(dt, min_segment_sec))
    return tuple(times)


def max_segment_velocity(
    waypoints: Sequence[Sequence[float]],
    times: Sequence[float],
) -> np.ndarray:
    """Per-joint max finite-difference velocity across all segments."""
    worst = np.zeros(ARM_DOF)
    for (a, b, ta, tb) in zip(waypoints[:-1], waypoints[1:],
                              times[:-1], times[1:]):
        dq = np.abs(np.asarray(b, dtype=np.float64)[:ARM_DOF]
                    - np.asarray(a, dtype=np.float64)[:ARM_DOF])
        worst = np.maximum(worst, dq / (tb - ta))
    return worst
