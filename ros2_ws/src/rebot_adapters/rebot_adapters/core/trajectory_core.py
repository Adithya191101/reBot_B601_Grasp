"""Pure validation/conversion core for the reBot trajectory adapter.

Implements every pre-forwarding validation gate from design doc sec. 9.2:

1.  joint names must be exactly the six canonical arm joints;
2.  reorder by name only if configured; default is reject;
3.  every point must contain six finite positions;
4.  optional velocity/acceleration arrays must be empty or length six;
5.  ``time_from_start`` must be strictly increasing;
6.  position limits must fit the canonical URDF;
7.  finite-difference segment velocity must not exceed the canonical
    5/3 rad/s limits;
8.  only one downstream goal may be active per adapter instance
    (:class:`SingleGoalGate`).

Gates 9 (cancellation propagation) and 10 (result/feedback mapping) are
node-level responsibilities; the node uses this module for everything that
can be decided without ROS.

This module is pure Python and must never import rclpy.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from .limits import ARM_JOINT_NAMES, ARM_POSITION_LIMITS, ARM_VELOCITY_LIMITS

# control_msgs/action/FollowJointTrajectory result error codes.
SUCCESSFUL = 0
INVALID_GOAL = -1
INVALID_JOINTS = -2

_EPS_POS = 1e-9   # absolute slack on position bounds
_EPS_VEL = 1e-9   # relative slack on velocity bounds


@dataclass(frozen=True)
class TrajectoryPoint:
    """ROS-free mirror of trajectory_msgs/JointTrajectoryPoint."""

    time_from_start: float                      # seconds
    positions: Tuple[float, ...]                # rad
    velocities: Tuple[float, ...] = ()          # rad/s, empty or len 6
    accelerations: Tuple[float, ...] = ()       # rad/s^2, empty or len 6


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of :func:`validate_trajectory`.

    ``error_code`` uses FollowJointTrajectory result codes:
    0 = SUCCESSFUL, -1 = INVALID_GOAL, -2 = INVALID_JOINTS.
    On success ``joint_names``/``points`` hold the normalized trajectory
    (canonical joint order) ready for downstream forwarding.
    """

    ok: bool
    error_code: int = SUCCESSFUL
    reason: str = ""
    joint_names: Tuple[str, ...] = ()
    points: Tuple[TrajectoryPoint, ...] = ()

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def _fail(code: int, reason: str) -> ValidationResult:
    return ValidationResult(ok=False, error_code=code, reason=reason)


def _all_finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def _reorder(values: Tuple[float, ...], perm: Sequence[int]) -> Tuple[float, ...]:
    return tuple(values[i] for i in perm)


def validate_trajectory(
    joint_names: Sequence[str],
    points: Sequence[TrajectoryPoint],
    *,
    reorder_joint_names: bool = False,
    validate_segment_velocity: bool = True,
    position_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    velocity_limits: Optional[Dict[str, float]] = None,
) -> ValidationResult:
    """Run all static validation gates on a FollowJointTrajectory goal.

    Returns a :class:`ValidationResult`; when ``ok`` the returned
    ``joint_names``/``points`` are in canonical order (columns permuted if
    ``reorder_joint_names`` allowed a name-order mismatch).
    """
    position_limits = position_limits or ARM_POSITION_LIMITS
    velocity_limits = velocity_limits or ARM_VELOCITY_LIMITS

    names = tuple(joint_names)

    # Gate 1/2: joint-name set, duplicates, order.
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        return _fail(INVALID_JOINTS, f"duplicate joint names: {dupes}")
    canonical = set(ARM_JOINT_NAMES)
    unknown = sorted(set(names) - canonical)
    if unknown:
        return _fail(INVALID_JOINTS, f"unknown joint names: {unknown}")
    missing = sorted(canonical - set(names))
    if missing:
        return _fail(INVALID_JOINTS, f"missing joint names: {missing}")
    if names != ARM_JOINT_NAMES:
        if not reorder_joint_names:
            return _fail(
                INVALID_JOINTS,
                "joint names not in canonical order "
                f"{list(ARM_JOINT_NAMES)} (got {list(names)}); "
                "reorder_joint_names is disabled",
            )
        perm: Tuple[int, ...] = tuple(names.index(n) for n in ARM_JOINT_NAMES)
    else:
        perm = tuple(range(len(ARM_JOINT_NAMES)))

    if not points:
        return _fail(INVALID_GOAL, "trajectory has no points")

    n = len(ARM_JOINT_NAMES)
    normalized = []
    prev_t: Optional[float] = None
    for idx, pt in enumerate(points):
        # Gate 3: six finite positions per point.
        if len(pt.positions) != n:
            return _fail(
                INVALID_GOAL,
                f"point {idx}: expected {n} positions, got {len(pt.positions)}",
            )
        if not _all_finite(pt.positions):
            return _fail(INVALID_GOAL, f"point {idx}: non-finite position")

        # Gate 4: optional arrays empty or length six (and finite).
        for label, arr in (("velocities", pt.velocities),
                           ("accelerations", pt.accelerations)):
            if len(arr) not in (0, n):
                return _fail(
                    INVALID_GOAL,
                    f"point {idx}: {label} must be empty or length {n}, "
                    f"got {len(arr)}",
                )
            if arr and not _all_finite(arr):
                return _fail(INVALID_GOAL, f"point {idx}: non-finite {label}")

        # Gate 5: strictly increasing, non-negative time_from_start.
        t = pt.time_from_start
        if not math.isfinite(t) or t < 0.0:
            return _fail(
                INVALID_GOAL,
                f"point {idx}: invalid time_from_start {t!r}",
            )
        if prev_t is not None and t <= prev_t:
            return _fail(
                INVALID_GOAL,
                f"point {idx}: time_from_start {t} not strictly greater "
                f"than previous {prev_t}",
            )
        prev_t = t

        positions = _reorder(tuple(pt.positions), perm)
        velocities = _reorder(tuple(pt.velocities), perm) if pt.velocities else ()
        accelerations = (
            _reorder(tuple(pt.accelerations), perm) if pt.accelerations else ()
        )

        # Gate 6: canonical URDF position limits.
        for joint, q in zip(ARM_JOINT_NAMES, positions):
            lo, hi = position_limits[joint]
            if q < lo - _EPS_POS or q > hi + _EPS_POS:
                return _fail(
                    INVALID_GOAL,
                    f"point {idx}: {joint} position {q} outside "
                    f"canonical limits [{lo}, {hi}]",
                )

        normalized.append(
            TrajectoryPoint(
                time_from_start=t,
                positions=positions,
                velocities=velocities,
                accelerations=accelerations,
            )
        )

    # Gate 7: finite-difference segment velocity vs 5/3 rad/s limits.
    if validate_segment_velocity:
        for idx in range(1, len(normalized)):
            a, b = normalized[idx - 1], normalized[idx]
            dt = b.time_from_start - a.time_from_start  # > 0 by gate 5
            for joint, qa, qb in zip(ARM_JOINT_NAMES, a.positions, b.positions):
                vmax = velocity_limits[joint]
                v = abs(qb - qa) / dt
                if v > vmax * (1.0 + _EPS_VEL) + _EPS_VEL:
                    return _fail(
                        INVALID_GOAL,
                        f"segment {idx - 1}->{idx}: {joint} finite-difference "
                        f"velocity {v:.4f} rad/s exceeds limit {vmax} rad/s",
                    )

    return ValidationResult(
        ok=True,
        error_code=SUCCESSFUL,
        reason="",
        joint_names=ARM_JOINT_NAMES,
        points=tuple(normalized),
    )


def trajectory_duration(points: Sequence[TrajectoryPoint]) -> float:
    """Duration in seconds of a (validated) trajectory."""
    return points[-1].time_from_start if points else 0.0


class SingleGoalGate:
    """Gate 8: at most one downstream goal active per adapter instance.

    Thread-safe; owner is identified by an opaque hashable goal id.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Optional[object] = None

    @property
    def active_goal(self) -> Optional[object]:
        with self._lock:
            return self._active

    def try_acquire(self, goal_id: object) -> bool:
        """Claim the gate; False if another goal is already active."""
        if goal_id is None:
            raise ValueError("goal_id must not be None")
        with self._lock:
            if self._active is not None:
                return False
            self._active = goal_id
            return True

    def release(self, goal_id: object) -> bool:
        """Release the gate; only the owning goal id may release it."""
        with self._lock:
            if self._active == goal_id:
                self._active = None
                return True
            return False
