"""Pure trajectory-sampling core for the sim-JTC shim.

The Isaac side of the M5 profile consumes plain JointState position targets
on ``/isaac_joint_commands`` each physics tick (the rearm environment's
verified closed loop); FollowJointTrajectory semantics therefore live here,
container-side.  This module holds the ROS-free logic: goal validation and
time-parameterized linear interpolation, matching the upstream Seeed FJT
behavior the design doc records ("linearly interpolates between points",
sec. 9.2) so sim and real execute goals the same way.

This module is pure Python and must never import rclpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


class TrajectoryGoalError(ValueError):
    """Raised when an FJT goal violates the shim's contract."""


@dataclass(frozen=True)
class ShimPoint:
    """ROS-free mirror of trajectory_msgs/JointTrajectoryPoint."""

    time_from_start: float
    positions: Tuple[float, ...]


@dataclass(frozen=True)
class ShimTrajectory:
    """A validated goal, positions reordered to the shim's joint order."""

    joint_names: Tuple[str, ...]
    points: Tuple[ShimPoint, ...]

    @property
    def duration(self) -> float:
        return self.points[-1].time_from_start


def validate_goal(
    configured_joints: Sequence[str],
    goal_joint_names: Sequence[str],
    points: Sequence[ShimPoint],
) -> ShimTrajectory:
    """Validate an FJT goal and reorder its columns to configured order.

    * goal joints must be exactly the configured set (any order);
    * at least one point; every point full-width, finite positions;
    * ``time_from_start`` strictly increasing and positive.

    Velocity/limit gates are NOT duplicated here: the canonical trajectory
    adapter already enforced them upstream (design doc sec. 9.2 gates 1-7);
    the shim is the downstream controller.
    """
    configured = tuple(configured_joints)
    names = tuple(goal_joint_names)
    if sorted(names) != sorted(configured):
        raise TrajectoryGoalError(
            f"joint names {list(names)} != configured {list(configured)}")
    if not points:
        raise TrajectoryGoalError("empty trajectory")

    column = [names.index(j) for j in configured]
    previous_t = 0.0
    reordered = []
    for k, point in enumerate(points):
        if len(point.positions) != len(configured):
            raise TrajectoryGoalError(
                f"point {k}: {len(point.positions)} positions, "
                f"want {len(configured)}")
        if not all(math.isfinite(p) for p in point.positions):
            raise TrajectoryGoalError(f"point {k}: non-finite position")
        if point.time_from_start <= previous_t:
            raise TrajectoryGoalError(
                f"point {k}: time_from_start {point.time_from_start} not "
                f"strictly increasing (previous {previous_t})")
        previous_t = point.time_from_start
        reordered.append(ShimPoint(
            time_from_start=point.time_from_start,
            positions=tuple(point.positions[c] for c in column)))
    return ShimTrajectory(joint_names=configured, points=tuple(reordered))


def sample(
    trajectory: ShimTrajectory,
    start_positions: Sequence[float],
    elapsed: float,
) -> Tuple[float, ...]:
    """Linearly interpolated position command at ``elapsed`` seconds.

    Before the first point's time the command ramps from the measured start
    positions (matching how a JTC treats the implicit t=0 state); past the
    last point it holds the final positions.
    """
    points = trajectory.points
    if elapsed >= points[-1].time_from_start:
        return points[-1].positions

    prev_t = 0.0
    prev_q: Tuple[float, ...] = tuple(float(v) for v in start_positions)
    for point in points:
        if elapsed < point.time_from_start:
            span = point.time_from_start - prev_t
            f = 0.0 if span <= 0.0 else (elapsed - prev_t) / span
            return tuple(a + f * (b - a)
                         for a, b in zip(prev_q, point.positions))
        prev_t = point.time_from_start
        prev_q = point.positions
    return points[-1].positions  # pragma: no cover - loop covers all cases


def goal_error(
    target: Sequence[float],
    measured_by_name: dict,
    joint_names: Sequence[str],
) -> Optional[float]:
    """Max |target - measured| over the shim's joints; None if any missing."""
    worst = 0.0
    for name, want in zip(joint_names, target):
        if name not in measured_by_name:
            return None
        worst = max(worst, abs(float(want) - float(measured_by_name[name])))
    return worst
