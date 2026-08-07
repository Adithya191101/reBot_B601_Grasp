"""Unit tests for rebot_adapters.core.trajectory_core.

Covers the design-doc sec. 9.2 validation gates and the sec. 20.3
trajectory-adapter test list (static portion).
"""

import math

import pytest

from rebot_adapters.core.limits import ARM_JOINT_NAMES
from rebot_adapters.core.trajectory_core import (
    INVALID_GOAL,
    INVALID_JOINTS,
    SUCCESSFUL,
    SingleGoalGate,
    TrajectoryPoint,
    ValidationResult,
    trajectory_duration,
    validate_trajectory,
)

NAMES = list(ARM_JOINT_NAMES)


def pt(t, positions, velocities=(), accelerations=()):
    return TrajectoryPoint(
        time_from_start=t,
        positions=tuple(positions),
        velocities=tuple(velocities),
        accelerations=tuple(accelerations),
    )


def valid_points():
    # All positions inside canonical limits; slow motion (well under 5/3 rad/s).
    return [
        pt(0.5, [0.0, -0.5, -0.5, 0.0, 0.0, 0.0]),
        pt(1.5, [0.2, -0.7, -0.6, 0.1, 0.1, 0.2]),
        pt(2.5, [0.4, -0.9, -0.7, 0.2, 0.2, 0.4]),
    ]


class TestJointNameGates:
    def test_accepts_valid_six_joint_trajectory(self):
        res = validate_trajectory(NAMES, valid_points())
        assert res.ok
        assert res.error_code == SUCCESSFUL
        assert res.joint_names == ARM_JOINT_NAMES
        assert len(res.points) == 3

    def test_rejects_missing_joint(self):
        res = validate_trajectory(NAMES[:5], [pt(1.0, [0.0] * 5)])
        assert not res.ok
        assert res.error_code == INVALID_JOINTS
        assert "missing" in res.reason

    def test_rejects_duplicate_joint(self):
        names = NAMES[:5] + ["joint1"]
        res = validate_trajectory(names, [pt(1.0, [0.0] * 6)])
        assert not res.ok
        assert res.error_code == INVALID_JOINTS
        assert "duplicate" in res.reason

    def test_rejects_unknown_joint(self):
        names = NAMES[:5] + ["panda_joint7"]
        res = validate_trajectory(names, [pt(1.0, [0.0] * 6)])
        assert not res.ok
        assert res.error_code == INVALID_JOINTS
        assert "unknown" in res.reason

    def test_rejects_out_of_order_by_default(self):
        names = list(reversed(NAMES))
        res = validate_trajectory(names, [pt(1.0, [0.0] * 6)])
        assert not res.ok
        assert res.error_code == INVALID_JOINTS
        assert "order" in res.reason

    def test_reorder_when_configured(self):
        names = ["joint2", "joint1", "joint3", "joint4", "joint5", "joint6"]
        # joint2 column first: value -1.0 belongs to joint2, 0.5 to joint1.
        points = [pt(1.0, [-1.0, 0.5, -0.5, 0.0, 0.0, 0.0])]
        res = validate_trajectory(names, points, reorder_joint_names=True)
        assert res.ok
        assert res.joint_names == ARM_JOINT_NAMES
        assert res.points[0].positions[0] == 0.5    # joint1
        assert res.points[0].positions[1] == -1.0   # joint2

    def test_reorder_also_permutes_velocities(self):
        names = ["joint6", "joint1", "joint2", "joint3", "joint4", "joint5"]
        points = [
            pt(
                1.0,
                [0.9, 0.1, -0.2, -0.3, 0.4, 0.5],
                velocities=[0.6, 0.1, 0.2, 0.3, 0.4, 0.5],
            )
        ]
        res = validate_trajectory(names, points, reorder_joint_names=True)
        assert res.ok
        assert res.points[0].positions == (0.1, -0.2, -0.3, 0.4, 0.5, 0.9)
        assert res.points[0].velocities == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


class TestPointGates:
    def test_rejects_empty_trajectory(self):
        res = validate_trajectory(NAMES, [])
        assert not res.ok
        assert res.error_code == INVALID_GOAL

    def test_rejects_wrong_position_count(self):
        res = validate_trajectory(NAMES, [pt(1.0, [0.0] * 5)])
        assert not res.ok
        assert res.error_code == INVALID_GOAL
        assert "positions" in res.reason

    def test_rejects_nonfinite_position(self):
        res = validate_trajectory(
            NAMES, [pt(1.0, [0.0, -0.5, float("nan"), 0.0, 0.0, 0.0])]
        )
        assert not res.ok
        assert res.error_code == INVALID_GOAL
        assert "non-finite" in res.reason

    def test_rejects_infinite_position(self):
        res = validate_trajectory(
            NAMES, [pt(1.0, [math.inf, -0.5, -0.5, 0.0, 0.0, 0.0])]
        )
        assert not res.ok

    def test_rejects_bad_velocity_array_length(self):
        res = validate_trajectory(
            NAMES,
            [pt(1.0, [0.0, -0.5, -0.5, 0.0, 0.0, 0.0], velocities=[0.1] * 3)],
        )
        assert not res.ok
        assert res.error_code == INVALID_GOAL
        assert "velocities" in res.reason

    def test_accepts_empty_velocity_and_acceleration_arrays(self):
        res = validate_trajectory(NAMES, valid_points())
        assert res.ok

    def test_accepts_length_six_optional_arrays(self):
        points = [
            pt(
                1.0,
                [0.0, -0.5, -0.5, 0.0, 0.0, 0.0],
                velocities=[0.1] * 6,
                accelerations=[0.1] * 6,
            )
        ]
        assert validate_trajectory(NAMES, points).ok

    def test_rejects_nonfinite_velocity_entry(self):
        points = [
            pt(
                1.0,
                [0.0, -0.5, -0.5, 0.0, 0.0, 0.0],
                velocities=[0.1, 0.1, math.nan, 0.1, 0.1, 0.1],
            )
        ]
        assert not validate_trajectory(NAMES, points).ok

    def test_rejects_bad_acceleration_array_length(self):
        points = [
            pt(1.0, [0.0, -0.5, -0.5, 0.0, 0.0, 0.0], accelerations=[0.1] * 2)
        ]
        res = validate_trajectory(NAMES, points)
        assert not res.ok
        assert "accelerations" in res.reason


class TestTimingGates:
    def test_rejects_nonmonotonic_timestamps(self):
        points = [
            pt(1.0, [0.0, -0.5, -0.5, 0.0, 0.0, 0.0]),
            pt(0.5, [0.1, -0.5, -0.5, 0.0, 0.0, 0.0]),
        ]
        res = validate_trajectory(NAMES, points)
        assert not res.ok
        assert res.error_code == INVALID_GOAL
        assert "strictly greater" in res.reason

    def test_rejects_equal_timestamps(self):
        points = [
            pt(1.0, [0.0, -0.5, -0.5, 0.0, 0.0, 0.0]),
            pt(1.0, [0.1, -0.5, -0.5, 0.0, 0.0, 0.0]),
        ]
        assert not validate_trajectory(NAMES, points).ok

    def test_rejects_negative_time(self):
        res = validate_trajectory(
            NAMES, [pt(-0.1, [0.0, -0.5, -0.5, 0.0, 0.0, 0.0])]
        )
        assert not res.ok
        assert "time_from_start" in res.reason


class TestLimitGates:
    def test_rejects_position_above_urdf_bound(self):
        # joint2 upper bound is 0.0 rad in the canonical URDF.
        res = validate_trajectory(
            NAMES, [pt(1.0, [0.0, 0.5, -0.5, 0.0, 0.0, 0.0])]
        )
        assert not res.ok
        assert res.error_code == INVALID_GOAL
        assert "joint2" in res.reason

    def test_rejects_position_below_urdf_bound(self):
        # joint1 lower bound is -2.8 rad.
        res = validate_trajectory(
            NAMES, [pt(1.0, [-3.0, -0.5, -0.5, 0.0, 0.0, 0.0])]
        )
        assert not res.ok
        assert "joint1" in res.reason

    def test_accepts_positions_exactly_on_bounds(self):
        res = validate_trajectory(
            NAMES, [pt(1.0, [2.8, 0.0, 0.0, 1.57, -1.57, 3.14])]
        )
        assert res.ok

    def test_rejects_segment_velocity_above_limit_joint1(self):
        # joint1 limit 5 rad/s: 1.2 rad in 0.2 s = 6 rad/s.
        points = [
            pt(0.1, [0.0, -0.5, -0.5, 0.0, 0.0, 0.0]),
            pt(0.3, [1.2, -0.5, -0.5, 0.0, 0.0, 0.0]),
        ]
        res = validate_trajectory(NAMES, points)
        assert not res.ok
        assert res.error_code == INVALID_GOAL
        assert "joint1" in res.reason and "velocity" in res.reason

    def test_rejects_segment_velocity_above_limit_joint4(self):
        # joint4 limit is 3 rad/s (wrist group): 0.8 rad in 0.2 s = 4 rad/s.
        points = [
            pt(0.1, [0.0, -0.5, -0.5, 0.0, 0.0, 0.0]),
            pt(0.3, [0.0, -0.5, -0.5, 0.8, 0.0, 0.0]),
        ]
        res = validate_trajectory(NAMES, points)
        assert not res.ok
        assert "joint4" in res.reason

    def test_accepts_velocity_between_wrist_and_base_limits_on_joint1(self):
        # 4 rad/s is legal for joint1 (limit 5) but not joint4 (limit 3).
        points = [
            pt(0.1, [0.0, -0.5, -0.5, 0.0, 0.0, 0.0]),
            pt(0.6, [2.0, -0.5, -0.5, 0.0, 0.0, 0.0]),
        ]
        assert validate_trajectory(NAMES, points).ok

    def test_velocity_gate_can_be_disabled(self):
        points = [
            pt(0.1, [0.0, -0.5, -0.5, 0.0, 0.0, 0.0]),
            pt(0.3, [1.2, -0.5, -0.5, 0.0, 0.0, 0.0]),
        ]
        res = validate_trajectory(
            NAMES, points, validate_segment_velocity=False
        )
        assert res.ok


class TestHelpers:
    def test_trajectory_duration(self):
        res = validate_trajectory(NAMES, valid_points())
        assert trajectory_duration(res.points) == pytest.approx(2.5)
        assert trajectory_duration(()) == 0.0

    def test_result_is_truthy_only_on_ok(self):
        assert ValidationResult(ok=True)
        assert not ValidationResult(ok=False, error_code=INVALID_GOAL)


class TestSingleGoalGate:
    def test_second_acquire_fails_until_release(self):
        gate = SingleGoalGate()
        assert gate.try_acquire("goal-a")
        assert not gate.try_acquire("goal-b")
        assert gate.active_goal == "goal-a"
        assert gate.release("goal-a")
        assert gate.try_acquire("goal-b")

    def test_release_by_non_owner_is_ignored(self):
        gate = SingleGoalGate()
        assert gate.try_acquire("goal-a")
        assert not gate.release("goal-b")
        assert gate.active_goal == "goal-a"

    def test_none_goal_id_rejected(self):
        gate = SingleGoalGate()
        with pytest.raises(ValueError):
            gate.try_acquire(None)
