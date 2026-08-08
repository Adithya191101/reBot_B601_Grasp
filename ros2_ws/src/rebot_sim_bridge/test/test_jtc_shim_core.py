"""Unit tests for the rclpy-free sim-JTC shim core."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rebot_sim_bridge.core import jtc_shim_core as core  # noqa: E402

JOINTS = ("joint1", "joint2", "joint3")


def _point(t, positions):
    return core.ShimPoint(time_from_start=t, positions=tuple(positions))


class TestValidateGoal:
    def test_accepts_and_reorders_by_name(self):
        traj = core.validate_goal(
            JOINTS, ("joint3", "joint1", "joint2"),
            [_point(1.0, (30.0, 10.0, 20.0))])
        assert traj.joint_names == JOINTS
        assert traj.points[0].positions == (10.0, 20.0, 30.0)

    def test_rejects_wrong_joint_set(self):
        with pytest.raises(core.TrajectoryGoalError):
            core.validate_goal(JOINTS, ("joint1", "joint2", "jointX"),
                               [_point(1.0, (0.0, 0.0, 0.0))])

    def test_rejects_empty_trajectory(self):
        with pytest.raises(core.TrajectoryGoalError):
            core.validate_goal(JOINTS, JOINTS, [])

    def test_rejects_short_point(self):
        with pytest.raises(core.TrajectoryGoalError):
            core.validate_goal(JOINTS, JOINTS, [_point(1.0, (0.0, 0.0))])

    def test_rejects_non_finite(self):
        with pytest.raises(core.TrajectoryGoalError):
            core.validate_goal(JOINTS, JOINTS,
                               [_point(1.0, (0.0, math.nan, 0.0))])

    def test_rejects_non_increasing_times(self):
        with pytest.raises(core.TrajectoryGoalError):
            core.validate_goal(JOINTS, JOINTS, [
                _point(1.0, (0.0, 0.0, 0.0)),
                _point(1.0, (1.0, 1.0, 1.0)),
            ])

    def test_rejects_zero_first_time(self):
        with pytest.raises(core.TrajectoryGoalError):
            core.validate_goal(JOINTS, JOINTS, [_point(0.0, (0.0, 0.0, 0.0))])


class TestSample:
    def _traj(self):
        return core.validate_goal(JOINTS, JOINTS, [
            _point(1.0, (1.0, 2.0, 3.0)),
            _point(3.0, (3.0, 2.0, 1.0)),
        ])

    def test_ramps_from_measured_start(self):
        got = core.sample(self._traj(), (0.0, 0.0, 0.0), 0.5)
        assert got == pytest.approx((0.5, 1.0, 1.5))

    def test_interpolates_between_points(self):
        got = core.sample(self._traj(), (0.0, 0.0, 0.0), 2.0)
        assert got == pytest.approx((2.0, 2.0, 2.0))

    def test_holds_final_past_duration(self):
        got = core.sample(self._traj(), (0.0, 0.0, 0.0), 99.0)
        assert got == pytest.approx((3.0, 2.0, 1.0))

    def test_exact_waypoint_times(self):
        assert core.sample(self._traj(), (0.0, 0.0, 0.0), 1.0) == \
            pytest.approx((1.0, 2.0, 3.0))
        assert core.sample(self._traj(), (0.0, 0.0, 0.0), 3.0) == \
            pytest.approx((3.0, 2.0, 1.0))


class TestGoalError:
    def test_reports_worst_joint(self):
        err = core.goal_error(
            (1.0, 2.0, 3.0),
            {"joint1": 1.0, "joint2": 2.5, "joint3": 3.1}, JOINTS)
        assert err == pytest.approx(0.5)

    def test_missing_joint_returns_none(self):
        assert core.goal_error((1.0,), {}, ("joint1",)) is None
