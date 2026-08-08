"""Unit tests for rebot_planner.core.path_core.

Waypoint continuity, branch/shoulder guards, and velocity-limit timing on
the canonical URDF.
"""

import numpy as np
import pytest

from rebot_planner.core import path_core
from rebot_planner.core.ik_core import ARM_DOF

from planner_testlib import make_tcp


def _max_wp_step(waypoints):
    wp = np.asarray(waypoints)
    return float(np.max(np.abs(np.diff(wp, axis=0))))


def _yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---- nominal plans -------------------------------------------------------


def test_descent_waypoint_continuity(kin, ready):
    """A straight 18 cm descent: continuous, in-limit waypoints reaching
    the goal, starting exactly at the start configuration."""
    T_goal = make_tcp(ready["R"], (0.30, 0.0, 0.12))
    plan = path_core.plan_linear(kin, ready["q"], T_goal)
    assert plan.ok, plan.reason
    assert np.allclose(plan.waypoints[0], ready["q"])
    assert len(plan.waypoints) >= path_core.DEFAULT_STEPS + 1
    # continuity: no per-joint step exceeds the branch-jump guard
    assert _max_wp_step(plan.waypoints) <= path_core.DEFAULT_BRANCH_JUMP_RAD
    for q in plan.waypoints:
        assert kin.within_limits(q)
    # the final waypoint reaches the goal TCP
    T_end = kin.fk_tcp(plan.waypoints[-1])
    assert np.linalg.norm(T_end[:3, 3] - T_goal[:3, 3]) < 2e-3


def test_rotation_stage_slerps_in_place(kin, ready):
    """45 deg world-yaw reorientation: slerped waypoints, position held."""
    T_goal = make_tcp(_yaw(np.pi / 4) @ ready["R"], (0.30, 0.0, 0.30))
    plan = path_core.plan_linear(kin, ready["q"], T_goal)
    assert plan.ok, plan.reason
    assert plan.n_rotation_waypoints >= 2  # 0.785 rad gap, 0.35 rad steps
    # rotation-stage waypoints keep the TCP position (rotate-then-translate)
    for q in plan.waypoints[1:1 + plan.n_rotation_waypoints]:
        T = kin.fk_tcp(q)
        assert np.linalg.norm(T[:3, 3] - [0.30, 0.0, 0.30]) < 2e-3
    # final orientation reached
    T_end = kin.fk_tcp(plan.waypoints[-1])
    R_err = T_end[:3, :3].T @ T_goal[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
    assert ang < 0.5


# ---- guards --------------------------------------------------------------


def test_branch_jump_guard_refuses(kin, ready):
    """The same feasible descent is REFUSED when the per-waypoint jump
    budget is tightened below its actual (measured ~0.107 rad) step."""
    T_goal = make_tcp(ready["R"], (0.30, 0.0, 0.12))
    plan = path_core.plan_linear(kin, ready["q"], T_goal,
                                 branch_jump_rad=0.05)
    assert not plan.ok
    assert plan.reason == "branch jump refused"
    assert plan.detail["jump_rad"] > 0.05


def test_shoulder_guard_blocks_reorientation(kin, ready):
    """The 45 deg yaw reorientation drags joint1 ~0.32 rad; a tightened
    shoulder budget must refuse it, the default must allow it."""
    T_goal = make_tcp(_yaw(np.pi / 4) @ ready["R"], (0.30, 0.0, 0.30))
    tight = path_core.plan_linear(kin, ready["q"], T_goal,
                                  shoulder_jump_rad=0.25)
    assert not tight.ok
    assert tight.reason == "reorientation would move shoulder joints"
    assert tight.detail["shoulder_jump_rad"] > 0.25
    ok = path_core.plan_linear(kin, ready["q"], T_goal)
    assert ok.ok, ok.reason


def test_goal_behind_arm_is_refused_not_swept(kin, ready):
    """A goal straight behind the base would need a shoulder-flipped
    branch; the planner must refuse instead of sweeping through (the
    demo's measured 4.000 rad tracking-error failure mode)."""
    T_goal = make_tcp(ready["R"], (-0.30, 0.0, 0.30))
    plan = path_core.plan_linear(kin, ready["q"], T_goal)
    assert not plan.ok
    assert plan.reason in (
        "local IK failed",
        "branch jump refused",
        "outside joint limits",
        "in-place reorientation unreachable",
        "reorientation would move shoulder joints",
    )


def test_start_outside_limits_rejected(kin, ready):
    q_bad = np.asarray(ready["q"]).copy()
    q_bad[1] = kin.lower[1] - 0.2
    plan = path_core.plan_linear(kin, q_bad, ready["T"])
    assert not plan.ok
    assert plan.reason == "start configuration outside joint limits"


# ---- timing --------------------------------------------------------------


def test_timing_respects_velocity_limits(kin, ready):
    T_goal = make_tcp(ready["R"], (0.30, 0.0, 0.12))
    plan = path_core.plan_linear(kin, ready["q"], T_goal)
    assert plan.ok
    for sf in (1.0, 0.5, 0.25):
        times = path_core.time_waypoints(plan.waypoints, kin.velocity_limits,
                                         safety_factor=sf)
        assert times[0] == 0.0
        assert all(b > a for a, b in zip(times[:-1], times[1:]))
        vmax = path_core.max_segment_velocity(plan.waypoints, times)
        assert np.all(vmax <= kin.velocity_limits * sf + 1e-9)
        # and therefore inside the canonical adapter gate (5/3 rad/s)
        assert np.all(vmax <= kin.velocity_limits)


def test_safety_factor_scales_duration(kin, ready):
    T_goal = make_tcp(ready["R"], (0.30, 0.0, 0.12))
    plan = path_core.plan_linear(kin, ready["q"], T_goal)
    assert plan.ok
    # tiny floor so the velocity constraint, not the floor, dominates
    t_half = path_core.time_waypoints(plan.waypoints, kin.velocity_limits,
                                      safety_factor=0.5,
                                      min_segment_sec=1e-4)[-1]
    t_quarter = path_core.time_waypoints(plan.waypoints, kin.velocity_limits,
                                         safety_factor=0.25,
                                         min_segment_sec=1e-4)[-1]
    assert t_quarter == pytest.approx(2.0 * t_half, rel=1e-6)


def test_min_segment_floor_and_validation(kin):
    wps = [np.zeros(ARM_DOF), np.zeros(ARM_DOF)]  # zero motion
    times = path_core.time_waypoints(wps, kin.velocity_limits,
                                     min_segment_sec=0.2)
    assert times == (0.0, 0.2)
    with pytest.raises(ValueError):
        path_core.time_waypoints(wps, kin.velocity_limits, safety_factor=0.0)
    with pytest.raises(ValueError):
        path_core.time_waypoints(wps, kin.velocity_limits, safety_factor=1.2)
    with pytest.raises(ValueError):
        path_core.time_waypoints(wps, kin.velocity_limits,
                                 min_segment_sec=0.0)
    with pytest.raises(ValueError):
        path_core.time_waypoints(wps, [5, 5, 5, 3, 3, 0.0])
