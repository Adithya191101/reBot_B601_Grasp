"""Unit tests for rebot_adapters.core.gripper_core.

Covers the design-doc sec. 20.3 gripper-adapter test list (core portion):
interpolation, clamping, monotonic map validation, inverse mapping, sim
conversion, and the sec. 9.3 state machine.
"""

import math

import pytest

from rebot_adapters.core.gripper_core import (
    CalibrationError,
    CalibrationSample,
    CalibrationTable,
    GripperGoalState,
    GripperStateMachine,
    InvalidTransition,
    RangeError,
    plan_real_command,
    reached_goal,
    sim_jaw_positions,
)


def make_table(**kwargs):
    """Realistic table: motor 0.0 closed -> -5.0 open (design doc sec. 4.2)."""
    samples = [
        CalibrationSample(q_jaw_m=0.000, aperture_m=0.000, motor_rad=0.0),
        CalibrationSample(q_jaw_m=0.015, aperture_m=0.030, motor_rad=-1.8),
        CalibrationSample(q_jaw_m=0.030, aperture_m=0.060, motor_rad=-3.4),
        CalibrationSample(q_jaw_m=0.045, aperture_m=0.090, motor_rad=-5.0),
    ]
    defaults = dict(q_jaw_min_m=0.0, q_jaw_max_m=0.045)
    defaults.update(kwargs)
    return CalibrationTable(samples, **defaults)


class TestMonotonicMapValidation:
    def test_empty_table_refused(self):
        with pytest.raises(CalibrationError, match="empty"):
            CalibrationTable([])

    def test_single_sample_refused(self):
        with pytest.raises(CalibrationError, match="two samples"):
            CalibrationTable(
                [CalibrationSample(q_jaw_m=0.0, aperture_m=0.0, motor_rad=0.0)]
            )

    def test_nonmonotonic_q_refused(self):
        samples = [
            CalibrationSample(q_jaw_m=0.00, aperture_m=0.00, motor_rad=0.0),
            CalibrationSample(q_jaw_m=0.02, aperture_m=0.04, motor_rad=-2.0),
            CalibrationSample(q_jaw_m=0.02, aperture_m=0.05, motor_rad=-3.0),
        ]
        with pytest.raises(CalibrationError, match="q_jaw_m"):
            CalibrationTable(samples)

    def test_nonmonotonic_motor_refused(self):
        samples = [
            CalibrationSample(q_jaw_m=0.00, aperture_m=0.00, motor_rad=0.0),
            CalibrationSample(q_jaw_m=0.02, aperture_m=0.04, motor_rad=-2.0),
            CalibrationSample(q_jaw_m=0.04, aperture_m=0.08, motor_rad=-1.0),
        ]
        with pytest.raises(CalibrationError, match="motor_rad"):
            CalibrationTable(samples)

    def test_nonfinite_sample_refused(self):
        samples = [
            CalibrationSample(q_jaw_m=0.00, aperture_m=0.0, motor_rad=0.0),
            CalibrationSample(q_jaw_m=0.02, aperture_m=0.04, motor_rad=math.nan),
        ]
        with pytest.raises(CalibrationError, match="non-finite"):
            CalibrationTable(samples)

    def test_unsupported_interpolation_refused(self):
        samples = [
            CalibrationSample(q_jaw_m=0.00, aperture_m=0.0, motor_rad=0.0),
            CalibrationSample(q_jaw_m=0.02, aperture_m=0.04, motor_rad=-2.0),
        ]
        with pytest.raises(CalibrationError, match="interpolation"):
            CalibrationTable(samples, interpolation="cubic_spline")

    def test_valid_decreasing_motor_table_accepted(self):
        table = make_table()
        assert len(table.samples) == 4

    def test_from_dict_matches_doc_yaml_layout(self):
        data = {
            "version": 1,
            "operational_range": {"q_jaw_min_m": 0.0, "q_jaw_max_m": 0.045},
            "samples": [
                {"q_jaw_m": 0.0, "aperture_m": 0.0, "motor_rad": 0.0},
                {"q_jaw_m": 0.045, "aperture_m": 0.09, "motor_rad": -5.0},
            ],
            "interpolation": "monotonic_piecewise_linear",
            "clamp": True,
            "goal_tolerance_q_m": 0.0015,
        }
        table = CalibrationTable.from_dict(data)
        assert table.goal_tolerance_q_m == pytest.approx(0.0015)
        assert table.q_range == (0.0, 0.045)

    def test_from_dict_empty_samples_refused(self):
        data = {
            "samples": [],
            "interpolation": "monotonic_piecewise_linear",
        }
        with pytest.raises(CalibrationError):
            CalibrationTable.from_dict(data)


class TestInterpolation:
    def test_forward_map_on_sample_points(self):
        table = make_table()
        assert table.q_to_motor(0.000) == pytest.approx(0.0)
        assert table.q_to_motor(0.015) == pytest.approx(-1.8)
        assert table.q_to_motor(0.045) == pytest.approx(-5.0)

    def test_forward_map_between_samples(self):
        table = make_table()
        # midpoint of [0.0, 0.015] -> midpoint of [0.0, -1.8]
        assert table.q_to_motor(0.0075) == pytest.approx(-0.9)

    def test_inverse_map_on_sample_points(self):
        table = make_table()
        assert table.motor_to_q(0.0) == pytest.approx(0.000)
        assert table.motor_to_q(-3.4) == pytest.approx(0.030)
        assert table.motor_to_q(-5.0) == pytest.approx(0.045)

    def test_inverse_map_between_samples(self):
        table = make_table()
        assert table.motor_to_q(-0.9) == pytest.approx(0.0075)

    def test_roundtrip_opening_and_closing(self):
        table = make_table()
        for q in [0.0, 0.005, 0.0125, 0.0225, 0.033, 0.045]:
            assert table.motor_to_q(table.q_to_motor(q)) == pytest.approx(q)
        # closing sweep (descending commands) uses the same monotonic map
        for q in [0.045, 0.03, 0.01, 0.0]:
            assert table.motor_to_q(table.q_to_motor(q)) == pytest.approx(q)

    def test_slope_q_per_motor(self):
        table = make_table()
        # first segment: dq/dm = 0.015 / -1.8
        assert table.slope_q_per_motor(-0.5) == pytest.approx(0.015 / -1.8)


class TestClampBehavior:
    def test_command_above_range_clamps_to_max(self):
        table = make_table()
        assert table.clamp_q(0.10) == pytest.approx(0.045)
        assert table.q_to_motor(0.10) == pytest.approx(-5.0)

    def test_command_below_range_clamps_to_min(self):
        table = make_table()
        assert table.clamp_q(-0.01) == pytest.approx(0.0)
        assert table.q_to_motor(-0.01) == pytest.approx(0.0)

    def test_clamp_disabled_raises(self):
        table = make_table(clamp=False)
        with pytest.raises(RangeError):
            table.q_to_motor(0.10)

    def test_nonfinite_command_always_raises(self):
        table = make_table()
        with pytest.raises(RangeError):
            table.q_to_motor(math.nan)

    def test_operational_range_narrower_than_samples(self):
        table = make_table(q_jaw_max_m=0.030)
        assert table.q_range == (0.0, 0.030)
        assert table.q_to_motor(0.045) == pytest.approx(-3.4)  # clamped

    def test_inverse_map_clamps_out_of_table_motor(self):
        table = make_table()
        assert table.motor_to_q(-7.0) == pytest.approx(0.045)
        assert table.motor_to_q(1.0) == pytest.approx(0.0)


class TestRealConversion:
    def test_plan_real_command(self):
        table = make_table()
        cmd = plan_real_command(0.0075, table)
        assert cmd.motor_rad == pytest.approx(-0.9)
        assert cmd.q_jaw_m_clamped == pytest.approx(0.0075)

    def test_plan_real_command_clamps_first(self):
        table = make_table()
        cmd = plan_real_command(1.0, table)
        assert cmd.q_jaw_m_clamped == pytest.approx(0.045)
        assert cmd.motor_rad == pytest.approx(-5.0)


class TestSimConversion:
    def test_equal_jaw_positions(self):
        assert sim_jaw_positions(0.02) == (0.02, 0.02)

    def test_clamps_to_operational_range(self):
        q0, q1 = sim_jaw_positions(0.10)
        assert q0 == q1 == pytest.approx(0.045)

    def test_clamps_negative_to_zero(self):
        assert sim_jaw_positions(-0.5) == (0.0, 0.0)

    def test_respects_urdf_bound_even_with_wide_operational_range(self):
        q0, q1 = sim_jaw_positions(0.2, q_max_m=1.0)
        assert q0 == q1 == pytest.approx(0.0715)

    def test_clamp_disabled_raises(self):
        with pytest.raises(RangeError):
            sim_jaw_positions(0.10, clamp=False)

    def test_nonfinite_raises(self):
        with pytest.raises(RangeError):
            sim_jaw_positions(math.inf)


class TestReachedGoal:
    def test_within_tolerance(self):
        assert reached_goal(0.0301, 0.030, tolerance_m=0.0015)

    def test_outside_tolerance(self):
        assert not reached_goal(0.034, 0.030, tolerance_m=0.0015)


class TestStateMachine:
    def test_nominal_success_path(self):
        m = GripperStateMachine()
        assert m.state is GripperGoalState.IDLE
        m.to(GripperGoalState.CONVERTING)
        m.to(GripperGoalState.FORWARDED)
        m.to(GripperGoalState.TRACKING)
        m.to(GripperGoalState.SUCCEEDED)
        assert m.is_terminal
        m.reset()
        assert m.state is GripperGoalState.IDLE

    @pytest.mark.parametrize(
        "start",
        [
            GripperGoalState.CONVERTING,
            GripperGoalState.FORWARDED,
            GripperGoalState.TRACKING,
        ],
    )
    @pytest.mark.parametrize(
        "terminal", [GripperGoalState.ABORTED, GripperGoalState.CANCELED]
    )
    def test_abort_and_cancel_from_any_active_state(self, start, terminal):
        m = GripperStateMachine()
        path = {
            GripperGoalState.CONVERTING: [GripperGoalState.CONVERTING],
            GripperGoalState.FORWARDED: [
                GripperGoalState.CONVERTING,
                GripperGoalState.FORWARDED,
            ],
            GripperGoalState.TRACKING: [
                GripperGoalState.CONVERTING,
                GripperGoalState.FORWARDED,
                GripperGoalState.TRACKING,
            ],
        }[start]
        for s in path:
            m.to(s)
        m.to(terminal)
        assert m.is_terminal

    def test_success_only_from_tracking(self):
        m = GripperStateMachine()
        m.to(GripperGoalState.CONVERTING)
        with pytest.raises(InvalidTransition):
            m.to(GripperGoalState.SUCCEEDED)

    def test_cannot_skip_converting(self):
        m = GripperStateMachine()
        with pytest.raises(InvalidTransition):
            m.to(GripperGoalState.FORWARDED)

    def test_terminal_states_only_reset(self):
        m = GripperStateMachine()
        m.to(GripperGoalState.CONVERTING)
        m.to(GripperGoalState.ABORTED)
        with pytest.raises(InvalidTransition):
            m.to(GripperGoalState.CONVERTING)
        m.reset()
        m.to(GripperGoalState.CONVERTING)  # reusable after reset

    def test_is_active_flags(self):
        m = GripperStateMachine()
        assert not m.is_active
        m.to(GripperGoalState.CONVERTING)
        assert m.is_active
        m.to(GripperGoalState.CANCELED)
        assert not m.is_active
