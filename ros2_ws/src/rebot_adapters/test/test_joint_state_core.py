"""Unit tests for rebot_adapters.core.joint_state_core.

Covers the design-doc sec. 20.3 joint-state-adapter test list (core
portion): exact canonical names/order, stale-input behavior, missing-joint
behavior, correct units (q_jaw_m in metres via the inverse map), and
coherent timestamps.
"""

import pytest

from rebot_adapters.core.gripper_core import CalibrationSample, CalibrationTable
from rebot_adapters.core.joint_state_core import (
    JointSample,
    JointStateError,
    MotorSample,
    is_fresh,
    merge_real,
    merge_sim,
)
from rebot_adapters.core.limits import ARM_JOINT_NAMES, CANONICAL_JOINT_NAMES

ARM = list(ARM_JOINT_NAMES)


@pytest.fixture()
def table():
    # motor 0.0 closed -> -5.0 open over q_jaw 0 .. 0.045 m (sec. 4.2)
    return CalibrationTable(
        [
            CalibrationSample(q_jaw_m=0.000, aperture_m=0.000, motor_rad=0.0),
            CalibrationSample(q_jaw_m=0.045, aperture_m=0.090, motor_rad=-5.0),
        ],
        q_jaw_min_m=0.0,
        q_jaw_max_m=0.045,
    )


def arm_sample(names=None, positions=None, **kwargs):
    names = tuple(names if names is not None else ARM)
    positions = tuple(
        positions if positions is not None else
        [0.1, -0.2, -0.3, 0.4, 0.5, 0.6][: len(names)]
    )
    defaults = dict(
        velocities=tuple(0.01 * i for i in range(len(names))),
        efforts=tuple(1.0 * i for i in range(len(names))),
        stamp_sec=100.0,
    )
    defaults.update(kwargs)
    return JointSample(names=names, positions=positions, **defaults)


class TestMergeReal:
    def test_canonical_name_order(self, table):
        merged = merge_real(arm_sample(), MotorSample(position_rad=-2.5), table)
        assert merged.names == CANONICAL_JOINT_NAMES

    def test_reorders_shuffled_arm_input(self, table):
        shuffled = ["joint3", "joint1", "joint2", "joint6", "joint4", "joint5"]
        positions = [-0.3, 0.1, -0.2, 0.6, 0.4, 0.5]
        merged = merge_real(
            arm_sample(names=shuffled, positions=positions,
                       velocities=(), efforts=()),
            MotorSample(position_rad=0.0),
            table,
        )
        assert merged.positions[:6] == (0.1, -0.2, -0.3, 0.4, 0.5, 0.6)

    def test_jaw_positions_equal_and_in_metres(self, table):
        # motor -2.5 rad is halfway: q_jaw = 0.0225 m through the inverse map
        merged = merge_real(arm_sample(), MotorSample(position_rad=-2.5), table)
        assert merged.positions[6] == pytest.approx(0.0225)
        assert merged.positions[7] == pytest.approx(0.0225)

    def test_jaw_velocity_from_inverse_slope(self, table):
        # slope dq/dm = 0.045 / -5.0 = -0.009 m/rad; motor vel -1 rad/s
        merged = merge_real(
            arm_sample(),
            MotorSample(position_rad=-2.5, velocity_rad_s=-1.0),
            table,
        )
        assert merged.velocities[6] == pytest.approx(0.009)
        assert merged.velocities[7] == pytest.approx(0.009)

    def test_stamp_from_newest_input(self, table):
        merged = merge_real(
            arm_sample(stamp_sec=100.0),
            MotorSample(position_rad=0.0, stamp_sec=100.5),
            table,
        )
        assert merged.stamp_sec == pytest.approx(100.5)
        merged = merge_real(
            arm_sample(stamp_sec=101.0),
            MotorSample(position_rad=0.0, stamp_sec=100.5),
            table,
        )
        assert merged.stamp_sec == pytest.approx(101.0)

    def test_unknown_arm_joint_rejected(self, table):
        with pytest.raises(JointStateError, match="unknown"):
            merge_real(
                arm_sample(names=ARM[:5] + ["panda_joint7"]),
                MotorSample(position_rad=0.0),
                table,
            )

    def test_missing_arm_joint_rejected_when_required(self, table):
        with pytest.raises(JointStateError, match="missing"):
            merge_real(
                arm_sample(names=ARM[:5], velocities=(), efforts=()),
                MotorSample(position_rad=0.0),
                table,
            )

    def test_missing_arm_joint_allowed_when_not_required(self, table):
        merged = merge_real(
            arm_sample(names=ARM[:5], velocities=(), efforts=()),
            MotorSample(position_rad=0.0),
            table,
            require_all_arm_joints=False,
        )
        # canonical-order subset of present arm joints + both jaws
        assert merged.names == tuple(ARM[:5]) + (
            "gripper_joint1",
            "gripper_joint2",
        )

    def test_duplicate_arm_names_rejected(self, table):
        with pytest.raises(JointStateError, match="duplicate"):
            merge_real(
                arm_sample(names=ARM[:5] + ["joint1"]),
                MotorSample(position_rad=0.0),
                table,
            )

    def test_array_length_mismatch_rejected(self, table):
        with pytest.raises(JointStateError, match="positions"):
            merge_real(
                arm_sample(positions=[0.1, 0.2, 0.3]),
                MotorSample(position_rad=0.0),
                table,
            )

    def test_bad_optional_array_length_rejected(self, table):
        with pytest.raises(JointStateError, match="velocities"):
            merge_real(
                arm_sample(velocities=(0.1, 0.2)),
                MotorSample(position_rad=0.0),
                table,
            )

    def test_nonfinite_position_rejected(self, table):
        with pytest.raises(JointStateError, match="non-finite"):
            merge_real(
                arm_sample(
                    positions=[0.1, float("nan"), -0.3, 0.4, 0.5, 0.6]
                ),
                MotorSample(position_rad=0.0),
                table,
            )

    def test_nonfinite_motor_position_rejected(self, table):
        with pytest.raises(JointStateError, match="motor"):
            merge_real(
                arm_sample(),
                MotorSample(position_rad=float("inf")),
                table,
            )


class TestMergeSim:
    def raw(self, names, positions, **kwargs):
        return arm_sample(names=names, positions=positions, **kwargs)

    def test_full_raw_state_with_both_jaws(self):
        names = ARM + ["gripper_joint1", "gripper_joint2"]
        positions = [0.1, -0.2, -0.3, 0.4, 0.5, 0.6, 0.03, 0.031]
        merged = merge_sim(
            self.raw(names, positions, velocities=(), efforts=())
        )
        assert merged.names == CANONICAL_JOINT_NAMES
        # mimic (gripper_joint2) state suppressed: master value used twice
        assert merged.positions[6] == pytest.approx(0.03)
        assert merged.positions[7] == pytest.approx(0.03)

    def test_only_master_jaw_present(self):
        names = ARM + ["gripper_joint1"]
        positions = [0.1, -0.2, -0.3, 0.4, 0.5, 0.6, 0.02]
        merged = merge_sim(
            self.raw(names, positions, velocities=(), efforts=())
        )
        assert merged.positions[6:] == (0.02, 0.02)

    def test_only_mimic_jaw_present_used_as_master(self):
        names = ARM + ["gripper_joint2"]
        positions = [0.1, -0.2, -0.3, 0.4, 0.5, 0.6, 0.02]
        merged = merge_sim(
            self.raw(names, positions, velocities=(), efforts=())
        )
        assert merged.positions[6:] == (0.02, 0.02)

    def test_no_jaw_state_rejected(self):
        with pytest.raises(JointStateError, match="gripper"):
            merge_sim(
                self.raw(
                    ARM,
                    [0.1, -0.2, -0.3, 0.4, 0.5, 0.6],
                    velocities=(),
                    efforts=(),
                )
            )

    def test_unknown_joint_rejected(self):
        names = ARM + ["gripper_joint1", "mystery_joint"]
        with pytest.raises(JointStateError, match="unknown"):
            merge_sim(
                self.raw(
                    names,
                    [0.1, -0.2, -0.3, 0.4, 0.5, 0.6, 0.02, 0.0],
                    velocities=(),
                    efforts=(),
                )
            )

    def test_missing_arm_joint_rejected(self):
        names = ARM[:5] + ["gripper_joint1"]
        with pytest.raises(JointStateError, match="missing arm"):
            merge_sim(
                self.raw(
                    names,
                    [0.1, -0.2, -0.3, 0.4, 0.5, 0.02],
                    velocities=(),
                    efforts=(),
                )
            )

    def test_rename_map_translates_raw_articulation_names(self):
        names = [f"rebot/{n}" for n in ARM] + ["rebot/left_jaw"]
        rename = {f"rebot/{n}": n for n in ARM}
        rename["rebot/left_jaw"] = "gripper_joint1"
        merged = merge_sim(
            self.raw(
                names,
                [0.1, -0.2, -0.3, 0.4, 0.5, 0.6, 0.02],
                velocities=(),
                efforts=(),
            ),
            rename_map=rename,
        )
        assert merged.names == CANONICAL_JOINT_NAMES
        assert merged.positions[6:] == (0.02, 0.02)

    def test_sim_stamp_passthrough(self):
        names = ARM + ["gripper_joint1"]
        merged = merge_sim(
            self.raw(
                names,
                [0.1, -0.2, -0.3, 0.4, 0.5, 0.6, 0.02],
                velocities=(),
                efforts=(),
                stamp_sec=42.5,
            )
        )
        assert merged.stamp_sec == pytest.approx(42.5)


class TestStaleness:
    def test_fresh_sample(self):
        assert is_fresh(stamp_sec=10.00, now_sec=10.15, stale_timeout_sec=0.20)

    def test_stale_sample(self):
        assert not is_fresh(
            stamp_sec=10.00, now_sec=10.25, stale_timeout_sec=0.20
        )

    def test_boundary_is_fresh(self):
        assert is_fresh(stamp_sec=10.0, now_sec=10.2, stale_timeout_sec=0.20)
