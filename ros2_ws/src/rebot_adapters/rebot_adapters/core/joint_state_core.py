"""Pure merge/validation core for the reBot joint-state adapter.

Design doc sec. 9.1: publish exactly one canonical ``/joint_states`` stream
with joint order::

    joint1..joint6, gripper_joint1, gripper_joint2

Real profile: merge ``/rebotarm/joint_states`` (six arm joints, rad) with
``/rebotarm/gripper/state`` (one motor state, rad) converted through the
inverse calibration map to the master jaw coordinate ``q_jaw_m`` (metres);
publish equal positions for both jaw joints; stamp from the newest coherent
input sample.

Sim profile: rename/filter ``/rebot_sim/joint_states_raw`` to canonical
names, suppress duplicate mimic state, publish both jaw joints with equal q,
reject unknown or missing arm joints.

This module is pure Python and must never import rclpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from .gripper_core import CalibrationTable
from .limits import (
    ARM_JOINT_NAMES,
    CANONICAL_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    GRIPPER_MASTER_JOINT,
    GRIPPER_MIMIC_JOINT,
)


class JointStateError(ValueError):
    """Raised when an input sample violates the canonical contract."""


@dataclass(frozen=True)
class JointSample:
    """ROS-free mirror of sensor_msgs/JointState."""

    names: Tuple[str, ...]
    positions: Tuple[float, ...]
    velocities: Tuple[float, ...] = ()
    efforts: Tuple[float, ...] = ()
    stamp_sec: float = 0.0


@dataclass(frozen=True)
class MotorSample:
    """ROS-free mirror of rebotarm_msgs/JointMotorState (one motor)."""

    position_rad: float
    velocity_rad_s: float = 0.0
    torque_nm: float = 0.0
    stamp_sec: float = 0.0


@dataclass(frozen=True)
class MergedJointState:
    """Canonical /joint_states payload (canonical name order)."""

    names: Tuple[str, ...]
    positions: Tuple[float, ...]
    velocities: Tuple[float, ...]
    efforts: Tuple[float, ...]
    stamp_sec: float


def is_fresh(stamp_sec: float, now_sec: float, stale_timeout_sec: float) -> bool:
    """True when a sample is recent enough to republish (sec. 9.1)."""
    return (now_sec - stamp_sec) <= stale_timeout_sec


def _validate_sample_arrays(sample: JointSample, what: str) -> None:
    n = len(sample.names)
    if len(sample.positions) != n:
        raise JointStateError(
            f"{what}: {n} names but {len(sample.positions)} positions"
        )
    for label, arr in (("velocities", sample.velocities),
                      ("efforts", sample.efforts)):
        if len(arr) not in (0, n):
            raise JointStateError(
                f"{what}: {label} must be empty or length {n}, got {len(arr)}"
            )
    if not all(math.isfinite(p) for p in sample.positions):
        raise JointStateError(f"{what}: non-finite position")
    if len(set(sample.names)) != n:
        raise JointStateError(f"{what}: duplicate joint names")


def _index_by_name(sample: JointSample) -> Dict[str, int]:
    return {name: i for i, name in enumerate(sample.names)}


def _column(sample: JointSample, arr: Tuple[float, ...], i: int) -> float:
    return arr[i] if arr else 0.0


def merge_real(
    arm: JointSample,
    gripper: MotorSample,
    table: CalibrationTable,
    *,
    require_all_arm_joints: bool = True,
) -> MergedJointState:
    """Merge real arm + gripper-motor samples into the canonical state.

    * validates arm names and array lengths;
    * copies arm positions/velocities/efforts in canonical order;
    * converts motor angle to q_jaw_m through the inverse calibration map
      (jaw velocity via the local inverse-map slope; jaw effort is not
      published because motor torque is not a jaw force);
    * publishes equal positions for both jaw joints;
    * stamps from the newest input sample.
    """
    _validate_sample_arrays(arm, "arm sample")
    unknown = sorted(set(arm.names) - set(ARM_JOINT_NAMES))
    if unknown:
        raise JointStateError(f"arm sample: unknown joints {unknown}")
    missing = sorted(set(ARM_JOINT_NAMES) - set(arm.names))
    if missing and require_all_arm_joints:
        raise JointStateError(f"arm sample: missing joints {missing}")

    idx = _index_by_name(arm)
    arm_names = tuple(n for n in ARM_JOINT_NAMES if n in idx)

    if not math.isfinite(gripper.position_rad):
        raise JointStateError("gripper sample: non-finite motor position")
    q_jaw = table.motor_to_q(gripper.position_rad)
    jaw_vel = (
        gripper.velocity_rad_s * table.slope_q_per_motor(gripper.position_rad)
        if math.isfinite(gripper.velocity_rad_s)
        else 0.0
    )

    names = arm_names + GRIPPER_JOINT_NAMES
    positions = tuple(arm.positions[idx[n]] for n in arm_names) + (q_jaw, q_jaw)
    velocities = tuple(
        _column(arm, arm.velocities, idx[n]) for n in arm_names
    ) + (jaw_vel, jaw_vel)
    efforts = tuple(
        _column(arm, arm.efforts, idx[n]) for n in arm_names
    ) + (0.0, 0.0)

    return MergedJointState(
        names=names,
        positions=positions,
        velocities=velocities,
        efforts=efforts,
        stamp_sec=max(arm.stamp_sec, gripper.stamp_sec),
    )


def merge_sim(
    raw: JointSample,
    *,
    rename_map: Optional[Mapping[str, str]] = None,
) -> MergedJointState:
    """Filter/rename a raw sim articulation sample to the canonical state.

    * optional ``rename_map`` translates raw articulation names to
      canonical names before validation;
    * unknown joints are rejected;
    * all six arm joints are required;
    * the mimic jaw (gripper_joint2) state is suppressed: the master value
      is used for both jaws (equal q).
    """
    if rename_map:
        raw = JointSample(
            names=tuple(rename_map.get(n, n) for n in raw.names),
            positions=raw.positions,
            velocities=raw.velocities,
            efforts=raw.efforts,
            stamp_sec=raw.stamp_sec,
        )
    _validate_sample_arrays(raw, "sim sample")
    unknown = sorted(set(raw.names) - set(CANONICAL_JOINT_NAMES))
    if unknown:
        raise JointStateError(f"sim sample: unknown joints {unknown}")
    missing_arm = sorted(set(ARM_JOINT_NAMES) - set(raw.names))
    if missing_arm:
        raise JointStateError(f"sim sample: missing arm joints {missing_arm}")

    idx = _index_by_name(raw)
    if GRIPPER_MASTER_JOINT in idx:
        jaw_i = idx[GRIPPER_MASTER_JOINT]  # suppress duplicate mimic state
    elif GRIPPER_MIMIC_JOINT in idx:
        jaw_i = idx[GRIPPER_MIMIC_JOINT]   # mimic equals master (x1.0)
    else:
        raise JointStateError("sim sample: no gripper joint state present")

    q_jaw = raw.positions[jaw_i]
    jaw_vel = _column(raw, raw.velocities, jaw_i)
    jaw_eff = _column(raw, raw.efforts, jaw_i)

    positions = tuple(
        raw.positions[idx[n]] for n in ARM_JOINT_NAMES
    ) + (q_jaw, q_jaw)
    velocities = tuple(
        _column(raw, raw.velocities, idx[n]) for n in ARM_JOINT_NAMES
    ) + (jaw_vel, jaw_vel)
    efforts = tuple(
        _column(raw, raw.efforts, idx[n]) for n in ARM_JOINT_NAMES
    ) + (jaw_eff, jaw_eff)

    return MergedJointState(
        names=CANONICAL_JOINT_NAMES,
        positions=positions,
        velocities=velocities,
        efforts=efforts,
        stamp_sec=raw.stamp_sec,
    )
