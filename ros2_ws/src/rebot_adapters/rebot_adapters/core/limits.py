"""Canonical reBot B601-DM joint names and limits.

Single source of truth for the adapter layer, following
``rebot_b601dm_isaac_ros_implementation_design_v2.md``:

* canonical joint order (sec. 9.1);
* corrected canonical-URDF velocity limits 5/3 rad/s (sec. 3.5, 20.1);
* arm position limits taken from the upstream B601-DM with-gripper URDF
  (the canonical URDF corrects only the velocity limits, positions are
  preserved);
* gripper master-joint (``q_jaw_m``) ranges (sec. 4.2-4.5).

This module is pure Python and must never import rclpy.
"""

from __future__ import annotations

from typing import Dict, Tuple

# --- Canonical joint names (design doc sec. 9.1) -------------------------

ARM_JOINT_NAMES: Tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)

GRIPPER_MASTER_JOINT: str = "gripper_joint1"
GRIPPER_MIMIC_JOINT: str = "gripper_joint2"

GRIPPER_JOINT_NAMES: Tuple[str, ...] = (GRIPPER_MASTER_JOINT, GRIPPER_MIMIC_JOINT)

CANONICAL_JOINT_NAMES: Tuple[str, ...] = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

# --- Arm limits (canonical URDF) -----------------------------------------

#: rad; (lower, upper) from the upstream B601-DM with-gripper URDF.
ARM_POSITION_LIMITS: Dict[str, Tuple[float, float]] = {
    "joint1": (-2.8, 2.8),
    "joint2": (-3.14, 0.0),
    "joint3": (-3.14, 0.0),
    "joint4": (-1.87, 1.57),
    "joint5": (-1.57, 1.57),
    "joint6": (-3.14, 3.14),
}

#: rad/s; corrected canonical limits (SDK POS_VEL.vlim), design doc sec. 3.5.
ARM_VELOCITY_LIMITS: Dict[str, float] = {
    "joint1": 5.0,
    "joint2": 5.0,
    "joint3": 5.0,
    "joint4": 3.0,
    "joint5": 3.0,
    "joint6": 3.0,
}

# --- Gripper master-joint coordinate (metres) ----------------------------

#: URDF/SRDF geometric range of the master prismatic joint (sec. 4.3).
Q_JAW_URDF_MIN_M: float = 0.0
Q_JAW_URDF_MAX_M: float = 0.0715

#: Upstream DM-demo operational baseline (sec. 4.5); replace after measurement.
Q_JAW_DEMO_MAX_M: float = 0.045

#: Default goal tolerance on q_jaw_m (sec. 4.5 calibration map template).
DEFAULT_GOAL_TOLERANCE_Q_M: float = 0.0015
