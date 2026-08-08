"""Pinocchio kinematics core for the reBot B601-DM planner.

This module replaces MoveIt's IK with the solver PROVEN in the Isaac Sim
grasping track (``scripts/b601_banana_demo.py``): a step-clamped,
joint-limit-projected damped-least-squares solve on the ``gripper_link``
frame.  Local by construction -- each iterate moves at most ``step_clamp``
rad per joint, so the solution cannot hop to a flipped branch the way the
vendor CLIK's backtracking line-search was observed to (2.78 rad jumps on
3 cm waypoints).

Frames (KDR-001):

* canonical model: ``urdf/rebot_b601dm_canonical.urdf`` (vendor
  with_gripper URDF with velocity limits corrected to 5/3 rad/s);
* end-effector frame: ``gripper_link``;
* canonical TCP: ``gripper_tcp`` = ``gripper_link`` (+) translation
  [-0.041763, 0.000008, 0.003427] m (P3-calibrated jaw midpoint).

Joint-position AND velocity limits are read from the model, so the
canonical URDF stays the single source of truth.

This module is rclpy-free and must never import any ROS package.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin

#: canonical arm joints; the model additionally carries the two gripper
#: prismatic joints, which the planner pins at 0 (closed) for IK.
ARM_DOF = 6

DEFAULT_EE_FRAME = "gripper_link"

#: KDR-001 canonical TCP translation (gripper_link frame), metres.
DEFAULT_TCP_OFFSET_M: Tuple[float, float, float] = (-0.041763, 0.000008, 0.003427)

#: Acceptance threshold on the residual SE3 log-metric.  The solver often
#: converges to ~1e-3 in the combined metric and stalls -- a millimetre-scale
#: residual, not an unreachable pose (banana-demo lesson).
IK_ERR_ACCEPT = 5.0e-3

#: Margin used when CHECKING a solution against the limits (rad).
LIMIT_CHECK_MARGIN_RAD = 1.0e-3

#: Margin used when PROJECTING DLS iterates inside the limits (rad).
LIMIT_PROJECT_MARGIN_RAD = 2.0e-3


def quat_xyzw_to_rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    """3x3 rotation matrix from a (x, y, z, w) unit quaternion (ROS order)."""
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    # Row 3 is [2(xz - yw), 2(yz + xw), 1 - 2(x^2 + y^2)].  An earlier
    # version had its first two elements swapped -- invisible for identity
    # and yaw-only quaternions (x = y = 0 zeroes both terms, which is all
    # the unit tests exercised) but wrong by tens of degrees for general
    # orientations; caught by the M5 parity test's ready-pose goal.
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_to_transform(position: Sequence[float],
                      quat_xyzw: Sequence[float]) -> np.ndarray:
    """4x4 homogeneous transform from position + (x, y, z, w) quaternion."""
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_rotation(*quat_xyzw)
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    return T


class KinematicsCore:
    """Canonical-URDF FK/IK, parameterized by model path + TCP transform.

    ``tcp_offset_m`` is the translation of the canonical TCP in the
    end-effector frame; a full 4x4 ``tcp_transform`` may be given instead
    (it wins when both are supplied).
    """

    def __init__(
        self,
        urdf_path: str,
        *,
        ee_frame: str = DEFAULT_EE_FRAME,
        tcp_offset_m: Sequence[float] = DEFAULT_TCP_OFFSET_M,
        tcp_transform: Optional[np.ndarray] = None,
    ) -> None:
        self.urdf_path = str(urdf_path)
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()
        if not self.model.existFrame(ee_frame):
            raise ValueError(f"frame {ee_frame!r} not in {urdf_path}")
        self.ee_frame = ee_frame
        self.ee_frame_id = self.model.getFrameId(ee_frame)

        if tcp_transform is not None:
            T = np.asarray(tcp_transform, dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError("tcp_transform must be 4x4")
            self.T_ee_tcp = T.copy()
        else:
            self.T_ee_tcp = np.eye(4)
            self.T_ee_tcp[:3, 3] = np.asarray(tcp_offset_m, dtype=np.float64)
        self._T_tcp_ee = np.linalg.inv(self.T_ee_tcp)

        # Arm limits straight from the canonical model.
        self.lower = np.asarray(self.model.lowerPositionLimit[:ARM_DOF])
        self.upper = np.asarray(self.model.upperPositionLimit[:ARM_DOF])
        self.velocity_limits = np.asarray(self.model.velocityLimit[:ARM_DOF])

    # -- configuration helpers -------------------------------------------

    def full_q(self, q6: Sequence[float]) -> np.ndarray:
        """Model-sized configuration with gripper joints pinned at 0."""
        q = np.zeros(self.model.nq)
        q[:ARM_DOF] = np.asarray(q6, dtype=np.float64)[:ARM_DOF]
        return q

    def within_limits(self, q6: Sequence[float],
                      margin: float = LIMIT_CHECK_MARGIN_RAD) -> bool:
        q6 = np.asarray(q6, dtype=np.float64)[:ARM_DOF]
        return bool(np.all(q6 >= self.lower + margin)
                    and np.all(q6 <= self.upper - margin))

    def limit_violations(self, q6: Sequence[float],
                         margin: float = LIMIT_CHECK_MARGIN_RAD):
        """[(joint_index_1based, value), ...] outside the margined limits."""
        q6 = np.asarray(q6, dtype=np.float64)[:ARM_DOF]
        return [
            (i + 1, round(float(q6[i]), 3))
            for i in range(ARM_DOF)
            if not (self.lower[i] + margin <= q6[i] <= self.upper[i] - margin)
        ]

    # -- forward kinematics ----------------------------------------------

    def fk_ee(self, q6: Sequence[float]) -> np.ndarray:
        """4x4 base_link -> ee_frame transform."""
        q = self.full_q(q6)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return np.asarray(self.data.oMf[self.ee_frame_id].homogeneous).copy()

    def fk_tcp(self, q6: Sequence[float]) -> np.ndarray:
        """4x4 base_link -> gripper_tcp transform."""
        return self.fk_ee(q6) @ self.T_ee_tcp

    def ee_target_from_tcp(self, T_base_tcp: np.ndarray) -> np.ndarray:
        """Convert a TCP goal into the ee-frame goal the solver works on."""
        return np.asarray(T_base_tcp, dtype=np.float64) @ self._T_tcp_ee

    # -- inverse kinematics ----------------------------------------------

    def ik_local_dls(
        self,
        T_ee_target: np.ndarray,
        q0: Sequence[float],
        *,
        iters: int = 120,
        damping: float = 1e-6,
        step_clamp: float = 0.12,
        tol: float = 1e-4,
    ) -> Tuple[np.ndarray, float]:
        """Step-clamped damped-least-squares IK on the ee frame.

        Ported unchanged from the proven ``b601_banana_demo.ik_local_dls``.
        Local by construction: each iterate moves at most ``step_clamp`` rad
        per joint, so the solution cannot hop to a flipped branch.  Projected
        DLS: iterates are clipped inside the joint limits so the solver finds
        the LEGAL wrist branch instead of walking joint6 past +pi.

        Returns ``(q6, residual)`` where ``residual`` is the SE3 log-metric
        norm; compare against :data:`IK_ERR_ACCEPT`.
        """
        model, data = self.model, self.data
        q = np.zeros(model.nq)
        q[:ARM_DOF] = np.asarray(q0, dtype=np.float64)[:ARM_DOF]
        T = np.asarray(T_ee_target, dtype=np.float64)
        target = pin.SE3(T[:3, :3].copy(), T[:3, 3].copy())
        err = np.full(6, np.inf)
        for _ in range(iters):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            cur = data.oMf[self.ee_frame_id]
            err = pin.log6(cur.actInv(target)).vector
            if float(np.linalg.norm(err)) < tol:
                return q[:ARM_DOF].copy(), float(np.linalg.norm(err))
            J = pin.computeFrameJacobian(model, data, q, self.ee_frame_id,
                                         pin.ReferenceFrame.LOCAL)
            JJt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            dq = np.clip(dq, -step_clamp, step_clamp)
            q = pin.integrate(model, q, dq)
            # Projected DLS: keep iterates inside joint limits so the solver
            # finds the LEGAL wrist branch instead of walking joint6 past +pi.
            q[:ARM_DOF] = np.clip(q[:ARM_DOF],
                                  self.lower + LIMIT_PROJECT_MARGIN_RAD,
                                  self.upper - LIMIT_PROJECT_MARGIN_RAD)
            q[ARM_DOF:] = 0.0
        return q[:ARM_DOF].copy(), float(np.linalg.norm(err))

    def solve_tcp(
        self,
        T_base_tcp: np.ndarray,
        q0: Sequence[float],
        **kwargs,
    ) -> Tuple[np.ndarray, float]:
        """IK for a canonical-TCP goal: convert frame, then local DLS."""
        return self.ik_local_dls(self.ee_target_from_tcp(T_base_tcp), q0,
                                 **kwargs)
