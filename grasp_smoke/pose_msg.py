"""Grasp -> ``geometry_msgs/PoseStamped``, as a plain dict.

Kept ROS-free on purpose. The pure library must stay importable and testable
without ROS 2 installed (PLAN.md 5.2.7), so the message *content* -- frame,
stamp, position, orientation -- is built and tested here, and
``ros2_iface/grasp_node.py`` does nothing but copy these fields into a real
message. That way the part that can be tested today is tested today, and the
untestable part is a thin shell with no logic to get wrong.

Pose convention: position and orientation are expressed in the **base frame**,
stamped with the source image's timestamp -- not the wall clock at publish. A
grasp stamped "now" cannot be transformed correctly by any consumer.

Two orientation conventions are intentionally exposed as separate functions:

* :func:`grasp_to_pose_stamped` preserves the original vision basis
  ``[grip, open, approach]`` for backward-compatible offline analysis.
* :func:`grasp_to_b601_tcp_pose_stamped` maps that basis to the vendor B601 TCP
  convention before publishing a control target. Control code must use the
  named TCP function rather than silently reinterpreting the legacy quaternion.
"""

from __future__ import annotations

import numpy as np

from .geometry import quaternion_from_matrix, transform_direction, transform_point


def vision_grasp_basis_to_b601_tcp_rotation(vision_rotation: np.ndarray) -> np.ndarray:
    """Convert ``[grip, open, approach]`` into the vendor B601 TCP basis.

    The output columns are the B601 TCP axes expressed in the input frame:

    * TCP X = ``-approach`` (tool-forward, into the object)
    * TCP Y = ``open`` after projection orthogonal to TCP X
    * TCP Z = ``cross(TCP X, TCP Y)``

    The input axes need not be perfectly unit or orthogonal. They are validated
    and Gram--Schmidt orthonormalized explicitly. The vision grip axis fixes the
    otherwise grasp-equivalent sign of TCP Y/Z, matching the pinned vendor
    implementation for a parallel gripper.
    """
    vision_rotation = np.asarray(vision_rotation, dtype=np.float64)
    if vision_rotation.shape != (3, 3):
        raise ValueError(
            f"vision_rotation must be a 3x3 [grip, open, approach] basis, "
            f"got {vision_rotation.shape}"
        )
    if not np.all(np.isfinite(vision_rotation)):
        raise ValueError("vision_rotation contains non-finite values")

    grip = vision_rotation[:, 0]
    open_axis = vision_rotation[:, 1]
    approach = vision_rotation[:, 2]

    approach_norm = float(np.linalg.norm(approach))
    grip_norm = float(np.linalg.norm(grip))
    if approach_norm < 1e-8:
        raise ValueError("vision approach axis is degenerate")
    if grip_norm < 1e-8:
        raise ValueError("vision grip axis is degenerate")

    tcp_x = -approach / approach_norm
    tcp_y = open_axis - float(np.dot(open_axis, tcp_x)) * tcp_x
    tcp_y_norm = float(np.linalg.norm(tcp_y))
    if tcp_y_norm < 1e-8:
        raise ValueError("vision open axis is parallel to the approach axis")
    tcp_y /= tcp_y_norm

    tcp_z = np.cross(tcp_x, tcp_y)
    tcp_z_norm = float(np.linalg.norm(tcp_z))
    if tcp_z_norm < 1e-8:
        raise ValueError("B601 TCP right-handed completion is degenerate")
    tcp_z /= tcp_z_norm

    grip_unit = grip / grip_norm
    grip_alignment = float(np.dot(tcp_z, grip_unit))
    if abs(grip_alignment) < 1e-8:
        raise ValueError("vision grip axis is inconsistent with open/approach axes")
    if grip_alignment < 0.0:
        # A parallel gripper is symmetric under a 180-degree opening-axis flip.
        # Select the equivalent branch whose TCP Z agrees with the vision grip.
        tcp_y = -tcp_y
        tcp_z = -tcp_z

    tcp_rotation = np.column_stack([tcp_x, tcp_y, tcp_z])
    if not np.allclose(tcp_rotation.T @ tcp_rotation, np.eye(3), atol=1e-10):
        raise ValueError("B601 TCP basis failed orthonormalization")
    determinant = float(np.linalg.det(tcp_rotation))
    if not np.isclose(determinant, 1.0, atol=1e-10):
        raise ValueError(f"B601 TCP basis is not right-handed (det={determinant})")
    return tcp_rotation


def _pose_stamped_payload(
    estimate,
    frame,
    rotation_cam: np.ndarray,
    branch: str,
    orientation_convention: str,
) -> dict:
    """Build shared PoseStamped fields for an explicitly selected convention."""
    if estimate is None or not estimate.is_valid:
        raise ValueError("cannot build a PoseStamped from an invalid grasp")

    position_base = transform_point(frame.T_base_cam, estimate.position)
    rotation_base = frame.T_base_cam[:3, :3] @ np.asarray(rotation_cam, dtype=np.float64)
    quat = quaternion_from_matrix(rotation_base)

    return {
        "header": {
            "stamp": {
                "sec": int(frame.stamp_ns // 1_000_000_000),
                "nanosec": int(frame.stamp_ns % 1_000_000_000),
            },
            "frame_id": frame.base_frame_id,
        },
        "pose": {
            "position": {
                "x": float(position_base[0]),
                "y": float(position_base[1]),
                "z": float(position_base[2]),
            },
            "orientation": {
                "x": float(quat[0]), "y": float(quat[1]),
                "z": float(quat[2]), "w": float(quat[3]),
            },
        },
        "_meta": {
            "branch": branch,
            "scene_id": frame.scene_id,
            "orientation_convention": orientation_convention,
            "open_axis_base": [
                float(v) for v in transform_direction(frame.T_base_cam, estimate.open_axis)
            ],
            "jaw_width_m": float(estimate.jaw_width_m),
            "z_m": float(estimate.z_m),
        },
    }


def grasp_to_pose_stamped(estimate, frame, branch: str = "") -> dict:
    """Build a legacy vision-basis PoseStamped in the base frame.

    Backward-compatible semantics: the quaternion represents the vision basis
    ``[grip, open, approach]``. It is useful for analysis but is **not** a B601
    end-effector target. Use :func:`grasp_to_b601_tcp_pose_stamped` for control.
    """
    if estimate is None or not estimate.is_valid:
        raise ValueError("cannot build a PoseStamped from an invalid grasp")
    return _pose_stamped_payload(
        estimate,
        frame,
        estimate.rotation,
        branch,
        orientation_convention="vision_grasp",
    )


def grasp_to_b601_tcp_pose_stamped(estimate, frame, branch: str = "") -> dict:
    """Build a vendor-convention B601 TCP target PoseStamped in the base frame."""
    if estimate is None or not estimate.is_valid:
        raise ValueError("cannot build a PoseStamped from an invalid grasp")
    tcp_rotation_cam = vision_grasp_basis_to_b601_tcp_rotation(estimate.rotation)
    return _pose_stamped_payload(
        estimate,
        frame,
        tcp_rotation_cam,
        branch,
        orientation_convention="b601_tcp",
    )


def pose_stamped_position(msg: dict) -> np.ndarray:
    p = msg["pose"]["position"]
    return np.array([p["x"], p["y"], p["z"]], dtype=np.float64)


def pose_stamped_stamp_ns(msg: dict) -> int:
    s = msg["header"]["stamp"]
    return int(s["sec"]) * 1_000_000_000 + int(s["nanosec"])
