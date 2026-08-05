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
"""

from __future__ import annotations

import numpy as np

from .geometry import quaternion_from_matrix, transform_direction, transform_point


def grasp_to_pose_stamped(estimate, frame, branch: str = "") -> dict:
    """Build the PoseStamped payload for a valid grasp, in the base frame."""
    if estimate is None or not estimate.is_valid:
        raise ValueError("cannot build a PoseStamped from an invalid grasp")

    position_base = transform_point(frame.T_base_cam, estimate.position)
    rotation_base = frame.T_base_cam[:3, :3] @ estimate.rotation
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
            "open_axis_base": [
                float(v) for v in transform_direction(frame.T_base_cam, estimate.open_axis)
            ],
            "jaw_width_m": float(estimate.jaw_width_m),
            "z_m": float(estimate.z_m),
        },
    }


def pose_stamped_position(msg: dict) -> np.ndarray:
    p = msg["pose"]["position"]
    return np.array([p["x"], p["y"], p["z"]], dtype=np.float64)


def pose_stamped_stamp_ns(msg: dict) -> int:
    s = msg["header"]["stamp"]
    return int(s["sec"]) * 1_000_000_000 + int(s["nanosec"])
