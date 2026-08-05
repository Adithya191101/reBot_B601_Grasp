"""Shared topic names and frame-validation rules.

Publisher and subscriber import the **same** defaults from here. Previously the
publisher advertised ``~/rgb`` (node-private) while the subscriber listened on
``rgb`` (relative) -- two different resolved names, so the graph would have come
up looking healthy and delivered nothing. That class of bug is invisible until
something is meant to arrive, which is exactly when it costs the most.

No ROS imports in this module, so the validation rules are unit-testable without
ROS 2 installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Millimetre depth is the classic 1000x error. Only metres are accepted, and an
#: unexpected encoding is refused rather than guessed at.
DEPTH_ENCODING = "32FC1"
RGB_ENCODING = "rgb8"


@dataclass(frozen=True)
class TopicNames:
    """One source of truth for both ends of the graph."""

    rgb: str = "/dataset/rgb"
    depth: str = "/dataset/depth"
    camera_info: str = "/dataset/camera_info"
    grasp_pose: str = "/grasp/pose"
    clock: str = "/clock"

    @classmethod
    def with_namespace(cls, namespace: str) -> "TopicNames":
        ns = namespace.rstrip("/")
        if not ns:
            return cls()
        return cls(
            rgb=f"{ns}/rgb",
            depth=f"{ns}/depth",
            camera_info=f"{ns}/camera_info",
            grasp_pose=f"{ns}/grasp_pose",
            clock="/clock",          # /clock is global by convention, never namespaced
        )


def add_topic_arguments(parser) -> None:
    """Identical CLI surface on both nodes, so they cannot drift apart."""
    defaults = TopicNames()
    parser.add_argument("--namespace", default="",
                        help="apply one namespace to rgb/depth/camera_info/grasp_pose")
    parser.add_argument("--rgb-topic", default=None)
    parser.add_argument("--depth-topic", default=None)
    parser.add_argument("--camera-info-topic", default=None)
    parser.add_argument("--grasp-pose-topic", default=None)
    parser.set_defaults(_topic_defaults=defaults)


def resolve_topics(args) -> TopicNames:
    base = TopicNames.with_namespace(getattr(args, "namespace", "") or "")
    return TopicNames(
        rgb=getattr(args, "rgb_topic", None) or base.rgb,
        depth=getattr(args, "depth_topic", None) or base.depth,
        camera_info=getattr(args, "camera_info_topic", None) or base.camera_info,
        grasp_pose=getattr(args, "grasp_pose_topic", None) or base.grasp_pose,
        clock=base.clock,
    )


def validate_triplet(
    rgb_height: int, rgb_width: int, rgb_step: int, rgb_encoding: str,
    rgb_is_bigendian: int, rgb_frame_id: str, rgb_stamp_ns: int,
    depth_height: int, depth_width: int, depth_step: int, depth_encoding: str,
    depth_is_bigendian: int, depth_frame_id: str, depth_stamp_ns: int,
    info_height: int, info_width: int, info_frame_id: str, info_stamp_ns: int,
) -> Optional[str]:
    """Return the first inconsistency in an rgb/depth/camera_info triplet, or None.

    ``ApproximateTimeSynchronizer`` will happily hand over a triplet whose stamps
    differ by up to its slop, and nothing checks that the three describe the same
    camera at the same instant. Each rule below has a failure mode that otherwise
    surfaces as a wrong grasp rather than an error.
    """
    if rgb_encoding != RGB_ENCODING:
        return f"rgb encoding {rgb_encoding!r}, expected {RGB_ENCODING!r}"
    if depth_encoding != DEPTH_ENCODING:
        return (f"depth encoding {depth_encoding!r}, expected {DEPTH_ENCODING!r} "
                f"(metres); refusing to guess units")
    if rgb_is_bigendian or depth_is_bigendian:
        return "big-endian image data is not supported by this decoder"

    if not (rgb_stamp_ns == depth_stamp_ns == info_stamp_ns):
        return (f"stamps differ: rgb={rgb_stamp_ns} depth={depth_stamp_ns} "
                f"info={info_stamp_ns}; exact-stamp agreement is required")
    if not (rgb_frame_id == depth_frame_id == info_frame_id):
        return (f"frame_ids differ: rgb={rgb_frame_id!r} depth={depth_frame_id!r} "
                f"info={info_frame_id!r}")
    if not rgb_frame_id:
        return "empty frame_id"

    if (rgb_height, rgb_width) != (depth_height, depth_width):
        return (f"rgb {rgb_height}x{rgb_width} != depth {depth_height}x{depth_width}")
    if (rgb_height, rgb_width) != (info_height, info_width):
        return (f"image {rgb_height}x{rgb_width} != camera_info "
                f"{info_height}x{info_width}; intrinsics would be misapplied")

    if rgb_step != rgb_width * 3:
        return f"rgb step {rgb_step}, expected {rgb_width * 3} (padded rows unsupported)"
    if depth_step != depth_width * 4:
        return f"depth step {depth_step}, expected {depth_width * 4}"
    return None
