"""Collision checking for the reBot B601-DM planner.

Pinocchio's collision API against

* the canonical URDF's COLLISION geometry (``pin.buildGeomFromUrdf``) with
  the vendor SRDF's disabled self-collision pairs removed, and
* fixed world boxes loaded from ``config/cell_geometry.yaml`` (table slab
  + parametric gantry crossbar, design doc sec. 6.3).

A waypoint path is validated by checking interpolated configurations
(joint-space, resolution-bounded) with ``pin.computeCollisions``.

**Scope limit (by design):** the planner only REJECTS colliding paths; it
does not plan around obstacles.  Obstacle-avoiding planning arrives later
via cuMotion (design doc M6/M7) -- until then a rejected goal must be
re-posed by the caller.

This module is rclpy-free and must never import any ROS package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin

try:  # pinocchio >= 3 bundles coal; 2.x uses hppfcl
    import hppfcl as _fcl
except ImportError:  # pragma: no cover - depends on installed stack
    import coal as _fcl

from .ik_core import ARM_DOF, KinematicsCore

#: Self-collision pairs disabled by the vendor SRDF
#: (rebotarm_moveit_config/config/rebotarm.srdf @ pinned 39fbea5):
#: Adjacent pairs plus the "Never" finger pairs.  Kept inline so the
#: planner does not depend on the vendor tree at runtime.
DISABLED_SELF_COLLISION_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("base_link", "link1"),
    ("link1", "link2"),
    ("link2", "link3"),
    ("link3", "link4"),
    ("link4", "link5"),
    ("link5", "link6"),
    ("link6", "gripper_link"),
    ("gripper_link", "gripper_left"),
    ("gripper_link", "gripper_right"),
    ("gripper_left", "gripper_right"),
)

#: Links whose collision vs a world box is skipped by default: the base is
#: bolted to the work surface and link1 only yaws directly above it.
DEFAULT_WORLD_EXCLUDE_LINKS: Tuple[str, ...] = ("base_link", "link1")

#: Interpolation resolution for path checking: no joint moves more than
#: this between two checked configurations.
DEFAULT_MAX_STEP_RAD = 0.05


def default_package_dirs(urdf_path: str) -> List[str]:
    """mesh `package://` roots to try, derived from the URDF's repo layout.

    The canonical URDF lives at ``<repo>/urdf/...`` and references
    ``package://rebotarm_bringup/...`` which resolves under the pinned
    vendor tree (host) or the container workspace mounts.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(urdf_path)))
    candidates = [
        os.path.join(repo, "src", "reBotArmController_ROS2", "src"),
        os.path.join(repo, "ros2_ws", "src"),
        "/work/src/reBotArmController_ROS2/src",
    ]
    return [c for c in candidates if os.path.isdir(c)]


# ---- cell geometry ------------------------------------------------------


@dataclass(frozen=True)
class WorldBox:
    """Axis-aligned-in-frame box obstacle (frame per the YAML header)."""

    name: str
    size: Tuple[float, float, float]        # metres
    center: Tuple[float, float, float]      # metres
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    exclude_links: Tuple[str, ...] = ()


def _gantry_box(name: str, spec: Dict) -> WorldBox:
    """Parametric gantry crossbar (design doc sec. 6.3) -> box.

    Parameters: ``center_xy``, ``span_axis`` (x|y), ``span_m``,
    ``bar_width_m``, ``bar_height_m``, ``lower_edge_z_m``.
    """
    cx, cy = (float(v) for v in spec["center_xy"])
    span = float(spec["span_m"])
    width = float(spec.get("bar_width_m", 0.04))
    height = float(spec.get("bar_height_m", 0.04))
    z_lo = float(spec["lower_edge_z_m"])
    axis = str(spec.get("span_axis", "x")).lower()
    if axis == "x":
        size = (span, width, height)
    elif axis == "y":
        size = (width, span, height)
    else:
        raise ValueError(f"gantry span_axis must be 'x' or 'y', got {axis!r}")
    return WorldBox(
        name=name,
        size=size,
        center=(cx, cy, z_lo + height / 2.0),
        exclude_links=tuple(spec.get("exclude_links", ())),
    )


def load_cell_geometry(yaml_path: str) -> Tuple[List[WorldBox], Dict]:
    """Parse ``cell_geometry.yaml`` -> (obstacle boxes, zones dict).

    Zones (pick/place) are task metadata, NOT obstacles -- the arm must
    reach into them; they are returned untouched for task-layer use.
    """
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    boxes: List[WorldBox] = []
    for name, spec in (doc.get("obstacles") or {}).items():
        kind = str(spec.get("type", "box"))
        if kind == "box":
            boxes.append(WorldBox(
                name=str(name),
                size=tuple(float(v) for v in spec["size"]),
                center=tuple(float(v) for v in spec["center"]),
                rpy=tuple(float(v) for v in spec.get("rpy", (0.0, 0.0, 0.0))),
                exclude_links=tuple(spec.get("exclude_links", ())),
            ))
        elif kind == "gantry":
            boxes.append(_gantry_box(str(name), spec))
        else:
            raise ValueError(f"obstacle {name!r}: unknown type {kind!r}")
    return boxes, dict(doc.get("zones") or {})


# ---- results ------------------------------------------------------------


@dataclass(frozen=True)
class CollisionReport:
    ok: bool                                     # True = collision-free
    pairs: Tuple[Tuple[str, str], ...] = ()      # colliding geometry names

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


@dataclass(frozen=True)
class PathCheckResult:
    ok: bool
    reason: str = ""
    #: index of the SEGMENT (between waypoints i and i+1) that failed;
    #: -1 when ok, i for a failure between waypoint i and i+1 (a failure at
    #: waypoint 0 itself reports segment 0).
    failed_segment: int = -1
    q_colliding: Tuple[float, ...] = ()
    pairs: Tuple[Tuple[str, str], ...] = ()
    checked_configurations: int = 0

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


# ---- the checker --------------------------------------------------------


class CollisionCore:
    """Self- + world-box collision checks on the canonical model.

    Shares the :class:`KinematicsCore` model (same URDF, same joint
    ordering); builds the COLLISION geometry with ``pin.buildGeomFromUrdf``.
    """

    def __init__(
        self,
        kin: KinematicsCore,
        *,
        package_dirs: Optional[Sequence[str]] = None,
        cell_geometry_yaml: Optional[str] = None,
        world_exclude_links: Sequence[str] = DEFAULT_WORLD_EXCLUDE_LINKS,
    ) -> None:
        self.kin = kin
        model = kin.model
        dirs = list(package_dirs) if package_dirs else default_package_dirs(
            kin.urdf_path)
        self.geom_model = pin.buildGeomFromUrdf(
            model, kin.urdf_path, pin.GeometryType.COLLISION,
            package_dirs=dirs)
        self._n_robot_geoms = self.geom_model.ngeoms

        # geometry -> link name (via the parent frame).
        self._geom_link = [
            model.frames[g.parentFrame].name
            for g in self.geom_model.geometryObjects
        ]

        # Self-collision pairs: all pairs minus the SRDF-disabled set.
        self.geom_model.addAllCollisionPairs()
        disabled = {frozenset(p) for p in DISABLED_SELF_COLLISION_PAIRS}
        to_remove = [
            (p.first, p.second)
            for p in self.geom_model.collisionPairs
            if frozenset((self._geom_link[p.first],
                          self._geom_link[p.second])) in disabled
        ]
        for first, second in to_remove:
            self.geom_model.removeCollisionPair(pin.CollisionPair(first,
                                                                  second))

        # World boxes.
        self.world_boxes: List[WorldBox] = []
        self.zones: Dict = {}
        if cell_geometry_yaml:
            boxes, self.zones = load_cell_geometry(cell_geometry_yaml)
            for box in boxes:
                self._add_world_box(box, tuple(world_exclude_links))

        self.geom_data = pin.GeometryData(self.geom_model)

    def _add_world_box(self, box: WorldBox,
                       default_exclude: Tuple[str, ...]) -> None:
        placement = pin.SE3(
            pin.rpy.rpyToMatrix(*(float(v) for v in box.rpy)),
            np.asarray(box.center, dtype=np.float64))
        geometry = _fcl.Box(*box.size)
        try:
            # pinocchio >= 3.2/4 signature: (name, parent_joint, placement,
            # collision_geometry) -- what ros-jazzy-pinocchio 4.0 binds.
            go = pin.GeometryObject(f"world/{box.name}", 0, placement,
                                    geometry)
        except TypeError:  # Boost.Python.ArgumentError subclasses TypeError
            # older pip pinocchio: (name, parent_joint, geometry, placement)
            go = pin.GeometryObject(f"world/{box.name}", 0, geometry,
                                    placement)
        idx = self.geom_model.addGeometryObject(go)
        self._geom_link.append(f"world/{box.name}")
        exclude = set(box.exclude_links or default_exclude)
        for gi in range(self._n_robot_geoms):
            if self._geom_link[gi] in exclude:
                continue
            self.geom_model.addCollisionPair(pin.CollisionPair(gi, idx))
        self.world_boxes.append(box)

    # -- queries ----------------------------------------------------------

    def check_config(self, q6: Sequence[float]) -> CollisionReport:
        """Full collision report for one arm configuration (gripper at 0)."""
        q = self.kin.full_q(q6)
        pin.computeCollisions(self.kin.model, self.kin.data,
                              self.geom_model, self.geom_data, q, False)
        pairs = tuple(
            (self.geom_model.geometryObjects[p.first].name,
             self.geom_model.geometryObjects[p.second].name)
            for i, p in enumerate(self.geom_model.collisionPairs)
            if self.geom_data.collisionResults[i].isCollision()
        )
        return CollisionReport(ok=not pairs, pairs=pairs)

    def in_collision(self, q6: Sequence[float]) -> bool:
        """Fast boolean check (stops at the first colliding pair)."""
        q = self.kin.full_q(q6)
        return bool(pin.computeCollisions(
            self.kin.model, self.kin.data, self.geom_model, self.geom_data,
            q, True))

    def check_path(
        self,
        waypoints: Sequence[Sequence[float]],
        *,
        max_step_rad: float = DEFAULT_MAX_STEP_RAD,
    ) -> PathCheckResult:
        """Validate a waypoint path by checking interpolated configurations.

        Joint-space linear interpolation between consecutive waypoints, with
        enough samples that no joint moves more than ``max_step_rad``
        between checked configurations (waypoint 0 included).  The planner
        REJECTS a colliding path; it cannot re-route (cuMotion, later).
        """
        if len(waypoints) == 0:
            return PathCheckResult(ok=False, reason="empty path")
        checked = 0
        for seg in range(max(1, len(waypoints) - 1)):
            a = np.asarray(waypoints[seg], dtype=np.float64)[:ARM_DOF]
            if len(waypoints) == 1:
                b = a
            else:
                b = np.asarray(waypoints[seg + 1], dtype=np.float64)[:ARM_DOF]
            n = max(1, int(np.ceil(float(np.max(np.abs(b - a)))
                                   / max_step_rad)))
            # segment 0 checks its start config too; later segments start
            # at t>0 (their start was the previous segment's end).
            t0 = 0 if seg == 0 else 1
            for j in range(t0, n + 1):
                qj = a + (b - a) * (j / n)
                checked += 1
                if self.in_collision(qj):
                    report = self.check_config(qj)
                    return PathCheckResult(
                        ok=False, reason="path in collision",
                        failed_segment=seg,
                        q_colliding=tuple(float(v) for v in qj),
                        pairs=report.pairs,
                        checked_configurations=checked)
        return PathCheckResult(ok=True, checked_configurations=checked)
