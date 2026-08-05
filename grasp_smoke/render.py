"""Exact analytic renderer for a textured box on a ground plane.

Pure NumPy ray casting. Every quantity it emits is closed-form, not sampled:
optical-axis Z depth, the instance mask, and ``grasp_gt``. That exactness is the
point -- it makes the A0 red tests (PLAN.md 5.2.4) meaningful, because any error
they report is in :mod:`grasp_smoke.grasp` or :mod:`grasp_smoke.geometry` and
nowhere else.

It doubles as the ``analytic`` capture backend, so the full chain
(randomize -> capture -> replay -> masks -> PoseStamped -> scorer) is runnable
and testable without a simulator. Datasets record which backend produced them in
``manifest.json``; analytic results are never reported as Isaac Sim results.

The target follows PLAN.md 5.2.3: a **flat-topped, non-square, textured** box.
Non-square matters -- a square top face has a degenerate opening axis and the
angular metric becomes noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import (
    invert_transform,
    make_transform,
    normalize,
    rotation_about_axis,
    transform_direction,
    transform_point,
)


@dataclass
class BoxTarget:
    """Axis-aligned in its own frame; ``T_base_object`` places it in the world."""

    half_extents: np.ndarray      # (3,) metres. x=long, y=short, z=height
    T_base_object: np.ndarray     # (4, 4)
    base_color: np.ndarray        # (3,) float 0..1
    checker_m: float = 0.012      # texture cell size, metres

    @property
    def dims_m(self) -> tuple:
        return tuple(float(2.0 * v) for v in self.half_extents)

    def grasp_gt(self) -> tuple:
        """(position_base, open_axis_base) for the canonical top-face grasp.

        Position: centre of the top face -- *not* the object origin, which is the
        distinction PLAN.md 5.2.3 insists on.
        Opening axis: the box's short horizontal axis (its local +y), because a
        parallel-jaw gripper closes across the short dimension.
        """
        top_local = np.array([0.0, 0.0, float(self.half_extents[2])])
        position = transform_point(self.T_base_object, top_local)
        open_axis = normalize(transform_direction(self.T_base_object, np.array([0.0, 1.0, 0.0])))
        return position, open_axis


@dataclass
class RenderResult:
    rgb: np.ndarray        # uint8 (H, W, 3)
    depth_m: np.ndarray    # float32 (H, W), optical-axis Z, 0 where nothing hit
    mask: np.ndarray       # uint8 (H, W), 1 on the target


def _pixel_ray_directions(width: int, height: int, K: np.ndarray) -> np.ndarray:
    """Unit ray directions in the optical frame, one per pixel, shape (H, W, 3)."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = np.arange(width, dtype=np.float64) + 0.5
    v = np.arange(height, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)


def _ray_box(origin: np.ndarray, dirs: np.ndarray, half: np.ndarray) -> tuple:
    """Slab method. Returns (t_hit, hit_mask, face_axis) for a box at the origin."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t1 = (-half - origin) * inv
        t2 = (half - origin) * inv
    t_near_axis = np.minimum(t1, t2)
    t_far_axis = np.maximum(t1, t2)
    t_near = np.nanmax(t_near_axis, axis=-1)
    t_far = np.nanmin(t_far_axis, axis=-1)
    hit = (t_far >= np.maximum(t_near, 0.0)) & np.isfinite(t_near)
    face_axis = np.argmax(np.nan_to_num(t_near_axis, nan=-np.inf), axis=-1)
    return t_near, hit, face_axis


def _ray_plane_z0(origin: np.ndarray, dirs: np.ndarray) -> tuple:
    """Intersect z=0 in the frame the ray is expressed in."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -origin[2] / dirs[..., 2]
    return t, np.isfinite(t) & (t > 0)


def render(
    target: BoxTarget,
    T_base_cam: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    rng: Optional[np.random.Generator] = None,
    include_target: bool = True,
) -> RenderResult:
    """Ray-cast the scene. Depth is exact optical-axis Z in metres."""
    rng = rng or np.random.default_rng(0)
    dirs_opt = _pixel_ray_directions(width, height, K)

    # Ground plane, in base coordinates.
    T_cam_base = invert_transform(T_base_cam)
    origin_base = T_base_cam[:3, 3]
    dirs_base = dirs_opt @ T_base_cam[:3, :3].T
    t_ground, ground_hit = _ray_plane_z0(origin_base, dirs_base)

    # Target box, in object coordinates.
    T_object_cam = invert_transform(target.T_base_object) @ T_base_cam
    origin_obj = T_object_cam[:3, 3]
    dirs_obj = dirs_opt @ T_object_cam[:3, :3].T
    t_box, box_hit, face_axis = _ray_box(origin_obj, dirs_obj, np.asarray(target.half_extents))
    box_hit = box_hit & (t_box > 0)
    if not include_target:
        box_hit = np.zeros_like(box_hit)

    take_box = box_hit & (~ground_hit | (t_box < t_ground))
    t = np.where(take_box, t_box, np.where(ground_hit, t_ground, 0.0))

    # Optical-axis Z: the z component of the hit point in the optical frame.
    depth = t * dirs_opt[..., 2]
    depth = np.where(take_box | ground_hit, depth, 0.0)

    rgb = np.zeros((height, width, 3), dtype=np.float64)

    # Ground: low-contrast procedural texture so segmentation has to work a little.
    gp = origin_base[None, None, :] + t_ground[..., None] * dirs_base
    gcheck = ((np.floor(gp[..., 0] / 0.05) + np.floor(gp[..., 1] / 0.05)) % 2.0)
    ground_rgb = 0.32 + 0.06 * gcheck
    rgb[ground_hit] = ground_rgb[ground_hit, None]

    # Target: checkerboard in object coordinates, shaded per face.
    bp = origin_obj[None, None, :] + t_box[..., None] * dirs_obj
    bcheck = (
        np.floor(bp[..., 0] / target.checker_m) + np.floor(bp[..., 1] / target.checker_m)
    ) % 2.0
    shade = np.choose(np.clip(face_axis, 0, 2), [0.78, 0.88, 1.0])
    tex = (0.70 + 0.30 * bcheck) * shade
    rgb[take_box] = np.clip(target.base_color[None, :] * tex[take_box, None], 0.0, 1.0)

    rgb = np.clip(rgb + rng.normal(0.0, 0.004, rgb.shape), 0.0, 1.0)
    return RenderResult(
        rgb=(rgb * 255.0).astype(np.uint8),
        depth_m=depth.astype(np.float32),
        mask=take_box.astype(np.uint8),
    )


# --------------------------------------------------------------------------
# Fixtures and randomisation
# --------------------------------------------------------------------------


def fronto_parallel_fixture(
    z_m: float = 0.60,
    yaw_deg: float = 0.0,
    width: int = 640,
    height: int = 480,
    fx: float = 600.0,
    half_extents=(0.060, 0.030, 0.020),
) -> tuple:
    """The A0 fixture: camera straight down, top face at exactly constant depth.

    Returns ``(target, T_base_cam, K, expected_position_base, expected_axis_base)``.
    The camera looks along base -Z from height ``z_m + top``, so every point of
    the top face sits at optical-axis Z = ``z_m``. Under those conditions the
    vendor algorithm is exact and the >=1 mm / >=1 deg bar is a real red test.
    """
    from .geometry import make_intrinsics

    half = np.asarray(half_extents, dtype=np.float64)
    R_obj = rotation_about_axis(np.array([0.0, 0.0, 1.0]), np.deg2rad(yaw_deg))
    T_base_object = make_transform(R_obj, np.array([0.0, 0.0, 0.0]))
    target = BoxTarget(
        half_extents=half,
        T_base_object=T_base_object,
        base_color=np.array([0.85, 0.35, 0.25]),
    )

    # Optical frame: +z down (forward), +x = base +x, +y = base -y (right-handed).
    R_cam = np.column_stack([
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
    ])
    cam_height = float(half[2]) + z_m
    T_base_cam = make_transform(R_cam, np.array([0.0, 0.0, cam_height]))
    K = make_intrinsics(fx, fx, (width - 1) / 2.0, (height - 1) / 2.0)

    expected_position, expected_axis = target.grasp_gt()
    return target, T_base_cam, K, expected_position, expected_axis


def randomize_scene(seed: int, tilt_deg: float = 0.0) -> tuple:
    """Deterministic per-scene randomisation. Returns (target, T_base_cam, K).

    ``tilt_deg`` is the A2 stress knob: 0 reproduces the fronto-parallel case,
    larger values tilt the camera off the object normal, which is exactly where
    the single-depth back-projection stops being exact.
    """
    from .geometry import look_at, make_intrinsics

    rng = np.random.default_rng(seed)
    half = np.array([
        rng.uniform(0.050, 0.075),
        rng.uniform(0.022, 0.034),
        rng.uniform(0.015, 0.025),
    ])
    # Guard the non-square requirement: long axis must dominate clearly.
    if half[0] / half[1] < 1.5:
        half[0] = half[1] * 1.8

    yaw = rng.uniform(-np.pi, np.pi)
    # The object frame is the box CENTRE, and the box rests on the ground, so the
    # centre sits at +half_z. Both capture backends must use this same convention:
    # if one of them additionally lifts the box by half_z, grasp_gt lands half a
    # box-height off and every position error is inflated by exactly that amount.
    T_base_object = make_transform(
        rotation_about_axis(np.array([0.0, 0.0, 1.0]), yaw),
        np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), float(half[2])]),
    )
    target = BoxTarget(
        half_extents=half,
        T_base_object=T_base_object,
        base_color=np.array([
            rng.uniform(0.55, 0.95), rng.uniform(0.25, 0.65), rng.uniform(0.20, 0.55),
        ]),
        checker_m=float(rng.uniform(0.008, 0.016)),
    )

    dist = rng.uniform(0.50, 0.70)
    tilt = np.deg2rad(tilt_deg)
    azim = rng.uniform(-np.pi, np.pi)
    center = T_base_object[:3, 3]
    eye = center + np.array([
        dist * np.sin(tilt) * np.cos(azim),
        dist * np.sin(tilt) * np.sin(azim),
        dist * np.cos(tilt),
    ])
    up = np.array([0.0, 1.0, 0.0]) if tilt_deg > 1e-6 else np.array([0.0, 1.0, 0.0])
    T_base_cam = look_at(eye, T_base_object[:3, 3], up)
    fx = rng.uniform(580.0, 640.0)
    K = make_intrinsics(fx, fx, (640 - 1) / 2.0, (480 - 1) / 2.0)
    return target, T_base_cam, K
