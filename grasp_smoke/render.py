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
    distractors: Optional[list] = None,
) -> RenderResult:
    """Ray-cast the scene. Depth is exact optical-axis Z in metres.

    ``distractors`` are additional boxes rendered into RGB and depth but **never**
    into the mask. They are target-like on purpose: a scene containing only the
    target lets any "find the salient blob" heuristic score as though it were a
    detector.
    """
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

    # Distractors: same treatment as the target in RGB and depth, absent from the
    # mask. Occlusion is resolved against whatever is already nearest.
    for distractor in distractors or []:
        T_d_cam = invert_transform(distractor.T_base_object) @ T_base_cam
        o_d = T_d_cam[:3, 3]
        d_d = dirs_opt @ T_d_cam[:3, :3].T
        t_d, hit_d, face_d = _ray_box(o_d, d_d, np.asarray(distractor.half_extents))
        hit_d = hit_d & (t_d > 0)
        nearer = hit_d & ((t <= 0) | (t_d < t))
        if not np.any(nearer):
            continue
        dp = o_d[None, None, :] + t_d[..., None] * d_d
        dtex = _surface_texture(dp, distractor, face_d)
        rgb[nearer] = np.clip(distractor.base_color[None, :] * dtex[nearer, None], 0.0, 1.0)
        depth = np.where(nearer, t_d * dirs_opt[..., 2], depth)
        t = np.where(nearer, t_d, t)
        take_box = take_box & ~nearer          # a distractor in front occludes

    # Target: self-authored procedural texture, shaded per face.
    bp = origin_obj[None, None, :] + t_box[..., None] * dirs_obj
    tex = _surface_texture(bp, target, face_axis)
    rgb[take_box] = np.clip(target.base_color[None, :] * tex[take_box, None], 0.0, 1.0)

    rgb = np.clip(rgb + rng.normal(0.0, 0.004, rgb.shape), 0.0, 1.0)
    return RenderResult(
        rgb=(rgb * 255.0).astype(np.uint8),
        depth_m=depth.astype(np.float32),
        mask=take_box.astype(np.uint8),
    )


def _surface_texture(points_obj: np.ndarray, box: "BoxTarget", face_axis: np.ndarray) -> np.ndarray:
    """Self-authored multi-scale procedural texture in object coordinates.

    A plain checkerboard is nearly a solid colour once it is small in frame, which
    makes the target unrepresentatively easy to segment. This layers a coarse
    checker, a finer stripe, and a deterministic value ramp so the surface has
    structure at more than one scale -- without needing any external texture asset
    (PLAN.md 5.2.3 asks for a textured target).
    """
    u = points_obj[..., 0] / box.checker_m
    v = points_obj[..., 1] / box.checker_m
    checker = (np.floor(u) + np.floor(v)) % 2.0
    stripes = 0.5 * (1.0 + np.sin(6.0 * np.pi * u))
    ramp = 0.5 * (1.0 + np.cos(2.0 * np.pi * v * 0.5))
    shade = np.choose(np.clip(face_axis, 0, 2), [0.78, 0.88, 1.0])
    return (0.55 + 0.25 * checker + 0.12 * stripes + 0.08 * ramp) * shade


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


def randomize_scene(
    seed: int,
    tilt_deg: float = 0.0,
    aim_jitter: bool = False,
    n_distractors: int = 0,
) -> tuple:
    """Deterministic per-scene randomisation.

    Returns ``(target, T_base_cam, K, distractors)``.

    ``tilt_deg`` is the A2/B2 stress knob: 0 reproduces the fronto-parallel case,
    larger values tilt the camera off the object normal, which is where the
    single-depth back-projection stops being exact.

    ``aim_jitter`` perturbs where the camera *looks* and rolls it slightly, so the
    target stops landing dead-centre in every frame. Without it, principal-point
    errors and centre-biased heuristics both go undetected -- everything of
    interest sits where the lens is most honest.
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

    aim = np.array(center, dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0])
    if aim_jitter:
        # Off-axis framing: look a few centimetres away from the object centre and
        # roll the camera, so the target lands off-centre and non-axis-aligned.
        aim = aim + rng.uniform(-0.035, 0.035, size=3) * np.array([1.0, 1.0, 0.3])
        roll = rng.uniform(-np.deg2rad(12.0), np.deg2rad(12.0))
        up = rotation_about_axis(np.array([0.0, 0.0, 1.0]), roll) @ up
    T_base_cam = look_at(eye, aim, up)

    fx = rng.uniform(580.0, 640.0)
    K = make_intrinsics(fx, fx, (640 - 1) / 2.0, (480 - 1) / 2.0)

    distractors = []
    for _ in range(int(n_distractors)):
        d_half = np.array([
            rng.uniform(0.045, 0.075), rng.uniform(0.020, 0.034), rng.uniform(0.014, 0.026),
        ])
        if d_half[0] / d_half[1] < 1.5:
            d_half[0] = d_half[1] * 1.8
        angle = rng.uniform(-np.pi, np.pi)
        radius = rng.uniform(0.10, 0.16)
        distractors.append(BoxTarget(
            half_extents=d_half,
            T_base_object=make_transform(
                rotation_about_axis(np.array([0.0, 0.0, 1.0]), rng.uniform(-np.pi, np.pi)),
                center + np.array([radius * np.cos(angle), radius * np.sin(angle),
                                   float(d_half[2]) - float(half[2])]),
            ),
            base_color=np.array([
                rng.uniform(0.55, 0.95), rng.uniform(0.25, 0.65), rng.uniform(0.20, 0.55),
            ]),
            checker_m=float(rng.uniform(0.008, 0.016)),
        ))

    return target, T_base_cam, K, distractors
