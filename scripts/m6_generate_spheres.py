#!/usr/bin/env python3
"""M6: author collision spheres + XRDF for the B601-DM programmatically.

Design doc sec. 11 wants the spheres authored in the Isaac Sim Robot
Description Editor; this project authors them PROGRAMMATICALLY from the
canonical URDF's collision geometry instead (recorded M6 decision: the
editor is interactive/GUI-only, this is reproducible and hash-stable).

Fitting method (documented per deliverable):

  1. For every link in ``urdf/rebot_b601dm_canonical.urdf``, load the
     COLLISION mesh (vendor STL) and apply the URDF collision origin.
  2. Draw N area-weighted random surface samples (trimesh, fixed seed).
  3. Cluster the samples with seeded k-means (Lloyd, k-means++ init);
     k = clamp(ceil(major_extent / TARGET_SPAN_M), 4, 10) per link,
     honoring the deliverable's ~4-10 spheres/link.
  4. Each cluster becomes one sphere: Ritter approximate minimal
     enclosing sphere of the cluster's samples (guarantees every fitted
     sample is inside), radius padded by PAD_M to cover surface between
     samples.
  5. Coverage is re-verified against FRESH samples by
     ``scripts/m6_validate_spheres.py`` (different seed, pinocchio FK).

Special cases (all recorded in the XRDF header):

  * base_link mount region: the base sits flush on the work surface
    (banana-demo lesson: base plane == tabletop, z=0).  Sample points
    with z < BASE_FIT_MIN_Z_M are excluded from fitting AND validation,
    and base_link spheres are constrained to keep their lowest point at
    z >= BASE_SPHERE_MIN_Z_M so that cuMotion's world table (top face
    dropped to -10 mm, see m6_sample_poses) never collides with the
    robot's own pedestal.  base_link's world buffer is reduced to 4 mm
    accordingly (doc default 10 mm) -- the physical clearance argument:
    sphere bottom (>= +2 mm) to cuMotion table top (-10 mm) = 12 mm > 4 mm.
    The uncovered pedestal skirt (z in [0, 0.02]) is shielded by the
    world table + the moving links' 10 mm buffers, and every plan is
    re-checked against the true collision meshes by the trial gate.
  * gripper jaws: XRDF locks gripper_joint1/2 at JAW_OPEN_M (fully open,
    the doc's "conservative open shape"); this matches the sim bridge's
    seeded jaw state.
  * wrist_camera_mount (doc sec. 11.2 skeleton) does not exist in the
    canonical URDF (KDR-001: camera mount not yet authored) -- omitted.

Outputs:
  * config/rebot_b601dm.xrdf
  * urdf/rebot_b601dm_cumotion.urdf   (canonical + gripper_tcp fixed link,
    which cuMotion requires to exist for tool_frames)
  * artifacts/m6/sphere_fit_meta.json (method + per-link stats)

Run with the Isaac venv python (trimesh available):
  ~/isaaclab-venv/bin/python scripts/m6_generate_spheres.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
URDF_CANONICAL = REPO / "urdf" / "rebot_b601dm_canonical.urdf"
URDF_CUMOTION = REPO / "urdf" / "rebot_b601dm_cumotion.urdf"
XRDF_OUT = REPO / "config" / "rebot_b601dm.xrdf"
META_OUT = REPO / "artifacts" / "m6" / "sphere_fit_meta.json"
MESH_ROOT = REPO / "src" / "reBotArmController_ROS2" / "src"

FIT_SEED = 20260808
N_SAMPLES = 4000          # surface samples per link for fitting
TARGET_SPAN_M = 0.045     # aim for one sphere per ~45 mm of major extent
K_MIN, K_MAX = 4, 10      # deliverable: ~4-10 spheres per link
PAD_M = 0.005            # radius pad covering surface between samples

# base_link is a squat wide pedestal flush with the work surface; spheres
# constrained to stay above the table would need huge radii to also cover
# its lowest side walls.  Fit only z >= BASE_FIT_MIN_Z_M (the skirt below
# is represented by the world table obstacle + the moving links' 10 mm
# buffers, and the trial gate re-checks plans against the true meshes).
BASE_FIT_MIN_Z_M = 0.02
BASE_SPHERE_MIN_Z_M = 0.002  # lowest allowed point of any base_link sphere

# Per-link sphere-count overrides (still within the ~4-10 deliverable
# range).  The wrist cluster (link3..gripper_link) is compact: fewer,
# fatter spheres overlap NEIGHBORING links' spheres at ordinary poses,
# which would false-block cuMotion.  More, tighter spheres shrink the
# model instead of ignoring genuinely-possible collision pairs.
K_OVERRIDES = {
    "link2": 10, "link3": 10, "link4": 8, "link5": 8, "link6": 6,
    "gripper_link": 8, "gripper_left": 5, "gripper_right": 5,
}

JAW_OPEN_M = 0.0715       # locked (non-cspace) jaw value: conservative open

# KDR-001 canonical TCP (gripper_link frame), metres.
TCP_OFFSET = (-0.041763, 0.000008, 0.003427)

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]

# Doc sec. 11.2 buffer distances (base_link deviates, see module doc).
BUFFERS = {
    "base_link": 0.004,
    "link1": 0.010, "link2": 0.010, "link3": 0.010,
    "link4": 0.010, "link5": 0.010, "link6": 0.010,
    "gripper_link": 0.008,
    "gripper_left": 0.005, "gripper_right": 0.005,
}

# Doc sec. 11.2 self-collision ignore list == the vendor SRDF disabled
# pairs (rebot_planner.core.collision_core.DISABLED_SELF_COLLISION_PAIRS),
# minus the not-yet-authored wrist_camera_mount.
SELF_IGNORE = {
    "base_link": ["link1", "link2"],
    "link1": ["link2"],
    "link2": ["link3"],
    "link3": ["link4", "link5"],
    "link4": ["link5", "link6", "gripper_link"],
    "link5": ["link6", "gripper_link"],
    "link6": ["gripper_link", "gripper_left", "gripper_right"],
    "gripper_link": ["gripper_left", "gripper_right"],
    "gripper_left": ["gripper_right"],
}
# Ignore-list additions beyond the vendor SRDF adjacency (doc 11.3 rule:
# "adjacent links are ignored only where mechanically necessary").  Each
# added pair was MEASURED to false-trigger at 100% of 300 random
# mesh-collision-free configurations (sphere-model diagnostic, seed 4242)
# -- i.e. covering spheres of these links overlap at every ordinary pose:
#   * link5-gripper_link, link6-jaws: link6 is a 9.5 mm-thin flange ring,
#     so link5 / gripper_link / jaws are effectively adjacent;
#   * link4-link6 and link4-gripper_link: the wrist stack (link5: 80 mm,
#     link6: 9.5 mm) is shorter than the combined sphere radii;
#   * link3-link5: link4's 146 mm body is bridged by the link3/link5
#     covering spheres; real link3 contact stays guarded by the ACTIVE
#     link3-link6 and link3-gripper pairs (16%/13% trigger rates = real
#     wrist-fold protection);
#   * base_link-link2: link2's shoulder hub is rigidly ~30 mm above the
#     pedestal envelope at every yaw angle; deep folds toward the board
#     stay guarded by the world table box and the ACTIVE base_link-
#     link3/4/5/6/gripper pairs.
# Real contact in all ignored pairs is still caught by the mesh-based
# trial verifier (which keeps the vendor SRDF's full active-pair set).

# Doc sec. 11.2 default joint positions (verified collision-free by the
# validation script against the planner's mesh collision checker).
DEFAULT_Q = {"joint1": 0.0, "joint2": -0.75, "joint3": -0.55,
             "joint4": 0.0, "joint5": 0.0, "joint6": 0.0,
             "gripper_joint1": JAW_OPEN_M, "gripper_joint2": JAW_OPEN_M}


def rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def load_link_collision_points(urdf_path: Path, seed: int,
                               n_samples: int) -> dict:
    """{link_name: (N,3) surface samples in LINK frame} for every link
    with collision geometry (collision origin applied)."""
    tree = ET.parse(urdf_path)
    rng = np.random.default_rng(seed)
    out = {}
    for link in tree.getroot().findall("link"):
        name = link.get("name")
        pts_all = []
        for coll in link.findall("collision"):
            origin = coll.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin is not None:
                xyz = np.array([float(v) for v in
                                (origin.get("xyz") or "0 0 0").split()])
                rpy = np.array([float(v) for v in
                                (origin.get("rpy") or "0 0 0").split()])
            mesh_el = coll.find("geometry/mesh")
            if mesh_el is None:
                raise NotImplementedError(
                    f"{name}: only mesh collisions expected in this URDF")
            fn = mesh_el.get("filename")
            assert fn.startswith("package://"), fn
            rel = fn[len("package://"):]
            path = MESH_ROOT / rel
            if not path.is_file():
                raise FileNotFoundError(path)
            mesh = trimesh.load(str(path), force="mesh")
            pts, _ = trimesh.sample.sample_surface(
                mesh, n_samples, seed=int(rng.integers(0, 2**31)))
            R = rpy_to_matrix(*rpy)
            pts = pts @ R.T + xyz
            pts_all.append(np.asarray(pts))
        if pts_all:
            out[name] = np.vstack(pts_all)
    return out


def kmeans(points: np.ndarray, k: int, rng: np.random.Generator,
           iters: int = 60) -> np.ndarray:
    """Seeded Lloyd k-means with k-means++ init; returns labels."""
    n = len(points)
    centers = np.empty((k, 3))
    centers[0] = points[rng.integers(n)]
    d2 = np.sum((points - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        prob = d2 / d2.sum()
        centers[i] = points[rng.choice(n, p=prob)]
        d2 = np.minimum(d2, np.sum((points - centers[i]) ** 2, axis=1))
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(points[:, None, :] - centers[None, :, :],
                               axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for i in range(k):
            member = points[labels == i]
            if len(member):
                centers[i] = member.mean(axis=0)
            else:  # dead cluster: reseed at the point farthest from any center
                far = np.argmax(np.min(dists, axis=1))
                centers[i] = points[far]
    return labels


def ritter_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Ritter approximate minimal enclosing sphere (guaranteed enclosing)."""
    p0 = points[0]
    p1 = points[np.argmax(np.linalg.norm(points - p0, axis=1))]
    p2 = points[np.argmax(np.linalg.norm(points - p1, axis=1))]
    c = (p1 + p2) / 2.0
    r = float(np.linalg.norm(p1 - p2) / 2.0)
    for _ in range(3):  # a few growth passes
        d = np.linalg.norm(points - c, axis=1)
        i = int(np.argmax(d))
        if d[i] <= r + 1e-12:
            break
        # grow to include the farthest point
        r_new = (r + d[i]) / 2.0
        c = c + (points[i] - c) * ((d[i] - r_new) / d[i])
        r = float(r_new)
    # final exact enclosure guarantee
    r = float(np.max(np.linalg.norm(points - c, axis=1)))
    return c, r


def constrain_base_sphere(c: np.ndarray, r: float, points: np.ndarray,
                          pad: float) -> tuple[np.ndarray, float]:
    """Raise a base_link sphere so its lowest point stays at
    z >= BASE_SPHERE_MIN_Z_M while still enclosing all fitted samples
    WITH the pad (fixed-point iteration on the constrained height)."""
    if c[2] - r >= BASE_SPHERE_MIN_Z_M:
        return c, r
    c = c.copy()
    for _ in range(30):
        c[2] = BASE_SPHERE_MIN_Z_M + r
        r_needed = float(np.max(np.linalg.norm(points - c, axis=1))) + pad
        if abs(r_needed - r) < 1e-9:
            break
        r = r_needed
    c[2] = BASE_SPHERE_MIN_Z_M + r
    return c, r


def fit_link_spheres(name: str, pts: np.ndarray,
                     rng: np.random.Generator) -> list[dict]:
    if name == "base_link":
        pts = pts[pts[:, 2] >= BASE_FIT_MIN_Z_M]
        k = K_MAX  # wide flat pedestal: many small columns beat few fat balls
    elif name in K_OVERRIDES:
        k = K_OVERRIDES[name]
    else:
        extent = pts.max(axis=0) - pts.min(axis=0)
        k = int(np.clip(math.ceil(float(extent.max()) / TARGET_SPAN_M),
                        K_MIN, K_MAX))
    labels = kmeans(pts, k, rng)
    spheres = []
    for i in range(k):
        member = pts[labels == i]
        if len(member) < 3:
            continue
        c, r = ritter_sphere(member)
        r += PAD_M
        if name == "base_link":
            c, r = constrain_base_sphere(c, r, member, PAD_M)
            if c[2] - r < BASE_SPHERE_MIN_Z_M - 1e-9:
                raise AssertionError("base sphere constraint failed")
        if r > 0.09:
            raise AssertionError(
                f"{name}: sphere radius {r:.3f} m grossly exceeds the link "
                "envelope; refit with different clustering")
        spheres.append({"center": [round(float(v), 6) for v in c],
                        "radius": round(float(r), 6)})
    return spheres


def make_cumotion_urdf() -> None:
    """Canonical URDF + gripper_tcp massless fixed link (cuMotion requires
    tool_frames[0] to exist as a URDF link)."""
    text = URDF_CANONICAL.read_text()
    insert = (
        '  <!-- M6: canonical TCP as a URDF frame for cuMotion tool_frames\n'
        '       (KDR-001: gripper_tcp = gripper_link + [-0.041763, 0.000008,'
        ' 0.003427] m) -->\n'
        '  <link\n    name="gripper_tcp" />\n'
        '  <joint\n    name="gripper_tcp_joint"\n    type="fixed">\n'
        f'    <origin\n      xyz="{TCP_OFFSET[0]} {TCP_OFFSET[1]} '
        f'{TCP_OFFSET[2]}"\n      rpy="0 0 0" />\n'
        '    <parent\n      link="gripper_link" />\n'
        '    <child\n      link="gripper_tcp" />\n'
        '    <axis\n      xyz="0 0 0" />\n  </joint>\n'
    )
    assert "</robot>" in text and "gripper_tcp" not in text
    URDF_CUMOTION.write_text(text.replace("</robot>", insert + "</robot>"))


def emit_xrdf(link_spheres: dict) -> None:
    lines = []
    a = lines.append
    a("# reBot B601-DM XRDF for Isaac ROS 4.5 cuMotion (design doc sec. 11).")
    a("# GENERATED by scripts/m6_generate_spheres.py -- do not hand-edit.")
    a("#   * collision spheres fitted programmatically from the canonical")
    a("#     URDF collision meshes: area-weighted surface sampling ->")
    a("#     seeded k-means (k = 4..10/link) -> Ritter enclosing sphere")
    a(f"#     + {PAD_M * 1000:.0f} mm pad; validated by "
      "scripts/m6_validate_spheres.py.")
    a("#   * accel/jerk 1.0 / 10.0: doc commissioning values, not motor")
    a("#     maxima (velocity limits live in the URDF: 5/5/5/3/3/3 rad/s).")
    a("#   * jaws locked OPEN (0.0715 m) = doc 'conservative open shape';")
    a("#     matches the sim bridge's seeded jaw state.")
    a("#   * base_link: fitted only above z=+20 mm (k=10 columns), sphere")
    a("#     bottoms constrained to z>=+2 mm, buffer reduced to 4 mm; the")
    a("#     pedestal skirt below is represented by the world table model")
    a("#     (top dropped to -10 mm for cuMotion), so the robot's own base")
    a("#     never false-collides with the table it is bolted to.")
    a("#   * self_collision.ignore extends the vendor-SRDF adjacency ONLY")
    a("#     for pairs whose covering spheres measurably overlap at 100% of")
    a("#     300 random mesh-collision-free configs (see SELF_IGNORE notes")
    a("#     in the generator): link4-link6, link4-gripper_link,")
    a("#     link5-gripper_link, link6-jaws, link3-link5, base_link-link2.")
    a("#     Real contact in those pairs is still caught by the mesh-based")
    a("#     trial verifier (full vendor-SRDF active-pair set).")
    a("#   * wrist_camera_mount omitted: not yet authored (KDR-001).")
    a("format: xrdf")
    a("format_version: 1.0")
    a("")
    a("modifiers:")
    a("  - set_base_frame: base_link")
    a("")
    a("default_joint_positions:")
    for j, v in DEFAULT_Q.items():
        a(f"  {j}: {v}")
    a("")
    a("cspace:")
    a(f"  joint_names: [{', '.join(ARM_JOINTS)}]")
    a("  acceleration_limits: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]")
    a("  jerk_limits: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]")
    a("")
    a("tool_frames:")
    a("  - gripper_tcp")
    a("")
    a("collision:")
    a("  geometry: rebot_collision_spheres")
    a("  buffer_distance:")
    for link in link_spheres:
        a(f"    {link}: {BUFFERS[link]}")
    a("")
    a("self_collision:")
    a("  geometry: rebot_collision_spheres")
    a("  ignore:")
    for link, others in SELF_IGNORE.items():
        a(f"    {link}: [{', '.join(others)}]")
    a("")
    a("geometry:")
    a("  rebot_collision_spheres:")
    a("    spheres:")
    for link, spheres in link_spheres.items():
        a(f"      {link}:")
        for s in spheres:
            # fixed-decimal formatting: bare "6e-06" would be parsed as a
            # STRING by YAML 1.1 loaders (PyYAML), breaking consumers.
            c = [f"{v:.6f}" for v in s["center"]]
            a(f"        - center: [{c[0]}, {c[1]}, {c[2]}]")
            a(f"          radius: {s['radius']:.6f}")
    XRDF_OUT.write_text("\n".join(lines) + "\n")


def main() -> int:
    rng = np.random.default_rng(FIT_SEED)
    link_pts = load_link_collision_points(URDF_CANONICAL, FIT_SEED, N_SAMPLES)
    link_spheres = {}
    meta = {"method": "area-weighted surface sampling + seeded k-means "
                      "(Lloyd, k-means++ init) + Ritter enclosing sphere "
                      f"+ {PAD_M} m pad",
            "seed": FIT_SEED, "n_samples_per_link": N_SAMPLES,
            "target_span_m": TARGET_SPAN_M, "k_range": [K_MIN, K_MAX],
            "base_fit_min_z_m": BASE_FIT_MIN_Z_M,
            "base_sphere_min_z_m": BASE_SPHERE_MIN_Z_M,
            "links": {}}
    for link, pts in link_pts.items():
        spheres = fit_link_spheres(link, pts, rng)
        link_spheres[link] = spheres
        radii = [s["radius"] for s in spheres]
        meta["links"][link] = {
            "n_spheres": len(spheres),
            "radius_min_m": min(radii), "radius_max_m": max(radii),
            "fit_points": int(len(pts)),
        }
        print(f"{link:14s} {len(spheres):2d} spheres, "
              f"r=[{min(radii):.4f}..{max(radii):.4f}] m")
    emit_xrdf(link_spheres)
    make_cumotion_urdf()
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    META_OUT.write_text(json.dumps(meta, indent=2) + "\n")
    total = sum(len(s) for s in link_spheres.values())
    print(f"\nwrote {XRDF_OUT} ({total} spheres), {URDF_CUMOTION}, {META_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
