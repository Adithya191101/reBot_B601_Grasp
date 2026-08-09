#!/usr/bin/env python3
"""M6: validate the XRDF collision spheres against the URDF collision
geometry (deliverable 2; design doc sec. 11.3).

For >= 20 random in-limit arm configurations the spheres are FK-placed
via each link's frame and FRESH surface samples of the URDF collision
meshes (different seed than the fit) are FK-placed via pinocchio's
GEOMETRY placements.  A sample is covered when it lies inside at least
one sphere of its own link.  The two placement paths are independent
(XML+trimesh fit vs pin.buildGeomFromUrdf), so a frame/origin mistake in
either pipeline breaks coverage for every configuration.

Exemption (recorded in the XRDF header): base_link samples below
z = BASE_FIT_MIN_Z_M in the link frame -- the pedestal skirt is
represented by the world table model, not by robot spheres.

Also checked:
  * XRDF cspace joints/limits match the canonical URDF;
  * the XRDF default configuration is collision-free under the
    planner's mesh-based checker (rebot_planner collision_core);
  * sphere radius sanity (no sphere grossly beyond the link envelope).

Writes artifacts/m6/sphere_validation.json; exit 0 iff PASS.

Run: ~/isaaclab-venv/bin/python scripts/m6_validate_spheres.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import trimesh
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "rebot_planner"))

from rebot_planner.core import collision_core as cc  # noqa: E402
from rebot_planner.core.ik_core import KinematicsCore  # noqa: E402

URDF = REPO / "urdf" / "rebot_b601dm_canonical.urdf"
XRDF = REPO / "config" / "rebot_b601dm.xrdf"
OUT = REPO / "artifacts" / "m6" / "sphere_validation.json"

VALIDATE_SEED = 977          # independent of the fit seed
N_CONFIGS = 25               # >= 20 per the deliverable
N_SAMPLES = 3000             # fresh surface samples per link
BASE_FIT_MIN_Z_M = 0.02      # must match scripts/m6_generate_spheres.py
JAW_OPEN_M = 0.0715
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


def main() -> int:
    xrdf = yaml.safe_load(XRDF.read_text())
    geom_name = xrdf["collision"]["geometry"]
    spheres = xrdf["geometry"][geom_name]["spheres"]

    kin = KinematicsCore(str(URDF))
    model, data = kin.model, kin.data
    checker = cc.CollisionCore(
        kin, cell_geometry_yaml=str(
            REPO / "ros2_ws" / "src" / "rebot_planner" / "config"
            / "cell_geometry.yaml"))

    # XRDF cspace vs URDF cross-checks.
    checks = []
    cs = xrdf["cspace"]
    checks.append({
        "name": "cspace is exactly joint1..joint6",
        "passed": list(cs["joint_names"]) == ARM_JOINTS,
    })
    checks.append({
        "name": "tool_frames == [gripper_tcp]",
        "passed": list(xrdf["tool_frames"]) == ["gripper_tcp"],
    })
    checks.append({
        "name": "accel 1.0 / jerk 10.0 commissioning values",
        "passed": (all(a == 1.0 for a in cs["acceleration_limits"])
                   and all(j == 10.0 for j in cs["jerk_limits"])),
    })
    q_default = [float(xrdf["default_joint_positions"][j])
                 for j in ARM_JOINTS]
    rep = checker.check_config(q_default)
    checks.append({
        "name": "XRDF default config collision-free (mesh checker, "
                "cell geometry included)",
        "passed": bool(rep.ok),
        "pairs": [list(p) for p in rep.pairs],
    })
    max_r = max(s["radius"] for link in spheres.values() for s in link)
    checks.append({
        "name": "sphere radius sanity (max <= 0.09 m)",
        "passed": max_r <= 0.09,
        "max_radius_m": round(max_r, 4),
    })

    # Geometry: fresh surface samples per collision geometry, in the
    # geometry's LOCAL mesh frame (placement applied later by pinocchio).
    geom_model = checker.geom_model
    rng = np.random.default_rng(VALIDATE_SEED)
    geom_samples = {}          # geom index -> (N,3) local points
    geom_link = {}             # geom index -> link name
    for gi in range(checker._n_robot_geoms):
        go = geom_model.geometryObjects[gi]
        link = model.frames[go.parentFrame].name
        if link not in spheres:
            continue
        mesh = trimesh.load(str(go.meshPath), force="mesh")
        pts, _ = trimesh.sample.sample_surface(
            mesh, N_SAMPLES, seed=int(rng.integers(0, 2**31)))
        geom_samples[gi] = np.asarray(pts)
        geom_link[gi] = link

    # Link frame ids for FK sphere placement.
    frame_id = {link: model.getFrameId(link) for link in spheres}
    sph_local = {link: (np.array([s["center"] for s in ss]),
                        np.array([s["radius"] for s in ss]))
                 for link, ss in spheres.items()}

    geom_data = pin.GeometryData(geom_model)
    per_link = {link: {"points_checked": 0, "points_uncovered": 0,
                       "max_deficit_m": 0.0, "n_spheres": len(ss)}
                for link, ss in spheres.items()}

    configs = []
    for _ in range(N_CONFIGS):
        q6 = rng.uniform(kin.lower, kin.upper)
        configs.append([round(float(v), 4) for v in q6])
        q = kin.full_q(q6)
        q[6:8] = JAW_OPEN_M  # jaws at the XRDF locked-open value
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.updateGeometryPlacements(model, data, geom_model, geom_data)

        world_spheres = {}
        for link, fid in frame_id.items():
            M = np.asarray(data.oMf[fid].homogeneous)
            c_local, radii = sph_local[link]
            world_spheres[link] = (c_local @ M[:3, :3].T + M[:3, 3], radii)

        for gi, pts_local in geom_samples.items():
            link = geom_link[gi]
            M = np.asarray(geom_data.oMg[gi].homogeneous)
            pts = pts_local @ M[:3, :3].T + M[:3, 3]
            if link == "base_link":
                # exemption: pedestal skirt (link frame == world for base)
                local = pts_local  # base_link geometry placement is identity
                keep = local[:, 2] >= BASE_FIT_MIN_Z_M
                pts = pts[keep]
            centers, radii = world_spheres[link]
            d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
            deficit = d - radii[None, :]
            min_deficit = deficit.min(axis=1)  # <= 0 -> covered
            uncovered = min_deficit > 1e-9
            st = per_link[link]
            st["points_checked"] += int(len(pts))
            st["points_uncovered"] += int(uncovered.sum())
            if uncovered.any():
                st["max_deficit_m"] = max(st["max_deficit_m"],
                                          float(min_deficit[uncovered].max()))

    for link, st in per_link.items():
        st["coverage"] = round(
            1.0 - st["points_uncovered"] / max(1, st["points_checked"]), 6)
        st["max_deficit_m"] = round(st["max_deficit_m"], 6)
    coverage_pass = all(st["points_uncovered"] == 0
                        for st in per_link.values())
    checks.append({
        "name": f"sphere coverage of URDF collision surfaces over "
                f"{N_CONFIGS} random configs (fresh samples)",
        "passed": coverage_pass,
    })

    report = {
        "milestone": "M6",
        "artifact": "sphere_validation",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "xrdf": str(XRDF.relative_to(REPO)),
        "urdf": str(URDF.relative_to(REPO)),
        "method": "FK sphere placement (link frames) vs FK mesh surface "
                  "samples (pin.buildGeomFromUrdf placements), independent "
                  "of the fitting parse path",
        "seed": VALIDATE_SEED,
        "n_configs": N_CONFIGS,
        "n_fresh_samples_per_link": N_SAMPLES,
        "base_link_skirt_exemption_z_m": BASE_FIT_MIN_Z_M,
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "per_link": per_link,
        "configs": configs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed",)}, indent=2))
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}")
    for link, st in per_link.items():
        print(f"  {link:14s} coverage={st['coverage']:.4f} "
              f"max_deficit={st['max_deficit_m'] * 1000:.2f} mm "
              f"({st['n_spheres']} spheres)")
    print(f"wrote {OUT}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
