#!/usr/bin/env python3
"""M6: sample the 100-trial pose-to-pose chain for the cuMotion gate.

Generates 160 randomized TCP target poses -- 100 chained trials plus
spare candidates: cuMotion's collision-aware IK occasionally cannot solve
a pose that ik_core can (measured trial-7 failure, trajopt
INVERSE_KINEMATICS_FAILURE); the trial runner records such candidates as
REJECTED (the arm never moves, so the chain stays intact) and takes the
next candidate, still executing exactly 100 trials.  Every candidate is

  * reachable: produced by FK of a random in-limit configuration AND
    re-verified with rebot_planner's ik_core (deliverable requirement);
  * collision-free under the TRUE cell model: rebot_planner's mesh-based
    CollisionCore (canonical URDF collision meshes + cell_geometry.yaml
    table/gantry);
  * collision-free under CUMOTION'S OWN model: the XRDF spheres + world
    buffers vs the trial world boxes (table dropped 10 mm, see below) and
    the non-ignored self-collision sphere pairs -- so no submitted goal is
    rejected merely because the sphere model is (intentionally) fatter
    than the meshes;
  * inside a workspace box and >= MIN_STEP_M from the previous target.

The trial world (also written to the output JSON, single source for the
runner): cell_geometry.yaml's table + gantry, with the table's top face
dropped by TABLE_DROP_M = 10 mm for cuMotion only.  Rationale (recorded in
the XRDF header): the robot pedestal is bolted flush to the board, so a
z=0 tabletop would touch base_link's own collision spheres and every
start state would be "in collision" for cuMotion.  Dropping the modeled
top 10 mm below the physical top clears the (z >= +2 mm) base spheres,
while the moving links' 10 mm XRDF buffers still keep planned paths above
the PHYSICAL tabletop; the trial verifier re-checks every executed plan
against the true z=0 model.

Output: artifacts/m6/trial_poses.json

Run: ~/isaaclab-venv/bin/python scripts/m6_sample_poses.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "rebot_planner"))

from rebot_planner.core import collision_core as cc  # noqa: E402
from rebot_planner.core.ik_core import (  # noqa: E402
    IK_ERR_ACCEPT, KinematicsCore)

URDF = REPO / "urdf" / "rebot_b601dm_canonical.urdf"
XRDF = REPO / "config" / "rebot_b601dm.xrdf"
CELL = REPO / "ros2_ws" / "src" / "rebot_planner" / "config" / \
    "cell_geometry.yaml"
OUT = REPO / "artifacts" / "m6" / "trial_poses.json"

SEED = 60100
N_TARGETS = 160           # 100 trials + spare candidates (see below;
                              # measured cuMotion-IK rejection rate ~20%)
TABLE_DROP_M = 0.010          # cuMotion world table top at -10 mm
MIN_STEP_M = 0.08             # min TCP distance between consecutive targets
SPHERE_MARGIN_M = 0.005       # extra clearance beyond XRDF buffers
LIMIT_MARGIN_RAD = 0.15       # keep samples well off the hard joint
                              # limits: near-limit wrist configs make
                              # cuMotion's collision-aware IK fragile
                              # (measured: INVERSE_KINEMATICS_FAILURE at
                              # joint4 margin 0.155 rad)
JAW_OPEN_M = 0.0715           # XRDF locked jaw value

# TCP workspace box (base_link frame): over the board, above the gripper's
# own envelope (jaw spheres reach ~7 cm below/around the TCP).
POS_LO = np.array([0.10, -0.30, 0.12])
POS_HI = np.array([0.48, 0.30, 0.45])

#: bridge start pose (scripts/b601_sim_bridge.py START_Q_ARM).
START_Q_ARM = [-0.5420, -1.1215, -1.0309, 0.8831, -0.3264, -0.4406]


def rot_to_quat_xyzw(R: np.ndarray) -> list[float]:
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:  # fall back for near-pi rotations
        d = np.diag(R)
        i = int(np.argmax(d))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
        q = [0.0, 0.0, 0.0, 0.0]
        q[i] = s / 4.0
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        x, y, z = q[0], q[1], q[2]
        w = (R[k, j] - R[j, k]) / s
    q = np.array([x, y, z, w])
    return [float(v) for v in q / np.linalg.norm(q)]


class SphereModel:
    """cuMotion-model feasibility: XRDF spheres vs world boxes + self pairs."""

    def __init__(self, kin: KinematicsCore, xrdf: dict,
                 world_boxes: list[dict]) -> None:
        self.kin = kin
        geom = xrdf["geometry"][xrdf["collision"]["geometry"]]["spheres"]
        self.links = list(geom.keys())
        self.frame_ids = {ln: kin.model.getFrameId(ln) for ln in self.links}
        self.centers = {ln: np.array([s["center"] for s in ss])
                        for ln, ss in geom.items()}
        self.radii = {ln: np.array([s["radius"] for s in ss])
                      for ln, ss in geom.items()}
        self.buffer = {ln: float(xrdf["collision"]["buffer_distance"][ln])
                       for ln in self.links}
        ignore = xrdf["self_collision"]["ignore"]
        ignored = {frozenset((a, b))
                   for a, bs in ignore.items() for b in bs}
        self.self_pairs = [
            (a, b) for i, a in enumerate(self.links)
            for b in self.links[i + 1:]
            if frozenset((a, b)) not in ignored
        ]
        # world boxes as (lo, hi) AABBs (rpy must be zero).
        self.world = []
        for box in world_boxes:
            assert all(abs(v) < 1e-9 for v in box.get("rpy", [0, 0, 0]))
            c = np.array(box["center"])
            h = np.array(box["size"]) / 2.0
            self.world.append((box["name"], c - h, c + h))

    def world_spheres(self, q6) -> dict:
        import pinocchio as pin

        q = self.kin.full_q(q6)
        q[6:8] = JAW_OPEN_M
        pin.forwardKinematics(self.kin.model, self.kin.data, q)
        pin.updateFramePlacements(self.kin.model, self.kin.data)
        out = {}
        for ln in self.links:
            M = np.asarray(self.kin.data.oMf[self.frame_ids[ln]].homogeneous)
            out[ln] = self.centers[ln] @ M[:3, :3].T + M[:3, 3]
        return out

    def valid(self, q6, margin: float = SPHERE_MARGIN_M) -> bool:
        ws = self.world_spheres(q6)
        # robot vs world boxes
        for ln in self.links:
            need = self.radii[ln] + self.buffer[ln] + margin
            for _, lo, hi in self.world:
                nearest = np.clip(ws[ln], lo, hi)
                d = np.linalg.norm(ws[ln] - nearest, axis=1)
                if np.any(d <= need):
                    return False
        # self pairs (no XRDF self buffer -> radii + margin)
        for a, b in self.self_pairs:
            d = np.linalg.norm(ws[a][:, None, :] - ws[b][None, :, :], axis=2)
            need = self.radii[a][:, None] + self.radii[b][None, :] + margin
            if np.any(d <= need):
                return False
        return True


def load_world_boxes() -> list[dict]:
    """cell_geometry.yaml table+gantry -> cuMotion world boxes (table
    dropped by TABLE_DROP_M)."""
    boxes, _zones = cc.load_cell_geometry(str(CELL))
    out = []
    for b in boxes:
        center = list(b.center)
        if b.name == "table":
            center[2] -= TABLE_DROP_M
        out.append({"name": b.name, "size": list(b.size),
                    "center": [float(v) for v in center],
                    "rpy": list(b.rpy)})
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    kin = KinematicsCore(str(URDF))
    checker = cc.CollisionCore(kin, cell_geometry_yaml=str(CELL))
    xrdf = yaml.safe_load(XRDF.read_text())
    world_boxes = load_world_boxes()
    sphere_model = SphereModel(kin, xrdf, world_boxes)

    if not checker.check_config(START_Q_ARM).ok:
        raise AssertionError("bridge start pose fails the mesh checker")
    if not sphere_model.valid(START_Q_ARM):
        raise AssertionError("bridge start pose fails the cuMotion sphere "
                             "model -- XRDF/world geometry must be revisited")

    lo = kin.lower + LIMIT_MARGIN_RAD
    hi = kin.upper - LIMIT_MARGIN_RAD
    targets = []
    prev_pos = kin.fk_tcp(START_Q_ARM)[:3, 3]
    stats = {"attempts": 0, "mesh_reject": 0, "sphere_reject": 0,
             "workspace_reject": 0, "step_reject": 0, "ik_reject": 0}
    while len(targets) < N_TARGETS:
        stats["attempts"] += 1
        if stats["attempts"] > 300000:
            raise RuntimeError(f"sampling stalled: {stats}")
        q = rng.uniform(lo, hi)
        T = kin.fk_tcp(q)
        pos = T[:3, 3]
        if np.any(pos < POS_LO) or np.any(pos > POS_HI):
            stats["workspace_reject"] += 1
            continue
        if np.linalg.norm(pos - prev_pos) < MIN_STEP_M:
            stats["step_reject"] += 1
            continue
        if not checker.check_config(q).ok:
            stats["mesh_reject"] += 1
            continue
        if not sphere_model.valid(q):
            stats["sphere_reject"] += 1
            continue
        # ik_core reachability re-verification from a perturbed seed.
        q_seed = np.clip(q + rng.normal(0.0, 0.1, 6), lo, hi)
        q_sol, res = kin.solve_tcp(T, q_seed)
        if res > IK_ERR_ACCEPT or not kin.within_limits(q_sol):
            stats["ik_reject"] += 1
            continue
        targets.append({
            "index": len(targets) + 1,
            "position": [round(float(v), 6) for v in pos],
            "quat_xyzw": [round(v, 8) for v in rot_to_quat_xyzw(T[:3, :3])],
            "q_ref": [round(float(v), 5) for v in q],
            "ik_residual": float(res),
        })
        prev_pos = pos

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "milestone": "M6",
        "artifact": "trial_poses",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "n_targets": N_TARGETS,
        "n_trials_required": 100,
        "start_q": START_Q_ARM,
        "table_drop_m": TABLE_DROP_M,
        "tool_frame": "gripper_tcp",
        "base_frame": "base_link",
        "world_boxes": world_boxes,
        "sampling_stats": stats,
        "targets": targets,
    }, indent=2) + "\n")
    print(f"wrote {OUT} ({len(targets)} targets); stats={stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
