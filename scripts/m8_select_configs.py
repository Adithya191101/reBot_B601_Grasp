#!/usr/bin/env python3
"""M8: deterministic inputs for the mapping acceptance test (doc 15.4).

Host-side (isaaclab-venv python), the M6/M7 sampling machinery reused
unchanged.  Produces artifacts/m8/acceptance_configs.json with:

* ``map_configs`` -- N_MAP_CONFIGS + spares arm configurations, each
  mesh-collision-free (canonical URDF meshes, full cell: true-z table +
  gantry) AND valid under cuMotion's own XRDF sphere model, with
  pairwise-distinct TCP positions spread through the workspace.  The
  acceptance runner drives the arm through >= N_MAP_REQUIRED of them and
  the map must never contain the robot (doc 15.4 "robot body absent").
  Each config ships its world-frame XRDF sphere set (center + radius,
  jaws at the sim's locked-open state) so the in-container runner can
  test "occupied voxel inside the robot volume" without any kinematics
  dependency.
* ``pair`` (+ spares) -- an M7-style pose pair whose straight joint
  chord PROVABLY collides with the gantry and ONLY the gantry
  (chord_gantry_stats, both directions of the mesh proof).  Phase D
  plans it with NO statically-declared gantry: only the nvblox map can
  explain a re-route (doc 15.4 "cuMotion routes around the gantry",
  M8 deliverable "mapped, not statically-declared").
* world boxes for the goal-embedded static worlds: ``world_boxes``
  (table + gantry, M6 convention, safe transit world for setup moves)
  and ``world_boxes_table_only`` (phase D world).
* the true gantry box, nvblox workspace bounds and voxel size, and the
  numeric gates the runner/verifier apply.

Also regenerates artifacts/m7/cell_geometry_no_gantry.yaml (via the M7
helper) so the M8 verifier's contrast model always exists.

Run: ~/isaaclab-venv/bin/python scripts/m8_select_configs.py
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
sys.path.insert(0, str(REPO / "scripts"))

from rebot_planner.core import collision_core as cc  # noqa: E402

from m6_sample_poses import (  # noqa: E402
    JAW_OPEN_M, START_Q_ARM, SphereModel, load_world_boxes)
from m7_sample_pairs import (  # noqa: E402
    CHORD_STEP_RAD, MIN_GANTRY_HITS, OpenJawKin, PICK_HI, PICK_LO,
    PLACE_HI, PLACE_LO, chord_gantry_stats, write_no_gantry_cell)

URDF = REPO / "urdf" / "rebot_b601dm_canonical.urdf"
XRDF = REPO / "config" / "rebot_b601dm.xrdf"
CELL = REPO / "ros2_ws" / "src" / "rebot_planner" / "config" / \
    "cell_geometry.yaml"
CELL_NO_GANTRY = REPO / "artifacts" / "m7" / "cell_geometry_no_gantry.yaml"
ART = REPO / "artifacts" / "m8"
OUT = ART / "acceptance_configs.json"

SEED = 80100
N_MAP_REQUIRED = 5            # doc 15.4: ">= five configurations"
N_MAP_CONFIGS = 8             # + spares for cuMotion setup rejections
N_PAIRS = 3                   # 1 scored + spares
LIMIT_MARGIN_RAD = 0.15       # M6/M7 value
SPHERE_MARGIN_M = 0.005       # M6/M7 value
MIN_TCP_SEP_M = 0.12          # map configs must be distinct viewpoints
MAP_TCP_LO = np.array([0.15, -0.30, 0.13])
MAP_TCP_HI = np.array([0.55, 0.30, 0.45])
CONFIG_TRIES = 4000
PAIR_TRIES = 400
ENDPOINT_TRIES = 4000
MIN_TCP_STEP_M = 0.15         # M7: a pair must be a real transit

#: nvblox workspace (design doc 15.5) -- must match the bridge scene AND
#: the nvblox node parameters in scripts/m8_mapping_acceptance.sh.
WORKSPACE_MIN = [-0.10, -0.35, -0.05]
WORKSPACE_MAX = [0.65, 0.35, 0.65]
VOXEL_SIZE_M = 0.01

GATES = {
    # occupied ESDF voxel whose center is within (sphere_r + this) of any
    # robot sphere center => robot leaked into the map => FAIL.
    "robot_clearance_margin_m": 0.01,
    # gantry "present": >= this many occupied voxels inside the gantry box
    # dilated by 1 voxel (top-face shell alone is ~80 voxels).
    "gantry_min_occupied_voxels": 40,
    # gantry "cleared": <= this many occupied voxels remain after removal.
    "gantry_max_stale_voxels": 5,
    "map_clear_timeout_s": 90.0,
    "map_present_timeout_s": 90.0,
    # phase D re-route gates (M7 values).
    "dev_min_tcp_m": 0.020,
    "near_straight_tcp_m": 0.050,
    "contrast_ratio_min": 2.0,
}


def endpoint_ok(checker, sphere_model, q) -> bool:
    return checker.check_config(q).ok and sphere_model.valid(
        q, margin=SPHERE_MARGIN_M)


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    write_no_gantry_cell()  # keeps the M7-owned contrast model current

    rng = np.random.default_rng(SEED)
    kin = OpenJawKin(str(URDF))  # jaws at the sim state for ALL mesh checks
    checker = cc.CollisionCore(kin, cell_geometry_yaml=str(CELL))
    checker_ng = cc.CollisionCore(kin,
                                  cell_geometry_yaml=str(CELL_NO_GANTRY))
    xrdf = yaml.safe_load(XRDF.read_text())
    world_boxes = load_world_boxes()
    world_table_only = [b for b in world_boxes if b["name"] == "table"]
    sphere_model = SphereModel(kin, xrdf, world_boxes)
    lo = kin.lower + LIMIT_MARGIN_RAD
    hi = kin.upper - LIMIT_MARGIN_RAD

    assert endpoint_ok(checker, sphere_model, START_Q_ARM), \
        "bridge start pose fails the endpoint checks"

    # ---- map-check configs ------------------------------------------------
    configs = []
    tcps = [kin.fk_tcp(START_Q_ARM)[:3, 3]]
    tries = 0
    while len(configs) < N_MAP_CONFIGS:
        tries += 1
        if tries > CONFIG_TRIES * N_MAP_CONFIGS:
            raise RuntimeError("map-config sampling stalled")
        q = rng.uniform(lo, hi)
        tcp = kin.fk_tcp(q)[:3, 3]
        if np.any(tcp < MAP_TCP_LO) or np.any(tcp > MAP_TCP_HI):
            continue
        if any(np.linalg.norm(tcp - p) < MIN_TCP_SEP_M for p in tcps):
            continue
        if not endpoint_ok(checker, sphere_model, q):
            continue
        ws = sphere_model.world_spheres(q)
        spheres = []
        for ln in sphere_model.links:
            for c, r in zip(ws[ln], sphere_model.radii[ln]):
                spheres.append([round(float(v), 4) for v in c]
                               + [round(float(r), 4)])
        configs.append({
            "index": len(configs) + 1,
            "q": [round(float(v), 5) for v in q],
            "tcp": [round(float(v), 4) for v in tcp],
            "spheres_xyzr": spheres,
        })
        tcps.append(tcp)
    print(f"map configs: {len(configs)} sampled in {tries} tries; TCPs:")
    for c in configs:
        print(f"   #{c['index']}: tcp={c['tcp']}")

    # ---- gantry-blocked pair(s), M7 procedure -----------------------------
    def sample_endpoint(pos_lo, pos_hi):
        for _ in range(ENDPOINT_TRIES):
            q = rng.uniform(lo, hi)
            tcp = kin.fk_tcp(q)[:3, 3]
            if np.any(tcp < pos_lo) or np.any(tcp > pos_hi):
                continue
            if endpoint_ok(checker, sphere_model, q):
                return q, tcp
        raise RuntimeError("endpoint sampling stalled")

    pairs = []
    while len(pairs) < N_PAIRS:
        for _ in range(PAIR_TRIES):
            qa, tcp_a = sample_endpoint(PICK_LO, PICK_HI)
            qb, tcp_b = sample_endpoint(PLACE_LO, PLACE_HI)
            if np.linalg.norm(tcp_b - tcp_a) < MIN_TCP_STEP_M:
                continue
            st = chord_gantry_stats(checker, checker_ng, qa, qb,
                                    step=CHORD_STEP_RAD)
            if (st["n_gantry_hits"] >= MIN_GANTRY_HITS
                    and st["clean_without_gantry"]):
                pairs.append({
                    "index": len(pairs) + 1,
                    "A": {"q": [round(float(v), 5) for v in qa],
                          "tcp": [round(float(v), 4) for v in tcp_a]},
                    "B": {"q": [round(float(v), 5) for v in qb],
                          "tcp": [round(float(v), 4) for v in tcp_b]},
                    "chord_stats": st,
                })
                print(f"pair {len(pairs)}: gantry hits "
                      f"{st['n_gantry_hits']}, clean without gantry")
                break
        else:
            raise RuntimeError("pair sampling stalled")

    gantry = next(b for b in world_boxes if b["name"] == "gantry")
    doc = {
        "milestone": "M8",
        "artifact": "acceptance_configs",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "jaw_open_m": JAW_OPEN_M,
        "start_q_arm": START_Q_ARM,
        "n_map_required": N_MAP_REQUIRED,
        "map_configs": configs,
        "pairs": pairs,
        "world_boxes": world_boxes,
        "world_boxes_table_only": world_table_only,
        "gantry_box": {"center": list(gantry["center"]),
                       "size": list(gantry["size"])},
        "workspace": {"min": WORKSPACE_MIN, "max": WORKSPACE_MAX,
                      "voxel_size_m": VOXEL_SIZE_M},
        "gates": GATES,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
