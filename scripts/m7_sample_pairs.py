#!/usr/bin/env python3
"""M7: choose pose PAIRS whose straight-line joint path hits the gantry.

Design doc M7 gate: "static gantry collision avoidance -- visible
re-routing works".  This sampler produces the demonstration inputs:
pick-zone <-> place-zone transits for which the straight-line JOINT-space
path (linear interpolation qA -> qB) PROVABLY intersects the gantry
crossbar and nothing else, so any collision-free plan between the two
configurations must visibly deviate from the straight line.

Every pair (qA, qB) satisfies, verified with rebot_planner's mesh-based
collision_core (canonical URDF collision meshes, vendor SRDF pairs,
cell_geometry.yaml table at true z=0 + gantry):

  * both endpoints are collision-free under the TRUE cell model
    (meshes + table + gantry) AND under cuMotion's own sphere model
    (XRDF spheres + buffers vs the M6 trial world: table dropped 10 mm
    + gantry) -- so cuMotion accepts both as start/goal states;
  * both endpoints are reachable: FK-generated and re-verified with
    ik_core from a perturbed seed (M6 convention);
  * TCP(A) is on the pick side (y > 0) and TCP(B) on the place side
    (y < 0) of the cell, or vice versa (alternating), >= MIN_TCP_STEP_M
    apart -- i.e. a genuine under/around-the-gantry transit;
  * the straight-line joint chord qA -> qB, interpolated at
    <= CHORD_STEP_RAD per joint, COLLIDES with the gantry on at least
    MIN_GANTRY_HITS consecutive-resolution configurations, and is
    collision-free in an identical model WITHOUT the gantry (written to
    artifacts/m7/cell_geometry_no_gantry.yaml) -- so the gantry is
    provably the ONLY obstruction on the straight line.

The M6 sphere model, cuMotion world boxes (table dropped 10 mm) and
start pose are imported UNCHANGED from scripts/m6_sample_poses.py.

JAW STATE (measured M7 lesson, first gate run): every mesh check here
runs with the gripper jaws at their ACTUAL sim state -- locked OPEN at
0.0715 m (bridge seed, XRDF locked value) -- via :class:`OpenJawKin`.
``ik_core.full_q`` pins the jaws at 0 (shut), a configuration that never
exists during the trials; checking meshes there produced BOTH a phantom
1.4 mm fingertip graze on a properly clearing path AND a chord "hit"
through where the shut fingers would have been.  Jaw state does not
affect arm FK/IK (the finger joints are leaves under gripper_link), so
only the collision volumes change.

Output: artifacts/m7/pair_poses.json (pairs + both world-box sets, the
single source for scripts/m7_trial_runner.py) and
artifacts/m7/cell_geometry_no_gantry.yaml (gantry-removed cell model for
the runner's contrast world and the host verifier).

Run: ~/isaaclab-venv/bin/python scripts/m7_sample_pairs.py
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
from rebot_planner.core.ik_core import (  # noqa: E402
    IK_ERR_ACCEPT, KinematicsCore)

# M6 infrastructure, unchanged (sphere model, cuMotion world, start pose).
from m6_sample_poses import (  # noqa: E402
    JAW_OPEN_M, START_Q_ARM, SphereModel, load_world_boxes,
    rot_to_quat_xyzw)

URDF = REPO / "urdf" / "rebot_b601dm_canonical.urdf"
XRDF = REPO / "config" / "rebot_b601dm.xrdf"
CELL = REPO / "ros2_ws" / "src" / "rebot_planner" / "config" / \
    "cell_geometry.yaml"
ART = REPO / "artifacts" / "m7"
OUT = ART / "pair_poses.json"
CELL_NO_GANTRY = ART / "cell_geometry_no_gantry.yaml"

SEED = 70100
N_PAIRS_REQUIRED = 10         # scored with-gantry trials
N_PAIRS = 16                  # + spares for cuMotion-side rejections
N_CONTRAST = 3                # pairs re-run with the gantry REMOVED
LIMIT_MARGIN_RAD = 0.15       # M6 value: keep off the hard joint limits
SPHERE_MARGIN_M = 0.005       # M6 value: beyond the XRDF buffers
MIN_TCP_STEP_M = 0.15         # a pair must be a real transit
CHORD_STEP_RAD = 0.03         # straight-line interpolation resolution
MIN_GANTRY_HITS = 3           # >= 3 colliding configs: no epsilon graze
ENDPOINT_TRIES = 4000         # per-endpoint rejection budget
PAIR_TRIES = 400              # pair attempts per accepted pair

# TCP endpoint boxes (base_link frame).  The gantry bar spans
# x 0.35..0.55 at y 0.08..0.12, z 0.20..0.24 (cell_geometry.yaml); a
# pick(y+)->place(y-) transit with TCP radius/height near the bar sweeps
# the wrist/gripper straight through it.  z range chosen around the bar
# height; x range inside the M6 workspace box (x <= 0.48).
PICK_LO = np.array([0.28, 0.06, 0.13])
PICK_HI = np.array([0.48, 0.28, 0.33])
PLACE_LO = np.array([0.28, -0.28, 0.13])
PLACE_HI = np.array([0.48, -0.06, 0.33])


class OpenJawKin(KinematicsCore):
    """KinematicsCore whose collision configurations carry the jaws at
    the sim's locked-OPEN value (0.0715 m) instead of ik_core's shut
    default.  fk_tcp/solve_tcp are unaffected (finger joints are leaf
    joints; the TCP frame hangs off gripper_link)."""

    def full_q(self, q6):
        q = super().full_q(q6)
        q[6:8] = JAW_OPEN_M
        return q


def write_no_gantry_cell() -> None:
    """cell_geometry.yaml minus the gantry -> artifacts/m7 (single source
    of what 'gantry removed' means for the runner AND the verifier)."""
    doc = yaml.safe_load(CELL.read_text())
    obstacles = dict(doc.get("obstacles") or {})
    if "gantry" not in obstacles:
        raise AssertionError(f"no gantry obstacle in {CELL}")
    del obstacles["gantry"]
    out = {
        "frame": doc.get("frame", "base_link"),
        "obstacles": obstacles,
        "zones": doc.get("zones") or {},
    }
    CELL_NO_GANTRY.write_text(
        "# GENERATED by scripts/m7_sample_pairs.py: config/cell_geometry"
        ".yaml\n# with the gantry obstacle REMOVED (M7 contrast model).  "
        "Do not hand-edit.\n" + yaml.safe_dump(out, sort_keys=False))


def chord_configs(qa: np.ndarray, qb: np.ndarray,
                  step: float = CHORD_STEP_RAD) -> np.ndarray:
    """Linear joint-space interpolation, <= step rad per joint per config."""
    n = max(1, int(np.ceil(float(np.max(np.abs(qb - qa))) / step)))
    return np.array([qa + (qb - qa) * (j / n) for j in range(n + 1)])


def chord_gantry_stats(checker_full: cc.CollisionCore,
                       checker_no_gantry: cc.CollisionCore,
                       qa, qb, step: float = CHORD_STEP_RAD) -> dict:
    """Straight-line chord analysis against BOTH mesh models.

    Returns hits (configs colliding in the full model), clean_without
    (True iff the chord never collides once the gantry is removed), the
    colliding link pairs seen, and the hit span in rad (max joint metric).
    """
    qa = np.asarray(qa, dtype=np.float64)
    qb = np.asarray(qb, dtype=np.float64)
    qs = chord_configs(qa, qb, step)
    hit_idx: list[int] = []
    pairs: set[tuple[str, str]] = set()
    clean_without = True
    for i, q in enumerate(qs):
        if checker_no_gantry.in_collision(q):
            clean_without = False
            break
        if checker_full.in_collision(q):
            hit_idx.append(i)
            for p in checker_full.check_config(q).pairs:
                pairs.add(tuple(p))
    span_rad = 0.0
    if hit_idx:
        per_cfg = float(np.max(np.abs(qb - qa))) / (len(qs) - 1) \
            if len(qs) > 1 else 0.0
        span_rad = (hit_idx[-1] - hit_idx[0] + 1) * per_cfg
    return {
        "n_configs": len(qs),
        "n_gantry_hits": len(hit_idx),
        "hit_span_rad": round(span_rad, 4),
        "clean_without_gantry": clean_without,
        "colliding_pairs": sorted(sorted(p) for p in pairs),
    }


class EndpointSampler:
    """Rejection-sample one valid arm config with TCP inside a box."""

    def __init__(self, rng, kin: KinematicsCore,
                 checker_full: cc.CollisionCore,
                 sphere_model: SphereModel) -> None:
        self.rng = rng
        self.kin = kin
        self.checker = checker_full
        self.spheres = sphere_model
        self.lo = kin.lower + LIMIT_MARGIN_RAD
        self.hi = kin.upper - LIMIT_MARGIN_RAD
        self.stats = {"attempts": 0, "workspace": 0, "mesh": 0,
                      "sphere": 0, "ik": 0, "accepted": 0}

    def sample(self, pos_lo, pos_hi):
        """-> (q6, T_tcp) or None after ENDPOINT_TRIES rejections."""
        for _ in range(ENDPOINT_TRIES):
            self.stats["attempts"] += 1
            q = self.rng.uniform(self.lo, self.hi)
            T = self.kin.fk_tcp(q)
            pos = T[:3, 3]
            if np.any(pos < pos_lo) or np.any(pos > pos_hi):
                self.stats["workspace"] += 1
                continue
            if not self.checker.check_config(q).ok:
                self.stats["mesh"] += 1
                continue
            if not self.spheres.valid(q, margin=SPHERE_MARGIN_M):
                self.stats["sphere"] += 1
                continue
            q_seed = np.clip(q + self.rng.normal(0.0, 0.1, 6),
                             self.lo, self.hi)
            q_sol, res = self.kin.solve_tcp(T, q_seed)
            if res > IK_ERR_ACCEPT or not self.kin.within_limits(q_sol):
                self.stats["ik"] += 1
                continue
            self.stats["accepted"] += 1
            return q, T
        return None


def endpoint_record(q, T) -> dict:
    return {
        "q": [round(float(v), 5) for v in q],
        "position": [round(float(v), 6) for v in T[:3, 3]],
        "quat_xyzw": [round(v, 8) for v in rot_to_quat_xyzw(T[:3, :3])],
    }


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    write_no_gantry_cell()

    rng = np.random.default_rng(SEED)
    kin = OpenJawKin(str(URDF))
    checker_full = cc.CollisionCore(kin, cell_geometry_yaml=str(CELL))
    checker_ng = cc.CollisionCore(
        kin, cell_geometry_yaml=str(CELL_NO_GANTRY))
    xrdf = yaml.safe_load(XRDF.read_text())
    world_boxes = load_world_boxes()          # M6: table dropped 10 mm
    world_boxes_ng = [b for b in world_boxes if b["name"] != "gantry"]
    sphere_model = SphereModel(kin, xrdf, world_boxes)

    if not checker_full.check_config(START_Q_ARM).ok:
        raise AssertionError("bridge start pose fails the mesh checker")
    if not sphere_model.valid(START_Q_ARM):
        raise AssertionError("bridge start pose fails the sphere model")

    sampler = EndpointSampler(rng, kin, checker_full, sphere_model)
    pairs = []
    chord_stats = {"pair_attempts": 0, "endpoint_fail": 0,
                   "step_reject": 0, "no_gantry_hit": 0,
                   "not_clean_without": 0, "too_few_hits": 0}
    t0 = time.monotonic()
    while len(pairs) < N_PAIRS:
        # alternate transit direction: even pair index picks -> places.
        forward = (len(pairs) % 2 == 0)
        box_a = (PICK_LO, PICK_HI) if forward else (PLACE_LO, PLACE_HI)
        box_b = (PLACE_LO, PLACE_HI) if forward else (PICK_LO, PICK_HI)
        accepted = False
        for _ in range(PAIR_TRIES):
            chord_stats["pair_attempts"] += 1
            got_a = sampler.sample(*box_a)
            got_b = sampler.sample(*box_b)
            if got_a is None or got_b is None:
                chord_stats["endpoint_fail"] += 1
                continue
            (qa, Ta), (qb, Tb) = got_a, got_b
            if np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) < MIN_TCP_STEP_M:
                chord_stats["step_reject"] += 1
                continue
            st = chord_gantry_stats(checker_full, checker_ng, qa, qb)
            if not st["clean_without_gantry"]:
                chord_stats["not_clean_without"] += 1
                continue
            if st["n_gantry_hits"] == 0:
                chord_stats["no_gantry_hit"] += 1
                continue
            if st["n_gantry_hits"] < MIN_GANTRY_HITS:
                chord_stats["too_few_hits"] += 1
                continue
            pairs.append({
                "index": len(pairs) + 1,
                "direction": "pick_to_place" if forward
                             else "place_to_pick",
                "A": endpoint_record(qa, Ta),
                "B": endpoint_record(qb, Tb),
                "tcp_step_m": round(
                    float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), 4),
                "straight_line": st,
            })
            print(f"pair {len(pairs):2d} [{pairs[-1]['direction']}]: "
                  f"{st['n_gantry_hits']} gantry hits over "
                  f"{st['hit_span_rad']} rad, pairs={st['colliding_pairs']}")
            accepted = True
            break
        if not accepted:
            raise RuntimeError(
                f"pair sampling stalled after {PAIR_TRIES} attempts for "
                f"pair {len(pairs) + 1}: {chord_stats} / {sampler.stats}")

    OUT.write_text(json.dumps({
        "milestone": "M7",
        "artifact": "pair_poses",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "n_pairs": N_PAIRS,
        "n_pairs_required": N_PAIRS_REQUIRED,
        "n_contrast_required": N_CONTRAST,
        "start_q": START_Q_ARM,
        "tool_frame": "gripper_tcp",
        "base_frame": "base_link",
        "chord_step_rad": CHORD_STEP_RAD,
        "min_gantry_hits": MIN_GANTRY_HITS,
        "jaw_open_m": JAW_OPEN_M,
        "mesh_check_jaw_m": JAW_OPEN_M,
        "world_boxes": world_boxes,
        "world_boxes_no_gantry": world_boxes_ng,
        "sampling_stats": {"endpoints": sampler.stats,
                           "chords": chord_stats,
                           "wall_s": round(time.monotonic() - t0, 1)},
        "pairs": pairs,
    }, indent=2) + "\n")
    print(f"wrote {OUT} ({len(pairs)} pairs) in "
          f"{time.monotonic() - t0:.1f}s; endpoints={sampler.stats} "
          f"chords={chord_stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
