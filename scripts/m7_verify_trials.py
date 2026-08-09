#!/usr/bin/env python3
"""M7: independent verification + metrics -> gantry_avoidance.json.

Host-side (isaaclab-venv python), same role as scripts/m6_verify_trials.py
in the M6 gate.  For every SCORED trial in artifacts/m7/trials_raw.json:

  * COLLISION-FREE: the planned trajectory is re-checked against the TRUE
    cell model -- canonical URDF collision MESHES (pinocchio/hppfcl, full
    vendor-SRDF active pairs) + cell_geometry.yaml table at z=0 + gantry
    -- densely interpolated (<= 0.05 rad).  Different engine and stricter
    world than cuMotion's sphere model (M6 convention).  All mesh checks
    run with the jaws at their ACTUAL sim state, locked OPEN at 0.0715 m
    (OpenJawKin; see the m7_sample_pairs docstring for the measured
    phantom-graze failure that shut-jaw checking produced).
  * STRAIGHT LINE COLLIDES: the straight joint chord is re-proven to hit
    the gantry and ONLY the gantry, twice: (a) for the sampled reference
    pair (qA_ref -> qB_ref) and (b) for the ACTUAL executed trajectory
    endpoints (first -> last trajectory point) -- so the re-routing claim
    holds for the motion that really ran, not just the sampled one.
  * GOAL REACHED: last trajectory point within GOAL_TOL_RAD of qB_ref
    per joint (cspace goal), plus the runner's execution/convergence
    verdicts (FJT SUCCESSFUL, <= 0.02 rad tracking).
  * MAX DEVIATION: max over trajectory points of (i) joint-space distance
    to the straight chord segment [q_first, q_last] (L2, rad) and
    (ii) TCP distance to the chord's TCP polyline (m).  Gate: TCP
    deviation >= DEV_MIN_TCP_M -- the path measurably re-routes.
  * GANTRY CLEARANCE: min mesh distance (hppfcl) between any robot
    collision geometry and the gantry box over the densely interpolated
    planned path.  Gate: > 0 (reported per trial).

CONTRAST trials (gantry removed from the planner world) are checked
against the gantry-REMOVED mesh model (artifacts/m7/
cell_geometry_no_gantry.yaml), and gated NEAR-STRAIGHT: TCP deviation
<= NEAR_STRAIGHT_TCP_M and at most 1/CONTRAST_RATIO_MIN of the same
pair's with-gantry deviation.  Whether the direct plan actually sweeps
through the gantry volume is reported as ``crosses_gantry_volume``
(expected true -- that is the visible-re-routing evidence).

Aggregates everything (incl. peak VRAM) into
artifacts/m7/gantry_avoidance.json.  Exit 0 iff the M7 gate passes.

Run: ~/isaaclab-venv/bin/python scripts/m7_verify_trials.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "rebot_planner"))
sys.path.insert(0, str(REPO / "scripts"))

from rebot_planner.core import collision_core as cc  # noqa: E402

from m7_sample_pairs import (  # noqa: E402
    JAW_OPEN_M, OpenJawKin, chord_gantry_stats)

URDF = REPO / "urdf" / "rebot_b601dm_canonical.urdf"
CELL = REPO / "ros2_ws" / "src" / "rebot_planner" / "config" / \
    "cell_geometry.yaml"
ART = REPO / "artifacts" / "m7"
CELL_NO_GANTRY = ART / "cell_geometry_no_gantry.yaml"
PAIRS = ART / "pair_poses.json"
RAW = ART / "trials_raw.json"
OUT = ART / "gantry_avoidance.json"

N_REQUIRED = 10
N_CONTRAST = 3
GOAL_TOL_RAD = 0.01           # cspace goal: last point vs qB_ref
DEV_MIN_TCP_M = 0.020         # with-gantry: measurable re-route
NEAR_STRAIGHT_TCP_M = 0.050   # contrast: near-straight bound
CONTRAST_RATIO_MIN = 2.0      # with-gantry dev >= 2x contrast dev
PATH_STEP_RAD = 0.05          # mesh recheck resolution (M6 value)
CHORD_STEP_RAD = 0.02         # chord re-proof resolution (finer than
                              # the sampler's 0.03)
CHORD_POLYLINE_N = 200        # TCP polyline samples of the chord


def densify(waypoints: np.ndarray, step: float) -> np.ndarray:
    """<= step rad per joint between consecutive returned configs."""
    out = [waypoints[0]]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        n = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / step)))
        for j in range(1, n + 1):
            out.append(a + (b - a) * (j / n))
    return np.asarray(out)


def point_segment_dist(p: np.ndarray, a: np.ndarray,
                       b: np.ndarray) -> float:
    """Distance from p to segment [a, b] (any dimension)."""
    ab = b - a
    denom = float(ab @ ab)
    s = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom,
                                                0.0, 1.0))
    return float(np.linalg.norm(p - (a + s * ab)))


class Metrics:
    """Deviation + clearance metrics on the canonical mesh model."""

    def __init__(self) -> None:
        # OpenJawKin: mesh checks at the sim's ACTUAL jaw state (locked
        # open, 0.0715 m).  ik_core's shut-jaw default checks a
        # configuration that never existed during the trials -- measured
        # consequence in the first M7 gate run: a phantom 1.4 mm
        # fingertip graze on a path that cleared the gantry by >5 mm at
        # the real jaw state.
        self.kin = OpenJawKin(str(URDF))
        self.full = cc.CollisionCore(self.kin, cell_geometry_yaml=str(CELL))
        self.ng = cc.CollisionCore(
            self.kin, cell_geometry_yaml=str(CELL_NO_GANTRY))
        gm = self.full.geom_model
        self.gantry_pairs = [
            i for i, p in enumerate(gm.collisionPairs)
            if "world/gantry" in (gm.geometryObjects[p.first].name,
                                  gm.geometryObjects[p.second].name)]
        if not self.gantry_pairs:
            raise AssertionError("no robot-vs-gantry collision pairs")

    def deviation(self, traj: np.ndarray) -> dict:
        """Max deviation of traj from its own straight chord."""
        q0, qf = traj[0], traj[-1]
        chord = np.array([q0 + (qf - q0) * (j / (CHORD_POLYLINE_N - 1))
                          for j in range(CHORD_POLYLINE_N)])
        tcp_chord = np.array([self.kin.fk_tcp(q)[:3, 3] for q in chord])
        max_joint = 0.0
        max_tcp = 0.0
        for q in traj:
            max_joint = max(max_joint, point_segment_dist(q, q0, qf))
            p = self.kin.fk_tcp(q)[:3, 3]
            d = min(point_segment_dist(p, tcp_chord[i], tcp_chord[i + 1])
                    for i in range(len(tcp_chord) - 1))
            max_tcp = max(max_tcp, d)
        return {"max_joint_dev_rad": round(max_joint, 4),
                "max_tcp_dev_m": round(max_tcp, 4)}

    def min_gantry_clearance(self, traj: np.ndarray) -> float:
        """Min mesh distance robot vs gantry over the densified path."""
        kin, geom = self.kin, self.full
        d_min = float("inf")
        for q6 in densify(traj, PATH_STEP_RAD):
            q = kin.full_q(q6)
            pin.updateGeometryPlacements(kin.model, kin.data,
                                         geom.geom_model, geom.geom_data, q)
            for idx in self.gantry_pairs:
                res = pin.computeDistance(geom.geom_model, geom.geom_data,
                                          idx)
                d_min = min(d_min, float(res.min_distance))
        return d_min

    def crosses_gantry(self, traj: np.ndarray) -> int:
        """# densified configs whose ONLY collision is the gantry."""
        n = 0
        for q in densify(traj, PATH_STEP_RAD):
            if self.full.in_collision(q) and not self.ng.in_collision(q):
                n += 1
        return n


def chord_proof(m: Metrics, qa, qb) -> dict:
    st = chord_gantry_stats(m.full, m.ng, qa, qb, step=CHORD_STEP_RAD)
    st["hits_only_gantry"] = bool(st["n_gantry_hits"] > 0
                                  and st["clean_without_gantry"])
    return st


def agg(xs):
    if not xs:
        return None
    return {"min": round(float(np.min(xs)), 4),
            "mean": round(float(np.mean(xs)), 4),
            "max": round(float(np.max(xs)), 4)}


def main() -> int:
    raw = json.loads(RAW.read_text())
    pairs_doc = json.loads(PAIRS.read_text())
    pair_by_index = {p["index"]: p for p in pairs_doc["pairs"]}
    m = Metrics()

    scored_out = []
    stats = {"planning_time_s": [], "duration_s": [], "tracking_err_rad": [],
             "max_joint_dev_rad": [], "max_tcp_dev_m": [],
             "min_gantry_clearance_m": [], "goal_err_rad": []}
    for trial in raw["trials"]:
        idx = trial["pair_index"]
        pair = pair_by_index[idx]
        s = trial.get("scored") or {}
        v = {"pair_index": idx,
             "direction": pair["direction"],
             "planned": bool(s.get("planned")),
             "executed": bool(s.get("executed")),
             "converged": bool(s.get("converged")),
             "tracking_err_rad": s.get("tracking_err_rad"),
             "planning_time_s": s.get("planning_time_s"),
             "duration_s": s.get("duration_s")}
        if s.get("error"):
            v["error"] = s["error"]
        traj_pts = s.get("trajectory") or []
        if traj_pts:
            traj = np.array([p["q"] for p in traj_pts])
            v["n_points"] = len(traj)
            check = m.full.check_path([list(q) for q in traj],
                                      max_step_rad=PATH_STEP_RAD)
            v["collision_free"] = bool(check.ok)
            if not check.ok:
                v["collision_detail"] = {
                    "segment": check.failed_segment,
                    "pairs": [list(p) for p in check.pairs]}
            goal_err = float(np.max(np.abs(traj[-1]
                                           - np.array(pair["B"]["q"]))))
            v["goal_err_rad"] = round(goal_err, 5)
            v["goal_reached"] = goal_err <= GOAL_TOL_RAD
            v["straight_line_ref"] = chord_proof(
                m, pair["A"]["q"], pair["B"]["q"])
            v["straight_line_actual"] = chord_proof(m, traj[0], traj[-1])
            v.update(m.deviation(traj))
            clr = m.min_gantry_clearance(traj)
            v["min_gantry_clearance_m"] = round(clr, 4)
            v["deviation_measurable"] = v["max_tcp_dev_m"] >= DEV_MIN_TCP_M
            v["passed"] = all((
                v["planned"], v["executed"], v["converged"],
                v["collision_free"], v["goal_reached"],
                v["straight_line_ref"]["hits_only_gantry"],
                v["straight_line_actual"]["hits_only_gantry"],
                v["deviation_measurable"], clr > 0.0))
            if v["passed"]:
                stats["planning_time_s"].append(s["planning_time_s"])
                stats["duration_s"].append(s["duration_s"])
                stats["tracking_err_rad"].append(s["tracking_err_rad"])
                stats["max_joint_dev_rad"].append(v["max_joint_dev_rad"])
                stats["max_tcp_dev_m"].append(v["max_tcp_dev_m"])
                stats["min_gantry_clearance_m"].append(clr)
                stats["goal_err_rad"].append(goal_err)
        else:
            v["collision_free"] = False
            v["passed"] = False
        scored_out.append(v)

    contrast_out = []
    cstats = {"max_joint_dev_rad": [], "max_tcp_dev_m": [],
              "dev_ratio": []}
    scored_by_index = {t["pair_index"]: t for t in scored_out}
    for trial in raw.get("contrast_trials", []):
        idx = trial["pair_index"]
        s = trial.get("scored") or {}
        v = {"pair_index": idx,
             "world": "gantry REMOVED",
             "planned": bool(s.get("planned")),
             "executed": bool(s.get("executed")),
             "converged": bool(s.get("converged")),
             "tracking_err_rad": s.get("tracking_err_rad"),
             "duration_s": s.get("duration_s")}
        if s.get("error"):
            v["error"] = s["error"]
        traj_pts = s.get("trajectory") or []
        if traj_pts:
            traj = np.array([p["q"] for p in traj_pts])
            v["n_points"] = len(traj)
            check = m.ng.check_path([list(q) for q in traj],
                                    max_step_rad=PATH_STEP_RAD)
            v["collision_free_no_gantry"] = bool(check.ok)
            v.update(m.deviation(traj))
            n_cross = m.crosses_gantry(traj)
            v["crosses_gantry_volume"] = n_cross > 0
            v["n_configs_inside_gantry"] = n_cross
            if n_cross == 0:
                v["min_gantry_clearance_m"] = round(
                    m.min_gantry_clearance(traj), 4)
            paired = scored_by_index.get(idx, {})
            ref_dev = paired.get("max_tcp_dev_m")
            ratio = (ref_dev / v["max_tcp_dev_m"]
                     if ref_dev and v["max_tcp_dev_m"] > 1e-6
                     else float("inf"))
            v["with_gantry_max_tcp_dev_m"] = ref_dev
            v["dev_ratio_with_over_without"] = (
                round(ratio, 2) if np.isfinite(ratio) else None)
            v["near_straight"] = v["max_tcp_dev_m"] <= NEAR_STRAIGHT_TCP_M
            v["passed"] = all((
                v["planned"], v["executed"], v["converged"],
                v["collision_free_no_gantry"], v["near_straight"],
                ratio >= CONTRAST_RATIO_MIN))
            if v["passed"]:
                cstats["max_joint_dev_rad"].append(v["max_joint_dev_rad"])
                cstats["max_tcp_dev_m"].append(v["max_tcp_dev_m"])
                if np.isfinite(ratio):
                    cstats["dev_ratio"].append(ratio)
        else:
            v["collision_free_no_gantry"] = False
            v["passed"] = False
        contrast_out.append(v)

    n_required = int(raw.get("n_trials_required", N_REQUIRED))
    n_contrast = int(raw.get("n_contrast_required", N_CONTRAST))
    n_pass = sum(1 for t in scored_out if t.get("passed"))
    n_cpass = sum(1 for t in contrast_out if t.get("passed"))
    gate = (len(scored_out) == n_required and n_pass == n_required
            and len(contrast_out) == n_contrast and n_cpass == n_contrast)

    vram = {}
    baseline_f = ART / "vram_baseline.txt"
    samples_f = ART / "vram_samples.log"
    if baseline_f.is_file() and samples_f.is_file():
        samples = [int(x) for x in samples_f.read_text().split()
                   if x.strip()]
        baseline = int(baseline_f.read_text().split()[0])
        vram = {
            "baseline_before_stack_mib": baseline,
            "peak_during_run_mib": max(samples) if samples else None,
            "stack_delta_mib": (max(samples) - baseline) if samples else None,
            "samples": len(samples),
            "note": "nvidia-smi memory.used, whole GPU; stack = Isaac Sim "
                    "bridge + ROS containers + cuMotion",
        }

    report = {
        "milestone": "M7",
        "gate": "static gantry collision avoidance -- visible re-routing "
                "(design doc sec. 21 M7): 10 pose pairs whose straight "
                "joint path provably hits ONLY the gantry all plan "
                "collision-free, execute, converge and measurably "
                "deviate; 3 gantry-removed re-runs are near-straight",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": gate,
        "n_trials": len(scored_out),
        "n_passed": n_pass,
        "n_contrast": len(contrast_out),
        "n_contrast_passed": n_cpass,
        "n_pairs_rejected": raw.get("n_pairs_rejected", 0),
        "rejected_pairs": raw.get("rejected_pairs", []),
        "criteria": {
            "planned": "cumotion/motion_plan plan_cspace success "
                       "(joint goal = sampled pair endpoint)",
            "collision_free": "mesh recheck (pinocchio/hppfcl, full SRDF "
                              "active pairs, true z=0 table + gantry), "
                              f"<= {PATH_STEP_RAD} rad interpolation, "
                              f"jaws at sim state {JAW_OPEN_M} m open",
            "executed": "FJT SUCCESSFUL through adapter+shim",
            "converged": "tracking error <= 0.02 rad within 5 s (M5 gate)",
            "goal_reached": f"last trajectory point within {GOAL_TOL_RAD} "
                            "rad/joint of the pair's qB",
            "straight_line": "chord (ref AND actual endpoints) collides "
                             "with the gantry and with NOTHING else, "
                             f"mesh model, <= {CHORD_STEP_RAD} rad steps",
            "deviation_measurable": f"max TCP deviation from the straight "
                                    f"chord >= {DEV_MIN_TCP_M} m",
            "gantry_clearance": "min mesh distance to the gantry > 0 "
                                "over the whole planned path",
            "contrast_near_straight": f"gantry-removed max TCP deviation "
                                      f"<= {NEAR_STRAIGHT_TCP_M} m AND "
                                      f">= {CONTRAST_RATIO_MIN}x smaller "
                                      "than the same pair with the gantry",
        },
        "planner_decisions": {
            "planning_backend": "isaac_ros_cumotion 4.5 STANDALONE "
                                "MotionPlan action server, plan_cspace "
                                "joint goals; MoveIt not used (M6 "
                                "decision, unchanged)",
            "cspace_goal_rationale": "the M7 claim is about the straight "
                                     "joint path between two known "
                                     "configurations; a pose goal would "
                                     "let cuMotion's IK choose a "
                                     "different branch and detach the "
                                     "executed motion from the chord "
                                     "proof",
            "static_scene": "table (10 mm-dropped, M6/XRDF convention) + "
                            "gantry embedded per goal; contrast trials "
                            "embed the same world MINUS the gantry",
        },
        "gpu_vram": vram,
        "stats_with_gantry": {k: agg(v) for k, v in stats.items()},
        "stats_contrast": {k: agg(v) for k, v in cstats.items()},
        "trials": scored_out,
        "contrast_trials": contrast_out,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("passed", "n_trials", "n_passed", "n_contrast",
                       "n_contrast_passed", "gpu_vram",
                       "stats_with_gantry", "stats_contrast")}, indent=2))
    print(f"wrote {OUT}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
