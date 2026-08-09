#!/usr/bin/env python3
"""M6: independent verification of the cuMotion trials + gate aggregation.

Host-side (isaaclab-venv python).  For every trial in
artifacts/m6/trials_raw.json:

  * COLLISION-FREE: the full planned trajectory is re-checked against the
    TRUE cell model -- canonical URDF collision MESHES (pinocchio/hppfcl)
    with the vendor SRDF's full active self-pair set, plus the
    cell_geometry.yaml table (top at z=0, NOT the 10 mm-dropped cuMotion
    model) and gantry -- densely interpolated (<= 0.05 rad per step).
    This is deliberately a DIFFERENT collision engine and a stricter
    world model than the planner's own sphere model.
  * TCP ACCURACY: FK of the final trajectory point vs the requested pose
    (<= 20 mm / <= 0.1 rad -- generous hard gate; stats recorded).
  * EXECUTED + CONVERGED: taken from the runner record (FJT success +
    <= 0.02 rad tracking, the M5 gate value).

Aggregates the verdict + peak VRAM into artifacts/m6/trials.json.
Exit 0 iff the M6 gate passes: 100/100 collision-free planned, executed,
converged trials.

Run: ~/isaaclab-venv/bin/python scripts/m6_verify_trials.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "rebot_planner"))

from rebot_planner.core import collision_core as cc  # noqa: E402
from rebot_planner.core.ik_core import (  # noqa: E402
    KinematicsCore, pose_to_transform)

URDF = REPO / "urdf" / "rebot_b601dm_canonical.urdf"
CELL = REPO / "ros2_ws" / "src" / "rebot_planner" / "config" / \
    "cell_geometry.yaml"
RAW = REPO / "artifacts" / "m6" / "trials_raw.json"
ART = REPO / "artifacts" / "m6"
OUT = ART / "trials.json"

TCP_POS_TOL_M = 0.020
TCP_ROT_TOL_RAD = 0.10
N_REQUIRED = 100


def rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    tr = float(np.trace(Ra.T @ Rb))
    return float(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def main() -> int:
    raw = json.loads(RAW.read_text())
    kin = KinematicsCore(str(URDF))
    checker = cc.CollisionCore(kin, cell_geometry_yaml=str(CELL))

    trials_out = []
    stats = {"planning_time_s": [], "duration_s": [], "tracking_err_rad": [],
             "tcp_pos_err_m": [], "tcp_rot_err_rad": [], "n_points": []}
    for rec in raw["trials"]:
        v = {"index": rec["index"],
             "planned": bool(rec.get("planned")),
             "executed": bool(rec.get("executed")),
             "converged": bool(rec.get("converged"))}
        if rec.get("error"):
            v["error"] = rec["error"]
        traj = rec.get("trajectory") or []
        if traj:
            waypoints = [p["q"] for p in traj]
            check = checker.check_path(waypoints, max_step_rad=0.05)
            v["collision_free"] = bool(check.ok)
            v["collision_configs_checked"] = check.checked_configurations
            if not check.ok:
                v["collision_detail"] = {
                    "segment": check.failed_segment,
                    "pairs": [list(p) for p in check.pairs],
                    "q": list(check.q_colliding)}
            T_final = kin.fk_tcp(waypoints[-1])
            T_goal = pose_to_transform(rec["target"]["position"],
                                       rec["target"]["quat_xyzw"])
            pos_err = float(np.linalg.norm(T_final[:3, 3] - T_goal[:3, 3]))
            rot_err = rot_angle(T_final[:3, :3], T_goal[:3, :3])
            v["tcp_pos_err_m"] = round(pos_err, 5)
            v["tcp_rot_err_rad"] = round(rot_err, 5)
            v["tcp_ok"] = (pos_err <= TCP_POS_TOL_M
                           and rot_err <= TCP_ROT_TOL_RAD)
            stats["tcp_pos_err_m"].append(pos_err)
            stats["tcp_rot_err_rad"].append(rot_err)
            stats["n_points"].append(len(traj))
            if rec.get("planning_time_s") is not None:
                stats["planning_time_s"].append(rec["planning_time_s"])
            if rec.get("duration_s") is not None:
                stats["duration_s"].append(rec["duration_s"])
            if rec.get("tracking_err_rad") is not None:
                stats["tracking_err_rad"].append(rec["tracking_err_rad"])
        else:
            v["collision_free"] = False
            v["tcp_ok"] = False
        v["passed"] = all((v["planned"], v["executed"], v["converged"],
                           v["collision_free"], v["tcp_ok"]))
        trials_out.append(v)

    n_required = int(raw.get("n_trials_required", N_REQUIRED))
    n_pass = sum(1 for t in trials_out if t["passed"])
    gate = (len(trials_out) == n_required
            and n_pass == len(trials_out))

    def agg(xs):
        if not xs:
            return None
        return {"min": round(float(np.min(xs)), 4),
                "mean": round(float(np.mean(xs)), 4),
                "max": round(float(np.max(xs)), 4)}

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
        "milestone": "M6",
        "gate": "100 collision-free pose-to-pose cuMotion trials in a "
                "static scene (design doc sec. 13.5 item 7 / sec. 22 M6)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": gate,
        "n_trials": len(trials_out),
        "n_passed": n_pass,
        "n_candidates_rejected": raw.get("n_candidates_rejected", 0),
        "rejected_candidates": [
            {"candidate_index": r.get("candidate_index"),
             "error": r.get("error"),
             "target": r.get("target")}
            for r in raw.get("rejected_candidates", [])],
        "rejection_note": "candidates cuMotion's collision-aware IK could "
                          "not solve; arm never moved, chain intact, spare "
                          "candidate substituted (see m6_trial_runner "
                          "docstring)",
        "criteria": {
            "planned": "cumotion/motion_plan success",
            "collision_free": "mesh recheck (pinocchio/hppfcl, full SRDF "
                              "active pairs, true z=0 table + gantry), "
                              "<=0.05 rad interpolation",
            "executed": "FJT SUCCESSFUL through adapter+shim",
            "converged": f"tracking error <= 0.02 rad within 5 s",
            "tcp_ok": f"final TCP within {TCP_POS_TOL_M} m / "
                      f"{TCP_ROT_TOL_RAD} rad of the requested pose",
        },
        "planner_decisions": {
            "planning_backend": "isaac_ros_cumotion 4.5 STANDALONE "
                                "MotionPlan action server "
                                "(cumotion/motion_plan); MoveIt not used",
            "moveit_plugin_fallback_needed": False,
            "nvidia_enum_fork_needed": False,
            "enum_fork_note": "reference workflow launchers "
                              "(isaac_ros_manipulation_ros_python_utils "
                              "RobotType/GripperType) bypassed entirely by "
                              "driving the action server directly",
            "static_scene": "table+gantry from cell_geometry.yaml embedded "
                            "as CollisionObjects in every MotionPlan goal "
                            "(merged with the empty static-scene server "
                            "set); table top modeled 10 mm low for cuMotion "
                            "(see XRDF header), verified here at true z=0",
        },
        "gpu_vram": vram,
        "stats": {k: agg(v) for k, v in stats.items()},
        "trials": trials_out,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("passed", "n_trials", "n_passed", "gpu_vram",
                       "stats")}, indent=2))
    print(f"wrote {OUT}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
