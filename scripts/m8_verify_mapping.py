#!/usr/bin/env python3
"""M8: independent verification + gates -> mapping_acceptance.json.

Host-side (isaaclab-venv python), same role as m6/m7's verifiers.  Applies
the doc-15.4 mapping gates to artifacts/m8/acceptance_raw.json and
re-checks the phase-D trajectories on the canonical MESH model (the M7
Metrics class unchanged: pinocchio/hppfcl, vendor-SRDF active pairs,
true-z table + gantry, jaws locked open at the sim state):

  MAP  >= n_map_required held configs, every one:
       * move planned/executed/converged (M5/M6 gates);
       * robot ABSENT: zero occupied ESDF voxels inside the config's
         XRDF sphere volume (+1 cm margin);
       * gantry PRESENT: >= gantry_min_occupied_voxels in the dilated bar
         box;
       * segmentation evidence: nonempty robot mask AND masked-out pixels
         in the world-depth stream (the doc's "display" artifacts, saved
         as .npy under artifacts/m8/evidence/ and rendered to PNG here
         when PIL is available).
  D1   cuMotion re-routes around the MAPPED (never statically declared)
       gantry: scored move ok, mesh-collision-free vs the FULL cell,
       max TCP deviation from its own straight chord >= dev_min_tcp_m,
       min mesh clearance to the gantry > 0, and the straight chord of
       the EXECUTED endpoints provably hits only the gantry.
  C    stale map: gantry voxels fell to <= gantry_max_stale_voxels within
       map_clear_timeout_s of the sim-side removal.
  D2   contrast: same static world, cleared map -> near-straight
       (<= near_straight_tcp_m, AND >= contrast_ratio_min smaller than
       D1's deviation), mesh-collision-free vs the gantry-REMOVED cell,
       and the direct path sweeps through the old gantry volume.
  C2   map rebuilt after the gantry returned.

Also aggregates peak whole-GPU VRAM (DDR-001 stage-2 checkpoint).
Exit 0 iff every gate passes.

Run: ~/isaaclab-venv/bin/python scripts/m8_verify_mapping.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros2_ws" / "src" / "rebot_planner"))
sys.path.insert(0, str(REPO / "scripts"))

from m7_verify_trials import Metrics, chord_proof  # noqa: E402

ART = REPO / "artifacts" / "m8"
CFG = ART / "acceptance_configs.json"
RAW = ART / "acceptance_raw.json"
OUT = ART / "mapping_acceptance.json"
EVIDENCE = ART / "evidence"
PATH_STEP_RAD = 0.05


def render_evidence_pngs() -> list[str]:
    """Best-effort .npy -> .png so the report can show the doc's
    raw-depth / robot-mask / masked-depth triples."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        return []
    rendered = []
    for npy in sorted(EVIDENCE.glob("*.npy")):
        arr = np.load(npy)
        if arr.dtype == np.uint8:  # mask
            img = (arr > 0).astype(np.uint8) * 255
        else:  # depth (metres) -> normalized grayscale, invalid = black
            valid = np.isfinite(arr) & (arr > 0)
            img = np.zeros(arr.shape, dtype=np.uint8)
            if valid.any():
                lo, hi = arr[valid].min(), arr[valid].max()
                span = max(hi - lo, 1e-6)
                img[valid] = (255 * (1.0 - (arr[valid] - lo) / span)
                              ).astype(np.uint8)
        png = npy.with_suffix(".png")
        PILImage.fromarray(img).save(png)
        rendered.append(png.name)
    return rendered


def check_scored_move(m: Metrics, scored: dict, world: str) -> dict:
    """Mesh recheck + deviation metrics for a phase-D scored move."""
    v = {"planned": bool(scored.get("planned")),
         "executed": bool(scored.get("executed")),
         "converged": bool(scored.get("converged")),
         "tracking_err_rad": scored.get("tracking_err_rad"),
         "planning_time_s": scored.get("planning_time_s"),
         "duration_s": scored.get("duration_s")}
    pts = scored.get("trajectory") or []
    if not pts:
        v["error"] = "no trajectory"
        return v
    traj = np.array([p["q"] for p in pts])
    v["n_points"] = len(traj)
    checker = m.full if world == "full" else m.ng
    check = checker.check_path([list(q) for q in traj],
                               max_step_rad=PATH_STEP_RAD)
    v["collision_free"] = bool(check.ok)
    if not check.ok:
        v["collision_detail"] = {"segment": check.failed_segment,
                                 "pairs": [list(p) for p in check.pairs]}
    v.update(m.deviation(traj))
    if world == "full":
        v["min_gantry_clearance_m"] = round(m.min_gantry_clearance(traj), 4)
        v["straight_line_actual"] = chord_proof(m, traj[0], traj[-1])
    else:
        n_cross = m.crosses_gantry(traj)
        v["crosses_gantry_volume"] = n_cross > 0
        v["n_configs_inside_gantry"] = n_cross
    return v


def main() -> int:
    cfg = json.loads(CFG.read_text())
    raw = json.loads(RAW.read_text())
    gates = cfg["gates"]
    n_required = int(cfg["n_map_required"])
    m = Metrics()

    failures: list[str] = []

    # ---- map phase ------------------------------------------------------
    map_out = []
    for rec in raw.get("map_configs", []):
        mv = rec.get("move", {})
        img = rec.get("images", {})
        st = rec.get("map", {})
        v = {"index": rec["index"], "tcp": rec.get("tcp"),
             "move_ok": bool(mv.get("planned") and mv.get("executed")
                             and mv.get("converged")),
             "robot_occupied_voxels": st.get("robot_occupied_voxels"),
             "gantry_occupied_voxels": st.get("gantry_occupied_voxels"),
             "n_occupied": st.get("n_occupied"),
             "mask_px": img.get("mask_px"),
             "masked_out_px": img.get("masked_out_px"),
             "robot_absent": st.get("robot_occupied_voxels") == 0,
             "gantry_present": (st.get("gantry_occupied_voxels", 0)
                                >= gates["gantry_min_occupied_voxels"]),
             "segmentation_active": (img.get("mask_px", 0) > 0
                                     and img.get("masked_out_px", 0) > 0)}
        v["passed"] = all((v["move_ok"], v["robot_absent"],
                           v["gantry_present"], v["segmentation_active"]))
        if st.get("worst_robot_voxel"):
            v["worst_robot_voxel"] = st["worst_robot_voxel"]
        map_out.append(v)
    n_map_pass = sum(1 for v in map_out if v["passed"])
    if len(map_out) < n_required:
        failures.append(f"only {len(map_out)} map configs completed "
                        f"(need {n_required})")
    if n_map_pass < len(map_out):
        failures.append(f"map configs passed {n_map_pass}/{len(map_out)}")

    # ---- phase D1 -------------------------------------------------------
    d1 = raw.get("phase_d1") or {}
    d1_out = None
    if d1.get("scored"):
        d1_out = check_scored_move(m, d1["scored"], world="full")
        d1_out["map_gantry_voxels_at_plan"] = (d1.get("map_before") or {}
                                               ).get("gantry_occupied_voxels")
        d1_out["deviation_measurable"] = (
            d1_out.get("max_tcp_dev_m", 0.0) >= gates["dev_min_tcp_m"])
        d1_out["passed"] = all((
            d1_out["planned"], d1_out["executed"], d1_out["converged"],
            d1_out.get("collision_free", False),
            d1_out["deviation_measurable"],
            d1_out.get("min_gantry_clearance_m", -1.0) > 0.0,
            d1_out.get("straight_line_actual", {}).get("hits_only_gantry",
                                                       False),
            (d1_out["map_gantry_voxels_at_plan"] or 0)
            >= gates["gantry_min_occupied_voxels"]))
        if not d1_out["passed"]:
            failures.append("phase D1 (mapped-gantry avoidance) failed")
    else:
        failures.append("phase D1 missing")

    # ---- phase C --------------------------------------------------------
    c1 = raw.get("phase_c_clear") or {}
    if not c1.get("ok"):
        failures.append("phase C (stale-map clear) failed")

    # ---- phase D2 -------------------------------------------------------
    d2 = raw.get("phase_d2") or {}
    d2_out = None
    if d2.get("scored") and d1_out:
        d2_out = check_scored_move(m, d2["scored"], world="no_gantry")
        d2_out["map_gantry_voxels_at_plan"] = (d2.get("map_before") or {}
                                               ).get("gantry_occupied_voxels")
        dev1 = d1_out.get("max_tcp_dev_m") or 0.0
        dev2 = d2_out.get("max_tcp_dev_m") or 0.0
        ratio = dev1 / dev2 if dev2 > 1e-6 else float("inf")
        d2_out["dev_ratio_d1_over_d2"] = (round(ratio, 2)
                                          if np.isfinite(ratio) else None)
        d2_out["near_straight"] = (dev2 <= gates["near_straight_tcp_m"])
        d2_out["passed"] = all((
            d2_out["planned"], d2_out["executed"], d2_out["converged"],
            d2_out.get("collision_free", False),
            d2_out["near_straight"],
            ratio >= gates["contrast_ratio_min"],
            d2_out.get("crosses_gantry_volume", False),
            (d2_out["map_gantry_voxels_at_plan"] or 0)
            <= gates["gantry_max_stale_voxels"]))
        if not d2_out["passed"]:
            failures.append("phase D2 (cleared-map contrast) failed")
    else:
        failures.append("phase D2 missing")

    # ---- phase C2 -------------------------------------------------------
    c2 = raw.get("phase_c2_recover") or {}
    if not c2.get("ok"):
        failures.append("phase C2 (map rebuild) failed")

    if raw.get("hard_failure"):
        failures.append(f"runner hard failure: {raw['hard_failure']}")

    # ---- VRAM (DDR-001 stage-2 checkpoint) ------------------------------
    vram = {}
    baseline_f = ART / "vram_baseline.txt"
    samples_f = ART / "vram_samples.log"
    if baseline_f.is_file() and samples_f.is_file():
        samples = [int(x) for x in samples_f.read_text().split() if x.strip()]
        baseline = int(baseline_f.read_text().split()[0])
        vram = {
            "baseline_before_stack_mib": baseline,
            "peak_during_run_mib": max(samples) if samples else None,
            "stack_delta_mib": (max(samples) - baseline) if samples else None,
            "samples": len(samples),
            "note": "nvidia-smi memory.used, whole GPU; stack = Isaac Sim "
                    "bridge (+overhead RTX camera) + adapter container + "
                    "segmenter/nvblox/cuMotion container",
        }

    rendered = render_evidence_pngs()
    gate = not failures
    report = {
        "milestone": "M8",
        "gate": "robot segmentation + nvblox ESDF mapping feeds cuMotion "
                "(design doc 14-15; acceptance test 15.4): >= 5 arm "
                "configs with the robot body ABSENT from the map and the "
                "gantry PRESENT, stale map clears after the gantry is "
                "removed (and rebuilds on return), and cuMotion visibly "
                "re-routes around the MAPPED (not statically declared) "
                "gantry with a cleared-map contrast plan",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": gate,
        "failures": failures,
        "n_map_configs": len(map_out),
        "n_map_passed": n_map_pass,
        "map_configs": map_out,
        "phase_d1_mapped_gantry": d1_out,
        "phase_c_stale_clear": c1,
        "phase_d2_cleared_contrast": d2_out,
        "phase_c2_rebuild": c2,
        "config_rejections": raw.get("rejected_configs", []),
        "gates": gates,
        "nvblox": {"voxel_size_m": cfg["workspace"]["voxel_size_m"],
                   "workspace": {"min": cfg["workspace"]["min"],
                                 "max": cfg["workspace"]["max"]},
                   "esdf_mode": "3d", "global_frame": "base_link",
                   "depth_source": "robot-masked world_depth "
                                   "(isaac_ros_cumotion_robot_segmenter)"},
        "gpu_vram": vram,
        "evidence_pngs": rendered,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("passed", "failures", "n_map_configs", "n_map_passed",
                       "gpu_vram")}, indent=2))
    if d1_out:
        print("D1 mapped-gantry: dev=%.3fm clearance=%.3fm" % (
            d1_out.get("max_tcp_dev_m", -1),
            d1_out.get("min_gantry_clearance_m", -1)))
    if d2_out:
        print("D2 contrast: dev=%.3fm ratio=%s crosses=%s" % (
            d2_out.get("max_tcp_dev_m", -1),
            d2_out.get("dev_ratio_d1_over_d2"),
            d2_out.get("crosses_gantry_volume")))
    print(f"wrote {OUT}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
