#!/usr/bin/env python3
"""The Aug 8 ten-scene vertical smoke test (PLAN.md 5.2.1).

Runs the complete chain and refuses to declare success on a partial one::

    randomize -> capture -> serialize -> replay -> A1 oracle mask
              -> B predicted mask -> PoseStamped -> scorer + overlay

Every stage is reported with its own status, so a failure names the last stage
that worked rather than collapsing to "it broke".

    python run_smoke.py --backend analytic
    python run_smoke.py --backend isaac      # requires Isaac Sim
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from grasp_smoke import dataset as ds                      # noqa: E402
from grasp_smoke.capture import capture_analytic, plan_smoke_scenes  # noqa: E402
from grasp_smoke.detect import build_predictor             # noqa: E402
from grasp_smoke.grasp import estimate_grasp               # noqa: E402
from grasp_smoke.overlay import render_overlay             # noqa: E402
from grasp_smoke.pose_msg import grasp_to_pose_stamped     # noqa: E402
from grasp_smoke.scorer import score_scene, summarize, write_results  # noqa: E402


def _log(stage: str, msg: str) -> None:
    print(f"[{stage:<9}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["analytic", "isaac"], default="analytic")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "smoke")
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--depth-quantile", type=float, default=0.75)
    ap.add_argument("--reuse-dataset", action="store_true",
                    help="score an existing dataset instead of recapturing")
    args = ap.parse_args()

    out = args.out
    data_root = out / "dataset"
    overlay_dir = out / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    stages = {}
    specs = plan_smoke_scenes(args.seed)

    # ---- 1. randomize + capture ------------------------------------------
    if args.reuse_dataset and (data_root / "manifest.json").exists():
        capture_stats = {"backend": "reused", "n_scenes": len(specs)}
        _log("capture", f"reusing {data_root}")
    elif args.backend == "analytic":
        _log("capture", f"analytic backend, {len(specs)} scenes -> {data_root}")
        capture_stats = capture_analytic(
            data_root, specs, seed=args.seed, depth_quantile=args.depth_quantile
        )
    else:
        _log("capture", "isaac backend: delegating to capture/isaac_capture.py")
        rc = subprocess.call([
            sys.executable, str(REPO / "capture" / "isaac_capture.py"),
            "--out", str(data_root), "--seed", str(args.seed),
        ])
        if rc != 0:
            stages["capture"] = f"FAILED (exit {rc})"
            print(json.dumps({"stages": stages}, indent=2))
            return rc
        capture_stats = json.loads((data_root / "capture_stats.json").read_text())
    stages["randomize+capture"] = "ok"
    stages["serialize"] = "ok"

    # ---- 2. replay integrity ---------------------------------------------
    bad = ds.verify_checksums(data_root)
    if bad:
        stages["replay"] = f"FAILED: {len(bad)} checksum mismatches"
        print(json.dumps({"stages": stages, "bad_files": bad}, indent=2))
        return 1
    manifest = ds.load_manifest(data_root)
    stages["replay"] = f"ok ({len(manifest['scenes'])} scenes, checksums verified)"
    _log("replay", f"checksums verified for {len(manifest['scenes'])} scenes")

    # ---- 3. branches ------------------------------------------------------
    predictor = build_predictor()
    _log("branch-B", f"predictor = {predictor.config.name} "
                     f"(provisional={predictor.config.provisional})")

    spec_by_id = {s.scene_id: s for s in specs}
    scores, pose_msgs = [], []
    t_infer = time.perf_counter()

    for scene_id in ds.scene_ids(data_root):
        frame = ds.load_frame(data_root, scene_id)          # sensor data only
        labels = ds.load_labels(data_root, scene_id)        # scorer + oracle only
        tilt = spec_by_id[scene_id].tilt_deg if scene_id in spec_by_id else 0.0

        for branch in ("A1", "B"):
            if branch == "A1":
                mask = labels.gt_mask if labels.target_present else None
                stratum = spec_by_id[scene_id].stratum if scene_id in spec_by_id else "A1"
                branch_key = "A2" if stratum == "A2" else "A1"
            else:
                mask = predictor.predict(frame)             # frame only, no GT
                branch_key = "B"

            est = (
                estimate_grasp(mask, frame.depth_m, frame.K, args.depth_quantile)
                if mask is not None else None
            )
            score = score_scene(
                scene_id, branch_key, est, mask, labels, frame.T_base_cam,
                iou_match_threshold=predictor.config.iou_match_threshold,
                tilt_deg=tilt,
            )
            scores.append(score)

            if est is not None and est.is_valid:
                pose_msgs.append(grasp_to_pose_stamped(est, frame, branch=branch_key))

            cv2.imwrite(
                str(overlay_dir / f"{scene_id}_{branch_key}.png"),
                render_overlay(frame, est, mask, labels, score, branch_key),
            )

    infer_s = time.perf_counter() - t_infer
    stages["A1 oracle mask"] = "ok"
    stages["B predicted mask"] = f"ok ({predictor.config.name})"
    stages["PoseStamped"] = f"ok ({len(pose_msgs)} messages built)"
    stages["scorer+overlay"] = "ok"

    # ---- 4. results -------------------------------------------------------
    summaries = {b: summarize(scores, b) for b in ("A1", "A2", "B")}
    payload = {
        "stages": stages,
        "capture": capture_stats,
        "manifest": {k: manifest[k] for k in
                     ("schema_version", "seed", "depth_policy", "capture_backend",
                      "depth_quantile")},
        "branch_b_config": predictor.config.to_dict(),
        "inference_seconds_total": round(infer_s, 3),
        "inference_seconds_per_scene": round(infer_s / max(len(specs), 1), 4),
        "summaries": {k: v for k, v in summaries.items()},
        "scenes": scores,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "example_pose_stamped": pose_msgs[0] if pose_msgs else None,
    }
    results_path = write_results(out / "results.json", payload)

    # ---- 5. human-readable summary ---------------------------------------
    print("\n" + "=" * 72)
    print("TEN-SCENE SMOKE TEST")
    print("=" * 72)
    for stage, status in stages.items():
        print(f"  {stage:<20} {status}")
    print(f"\ncapture backend      : {capture_stats.get('backend')}")
    if "seconds_per_scene" in capture_stats:
        print(f"  time/scene         : {capture_stats['seconds_per_scene']:.3f} s")
        print(f"  disk/scene         : {capture_stats['bytes_per_scene']/1e6:.2f} MB")
        print(f"  peak VRAM          : {capture_stats['peak_vram_mib']} MiB")
        print(f"  -> 300 scenes      : {capture_stats['extrapolated_300']['minutes']:.1f} min, "
              f"{capture_stats['extrapolated_300']['gigabytes']:.2f} GB")
    print(f"\nBranch B predictor   : {predictor.config.name} "
          f"(provisional={predictor.config.provisional})")
    for name in ("A1", "A2", "B"):
        s = summaries[name]
        if s.n_scenes == 0:
            continue
        print(f"\n-- {name} -- n={s.n_scenes} present={s.n_present} absent={s.n_absent}")
        print(f"   recall           : {s.recall}")
        print(f"   TP / FP          : {s.n_true_positive} / {s.n_false_positive}")
        print(f"   position median  : {s.position_median_mm} mm   p90 {s.position_p90_mm}")
        print(f"   axis median      : {s.angle_median_deg} deg    p90 {s.angle_p90_deg}")
        print(f"   end-to-end yield : {s.end_to_end_yield}   tol={s.tolerances}")
        if s.per_tilt:
            for tilt, row in sorted(s.per_tilt.items(), key=lambda kv: float(kv[0])):
                print(f"     tilt {tilt:>5}       : pos {row['position_median_mm']} mm, "
                      f"axis {row['angle_median_deg']} deg (n={row['n_true_positive']})")
    print(f"\nresults  : {results_path}")
    print(f"overlays : {overlay_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
