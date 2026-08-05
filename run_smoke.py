#!/usr/bin/env python3
"""The Aug 8 ten-scene vertical smoke test (PLAN.md 5.2.1).

A **gate**, not a report. It validates the dataset it was handed, runs inference
and scoring as separate passes, and **exits non-zero** when the Branch B path
fails to reach a valid PoseStamped and evaluator. A run that prints "ok" for
every stage while quietly scoring nothing looks like evidence, which is worse
than a crash.

    python run_smoke.py --backend analytic --predictor saturation
    python run_smoke.py --backend isaac --predictor yoloe   # needs Isaac + ultralytics

Exit codes: 0 all gates passed | 2 dataset validation | 3 capture |
4 replay integrity | 5 Branch B never produced a scored PoseStamped
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

from grasp_smoke import dataset as ds                              # noqa: E402
from grasp_smoke import inference as infer                         # noqa: E402
from grasp_smoke.capture import (                                  # noqa: E402
    BASELINE_DEPTH_QUANTILE, capture_analytic, expected_scene_ids,
    plan_smoke_scenes, validate_depth_quantile,
)
from grasp_smoke.detect import (                                   # noqa: E402
    PREDICTOR_CHOICES, PredictorUnavailable, build_predictor,
)
from grasp_smoke.grasp import estimate_grasp                       # noqa: E402
from grasp_smoke.overlay import render_overlay                     # noqa: E402
from grasp_smoke.pose_msg import grasp_to_pose_stamped             # noqa: E402
from grasp_smoke.scorer import score_scene, summarize, write_results  # noqa: E402
from grasp_smoke.validate import validate_dataset                  # noqa: E402

EXIT_OK, EXIT_VALIDATION, EXIT_CAPTURE, EXIT_REPLAY, EXIT_NO_BRANCH_B = 0, 2, 3, 4, 5


def _log(stage: str, msg: str) -> None:
    print(f"[{stage:<10}] {msg}", flush=True)


def _fail(stages: dict, stage: str, detail, code: int) -> int:
    stages[stage] = f"FAILED: {detail}"
    print("\n" + "=" * 72)
    print(f"SMOKE TEST FAILED at '{stage}' (exit {code})")
    print("=" * 72)
    for k, v in stages.items():
        print(f"  {k:<22} {v}")
    if isinstance(detail, (list, tuple)):
        print("\ndetail:")
        for item in detail:
            print(f"  - {item}")
    return code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["analytic", "isaac"], default="analytic")
    ap.add_argument("--predictor", choices=PREDICTOR_CHOICES, default="saturation",
                    help="explicit; yoloe fails closed rather than falling back")
    ap.add_argument("--target-class", default="box")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "smoke")
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--depth-quantile", type=float, default=BASELINE_DEPTH_QUANTILE)
    ap.add_argument("--allow-ablation", action="store_true",
                    help="permit a declared non-baseline depth quantile")
    ap.add_argument("--reuse-dataset", action="store_true")
    ap.add_argument("--empty-predictor", action="store_true",
                    help=argparse.SUPPRESS)   # regression hook: always predicts nothing
    args = ap.parse_args()

    stages: dict = {}
    out = Path(args.out)
    data_root = out / "dataset"
    pred_dir = out / "predictions"
    overlay_dir = out / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    try:
        depth_quantile = validate_depth_quantile(args.depth_quantile, args.allow_ablation)
    except ValueError as exc:
        return _fail(stages, "depth-quantile", str(exc), EXIT_VALIDATION)
    is_ablation = abs(depth_quantile - BASELINE_DEPTH_QUANTILE) > 1e-12

    specs = plan_smoke_scenes(args.seed)
    expected_ids = expected_scene_ids(args.seed)

    # ---- 1. capture -------------------------------------------------------
    if args.reuse_dataset and (data_root / "manifest.json").exists():
        manifest = ds.load_manifest(data_root)
        # A replay must not silently re-interpret a sealed dataset under different
        # estimator settings than the ones it was captured with.
        sealed_q = float(manifest.get("depth_quantile", -1))
        if not args.allow_ablation and abs(sealed_q - depth_quantile) > 1e-12:
            return _fail(
                stages, "replay-args",
                f"dataset was sealed with depth_quantile={sealed_q} but "
                f"--depth-quantile={depth_quantile} was requested. Pass "
                f"--allow-ablation to score a sealed dataset off-baseline.",
                EXIT_VALIDATION,
            )
        capture_stats = {"backend": "reused", "n_scenes": len(specs),
                         "sealed_depth_quantile": sealed_q}
        _log("capture", f"reusing sealed dataset at {data_root}")
    elif args.backend == "analytic":
        _log("capture", f"analytic backend, {len(specs)} scenes -> {data_root}")
        try:
            capture_stats = capture_analytic(
                data_root, specs, seed=args.seed, depth_quantile=depth_quantile
            )
        except Exception as exc:                                # noqa: BLE001
            return _fail(stages, "capture", f"{type(exc).__name__}: {exc}", EXIT_CAPTURE)
    else:
        _log("capture", "isaac backend -> capture/isaac_capture.py")
        rc = subprocess.call([
            sys.executable, str(REPO / "capture" / "isaac_capture.py"),
            "--out", str(data_root), "--seed", str(args.seed),
            "--depth-quantile", str(depth_quantile),
        ])
        if rc != 0:
            return _fail(stages, "capture", f"isaac_capture exit {rc}", EXIT_CAPTURE)
        capture_stats = json.loads((data_root / "capture_stats.json").read_text())
    stages["randomize+capture"] = "ok"
    stages["serialize"] = "ok"

    # ---- 2. dataset validation -------------------------------------------
    expected_backend = None if args.reuse_dataset else (
        "analytic" if args.backend == "analytic" else "isaac_sim_5.1"
    )
    problems = validate_dataset(
        data_root,
        expected_scene_ids=expected_ids,
        expected_seed=args.seed,
        expected_backend=expected_backend,
        expected_depth_policy=ds.DEPTH_POLICY_IMAGE_PLANE,
    )
    if problems:
        return _fail(stages, "validate", problems, EXIT_VALIDATION)
    manifest = ds.load_manifest(data_root)
    manifest_sha = ds.manifest_sha256(data_root)
    stages["validate"] = (f"ok ({len(manifest['scenes'])} scenes, schema "
                          f"{manifest['schema_version']}, seed {manifest['seed']})")
    stages["replay"] = f"ok (checksums verified, manifest {manifest_sha[:12]})"
    _log("validate", f"{len(expected_ids)} expected scene ids present, stamps unique")

    # ---- 3. inference pass (sensor frames only) --------------------------
    try:
        if args.empty_predictor:
            from tests.helpers import AlwaysEmptyPredictor
            predictor = AlwaysEmptyPredictor()
        elif args.predictor == "yoloe":
            predictor = build_predictor("yoloe", target_class=args.target_class)
        else:
            predictor = build_predictor("saturation")
    except PredictorUnavailable as exc:
        return _fail(stages, "predictor", str(exc), EXIT_VALIDATION)
    _log("branch-B", f"predictor={predictor.config.name} "
                     f"diagnostic_only={predictor.config.diagnostic_only}")

    pred_index = infer.run_inference(data_root, predictor, pred_dir, branch="B")
    bad_pred = infer.verify_predictions(pred_dir)
    if bad_pred:
        return _fail(stages, "inference", bad_pred, EXIT_REPLAY)
    stages["inference (B)"] = (f"ok ({pred_index['detections']}/{pred_index['attempts']} "
                               f"scenes produced a mask)")

    # ---- 4. scoring pass (loads GT) --------------------------------------
    spec_by_id = {s.scene_id: s for s in specs}
    scores, pose_msgs = [], []
    counters = {b: {"attempts": 0, "valid_estimates": 0, "pose_stamped": 0}
                for b in ("A1", "A2", "B1", "B2")}
    t_score = time.perf_counter()

    for scene_id in ds.scene_ids(data_root):
        frame = ds.load_frame(data_root, scene_id)
        labels = ds.load_labels(data_root, scene_id)       # GT enters only here
        spec = spec_by_id[scene_id]
        oblique = spec.tilt_deg > 0.0

        for kind in ("oracle", "predicted"):
            if kind == "oracle":
                branch = "A2" if oblique else "A1"
                mask = labels.gt_mask if labels.target_present else None
            else:
                branch = "B2" if oblique else "B1"
                mask = infer.load_mask(pred_dir, scene_id)

            counters[branch]["attempts"] += 1
            est = (estimate_grasp(mask, frame.depth_m, frame.K, depth_quantile)
                   if mask is not None else None)
            if est is not None and est.is_valid:
                counters[branch]["valid_estimates"] += 1

            score = score_scene(
                scene_id, branch, est, mask, labels, frame.T_base_cam,
                iou_match_threshold=predictor.config.metric_iou_match_threshold,
                tilt_deg=spec.tilt_deg,
            )
            scores.append(score)

            if est is not None and est.is_valid:
                pose_msgs.append(grasp_to_pose_stamped(est, frame, branch=branch))
                counters[branch]["pose_stamped"] += 1

            cv2.imwrite(str(overlay_dir / f"{scene_id}_{branch}.png"),
                        render_overlay(frame, est, mask, labels, score, branch))

    score_s = time.perf_counter() - t_score
    stages["A1/A2 oracle"] = (f"ok ({counters['A1']['pose_stamped']} + "
                              f"{counters['A2']['pose_stamped']} PoseStamped)")
    stages["B1/B2 predicted"] = (f"{counters['B1']['pose_stamped']} + "
                                 f"{counters['B2']['pose_stamped']} PoseStamped")
    stages["scorer+overlay"] = "ok"

    summaries = {b: summarize(scores, b) for b in ("A1", "A2", "B1", "B2")}
    for b, s in summaries.items():
        s.n_prediction_attempts = counters[b]["attempts"]
        s.n_valid_estimates = counters[b]["valid_estimates"]
        s.n_pose_stamped = counters[b]["pose_stamped"]

    payload = {
        "gate_passed": None,
        "stages": stages,
        "capture": capture_stats,
        "dataset": {
            "manifest_sha256": manifest_sha,
            "schema_version": manifest["schema_version"],
            "seed": manifest["seed"],
            "capture_backend": manifest["capture_backend"],
            "depth_policy": manifest["depth_policy"],
            "capture_depth_quantile": manifest.get("depth_quantile"),
            "expected_scene_ids": expected_ids,
        },
        "estimator": {
            "depth_quantile_used": depth_quantile,
            "baseline_depth_quantile": BASELINE_DEPTH_QUANTILE,
            "is_ablation": is_ablation,
        },
        "branch_b": {
            "predictor": predictor.config.to_dict(),
            "predictions_index": {k: pred_index[k] for k in
                                  ("attempts", "detections", "dataset_manifest_sha256")},
        },
        "counters": counters,
        "scoring_seconds": round(score_s, 3),
        "summaries": summaries,
        "scenes": scores,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "example_pose_stamped": pose_msgs[0] if pose_msgs else None,
    }

    # ---- 5. the gate ------------------------------------------------------
    b_pose = counters["B1"]["pose_stamped"] + counters["B2"]["pose_stamped"]
    b_scored = sum(1 for s in scores if s.branch in ("B1", "B2") and s.true_positive)
    if b_pose == 0 or b_scored == 0:
        payload["gate_passed"] = False
        write_results(out / "results.json", payload)
        return _fail(
            stages, "gate",
            f"Branch B produced {b_pose} PoseStamped and {b_scored} scored true "
            f"positives; the predicted path never reached the evaluator. "
            f"Predictor was {predictor.config.name!r}.",
            EXIT_NO_BRANCH_B,
        )
    payload["gate_passed"] = True
    results_path = write_results(out / "results.json", payload)

    # ---- 6. summary -------------------------------------------------------
    print("\n" + "=" * 72)
    print("TEN-SCENE SMOKE TEST -- GATE PASSED")
    print("=" * 72)
    for stage, status in stages.items():
        print(f"  {stage:<22} {status}")
    print(f"\ndataset manifest sha : {manifest_sha}")
    print(f"capture backend      : {manifest['capture_backend']}")
    print(f"estimator quantile   : {depth_quantile}"
          f"{'  (ABLATION)' if is_ablation else '  (frozen Seeed baseline)'}")
    if "seconds_per_scene" in capture_stats:
        print(f"  time/scene         : {capture_stats['seconds_per_scene']:.3f} s")
        print(f"  disk/scene         : {capture_stats['bytes_per_scene']/1e6:.2f} MB")
        print(f"  peak VRAM          : {capture_stats['peak_vram_mib']} MiB")
        print(f"  -> 300 scenes      : {capture_stats['extrapolated_300']['minutes']:.1f} min, "
              f"{capture_stats['extrapolated_300']['gigabytes']:.2f} GB")
    print(f"\nBranch B predictor   : {predictor.config.name} "
          f"(diagnostic_only={predictor.config.diagnostic_only})")

    for name in ("A1", "A2", "B1", "B2"):
        s = summaries[name]
        if s.n_scenes == 0:
            continue
        print(f"\n-- {name} -- n={s.n_scenes} present={s.n_present} absent={s.n_absent}")
        print(f"   attempts/valid/pose : {s.n_prediction_attempts}/"
              f"{s.n_valid_estimates}/{s.n_pose_stamped}")
        print(f"   recall              : {s.recall}")
        print(f"   TP / FP             : {s.n_true_positive} / {s.n_false_positive}")
        if s.n_absent:
            print(f"   ** FPR on absent    : {s.false_positive_rate_absent} "
                  f"({s.n_absent} negative scenes) **")
        print(f"   position median     : {s.position_median_mm} mm   p90 {s.position_p90_mm}")
        print(f"   axis median         : {s.angle_median_deg} deg    p90 {s.angle_p90_deg}")
        print(f"   end-to-end yield    : {s.end_to_end_yield}   tol={s.tolerances}")
        if s.per_tilt:
            for tilt, row in sorted(s.per_tilt.items(), key=lambda kv: float(kv[0])):
                print(f"     tilt {tilt:>5}          : pos {row['position_median_mm']} mm, "
                      f"axis {row['angle_median_deg']} deg (n={row['n_true_positive']})")

    print(f"\nresults     : {results_path}")
    print(f"predictions : {pred_dir}")
    print(f"overlays    : {overlay_dir}")
    print("=" * 72)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
