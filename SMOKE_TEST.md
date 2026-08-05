# Aug 8 ten-scene vertical smoke test — results

**Run:** 2026-08-04 · **Gate:** PLAN.md §5.2.1 · **Reproduce:** `./run_smoke.sh`

The gate asks one question: does the whole chain work end to end, and does the
capture cost extrapolate to 300 scenes? Both answered, on the real Isaac Sim
backend.

```
randomize -> capture -> serialize -> replay -> A1 oracle mask
          -> B predicted mask -> PoseStamped -> scorer + overlay
```

All eight stages pass. 38 unit tests pass.

## Resource budget — the extrapolation the gate exists for

Measured over 10 scenes, Isaac Sim 5.1, headless, RTX 5000 Ada:

| Measure | Ten scenes | Extrapolated to 300 |
|---|---|---|
| Wall clock | 2.18 s capture (0.218 s/scene) + 8.6 s app startup | **~1.1 min** + startup |
| Disk | 13.5 MB (1.35 MB/scene) | **~0.40 GB** |
| Peak VRAM | **2416 MiB** of 16376 | unchanged (scene count is sequential) |
| Annotator read | **max 0.0026 s** | unchanged |

**The 300-scene dataset is not a scheduling risk.** It is about a minute of
compute and under half a gigabyte. The Aug 10–20 window is bounded by writing
and validating the pipeline, not by generating data — so if schedule pressure
appears, cutting scene count buys nothing and costs the preregistration.

## ⚠️ The documented depth stall did not reproduce

PLAN.md §5.2.1 ranked `distance_to_image_plane` as rung 1 with a headless-stall
risk carried from prior notes. **Measured: it works headless, max read 0.0026 s
across 10 scenes.** Rungs 2 (`--headful`) and 3 (`--allow-radial-depth`) are
implemented and the radial→optical-Z conversion is unit-tested, but neither was
needed. The `--enable_cameras` / annotator-stall notes appear to be Isaac Lab
script behaviour, not Isaac Sim standalone Replicator — the hedge in PLAN.md §10
was correct.

## Three bugs the smoke test caught

Exactly what a vertical slice is for. All three would have been near-invisible
at 300 scenes and expensive to unpick later.

1. **Camera near-clip.** A USD camera defaults to a 1 m near plane; the scene
   works at 0.6 m. Depth came back **all `inf`** and segmentation all-background
   while `idToLabels` still listed the prims — so it presented as an annotator
   bug rather than a camera bug. Fix: set `clippingRange` explicitly.
2. **No lights.** An empty `World()` has no default lighting, and the RTX
   renderer returned an **all-black RGB buffer** while depth and segmentation
   stayed perfectly healthy. The failure surfaced downstream as "the detector
   finds nothing." Fix: add dome + distant lights.
3. **`grasp_gt` off by half a box height.** The analytic backend centred the box
   on the object origin; the Isaac backend additionally lifted it to rest on the
   ground. Same authored `grasp_gt`, geometry half a box-height apart — which
   showed up as a uniform **20.5 mm** A1 position error. Fix: one convention,
   stated in `render.randomize_scene` and enforced in both backends.

Bug 3 is the instructive one: a *plausible* 20 mm error. Without A0 fixtures and
an exact-oracle A1 stratum to say "this should be ~0", it reads as ordinary
perception error and ships.

## Results — Isaac Sim backend, 10 scenes

Strata per PLAN.md §5.2.4. Tolerances: A1/A2 ≤3 mm / ≤3°, B ≤5 mm / ≤5°.

| Stratum | n | Recall | Position median | Axis median | Yield |
|---|---|---|---|---|---|
| **A1** oracle, nadir | 4 present | 1.0 | **0.0004 mm** | **0.0076°** | **1.0** |
| **A2** oracle, oblique | 4 present | 1.0 | 18.2 mm | 19.2° | 0.0 |
| **B** predicted | 8 present, 2 absent | 1.0 | 3.4 mm | 5.96° | 0.5 |

**A1 at 0.4 µm and 0.008° is the headline.** It means the pinhole model, the
frame conventions, the TF chain, the depth policy, and Isaac's renderer all agree
essentially exactly. That is the unit test the A0/A1 split was designed to give.

### A2 degrades monotonically with tilt — and it is not a bug

| Camera tilt | Position | Opening axis |
|---|---|---|
| 0° | 0.0004 mm | 0.008° |
| 15° | 6.4 mm | 11.8° |
| 25° | 15.0 mm | 15.3° |
| 35° | 21.5 mm | 25.8° |
| 45° | 21.3 mm | 23.1° |

This is the vendor algorithm's documented approximation, measured, with a
**perfect** mask and **exact** depth. Two causes, both in PLAN.md §5.2.4:

* `estimate_grasp` collapses the whole mask to **one scalar depth** via a
  quantile. On an object with extent along the viewing axis that is deliberate
  approximation, not error.
* The opening axis is recovered by back-projecting a 2-D short-edge direction at
  that single depth. Under perspective, the true 3-D opening axis of a tilted
  object is not the back-projection of its image-plane short edge.

So: an A2 miss is **not** grounds to go debugging the code. It is the number to
report, and the argument for either a plane-fit depth model or a real 6-D pose
estimator. Pooling A1 and A2 would have hidden it completely — which is why A2 is
reported per tilt.

### Branch B

Recall 1.0 at IoU ≥ 0.5, and at nadir it tracks A1 to 0.17 mm / 0.02° — the
provisional segmenter reproduces the GT mask almost exactly on this target.
**2 false positives, both on the target-absent scenes**: Otsu always splits an
image into two classes, so it segments the ground when there is nothing there.
Honest and expected; a real detector with a confidence threshold is what fixes
it, and that is the Aug 10 freeze.

### Determinism holds for replay, not for Isaac recapture

Worth knowing before anyone treats a rerun as a regression. **Replay is exactly
deterministic** — the same sealed dataset always produces the same scores, which
is what the §5.2.2 contract requires and what `manifest.json` checksums prove.
**Isaac *recapture* is not bit-identical**: RTX sampling and denoising perturb
the RGB slightly, so across two captures of the same seed Branch B moved
3.409 → 3.431 mm while A1/A2 (which use the GT mask) stayed identical to every
printed digit.

So: compare runs against a **sealed dataset**, not against a recapture. The
analytic backend *is* bit-deterministic and `TestDatasetContract.test_capture_is_deterministic`
enforces that; the equivalent guarantee does not exist for Isaac and should not
be assumed.

## Provisional Branch B configuration

| Field | Value |
|---|---|
| Predictor | `saturation_largest_blob` (**provisional**) |
| Threshold | Otsu-adaptive on HSV saturation |
| Morphology | open+close, 5×5 |
| Min area | 400 px |
| IoU match rule | **≥ 0.5** |
| Checkpoint | none |

⚠️ **This is not a detector result and must never be reported as one.** The
pinned vendor `yoloe-26s-seg.pt` is present in the tree (nothing to download),
but **`ultralytics` is not installed** and installing it needs approval. The
YOLOE path is implemented (`grasp_smoke/detect.YoloeSegmenter`) and selected
automatically the moment the package is importable — `build_predictor()` prefers
it and falls back with a printed reason.

A fixed saturation threshold of 60, tuned on the analytic renderer, produced
**zero recall** on Isaac renders before the switch to Otsu. That is a small
preview of why the Aug 10 freeze (checkpoint, prompt, threshold, IoU rule) is a
blocking decision rather than a formality.

## What is NOT done

- **No 300-scene dataset.** Out of scope, by instruction.
- **ROS 2 never executed.** `/opt/ros` is absent. `ros2_iface/dataset_publisher.py`
  and `ros2_iface/grasp_node.py` are written but unrun. All message *content* is
  built and tested ROS-free in `grasp_smoke/pose_msg.py` (`TestPoseStamped`), so
  the untested surface is the thin shell that copies fields into real messages.
- **The Isaac target is flat-shaded, not textured.** PLAN.md §5.2.3 asks for a
  textured target; the analytic backend has a procedural checkerboard but the
  Isaac target is a single `displayColor`. Adding real texture needs a material
  asset or MDL work. This matters — an untextured target makes segmentation
  unrepresentatively easy.
- **No M3, B601 USD, BYOR, XRDF or cuMotion work.** Out of scope, by instruction.

## Artifacts

| Path | Contents |
|---|---|
| `artifacts/smoke_isaac/dataset/` | 10 scenes, `manifest.json` with per-file SHA-256 |
| `artifacts/smoke_isaac/results.json` | stage status, metrics, per-scene scores, env |
| `artifacts/smoke_isaac/overlays/` | 20 debug overlays (A1 and B per scene) |
| `artifacts/smoke_isaac/isaac_progress.log` | per-scene capture log |

All gitignored — regenerate with `./run_smoke.sh`.
