# Aug 8 ten-scene vertical smoke test — results

**Run:** 2026-08-04 · **Gate:** PLAN.md §5.2.1 · **Reproduce:** `./run_smoke.sh isaac`
**Hardened:** 2026-08-05 — see [Hardening pass](#hardening-pass-2026-08-05) for what changed.

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
needed.

⚠️ **The elapsed-time check around `get_data()` is a report, not a timeout.** It
measures how long the call took *after* it returned, so a genuine hang blocks
forever and the check never fires. Treating it as an enforceable timeout would be
wrong. A real timeout needs the annotator read on a separate thread or process
with a join deadline; that is not implemented. The `--enable_cameras` / annotator-stall notes appear to be Isaac Lab
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

### A2 degrades with tilt from 15° to 35° — and it is not a bug

| Camera tilt | Position | Opening axis |
|---|---|---|
| 0° | 0.0004 mm | 0.008° |
| 15° | 6.4 mm | 11.8° |
| 25° | 15.0 mm | 15.3° |
| 35° | 21.5 mm | 25.8° |
| 45° | 21.3 mm | 23.1° |

Error grows steadily from 15° to 35°; **45° is slightly lower than 35° on both
axes, so the trend is not monotonic across the full range** — with one scene per
tilt that reversal is well inside noise and should not be read as a finding
either way. The honest statement is: sharp degradation between 15° and 35°.

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

### Branch B — and its false-positive rate, which is the number to lead with

Recall 1.0 at IoU ≥ 0.5, and at nadir it tracks A1 to 0.17 mm / 0.02° — the
provisional segmenter reproduces the GT mask almost exactly on this target.

> ### ⚠️ **False-positive rate on target-absent scenes: 1.0 (2 of 2).**
> Otsu always splits an image into two classes, so it segments *something* even
> when there is nothing to find. Quoting Branch B's recall or pose error without
> this number attached is straightforwardly misleading: a detector that fires on
> every frame can have perfect recall and be useless.

That is what a confidence threshold and a real class prior are for, and it is the
Aug 10 freeze. The scorer now prints FPR on its own line for every stratum that
has negatives.

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

⚠️ **This is not a detector result and must never be reported as one.** It is
marked `diagnostic_only: true` in every results file so a consumer can refuse it
programmatically. The pinned vendor `yoloe-26s-seg.pt` is in the tree (nothing to
download), but **`ultralytics` is not installed** and installing it needs
approval.

**Selection is explicit and fails closed.** `--predictor yoloe` either produces
the real detector or exits non-zero; it never falls back to the saturation
stand-in. (Before the hardening pass it *did* fall back silently, which would
have produced a results file labelled "Branch B" while reporting a colour
threshold.)

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


---

## Hardening pass (2026-08-05)

A correctness pass before any dependency installation or 300-scene capture.
**No dependency was installed, no real YOLOE or ROS execution happened, and
`artifacts/smoke_isaac/` was left untouched.**

### The results above predate the scene-plan change

The Isaac numbers in this document were measured on the **pre-hardening scene
composition** (8 present + 2 empty negatives, no distractors, no aim jitter). The
hardened plan is 4 nominal + 4 oblique + 1 empty negative + **1 distractor-only
negative**, with deterministic aim jitter and a multi-scale procedural texture.
Reproducing the Isaac gate under the new plan needs a fresh capture into a new
output directory, which was deliberately not run so the existing sealed dataset
survives. On the **analytic** backend under the new plan the provisional
segmenter's recall drops to 0.75 and its absent-scene FPR is 0.5 — the harder
scenes do what they were added to do.

### What changed

**The smoke command is now a gate that can fail.** It validates schema, seed,
backend, depth policy, the exact expected scene-id roster, timestamp uniqueness,
and all ten scene records; cross-checks CameraInfo/TF/label stamps, frame ids,
array shapes and SE(3) rigidity; and rejects both missing and undeclared files.
It records prediction attempts, valid estimates and PoseStamped counts separately
per stratum, and **exits non-zero when Branch B never reaches a scored
PoseStamped**. Exit codes: 2 validation, 3 capture, 4 replay integrity, 5 Branch
B never reached the evaluator.

**Inference and scoring are separate passes.** `grasp_smoke/inference.py` reads
sensor frames only — a test parses its AST to prove it contains no executable
reference to `load_labels`, `gt_mask` or `grasp_gt_position` — and emits
checksummed prediction masks. Scoring loads ground truth afterwards.

**Pose orientation conventions are explicit.** The original
`grasp_to_pose_stamped()` remains a backward-compatible analysis output whose
quaternion is the vision basis `[grip, open, approach]`; it is not a robot TCP
target. `grasp_to_b601_tcp_pose_stamped()` is the control-boundary function and
maps that basis to vendor B601 TCP axes (`X=-approach`, `Y=open`, right-handed
`Z`) before applying `base <- camera`. No ROS node has been switched to the TCP
path yet; doing so requires the controller to name and validate its physical TCP.

**Dataset writes are locked.** Capturing into an existing non-empty directory
raises `DatasetExistsError` instead of overwriting a possibly-sealed dataset.
Capture metadata is written *before* the manifest is sealed, and the manifest's
own SHA-256 is recorded in every results file.

**The Seeed baseline is frozen at `depth_quantile=0.5`** — the shipped
`config/default.yaml:61` value, not the 0.75 code default. 0.75 is accepted only
with `--allow-ablation` and is flagged `is_ablation: true` in the results. The
estimator quantile is recorded separately from capture metadata, and replaying a
sealed dataset with a conflicting quantile is refused.

**YOLOE follows the pinned vendor contract**, verified against
`547faa08e5161af996892497c0aaa788401454fc`: `YOLO(checkpoint)` then
`set_classes(class_list)`; `predict(bgr, conf=0.25, iou=0.45)`; mask resized
NEAREST and thresholded at 0.5; and target selection by **exact class match, then
substring, then max confidence within that candidate set** — not the globally
most-confident mask, which returns a distractor whenever the detector is surer
about it. The checkpoint SHA-256 is verified at construction
(`6f62bc7e…39a17d`) and a mismatch fails closed. Package version, device, imgsz,
class list, confidence, NMS IoU and mask threshold are all recorded, and the
**metric IoU match rule is stored separately from the NMS IoU** so the two cannot
be confused. 18 mocked tests cover this; none of them runs real inference.

**ROS defects fixed** (still unexecuted): publisher and subscriber now share
topic names from `ros2_iface/topics.py` — previously the publisher advertised
`~/rgb` while the node subscribed to `rgb`, two different resolved names, so the
graph would have looked healthy and delivered nothing. The documented invocation
is now `python3 -m ros2_iface.<node>` and the relative import that broke it is
gone. `/clock` is published from recorded stamps with documented sim-time
semantics. Synchronized triplets are **queued until TF exists at their exact
stamp** rather than dropped, with a 5 s give-up and a bounded queue. Triplets are
validated for matching stamps, frame ids, sizes, step, endianness and encodings —
`16UC1` depth is refused rather than silently read as metres.
`ros2_iface/test_jazzy_integration.py` is a real gate that **skips loudly**; it
has never been executed.

### Test counts

| Suite | Tests | Runs where |
|---|---|---|
| `tests/test_a0_geometry.py` | 16 | anywhere |
| `tests/test_pipeline.py` | 22 | anywhere |
| `tests/test_detect_yoloe.py` | 18 | anywhere (mocked; no ultralytics) |
| `tests/test_gate.py` | 30 | anywhere (spawns the real command) |
| `tests/test_ros_contract.py` | 18 | anywhere (no ROS needed) |
| `ros2_iface/test_jazzy_integration.py` | 4 | **never run — needs ROS 2 Jazzy** |

### Still requiring approval

1. **`pip install ultralytics==8.4.35`** in an isolated inference environment.
   Until then Branch B has no detector. Weights are already local; only the
   package is missing.
2. **Installing ROS 2 Jazzy**, after which the integration gate can run for the
   first time.

Neither was done here.
