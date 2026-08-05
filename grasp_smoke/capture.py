"""Deterministic seeded capture of the ten smoke scenes.

Two backends behind one scene plan:

* ``analytic`` -- :mod:`grasp_smoke.render`, exact and dependency-free.
* ``isaac`` -- ``capture/isaac_capture.py``, Replicator inside Isaac Sim.

The plan is shared, so a dataset from either backend is scored by the same code.
``manifest.json`` records which backend ran; analytic numbers are never presented
as Isaac Sim numbers.

Strata (PLAN.md 5.2.4), reported separately and never pooled:

===========  ==============================================================
``A1``/``B1``  nominal: camera on the object normal
``A2``/``B2``  oblique stress at predefined tilts
``absent``     no target. Two kinds, and the distinction matters:
               an empty scene, and a **distractor-only** scene that contains
               target-like objects. Only the second one can catch a detector
               that has learned "something salient is present".
===========  ==============================================================
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import (
    DEPTH_POLICY_IMAGE_PLANE,
    DatasetWriter,
    Frame,
    Labels,
    directory_size_bytes,
)
from .render import randomize_scene, render

#: Fixed epoch. Deterministic on purpose -- capture must not embed the time it
#: happened to run, or reruns stop being comparable.
BASE_STAMP_NS = 1_700_000_000_000_000_000
STAMP_STEP_NS = 100_000_000  # 10 Hz

#: The frozen faithful-Seeed baseline. ``config/default.yaml:61`` sets
#: ``depth_quantile: 0.5``; the code default in ``ordinary_grasp.py:56,103`` is
#: 0.75. The **shipped configuration wins** -- that is what the vendor pipeline
#: actually runs -- and 0.75 is available only as a declared ablation.
BASELINE_DEPTH_QUANTILE = 0.5
ABLATION_DEPTH_QUANTILES = (0.75,)


def validate_depth_quantile(value: float, allow_ablation: bool = False) -> float:
    """Reject a quantile that silently departs from the frozen baseline."""
    value = float(value)
    if abs(value - BASELINE_DEPTH_QUANTILE) < 1e-12:
        return value
    if allow_ablation and any(abs(value - q) < 1e-12 for q in ABLATION_DEPTH_QUANTILES):
        return value
    raise ValueError(
        f"depth_quantile={value} is not the frozen baseline "
        f"{BASELINE_DEPTH_QUANTILE}. Declared ablations {ABLATION_DEPTH_QUANTILES} "
        f"require --allow-ablation."
    )


@dataclass
class SceneSpec:
    scene_id: str
    seed: int
    tilt_deg: float
    target_present: bool
    n_distractors: int = 0
    aim_jitter: bool = True

    @property
    def stratum(self) -> str:
        if not self.target_present:
            return "distractor_absent" if self.n_distractors else "absent"
        return "nominal" if self.tilt_deg == 0.0 else "oblique"


def plan_smoke_scenes(seed: int = 20260808) -> list:
    """Ten scenes: 4 nominal, 4 oblique, 1 empty negative, 1 distractor negative."""
    specs = []
    for i, tilt in enumerate([0.0, 0.0, 0.0, 0.0, 15.0, 25.0, 35.0, 45.0]):
        specs.append(SceneSpec(f"scene_{i:04d}", seed + i, tilt, True,
                               n_distractors=(1 if i % 2 else 0)))
    specs.append(SceneSpec("scene_0008", seed + 8, 0.0, False, n_distractors=0))
    specs.append(SceneSpec("scene_0009", seed + 9, 0.0, False, n_distractors=2))
    return specs


def expected_scene_ids(seed: int = 20260808) -> list:
    return [s.scene_id for s in plan_smoke_scenes(seed)]


class VramSampler:
    """Polls nvidia-smi in a thread and keeps the peak used-MiB seen."""

    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                self.peak_mib = max(self.peak_mib, int(out.stdout.strip().splitlines()[0]))
            except Exception:                        # noqa: BLE001
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return False


def capture_analytic(
    root: Path,
    specs: list,
    seed: int,
    width: int = 640,
    height: int = 480,
    depth_quantile: float = BASELINE_DEPTH_QUANTILE,
    locked: bool = True,
) -> dict:
    """Render and serialise every scene in ``specs``. Returns capture statistics."""
    root = Path(root)
    writer = DatasetWriter(
        root=root,
        seed=seed,
        capture_backend="analytic",
        depth_policy=DEPTH_POLICY_IMAGE_PLANE,
        depth_quantile=depth_quantile,
        locked=locked,
        notes=(
            "Exact ray-cast renderer. Depth is closed-form optical-axis Z, so the "
            "distance_to_image_plane stall risk does not apply to this backend."
        ),
    )
    writer.set_capture_metadata(
        backend="analytic",
        width=width,
        height=height,
        seed=seed,
        depth_quantile=depth_quantile,
        scene_plan=[{"scene_id": s.scene_id, "stratum": s.stratum, "tilt_deg": s.tilt_deg,
                     "n_distractors": s.n_distractors, "aim_jitter": s.aim_jitter}
                    for s in specs],
    )

    per_scene_s = []
    with VramSampler() as vram:
        t_all = time.perf_counter()
        for i, spec in enumerate(specs):
            t0 = time.perf_counter()
            target, T_base_cam, K, distractors = randomize_scene(
                spec.seed, tilt_deg=spec.tilt_deg, aim_jitter=spec.aim_jitter,
                n_distractors=spec.n_distractors,
            )
            scene = render(
                target, T_base_cam, K, width, height,
                rng=np.random.default_rng(spec.seed),
                include_target=spec.target_present,
                distractors=distractors,
            )
            gt_pos, gt_axis = target.grasp_gt()
            stamp = BASE_STAMP_NS + i * STAMP_STEP_NS

            writer.write_scene(
                Frame(
                    scene_id=spec.scene_id, rgb=scene.rgb, depth_m=scene.depth_m,
                    K=K, width=width, height=height, stamp_ns=stamp,
                    T_base_cam=T_base_cam,
                ),
                Labels(
                    scene_id=spec.scene_id, stamp_ns=stamp, gt_mask=scene.mask,
                    grasp_gt_position=gt_pos, grasp_gt_open_axis=gt_axis,
                    T_base_object=target.T_base_object,
                    object_dims_m=target.dims_m,
                    target_present=spec.target_present,
                ),
            )
            per_scene_s.append(time.perf_counter() - t0)
        total_s = time.perf_counter() - t_all

    writer.seal()

    size_b = directory_size_bytes(root)
    n = len(specs)
    return {
        "backend": "analytic",
        "n_scenes": n,
        "total_seconds": round(total_s, 3),
        "seconds_per_scene": round(total_s / n, 3),
        "seconds_per_scene_max": round(max(per_scene_s), 3),
        "bytes_total": size_b,
        "bytes_per_scene": int(size_b / n),
        "peak_vram_mib": vram.peak_mib,
        "depth_quantile": depth_quantile,
        "extrapolated_300": {
            "minutes": round(total_s / n * 300 / 60.0, 2),
            "gigabytes": round(size_b / n * 300 / 1e9, 3),
        },
    }
