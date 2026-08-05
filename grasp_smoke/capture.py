"""Deterministic seeded capture of the ten smoke scenes.

Two backends behind one scene plan:

* ``analytic`` -- :mod:`grasp_smoke.render`, exact and dependency-free. Always
  available, so the chain is testable without a simulator.
* ``isaac`` -- ``capture/isaac_capture.py``, Replicator inside Isaac Sim.

The plan is identical for both, so a dataset from either backend is scored by
the same code. ``manifest.json`` records which backend ran; analytic numbers are
never presented as Isaac Sim numbers.

Stratification follows PLAN.md 5.2.4: tilt 0 scenes are the A1 nominal baseline,
non-zero tilts are the A2 oblique stress strata reported per tilt, and
target-absent scenes exist only to give false positives somewhere to land.
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

#: Fixed wall-clock epoch for stamps. Deterministic on purpose -- capture must
#: not embed the time it happened to run, or reruns stop being comparable.
BASE_STAMP_NS = 1_700_000_000_000_000_000
STAMP_STEP_NS = 100_000_000  # 10 Hz


@dataclass
class SceneSpec:
    scene_id: str
    seed: int
    tilt_deg: float
    target_present: bool

    @property
    def stratum(self) -> str:
        if not self.target_present:
            return "absent"
        return "A1" if self.tilt_deg == 0.0 else "A2"


def plan_smoke_scenes(seed: int = 20260808) -> list:
    """The ten-scene smoke plan: 4x A1 nominal, 4x A2 oblique, 2x target-absent."""
    specs = []
    for i, tilt in enumerate([0.0, 0.0, 0.0, 0.0, 15.0, 25.0, 35.0, 45.0]):
        specs.append(SceneSpec(f"scene_{i:04d}", seed + i, tilt, True))
    for j in range(2):
        i = 8 + j
        specs.append(SceneSpec(f"scene_{i:04d}", seed + i, 0.0, False))
    return specs


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
    depth_quantile: float = 0.75,
) -> dict:
    """Render and serialise every scene in ``specs``. Returns capture statistics."""
    root = Path(root)
    writer = DatasetWriter(
        root=root,
        seed=seed,
        capture_backend="analytic",
        depth_policy=DEPTH_POLICY_IMAGE_PLANE,
        depth_quantile=depth_quantile,
        notes=(
            "Exact ray-cast renderer. Depth is closed-form optical-axis Z, so the "
            "distance_to_image_plane stall risk does not apply to this backend."
        ),
    )

    per_scene_s = []
    with VramSampler() as vram:
        t_all = time.perf_counter()
        for i, spec in enumerate(specs):
            t0 = time.perf_counter()
            target, T_base_cam, K = randomize_scene(spec.seed, tilt_deg=spec.tilt_deg)
            scene = render(
                target, T_base_cam, K, width, height,
                rng=np.random.default_rng(spec.seed),
                include_target=spec.target_present,
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

    writer.manifest.notes += f" | scene_plan={[s.stratum for s in specs]}"
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
        "extrapolated_300": {
            "minutes": round(total_s / n * 300 / 60.0, 2),
            "gigabytes": round(size_b / n * 300 / 1e9, 3),
        },
    }
