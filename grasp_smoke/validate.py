"""Dataset validation. Every check returns a reason, not a boolean.

``run_smoke.py`` is a **gate**, not a report: it has to be able to fail. A smoke
run that prints "ok" for every stage while silently scoring nine scenes, or a
dataset captured at a different seed, or two scenes sharing a timestamp, is worse
than a run that crashes -- it looks like evidence.

So each check below states exactly what it expected and what it found, and the
caller exits non-zero on any failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import dataset as ds
from .geometry import invert_transform


def _is_rigid(T: np.ndarray, tol: float = 1e-6) -> Optional[str]:
    """A 4x4 must actually be SE(3): orthonormal rotation, det +1, [0,0,0,1] row."""
    if T.shape != (4, 4):
        return f"shape {T.shape}, expected (4, 4)"
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-5):
        return "rotation block is not orthonormal"
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > 1e-5:
        return f"rotation determinant {det:.6f}, expected +1 (reflection?)"
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=tol):
        return f"bottom row {T[3, :].tolist()}, expected [0, 0, 0, 1]"
    return None


def validate_dataset(
    root: Path,
    expected_scene_ids: list,
    expected_seed: Optional[int] = None,
    expected_backend: Optional[str] = None,
    expected_depth_policy: Optional[str] = None,
    expected_schema: str = ds.SCHEMA_VERSION,
) -> list:
    """Return a list of human-readable problems. Empty list means the dataset is
    exactly what was asked for."""
    root = Path(root)
    problems = []

    if not (root / "manifest.json").exists():
        return [f"no manifest.json in {root}"]
    manifest = ds.load_manifest(root)

    # -- provenance -------------------------------------------------------
    if manifest.get("schema_version") != expected_schema:
        problems.append(
            f"schema_version {manifest.get('schema_version')!r}, expected {expected_schema!r}"
        )
    if expected_seed is not None and int(manifest.get("seed", -1)) != int(expected_seed):
        problems.append(f"seed {manifest.get('seed')}, expected {expected_seed}")
    if expected_backend is not None and manifest.get("capture_backend") != expected_backend:
        problems.append(
            f"capture_backend {manifest.get('capture_backend')!r}, expected {expected_backend!r}"
        )
    if expected_depth_policy is not None and manifest.get("depth_policy") != expected_depth_policy:
        problems.append(
            f"depth_policy {manifest.get('depth_policy')!r}, expected {expected_depth_policy!r}"
        )
    if manifest.get("depth_quantile") is None:
        problems.append("manifest does not record depth_quantile")

    # -- scene roster -----------------------------------------------------
    found_ids = [s["scene_id"] for s in manifest["scenes"]]
    if found_ids != list(expected_scene_ids):
        missing = sorted(set(expected_scene_ids) - set(found_ids))
        extra = sorted(set(found_ids) - set(expected_scene_ids))
        problems.append(
            f"scene roster mismatch: {len(found_ids)} scenes, expected "
            f"{len(expected_scene_ids)}"
            + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {extra}" if extra else "")
            + ("; order differs" if not missing and not extra else "")
        )

    stamps = [int(s["stamp_ns"]) for s in manifest["scenes"]]
    if len(set(stamps)) != len(stamps):
        dupes = sorted({t for t in stamps if stamps.count(t) > 1})
        problems.append(f"duplicate capture timestamps: {dupes}")

    # -- integrity --------------------------------------------------------
    bad = ds.verify_checksums(root)
    if bad:
        problems.append(f"{len(bad)} checksum/missing failures: {bad[:5]}")
    extra_files = ds.unexpected_files(root)
    if extra_files:
        problems.append(f"{len(extra_files)} files not declared in manifest: {extra_files[:5]}")

    # -- per-scene cross-checks ------------------------------------------
    for scene in manifest["scenes"]:
        sid = scene["scene_id"]
        try:
            frame = ds.load_frame(root, sid)
            labels = ds.load_labels(root, sid)
        except Exception as exc:                                # noqa: BLE001
            problems.append(f"{sid}: unreadable ({type(exc).__name__}: {exc})")
            continue

        if frame.stamp_ns != int(scene["stamp_ns"]):
            problems.append(
                f"{sid}: CameraInfo stamp {frame.stamp_ns} != manifest {scene['stamp_ns']}"
            )
        if labels.stamp_ns != frame.stamp_ns:
            problems.append(
                f"{sid}: labels stamp {labels.stamp_ns} != frame stamp {frame.stamp_ns}"
            )
        if frame.rgb.shape != (frame.height, frame.width, 3):
            problems.append(f"{sid}: rgb shape {frame.rgb.shape} vs {frame.height}x{frame.width}")
        if frame.depth_m.shape != (frame.height, frame.width):
            problems.append(f"{sid}: depth shape {frame.depth_m.shape}")
        if labels.gt_mask.shape != (frame.height, frame.width):
            problems.append(f"{sid}: gt_mask shape {labels.gt_mask.shape}")
        if not frame.base_frame_id or not frame.camera_frame_id:
            problems.append(f"{sid}: empty frame id(s)")
        if frame.base_frame_id == frame.camera_frame_id:
            problems.append(f"{sid}: base and camera frame ids are identical")

        for name, T in (("T_base_cam", frame.T_base_cam), ("T_base_object", labels.T_base_object)):
            why = _is_rigid(np.asarray(T, dtype=np.float64))
            if why:
                problems.append(f"{sid}: {name} is not a rigid transform -- {why}")
        # Round-tripping the inverse catches a transform that is rigid but stored
        # in the wrong direction only if it is also non-invertible, so check the
        # cheap invariant explicitly.
        T = np.asarray(frame.T_base_cam, dtype=np.float64)
        if _is_rigid(T) is None and not np.allclose(invert_transform(T) @ T, np.eye(4), atol=1e-9):
            problems.append(f"{sid}: T_base_cam does not invert cleanly")

        axis = np.asarray(labels.grasp_gt_open_axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if labels.target_present and abs(norm - 1.0) > 1e-6:
            problems.append(f"{sid}: grasp_gt_open_axis norm {norm:.6f}, expected 1")
        if labels.target_present and not labels.gt_mask.any():
            problems.append(f"{sid}: target_present but gt_mask is empty")
        if not labels.target_present and labels.gt_mask.any():
            problems.append(f"{sid}: target absent but gt_mask has {int(labels.gt_mask.sum())} px")

        K = np.asarray(frame.K, dtype=np.float64)
        if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
            problems.append(f"{sid}: implausible intrinsics {K.tolist()}")

    return problems
