"""Inference pass. Sensor frames in, checksummed prediction masks out.

Physically separated from scoring. This module imports :func:`load_frame` and
never :func:`load_labels`, and writes its masks to a directory the scorer reads
afterwards. Two consequences worth the extra file:

* GT leakage into Branch B is not merely discouraged, it is unreachable -- there
  is no code path from here to a label.
* Predictions become an artifact with their own checksums, so a scoring run can
  prove which masks it scored and a rerun can be compared against them.

The oracle strata (A1/A2) do not go through here. They are, by definition, a
scoring-time construct that reads the GT mask -- and saying so plainly is better
than pretending the oracle is a kind of prediction.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import dataset as ds
from .detect import MaskPredictor

PREDICTIONS_SCHEMA = "1.0.0"


def run_inference(
    dataset_root: Path,
    predictor: MaskPredictor,
    out_dir: Path,
    branch: str = "B",
) -> dict:
    """Predict a mask for every scene. Returns the prediction index."""
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    records = []
    attempts = 0
    t0 = time.perf_counter()

    for scene_id in ds.scene_ids(dataset_root):
        frame = ds.load_frame(dataset_root, scene_id)     # sensor data only
        attempts += 1
        prediction = predictor.predict(frame)

        record = {
            "scene_id": scene_id,
            "stamp_ns": frame.stamp_ns,
            "detected": prediction is not None,
            "mask_file": None,
            "mask_sha256": None,
            "mask_pixels": 0,
        }
        if prediction is not None:
            rel = f"masks/{scene_id}.png"
            path = out_dir / rel
            Image.fromarray((prediction.mask > 0).astype(np.uint8) * 255).save(path)
            record.update(
                mask_file=rel,
                mask_sha256=ds.sha256_file(path),
                mask_pixels=int((prediction.mask > 0).sum()),
                **prediction.to_dict(),
            )
        records.append(record)

    index = {
        "schema_version": PREDICTIONS_SCHEMA,
        "branch": branch,
        "dataset_manifest_sha256": ds.manifest_sha256(dataset_root),
        "predictor": predictor.config.to_dict(),
        "attempts": attempts,
        "detections": sum(1 for r in records if r["detected"]),
        "seconds_total": round(time.perf_counter() - t0, 3),
        "records": records,
    }
    (out_dir / "predictions.json").write_text(
        json.dumps(index, sort_keys=True, indent=2) + "\n"
    )
    return index


def load_predictions(out_dir: Path) -> dict:
    return json.loads((Path(out_dir) / "predictions.json").read_text())


def verify_predictions(out_dir: Path) -> list:
    """Re-hash every written mask. Returns paths that no longer match."""
    out_dir = Path(out_dir)
    index = load_predictions(out_dir)
    bad = []
    for record in index["records"]:
        rel = record.get("mask_file")
        if not rel:
            continue
        path = out_dir / rel
        if not path.exists():
            bad.append(f"{rel} (MISSING)")
        elif ds.sha256_file(path) != record["mask_sha256"]:
            bad.append(rel)
    return bad


def load_mask(out_dir: Path, scene_id: str):
    """Load one prediction mask, or None if nothing was detected."""
    index = load_predictions(out_dir)
    for record in index["records"]:
        if record["scene_id"] == scene_id:
            if not record.get("mask_file"):
                return None
            mask = np.asarray(
                Image.open(Path(out_dir) / record["mask_file"]).convert("L")
            )
            return (mask > 127).astype(np.uint8)
    raise KeyError(f"no prediction record for {scene_id}")
