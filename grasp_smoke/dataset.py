"""File-native immutable dataset with scorer-only label sidecars.

The contract from PLAN.md 5.2.2:

* The **dataset on disk is the source of truth**, not a bag. MCAP, if produced at
  all, is derived from replaying this.
* **Ground truth lives in ``labels/``, keyed by scene id and timestamp**, and is
  never reachable from the sensor-side loader. :func:`load_frame` physically
  cannot return a GT mask or ``grasp_gt`` -- that is what makes GT leakage into
  Branch B structurally impossible rather than a matter of discipline.
* Every file is checksummed into ``manifest.json`` along with the schema version,
  seed, and depth policy, so a replay can prove it read what was captured.

Layout::

    <root>/
      manifest.json
      frames/scene_0000/rgb.png            uint8 HxWx3, RGB order
      frames/scene_0000/depth.npy          float32 HxW, optical-axis Z, METRES
      frames/scene_0000/camera_info.json   K, size, frame ids, stamp
      frames/scene_0000/tf.json            base <- camera only
      labels/scene_0000.json               SCORER ONLY: grasp_gt, object pose
      labels/scene_0000_mask.png           SCORER ONLY: GT instance mask
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

SCHEMA_VERSION = "1.0.0"

#: The only depth policy the A0 tests and the grasp library are valid for.
DEPTH_POLICY_IMAGE_PLANE = "distance_to_image_plane:optical_axis_z:metres"
#: Only permitted after the radial->optical-Z conversion is tested (PLAN.md 5.2.1).
DEPTH_POLICY_CONVERTED_RADIAL = "distance_to_camera:converted_to_optical_z:metres"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump_json(path: Path, obj: dict) -> None:
    """Deterministic JSON: sorted keys, fixed separators, trailing newline."""
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")


@dataclass
class Frame:
    """Sensor-side data. Contains **no** ground truth, by construction."""

    scene_id: str
    rgb: np.ndarray            # uint8 (H, W, 3)
    depth_m: np.ndarray        # float32 (H, W), optical-axis Z in metres
    K: np.ndarray              # (3, 3)
    width: int
    height: int
    stamp_ns: int
    T_base_cam: np.ndarray     # (4, 4) base <- optical
    base_frame_id: str = "base_link"
    camera_frame_id: str = "camera_optical_frame"


@dataclass
class Labels:
    """Ground truth. Reachable only through :func:`load_labels`."""

    scene_id: str
    stamp_ns: int
    gt_mask: np.ndarray                 # uint8 (H, W), non-zero = target
    grasp_gt_position: np.ndarray       # (3,) metres, BASE frame
    grasp_gt_open_axis: np.ndarray      # (3,) unit, BASE frame
    T_base_object: np.ndarray           # (4, 4)
    object_dims_m: tuple = (0.0, 0.0, 0.0)
    target_present: bool = True


@dataclass
class Manifest:
    schema_version: str = SCHEMA_VERSION
    seed: int = 0
    depth_policy: str = DEPTH_POLICY_IMAGE_PLANE
    capture_backend: str = "unknown"
    depth_quantile: float = 0.75
    notes: str = ""
    scenes: list = field(default_factory=list)   # [{scene_id, files:{rel: sha256}}]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "depth_policy": self.depth_policy,
            "capture_backend": self.capture_backend,
            "depth_quantile": self.depth_quantile,
            "notes": self.notes,
            "scenes": self.scenes,
        }


class DatasetWriter:
    """Writes scenes, then seals the dataset with a checksummed manifest."""

    def __init__(self, root: Path, seed: int, capture_backend: str,
                 depth_policy: str = DEPTH_POLICY_IMAGE_PLANE,
                 depth_quantile: float = 0.75, notes: str = ""):
        self.root = Path(root)
        (self.root / "frames").mkdir(parents=True, exist_ok=True)
        (self.root / "labels").mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest(
            seed=seed, depth_policy=depth_policy, capture_backend=capture_backend,
            depth_quantile=depth_quantile, notes=notes,
        )

    def write_scene(self, frame: Frame, labels: Labels) -> None:
        if frame.scene_id != labels.scene_id:
            raise ValueError("frame and labels disagree on scene_id")
        if frame.stamp_ns != labels.stamp_ns:
            raise ValueError("frame and labels disagree on stamp_ns")

        scene_dir = self.root / "frames" / frame.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(frame.rgb.astype(np.uint8)).save(scene_dir / "rgb.png")
        np.save(scene_dir / "depth.npy", frame.depth_m.astype(np.float32))
        _dump_json(scene_dir / "camera_info.json", {
            "K": [[float(v) for v in row] for row in np.asarray(frame.K)],
            "width": int(frame.width),
            "height": int(frame.height),
            "distortion_model": "plumb_bob",
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "stamp_ns": int(frame.stamp_ns),
            "frame_id": frame.camera_frame_id,
        })
        _dump_json(scene_dir / "tf.json", {
            "stamp_ns": int(frame.stamp_ns),
            "transforms": [{
                "parent_frame_id": frame.base_frame_id,
                "child_frame_id": frame.camera_frame_id,
                "matrix": [[float(v) for v in row] for row in np.asarray(frame.T_base_cam)],
            }],
        })

        Image.fromarray((labels.gt_mask > 0).astype(np.uint8) * 255).save(
            self.root / "labels" / f"{labels.scene_id}_mask.png"
        )
        _dump_json(self.root / "labels" / f"{labels.scene_id}.json", {
            "scene_id": labels.scene_id,
            "stamp_ns": int(labels.stamp_ns),
            "grasp_gt": {
                "position_base": [float(v) for v in labels.grasp_gt_position],
                "open_axis_base": [float(v) for v in labels.grasp_gt_open_axis],
            },
            "T_base_object": [[float(v) for v in row] for row in np.asarray(labels.T_base_object)],
            "object_dims_m": [float(v) for v in labels.object_dims_m],
            "target_present": bool(labels.target_present),
        })

        rel = [
            f"frames/{frame.scene_id}/rgb.png",
            f"frames/{frame.scene_id}/depth.npy",
            f"frames/{frame.scene_id}/camera_info.json",
            f"frames/{frame.scene_id}/tf.json",
            f"labels/{labels.scene_id}.json",
            f"labels/{labels.scene_id}_mask.png",
        ]
        self.manifest.scenes.append({
            "scene_id": frame.scene_id,
            "stamp_ns": int(frame.stamp_ns),
            "files": {r: sha256_file(self.root / r) for r in rel},
        })

    def seal(self) -> Path:
        path = self.root / "manifest.json"
        _dump_json(path, self.manifest.to_dict())
        return path


# --------------------------------------------------------------------------
# Readers. load_frame() is the ONLY loader Branch B is permitted to call.
# --------------------------------------------------------------------------


def load_manifest(root: Path) -> dict:
    return json.loads((Path(root) / "manifest.json").read_text())


def scene_ids(root: Path) -> list:
    return [s["scene_id"] for s in load_manifest(root)["scenes"]]


def load_frame(root: Path, scene_id: str) -> Frame:
    """Sensor data only. There is no code path from here to ground truth."""
    root = Path(root)
    scene_dir = root / "frames" / scene_id
    info = json.loads((scene_dir / "camera_info.json").read_text())
    tf = json.loads((scene_dir / "tf.json").read_text())
    t = tf["transforms"][0]
    return Frame(
        scene_id=scene_id,
        rgb=np.asarray(Image.open(scene_dir / "rgb.png").convert("RGB"), dtype=np.uint8),
        depth_m=np.load(scene_dir / "depth.npy").astype(np.float64),
        K=np.asarray(info["K"], dtype=np.float64),
        width=int(info["width"]),
        height=int(info["height"]),
        stamp_ns=int(info["stamp_ns"]),
        T_base_cam=np.asarray(t["matrix"], dtype=np.float64),
        base_frame_id=t["parent_frame_id"],
        camera_frame_id=t["child_frame_id"],
    )


def load_labels(root: Path, scene_id: str) -> Labels:
    """SCORER AND ORACLE (A1) ONLY. Never call this from Branch B."""
    root = Path(root)
    data = json.loads((root / "labels" / f"{scene_id}.json").read_text())
    mask = np.asarray(Image.open(root / "labels" / f"{scene_id}_mask.png").convert("L"))
    return Labels(
        scene_id=scene_id,
        stamp_ns=int(data["stamp_ns"]),
        gt_mask=(mask > 127).astype(np.uint8),
        grasp_gt_position=np.asarray(data["grasp_gt"]["position_base"], dtype=np.float64),
        grasp_gt_open_axis=np.asarray(data["grasp_gt"]["open_axis_base"], dtype=np.float64),
        T_base_object=np.asarray(data["T_base_object"], dtype=np.float64),
        object_dims_m=tuple(data["object_dims_m"]),
        target_present=bool(data["target_present"]),
    )


def verify_checksums(root: Path) -> list:
    """Re-hash every file. Returns the list of paths that no longer match."""
    root = Path(root)
    bad = []
    for scene in load_manifest(root)["scenes"]:
        for rel, expected in scene["files"].items():
            path = root / rel
            if not path.exists() or sha256_file(path) != expected:
                bad.append(rel)
    return bad


def directory_size_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())
