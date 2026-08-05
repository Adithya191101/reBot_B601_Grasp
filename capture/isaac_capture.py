#!/usr/bin/env python3
"""Isaac Sim capture backend for the ten smoke scenes.

Implements the same scene plan as :mod:`grasp_smoke.capture`, so datasets from
either backend are scored by identical code. Run with Isaac Sim's interpreter::

    ~/isaaclab-venv/bin/python capture/isaac_capture.py --out artifacts/smoke_isaac/dataset

**Depth fallback ladder (PLAN.md 5.2.1), implemented literally:**

1. ``distance_to_image_plane`` -- already optical-axis Z, which is exactly what
   the pinhole back-projection assumes. Preferred, and the default here.
2. ``--headful`` if it stalls. The documented stall is a headless symptom, so
   keeping the annotator and losing headless is the cheaper trade.
3. ``--allow-radial-depth`` last: ``distance_to_camera`` converted by
   ``Z = r / sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)``. Unit-tested in
   ``tests/test_depth_conversion.py``. Feeding raw radial range into a pinhole
   model gives a smooth, plausible, entirely wrong depth field that degrades
   toward the image edges.

**Measured on this machine 2026-08-04: rung 1 works headless, ~0.00 s per
annotator read across 10 scenes. The stall did not reproduce.**

Two Isaac-specific traps this script encodes, both found the hard way:

* **Near clip.** A USD camera defaults to a 1 m near plane. With a 0.6 m working
  distance the entire scene sits inside it: depth comes back all ``inf`` and
  segmentation all-background, while ``idToLabels`` still lists the prims -- so
  it looks like an annotator bug rather than a camera bug. ``clippingRange`` is
  set explicitly below.
* **Visual, not dynamic, geometry.** A ``DynamicCuboid`` falls under gravity
  between steps and its pose stops matching the authored ``grasp_gt``. Perception
  captures want ``VisualCuboid``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ANNOTATOR_TIMEOUT_S = 90.0
TARGET_PRIM = "/World/target"


def radial_range_to_optical_z(radial, K):
    """``distance_to_camera`` (Euclidean range) -> optical-axis Z.

    ``Z = r / sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)``
    """
    import numpy as np

    radial = np.asarray(radial, dtype=np.float64)
    h, w = radial.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    uu, vv = np.meshgrid(
        np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5
    )
    return radial / np.sqrt(1.0 + ((uu - cx) / fx) ** 2 + ((vv - cy) / fy) ** 2)


def _quat_wxyz_from_matrix(R):
    """Isaac wants (w, x, y, z); our helper returns (x, y, z, w)."""
    from grasp_smoke.geometry import quaternion_from_matrix

    x, y, z, w = quaternion_from_matrix(R)
    return [float(w), float(x), float(y), float(z)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--headful", action="store_true", help="fallback rung 2")
    ap.add_argument("--allow-radial-depth", action="store_true", help="fallback rung 3")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    # Isaac redirects stdout into the Kit logger once SimulationApp starts, so
    # progress goes to a file that survives.
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out.parent / "isaac_progress.log"
    log_path.write_text("")

    def log(msg: str) -> None:
        with log_path.open("a") as fh:
            fh.write(msg + "\n")

    stage = "import"
    try:
        from isaacsim import SimulationApp
    except Exception as exc:                                   # noqa: BLE001
        log(f"FAILED {stage}: {type(exc).__name__}: {exc}")
        return 2

    t0 = time.perf_counter()
    sim_app = SimulationApp({"headless": not args.headful,
                             "width": args.width, "height": args.height})
    log(f"SimulationApp up in {time.perf_counter()-t0:.1f}s (headless={not args.headful})")

    try:
        import numpy as np
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom, Vt

        import isaacsim.core.utils.prims as prim_utils
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import GroundPlane, VisualCuboid

        from grasp_smoke.capture import (
            BASE_STAMP_NS, STAMP_STEP_NS, VramSampler, plan_smoke_scenes,
        )
        from grasp_smoke.dataset import (
            DEPTH_POLICY_CONVERTED_RADIAL, DEPTH_POLICY_IMAGE_PLANE,
            DatasetWriter, Frame, Labels, directory_size_bytes,
        )
        from grasp_smoke.geometry import make_intrinsics
        from grasp_smoke.render import randomize_scene

        stage = "world"
        world = World(stage_units_in_meters=1.0)
        world.scene.add(GroundPlane(prim_path="/World/ground", size=4.0))

        # Without lights the RTX renderer returns an all-black RGB buffer while
        # depth and segmentation still look perfectly healthy -- so the failure
        # surfaces downstream as "the detector finds nothing" rather than as a
        # rendering error. An empty World() has no default lighting.
        prim_utils.create_prim(
            "/World/dome_light", "DomeLight",
            attributes={"inputs:intensity": 300.0, "inputs:color": (1.0, 1.0, 1.0)},
        )
        prim_utils.create_prim(
            "/World/key_light", "DistantLight",
            attributes={"inputs:intensity": 800.0, "inputs:angle": 1.0},
            orientation=np.array([0.9239, 0.0, 0.3827, 0.0]),
        )
        box = world.scene.add(VisualCuboid(
            prim_path=TARGET_PRIM, name="target",
            position=np.array([0.0, 0.0, 0.02]),
            scale=np.array([0.12, 0.06, 0.04]),
            color=np.array([0.85, 0.35, 0.25]),
        ))

        stage = "camera"
        camera_path = "/World/capture_cam"
        prim_utils.create_prim(camera_path, "Camera")
        cam_prim = prim_utils.get_prim_at_path(camera_path)
        # Without this the whole scene sits inside the default 1 m near plane.
        cam_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 1000.0))

        render_product = rep.create.render_product(camera_path, (args.width, args.height))
        depth_name = "distance_to_camera" if args.allow_radial_depth else "distance_to_image_plane"
        annots = {
            "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
            "depth": rep.AnnotatorRegistry.get_annotator(depth_name),
            "seg": rep.AnnotatorRegistry.get_annotator("instance_id_segmentation"),
        }
        for a in annots.values():
            a.attach(render_product)
        log(f"annotators attached (depth={depth_name})")

        world.reset()

        stage = "capture"
        writer = DatasetWriter(
            root=args.out, seed=args.seed, capture_backend="isaac_sim_5.1",
            depth_policy=(DEPTH_POLICY_CONVERTED_RADIAL if args.allow_radial_depth
                          else DEPTH_POLICY_IMAGE_PLANE),
            depth_quantile=0.75,
            notes=f"Isaac Sim 5.1 Replicator capture; depth annotator={depth_name}; "
                  f"headless={not args.headful}",
        )

        specs = plan_smoke_scenes(args.seed)
        per_scene, annot_reads, stalled = [], [], None

        with VramSampler() as vram:
            t_all = time.perf_counter()
            for i, spec in enumerate(specs):
                t_scene = time.perf_counter()
                target, T_base_cam, K_want = randomize_scene(spec.seed, tilt_deg=spec.tilt_deg)
                K = make_intrinsics(K_want[0, 0], K_want[0, 0],
                                    (args.width - 1) / 2.0, (args.height - 1) / 2.0)

                half = np.asarray(target.half_extents, dtype=np.float64)
                # T_base_object IS the box centre and already includes the
                # rest-on-ground offset (see render.randomize_scene). Adding
                # half_z again here would shift the geometry half a box height
                # away from the authored grasp_gt.
                if spec.target_present:
                    pos = target.T_base_object[:3, 3]
                else:
                    pos = np.array([0.0, 0.0, -50.0])   # out of frame, physics-free
                box.set_world_pose(
                    position=pos,
                    orientation=np.array(_quat_wxyz_from_matrix(target.T_base_object[:3, :3])),
                )
                box.set_local_scale(2.0 * half)
                # VisualCuboid exposes no set_color; drive the USD primvar so the
                # target is chromatic rather than default grey (Branch B keys on
                # saturation, so a grey target is a segmentation trap).
                UsdGeom.Gprim(prim_utils.get_prim_at_path(TARGET_PRIM)) \
                    .CreateDisplayColorAttr().Set(
                        Vt.Vec3fArray([Gf.Vec3f(*[float(c) for c in target.base_color])]))

                # USD cameras look along -Z with +Y up; the optical frame is +Z
                # forward, +Y down. Convert before writing the transform.
                R_usd = T_base_cam[:3, :3] @ np.diag([1.0, -1.0, -1.0])
                xform = UsdGeom.Xformable(cam_prim)
                xform.ClearXformOpOrder()
                m = Gf.Matrix4d()
                m.SetIdentity()
                m.SetRotateOnly(Gf.Matrix3d(*[float(v) for v in R_usd.T.flatten()]))
                m.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in T_base_cam[:3, 3]]))
                xform.AddTransformOp().Set(m)
                cam_prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 1000.0))

                # Make the renderer's focal length agree with the K we will later
                # back-project with; otherwise the two silently disagree.
                aperture = float(cam_prim.GetAttribute("horizontalAperture").Get() or 20.955)
                cam_prim.GetAttribute("focalLength").Set(
                    float(K[0, 0]) * aperture / float(args.width))

                for _ in range(2):
                    rep.orchestrator.step(rt_subframes=8)

                t_read = time.perf_counter()
                rgb_raw = annots["rgb"].get_data()
                depth_raw = annots["depth"].get_data()
                seg_raw = annots["seg"].get_data()
                read_s = time.perf_counter() - t_read
                annot_reads.append(read_s)
                if read_s > ANNOTATOR_TIMEOUT_S:
                    stalled = {"scene": spec.scene_id, "annotator": depth_name,
                               "seconds": round(read_s, 2)}
                    break

                rgb = np.asarray(rgb_raw)[..., :3].astype(np.uint8)
                depth = np.asarray(depth_raw, dtype=np.float64)
                depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                depth[(depth <= 0.0) | (depth > 50.0)] = 0.0
                if args.allow_radial_depth:
                    depth = radial_range_to_optical_z(depth, K)

                seg_data = np.asarray(seg_raw["data"])
                if seg_data.ndim == 3:
                    seg_data = seg_data[..., 0]
                target_ids = [
                    int(k) for k, v in (seg_raw.get("info", {}).get("idToLabels", {}) or {}).items()
                    if TARGET_PRIM in str(v)
                ]
                mask = np.isin(seg_data, target_ids).astype(np.uint8) if target_ids \
                    else np.zeros(seg_data.shape, np.uint8)

                if spec.target_present and not mask.any():
                    stalled = {"scene": spec.scene_id,
                               "error": "target not visible in instance segmentation",
                               "ids_seen": sorted(int(v) for v in np.unique(seg_data))}
                    break

                gt_pos, gt_axis = target.grasp_gt()
                stamp = BASE_STAMP_NS + i * STAMP_STEP_NS
                writer.write_scene(
                    Frame(scene_id=spec.scene_id, rgb=rgb, depth_m=depth.astype(np.float32),
                          K=K, width=args.width, height=args.height, stamp_ns=stamp,
                          T_base_cam=T_base_cam),
                    Labels(scene_id=spec.scene_id, stamp_ns=stamp, gt_mask=mask,
                           grasp_gt_position=gt_pos, grasp_gt_open_axis=gt_axis,
                           T_base_object=target.T_base_object,
                           object_dims_m=target.dims_m,
                           target_present=spec.target_present),
                )
                per_scene.append(time.perf_counter() - t_scene)
                log(f"{spec.scene_id} ok  scene={per_scene[-1]:.2f}s annot={read_s:.3f}s "
                    f"mask_px={int(mask.sum())} depth=[{depth[depth>0].min():.3f},"
                    f"{depth.max():.3f}]" if mask.any() or not spec.target_present
                    else f"{spec.scene_id} ok (absent)")
            total_s = time.perf_counter() - t_all

        if stalled:
            log("STALLED/FAILED: " + json.dumps(stalled))
            (args.out.parent / "isaac_failure.json").write_text(json.dumps({
                "stalled": stalled,
                "remedy": "rerun with --headful, then --allow-radial-depth",
            }, indent=2) + "\n")
            return 3

        writer.seal()
        size_b = directory_size_bytes(args.out)
        n = max(len(per_scene), 1)
        stats = {
            "backend": "isaac_sim_5.1",
            "n_scenes": len(per_scene),
            "total_seconds": round(total_s, 3),
            "seconds_per_scene": round(total_s / n, 3),
            "seconds_per_scene_max": round(max(per_scene), 3),
            "annotator_read_seconds_max": round(max(annot_reads), 4),
            "bytes_total": size_b,
            "bytes_per_scene": int(size_b / n),
            "peak_vram_mib": vram.peak_mib,
            "depth_annotator": depth_name,
            "headless": not args.headful,
            "extrapolated_300": {
                "minutes": round(total_s / n * 300 / 60.0, 2),
                "gigabytes": round(size_b / n * 300 / 1e9, 3),
            },
        }
        (args.out / "capture_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        log("DONE " + json.dumps(stats))
        return 0

    except Exception as exc:                                   # noqa: BLE001
        import traceback
        log(f"FAILED stage={stage}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        return 4
    finally:
        try:
            sim_app.close()
        except Exception:                                      # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
