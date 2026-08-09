# DDR-002 — Camera architecture (PROVISIONAL)

**Status: PROVISIONAL** — recorded 2026-08-09 for milestone M8; the user
may revise before any hardware purchase or real-camera bring-up.
Supersedes nothing; constrains M8+ sim work and the M10 perception stack.

## Decision

Adopt the design doc's recommended two-camera cell (sec. 15.1):

| Camera | Role |
|---|---|
| **fixed overhead** (RealSense-class) | stable workspace depth, robot segmentation + nvblox mapping, coarse ROI/detection |
| **wrist D405** | close-range pose estimation and grasp verification |

**In simulation, the FIXED OVERHEAD camera is implemented first** (M8).
The wrist camera is deferred to the perception milestones: mapping wants a
stationary, always-valid view of the whole cell (a wrist camera swings
away during approach and creates moving-camera occlusion, per the doc),
and every M8 acceptance criterion (robot-masked map, gantry persistence,
stale-map clearing, ESDF-driven re-routing) is provable with the overhead
view alone.

## Sim implementation (M8, `scripts/b601_sim_bridge.py --m8-scene`)

* **Placement:** eye at **(0.275, 0.0, 1.10) m in `base_link`**, looking
  straight down (camera prim world orientation = identity quaternion —
  the recorded topdown-camera convention). Directly above the workspace
  centre and 0.45 m ABOVE the nvblox workspace ceiling (z max 0.65), so
  the camera body can never enter the map.
* **Optics:** 18.15 mm focal / 24 mm horizontal aperture (≈67° HFOV,
  D435-class, the rearm scene-camera values), 640×480 @ ~15 Hz
  (`frameSkipCount=3` on a 60 FPS render loop). Footprint on the table
  plane: x ∈ [−0.45, 1.00], y ∈ [−0.55, 0.55] — the whole mapped cell
  (x [−0.10, 0.65], y [−0.35, 0.35]) with margin.
* **Frames:** images and camera_info carry
  `overhead_camera_color_optical_frame` (ROS optical convention). The
  prim pair mirrors the proven wrist-camera construction from the rearm
  environment: an optical-frame Xform parent + a child Camera prim with
  the 180°-about-X flip. TF `base_link →
  overhead_camera_color_optical_frame` is published on `/tf_static` by
  the bridge (ROS2PublishRawTransformTree, staticPublisher), translation
  (0.275, 0, 1.10), rotation 180° about X — verified numerically by the
  robot segmenter's logged camera pose (diag(1,−1,−1)).
* **Topics** (doc 13.4 shape, aligned-depth naming so the real
  RealSense driver drops in unchanged; frozen in
  `config/sim_topics.yaml`):
  - `/overhead_camera/color/image_raw`
  - `/overhead_camera/color/camera_info`
  - `/overhead_camera/aligned_depth_to_color/image_raw` (32FC1, metres)
  - `/overhead_camera/aligned_depth_to_color/camera_info`
* **Pipeline:** IsaacCreateRenderProduct + ROS2CameraHelper /
  ROS2CameraInfoHelper OmniGraph nodes — the supported bridge path. The
  raw `distance_to_image_plane` annotator is NOT used: it hangs in
  headless snapshot contexts (recorded repo memory); the CameraHelper
  `type: depth` stream is the same DistanceToImagePlane range image
  delivered through the SDG pipeline, which is what the robot segmenter
  expects.
* Depth ground truth was validated pixel-by-pixel at bring-up: table top
  1.100 m, gantry top 0.860 m, floor 1.300 m, intrinsics fx=fy=484,
  cx=320, cy=240 — all exact.

## Real-hardware notes (for the revision pass)

* Mount the overhead camera OUTSIDE the mapped workspace, rigid to the
  cell frame; calibrate `cell_frame → camera` with the NVIDIA multi-pose
  hand-eye procedure (doc 15.2) — the sim's exact static TF is the
  stand-in for that calibration result.
* The aligned-depth topic names match the RealSense ROS driver's
  `align_depth.enable:=true` output, so the sim-vs-real swap is a
  namespace/launch change, not a topic-contract change.
* Wrist D405: still recommended for M10+ grasp verification; unmodelled
  in sim until KDR-001's wrist_camera_mount is authored (XRDF note).

## Consequences

* nvblox maps ONLY what the overhead camera sees: the gantry is a
  camera-facing shell (top + visible side faces); its occluded interior
  stays unknown. M8's mesh-verified clearance gates cover this: the
  planned path must clear the TRUE bar box, not just the mapped shell.
* A single fixed view means the arm shadows parts of the table while it
  moves; the robot segmenter turns those pixels into unknown (never
  obstacle) space, which is the doc-intended conservative behavior.
* Anything under the table plane is invisible; the workspace floor at
  z = −0.05 bounds the map from below.
