# M0 — Environment record and deployment-shape decision

Per design doc §0.1 (compute gate). Recorded 2026-08-07, before any Isaac ROS
implementation work.

## Environment record

| Field | Value |
|---|---|
| GPU model | NVIDIA RTX 5000 Ada Generation Laptop GPU |
| GPU VRAM | **16376 MiB (16 GB)** |
| System RAM | 62 GiB |
| Driver version | 580.173.02 (pinned to 580.x — driver 595 crashes Isaac Sim 5.1's rtx.scenedb plugin, see memory/known-issue record) |
| CUDA version | 13.0 (driver-reported) |
| OS | Ubuntu 24.04.4 LTS |
| Isaac Sim version | 5.1.0.0 (pip install, ~/isaaclab-venv, python 3.11) |
| Isaac ROS tag | release-4.5 (Jazzy / Ubuntu 24.04) — image digest recorded below once pulled |
| ROS_DOMAIN_ID | 42 |
| Single-host or split-host | Single host (see decision) |
| Docker | 29.x, user in docker group, nvidia-container-toolkit working (GPU passthrough verified) |
| No passwordless sudo | All ROS-side work runs in the Isaac ROS dev container; any host `sudo` step is surfaced to Adithya to run manually |

## Deployment-shape decision (DDR-001)

**Chosen shape: phased development on a single 16 GB host.** This machine is
below BOTH the doc's 25 GB requirement and its 24 GB "phased" example, so the
phased order is mandatory, with hard checkpoints:

1. **Stage 1 — articulation + cuMotion** (M5/M6): measure peak VRAM after the
   100-trial pose-to-pose gate.
2. **Stage 2 — + robot segmentation + nvblox** (M8): measure again. First
   relief valve: voxel_size_m 0.01 → 0.02, reduced camera resolution.
3. **Stage 3 — + FoundationPose** (M10): highest risk. If it does not fit
   alongside Isaac Sim, the recorded fallback applies.

**Fallback (pre-committed, per doc option 4):** perception (FoundationPose)
moves to real-camera-only operation; simulation stays at P1 ground-truth
object poses. Second fallback: split deployment (Isaac Sim host + ROS host)
per doc option 3, with Isaac Sim as the single `/clock` owner and every ROS
sim node running `use_sim_time:=true` — that clock rule applies in ALL shapes.

Peak-VRAM-per-launch-profile is a required metric in the final M14 report.

## Kinematic-model gate (doc §0.2) — status

Resolved by prior work (the sim-only grasping track): the canonical DM
with-gripper URDF is validated in Isaac Sim (import probe, teleop, full
perception-driven pick), TCP calibrated at [-41.763, 0.008, 3.427] mm in the
EE frame. The doc still requires the records — KDR-001 is milestone M3′ and
MUST be produced before grasp libraries are authored (task #3).
