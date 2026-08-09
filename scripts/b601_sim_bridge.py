#!/usr/bin/env python3
"""M5 sim side: headless Isaac Sim 5.1 bridge for the reBot B601-DM.

Run on the HOST with the pip Isaac Sim interpreter (prefer
``scripts/m5_parity_test.sh``, which sets all of this up)::

    TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N \
    ROS_DISTRO=jazzy ROS_DOMAIN_ID=42 \
    LD_LIBRARY_PATH=~/isaaclab-venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib \
      ~/isaaclab-venv/bin/python scripts/b601_sim_bridge.py --duration 600

LD_LIBRARY_PATH must point at the bridge's bundled Jazzy libs BEFORE the
process starts (the loader does not re-read it): without it the extension
fails with "librmw_implementation.so ... libament_index_cpp.so: cannot open
shared object file".  Same fix as the rearm env's isaac_sim_env.sh.

The ROS side (adapters + shim + planner) runs in the ``rebot-jazzy-baseline``
container with ``--network host`` (see ``scripts/m5_parity_test.sh``).

BASIS: the ready-made rearm Isaac environment
(``assets/rearm_isaac_scene/rebot_isaac_ws/``).  Per the recorded direction
change, the M5 sim side reuses that environment instead of building the scene
from the shipped vendor USD:

* **its USD** ``usd/rebot_b601dm.usd`` -- a URDF-importer product with 24/24
  import checks green (metric units, one ArticulationRootAPI on
  ``/rebot_b601dm/root_joint``, joint limits matching the URDF, a WORKING
  PhysX mimic on gripper_joint2, decimated-hull colliders, gripper_tcp
  authored).  Unlike the shipped vendor USD this needs no nested-Xform repair
  and no runtime PD-gain injection -- the importer authored working drives.
* **its scene bootstrap** (``sim/pick_scene.py``), from which the following
  proven decisions are ported verbatim, each guarding a measured failure:
  - the articulation root is DISCOVERED by ArticulationRootAPI, not
    hardcoded (it is ``/rebot_b601dm/root_joint``, NOT ``/rebot_b601dm``;
    the wrong prim makes every graph tick abort, silencing even /clock);
  - arm max-effort caps raised to 1000 N.m via the physics view (the
    imported 27 N.m caps are BELOW the static gravity torque of the
    extended arm -- the pose sags no matter how stiff the drive);
  - state AND drive-target are both seeded, with a re-asserting warmup
    (seeding state alone lets the arm fall before the target registers);
  - after warmup the graph's IsaacArticulationController is the SOLE writer
    of drive targets -- a Python ``apply_action`` per step silently overrides
    every ROS command in the same tick (measured there: closed loop dead);
  - the timeline must be PLAYING and every tick must be a RENDERED step
    (OnPlaybackTick fires from the render loop; ``step(render=False)`` would
    advance physics while publishing nothing).

TOPIC CONTRACT (config/sim_topics.yaml, updated by recorded decision):
  PUBLISH   /clock                  (sole owner, 60 Hz)
            /isaac_joint_states     (sensor_msgs/JointState, all 8 joints)
  SUBSCRIBE /isaac_joint_commands   (sensor_msgs/JointState position targets,
                                     applied to the named DOFs each tick)

The original frozen names (/rebot_sim/joint_states_raw and a
FollowJointTrajectory sim controller) are re-pointed at the rearm
environment's names: the JointState-command closed loop is what that
environment ships and verified, and the bridge has no ActionGraph JTC node
headless.  FollowJointTrajectory semantics are provided container-side by
``rebot_sim_bridge``'s sim-JTC shim, which interpolates FJT goals onto
/isaac_joint_commands at the sim rate.  The adapter-facing FJT action names
are unchanged.

DELIBERATE OMISSIONS versus pick_scene.py (M5 is articulation parity only):
no cameras, no /tf, no soup can, no cosmetic environment -- smallest VRAM
footprint (16 GB is the binding constraint, DDR-001 stage 1 must be
measured lean).

M8 EXTENSION (``--m8-scene``, default OFF so M5/M6/M7 behave bit-identically):
per docs/DDR-002-camera-architecture.md (PROVISIONAL) a FIXED OVERHEAD
RGB-D camera + the visible cell geometry are added for the robot-
segmentation + nvblox mapping milestone:

* visible cell geometry (VISUAL ONLY, no colliders -- cuMotion owns
  avoidance; a collider could deflect the arm in a failed plan): the
  cell_geometry.yaml table slab at true z<=0, the gantry crossbar box,
  and a large floor slab well below the nvblox workspace so every
  downward depth ray terminates (clean space-carving);
* an overhead camera at OVERHEAD_CAM_EYE looking straight down, authored
  as pick_scene.py's wrist-camera construction: parent Xform
  ``overhead_camera_color_optical_frame`` in ROS optical convention
  (+Z view), child Camera prim with the 180-deg X flip (identity world
  orientation = straight down per the topdown-camera memory note);
* RGB + depth + camera_info via IsaacCreateRenderProduct +
  ROS2CameraHelper/ROS2CameraInfoHelper (the supported bridge path --
  the raw ``distance_to_image_plane`` annotator HANGS in headless
  snapshot contexts, recorded repo memory), throttled by frameSkipCount;
* TF ``base_link -> overhead_camera_color_optical_frame`` via
  ROS2PublishRawTransformTree on /tf_static (staticPublisher), so the
  robot segmenter and nvblox can locate the depth stream;
* a polled scene-command file (``--scene-cmd-file``, JSON
  ``{"gantry_visible": bool}``) toggling the gantry prim's visibility --
  the mapping acceptance test removes/restores the REAL obstacle and
  watches the ESDF clear/rebuild (doc 15.4 stale-map criterion).

rclpy is NEVER imported here (repo process rule): the OmniGraph nodes
publish/subscribe through the bridge's bundled C++ Jazzy libraries, which is
why no system ROS needs to be sourced on the host.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Argument parsing BEFORE SimulationApp (it consumes sys.argv).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD = (REPO_ROOT / "assets" / "rearm_isaac_scene" / "rebot_isaac_ws"
               / "usd" / "rebot_b601dm.usd")

#: M5 parity start pose: vendor end-pose (0.28, -0.10, 0.20, pitch 0.9)
#: solved through the vendor's own IK on its driver model (end_link frame,
#: reBot-DevArm_fixend.urdf).  Same orientation family as the vendor ready
#: pose, so the Pinocchio planner's rotate-then-translate scheme (built for
#: pick moves, guarded against shoulder branch changes) applies; verified
#: offline: plan + collision gates pass from here to the ready pose.
START_Q_ARM = [-0.5420, -1.1215, -1.0309, 0.8831, -0.3264, -0.4406]

_ap = argparse.ArgumentParser(
    description=__doc__.split("\n\n")[0],
    formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--duration", type=float, default=600.0,
                 help="wall-clock seconds to serve before a clean exit")
_ap.add_argument("--gui", action="store_true",
                 help="open a viewport (default headless)")
_ap.add_argument("--usd", default=str(DEFAULT_USD),
                 help="robot USD (default: the rearm environment USD)")
_ap.add_argument("--start-q", type=float, nargs=6, default=START_Q_ARM,
                 metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
                 help="arm start pose, radians (default: M5 parity start)")
_ap.add_argument("--arm-gain-scale", type=float, default=5.0,
                 help="runtime multiplier on the imported arm drive "
                      "stiffness (damping x sqrt(scale)); the importer's "
                      "gains leave ~0.018 rad of static gravity droop, too "
                      "close to the 0.02 rad M5 parity gate")
_ap.add_argument("--no-realtime", action="store_true",
                 help="disable the 60 FPS wall-time rate limiter (sim then "
                      "free-runs; the container's sim-time timeouts shrink "
                      "in wall terms -- debug only)")
_ap.add_argument("--ready-file", default="",
                 help="write a small JSON marker here once the bridge serves")
_ap.add_argument("--m8-scene", action="store_true",
                 help="M8: add the visible cell geometry (table/gantry/floor), "
                      "the fixed overhead RGB-D camera + camera_info + "
                      "/tf_static, and the scene-command file poller "
                      "(docs/DDR-002-camera-architecture.md)")
_ap.add_argument("--scene-cmd-file",
                 default=str(REPO_ROOT / "artifacts" / "m8" /
                             "scene_cmd.json"),
                 help="M8 scene-command JSON polled each second "
                      "({\"gantry_visible\": bool}); the acceptance runner "
                      "writes it through the shared /work mount")
_ap.add_argument("--cam-skip", type=int, default=3,
                 help="frameSkipCount for the camera helpers (3 -> 15 Hz "
                      "publish at the 60 FPS render rate)")
_args, _unknown = _ap.parse_known_args()

# ---------------------------------------------------------------------------
# Environment guards (adapted from the rearm env's isaac_sim_env.sh).
# * ROS_DISTRO=jazzy: on this Ubuntu 24.04 host the bridge's system_default
#   already resolves to jazzy, but the rearm track's silent-failure lesson
#   (bridge on one distro, container on the other, nothing errors) makes the
#   explicit check worth keeping.
# * RMW: default FastDDS on BOTH sides (M5 brief).  The rearm env used
#   CycloneDDS because its container did; ours does not -- recorded diff.
# * CUDA MPS bypass: only needed when an MPS control pipe exists; harmless
#   and defensive here (the rearm host hung forever in Kit device init).
# ---------------------------------------------------------------------------
os.environ.setdefault("ROS_DISTRO", "jazzy")
if os.environ["ROS_DISTRO"] != "jazzy":
    sys.exit("ERROR: ROS_DISTRO=%r, the Jazzy container cannot read another "
             "distro's types" % os.environ["ROS_DISTRO"])
os.environ.setdefault("ROS_DOMAIN_ID", "42")
if os.path.exists("/tmp/nvidia-mps"):
    os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/isaacsim-no-mps")

USD_PATH = str(Path(_args.usd).expanduser().resolve())
if not Path(USD_PATH).is_file():
    sys.exit("ERROR: robot USD not found at %s" % USD_PATH)

ROBOT_PRIM = "/rebot_b601dm"
TOPIC_CLOCK = "/clock"
TOPIC_JOINT_STATES = "/isaac_joint_states"
TOPIC_JOINT_COMMANDS = "/isaac_joint_commands"
ARM_JOINTS = ["joint%d" % i for i in range(1, 7)]
JAW_JOINTS = ["gripper_joint1", "gripper_joint2"]
JAW_OPEN_M = 0.0715
PHYSICS_HZ = 60.0  # matches the USD's PhysicsScene timeStepsPerSecond

# --- M8 overhead camera + cell geometry (config/sim_topics.yaml m8 block,
# docs/DDR-002-camera-architecture.md).  All poses in base_link == world
# (the robot root is fixed at the origin).
CAM_OPTICAL_PRIM = "/World/overhead_camera_color_optical_frame"
CAM_PRIM = CAM_OPTICAL_PRIM + "/rgbd_camera"
CAM_FRAME_ID = "overhead_camera_color_optical_frame"
#: Above the workspace centre (x span -0.10..0.65), OUTSIDE the nvblox
#: workspace box (z max 0.65) so the camera body can never be mapped.
#: At 1.10 m with the 67x53 deg FOV below, the z=0 footprint is
#: x in [-0.45, 1.00], y in [-0.55, 0.55]: the whole mapped cell is seen.
CAM_EYE = (0.275, 0.0, 1.10)
CAM_W, CAM_H = 640, 480
CAM_FOCAL_MM = 18.15          # ~67 deg HFOV on a 24 mm aperture (rearm value)
CAM_APERTURE_MM = 24.0
TOPIC_CAM_RGB = "/overhead_camera/color/image_raw"
TOPIC_CAM_RGB_INFO = "/overhead_camera/color/camera_info"
TOPIC_CAM_DEPTH = "/overhead_camera/aligned_depth_to_color/image_raw"
TOPIC_CAM_DEPTH_INFO = "/overhead_camera/aligned_depth_to_color/camera_info"

#: Visible cell geometry, numerically identical to config/cell_geometry.yaml
#: (table slab; gantry via collision_core's _gantry_box conversion:
#: span x 0.20 m, bar 0.04x0.04 m, lower edge z=0.20 -> box centre z=0.22).
M8_TABLE = {"center": (0.30, 0.0, -0.025), "size": (0.60, 0.45, 0.05),
            "rgb": (0.42, 0.36, 0.30)}
M8_GANTRY = {"center": (0.45, 0.10, 0.22), "size": (0.20, 0.04, 0.04),
             "rgb": (0.85, 0.45, 0.10)}
#: Floor slab far below the nvblox workspace floor (z min -0.05) so every
#: downward depth ray terminates on real geometry -- nvblox then carves
#: free space cleanly instead of ignoring no-hit pixels.
M8_FLOOR = {"center": (0.0, 0.0, -0.21), "size": (4.0, 4.0, 0.02),
            "rgb": (0.18, 0.18, 0.20)}
GANTRY_PRIM = "/World/m8_scene/gantry"

# Hermetic Kit flags (pick_scene.py pattern): extensions are cached locally,
# remote registries are only latency.
sys.argv.extend([
    "--/app/extensions/registryEnabled=false",
    "--/persistent/app/omniverse/hubEnabled=false",
    "--/structuredLog/enable=false",
])
# Wall-lock /clock to 1x real time (MEASURED failure without this): the pip
# standalone app ships rateLimitEnabled=false, so SimulationContext skips its
# 60 FPS target and PhysX free-runs/catches-up -- observed sim time racing
# ~4x wall with multi-substep bursts.  Wall-bound DDS latencies (action
# discovery, goal round-trips) then consume the container's SIM-time budgets:
# the planner's `duration+8s` result deadline expired before the trajectory
# even started, and bursty stamps tripped the joint-state adapter's 0.2 s
# stale gate.  With the rate limiter on, SimulationContext(rendering_dt=1/60)
# locks the loop to 60 FPS wall with a fixed 1/60 manual step: sim == wall.
if not _args.no_realtime:
    sys.argv.extend([
        "--/app/runLoops/main/rateLimitEnabled=true",
        "--/app/runLoops/main/rateLimitFrequency=60",
    ])

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not _args.gui})

import numpy as np  # noqa: E402
import omni.graph.core as og  # noqa: E402
import omni.usd  # noqa: E402
import usdrt.Sdf  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()


def find_articulation_root(stage: Usd.Stage) -> str:
    """The prim carrying UsdPhysics.ArticulationRootAPI (pick_scene.py).

    NOT /rebot_b601dm: the URDF importer (fix_base=True) put the API on the
    generated root_joint.  The graph nodes resolve the articulation by EXACT
    prim path; the wrong prim aborts every graph tick, silencing /clock too.
    """
    roots = [str(p.GetPath()) for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        print("WARN: expected one ArticulationRootAPI prim, found %r" % roots,
              flush=True)
    if not roots:
        return ROBOT_PRIM
    return roots[0]


def build_action_graph(articulation_root: str) -> None:
    """Clock + joint-state publisher + the JointState-command closed loop.

    Verbatim subset of pick_scene.py's proven graph: SubscribeJointState's
    jointNames/position/velocity/effortCommand feed IsaacArticulationController
    on the articulation root, so a command on /isaac_joint_commands MOVES the
    articulation, which then reports on /isaac_joint_states.  The controller
    applies only the joints NAMED in the incoming message, and PhysX drive
    targets persist per-DOF, so arm-only and jaw-only publishers coexist.
    """
    og.Controller.edit(
        {"graph_path": "/SimBridgeGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("PublishJointState",
                 "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState",
                 "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController",
                 "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishClock.inputs:topicName", TOPIC_CLOCK),
                ("PublishJointState.inputs:topicName", TOPIC_JOINT_STATES),
                ("PublishJointState.inputs:targetPrim",
                 [usdrt.Sdf.Path(articulation_root)]),
                ("SubscribeJointState.inputs:topicName", TOPIC_JOINT_COMMANDS),
                ("ArticulationController.inputs:robotPath", articulation_root),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick",
                 "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick",
                 "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick",
                 "ArticulationController.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime",
                 "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime",
                 "PublishJointState.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames",
                 "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand",
                 "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand",
                 "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand",
                 "ArticulationController.inputs:effortCommand"),
            ],
        },
    )


def _m8_box(stage: Usd.Stage, path: str, spec: dict) -> None:
    """A colored VISUAL-ONLY box (pick_scene.py's _box, collision=False).

    No CollisionAPI on purpose: obstacle avoidance is cuMotion's job via
    the ESDF map; a PhysX collider here could deflect the arm on a failed
    plan and corrupt tracking-convergence verdicts.
    """
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*spec["rgb"])])
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*spec["center"]))
    xf.AddScaleOp().Set(Gf.Vec3f(*spec["size"]))


def build_m8_scene(stage: Usd.Stage) -> None:
    """Visible cell geometry + lights + the overhead camera prim pair.

    Camera construction is the wrist-camera pattern from the rearm
    pick_scene.py (its stated proven decision): a parent Xform authored in
    the ROS OPTICAL convention carries the frame the images/TF advertise,
    and the child Camera prim applies the 180-deg X flip so the USD camera
    (which views along its local -Z) looks along the parent's +Z.  For a
    straight-down view the child's WORLD orientation is then the identity
    quaternion -- the exact topdown-camera convention recorded in memory.
    """
    _m8_box(stage, "/World/m8_scene/floor", M8_FLOOR)
    _m8_box(stage, "/World/m8_scene/table", M8_TABLE)
    _m8_box(stage, GANTRY_PRIM, M8_GANTRY)

    # Lights (RGB stream only; depth is geometric): dim dome + distant key,
    # the rearm exposure recipe (a lone bright dome over-exposes).
    dome = UsdLux.DomeLight.Define(stage, "/World/m8_scene/dome_light")
    dome.CreateIntensityAttr(150.0)
    key = UsdLux.DistantLight.Define(stage, "/World/m8_scene/key_light")
    key.CreateIntensityAttr(1500.0)
    kxf = UsdGeom.Xformable(key)
    kxf.ClearXformOpOrder()
    kxf.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 20.0, 0.0))

    # Optical-frame Xform: x=+X world, y=-Y, z=-Z (view straight down)
    # == 180 deg about X.
    opt = UsdGeom.Xform.Define(stage, CAM_OPTICAL_PRIM)
    oxf = UsdGeom.Xformable(opt)
    oxf.ClearXformOpOrder()
    oxf.AddTranslateOp().Set(Gf.Vec3d(*CAM_EYE))
    oxf.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))

    cam = UsdGeom.Camera.Define(stage, CAM_PRIM)
    cam.CreateFocalLengthAttr(CAM_FOCAL_MM)
    cam.CreateHorizontalApertureAttr(CAM_APERTURE_MM)
    cam.CreateVerticalApertureAttr(CAM_APERTURE_MM * CAM_H / CAM_W)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 8.0))
    cxf = UsdGeom.Xformable(cam)
    cxf.ClearXformOpOrder()
    cxf.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))  # optical -> USD cam
    print("M8 SCENE: table/gantry/floor + overhead cam at %s (world "
          "orientation identity, straight down)" % (CAM_EYE,), flush=True)


def build_m8_camera_graph() -> None:
    """RGB + depth + 2x camera_info publishers on the overhead camera.

    Port of pick_scene.py's build_camera_graph (proven headless pattern):
    one IsaacCreateRenderProduct on the camera prim feeds
    ROS2CameraHelper (rgb), ROS2CameraHelper (depth ==
    DistanceToImagePlane, the range image the robot segmenter expects)
    and ROS2CameraInfoHelper nodes; a push/on-demand graph evaluated once
    so the SDG-pipeline publishers exist before the main loop steps.
    frameSkipCount throttles all four helpers identically so depth and
    camera_info stay stamp-synchronized for the segmenter's pairing.
    """
    skip = max(0, int(_args.cam_skip))
    keys = og.Controller.Keys
    (graph, _, _, _) = og.Controller.edit(
        {
            "graph_path": "/OverheadCamGraph",
            "evaluator_name": "push",
            "pipeline_stage":
                og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("RP", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("Rgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("InfoColor", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("InfoDepth", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("Depth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                ("RP.inputs:cameraPrim", [usdrt.Sdf.Path(CAM_PRIM)]),
                ("RP.inputs:width", CAM_W),
                ("RP.inputs:height", CAM_H),
                ("Rgb.inputs:frameId", CAM_FRAME_ID),
                ("Rgb.inputs:topicName", TOPIC_CAM_RGB),
                ("Rgb.inputs:type", "rgb"),
                ("Rgb.inputs:frameSkipCount", skip),
                ("InfoColor.inputs:frameId", CAM_FRAME_ID),
                ("InfoColor.inputs:topicName", TOPIC_CAM_RGB_INFO),
                ("InfoColor.inputs:frameSkipCount", skip),
                ("InfoDepth.inputs:frameId", CAM_FRAME_ID),
                ("InfoDepth.inputs:topicName", TOPIC_CAM_DEPTH_INFO),
                ("InfoDepth.inputs:frameSkipCount", skip),
                ("Depth.inputs:frameId", CAM_FRAME_ID),
                ("Depth.inputs:topicName", TOPIC_CAM_DEPTH),
                ("Depth.inputs:type", "depth"),
                ("Depth.inputs:frameSkipCount", skip),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "RP.inputs:execIn"),
                ("RP.outputs:execOut", "Rgb.inputs:execIn"),
                ("RP.outputs:execOut", "InfoColor.inputs:execIn"),
                ("RP.outputs:execOut", "InfoDepth.inputs:execIn"),
                ("RP.outputs:execOut", "Depth.inputs:execIn"),
                ("RP.outputs:renderProductPath",
                 "Rgb.inputs:renderProductPath"),
                ("RP.outputs:renderProductPath",
                 "InfoColor.inputs:renderProductPath"),
                ("RP.outputs:renderProductPath",
                 "InfoDepth.inputs:renderProductPath"),
                ("RP.outputs:renderProductPath",
                 "Depth.inputs:renderProductPath"),
            ],
        },
    )
    og.Controller.evaluate_sync(graph)
    print("M8 CAMERA GRAPH: %s {rgb,depth,2x info} @60/%d Hz, frame %s"
          % ("/overhead_camera", skip + 1, CAM_FRAME_ID), flush=True)


def build_m8_tf_graph() -> None:
    """Static TF base_link -> overhead_camera_color_optical_frame.

    ROS2PublishRawTransformTree with staticPublisher=true on /tf_static
    (transient-local, the tf2 static convention).  base_link is at the
    world origin with identity orientation (fixed-base import), so the
    edge equals the optical Xform's world pose: translation CAM_EYE,
    rotation 180 deg about X (IJKR = [1, 0, 0, 0]).  This single edge is
    the whole TF need of the M8 consumers: the robot segmenter looks up
    depth-frame <-> robot_base_frame and nvblox looks up
    global_frame(base_link) <-> depth frame; neither needs per-link TF
    (the segmenter poses the robot from /joint_states via cuRobo FK).
    """
    og.Controller.edit(
        {"graph_path": "/OverheadTfGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime",
                 "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("CamTf",
                 "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("CamTf.inputs:topicName", "tf_static"),
                ("CamTf.inputs:staticPublisher", True),
                ("CamTf.inputs:parentFrameId", "base_link"),
                ("CamTf.inputs:childFrameId", CAM_FRAME_ID),
                ("CamTf.inputs:translation", list(CAM_EYE)),
                # quatd IJKR layout: 180 deg about X (x=1, w=0) == the
                # optical Xform authored above.
                ("CamTf.inputs:rotation", [1.0, 0.0, 0.0, 0.0]),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "CamTf.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime",
                 "CamTf.inputs:timeStamp"),
            ],
        },
    )


class SceneCommandPoller:
    """Polls the scene-command JSON and applies gantry visibility.

    File-based on purpose: rclpy is banned in this process (repo rule)
    and the acceptance runner in the container shares the repo mount, so
    a JSON file is the simplest reliable cross-container channel.
    """

    def __init__(self, stage: Usd.Stage, path: str) -> None:
        self.stage = stage
        self.path = Path(path)
        self.mtime = -1.0
        self.gantry_visible = True

    def poll(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime == self.mtime:
            return
        self.mtime = mtime
        try:
            cmd = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            print("M8 SCENE CMD: unreadable %s (%s)" % (self.path, exc),
                  flush=True)
            return
        want = bool(cmd.get("gantry_visible", True))
        if want == self.gantry_visible:
            return
        prim = self.stage.GetPrimAtPath(GANTRY_PRIM)
        if not prim.IsValid():
            print("M8 SCENE CMD: gantry prim missing", flush=True)
            return
        img = UsdGeom.Imageable(prim)
        (img.MakeVisible() if want else img.MakeInvisible())
        self.gantry_visible = want
        print("M8 SCENE CMD: gantry_visible=%s" % want, flush=True)


def main() -> int:
    add_reference_to_stage(usd_path=USD_PATH, prim_path=ROBOT_PRIM)
    stage = omni.usd.get_context().get_stage()

    sim = SimulationContext(physics_dt=1.0 / PHYSICS_HZ,
                            rendering_dt=1.0 / PHYSICS_HZ,
                            stage_units_in_meters=1.0)

    articulation_root = find_articulation_root(stage)
    print("articulation root: %s" % articulation_root, flush=True)
    build_action_graph(articulation_root)
    scene_poller = None
    if _args.m8_scene:
        build_m8_scene(stage)
        build_m8_tf_graph()
        build_m8_camera_graph()
        scene_poller = SceneCommandPoller(stage, _args.scene_cmd_file)
    simulation_app.update()

    sim.initialize_physics()
    sim.play()  # physics silently no-ops while the timeline is paused

    from isaacsim.core.prims import SingleArticulation  # noqa: E402
    from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

    arm = SingleArticulation(prim_path=ROBOT_PRIM, name="rebot")
    arm.initialize()
    dof_names = list(arm.dof_names)

    # Raise arm effort caps (physics-view write, USD on disk untouched): the
    # imported 27 N.m caps are below static gravity torque -- see module doc.
    av = arm._articulation_view
    max_efforts = np.asarray(av.get_max_efforts(), dtype=float).ravel().copy()
    arm_idx = [dof_names.index(j) for j in ARM_JOINTS if j in dof_names]
    for i in arm_idx:
        max_efforts[i] = max(max_efforts[i], 1000.0)
    av.set_max_efforts(np.expand_dims(max_efforts, 0))

    # Stiffen the arm drives at runtime (physics view only, same rationale as
    # the effort raise and the probe track's RUNTIME_KP pattern): measured
    # with the imported gains, the arm settles ~0.018 rad below its position
    # target under gravity -- inside the pose_scene's tolerance but NOT
    # comfortably inside the 0.02 rad M5 convergence gate.
    scale = max(1.0, float(_args.arm_gain_scale))
    kps = np.asarray(av.get_gains()[0], dtype=float).reshape(-1).copy()
    kds = np.asarray(av.get_gains()[1], dtype=float).reshape(-1).copy()
    before = (kps.copy(), kds.copy())
    for i in arm_idx:
        kps[i] *= scale
        kds[i] *= scale ** 0.5
    av.set_gains(np.expand_dims(kps, 0), np.expand_dims(kds, 0),
                 save_to_usd=False)
    print("arm drive gains x%.1f: kp %s -> %s" % (
        scale, np.round(before[0][:6], 1).tolist(),
        np.round(kps[:6], 1).tolist()), flush=True)

    # Seed state AND drive target; jaws OPEN (0.0 = shut).
    target_q = {name: 0.0 for name in dof_names}
    target_q.update(dict(zip(ARM_JOINTS, [float(v) for v in _args.start_q])))
    for jaw in JAW_JOINTS:
        target_q[jaw] = JAW_OPEN_M
    q0 = np.array([target_q.get(n, 0.0) for n in dof_names])
    arm.set_joint_positions(q0)
    arm.apply_action(ArticulationAction(joint_positions=q0))
    for _ in range(30):  # re-asserting warmup; graph owns the drives after
        arm.apply_action(ArticulationAction(joint_positions=q0))
        sim.step(render=True)
    settled = np.asarray(arm.get_joint_positions(), dtype=float)

    print("=" * 70, flush=True)
    print("B601 SIM BRIDGE (M5) usd=%s" % USD_PATH, flush=True)
    print("  ROS_DISTRO=%s RMW=%s DOMAIN=%s headless=%s"
          % (os.environ.get("ROS_DISTRO"),
             os.environ.get("RMW_IMPLEMENTATION", "default(fastdds)"),
             os.environ.get("ROS_DOMAIN_ID"), not _args.gui), flush=True)
    print("  PUBLISH   %-24s 60 Hz, sole owner" % TOPIC_CLOCK, flush=True)
    print("  PUBLISH   %-24s all %d joints" % (TOPIC_JOINT_STATES,
                                               len(dof_names)), flush=True)
    print("  SUBSCRIBE %-24s JointState position targets"
          % TOPIC_JOINT_COMMANDS, flush=True)
    print("  DOF order: %s" % dof_names, flush=True)
    print("  start q:   %s" % {n: round(float(v), 4)
                               for n, v in zip(dof_names, settled)}, flush=True)
    print("BRIDGE READY", flush=True)
    if _args.m8_scene:
        print("  PUBLISH   %-24s rgb (60/%d Hz)"
              % (TOPIC_CAM_RGB, _args.cam_skip + 1), flush=True)
        print("  PUBLISH   %-24s depth 32FC1" % TOPIC_CAM_DEPTH, flush=True)
        print("  PUBLISH   %s + %s" % (TOPIC_CAM_RGB_INFO,
                                       TOPIC_CAM_DEPTH_INFO), flush=True)
        print("  PUBLISH   /tf_static base_link->%s" % CAM_FRAME_ID,
              flush=True)
        print("  SCENE CMD %s" % _args.scene_cmd_file, flush=True)
    if _args.ready_file:
        ready = {
            "usd": USD_PATH,
            "articulation_root": articulation_root,
            "dof_names": dof_names,
            "settled_q": [float(v) for v in settled],
            "topics": {"clock": TOPIC_CLOCK,
                       "joint_states": TOPIC_JOINT_STATES,
                       "joint_commands": TOPIC_JOINT_COMMANDS},
        }
        if _args.m8_scene:
            ready["m8_scene"] = {
                "camera_frame": CAM_FRAME_ID,
                "camera_eye": list(CAM_EYE),
                "camera_resolution": [CAM_W, CAM_H],
                "topics": {"rgb": TOPIC_CAM_RGB, "depth": TOPIC_CAM_DEPTH,
                           "rgb_info": TOPIC_CAM_RGB_INFO,
                           "depth_info": TOPIC_CAM_DEPTH_INFO},
                "scene_cmd_file": str(_args.scene_cmd_file),
                "gantry": M8_GANTRY, "table": M8_TABLE,
            }
        Path(_args.ready_file).write_text(json.dumps(ready, indent=2) + "\n")

    # Main loop: rendered steps only (OnPlaybackTick fires from the render
    # loop).  The Kit rate limiter (argv flags above) locks stepping to
    # 60 FPS wall; sleeping here instead was measured to make PhysX burst
    # multiple catch-up substeps per frame, racing /clock ~4x wall.
    wall_start = time.monotonic()
    sim_start = float(sim.current_time)
    deadline = wall_start + _args.duration
    next_report = wall_start + 30.0
    frame = 0
    while simulation_app.is_running() and time.monotonic() < deadline:
        sim.step(render=True)
        frame += 1
        if scene_poller is not None and frame % 60 == 0:
            scene_poller.poll()
        if time.monotonic() >= next_report:
            q_now = np.asarray(arm.get_joint_positions(), dtype=float)
            wall_elapsed = time.monotonic() - wall_start
            sim_elapsed = float(sim.current_time) - sim_start
            print("  ... wall %.1f s, sim %.1f s (rate %.2fx), q=%s"
                  % (wall_elapsed, sim_elapsed,
                     sim_elapsed / max(wall_elapsed, 1e-9),
                     np.round(q_now, 3).tolist()), flush=True)
            next_report += 30.0

    print("sim bridge done, sim time %.2f s" % sim.current_time, flush=True)
    sim.stop()
    return 0


if __name__ == "__main__":
    _rc = 1
    try:
        _rc = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
    sys.exit(_rc)
