#!/usr/bin/env bash
# M8 gate: robot segmentation + nvblox ESDF mapping feeding cuMotion's world
# (design doc sections 14-15; mapping acceptance test = doc 15.4).
#
# The M6/M7 orchestration flow REUSED with the M8 sim scene and container:
#
#   host:       scripts/b601_sim_bridge.py --m8-scene   (Isaac Sim 5.1 +
#               overhead RGB-D camera + visible table/gantry + /tf_static)
#   container1: rebot-jazzy-baseline       (adapters + sim-JTC shims)
#   container2: rebot-m8-nvblox            (rebot-m6-cumotion + apt
#               ros-jazzy-isaac-ros-cumotion-robot-segmenter +
#               ros-jazzy-isaac-ros-nvblox, committed locally) running ONE
#               component container with: RobotSegmenter -> NvbloxNode
#               (TSDF/ESDF, workspace-bounded) -> CumotionPlanner
#               (read_esdf_world:=true) + StaticPlanningSceneServer
#   sampler:    scripts/m8_select_configs.py   (host, deterministic)
#   runner:     scripts/m8_acceptance_runner.py (inside container2)
#   verify:     scripts/m8_verify_mapping.py    (host, mesh recheck + gates)
#
# All ROS participants: ROS_DOMAIN_ID=42, default FastDDS with the UDPv4
# profile config/fastdds_udp.xml on EVERY side (M5 lesson: same-host SHM
# transport silently drops host<->container DATA).
#
# Writes artifacts/m8/mapping_acceptance.json (verdict + peak VRAM,
# DDR-001 stage-2 checkpoint).  Exit 0 iff the gate passed.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_PY="${ISAAC_PY:-$HOME/isaaclab-venv/bin/python}"
BRIDGE_JAZZY_LIB="$(dirname "$(dirname "$ISAAC_PY")")/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib"
if [ ! -d "$BRIDGE_JAZZY_LIB" ]; then
    echo "FATAL: bundled Jazzy libs not found at $BRIDGE_JAZZY_LIB"
    exit 1
fi
IMAGE_BASE="${IMAGE_BASE:-rebot-jazzy-baseline:latest}"
IMAGE_M8="${IMAGE_M8:-rebot-m8-nvblox:latest}"
STACK_NAME="rebot_m8_stack"
CU_NAME="rebot_m8_cu"
ART="$REPO/artifacts/m8"
BRIDGE_TIMEOUT="${BRIDGE_TIMEOUT:-300}"
BRIDGE_DURATION="${BRIDGE_DURATION:-5400}"

mkdir -p "$ART"
rm -f "$ART/bridge_ready.json" "$ART/acceptance_raw.json" \
      "$ART/mapping_acceptance.json" "$ART/scene_cmd.json"
# Pre-create host-owned so the root-running container writes INTO it and a
# later host-side cleanup can still delete the files.
mkdir -p "$ART/evidence"
rm -f "$ART"/evidence/*

BRIDGE_PID=""
VRAM_PID=""
cleanup() {
    docker rm -f "$STACK_NAME" "$CU_NAME" >/dev/null 2>&1 || true
    [ -n "$BRIDGE_PID" ] && kill "$BRIDGE_PID" >/dev/null 2>&1 || true
    # A stray bridge poisons the next run (duplicate /clock publishers,
    # measured in M5): kill hard after a grace period.
    pkill -f "scripts/b601_sim_bridge.py" >/dev/null 2>&1 || true
    sleep 3
    pkill -9 -f "scripts/b601_sim_bridge.py" >/dev/null 2>&1 || true
    [ -n "$VRAM_PID" ] && kill "$VRAM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if pgrep -f "scripts/b601_sim_bridge.py" >/dev/null 2>&1; then
    echo "WARN: killing stale b601_sim_bridge from a previous run"
    pkill -f "scripts/b601_sim_bridge.py" || true
    sleep 5
fi
docker rm -f "$STACK_NAME" "$CU_NAME" >/dev/null 2>&1 || true

echo "== [0/9] baseline VRAM + sampler =="
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | head -1 > "$ART/vram_baseline.txt"
: > "$ART/vram_samples.log"
( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
          | head -1 >> "$ART/vram_samples.log"
      sleep 2
  done ) &
VRAM_PID=$!

echo "== [1/9] acceptance configs (host, deterministic) =="
"$ISAAC_PY" "$REPO/scripts/m8_select_configs.py" || exit 1

echo "== [2/9] building the container workspace (colcon) =="
docker run --rm --network host -v "$REPO":/work "$IMAGE_BASE" bash -lc '
    source /opt/ros/jazzy/setup.bash
    cd /work/ros2_ws
    colcon build --packages-select rebot_planner_msgs rebot_adapters \
        rebot_planner rebot_sim_bridge 2>&1 | tail -3
' > "$ART/colcon_build.log" 2>&1
if ! grep -q "packages finished" "$ART/colcon_build.log" \
        || grep -qE "failed|aborted" "$ART/colcon_build.log"; then
    echo "FATAL: colcon build failed; see $ART/colcon_build.log"
    cat "$ART/colcon_build.log"
    exit 1
fi

echo "== [3/9] starting the Isaac Sim bridge (host, --m8-scene) =="
env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N \
    ROS_DISTRO=jazzy ROS_DOMAIN_ID=42 \
    FASTRTPS_DEFAULT_PROFILES_FILE="$REPO/config/fastdds_udp.xml" \
    LD_LIBRARY_PATH="$BRIDGE_JAZZY_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$ISAAC_PY" "$REPO/scripts/b601_sim_bridge.py" \
        --m8-scene \
        --duration "$BRIDGE_DURATION" \
        --ready-file "$ART/bridge_ready.json" \
    > "$ART/sim_bridge.log" 2>&1 &
BRIDGE_PID=$!
echo -n "   waiting for BRIDGE READY (timeout ${BRIDGE_TIMEOUT}s) "
for _ in $(seq "$BRIDGE_TIMEOUT"); do
    [ -f "$ART/bridge_ready.json" ] && break
    if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        echo; echo "FATAL: bridge exited early:"; tail -30 "$ART/sim_bridge.log"
        exit 1
    fi
    echo -n "."; sleep 1
done
echo
[ -f "$ART/bridge_ready.json" ] || {
    echo "FATAL: bridge not ready"; tail -30 "$ART/sim_bridge.log"; exit 1; }
echo "   bridge is up."

echo "== [4/9] starting the adapter/shim stack (container 1) =="
docker run -d --name "$STACK_NAME" --network host \
    -e ROS_DOMAIN_ID=42 \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/work/config/fastdds_udp.xml \
    -v "$REPO":/work "$IMAGE_BASE" bash -lc '
    source /opt/ros/jazzy/setup.bash
    source /work/ros2_ws/install/setup.bash
    exec ros2 launch rebot_sim_bridge sim_profile.launch.py
' >/dev/null
sleep 8
if ! docker ps -q -f name="$STACK_NAME" | grep -q .; then
    echo "FATAL: adapter stack exited:"; docker logs "$STACK_NAME" | tail -30
    exit 1
fi
echo "   adapter stack is up."

echo "== [5/9] starting segmenter + nvblox + cuMotion (container 2) =="
# One component_container_mt; load order matters only for readability --
# every node discovers its peers over DDS.  nvblox: workspace-bounded
# (doc 15.5 values, = the sampler's WORKSPACE_*), 1 cm voxels, 3d ESDF,
# global frame base_link, DEPTH ONLY (color integration off: nothing
# consumes a colored mesh and VRAM is the binding constraint).  The
# segmenter consumes the overhead depth + /joint_states and republishes
# robot-masked depth, which is nvblox's ONLY depth input -- the arm can
# never enter the map (doc 15.3: "the robot segmenter is not optional").
docker run -d --name "$CU_NAME" --network host --gpus all \
    -e ROS_DOMAIN_ID=42 \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/work/config/fastdds_udp.xml \
    -v "$REPO":/work "$IMAGE_M8" bash -c '
    source /opt/ros/jazzy/setup.bash
    ros2 run rclcpp_components component_container_mt --ros-args \
        -r __node:=cumotion_container -p use_sim_time:=true &
    CONTAINER_PID=$!
    sleep 5
    ros2 component load /cumotion_container isaac_ros_cumotion_robot_segmenter \
        nvidia::isaac_ros::manipulator::RobotSegmenter \
        --node-name robot_segmenter_1 \
        -p use_sim_time:=true \
        -p urdf_path:=/work/urdf/rebot_b601dm_cumotion.urdf \
        -p xrdf_path:=/work/config/rebot_b601dm.xrdf \
        -p robot_base_frame:=base_link \
        -p additional_buffer_distance:=0.05 \
        -p input_qos:=DEFAULT -p output_qos:=DEFAULT \
        -r depth_image:=/overhead_camera/aligned_depth_to_color/image_raw \
        -r camera_info_depth:=/overhead_camera/aligned_depth_to_color/camera_info \
        -r joint_states:=/joint_states \
        -r robot_mask:=/cumotion/camera_1/robot_mask \
        -r robot_depth:=/cumotion/camera_1/world_depth
    ros2 component load /cumotion_container nvblox_ros nvblox::NvbloxNode \
        --node-name nvblox_node \
        -p use_sim_time:=true \
        -p global_frame:=base_link \
        -p num_cameras:=1 \
        -p use_color:=false -p use_depth:=true -p use_lidar:=false \
        -p voxel_size:=0.01 \
        -p esdf_mode:=3d \
        -p publish_esdf_distance_slice:=false \
        -p input_qos:=DEFAULT \
        -p map_clearing_frame_id:=base_link \
        -p static_mapper.workspace_bounds_type:=bounding_box \
        -p static_mapper.workspace_bounds_min_corner_x_m:=-0.10 \
        -p static_mapper.workspace_bounds_min_corner_y_m:=-0.35 \
        -p static_mapper.workspace_bounds_min_height_m:=-0.05 \
        -p static_mapper.workspace_bounds_max_corner_x_m:=0.65 \
        -p static_mapper.workspace_bounds_max_corner_y_m:=0.35 \
        -p static_mapper.workspace_bounds_max_height_m:=0.65 \
        -p static_mapper.projective_integrator_max_integration_distance_m:=2.0 \
        -p static_mapper.esdf_integrator_max_distance_m:=0.5 \
        -r camera_0/depth/image:=/cumotion/camera_1/world_depth \
        -r camera_0/depth/camera_info:=/overhead_camera/aligned_depth_to_color/camera_info
    ros2 component load /cumotion_container isaac_ros_cumotion \
        nvidia::isaac_ros::cumotion::StaticPlanningSceneServer \
        -p use_sim_time:=true
    ros2 component load /cumotion_container isaac_ros_cumotion \
        nvidia::isaac_ros::cumotion::CumotionPlanner \
        -p use_sim_time:=true \
        -p urdf_file_path:=/work/urdf/rebot_b601dm_cumotion.urdf \
        -p xrdf_file_path:=/work/config/rebot_b601dm.xrdf \
        -p read_esdf_world:=true \
        -p update_esdf_on_request:=true \
        -p esdf_service_name:=/nvblox_node/get_esdf_and_gradient \
        -p publish_world_collision_spheres:=false \
        -p time_dilation_factor:=0.5 \
        -p interpolation_dt:=0.05
    wait $CONTAINER_PID
' >/dev/null
echo -n "   waiting for cumotion/motion_plan + ESDF service "
CU_OK=0
for _ in $(seq 120); do
    if docker exec "$CU_NAME" bash -c \
        'source /opt/ros/jazzy/setup.bash \
         && timeout 10 ros2 action list 2>/dev/null | grep -q cumotion/motion_plan \
         && timeout 10 ros2 service list 2>/dev/null | grep -q get_esdf_and_gradient' \
        2>/dev/null; then
        CU_OK=1; break
    fi
    if ! docker ps -q -f name="$CU_NAME" | grep -q .; then
        echo; echo "FATAL: M8 container exited:"
        docker logs "$CU_NAME" | tail -40
        exit 1
    fi
    echo -n "."; sleep 2
done
echo
if [ "$CU_OK" != 1 ]; then
    echo "FATAL: cuMotion/nvblox not up; container log:"
    docker logs "$CU_NAME" | tail -40
    exit 1
fi
echo "   segmenter + nvblox + cuMotion are up."

echo "== [6/9] running the mapping acceptance phases =="
docker exec -e ROS_DOMAIN_ID=42 \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/work/config/fastdds_udp.xml \
    "$CU_NAME" bash -c '
    source /opt/ros/jazzy/setup.bash
    python3 /work/scripts/m8_acceptance_runner.py
' 2>&1 | tee "$ART/acceptance_runner.log"
RUNNER_RC=${PIPESTATUS[0]}

echo "== [7/9] collecting logs + stopping the stack =="
docker logs "$STACK_NAME" > "$ART/stack.log" 2>&1 || true
docker logs "$CU_NAME" > "$ART/m8_container.log" 2>&1 || true
kill "$VRAM_PID" >/dev/null 2>&1 || true; VRAM_PID=""
docker rm -f "$STACK_NAME" "$CU_NAME" >/dev/null 2>&1 || true
kill "$BRIDGE_PID" >/dev/null 2>&1 || true

echo "== [8/9] host-side verification + gate verdict =="
if [ ! -f "$ART/acceptance_raw.json" ]; then
    echo "FATAL: no acceptance_raw.json produced (runner rc=$RUNNER_RC)"
    exit 1
fi
"$ISAAC_PY" "$REPO/scripts/m8_verify_mapping.py"
VERIFY_RC=$?

echo "== [9/9] verdict =="
if [ "$VERIFY_RC" -eq 0 ]; then
    echo "M8 MAPPING GATE: PASS"
else
    echo "M8 MAPPING GATE: FAIL (runner rc=$RUNNER_RC; see" \
         "$ART/acceptance_runner.log, $ART/m8_container.log, $ART/stack.log)"
fi
exit "$VERIFY_RC"
