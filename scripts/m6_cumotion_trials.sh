#!/usr/bin/env bash
# M6 gate: 100 collision-free pose-to-pose cuMotion trials, static scene.
#
#   host:       scripts/b601_sim_bridge.py        (Isaac Sim 5.1, isaaclab-venv)
#   container1: rebot-jazzy-baseline              (adapters + sim-JTC shims)
#   container2: rebot-m6-cumotion                 (Isaac ROS 4.5 cuMotion
#               standalone action server; image = NVIDIA isaac_ros dev image
#               + apt ros-jazzy-isaac-ros-cumotion, committed locally)
#   runner:     scripts/m6_trial_runner.py        (inside container2)
#   verify:     scripts/m6_verify_trials.py       (host, mesh recheck)
#
# All ROS participants: ROS_DOMAIN_ID=42, default FastDDS with the UDPv4
# profile config/fastdds_udp.xml on EVERY side (M5 lesson: same-host SHM
# transport silently drops host<->container DATA).
#
# Writes artifacts/m6/trials.json (verdict + peak VRAM).  Exit 0 iff the
# gate passed.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_PY="${ISAAC_PY:-$HOME/isaaclab-venv/bin/python}"
BRIDGE_JAZZY_LIB="$(dirname "$(dirname "$ISAAC_PY")")/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib"
if [ ! -d "$BRIDGE_JAZZY_LIB" ]; then
    echo "FATAL: bundled Jazzy libs not found at $BRIDGE_JAZZY_LIB"
    exit 1
fi
IMAGE_BASE="${IMAGE_BASE:-rebot-jazzy-baseline:latest}"
IMAGE_CU="${IMAGE_CU:-rebot-m6-cumotion:latest}"
STACK_NAME="rebot_m6_stack"
CU_NAME="rebot_m6_cumotion_stack"
ART="$REPO/artifacts/m6"
BRIDGE_TIMEOUT="${BRIDGE_TIMEOUT:-300}"
BRIDGE_DURATION="${BRIDGE_DURATION:-5400}"

mkdir -p "$ART"
rm -f "$ART/bridge_ready.json" "$ART/trials_raw.json" "$ART/trials.json"

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

echo "== [0/8] baseline VRAM + sampler =="
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | head -1 > "$ART/vram_baseline.txt"
: > "$ART/vram_samples.log"
( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
          | head -1 >> "$ART/vram_samples.log"
      sleep 2
  done ) &
VRAM_PID=$!

echo "== [1/8] trial poses (host, deterministic, always regenerated) =="
"$ISAAC_PY" "$REPO/scripts/m6_sample_poses.py" || exit 1

echo "== [2/8] building the container workspace (colcon) =="
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

echo "== [3/8] starting the Isaac Sim bridge (host, background) =="
env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N \
    ROS_DISTRO=jazzy ROS_DOMAIN_ID=42 \
    FASTRTPS_DEFAULT_PROFILES_FILE="$REPO/config/fastdds_udp.xml" \
    LD_LIBRARY_PATH="$BRIDGE_JAZZY_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$ISAAC_PY" "$REPO/scripts/b601_sim_bridge.py" \
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

echo "== [4/8] starting the adapter/shim stack (container 1) =="
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

echo "== [5/8] starting cuMotion (container 2, standalone action server) =="
docker run -d --name "$CU_NAME" --network host --gpus all \
    -e ROS_DOMAIN_ID=42 \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/work/config/fastdds_udp.xml \
    -v "$REPO":/work "$IMAGE_CU" bash -c '
    source /opt/ros/jazzy/setup.bash
    ros2 run rclcpp_components component_container_mt --ros-args \
        -r __node:=cumotion_container -p use_sim_time:=true &
    CONTAINER_PID=$!
    sleep 5
    ros2 component load /cumotion_container isaac_ros_cumotion \
        nvidia::isaac_ros::cumotion::StaticPlanningSceneServer \
        -p use_sim_time:=true
    ros2 component load /cumotion_container isaac_ros_cumotion \
        nvidia::isaac_ros::cumotion::CumotionPlanner \
        -p use_sim_time:=true \
        -p urdf_file_path:=/work/urdf/rebot_b601dm_cumotion.urdf \
        -p xrdf_file_path:=/work/config/rebot_b601dm.xrdf \
        -p read_esdf_world:=false \
        -p publish_world_collision_spheres:=false \
        -p time_dilation_factor:=0.5 \
        -p interpolation_dt:=0.05
    wait $CONTAINER_PID
' >/dev/null
echo -n "   waiting for cumotion/motion_plan action server "
CU_OK=0
for _ in $(seq 120); do
    if docker exec "$CU_NAME" bash -c \
        'source /opt/ros/jazzy/setup.bash && timeout 10 ros2 action list 2>/dev/null' \
        2>/dev/null | grep -q "cumotion/motion_plan"; then
        CU_OK=1; break
    fi
    if ! docker ps -q -f name="$CU_NAME" | grep -q .; then
        echo; echo "FATAL: cumotion container exited:"
        docker logs "$CU_NAME" | tail -40
        exit 1
    fi
    echo -n "."; sleep 2
done
echo
if [ "$CU_OK" != 1 ]; then
    echo "FATAL: cumotion action server not up; container log:"
    docker logs "$CU_NAME" | tail -40
    exit 1
fi
echo "   cuMotion is up."

echo "== [6/8] running 100 pose-to-pose trials =="
docker exec -e ROS_DOMAIN_ID=42 \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/work/config/fastdds_udp.xml \
    "$CU_NAME" bash -c '
    source /opt/ros/jazzy/setup.bash
    python3 /work/scripts/m6_trial_runner.py
' 2>&1 | tee "$ART/trial_runner.log"
RUNNER_RC=${PIPESTATUS[0]}

echo "== [7/8] collecting logs + stopping the stack =="
docker logs "$STACK_NAME" > "$ART/stack.log" 2>&1 || true
docker logs "$CU_NAME" > "$ART/cumotion.log" 2>&1 || true
kill "$VRAM_PID" >/dev/null 2>&1 || true; VRAM_PID=""
docker rm -f "$STACK_NAME" "$CU_NAME" >/dev/null 2>&1 || true
kill "$BRIDGE_PID" >/dev/null 2>&1 || true

echo "== [8/8] host-side mesh verification + gate verdict =="
if [ ! -f "$ART/trials_raw.json" ]; then
    echo "FATAL: no trials_raw.json produced (runner rc=$RUNNER_RC)"
    exit 1
fi
"$ISAAC_PY" "$REPO/scripts/m6_verify_trials.py"
VERIFY_RC=$?

if [ "$VERIFY_RC" -eq 0 ]; then
    echo "M6 CUMOTION GATE: PASS"
else
    echo "M6 CUMOTION GATE: FAIL (runner rc=$RUNNER_RC; see" \
         "$ART/trial_runner.log, $ART/cumotion.log, $ART/stack.log)"
fi
exit "$VERIFY_RC"
