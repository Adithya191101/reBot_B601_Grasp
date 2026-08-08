#!/usr/bin/env bash
# M5 end-to-end parity test: Isaac Sim articulation driven by the ROS 2 stack.
#
#   host:      scripts/b601_sim_bridge.py  (Isaac Sim 5.1, ~/isaaclab-venv)
#   container: rebot-jazzy-baseline, --network host, repo mounted at /work:
#              ros2 launch rebot_sim_bridge sim_profile.launch.py
#   check:     scripts/m5_parity_check.py (inside the container)
#
# Writes artifacts/m5/parity.json (container verdict + peak GPU VRAM).
# Exit 0 iff the parity gate passed.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_PY="${ISAAC_PY:-$HOME/isaaclab-venv/bin/python}"
# The bridge's bundled Jazzy client libraries resolve each other through the
# dynamic loader, and LD_LIBRARY_PATH is fixed at process start -- without
# this the extension fails with "librmw_implementation.so: ...
# libament_index_cpp.so: cannot open shared object file" (same fix as the
# rearm environment's isaac_sim_env.sh; no system ROS is sourced instead).
BRIDGE_JAZZY_LIB="$(dirname "$(dirname "$ISAAC_PY")")/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib"
if [ ! -d "$BRIDGE_JAZZY_LIB" ]; then
    echo "FATAL: bundled Jazzy libs not found at $BRIDGE_JAZZY_LIB"
    exit 1
fi
IMAGE="${IMAGE:-rebot-jazzy-baseline:latest}"
STACK_NAME="rebot_m5_stack"
ART="$REPO/artifacts/m5"
BRIDGE_TIMEOUT="${BRIDGE_TIMEOUT:-300}"   # Isaac boot can take ~90 s+
BRIDGE_DURATION="${BRIDGE_DURATION:-900}"

mkdir -p "$ART"
rm -f "$ART/bridge_ready.json" "$ART/parity_container.json" "$ART/parity.json"

BRIDGE_PID=""
VRAM_PID=""
cleanup() {
    docker rm -f "$STACK_NAME" >/dev/null 2>&1 || true
    [ -n "$BRIDGE_PID" ] && kill "$BRIDGE_PID" >/dev/null 2>&1 || true
    # Belt and braces: a stray bridge is CATASTROPHIC for the next run --
    # two Isaac processes both publish /clock and /isaac_joint_states on
    # domain 42, the container mixes the epochs/robots, and every sim-time
    # timeout and state check goes quietly insane (measured: attempts 3-7).
    pkill -f "scripts/b601_sim_bridge.py" >/dev/null 2>&1 || true
    sleep 3
    # Kit teardown can hang for minutes on SIGTERM; do not leave it around.
    pkill -9 -f "scripts/b601_sim_bridge.py" >/dev/null 2>&1 || true
    [ -n "$VRAM_PID" ] && kill "$VRAM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Refuse to start over a stale bridge or stack from an earlier run.
if pgrep -f "scripts/b601_sim_bridge.py" >/dev/null 2>&1; then
    echo "WARN: killing stale b601_sim_bridge processes from a previous run"
    pkill -f "scripts/b601_sim_bridge.py" || true
    sleep 5
fi
docker rm -f "$STACK_NAME" >/dev/null 2>&1 || true

echo "== [0/6] baseline VRAM + sampler =="
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | head -1 > "$ART/vram_baseline.txt"
: > "$ART/vram_samples.log"
( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
          | head -1 >> "$ART/vram_samples.log"
      sleep 2
  done ) &
VRAM_PID=$!

echo "== [1/6] building the container workspace (colcon) =="
docker run --rm --network host -v "$REPO":/work "$IMAGE" bash -lc '
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
cat "$ART/colcon_build.log"

echo "== [2/6] starting the Isaac Sim bridge (host, background) =="
# env + direct exec (no subshell): $! must be the PYTHON process itself --
# killing a wrapping subshell leaves the bridge alive, and a surviving
# bridge poisons every later run (see cleanup note above).
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
    if [ -f "$ART/bridge_ready.json" ]; then break; fi
    if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        echo; echo "FATAL: bridge exited early; tail of $ART/sim_bridge.log:"
        tail -30 "$ART/sim_bridge.log"
        exit 1
    fi
    echo -n "."
    sleep 1
done
echo
if [ ! -f "$ART/bridge_ready.json" ]; then
    echo "FATAL: bridge not ready after ${BRIDGE_TIMEOUT}s; tail of log:"
    tail -30 "$ART/sim_bridge.log"
    exit 1
fi
echo "   bridge is up."

echo "== [3/6] starting the container node stack =="
docker rm -f "$STACK_NAME" >/dev/null 2>&1 || true
# FASTRTPS_DEFAULT_PROFILES_FILE (both sides): default FastDDS picks a
# shared-memory transport between same-host participants; host<->container
# discovery then works but DATA delivery dies silently (observed flaky:
# attempt 2 delivered /clock, attempts 3-4 did not, --ipc/--pid host did
# not help; every sim-time timer freezes with zero warnings).  NVIDIA's
# documented remedy for Isaac Sim + Docker ROS nodes is a UDPv4-only
# profile -- see config/fastdds_udp.xml.  RMW stays default fastrtps.
docker run -d --name "$STACK_NAME" --network host \
    -e ROS_DOMAIN_ID=42 \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/work/config/fastdds_udp.xml \
    -v "$REPO":/work "$IMAGE" bash -lc '
    source /opt/ros/jazzy/setup.bash
    source /work/ros2_ws/install/setup.bash
    exec ros2 launch rebot_sim_bridge sim_profile.launch.py
' >/dev/null
sleep 8
docker logs "$STACK_NAME" > "$ART/stack.log" 2>&1
if ! docker ps -q -f name="$STACK_NAME" | grep -q .; then
    echo "FATAL: node stack exited; log:"
    tail -40 "$ART/stack.log"
    exit 1
fi
echo "   stack is up."

echo "== [4/6] running the parity check inside the container =="
docker exec -e ROS_DOMAIN_ID=42 "$STACK_NAME" bash -lc '
    source /opt/ros/jazzy/setup.bash
    source /work/ros2_ws/install/setup.bash
    python3 /work/scripts/m5_parity_check.py \
        --out /work/artifacts/m5/parity_container.json
' 2>&1 | tee "$ART/parity_check.log"
CHECK_RC=${PIPESTATUS[0]}

echo "== [5/6] collecting logs + VRAM peak =="
docker logs "$STACK_NAME" > "$ART/stack.log" 2>&1 || true
kill "$VRAM_PID" >/dev/null 2>&1 || true; VRAM_PID=""

echo "== [6/6] writing $ART/parity.json =="
python3 - "$ART" "$CHECK_RC" <<'EOF'
import json, sys, time
from pathlib import Path

art = Path(sys.argv[1])
check_rc = int(sys.argv[2])

samples = [int(line) for line in
           (art / "vram_samples.log").read_text().split() if line.strip()]
baseline = int((art / "vram_baseline.txt").read_text().split()[0])
container = {}
p = art / "parity_container.json"
if p.is_file():
    container = json.loads(p.read_text())

bridge_ready = {}
p = art / "bridge_ready.json"
if p.is_file():
    bridge_ready = json.loads(p.read_text())

report = {
    "milestone": "M5",
    "gate": "state/command parity (design doc sec. 21)",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "passed": check_rc == 0 and bool(container.get("passed")),
    "gpu_vram": {
        "baseline_before_bridge_mib": baseline,
        "peak_during_run_mib": max(samples) if samples else None,
        "bridge_delta_mib": (max(samples) - baseline) if samples else None,
        "samples": len(samples),
        "note": "nvidia-smi memory.used is whole-GPU; delta vs baseline "
                "isolates the sim bridge stack",
    },
    "sim_bridge": bridge_ready,
    "container_verdict": container,
}
(art / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in ("passed", "gpu_vram")}, indent=2))
EOF

if [ "$CHECK_RC" -eq 0 ]; then
    echo "M5 PARITY TEST: PASS"
else
    echo "M5 PARITY TEST: FAIL (see $ART/parity_check.log, $ART/stack.log," \
         "$ART/sim_bridge.log)"
fi
exit "$CHECK_RC"
