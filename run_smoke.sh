#!/usr/bin/env bash
# Reproduce the ten-scene smoke test.
#
#   ./run_smoke.sh isaac       # THE REAL GATE. Fails if Isaac Sim is unavailable.
#   ./run_smoke.sh analytic    # development mode: dependency-free, NOT the gate
#   ./run_smoke.sh             # same as `isaac`
#
# Installs nothing, downloads nothing, needs no sudo, and does not need ROS 2.
# It will not clone anything either -- if the pinned vendor sources are missing
# it tells you the command and stops, because "downloads nothing" and "runs
# vcs import for you" cannot both be true.
#
# It never deletes a sealed dataset. Captures refuse to write into a non-empty
# output directory; pick a new --out or remove the old one yourself, deliberately.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-$HOME/isaaclab-venv/bin/python}"
BACKEND="${1:-isaac}"
PREDICTOR="${PREDICTOR:-saturation}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: no interpreter at $PYTHON (override with PYTHON=/path/to/python)" >&2
  exit 1
fi

echo "=== 0/4  environment ==="
"$PYTHON" -c "import numpy, cv2, PIL, platform; print('python', platform.python_version(),
      '| numpy', numpy.__version__, '| opencv', cv2.__version__)"

if [[ ! -d src/reBot-DevArm-Grasp ]]; then
  echo
  echo "error: pinned vendor sources are missing." >&2
  echo "       Restore them yourself (this script does not fetch anything):" >&2
  echo "           vcs import src < upstream.repos" >&2
  exit 1
fi
echo "pinned vendor sources: present"

echo
echo "=== 1/4  unit tests ==="
"$PYTHON" -m unittest discover -s tests -p "test_*.py" 2>&1 | tail -4
echo "note: ros2_iface/test_jazzy_integration.py is NOT part of this run and has"
echo "      never been executed -- it needs ROS 2 Jazzy installed."

echo
echo "=== 2/4  select backend ==="
case "$BACKEND" in
  isaac)
    if ! "$PYTHON" -c "import isaacsim" >/dev/null 2>&1; then
      echo "error: Isaac Sim is not importable with $PYTHON." >&2
      echo "       The real gate requires it. For development only, run:" >&2
      echo "           ./run_smoke.sh analytic" >&2
      exit 2
    fi
    OUT="${OUT:-artifacts/smoke_isaac}"
    echo "backend: Isaac Sim 5.1  (THE GATE)"
    ;;
  analytic)
    OUT="${OUT:-artifacts/smoke_analytic}"
    echo "backend: analytic  (DEVELOPMENT MODE -- exercises the chain, not the simulator)"
    ;;
  *)
    echo "usage: $0 [isaac|analytic]" >&2
    exit 64
    ;;
esac

if [[ -e "$OUT/dataset/manifest.json" ]]; then
  echo
  echo "error: a sealed dataset already exists at $OUT/dataset" >&2
  echo "       This script never deletes one. Either:" >&2
  echo "         OUT=artifacts/smoke_$(date +%s) $0 $BACKEND" >&2
  echo "       or remove it yourself if you are sure it is not the locked set." >&2
  exit 3
fi

echo
echo "=== 3/4  capture ten scenes ==="
if [[ "$BACKEND" == "isaac" ]]; then
  echo "progress -> $OUT/isaac_progress.log"
  "$PYTHON" capture/isaac_capture.py --out "$OUT/dataset" >/dev/null 2>&1 || {
    echo "Isaac capture failed; see $OUT/isaac_progress.log" >&2
    cat "$OUT/isaac_progress.log" >&2 || true
    exit 4
  }
  tail -3 "$OUT/isaac_progress.log"
fi

echo
echo "=== 4/4  validate -> inference -> scoring -> overlay ==="
if [[ "$BACKEND" == "isaac" ]]; then
  "$PYTHON" run_smoke.py --out "$OUT" --predictor "$PREDICTOR" --reuse-dataset
else
  "$PYTHON" run_smoke.py --out "$OUT" --backend analytic --predictor "$PREDICTOR"
fi

echo
echo "artifacts: $REPO/$OUT   (results.json, predictions/, overlays/, dataset/)"
