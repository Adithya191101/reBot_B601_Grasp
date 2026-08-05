#!/usr/bin/env bash
# Reproduce the complete Aug 8 ten-scene smoke test from a clean checkout.
#
#   ./run_smoke.sh              # Isaac Sim backend if available, else analytic
#   ./run_smoke.sh analytic     # force the dependency-free analytic backend
#   ./run_smoke.sh isaac        # force Isaac Sim; fail loudly if it will not run
#
# Requires only Isaac Sim's interpreter (numpy, opencv, pillow are already in it).
# Installs nothing, downloads nothing, needs no sudo, and does not need ROS 2.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-$HOME/isaaclab-venv/bin/python}"
BACKEND="${1:-auto}"

if [[ ! -x "$PYTHON" ]]; then
  echo "error: no interpreter at $PYTHON (override with PYTHON=/path/to/python)" >&2
  exit 1
fi

echo "=== 0/4  environment ==="
"$PYTHON" -c "import numpy, cv2, PIL; print('python', __import__('platform').python_version(),
      '| numpy', numpy.__version__, '| opencv', cv2.__version__)"

echo
echo "=== 1/4  restore pinned upstream sources (if missing) ==="
if [[ ! -d src/reBot-DevArm-Grasp ]]; then
  if command -v vcs >/dev/null 2>&1; then
    mkdir -p src && vcs import src < upstream.repos
  else
    echo "note: vcstool not installed and src/ is absent."
    echo "      The smoke test does not need the vendor sources; clone them with:"
    echo "      vcs import src < upstream.repos"
  fi
else
  echo "src/ present"
fi

echo
echo "=== 2/4  unit tests (A0 red tests + dataset/scorer/pose/depth contracts) ==="
"$PYTHON" -m unittest discover -s tests -p "test_*.py" -v 2>&1 | tail -5

echo
echo "=== 3/4  capture ten scenes ==="
if [[ "$BACKEND" == "analytic" ]]; then
  USE_ISAAC=0
elif [[ "$BACKEND" == "isaac" ]]; then
  USE_ISAAC=1
else
  if "$PYTHON" -c "import isaacsim" >/dev/null 2>&1; then USE_ISAAC=1; else USE_ISAAC=0; fi
fi

if [[ "$USE_ISAAC" == "1" ]]; then
  OUT="artifacts/smoke_isaac"
  echo "backend: Isaac Sim 5.1 (progress -> $OUT/isaac_progress.log)"
  rm -rf "$OUT"
  "$PYTHON" capture/isaac_capture.py --out "$OUT/dataset" >/dev/null 2>&1 || {
    echo "Isaac capture failed; see $OUT/isaac_progress.log" >&2
    cat "$OUT/isaac_progress.log" >&2 || true
    exit 3
  }
  tail -3 "$OUT/isaac_progress.log"
else
  OUT="artifacts/smoke"
  echo "backend: analytic"
  rm -rf "$OUT"
fi

echo
echo "=== 4/4  replay -> A1/A2 + B -> PoseStamped -> scorer + overlay ==="
if [[ "$USE_ISAAC" == "1" ]]; then
  "$PYTHON" run_smoke.py --out "$OUT" --reuse-dataset
else
  "$PYTHON" run_smoke.py --out "$OUT" --backend analytic
fi

echo
echo "artifacts: $REPO/$OUT   (results.json, overlays/, dataset/)"
