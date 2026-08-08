"""Make the package importable without installation or ROS.

The cores under rebot_planner/core are rclpy-free by design; these tests
run on a plain Python interpreter with pinocchio installed (e.g.
``~/isaaclab-venv/bin/python -m pytest``), no ROS environment sourced.
"""

import os
import sys

import numpy as np
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_TEST_DIR)
for _p in (_PKG_ROOT, _TEST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from planner_testlib import make_tcp, rotation_pitch_forward  # noqa: E402

# <repo>/ros2_ws/src/rebot_planner -> <repo>
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_PKG_ROOT)))

URDF_PATH = os.path.join(_REPO, "urdf", "rebot_b601dm_canonical.urdf")
CELL_YAML = os.path.join(_PKG_ROOT, "config", "cell_geometry.yaml")
MESH_ROOT = os.path.join(_REPO, "src", "reBotArmController_ROS2", "src")


def _require(path: str) -> str:
    if not os.path.exists(path):
        pytest.skip(f"required asset missing: {path}", allow_module_level=True)
    return path


@pytest.fixture(scope="session")
def urdf_path() -> str:
    return _require(URDF_PATH)


@pytest.fixture(scope="session")
def cell_yaml() -> str:
    return _require(CELL_YAML)


@pytest.fixture(scope="session")
def kin(urdf_path):
    from rebot_planner.core.ik_core import KinematicsCore

    return KinematicsCore(urdf_path)


@pytest.fixture(scope="session")
def collision(kin, cell_yaml):
    from rebot_planner.core.collision_core import CollisionCore

    _require(os.path.join(MESH_ROOT, "rebotarm_bringup"))
    return CollisionCore(kin, package_dirs=[MESH_ROOT],
                         cell_geometry_yaml=cell_yaml)


@pytest.fixture(scope="session")
def collision_bare(kin):
    """Self-collision-only checker (no world boxes)."""
    from rebot_planner.core.collision_core import CollisionCore

    _require(os.path.join(MESH_ROOT, "rebotarm_bringup"))
    return CollisionCore(kin, package_dirs=[MESH_ROOT])


@pytest.fixture(scope="session")
def ready(kin):
    """Deterministic collision-free start: vendor-style ready pose
    (TCP at (0.30, 0, 0.30), approach pitched 0.7 rad)."""
    R0 = rotation_pitch_forward(0.7)
    T0 = make_tcp(R0, (0.30, 0.0, 0.30))
    q0, err = kin.solve_tcp(T0, np.zeros(6), iters=600)
    assert err < 1e-4, f"ready-pose IK failed to converge: {err}"
    return {"q": q0, "T": T0, "R": R0}
