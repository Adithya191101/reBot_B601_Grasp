"""Shared pure helpers for the rebot_planner test suite.

Kept out of conftest.py so tests can import them by a unique module name
(``from planner_testlib import ...``) -- importing from ``conftest``
breaks when several packages' suites run in one pytest invocation.
"""

import numpy as np


def rotation_pitch_forward(pitch: float) -> np.ndarray:
    """Demo-style TCP rotation: approach pitched down from +x by ``pitch``.

    Column 0 (TCP x) is the approach/jaw direction, column 1 the opening
    axis; composed with the pi roll that keeps joint6 away from its
    +/-pi limit (the banana demo's grasp-equivalent branch choice).
    """
    a = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    y = np.array([0.0, 1.0, 0.0])
    z = np.cross(a, y)
    z /= np.linalg.norm(z)
    y2 = np.cross(z, a)
    return np.column_stack([a, y2, z]) @ np.diag([1.0, -1.0, -1.0])


def make_tcp(R: np.ndarray, pos) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T
