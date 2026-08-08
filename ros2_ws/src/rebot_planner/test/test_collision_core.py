"""Unit tests for rebot_planner.core.collision_core.

Self-collision + world-box (cell geometry) checking on the canonical URDF
collision meshes.
"""

import numpy as np
import pytest

from rebot_planner.core import path_core
from rebot_planner.core.collision_core import (
    DISABLED_SELF_COLLISION_PAIRS,
    _gantry_box,
    load_cell_geometry,
)

from planner_testlib import make_tcp, rotation_pitch_forward


#: Found by random sampling (seed 3): the arm folded back onto its base.
SELF_COLLIDING_Q = (1.965, -2.609, -0.112, 0.276, 0.336, 2.955)


# ---- construction --------------------------------------------------------


def test_world_boxes_loaded(collision):
    names = [b.name for b in collision.world_boxes]
    assert names == ["table", "gantry"]
    table = collision.world_boxes[0]
    assert table.size == (0.60, 0.45, 0.05)
    assert table.center[2] == pytest.approx(-0.025)  # top face at z=0
    gantry = collision.world_boxes[1]
    assert gantry.size == (0.20, 0.04, 0.04)         # span along x
    assert gantry.center[2] == pytest.approx(0.22)   # lower edge 0.20 + h/2


def test_zones_are_metadata_not_obstacles(collision):
    assert set(collision.zones) == {"pick_zone", "place_zone"}
    assert collision.zones["pick_zone"]["center_xy"] == [0.30, 0.14]
    assert collision.zones["place_zone"]["size_xy"] == [0.14, 0.14]
    # zones must NOT appear among collision geometries
    geom_names = [g.name for g in collision.geom_model.geometryObjects]
    assert not any("zone" in n for n in geom_names)


def test_disabled_adjacent_pairs_removed(collision_bare):
    gm = collision_bare.geom_model
    link_of = collision_bare._geom_link
    active = {frozenset((link_of[p.first], link_of[p.second]))
              for p in gm.collisionPairs}
    for pair in DISABLED_SELF_COLLISION_PAIRS:
        assert frozenset(pair) not in active, f"SRDF pair {pair} still active"
    # but non-adjacent pairs remain (e.g. base_link vs the gripper)
    assert frozenset(("base_link", "gripper_link")) in active


def test_gantry_parametrization():
    box = _gantry_box("g", {
        "center_xy": [0.40, -0.05],
        "span_axis": "y",
        "span_m": 0.18,
        "bar_width_m": 0.03,
        "bar_height_m": 0.05,
        "lower_edge_z_m": 0.25,
    })
    assert box.size == (0.03, 0.18, 0.05)
    assert box.center == (0.40, -0.05, 0.25 + 0.025)
    with pytest.raises(ValueError):
        _gantry_box("g", {"center_xy": [0, 0], "span_axis": "z",
                          "span_m": 0.2, "lower_edge_z_m": 0.2})


def test_load_cell_geometry_rejects_unknown_type(tmp_path):
    bad = tmp_path / "cell.yaml"
    bad.write_text("obstacles:\n  blob:\n    type: sphere\n")
    with pytest.raises(ValueError):
        load_cell_geometry(str(bad))


# ---- configuration checks ------------------------------------------------


def test_neutral_and_ready_configs_are_free(collision, ready):
    assert collision.check_config(np.zeros(6)).ok
    assert collision.check_config(ready["q"]).ok


def test_table_box_collision_detected(kin, collision):
    """A configuration whose fingers dip below the tabletop must collide
    with the table world box."""
    T = make_tcp(rotation_pitch_forward(np.pi / 2), (0.35, 0.0, -0.03))
    q, err = kin.solve_tcp(T, np.zeros(6), iters=600)
    assert err < 1e-4
    report = collision.check_config(q)
    assert not report.ok
    assert any("world/table" in pair for pair in report.pairs)
    assert collision.in_collision(q)


def test_gantry_collision_detected(kin, collision):
    T = make_tcp(rotation_pitch_forward(0.7), (0.45, 0.10, 0.22))
    q, err = kin.solve_tcp(T, np.zeros(6), iters=600)
    assert err < 1e-3
    report = collision.check_config(q)
    assert not report.ok
    assert any("world/gantry" in pair for pair in report.pairs)


def test_self_collision_detected(collision_bare):
    report = collision_bare.check_config(np.asarray(SELF_COLLIDING_Q))
    assert not report.ok
    assert ("base_link_0", "link4_0") in report.pairs


# ---- path checks ---------------------------------------------------------


def test_path_check_rejects_table_sweep(kin, collision, ready):
    """A kinematically valid descent THROUGH the tabletop is planned by
    path_core (it knows nothing of obstacles) and must then be rejected by
    the collision gate -- the planner rejects, it cannot re-route
    (cuMotion later)."""
    T_goal = make_tcp(ready["R"], (0.30, 0.0, -0.02))
    plan = path_core.plan_linear(kin, ready["q"], T_goal)
    assert plan.ok, plan.reason  # kinematics alone cannot see the table
    check = collision.check_path(plan.waypoints)
    assert not check.ok
    assert check.reason == "path in collision"
    assert check.failed_segment >= 0
    assert any("world/table" in pair for pair in check.pairs)
    assert check.checked_configurations > len(plan.waypoints)


def test_path_check_passes_safe_descent(kin, collision, ready):
    T_goal = make_tcp(ready["R"], (0.30, 0.0, 0.12))
    plan = path_core.plan_linear(kin, ready["q"], T_goal)
    assert plan.ok, plan.reason
    check = collision.check_path(plan.waypoints)
    assert check.ok, (check.reason, check.pairs)
    # interpolation really is denser than the waypoints
    assert check.checked_configurations > len(plan.waypoints)


def test_path_check_interpolates_between_waypoints(collision):
    """Two collision-free endpoints whose straight joint-space segment
    sweeps the tool through the table must still be rejected -- proof the
    checker tests INTERPOLATED configurations, not just waypoints."""
    # Found by random sampling (seed 11): both endpoints free, midpoint
    # drives the gripper into the table box.
    q_a = np.array([-2.664, -2.486, -1.837, 0.477, -0.187, 0.082])
    q_b = np.array([1.435, -2.875, -1.028, -1.657, -0.764, 1.374])
    assert collision.check_config(q_a).ok
    assert collision.check_config(q_b).ok
    mid = 0.5 * (q_a + q_b)
    assert not collision.check_config(mid).ok
    check = collision.check_path([q_a, q_b])
    assert not check.ok
    assert any("world/" in name for pair in check.pairs for name in pair)


def test_empty_path_rejected(collision):
    assert not collision.check_path([]).ok
