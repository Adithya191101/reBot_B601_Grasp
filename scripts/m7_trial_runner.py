#!/usr/bin/env python3
"""M7: gantry-avoidance pair trials against the M5/M6 sim stack.

Runs INSIDE the Isaac ROS 4.5 cuMotion container (rebot-m6-cumotion
image, repo at /work, --network host, ROS_DOMAIN_ID=42, FastDDS UDP
profile) -- orchestrated by scripts/m7_gantry_avoidance.sh, which reuses
the M6 flow (scripts/m6_cumotion_trials.sh) unchanged apart from the M7
sampler/runner/verifier and artifacts/m7.

Infrastructure is imported UNCHANGED from scripts/m6_trial_runner.py
(TrialRunner node: /joint_states tracking, action helpers, readiness,
CollisionObject builder, gate constants).  What M7 changes is the GOAL:

  * moves are planned with ``plan_cspace`` + an explicit 6-joint
    ``goal_state`` instead of ``plan_pose``.  The M7 deliverable is a
    statement about the straight-line JOINT path between two known
    configurations, so the scored motion must end at exactly the joint
    configuration the chord proof (m7_sample_pairs) was computed for --
    a pose goal would let cuMotion's IK pick another branch and detach
    the executed motion from the proof.

Per scored trial (pairs from artifacts/m7/pair_poses.json, straight
chord qA->qB mesh-proven to hit ONLY the gantry):

  1. SETUP move to qA (world = table + gantry).  A setup PLANNING
     failure leaves the arm unmoved -> the pair is REJECTED (M6
     candidate-rejection convention) and a spare pair is used.
  2. SCORED move qA -> qB (world = table + gantry): plan, execute via
     /rebot_controller/follow_joint_trajectory (t=0 point dropped, M6
     convention), verify tracking convergence (<= 0.02 rad, M5 gate).
     A scored PLANNING failure also rejects the pair (arm stays at qA,
     chain intact); any EXECUTION or CONVERGENCE failure is a hard
     trial failure and fails the gate.

After N_REQUIRED scored trials, the first N_CONTRAST successful pairs
are re-run with the GANTRY REMOVED from the goal world (setup move back
to qA still uses the with-gantry world): same plan/execute/converge
criteria.  These contrast plans are expected to be near-straight -- the
"visible re-routing" evidence is scored host-side by
scripts/m7_verify_trials.py.

Writes artifacts/m7/trials_raw.json.  Exit 0 iff every scored and every
contrast trial planned, executed and converged (mesh verification and
deviation metrics happen on the host).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from control_msgs.action import FollowJointTrajectory  # noqa: E402
from isaac_ros_cumotion_interfaces.action import MotionPlan  # noqa: E402

# M6 infrastructure, unchanged.
from m6_trial_runner import (  # noqa: E402
    ARM_JOINTS, CONV_TIMEOUT_S, CONV_TOL_RAD, PLAN_TIMEOUT_FIRST_S,
    PLAN_TIMEOUT_S, TIME_DILATION, TrialRunner, make_collision_objects)

WORK = Path("/work")
PAIRS = WORK / "artifacts" / "m7" / "pair_poses.json"
OUT = WORK / "artifacts" / "m7" / "trials_raw.json"


class M7Runner(TrialRunner):
    """M6 TrialRunner with a cspace (joint-goal) move primitive."""

    def __init__(self) -> None:
        # Same wiring as TrialRunner.__init__ (M6, frozen) but under the
        # milestone's own node name.
        Node.__init__(
            self, "m7_trial_runner",
            parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.q_now = {}
        self.js_count = 0
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState
        self.create_subscription(JointState, "/joint_states",
                                 self._on_js, 10)
        self.plan_client = ActionClient(self, MotionPlan,
                                        "cumotion/motion_plan")
        self.fjt_client = ActionClient(
            self, FollowJointTrajectory,
            "/rebot_controller/follow_joint_trajectory")

    # -- cspace move ------------------------------------------------------

    def run_move(self, label: str, goal_q, world_objs,
                 first: bool) -> dict:
        """Plan (plan_cspace) -> execute (FJT) -> converge; M6 semantics."""
        rec = {"label": label,
               "goal_q": [float(v) for v in goal_q],
               "q_start": [self.q_now[j] for j in ARM_JOINTS]}

        goal = MotionPlan.Goal()
        goal.plan_cspace = True
        goal.use_current_state = True
        goal.use_planning_scene = True
        goal.world.collision_objects = world_objs
        goal.time_dilation_factor = float(TIME_DILATION)
        goal.goal_state.name = list(ARM_JOINTS)
        goal.goal_state.position = [float(v) for v in goal_q]

        t_plan0 = time.monotonic()
        result, err = self.await_action(
            self.plan_client, goal,
            PLAN_TIMEOUT_FIRST_S if first else PLAN_TIMEOUT_S)
        rec["plan_wall_s"] = round(time.monotonic() - t_plan0, 3)
        if result is None:
            rec["planned"] = False
            rec["error"] = f"motion_plan: {err}"
            return rec
        rec["planned"] = bool(result.success)
        rec["error_code"] = int(result.error_code.val)
        rec["planning_time_s"] = round(float(result.planning_time), 4)
        if not result.success or not result.planned_trajectory:
            rec["error"] = (f"cumotion planning failed "
                            f"(error_code={result.error_code.val})")
            return rec

        jt = result.planned_trajectory[0].joint_trajectory
        rec["joint_names"] = list(jt.joint_names)
        if list(jt.joint_names) != ARM_JOINTS:
            rec["error"] = f"unexpected joint order {list(jt.joint_names)}"
            rec["planned"] = False
            return rec
        points = [{
            "t": p.time_from_start.sec + p.time_from_start.nanosec * 1e-9,
            "q": [float(v) for v in p.positions],
        } for p in jt.points]
        rec["trajectory"] = points
        rec["duration_s"] = round(points[-1]["t"], 3) if points else 0.0

        # M6 convention: drop leading t=0 point(s) for the sim-JTC shim.
        fjt = FollowJointTrajectory.Goal()
        fjt.trajectory.joint_names = list(jt.joint_names)
        fjt.trajectory.points = [p for p in jt.points
                                 if (p.time_from_start.sec
                                     + p.time_from_start.nanosec * 1e-9)
                                 > 1e-9]
        if not fjt.trajectory.points:
            rec["error"] = "trajectory empty after dropping t=0 point"
            rec["executed"] = False
            return rec

        t_exec0 = time.monotonic()
        exec_result, err = self.await_action(
            self.fjt_client, fjt, rec["duration_s"] + 30.0)
        rec["exec_wall_s"] = round(time.monotonic() - t_exec0, 3)
        if exec_result is None:
            rec["executed"] = False
            rec["error"] = f"fjt: {err}"
            return rec
        rec["executed"] = (exec_result.error_code
                           == FollowJointTrajectory.Result.SUCCESSFUL)
        if not rec["executed"]:
            rec["error"] = (f"fjt error_code={exec_result.error_code} "
                            f"{exec_result.error_string!r}")
            return rec

        q_final = points[-1]["q"]
        t0 = time.monotonic()
        conv_err = float("inf")
        while time.monotonic() - t0 < CONV_TIMEOUT_S:
            rclpy.spin_once(self, timeout_sec=0.05)
            conv_err = max(abs(self.q_now[j] - q_final[i])
                           for i, j in enumerate(ARM_JOINTS))
            if conv_err <= CONV_TOL_RAD:
                break
        rec["converged"] = conv_err <= CONV_TOL_RAD
        rec["tracking_err_rad"] = round(conv_err, 5)
        if not rec["converged"]:
            rec["error"] = f"no convergence: {conv_err:.4f} rad"
        return rec


def move_ok(rec: dict) -> bool:
    return bool(rec.get("planned") and rec.get("executed")
                and rec.get("converged"))


def main() -> int:
    doc = json.loads(PAIRS.read_text())
    n_required = int(doc.get("n_pairs_required", 10))
    n_contrast = int(doc.get("n_contrast_required", 3))
    rclpy.init()
    node = M7Runner()
    trials: list[dict] = []
    contrast: list[dict] = []
    rejected: list[dict] = []
    try:
        node.wait_ready()
        stamp = node.get_clock().now().to_msg()
        world_gantry = make_collision_objects(doc["world_boxes"], stamp)
        world_no_gantry = make_collision_objects(
            doc["world_boxes_no_gantry"], stamp)
        assert any(o.id == "gantry" for o in world_gantry)
        assert not any(o.id == "gantry" for o in world_no_gantry)

        first = True
        for pair in doc["pairs"]:
            if len(trials) >= n_required:
                break
            idx = pair["index"]
            setup = node.run_move(f"pair{idx:02d}_setup", pair["A"]["q"],
                                  world_gantry, first=first)
            first = False
            if not setup.get("planned"):
                rejected.append({"pair_index": idx, "stage": "setup",
                                 "error": setup.get("error", "?")})
                node.get_logger().warn(
                    f"pair {idx:2d} REJECTED at setup "
                    f"({setup.get('error', '?')})")
                continue
            if not move_ok(setup):
                trials.append({"pair_index": idx, "setup": setup,
                               "scored": None, "ok": False})
                node.get_logger().error(
                    f"pair {idx:2d} setup HARD FAILURE "
                    f"({setup.get('error', '?')})")
                continue
            scored = node.run_move(f"pair{idx:02d}_scored", pair["B"]["q"],
                                   world_gantry, first=False)
            if not scored.get("planned"):
                rejected.append({"pair_index": idx, "stage": "scored",
                                 "error": scored.get("error", "?")})
                node.get_logger().warn(
                    f"pair {idx:2d} REJECTED at scored plan "
                    f"({scored.get('error', '?')}); arm holds at A")
                continue
            ok = move_ok(scored)
            trials.append({"pair_index": idx, "setup": setup,
                           "scored": scored, "ok": ok})
            node.get_logger().info(
                f"trial {len(trials):2d}/{n_required} (pair {idx:2d}): "
                f"{'OK' if ok else 'FAIL ' + scored.get('error', '?')} "
                f"plan={scored.get('planning_time_s', -1):.3f}s "
                f"dur={scored.get('duration_s', -1):.1f}s "
                f"track={scored.get('tracking_err_rad', -1):.4f}rad")

        # -- contrast phase: gantry REMOVED from the scored goal world --
        pair_by_index = {p["index"]: p for p in doc["pairs"]}
        for trial in [t for t in trials if t["ok"]][:n_contrast]:
            pair = pair_by_index[trial["pair_index"]]
            idx = pair["index"]
            setup = node.run_move(f"pair{idx:02d}_contrast_setup",
                                  pair["A"]["q"], world_gantry,
                                  first=False)
            if not move_ok(setup):
                contrast.append({"pair_index": idx, "setup": setup,
                                 "scored": None, "ok": False})
                node.get_logger().error(
                    f"contrast pair {idx:2d} setup FAILURE "
                    f"({setup.get('error', '?')})")
                continue
            scored = node.run_move(f"pair{idx:02d}_contrast",
                                   pair["B"]["q"], world_no_gantry,
                                   first=False)
            ok = move_ok(scored)
            contrast.append({"pair_index": idx, "setup": setup,
                             "scored": scored, "ok": ok})
            node.get_logger().info(
                f"contrast (pair {idx:2d}, NO gantry): "
                f"{'OK' if ok else 'FAIL ' + scored.get('error', '?')} "
                f"dur={scored.get('duration_s', -1):.1f}s "
                f"track={scored.get('tracking_err_rad', -1):.4f}rad")
    finally:
        n_ok = sum(1 for t in trials if t["ok"])
        n_cok = sum(1 for t in contrast if t["ok"])
        OUT.write_text(json.dumps({
            "milestone": "M7",
            "artifact": "trials_raw",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
            "planner": "isaac_ros_cumotion 4.5 standalone action server "
                       "cumotion/motion_plan, plan_cspace joint goals "
                       "(no MoveIt)",
            "time_dilation_factor": TIME_DILATION,
            "n_trials_required": n_required,
            "n_contrast_required": n_contrast,
            "n_trials": len(trials),
            "n_trials_ok": n_ok,
            "n_contrast": len(contrast),
            "n_contrast_ok": n_cok,
            "n_pairs_rejected": len(rejected),
            "rejected_pairs": rejected,
            "trials": trials,
            "contrast_trials": contrast,
        }, indent=2) + "\n")
        node.get_logger().info(
            f"wrote {OUT} ({n_ok}/{len(trials)} scored ok, "
            f"{n_cok}/{len(contrast)} contrast ok, "
            f"{len(rejected)} pairs rejected)")
        node.destroy_node()
        rclpy.shutdown()
    return 0 if (len(trials) == n_required and n_ok == n_required
                 and len(contrast) == n_contrast
                 and n_cok == n_contrast) else 1


if __name__ == "__main__":
    sys.exit(main())
