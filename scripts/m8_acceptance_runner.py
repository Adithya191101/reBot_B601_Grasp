#!/usr/bin/env python3
"""M8: mapping acceptance test runner (doc 15.4) against the live stack.

Runs INSIDE the Isaac ROS 4.5 container (rebot-m8-nvblox image, repo at
/work, --network host, ROS_DOMAIN_ID=42, FastDDS UDP profile) --
orchestrated by scripts/m8_mapping_acceptance.sh.  The M6/M7 runner
infrastructure is reused unchanged (M7Runner: plan_cspace moves through
the standalone cuMotion action server + FJT execution + convergence).

Phases (inputs from artifacts/m8/acceptance_configs.json):

  A/B  drive the arm through >= n_map_required distinct configurations
       (cuMotion cspace plans, goal-embedded static table+gantry world --
       the safe-transit convention; the ESDF world is ALWAYS also active
       since the planner runs read_esdf_world:=true).  At each held
       config: dwell for map integration, capture one raw-depth /
       robot-mask / masked-depth triple (stats + saved arrays = the doc's
       "display" evidence), query the nvblox ESDF grid and require
         * ZERO occupied voxels inside the robot's XRDF sphere volume
           (+margin) -- robot body absent from the map;
         * >= gantry_min_occupied_voxels inside the (1-voxel-dilated)
           gantry box -- gantry present.
  D1   setup to pair.A, then scored plan A->B with static world = TABLE
       ONLY: the gantry exists ONLY in the nvblox map, so any re-route is
       attributable to the mapped obstacle.  Execute + converge.
  C    hide the gantry in the sim (scene-command file through the shared
       /work mount), poll the ESDF until the stale gantry voxels clear.
  D2   contrast: setup back to A (map now clear), scored plan A->B with
       the SAME table-only static world -- expected near-straight through
       the old gantry volume (host verifier gates the deviation ratio).
  C2   restore the gantry, poll until the map contains it again (stale
       map "cleared or updated when the gantry moves", both directions).

Writes artifacts/m8/acceptance_raw.json (+ evidence arrays under
artifacts/m8/evidence/).  Exit 0 iff every phase completed; the numeric
gates are re-applied host-side by scripts/m8_verify_mapping.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geometry_msgs.msg import Point, Vector3  # noqa: E402
from nvblox_msgs.srv import EsdfAndGradients  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

from m6_trial_runner import make_collision_objects  # noqa: E402
from m7_trial_runner import M7Runner, move_ok  # noqa: E402

WORK = Path("/work")
CFG = WORK / "artifacts" / "m8" / "acceptance_configs.json"
OUT = WORK / "artifacts" / "m8" / "acceptance_raw.json"
EVIDENCE = WORK / "artifacts" / "m8" / "evidence"
SCENE_CMD = WORK / "artifacts" / "m8" / "scene_cmd.json"

ESDF_SERVICE = "/nvblox_node/get_esdf_and_gradient"
TOPIC_RAW_DEPTH = "/overhead_camera/aligned_depth_to_color/image_raw"
TOPIC_MASK = "/cumotion/camera_1/robot_mask"
TOPIC_WORLD_DEPTH = "/cumotion/camera_1/world_depth"

DWELL_S = 4.0                 # map-integration dwell at each held config
UNKNOWN_VALUE = -1000.0       # nvblox EsdfAndGradients unobserved marker
ESDF_TIMEOUT_S = 30.0
POLL_PERIOD_S = 3.0


class M8Runner(M7Runner):
    """M7 cspace-move runner + ESDF service client + image capture."""

    def __init__(self) -> None:
        Node.__init__(
            self, "m8_acceptance_runner",
            parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.q_now = {}
        self.js_count = 0
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState
        from control_msgs.action import FollowJointTrajectory
        from isaac_ros_cumotion_interfaces.action import MotionPlan
        self.create_subscription(JointState, "/joint_states",
                                 self._on_js, 10)
        self.plan_client = ActionClient(self, MotionPlan,
                                        "cumotion/motion_plan")
        self.fjt_client = ActionClient(
            self, FollowJointTrajectory,
            "/rebot_controller/follow_joint_trajectory")
        self.esdf_client = self.create_client(EsdfAndGradients, ESDF_SERVICE)
        self._img = {}
        self._img_want = set()
        self.create_subscription(Image, TOPIC_RAW_DEPTH,
                                 lambda m: self._on_img("raw_depth", m), 5)
        self.create_subscription(Image, TOPIC_MASK,
                                 lambda m: self._on_img("mask", m), 5)
        self.create_subscription(Image, TOPIC_WORLD_DEPTH,
                                 lambda m: self._on_img("world_depth", m), 5)
        ws = json.loads(CFG.read_text())["workspace"]
        self.ws_min = [float(v) for v in ws["min"]]
        self.ws_size = [float(b - a) for a, b in zip(ws["min"], ws["max"])]

    def await_action(self, client, goal, timeout_s: float):
        """M8 addition: every MotionPlan goal must refresh the planner's
        ESDF grid.  MEASURED failure without this: the 4.5 planner calls
        the nvblox service only once at startup ("Initialized grid from
        nvblox" appears a single time), so every later plan runs against
        that stale first grid -- D2 kept re-routing around a gantry that
        nvblox had already cleared (dev ratio 1.01).  The refresh is a
        PER-GOAL bool in MotionPlan.action (update_esdf), which the
        frozen M6/M7 runners never set; the node-level
        update_esdf_on_request parameter alone does not re-query.
        cuMotion uses the ESDF at planning time only (upstream issue
        NVIDIA-ISAAC-ROS/isaac_ros_cumotion#45), so per-plan refresh is
        the supported granularity."""
        from isaac_ros_cumotion_interfaces.action import MotionPlan
        if isinstance(goal, MotionPlan.Goal):
            goal.update_esdf = True
        return super().await_action(client, goal, timeout_s)

    # -- image capture ----------------------------------------------------

    def _on_img(self, key: str, msg: Image) -> None:
        if key in self._img_want:
            self._img[key] = msg
            self._img_want.discard(key)

    def grab_images(self, timeout_s: float = 15.0) -> dict:
        """One fresh raw-depth + mask + masked-depth triple -> stats."""
        self._img.clear()
        self._img_want = {"raw_depth", "mask", "world_depth"}
        t0 = time.monotonic()
        while self._img_want and time.monotonic() - t0 < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)
        out = {"captured": sorted(self._img)}
        raw = mask = wd = None
        if "raw_depth" in self._img:
            m = self._img["raw_depth"]
            raw = np.frombuffer(m.data, dtype=np.float32).reshape(
                m.height, m.width)
            out["raw_valid_px"] = int(np.isfinite(raw).sum())
        if "mask" in self._img:
            m = self._img["mask"]
            dtype = {"mono8": np.uint8, "8UC1": np.uint8,
                     "mono16": np.uint16, "16UC1": np.uint16,
                     "32FC1": np.float32}.get(m.encoding)
            out["mask_encoding"] = m.encoding
            if dtype is not None:
                mask = np.frombuffer(m.data, dtype=dtype).reshape(
                    m.height, m.width)
                valid = (np.nan_to_num(mask.astype(np.float32), nan=0.0)
                         > 0.0)
                out["mask_px"] = int(valid.sum())
        if "world_depth" in self._img:
            m = self._img["world_depth"]
            wd = np.frombuffer(m.data, dtype=np.float32).reshape(
                m.height, m.width)
            out["world_depth_invalid_px"] = int(
                (~np.isfinite(wd) | (wd <= 0.0)).sum())
        if raw is not None and wd is not None:
            removed = (np.isfinite(raw) & (raw > 0)
                       & (~np.isfinite(wd) | (wd <= 0.0)))
            out["masked_out_px"] = int(removed.sum())
        return out, {"raw_depth": raw, "mask": mask, "world_depth": wd}

    # -- ESDF -------------------------------------------------------------

    def query_esdf(self, update: bool = True):
        req = EsdfAndGradients.Request()
        req.update_esdf = bool(update)
        req.use_aabb = True
        req.frame_id = "base_link"
        req.aabb_min_m = Point(x=self.ws_min[0], y=self.ws_min[1],
                               z=self.ws_min[2])
        req.aabb_size_m = Vector3(x=self.ws_size[0], y=self.ws_size[1],
                                  z=self.ws_size[2])
        fut = self.esdf_client.call_async(req)
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > ESDF_TIMEOUT_S:
                raise RuntimeError("ESDF service timeout")
            rclpy.spin_once(self, timeout_sec=0.1)
        res = fut.result()
        if not res.success:
            raise RuntimeError("ESDF service returned success=false")
        dims = [d.size for d in res.esdf_and_gradients.layout.dim]
        grid = np.array(res.esdf_and_gradients.data,
                        dtype=np.float32).reshape(dims)
        origin = np.array([res.origin_m.x, res.origin_m.y, res.origin_m.z])
        return grid, origin, float(res.voxel_size_m)

    @staticmethod
    def occupied_centers(grid, origin, voxel):
        """World-frame centers of occupied voxels (esdf<0, not unknown)."""
        occ = (grid < 0.0) & (grid != UNKNOWN_VALUE)
        idx = np.argwhere(occ)
        return origin[None, :] + (idx + 0.5) * voxel, occ

    def map_stats(self, spheres_xyzr, gantry_box, margin: float) -> dict:
        grid, origin, voxel = self.query_esdf(update=True)
        centers, occ = self.occupied_centers(grid, origin, voxel)
        st = {"n_occupied": int(occ.sum()),
              "n_unknown": int((grid == UNKNOWN_VALUE).sum()),
              "n_voxels": int(grid.size)}
        # robot-volume check
        n_robot = 0
        worst = None
        if spheres_xyzr and len(centers):
            sph = np.asarray(spheres_xyzr, dtype=float)
            d = np.linalg.norm(centers[:, None, :] - sph[None, :, :3],
                               axis=2) - sph[None, :, 3]
            inside = d <= margin
            n_robot = int(np.any(inside, axis=1).sum())
            if n_robot:
                i, j = np.unravel_index(np.argmin(d), d.shape)
                worst = {"voxel_center": [round(float(v), 3)
                                          for v in centers[i]],
                         "sphere": [round(float(v), 3) for v in sph[j]],
                         "depth_m": round(float(-d[i, j]), 4)}
        st["robot_occupied_voxels"] = n_robot
        if worst:
            st["worst_robot_voxel"] = worst
        # gantry box (1-voxel dilated) check
        c = np.asarray(gantry_box["center"], dtype=float)
        h = np.asarray(gantry_box["size"], dtype=float) / 2.0 + voxel
        if len(centers):
            in_box = np.all(np.abs(centers - c[None, :]) <= h[None, :],
                            axis=1)
            st["gantry_occupied_voxels"] = int(in_box.sum())
        else:
            st["gantry_occupied_voxels"] = 0
        return st

    # -- scene command ----------------------------------------------------

    def set_gantry(self, visible: bool) -> None:
        SCENE_CMD.write_text(json.dumps({"gantry_visible": bool(visible)})
                             + "\n")
        self.get_logger().info(f"scene command: gantry_visible={visible}")

    def wait_gantry_state(self, present: bool, threshold_lo: int,
                          threshold_hi: int, timeout_s: float,
                          gantry_box) -> dict:
        """Poll the ESDF until the gantry voxel count crosses a gate."""
        t0 = time.monotonic()
        history = []
        while True:
            st = self.map_stats([], gantry_box, 0.0)
            n = st["gantry_occupied_voxels"]
            history.append({"t_s": round(time.monotonic() - t0, 1),
                            "gantry_occupied_voxels": n})
            ok = (n >= threshold_hi) if present else (n <= threshold_lo)
            if ok:
                return {"ok": True, "elapsed_s": round(
                    time.monotonic() - t0, 1), "history": history}
            if time.monotonic() - t0 > timeout_s:
                return {"ok": False, "elapsed_s": round(
                    time.monotonic() - t0, 1), "history": history}
            end = time.monotonic() + POLL_PERIOD_S
            while time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.1)

    def wait_ready(self) -> None:  # extends the M6 readiness with ESDF
        super().wait_ready()
        if not self.esdf_client.wait_for_service(timeout_sec=60.0):
            raise RuntimeError("ESDF service not available")


def save_evidence(tag: str, arrays: dict) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for key, arr in arrays.items():
        if arr is not None:
            np.save(EVIDENCE / f"{tag}_{key}.npy", arr)


def main() -> int:
    doc = json.loads(CFG.read_text())
    gates = doc["gates"]
    gantry_box = doc["gantry_box"]
    n_required = int(doc["n_map_required"])
    margin = float(gates["robot_clearance_margin_m"])

    rclpy.init()
    node = M8Runner()
    result = {"map_configs": [], "rejected_configs": [],
              "phase_d1": None, "phase_c_clear": None,
              "phase_d2": None, "phase_c2_recover": None,
              "hard_failure": None}
    rc = 1
    try:
        node.wait_ready()
        stamp = node.get_clock().now().to_msg()
        world_full = make_collision_objects(doc["world_boxes"], stamp)
        world_table = make_collision_objects(doc["world_boxes_table_only"],
                                             stamp)
        assert any(o.id == "gantry" for o in world_full)
        assert not any(o.id == "gantry" for o in world_table)
        node.set_gantry(True)  # deterministic start state

        # ---- phase A/B: >= 5 configs, robot absent + gantry present -----
        first = True
        for cfg in doc["map_configs"]:
            if len(result["map_configs"]) >= n_required:
                break
            move = node.run_move(f"mapcfg{cfg['index']:02d}", cfg["q"],
                                 world_full, first=first)
            first = False
            if not move.get("planned"):
                result["rejected_configs"].append(
                    {"index": cfg["index"],
                     "error": move.get("error", "?")})
                node.get_logger().warn(
                    f"map config {cfg['index']} REJECTED at planning "
                    f"({move.get('error', '?')})")
                continue
            if not move_ok(move):
                result["hard_failure"] = (f"map config {cfg['index']} "
                                          f"execution: "
                                          f"{move.get('error', '?')}")
                break
            node.spin_for(DWELL_S)
            img_stats, arrays = node.grab_images()
            save_evidence(f"config{cfg['index']:02d}", arrays)
            st = node.map_stats(cfg["spheres_xyzr"], gantry_box, margin)
            rec = {"index": cfg["index"], "q": cfg["q"], "tcp": cfg["tcp"],
                   "move": {k: move[k] for k in
                            ("planned", "executed", "converged",
                             "tracking_err_rad", "planning_time_s")
                            if k in move},
                   "images": img_stats, "map": st,
                   "robot_absent": st["robot_occupied_voxels"] == 0,
                   "gantry_present": (st["gantry_occupied_voxels"]
                                      >= gates["gantry_min_occupied_voxels"])}
            result["map_configs"].append(rec)
            node.get_logger().info(
                f"map config {cfg['index']}: robot_occ="
                f"{st['robot_occupied_voxels']} gantry_occ="
                f"{st['gantry_occupied_voxels']} mask_px="
                f"{img_stats.get('mask_px')} -> "
                f"{'OK' if rec['robot_absent'] and rec['gantry_present'] else 'BAD'}")
        if result["hard_failure"]:
            raise RuntimeError(result["hard_failure"])
        if len(result["map_configs"]) < n_required:
            raise RuntimeError("not enough map configs completed")

        # ---- phase D1: avoid the MAPPED gantry --------------------------
        pair = doc["pairs"][0]
        d1 = {"pair_index": pair["index"]}
        setup = node.run_move("d1_setup_A", pair["A"]["q"], world_full,
                              first=False)
        d1["setup"] = setup
        if not move_ok(setup):
            raise RuntimeError(f"D1 setup failed: {setup.get('error')}")
        node.spin_for(2.0)
        d1["map_before"] = node.map_stats([], gantry_box, margin)
        scored = node.run_move("d1_scored_mapped_gantry", pair["B"]["q"],
                               world_table, first=False)
        d1["scored"] = scored
        result["phase_d1"] = d1
        if not move_ok(scored):
            raise RuntimeError(f"D1 scored failed: {scored.get('error')}")

        # ---- phase C: remove the gantry, stale map must clear -----------
        node.set_gantry(False)
        clear = node.wait_gantry_state(
            present=False,
            threshold_lo=int(gates["gantry_max_stale_voxels"]),
            threshold_hi=int(gates["gantry_min_occupied_voxels"]),
            timeout_s=float(gates["map_clear_timeout_s"]),
            gantry_box=gantry_box)
        result["phase_c_clear"] = clear
        if not clear["ok"]:
            raise RuntimeError("stale gantry did not clear from the map")

        # ---- phase D2: same plan, map cleared -> near-straight ----------
        d2 = {"pair_index": pair["index"]}
        setup = node.run_move("d2_setup_A", pair["A"]["q"], world_table,
                              first=False)
        d2["setup"] = setup
        if not move_ok(setup):
            raise RuntimeError(f"D2 setup failed: {setup.get('error')}")
        d2["map_before"] = node.map_stats([], gantry_box, margin)
        scored = node.run_move("d2_scored_cleared_map", pair["B"]["q"],
                               world_table, first=False)
        d2["scored"] = scored
        result["phase_d2"] = d2
        if not move_ok(scored):
            raise RuntimeError(f"D2 scored failed: {scored.get('error')}")

        # ---- phase C2: restore the gantry, map must rebuild -------------
        node.set_gantry(True)
        recover = node.wait_gantry_state(
            present=True,
            threshold_lo=int(gates["gantry_max_stale_voxels"]),
            threshold_hi=int(gates["gantry_min_occupied_voxels"]),
            timeout_s=float(gates["map_present_timeout_s"]),
            gantry_box=gantry_box)
        result["phase_c2_recover"] = recover
        if not recover["ok"]:
            raise RuntimeError("gantry did not reappear in the map")
        rc = 0
    except Exception as exc:  # recorded, gate fails
        result["hard_failure"] = result["hard_failure"] or str(exc)
        node.get_logger().error(f"M8 runner failure: {exc}")
    finally:
        result.update({
            "milestone": "M8",
            "artifact": "acceptance_raw",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
            "esdf_service": ESDF_SERVICE,
            "unknown_value": UNKNOWN_VALUE,
            "dwell_s": DWELL_S,
        })
        OUT.write_text(json.dumps(result, indent=2) + "\n")
        node.get_logger().info(f"wrote {OUT} (rc={rc})")
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
