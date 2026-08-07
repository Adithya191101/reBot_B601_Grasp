"""Tensor -> USD link-transform sync for the session-repaired DM asset.

Why this exists, proven by ``b601_render_experiment.py`` (artifacts/render_exp/):

The DM USD's nine nested rigid bodies need the P0 session repair --
``resetXformStack`` plus an explicit transform op -- to be a valid PhysX
articulation at all. But **resetXformStack disables PhysX's own physics->USD
writeback on those prims**: with the repair active, PhysX still writes live
``xformOp:translate/orient`` values (observed in forensics), yet the composed
transform ignores them, and re-authoring the repair as translate/orient ops
does not help (measured: USD range 0.00 mm over a 77 mm sweep). The renderer
draws the composed USD result, so the arm renders frozen while plain rigid
bodies (whose writeback works) render fine.

Fix: push the PhysX tensor-API link transforms into the repair ops whenever a
correct picture is needed. Measured result: USD tracks physics to 0.03 mm and
a joint sweep changes ~109,000 pixels where the frozen render changed ~130.

Throttle the calls (per captured frame, not per physics step) and batch them in
an ``Sdf.ChangeBlock`` -- unthrottled per-step writes destabilised Kit.
"""

from __future__ import annotations

import numpy as np


class LinkUsdSync:
    """Writes articulation link world transforms into the session repair ops."""

    def __init__(self, stage, monitor, repair_body_paths: list):
        from pxr import UsdGeom

        self._stage = stage
        self._monitor = monitor
        self._targets = []
        for path in repair_body_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
            transform_ops = [o for o in ops
                             if o.GetOpType() == UsdGeom.XformOp.TypeTransform]
            name = path.rsplit("/", 1)[-1]
            if transform_ops and name in monitor.body_names:
                self._targets.append(
                    (monitor.body_names.index(name), transform_ops[0]))

    @property
    def link_count(self) -> int:
        return len(self._targets)

    def rebind(self, monitor) -> None:
        """Point the sync at a fresh StateMonitor after world.reset()."""
        self._monitor = monitor

    def push(self) -> None:
        """Sync all repaired links to the current physics state.

        world.reset() rebuilds the PhysX simulation view and steps the world
        while doing so; captures fired inside that window see an invalidated
        tensor handle. Skipping the frame is correct -- the previous synced
        transforms stay on the prims, and the caller re-binds a fresh monitor
        right after the reset.
        """
        from pxr import Gf, Sdf, Usd

        from grasp_smoke.geometry import rotation_from_quaternion

        try:
            links = self._monitor._link_transforms()
        except Exception:                                      # noqa: BLE001
            return
        with Usd.EditContext(self._stage,
                             Usd.EditTarget(self._stage.GetSessionLayer())), \
                Sdf.ChangeBlock():
            for li, op in self._targets:
                p = links[li, :3]
                q = links[li, 3:]                    # xyzw
                R = rotation_from_quaternion(np.asarray(q, dtype=np.float64))
                m = Gf.Matrix4d()
                m.SetIdentity()
                m.SetRotateOnly(Gf.Matrix3d(*[float(v) for v in R.T.flatten()]))
                m.SetTranslateOnly(Gf.Vec3d(float(p[0]), float(p[1]),
                                            float(p[2])))
                op.Set(m)
