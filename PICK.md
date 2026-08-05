# P1 + P2 — a physics-driven B601-DM grasp and lift

**Run:** 2026-08-05 · **Gate:** PLAN.md §3.1 · **Result: PASS, 20/20 checks, 3/3 repetitions**
**Record:** `artifacts/b601_pick/b601_pick.json`

Reproduce:

```bash
cd ~/reBot_B601_Grasp
TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=N \
  ~/isaaclab-venv/bin/python scripts/b601_pick.py \
    --repair-nested-xforms --out artifacts/b601_pick/b601_pick.json
```

## The result

| Measure | Value | Gate |
|---|---|---|
| Object rise | **59.8 mm** | ≥ 50 mm ✓ |
| Hold duration | 1.0 s | ≥ 1.0 s ✓ |
| Object z drift while held | **0.30 µm** | < 10 mm ✓ |
| Fall on release | **321.6 mm** | ≥ 10 mm ✓ |
| Jaw rise (arm) | 80.7 mm | — |
| Base drift through the pick | 2.1 µm / 2.9e-05 rad | within P0 tolerances ✓ |
| Repetitions | **3/3 identical** to every printed digit | 3/3 ✓ |

Nothing is teleported, parented, or attached. The arm and both fingers are driven
by articulation position targets while physics steps; the object moves only
because it is squeezed between two colliders. `set_joints_default_state` is used
once, to seed the arm's pose at reset — an explicit reset, which PLAN.md §3.1
permits and which is not scored motion.

**The release check is what makes the lift meaningful.** An object that stays put
when the jaw opens was never held by contact. This one falls 321.6 mm.

## Asset defect #4 — the one that blocked the grasp

P0 found three defects. Trying to actually grasp with the asset found a fourth,
and it was the blocker.

**The finger colliders are authored `physics:approximation = "convexHull"`.** A
convex hull fills in the concave inner face of a jaw, so each finger becomes a
solid blob spanning the throat. Measured consequences:

- The two hulls **overlap when closed** — which is *why* PhysX self-collision had
  to be disabled in P0 at all. That was a symptom, not the root cause.
- At **full open** the free gap is roughly **24 mm**, against an authored finger
  travel of **143 mm**.
- A 40 mm object cannot enter the jaw. It is ejected on contact: in one run the
  cube was flung to (0.502, 0.136) and landed on the floor.

Session-only fix, in the same spirit as P0's three: set
`physics:approximation = "convexDecomposition"` on both finger collider meshes.
The geometry is authored `instanceable = true`, so the colliders live in a
prototype where a plain traversal never reaches them and an instance proxy cannot
carry an opinion — the two finger subtrees are de-instanced in the session layer
first (10 prims). **The vendor USD on disk is not modified**; its SHA-256 is
recorded in the result and is unchanged.

### The measurement that proves it

The first-contact aperture, measured against the object by closing in 0.5 mm
increments until both fingers show drive-tracking resistance on two consecutive
steps:

| Collider approximation | First contact | Implied inset per finger |
|---|---|---|
| `convexHull` (as shipped) | 60.0 mm per finger | **40 mm** |
| `convexDecomposition` (session fix) | **20.5 mm per finger** | **0.5 mm** |

41.0 mm measured aperture for a 40.0 mm object. **This is the number P0 could not
give.** P0 measured finger *link-origin* separation (0.059 / 71.445 / 142.943 mm),
which describes the joints, not the jaw — with the shipped hulls it overstated the
usable aperture by roughly 80 mm.

## Four bugs found on the way

Each one was a plausible-looking failure that a less instrumented run would have
misattributed.

1. **Stale pose reads.** `get_world_pose` reads the USD stage, which PhysX does
   not write back during simulation, so every measured lift was exactly
   `-1.9e-06 m`. Live poses must come from the PhysX tensor API — the same path
   `StateMonitor` uses.
2. **Lift direction assumed, not measured.** The first candidate sweep drove the
   tool *down* monotonically (−24 mm to −216 mm). The lift axis is `+joint2` at
   roughly 0.24 m/rad near this pose. The script now sweeps candidates and
   *measures* the rise rather than reasoning about the kinematics.
3. **Invalidated physics view.** `world.reset()` after spawning scene objects
   rebuilds the PhysX simulation view and invalidates the cached tensor handle
   ("Failed to get link transforms from backend").
4. **Approach path collision.** The support column stands where the jaw works, so
   the default-pose → grasp-pose sweep hit it and left the jaw 86 mm off, placing
   the object beside its support. Fixed by seeding the arm's default state so it
   materialises at the grasp pose instead of sweeping to it, plus a guard that
   fails if the jaw shifts more than 5 mm between calibration and execution.

## Object geometry — a deliberate, stated deviation

PLAN.md §3.1 says "a lightweight ~40 mm object". The object here is **40 × 40 ×
100 mm**, 50 g: 40 mm across the closing direction — the dimension that defines
the grasp — but tall enough that the jaw works **50 mm above whatever supports
it**. The finger colliders are ~89 × 93 × 39 mm, so with a 40 mm cube on a
pedestal the fingers clamp the *pedestal* and the lift leaves the cube behind.
This is also closer to what the vendor demo actually grasps (bottles, cups).

If a literal 40 mm cube is wanted, the support has to be removed from the jaw's
working volume — a cantilevered shelf, or lowering the whole task to the ground
plane — not a change to the gripper.

## What this does and does not establish

**Does:** the shipped DM asset can be driven through contact to grasp and lift an
object in Isaac Sim, repeatably, with the fingers doing the work.

**Does not:** it is a *fixed* grasp pose with known object placement. There is no
IK (P3), no hand-eye calibration (P3b), and no perception in the loop (P4). The
grasp pose and the lift pose are calibrated by measurement inside the script, not
computed from an object pose.
