"""Pure validation/conversion core for the reBot gripper adapter.

Design doc sec. 4 and 9.3:

* the public GripperCommand ``position`` is the master prismatic-joint
  coordinate ``q_jaw_m`` in METRES (never total aperture, never motor rad);
* real conversion: clamp to the calibrated model range, then
  monotonic piecewise-linear interpolation to a DM4310 ``motor_rad`` goal;
  motor state maps back through the inverse table to ``q_jaw_m`` feedback;
* sim conversion: trajectory positions ``[q_jaw_m, q_jaw_m]`` for the two
  mimic-coupled jaw joints;
* real mode must refuse to start with an empty or non-monotonic
  calibration table (sec. 4.5);
* per-goal state machine:
  ``IDLE -> CONVERTING -> FORWARDED -> TRACKING -> SUCCEEDED``
  with ``ABORTED``/``CANCELED`` branches from any active state.

This module is pure Python and must never import rclpy.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .limits import (
    DEFAULT_GOAL_TOLERANCE_Q_M,
    Q_JAW_DEMO_MAX_M,
    Q_JAW_URDF_MAX_M,
    Q_JAW_URDF_MIN_M,
)

SUPPORTED_INTERPOLATION = "monotonic_piecewise_linear"


class CalibrationError(ValueError):
    """Raised when a calibration table is unusable for real mode."""


class RangeError(ValueError):
    """Raised when a command is out of range and clamping is disabled."""


@dataclass(frozen=True)
class CalibrationSample:
    """One calibration sample: all three gripper coordinates (sec. 4.5)."""

    q_jaw_m: float      # canonical master-joint coordinate, metres
    aperture_m: float   # measured inner opening, metres (diagnostics only)
    motor_rad: float    # measured DM4310 motor angle, radians


class CalibrationTable:
    """Monotonic piecewise-linear q_jaw_m <-> motor_rad map.

    Validates on construction. Real mode must construct this and treat
    :class:`CalibrationError` as a refusal to start.
    """

    def __init__(
        self,
        samples: Sequence[CalibrationSample],
        *,
        q_jaw_min_m: float = Q_JAW_URDF_MIN_M,
        q_jaw_max_m: float = Q_JAW_DEMO_MAX_M,
        clamp: bool = True,
        goal_tolerance_q_m: float = DEFAULT_GOAL_TOLERANCE_Q_M,
        interpolation: str = SUPPORTED_INTERPOLATION,
    ) -> None:
        if interpolation != SUPPORTED_INTERPOLATION:
            raise CalibrationError(
                f"unsupported interpolation {interpolation!r}; "
                f"expected {SUPPORTED_INTERPOLATION!r}"
            )
        if not samples:
            raise CalibrationError(
                "calibration table is empty; real mode must refuse to start"
            )
        if len(samples) < 2:
            raise CalibrationError(
                "calibration table needs at least two samples to interpolate"
            )
        ordered = sorted(samples, key=lambda s: s.q_jaw_m)
        for s in ordered:
            if not (
                math.isfinite(s.q_jaw_m)
                and math.isfinite(s.motor_rad)
                and math.isfinite(s.aperture_m)
            ):
                raise CalibrationError(f"non-finite calibration sample: {s}")
        q_vals = [s.q_jaw_m for s in ordered]
        m_vals = [s.motor_rad for s in ordered]
        if any(b <= a for a, b in zip(q_vals, q_vals[1:])):
            raise CalibrationError(
                "q_jaw_m samples are not strictly monotonic; refusing table"
            )
        increasing = all(b > a for a, b in zip(m_vals, m_vals[1:]))
        decreasing = all(b < a for a, b in zip(m_vals, m_vals[1:]))
        if not (increasing or decreasing):
            raise CalibrationError(
                "motor_rad samples are not strictly monotonic; refusing table"
            )
        if not (q_jaw_min_m < q_jaw_max_m):
            raise CalibrationError(
                f"invalid operational range [{q_jaw_min_m}, {q_jaw_max_m}]"
            )

        self._samples: Tuple[CalibrationSample, ...] = tuple(ordered)
        self._q: List[float] = q_vals
        self._m: List[float] = m_vals
        self._motor_increasing = increasing
        self.q_jaw_min_m = q_jaw_min_m
        self.q_jaw_max_m = q_jaw_max_m
        self.clamp = clamp
        self.goal_tolerance_q_m = goal_tolerance_q_m

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict) -> "CalibrationTable":
        """Build from the parsed ``config/gripper_calibration.yaml`` layout."""
        try:
            raw_samples = data["samples"] or []
            op = data.get("operational_range", {})
            samples = [
                CalibrationSample(
                    q_jaw_m=float(s["q_jaw_m"]),
                    aperture_m=float(s["aperture_m"]),
                    motor_rad=float(s["motor_rad"]),
                )
                for s in raw_samples
            ]
        except (KeyError, TypeError) as exc:
            raise CalibrationError(f"malformed calibration data: {exc}") from exc
        return cls(
            samples,
            q_jaw_min_m=float(op.get("q_jaw_min_m", Q_JAW_URDF_MIN_M)),
            q_jaw_max_m=float(op.get("q_jaw_max_m", Q_JAW_DEMO_MAX_M)),
            clamp=bool(data.get("clamp", True)),
            goal_tolerance_q_m=float(
                data.get("goal_tolerance_q_m", DEFAULT_GOAL_TOLERANCE_Q_M)
            ),
            interpolation=data.get("interpolation", SUPPORTED_INTERPOLATION),
        )

    @classmethod
    def from_yaml_file(cls, path: str) -> "CalibrationTable":
        import yaml  # local import: keep core importable without PyYAML

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise CalibrationError(f"calibration file {path!r} is not a mapping")
        return cls.from_dict(data)

    # -- properties --------------------------------------------------------

    @property
    def samples(self) -> Tuple[CalibrationSample, ...]:
        return self._samples

    @property
    def q_range(self) -> Tuple[float, float]:
        """Usable command range: operational range ∩ sample support."""
        lo = max(self.q_jaw_min_m, self._q[0])
        hi = min(self.q_jaw_max_m, self._q[-1])
        return lo, hi

    # -- mappings ----------------------------------------------------------

    def clamp_q(self, q_jaw_m: float) -> float:
        """Clamp a requested q_jaw_m to the calibrated model range."""
        if not math.isfinite(q_jaw_m):
            raise RangeError(f"non-finite q_jaw_m command: {q_jaw_m!r}")
        lo, hi = self.q_range
        if q_jaw_m < lo or q_jaw_m > hi:
            if not self.clamp:
                raise RangeError(
                    f"q_jaw_m {q_jaw_m} outside calibrated range [{lo}, {hi}] "
                    "and clamping is disabled"
                )
            return min(max(q_jaw_m, lo), hi)
        return q_jaw_m

    def q_to_motor(self, q_jaw_m: float) -> float:
        """Requested q_jaw_m -> motor_rad goal (clamps first)."""
        q = self.clamp_q(q_jaw_m)
        q = min(max(q, self._q[0]), self._q[-1])
        i = self._segment_index(self._q, q)
        q0, q1 = self._q[i], self._q[i + 1]
        m0, m1 = self._m[i], self._m[i + 1]
        return m0 + (q - q0) * (m1 - m0) / (q1 - q0)

    def motor_to_q(self, motor_rad: float) -> float:
        """Measured motor_rad -> q_jaw_m feedback (inverse map, clamped)."""
        if not math.isfinite(motor_rad):
            raise RangeError(f"non-finite motor_rad: {motor_rad!r}")
        m_lo = min(self._m[0], self._m[-1])
        m_hi = max(self._m[0], self._m[-1])
        m = min(max(motor_rad, m_lo), m_hi)
        for i in range(len(self._m) - 1):
            a, b = self._m[i], self._m[i + 1]
            if min(a, b) <= m <= max(a, b):
                if a == b:  # unreachable: strict monotonicity enforced
                    continue
                q = self._q[i] + (m - a) * (self._q[i + 1] - self._q[i]) / (b - a)
                return min(max(q, self.q_range[0]), self.q_range[1])
        raise RangeError(f"motor_rad {motor_rad} not covered by table")

    def slope_q_per_motor(self, motor_rad: float) -> float:
        """d(q_jaw_m)/d(motor_rad) on the segment containing motor_rad."""
        m_lo = min(self._m[0], self._m[-1])
        m_hi = max(self._m[0], self._m[-1])
        m = min(max(motor_rad, m_lo), m_hi)
        for i in range(len(self._m) - 1):
            a, b = self._m[i], self._m[i + 1]
            if min(a, b) <= m <= max(a, b):
                return (self._q[i + 1] - self._q[i]) / (b - a)
        raise RangeError(f"motor_rad {motor_rad} not covered by table")

    @staticmethod
    def _segment_index(grid: List[float], x: float) -> int:
        for i in range(len(grid) - 1):
            if grid[i] <= x <= grid[i + 1]:
                return i
        raise RangeError(f"value {x} not covered by grid")


# -- conversions -----------------------------------------------------------


@dataclass(frozen=True)
class RealGripperCommand:
    """Downstream real command: /rebotarm/gripper/command position (rad)."""

    motor_rad: float
    q_jaw_m_clamped: float


def plan_real_command(q_jaw_m: float, table: CalibrationTable) -> RealGripperCommand:
    """CONVERTING step, real profile: clamp then piecewise interpolation."""
    q = table.clamp_q(q_jaw_m)
    return RealGripperCommand(motor_rad=table.q_to_motor(q), q_jaw_m_clamped=q)


def sim_jaw_positions(
    q_jaw_m: float,
    *,
    q_min_m: float = Q_JAW_URDF_MIN_M,
    q_max_m: float = Q_JAW_DEMO_MAX_M,
    clamp: bool = True,
) -> Tuple[float, float]:
    """CONVERTING step, sim profile: trajectory positions [q, q].

    Both jaw joints receive the same value (mimic multiplier +1.0). The
    result is additionally bounded by the URDF joint range [0, 0.0715] m.
    """
    if not math.isfinite(q_jaw_m):
        raise RangeError(f"non-finite q_jaw_m command: {q_jaw_m!r}")
    lo = max(q_min_m, Q_JAW_URDF_MIN_M)
    hi = min(q_max_m, Q_JAW_URDF_MAX_M)
    if q_jaw_m < lo or q_jaw_m > hi:
        if not clamp:
            raise RangeError(
                f"q_jaw_m {q_jaw_m} outside sim range [{lo}, {hi}] "
                "and clamping is disabled"
            )
        q_jaw_m = min(max(q_jaw_m, lo), hi)
    return (q_jaw_m, q_jaw_m)


def reached_goal(
    q_measured_m: float,
    q_goal_m: float,
    tolerance_m: float = DEFAULT_GOAL_TOLERANCE_Q_M,
) -> bool:
    """GripperCommand ``reached_goal``: q estimate within tolerance."""
    return abs(q_measured_m - q_goal_m) <= tolerance_m


# -- state machine ---------------------------------------------------------


class GripperGoalState(enum.Enum):
    IDLE = "IDLE"
    CONVERTING = "CONVERTING"
    FORWARDED = "FORWARDED"
    TRACKING = "TRACKING"
    SUCCEEDED = "SUCCEEDED"
    ABORTED = "ABORTED"
    CANCELED = "CANCELED"


class InvalidTransition(RuntimeError):
    """Raised on a state transition the design doc does not allow."""


_S = GripperGoalState
_ALLOWED: Dict[GripperGoalState, frozenset] = {
    _S.IDLE: frozenset({_S.CONVERTING}),
    _S.CONVERTING: frozenset({_S.FORWARDED, _S.ABORTED, _S.CANCELED}),
    _S.FORWARDED: frozenset({_S.TRACKING, _S.ABORTED, _S.CANCELED}),
    _S.TRACKING: frozenset({_S.SUCCEEDED, _S.ABORTED, _S.CANCELED}),
    _S.SUCCEEDED: frozenset({_S.IDLE}),
    _S.ABORTED: frozenset({_S.IDLE}),
    _S.CANCELED: frozenset({_S.IDLE}),
}

TERMINAL_STATES = frozenset({_S.SUCCEEDED, _S.ABORTED, _S.CANCELED})


class GripperStateMachine:
    """Per-goal state machine (design doc sec. 9.3)."""

    def __init__(self) -> None:
        self._state = GripperGoalState.IDLE

    @property
    def state(self) -> GripperGoalState:
        return self._state

    @property
    def is_terminal(self) -> bool:
        return self._state in TERMINAL_STATES

    @property
    def is_active(self) -> bool:
        return self._state not in TERMINAL_STATES and self._state is not _S.IDLE

    def to(self, new_state: GripperGoalState) -> GripperGoalState:
        if new_state not in _ALLOWED[self._state]:
            raise InvalidTransition(
                f"illegal gripper state transition "
                f"{self._state.value} -> {new_state.value}"
            )
        self._state = new_state
        return self._state

    def reset(self) -> None:
        """Return to IDLE; only legal from a terminal state (or IDLE)."""
        if self._state is _S.IDLE:
            return
        self.to(GripperGoalState.IDLE)
