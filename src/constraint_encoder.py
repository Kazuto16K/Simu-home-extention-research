"""Component 2: IR -> Z3 constraints over real-valued time, SAT/UNSAT check.

Time is encoded as a single Real variable per goal: minutes elapsed since
the episode's `base_time` ("now" = 0). Each TimeReading on a goal produces
one equality/inequality constraint on that variable. A goal is feasible iff
all its readings can be satisfied simultaneously (by the *same* time point,
within tolerance) — this is exactly the check SimuHome's `schedule_workflow`
never performs, and it is the "Contradiction Blindness" fix.

CPU-only, no GPU, no SLM required to run or test this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import z3

from src.ir_schema import Goal, IntentIR, TimeReading


def _parse_hms_to_minutes_of_day(hms: str) -> float:
    parts = [int(x) for x in hms.split(":")]
    h, m = parts[0], parts[1]
    s = parts[2] if len(parts) > 2 else 0
    return h * 60.0 + m + s / 60.0


def _is_minute_granular(hms: str) -> bool:
    return len(hms.split(":")) < 3


def _base_time_minutes_of_day(base_time: str) -> float:
    # base_time: "YYYY-MM-DD HH:MM:SS"
    hms = base_time.split(" ", 1)[1]
    return _parse_hms_to_minutes_of_day(hms)


def reading_to_absolute_minutes(reading: TimeReading, base_time: str) -> Optional[float]:
    """Resolve a single reading to minutes-since-base_time, when it is
    self-contained (i.e. an absolute clock-time reading anchored to "now").
    Device-event anchored readings can't be resolved without simulator
    state, so they stay symbolic in the Z3 encoding instead.
    """
    if reading.absolute_time is not None and reading.anchor.kind == "now":
        base_minutes_of_day = _base_time_minutes_of_day(base_time)
        target_minutes_of_day = _parse_hms_to_minutes_of_day(reading.absolute_time)
        delta = target_minutes_of_day - base_minutes_of_day
        # Clock times are same-day in SimuHome QT4 episodes; no midnight wrap handling needed.
        return delta
    return None


@dataclass
class GoalCheckResult:
    goal_id: int
    sat: bool
    reason: Optional[str] = None  # populated on UNSAT
    unsat_core: Optional[list[str]] = None
    resolved_minutes: Optional[float] = None  # populated on SAT


@dataclass
class FeasibilityResult:
    feasible: bool
    goal_results: list[GoalCheckResult]

    def explanation(self) -> str:
        bad = [g for g in self.goal_results if not g.sat]
        if not bad:
            return "All scheduling constraints are consistent."
        lines = [f"Goal {g.goal_id}: {g.reason}" for g in bad]
        return "Infeasible schedule — " + "; ".join(lines)


def _encode_goal(goal: Goal, base_time: str) -> GoalCheckResult:
    if goal.requires_duration_model:
        return GoalCheckResult(
            goal_id=goal.goal_id,
            sat=True,
            reason="skipped: requires a device duration model (not checked)",
        )

    solver = z3.Solver()
    t = z3.Real(f"t_goal_{goal.goal_id}")

    named_constraints: list[tuple[str, z3.BoolRef]] = []

    for i, reading in enumerate(goal.readings):
        extra = 1.0 if (reading.absolute_time and _is_minute_granular(reading.absolute_time)) else 0.0
        # An "HH:MM" clock time is only known to the minute, so widen by one minute.
        tol = z3.RealVal(reading.tolerance_minutes + extra)
        label = f"reading_{i}"

        if reading.anchor.kind == "device_event":
            # Anchor time is not known symbolically here (no simulator
            # state at IR-build time); readings against the SAME anchor
            # device are still comparable to each other via a shared
            # per-anchor symbolic variable.
            anchor_var = z3.Real(f"anchor_{goal.goal_id}_{reading.anchor.device_id}")
            if reading.resolved_at_minutes is not None:
                # Ground-truth / runtime-known anchor value (e.g. fetched by
                # the agent via get_attribute CountdownTime before invoking
                # the solver). Without this the anchor stays symbolic and a
                # device-event reading alone can never conflict with itself.
                anchor_label = f"{label}_anchor_known"
                solver.assert_and_track(
                    anchor_var == z3.RealVal(reading.resolved_at_minutes), z3.Bool(anchor_label)
                )
            if reading.offset_minutes is None:
                continue
            offset = z3.RealVal(reading.offset_minutes)
            if reading.relation == "after":
                target = anchor_var + offset
            elif reading.relation == "before":
                target = anchor_var - offset
            else:
                target = anchor_var
            named_constraints.append(
                (label, z3.And(t >= target - tol, t <= target + tol))
            )
        else:
            # "now"-anchored: either a relative offset from 0, or a stated
            # absolute clock time resolved to minutes-since-base_time.
            resolved = reading_to_absolute_minutes(reading, base_time)
            if resolved is not None:
                target = z3.RealVal(resolved)
            elif reading.offset_minutes is not None:
                offset = z3.RealVal(reading.offset_minutes)
                target = offset if reading.relation != "before" else -offset
            else:
                continue
            named_constraints.append(
                (label, z3.And(t >= target - tol, t <= target + tol))
            )

    if not named_constraints:
        return GoalCheckResult(goal_id=goal.goal_id, sat=True, reason=None)

    tracker_bools = []
    for label, constraint in named_constraints:
        b = z3.Bool(label)
        solver.assert_and_track(constraint, b)
        tracker_bools.append(b)

    result = solver.check()
    if result == z3.sat:
        model = solver.model()
        resolved_t = model.eval(t, model_completion=True)
        return GoalCheckResult(
            goal_id=goal.goal_id,
            sat=True,
            resolved_minutes=float(resolved_t.as_fraction()),
        )

    core = [str(c) for c in solver.unsat_core()]
    reading_summaries = {
        f"reading_{i}": _describe_reading(r) for i, r in enumerate(goal.readings)
    }
    # "*_anchor_known" trackers exist only to pin a symbolic anchor to a
    # runtime-known value; they aren't a distinct "reading" worth naming in
    # the human-facing explanation.
    described = [
        reading_summaries[c] for c in core if c in reading_summaries
    ]
    reason = " vs. ".join(described) if described else "unsatisfiable"
    return GoalCheckResult(goal_id=goal.goal_id, sat=False, reason=reason, unsat_core=core)


def _describe_reading(reading: TimeReading) -> str:
    if reading.absolute_time is not None:
        return f"stated clock time {reading.absolute_time}"
    if reading.offset_minutes is not None:
        anchor_desc = (
            "now"
            if reading.anchor.kind == "now"
            else f"{reading.anchor.device_id} finishing"
        )
        return f"{reading.offset_minutes} min {reading.relation} {anchor_desc}"
    return "unspecified time reading"


def check_feasibility(ir: IntentIR) -> FeasibilityResult:
    goal_results = [_encode_goal(goal, ir.base_time) for goal in ir.goals]
    feasible = all(g.sat for g in goal_results)
    return FeasibilityResult(feasible=feasible, goal_results=goal_results)
