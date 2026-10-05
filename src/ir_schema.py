"""Typed intermediate representation (IR) for SimuHome QT4 scheduling requests.

An IR describes one `schedule_workflow`-shaped user request as a set of
target device assertions, each gated by one or more *time readings* (ways
the natural-language query expresses when the action should happen).
A feasible request has exactly one consistent time reading per goal.
An infeasible request states two readings that resolve to different times
for the same action (this is the "Contradiction Blindness" failure mode
SimuHome documents: the base agent never compares the readings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Relation = Literal["after", "before", "at"]


@dataclass
class Anchor:
    """What a time offset is measured from."""

    kind: Literal["now", "device_event"] = "now"
    room_id: Optional[str] = None
    device_id: Optional[str] = None
    device_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "room_id": self.room_id,
            "device_id": self.device_id,
            "device_type": self.device_type,
        }


@dataclass
class TimeReading:
    """One way the query expresses *when* an action should occur.

    Exactly one of (offset_minutes with anchor) or (absolute_time) is the
    "primary" reading normally used to compute `at_minutes`. A second
    TimeReading on the same goal, disagreeing with the first once resolved
    to absolute minutes-since-base_time, is what makes a request infeasible.
    """

    relation: Relation
    anchor: Anchor
    offset_minutes: Optional[float] = None  # relative reading
    absolute_time: Optional[str] = None  # "HH:MM:SS", clock-time reading
    resolved_at_minutes: Optional[float] = None  # ground truth, minutes since base_time
    tolerance_minutes: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "anchor": self.anchor.to_dict(),
            "offset_minutes": self.offset_minutes,
            "absolute_time": self.absolute_time,
            "resolved_at_minutes": self.resolved_at_minutes,
            "tolerance_minutes": self.tolerance_minutes,
        }


@dataclass
class Assertion:
    attribute: str
    value: Any
    value_type: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "value": self.value,
            "value_type": self.value_type,
            "description": self.description,
        }


@dataclass
class Target:
    room_id: str
    device_id: str
    device_type: Optional[str]
    asserts: list[Assertion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "device_id": self.device_id,
            "device_type": self.device_type,
            "asserts": [a.to_dict() for a in self.asserts],
        }


@dataclass
class Goal:
    """One scheduled effect: a set of device assertions gated by time reading(s)."""

    goal_id: int
    readings: list[TimeReading]
    targets: list[Target]
    # Set when the conflict is a duration/resource feasibility issue rather
    # than a stated-time mismatch (e.g. "finish washer exactly when
    # dishwasher finishes" but the washer cycle is longer). The Z3 encoder
    # cannot check these without a device duration model; flagged so callers
    # know to skip or handle them separately (see handoff Sec 6, open items).
    requires_duration_model: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "readings": [r.to_dict() for r in self.readings],
            "targets": [t.to_dict() for t in self.targets],
            "requires_duration_model": self.requires_duration_model,
        }


@dataclass
class IntentIR:
    query: str
    query_type: str
    case: Literal["feasible", "infeasible", "unknown"]
    base_time: str  # "YYYY-MM-DD HH:MM:SS"
    goals: list[Goal] = field(default_factory=list)
    conflict_type: Optional[str] = None  # ground-truth label, training-time only

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "case": self.case,
            "base_time": self.base_time,
            "goals": [g.to_dict() for g in self.goals],
            "conflict_type": self.conflict_type,
        }


# ---------------------------------------------------------------------------
# Compact wire format (what the SLM is trained to emit)
# ---------------------------------------------------------------------------
# The full IntentIR.to_dict() is ~520 tokens: it echoes the query, repeats null
# fields, and includes labels the model must NOT see or produce (`case`,
# `conflict_type`) and simulator facts it cannot know (`base_time`, tolerances).
# The compact form keeps only what the query text determines: per goal, the time
# readings and the targets. Z3 decides feasibility; the model never says it.

DEFAULT_TOLERANCE_MINUTES = 0.5


def compact_ir_dict(ir_dict: dict, *, keep_resolved: bool = False) -> dict:
    """Full IR dict -> compact dict. Clock times are cut to HH:MM (the query
    states minutes, never the simulator's seconds)."""
    goals = []
    for g in ir_dict["goals"]:
        readings = []
        for r in g["readings"]:
            anchor = {"kind": r["anchor"]["kind"]}
            if r["anchor"].get("device_id"):
                anchor["device_id"] = r["anchor"]["device_id"]
            if r["anchor"].get("room_id"):
                anchor["room_id"] = r["anchor"]["room_id"]
            out = {"relation": r["relation"], "anchor": anchor}
            if r.get("offset_minutes") is not None:
                out["offset_minutes"] = r["offset_minutes"]
            if r.get("absolute_time") is not None:
                out["absolute_time"] = ":".join(r["absolute_time"].split(":")[:2])
            if keep_resolved and r.get("resolved_at_minutes") is not None:
                out["resolved_at_minutes"] = r["resolved_at_minutes"]
            readings.append(out)
        targets = [
            {
                "device_id": t["device_id"],
                "asserts": [{"attribute": a["attribute"], "value": a["value"]} for a in t.get("asserts", [])],
            }
            for t in g["targets"]
        ]
        goals.append({"readings": readings, "targets": targets})
    return {"goals": goals}


def ir_from_compact(compact: dict, query: str, base_time: str) -> "IntentIR":
    """Compact dict (+ the live query and simulator base_time) -> IntentIR."""
    goals = []
    for i, g in enumerate(compact["goals"]):
        readings = []
        for r in g["readings"]:
            a = r.get("anchor") or {}
            readings.append(
                TimeReading(
                    relation=r.get("relation", "after"),
                    anchor=Anchor(
                        kind=a.get("kind", "now"),
                        room_id=a.get("room_id"),
                        device_id=a.get("device_id"),
                    ),
                    offset_minutes=r.get("offset_minutes"),
                    absolute_time=r.get("absolute_time"),
                    resolved_at_minutes=r.get("resolved_at_minutes"),
                    tolerance_minutes=r.get("tolerance_minutes", DEFAULT_TOLERANCE_MINUTES),
                )
            )
        targets = [
            Target(
                room_id=t.get("room_id", ""),
                device_id=t["device_id"],
                device_type=None,
                asserts=[Assertion(attribute=a["attribute"], value=a["value"]) for a in t.get("asserts", [])],
            )
            for t in g.get("targets", [])
        ]
        goals.append(Goal(goal_id=i, readings=readings, targets=targets))
    return IntentIR(query=query, query_type="qt4", case="unknown", base_time=base_time, goals=goals)
