"""Component 3 (approach 2): the per-episode gate around `schedule_workflow`.

Approach 1 re-parsed the user's request on every `schedule_workflow` call, checked
only the request (never the agent's arguments) and sent back a 409 the agent
could not act on, so it kept retrying with other times. Approach 2 changes this:

  A. Pre-flight: parse once and run Z3 once, before the agent's first step. Device
     anchors are read straight from the simulator (no LLM call). If the request is
     contradictory, either tell the agent at step 0 ("hint") or answer the user
     directly without calling the agent at all ("answer").
  B. Final rejection: when the contradiction is in the request itself, the 409 says
     that no start_time can fix it and that the agent should call finish.
  C. Plan check, for consistent requests:
     C1 the call's start_time must fall inside the window Z3 computes for the goal
        whose devices the call touches;
     C2 the steps are dry-run on a copy of the home at that start_time
        (`shadow_sim`): steps that would fail are rejected with SimuHome's own error
        message; target attributes that would not reach the requested value are
        reported (warn) or rejected.
  Every rejection kind is capped per episode (`max_plan_rejections`); after that the
  gate fails open, so it can slow the agent down but never trap it in a loop.

Nothing here imports SimuHome: tool calls, the parser and the dry run are passed in,
so the logic is unit-testable without a simulator or GPU.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .constraint_encoder import (
    FeasibilityResult,
    GoalWindow,
    check_feasibility,
    clock_to_minutes,
    goal_windows,
    minutes_to_clock,
)
from .grounding import ground_ir, repair_device_ids
from .ir_schema import ir_from_compact, uses_goal_anchors

ToolFn = Callable[[str, dict], dict]


@dataclass
class GateConfig:
    preflight: str = "hint"  # "off" | "hint" | "answer"
    check_at_schedule: bool = True  # enforce the request-level verdict inside schedule_workflow
    terminal_rejection: bool = True  # B: say the contradiction cannot be fixed by another time
    time_check: str = "auto"  # "off" | "on" | "auto" (= on only for v3 IRs, which have goal anchors)
    dry_run: str = "reject"  # "off" | "warn" | "reject": steps that would fail when they run
    target_check: str = "warn"  # "off" | "warn" | "reject": target attributes not reaching the value
    sat_hints: bool = False  # also tell the agent the computed times when the request is consistent
    ground_clock_times: bool = True  # drop clock times the parser wrote but the user never said (grounding.py)
    ground_device_names: bool = True  # map device ids that do not exist in the house to the closest real one
    max_plan_rejections: int = 2  # per episode and per kind (time / dry_run / target); then fail open
    max_unsat_rejections: int = 3  # per episode, for an infeasible request
    slack_minutes: float = 0.25  # extra slack on every window (SimuHome scores within +/-30 s)
    # No dry run for start times further ahead than this (the copy must be fast-forwarded the whole
    # way: ~1 s per 10 simulated minutes). QT4 goals are at most ~116 min after the request.
    dry_run_horizon_minutes: float = 150.0
    ir_version: int = 2  # 3 when the parser was trained on v3 labels


# The arms of the experiment. `None` = SimuHome's original tool, no gate.
ARMS: dict[str, Optional[GateConfig]] = {
    "baseline": None,
    "selfcheck": None,  # + SELF_CHECK_PROMPT appended to the query (see the notebook)
    # approach 1 reproduced inside the same harness: check at schedule time only
    "v1_gate": GateConfig(preflight="off", terminal_rejection=False, time_check="off", dry_run="off", target_check="off",
                          ground_clock_times=False, ground_device_names=False),
    "pre_hint": GateConfig(preflight="hint", time_check="off", dry_run="off", target_check="off"),
    "pre_answer": GateConfig(preflight="answer", time_check="off", dry_run="off", target_check="off"),
    "full": GateConfig(preflight="hint"),
    "full_answer": GateConfig(preflight="answer"),
}


def _response(code: int, message: str, err_type: str, detail: str) -> dict:
    return {"status": {"code": code, "message": message}, "data": None, "error": {"type": err_type, "detail": detail}}


def _ok(resp: Any) -> bool:
    return isinstance(resp, dict) and (resp.get("status") or {}).get("code") == 200


def _tool_data(resp: Any) -> Any:
    return resp.get("data") if isinstance(resp, dict) else None


def explain_contradiction(result: FeasibilityResult) -> str:
    """Plain-English sentence naming the clashing statements (from the unsat core)."""
    bad = [g for g in result.goal_results if not g.sat]
    parts = [g.reason for g in bad if g.reason]
    return "; ".join(parts) if parts else "two parts of the request give the same action different times"


def user_answer(result: FeasibilityResult) -> str:
    """The reply sent to the user when the gate answers without the agent ("answer" pre-flight)."""
    return (
        "I can't schedule this as requested, because the timing contradicts itself: "
        f"{explain_contradiction(result)}. Both cannot be true, so I have not scheduled anything. "
        "Please tell me which time you want."
    )


def devices_in_steps(steps: list) -> list[str]:
    out = []
    for st in steps or []:
        args = st.get("args") if isinstance(st, dict) else None
        if isinstance(args, dict) and isinstance(args.get("device_id"), str) and args["device_id"] not in out:
            out.append(args["device_id"])
    return out


@dataclass
class GateStats:
    preflight: str = "not_run"  # not_run | sat | unsat | parse_fail | error
    preflight_seconds: float = 0.0
    clock_readings_dropped: int = 0  # parser clock times not found in the request (grounding filter)
    device_names_repaired: int = 0  # parser device ids not in the house, mapped to the closest real device
    device_names_unresolved: int = 0  # ... and ones with no close match (left as they are)
    schedule_calls: int = 0
    passed: int = 0
    unsat_rejections: int = 0
    time_rejections: int = 0
    dry_run_rejections: int = 0
    target_rejections: int = 0
    target_warnings: int = 0
    unmatched_calls: int = 0  # touched no device the IR knows: not checked
    fail_open: int = 0  # a cap was reached, or a check could not run
    dry_run_seconds: float = 0.0
    events: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: (v if k != "events" else list(v)) for k, v in self.__dict__.items()}


class EpisodeGate:
    """All gate state for ONE episode (one user request)."""

    def __init__(
        self,
        query: str,
        base_time: str,
        compact_ir: Optional[dict],
        cfg: GateConfig,
        call_tool: ToolFn,
        dry_run_fn: Optional[Callable[..., Any]] = None,
    ):
        self.query = query
        self.base_time = base_time  # when the user spoke = minute 0 of every reading
        self.compact_ir = compact_ir
        self.cfg = cfg
        self.call_tool = call_tool
        self.dry_run_fn = dry_run_fn
        self.stats = GateStats()
        self.result: Optional[FeasibilityResult] = None
        self.windows: dict[int, GoalWindow] = {}
        self.ir = None
        self.committed: list[dict] = []
        self._verdict_done = False

    # ------------------------------------------------------------------ verdict
    def _house_device_ids(self) -> list[str]:
        """Every device id in the house, from the simulator (one call; no answer key involved)."""
        snap = _tool_data(self.call_tool("get_home_state", {})) or {}
        return [d.get("device_id") for room in (snap.get("rooms") or {}).values() for d in (room.get("devices") or [])
                if d.get("device_id")]

    def _now_minutes(self) -> float:
        now = (_tool_data(self.call_tool("get_current_time", {})) or {}).get("now")
        return clock_to_minutes(self.base_time, now) if now else 0.0

    def _resolve_device_anchors(self, compact: dict) -> None:
        """Fill resolved_at_minutes for device anchors from the live simulator:
        finish time = now + CountdownTime. Idle devices stay unresolved (symbolic)."""
        needed = {
            r["anchor"]["device_id"]
            for g in compact["goals"]
            for r in g["readings"]
            if (r.get("anchor") or {}).get("kind") == "device_event" and r["anchor"].get("device_id")
            and r.get("resolved_at_minutes") is None
        }
        if not needed:
            return
        now_min = self._now_minutes()
        finish: dict[str, Optional[float]] = {}
        for dev in needed:
            resp = self.call_tool("get_attribute", {
                "device_id": dev, "endpoint_id": 1, "cluster_id": "OperationalState", "attribute_id": "CountdownTime",
            })
            data = _tool_data(resp)
            value = data.get("value") if isinstance(data, dict) else data
            ok = isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
            finish[dev] = now_min + float(value) / 60.0 if ok else None
        for g in compact["goals"]:
            for r in g["readings"]:
                a = r.get("anchor") or {}
                if a.get("kind") == "device_event" and r.get("resolved_at_minutes") is None:
                    r["resolved_at_minutes"] = finish.get(a.get("device_id"))

    def ensure_verdict(self) -> None:
        if self._verdict_done:
            return
        self._verdict_done = True
        t0 = time.time()
        if self.compact_ir is None:
            self.stats.preflight = "parse_fail"
            return
        try:
            compact = copy.deepcopy(self.compact_ir)
            if self.cfg.ground_clock_times:
                compact, self.stats.clock_readings_dropped = ground_ir(compact, self.query)
            if self.cfg.ground_device_names:
                known = self._house_device_ids()
                if known:
                    compact, self.stats.device_names_repaired, self.stats.device_names_unresolved =                         repair_device_ids(compact, known)
            self._resolve_device_anchors(compact)
            self.ir = ir_from_compact(compact, self.query, self.base_time)
            self.result = check_feasibility(self.ir)
            if self.result.feasible and self._time_check_on():
                self.windows = goal_windows(self.ir)
            self.stats.preflight = "sat" if self.result.feasible else "unsat"
        except Exception as e:  # a bug in the check must never break the episode: fail open
            self.result = None
            self.stats.preflight = "error"
            self.stats.events.append(("verdict_error", repr(e)[:200]))
        finally:
            self.stats.preflight_seconds += time.time() - t0

    def _time_check_on(self) -> bool:
        mode = self.cfg.time_check
        if mode == "auto":
            return self.cfg.ir_version >= 3 or bool(self.compact_ir and uses_goal_anchors(self.compact_ir))
        return mode == "on"

    @property
    def infeasible(self) -> bool:
        return self.result is not None and not self.result.feasible

    # ---------------------------------------------------------------- pre-flight
    def preflight_action(self) -> tuple[str, str]:
        """("none" | "hint" | "answer", text) to apply before the agent's first step."""
        if self.cfg.preflight == "off":
            return "none", ""
        self.ensure_verdict()
        if self.infeasible:
            if self.cfg.preflight == "answer":
                return "answer", user_answer(self.result)
            return "hint", (
                "\n\n[Note from the scheduling verifier, computed before you started: this request contains a "
                f"timing contradiction: {explain_contradiction(self.result)}. No start time can satisfy both "
                "statements. Do not call schedule_workflow; call finish and explain the conflict to the user.]"
            )
        if self.result is not None and self.cfg.sat_hints:
            parts = []
            for g in self.result.goal_results:
                if g.resolved_minutes is None:
                    continue
                devs = ", ".join(t.device_id for t in self.ir.goals[g.goal_id].targets) or "?"
                parts.append(f"{devs} at {minutes_to_clock(self.base_time, g.resolved_minutes)}")
            if parts:
                return "hint", "\n\n[Note from the scheduling verifier: the requested times work out to " + "; ".join(parts) + ".]"
        return "none", ""

    # ------------------------------------------------------------ schedule call
    def on_schedule(self, args: dict, real_tool: Callable[[dict], dict]) -> dict:
        self.stats.schedule_calls += 1
        if not self.cfg.check_at_schedule:
            return self._commit(args, real_tool)
        self.ensure_verdict()
        if self.result is None:  # parse failure or check error: fail open
            self.stats.fail_open += 1
            return self._commit(args, real_tool)

        if self.infeasible:
            if self.stats.unsat_rejections >= self.cfg.max_unsat_rejections:
                self.stats.fail_open += 1
                return self._commit(args, real_tool)
            self.stats.unsat_rejections += 1
            why = explain_contradiction(self.result)
            if self.cfg.terminal_rejection:
                detail = (
                    f"The request itself is contradictory: {why}. No start_time can satisfy it, so changing the "
                    "time will not help. Do not call schedule_workflow again for this request; call finish and "
                    "explain the conflict to the user."
                )
                return _response(409, "REQUEST_CONTRADICTION", "TEMPORAL_CONTRADICTION_IN_REQUEST", detail)
            detail = "I cannot schedule this as stated -- the timing is inconsistent: " + self.result.explanation()
            return _response(409, "INFEASIBLE_SCHEDULE", "TEMPORAL_CONTRADICTION", detail)

        steps = args.get("steps") if isinstance(args, dict) else None
        start_time = args.get("start_time") if isinstance(args, dict) else None
        if not isinstance(steps, list) or not isinstance(start_time, str):
            return self._commit(args, real_tool)  # malformed call: SimuHome's own validation answers it
        try:
            start_min = clock_to_minutes(self.base_time, start_time)
        except Exception:
            return self._commit(args, real_tool)

        devices = devices_in_steps(steps)
        candidates = [g for g in self.ir.goals if any(t.device_id in devices for t in g.targets)]
        if not candidates:
            self.stats.unmatched_calls += 1

        # C1: start time against the goal windows
        matched = candidates
        if candidates and self.windows:
            inside = [g for g in candidates if self.windows.get(g.goal_id) is None
                      or self.windows[g.goal_id].contains(start_min, self.cfg.slack_minutes)]
            if inside:
                matched = inside
            elif self.stats.time_rejections < self.cfg.max_plan_rejections:
                self.stats.time_rejections += 1
                return _response(409, "TIME_MISMATCH", "START_TIME_DOES_NOT_MATCH_REQUEST",
                                 self._time_mismatch_detail(start_time, candidates))
            else:
                self.stats.fail_open += 1

        # C2: dry run on a copy of the home
        notes = []
        if start_min > self.cfg.dry_run_horizon_minutes and self.cfg.dry_run != "off":
            self.stats.events.append(("dry_run_skipped_beyond_horizon", start_time))
        elif self.cfg.dry_run != "off" and self.dry_run_fn is not None:
            expected = self._expected_targets(self._closest_goals(matched, start_min), devices) \
                if self.cfg.target_check != "off" else []
            dr = self._dry_run(start_time, steps, expected)
            if dr is not None:
                if dr.status == "failed":
                    if self.cfg.dry_run == "reject" and self.stats.dry_run_rejections < self.cfg.max_plan_rejections:
                        self.stats.dry_run_rejections += 1
                        return _response(409, "DRY_RUN_FAILED", "WORKFLOW_STEP_WOULD_FAIL", (
                            f"Scheduling refused: when this workflow runs at {start_time}, a step would fail: "
                            f"{dr.error}. Check the device's commands and attributes with get_device_structure, "
                            "fix the step, and schedule again."
                        ))
                    notes.append(f"Dry run: a step will fail at {start_time}: {dr.error}.")
                if dr.target_mismatches:
                    msg = "; ".join(f"{m.device_id} {m.attribute} would be {m.actual!r}, the request asks for {m.expected!r}"
                                    for m in dr.target_mismatches)
                    if self.cfg.target_check == "reject" and self.stats.target_rejections < self.cfg.max_plan_rejections:
                        self.stats.target_rejections += 1
                        return _response(409, "TARGET_MISMATCH", "WORKFLOW_WOULD_NOT_REACH_REQUESTED_STATE",
                                         f"Scheduling refused: after this workflow runs at {start_time}, {msg}. "
                                         "Fix the steps and schedule again.")
                    if self.cfg.target_check != "off":
                        self.stats.target_warnings += 1
                        notes.append(f"Dry run after {start_time}: {msg}.")
        resp = self._commit(args, real_tool)
        if notes and _ok(resp) and isinstance(resp.get("data"), dict):
            resp = copy.deepcopy(resp)
            resp["data"]["verifier_note"] = " ".join(notes)
        return resp

    def _closest_goals(self, goals, start_min: float):
        """Per device, keep only the goal whose computed time is closest to start_min.
        (Two goals on one device, e.g. dimmer to 30% at +13 min and to 20% at +21 min,
        must not both be expected from a single call.)"""
        resolved = {g.goal_id: g.resolved_minutes for g in self.result.goal_results}
        best: dict[str, Any] = {}
        for g in goals:
            dist = abs(resolved[g.goal_id] - start_min) if resolved.get(g.goal_id) is not None else float("inf")
            for t in g.targets:
                if t.device_id not in best or dist < best[t.device_id][0]:
                    best[t.device_id] = (dist, g)
        return best

    @staticmethod
    def _expected_targets(best: dict, devices: list[str]) -> list[tuple[str, str, Any]]:
        out = []
        for dev in devices:
            if dev in best:
                g = best[dev][1]
                out.extend((t.device_id, a.attribute, a.value) for t in g.targets if t.device_id == dev for a in t.asserts)
        return out

    def _dry_run(self, start_time: str, steps: list, expected: list):
        t0 = time.time()
        try:
            snapshot = _tool_data(self.call_tool("get_home_state", {}))
            if not isinstance(snapshot, dict):
                raise RuntimeError("no home snapshot")
            dr = self.dry_run_fn(snapshot, {"start_time": start_time, "steps": steps}, self._pending_committed(), expected)
            if dr.status == "error":
                self.stats.events.append(("dry_run_error", (dr.error or "")[:200]))
                self.stats.fail_open += 1
                return None
            return dr
        except Exception as e:
            self.stats.events.append(("dry_run_error", repr(e)[:200]))
            self.stats.fail_open += 1
            return None
        finally:
            self.stats.dry_run_seconds += time.time() - t0

    def _pending_committed(self) -> list[dict]:
        """Workflows this episode scheduled that are still pending in the real home. Ones the
        agent cancelled (cancel_workflow) or that already ran (their effect is in the snapshot)
        must not be replayed on the copy."""
        out = []
        for wf in self.committed:
            wid = wf.get("workflow_id")
            if wid is None:
                out.append(wf)
                continue
            status = (_tool_data(self.call_tool("get_workflow_status", {"workflow_id": wid})) or {}).get("status")
            if status in (None, "pending"):
                out.append(wf)
        return out

    def _time_mismatch_detail(self, start_time: str, goals) -> str:
        parts = []
        for g in goals:
            w = self.windows.get(g.goal_id)
            if w is None:
                continue
            lo = minutes_to_clock(self.base_time, w.lo) if w.lo is not None else None
            hi = minutes_to_clock(self.base_time, w.hi) if w.hi is not None else None
            devs = ", ".join(t.device_id for t in g.targets)
            if lo and hi:
                rng = f"between {lo} and {hi}"
            elif hi:
                rng = f"no later than {hi}"
            else:
                rng = f"no earlier than {lo}"
            parts.append(f"for {devs} the request works out to {rng}")
        return (
            f"Scheduling refused: start_time {start_time} does not match the request ({'; '.join(parts)}). "
            "Recompute the time from the request and the current device states, then schedule again."
        )

    def _commit(self, args: dict, real_tool: Callable[[dict], dict]) -> dict:
        resp = real_tool(args)
        if _ok(resp):
            self.stats.passed += 1
            if isinstance(args, dict) and isinstance(args.get("steps"), list) and isinstance(args.get("start_time"), str):
                wid = (resp.get("data") or {}).get("workflow_id") if isinstance(resp.get("data"), dict) else None
                self.committed.append({"start_time": args["start_time"], "steps": args["steps"], "workflow_id": wid})
        return resp
