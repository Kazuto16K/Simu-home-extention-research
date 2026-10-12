"""Component 4: dry run of a proposed workflow on a copy of the home.

SimuHome's `schedule_workflow` only checks the tool name and that the start time
is in the future. Whether the steps can actually execute (right device, command,
cluster, argument names, writable attribute) is discovered only when the workflow
fires, long after the agent has called `finish`. This module finds out at
scheduling time:

    live snapshot (GET /home/state)
      -> SimulationConfig with base_time = the snapshot's current time
      -> a fresh, thread-less Home (SimuHome's own initializer)
      -> re-schedule the workflows already committed in this episode
      -> schedule the candidate, fast-forward to its due tick (+1)
      -> read the workflow status and the target attributes

Everything goes through SimuHome's own code (`_initialize_home_locally`,
`_convert_snapshot_to_simulation_config`, `Home._fast_forward_to`), so nothing in
SimuHome is edited (its licence is CC BY-NC-ND 4.0). This is a lightweight form of
the "simulation-based pre-validation" the SimuHome paper lists as future work.

Imports SimuHome lazily, so the rest of the package works without it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TargetMismatch:
    device_id: str
    attribute: str
    expected: Any
    actual: Any


@dataclass
class DryRunResult:
    ok: bool  # workflow completed in the dry run
    status: str  # completed | failed | pending | not_scheduled | error
    error: Optional[str] = None
    target_mismatches: list[TargetMismatch] = field(default_factory=list)
    seconds: float = 0.0


def _split_attribute(path: str) -> tuple[int, str, str]:
    endpoint, cluster, attribute = path.split(".", 2)
    return int(endpoint), cluster, attribute


def values_match(expected: Any, actual: Any) -> bool:
    """Plain equality, the same rule SimuHome's QT4 scorer uses (`_verify_asserts`).
    Values are raw attribute values (e.g. LevelControl.CurrentLevel 30), not percentages."""
    return actual == expected


def build_shadow_home(snapshot: dict):
    """A thread-less SimuHome Home whose tick 0 is the snapshot's current time."""
    from src.pipelines.episode_generation.core.home_augmenter import (  # SimuHome's own helpers
        _convert_snapshot_to_simulation_config,
        _initialize_home_locally,
    )

    config = _convert_snapshot_to_simulation_config(snapshot, base_time=snapshot["current_time"])
    home, _ = _initialize_home_locally(config)
    return home


def dry_run(
    snapshot: dict,
    candidate: dict,
    committed: Optional[list[dict]] = None,
    expected_targets: Optional[list[tuple[str, str, Any]]] = None,
) -> DryRunResult:
    """Run `candidate` = {"start_time", "steps"} on a copy of the home.

    `committed`: workflows the agent already scheduled in this episode (same
    shape); replayed first so the copy sees their effects. `expected_targets`:
    (device_id, "1.Cluster.Attribute", value) triples to read after the candidate
    ran; mismatches are reported, never raised.
    """
    t0 = time.time()
    try:
        home = build_shadow_home(snapshot)
    except Exception as e:  # snapshot the initializer cannot rebuild: say so, don't guess
        return DryRunResult(ok=False, status="error", error=f"shadow build failed: {e!r}"[:300], seconds=time.time() - t0)

    for wf in committed or []:
        home.schedule_workflow(wf["start_time"], wf["steps"])  # ones already in the past are skipped by Home

    res = home.schedule_workflow(candidate["start_time"], candidate["steps"])
    if not res.success:
        return DryRunResult(ok=False, status="not_scheduled", error=res.error_message, seconds=time.time() - t0)
    wf_id = res.data["workflow_id"]
    wf = home.workflows_by_id[wf_id]

    ff = home._fast_forward_to(int(wf["due_tick"]) + 1)  # noqa: SLF001 (SimuHome's own fast-forward)
    if not ff.success:
        return DryRunResult(ok=False, status="error", error=f"fast-forward failed: {ff.error_message}", seconds=time.time() - t0)

    mismatches = []
    for device_id, attr_path, expected in expected_targets or []:
        try:
            endpoint, cluster, attribute = _split_attribute(attr_path)
            got = home._get_attribute(device_id, endpoint, cluster, attribute)  # noqa: SLF001
            actual = got.data if got.success else None
        except Exception:
            actual = None
        if not values_match(expected, actual):
            mismatches.append(TargetMismatch(device_id, attr_path, expected, actual))

    status = wf.get("status", "pending")
    return DryRunResult(
        ok=status == "completed",
        status=status,
        error=wf.get("error"),
        target_mismatches=mismatches,
        seconds=time.time() - t0,
    )
