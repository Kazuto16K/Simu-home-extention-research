# Approach 2: design, evidence and limits

Read `../neuro_symbolic_scheduler/ARCHITECTURE.md` first for the benchmark, the IR
and approach 1. This file covers only what approach 2 changes, why, and what is
still unproven.

## 1. What the session-1 traces showed

From `../neuro_symbolic_scheduler/results/session1_qt4-1` (agent Gemma-31B, QT4-1,
10 held-out episodes per cell):

| Run | agent calls/ep | seconds/ep | schedule calls/ep | accuracy |
|---|---|---|---|---|
| baseline, infeasible | 3.4 | 159 | 0.7 | 3/10 |
| approach 1, infeasible | 4.4 | 208 | 2.0 | 10/10 |
| baseline, feasible | 7.5 | 218 | 2.7 | 2/10 |
| approach 1, feasible | 8.7 | 378 | 3.9 | 2/10 |

Causes, read from the traces and the code:
1. **The gate checked the request, not the call.** `verified_schedule_workflow` ignored
   `start_time` and `steps`. On infeasible requests every retry got the same 409 (in
   `qt4-1_infeasible_seed_15` the agent tried 12:23, then 12:12:33, then gave up: 6 calls
   instead of 2). On feasible requests the gate could not see the agent's own mistakes.
2. **SimuHome's `schedule_workflow` accepts any steps.** `Home.schedule_workflow` checks
   only the tool name and that the start time is in the future; invalid commands fail
   silently when the workflow fires. That is the feasible failure mode the traces show
   (invented `SetFanSpeed`, wrong argument names).
3. **Time base.** SimuHome's clock runs in real time during an episode, but the scorer
   measures goals from tick 0. Approach 1 used the time of the check as "now"; approach 2
   uses the episode's `base_time` (the time the user spoke, which SimuHome passes to
   `ReActAgent.run` as `current_time`), and converts live CountdownTime readings onto it.

## 2. The pipeline

```
request -> [A] pre-flight, once: parse (cached) -> device finish times from the simulator
                                 -> ONE joint Z3 system -> UNSAT or SAT + windows W
   UNSAT + hint:   note appended to the agent's query; agent should call finish
   UNSAT + answer: the gate writes the reply; the agent is never called
   SAT:            the agent runs as usual
agent calls schedule_workflow(start_time, steps)
   [B]  request UNSAT  -> 409 REQUEST_CONTRADICTION ("no start_time can satisfy it; call finish")
   [C1] start_time outside W of every goal touching these devices -> 409 TIME_MISMATCH with the window
   [C2] dry run on a copy of the home at start_time:
          a step fails       -> 409 WORKFLOW_STEP_WOULD_FAIL with SimuHome's error text
          targets not reached -> note in the success response (warn) or 409 (reject)
   otherwise -> SimuHome's real tool; the workflow is remembered for later dry runs
```

Every rejection kind is capped per episode (`max_plan_rejections` = 2,
`max_unsat_rejections` = 3); after the cap the gate passes the call through. Parse
failures and internal errors also fail open. So the gate can add a few agent steps but
cannot trap the agent or crash an episode.

## 3. IR v3 and the joint encoder

Two labelling rules, both read off `eval.goals[].when` (`src/extract_ir_from_episode.py`):

| Pattern in the benchmark | v2 label | v3 label | Episodes |
|---|---|---|---|
| no device anchor, `reference_minutes > 0` ("8 min after the previous action") | "8 min after now" | `anchor: {kind: goal, goal_ref: j}` | 42/50 QT4-1 feasible |
| `relation: before`, `tolerance_ticks: 0` (QT4-3 "start the dryer now, pause it when the washer finishes") | point 1 min before the device ends | `relation: by` (deadline) anchored to the other goal | all 100 QT4-3 |

Finding while building this: in QT4-3 the deadline goal's `reference_minutes` is the
**other goal's** time (e.g. 89 + 14 = 103), not the labelled device's finish (89). A
shared device-anchor variable made 11 feasible episodes UNSAT until the deadline was
anchored to the other goal, which is also the right meaning ("start before you pause").

`src/constraint_encoder.py` puts all goals in one solver (approach 1 solved each goal
alone): goal anchors, a device anchor shared by several goals, and deadlines become
constraints between goal variables. `goal_windows` gets each goal's [lo, hi] by Z3
`Optimize` (minimise, then maximise `t_goal`). On QT4-1 chained goals the window of
goal 1 comes out as [20, 22] for "13 min from now, then 8 min after that", so C1 is not
a one-line comparison.

The joint encoding keeps every approach-1 verdict (286/286) and puts every benchmark
goal time inside its window (301/301; checked against `at_tick`, because SimuHome
rounds `at_minutes`).

## 4. The dry run (`src/shadow_sim.py`)

`GET /home/state` -> `_convert_snapshot_to_simulation_config` (base_time = the
snapshot's current time) -> `_initialize_home_locally` -> replay this episode's committed
workflows -> schedule the candidate -> `_fast_forward_to(due_tick + 1)` -> read the
workflow status and the target attributes. All SimuHome functions; nothing edited.

Measured locally:
- **Cost**: about 0.7 s for a 13-minute fast-forward, about 5 s for 50 minutes.
- **Fidelity**: after 50 simulated minutes the copy and the real home agree on every
  device on/off state, mode, operational state and countdown (within 1 s). They differ on
  ambient aggregates (room temperature, humidity, illuminance, PM10), which the copy
  rebuilds from scratch. QT4 never asserts on those, but this is a real limit: a request
  whose target depended on ambient values could be misjudged.
- **Detection**: wrong command (`SetLevel`), wrong argument name (`level` vs `Level`),
  unknown device, writing a read-only attribute: each refused with SimuHome's message.
- Target values are compared with plain `==`, the same rule as SimuHome's QT4 scorer
  (raw attribute values, e.g. `CurrentLevel` 30).

## 5. Experiment arms (`src/gate.py: ARMS`)

| arm | pre-flight | B | C1 | C2 dry run | C2 targets |
|---|---|---|---|---|---|
| `baseline` | - | - | - | - | - |
| `selfcheck` | prompt-only instruction | - | - | - | - |
| `v1_gate` | - (request checked inside the call, plain 409) | - | - | - | - |
| `pre_hint` | hint | yes | - | - | - |
| `pre_answer` | answer | yes | - | - | - |
| `full` | hint | yes | v3 only | reject | warn |
| `full_answer` | answer | yes | v3 only | reject | warn |

`v1_gate` reimplements approach 1 inside the same harness, so the comparison is fair.
The one difference: it uses the episode's `base_time`, not the check time (see §1.3).

Report per cell: accuracy, false rejections (feasible episodes the gate made fail),
agent calls/episode, seconds/episode, schedule calls/episode, and gate telemetry
(`gate_telemetry.json`: one row per episode with every rejection kind and the dry-run
time).

## 6. What is NOT shown yet (be honest in the paper)

- **No real-agent E1 numbers for approach 2.** Every number above comes from unit and
  integration tests with the gold IR and a scripted agent.
- **The v3 parser is not trained yet.** C1 is only as safe as the parser's chain
  labels; notebook 01 reports "window-safe goals" (benchmark time inside the window from
  the predicted IR). If that is below ~100%, C1 will cause false rejections; it is
  capped and fails open, but the cost should be reported.
- **`answer` mode depends on parser verdict accuracy.** v2 reached 42/46 = 91% on
  held-out verdicts; a false UNSAT in `answer` mode fails a feasible episode with no
  recovery. `hint` mode lets the agent overrule a wrong note. Report both.
- **Target check uses the parser's targets**, and v2 exact match was only 20/54; hence
  the default is warn, not reject. Notebook 01 measures target accuracy.
- **Benchmark-derived labels.** The parser learns the shape of SimuHome's `eval.goals`.
  This is the same setup as approach 1 and is held-out by seed, but reviewers may call it
  learning the grader's format. A rephrasing test (E-robust) is still to do.
- **Duration conflicts** (14 `completion_vs_pause` episodes) are still out of scope.

## 7. Positioning for the paper

- SimuHome App. L proposes "simulation-based pre-validation" as future work; C2 is a
  working, lightweight version (a copy of the home, SimuHome's own fast-forward).
- Small-large collaboration, measured: the 1.7B parser + solver + dry run decide *before*
  the large model acts; `answer` mode removes large-model calls entirely on contradictory
  requests. The cost columns (agent calls, seconds) are results, not side notes.
- The joint encoding answers approach 1's weak point ("Z3 is one-line arithmetic"): goal
  chains, shared anchors and deadlines need simultaneous constraints and windows, not one
  subtraction. Still report the honest ablation (Z3 vs a hand-written checker) on QT4.
