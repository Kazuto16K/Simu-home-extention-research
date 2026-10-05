# What this project is, how it works, and how it is built

> **Status snapshot (Oct 2026).** The whole pipeline is built and has run end to
> end on Kaggle. Preliminary E1 result (QT4-1, 10 held-out episodes per cell,
> agent = `google/gemma-4-31b-it`): infeasible **30% -> 100%**, feasible
> **20% -> 20%** with **0 false rejections**. Parser: 54/54 valid JSON, **91%**
> verdict accuracy (42/46). Small sample, one sub-type: see Section 13 for the
> full results table and Section 14 for an honest assessment of the contribution.
> A figure of the architecture is in `docs/architecture.png`; a presentation-style
> write-up is `docs/PROJECT_REPORT.md`.

## 1. The problem in one paragraph

**SimuHome** (ICLR 2026) is a simulated smart home (rooms, lights, washers,
dishwashers, ACs, ...) plus a benchmark of 600 natural-language requests. An LLM
agent (a ReAct loop) reads a request like

> "20 minutes after dishwasher 1 in the kitchen finishes, turn on the light in the dining room"

and calls tools (`get_room_devices`, `get_attribute`, `schedule_workflow`, ...)
to make it happen.

The benchmark's hardest category is **QT4: scheduling**: do something in the
future, after another device finishes, or at the same time as another device.
GPT-4.1 only succeeds 34-50% of the time. The paper's diagnosis (their Sec 6.2):
the `schedule_workflow` tool **accepts anything**. It returns "scheduled OK"
without checking whether the schedule is possible, so the agent never finds out
its plan is contradictory. On the infeasible "trap" episodes, 75-91% of errors
are **Contradiction Blindness**: the request contains a timing contradiction and
the agent never notices.

Example of a trap (a real episode, `qt4-1_infeasible_seed_1`):

> "At 11:35 AM, that is 13 minutes from now, turn on dimmer light 1 ..."

The simulator clock says it is 11:36:54. "13 minutes from now" is 11:49:54, but
the user also said 11:35 AM, which is already in the past. Both cannot be true.
A good agent should say so. The baseline schedules it blindly.

## 1b. The SimuHome paper vs this project

| Aspect | SimuHome paper (ICLR 2026, arXiv 2509.24282) | This project |
|---|---|---|
| Type of work | Benchmark + simulator; evaluates 18 LLM agents (ReAct) | A **fix** for one failure the benchmark exposes |
| What it observes | QT4 is the hardest category. On infeasible QT4 the dominant error is **contradiction blindness** (Fig. 4d: 85%; Table 11: 40/44, 25/33, 30/34 for QT4-1/2/3). `schedule_workflow` only confirms registration, with no validation (Sec. 5.3) | Takes exactly this failure as the target |
| Fixes tried | HiAgent framework (mixed); **self-correction prompting** (recovery at most 18.5% on QT4-2, 0% on QT4-3, App. M); oracle failure notice (55-67% recovery, an upper bound, impractical); SFT of the *agent* on 204 GPT-5.1 trajectories (Gemma3-4B, Qwen3-32B: infeasible detection up to +26 points, QT4-3-F no gain) | A **pre-execution gate**: small trained parser + checker in front of the tool; the agent is not fine-tuned |
| Proposed future work | App. L: **simulation-based pre-validation**, using the simulator as a runtime world model, "going beyond simple API checkers" | A lightweight alternative at *request* level: checks whether the stated times are mutually consistent before anything is scheduled (does not simulate execution) |
| Training | None of its own method; SFT baseline on agent trajectories | LoRA fine-tune of Qwen3-1.7B on (query, IR) pairs generated automatically from the benchmark's own `eval.goals` / `temporal_conflict` (no manual labels) |
| Representation | Natural language + tool calls | Typed IR of time readings; Z3 constraints; unsat core |
| Feedback to the agent | "Registered OK" | `409 TEMPORAL_CONTRADICTION` plus the clashing statements in plain words |
| Scope and scale | 600 episodes, 12 categories, 18 models | QT4 only; QT4-1 run so far (10 held-out episodes per cell); one agent (Gemma-31B via NVIDIA free API) |
| Scoring | Simulator state (feasible), LLM judge (infeasible) | Same evaluators, so results are comparable in kind (not directly comparable in numbers: different agent and sample) |

**Reference numbers from the paper (Table 1, success %, read from the PDF text layer; confirm against the table before quoting):**

| Agent | QT4-1 F / IF | QT4-2 F / IF | QT4-3 F / IF |
|---|---|---|---|
| GPT-4.1 | 50 / 12 | 46 / 34 | 34 / 32 |
| GPT-5.1 (reasoning) | 60 / 100 | 72 / 92 | 56 / 44 |

Reasoning models take roughly 3-5x longer per episode (GPT-5.1 over 100 s on the hardest tasks, Table 2).

**What this means for positioning**

- Our QT4-1 infeasible result (30% -> 100%, Gemma-31B) is real but **not exceptional**: a frontier reasoning agent already reaches 100% on QT4-1-IF with no verifier. The case for the verifier is a cheap/fast non-reasoning agent, latency (a 1.7B parser + checker vs a 100 s reasoning model), and the harder sub-types, especially **QT4-3-IF, where even GPT-5.1 gets 44%**.
- The paper explicitly leaves pre-validation to future work and argues for a simulator-in-the-loop version. Reviewers will compare against that. Our framing: a cheap request-level consistency check, not an execution-outcome check.
- The paper's weak self-correction result (App. M) is about recovery *after* scheduling. Our `selfcheck` arm tests a *pre-scheduling* prompt check and still has to be run.
- Our weak feasible results (20%) match the paper's picture: feasible QT4 is dominated by device-control errors (40% of QT4-F errors) and temporal errors (25%), which a temporal verifier does not address.

## 2. The idea: put a logic check in front of `schedule_workflow`

We don't retrain the big agent LLM. We add a **verifier** between the agent and
the tool:

```
User query (natural language)
        |
        v
+---------------------+     info tools: get_current_time,
|  Base ReAct agent   |---> get_room_devices, get_attribute (CountdownTime)
|  (SimuHome's own)   |
+----------+----------+
           | agent calls schedule_workflow(...)   <-- WE INTERCEPT HERE
           v
+-----------------------------+
| Component 1: Intent Parser  |   small LLM (Qwen3-1.7B + LoRA)
| NL query -> typed IR (JSON) |
+--------------+--------------+
               v
+-----------------------------+
| Component 2: Z3 encoder     |   IR -> real-valued time constraints
| solver.check()              |
+------+---------------+------+
     SAT             UNSAT
       |               |
       v               v
  call the real   Component 3: verbalizer
  schedule_       unsat core -> plain-English reason
  workflow        returned to the agent as the tool result
```

- **Neural part**: a small fine-tuned model reads the messy English and writes a
  structured form (IR).
- **Symbolic part**: the Z3 SMT solver *proves* whether the timing in that form
  is consistent. It doesn't guess; it either finds a consistent schedule or
  returns the minimal conflicting set of constraints.

**Novelty claim.** Prior work (Agent-C, UIUC/Meta) also puts an SMT solver in
front of agent tool calls, but its logic only talks about *position in the call
sequence* ("A happens before B"). It has no notion of wall-clock minutes or
durations. We extend this to real-valued metric time ("20 minutes after the
dishwasher finishes", "at 11:35 AM"). **Caveat:** on QT4 as it stands this extension
is thin (see Section 14); it only becomes a strong claim with richer constraints.

## 3. Repository layout

```
D:\IoT and Agents\
  SimuHome\                    cloned benchmark (read-only; CC BY-NC-ND 4.0)
  agent-c-reference\           placeholder only: Agent-C code is not released yet
  neuro_symbolic_scheduler\    <- all of our code
    src\                       Python modules, tested locally
      ir_schema.py               IR data types
      constraint_encoder.py      Component 2 (Z3)
      extract_ir_from_episode.py (query, IR) training-label extraction
    tests\test_pipeline.py     validates encoder+extractor on real episodes
    notebooks\                 what runs on Kaggle
      00_full_pipeline.ipynb     one notebook that does everything
      01..06_*.ipynb             same content split into 6 separate notebooks
    scripts_build_notebooks.py generates every .ipynb from src/*.py
    README.md, ARCHITECTURE.md
```

The notebooks are **generated**. Edit `src/*.py` or `scripts_build_notebooks.py`,
then run `python scripts_build_notebooks.py`. Kaggle can't import this repo's
`src/`, so the generator pastes the three `src` files into notebook cells with
`%%writefile` and renames the package to `irlib` (SimuHome also has a top-level
`src` package, and two same-named packages in one kernel shadow each other).

## 4. What a SimuHome episode looks like (the raw material)

Each file in `SimuHome/data/benchmark/qt4-*.json` has:

```jsonc
{
  "meta":  { "query_type": "qt4-2", "case": "feasible|infeasible", "conflict_type": "..." },
  "query": "20 minutes after the dishwasher 1 in the kitchen finishes, power on light 1 in the dining room ...",
  "eval": {
    "required_actions": [ {"tool": "get_room_devices", "params": {"room_id": "dining_room"}} ],
    "goals": [{
      "when":   { "relation": "after", "offset_minutes": 20, "at_minutes": 109,
                  "tolerance_ticks": 300 },
      "anchor": { "room_id": "kitchen", "device_id": "kitchen_dishwasher_1" },
      "targets": [{ "device_id": "dining_room_on_off_light_1",
                    "asserts": [{ "attribute": "1.OnOff.OnOff", "value": true }] }]
    }]
  },
  "temporal_conflict": { ... },           // only on infeasible episodes
  "initial_home_config": { "base_time": "2025-08-23 11:36:54", "tick_interval": 0.1, "rooms": { ... } }
}
```

The key fact that makes this project cheap: **the benchmark already contains the
answer key.** `eval.goals` is the structured meaning of the query, and
`temporal_conflict` (infeasible episodes only) holds the two contradicting times,
already computed by SimuHome's generator. So training labels need no manual
annotation and no LLM calls.

Time is measured in **ticks** (`tick_interval` = 0.1 s per tick). 300 ticks of
tolerance = 30 s.

## 5. Component 1: the IR (intermediate representation)

File: `src/ir_schema.py`. The IR is a handful of dataclasses:

```
IntentIR
 +- query, query_type, case, base_time, conflict_type
 +- goals: [Goal]
      +- goal_id
      +- readings: [TimeReading]      <- "ways the query says WHEN"
      |    +- relation: after | before | at
      |    +- anchor: Anchor(kind="now" | "device_event", device_id, room_id, ...)
      |    +- offset_minutes          (relative: "20 min after ...")
      |    +- absolute_time "HH:MM:SS" (clock time: "at 11:35 AM")
      |    +- resolved_at_minutes     (known anchor time, see below)
      |    +- tolerance_minutes
      +- targets: [Target(device_id, asserts=[Assertion(attribute, value)])]
      +- requires_duration_model      (flag, see Known limitation)
```

**The central design idea:** a goal has a *list* of `TimeReading`s. A feasible
request has one consistent reading. An infeasible one has **two readings that
resolve to different times for the same action**: exactly the
Contradiction-Blindness pattern.

IR of the 11:35 AM example (two readings on one goal):

```json
{
  "goals": [{
    "goal_id": 0,
    "readings": [
      {"relation": "after", "anchor": {"kind": "now"},
       "offset_minutes": 13, "absolute_time": null},
      {"relation": "at",    "anchor": {"kind": "now"},
       "offset_minutes": null, "absolute_time": "11:35:54"}
    ],
    "targets": [{"device_id": "living_room_dimmable_light_1",
                 "asserts": [{"attribute": "1.OnOff.OnOff", "value": true}]}]
  }]
}
```

Anchors: `kind="now"` means measured from the current simulator time (minute 0).
`kind="device_event"` means measured from when another device finishes (e.g. the
dishwasher). Whether the dishwasher finishes at minute 89 or 120 is not in the
query text. It lives in the live simulator state (`CountdownTime`). So the SLM
never outputs a number for it; the runtime fetches it (Section 8).

## 6. Training-data extraction (no manual labeling)

File: `src/extract_ir_from_episode.py`. For each episode it builds the IR
mechanically.

**Feasible episodes**: one `TimeReading` per `eval.goals[]` entry, copied from
`when` and `anchor`:

```python
def _goal_from_eval_goal(goal_id, raw_goal, tick_interval):
    when = raw_goal["when"]
    raw_anchor = raw_goal.get("anchor") or {}
    if raw_anchor.get("device_id"):
        anchor = Anchor(kind="device_event", room_id=raw_anchor.get("room_id"),
                        device_id=raw_anchor.get("device_id"), device_type=raw_anchor.get("device_type"))
    else:
        anchor = Anchor(kind="now")
    reading = TimeReading(relation=when.get("relation", "after"), anchor=anchor,
                          offset_minutes=float(when.get("offset_minutes", 0.0)),
                          tolerance_minutes=_tolerance_minutes(when, tick_interval))
    ...
```

**Infeasible episodes**: take the goal, then *inject the second, contradicting
reading* from `temporal_conflict`: the stated clock time (`conflict_time`)
becomes a second `TimeReading` with `absolute_time`. If the conflict has an
`anchor_end_at` (e.g. the dishwasher end time) the primary reading is
device-anchored; otherwise it is "now"-anchored.

```python
goal.readings.append(TimeReading(relation="at", anchor=Anchor(kind="now"),
                                 absolute_time=conflict_time.split(" ")[-1],
                                 tolerance_minutes=goal.readings[0].tolerance_minutes))
```

**Two outputs per episode** (this matters):

| Field | Contains | Used for |
|---|---|---|
| `ir_for_training` | device-anchor times left empty | the SLM's training target: it must be derivable from the query text alone |
| `debug_ground_truth` | device-anchor times filled from `anchor_end_at` | testing the Z3 encoder without a live simulator |

If the training IR contained the dishwasher's finish minute, the SLM would be
asked to guess a number that appears nowhere in the query.

**Compact IR (what the SLM actually outputs).** `ir_for_training` is the compact
form (`compact_ir_dict` in `src/ir_schema.py`): it drops `query`, `case`,
`conflict_type`, `base_time`, tolerances and nulls, and truncates clocks to HH:MM.
The runtime restores them (`ir_from_compact`: default tolerance 0.5 min, plus 1
min when only HH:MM is given). The first, verbose IR gave 8/30 invalid JSON on
the parser; the compact one (with a larger generation cap) gives 54/54 valid.

**Train/val/test split (as actually used).** Generation of extra episodes is
switched off (`RUN_GENERATION=False`); the data are the 300 native QT4 episodes,
split by `seed % 5` into **221 train / 25 val / 54 held-out test**, with a
leakage check (no query text shared across splits). E1 only scores the held-out
54. File names carry a `_v2` suffix (`qt4_ir_v2_*.jsonl`,
`qwen3_ir_parser_adapter_v2`). A synthetic-data scale-up (2,000-5,000 pairs) is
future work.

## 7. Component 2: the Z3 constraint encoder

File: `src/constraint_encoder.py`. For each goal it creates **one Real variable**
`t` = "minutes since now at which this action happens" and turns each
`TimeReading` into a constraint on `t`, within tolerance:

```python
t = z3.Real(f"t_goal_{goal.goal_id}")
...
# "now"-anchored, relative:   t ~ offset
# "now"-anchored, clock time: t ~ (clock time - base_time) in minutes
# device-anchored:            t ~ anchor_var +/- offset
named_constraints.append((label, z3.And(t >= target - tol, t <= target + tol)))
```

Each constraint is registered with `solver.assert_and_track(constraint, bool)`
so Z3 can report **which constraints conflict**:

```python
result = solver.check()
if result == z3.sat:
    ...                                   # consistent: one time works for all readings
core = [str(c) for c in solver.unsat_core()]   # the minimal conflicting set
```

Worked example, the 11:35 AM trap: reading 0 says `t ~ 13` (13 minutes from
now). Reading 1: clock time 11:35:54 minus base 11:36:54 is `-1` minute, so
`t ~ -1`. With tolerance 5 min: `8 <= t <= 18` and `-6 <= t <= 4`. No overlap:
**UNSAT**. The unsat core is `{reading_0, reading_1}`, which becomes:

```
Infeasible schedule - Goal 0: 13.0 min after now vs. stated clock time 11:35:54
```

For device-anchored readings, `anchor_var` is a symbolic Real. If the runtime
knows the anchor's real time it adds `anchor_var == resolved_at_minutes`;
without it a device-anchored reading alone can never conflict with itself.

**Real-valued time** (Z3 `Real`, not integer position) is the extension over
Agent-C.

### Validation done locally (no GPU needed)

`tests/test_pipeline.py` runs the extractor + encoder on all 300 downloaded QT4
episodes using `debug_ground_truth`:

```
feasible:   150 correct, 0 wrong
infeasible: 136 correct, 0 wrong, 14 skipped (duration-model conflicts)
Accuracy 286/286
```

This proves the encoder logic is right *given a correct IR*. It does **not**
prove the SLM can produce a correct IR. That is what the fine-tuning and
evaluation steps measure.

### Known limitation

14 `completion_vs_pause` episodes are infeasible for a different reason: e.g.
"finish the washer exactly when the dishwasher finishes" but the washer's cycle
is simply longer than the time available. No two stated times contradict; you
need a device duration model. `Goal.requires_duration_model=True` marks these;
the encoder returns SAT-by-default ("skipped") rather than lie. The handoff
scopes this as optional bonus work.

## 8. Component 3: verbalizer + the interception hook

The verbalizer is intentionally trivial and auditable (notebook section 6):

```python
def verbalize_rejection(result) -> str:
    return 'I cannot schedule this as stated -- the timing is inconsistent: ' + result.explanation()
```

The interception replaces SimuHome's tool in its registry with a wrapper:

```python
def verified_schedule_workflow(args):
    query = _last_user_query['text']
    ir_dict = parse_intent(query)                         # Component 1 (the SLM)
    base_time = run_tool('get_current_time', {})['data']['now']
    for goal in ir_dict['goals']:                         # resolve device anchors LIVE
        for reading in goal['readings']:
            if reading['anchor']['kind'] == 'device_event' and reading.get('resolved_at_minutes') is None:
                reading['resolved_at_minutes'] = resolve_anchor_minutes(reading['anchor']['device_id'])
    result = check_feasibility(_dict_to_ir(ir_dict, query, base_time))   # Component 2 (Z3)
    if result.feasible:
        return tool_schedule_workflow(args)               # commit the real schedule
    return {'status': {'code': 409, 'message': 'INFEASIBLE_SCHEDULE'},   # Component 3
            'data': None,
            'error': {'type': 'TEMPORAL_CONTRADICTION', 'detail': verbalize_rejection(result)}}

TOOL_REGISTRY['schedule_workflow'] = verified_schedule_workflow
```

`resolve_anchor_minutes` calls `get_attribute(... 'OperationalState', 'CountdownTime')`
on the anchor device and converts seconds to minutes.

Where does the query text come from? The tool only receives `args`, not the
user's sentence. So we wrap `ReActAgent.run` to stash it first:

```python
_original_run = ReActAgent.run
def _patched_run(self, user_query, **kwargs):
    _last_user_query['text'] = user_query
    return _original_run(self, user_query, **kwargs)
ReActAgent.run = _patched_run
```

We **monkeypatch instead of editing SimuHome's source** because the code is
CC BY-NC-ND 4.0 (no derivatives to redistribute).

Failure policy: if the parser errors, the wrapper **fails open** (calls the real
tool) so the verifier can't make the agent worse than baseline by crashing.

**Why everything runs in-process.** SimuHome's evaluator
(`parallel_model_evaluation`) runs episodes in an `mp.Pool` of separate worker
processes. Patches made in the notebook (the request shim, the verifier, their
counters, the GPU-resident parser) do not exist there, and forked workers cannot
use the CUDA context. The notebook therefore swaps in `_InlinePool`, a drop-in
that runs the same work in the notebook process, and calls
`parallel_model_evaluation.main()` through `run_inproc(...)` for **every** arm,
baseline included, so all arms are measured the same way. `set_verifier(on)`
installs or removes the patched `schedule_workflow`.

**Third arm (`selfcheck`).** `_patched_run` can also append a short instruction
to the agent's query ("work out every time first; if two parts of the request
give the same event two different times, do not schedule, call finish and explain").
The verifier still parses the *original* query. This prompt-only arm is the
fair comparison for "why not just ask the LLM to double-check?".

## 9. The three models in play (don't confuse them)

| Role | Model | Where it runs | Trained? |
|---|---|---|---|
| Agent LLM (does the ReAct loop, picks tools) | `google/gemma-4-31b-it` on NVIDIA's free hosted API (OpenAI-compatible endpoint; ~40 req/min, ~45 s per call observed) | NVIDIA cloud | No, used as-is |
| Judge LLM (scores infeasible episodes) | the model configured in the eval spec (same hosted endpoint) | same | No |
| Intent Parser SLM (query -> IR) | `Qwen/Qwen3-1.7B` + LoRA adapter | Kaggle T4 GPU 0 | **Yes, this is the only thing we train** |

Agent-model history (why Gemma): Llama-3.3-70B on NVIDIA is retired (HTTP 410),
free OpenRouter models were retired or capped at 50 requests/day, and a local
Qwen2.5-7B-AWQ in vLLM scored 0/5 (broke the JSON action format, invented a
device). Gemma-31B was the only free option that passed the agent-format test.
Free hosted models can disappear, so record the id and dates of every run.

The same agent model is used for every arm, so any score difference is
attributable to the verifier. Request plumbing: a shim patches
`openai...Completions.create` (throttle, call counter, drop unsupported params,
strip `strict`/`additionalProperties`); a preflight cell checks the model's
reply parses as `{thought, action, action_input}` before any run.

## 10. Fine-tuning the intent parser (notebook section 5)

- Data: `data/qt4_ir_train.jsonl` / `qt4_ir_val.jsonl`, each row =
  `(query, ir_for_training)`.
- Prompt format: system instruction ("output ONLY the JSON IR") + the query;
  the target is the IR serialized with `json.dumps`.
- LoRA: rank 16, alpha 32, on all attention and MLP projections
  (`q,k,v,o,gate,up,down`). About 1% of weights are trainable (17.4M of 1.74B).
- Trainer: TRL `SFTTrainer` (`peft_config=`, `loss_type='nll'`), `bf16=True` (the T4 emulates
  bf16 and it ran; the "no bf16, use fp16" note applies only to the local vLLM server), pinned to GPU 0, checkpoint every 100 steps so a killed Kaggle
  session resumes. A saved adapter in `/kaggle/input/*/qwen3_ir_parser_adapter_v2`
  is reused instead of retraining.
- Parser evaluation (`eval_parser`, written to `parser_eval.json`): valid JSON,
  exact IR match, and **verdict accuracy** = does Z3, run on the *predicted* IR,
  give the right feasible/infeasible answer. This is the metric that matters.
- Still TODO: a few-shot, no-training baseline to show fine-tuning buys something
  real.

## 11. Evaluation (notebook section 7)

**E1, the main experiment.** For each QT4 sub-type (1/2/3) and case
(feasible/infeasible), on the held-out native episodes (`E1_MAX_SEEDS`, 10 so far),
three arms with the same agent model:

| Arm | What it is |
|---|---|
| `baseline` | the agent alone |
| `selfcheck` | agent + prompt-only consistency instruction (implemented, **not yet run**) |
| `verified` | agent + parser + Z3 verifier |

Each run is aggregated per model sub-folder (`aggregate_run`; SimuHome's aggregator
only reads `*.json` directly inside the folder it is given). Results save after
every run (`e1_aggregates.json`, `verifier_telemetry.json`); a run is "done" only
if episodes were evaluated and `infra_errors == 0`, otherwise it is redone.
`inspect_run(run_id)` explains a run per episode (score, required lookups, action
list, final answer, plus a verdict on API/format/accuracy problems).

Scoring uses SimuHome's own evaluators, so numbers compare to the paper:
*feasible* = simulator state at the goal ticks plus required lookups; *infeasible*
= LLM judge against `temporal_conflict`.

**Ablations E2-E6** (to do): parser without Z3, oracle IR vs predicted IR,
verbalizer template vs raw unsat core, per-conflict-type breakdown, latency
overhead, plus Z3 vs a plain-Python consistency function on the same IR (see
Section 14).

## 12. Running it (summary)

Local (only needed after editing `src/`):

```bash
pip install z3-solver
python tests/test_pipeline.py ../SimuHome/data/benchmark     # expect 286/286
python scripts_build_notebooks.py                            # regenerate notebooks
python docs/make_figures.py                                  # regenerate the two figures
```

Kaggle: upload `notebooks/00_full_pipeline.ipynb`; GPU T4 x2, Internet ON; Kaggle
secret `NVIDIA_API_KEY` (key from build.nvidia.com; never paste keys into chat or
notebooks). Run setup, the backend cell, the shim, the **preflight** (must print
`LLM preflight OK` and `agent JSON format: OK`), then the simulator cell, the
baseline smoke test with `inspect_run`, then the parser sections (reuse the saved
adapter dataset to skip training), the verifier cells, and E1. Trim `E1_CELLS`
per session: one arm of one cell is about 1 hour on the free API (~45 s per LLM
call). Download results with the zip cell. The six split notebooks `01`-`06` are
stale; `00` is the maintained path.

## 13. Status and results

| Part | Status |
|---|---|
| IR schema, extractor, Z3 encoder | Built; **286/286** on real data with ground-truth IR |
| Parser (Qwen3-1.7B + LoRA, compact IR) | Trained; held-out **54/54 valid JSON, 42/46 = 91% verdict accuracy**, 20/54 exact match (val: 25/25 valid, 23/23 verdicts) |
| Verifier hook, in-process evaluator | Works end to end on Kaggle (0 infra errors, 0 parse failures) |
| E1, QT4-1 (10 episodes per cell), baseline and verified | **Done** (table below) |
| E1 `selfcheck` arm, QT4-2, QT4-3 | Implemented in the notebook, **not yet run** |
| Duration/resource conflicts (14 episodes) | Out of scope, flagged |

**E1 results (agent `google/gemma-4-31b-it`, held-out QT4-1 episodes, n = 10 each):**

| Case | Baseline | Verified | Verifier activity |
|---|---|---|---|
| Infeasible | 3/10 (30%) | **10/10 (100%)** | 14 rejected, 0 parse failures |
| Feasible | 2/10 (20%) | 2/10 (20%) | 28 passed, **0 rejected** |

Reading: the verifier fixes contradiction blindness and never blocked a valid
schedule. Caveats: n = 10 (not statistically strong), one sub-type, templated
queries, one agent model.

**Why feasible accuracy is only 20% for both arms.** Trace analysis shows the
times are computed correctly (e.g. "+15 min" and "+23 min after the previous
action" came out right). The failures are device-command errors: invented
commands/arguments (`SetFanSpeed`, `MoveToLevel` instead of writing the actual
attribute, `temperature: 100` on a refrigerator), and episodes where the agent
never called `get_device_structure` mostly fail. That is a weakness of this agent
model, outside what a temporal verifier can fix.

## 14. Is this a significant contribution? (honest assessment)

**The weakness, stated plainly.** For QT4 as written, the consistency check is
one-line arithmetic: "does the stated clock time equal now + offset (within a
tolerance)?". Once the parser has produced the IR, a plain Python function would
reach the same verdicts, and E1 would not change. So **Z3 is not what earns the
result here**; the parser (language to structure) and the *placement* of a
checker in front of `schedule_workflow` are. Claiming "an SMT solver" as the
contribution would not survive review, and Agent-C (same idea, SMT in front of
tool calls) already occupies that ground.

**What is genuinely supportable now:** (a) a small fine-tuned parser reaches 91%
verdict accuracy with perfect JSON validity on held-out episodes; (b) a
plug-in, agent-agnostic gate fixes the benchmark's dominant failure mode
(30% -> 100% preliminary) with no false rejections; (c) rejections come with a
specific, grounded reason the agent can act on.

**How to make it a real contribution (pick one direction):**

1. **Make the constraint class genuinely need a solver.** Dependency chains
   ("after B, which is after A, but before C"), concurrency, device durations and
   resource limits (the 14 skipped `completion_vs_pause` episodes), user
   preferences/quiet hours, multi-device overlaps. Here a hand-written function
   stops scaling and minimal unsat cores give explanations that ad hoc code
   cannot. Then add the ablation "Z3 vs hand-coded checker" and show the
   difference on these harder cases.
2. **Reposition as a systems/benchmark-methods paper:** a verified, auditable,
   edge-deployable gate (1.7B parser + checker) versus asking a large LLM to
   self-check; the evidence is the three-arm comparison, latency and cost, plus
   robustness to paraphrase and to different agent models.
3. **Both:** richer constraints (1) evaluated with the three-arm design (2).

**Must-do regardless:** the `selfcheck` baseline (a strong prompt may already
close much of the gap, which would weaken the claim); a Z3-vs-plain-Python
ablation reported honestly; 30+ episodes per cell on all six cells; paraphrase
robustness; a second agent model; error analysis of the 4 wrong verdicts.

## 15. Open items

Whether to reuse Agent-C's constrained generation (its repo was a placeholder),
final LoRA hyperparameters, a periodic check that nobody has published a similar
SMT extension of SimuHome, and porting the structural fixes to the six split
notebooks (not planned; `00` is maintained).
