# CODE_EXPLAINS: a walkthrough of `notebooks/00_full_pipeline.ipynb`

This file explains the one notebook that runs the whole project, cell by cell, in plain language, so
you can read it away from Kaggle. Cell numbers are positions in the current notebook (0 = first cell);
if the notebook is regenerated they can shift by one or two, so use the section names too.

**Reminder:** the notebook is *generated* by `scripts_build_notebooks.py`. To change it, edit that script
(or `src/*.py`) and run `python scripts_build_notebooks.py`. Do not hand-edit the `.ipynb`.

---

## 0. The big picture (read this first)

The notebook does five jobs, in this order:

```
1  Setup            install libraries, clone SimuHome, write our 3 Python modules (irlib)
2  Check the logic  prove the Z3 checker is right on the 300 benchmark episodes (no GPU, no API)
3  Agent plumbing   connect an agent LLM through an API, start the simulator, build an evaluation harness
4  Train/evaluate   build (query -> IR) data, fine-tune/load the small parser, measure it
5  Experiments      install the verifier in front of schedule_workflow and run the arms (E1)
```

Who runs where:

| Thing | Runs on | Needs |
|---|---|---|
| Agent LLM (Gemma-31B) | NVIDIA's servers | internet + `NVIDIA_API_KEY` |
| Judge LLM (scores infeasible episodes) | the same hosted endpoint | same |
| Intent parser (Qwen3-1.7B + LoRA) | Kaggle GPU 0 | GPU |
| Z3 solver, simulator, evaluator | the notebook's CPU | nothing |

**Why the evaluator runs inside the notebook process.** SimuHome normally starts separate worker
processes to run episodes. Anything we patch in the notebook (the request throttle, the verifier, the
GPU-loaded parser) would not exist in those workers. So the notebook swaps in a tiny `_InlinePool`
that runs the episodes in the same process. This is the single most important engineering decision.

---

## 1. Cell map

| Cells | Section | What it does |
|---|---|---|
| 0-1 | Title, "1. Setup" | Markdown only |
| 2 | Install | `pip install z3-solver transformers peft accelerate bitsandbytes trl datasets torchao` |
| 3 | Environment | pins the kernel to GPU 0, remembers `PROJECT_ROOT`, clones SimuHome, installs it |
| 4-8 | Write `irlib` | writes our three modules to disk with `%%writefile`, plus `__init__.py` |
| 9-11 | "2. Z3 sanity check" | one hand-built example, then all 300 benchmark episodes |
| 12-13 | "3. Baseline", backend | choose the agent LLM, API base, model, secret, throttle |
| 14-15 | Request shim | wraps every OpenAI call (throttle, counter, drop unsupported options) |
| 16-17 | Preflight | proves the agent LLM answers *and* produces the agent's JSON format |
| 18 | Simulator | starts SimuHome's FastAPI simulator on port 8000 |
| 19-20 | Eval harness | `build_eval_spec`, `_InlinePool`, `run_inproc`, `aggregate_run` |
| 21-22 | Baseline smoke test | 3-episode check (off by default) |
| 23-24 | `inspect_run` | per-episode diagnosis of any run |
| 25-27 | "4. Generate data" | optional generation of extra episodes (off by default) |
| 28 | Split | builds train / val / held-out test files |
| 29-34 | "5. LoRA fine-tune" | load the adapter (or train), `parse_intent`, `eval_parser` |
| 35-36 | Parser ablations | few-shot baseline and error analysis |
| 37-40 | "6. Verifier" | the gate in front of `schedule_workflow`, and the on/off switches |
| 41-45 | "7. E1" | the plan runner: baseline / selfcheck / verified arms |
| 46-47 | Results | summary table, CSV and the download zip |
| 48 | Ablation notes | markdown: what is still to do |

---

## 2. Section by section

### Cells 2-3: setup and the two traps

- `os.environ['CUDA_VISIBLE_DEVICES'] = '0'` hides GPU 1 from this kernel. It was needed when a local agent
  model used GPU 1, and prevents the trainer from spreading across two GPUs and running out of memory.
  It must be set before any CUDA call.
- `PROJECT_ROOT = os.getcwd()` is captured **once**. Later cells `os.chdir` into `SimuHome/` and back;
  without a fixed root, re-running a cell nested the folders (`SimuHome/SimuHome/...`) and broke paths.
- SimuHome is installed with `--ignore-requires-python` because its `pyproject.toml` says Python 3.13
  while Kaggle may be older; the code does not actually need 3.13.

### Cells 4-8: the `irlib` package (our three modules)

These cells write `ir_schema.py`, `constraint_encoder.py` and `extract_ir_from_episode.py` to `irlib/`.
They are copies of `src/*.py` with `from src.` rewritten to `from irlib.`. The rename exists because
SimuHome has its *own* top-level package called `src`; two packages with the same name in one Python
process shadow each other.

What each module does:

- **`ir_schema.py`**: the data types. An `IntentIR` has `goals`; each `Goal` has `readings` (ways the
  request states *when*) and `targets` (which device attribute should have which value). A `TimeReading`
  has a `relation` (`after`/`before`/`at`), an `anchor` (`now`, or a `device_event` such as "when the
  dishwasher finishes"), and either an `offset_minutes` or an `absolute_time`.
  - `compact_ir_dict` shrinks the IR for the small model (drops fields that can be rebuilt).
  - `ir_from_compact` rebuilds the full IR at run time (default tolerance 0.5 min, plus 1 min when the
    clock time is only given to the minute).
- **`constraint_encoder.py`**: turns an IR into Z3 constraints. `check_feasibility(ir)` returns a result
  with `.feasible` and `.explanation()` (see "How the Z3 check works" below).
- **`extract_ir_from_episode.py`**: builds training labels with no manual work. SimuHome episodes already
  contain `eval.goals` (the structured meaning of the query) and, for infeasible episodes,
  `temporal_conflict` (the two contradicting times). `extract_pair(episode)` returns the question, the
  compact IR to train on (`ir_for_training`) and a full IR with resolved anchor times for testing
  (`debug_ground_truth`).

### Cells 9-11: is the logic itself right?

- Cell 10 builds the "11:35 AM, that is 13 minutes from now" example by hand and asserts Z3 says infeasible.
- Cell 11 runs the extractor + Z3 on **all 300 benchmark QT4 episodes** using the *ground-truth* IR and
  asserts zero mistakes (286/286 judged; 14 `completion_vs_pause` episodes are skipped because they need a
  device duration model). This proves the *checker* is right given a correct IR. It says nothing about
  whether the *parser* can produce a correct IR; that is measured later.

### Cells 12-13: choosing the agent LLM

`LLM_BACKEND = 'custom'` means "any OpenAI-compatible API". Defaults:

```
LLM_API_BASE   = https://integrate.api.nvidia.com/v1
LLM_MODEL      = google/gemma-4-31b-it
LLM_API_KEY_ENV= NVIDIA_API_KEY      (a Kaggle secret; the key is read at run time, never written in the file)
MIN_SECONDS_BETWEEN_CALLS = 2.0      (the free tier allows about 40 requests/minute)
DROP_PARAMS    = ['seed', 'response_format']   (options NVIDIA does not support are removed)
```

Other backends are kept for fallback: `openrouter`, and `local_vllm` (serve a model on the notebook's own
GPU; slow to set up and the local 7B model scored 0/5, so it is not recommended). Change only these lines
to try another provider.

### Cells 14-15: the request shim

SimuHome calls the model through the `openai` Python library. The shim replaces
`Completions.create` with a wrapper that, for every call: removes unsupported parameters, strips
`strict`/`additionalProperties` from the JSON schema, waits so calls stay under the rate limit, and counts
the call (`LLM_CALLS['count']`). It also adds two protections for unattended runs:

- **Retries (`SHIM_RETRIES = 4`).** On provider-side errors (5xx such as 504, 429, timeouts, dropped
  connections) it waits 20, 40, 80, 160 s and tries again, printing `[shim] ...: retry n/4`.
- **Hard call cap (`LLM_CALL_CAP = 1500`).** Every attempt, retries included, counts. Past the cap the shim
  raises instead of calling the API, and the plan runner will not start a run that could pass it.

It is installed once and is safe to re-run. Nothing in SimuHome's source is modified; this is monkeypatching.

### Cells 16-17: the preflight (a cheap guard)

Sends two tiny requests: a plain one, and one shaped like the real agent's request. If the API fails (after the
shim's retries) it **raises**, so an unattended run stops in minutes. It also tries to parse the reply as
`{thought, action, action_input}`, but this probe uses a toy prompt (the real agent has a long system prompt
demanding JSON), so it is only a hint: `STOP_ON_BAD_FORMAT = False` by default and the real format check is the
agent smoke test in the E1 cell.

### Cell 18: the simulator

Starts SimuHome's simulator (`uvicorn src.simulator.api.app:app`, port 8000) in the background and waits
until it answers. The agent's tools (`get_room_devices`, `schedule_workflow`, ...) are HTTP calls to it.

### Cells 19-20: the evaluation harness

- `build_eval_spec(run_id, qt, case, seeds)` builds the YAML-shaped dict SimuHome's evaluator expects:
  which episodes, which model, ReAct strategy (`temperature 0`, `max_steps 20`, `timeout 120`),
  one worker, and the judge model.
- `_InlinePool` is the in-process replacement for the multiprocessing pool (see section 0). It also
  restores `sys.stdout/stderr`, because the evaluator's worker initializer silences them.
- `run_inproc(run_id, qt, case, seeds)` writes the spec, **deletes any old run folder** (SimuHome refuses an
  existing one), calls SimuHome's `parallel_model_evaluation.main()`, and checks `run_summary.json` exists.
- `aggregate_run(run_id)` runs SimuHome's `aggregate` command on each model sub-folder and returns the
  accuracy summary. (The aggregator only reads JSON files directly inside the folder it is given, and the
  results sit one level deeper, hence the loop.)

### Cells 21-24: the baseline smoke test and `inspect_run`

- `RUN_BASELINE_SMOKE = False`: a 3-episode baseline is off by default (it costs 15-30 minutes); turn it on
  the first time you try a new model.
- `inspect_run(run_id)` prints, per episode, the score, error type, how many required lookups were made,
  the list of actions and the final answer, plus a plain verdict. Reading guide:
  - `infra_errors > 0`: the API/simulator failed; the accuracy is meaningless.
  - `schema_errors > 0`: the model broke the JSON action format.
  - accuracy 0 with no errors: it ran but solved nothing (look at the actions).
  - accuracy > 0, no errors: the agent works.

### Cells 25-28: training data

- Cells 26-27 can **generate extra episodes** with SimuHome's own generator (`RUN_GENERATION = False`, so it
  is skipped). It needs a `home:` block in the spec; that was a past error.
- **Cell 28 builds the files actually used.** If no generated episodes exist, it falls back to the 300 native
  benchmark episodes: **every episode whose seed is divisible by 5 becomes the held-out test set**
  (`seed % 5 == 0`), the rest is split 90/10 into train and validation (`random.seed(0)`). It asserts that no
  test query text also appears in train/val. Outputs: `data/qt4_ir_v2_train.jsonl`, `..._val.jsonl`,
  `..._test_held_out.jsonl`. Result: 221 train / 25 val / 54 test (about 10 test episodes per
  sub-type-and-case cell).
  Each row is `{query, ir_for_training, debug_ground_truth, meta}`.

A concrete row (validation set):

```
query : "At 10:26 AM, that's 21 minutes from now, set freezer 1 in the kitchen to -20 C ..."
IR    : {"goals":[{"readings":[
           {"relation":"after","anchor":{"kind":"now"},"offset_minutes":21.0},
           {"relation":"at",   "anchor":{"kind":"now"},"absolute_time":"10:26"}],
         "targets":[{"device_id":"kitchen_freezer_1",
                     "asserts":[{"attribute":"1.TemperatureControl.TemperatureSetpoint","value":-2000}]}]}]}
```

Two readings for one action ("21 minutes from now" and "10:26") are what make it a trap: they only agree if the
clock happens to say 10:05.

### Cells 29-34: the parser (Qwen3-1.7B + LoRA)

- Cell 31 fixes the prompt format: a system line ("output ONLY the JSON IR"), the user query, then the target
  JSON. `SYSTEM_PROMPT` is shared by training, inference and the few-shot baseline.
- Cell 32 loads `Qwen/Qwen3-1.7B`. **If a trained adapter folder `qwen3_ir_parser_adapter_v2` exists** (or is
  attached as a Kaggle dataset under `/kaggle/input/`), it is loaded and **training is skipped**.
- Cell 33 trains only if there is no adapter: LoRA rank 16, alpha 32, dropout 0.05 on all attention and MLP
  projections (about 1% of weights trainable), 3 epochs, learning rate 2e-4, effective batch 16
  (4 x accumulation 4), sequence length 1024, checkpoints every 100 steps so a killed session resumes.
  Two library workarounds are in there: pass `peft_config=` to the trainer instead of pre-wrapping the model,
  and use `loss_type='nll'`.
- Cell 34:
  - `parse_intent(query)` generates the IR greedily (no sampling, up to 700 new tokens) and `json.loads` it.
  - `eval_parser(rows)` measures three things: **valid JSON**, **exact match** with the target IR, and
    **verdict accuracy** (does Z3, fed with the *predicted* IR, say the right feasible/infeasible?).
    Verdict accuracy is the number that matters, since small wording differences in the IR do not change
    the verdict. For device-anchored readings it fills the anchor time from the gold IR, as the runtime would
    get it from the simulator.
  - Reported so far on held-out: 54/54 valid, 42/46 verdicts (91%), 20/54 exact. Saved to `parser_eval.json`.

### Cells 35-36: parser ablations (new)

- **Few-shot baseline**: the same model with the adapter *switched off* (`disable_adapter()`), shown 4 training
  examples in the prompt. It answers: does fine-tuning actually matter?
- **Error analysis**: every held-out query the fine-tuned parser got wrong is written to `parser_errors.json`.
- The whole cell is wrapped in `try/except`, so a problem here can never stop an unattended run.

### Cells 37-40: the verifier (the core idea, in code)

`verified_schedule_workflow(args)` replaces SimuHome's `schedule_workflow` tool in `TOOL_REGISTRY`. When the
agent calls it:

1. Fetch the user's original sentence (`_last_user_query`, stored by a wrapper around `ReActAgent.run`,
   because the tool itself only receives its arguments).
2. `parse_intent(query)` -> IR. If this fails, **fail open**: call the real tool (the verifier must never make
   the agent worse by crashing).
3. Ask the simulator for the current time (`get_current_time`), and for each "device finishes" anchor fetch its
   remaining time (`get_attribute ... CountdownTime`).
4. `ir_from_compact` -> full IR -> `check_feasibility`.
5. Feasible: call the real `tool_schedule_workflow`. Infeasible: return a `409 TEMPORAL_CONTRADICTION`
   with a readable reason; the agent sees this as the tool's answer and can tell the user or re-plan.

Telemetry is kept in `VERIFIER_STATS` (calls, passed, rejected, parse failures, and total `seconds` spent
verifying) and `VERIFIER_LOG` (query, verdict, explanation for every call).

The verifier wrapper is timed (the latency overhead a paper should report) and then registered as the tool.

Switches (cell 40 and the patched `ReActAgent.run`):

- `set_verifier(on)`: installs/removes the verified tool.
- `set_selfcheck(on)`: appends one sentence to the agent's query ("work out every time first; if two parts
  contradict, do not schedule, call finish and explain"). The verifier still parses the *original* query.

### Cells 41-45: E1, the experiment runner

Edit **only** the block marked `EDIT THIS BLOCK ONLY`:

```
E1_MAX_SEEDS = 10     # episodes per cell per arm (about all that is held out)
BUDGET_HOURS = 6.5    # never START a run that could overrun this
PLAN = [ ('qt4-2','infeasible','baseline'), ('qt4-2','infeasible','selfcheck'), ... ]
```

The three **arms** (versions compared on the same episodes and agent):

| Arm | Verifier | Prompt addition | Meaning |
|---|---|---|---|
| `baseline` | off | none | the agent alone |
| `selfcheck` | off | consistency instruction | "why not just ask the model to be careful?" |
| `verified` | on | none | agent + parser + Z3 |

**Agent smoke test (`AGENT_SMOKE_TEST = True`).** Before the plan, one real episode (qt4-1 feasible) is run
through the actual agent. If it shows `schema_errors`, `infra_errors` or zero evaluated episodes, the cell raises
with a clear message. This costs about 8 LLM calls and a few minutes, and is the faithful check that the model
can run the ReAct loop.

For each plan item the runner: skips it if already complete; stops if the time budget would be exceeded;
sets the arm's switches; runs `run_inproc`; aggregates; **saves everything after every run**; logs time, number
of LLM calls and verifier counters. It counts a run as "bad" if it crashes, has API errors, or has more than 3 schema errors, and **stops the
plan after two bad runs in a row**. Seeds come from the held-out file, so the parser never saw those episodes.

### Cells 46-47: results table and bundle

Prints a table (sub-type, case, arm, correct/evaluated, accuracy, infra and schema errors), writes
`e1_summary.csv`, and zips the important files into `/kaggle/working/overnight_results.zip`
(aggregates, run log, telemetry, parser evaluation and errors, held-out list, adapter, per-episode traces).

---

## 3. How the Z3 check works (the symbolic core)

For each goal the encoder creates **one real-valued variable `t`**: "minutes after now at which this action
happens". Each time reading becomes a band around a target value:

```
"21 minutes from now"          t in [21 - tol, 21 + tol]
"at 10:26" (now is 10:05)      t in [21 - tol, 21 + tol]   -> consistent
"at 10:26" (now is 10:40)      t in [-14 - tol, -14 + tol] -> no overlap with the first band: UNSAT
```

Every constraint is registered with `assert_and_track`, so when Z3 answers UNSAT, `unsat_core()` returns the
minimal clashing set, which `explanation()` turns into text such as
`Infeasible schedule: Goal 0: stated clock time 19:00 vs. 25.0 min after now`.

For "after the dishwasher finishes", the anchor time is another variable pinned to the value read from the
simulator. A goal with `requires_duration_model` is reported as skipped (not checked) rather than guessed.

**Honest note (also in `ARCHITECTURE.md` section 14):** for QT4 as it stands, this check is simple arithmetic.
A plain Python function over the same IR would give the same verdicts. The value today is the learned parser
plus the placement of the gate; the solver becomes essential only with richer constraints (chains,
concurrency, durations).

---

## 4. Files the notebook writes

| File | Content |
|---|---|
| `data/qt4_ir_v2_{train,val,test_held_out}.jsonl` | the splits |
| `qwen3_ir_parser_adapter_v2/` | the LoRA adapter (also reused from a Kaggle dataset) |
| `parser_eval.json` | valid/exact/verdict metrics (val, test, few-shot) |
| `parser_errors.json` | every wrong held-out parser output |
| `e1_aggregates.json` | SimuHome's aggregate per run and arm |
| `e1_run_log.json` | per run: seconds, LLM calls, verifier counters, accuracy |
| `verifier_telemetry.json` | verifier stats and the per-call log |
| `e1_summary.csv` | the results table |
| `SimuHome/experiments/<run_id>/<model>/qt4-*_seed_*.json` | per-episode traces (steps, final answer, score) |
| `overnight_results.zip` | the bundle to download |

---

## 5. Things people usually get wrong (and the symptoms)

| Symptom | Cause |
|---|---|
| `FileExistsError: run directory already exists` | an earlier run left a folder; `run_inproc` now deletes it |
| `evaluated 0/10` and the folder looks empty | the aggregator was given the run folder, not the model sub-folder |
| verifier counters stay 0 on the verified arm | the patch was applied to a second copy of the module, or the evaluator ran in a separate process |
| `infra_errors > 0` | quota, 429/5xx, or a retired model; not an agent problem |
| `HTTP 410` | the hosted model was retired; change `LLM_MODEL` |
| `504` / `5xx` in the preflight | provider overloaded; the shim retries for about 5 minutes, then fails; try later |
| preflight says "NOT PARSEABLE" | only a hint (toy prompt); the agent smoke test decides |
| `LLM call cap reached` | the safety cap `LLM_CALL_CAP` was hit; raise it only if you mean to |
| `schema_errors > 0` | the model cannot produce `{thought, action, action_input}` |
| very slow runs | about 45 s per call on the free tier; that is network latency, not the notebook |

---

## 6. A suggested way to study this

1. Read cells 9-11 and `src/constraint_encoder.py` together until the Z3 example feels obvious.
2. Read the example row in section 2 and the `extract_pair` function: that is where the labels come from.
3. Read cells 37-40: the entire contribution is about 40 lines.
4. Open `results/session1_qt4-1/experiments/` and read two traces (one correct, one wrong) next to
   `docs/PROJECT_REPORT.md` section 5.
5. Then read `ARCHITECTURE.md` sections 13-14 for the results and the honest assessment.
