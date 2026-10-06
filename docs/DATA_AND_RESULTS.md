# Data and results: what we have, and how each piece is used for our purpose

This note answers four questions for the project: where the data comes from, what we extract from it,
what each result shows, and how each result is used in the argument. Numbers are from the Kaggle
session 1 outputs in `results/session1_qt4-1/` and from re-running the extraction locally.
Companion files: `PROJECT_REPORT.md` (the story), `../ARCHITECTURE.md` (the design),
`../CODE_EXPLAINS.md` (the notebook).

---

## 1. Where the data comes from

We did **not** collect or generate any data ourselves. It is the **SimuHome** benchmark (ICLR 2026),
specifically its **QT4 (workflow scheduling)** episodes:

| Fact | Detail |
|---|---|
| Benchmark | 600 episodes over 12 categories (6 query types x feasible/infeasible), 50 episodes per category |
| What we use | The 6 QT4 categories = **300 episodes** |
| Who wrote the requests | **GPT-5 mini** drafted them from structured goals; **two graduate students** reviewed and corrected them (Cohen's kappa 0.92). Source: SimuHome Sec. 4.2, Step 3 |
| Who made the contradictions | The authors' generator deliberately samples contradictory times (e.g. a wrong assumed time paired with the real current time) |
| Licence | Code CC BY-NC-ND 4.0: used here for non-commercial research, not redistributed (it is git-ignored) |

The 300 QT4 episodes:

| Sub-type | Meaning | Feasible | Infeasible (conflict type, count) |
|---|---|---|---|
| QT4-1 | time-based ("in 15 minutes") | 50 | `absolute_time_mismatch_nonop_multi` 50 |
| QT4-2 | event-driven ("20 min after the dishwasher finishes") | 50 | `timing_mismatch_op_nonop` 50 |
| QT4-3 | coordinated (devices finish together) | 50 | `absolute_time_mismatch` 15, `delay_time_mismatch` 11, `impossible_early_end` 10, `completion_vs_pause` 14 |

---

## 2. What an episode holds, and what we take from it

Each episode is a JSON file. The parts we use:

| Field | What it is | How we use it |
|---|---|---|
| `query` | the natural-language request | the **input** the parser learns to read |
| `eval.goals` | the structured meaning: which device, which attribute value, when (relation, offset, anchor) | becomes the **time readings and targets** of the IR label |
| `temporal_conflict` (infeasible only) | the two contradicting times, already computed by SimuHome | becomes the **second, conflicting time reading** in the IR label |
| `initial_home_config.base_time` | the simulator's "now" | converts clock times to minutes for Z3 |
| `eval.required_actions`, scoring code | what the agent must look up; how success is judged | reused unchanged to score our experiments |

This is why no manual labelling was needed: the benchmark already contains the answer key.

---

## 3. Extraction: how a label is made

`src/extract_ir_from_episode.py` (deterministic, no LLM):

1. **Feasible episode:** one time reading per goal in `eval.goals`.
2. **Infeasible episode:** the same, plus a second reading taken from `temporal_conflict`
   (a stated clock time that disagrees with the first reading).
3. **Two outputs per episode:**
   - `ir_for_training`: the compact IR, with device-anchor times left out (they are not in the text),
     used as the parser's target.
   - `debug_ground_truth`: the full IR with resolved anchor times, used only to test Z3.
4. **Duration-model conflicts** (`completion_vs_pause`) are **flagged, not guessed**
   (`requires_duration_model`), because no two stated times contradict; they need a device duration model.

Coverage on the 300 episodes: **300 extracted**, of which **286 are checkable by Z3** and **14 are
flagged** (all `completion_vs_pause`, all QT4-3 infeasible).

---

## 4. How the 300 episodes are split and used

| Split | Episodes | Selection | Used for |
|---|---|---|---|
| Train | 221 | 90% of episodes with seed not divisible by 5 | fine-tuning the parser |
| Validation | 25 | remaining 10% of those | monitoring training, quick parser check |
| Held-out test | 54 | every episode with seed divisible by 5 | parser evaluation **and** E1 (agent experiments) |

All 300 are used; none are dropped. A check in the notebook asserts that no test request text also
appears in train or validation.

Per cell:

| Sub-type, case | Train | Val | Test |
|---|---|---|---|
| QT4-1 feasible / infeasible | 37 / 36 | 3 / 4 | 10 / 10 |
| QT4-2 feasible / infeasible | 38 / 37 | 4 / 5 | 8 / 8 |
| QT4-3 feasible / infeasible | 38 / 35 | 3 / 6 | 9 / 9 |

Consequences: E1 for QT4-2 runs 8 episodes per arm, not 10, and 5 of the 9 QT4-3 infeasible test
episodes are duration-model cases, so only 4 are checkable by Z3. The 9 duration-model episodes in
train/validation carry labels with a single time reading, which cannot teach the parser what a
contradiction looks like.

The same data serve **four different purposes**, which is the point of the pipeline:

| Purpose | Data | Question it answers |
|---|---|---|
| Is the **checker** correct? | all 300, ground-truth IR | Does Z3 reach the right verdict if the IR is right? |
| Can the **parser** learn the format? | train + val | Does it produce valid, correct IR? |
| Does the parser **generalise**? | held-out 54 | Is it right on episodes it never saw? |
| Does the **whole system** help an agent? | held-out episodes, run through the simulator | Do outcomes improve with the verifier? |

---

## 5. Results, what each shows, and how we use it

### 5.1 Z3 encoder on ground truth: 286 / 286

- **What:** with the correct IR, Z3 classifies every checkable episode correctly (150 feasible, 136 infeasible).
- **Shows:** the checking logic and the extractor agree on this benchmark.
- **Does not show:** that the parser can produce a correct IR, or that the logic covers scheduling in general.
- **Use:** justifies trusting the checker. Any later error is then attributable to the parser or the agent.

### 5.2 Parser (Qwen3-1.7B + LoRA), held-out: 54/54 valid JSON, 42/46 verdicts (91%)

| Split | n | Valid JSON | Exact IR match | Verdict correct |
|---|---|---|---|---|
| Validation | 25 | 25 | 9 | 23 / 23 |
| Held-out test | 54 | 54 | 20 | 42 / 46 |

- **Shows:** a 1.7B model can turn these requests into IR the solver can use, on unseen episodes.
- **Verdict accuracy** (does Z3, fed the predicted IR, give the right feasible/infeasible answer) is the
  metric that matters; exact match is stricter and less relevant. The denominators exclude duration-model episodes.
- **Use:** the evidence for the "neural" half of the system. With n = 46 the true rate could be roughly 80-97%.

### 5.3 Training run: 221 examples, 42 steps

- Trainable parameters 17.4M of 1.74B (1.0%); final training loss 0.439, validation loss 0.448,
  token accuracy 0.90, from the one logged evaluation row.
- **Use:** shows training converged without a large train/validation gap. It is **not** a curve, since only
  the final step was logged, and the loss probably includes the request text as well as the JSON (not
  confirmed), so it is not a task-accuracy measure. The notebook now saves `train_log.json` on the next training run.

### 5.4 End-to-end E1, QT4-1, agent `google/gemma-4-31b-it`, 10 held-out episodes per cell

| Case | Agent alone | Agent + verifier | Verifier activity |
|---|---|---|---|
| Infeasible | 3/10 (30%) | **10/10 (100%)** | 14 calls rejected, 0 parse failures |
| Feasible | 2/10 (20%) | 2/10 (20%) | 28 calls passed, 0 rejected |

- **Shows:** on contradictory requests the gate turns missed contradictions into correct refusals, and on valid
  requests it never blocked anything (0 of 28), so it did no harm.
- **Use:** the headline claim. It is preliminary: n = 10, one sub-type, one agent, no prompt-only comparison yet.
- **Why feasible stays at 20% in both arms:** the traces show the timing is computed correctly, but the agent
  invents device commands. In the 4 episodes where it never called `get_device_structure`, all 4 failed; where
  it did, 2 of 6 succeeded. That is an agent-capability problem outside what a temporal verifier addresses.
- **Latency note:** average seconds per episode were higher with the verifier in session 1 (feasible 218 s vs
  377 s; infeasible 159 s vs 208 s), but those numbers are dominated by the hosted API's variable speed. Verifier
  time itself was not measured then; the notebook now records it.

---

## 6. Traceability: where each number lives

| Claim | File |
|---|---|
| Z3 286/286 | `tests/test_pipeline.py` (run locally) |
| Parser metrics | `results/session1_qt4-1/parser_eval.json` |
| Training table | Kaggle executed notebook, training cell (see `../CODE_EXPLAINS.md`) |
| E1 per run and arm | `results/session1_qt4-1/e1_aggregates.json` |
| Verifier counts and per-call log | `results/session1_qt4-1/verifier_telemetry.json` |
| Per-episode traces | `results/session1_qt4-1/experiments/e1_*/...` |
| Split construction | notebook split cell; `src/extract_ir_from_episode.py` |

---

## 7. Limits of the data, and what they mean for the claims

| Limit | Consequence | Planned handling |
|---|---|---|
| Requests are machine-written (GPT-5 mini), template-like | 91% says nothing about real human phrasing | paraphrased and hand-written test set |
| Only 54 held-out episodes (8-10 per cell) | wide confidence intervals; some cells too small for strong claims | report intervals; consider cross-fitting the parser to test on all 300 |
| 14 duration-model episodes unchecked, 9 of them in training | QT4-3 infeasible is only partly covered, and training sees unlearnable labels | exclude or model durations |
| One agent model, one sub-type tested | effect may be specific | second agent, QT4-2/3, prompt-only self-check arm |
| The extractor and IR cover only after / before / at with anchors | not general scheduling | richer constraints (chains, concurrency, durations) |

---

## 8. How the results feed the next steps

1. **Same data, new question:** run the QT4-2 and QT4-3 cells and the prompt-only `selfcheck` arm on the
   held-out episodes. The checker and parser are reused unchanged.
2. **Same parser, tougher test:** a paraphrased test set tells us whether the 91% survives real wording,
   which decides whether more training data is needed.
3. **Same pipeline, richer constraints:** the 14 flagged episodes are the natural first step toward
   constraints that genuinely need a solver, which is where the contribution can become stronger.
