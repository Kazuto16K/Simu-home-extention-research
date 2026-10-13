# Findings from the first approach-2 experiment (12 Oct 2026)

This document explains, in plain language, what the first full run of approach 2 showed,
why each result happened (checked against the per-episode traces, not guessed), what it
means for the paper, and the decisions to make next. Every number here comes from the
files in `approach-2-results/` (one JSON record per episode, plus `gate_telemetry.json`).

If a term is unfamiliar (episode, seed, arm, held-out), see `docs/GUIDE.md`.

---

## 1. What was run

| Item | Value |
|---|---|
| Benchmark | SimuHome QT4 (scheduling), held-out episodes only (seed divisible by 5) |
| Episodes per group | QT4-1: 10, QT4-2: 8, QT4-3: 9 (for both possible and impossible requests) |
| Agent (large model) | `google/gemma-4-31b-it` on NVIDIA's free API |
| Judge (scores impossible requests) | the same model, asked 3 times, majority vote |
| Small model (parser) | Qwen3-1.7B + LoRA, version 3.1 (adapter `d9b01f65`) |
| Episodes run | 206 (21 runs), 1 lost to an error |

The **arms** (versions of the system) that were run:

| Arm | What it does |
|---|---|
| `baseline` | the agent alone, as in the SimuHome paper |
| `selfcheck` | the agent alone, but told in its prompt to check that the times are consistent |
| `v1_gate` | approach 1 (the old gate inside `schedule_workflow`), rebuilt for a fair comparison |
| `pre_hint` | approach 2: impossible requests are detected before the agent starts; the agent gets a warning |
| `full_answer` | approach 2: impossible requests are answered by the gate itself; the agent is never called |
| `full` | approach 2 with all checks on the agent's scheduling calls (time window + dry run) |

---

## 2. The results table

"calls" = large-model calls per episode (the cost). Seconds per episode are not shown
because they were inflated by API timeouts and retries, which have nothing to do with the
method.

| Group | baseline | selfcheck | v1_gate | pre_hint | full | full_answer |
|---|---|---|---|---|---|---|
| QT4-1 impossible | 4/10 (3.1 calls) | 10/10 (1.1) | 10/10 (4.3) | 10/10 (1.0) | - | **10/10 (0)** |
| QT4-2 impossible | 0/8 (6.9) | 8/8 (3.5) | - | 6/8 (1.0) | - | **8/8 (0)** |
| QT4-3 impossible | 4/9 (7.9) | 5/9 (5.8) | - | 4/9 (5.7) | - | 2/8 + 1 error (3.9) |
| QT4-1 possible | 2/10 (7.7) | not run | 3/10 (7.8) | - | **5/10** (9.8) | - |
| QT4-2 possible | 5/8 (7.1) | not run | not run | - | 4/8 (8.3) | - |
| QT4-3 possible | 2/9 (9.7) | not run | not run | - | 1/9 (10.3) | - |

"-" means the arm does not apply or is not meaningful for that group.

---

## 3. Findings, one by one

Each finding has: what we saw, the evidence, and what it means.

### Finding 1. Without help, the agent misses most impossible requests

**Saw:** baseline caught 4 of 10 QT4-1 and 0 of 8 QT4-2 contradictions.

**Means:** this reproduces the SimuHome paper's main failure ("contradiction blindness") with
a different, newer agent. It confirms the problem we are solving is real for current models.

### Finding 2. Answer mode catches every QT4-1 and QT4-2 contradiction with zero large-model calls

**Saw:** `full_answer` scored 10/10 and 8/8. In all 18 episodes the large model was never
called; the small model + Z3 decided, and the gate replied to the user itself.

**Means:** this is the strongest result so far and fits the IoT-J special issue directly
("deciding when to invoke large models"). For these requests the system needs no cloud
model at all.

### Finding 3. A simple prompt is just as accurate on impossible requests (the main threat)

**Saw:** `selfcheck` (one extra sentence in the prompt: "check the times; if two disagree,
don't schedule, explain") scored 10/10 and 8/8, the same as the gate.

**Evidence:** the selfcheck agents read the device's state themselves (via
`get_device_structure`) and quoted the real remaining time.

**Means:** for this agent, the gate's advantage on impossible requests is **cost** (0 calls
instead of 1.1 to 3.5) and not accuracy. The paper must say this honestly. The open
question that decides how strong the paper is:

> Does the selfcheck prompt make the agent **wrongly refuse possible requests**? Telling an
> agent to look for contradictions often makes it see contradictions that are not there.
> **This has not been measured yet** (selfcheck was only run on impossible requests).

If selfcheck causes false refusals on possible requests and the gate does not, the gate
wins on both sides. If not, the paper rests on cost, plus whatever the possible-request
results show.

### Finding 4. Our warning text misled the agent (QT4-2 hint mode, 6/8)

**Saw:** in seeds 30 and 35 the agent correctly said "there is a contradiction" but was
judged wrong.

**Evidence (seed 30):** the washer really finishes at 18:50. The user said "6:13 PM, which
is 24 minutes before the washer finishes". Our warning said:
*"24 min before utility_room_laundry_washer_1 finishing (= 18:26:21) vs. stated clock time
18:13"*. We meant "24 minutes before the finish is 18:26". The agent read it as "the washer
finishes at 18:26" and told the user that, which is false. The judge, who knows the true
finish time (18:50), failed it. Seed 35 failed the same way.

**Means:** a wording bug in our explanation, not a weakness of the method. Fix: always say
explicitly *"washer 1 finishes at 18:50:18 (59 min left on its timer); 24 minutes before
that is 18:26; you said 18:13"*.

### Finding 5. The judge requires evidence that the agent itself read the device's timer

**Saw:** the QT4-2 and QT4-3 judge instructions (in SimuHome's `prompts/qt4_2/...` and
`qt4_3/...`) say: pass only if *"the agent must correctly recognize the actual completion
time of the anchor by referencing the CountdownTime attribute"*, and fail if it *"does not
obtain CountdownTime before the final response"*. QT4-1's judge has no such rule (which is
why everything passes there).

**Evidence:** in hint and answer mode the **gate** read the timer, but the judge only sees
the **agent's** steps, where no such reading appears. The judge also applies the rule
inconsistently: in QT4-3 hint mode, seeds 20 and 30 passed and seed 75 failed with equally
good answers and no tool calls in any of them.

**Means:** our answers need to show their evidence. In answer mode the gate's real timer
readings can be put into the trajectory (they really happened, so this is honest). In hint
mode the agent can be asked to read the timer itself before answering (costs one extra
call). Separately: **the judge is noisy**, which is one more reason to report results over
more episodes and to hand-check a sample of judge decisions for the paper.

### Finding 6. QT4-3 impossible requests are mostly out of the gate's reach (by design)

**Saw:** gate arms did not beat baseline on QT4-3 impossible (pre_hint 4/9, baseline 4/9).

**Evidence:** of the 9 test episodes, **5 are "completion vs pause" conflicts**: the request
is impossible because a machine's cycle is longer than the time available. No two stated
times contradict, so a timing solver cannot see it; it needs a model of how long each
device runs. We declared this out of scope from the start. That leaves at most 4 episodes
the gate can affect. In answer mode, 2 of those 4 failed because our reply was jargon
(*"no later than 1 min before goal 1 vs. stated clock time 18:56"*), which neither a user
nor the judge can follow.

**Means:** for QT4-3, report results on the in-scope episodes separately, and fix the reply
wording (Finding 4). Supporting device run times is possible future work, not for now.

### Finding 7. On possible requests, QT4-1 looks promising, but part of it is noise

**Saw:** `full` 5/10 vs baseline 2/10 on QT4-1 possible.

**Evidence, per episode:** `full` won seeds 5, 10 and 35 and lost none.
- Seed 35 is a genuine gate win: the dry run refused a broken command twice, the agent fixed
  it, and the episode succeeded.
- Seed 10 may be the gate (a dry-run note was attached).
- Seed 5 flipped **with no gate activity at all**.

**Means:** the hosted model does not give exactly the same answers every time, even at
temperature 0. `v1_gate` (which never acts on possible requests, so should equal baseline)
also differed from baseline by one episode. **Expect about one episode of random difference
per group.** With 8 to 10 episodes per group, a difference of 1 is noise, and even 3 is not
yet statistically convincing (a paired test needs about 6 one-directional changes). This is
the case for running more episodes (the 5-fold option in `GUIDE.md`) if the gain holds up
after the fixes.

### Finding 8. The agent ignores warnings attached to successful scheduling replies

**Saw:** in QT4-3 possible, `full` lost seeds 30 and 45 that baseline won.

**Evidence, seed 45:** the agent started the washer *before* powering it on, so the start
failed silently. When it scheduled the pause, the dry run correctly warned: *"at 14:10 the
washer would be stopped (0), the request asks for paused (2)"*. That warning was only a note
attached to a successful reply, and the agent finished without acting on it. (Baseline
succeeded because it happened to send the start command twice.)

**Means:** a warning that the agent can ignore achieves nothing in practice. The useful
checks are the ones that refuse and say why. A safe way to refuse here without trusting the
small model: **refuse a workflow that would do nothing** (pausing a washer that is not
running). That check does not depend on the parser at all.

### Finding 9. A scheduling call was matched to the wrong goal (QT4-3 seed 30)

**Saw:** the agent scheduled "pause dryer 1" about one minute too early, and the time
check did not catch it.

**Evidence:** the request has two goals for dryer 1: "keep it running until shortly before
the pause" (a deadline with no lower time limit) and "pause it when dryer 2 finishes". The
gate matched the pause call to the first goal, whose window accepts any earlier time, so the
early time passed. The dry-run note also compared against the wrong goal ("would be paused,
request asks running").

**Means:** a design flaw. Calls should be matched to the goal whose target state their
commands actually produce (a pause matches the "paused" goal), not to whichever goal is
closest in time.

> **Correction (13 Oct, found while implementing the fix).** The early pause was *not* why
> seed 30 failed. SimuHome scores a goal as met if the requested state holds at **any moment
> within +/-30 s of the goal time**; it never checks when the action happened. A pause one
> minute early still leaves the dryer paused at the check time, so it passes (the baseline run
> of this episode did exactly that and succeeded). The `full` run failed for another reason
> (most likely the dryness level it wrote). The matching flaw is still real (the dry-run note
> compared against the wrong goal), so the fix stays, but the time check was also changed to
> follow SimuHome's definition of success: a call is refused only if it is **too late** (after
> its goal's window) or so **early that it would overwrite an earlier goal** on the same
> device. See section 9.

### Finding 10. Smaller observations

- The one error (QT4-3 answer mode, seed 70) is the agent never calling `finish` within 20
  steps on a request the gate could not judge. SimuHome labels this an infrastructure error.
- The parser's one known false alarm (QT4-1 possible, seed 25: "8 minutes after the
  previous action" read as "8 minutes from now") cost nothing here because baseline also
  failed that episode.
- Approach 1 (`v1_gate`) still loops on impossible requests: 4.3 large-model calls per
  episode versus 1.0 for hint mode and 0 for answer mode on the same requests.

---

## 4. What this means for the paper

| Claim | Status |
|---|---|
| Current agents miss timing contradictions (problem is real) | **Supported** (Finding 1) |
| The gate detects them before the large model acts, at zero large-model cost | **Supported** for QT4-1 and QT4-2 (Finding 2) |
| The gate is more *accurate* than prompting the agent | **Not supported** for this agent (Finding 3); needs selfcheck on possible requests |
| Checking the agent's plan raises success on possible requests | **Promising, not proven** (Findings 7 to 9) |
| It handles all of QT4-3 | **No**: 5 of 9 impossible test episodes are out of scope by design (Finding 6) |

The honest current story: *a small model plus a solver can take over the detection of
timing contradictions completely, removing the large model from those requests; checking
the large model's scheduling calls catches some of its mistakes; prompting alone can match
detection accuracy for a strong agent but costs more calls (and possibly false refusals, to
be measured).*

---

## 5. Proposed fixes (system v2.1), for you to decide

| Fix | What it changes | Why | Risk |
|---|---|---|---|
| **F1** Clear explanations | Plain language with explicit times: "washer 1 finishes at 18:50 (59 min left); 24 min before that is 18:26; you said 18:13". Device names instead of "goal 1". | Findings 4 and 6: the wording misled the agent and confused the judge | None; wording only |
| **F2** Show the evidence | Answer mode: the gate's real timer readings appear as steps in the trajectory. Hint mode: the agent is asked to read the timer itself before answering. | Finding 5: the judge requires it, and the answer should be grounded anyway | Hint mode costs ~1 more call |
| **F3** Match calls by effect | A scheduling call is matched to the goal whose target state its commands produce. | Finding 9 | Small; tested on the benchmark before use |
| **F4** Refuse workflows that do nothing (optional) | If the dry run shows the steps change nothing (e.g. pausing a stopped washer), refuse and say why. | Finding 8; does not depend on the small model | Possible false refusals if the copy of the house differs from the real one (measured locally: device states match) |

After F1 to F3 (and F4 if chosen), the gate arms have to be **re-run** so that all gate
results come from one version of the system:
`pre_hint`, `full_answer` (and `pre_answer`) on the three impossible groups, and `full` on
the three possible groups. `baseline` and `selfcheck` results stay valid (they do not use
the gate).

---

## 6. Runs still missing (independent of the fixes)

1. **`selfcheck` on the three possible groups.** Most important: does the prompt cause
   false refusals? (Finding 3)
2. **`baseline` on the three possible groups, a second time.** Measures how much results
   change between identical runs (Finding 7). The paper needs this number.
3. `v1_gate` on QT4-2 and QT4-3 (completes the approach-1 comparison).

```python
PLAN = [
    ('qt4-1', 'feasible', 'selfcheck'), ('qt4-2', 'feasible', 'selfcheck'), ('qt4-3', 'feasible', 'selfcheck'),
    ('qt4-1', 'feasible', 'baseline'),  ('qt4-2', 'feasible', 'baseline'),  ('qt4-3', 'feasible', 'baseline'),
]
```
(Run this in a new Kaggle session so the baseline runs are genuinely repeated.)

---

## 7. Decisions for tomorrow

1. Implement F1, F2, F3? (Recommended: yes; F1 and F2 fix bugs, F3 fixes a design flaw.)
2. Add F4? (Recommended: yes; it is the only check that turned a real failure into
   something the agent cannot ignore, and it does not depend on the small model.)
3. Run the missing selfcheck and repeated-baseline runs now (they do not depend on 1 and 2).
4. After the re-runs: if the possible-request gain holds, run the 5-fold option for 50
   episodes per group.

---

## 8. Where the evidence is

| File | Contents |
|---|---|
| `approach-2-results/e1_summary_v2.csv` | the table in section 2 |
| `approach-2-results/SimuHome/experiments/a2_<arm>_<group>/` | one JSON per episode: every agent step, tool call, observation, final answer, judge votes |
| `approach-2-results/gate_telemetry.json` | per episode: gate verdict, every refusal, warnings, device-name repairs, timings |
| `results/e1_session1_analysis.md` | the short technical version of this document |
| `results/parser_v3.0_analysis.md` | how the parser was checked and improved (v3.0 to v3.1) |

---

## 9. Update (13 Oct): fixes F1-F4 implemented as gate v2.1

All four fixes are in the code (`src/explain.py`, `src/gate.py`, `src/shadow_sim.py`,
`src/harness.py`). Gate runs are now saved with a version tag (`..._g21`) so they never mix
with the session-1 results above.

**One more change, found while testing:** SimuHome scores a goal as met if the requested
state holds at any moment within +/-30 s of the goal time, and never checks when the action
happened (see the correction under Finding 9). The time check now follows that definition: a
scheduling call is refused only if it is too late (after its goal's window) or so early that
it would overwrite an earlier goal on the same device.

### How v2.1 was checked before spending any API calls

| Check | Result |
|---|---|
| Joint Z3 verdicts / goal times inside windows (gold labels) | 286/286, 301/301 |
| Gate on all 300 episodes with live device timers: correct times refused | 0 of 301 |
| Calls 5 min after a device's last goal refused | 251 of 251 |
| QT4-3 seed 45 (washer never started, then paused) | refused: "would change nothing" (F4); correct order passes |
| QT4-3 seed 30 (pause): 1 min early / on time / 1.5 min late | passes / passes / refused |
| Answer-mode trace (what the judge sees) | contains the timer reading and the true finish time |
| **Replay of all 151 real scheduling calls from session 1** (64 possible-request episodes, real parser output, real simulator) | **0 correct schedules blocked**; mistakes flagged in **23 of 42 failed episodes** (29 failing commands, 12 do-nothing workflows, 4 calls on impossible requests, 2 too-late times) |

Before the timing change, the same replay showed the gate would have wrongly blocked 3
successful QT4-3 episodes; that is why the change was made.

What the replay cannot show: whether the agent, once refused, would have fixed its mistake.
That needs the new E1 run (`notebooks/00_approach2_pipeline.ipynb`, plan parts A and B).
Replay data: `approach-2-results/replay_gate_v21.json`; script: `tests/replay_traces.py`.

### Does this work for any home?

The method is not tied to one house: every SimuHome episode is a different, randomly
generated home, and all results are on homes the parser never saw. But the evidence is so far
only SimuHome. The Z3 check, time windows and safeguards work on any device list; reading
finish times needs devices that report remaining time (Matter's CountdownTime); the dry run
needs a simulator of the home (exact here, approximate in a real home); and the small model
was trained only on SimuHome's template-generated requests. For a claim beyond "SimuHome's
generated homes", the paper needs a rephrasing test (an LLM rewords the test requests) and
fresh homes from SimuHome's generator with unseen seeds.
