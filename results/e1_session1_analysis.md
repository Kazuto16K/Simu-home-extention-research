# E1 session 1 (approach 2), 2026-10-12: analysis

Agent and judge: `google/gemma-4-31b-it` (NVIDIA free API). Parser v3.1 (adapter `d9b01f65`).
Held-out episodes (seed % 5 == 0): 10 / 8 / 9 per cell for QT4-1 / QT4-2 / QT4-3.
Raw data: `approach-2-results/` (per-episode SimuHome records, `gate_telemetry.json`).

## Summary table

| cell | baseline | selfcheck | v1_gate | pre_hint | full | full_answer |
|---|---|---|---|---|---|---|
| QT4-1 impossible | 4/10 (3.1 calls) | 10/10 (1.1) | 10/10 (4.3) | 10/10 (1.0) | - | **10/10 (0 calls)** |
| QT4-2 impossible | 0/8 (6.9) | 8/8 (3.5) | - | 6/8 (1.0) | - | **8/8 (0)** |
| QT4-3 impossible | 4/9 (7.9) | 5/9 (5.8) | - | 4/9 (5.7) | - | 2/8 + 1 infra (3.9) |
| QT4-1 possible | 2/10 (7.7) | - | 3/10 (7.8) | - | 5/10 (9.8) | - |
| QT4-2 possible | 5/8 (7.1) | - | - | - | 4/8 (8.3) | - |
| QT4-3 possible | 2/9 (9.7) | - | - | - | 1/9 (10.3) | - |

Seconds per episode are dominated by API timeouts/retries; use calls per episode for cost.

## Findings

1. **Contradiction blindness reproduced.** Baseline: 4/10 and 0/8 on QT4-1/QT4-2 impossible.
2. **Answer mode: 18/18 on QT4-1/QT4-2 impossible with zero large-model calls.**
3. **A one-line prompt (selfcheck) also reaches 10/10 and 8/8** for this agent. On impossible
   requests the gate's advantage is cost (0 vs 1.1-3.5 calls), not accuracy. Selfcheck's cost on
   POSSIBLE requests (false alarms) has not been measured yet; it decides how strong the claim is.
4. **QT4-2 pre_hint 6/8: our hint wording misled the agent.** The hint said "24 min before
   washer finishing (= 18:26:21)"; 18:26 is the target time, but the agent told the user "the
   washer finishes at 18:26" (true: 18:50) and the judge failed it (seeds 30, 35).
5. **The QT4-2/QT4-3 judges require the agent to have read CountdownTime** ("fails to obtain
   CountdownTime before the final response" -> B). Selfcheck agents read it via
   `get_device_structure` and quote the real remaining time. In hint/answer mode the gate read it,
   but that is invisible in the trajectory the judge sees. The judge applies the rule
   inconsistently (QT4-3 pre_hint: seeds 20, 30 passed and 75 failed with equivalent answers).
6. **QT4-3 impossible: 5 of 9 test episodes are `completion_vs_pause`** (duration conflicts, out
   of the gate's scope by design). The gate can affect at most 4. answer-mode replies there were
   jargon ("no later than 1 min before goal 1 vs. stated clock time 18:56") and failed (seeds 30, 75).
7. **Possible requests, QT4-1: full 5/10 vs baseline 2/10**, but only seed 35 is clearly the
   gate's doing (dry run refused a broken command twice, agent fixed it). Seed 5 flipped with no
   gate activity: **the hosted model is not deterministic** (v1_gate, which never acts on possible
   requests, also differs from baseline by one episode). Expect +/-1 episode of noise per cell.
8. **QT4-3 possible: target warnings were ignored by the agent.** Seed 45: a true warning (washer
   would be stopped, not paused: the agent started it before powering it on) was ignored. Seed 30:
   the pause call was matched to the "keep running until the pause" deadline goal (no lower
   bound), so a pause scheduled ~1 min early passed the time check, and the warning compared
   against the wrong goal.
9. QT4-1 parser error seed 25 (possible request judged impossible) cost nothing here (baseline also 0).
10. The one "infra" error (QT4-3 full_answer seed 70) is the agent never calling finish within 20 steps.

## Fixes (system v2.1)

- F1 Verbalizer: plain language, explicit device finish times ("washer 1 finishes at 18:50:18,
  59 min left on its timer; 24 min before that is 18:26:18, but you said 18:13"), no "goal 1".
- F2 Evidence: answer mode records the gate's CountdownTime reads as steps in the trajectory
  (they did happen); hint mode asks the agent to confirm by reading CountdownTime before finishing.
- F3 Goal matching by effect: match a schedule call to the goal whose target state its steps
  produce (from the dry run), not the goal closest in time; deadline goals no longer swallow calls.
- F4 (optional) No-effect check: refuse a workflow whose steps change nothing on the copy of the
  house (e.g. Pause on a washer that is not running). Parser-free.

All gate arms on impossible requests and `full` on possible requests must be re-run after F1-F3
so that every gate arm uses one system version. Baseline and selfcheck stay valid.
