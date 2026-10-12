# Parser v3.0 (first notebook-01 run, 2026-10-11): analysis

Source: `parser_v3_outputs/` (Qwen3-1.7B + LoRA, 3 epochs, labels v3.0), 54 held-out
QT4 requests (seed % 5 == 0).

## Reported by the notebook (metric v3.0)

| | val (25) | test (54) |
|---|---|---|
| valid JSON | 24 | 54 |
| exact IR | 7 | 18 |
| verdict accuracy | 21/23 | 44/49 |
| "window-safe goals" | 14/18 | 38/47 |
| targets exact | 8/24 | 22/54 |

Validation loss per epoch: 0.71, 0.49, 0.45 (still falling at the end).

## What was really wrong

1. **The window metric was wrong, not the parser (5 of 9 window failures).** It compared
   benchmark goal *i* with predicted goal *i*. The parser lists goals in sentence order,
   the benchmark in another order; the chains themselves were right. The gate matches a
   call to goals by device, so the metric now does the same.
2. **Invented clock times (4 possible requests called impossible).** E.g. "at 16:00" for
   "start the dishwasher now". 3 of 4 were QT4-3. Cause, partly ours: v3.0 labels for
   QT4-3 impossible requests replaced goal 0's reading instead of only adding the clock
   time, so possible and impossible labels differed in structure.
3. **Wrong device names** (e.g. `kitchen_fan_controller_1` for `kitchen_fan_1`): the gate
   cannot match those goals and lets them through unchecked.
4. **One dropped goal** (qt4-1 seed 5: the AC's second action).

## Measured as the gate would behave (device matching)

| | parser as-is | + grounding filter |
|---|---|---|
| verdicts correct | 43/49 | **47/49** |
| possible requests called impossible | 4 | **0** |
| correct schedules wrongly refused (goals) | 2/47 | 4/54 |
| goals unmatched (wrong device name, unchecked) | 6 | 8 |

The grounding filter (`src/grounding.py`) drops a clock time unless the request states it
(or it is a stated time +/- an offset in the same IR). It leaves all 300 gold labels
untouched.

## Fixes for v3.1 (notebook 01 re-run)

- QT4-3 impossible labels = the possible twin + one clock reading on the goal it is about
  (start / delayed start / finish by conflict type). Encoder still 286/286, 301/301.
- 5 epochs, best epoch by validation loss kept.
- Evaluation now matches the gate: grounding on, device matching; reports goals wrongly
  refused, goals unmatched, clock times dropped.
- Grounding is part of the gate (`GateConfig.ground_clock_times`, off for `v1_gate`).

---

# Parser v3.1 (second run, 2026-10-12) - the one to use

Source: `parser_v3_outputs_new/` (adapter sha256 `d9b01f65cd440c13`, run tag
`labels=1218ff0b464b|epochs=5`, fingerprint verified against the local extractor).
Validation loss per epoch: 0.69, 0.46, 0.40, 0.37, 0.36.

Note: a first "v3.1" upload was the v3.0 adapter again (identical bytes): the notebook
skipped training because an old adapter folder existed. Notebook 01 now fingerprints
adapters and checkpoints and retrains on any mismatch.

| held-out (54) | v3.0 | v3.1 | v3.1 + device-name repair (gate) |
|---|---|---|---|
| exact IR | 17 | 33 | - |
| verdicts correct (with clock grounding) | 47/49 | 47/49 | **48/49** |
| invented clock times | 5 | 0 | 0 |
| correct schedules wrongly refused | 4 | 1 | 1 |
| goals unmatched (unchecked) | 8 | 2 | 2 |
| targets exact | 22 | 34 | - |

Remaining errors: qt4-1_feasible_seed_25 (parser wrote "8 min after now" instead of "8 min
after the previous action": a possible request called impossible); qt4-1_feasible_seed_20
(dropped one goal: one correct schedule would be refused, capped at 2 refusals);
qt4-1_feasible_seed_30 (picked a real but wrong device: on/off instead of dimmable light).

Device-name repair (`grounding.repair_device_ids`, in the gate): maps a device id that does
not exist in the house to the closest real one in the same room. Fired once on held-out
(`utility_laundry_dryer_1` -> `utility_room_laundry_dryer_1`, correct), never touches gold.
