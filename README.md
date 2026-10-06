# Neuro-Symbolic Verification for LLM Smart-Home Scheduling Agents

MTech research project targeting the IEEE IoT-J special issue "AI Agent Enabled
Small-Large Model Collaboration for IoT" (deadline Dec 15, 2026).

**Idea in one line:** a small fine-tuned model (Qwen3-1.7B + LoRA) turns a smart-home
scheduling request into a structured form, a Z3 solver checks that the stated times
are consistent, and a gate in front of SimuHome's `schedule_workflow` rejects
contradictory requests with a plain-language reason. The big agent LLM is not retrained.

## Where to start reading (study order)

1. `docs/PROJECT_REPORT.md` - the story for a presentation: problem, idea, results,
   limits, next steps, slide outline, glossary. Figures: `docs/architecture.png`,
   `docs/e1_results.png`.
2. `ARCHITECTURE.md` - the detailed technical write-up (IR, Z3 encoding, hook, evaluation,
   honest assessment of the contribution in section 14).
3. `CODE_EXPLAINS.md` - a cell-by-cell walkthrough of `notebooks/00_full_pipeline.ipynb`.
4. `papers/SimuHome_ICLR2026_arXiv-2509.24282.pdf` - the benchmark paper this builds on.
5. `src/` and `tests/` - the small, locally testable core.

## Status (October 2026)

- Pipeline built and run end to end on Kaggle. Z3 encoder: 286/286 on benchmark ground truth.
- Parser: 54/54 valid JSON and 91% verdict accuracy (42/46) on held-out episodes.
- First E1 result (QT4-1, 10 held-out episodes per cell, agent `google/gemma-4-31b-it`):
  infeasible 30% -> 100%, feasible 20% -> 20% with 0 false rejections. **Preliminary:**
  small sample, one sub-type. Raw outputs: `results/session1_qt4-1/`.
- Not done yet: the prompt-only `selfcheck` comparison, QT4-2/QT4-3, harder constraints,
  more episodes, a second agent model, paraphrase robustness. See `docs/PROJECT_REPORT.md`
  section 7 and `ARCHITECTURE.md` section 14.

## Repo layout

```
README.md, ARCHITECTURE.md, CODE_EXPLAINS.md   documentation
docs/                      PROJECT_REPORT.md, DATA_AND_RESULTS.md, Project_Presentation.pptx (5 slides), architecture.png, simuhome_vs_ours.png, e1_results.png, make_figures.py
papers/                    the SimuHome paper (see papers/README.md for licence notes)
results/session1_qt4-1/    raw E1 outputs from the first Kaggle session (JSON + per-episode traces)
src/                       ir_schema.py, constraint_encoder.py, extract_ir_from_episode.py
tests/test_pipeline.py     regression test against the real benchmark episodes
notebooks/00_full_pipeline.ipynb   the one notebook that runs everything on Kaggle
scripts_build_notebooks.py generates the notebook from src/*.py (edit this, not the .ipynb)
```

Not in the repo on purpose (see `.gitignore`): the SimuHome benchmark itself (clone it
from github.com/holi-lab/SimuHome, CC BY-NC-ND 4.0), the trained adapter (70 MB, regenerate
or keep as a Kaggle dataset), generated data, the stale split notebooks `01`-`06`, and any
API keys.

## Running locally (no GPU needed)

```bash
git clone https://github.com/holi-lab/SimuHome.git ../SimuHome
pip install z3-solver
python tests/test_pipeline.py ../SimuHome/data/benchmark     # expect 286/286
python scripts_build_notebooks.py                            # regenerate notebooks/00_full_pipeline.ipynb
python docs/make_figures.py                                  # regenerate the two figures (needs matplotlib)
```

## Running on Kaggle

Upload `notebooks/00_full_pipeline.ipynb`. Settings: GPU T4 x2, Internet On, secret
`NVIDIA_API_KEY` (key from build.nvidia.com), and attach a dataset containing the trained
`qwen3_ir_parser_adapter_v2` folder to skip retraining. Use *Save Version -> Save & Run All*
for an unattended run. Edit only the `PLAN` / `BUDGET_HOURS` block in the E1 cell. Details
for every cell are in `CODE_EXPLAINS.md`. Free hosted models can disappear; the preflight
cell stops the run if the agent model is unusable.

The notebook embeds the three `src/` files as `irlib/` (not `src/`) because SimuHome has its
own top-level `src` package and two same-named packages in one kernel shadow each other.
`scripts_build_notebooks.py` rewrites the imports automatically; rerun it after editing `src/`.

## Key findings from source inspection (SimuHome)

- **Episode schema** (`data/benchmark/qt4-*.json`): `meta` (query_type,
  case, conflict_type), `query` (NL), `eval.required_actions` (expected
  info-gathering tool calls), `eval.goals[]` (`when` + `anchor` + `targets`
  with device attribute `asserts`), `initial_home_config` (tick_interval,
  base_time, full device/room state). Infeasible episodes add
  `temporal_conflict` with the two conflicting time readings already
  resolved by SimuHome's own generator — this is what makes deterministic
  IR-label extraction possible with zero manual annotation.
- **Tool signatures**: `src/agents/tools.py` — `schedule_workflow(start_time,
  steps)` is the integration point; it does no feasibility check
  (confirmed — this is exactly the paper's Sec 6.2 root cause). Full
  registry: `get_current_time`, `get_room_devices`, `get_device_structure`,
  `get_attribute`, `execute_command`, `write_attribute`,
  `get_environment_control_rules`, `finish`, etc.
- **ReAct loop**: `src/agents/strategies/react_agent.py` — JSON
  `{thought, action, action_input}` per turn, `finish` terminates. The
  verified-scheduling wrapper (notebook section 6) monkeypatches `TOOL_REGISTRY`
  and `ReActAgent.run` rather than forking the vendored source, out of
  care for the CC BY-NC-ND 4.0 license on republishing modified code.
- **Evaluation**: feasible QT4 scored by simulator-state diff at
  goal ticks (`episode_evaluation/qt4/feasible.py`); infeasible QT4 scored
  by an LLM-judge panel comparing the agent's final answer against
  `temporal_conflict` (`infeasible_{1,2,3}.py`). Reuse these, don't
  reimplement scoring (used by the E1 section of the notebook).
- **Conflict type taxonomy found across the 300 downloaded QT4 episodes**:
  `absolute_time_mismatch(_nonop_multi)`, `delay_time_mismatch`,
  `timing_mismatch_op_nonop` — all stated-time-vs-time contradictions, all
  caught by the Z3 encoder. `completion_vs_pause` — a duration/resource
  contradiction (device cycle length vs. available window), **not** caught;
  see Known limitation.

## Known limitation (flagged per handoff Sec 6, not hidden)

`completion_vs_pause` conflicts (14/150 infeasible QT4-3 episodes) require
a device duration model (e.g. "Heavy mode washer cycle ≈ 60 min") to
detect — no two stated time readings contradict each other, the
contradiction is between a stated timing goal and a device's own physics.
`Goal.requires_duration_model=True` marks these; the encoder currently
treats them as SAT-by-default (unverified) rather than silently
mis-classifying them as feasible with false confidence. The handoff scopes
this as optional bonus scope (Sec 3, Component 2) — revisit only if time
remains after the core E1 result is in hand.

SimuHome's `pyproject.toml` pins `requires-python = ">=3.13"`. Its actual
code does not use any 3.13-only syntax (checked directly), so every install
cell uses `--ignore-requires-python` to bypass the gate rather than fail on
a Kaggle image still on 3.10/3.11. If a genuine 3.13-only runtime behavior
ever surfaces, this is where to look first.

## Open items carried over from the handoff (unresolved, don't re-litigate)

- Agent-C's `Gen-Call`/`Gen-Call-Reprompt` enforcement machinery is **not
  yet public** — `github.com/structuredllm/agent-c` currently contains only
  a placeholder README ("released soon"). Re-check before deciding whether
  to reuse it or build the LLM-constraint interleaving from scratch.
- LoRA target model size (Qwen3-1.7B vs 4B) — the notebook defaults to
  1.7B and says to move up only if it underfits; not decided a priori.
- Novelty check ("cited by SimuHome" on Scholar) — standing task, not done
  here.
