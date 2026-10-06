"""Generates the project's Jupyter notebooks (01-06) as plain .ipynb JSON.

Run once locally: `python scripts_build_notebooks.py`
Regenerate after editing src/*.py, since notebooks 02/04/05 embed those
files verbatim via %%writefile cells (Kaggle notebooks don't have this
repo's `src/` package available, so each notebook must be self-contained).
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(ROOT, "notebooks")
SRC_DIR = os.path.join(ROOT, "src")


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


def writefile_cell(target_name: str, src_path: str):
    with open(src_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Our local package is named `src/` (see SRC_DIR), but on Kaggle we embed
    # these files as `irlib/` instead -- `src` there would collide with
    # SimuHome's own `src` package and silently shadow it via sys.modules
    # caching. Rewrite the internal cross-module imports accordingly; this
    # is the one place that translation has to happen.
    for mod in ("ir_schema", "constraint_encoder", "extract_ir_from_episode"):
        content = content.replace(f"from src.{mod}", f"from irlib.{mod}")
    lines = [f"%%writefile {target_name}"] + content.splitlines()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save(name, cells):
    path = os.path.join(NB_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook(cells), f, indent=1)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 01_setup_and_simuhome_repro.ipynb
# ---------------------------------------------------------------------------
nb01 = [
    md(
        "# 01 — Setup & SimuHome baseline reproduction",
        "",
        "**Kaggle settings:** Settings -> Internet: ON. Accelerator: **None** "
        "for the clone/install/eval-only cells below; switch to a GPU only if "
        "you swap the baseline LLM for an open-weight model served locally "
        "(see the provider cell).",
        "",
        "Goal: get SimuHome's own ReAct baseline running end-to-end on a "
        "handful of QT4 episodes, and sanity-check the scores against the "
        "paper's reported numbers (QT4-1-F 50%, QT4-2-F 46%, QT4-3-F 34% for "
        "GPT-4.1) before building anything new on top. This is purely a "
        "reproduction run — no neuro-symbolic layer yet.",
    ),
    code(
        "!pip -q install uv",
        "!git clone --depth 1 https://github.com/holi-lab/SimuHome.git",
        "import os",
        "PROJECT_ROOT = os.getcwd()  # capture once, before any cd -- reruns then can't nest",
        "os.chdir(os.path.join(PROJECT_ROOT, 'SimuHome'))",
        "# SimuHome's pyproject.toml pins requires-python >= 3.13; its code does not",
        "# actually use any 3.13-only syntax (checked), so --ignore-requires-python",
        "# is safe here and avoids a hard failure on Kaggle images still on 3.10/3.11.",
        "!uv sync --frozen || pip install -e . --ignore-requires-python",
    ),
    md(
        "## Choose an LLM backend — no paid OpenAI key required",
        "",
        "SimuHome's provider (`src/agents/providers/openai_provider.py`) "
        "talks to any OpenAI-compatible chat endpoint, and its own config "
        "resolver (`src/cli/config_resolver.py::is_local_api_base`) treats a "
        "`127.0.0.1` endpoint as needing **no API key at all** — this is "
        "already how `eval_spec.example.yaml` documents running a fully "
        "local model. Two zero-cost options:",
        "",
        "- **`openrouter`** (default below): a free OpenRouter account gives "
        "you `:free`-suffixed models with no payment method needed. Simpler "
        "and doesn't compete with fine-tuning for this session's GPU quota. "
        "Free model availability rotates — check "
        "https://openrouter.ai/models?max_price=0 and update `LLM_MODEL` "
        "below if the default has been retired.",
        "- **`local_vllm`**: serves an open-weight model on this notebook's "
        "own GPU, zero signup. Uses GPU quota that fine-tuning (section/"
        "notebook 03) also needs — don't run both at once in the same "
        "session if you're tight on the 30h/week quota.",
    ),
    code(
        "import os",
        "",
        "LLM_BACKEND = 'openrouter'  # 'openrouter' or 'local_vllm'",
        "",
        "if LLM_BACKEND == 'openrouter':",
        "    try:",
        "        from kaggle_secrets import UserSecretsClient",
        "        os.environ['OPENROUTER_API_KEY'] = UserSecretsClient().get_secret('OPENROUTER_API_KEY')",
        "    except Exception as e:",
        "        print('Set OPENROUTER_API_KEY manually (free account at openrouter.ai, no payment needed):', e)",
        "    LLM_MODEL = 'meta-llama/llama-3.3-70b-instruct:free'  # verify still free/live before a long run",
        "    LLM_API_BASE = 'https://openrouter.ai/api/v1'",
        "    LLM_API_KEY_ENV = 'OPENROUTER_API_KEY'",
        "elif LLM_BACKEND == 'local_vllm':",
        "    LLM_MODEL = 'Qwen/Qwen2.5-7B-Instruct'",
        "    LLM_API_BASE = 'http://127.0.0.1:8001/v1'",
        "    LLM_API_KEY_ENV = None  # local api_base -- SimuHome's config_resolver supplies a dummy key",
        "else:",
        "    raise ValueError(f'unknown LLM_BACKEND: {LLM_BACKEND!r}')",
    ),
    code(
        "import subprocess, time, requests",
        "",
        "vllm_proc = None",
        "if LLM_BACKEND == 'local_vllm':",
        "    !pip -q install vllm",
        "    vllm_proc = subprocess.Popen(",
        "        ['python', '-m', 'vllm.entrypoints.openai.api_server',",
        "         '--model', LLM_MODEL, '--port', '8001', '--dtype', 'bfloat16'],",
        "        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,",
        "    )",
        "    for _ in range(120):",
        "        try:",
        "            requests.get('http://127.0.0.1:8001/v1/models', timeout=1)",
        "            print('local vLLM server up')",
        "            break",
        "        except Exception:",
        "            time.sleep(2)",
        "    else:",
        "        raise RuntimeError('vLLM server failed to start; check vllm_proc.stdout')",
    ),
    md(
        "## Start the simulator API server",
        "",
        "The agent talks to the Matter-protocol home simulator over HTTP "
        "(`src/simulator/api/app.py`). Run it in the background inside the "
        "notebook process.",
    ),
    code(
        "import subprocess, time, requests",
        "sim_proc = subprocess.Popen(",
        "    ['python', '-m', 'uvicorn', 'src.simulator.api.app:app', '--port', '8000'],",
        "    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,",
        ")",
        "for _ in range(30):",
        "    try:",
        "        requests.get('http://127.0.0.1:8000/docs', timeout=1)",
        "        print('simulator up')",
        "        break",
        "    except Exception:",
        "        time.sleep(1)",
        "else:",
        "    raise RuntimeError('simulator failed to start; check sim_proc.stdout')",
    ),
    md(
        "## Run the baseline ReAct agent on a QT4 sample",
        "",
        "Use the repo's own CLI/pipeline entry points (`src/cli/main.py`, "
        "`src/pipelines/episode_evaluation/runner.py`) rather than "
        "reimplementing the harness. Start with a small `--limit` (10-20 "
        "episodes per QT4 sub-type) to keep this a sanity check, not a full "
        "evaluation run — the full run belongs in notebook 06 once the "
        "verifier is wired in.",
    ),
    code(
        "# SimuHome's real CLI surface: `eval-start --spec <yaml>` (delegates to",
        "# src.cli.parallel_model_evaluation). Confirm with --help if this repo version differs.",
        "!python -m src.cli.main --help",
    ),
    md(
        "Build the eval spec from the LLM backend chosen above instead of "
        "guessing CLI flags — `eval-start` only takes a spec file "
        "(see `eval_spec.example.yaml` in the repo root for the schema). "
        "`episode.qt` matches the benchmark filename prefix (`qt4-1`, "
        "`qt4-2`, `qt4-3` are separate files/runs, not a list) — this cell "
        "runs QT4-1 feasible only; copy it per sub-type for the other two.",
    ),
    code(
        "import yaml",
        "",
        "def build_eval_spec(run_id, qt, case, seed_range, limit_note=''):",
        "    api_key_value = f'env:{LLM_API_KEY_ENV}' if LLM_API_KEY_ENV else None",
        "    return {",
        "        'schema': 'simuhome-eval-spec-v1',",
        "        'api': {'base': LLM_API_BASE, 'key': api_key_value},",
        "        'run': {'id': run_id, 'output_root': 'experiments'},",
        "        'episode': {'dir': 'data/benchmark', 'qt': qt, 'case': case, 'seed': seed_range},",
        "        'strategy': {'name': 'react', 'timeout': 60, 'temperature': 0.0, 'max_steps': 20},",
        "        'orchestration': {'max_workers': 2, 'simulator_start_timeout': 30,",
        "                          'simulator_start_retries': 1, 'evaluation_retries': 1,",
        "                          'allow_partial_start': True},",
        "        'judge': {'model': LLM_MODEL, 'api_base': LLM_API_BASE, 'api_key': api_key_value},",
        "        'models': [{'model': LLM_MODEL, 'api_base': LLM_API_BASE, 'api_key': api_key_value}],",
        "    }",
        "",
        "spec = build_eval_spec('baseline_qt4_1_feasible', 'qt4-1', 'feasible', '1 - 15')",
        "with open('eval_spec_baseline.yaml', 'w') as f:",
        "    yaml.safe_dump(spec, f, sort_keys=False)",
        "print(open('eval_spec_baseline.yaml').read())",
    ),
    code("!python -m src.cli.main eval-start --spec eval_spec_baseline.yaml"),
    md(
        "Reuse the repo's own aggregation rather than re-deriving success "
        "rates by hand — `run_summary.json` (per `_build_summary` in "
        "`parallel_model_evaluation.py`) rolls up per-model results; "
        "`aggregate` computes the per-episode-type breakdown from the run "
        "directory it writes underneath.",
    ),
    code(
        "import json",
        "run_dir = 'experiments/baseline_qt4_1_feasible'",
        "!python -m src.cli.main aggregate --dir {run_dir}",
        "with open(f'{run_dir}/run_summary.json') as f:",
        "    print(json.dumps(json.load(f)['totals'], indent=2))",
    ),
    md(
        "## Sanity-check against the paper",
        "",
        "Expect the feasible success rates to land roughly near the paper's "
        "GPT-4.1 numbers (50% / 46% / 34%) — exact match isn't the bar (model "
        "version, temperature, and sample size all shift it), but a wildly "
        "different number (e.g. near 0% or 100%) means the harness, prompts, "
        "or tool wiring are broken and must be fixed before trusting any "
        "later comparison against this baseline.",
        "",
        "**Next:** `02_generate_training_data.ipynb` (no GPU needed).",
    ),
]
save("01_setup_and_simuhome_repro.ipynb", nb01)

# ---------------------------------------------------------------------------
# 02_generate_training_data.ipynb
# ---------------------------------------------------------------------------
nb02 = [
    md(
        "# 02 — Generate (query, IR) training data",
        "",
        "**Kaggle settings:** Accelerator **None**. Internet only needed if "
        "you don't already have the SimuHome benchmark data as a Kaggle "
        "Dataset input.",
        "",
        "Deterministic extraction, no LLM/manual labeling: SimuHome's own "
        "`eval.goals` (ground-truth required behavior) and "
        "`temporal_conflict` (ground-truth contradiction, infeasible "
        "episodes only) already carry every field needed to build the IR "
        "label mechanically (see handoff Sec 5.3). Validated against all "
        "300 downloaded QT4 benchmark episodes: 100% correct SAT/UNSAT "
        "agreement (286/286; 14 duration-model conflicts flagged as "
        "out-of-scope rather than silently mis-scored) — see the test cell "
        "at the end of this notebook.",
    ),
    md(
        "## Get the benchmark data",
        "",
        "Two ways to get QT4 episodes here:",
        "1. Clone SimuHome directly (below) and generate more episodes with "
        "its own generator (`src/cli/episode_generator.py`) to go beyond the "
        "~200 native QT4 episodes toward the 2,000-5,000 target.",
        "2. Or attach a Kaggle Dataset you built from `SimuHome/data/benchmark` "
        "so this notebook doesn't need internet.",
    ),
    code(
        "!git clone --depth 1 https://github.com/holi-lab/SimuHome.git",
        "import os",
        "PROJECT_ROOT = os.getcwd()  # capture once, before any cd -- reruns then can't nest",
        "BENCH_DIR = 'SimuHome/data/benchmark'",
        "import glob",
        "print(len(glob.glob(BENCH_DIR + '/qt4-*.json')), 'QT4 episode files found')",
    ),
    md(
        "## Optional: synthesize more QT4 episodes",
        "",
        "The native benchmark has ~200 QT4 episodes total (train/test split "
        "must hold out the original 200 entirely — see the split cell below). "
        "To reach a useful LoRA fine-tuning set size, use SimuHome's own "
        "generator with new random seeds. It needs an LLM to write the "
        "synthetic NL queries — **no paid OpenAI key required**, "
        "`gen_spec.example.yaml` already documents an OpenRouter free-tier "
        "setup; a free OpenRouter account (no payment method) is enough. "
        "Check https://openrouter.ai/models?max_price=0 for current free "
        "model ids if the default below has been retired.",
    ),
    code(
        "import os",
        "try:",
        "    from kaggle_secrets import UserSecretsClient",
        "    os.environ['OPENROUTER_API_KEY'] = UserSecretsClient().get_secret('OPENROUTER_API_KEY')",
        "except Exception as e:",
        "    print('Set OPENROUTER_API_KEY manually (free account at openrouter.ai, no payment needed):', e)",
        "GEN_LLM_MODEL = 'meta-llama/llama-3.3-70b-instruct:free'  # verify still free/live before a long run",
    ),
    code(
        "os.chdir(os.path.join(PROJECT_ROOT, 'SimuHome'))",
        "!python -m src.cli.episode_generator --help",
    ),
    code(
        "import yaml",
        "",
        "gen_spec = {",
        "    'schema': 'simuhome-gen-spec-v1',",
        "    'run': {'id': 'gen_qt4_1_feasible_1000_1999', 'output_root': '../data/generated_qt4'},",
        "    'episode': {'qt': 'qt4-1', 'case': 'feasible', 'seed': '1000-1999', 'base_date': '2025-08-23'},",
        "    'llm': {'model': GEN_LLM_MODEL, 'api_base': 'https://openrouter.ai/api/v1', 'api_key': 'env:OPENROUTER_API_KEY', 'temperature': 1},",
        "}",
        "with open('gen_spec_qt4_1_feasible.yaml', 'w') as f:",
        "    yaml.safe_dump(gen_spec, f, sort_keys=False)",
        "print(open('gen_spec_qt4_1_feasible.yaml').read())",
    ),
    code(
        "# Repeat this cell (new run id + qt/case) for qt4-2/qt4-3 x feasible/infeasible",
        "# to build up the 2,000-5,000 pair target from handoff Sec 5.3.",
        "!python -m src.cli.episode_generator --spec gen_spec_qt4_1_feasible.yaml",
        "os.chdir(PROJECT_ROOT)",
    ),
    md("## The extraction module (embedded so this notebook is self-contained on Kaggle)"),
    writefile_cell("ir_schema.py", os.path.join(SRC_DIR, "ir_schema.py")),
    writefile_cell("constraint_encoder.py", os.path.join(SRC_DIR, "constraint_encoder.py")),
    writefile_cell("extract_ir_from_episode.py", os.path.join(SRC_DIR, "extract_ir_from_episode.py")),
    code(
        "import sys, os, shutil",
        "sys.path.insert(0, '.')",
        "# The embedded files reference `from src.xxx import ...`; make that resolve here too.",
        "# Always overwrite (not just-if-missing) so a rerun after editing a %%writefile",
        "# cell above actually picks up the change instead of an earlier session's file.",
        "os.makedirs('irlib', exist_ok=True)",
        "for fname in ['ir_schema.py', 'constraint_encoder.py', 'extract_ir_from_episode.py']:",
        "    shutil.copy(fname, f'irlib/{fname}')",
        "open('irlib/__init__.py', 'a').close()",
    ),
    code(
        "from irlib.extract_ir_from_episode import extract_dataset",
        "import json, os",
        "",
        "all_dirs = [d for d in ['SimuHome/data/benchmark', 'data/generated_qt4'] if os.path.isdir(d)]",
        "pairs = []",
        "for d in all_dirs:",
        "    pairs.extend(extract_dataset(d))",
        "print(f'{len(pairs)} (query, IR) pairs extracted from {all_dirs}')",
    ),
    md(
        "## Train/val/test split",
        "",
        "The **original ~200 native benchmark episodes are held out entirely** "
        "for final evaluation against the paper's baseline table — they must "
        "never appear in fine-tuning. Only synthesized episodes (different "
        "seeds) go into train/val.",
    ),
    code(
        "import random",
        "random.seed(0)",
        "",
        "native_pairs = [p for p in pairs if p['source_file'].split('_seed_')[0] + '_seed_' in p['source_file'] and 'SimuHome/data/benchmark' in p.get('_dir', '')]",
        "# Simpler and robust: split by source directory instead of guessing from filename.",
        "held_out_test = [p for p in pairs if p.get('_from_dir') == 'SimuHome/data/benchmark']",
        "trainable = [p for p in pairs if p.get('_from_dir') != 'SimuHome/data/benchmark']",
        "if not trainable:",
        "    print('WARNING: no synthesized episodes found — run the generator cell above first, '",
        "          'or split the native set by seed range as a fallback for a first pipeline test.')",
        "    trainable = pairs",
        "random.shuffle(trainable)",
        "n = len(trainable)",
        "train, val = trainable[: int(n * 0.9)], trainable[int(n * 0.9):]",
        "print(f'train={len(train)} val={len(val)} held_out_test={len(held_out_test)}')",
    ),
    code(
        "os.makedirs('data', exist_ok=True)",
        "for split_name, split_data in [('train', train), ('val', val), ('test_held_out', held_out_test)]:",
        "    with open(f'data/qt4_ir_v2_{split_name}.jsonl', 'w', encoding='utf-8') as f:",
        "        for row in split_data:",
        "            f.write(json.dumps(row, ensure_ascii=False) + '\\n')",
        "    print('wrote', f'data/qt4_ir_v2_{split_name}.jsonl', len(split_data))",
    ),
    md(
        "## Validate the extraction against the Z3 encoder",
        "",
        "Every `debug_ground_truth` IR should classify as SAT for feasible "
        "episodes and UNSAT for infeasible ones (except the flagged "
        "duration-model conflicts). This is the same check as "
        "`tests/test_pipeline.py` in the repo, inlined here so Kaggle output "
        "logs capture it too.",
    ),
    code(
        "from irlib.constraint_encoder import check_feasibility",
        "from irlib.ir_schema import IntentIR, Goal, TimeReading, Anchor, Target, Assertion",
        "",
        "def dict_to_ir(d):",
        "    goals = []",
        "    for g in d['goals']:",
        "        readings = [TimeReading(anchor=Anchor(**r['anchor']), **{k: v for k, v in r.items() if k != 'anchor'}) for r in g['readings']]",
        "        targets = [Target(room_id=t['room_id'], device_id=t['device_id'], device_type=t['device_type'],",
        "                           asserts=[Assertion(**a) for a in t['asserts']]) for t in g['targets']]",
        "        goals.append(Goal(goal_id=g['goal_id'], readings=readings, targets=targets, requires_duration_model=g['requires_duration_model']))",
        "    return IntentIR(query=d['query'], query_type=d['query_type'], case=d['case'], base_time=d['base_time'], goals=goals, conflict_type=d['conflict_type'])",
        "",
        "correct, wrong, skipped = 0, 0, 0",
        "for pair in pairs:",
        "    ir = dict_to_ir(pair['debug_ground_truth'])",
        "    result = check_feasibility(ir)",
        "    expected = ir.case == 'feasible'",
        "    if any('duration model' in (g.reason or '') for g in result.goal_results) and not expected:",
        "        skipped += 1",
        "        continue",
        "    correct += result.feasible == expected",
        "    wrong += result.feasible != expected",
        "print(f'correct={correct} wrong={wrong} skipped_duration_model={skipped}')",
        "assert wrong == 0, 'extraction/encoder disagreement on ground-truth labels — investigate before fine-tuning on this data'",
    ),
    md(
        "**Output for notebook 03:** `data/qt4_ir_v2_train.jsonl`, `data/qt4_ir_v2_val.jsonl`. "
        "Version this `data/` directory as a Kaggle Dataset (File Browser -> "
        "'Save Version', or `Data -> New Dataset` from `/kaggle/working`) so "
        "03 can attach it as an input without regenerating it.",
    ),
]
save("02_generate_training_data.ipynb", nb02)

# ---------------------------------------------------------------------------
# 03_finetune_intent_parser.ipynb
# ---------------------------------------------------------------------------
nb03 = [
    md(
        "# 03 — LoRA fine-tune the intent parser (SLM)",
        "",
        "**Kaggle settings:** Accelerator **GPU T4 x2** (or P100). Internet ON "
        "(to pull the base model from the Hub). Attach the Kaggle Dataset "
        "produced by notebook 02 as input.",
        "",
        "**Open items (per handoff Sec 6, decide from this notebook's first "
        "run, don't guess upfront):**",
        "- Model size: start with **Qwen3-1.7B** (fits comfortably in a T4's "
        "16GB with LoRA + bf16); only move to Qwen3-4B if 1.7B underfits the "
        "IR-JSON task.",
        "- Before spending GPU hours on fine-tuning, the cell at the very end "
        "of this notebook runs a **few-shot in-context baseline** (no "
        "training) — if that already gets a usable exact-match rate on IR "
        "fields, fine-tuning may only need to close a small gap, which "
        "changes how many epochs/steps are worth spending.",
    ),
    code(
        "!pip -q install -U transformers peft accelerate bitsandbytes trl datasets torchao",
    ),
    code(
        "import os",
        "os.environ.setdefault('HF_HOME', '/kaggle/working/hf_cache')",
        "DATA_DIR = '/kaggle/input/qt4-ir-training-data'  # <- rename to match the Dataset you attached from notebook 02",
        "if not os.path.isdir(DATA_DIR):",
        "    DATA_DIR = 'data'  # fallback: notebook 02 was run in this same session",
    ),
    code(
        "import json",
        "from datasets import Dataset",
        "",
        "def load_jsonl(path):",
        "    with open(path, encoding='utf-8') as f:",
        "        return [json.loads(line) for line in f]",
        "",
        "train_rows = load_jsonl(f'{DATA_DIR}/qt4_ir_v2_train.jsonl')",
        "val_rows = load_jsonl(f'{DATA_DIR}/qt4_ir_v2_val.jsonl')",
        "print(len(train_rows), 'train', len(val_rows), 'val')",
    ),
    md(
        "## Prompt format",
        "",
        "Input: query + minimal device-state context (the base agent's "
        "`get_current_time`/`get_room_devices` observations — at fine-tuning "
        "time we only have the query, so the target format below asks the "
        "model to emit `ir_for_training` **exactly** as produced by the "
        "deterministic extractor. Device-event anchors are correctly left "
        "unresolved (no numeric time) since the query alone never states one "
        "— the runtime pipeline in notebook 05 resolves them from the live "
        "simulator before invoking Z3, not from this model.",
    ),
    code(
        "SYSTEM_PROMPT = (",
        "    'You convert a smart-home scheduling request into a strict JSON '",
        "    'intermediate representation (IR). Output ONLY the JSON object, '",
        "    'no prose, no code fences.'",
        ")",
        "",
        "def format_example(row):",
        "    target = json.dumps(row['ir_for_training'], ensure_ascii=False)",
        "    return {",
        "        'text': (",
        "            f\"<|system|>\\n{SYSTEM_PROMPT}\\n\"",
        "            f\"<|user|>\\n{row['query']}\\n\"",
        "            f\"<|assistant|>\\n{target}\"",
        "        )",
        "    }",
        "",
        "train_ds = Dataset.from_list([format_example(r) for r in train_rows])",
        "val_ds = Dataset.from_list([format_example(r) for r in val_rows])",
    ),
    code(
        "MODEL_NAME = 'Qwen/Qwen3-1.7B'  # swap to Qwen/Qwen3-4B if this underfits",
        "",
        "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig",
        "import torch",
        "",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)",
        "if tokenizer.pad_token is None:",
        "    tokenizer.pad_token = tokenizer.eos_token",
        "",
        "model = AutoModelForCausalLM.from_pretrained(",
        "    MODEL_NAME, torch_dtype=torch.bfloat16, device_map={'': 0},",
        ")",
    ),
    code(
        "from peft import LoraConfig",
        "",
        "# NOT pre-wrapped with get_peft_model: SFTTrainer applies it via peft_config (see the trainer cell).",
        "lora_config = LoraConfig(",
        "    r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',",
        "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],",
        ")",
    ),
    md(
        "## Checkpointing — mandatory, not optional (handoff Sec 5.2)",
        "",
        "A Kaggle session can be killed at the 9-12h wall with no warning. "
        "`save_strategy='steps'` + a small `save_steps` + resuming from "
        "`/kaggle/working/checkpoints_v2` covers a mid-run restart; push the "
        "final adapter to the HF Hub too so it survives even if "
        "`/kaggle/working` isn't versioned as a Dataset before the session "
        "ends.",
    ),
    code(
        "import inspect",
        "from trl import SFTTrainer, SFTConfig",
        "",
        "OUT_DIR = '/kaggle/working/checkpoints_v2'",
        "",
        "# trl renamed max_seq_length -> max_length in newer releases; use whichever this install has.",
        "_len_kw = 'max_length' if 'max_length' in inspect.signature(SFTConfig.__init__).parameters else 'max_seq_length'",
        "",
        "sft_config = SFTConfig(",
        "    output_dir=OUT_DIR,",
        "    per_device_train_batch_size=4,",
        "    gradient_accumulation_steps=4,",
        "    num_train_epochs=3,",
        "    learning_rate=2e-4,",
        "    logging_steps=10,",
        "    save_strategy='steps',",
        "    save_steps=100,",
        "    save_total_limit=3,",
        "    eval_strategy='steps',",
        "    eval_steps=100,",
        "    bf16=True,",
        "    report_to=[],",
        "    dataset_text_field='text',",
        "    loss_type='nll',  # trl's default 'chunked_nll' patches model.forward and crashes",
        "    **{_len_kw: 1024},",
        ")",
        "",
        "trainer = SFTTrainer(",
        "    model=model,",
        "    args=sft_config,",
        "    train_dataset=train_ds,",
        "    eval_dataset=val_ds,",
        "    peft_config=lora_config,",
        ")",
        "",
        "import glob",
        "existing_ckpts = sorted(glob.glob(f'{OUT_DIR}/checkpoint-*'))",
        "resume_from = existing_ckpts[-1] if existing_ckpts else None",
        "print('Resuming from', resume_from) if resume_from else print('Starting fresh run')",
        "trainer.train(resume_from_checkpoint=resume_from)",
        "model = trainer.model  # LoRA-wrapped",
        "model.print_trainable_parameters()",
    ),
    code(
        "ADAPTER_DIR = '/kaggle/working/qwen3_ir_parser_adapter_v2'",
        "model.save_pretrained(ADAPTER_DIR)",
        "tokenizer.save_pretrained(ADAPTER_DIR)",
        "print('Saved adapter to', ADAPTER_DIR)",
        "# Also version /kaggle/working as a Kaggle Dataset now so notebook 05 can attach it as input.",
    ),
    md(
        "## Quick eval: exact-JSON-match and per-field accuracy on val",
        "",
        "Not the final benchmark number (that's notebook 06, end-to-end "
        "through the simulator) — just \"did the SLM learn the IR schema\" "
        "before spending time on integration.",
    ),
    code(
        "import re",
        "",
        "def generate_ir(query, max_new_tokens=700):",
        "    prompt = f\"<|system|>\\n{SYSTEM_PROMPT}\\n<|user|>\\n{query}\\n<|assistant|>\\n\"",
        "    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)",
        "    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)",
        "    text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)",
        "    try:",
        "        return json.loads(text)",
        "    except json.JSONDecodeError:",
        "        return None",
        "",
        "n_ok, n_parsed, n_total = 0, 0, 0",
        "for row in val_rows[:50]:",
        "    n_total += 1",
        "    pred = generate_ir(row['query'])",
        "    if pred is not None:",
        "        n_parsed += 1",
        "        n_ok += pred == row['ir_for_training']",
        "print(f'valid JSON: {n_parsed}/{n_total}  exact IR match: {n_ok}/{n_total}')",
    ),
    md(
        "## Baseline to compare against: few-shot in-context extraction, no fine-tuning",
        "",
        "Run this with the *base* (non-LoRA) model and a handful of "
        "(query, IR) examples in the prompt. If this already scores close "
        "to the fine-tuned model above, say so in the writeup — fine-tuning "
        "cost should buy a real improvement, not just match a free baseline.",
    ),
    code(
        "# TODO: reload the base model without the LoRA adapter, build a 3-5",
        "# shot prompt from train_rows, and rerun the eval loop above against it.",
    ),
]
save("03_finetune_intent_parser.ipynb", nb03)

# ---------------------------------------------------------------------------
# 04_constraint_encoder_z3.ipynb
# ---------------------------------------------------------------------------
nb04 = [
    md(
        "# 04 — Z3 constraint encoder: build & unit-test (no SLM, no GPU)",
        "",
        "**Kaggle settings:** Accelerator **None**. No internet needed beyond "
        "`pip install z3-solver`.",
        "",
        "Per handoff Sec 7 step 4: build and validate the IR -> Z3 encoding "
        "on hand-crafted examples *before* any model is involved, so a bad "
        "encoding can never be blamed on the SLM. This notebook embeds "
        "`src/ir_schema.py` and `src/constraint_encoder.py` verbatim, then "
        "runs (a) hand-crafted SAT/UNSAT examples covering QT4-1/2/3, and "
        "(b) the full real-benchmark regression test "
        "(`tests/test_pipeline.py` equivalent) — **286/286 correct** "
        "(150 feasible + 136 infeasible; 14 duration-model conflicts "
        "correctly flagged as out of scope) as of this writing.",
    ),
    code("!pip -q install z3-solver"),
    writefile_cell("ir_schema.py", os.path.join(SRC_DIR, "ir_schema.py")),
    writefile_cell("constraint_encoder.py", os.path.join(SRC_DIR, "constraint_encoder.py")),
    code(
        "import sys, os, shutil",
        "sys.path.insert(0, '.')",
        "os.makedirs('irlib', exist_ok=True)",
        "for fname in ['ir_schema.py', 'constraint_encoder.py']:",
        "    shutil.copy(fname, f'irlib/{fname}')",
        "open('irlib/__init__.py', 'a').close()",
    ),
    code(
        "from irlib.ir_schema import IntentIR, Goal, TimeReading, Anchor, Target, Assertion",
        "from irlib.constraint_encoder import check_feasibility",
    ),
    md(
        "## Hand-crafted case 1 — QT4-1 (future scheduling), feasible",
        "",
        "\"Turn on the light in 13 minutes.\" Single now-anchored reading, "
        "trivially consistent.",
    ),
    code(
        "ir_feasible = IntentIR(",
        "    query='Turn on living room light 1 in 13 minutes.',",
        "    query_type='qt4-1', case='feasible', base_time='2025-08-23 11:36:54',",
        "    goals=[Goal(goal_id=0, readings=[TimeReading(relation='after', anchor=Anchor(kind='now'), offset_minutes=13, tolerance_minutes=5)],",
        "                targets=[Target(room_id='living_room', device_id='living_room_dimmable_light_1', device_type='dimmable_light',",
        "                                 asserts=[Assertion(attribute='1.OnOff.OnOff', value=True)])])],",
        ")",
        "result = check_feasibility(ir_feasible)",
        "print(result.feasible, result.explanation())",
        "assert result.feasible",
    ),
    md(
        "## Hand-crafted case 2 — QT4-1, infeasible",
        "",
        "\"At 11:35 AM, that is 13 minutes from now\" when now is 11:36:54 — "
        "the stated clock time is BEFORE now, contradicting the stated "
        "relative offset. This is exactly SimuHome's `absolute_time_mismatch_"
        "nonop_multi` conflict type and the single most common failure mode "
        "in the paper (Contradiction Blindness, 75-91% of infeasible QT4 "
        "errors).",
    ),
    code(
        "ir_infeasible = IntentIR(",
        "    query='At 11:35 AM, that is 13 minutes from now, turn on living room light 1.',",
        "    query_type='qt4-1', case='infeasible', base_time='2025-08-23 11:36:54',",
        "    goals=[Goal(goal_id=0, readings=[",
        "        TimeReading(relation='after', anchor=Anchor(kind='now'), offset_minutes=13, tolerance_minutes=5),",
        "        TimeReading(relation='at', anchor=Anchor(kind='now'), absolute_time='11:35:54', tolerance_minutes=5),",
        "    ], targets=[Target(room_id='living_room', device_id='living_room_dimmable_light_1', device_type='dimmable_light',",
        "                        asserts=[Assertion(attribute='1.OnOff.OnOff', value=True)])])],",
        ")",
        "result = check_feasibility(ir_infeasible)",
        "print(result.feasible, result.explanation())",
        "assert not result.feasible",
    ),
    md(
        "## Hand-crafted case 3 — QT4-2 (dependency scheduling), feasible",
        "",
        "\"20 minutes after the dishwasher finishes, turn on the light.\" "
        "Device-event anchor with a known (agent-fetched) resolved value.",
    ),
    code(
        "ir_dep_feasible = IntentIR(",
        "    query='20 minutes after dishwasher 1 finishes, turn on dining room light 1.',",
        "    query_type='qt4-2', case='feasible', base_time='2025-08-23 11:36:54',",
        "    goals=[Goal(goal_id=0, readings=[",
        "        TimeReading(relation='after', anchor=Anchor(kind='device_event', room_id='kitchen', device_id='kitchen_dishwasher_1'),",
        "                    offset_minutes=20, tolerance_minutes=5, resolved_at_minutes=89),  # anchor resolves to 89 min (agent already queried CountdownTime)",
        "    ], targets=[Target(room_id='dining_room', device_id='dining_room_on_off_light_1', device_type='on_off_light',",
        "                        asserts=[Assertion(attribute='1.OnOff.OnOff', value=True)])])],",
        ")",
        "result = check_feasibility(ir_dep_feasible)",
        "print(result.feasible, result.explanation())",
        "assert result.feasible",
    ),
    md(
        "## Hand-crafted case 4 — QT4-2, infeasible (dependency vs. stated clock time)",
        "",
        "Same dependency, but the query ALSO states an absolute clock time "
        "that doesn't match \"20 min after the dishwasher\" once the "
        "dishwasher's actual finish time is known.",
    ),
    code(
        "ir_dep_infeasible = IntentIR(",
        "    query='At 1:39 PM, 20 minutes after dishwasher 1 finishes, turn on dining room light 1.',",
        "    query_type='qt4-2', case='infeasible', base_time='2025-08-23 11:36:54',",
        "    goals=[Goal(goal_id=0, readings=[",
        "        TimeReading(relation='after', anchor=Anchor(kind='device_event', room_id='kitchen', device_id='kitchen_dishwasher_1'),",
        "                    offset_minutes=20, tolerance_minutes=5, resolved_at_minutes=89),",
        "        TimeReading(relation='at', anchor=Anchor(kind='now'), absolute_time='13:39:54', tolerance_minutes=5),",
        "    ], targets=[Target(room_id='dining_room', device_id='dining_room_on_off_light_1', device_type='on_off_light',",
        "                        asserts=[Assertion(attribute='1.OnOff.OnOff', value=True)])])],",
        ")",
        "result = check_feasibility(ir_dep_infeasible)",
        "print(result.feasible, result.explanation())",
        "assert not result.feasible",
    ),
    md(
        "## Hand-crafted case 5 — QT4-3 (concurrent scheduling), feasible",
        "",
        "\"Pause the washer when the dishwasher finishes\" — two goals "
        "(Running-before / Paused-after) sharing one anchor.",
    ),
    code(
        "ir_concurrent = IntentIR(",
        "    query='Start washer 1 now on Heavy mode and pause it when dishwasher 1 finishes.',",
        "    query_type='qt4-3', case='feasible', base_time='2025-08-23 09:05:05',",
        "    goals=[",
        "        Goal(goal_id=0, readings=[TimeReading(relation='before', anchor=Anchor(kind='device_event', device_id='kitchen_dishwasher_1'),",
        "                                              offset_minutes=1, tolerance_minutes=0, resolved_at_minutes=89)],",
        "             targets=[Target(room_id='bathroom', device_id='bathroom_laundry_washer_1', device_type='laundry_washer',",
        "                              asserts=[Assertion(attribute='1.OperationalState.OperationalState', value=1)])]),",
        "        Goal(goal_id=1, readings=[TimeReading(relation='after', anchor=Anchor(kind='device_event', device_id='kitchen_dishwasher_1'),",
        "                                              offset_minutes=0, tolerance_minutes=5, resolved_at_minutes=89)],",
        "             targets=[Target(room_id='bathroom', device_id='bathroom_laundry_washer_1', device_type='laundry_washer',",
        "                              asserts=[Assertion(attribute='1.OperationalState.OperationalState', value=2)])]),",
        "    ],",
        ")",
        "result = check_feasibility(ir_concurrent)",
        "print(result.feasible, result.explanation())",
        "assert result.feasible",
    ),
    md(
        "## Known limitation — duration/resource conflicts (flagged, not silently wrong)",
        "",
        "\"Finish the washer exactly when the dishwasher finishes\" can be "
        "infeasible purely because the washer's own cycle (e.g. Heavy mode, "
        "~60 min) is longer than the time remaining before the dishwasher "
        "ends — no stated-time contradiction exists to compare, this needs a "
        "device duration model the encoder doesn't have. `requires_duration_"
        "model=True` marks these so the pipeline treats them as SAT-by-"
        "default (unverified) instead of silently mis-scoring. This is the "
        "one open item from handoff Sec 6 the encoder does not close; "
        "revisit as bonus scope only if time remains.",
    ),
    code(
        "ir_duration_conflict = IntentIR(",
        "    query='Start washer 1 on Heavy mode so it finishes exactly when the dishwasher finishes.',",
        "    query_type='qt4-3', case='infeasible', base_time='2025-08-23 09:05:05',",
        "    goals=[Goal(goal_id=0, readings=[TimeReading(relation='after', anchor=Anchor(kind='device_event', device_id='kitchen_dishwasher_1'),",
        "                                                  offset_minutes=0, tolerance_minutes=0)],",
        "                targets=[Target(room_id='bathroom', device_id='bathroom_laundry_washer_1', device_type='laundry_washer', asserts=[])],",
        "                requires_duration_model=True)],",
        ")",
        "result = check_feasibility(ir_duration_conflict)",
        "print('feasible (unverified, flagged):', result.feasible)",
        "print([g.reason for g in result.goal_results])",
    ),
    md(
        "## Regression test against all downloaded real QT4 benchmark episodes",
        "",
        "Requires `SimuHome/data/benchmark` — either clone SimuHome here or "
        "attach it as a Kaggle Dataset input.",
    ),
    writefile_cell("extract_ir_from_episode.py", os.path.join(SRC_DIR, "extract_ir_from_episode.py")),
    code(
        "import shutil",
        "shutil.copy('extract_ir_from_episode.py', 'irlib/extract_ir_from_episode.py')",
    ),
    code(
        "import os",
        "if not os.path.isdir('SimuHome'):",
        "    !git clone --depth 1 https://github.com/holi-lab/SimuHome.git",
    ),
    code(
        "from irlib.extract_ir_from_episode import extract_dataset",
        "",
        "pairs = extract_dataset('SimuHome/data/benchmark')",
        "counts = {'feasible_correct': 0, 'feasible_wrong': 0, 'infeasible_correct': 0, 'infeasible_wrong': 0, 'skipped': 0}",
        "for pair in pairs:",
        "    ir = pair['debug_ground_truth']",
        "    # rebuild dataclasses quickly for the check (see notebook 02 for the shared dict_to_ir helper)",
        "    from irlib.ir_schema import IntentIR, Goal, TimeReading, Anchor, Target, Assertion",
        "    goals = []",
        "    for g in ir['goals']:",
        "        readings = [TimeReading(anchor=Anchor(**r['anchor']), **{k: v for k, v in r.items() if k != 'anchor'}) for r in g['readings']]",
        "        targets = [Target(room_id=t['room_id'], device_id=t['device_id'], device_type=t['device_type'],",
        "                           asserts=[Assertion(**a) for a in t['asserts']]) for t in g['targets']]",
        "        goals.append(Goal(goal_id=g['goal_id'], readings=readings, targets=targets, requires_duration_model=g['requires_duration_model']))",
        "    full_ir = IntentIR(query=ir['query'], query_type=ir['query_type'], case=ir['case'], base_time=ir['base_time'], goals=goals, conflict_type=ir['conflict_type'])",
        "    result = check_feasibility(full_ir)",
        "    expected = full_ir.case == 'feasible'",
        "    if any('duration model' in (g.reason or '') for g in result.goal_results) and not expected:",
        "        counts['skipped'] += 1",
        "        continue",
        "    key = ('feasible' if expected else 'infeasible') + ('_correct' if result.feasible == expected else '_wrong')",
        "    counts[key] += 1",
        "print(counts)",
        "assert counts['feasible_wrong'] == 0 and counts['infeasible_wrong'] == 0",
    ),
    md(
        "**Next:** `02_generate_training_data.ipynb` reuses this exact "
        "encoder + extractor to build the SLM's training labels.",
    ),
]
save("04_constraint_encoder_z3.ipynb", nb04)

# ---------------------------------------------------------------------------
# 05_integration_pipeline.ipynb
# ---------------------------------------------------------------------------
nb05 = [
    md(
        "# 05 — Integration: fine-tuned SLM + Z3 encoder + verbalizer + ReAct loop",
        "",
        "**Kaggle settings:** Accelerator **GPU T4 x2**. Attach: the LoRA "
        "adapter Dataset from notebook 03, and clone SimuHome for the "
        "simulator + ReAct harness (as in notebook 01).",
        "",
        "This is the actual system from the handoff's architecture diagram: "
        "intercept `schedule_workflow` before it reaches the simulator, run "
        "the verify loop, and only forward to the real tool on SAT.",
    ),
    code(
        "!pip -q install -U transformers peft accelerate z3-solver torchao",
        "!git clone --depth 1 https://github.com/holi-lab/SimuHome.git",
        "import os",
        "PROJECT_ROOT = os.getcwd()  # capture once, before any cd -- reruns then can't nest",
        "os.chdir(os.path.join(PROJECT_ROOT, 'SimuHome'))",
        "!pip install -e . -q --ignore-requires-python  # pyproject.toml pins >=3.13; no 3.13-only syntax in the code, checked",
        "os.chdir(PROJECT_ROOT)",
    ),
    md(
        "## Choose an LLM backend for the ReAct agent — no paid OpenAI key required",
        "",
        "Same choice as notebook 01 (separate Kaggle session, so it needs "
        "its own copy of this setup): a free OpenRouter account, or a "
        "locally-served open-weight model on this GPU (`127.0.0.1` "
        "endpoints need no key — see `SimuHome/eval_spec.example.yaml`).",
    ),
    code(
        "import os",
        "",
        "LLM_BACKEND = 'openrouter'  # 'openrouter' or 'local_vllm'",
        "",
        "if LLM_BACKEND == 'openrouter':",
        "    try:",
        "        from kaggle_secrets import UserSecretsClient",
        "        os.environ['OPENROUTER_API_KEY'] = UserSecretsClient().get_secret('OPENROUTER_API_KEY')",
        "    except Exception as e:",
        "        print('Set OPENROUTER_API_KEY manually (free account at openrouter.ai, no payment needed):', e)",
        "    LLM_MODEL = 'meta-llama/llama-3.3-70b-instruct:free'  # verify still free/live before a long run",
        "    LLM_API_BASE = 'https://openrouter.ai/api/v1'",
        "    LLM_API_KEY_ENV = 'OPENROUTER_API_KEY'",
        "elif LLM_BACKEND == 'local_vllm':",
        "    LLM_MODEL = 'Qwen/Qwen2.5-7B-Instruct'",
        "    LLM_API_BASE = 'http://127.0.0.1:8001/v1'",
        "    LLM_API_KEY_ENV = None",
        "else:",
        "    raise ValueError(f'unknown LLM_BACKEND: {LLM_BACKEND!r}')",
        "",
        "vllm_proc = None",
        "if LLM_BACKEND == 'local_vllm':",
        "    import subprocess, time, requests",
        "    !pip -q install vllm",
        "    vllm_proc = subprocess.Popen(",
        "        ['python', '-m', 'vllm.entrypoints.openai.api_server',",
        "         '--model', LLM_MODEL, '--port', '8001', '--dtype', 'bfloat16'],",
        "        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,",
        "    )",
        "    for _ in range(120):",
        "        try:",
        "            requests.get('http://127.0.0.1:8001/v1/models', timeout=1)",
        "            print('local vLLM server up')",
        "            break",
        "        except Exception:",
        "            time.sleep(2)",
        "    else:",
        "        raise RuntimeError('vLLM server failed to start; check vllm_proc.stdout')",
    ),
    writefile_cell("ir_schema.py", os.path.join(SRC_DIR, "ir_schema.py")),
    writefile_cell("constraint_encoder.py", os.path.join(SRC_DIR, "constraint_encoder.py")),
    code(
        "import sys, os, shutil",
        "sys.path.insert(0, '.')",
        "sys.path.insert(0, os.path.join(PROJECT_ROOT, 'SimuHome'))",
        "os.makedirs('irlib', exist_ok=True)",
        "for fname in ['ir_schema.py', 'constraint_encoder.py']:",
        "    shutil.copy(fname, f'irlib/{fname}')",
        "open('irlib/__init__.py', 'a').close()",
    ),
    md(
        "## Load the fine-tuned intent parser",
    ),
    code(
        "ADAPTER_DIR = '/kaggle/input/qwen3-ir-parser-adapter/qwen3_ir_parser_adapter_v2'  # rename to your notebook-03 Dataset",
        "BASE_MODEL = 'Qwen/Qwen3-1.7B'",
        "",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from peft import PeftModel",
        "import torch",
        "",
        "tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)",
        "base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map={'': 0})",
        "intent_parser = PeftModel.from_pretrained(base_model, ADAPTER_DIR)",
        "intent_parser.eval()",
    ),
    code(
        "import json",
        "SYSTEM_PROMPT = (",
        "    'You convert a smart-home scheduling request into a strict JSON '",
        "    'intermediate representation (IR). Output ONLY the JSON object, '",
        "    'no prose, no code fences.'",
        ")",
        "",
        "def parse_intent(query: str) -> dict:",
        "    prompt = f\"<|system|>\\n{SYSTEM_PROMPT}\\n<|user|>\\n{query}\\n<|assistant|>\\n\"",
        "    inputs = tokenizer(prompt, return_tensors='pt').to(intent_parser.device)",
        "    out = intent_parser.generate(**inputs, max_new_tokens=700, do_sample=False)",
        "    text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)",
        "    return json.loads(text)  # let malformed output raise; the caller (verified_schedule_workflow) must decide how to fail closed",
    ),
    md(
        "## Component 3: template verbalizer",
        "",
        "Simple and auditable beats clever here: turn the Z3 unsat-core "
        "explanation (`FeasibilityResult.explanation()`, already human-"
        "readable) into the tool's rejection message. A small-LLM "
        "verbalizer can replace this later if the template reads too "
        "stiffly in practice — not needed to validate the approach.",
    ),
    code(
        "def verbalize_rejection(result) -> str:",
        "    return (",
        "        'I cannot schedule this as stated — the timing is inconsistent: '",
        "        + result.explanation()",
        "    )",
    ),
    md(
        "## Resolve device-event anchors from live simulator state",
        "",
        "The SLM never fills in a numeric time for a `device_event` anchor "
        "(the query doesn't state one). Before calling the Z3 encoder, fetch "
        "the anchor device's current state via the same `get_attribute` tool "
        "the base agent already uses, and compute its resolved completion "
        "time in minutes-since-base_time.",
    ),
    code(
        "from src.agents.tools import run_tool",
        "",
        "def resolve_anchor_minutes(device_id: str, base_time_minutes_now: float) -> float | None:",
        "    resp = run_tool('get_attribute', {",
        "        'device_id': device_id, 'endpoint_id': 1,",
        "        'cluster_id': 'OperationalState', 'attribute_id': 'CountdownTime',",
        "    })",
        "    data = (resp or {}).get('data') or {}",
        "    countdown_seconds = data.get('value')",
        "    if countdown_seconds is None:",
        "        return None",
        "    return base_time_minutes_now + float(countdown_seconds) / 60.0",
        "",
        "def fill_anchor_resolutions(ir_dict: dict, base_time_minutes_now: float) -> dict:",
        "    for goal in ir_dict['goals']:",
        "        for reading in goal['readings']:",
        "            if reading['anchor']['kind'] == 'device_event' and reading.get('resolved_at_minutes') is None:",
        "                reading['resolved_at_minutes'] = resolve_anchor_minutes(reading['anchor']['device_id'], base_time_minutes_now)",
        "    return ir_dict",
    ),
    md(
        "## The verified `schedule_workflow` wrapper",
        "",
        "Drop-in replacement registered in `TOOL_REGISTRY` in place of "
        "`tool_schedule_workflow`: parse -> resolve anchors -> Z3 check -> "
        "commit on SAT / reject with explanation on UNSAT. This is the "
        "single integration point — the rest of the ReAct loop "
        "(`src/agents/strategies/react_agent.py`) is untouched.",
    ),
    writefile_cell("extract_ir_from_episode.py", os.path.join(SRC_DIR, "extract_ir_from_episode.py")),
    code("import shutil; shutil.copy('extract_ir_from_episode.py', 'irlib/extract_ir_from_episode.py')"),
    code(
        "from irlib.constraint_encoder import check_feasibility",
        "from irlib.ir_schema import IntentIR, Goal, TimeReading, Anchor, Target, Assertion, ir_from_compact",
        "from src.agents.tools import tool_schedule_workflow, TOOL_REGISTRY, get_tool_config",
        "from datetime import datetime",
        "",
        "def _dict_to_ir(d, query, base_time):",
        "    goals = []",
        "    for g in d['goals']:",
        "        readings = [TimeReading(anchor=Anchor(**r['anchor']), **{k: v for k, v in r.items() if k != 'anchor'}) for r in g['readings']]",
        "        targets = [Target(room_id=t['room_id'], device_id=t['device_id'], device_type=t['device_type'],",
        "                           asserts=[Assertion(**a) for a in t['asserts']]) for t in g['targets']]",
        "        goals.append(Goal(goal_id=g['goal_id'], readings=readings, targets=targets, requires_duration_model=d.get('requires_duration_model', False)))",
        "    return IntentIR(query=query, query_type=d.get('query_type', 'qt4'), case='unknown', base_time=base_time, goals=goals, conflict_type=None)",
        "",
        "_last_user_query = {'text': None}  # set by the ReAct loop wiring below before each schedule_workflow call",
        "",
        "def verified_schedule_workflow(args):",
        "    query = _last_user_query['text']",
        "    if not query:",
        "        return tool_schedule_workflow(args)  # no query context captured -- fail open to the original tool rather than block silently",
        "    try:",
        "        ir_dict = parse_intent(query)",
        "    except Exception as e:",
        "        return tool_schedule_workflow(args)  # parser failure -- fail open; log for offline review",
        "    now_resp = run_tool('get_current_time', {})",
        "    base_time = (now_resp.get('data') or {}).get('now')",
        "    base_time_minutes_now = 0.0",
        "    ir_dict = fill_anchor_resolutions(ir_dict, base_time_minutes_now)",
        "    ir = ir_from_compact(ir_dict, query, base_time)",
        "    result = check_feasibility(ir)",
        "    if result.feasible:",
        "        return tool_schedule_workflow(args)",
        "    return {",
        "        'status': {'code': 409, 'message': 'INFEASIBLE_SCHEDULE'},",
        "        'data': None,",
        "        'error': {'type': 'TEMPORAL_CONTRADICTION', 'detail': verbalize_rejection(result)},",
        "    }",
        "",
        "TOOL_REGISTRY['schedule_workflow'] = verified_schedule_workflow",
    ),
    md(
        "## Wire `_last_user_query` from the ReAct loop",
        "",
        "`ReActAgent.run()` (src/agents/strategies/react_agent.py) takes "
        "`user_query` as a parameter but doesn't expose it to individual "
        "tool calls. Simplest non-invasive hook: monkeypatch it to stash the "
        "query before delegating to the original `run`, rather than editing "
        "the vendored SimuHome source (which the CC BY-NC-ND 4.0 license "
        "makes us wary of forking to publish).",
    ),
    code(
        "from src.agents.strategies.react_agent import ReActAgent",
        "_original_run = ReActAgent.run",
        "",
        "def _patched_run(self, user_query, **kwargs):",
        "    _last_user_query['text'] = user_query",
        "    return _original_run(self, user_query, **kwargs)",
        "",
        "ReActAgent.run = _patched_run",
    ),
    md(
        "## Important: run the verified pipeline in-process, not via the CLI",
        "",
        "`src.cli.main eval-start` (what notebook 01's baseline uses) always "
        "shells out to a **fresh subprocess** for "
        "`src.cli.parallel_model_evaluation` (`main.py::_handle_eval_start` "
        "-> `_run_module`, a real `subprocess.run`). A fresh subprocess "
        "re-imports everything from scratch, so it would run with the "
        "**unpatched** `TOOL_REGISTRY` — the monkeypatch above would be "
        "silently ignored and this would score the unmodified baseline "
        "twice. (`parallel_model_evaluation` itself only subprocesses the "
        "*simulator*, driving the agent and `TOOL_REGISTRY` from this same "
        "Python process via a `ThreadPoolExecutor` — that part is fine, and "
        "is why the monkeypatch above works at all.) So: call the module "
        "directly, in-process, instead of shelling out through the CLI, "
        "for every verified-pipeline run.",
    ),
    code(
        "import sys, yaml, importlib",
        "",
        "def run_verified_eval(run_id, qt, case, seed_range, llm_model, llm_api_base, llm_api_key_env):",
        "    api_key_value = f'env:{llm_api_key_env}' if llm_api_key_env else None",
        "    spec = {",
        "        'schema': 'simuhome-eval-spec-v1',",
        "        'api': {'base': llm_api_base, 'key': api_key_value},",
        "        'run': {'id': run_id, 'output_root': 'experiments'},",
        "        'episode': {'dir': 'data/benchmark', 'qt': qt, 'case': case, 'seed': seed_range},",
        "        'strategy': {'name': 'react', 'timeout': 60, 'temperature': 0.0, 'max_steps': 20},",
        "        'orchestration': {'max_workers': 1, 'simulator_start_timeout': 30,",
        "                          'simulator_start_retries': 1, 'evaluation_retries': 1,",
        "                          'allow_partial_start': True},",
        "        'judge': {'model': llm_model, 'api_base': llm_api_base, 'api_key': api_key_value},",
        "        'models': [{'model': llm_model, 'api_base': llm_api_base, 'api_key': api_key_value}],",
        "    }",
        "    spec_path = f'{run_id}.yaml'",
        "    with open(spec_path, 'w') as f:",
        "        yaml.safe_dump(spec, f, sort_keys=False)",
        "    pme = importlib.import_module('src.cli.parallel_model_evaluation')",
        "    old_argv = sys.argv",
        "    sys.argv = ['parallel_model_evaluation', '--spec', spec_path]",
        "    try:",
        "        pme.main()  # in-process: TOOL_REGISTRY['schedule_workflow'] patch above stays in effect",
        "    finally:",
        "        sys.argv = old_argv",
        "",
        "os.chdir(os.path.join(PROJECT_ROOT, 'SimuHome'))  # run_verified_eval uses paths relative to here",
        "",
        "# Smoke test: one infeasible + one feasible qt4-1 episode. Full-scale",
        "# comparison against the baseline belongs in notebook 06.",
        "run_verified_eval('smoke_verified_qt4_1_infeasible', 'qt4-1', 'infeasible', '1', LLM_MODEL, LLM_API_BASE, LLM_API_KEY_ENV)",
        "run_verified_eval('smoke_verified_qt4_1_feasible', 'qt4-1', 'feasible', '1', LLM_MODEL, LLM_API_BASE, LLM_API_KEY_ENV)",
        "# Confirm: infeasible -> agent's finish answer reflects the verbalized rejection;",
        "# feasible -> schedule_workflow succeeds exactly as under the unmodified baseline",
        "# (check experiments/smoke_verified_.../run_summary.json for both).",
    ),
]
save("05_integration_pipeline.ipynb", nb05)

# ---------------------------------------------------------------------------
# 06_evaluation_and_ablations.ipynb
# ---------------------------------------------------------------------------
nb06 = [
    md(
        "# 06 — Full evaluation & ablations",
        "",
        "**Kaggle settings:** GPU T4 x2 if using a local model; CPU-only "
        "(None) if only calling a hosted API. Run after 01 (baseline) and 05 "
        "(verified pipeline) both work standalone.",
        "",
        "**E1 (main result):** verified pipeline vs. unmodified baseline, on "
        "the held-out native QT4 episodes (the ~200 never used in "
        "fine-tuning — see notebook 02's split), all three sub-types x both "
        "cases. Report success rate per (query_type, case) cell, matching "
        "the paper's table layout so it's directly comparable.",
        "",
        "**Ablations (E2-E6, exact designs to fill in once E1 is running — "
        "don't lock these in speculatively):**",
        "- E2: SLM intent parser only, no Z3 (does structured IR alone help "
        "the agent, or is the solver doing the real work?)",
        "- E3: Z3 with oracle IR (ground-truth `debug_ground_truth`) vs. "
        "SLM-produced IR — isolates parser error from solver error.",
        "- E4: verbalizer ablation — return the raw unsat core vs. the "
        "templated NL explanation, measure whether the agent's downstream "
        "answer quality differs.",
        "- E5: per-conflict-type breakdown (the conflict_type field survives "
        "in `meta`) — does the fix help evenly across mismatch types, or "
        "mainly on the dominant `timing_mismatch_op_nonop`/`absolute_time_"
        "mismatch*` types?",
        "- E6: cost/latency overhead of the verify step vs. baseline.",
    ),
    code(
        "import json, glob, os, pandas as pd",
        "",
        "HELD_OUT_DIR = 'SimuHome/data/benchmark'  # the untouched native episodes",
        "assert os.path.isdir(HELD_OUT_DIR), 'clone/attach SimuHome first (see notebook 01)'",
    ),
    md(
        "## E1 — main result",
        "",
        "Reuse SimuHome's own `src/pipelines/episode_evaluation` scoring "
        "(`qt4/feasible.py` for feasible episodes: simulator-state diff at "
        "the goal ticks; `qt4/infeasible_{1,2,3}.py` for infeasible episodes: "
        "LLM-judge panel on the final answer against `temporal_conflict`) so "
        "scores are computed exactly the way the paper computes them — don't "
        "reimplement scoring here.",
    ),
    code(
        "# TODO: for each QT4 sub-type x case, run:",
        "#   (a) the unmodified ReAct agent (notebook 01's harness)",
        "#   (b) the verified pipeline (notebook 05's TOOL_REGISTRY['schedule_workflow'] override)",
        "# on the same held-out episode set and the same base LLM, then score both",
        "# with src.pipelines.episode_evaluation.qt4.{feasible,infeasible_1,infeasible_2,infeasible_3}.evaluate().",
        "results = []  # append {'system': 'baseline'|'verified', 'query_type':..., 'case':..., 'score':...}",
    ),
    code(
        "df = pd.DataFrame(results)",
        "if len(df):",
        "    table = df.pivot_table(index=['query_type', 'case'], columns='system', values='score', aggfunc='mean')",
        "    print(table)",
        "    table.to_csv('/kaggle/working/e1_main_result.csv')",
    ),
    md("## E2-E6 — ablations"),
    code(
        "# TODO: each ablation reuses the same held-out set and scoring functions,",
        "# swapping one component per E2-E6 above. Save each as its own CSV/JSON",
        "# under /kaggle/working/ so the ablation table can be assembled without rerunning everything.",
    ),
    md(
        "## Save everything",
        "",
        "Version `/kaggle/working` as a Kaggle Dataset once this notebook "
        "finishes — `results/` (this project's local copy) should mirror it "
        "for the writeup.",
    ),
]
save("06_evaluation_and_ablations.ipynb", nb06)

# ---------------------------------------------------------------------------
# 00_full_pipeline.ipynb — everything above, one notebook, one Kaggle session
# ---------------------------------------------------------------------------
# Not a blind concatenation of 01-06: setup/clone/writefile steps that were
# duplicated across notebooks (because each had to be Dataset-attachable and
# self-contained on its own) happen exactly once here, and steps that
# previously read a sibling notebook's output from a Kaggle Dataset
# (`/kaggle/input/...`) now just read the local file this same session wrote
# a few cells up. Every expensive step (data generation, fine-tuning) checks
# for its own output first and skips itself if already done, so re-running
# the notebook after a restart (9-12h Kaggle session cap) resumes instead of
# redoing work.

nb_full = [
    md(
        "# Full pipeline — neuro-symbolic verification for SimuHome QT4 scheduling",
        "",
        "One notebook, one Kaggle session, run top to bottom:",
        "1. Setup (clone SimuHome, install deps, write `src/`)",
        "2. Z3 encoder sanity check (no GPU, no data, ~seconds)",
        "3. Baseline reproduction (no GPU if using a hosted API; needs `OPENAI_API_KEY`)",
        "4. Generate (query, IR) training data (no GPU)",
        "5. LoRA fine-tune the intent parser (**needs GPU**)",
        "6. Wire the fine-tuned parser + Z3 + verbalizer into `schedule_workflow`",
        "7. Full evaluation (E1) + ablations",
        "",
        "**Kaggle settings:** Accelerator **GPU T4 x2**, Internet **ON**. "
        "A T4 session is capped at ~9-12h continuous — if steps 5-7 don't "
        "fit in one sitting, just re-run this same notebook in a new "
        "session after 'Save Version': every expensive cell below checks "
        "for its own output first and skips itself if the work is already "
        "on disk, so nothing is repeated. Version `/kaggle/working` as a "
        "Kaggle Dataset between sessions so the skip-checks have something "
        "to find.",
        "",
        "This replaces running `01`-`06` as six separate notebooks/sessions "
        "and manually passing Kaggle Datasets between them — same content, "
        "same tested `src/` modules, single linear run.",
    ),
    md("## 1. Setup"),
    code(
        "!pip -q install -U z3-solver transformers peft accelerate bitsandbytes trl datasets torchao",
    ),
    code(
        "import os",
        "# This kernel (parser, fine-tuning) may only use GPU 0. GPU 1 belongs to the vLLM server, and with both",
        "# visible HF Trainer would wrap training in DataParallel and OOM there. Must be set before CUDA is first used.",
        "# (The vLLM subprocess sets its own CUDA_VISIBLE_DEVICES, so it is unaffected.)",
        "os.environ['CUDA_VISIBLE_DEVICES'] = '0'",
        "PROJECT_ROOT = os.getcwd()  # capture once, before any cd -- reruns then can't nest",
        "if not os.path.isdir('SimuHome'):",
        "    !git clone --depth 1 https://github.com/holi-lab/SimuHome.git",
        "# pyproject.toml pins requires-python >= 3.13; no 3.13-only syntax in the",
        "# code (checked), so bypass the gate rather than fail on a 3.10/3.11 image.",
        "!pip install -e ./SimuHome -q --ignore-requires-python",
        "import sys",
        "sys.path.insert(0, os.path.join(PROJECT_ROOT, 'SimuHome'))",
    ),
    code("import os; os.makedirs('irlib', exist_ok=True)"),
    writefile_cell("irlib/ir_schema.py", os.path.join(SRC_DIR, "ir_schema.py")),
    writefile_cell("irlib/constraint_encoder.py", os.path.join(SRC_DIR, "constraint_encoder.py")),
    writefile_cell("irlib/extract_ir_from_episode.py", os.path.join(SRC_DIR, "extract_ir_from_episode.py")),
    code(
        "open('irlib/__init__.py', 'a').close()",
        "import sys",
        "sys.path.insert(0, '.')",
    ),
    md(
        "## 2. Z3 encoder sanity check",
        "",
        "Hand-crafted QT4-1/2/3 feasible + infeasible examples, checked "
        "before any model is involved (per the project handoff: never blame "
        "the SLM for a bad encoding). Locally, this scores **286/286** "
        "against all 300 downloaded real QT4 benchmark episodes (150 "
        "feasible + 136 infeasible correct, 14 duration-model conflicts "
        "correctly flagged as out of scope — see the README's Known "
        "limitation section).",
    ),
    code(
        "from irlib.ir_schema import IntentIR, Goal, TimeReading, Anchor, Target, Assertion",
        "from irlib.constraint_encoder import check_feasibility",
        "",
        "# QT4-1 infeasible: stated clock time contradicts stated relative offset.",
        "ir_infeasible = IntentIR(",
        "    query='At 11:35 AM, that is 13 minutes from now, turn on living room light 1.',",
        "    query_type='qt4-1', case='infeasible', base_time='2025-08-23 11:36:54',",
        "    goals=[Goal(goal_id=0, readings=[",
        "        TimeReading(relation='after', anchor=Anchor(kind='now'), offset_minutes=13, tolerance_minutes=5),",
        "        TimeReading(relation='at', anchor=Anchor(kind='now'), absolute_time='11:35:54', tolerance_minutes=5),",
        "    ], targets=[Target(room_id='living_room', device_id='living_room_dimmable_light_1', device_type='dimmable_light',",
        "                        asserts=[Assertion(attribute='1.OnOff.OnOff', value=True)])])],",
        ")",
        "result = check_feasibility(ir_infeasible)",
        "print(result.feasible, result.explanation())",
        "assert not result.feasible",
    ),
    code(
        "from irlib.extract_ir_from_episode import extract_dataset",
        "",
        "pairs = extract_dataset('SimuHome/data/benchmark')",
        "counts = {'feasible_correct': 0, 'feasible_wrong': 0, 'infeasible_correct': 0, 'infeasible_wrong': 0, 'skipped': 0}",
        "for pair in pairs:",
        "    d = pair['debug_ground_truth']",
        "    goals = []",
        "    for g in d['goals']:",
        "        readings = [TimeReading(anchor=Anchor(**r['anchor']), **{k: v for k, v in r.items() if k != 'anchor'}) for r in g['readings']]",
        "        targets = [Target(room_id=t['room_id'], device_id=t['device_id'], device_type=t['device_type'],",
        "                           asserts=[Assertion(**a) for a in t['asserts']]) for t in g['targets']]",
        "        goals.append(Goal(goal_id=g['goal_id'], readings=readings, targets=targets, requires_duration_model=g['requires_duration_model']))",
        "    ir = IntentIR(query=d['query'], query_type=d['query_type'], case=d['case'], base_time=d['base_time'], goals=goals, conflict_type=d['conflict_type'])",
        "    r = check_feasibility(ir)",
        "    expected = ir.case == 'feasible'",
        "    if any('duration model' in (g.reason or '') for g in r.goal_results) and not expected:",
        "        counts['skipped'] += 1",
        "        continue",
        "    key = ('feasible' if expected else 'infeasible') + ('_correct' if r.feasible == expected else '_wrong')",
        "    counts[key] += 1",
        "print(counts)",
        "assert counts['feasible_wrong'] == 0 and counts['infeasible_wrong'] == 0",
    ),
    md(
        "## 3. Baseline reproduction (agent LLM: a hosted API, NVIDIA Gemma by default)",
        "",
        "Sanity-check SimuHome's own ReAct baseline before building anything on top. The agent LLM is a **hosted API called through its OpenAI-compatible endpoint**. The default is `google/gemma-4-31b-it` on NVIDIA's free hosted API (the only strong-ish model that still worked when tested; Llama-3.3-70B and most others there have been retired, and free hosted models can disappear, so record the exact model id and the dates you ran). Free tiers are rate- and quota-limited, so this notebook:",
        "- throttles every LLM call (`MIN_SECONDS_BETWEEN_CALLS`) and counts them,",
        "- runs the evaluator **in this process** for BOTH arms, so the throttle, the counter and the request fixes apply to both,",
        "- runs E1 a few (sub-type, case) cells at a time (`E1_CELLS`) and saves after each run, so you can spread it over several days.",
        "",
        "Get a key at https://build.nvidia.com (Get API Key), add it as the Kaggle secret `NVIDIA_API_KEY`. To use another provider (Gemini, Mistral, Groq...), change the four lines marked below in the backend cell: base URL, model id, secret name, and the throttle.",
    ),
    code(
        "import os, sys, time, subprocess, requests",
        "",
        "LLM_BACKEND = 'custom'   # 'custom' (any OpenAI-compatible API; default = Gemini) | 'local_vllm' | 'openrouter'",
        "",
        "if LLM_BACKEND == 'custom':",
        "    LLM_API_BASE = 'https://integrate.api.nvidia.com/v1'    # Gemini alt: 'https://generativelanguage.googleapis.com/v1beta/openai'",
        "    LLM_MODEL = 'google/gemma-4-31b-it'         # the only NVIDIA-hosted model that passed our agent-format test; 'gemini-2.5-flash' for Gemini",
        "    LLM_API_KEY_ENV = 'NVIDIA_API_KEY'          # name of the Kaggle secret that holds your key",
        "    from kaggle_secrets import UserSecretsClient",
        "    os.environ[LLM_API_KEY_ENV] = UserSecretsClient().get_secret(LLM_API_KEY_ENV)",
        "elif LLM_BACKEND == 'openrouter':",
        "    try:",
        "        from kaggle_secrets import UserSecretsClient",
        "        os.environ['OPENROUTER_API_KEY'] = UserSecretsClient().get_secret('OPENROUTER_API_KEY')",
        "    except Exception as e:",
        "        print('Set OPENROUTER_API_KEY manually:', e)",
        "    LLM_MODEL = 'meta-llama/llama-3.3-70b-instruct'",
        "    LLM_API_BASE = 'https://openrouter.ai/api/v1'",
        "    LLM_API_KEY_ENV = 'OPENROUTER_API_KEY'",
        "elif LLM_BACKEND == 'local_vllm':",
        "    LLM_MODEL = 'Qwen/Qwen2.5-7B-Instruct-AWQ'",
        "    LLM_API_BASE = 'http://127.0.0.1:8001/v1'",
        "    LLM_API_KEY_ENV = None",
        "else:",
        "    raise ValueError(f'unknown LLM_BACKEND: {LLM_BACKEND!r}')",
        "",
        "# ---- free-tier guards, applied by the shim in the next cell ----",
        "MIN_SECONDS_BETWEEN_CALLS = 2.0   # NVIDIA free tier is 40 requests/min (1.5 s); raise it if you see 429s",
        "LLM_CALL_CAP = 1500              # hard stop: no more than this many LLM requests in one session (a run is about 80; the 7-run plan is about 600)",
        "DROP_PARAMS = ['seed', 'response_format']   # NVIDIA lists structured output as unsupported; the agent's JSON is then parsed from text",
        "STRIP_SCHEMA_STRICT = True        # drop `strict` / `additionalProperties` from the JSON schema (Gemini accepts a subset)",
        "EXTRA_KWARGS = {}                 # e.g. {'reasoning_effort': 'low'} to cut Gemini thinking time/tokens",
        "if LLM_BACKEND == 'local_vllm':",
        "    MIN_SECONDS_BETWEEN_CALLS, DROP_PARAMS, STRIP_SCHEMA_STRICT = 0.0, [], False",
        "",
        "vllm_proc = None",
        "if LLM_BACKEND == 'local_vllm':",
        "    VLLM_PY = '/tmp/vllm_env/bin/python'",
        "    # Clean env for the venv: Kaggle's PYTHONPATH/sitecustomize import packages that don't exist in the venv.",
        "    clean_env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}",
        "    clean_env['PYTHONNOUSERSITE'] = '1'",
        "    def _vllm_ok():",
        "        if not os.path.exists(VLLM_PY): return False",
        "        return subprocess.run([VLLM_PY, '-c', 'import vllm'], capture_output=True, env=clean_env).returncode == 0",
        "    if not _vllm_ok():",
        "        # uv creates the venv (with or without pip) and installs much faster than pip.",
        "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)",
        "        subprocess.run([sys.executable, '-m', 'uv', 'venv', '/tmp/vllm_env', '--python', sys.executable, '--clear'], check=True, env=clean_env)",
        "        r = subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', VLLM_PY, 'vllm'],",
        "                           capture_output=True, text=True, env=clean_env)",
        "        print(r.stdout[-800:], r.stderr[-2000:])",
        "        if not _vllm_ok():",
        "            raise RuntimeError('vLLM is still not importable in /tmp/vllm_env (install output above)')",
        "    print('vLLM importable in the isolated venv')",
        "    n_gpu = len(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip().splitlines())",
        "    env = dict(clean_env)",
        "    if n_gpu >= 2:",
        "        env['CUDA_VISIBLE_DEVICES'] = '1'; util = '0.88'   # leave GPU 0 for the parser / fine-tuning",
        "    else:",
        "        util = '0.45'",
        "    print(f'{n_gpu} GPU(s); starting vLLM with gpu-memory-utilization={util}')",
        "    log = open('/tmp/vllm_server.log', 'w')",
        "    vllm_proc = subprocess.Popen(",
        "        [VLLM_PY, '-m', 'vllm.entrypoints.openai.api_server', '--model', LLM_MODEL, '--served-model-name', LLM_MODEL,",
        "         '--port', '8001', '--dtype', 'half', '--max-model-len', '8192', '--gpu-memory-utilization', util],",
        "        env=env, stdout=log, stderr=subprocess.STDOUT)",
        "    for _ in range(450):   # up to 15 min (first run downloads the weights)",
        "        if vllm_proc.poll() is not None:",
        "            print(open('/tmp/vllm_server.log').read()[-3000:]); raise RuntimeError('vLLM exited early (log above)')",
        "        try:",
        "            requests.get('http://127.0.0.1:8001/v1/models', timeout=1); print('local vLLM server up'); break",
        "        except Exception:",
        "            time.sleep(2)",
        "    else:",
        "        print(open('/tmp/vllm_server.log').read()[-3000:]); raise RuntimeError('vLLM did not come up in 15 min')",
    ),
    md(
        "### Request shim (free-tier guards)",
        "",
        "SimuHome builds its own OpenAI client, so we patch the SDK's `chat.completions.create` once, in this process: it spaces calls out, counts them, drops parameters the provider rejects, and relaxes the JSON schema for providers that accept only a subset. It affects the baseline AND verified arms identically.",
    ),
    code(
        "import threading, copy",
        "from openai.resources.chat.completions import Completions",
        "import openai",
        "SHIM_RETRIES = 4   # extra attempts on provider errors, with backoff (so a brief 504 does not kill an unattended run)",
        "_RETRYABLE = (openai.InternalServerError, openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)",
        "",
        "LLM_CALLS = {'count': 0, 'last': 0.0}",
        "_llm_lock = threading.Lock()",
        "_orig_create = getattr(Completions.create, '_orig', Completions.create)   # idempotent if this cell is re-run",
        "",
        "def _strip_schema(node):",
        "    if isinstance(node, dict):",
        "        node.pop('additionalProperties', None)",
        "        for v in node.values():",
        "            _strip_schema(v)",
        "    elif isinstance(node, list):",
        "        for v in node:",
        "            _strip_schema(v)",
        "",
        "def _shimmed_create(self, *args, **kwargs):",
        "    for k in DROP_PARAMS:",
        "        kwargs.pop(k, None)",
        "    kwargs.update(EXTRA_KWARGS)",
        "    rf = kwargs.get('response_format')",
        "    if STRIP_SCHEMA_STRICT and isinstance(rf, dict) and rf.get('type') == 'json_schema':",
        "        rf = copy.deepcopy(rf)",
        "        rf['json_schema'].pop('strict', None)",
        "        _strip_schema(rf['json_schema'].get('schema', {}))",
        "        kwargs['response_format'] = rf",
        "    _cap = globals().get('LLM_CALL_CAP')",
        "    for attempt in range(SHIM_RETRIES + 1):",
        "        with _llm_lock:",
        "            if _cap and LLM_CALLS['count'] >= _cap:",
        "                raise RuntimeError(f'LLM call cap reached ({_cap}); raise LLM_CALL_CAP if you really want more')",
        "            wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - LLM_CALLS['last'])",
        "            if wait > 0:",
        "                time.sleep(wait)",
        "            LLM_CALLS['last'] = time.time()",
        "            LLM_CALLS['count'] += 1          # every attempt counts toward the cap, retries included",
        "        try:",
        "            return _orig_create(self, *args, **kwargs)",
        "        except _RETRYABLE as e:              # provider-side trouble (504/5xx, 429, timeouts, dropped connections)",
        "            if attempt == SHIM_RETRIES:",
        "                raise",
        "            delay = 20 * 2 ** attempt        # 20, 40, 80, 160 s",
        "            print(f'[shim] {type(e).__name__}: retry {attempt + 1}/{SHIM_RETRIES} in {delay}s', flush=True)",
        "            time.sleep(delay)",
        "",
        "_shimmed_create._orig = _orig_create",
        "Completions.create = _shimmed_create",
        "print('request shim installed; min seconds between calls =', MIN_SECONDS_BETWEEN_CALLS)",
    ),
    md(
        "### Preflight: prove the LLM answers BEFORE spending quota on it",
        "",
        "Sends the request exactly the way SimuHome's ReAct agent does (temperature 0, seed, strict JSON-schema output with its `thought/action/action_input` schema), through the shim above. It **stops** with the provider's raw error if it fails. Common causes: quota exhausted (429), wrong model id (404), bad key (401/403).",
    ),
    code(
        "import json",
        "from openai import OpenAI",
        "",
        "STOP_ON_BAD_FORMAT = False   # the probe below uses a toy prompt, so it is only a hint; the real check is the agent smoke test in the E1 cell",
        "",
        "def llm_preflight():",
        "    client = OpenAI(api_key=(os.environ.get(LLM_API_KEY_ENV, '') if LLM_API_KEY_ENV else 'dummy'), base_url=LLM_API_BASE)",
        "    schema = {'type': 'json_schema', 'json_schema': {'name': 'react_response', 'strict': True, 'schema': {",
        "        'type': 'object',",
        "        'properties': {'thought': {'type': 'string'}, 'action': {'type': 'string'}, 'action_input': {'type': 'string', 'default': '{}'}},",
        "        'required': ['thought', 'action', 'action_input'], 'additionalProperties': False}}}",
        "    msgs = [{'role': 'user', 'content': 'Use the tool get_current_time. Answer with thought, action and action_input (a JSON string).'}]",
        "    for label, extra in [('plain', {}), ('json_schema, as the ReAct agent sends it', {'response_format': schema})]:",
        "        try:",
        "            r = client.chat.completions.create(model=LLM_MODEL, temperature=0.0, seed=42, messages=msgs, **extra)",
        "        except Exception as e:",
        "            print(f'[{label}] FAILED: {repr(e)[:900]}')",
        "            raise RuntimeError(f'LLM preflight failed ({label}); see the provider error above')",
        "        txt = r.choices[0].message.content or ''",
        "        print(f'[{label}] OK: {txt[:220]!r}')",
        "        if extra:   # the agent-format check: the reply must parse as {thought, action, action_input}",
        "            try:",
        "                d = json.loads(txt[txt.find('{'):txt.rfind('}') + 1])",
        "                fmt_ok = {'thought', 'action', 'action_input'} <= set(d)",
        "            except Exception:",
        "                fmt_ok = False",
        "            print('   agent JSON format (thought/action/action_input):', 'OK' if fmt_ok else 'NOT PARSEABLE with this toy prompt (only a hint; the agent smoke test before E1 is the real check)')",
        "            if not fmt_ok and STOP_ON_BAD_FORMAT:",
        "                raise RuntimeError('agent model did not return the ReAct JSON format on the probe; change LLM_MODEL (or set STOP_ON_BAD_FORMAT=False)')",
        "    print('LLM preflight OK; calls used so far:', LLM_CALLS['count'])",
        "",
        "llm_preflight()",
    ),
    code(
        "import subprocess, time, requests",
        "sim_proc = subprocess.Popen(",
        "    ['python', '-m', 'uvicorn', 'src.simulator.api.app:app', '--port', '8000'],",
        "    cwd='SimuHome', stdout=subprocess.PIPE, stderr=subprocess.STDOUT,",
        ")",
        "for _ in range(30):",
        "    try:",
        "        requests.get('http://127.0.0.1:8000/docs', timeout=1)",
        "        print('simulator up')",
        "        break",
        "    except Exception:",
        "        time.sleep(1)",
        "else:",
        "    raise RuntimeError('simulator failed to start; check sim_proc.stdout')",
    ),
    md(
        "### Run SimuHome's evaluator in this process",
        "",
        "`main.py eval-start` runs the evaluator in a fresh subprocess, where none of the guards above (or the verifier patch in section 6) would apply. So both arms call `parallel_model_evaluation.main()` directly. `max_workers` is 1: the shim serialises calls anyway and a free tier can't use more.",
    ),
    code(
        "import os, sys, json, yaml, importlib, shutil",
        "os.chdir(os.path.join(PROJECT_ROOT, 'SimuHome'))",
        "",
        "def build_eval_spec(run_id, qt, case, seed_range):",
        "    api_key_value = f'env:{LLM_API_KEY_ENV}' if LLM_API_KEY_ENV else None",
        "    return {",
        "        'schema': 'simuhome-eval-spec-v1',",
        "        'api': {'base': LLM_API_BASE, 'key': api_key_value},",
        "        'run': {'id': run_id, 'output_root': 'experiments'},",
        "        'episode': {'dir': 'data/benchmark', 'qt': qt, 'case': case, 'seed': seed_range},",
        "        'strategy': {'name': 'react', 'timeout': 120, 'temperature': 0.0, 'max_steps': 20},",
        "        'orchestration': {'max_workers': 1, 'simulator_start_timeout': 30,",
        "                          'simulator_start_retries': 1, 'evaluation_retries': 1,",
        "                          'allow_partial_start': True},",
        "        'judge': {'model': LLM_MODEL, 'api_base': LLM_API_BASE, 'api_key': api_key_value},",
        "        'models': [{'model': LLM_MODEL, 'api_base': LLM_API_BASE, 'api_key': api_key_value}],",
        "    }",
        "",
        "import types, multiprocessing",
        "",
        "class _InlinePool:",
        "    \"\"\"SimuHome's evaluator runs each model's episodes in an mp.Pool of SEPARATE worker processes. Anything patched in",
        "    this notebook (the request shim, the verifier, their counters, the GPU-resident parser) would not exist there, and",
        "    on Linux forked workers cannot use the CUDA context at all. This drop-in runs the same work in THIS process.\"\"\"",
        "    def __init__(self, processes=None, initializer=None, initargs=()):",
        "        # The worker initializer points sys.stdout/sys.stderr at /dev/null (fine in a worker process, but here it would",
        "        # silence this notebook), so remember them and put them back when the pool is closed.",
        "        self._saved = (sys.stdout, sys.stderr)",
        "        if initializer:",
        "            initializer(*initargs)",
        "    def __enter__(self):",
        "        return self",
        "    def __exit__(self, *exc):",
        "        sys.stdout, sys.stderr = self._saved",
        "        return False",
        "    def map_async(self, fn, iterable):",
        "        results = [fn(x) for x in iterable]",
        "        class _Done:",
        "            def ready(self): return True",
        "            def get(self): return results",
        "        return _Done()",
        "",
        "def run_inproc(run_id, qt, case, seed_range):",
        "    spec_path = f'{run_id}.yaml'",
        "    shutil.rmtree(f'experiments/{run_id}', ignore_errors=True)   # SimuHome refuses an existing run folder; a failed earlier attempt leaves one",
        "    with open(spec_path, 'w') as f:",
        "        yaml.safe_dump(build_eval_spec(run_id, qt, case, seed_range), f, sort_keys=False)",
        "    pme = importlib.import_module('src.cli.parallel_model_evaluation')",
        "    pme.mp = types.SimpleNamespace(Queue=multiprocessing.Queue, Pool=_InlinePool)   # see _InlinePool above",
        "    old_argv = sys.argv",
        "    sys.argv = ['parallel_model_evaluation', '--spec', spec_path]",
        "    try:",
        "        pme.main()",
        "    except SystemExit as e:",
        "        print('evaluator exited with code', e.code)",
        "    finally:",
        "        sys.argv = old_argv",
        "    if not os.path.exists(f'experiments/{run_id}/run_summary.json'):",
        "        raise RuntimeError(f'no run_summary.json for {run_id}; read the evaluator output above')",
        "",
        "def aggregate_run(run_id):",
        "    # Per-episode results live in a per-model SUBFOLDER, and the aggregator only reads *.json directly inside the",
        "    # folder it is given, so aggregate each subfolder (the run folder itself yields 0 episodes).",
        "    base = f'experiments/{run_id}'",
        "    if not os.path.isdir(base):",
        "        return {}",
        "    out = {}",
        "    for sub in sorted(d for d in os.listdir(base) if os.path.isdir(f'{base}/{d}')):",
        "        subprocess.run([sys.executable, '-m', 'src.cli.main', 'aggregate', '--dir', f'{base}/{sub}'], capture_output=True, text=True)",
        "        f = f'{base}/{sub}/aggregate_summary.json'",
        "        if os.path.exists(f):",
        "            out[sub] = json.load(open(f))",
        "    return out",
    ),
    md(
        "### Baseline sanity check (5 episodes, about 50 LLM calls)",
        "",
        "Read `infra_errors` first: above 0 means the API failed (quota, 429, rejected request), which says nothing about the agent. A healthy result is infra_errors 0 and an accuracy well above 0. If accuracy is 0 with `schema_errors` > 0, the model cannot produce SimuHome's JSON action format.",
    ),
    code(
        "RUN_BASELINE_SMOKE = False   # True only on a first, attended session (about 15-30 min); the preflight already checks the API",
        "BASELINE_SEEDS = '1 - 3'   # about 30 LLM calls; raise once you know your quota",
        "if RUN_BASELINE_SMOKE and not os.path.exists('experiments/baseline_qt4_1_feasible/run_summary.json'):",
        "    run_inproc('baseline_qt4_1_feasible', 'qt4-1', 'feasible', BASELINE_SEEDS)",
        "for model_dir, agg in aggregate_run('baseline_qt4_1_feasible').items():",
        "    print(model_dir, json.dumps(agg['overall'], indent=2))",
        "print('LLM calls used so far:', LLM_CALLS['count'])",
        "os.chdir(PROJECT_ROOT)",
    ),
    md(
        "### Baseline check: is the agent working at all?",
        "",
        "The numbers above only say how many episodes passed. This cell explains *why*, using the per-episode result files. Run it after the baseline (and reuse `inspect_run(run_id)` later on any E1 run).",
        "",
        "**How to read it**",
        "",
        "| What you see | What it means | What to do |",
        "|---|---|---|",
        "| `infra_errors` > 0 | The API or simulator failed (quota, 429/5xx, retired model). The accuracy is **not meaningful**. | Read the provider message, fix quota or model, then delete `experiments/<run>` and rerun. |",
        "| `schema_errors` > 0 | The model broke SimuHome's JSON action format (`thought` / `action` / `action_input`). | A stronger model is needed. |",
        "| accuracy 0, no errors | It runs but solves nothing. Look at the per-episode lines. | See the list below. |",
        "| accuracy > 0, no errors | The agent works. | Proceed to E1, but 3 episodes is a tiny sample. |",
        "",
        "**Per-episode lines:** `required lookups 0/2` means it never called `get_room_devices`, which the scorer requires (and usually means it invented device ids). An `actions` list ending in `finish` with no `schedule_workflow` means it answered without scheduling anything. A `schedule_workflow` with the wrong time means a time-arithmetic error.",
    ),
    code(
        "import glob, json, os",
        "os.chdir(os.path.join(PROJECT_ROOT, 'SimuHome'))",
        "",
        "def inspect_run(run_id, show=10):",
        "    base = f'experiments/{run_id}'",
        "    runs = aggregate_run(run_id)",
        "    if not runs:",
        "        print(f'{run_id}: no per-model results found under {base}'); return",
        "    for model_dir, agg in runs.items():",
        "        o = agg['overall']",
        "        print(f'== {run_id}   (model folder: {model_dir})')",
        "        print(f\"   episodes evaluated: {o['evaluated_total']}/{o['total']}    correct: {o['correct']}    accuracy: {o['accuracy']:.2f}\")",
        "        print(f\"   infra_errors (API/simulator failed): {o['infra_errors']}    schema_errors (bad JSON action format): {o['schema_errors']}\")",
        "        print(f\"   avg seconds per episode: {o['avg_duration_sec']}    avg tool calls per episode: {o['avg_actions']}\")",
        "        notes = []",
        "        if o['evaluated_total'] == 0:",
        "            notes.append('NO episodes were evaluated: read the evaluator output above for the real error.')",
        "        if o['infra_errors']:",
        "            notes.append('API/simulator errors: check your quota and the provider message. The accuracy is NOT meaningful yet.')",
        "        if o['schema_errors']:",
        "            notes.append('The model broke the JSON action format in some episodes (a stronger model is needed).')",
        "        if not notes and o['accuracy'] == 0:",
        "            notes.append('Runs mechanically but solved nothing: see the per-episode lines (skipped lookups? invented device ids? wrong times?).')",
        "        if not notes:",
        "            notes.append('GO: the agent works. With only a few episodes treat this as a smoke test, not a result.')",
        "        for n in notes:",
        "            print('   ->', n)",
        "        files = [f for f in sorted(glob.glob(f'{base}/{model_dir}/*.json')) if not f.endswith('aggregate_summary.json')]",
        "        print('   per-episode:')",
        "        for f in files[:show]:",
        "            d = json.load(open(f))",
        "            ev = d.get('evaluation_result') or {}",
        "            steps = d.get('steps') or []",
        "            actions = [s.get('action') for s in steps]",
        "            req = ev.get('required_actions') or []",
        "            req_txt = f\"{sum(1 for r in req if r.get('invoked'))}/{len(req)}\" if req else 'n/a'",
        "            answer = ''",
        "            for s in steps:",
        "                if s.get('action') == 'finish':",
        "                    ai = s.get('action_input')",
        "                    answer = (ai.get('answer') if isinstance(ai, dict) else str(ai))[:90]",
        "            print(f\"   seed {d.get('seed')}: score={ev.get('score')}  error={ev.get('error_type') or '-'}  required lookups {req_txt}\")",
        "            print(f'            actions: {actions}')",
        "            if answer:",
        "                print(f'            final answer: {answer!r}')",
        "        print()",
        "",
        "inspect_run('baseline_qt4_1_feasible')",
        "print('LLM calls used so far:', LLM_CALLS['count'])",
        "os.chdir(PROJECT_ROOT)",
    ),
    md(
        "## 4. Generate (query, IR) training data",
        "",
        "Deterministic extraction (no LLM calls) — see "
        "`src/extract_ir_from_episode.py`. The original ~200 native "
        "benchmark episodes are held out entirely from fine-tuning; only "
        "synthesized episodes (new seeds, via SimuHome's own generator) go "
        "into train/val. Generation itself needs an LLM to write the "
        "synthetic NL queries — reuses the same free `LLM_MODEL` chosen in "
        "section 3 (OpenRouter's free tier or the local vLLM server).",
    ),
    code(
        "os.chdir(os.path.join(PROJECT_ROOT, 'SimuHome'))",
        "!python -m src.cli.episode_generator --help",
    ),
    code(
        "import os, sys, glob, json, collections, subprocess, yaml",
        "",
        "# episode.home is REQUIRED by SimuHome's generator (the earlier run died with",
        "# 'episode.home must be a mapping'); this block is copied from gen_spec.example.yaml.",
        "HOME_SCHEMA = {'room_count': 5, 'devices_per_room': {'min': 4, 'max': 7},",
        "               'environment': {'temperature_c': {'min': 22, 'max': 32}, 'humidity_pct': {'min': 35, 'max': 65},",
        "                               'illuminance_lux': {'min': 100, 'max': 1500}, 'pm10_ugm3': {'min': 20, 'max': 100}}}",
        "RUN_GENERATION = False   # True only to build a larger training set; then delete data/qt4_ir_v2_* and the v2 adapter to retrain",
        "GEN_SEEDS = '1000-1049'   # 50 per (sub-type, case) x 6 = 300 episodes; widen once the free API proves fast enough",
        "api_key_value = f'env:{LLM_API_KEY_ENV}' if LLM_API_KEY_ENV else None",
        "",
        "for qt in ['qt4-1', 'qt4-2', 'qt4-3']:",
        "    for case in ['feasible', 'infeasible']:",
        "        if not RUN_GENERATION:",
        "            continue",
        "        run_id = f'gen_{qt}_{case}'.replace('-', '_')",
        "        episodes_dir = f'../data/generated_qt4/{run_id}/episodes'",
        "        if glob.glob(f'{episodes_dir}/*.json'):",
        "            print('skip (already generated):', run_id)",
        "            continue",
        "        spec = {'schema': 'simuhome-gen-spec-v1',",
        "                'run': {'id': run_id, 'output_root': '../data/generated_qt4'},",
        "                'episode': {'qt': qt, 'case': case, 'seed': GEN_SEEDS, 'base_date': '2025-08-23', 'home': HOME_SCHEMA},",
        "                'llm': {'model': LLM_MODEL, 'api_base': LLM_API_BASE, 'api_key': api_key_value, 'temperature': 1}}",
        "        spec_path = f'{run_id}.yaml'",
        "        with open(spec_path, 'w') as f:",
        "            yaml.safe_dump(spec, f, sort_keys=False)",
        "        proc = subprocess.run([sys.executable, '-m', 'src.cli.episode_generator', '--spec', spec_path],",
        "                              capture_output=True, text=True)",
        "        n = len(glob.glob(f'{episodes_dir}/*.json'))",
        "        print(f'{run_id}: exit={proc.returncode} episodes_written={n}')",
        "        if proc.returncode != 0 or n == 0:",
        "            # Real cause is recorded per seed in run_state.json (the console tail is only progress bars).",
        "            state_path = f'../data/generated_qt4/{run_id}/run_state.json'",
        "            if os.path.exists(state_path):",
        "                seeds_state = json.load(open(state_path)).get('seeds', {})",
        "                errs = collections.Counter((v.get('error') or '')[:300] for v in seeds_state.values() if v.get('status') == 'failed')",
        "                for msg, cnt in errs.most_common(3):",
        "                    print(f'   {cnt} seeds failed with: {msg}')",
        "            else:",
        "                print('   no run_state.json written; console tail:', [l for l in (proc.stdout + proc.stderr).replace(chr(13), chr(10)).splitlines() if l.strip() and '%|' not in l][-8:])",
        "os.chdir(PROJECT_ROOT)",
    ),
    code(
        "import json, os, random",
        "",
        "if not os.path.exists('data/qt4_ir_v2_train.jsonl'):",
        "    native = extract_dataset('SimuHome/data/benchmark')",
        "    generated = extract_dataset('data/generated_qt4') if os.path.isdir('data/generated_qt4') else []",
        "    print(f'native={len(native)} generated={len(generated)}')",
        "",
        "    if generated:",
        "        # Clean split: train/val come ONLY from generated episodes; ALL native episodes are test.",
        "        trainable, held_out_test = generated, native",
        "    else:",
        "        # Generation produced nothing. Do NOT train on the whole native set (that contaminates E1).",
        "        # Hold out every 5th seed of the native set as test and train only on the rest.",
        "        print('WARNING: no generated episodes -- falling back to a seed-based split of the NATIVE set. '",
        "              'This is a small, weaker setup; fix generation (see the cell above) for the real run.')",
        "        held_out_test = [p for p in native if int(p['meta']['seed']) % 5 == 0]",
        "        trainable = [p for p in native if int(p['meta']['seed']) % 5 != 0]",
        "",
        "    random.seed(0)",
        "    random.shuffle(trainable)",
        "    n = len(trainable)",
        "    train, val = trainable[: int(n * 0.9)], trainable[int(n * 0.9):]",
        "    overlap = {p['query'] for p in train + val} & {p['query'] for p in held_out_test}",
        "    assert not overlap, f'{len(overlap)} test queries also appear in train/val -- split is contaminated'",
        "",
        "    os.makedirs('data', exist_ok=True)",
        "    for split_name, split_data in [('train', train), ('val', val), ('test_held_out', held_out_test)]:",
        "        with open(f'data/qt4_ir_v2_{split_name}.jsonl', 'w', encoding='utf-8') as f:",
        "            for row in split_data:",
        "                f.write(json.dumps(row, ensure_ascii=False) + '\\n')",
        "    print(f'train={len(train)} val={len(val)} held_out_test={len(held_out_test)}')",
        "else:",
        "    print('data/qt4_ir_v2_train.jsonl already exists, skipping extraction')",
    ),
    md(
        "## 5. LoRA fine-tune the intent parser",
        "",
        "**Open items — decide from this run, don't guess upfront:** start "
        "with Qwen3-1.7B (fits a T4 with LoRA + bf16); move to 4B only if "
        "1.7B underfits the IR-JSON task. Checkpointing is mandatory, not "
        "optional, per the handoff — a session can be killed mid-run with "
        "no warning.",
    ),
    code(
        "import json",
        "",
        "def load_jsonl(path):",
        "    with open(path, encoding='utf-8') as f:",
        "        return [json.loads(line) for line in f]",
        "",
        "train_rows = load_jsonl('data/qt4_ir_v2_train.jsonl')",
        "val_rows = load_jsonl('data/qt4_ir_v2_val.jsonl')",
        "print(len(train_rows), 'train', len(val_rows), 'val')",
    ),
    code(
        "from datasets import Dataset",
        "",
        "SYSTEM_PROMPT = (",
        "    'You convert a smart-home scheduling request into a strict JSON '",
        "    'intermediate representation (IR). Output ONLY the JSON object, '",
        "    'no prose, no code fences.'",
        ")",
        "",
        "def format_example(row):",
        "    target = json.dumps(row['ir_for_training'], ensure_ascii=False)",
        "    return {'text': f\"<|system|>\\n{SYSTEM_PROMPT}\\n<|user|>\\n{row['query']}\\n<|assistant|>\\n{target}\"}",
        "",
        "train_ds = Dataset.from_list([format_example(r) for r in train_rows])",
        "val_ds = Dataset.from_list([format_example(r) for r in val_rows])",
    ),
    code(
        "MODEL_NAME = 'Qwen/Qwen3-1.7B'  # bump to Qwen/Qwen3-4B only if this underfits",
        "ADAPTER_DIR = 'qwen3_ir_parser_adapter_v2'",
        "",
        "import torch, os",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)",
        "if tokenizer.pad_token is None:",
        "    tokenizer.pad_token = tokenizer.eos_token",
        "",
        "# Reuse a previously trained adapter if you attached one as a Kaggle Dataset (skips retraining).",
        "import glob, shutil",
        "if not os.path.isdir(ADAPTER_DIR):",
        "    for cand in glob.glob('/kaggle/input/*/qwen3_ir_parser_adapter_v2') + glob.glob('/kaggle/input/*/*/qwen3_ir_parser_adapter_v2'):",
        "        shutil.copytree(cand, ADAPTER_DIR)",
        "        print('reusing adapter from', cand)",
        "        break",
        "",
        "if os.path.isdir(ADAPTER_DIR):",
        "    print(f'{ADAPTER_DIR} already exists -- loading the fine-tuned adapter instead of retraining')",
        "    from peft import PeftModel",
        "    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={'': 0})",
        "    intent_parser = PeftModel.from_pretrained(base_model, ADAPTER_DIR)",
        "else:",
        "    intent_parser = None  # trained in the next cell",
    ),
    code(
        "if intent_parser is None:",
        "    from peft import LoraConfig, get_peft_model",
        "    from trl import SFTTrainer, SFTConfig",
        "    import glob",
        "",
        "    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map={'': 0})",
        "    lora_config = LoraConfig(",
        "        r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',",
        "        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],",
        "    )",
        "    # NOT pre-wrapped with get_peft_model: TRL applies LoRA itself via peft_config. Pre-wrapping makes",
        "    # model.forward a functools.partial and trl's chunked-loss patch then crashes on `__func__`.",
        "",
        "    import inspect",
        "    OUT_DIR = 'checkpoints_v2'",
        "    # trl renamed max_seq_length -> max_length in newer releases; use whichever this install has.",
        "    _len_kw = 'max_length' if 'max_length' in inspect.signature(SFTConfig.__init__).parameters else 'max_seq_length'",
        "    sft_config = SFTConfig(",
        "        output_dir=OUT_DIR, per_device_train_batch_size=4, gradient_accumulation_steps=4,",
        "        num_train_epochs=3, learning_rate=2e-4, logging_steps=10,",
        "        save_strategy='steps', save_steps=14, save_total_limit=3,",
        "        eval_strategy='steps', eval_steps=14, bf16=True, report_to=[],",
        "        dataset_text_field='text', **{_len_kw: 1024},",
        "        loss_type='nll',  # trl's default 'chunked_nll' patches model.forward and crashes here",
        "    )",
        "    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_ds, eval_dataset=val_ds, peft_config=lora_config)",
        "",
        "    existing_ckpts = sorted(glob.glob(f'{OUT_DIR}/checkpoint-*'))",
        "    resume_from = existing_ckpts[-1] if existing_ckpts else None",
        "    print('Resuming from', resume_from) if resume_from else print('Starting fresh run')",
        "    trainer.train(resume_from_checkpoint=resume_from)",
        "    json.dump(trainer.state.log_history, open('train_log.json', 'w'), indent=2)   # per-step loss / eval loss / token accuracy, for the paper",
        "",
        "    model = trainer.model   # the LoRA-wrapped model (the local `model` is still the bare base model)",
        "    model.print_trainable_parameters()",
        "    model.save_pretrained(ADAPTER_DIR)",
        "    tokenizer.save_pretrained(ADAPTER_DIR)",
        "    intent_parser = model",
        "    print('Saved adapter to', ADAPTER_DIR)",
        "intent_parser.eval()",
    ),
    code(
        "def parse_intent(query: str, max_new_tokens=700) -> dict:",
        "    prompt = f\"<|system|>\\n{SYSTEM_PROMPT}\\n<|user|>\\n{query}\\n<|assistant|>\\n\"",
        "    inputs = tokenizer(prompt, return_tensors='pt').to(intent_parser.device)",
        "    out = intent_parser.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)",
        "    text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)",
        "    return json.loads(text)",
        "",
        "",
        "# Parser quality, three ways (exact match alone is too harsh -- one extra field fails it):",
        "#  valid JSON | exact match | VERDICT accuracy = does Z3 reach the right feasible/infeasible",
        "#  answer from the predicted IR? Verdict is what actually matters for the verifier.",
        "from irlib.ir_schema import ir_from_compact",
        "from irlib.constraint_encoder import check_feasibility",
        "",
        "def eval_parser(rows, limit=60, parse_fn=None):",
        "    stats = dict(n=0, valid=0, exact=0, verdict_ok=0, verdict_n=0)",
        "    failures = []",
        "    for row in rows[:limit]:",
        "        stats['n'] += 1",
        "        try:",
        "            pred = (parse_fn or parse_intent)(row['query'])",
        "            stats['valid'] += 1",
        "        except Exception as e:",
        "            failures.append((row['query'], 'invalid JSON'))",
        "            continue",
        "        stats['exact'] += pred == row['ir_for_training']",
        "        gold = row['debug_ground_truth']",
        "        if gold['conflict_type'] == 'completion_vs_pause':",
        "            continue  # needs a duration model; Z3 cannot judge these",
        "        # Give device-event anchors the simulator-known time from the gold IR (as the runtime would).",
        "        try:",
        "            for gi, g in enumerate(pred['goals']):",
        "                for ri, r in enumerate(g['readings']):",
        "                    if r['anchor']['kind'] == 'device_event':",
        "                        r['resolved_at_minutes'] = gold['goals'][gi]['readings'][ri].get('resolved_at_minutes')",
        "            verdict = check_feasibility(ir_from_compact(pred, row['query'], gold['base_time'])).feasible",
        "            stats['verdict_n'] += 1",
        "            stats['verdict_ok'] += verdict == (row['meta']['case'] == 'feasible')",
        "            if verdict != (row['meta']['case'] == 'feasible'):",
        "                failures.append((row['query'], f\"wrong verdict: predicted {'feasible' if verdict else 'infeasible'}, gold {row['meta']['case']}\"))",
        "        except Exception as e:",
        "            failures.append((row['query'], f'bad structure: {type(e).__name__}'))",
        "    return stats, failures",
        "",
        "val_stats, val_failures = eval_parser(val_rows)",
        "print(val_stats)",
        "print(f\"valid JSON {val_stats['valid']}/{val_stats['n']} | exact {val_stats['exact']}/{val_stats['n']} | \"",
        "      f\"verdict accuracy {val_stats['verdict_ok']}/{val_stats['verdict_n']}\")",
        "for q, why in val_failures[:5]:",
        "    print(' -', why, '|', q[:80])",
        "",
        "# Same metric on the HELD-OUT native test episodes (never trained on) -- this is the number to report.",
        "test_rows = load_jsonl('data/qt4_ir_v2_test_held_out.jsonl')",
        "test_stats, test_failures = eval_parser(test_rows, limit=100)",
        "print('HELD-OUT', test_stats)",
        "import json as _json",
        "_json.dump({'val': val_stats, 'test': test_stats}, open('parser_eval.json', 'w'), indent=2)",
    ),
    md(
        "### Parser ablations: few-shot (no fine-tuning) baseline, and error analysis",
        "",
        "**Few-shot baseline.** The *same* Qwen3-1.7B with the LoRA adapter switched off, shown 4 training examples in the prompt. If this scores close to the fine-tuned parser, fine-tuning is not buying much; if it is far below, fine-tuning is justified.",
        "",
        "**Error analysis.** Every held-out query the fine-tuned parser got wrong (invalid JSON, bad structure, or the wrong feasible/infeasible verdict) is saved to `parser_errors.json`, so you can read them and describe the failure types in the paper.",
    ),
    code(
        "try:",
        "    def fewshot_parse(query, k=4, max_new_tokens=700):",
        "        feas = [r for r in train_rows if r['meta']['case'] == 'feasible'][:k // 2]",
        "        infe = [r for r in train_rows if r['meta']['case'] == 'infeasible'][:k // 2]",
        "        shots = [x for pair in zip(feas, infe) for x in pair]",
        "        prompt = f\"<|system|>\\n{SYSTEM_PROMPT}\\n\"",
        "        for r in shots:",
        "            prompt += f\"<|user|>\\n{r['query']}\\n<|assistant|>\\n{json.dumps(r['ir_for_training'], ensure_ascii=False)}\\n\"",
        "        prompt += f\"<|user|>\\n{query}\\n<|assistant|>\\n\"",
        "        inputs = tokenizer(prompt, return_tensors='pt').to(intent_parser.device)",
        "        with intent_parser.disable_adapter():   # the untouched base model",
        "            out = intent_parser.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,",
        "                                         stop_strings=['<|user|>'], tokenizer=tokenizer)",
        "        text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()",
        "        return json.JSONDecoder().raw_decode(text)[0]   # tolerate trailing text after the JSON object",
        "",
        "    fs_stats, fs_failures = eval_parser(test_rows, limit=100, parse_fn=fewshot_parse)",
        "    print('FEW-SHOT (no fine-tuning), held-out:', fs_stats)",
        "    print(f\"valid JSON {fs_stats['valid']}/{fs_stats['n']} | verdict accuracy {fs_stats['verdict_ok']}/{fs_stats['verdict_n']}   \"",
        "          f\"(fine-tuned: {test_stats['valid']}/{test_stats['n']} valid, {test_stats['verdict_ok']}/{test_stats['verdict_n']} verdicts)\")",
        "",
        "    _pe = json.load(open('parser_eval.json'))",
        "    _pe['fewshot_test'] = fs_stats",
        "    json.dump(_pe, open('parser_eval.json', 'w'), indent=2)",
        "",
        "    json.dump({'held_out_failures': [{'query': q, 'why': w} for q, w in test_failures],",
        "               'fewshot_failures_count': len(fs_failures)}, open('parser_errors.json', 'w'), indent=2, ensure_ascii=False)",
        "    print()",
        "    print(f'Fine-tuned parser: {len(test_failures)} held-out problems (saved to parser_errors.json):')",
        "    for q, why in test_failures:",
        "        print(' -', why, '|', q[:110])",
        "except Exception as _e:   # must never abort an unattended run before E1",
        "    import traceback; traceback.print_exc(); print('parser ablation skipped (does not affect E1):', repr(_e)[:300])",
    ),
    md(
        "## 6. Wire the verified `schedule_workflow` into the ReAct loop",
        "",
        "Monkeypatches `TOOL_REGISTRY['schedule_workflow']` and `ReActAgent.run` "
        "rather than forking SimuHome's vendored source (CC BY-NC-ND 4.0).",
    ),
    code(
        "from irlib.constraint_encoder import check_feasibility",
        "from irlib.ir_schema import ir_from_compact",
        "# MUST be the same module objects SimuHome's evaluator imports (`src.agents...`). Importing them as",
        "# `SimuHome.src.agents...` creates a second copy, and patching that copy changes nothing.",
        "from src.agents.tools import tool_schedule_workflow, TOOL_REGISTRY, run_tool",
        "from src.agents.strategies.react_agent import ReActAgent",
        "",
        "def verbalize_rejection(result) -> str:",
        "    return 'I cannot schedule this as stated -- the timing is inconsistent: ' + result.explanation()",
        "",
        "def resolve_anchor_minutes(device_id: str) -> float | None:",
        "    resp = run_tool('get_attribute', {",
        "        'device_id': device_id, 'endpoint_id': 1,",
        "        'cluster_id': 'OperationalState', 'attribute_id': 'CountdownTime',",
        "    })",
        "    countdown_seconds = ((resp or {}).get('data') or {}).get('value')",
        "    return None if countdown_seconds is None else float(countdown_seconds) / 60.0",
        "",
        "_last_user_query = {'text': None}",
        "",
        "import time as _time",
        "VERIFIER_STATS = {'calls': 0, 'no_query': 0, 'parse_fail': 0, 'passed': 0, 'rejected': 0, 'seconds': 0.0}   # seconds = total verify time (parse + Z3)",
        "VERIFIER_LOG = []  # (query, verdict, explanation) for every call -- inspect after the run",
        "",
        "def verified_schedule_workflow(args):",
        "    VERIFIER_STATS['calls'] += 1",
        "    query = _last_user_query['text']",
        "    if not query:",
        "        VERIFIER_STATS['no_query'] += 1",
        "        return tool_schedule_workflow(args)",
        "    try:",
        "        ir_dict = parse_intent(query)",
        "    except Exception as e:",
        "        VERIFIER_STATS['parse_fail'] += 1",
        "        VERIFIER_LOG.append((query[:100], 'parse_fail', repr(e)[:100]))",
        "        return tool_schedule_workflow(args)",
        "    now_resp = run_tool('get_current_time', {})",
        "    base_time = (now_resp.get('data') or {}).get('now')",
        "    for goal in ir_dict['goals']:",
        "        for reading in goal['readings']:",
        "            if reading['anchor']['kind'] == 'device_event' and reading.get('resolved_at_minutes') is None:",
        "                reading['resolved_at_minutes'] = resolve_anchor_minutes(reading['anchor']['device_id'])",
        "    ir = ir_from_compact(ir_dict, query, base_time)",
        "    result = check_feasibility(ir)",
        "    VERIFIER_LOG.append((query[:100], 'passed' if result.feasible else 'rejected', result.explanation()[:200]))",
        "    if result.feasible:",
        "        VERIFIER_STATS['passed'] += 1",
        "        return tool_schedule_workflow(args)",
        "    VERIFIER_STATS['rejected'] += 1",
        "    return {",
        "        'status': {'code': 409, 'message': 'INFEASIBLE_SCHEDULE'},",
        "        'data': None,",
        "        'error': {'type': 'TEMPORAL_CONTRADICTION', 'detail': verbalize_rejection(result)},",
        "    }",
        "",
        "_verified_impl = verified_schedule_workflow",
        "def verified_schedule_workflow(args):   # same function, timed (the latency overhead the paper should report)",
        "    _t0 = _time.time()",
        "    try:",
        "        return _verified_impl(args)",
        "    finally:",
        "        VERIFIER_STATS['seconds'] += _time.time() - _t0",
        "",
        "TOOL_REGISTRY['schedule_workflow'] = verified_schedule_workflow",
        "",
        "_original_run = ReActAgent.run",
        "# Prompt-only comparison arm (E1 'selfcheck'): the SAME agent, no Z3, just told to check time consistency itself.",
        "SELF_CHECK = {'on': False}",
        "SELF_CHECK_PROMPT = (' IMPORTANT: before scheduling anything, work out the exact time of every requested action. If two parts of the request'",
        "                     ' give the same event two different times (a contradiction), do NOT call schedule_workflow; call finish and explain the conflict.')",
        "def _patched_run(self, user_query, **kwargs):",
        "    _last_user_query['text'] = user_query            # the verifier parses the ORIGINAL query",
        "    if SELF_CHECK['on']:",
        "        user_query = user_query + SELF_CHECK_PROMPT",
        "    return _original_run(self, user_query, **kwargs)",
        "ReActAgent.run = _patched_run",
    ),
    md(
        "### Verifier on/off switch",
        "",
        "Both E1 arms run in this process (see section 3). `set_verifier(False)` restores SimuHome's original `schedule_workflow` for the baseline arm; `set_verifier(True)` installs the verified one. It starts OFF.",
    ),
    code(
        "_orig_schedule = tool_schedule_workflow",
        "",
        "def set_verifier(on):",
        "    TOOL_REGISTRY['schedule_workflow'] = verified_schedule_workflow if on else _orig_schedule",
        "",
        "def set_selfcheck(on):",
        "    SELF_CHECK['on'] = on",
        "",
        "set_verifier(False)",
        "print('verifier is OFF until E1 turns it on for the verified arm')",
    ),
    md(
        "## 7. Full evaluation (E1) + ablations",
        "",
        "Reuse SimuHome's own scoring "
        "(`src/pipelines/episode_evaluation/qt4/{feasible,infeasible_1,infeasible_2,infeasible_3}.py`) "
        "so numbers are directly comparable to the paper's table. Run on the "
        "**held-out native episodes only** (`data/qt4_ir_v2_test_held_out.jsonl` "
        "lists exactly which ones — never the synthesized episodes used in "
        "fine-tuning).",
    ),
    md(
        "**Important:** every arm (baseline, selfcheck, verified) runs through `run_inproc` (section 3), which executes SimuHome's evaluator **inside this notebook process**. Never use SimuHome's `eval-start` CLI for E1: it starts a separate process, so the request shim and the verifier patch would silently not apply.",
    ),
    md(
        "### E1 overnight plan runner",
        "",
        "Edit **only the `PLAN` block** below. Each entry is `(sub-type, case, arm)`, run in the listed order, one run at a time:",
        "",
        "| arm | what runs |",
        "|---|---|",
        "| `baseline` | the agent alone |",
        "| `selfcheck` | the agent + a prompt-only instruction to check time consistency (no parser, no Z3) |",
        "| `verified` | the agent + parser + Z3 verifier |",
        "",
        "Safety features for an unattended run: finished runs are skipped; no new run **starts** once `BUDGET_HOURS` would be exceeded; a run that crashes is logged and skipped; two bad runs in a row (crash or API errors) stop the plan so it does not burn the night on a dead API; everything is saved after every run.",
    ),
    code(
        "import json, os, collections, shutil, time",
        "os.chdir(os.path.join(PROJECT_ROOT, 'SimuHome'))",
        "",
        "# Seeds come from the held-out file, so E1 only scores episodes the parser never trained on.",
        "held_out = [json.loads(l) for l in open(os.path.join(PROJECT_ROOT, 'data/qt4_ir_v2_test_held_out.jsonl'), encoding='utf-8')]",
        "seeds_by_cell = collections.defaultdict(list)",
        "for row in held_out:",
        "    seeds_by_cell[(row['meta']['query_type'], row['meta']['case'])].append(int(row['meta']['seed']))",
        "",
        "# ============================ EDIT THIS BLOCK ONLY ============================",
        "E1_MAX_SEEDS = 10      # episodes per (sub-type, case) per arm (the held-out split has about 10 per cell)",
        "BUDGET_HOURS = 6.5     # never START a run if it could push this cell past this many hours",
        "PLAN = [               # (sub-type, case, arm), in priority order; about 1 hour per run at 10 episodes",
        "    ('qt4-1', 'infeasible', 'selfcheck'),",
        "    ('qt4-2', 'infeasible', 'baseline'),",
        "    ('qt4-2', 'infeasible', 'selfcheck'),",
        "    ('qt4-2', 'infeasible', 'verified'),",
        "    ('qt4-2', 'feasible',   'baseline'),",
        "    ('qt4-2', 'feasible',   'verified'),",
        "    ('qt4-2', 'feasible',   'selfcheck'),",
        "    # Later session (hardest sub-type):",
        "    # ('qt4-3', 'infeasible', 'baseline'), ('qt4-3', 'infeasible', 'selfcheck'), ('qt4-3', 'infeasible', 'verified'),",
        "]",
        "# ==============================================================================",
        "ARMS = {'baseline': (False, False), 'selfcheck': (False, True), 'verified': (True, False)}   # (verifier_on, selfcheck_on)",
        "",
        "def seed_spec(qt, case):",
        "    return ','.join(str(s) for s in sorted(seeds_by_cell[(qt, case)])[:E1_MAX_SEEDS])",
        "",
        "def run_done(run_id):",
        "    # complete = summary exists, episodes were evaluated, and NO infra errors (quota/429/rejected requests)",
        "    if not os.path.exists(f'experiments/{run_id}/run_summary.json'):",
        "        return False",
        "    agg = aggregate_run(run_id)",
        "    return bool(agg) and all(a['overall']['evaluated_total'] > 0 and a['overall']['infra_errors'] == 0 for a in agg.values())",
        "",
        "agg_file = os.path.join(PROJECT_ROOT, 'e1_aggregates.json')",
        "log_file = os.path.join(PROJECT_ROOT, 'e1_run_log.json')",
        "aggregates = json.load(open(agg_file)) if os.path.exists(agg_file) else {}",
        "run_log = json.load(open(log_file)) if os.path.exists(log_file) else []",
        "",
        "AGENT_SMOKE_TEST = True   # one real episode through the actual agent before the plan (a few minutes); stops the run early if the model cannot follow the ReAct format",
        "if AGENT_SMOKE_TEST and PLAN:",
        "    set_verifier(False); set_selfcheck(False)",
        "    _seed = (sorted(seeds_by_cell[('qt4-1', 'feasible')]) or [1])[0]",
        "    run_inproc('smoke_agent_check', 'qt4-1', 'feasible', str(_seed))",
        "    _o = list(aggregate_run('smoke_agent_check').values())[0]['overall']",
        "    print(f\"agent smoke test: evaluated={_o['evaluated_total']} infra_errors={_o['infra_errors']} schema_errors={_o['schema_errors']} (calls so far {LLM_CALLS['count']})\")",
        "    if _o['evaluated_total'] == 0 or _o['infra_errors'] or _o['schema_errors']:",
        "        raise RuntimeError('agent smoke test failed (see the counts above and the evaluator output): the model cannot run the ReAct loop here; change LLM_MODEL or fix the API error before spending hours on E1')",
        "",
        "T0, slowest, bad_streak = time.time(), 0.0, 0",
        "for qt, case, system in PLAN:",
        "    run_id = f'e1_{system}_{qt}_{case}'.replace('-', '_')",
        "    if run_done(run_id):",
        "        print('already complete:', run_id); continue",
        "    if not seed_spec(qt, case):",
        "        print('no held-out seeds for', qt, case); continue",
        "    if LLM_CALL_CAP and LLM_CALLS['count'] + 150 > LLM_CALL_CAP:",
        "        print(f'call cap nearly reached ({LLM_CALLS[\"count\"]}/{LLM_CALL_CAP}); stopping before {run_id}'); break",
        "    if (time.time() - T0 + slowest) / 3600 > BUDGET_HOURS:",
        "        print(f'time budget reached ({(time.time() - T0) / 3600:.1f} h used); stopping before {run_id}'); break",
        "    shutil.rmtree(f'experiments/{run_id}', ignore_errors=True)   # an earlier attempt crashed or hit a quota/infra error",
        "    verifier_on, selfcheck_on = ARMS[system]",
        "    set_verifier(verifier_on); set_selfcheck(selfcheck_on)",
        "    calls_before, stats_before, t_run = LLM_CALLS['count'], dict(VERIFIER_STATS), time.time()",
        "    entry = {'run_id': run_id, 'system': system, 'cell': f'{qt}|{case}', 'model': LLM_MODEL, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}",
        "    try:",
        "        run_inproc(run_id, qt, case, seed_spec(qt, case))",
        "        per_model = aggregate_run(run_id)",
        "        aggregates[f'{system}|{qt}|{case}'] = per_model",
        "        infra = 0",
        "        for m, a in per_model.items():",
        "            o = a['overall']; infra += o['infra_errors']",
        "            print(f'{system:9s} {qt} {case:10s} accuracy={o[\"accuracy\"]:.2f} ({o[\"correct\"]}/{o[\"evaluated_total\"]}) infra_errors={o[\"infra_errors\"]} schema_errors={o[\"schema_errors\"]}')",
        "            entry.update(accuracy=o['accuracy'], correct=o['correct'], evaluated=o['evaluated_total'], infra_errors=o['infra_errors'], schema_errors=o['schema_errors'])",
        "        bad_streak = bad_streak + 1 if (infra or entry.get('schema_errors', 0) > 3) else 0",
        "        if infra:",
        "            print('   -> API errors in this run (quota?); it will be redone next time.')",
        "    except Exception as e:",
        "        print('RUN FAILED:', run_id, repr(e)[:300]); entry['error'] = repr(e)[:300]; bad_streak += 1",
        "    entry.update(seconds=round(time.time() - t_run), llm_calls=LLM_CALLS['count'] - calls_before,",
        "                 verifier={k: round(VERIFIER_STATS[k] - stats_before[k], 2) for k in VERIFIER_STATS})",
        "    slowest = max(slowest, entry['seconds'])",
        "    run_log.append(entry)",
        "    print('   ', entry['seconds'] // 60, 'min |', entry['llm_calls'], 'LLM calls | verifier:', entry['verifier'])",
        "    json.dump(aggregates, open(agg_file, 'w'), indent=2)   # saved after EVERY run",
        "    json.dump(run_log, open(log_file, 'w'), indent=2)",
        "    json.dump({'stats': VERIFIER_STATS, 'log': VERIFIER_LOG}, open(os.path.join(PROJECT_ROOT, 'verifier_telemetry.json'), 'w'), indent=2)",
        "    if bad_streak >= 2:",
        "        print('two bad runs in a row (API errors or crashes): stopping the plan. Check the provider message above.'); break",
        "set_verifier(False); set_selfcheck(False)",
        "print(f'plan finished after {(time.time() - T0) / 3600:.1f} h')",
        "os.chdir(PROJECT_ROOT)",
    ),
    md(
        "`e1_aggregates.json` holds SimuHome's own per-episode aggregate for every run, and "
        "`verifier_telemetry.json` shows how many times the verifier parsed, passed, "
        "rejected, or failed to parse. Download both: the success-rate table is built "
        "from those files, not from `run_summary.json` (whose `totals` only says "
        "whether each *model run* finished, which is why the earlier table was all 1.0).",
    ),
    md(
        "### Results table and download bundle",
        "",
        "Prints every finished run and writes `overnight_results.zip` to `/kaggle/working/`. In a *Save & Run All* session the zip appears under the notebook's **Output** tab.",
    ),
    code(
        "import csv, zipfile, glob",
        "os.chdir(PROJECT_ROOT)",
        "rows = []",
        "for key, per_model in json.load(open('e1_aggregates.json')).items():",
        "    system, qt, case = key.split('|')",
        "    for m, a in per_model.items():",
        "        o = a['overall']",
        "        rows.append([qt, case, system, o['correct'], o['evaluated_total'], round(o['accuracy'], 3), o['infra_errors'], o['schema_errors']])",
        "rows.sort()",
        "print(f\"{'sub-type':8s} {'case':10s} {'arm':10s} correct/n  acc   infra schema\")",
        "for r in rows:",
        "    print(f'{r[0]:8s} {r[1]:10s} {r[2]:10s} {r[3]:>3}/{r[4]:<3}   {r[5]:.2f}  {r[6]:>3}   {r[7]:>3}')",
        "with open('e1_summary.csv', 'w', newline='') as f:",
        "    w = csv.writer(f); w.writerow(['sub_type', 'case', 'arm', 'correct', 'evaluated', 'accuracy', 'infra_errors', 'schema_errors']); w.writerows(rows)",
        "",
        "OUT = '/kaggle/working/overnight_results.zip'",
        "keep = ['e1_aggregates.json', 'e1_run_log.json', 'e1_summary.csv', 'verifier_telemetry.json', 'parser_eval.json', 'parser_errors.json', 'train_log.json',",
        "        'data/qt4_ir_v2_test_held_out.jsonl', 'qwen3_ir_parser_adapter_v2'] + glob.glob('SimuHome/experiments/e1_*') + glob.glob('SimuHome/experiments/baseline_*')",
        "with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:",
        "    for pth in keep:",
        "        if os.path.isfile(pth):",
        "            z.write(pth)",
        "        elif os.path.isdir(pth):",
        "            for root, _, files in os.walk(pth):",
        "                for fn in files:",
        "                    z.write(os.path.join(root, fn))",
        "print(OUT, round(os.path.getsize(OUT) / 1e6, 1), 'MB')",
    ),
    md(
        "**Ablations (E2-E6, design once E1 is running):** SLM-only (no Z3), "
        "oracle-IR-vs-SLM-IR, verbalizer template-vs-raw-core, per-"
        "conflict-type breakdown, verify-step latency overhead. Same "
        "held-out set and scoring functions as E1, swap one component per "
        "ablation, save each to its own CSV under this notebook's directory.",
        "",
        "**When done:** click 'Save Version' so `/kaggle/working` (checkpoints, "
        "adapter, `data/`, result CSVs) is versioned — that's what should "
        "come back into this project's local directory.",
    ),
]
save("00_full_pipeline.ipynb", nb_full)

print("\nAll notebooks written to", NB_DIR)
