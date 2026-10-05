"""Sanity check: extractor + Z3 encoder against real SimuHome QT4 episodes.

Run: python tests/test_pipeline.py [path/to/SimuHome/data/benchmark]

CPU-only, no GPU/API keys needed. This is the "confirm SAT/UNSAT results
match expectations manually before any model is involved" step from the
handoff (Sec 7, step 4), run against real benchmark data instead of
hand-crafted examples.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constraint_encoder import check_feasibility
from src.extract_ir_from_episode import extract_episode


def main(bench_dir: str) -> None:
    paths = sorted(glob.glob(os.path.join(bench_dir, "qt4-*.json")))
    assert paths, f"no qt4-*.json files found under {bench_dir}"

    counts = {
        "feasible_correct": 0,
        "feasible_wrong": 0,
        "infeasible_correct": 0,
        "infeasible_wrong": 0,
        "infeasible_skipped_duration_model": 0,
    }
    wrong_examples: list[str] = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            episode = json.load(f)

        ir = extract_episode(episode, for_training=False)
        result = check_feasibility(ir)
        expected_feasible = episode["meta"]["case"] == "feasible"

        skipped = any(g.reason and "requires a device duration model" in (g.reason or "") for g in result.goal_results)

        if not expected_feasible and skipped:
            counts["infeasible_skipped_duration_model"] += 1
            continue

        if result.feasible == expected_feasible:
            counts["feasible_correct" if expected_feasible else "infeasible_correct"] += 1
        else:
            counts["feasible_wrong" if expected_feasible else "infeasible_wrong"] += 1
            wrong_examples.append(
                f"{os.path.basename(path)}: expected_feasible={expected_feasible} "
                f"got_feasible={result.feasible} explanation={result.explanation()!r} "
                f"query={episode['query'][:80]!r}"
            )

    print(json.dumps(counts, indent=2))
    total_checked = sum(counts.values()) - counts["infeasible_skipped_duration_model"]
    total_correct = counts["feasible_correct"] + counts["infeasible_correct"]
    print(f"Accuracy (excluding duration-model-only conflicts): {total_correct}/{total_checked}")

    if wrong_examples:
        print(f"\n{len(wrong_examples)} mismatches (first 15):")
        for line in wrong_examples[:15]:
            print(" -", line)


if __name__ == "__main__":
    bench_dir = sys.argv[1] if len(sys.argv) > 1 else "../SimuHome/data/benchmark"
    main(bench_dir)
