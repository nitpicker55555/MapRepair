"""Experiment 6: difficulty curves — how does heuristic repair scale as the
number of injected errors grows?

For each (family, size) pair we sweep `num_errors` from 1 to 10 and run both
the HeuristicRepairAgent and a random-rotation baseline. We report:

  * `conflict_free_rate`     - % of scenarios that end with zero conflicts
  * `gt_direction_accuracy`  - mean directional correctness on the repaired graph
  * `recovered_errors`       - mean # of injected errors that the agent removed
                                or relabelled away from their bad direction

The hypothesis: heuristic recovery scales sub-linearly with error count while
random baseline collapses immediately. This is a falsifiable difficulty curve.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

from maprepair.conflict import detect_all
from maprepair.synth import make_synthetic
from maprepair.agents.heuristic import HeuristicRepairAgent
from experiments.exp03_heuristic_repair import random_repair, evaluate


def _run(family: str, size: int, err_type: str, num_errors: int, seed: int) -> Dict:
    if family == "grid":
        size = max(size, 3)
        spec = make_synthetic("grid", rows=size, cols=size,
                              error_mix={err_type: num_errors}, seed=seed)
    else:
        spec = make_synthetic(family, size=size,
                              error_mix={err_type: num_errors}, seed=seed)
    if not detect_all(spec.graph):
        return {}
    heur = HeuristicRepairAgent().repair(spec.graph.copy(), max_iterations=30)
    rand = random_repair(spec.graph.copy(), max_iterations=30, seed=seed)
    h_ev = evaluate(spec, heur.graph_after)
    r_ev = evaluate(spec, rand.graph_after)
    return {
        "family": family, "size": size, "err_type": err_type,
        "num_errors": num_errors, "seed": seed,
        "num_conflicts": len(detect_all(spec.graph)),
        "heur_conflict_free": heur.success,
        "heur_iters": heur.iterations,
        "heur_dir_acc": h_ev["gt_direction_accuracy"],
        "heur_recovered": h_ev["recovered_errors"],
        "rand_conflict_free": rand.success,
        "rand_iters": rand.iterations,
        "rand_dir_acc": r_ev["gt_direction_accuracy"],
        "rand_recovered": r_ev["recovered_errors"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("results/exp06"))
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()

    rows: List[Dict] = []
    plan = [
        ("tree",   5, "direction", list(range(1, 6))),
        ("tree",   5, "topology",  list(range(1, 6))),
        ("random", 60, "direction", list(range(1, 9))),
        ("random", 60, "topology",  list(range(1, 9))),
        ("grid",   5, "direction", list(range(1, 6))),
        ("grid",   5, "topology",  list(range(1, 6))),
    ]
    for family, size, err, n_list in plan:
        for n_err in n_list:
            for seed in range(args.seeds):
                row = _run(family, size, err, n_err, seed)
                if row:
                    rows.append(row)

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    # Per (family, err_type, num_errors) cohort
    cohort: Dict[Tuple[str, str, int], List[Dict]] = {}
    for r in rows:
        cohort.setdefault((r["family"], r["err_type"], r["num_errors"]), []).append(r)

    md = [
        "# Experiment 6 - Difficulty curves (heuristic vs random vs num_errors)\n",
        f"Total scenarios: {len(rows)}\n",
        "Cell values are mean over `seeds`. Use the CSV / raw JSON for distributions.\n",
        "| family | err_type | num_err | n | heur_cf% | heur_dir% | heur_rec | rand_cf% | rand_dir% | rand_rec |",
        "|--------|----------|--------:|--:|---------:|----------:|---------:|---------:|----------:|---------:|",
    ]
    for (family, err, n_err), group in sorted(cohort.items()):
        n = len(group)
        def avg(field): return statistics.mean(r.get(field, 0) or 0 for r in group)
        def pct(field): return 100 * sum(1 for r in group if r.get(field)) / n
        md.append(f"| {family} | {err} | {n_err} | {n} | "
                  f"{pct('heur_conflict_free'):.1f} | "
                  f"{100*avg('heur_dir_acc'):.1f} | "
                  f"{avg('heur_recovered'):.2f} | "
                  f"{pct('rand_conflict_free'):.1f} | "
                  f"{100*avg('rand_dir_acc'):.1f} | "
                  f"{avg('rand_recovered'):.2f} |")
    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out_root}/summary.md")
    print("\n".join(md[:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
