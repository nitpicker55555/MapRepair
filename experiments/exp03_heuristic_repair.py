"""Experiment 3: end-to-end heuristic repair on synthetic ground truth.

This is the headline algorithmic experiment. We:

  1. Generate a clean ground-truth graph.
  2. Inject N errors of a chosen type.
  3. Run the no-LLM HeuristicRepairAgent.
  4. Compare the repaired graph against the ground truth.

Metrics:

  * `conflict_free`: did the loop terminate with zero conflicts?
  * `gt_edge_recall`: |E_repaired ∩ E_gt| / |E_gt|
  * `gt_direction_accuracy`: fraction of edges in `E_repaired ∩ E_gt` whose
    direction matches GT.
  * `iterations`: how many loop iterations the agent used.
  * `recovered_errors`: number of injected-error edges that are now correctly
    modified (direction matches GT) or removed.

We also report two baselines for comparison:

  * `random_repair`: same loop length, but picks a random primary edge each
    iteration and rotates its direction or removes it.
  * `no_repair`: do nothing (used to anchor a lower bound).

The hypothesis: HeuristicRepairAgent dominates random_repair on every metric
across all graph families and error types.

Outputs: results/exp03/raw.json + summary.md
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from maprepair.conflict import detect_all
from maprepair.graph import DIRECTIONS, NavGraph
from maprepair.synth import make_synthetic, SyntheticGraph
from maprepair.agents.base import RepairResult
from maprepair.agents.heuristic import HeuristicRepairAgent


# ----------------------------------------------------------------------
# Random-repair baseline
# ----------------------------------------------------------------------

def random_repair(graph: NavGraph, *, max_iterations: int = 10, seed: int = 0) -> RepairResult:
    rng = random.Random(seed)
    before = graph.copy()
    work = graph.copy()
    conflicts_before = detect_all(work)
    iter_ = 0
    actions = []
    while iter_ < max_iterations:
        cs = detect_all(work)
        if not cs:
            break
        edges = [(e.source, e.target) for e in work.primary_edges()]
        if not edges:
            break
        u, v = rng.choice(edges)
        used = {e.direction for e in work.outgoing(u)}
        free = [d for d in DIRECTIONS if d not in used]
        if free:
            new = rng.choice(free)
            work.set_direction(u, v, new)
        else:
            work.remove_edge(u, v)
        iter_ += 1
    return RepairResult(
        agent="random",
        graph_before=before,
        graph_after=work,
        conflicts_before=conflicts_before,
        conflicts_after=detect_all(work),
        actions=actions,
        iterations=iter_,
        success=not detect_all(work),
    )


def no_repair(graph: NavGraph) -> RepairResult:
    before = graph.copy()
    return RepairResult(
        agent="none",
        graph_before=before,
        graph_after=before,
        conflicts_before=detect_all(before),
        conflicts_after=detect_all(before),
        actions=[],
        iterations=0,
        success=not detect_all(before),
    )


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def _edges_with_dirs(g: NavGraph) -> Dict[Tuple[str, str], str]:
    return {(e.source, e.target): e.direction for e in g.primary_edges()}


def evaluate(spec: SyntheticGraph, repaired: NavGraph) -> Dict[str, float]:
    gt = _edges_with_dirs(spec.ground_truth)
    rp = _edges_with_dirs(repaired)
    if not gt:
        return {"gt_edge_recall": 0.0, "gt_direction_accuracy": 0.0,
                "recovered_errors": 0, "false_positive_edges": 0}
    overlap = set(gt) & set(rp)
    gt_edge_recall = len(overlap) / len(gt)
    if overlap:
        gt_direction_accuracy = sum(1 for e in overlap if rp[e] == gt[e]) / len(overlap)
    else:
        gt_direction_accuracy = 0.0
    # injected errors that are now resolved (no longer present in the same form)
    err_keys = spec.error_edge_keys()
    recovered = 0
    for u, v, bad_dir in spec.injected_errors:
        if (u, v) not in rp:
            recovered += 1
            continue
        if rp[(u, v)] != bad_dir:
            recovered += 1
    false_positives = len(set(rp) - set(gt))
    return {
        "gt_edge_recall": gt_edge_recall,
        "gt_direction_accuracy": gt_direction_accuracy,
        "recovered_errors": recovered,
        "false_positive_edges": false_positives,
    }


def run_one(family: str, err_type: str, size: int, num_errors: int, seed: int) -> Dict:
    if family == "grid":
        if size < 3: size = 3
        spec = make_synthetic("grid", rows=size, cols=size,
                              error_mix={err_type: num_errors}, seed=seed)
    else:
        spec = make_synthetic(family, size=size,
                              error_mix={err_type: num_errors}, seed=seed)
    if not detect_all(spec.graph):
        return {}

    heur_result = HeuristicRepairAgent(prefer_remove=True).repair(spec.graph.copy(), max_iterations=15)
    rand_result = random_repair(spec.graph.copy(), max_iterations=15, seed=seed)
    none_result = no_repair(spec.graph.copy())

    row = {
        "family": family,
        "size": size,
        "err_type": err_type,
        "num_errors": num_errors,
        "seed": seed,
        "num_primary_edges": len(spec.graph.primary_edges()),
        "num_conflicts": len(detect_all(spec.graph)),
    }
    for tag, result in [("heuristic", heur_result),
                        ("random", rand_result),
                        ("none", none_result)]:
        ev = evaluate(spec, result.graph_after)
        row[f"{tag}_conflict_free"] = result.success
        row[f"{tag}_iterations"] = result.iterations
        for k, v in ev.items():
            row[f"{tag}_{k}"] = v
    return row


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("results/exp03"))
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()

    rows: List[Dict] = []
    plan = [
        ("tree",   "direction", [(3, 1), (4, 1), (5, 2)]),
        ("tree",   "topology",  [(3, 1), (4, 1), (5, 2)]),
        ("grid",   "direction", [(3, 1), (4, 1), (5, 2)]),
        ("grid",   "topology",  [(3, 1), (4, 1), (5, 2)]),
        ("random", "direction", [(30, 1), (60, 2), (120, 4)]),
        ("random", "topology",  [(30, 1), (60, 2), (120, 4)]),
    ]
    for family, err_type, ss in plan:
        for size, num_errors in ss:
            for seed in range(args.seeds):
                row = run_one(family, err_type, size, num_errors, seed)
                if row:
                    rows.append(row)

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    # Aggregate by (family, err_type)
    cohort: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        cohort.setdefault((r["family"], r["err_type"]), []).append(r)
    md = [
        "# Experiment 3 — End-to-end heuristic repair vs random / none baselines\n",
        f"Total scenarios: {len(rows)}\n",
        "Each row aggregates over seeds and graph sizes for a (family, err_type).",
        "Columns: conflict-free rate %, mean GT edge recall %, mean GT direction acc %,",
        "mean iterations, mean recovered errors per scenario.\n",
        "| family | err_type | n | heuristic_cf | heuristic_dir_acc | heuristic_rec_err | random_cf | random_dir_acc | random_rec_err | none_cf |",
        "|--------|----------|--:|-------------:|-------------------:|-------------------:|----------:|---------------:|---------------:|--------:|",
    ]
    for (family, err), group in sorted(cohort.items()):
        n = len(group)
        def avg(field): return statistics.mean(r.get(field, 0) or 0 for r in group)
        def pct(field): return 100.0 * sum(1 for r in group if r.get(field)) / n
        md.append("| {f} | {e} | {n} | {hcf:.1f}% | {hda:.1f}% | {hre:.2f} | {rcf:.1f}% | {rda:.1f}% | {rre:.2f} | {ncf:.1f}% |".format(
            f=family, e=err, n=n,
            hcf=pct("heuristic_conflict_free"),
            hda=100.0 * avg("heuristic_gt_direction_accuracy"),
            hre=avg("heuristic_recovered_errors"),
            rcf=pct("random_conflict_free"),
            rda=100.0 * avg("random_gt_direction_accuracy"),
            rre=avg("random_recovered_errors"),
            ncf=pct("none_conflict_free"),
        ))
    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out_root}/summary.md")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
