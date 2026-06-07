"""Experiment 1: scaled-up LCA candidate-filtering study.

Strengthens the paper's six hand-crafted TC1-TC6 cases by sampling a *large*
number of synthetic graphs across the three families (tree, grid, random) and
the three conflict types. For each (family, conflict_type, size, seed) we
compute:

  * |E|           - total primary edges in the broken graph
  * |E_LCA|       - candidate edges after LCA filtering, deduplicated across
                    all conflicts in the graph
  * reduction     - 1 - |E_LCA| / |E|
  * truth_in_lca  - fraction of injected error edges that are in the candidate
                    set (i.e. the algorithm did not miss the truth)
  * rank_of_truth - position of the highest-ranked injected error edge in the
                    score-ranked candidate list (lower is better; -1 if missed)

Outputs:
  results/exp01/raw.json
  results/exp01/summary.csv
  results/exp01/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from maprepair.conflict import detect_all
from maprepair.localizer import Localizer
from maprepair.scoring import score_edges
from maprepair.synth import make_synthetic, SyntheticGraph


def _run_single(family: str, size: int, err_type: str, seed: int) -> Optional[Dict]:
    error_mix: Dict[str, int] = {err_type: 1}
    if family == "grid":
        if size < 3:
            size = 3
        spec = make_synthetic("grid", rows=size, cols=size, error_mix=error_mix, seed=seed)
    else:
        spec = make_synthetic(family, size=size, error_mix=error_mix, seed=seed)
    conflicts = detect_all(spec.graph)
    if not conflicts:
        return None
    loc = Localizer()
    primary_set = {(e.source, e.target) for e in spec.graph.primary_edges()}
    cand_set: set = set()
    for c in conflicts:
        for e in loc.localize(spec.graph, c):
            if e in primary_set:
                cand_set.add(e)
    total = len(primary_set)
    if total == 0:
        return None
    reduction = 1.0 - (len(cand_set) / total)

    scored = score_edges(spec.graph, conflicts=conflicts)
    edge_rank = {s.edge: i for i, s in enumerate(scored)}

    err_keys = spec.error_edge_keys()
    truth_in_lca = sum(1 for e in err_keys if e in cand_set) / max(1, len(err_keys))
    ranks = [edge_rank[e] for e in err_keys if e in edge_rank]
    rank_of_truth = min(ranks) if ranks else -1

    return {
        "family": family,
        "size": size,
        "err_type": err_type,
        "seed": seed,
        "num_nodes": spec.graph.num_nodes(),
        "num_primary_edges": total,
        "num_candidates": len(cand_set),
        "reduction": reduction,
        "num_errors_injected": len(err_keys),
        "truth_in_lca_fraction": truth_in_lca,
        "rank_of_truth": rank_of_truth,
        "rank_of_truth_pct": (rank_of_truth / max(1, len(cand_set))) if rank_of_truth >= 0 else None,
        "num_conflicts": len(conflicts),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("results/exp01"))
    ap.add_argument("--seeds", type=int, default=30,
                    help="number of seeds per (family,size,err_type) triple")
    args = ap.parse_args()

    rows: List[Dict] = []
    families_sizes = [
        ("tree", [3, 4, 5, 6]),
        ("grid", [3, 4, 5, 6, 7]),
        ("random", [15, 30, 60, 120]),
    ]
    err_types = ["direction", "topology", "naming"]

    for family, sizes in families_sizes:
        for size in sizes:
            for err in err_types:
                for seed in range(args.seeds):
                    row = _run_single(family, size, err, seed)
                    if row is not None:
                        rows.append(row)

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    # summary.csv: per (family, size, err_type) aggregates
    keys = ["family", "size", "err_type"]
    agg: Dict[Tuple, List[Dict]] = {}
    for r in rows:
        k = tuple(r[k_] for k_ in keys)
        agg.setdefault(k, []).append(r)

    csv_path = args.out_root / "summary.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "family", "size", "err_type", "n", "mean_reduction",
            "median_reduction", "stdev_reduction",
            "mean_truth_in_lca", "median_rank_of_truth",
            "mean_rank_pct", "median_rank_pct",
        ])
        for k, group in sorted(agg.items()):
            reductions = [r["reduction"] for r in group]
            t_in_lca = [r["truth_in_lca_fraction"] for r in group]
            ranks = [r["rank_of_truth"] for r in group if r["rank_of_truth"] >= 0]
            rank_pcts = [r["rank_of_truth_pct"] for r in group if r["rank_of_truth_pct"] is not None]
            w.writerow([
                k[0], k[1], k[2], len(group),
                f"{statistics.mean(reductions):.3f}",
                f"{statistics.median(reductions):.3f}",
                f"{statistics.pstdev(reductions):.3f}" if len(reductions) > 1 else "0.000",
                f"{statistics.mean(t_in_lca):.3f}",
                f"{statistics.median(ranks):.0f}" if ranks else "-",
                f"{statistics.mean(rank_pcts):.3f}" if rank_pcts else "-",
                f"{statistics.median(rank_pcts):.3f}" if rank_pcts else "-",
            ])

    # write a quick markdown summary
    overall = {
        "n": len(rows),
        "mean_reduction": statistics.mean(r["reduction"] for r in rows),
        "mean_truth_in_lca": statistics.mean(r["truth_in_lca_fraction"] for r in rows),
    }
    md = [
        "# Experiment 1 — Scaling localization study\n",
        f"Total scenarios run: **{overall['n']}**",
        f"Mean LCA candidate-set reduction: **{overall['mean_reduction']*100:.2f}%**",
        f"Mean fraction of injected truth retained in LCA set: "
        f"**{overall['mean_truth_in_lca']*100:.2f}%**",
        "\n## Per-cohort summary (see summary.csv for details)\n",
    ]
    # break out by err_type
    by_err: Dict[str, List[Dict]] = {}
    for r in rows:
        by_err.setdefault(r["err_type"], []).append(r)
    md.append("| err_type | n | mean_reduction | mean_truth_in_lca | median_rank_of_truth |")
    md.append("|----------|---|----------------|--------------------|----------------------|")
    for err, group in sorted(by_err.items()):
        red = statistics.mean(r["reduction"] for r in group)
        t = statistics.mean(r["truth_in_lca_fraction"] for r in group)
        ranks = [r["rank_of_truth"] for r in group if r["rank_of_truth"] >= 0]
        med_rank = statistics.median(ranks) if ranks else "-"
        md.append(f"| {err} | {len(group)} | {red*100:.2f}% | {t*100:.2f}% | {med_rank} |")
    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out_root}/summary.md")
    print("\n" + "\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
