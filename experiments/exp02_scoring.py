"""Experiment 2: edge impact scoring vs actual cascade size.

The paper's cascade-correlation experiment uses a single 36-edge graph with
5 hand-tuned chains. Here we test the same hypothesis at scale:

  1. Generate many random navigation graphs.
  2. For each generated graph, pick a *single* edge and treat it as the error.
     The actual cascade is the count of nodes that become unreachable from the
     graph root when that edge is removed (a more conservative definition
     than the paper's "rooms that overlap downstream").
  3. Compute the predicted impact score for that edge BEFORE injecting an
     error (i.e. on the clean graph). The hypothesis is: edges with higher
     impact score have larger downstream cascades.
  4. Report Spearman and Kendall correlations between predicted score and
     actual cascade across all sampled edges.

This generalises the paper's ρ=1.0 result, which was on a hand-crafted setup;
we want to see how the correlation behaves across families and densities.

Outputs:
  results/exp02/raw.json
  results/exp02/summary.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
from scipy.stats import spearmanr, kendalltau

from maprepair.scoring import score_edges
from maprepair.synth import gen_tree, gen_grid, gen_random


def _cascade_size(graph_nx: nx.DiGraph, edge: Tuple[str, str], root: str) -> int:
    """Count nodes that become unreachable from `root` when `edge` is severed."""
    if not graph_nx.has_edge(*edge):
        return 0
    g = graph_nx.copy()
    g.remove_edge(*edge)
    reachable_after = nx.descendants(g, root) | {root}
    reachable_before = nx.descendants(graph_nx, root) | {root}
    return len(reachable_before) - len(reachable_after)


def _sample_one(family: str, size: int, seed: int) -> Dict:
    if family == "tree":
        g = gen_tree(depth=size, branching=3, seed=seed)
    elif family == "grid":
        g = gen_grid(size, size)
    else:
        g = gen_random(num_nodes=size, branching=3, seed=seed)
    primaries = g.primary_edges()
    if not primaries:
        return {}
    scored = score_edges(g)
    # Determine the "root": for trees/random it's "r0" / "n0"; for grid it's (0,0)
    if family == "tree":
        root = "r0"
    elif family == "grid":
        root = "(0,0)"
    else:
        root = "n0"
    if root not in g.nx:
        return {}
    cascades: List[int] = []
    preds: List[float] = []
    for s in scored:
        u, v = s.edge
        cascade = _cascade_size(g.nx, (u, v), root)
        cascades.append(cascade)
        preds.append(s.score)
    return {
        "family": family,
        "size": size,
        "seed": seed,
        "num_edges": len(scored),
        "cascades": cascades,
        "scores": preds,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("results/exp02"))
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    out: List[Dict] = []
    cases = [("tree", s) for s in (3, 4, 5, 6)] + \
            [("grid", s) for s in (3, 4, 5)] + \
            [("random", s) for s in (30, 60, 120)]
    for family, size in cases:
        for seed in range(args.seeds):
            row = _sample_one(family, size, seed)
            if not row or len(row["scores"]) < 3:
                continue
            try:
                rho, _ = spearmanr(row["scores"], row["cascades"])
                tau, _ = kendalltau(row["scores"], row["cascades"])
            except Exception:
                rho = tau = float("nan")
            row["spearman"] = rho
            row["kendall"] = tau
            out.append(row)

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "raw.json").write_text(json.dumps(out, indent=2))

    rho_vals = [r["spearman"] for r in out if r.get("spearman") == r.get("spearman")]
    tau_vals = [r["kendall"] for r in out if r.get("kendall") == r.get("kendall")]
    md = [
        "# Experiment 2 — Edge impact scoring vs cascade size\n",
        f"Sampled {len(out)} graphs across tree / grid / random families.\n",
        f"Mean Spearman ρ: **{statistics.mean(rho_vals):.3f}** ± "
        f"{statistics.pstdev(rho_vals):.3f}",
        f"Mean Kendall τ:  **{statistics.mean(tau_vals):.3f}** ± "
        f"{statistics.pstdev(tau_vals):.3f}",
        "\n## Per-cohort (family × size)\n",
        "| family | size | n | mean ρ | mean τ | median ρ |",
        "|--------|-----:|--:|-------:|-------:|---------:|",
    ]
    cohort: Dict[Tuple[str, int], List[Dict]] = {}
    for r in out:
        cohort.setdefault((r["family"], r["size"]), []).append(r)
    for (family, size), rows in sorted(cohort.items()):
        rs = [r["spearman"] for r in rows if r["spearman"] == r["spearman"]]
        ts = [r["kendall"] for r in rows if r["kendall"] == r["kendall"]]
        if not rs:
            continue
        md.append(f"| {family} | {size} | {len(rows)} | "
                  f"{statistics.mean(rs):.3f} | "
                  f"{statistics.mean(ts):.3f} | "
                  f"{statistics.median(rs):.3f} |")
    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out_root}/summary.md")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
