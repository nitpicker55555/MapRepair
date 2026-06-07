"""Experiment 16 harness: controlled-noise benchmark.

For each (game × regime × method × seed) we:
  1. Load the clean GT graph (data_fixed/<game>).
  2. Inject noise according to the regime config.
  3. Run a repair method on the noised copy.
  4. Compute metrics: conflict-free, GT recall/precision, node recall, etc.

Outputs (all under results/exp16/):
  raw.json              -- list of run records
  summary.md            -- aggregate tables (the headline)
  per_graph/<game>.md   -- per-game breakdowns
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from maprepair.agents.base import RepairResult
from maprepair.agents.heuristic import HeuristicRepairAgent
from maprepair.conflict import detect_all
from maprepair.graph import DIRECTIONS, NavGraph
from maprepair.synth import gen_grid, gen_random, gen_tree

from experiments.exp16_noise import REGIMES, apply_regime, decorate_with_prefixes, regime_by_name


DEFAULT_GRAPHS = ("grid_4x4", "grid_5x6", "tree_d3b3", "random_24", "random_40")
DEFAULT_REGIMES = ("edge_minimal", "edge_clean", "edge_heavy",
                    "node_only", "node_heavy", "mango_like")
DEFAULT_METHODS = ("no_repair", "random", "heuristic_remove", "heuristic_modify")
DEFAULT_SEEDS = (0, 1, 2)


def build_graph(name: str, seed: int = 0) -> NavGraph:
    """Make a synthetic graph and decorate with prefix-shared names so
    N5 node-collapse has realistic structure to attack."""
    if name == "grid_4x4":
        g = gen_grid(4, 4)
    elif name == "grid_5x6":
        g = gen_grid(5, 6)
    elif name == "tree_d3b3":
        g = gen_tree(depth=3, branching=3, seed=seed)
    elif name == "random_24":
        g = gen_random(24, branching=3, seed=seed)
    elif name == "random_40":
        g = gen_random(40, branching=4, seed=seed)
    else:
        raise ValueError(f"unknown graph: {name}")
    # decorate ~half the nodes with prefix-shared names
    n_nodes = g.num_nodes()
    decorate_with_prefixes(g, n_groups=max(2, n_nodes // 5),
                              members_per_group=3, seed=seed)
    return g


# ----------------------------------------------------------------------
# Repair methods
# ----------------------------------------------------------------------

def repair_no(work: NavGraph, *, seed: int = 0) -> RepairResult:
    before = work.copy()
    cb = detect_all(work)
    return RepairResult(agent="no_repair", graph_before=before, graph_after=work,
                        conflicts_before=cb, conflicts_after=cb, iterations=0,
                        success=not cb)


def repair_random(work: NavGraph, *, seed: int = 0, max_iterations: int = 50) -> RepairResult:
    import random
    rng = random.Random(seed)
    before = work.copy()
    cb = detect_all(work)
    iter_ = 0
    while iter_ < max_iterations:
        cs = detect_all(work)
        if not cs:
            break
        edges = [(e.source, e.target) for e in work.primary_edges()]
        if not edges:
            break
        u, v = rng.choice(edges)
        # rotate direction or remove
        used = {e.direction for e in work.outgoing(u)}
        free = [d for d in DIRECTIONS if d not in used]
        if free and rng.random() < 0.6:
            try:
                work.set_direction(u, v, rng.choice(free))
            except Exception:
                pass
        else:
            work.remove_edge(u, v)
            work.remove_edge(v, u)
        iter_ += 1
    return RepairResult(agent="random", graph_before=before, graph_after=work,
                        conflicts_before=cb, conflicts_after=detect_all(work),
                        iterations=iter_, success=not detect_all(work))


def repair_heuristic(work: NavGraph, *, seed: int = 0, prefer_remove: bool = True,
                       max_iterations: int = 50) -> RepairResult:
    agent = HeuristicRepairAgent(prefer_remove=prefer_remove)
    return agent.repair(work, max_iterations=max_iterations)


def run_method(method: str, work: NavGraph, *, seed: int = 0) -> RepairResult:
    if method == "no_repair":
        return repair_no(work, seed=seed)
    if method == "random":
        return repair_random(work, seed=seed)
    if method == "heuristic_remove":
        return repair_heuristic(work, seed=seed, prefer_remove=True)
    if method == "heuristic_modify":
        return repair_heuristic(work, seed=seed, prefer_remove=False)
    raise ValueError(f"unknown method: {method}")


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def _primary_pairs(g: NavGraph) -> Set[Tuple[str, str]]:
    return {(e.source, e.target) for e in g.primary_edges()}


def _primary_with_dir(g: NavGraph) -> Set[Tuple[str, str, str]]:
    return {(e.source, e.target, e.direction) for e in g.primary_edges()}


def _noise_breakdown(noise_recs: List[Dict]) -> Dict[str, int]:
    out = Counter(r["type"] for r in noise_recs)
    return dict(out)


def measure(gt: NavGraph, before: NavGraph, after: NavGraph,
             result: RepairResult, noise_recs: List[Dict]) -> Dict:
    gt_pairs = _primary_pairs(gt)
    gt_with_dir = _primary_with_dir(gt)
    gt_nodes = set(gt.nodes())

    b_pairs = _primary_pairs(before)
    a_pairs = _primary_pairs(after)
    a_with_dir = _primary_with_dir(after)
    a_nodes = set(after.nodes())

    def prf(pred: Set, gold: Set) -> Tuple[float, float, float]:
        if not gold or not pred:
            return 0.0, 0.0, 0.0
        tp = len(pred & gold)
        p = tp / len(pred); r = tp / len(gold)
        f = (2 * p * r) / (p + r) if (p + r) else 0.0
        return p, r, f

    bp, br, bf = prf(b_pairs, gt_pairs)
    ap, ar, af = prf(a_pairs, gt_pairs)
    _adp, adr, _adf = prf(a_with_dir, gt_with_dir)
    n_recall = len(a_nodes & gt_nodes) / max(1, len(gt_nodes))
    n_kept_after_noise = len(set(before.nodes()) & gt_nodes) / max(1, len(gt_nodes))

    cb = len(result.conflicts_before)
    ca = len(result.conflicts_after)

    actions = [a.kind for a in result.actions]
    action_counts = Counter(actions)

    return {
        "n_noise_injections": len(noise_recs),
        "n_conflicts_before": cb,
        "n_conflicts_after": ca,
        "conflict_free": (ca == 0),
        "conflict_reduction": (cb - ca) / max(1, cb),
        "gt_edge_recall_before_repair": br,
        "gt_edge_precision_before_repair": bp,
        "gt_edge_recall_after": ar,
        "gt_edge_precision_after": ap,
        "gt_edge_with_dir_recall_after": adr,
        "gt_node_recall_after": n_recall,
        "gt_node_recall_post_noise": n_kept_after_noise,
        "iterations": result.iterations,
        "actions_modify": action_counts.get("modify_edge", 0),
        "actions_remove": action_counts.get("remove_edge", 0),
        "actions_total": len(result.actions),
        "success": result.success,
        "noise_breakdown": _noise_breakdown(noise_recs),
    }


# ----------------------------------------------------------------------
# Single run
# ----------------------------------------------------------------------

def run_one(graph_name: str, regime_name: str, method: str, seed: int,
            gt_root: Path) -> Dict:
    # build synthetic GT graph fresh
    gt = build_graph(graph_name, seed=seed)
    work = gt.copy()
    # inject noise
    regime = regime_by_name(regime_name)
    recs = apply_regime(work, regime, seed=seed)
    noise_dicts = [r.to_dict() for r in recs]
    # repair
    start = time.time()
    result = run_method(method, work, seed=seed)
    elapsed = time.time() - start
    # measure — note: result.graph_before is *post-noise* state
    metrics = measure(gt, result.graph_before, result.graph_after, result, noise_dicts)
    metrics.update({
        "graph": graph_name, "regime": regime_name, "method": method, "seed": seed,
        "elapsed_sec": elapsed,
    })
    return metrics


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default=",".join(DEFAULT_GRAPHS))
    ap.add_argument("--regimes", default=",".join(DEFAULT_REGIMES))
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--out-root", type=Path, default=Path("results/exp16"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gt-root", type=Path,
                    default=Path("/Users/puzhen/Downloads/maprepair/data_fixed"))
    args = ap.parse_args()

    graphs = args.graphs.split(",")
    regimes = args.regimes.split(",")
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"graphs={graphs}\nregimes={regimes}\nmethods={methods}\nseeds={seeds}")
    n_total = len(graphs) * len(regimes) * len(methods) * len(seeds)
    print(f"Total runs: {n_total}")

    rows: List[Dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for g in graphs:
            for r in regimes:
                for m in methods:
                    for s in seeds:
                        futs[ex.submit(run_one, g, r, m, s, args.gt_root)] = (g, r, m, s)
        done = 0
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"graph": key[0], "regime": key[1], "method": key[2],
                        "seed": key[3], "error": str(e)}
            rows.append(row)
            done += 1
            if done % 20 == 0 or done == n_total:
                print(f"  [{done}/{n_total}]")

    (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))
    print(f"Wrote {args.out_root}/raw.json")

    # ---- aggregate
    by_regime_method: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        by_regime_method[(r["regime"], r["method"])].append(r)

    md = [
        "# Experiment 16 — controlled-noise benchmark\n",
        f"Graphs:  {graphs}",
        f"Regimes: {regimes}",
        f"Methods: {methods}",
        f"Seeds:   {seeds}",
        f"Total runs: {len(rows)} (errors: {sum(1 for r in rows if 'error' in r)})\n",
        "## Aggregate by regime × method\n",
        "| regime | method | n | conflict-free % | conf reduction % | GT edge recall after | GT edge prec after | GT node recall after | actions/run | iters/run |",
        "|--------|--------|--:|----------------:|-----------------:|---------------------:|-------------------:|---------------------:|------------:|----------:|",
    ]
    for regime in regimes:
        for method in methods:
            rs = by_regime_method.get((regime, method), [])
            if not rs:
                continue
            n = len(rs)
            cf = 100 * sum(1 for r in rs if r["conflict_free"]) / n
            cred = 100 * statistics.mean(r["conflict_reduction"] for r in rs)
            er = 100 * statistics.mean(r["gt_edge_recall_after"] for r in rs)
            ep = 100 * statistics.mean(r["gt_edge_precision_after"] for r in rs)
            nr = 100 * statistics.mean(r["gt_node_recall_after"] for r in rs)
            acts = statistics.mean(r["actions_total"] for r in rs)
            its = statistics.mean(r["iterations"] for r in rs)
            md.append(f"| {regime} | {method} | {n} | {cf:.1f} | {cred:.1f} | "
                      f"{er:.1f} | {ep:.1f} | {nr:.1f} | {acts:.1f} | {its:.1f} |")

    md.append("\n## Method lift over no_repair (per regime, edge-recall delta)\n")
    md.append("| regime | no_repair | random | heur_remove | heur_modify | best lift |")
    md.append("|--------|----------:|-------:|------------:|------------:|----------:|")
    for regime in regimes:
        baseline = by_regime_method.get((regime, "no_repair"), [])
        if not baseline:
            md.append(f"| {regime} | n/a | | | | |")
            continue
        base = statistics.mean(r["gt_edge_recall_after"] for r in baseline) * 100
        row = [f"| {regime} | {base:.1f}"]
        lifts = []
        for method in ("random", "heuristic_remove", "heuristic_modify"):
            rs = by_regime_method.get((regime, method), [])
            if not rs:
                row.append(""); continue
            v = statistics.mean(r["gt_edge_recall_after"] for r in rs) * 100
            row.append(f"{v:.1f}")
            lifts.append((method, v - base))
        if lifts:
            best = max(lifts, key=lambda x: x[1])
            row.append(f"{best[1]:+.1f}pp ({best[0]})")
        md.append(" | ".join(row) + " |")

    # noise-type accounting (averaged across no_repair runs only, since
    # noise is set before repair starts)
    md.append("\n## Mean injection counts per regime\n")
    NOISE_TYPES = ("N1_direction", "N2_topology", "N3_endpoint_swap",
                    "N4_spurious_pair", "N5_node_collapse", "N5_collapse_fallback",
                    "N6_duplicate", "N7_hallucinated")
    md.append("| regime | total | " + " | ".join(NOISE_TYPES) + " |")
    md.append("|--------|------:|" + "|".join(["---:"] * len(NOISE_TYPES)) + "|")
    for regime in regimes:
        rs = by_regime_method.get((regime, "no_repair"), [])
        if not rs:
            continue
        totals = Counter()
        for r in rs:
            nb = r.get("noise_breakdown", {})
            for k, v in nb.items():
                totals[k] += v
        n = len(rs)
        means = [totals.get(t, 0) / n for t in NOISE_TYPES]
        md.append(f"| {regime} | {sum(means):.1f} | "
                  + " | ".join(f"{m:.1f}" for m in means) + " |")

    # Per-graph breakdown
    md.append("\n## Per-graph aggregate (edge-recall after repair, per method)\n")
    by_graph_regime_method = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        by_graph_regime_method[(r["graph"], r["regime"], r["method"])].append(r)
    md.append("| graph | regime | no_repair | random | heur_remove | heur_modify |")
    md.append("|-------|--------|----------:|-------:|------------:|------------:|")
    for graph in graphs:
        for regime in regimes:
            cells = [f"| {graph} | {regime}"]
            for method in ("no_repair", "random", "heuristic_remove", "heuristic_modify"):
                rs = by_graph_regime_method.get((graph, regime, method), [])
                if rs:
                    v = 100 * statistics.mean(r["gt_edge_recall_after"] for r in rs)
                    cells.append(f"{v:.1f}")
                else:
                    cells.append("")
            md.append(" | ".join(cells) + " |")

    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out_root}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
