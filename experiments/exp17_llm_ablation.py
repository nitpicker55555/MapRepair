"""Experiment 17: LLM repair ablation on synthetic GT graphs.

Mirrors exp03's experimental design (synthetic graph × single-noise-type ×
small error counts) but replaces the heuristic agent with the four
LLMRepairAgent modes that constitute the paper's ablation:

  baseline    — vanilla prompt, no LCA/EIS/VC
  edge_impact — LCA-filtered candidates + impact scores
  vc_only     — LCA-filtered candidates + version history (rollback)
  vc_ei       — everything (paper's full pipeline)

This is the head-to-head test that validates the paper's algorithmic
contributions. Together with exp03 (heuristic on the same shape of
problem) we have:

  no_repair  ← lower bound
  random     ← random-edit lower bound
  heuristic  ← non-LLM algorithmic baseline (exp03)
  LLM modes  ← LLM-in-loop ablation (this experiment)

Outputs:
  results/exp17/raw.json
  results/exp17/summary.md
  results/exp17/ablation_table.md  (the paper headline)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from maprepair.agents.llm_agent import LLMRepairAgent
from maprepair.conflict import detect_all
from maprepair.graph import NavGraph
from maprepair.synth import make_synthetic, SyntheticGraph


MODES = ("baseline", "edge_impact", "vc_only", "vc_ei")

# Mirrors exp03 plan: (family, err_type, [(size, num_errors)])
DEFAULT_PLAN = [
    ("tree",   "direction", [(3, 1), (4, 1), (5, 2)]),
    ("tree",   "topology",  [(3, 1), (4, 1), (5, 2)]),
    ("grid",   "direction", [(3, 1), (4, 1), (5, 2)]),
    ("grid",   "topology",  [(3, 1), (4, 1), (5, 2)]),
    ("random", "direction", [(24, 1), (36, 2)]),
    ("random", "topology",  [(24, 1), (36, 2)]),
]


def gt_recall(graph: NavGraph, gt: NavGraph) -> float:
    gt_pairs = {(e.source, e.target) for e in gt.primary_edges()}
    pred_pairs = {(e.source, e.target) for e in graph.primary_edges()}
    if not gt_pairs:
        return 0.0
    return len(pred_pairs & gt_pairs) / len(gt_pairs)


def gt_direction_accuracy(graph: NavGraph, gt: NavGraph) -> float:
    gt_dir = {(e.source, e.target): e.direction for e in gt.primary_edges()}
    matches = 0; total = 0
    for e in graph.primary_edges():
        if (e.source, e.target) in gt_dir:
            total += 1
            if gt_dir[(e.source, e.target)] == e.direction:
                matches += 1
    return matches / total if total else 0.0


def run_one(family: str, err_type: str, size: int, num_errors: int,
            mode: str, seed: int, model: str,
            max_iterations: int = 15,
            max_attempts: int = 3) -> Dict:
    if family == "grid":
        if size < 3:
            size = 3
        spec = make_synthetic("grid", rows=size, cols=size,
                               error_mix={err_type: num_errors}, seed=seed)
    else:
        spec = make_synthetic(family, size=size,
                               error_mix={err_type: num_errors}, seed=seed)
    if not detect_all(spec.graph):
        return {"skip": True, "reason": "no conflicts after injection"}

    agent = LLMRepairAgent(model=model, mode=mode,
                             max_attempts_per_conflict=max_attempts)
    t0 = time.time()
    r = agent.repair(spec.graph.copy(), max_iterations=max_iterations)
    elapsed = time.time() - t0
    gt = spec.ground_truth
    return {
        "family": family, "err_type": err_type, "size": size,
        "num_errors": num_errors, "mode": mode, "seed": seed,
        "n_initial_conflicts": len(r.conflicts_before),
        "n_remaining_conflicts": len(r.conflicts_after),
        "conflict_free": r.success,
        "gt_edge_recall": gt_recall(r.graph_after, gt),
        "gt_direction_accuracy": gt_direction_accuracy(r.graph_after, gt),
        "n_actions": len(r.actions),
        "iterations": r.iterations,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--seeds", type=int, default=5,
                    help="Seeds per (family, err_type, size) combination.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-iterations", type=int, default=15)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--out-root", type=Path, default=Path("results/exp17"))
    args = ap.parse_args()

    modes = args.modes.split(",")
    args.out_root.mkdir(parents=True, exist_ok=True)

    # build job list
    jobs = []
    for family, err_type, ss in DEFAULT_PLAN:
        for size, num_errors in ss:
            for seed in range(args.seeds):
                for mode in modes:
                    jobs.append((family, err_type, size, num_errors, mode, seed))
    print(f"Total runs: {len(jobs)} (modes={modes}, seeds={args.seeds})")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_one, *job, model=args.model,
                       max_iterations=args.max_iterations,
                       max_attempts=args.max_attempts): job
            for job in jobs
        }
        done = 0
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"family": job[0], "err_type": job[1], "size": job[2],
                        "num_errors": job[3], "mode": job[4], "seed": job[5],
                        "error": str(e)}
            rows.append(row)
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}]")

    (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if "error" not in r and "skip" not in r]
    by_mode = defaultdict(list)
    by_mode_errtype = defaultdict(list)
    for r in ok:
        by_mode[r["mode"]].append(r)
        by_mode_errtype[(r["mode"], r["err_type"])].append(r)

    def stats(rs):
        if not rs: return {}
        return {
            "n": len(rs),
            "cf_pct": 100 * sum(1 for r in rs if r["conflict_free"]) / len(rs),
            "gt_recall": 100 * statistics.mean(r["gt_edge_recall"] for r in rs),
            "gt_dir_acc": 100 * statistics.mean(r["gt_direction_accuracy"] for r in rs),
            "mean_iters": statistics.mean(r["iterations"] for r in rs),
            "mean_actions": statistics.mean(r["n_actions"] for r in rs),
        }

    md = [
        f"# Experiment 17 — LLM ablation on synthetic GT (model={args.model})\n",
        f"Total runs: {len(rows)}  ({len(ok)} valid, "
        f"{sum(1 for r in rows if 'skip' in r)} skipped, "
        f"{sum(1 for r in rows if 'error' in r)} errors)\n",
        "## Aggregate by mode\n",
        "| mode | n | conflict-free % | GT edge recall % | GT dir acc % | mean iters | mean actions |",
        "|------|--:|----------------:|-----------------:|-------------:|-----------:|-------------:|",
    ]
    for mode in modes:
        s = stats(by_mode[mode])
        if not s: continue
        md.append(f"| {mode} | {s['n']} | {s['cf_pct']:.1f} | {s['gt_recall']:.1f} | "
                  f"{s['gt_dir_acc']:.1f} | {s['mean_iters']:.1f} | {s['mean_actions']:.1f} |")

    md.append("\n## By mode × error type\n")
    md.append("| mode | err_type | n | conflict-free % | GT recall % | dir acc % | iters |")
    md.append("|------|----------|--:|----------------:|------------:|----------:|------:|")
    for mode in modes:
        for et in ("direction", "topology"):
            s = stats(by_mode_errtype[(mode, et)])
            if not s: continue
            md.append(f"| {mode} | {et} | {s['n']} | {s['cf_pct']:.1f} | "
                      f"{s['gt_recall']:.1f} | {s['gt_dir_acc']:.1f} | "
                      f"{s['mean_iters']:.1f} |")

    md.append("\n## By mode × family × error type × num_errors\n")
    md.append("| family | err_type | n_err | mode | CF % | GT recall | iters |")
    md.append("|--------|----------|------:|------|-----:|----------:|------:|")
    keyed = defaultdict(list)
    for r in ok:
        keyed[(r["family"], r["err_type"], r["num_errors"], r["mode"])].append(r)
    for (fam, et, ne, mode), rs in sorted(keyed.items()):
        s = stats(rs)
        md.append(f"| {fam} | {et} | {ne} | {mode} | {s['cf_pct']:.1f} | "
                  f"{s['gt_recall']:.1f} | {s['mean_iters']:.1f} |")

    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out_root}/summary.md")

    # Also write a focused ablation table
    abl_md = [
        f"# LLM-MapRepair ablation (model={args.model})\n",
        "Each mode adds one capability vs the previous:\n",
        "  - baseline:    raw LLM, sees only involved edges of the current conflict",
        "  - edge_impact: + LCA-filtered candidates ranked by impact score",
        "  - vc_only:     + version history (can rollback to a prior commit)",
        "  - vc_ei:       both (= full LLM-MapRepair pipeline)\n",
        "## Headline\n",
        "| mode | CF % | GT recall % | GT dir acc % | mean iters | Δ CF vs baseline |",
        "|------|-----:|------------:|-------------:|-----------:|-----------------:|",
    ]
    base_cf = stats(by_mode["baseline"]).get("cf_pct", 0) if by_mode["baseline"] else 0
    for mode in modes:
        s = stats(by_mode[mode])
        if not s: continue
        delta = s["cf_pct"] - base_cf
        abl_md.append(f"| {mode} | {s['cf_pct']:.1f} | {s['gt_recall']:.1f} | "
                       f"{s['gt_dir_acc']:.1f} | {s['mean_iters']:.1f} | "
                       f"{delta:+.1f}pp |")

    (args.out_root / "ablation_table.md").write_text("\n".join(abl_md) + "\n")
    print(f"Wrote {args.out_root}/ablation_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
