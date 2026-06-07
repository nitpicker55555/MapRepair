"""Experiment 29: focused complementary-roles experiment.

Hypothesis (from exp19 aggregation):
  Edge-Impact dominates topology conflicts at all error densities.
  Version Control dominates direction conflicts at high error density
  (>=4 errors).  This is "complementary roles", not "synergy".

exp19's LARGE configs (random graphs at size 60, num_errors 4 and 8)
had only 8 seeds per cell -- too thin.  exp29 boosts to 20 seeds
per cell on the same 4 LARGE configs, keeping only the 4 core LLM
modes (no lookahead, no heuristic).

Total runs: 4 cells x 4 modes x 20 seeds = 320 runs.
At ~5-30s per run with workers=8, wall-clock 250-1500s.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from maprepair.agents.llm_agent import LLMRepairAgent
from maprepair.conflict import detect_all
from maprepair.synth import make_synthetic


CELLS = [
    ("random", "direction", 60, 4),
    ("random", "direction", 60, 8),
    ("random", "topology",  60, 4),
    ("random", "topology",  60, 8),
]

MODES = ["llm_baseline", "llm_edge_impact", "llm_vc_only", "llm_vc_ei"]


def gt_recall(graph, gt) -> float:
    gt_pairs = {(e.source, e.target) for e in gt.primary_edges()}
    pred_pairs = {(e.source, e.target) for e in graph.primary_edges()}
    if not gt_pairs: return 0.0
    return len(pred_pairs & gt_pairs) / len(gt_pairs)


def gt_direction_accuracy(graph, gt) -> float:
    gt_dir = {(e.source, e.target): e.direction for e in gt.primary_edges()}
    matches = 0; total = 0
    for e in graph.primary_edges():
        if (e.source, e.target) in gt_dir:
            total += 1
            if gt_dir[(e.source, e.target)] == e.direction:
                matches += 1
    return matches / total if total else 0.0


def run_one(family: str, err_type: str, size: int, num_errors: int,
            mode_label: str, seed: int, model: str,
            max_iterations: int = 20, max_attempts: int = 3) -> Dict:
    spec = make_synthetic(family, size=size,
                          error_mix={err_type: num_errors}, seed=seed)
    work = spec.graph
    gt = spec.ground_truth
    n_init = len(detect_all(work))

    mode = mode_label.replace("llm_", "")
    agent = LLMRepairAgent(model=model, mode=mode,
                           max_attempts_per_conflict=max_attempts,
                           lookahead=False)
    t0 = time.time()
    r = agent.repair(work.copy(), max_iterations=max_iterations)
    elapsed = time.time() - t0
    n_remaining = len(r.conflicts_after)
    cf = (n_remaining == 0)
    actions_kinds = [a.kind for a in r.actions]
    return {
        "family": family, "err_type": err_type, "size": size,
        "num_errors": num_errors, "mode": mode_label, "seed": seed,
        "model": model,
        "n_initial_conflicts": n_init,
        "n_remaining_conflicts": n_remaining,
        "conflict_free": cf,
        "gt_edge_recall": gt_recall(r.graph_after, gt),
        "gt_direction_accuracy": gt_direction_accuracy(r.graph_after, gt),
        "n_actions": len(r.actions),
        "n_modify": actions_kinds.count("modify_edge"),
        "n_remove": actions_kinds.count("remove_edge"),
        "n_rollback": actions_kinds.count("rollback"),
        "iterations": r.iterations,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-iterations", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--out-root", type=Path, default=Path("results/exp29"))
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for family, et, sz, ne in CELLS:
        for seed in range(args.seeds):
            for m in MODES:
                jobs.append((family, et, sz, ne, m, seed))
    print(f"Total runs: {len(jobs)} ({len(CELLS)} cells x {len(MODES)} modes x {args.seeds} seeds)", flush=True)

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, *job, model=args.model,
                          max_iterations=args.max_iterations,
                          max_attempts=args.max_attempts): job for job in jobs}
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
                print(f"  [{done}/{len(jobs)}] {job[1]} ne={job[3]} {job[4]} seed={job[5]} t={time.time()-t0:.0f}s", flush=True)
            (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if "error" not in r]
    by_cell_mode = defaultdict(list)
    for r in ok:
        by_cell_mode[(r["err_type"], r["num_errors"], r["mode"])].append(r)

    md = [f"# exp29 — Complementary roles (model={args.model})\n",
          f"Cells: {CELLS}\nModes: {MODES}\nSeeds per cell: {args.seeds}\n",
          "## Aggregate by err_type x num_errors x mode\n",
          "| err_type | n_err | mode | n | CF % | GT recall | dir acc | iters | actions |",
          "|----------|------:|------|--:|-----:|----------:|--------:|------:|--------:|"]
    for (et, ne, m), rs in sorted(by_cell_mode.items()):
        cf = 100 * sum(1 for r in rs if r.get("conflict_free")) / len(rs)
        gtr = 100 * statistics.mean(r.get("gt_edge_recall", 0) for r in rs)
        da = 100 * statistics.mean(r.get("gt_direction_accuracy", 0) for r in rs)
        it = statistics.mean(r.get("iterations", 0) for r in rs)
        act = statistics.mean(r.get("n_actions", 0) for r in rs)
        md.append(f"| {et} | {ne} | {m} | {len(rs)} | {cf:.1f} | {gtr:.2f} | {da:.2f} | {it:.1f} | {act:.1f} |")

    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out_root}/summary.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
