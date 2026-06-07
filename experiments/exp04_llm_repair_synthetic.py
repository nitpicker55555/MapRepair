"""Experiment 4: LLM-driven end-to-end repair on synthetic ground truth.

Same evaluation harness as Exp 3, but now the repair agent is the LLM in four
configurations (baseline / edge_impact / vc_only / vc_ei). Because GT is
known by construction, we can report the same accuracy metric the paper uses
on MANGO -- but interpreted strictly (a fix is "correct" iff the resulting
graph state on touched edges matches GT).

Sample count is intentionally small (LLM calls are expensive). We pin a fixed
set of (family, err_type, seed) cases and run all four modes on each so the
ablation comparison is paired.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from maprepair.conflict import detect_all
from maprepair.synth import make_synthetic, SyntheticGraph
from maprepair.agents.llm_agent import LLMRepairAgent

# Re-use Exp 3's evaluation helpers.
from experiments.exp03_heuristic_repair import evaluate


MODES = ("baseline", "edge_impact", "vc_only", "vc_ei")


def _make_spec(family: str, size: int, err_type: str, num_err: int, seed: int) -> SyntheticGraph:
    if family == "grid":
        return make_synthetic("grid", rows=size, cols=size,
                              error_mix={err_type: num_err}, seed=seed)
    return make_synthetic(family, size=size,
                          error_mix={err_type: num_err}, seed=seed)


def _run_one_mode(family: str, size: int, err_type: str, num_err: int,
                   seed: int, model: str, mode: str) -> Dict:
    spec = _make_spec(family, size, err_type, num_err, seed)
    agent = LLMRepairAgent(model=model, mode=mode, max_attempts_per_conflict=5)
    result = agent.repair(spec.graph.copy(), max_iterations=20)
    ev = evaluate(spec, result.graph_after)
    return {
        "family": family,
        "size": size,
        "err_type": err_type,
        "num_err": num_err,
        "seed": seed,
        "model": model,
        "mode": mode,
        "num_conflicts": len(detect_all(spec.graph)),
        "iterations": result.iterations,
        "conflict_free": result.success,
        "remaining_conflicts": len(result.conflicts_after),
        **ev,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("results/exp04"))
    ap.add_argument("--model", default="gpt-4.1-mini",
                    help="LLM model name (Azure deployment).")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cases: List[Tuple[str, int, str, int]] = [
        ("tree",   4, "direction", 1),
        ("tree",   4, "topology",  1),
        ("tree",   5, "direction", 2),
        ("random", 30, "direction", 1),
        ("random", 30, "topology",  1),
        ("random", 60, "direction", 2),
        ("grid",   4, "direction", 1),
        ("grid",   4, "topology",  1),
    ]
    work: List[Tuple] = []
    for (family, size, err, n_err) in cases:
        for seed in range(args.seeds):
            for mode in MODES:
                work.append((family, size, err, n_err, seed, args.model, mode))

    print(f"Total runs: {len(work)} with {args.workers} workers, model={args.model}")
    rows: List[Dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one_mode, *args_t): args_t for args_t in work}
        completed = 0
        for fut in as_completed(futs):
            args_t = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"error": str(e), "args": args_t}
            rows.append(row)
            completed += 1
            if completed % 20 == 0:
                print(f"  {completed}/{len(work)} done")

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    md = [
        f"# Experiment 4 — LLM end-to-end repair on synthetic GT (model={args.model})\n",
        f"Total runs: {len(rows)} (4 modes x {args.seeds} seeds x {len(cases)} cases)\n",
        "Headline metric: % of runs that end with **conflict-free graph that matches GT** "
        "(strict: GT direction accuracy = 100% and GT edge recall = 100%).\n",
    ]
    by_mode: Dict[str, List[Dict]] = {}
    for r in rows:
        if "mode" not in r:
            continue
        by_mode.setdefault(r["mode"], []).append(r)

    md.append("## Aggregate by mode")
    md.append("| mode | n | conflict_free %% | gt_recall %% | gt_dir_acc %% | strict_correct %% | mean_iters |")
    md.append("|------|--:|-----------------:|--------------:|--------------:|-------------------:|-----------:|")
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        cf = 100 * sum(1 for r in rs if r["conflict_free"]) / len(rs)
        rec = 100 * statistics.mean(r["gt_edge_recall"] for r in rs)
        dir_ = 100 * statistics.mean(r["gt_direction_accuracy"] for r in rs)
        strict = 100 * sum(1 for r in rs
                            if r["conflict_free"]
                            and r["gt_edge_recall"] >= 0.999
                            and r["gt_direction_accuracy"] >= 0.999) / len(rs)
        iters = statistics.mean(r["iterations"] for r in rs)
        md.append(f"| {mode} | {len(rs)} | {cf:.1f} | {rec:.1f} | {dir_:.1f} | {strict:.1f} | {iters:.2f} |")

    md.append("\n## By (mode x err_type)\n")
    md.append("| mode | err_type | n | conflict_free % | gt_dir_acc % | mean_iters |")
    md.append("|------|----------|--:|----------------:|--------------:|-----------:|")
    cohort: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        if "mode" not in r: continue
        cohort.setdefault((r["mode"], r["err_type"]), []).append(r)
    for (mode, err), rs in sorted(cohort.items()):
        cf = 100 * sum(1 for r in rs if r["conflict_free"]) / len(rs)
        dir_ = 100 * statistics.mean(r["gt_direction_accuracy"] for r in rs)
        iters = statistics.mean(r["iterations"] for r in rs)
        md.append(f"| {mode} | {err} | {len(rs)} | {cf:.1f} | {dir_:.1f} | {iters:.2f} |")
    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out_root}/summary.md")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
