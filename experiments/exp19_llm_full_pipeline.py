"""Experiment 19: LLM repair with remove_edge action + optional lookahead.

After exp18 showed LLM modes were beaten 10x by heuristic_remove on
direction conflicts, root-cause analysis identified TWO missing
primitives in the LLM action space:

  (1) `remove_edge` — heuristic_remove succeeds primarily by deleting
      noise edges. Previous LLM modes could only modify_edge, leaving
      them unable to delete a spurious sibling.
  (2) `lookahead` — before applying an action, check whether it
      strictly decreases conflict count; reject and retry otherwise.

exp19 sweep:
  4 LLM modes × {with, without lookahead} = 8 LLM conditions
  + 2 heuristic baselines (reference)
  × 16 core configs (15 seeds) + 4 large configs (8 seeds)
  = ~2720 runs total

Headlines we want to extract:
  a) lift from remove_edge access (vs exp18)
  b) lift from lookahead on top of remove_edge
  c) does (any LLM mode + remove + lookahead) match heuristic_remove?
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

from maprepair.agents.heuristic import HeuristicRepairAgent
from maprepair.agents.llm_agent import LLMRepairAgent
from maprepair.conflict import detect_all
from maprepair.synth import make_synthetic


CORE = [
    ("tree",   "direction", 3, 1),
    ("tree",   "direction", 4, 1),
    ("tree",   "direction", 5, 2),
    ("tree",   "topology",  3, 1),
    ("tree",   "topology",  4, 1),
    ("tree",   "topology",  5, 2),
    ("grid",   "direction", 3, 1),
    ("grid",   "direction", 4, 1),
    ("grid",   "direction", 5, 2),
    ("grid",   "topology",  3, 1),
    ("grid",   "topology",  4, 1),
    ("grid",   "topology",  5, 2),
    ("random", "direction", 24, 1),
    ("random", "direction", 36, 2),
    ("random", "topology",  24, 1),
    ("random", "topology",  36, 2),
]
LARGE = [
    ("random", "direction", 60, 4),
    ("random", "direction", 60, 8),
    ("random", "topology",  60, 4),
    ("random", "topology",  60, 8),
]


def gt_recall(graph, gt) -> float:
    gt_pairs = {(e.source, e.target) for e in gt.primary_edges()}
    pred_pairs = {(e.source, e.target) for e in graph.primary_edges()}
    if not gt_pairs:
        return 0.0
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
    if family == "grid":
        spec = make_synthetic("grid", rows=size, cols=size,
                               error_mix={err_type: num_errors}, seed=seed)
    else:
        spec = make_synthetic(family, size=size,
                               error_mix={err_type: num_errors}, seed=seed)
    if not detect_all(spec.graph):
        return {"skip": True, "reason": "no conflicts"}
    t0 = time.time()
    if mode_label.startswith("llm_"):
        # parse mode_label like "llm_vc_ei+lookahead" or "llm_baseline"
        rest = mode_label[4:]
        lookahead = rest.endswith("+lookahead")
        if lookahead:
            rest = rest[: -len("+lookahead")]
        agent = LLMRepairAgent(model=model, mode=rest,
                                 max_attempts_per_conflict=max_attempts,
                                 lookahead=lookahead, lookahead_retries=3)
        r = agent.repair(spec.graph.copy(), max_iterations=max_iterations)
    elif mode_label == "heuristic_remove":
        r = HeuristicRepairAgent(prefer_remove=True).repair(
            spec.graph.copy(), max_iterations=max_iterations)
    elif mode_label == "heuristic_modify":
        r = HeuristicRepairAgent(prefer_remove=False).repair(
            spec.graph.copy(), max_iterations=max_iterations)
    else:
        raise ValueError(mode_label)
    elapsed = time.time() - t0
    gt = spec.ground_truth
    actions_kinds = [a.kind for a in r.actions]
    return {
        "family": family, "err_type": err_type, "size": size,
        "num_errors": num_errors, "mode": mode_label, "seed": seed,
        "n_initial_conflicts": len(r.conflicts_before),
        "n_remaining_conflicts": len(r.conflicts_after),
        "conflict_free": r.success,
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
    ap.add_argument("--seeds-core", type=int, default=15)
    ap.add_argument("--seeds-large", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-iterations", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--out-root", type=Path, default=Path("results/exp19"))
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    # 4 LLM modes × 2 lookahead = 8 conditions + 2 heuristic
    modes = []
    for base in ("baseline", "edge_impact", "vc_only", "vc_ei"):
        modes.append(f"llm_{base}")
        modes.append(f"llm_{base}+lookahead")
    modes.append("heuristic_remove")
    modes.append("heuristic_modify")

    jobs: List[Tuple] = []
    for family, err_type, size, ne in CORE:
        for seed in range(args.seeds_core):
            for mode in modes:
                jobs.append((family, err_type, size, ne, mode, seed))
    for family, err_type, size, ne in LARGE:
        for seed in range(args.seeds_large):
            for mode in modes:
                jobs.append((family, err_type, size, ne, mode, seed))
    print(f"Total runs: {len(jobs)} (modes={len(modes)})", flush=True)

    rows: List[Dict] = []
    t0 = time.time()
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
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  [{done}/{len(jobs)}] {el:.0f}s", flush=True)

    (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if "error" not in r and "skip" not in r]

    def stats(rs):
        if not rs: return {"n": 0}
        return {
            "n": len(rs),
            "cf": 100 * sum(1 for r in rs if r["conflict_free"]) / len(rs),
            "gtr": 100 * statistics.mean(r["gt_edge_recall"] for r in rs),
            "gtd": 100 * statistics.mean(r["gt_direction_accuracy"] for r in rs),
            "iters": statistics.mean(r["iterations"] for r in rs),
            "modify": statistics.mean(r.get("n_modify", 0) for r in rs),
            "remove": statistics.mean(r.get("n_remove", 0) for r in rs),
        }

    by_mode = defaultdict(list)
    by_mode_err = defaultdict(list)
    by_full = defaultdict(list)
    for r in ok:
        by_mode[r["mode"]].append(r)
        by_mode_err[(r["mode"], r["err_type"])].append(r)
        by_full[(r["family"], r["err_type"], r["size"], r["num_errors"], r["mode"])].append(r)

    md = [
        f"# Experiment 19 — LLM with remove_edge + optional lookahead (model={args.model})\n",
        f"Total runs: {len(rows)}  (valid {len(ok)}, errors {sum(1 for r in rows if 'error' in r)}, "
        f"skipped {sum(1 for r in rows if 'skip' in r)})\n",
        f"Core configs: {len(CORE)} × {args.seeds_core} seeds  |  "
        f"Large configs: {len(LARGE)} × {args.seeds_large} seeds.\n",
        "## Aggregate by mode\n",
        "| mode | n | CF % | GT recall % | dir acc % | iters | modify/run | remove/run |",
        "|------|--:|-----:|------------:|----------:|------:|-----------:|-----------:|",
    ]
    for mode in modes:
        s = stats(by_mode[mode])
        if s["n"] == 0: continue
        md.append(f"| {mode} | {s['n']} | {s['cf']:.1f} | {s['gtr']:.1f} | "
                  f"{s['gtd']:.1f} | {s['iters']:.1f} | {s['modify']:.1f} | {s['remove']:.1f} |")

    md.append("\n## By mode × err_type\n")
    md.append("| mode | err_type | n | CF % | GT recall | dir acc | iters | modify | remove |")
    md.append("|------|----------|--:|-----:|----------:|--------:|------:|-------:|-------:|")
    for mode in modes:
        for et in ("direction", "topology"):
            s = stats(by_mode_err[(mode, et)])
            if s["n"] == 0: continue
            md.append(f"| {mode} | {et} | {s['n']} | {s['cf']:.1f} | {s['gtr']:.1f} | "
                      f"{s['gtd']:.1f} | {s['iters']:.1f} | {s['modify']:.1f} | {s['remove']:.1f} |")

    md.append("\n## Detailed: family × err_type × size × num_errors × mode\n")
    md.append("| family | err | size | n_err | mode | n | CF % | GT recall | dir acc | iters |")
    md.append("|--------|-----|-----:|------:|------|--:|-----:|----------:|--------:|------:|")
    for (fam, et, sz, ne, mode), rs in sorted(by_full.items()):
        s = stats(rs)
        md.append(f"| {fam} | {et} | {sz} | {ne} | {mode} | {s['n']} | {s['cf']:.1f} | "
                  f"{s['gtr']:.1f} | {s['gtd']:.1f} | {s['iters']:.1f} |")

    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out_root}/summary.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
