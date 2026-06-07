"""Experiment 25: 2026 frontier-model robustness check.

Paper accepted with 2024-2025 model data (gpt-4o/4.1/4.1-mini in exp22).
This experiment validates the method holds up on June-2026 frontier
models from three vendors (OpenAI / Anthropic / Google) — without
touching any existing data (purely additive rows for camera-ready
or arxiv update).

Models (all via api.openai-hub.com proxy):
  gpt-5.5                       — OpenAI flagship 2026
  gpt-5-mini                    — OpenAI value-tier, reasoning-aware
  claude-sonnet-4-6             — Anthropic frontier (non-thinking)
  gemini-2.5-flash              — Google flash (fast + reliable)
  o4-mini                       — OpenAI reasoning model

Conditions (per model, per config, per seed):
  llm_baseline      — raw LLM, no LCA / no EIS  (ablation lower bound)
  llm_edge_impact   — LCA + EIS  (OUR METHOD)
  heuristic_remove  — model-independent reference (run once)

Configs: 8-subset of exp22's 16 (one per family × err_type × size-band)
Seeds:   5 per cell
Total:   8 × 5 × 5 models × 2 LLM-modes + 8 × 5 heuristic = 440 runs
Est cost ~$30-50, ~30-60 min wallclock with 8 workers.
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
from maprepair.llm_client_proxy import chat_json as proxy_chat_json


# Subset of exp22 configs that span the parameter space efficiently
CONFIGS = [
    ("tree",   "direction", 4, 1),
    ("tree",   "topology",  5, 2),
    ("grid",   "direction", 4, 1),
    ("grid",   "topology",  5, 2),
    ("random", "direction", 24, 1),
    ("random", "topology",  24, 1),
    ("random", "direction", 36, 2),
    ("random", "topology",  36, 2),
]

DEFAULT_MODELS = (
    "gpt-5.5",
    "gpt-5-mini",
    "claude-sonnet-4-6",
    "gemini-2.5-flash",
    "o4-mini",
)

LLM_MODES = ("baseline", "edge_impact")


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
            mode: str, model: str, seed: int,
            max_iterations: int, max_attempts: int) -> Dict:
    if family == "grid":
        spec = make_synthetic("grid", rows=size, cols=size,
                               error_mix={err_type: num_errors}, seed=seed)
    else:
        spec = make_synthetic(family, size=size,
                               error_mix={err_type: num_errors}, seed=seed)
    if not detect_all(spec.graph):
        return {"skip": True, "reason": "no conflicts"}
    t0 = time.time()
    if mode in LLM_MODES:
        agent = LLMRepairAgent(model=model, mode=mode,
                                 max_attempts_per_conflict=max_attempts,
                                 lookahead=False,
                                 chat_json_fn=proxy_chat_json)
        r = agent.repair(spec.graph.copy(), max_iterations=max_iterations)
    elif mode == "heuristic_remove":
        r = HeuristicRepairAgent(prefer_remove=True).repair(
            spec.graph.copy(), max_iterations=max_iterations)
    else:
        raise ValueError(mode)
    elapsed = time.time() - t0
    gt = spec.ground_truth
    actions_kinds = [a.kind for a in r.actions]
    return {
        "family": family, "err_type": err_type, "size": size,
        "num_errors": num_errors, "mode": mode, "model": model, "seed": seed,
        "n_initial_conflicts": len(r.conflicts_before),
        "n_remaining_conflicts": len(r.conflicts_after),
        "conflict_free": r.success,
        "gt_edge_recall": gt_recall(r.graph_after, gt),
        "gt_direction_accuracy": gt_direction_accuracy(r.graph_after, gt),
        "n_actions": len(r.actions),
        "n_modify": actions_kinds.count("modify_edge"),
        "n_remove": actions_kinds.count("remove_edge"),
        "iterations": r.iterations,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-iterations", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--out-root", type=Path, default=Path("results/exp25"))
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    models = args.models.split(",")

    jobs: List[Tuple] = []
    # LLM modes × models
    for model in models:
        for mode in LLM_MODES:
            for family, err_type, size, ne in CONFIGS:
                for seed in range(args.seeds):
                    jobs.append((family, err_type, size, ne, mode, model, seed))
    # Heuristic reference (run once total, model-independent)
    for family, err_type, size, ne in CONFIGS:
        for seed in range(args.seeds):
            jobs.append((family, err_type, size, ne, "heuristic_remove", "n/a", seed))

    print(f"Models: {models}", flush=True)
    print(f"Configs: {len(CONFIGS)}  Seeds: {args.seeds}", flush=True)
    print(f"Total runs: {len(jobs)}", flush=True)

    rows: List[Dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_one, *job,
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
                        "num_errors": job[3], "mode": job[4], "model": job[5],
                        "seed": job[6], "error": f"{type(e).__name__}: {e}"}
            rows.append(row)
            done += 1
            if done % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                # incremental save every 20
                (args.out_root / "raw.json").write_text(
                    json.dumps(rows, indent=2)
                )
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
            "elapsed": statistics.mean(r["elapsed_sec"] for r in rs),
            "modify": statistics.mean(r.get("n_modify", 0) for r in rs),
            "remove": statistics.mean(r.get("n_remove", 0) for r in rs),
        }

    by_mode_model = defaultdict(list)
    for r in ok:
        by_mode_model[(r["mode"], r["model"])].append(r)

    # Headline table
    md = [
        f"# Experiment 25 — 2026 frontier-model robustness check\n",
        f"Total runs: {len(rows)} (valid {len(ok)}, "
        f"errors {sum(1 for r in rows if 'error' in r)}, "
        f"skipped {sum(1 for r in rows if 'skip' in r)})\n",
        f"Models: {models}\n",
        "## Headline: CF % per model (n per cell shown)\n",
        "| Model | baseline LLM | edge_impact (ours) | Δ lift | Δ vs heuristic |",
        "|-------|-------------:|-------------------:|-------:|---------------:|",
    ]
    heur_stats = stats(by_mode_model[("heuristic_remove", "n/a")])
    heur_cf = heur_stats.get("cf", 0)
    for model in models:
        sb = stats(by_mode_model[("baseline", model)])
        se = stats(by_mode_model[("edge_impact", model)])
        if sb["n"] == 0 or se["n"] == 0: continue
        delta = se["cf"] - sb["cf"]
        delta_heur = se["cf"] - heur_cf
        md.append(f"| {model} | {sb['cf']:.1f}% (n={sb['n']}) | "
                  f"**{se['cf']:.1f}%** (n={se['n']}) | "
                  f"**{delta:+.1f}pp** | **{delta_heur:+.1f}pp** |")
    md.append(f"| heuristic_remove (ref) | — | — | — | {heur_cf:.1f}% baseline |")

    # Aggregate by mode×model
    md.append("\n## Detailed: by (mode × model)\n")
    md.append("| mode | model | n | CF % | GT recall | dir acc | iters | elapsed |")
    md.append("|------|-------|--:|-----:|----------:|--------:|------:|--------:|")
    for mode in LLM_MODES:
        for model in models:
            s = stats(by_mode_model[(mode, model)])
            if s["n"] == 0: continue
            md.append(f"| {mode} | {model} | {s['n']} | {s['cf']:.1f} | "
                      f"{s['gtr']:.1f} | {s['gtd']:.1f} | {s['iters']:.1f} | "
                      f"{s['elapsed']:.0f}s |")
    s = heur_stats
    if s["n"] > 0:
        md.append(f"| heuristic_remove | n/a | {s['n']} | {s['cf']:.1f} | "
                  f"{s['gtr']:.1f} | {s['gtd']:.1f} | {s['iters']:.1f} | "
                  f"{s['elapsed']:.1f}s |")

    # by err_type
    md.append("\n## By model × err_type\n")
    md.append("| model | err_type | mode | n | CF % | dir acc |")
    md.append("|-------|----------|------|--:|-----:|--------:|")
    by_full = defaultdict(list)
    for r in ok:
        by_full[(r["mode"], r["model"], r["err_type"])].append(r)
    for model in models:
        for et in ("direction", "topology"):
            for mode in LLM_MODES:
                s = stats(by_full[(mode, model, et)])
                if s["n"] == 0: continue
                md.append(f"| {model} | {et} | {mode} | {s['n']} | {s['cf']:.1f} | "
                          f"{s['gtd']:.1f} |")

    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out_root}/summary.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
