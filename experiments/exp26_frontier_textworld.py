"""Experiment 26: 2026 frontier models on TextWorld real-language data.

The combined synthesis: exp25 showed cross-vendor lift on synthetic
benchmarks (claude-sonnet +12.5pp aggregate, gpt-5.5 +50pp on direction).
exp23 showed gpt-4.1 achieves +10pp lift over heuristic on TextWorld's
mango_like regime (real-language room names, real game topology). This
experiment bridges the two: do the 5 2026 frontier models retain the
TextWorld real-language win?

Setup:
  - 10 TextWorld games (reuse exp23's GT graphs — sequential pre-load to
    avoid the textworld concurrency bug fixed in exp23)
  - 1 noise regime: mango_like (the most realistic noise mix, mirroring
    MANGO's failure-mode distribution)
  - 5 frontier models from 3 vendors (matches exp25):
      gpt-5.5, gpt-5-mini, claude-sonnet-4-6,
      gemini-2.5-flash, o4-mini
  - 4 repair methods (heuristic_remove and no_repair are
    model-independent, so run only once):
      no_repair
      heuristic_remove
      llm_baseline (per model)
      llm_edge_impact (per model — OUR METHOD)
  - 3 seeds per cell

Total: 10 games × (2 model-independent methods × 3 seeds +
       5 models × 2 LLM methods × 3 seeds)
     = 10 × (6 + 30) = 360 runs (300 LLM, 60 fast).

Estimated 30-60 min, $15-25.

Outputs:
  results/exp26/raw.json
  results/exp26/summary.md
  results/exp26/frontier_textworld_paper_narrative.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from maprepair.agents.heuristic import HeuristicRepairAgent
from maprepair.agents.llm_agent import LLMRepairAgent
from maprepair.conflict import detect_all
from maprepair.graph import NavGraph
from maprepair.llm_client_proxy import chat_json as proxy_chat_json

# Reuse exp23's TextWorld GT loading (avoids concurrent parsing bug)
from experiments.exp23_textworld_noise import gt_to_navgraph
from experiments.exp16_noise import apply_regime, regime_by_name


REGIME_NAME = "mango_like"

LLM_MODES = ("baseline", "edge_impact")
HEUR_MODES = ("heuristic_remove",)
ALL_METHODS = ("no_repair",) + HEUR_MODES + LLM_MODES

DEFAULT_MODELS = (
    "gpt-5.5",
    "gpt-5-mini",
    "claude-sonnet-4-6",
    "gemini-2.5-flash",
    "o4-mini",
)


def gt_recall_precision(work: NavGraph, gt: NavGraph) -> Tuple[float, float]:
    gt_pairs = {(e.source, e.target) for e in gt.primary_edges()}
    pred_pairs = {(e.source, e.target) for e in work.primary_edges()}
    if not gt_pairs:
        return 0.0, 0.0
    if not pred_pairs:
        return 0.0, 0.0
    tp = len(pred_pairs & gt_pairs)
    return tp / len(gt_pairs), tp / len(pred_pairs)


def gt_dir_acc(work: NavGraph, gt: NavGraph) -> float:
    gt_dir = {(e.source, e.target): e.direction for e in gt.primary_edges()}
    matches = 0; total = 0
    for e in work.primary_edges():
        if (e.source, e.target) in gt_dir:
            total += 1
            if gt_dir[(e.source, e.target)] == e.direction:
                matches += 1
    return matches / total if total else 0.0


def run_method(method: str, work: NavGraph, model: str,
                max_iter: int, max_att: int):
    if method == "no_repair":
        cb = detect_all(work)
        return {"graph_after": work, "conflicts_before": cb,
                "conflicts_after": cb, "actions": [], "iterations": 0,
                "success": not cb}
    if method == "heuristic_remove":
        r = HeuristicRepairAgent(prefer_remove=True).repair(
            work.copy(), max_iterations=max_iter)
    elif method in ("llm_baseline", "llm_edge_impact"):
        mode = method[4:]
        agent = LLMRepairAgent(model=model, mode=mode,
                                 max_attempts_per_conflict=max_att,
                                 lookahead=False,
                                 chat_json_fn=proxy_chat_json)
        r = agent.repair(work.copy(), max_iterations=max_iter)
    else:
        raise ValueError(method)
    return {"graph_after": r.graph_after,
            "conflicts_before": r.conflicts_before,
            "conflicts_after": r.conflicts_after,
            "actions": r.actions,
            "iterations": r.iterations,
            "success": r.success}


def run_one(game_id: str, gt: NavGraph, method: str, model: str,
             seed: int, max_iter: int, max_att: int) -> Dict:
    work = gt.copy()
    recs = apply_regime(work, regime_by_name(REGIME_NAME), seed=seed)
    n_cb = len(detect_all(work))
    er_b, _ = gt_recall_precision(work, gt)
    t0 = time.time()
    out = run_method(method, work, model, max_iter, max_att)
    elapsed = time.time() - t0
    repaired = out["graph_after"]
    er_a, ep_a = gt_recall_precision(repaired, gt)
    n_ca = len(out["conflicts_after"])
    dir_acc = gt_dir_acc(repaired, gt)
    pred_nodes = len(set(repaired.nodes()) & set(gt.nodes()))
    n_gt_nodes = gt.num_nodes()
    node_recall = pred_nodes / n_gt_nodes if n_gt_nodes else 0.0
    action_kinds = [a.kind for a in out["actions"]]
    return {
        "game": game_id, "regime": REGIME_NAME, "method": method,
        "model": model, "seed": seed,
        "n_gt_rooms": n_gt_nodes,
        "n_gt_edges": len(gt.primary_edges()),
        "n_noise": len(recs),
        "n_conflicts_before": n_cb,
        "n_conflicts_after": n_ca,
        "conflict_free": (n_ca == 0),
        "conflict_reduction_pct": 100 * (n_cb - n_ca) / max(1, n_cb),
        "edge_recall_before_repair": er_b,
        "edge_recall_after": er_a,
        "edge_recall_delta": er_a - er_b,
        "edge_precision_after": ep_a,
        "direction_accuracy_after": dir_acc,
        "node_recall_after": node_recall,
        "n_actions": len(out["actions"]),
        "n_modify": action_kinds.count("modify_edge"),
        "n_remove": action_kinds.count("remove_edge"),
        "iterations": out["iterations"],
        "elapsed_sec": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-iterations", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--out-root", type=Path, default=Path("results/exp26"))
    ap.add_argument("--games-root", type=Path,
                    default=Path("results/exp23/games"))
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    models = args.models.split(",")

    # Pre-load all TextWorld GTs sequentially (avoid concurrent textworld bug)
    game_dirs = sorted(args.games_root.iterdir())
    game_gts: Dict[str, NavGraph] = {}
    print(f"Pre-loading TextWorld GTs from {args.games_root}...", flush=True)
    for gdir in game_dirs:
        if not gdir.is_dir():
            continue
        gt_path = gdir / "game.json"
        if not gt_path.exists():
            continue
        try:
            game_gts[gdir.name] = gt_to_navgraph(gt_path)
        except Exception as e:
            print(f"  failed {gdir.name}: {e}", flush=True)
    print(f"  loaded {len(game_gts)} games", flush=True)
    if not game_gts:
        print("No games loaded — run exp23 first to generate the GTs.")
        return 1

    jobs: List[Tuple] = []
    # Model-independent methods (no_repair, heuristic_remove) — single run per (game, seed)
    for gid in game_gts:
        for method in ("no_repair", "heuristic_remove"):
            for seed in range(args.seeds):
                jobs.append((gid, game_gts[gid], method, "n/a", seed))
    # LLM methods — per model
    for model in models:
        for gid in game_gts:
            for method in ("llm_baseline", "llm_edge_impact"):
                for seed in range(args.seeds):
                    jobs.append((gid, game_gts[gid], method, model, seed))

    print(f"Models: {models}", flush=True)
    print(f"Regime: {REGIME_NAME}", flush=True)
    print(f"Total runs: {len(jobs)}", flush=True)

    rows: List[Dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_one, *job,
                       max_iter=args.max_iterations,
                       max_att=args.max_attempts): job
            for job in jobs
        }
        done = 0
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"game": j[0], "method": j[2], "model": j[3],
                        "seed": j[4], "regime": REGIME_NAME,
                        "error": f"{type(e).__name__}: {e}"}
            rows.append(row)
            done += 1
            if done % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))
                print(f"  [{done}/{len(jobs)}] {el:.0f}s", flush=True)

    (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if "error" not in r]

    def stats(rs):
        if not rs: return {"n": 0}
        return {
            "n": len(rs),
            "cf": 100 * sum(1 for r in rs if r["conflict_free"]) / len(rs),
            "gtr_before": 100 * statistics.mean(r["edge_recall_before_repair"] for r in rs),
            "gtr_after": 100 * statistics.mean(r["edge_recall_after"] for r in rs),
            "gtr_delta": 100 * statistics.mean(r["edge_recall_delta"] for r in rs),
            "gtp_after": 100 * statistics.mean(r["edge_precision_after"] for r in rs),
            "gtd_after": 100 * statistics.mean(r["direction_accuracy_after"] for r in rs),
            "cred": statistics.mean(r["conflict_reduction_pct"] for r in rs),
            "iters": statistics.mean(r["iterations"] for r in rs),
            "actions": statistics.mean(r["n_actions"] for r in rs),
            "remove": statistics.mean(r.get("n_remove", 0) for r in rs),
            "elapsed": statistics.mean(r["elapsed_sec"] for r in rs),
        }

    by_meth_model = defaultdict(list)
    for r in ok:
        by_meth_model[(r["method"], r["model"])].append(r)

    # Aggregate per-model headline
    md = [
        f"# Experiment 26 — 2026 frontier models on TextWorld real-language\n",
        f"Total runs: {len(rows)} (valid {len(ok)}, "
        f"errors {sum(1 for r in rows if 'error' in r)})\n",
        f"Models: {models}",
        f"Regime: {REGIME_NAME} (mirrors MANGO's failure-mode mix)",
        f"Games: {len(game_gts)} TextWorld games (real-prose room names)",
        f"Seeds per cell: {args.seeds}\n",
        "## Headline: real-language CF % per model\n",
        "| Model | baseline LLM | edge_impact (ours) | Δ lift | edge recall after | Δ recall |",
        "|-------|-------------:|-------------------:|-------:|------------------:|---------:|",
    ]
    s_nr = stats(by_meth_model[("no_repair", "n/a")])
    s_h = stats(by_meth_model[("heuristic_remove", "n/a")])
    for model in models:
        sb = stats(by_meth_model[("llm_baseline", model)])
        se = stats(by_meth_model[("llm_edge_impact", model)])
        if sb["n"] == 0 or se["n"] == 0: continue
        delta = se["cf"] - sb["cf"]
        md.append(f"| {model} | {sb['cf']:.1f}% (n={sb['n']}) | "
                  f"**{se['cf']:.1f}%** (n={se['n']}) | "
                  f"**{delta:+.1f}pp** | {se['gtr_after']:.1f}% | "
                  f"{se['gtr_delta']:+.2f}pp |")
    md.append(f"| no_repair (ref)        | — | {s_nr['cf']:.1f}% | — | "
              f"{s_nr['gtr_after']:.1f}% | 0 |")
    md.append(f"| heuristic_remove (ref) | — | {s_h['cf']:.1f}% | — | "
              f"{s_h['gtr_after']:.1f}% | {s_h['gtr_delta']:+.2f}pp |")

    md.append("\n## Detailed: by (method × model)\n")
    md.append("| method | model | n | CF % | conf reduce | edge recall after | edge recall Δ | dir acc | iters |")
    md.append("|--------|-------|--:|-----:|-----------:|------------------:|--------------:|--------:|------:|")
    for method in ALL_METHODS:
        ms = models if method in ("llm_baseline", "llm_edge_impact") else ("n/a",)
        for model in ms:
            s = stats(by_meth_model[(method, model)])
            if s["n"] == 0: continue
            md.append(f"| {method} | {model} | {s['n']} | {s['cf']:.1f} | "
                      f"{s['cred']:.1f} | {s['gtr_after']:.1f} | "
                      f"{s['gtr_delta']:+.2f}pp | {s['gtd_after']:.1f} | "
                      f"{s['iters']:.1f} |")

    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out_root}/summary.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
