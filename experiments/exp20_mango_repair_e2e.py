"""Experiment 20: end-to-end repair on real LLM-built MANGO maps.

The "real benchmark" validation. Pipeline:

  exp14 V3 clean walkthroughs       (real MANGO game traces)
   -> gpt-4.1 incremental mapping   (real LLM construction stage)
   -> noisy NavGraph                (one per game, with conflicts)
   -> exp19-validated repair        (LLM-MapRepair edge_impact mode)
   -> measure GT edge recall lift   (real-benchmark metric)

For each of the 53 games we compare four repair conditions:

  no_repair         baseline (LLM-built map untouched)
  heuristic_remove  non-LLM algorithmic baseline
  llm_baseline      LLM with no LCA / no EIS (action space: modify/remove/skip)
  llm_edge_impact   LLM + LCA candidate filter + impact scoring (OUR METHOD)

Outputs:
  results/exp20/raw.json
  results/exp20/summary.md
  results/exp20/per_game/<game>.md  (optional)
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
from maprepair.mango import ground_truth_graph


MODES = ("no_repair", "heuristic_remove", "llm_baseline", "llm_edge_impact")


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def load_llm_map(edges_path: Path) -> NavGraph:
    """Build a NavGraph from exp14's per-game edges JSON."""
    g = NavGraph()
    edges = json.loads(edges_path.read_text())
    for e in edges:
        u = _norm(e.get("src_node"))
        v = _norm(e.get("dst_node"))
        a = _norm(e.get("action"))
        if not u or not v or u == v or not a:
            continue
        if g.has_edge(u, v):
            continue
        try:
            g.add_edge(u, v, a, add_auto_reverse=True)
        except Exception:
            continue
    return g


def measure(work: NavGraph, gt: NavGraph) -> Dict:
    gt_pairs: Set[Tuple[str, str]] = {(e.source, e.target) for e in gt.primary_edges()}
    gt_with_dir: Set[Tuple[str, str, str]] = {
        (e.source, e.target, e.direction) for e in gt.primary_edges()
    }
    gt_nodes = set(gt.nodes())

    pred_pairs = {(e.source, e.target) for e in work.primary_edges()}
    pred_with_dir = {(e.source, e.target, e.direction) for e in work.primary_edges()}
    pred_nodes = set(work.nodes())

    def prf(pred, gold):
        if not gold or not pred:
            return 0.0, 0.0
        tp = len(pred & gold)
        return tp / len(pred), tp / len(gold)

    edge_prec, edge_rec = prf(pred_pairs, gt_pairs)
    dir_prec, dir_rec = prf(pred_with_dir, gt_with_dir)
    node_rec = (len(pred_nodes & gt_nodes) / len(gt_nodes)) if gt_nodes else 0.0
    overlap = pred_pairs & gt_pairs
    gt_dir_map = {(e.source, e.target): e.direction for e in gt.primary_edges()}
    pred_dir_map = {(u, v): d for (u, v, d) in pred_with_dir}
    dir_correct = sum(1 for e in overlap if pred_dir_map.get(e) == gt_dir_map.get(e))
    dir_acc = dir_correct / len(overlap) if overlap else 0.0
    return {
        "n_pred_edges": len(pred_pairs),
        "n_gt_edges": len(gt_pairs),
        "edge_recall": edge_rec,
        "edge_precision": edge_prec,
        "edge_with_dir_recall": dir_rec,
        "edge_with_dir_precision": dir_prec,
        "direction_accuracy": dir_acc,
        "node_recall": node_rec,
    }


def run_one(game: str, mode: str, model: str, exp14_dir: Path,
             max_iterations: int, max_attempts: int) -> Dict:
    edges_path = exp14_dir / f"{game}_edges.json"
    if not edges_path.exists():
        return {"game": game, "mode": mode, "error": "no edges file"}
    work = load_llm_map(edges_path)
    gt = ground_truth_graph(game)

    metrics_before = measure(work, gt)
    n_conflicts_before = len(detect_all(work))

    t0 = time.time()
    actions = []
    iterations = 0
    if mode == "no_repair":
        repaired = work
    elif mode == "heuristic_remove":
        r = HeuristicRepairAgent(prefer_remove=True).repair(
            work.copy(), max_iterations=max_iterations)
        repaired = r.graph_after; actions = r.actions; iterations = r.iterations
    elif mode == "llm_baseline":
        agent = LLMRepairAgent(model=model, mode="baseline",
                                 max_attempts_per_conflict=max_attempts,
                                 lookahead=False)
        r = agent.repair(work.copy(), max_iterations=max_iterations)
        repaired = r.graph_after; actions = r.actions; iterations = r.iterations
    elif mode == "llm_edge_impact":
        agent = LLMRepairAgent(model=model, mode="edge_impact",
                                 max_attempts_per_conflict=max_attempts,
                                 lookahead=False)
        r = agent.repair(work.copy(), max_iterations=max_iterations)
        repaired = r.graph_after; actions = r.actions; iterations = r.iterations
    else:
        raise ValueError(mode)
    elapsed = time.time() - t0

    n_conflicts_after = len(detect_all(repaired))
    metrics_after = measure(repaired, gt)
    action_kinds = [a.kind for a in actions]

    return {
        "game": game,
        "mode": mode,
        "n_conflicts_before": n_conflicts_before,
        "n_conflicts_after": n_conflicts_after,
        "conflict_reduction_pct": (
            100 * (n_conflicts_before - n_conflicts_after) / max(1, n_conflicts_before)
        ),
        "edge_recall_before": metrics_before["edge_recall"],
        "edge_recall_after": metrics_after["edge_recall"],
        "edge_recall_delta": metrics_after["edge_recall"] - metrics_before["edge_recall"],
        "edge_precision_after": metrics_after["edge_precision"],
        "edge_with_dir_recall_after": metrics_after["edge_with_dir_recall"],
        "direction_accuracy_after": metrics_after["direction_accuracy"],
        "node_recall_after": metrics_after["node_recall"],
        "n_pred_edges_before": metrics_before["n_pred_edges"],
        "n_pred_edges_after": metrics_after["n_pred_edges"],
        "n_actions": len(actions),
        "n_modify": action_kinds.count("modify_edge"),
        "n_remove": action_kinds.count("remove_edge"),
        "iterations": iterations,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp14-dir", type=Path,
                    default=Path("results/exp14/gpt-4.1"))
    ap.add_argument("--out-root", type=Path, default=Path("results/exp20"))
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-iterations", type=int, default=40)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--games", default="")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.games:
        games = [g.strip() for g in args.games.split(",") if g.strip()]
    else:
        games = sorted(
            p.stem.replace("_edges", "")
            for p in args.exp14_dir.glob("*_edges.json")
        )
    print(f"Games: {len(games)}; modes: {MODES}")

    jobs: List[Tuple[str, str]] = []
    for g in games:
        for m in MODES:
            jobs.append((g, m))
    print(f"Total runs: {len(jobs)}", flush=True)

    rows: List[Dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_one, g, m, args.model, args.exp14_dir,
                       args.max_iterations, args.max_attempts): (g, m)
            for g, m in jobs
        }
        done = 0
        for fut in as_completed(futs):
            g, m = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"game": g, "mode": m, "error": str(e)}
            rows.append(row)
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] {time.time()-t0:.0f}s", flush=True)

    (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if "error" not in r]
    by_mode = defaultdict(list)
    for r in ok:
        by_mode[r["mode"]].append(r)

    def stats(rs):
        if not rs:
            return {}
        return {
            "n": len(rs),
            "n_pred_before": statistics.mean(r["n_pred_edges_before"] for r in rs),
            "n_pred_after": statistics.mean(r["n_pred_edges_after"] for r in rs),
            "edge_recall_before": 100 * statistics.mean(r["edge_recall_before"] for r in rs),
            "edge_recall_after": 100 * statistics.mean(r["edge_recall_after"] for r in rs),
            "edge_recall_delta_pp": 100 * statistics.mean(r["edge_recall_delta"] for r in rs),
            "edge_precision_after": 100 * statistics.mean(r["edge_precision_after"] for r in rs),
            "edge_with_dir_recall_after": 100 * statistics.mean(r["edge_with_dir_recall_after"] for r in rs),
            "direction_accuracy_after": 100 * statistics.mean(r["direction_accuracy_after"] for r in rs),
            "node_recall_after": 100 * statistics.mean(r["node_recall_after"] for r in rs),
            "conflict_reduction_pct": statistics.mean(r["conflict_reduction_pct"] for r in rs),
            "conflicts_before": statistics.mean(r["n_conflicts_before"] for r in rs),
            "conflicts_after": statistics.mean(r["n_conflicts_after"] for r in rs),
            "iterations": statistics.mean(r["iterations"] for r in rs),
            "actions": statistics.mean(r["n_actions"] for r in rs),
            "remove_per_run": statistics.mean(r["n_remove"] for r in rs),
            "modify_per_run": statistics.mean(r["n_modify"] for r in rs),
        }

    md = [
        f"# Experiment 20 — End-to-end MANGO repair (model={args.model})\n",
        f"Real-benchmark validation. Input: exp14 V3 gpt-4.1 LLM-built maps "
        f"on the 53 refined MANGO games. Compared 4 repair conditions per game.\n",
        f"Total runs: {len(rows)} (valid {len(ok)}, errors {sum(1 for r in rows if 'error' in r)}).\n",
        "## Aggregate by mode (macro across 53 games)\n",
        "| mode | n | conf reduction % | edge recall before | edge recall after | Δ edge recall | edge prec after | dir acc after | iters | actions/run | remove/run |",
        "|------|--:|-----------------:|-------------------:|------------------:|--------------:|----------------:|--------------:|------:|------------:|-----------:|",
    ]
    for mode in MODES:
        s = stats(by_mode[mode])
        if not s:
            continue
        md.append(
            f"| {mode} | {s['n']} | {s['conflict_reduction_pct']:.1f} | "
            f"{s['edge_recall_before']:.1f} | {s['edge_recall_after']:.1f} | "
            f"{s['edge_recall_delta_pp']:+.2f}pp | {s['edge_precision_after']:.1f} | "
            f"{s['direction_accuracy_after']:.1f} | {s['iterations']:.1f} | "
            f"{s['actions']:.1f} | {s['remove_per_run']:.1f} |"
        )

    # Per-game (edge_impact only, sorted by lift)
    md.append("\n## Per-game lift (llm_edge_impact)\n")
    md.append("| game | edges before | conflicts before | edges after | conflicts after | edge recall before | after | Δ | actions | remove |")
    md.append("|------|-------------:|-----------------:|------------:|----------------:|-------------------:|------:|--:|--------:|-------:|")
    rows_ei = by_mode.get("llm_edge_impact", [])
    for r in sorted(rows_ei, key=lambda x: -x["edge_recall_delta"]):
        md.append(
            f"| {r['game']} | {r['n_pred_edges_before']} | {r['n_conflicts_before']} | "
            f"{r['n_pred_edges_after']} | {r['n_conflicts_after']} | "
            f"{100*r['edge_recall_before']:.1f} | {100*r['edge_recall_after']:.1f} | "
            f"{100*r['edge_recall_delta']:+.2f}pp | {r['n_actions']} | {r['n_remove']} |"
        )

    # Identify subset where edge_impact gave a meaningful lift (>= +1pp)
    lifters = [r for r in rows_ei if r["edge_recall_delta"] >= 0.01]
    flat   = [r for r in rows_ei if -0.01 < r["edge_recall_delta"] < 0.01]
    hurt   = [r for r in rows_ei if r["edge_recall_delta"] <= -0.01]
    md.append("\n## Subset summary (llm_edge_impact)\n")
    md.append(f"- Games with **>= +1pp** edge-recall lift: **{len(lifters)}**/{len(rows_ei)}")
    md.append(f"- Games **flat** (|Δ| < 1pp): {len(flat)}/{len(rows_ei)}")
    md.append(f"- Games **hurt** (<= -1pp): {len(hurt)}/{len(rows_ei)}")
    if lifters:
        avg_lift = 100 * statistics.mean(r["edge_recall_delta"] for r in lifters)
        md.append(f"- Mean lift on the lifter subset: **+{avg_lift:.2f}pp** edge recall")

    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out_root}/summary.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
