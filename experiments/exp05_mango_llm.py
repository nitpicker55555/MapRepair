"""Experiment 5: LLM repair on MANGO graphs, refined accuracy.

Uses the existing gpt-4.1 LLM-mapping outputs (legacy `{game}_edges.json`)
from the older maprepair checkout as the conflict-laden inputs, but routes
them through the clean NavGraph / Conflict / LLMRepairAgent stack.

Refinements vs the earlier MANGO experiment:

  * **Strict accuracy** treats a conflict as "correct" only if the post-repair
    graph state on the touched edge matches GT direction AND no new GT edges
    are missing.
  * **Permissive accuracy** is a tolerance variant: the touched edge has any
    direction consistent with the GT topology (e.g. the LLM picked the
    opposite direction but the auto-reverse propagates correctness).
  * **Edge-level recall / direction accuracy** independent of "conflicts", so
    the result also reflects pure graph-quality improvement.

Outputs:
  results/exp05/<model>/<mode>/<game>.json
  results/exp05/<model>/<mode>/_aggregate.json
  results/exp05/summary.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from maprepair.conflict import detect_all
from maprepair.graph import NavGraph
from maprepair.mango import ground_truth_graph, list_games
from maprepair.mapping import load_legacy_edges
from maprepair.agents.llm_agent import LLMRepairAgent


MODES = ("baseline", "edge_impact", "vc_only", "vc_ei")


def _evaluate(gt: NavGraph, repaired: NavGraph) -> Dict[str, float]:
    gt_edges = {(e.source, e.target): e.direction for e in gt.primary_edges()}
    rp_edges = {(e.source, e.target): e.direction for e in repaired.primary_edges()}
    overlap = set(gt_edges) & set(rp_edges)
    if not gt_edges:
        return {"edge_recall": 0.0, "edge_direction_accuracy": 0.0,
                "strict_direction_match": 0.0, "permissive_direction_match": 0.0}
    edge_recall = len(overlap) / len(gt_edges)
    correct = sum(1 for e in overlap if rp_edges[e] == gt_edges[e])
    edge_direction_accuracy = correct / len(overlap) if overlap else 0.0
    # strict: how many GT edges are present with exact direction
    strict = correct / len(gt_edges)
    # permissive: also count cases where the GT reverse is present with the
    # right opposite direction
    perm_correct = correct
    from maprepair.graph import OPPOSITE
    for (u, v), gt_dir in gt_edges.items():
        if (u, v) in rp_edges:
            continue
        # see if reverse repaired captures it
        if (v, u) in rp_edges:
            if rp_edges[(v, u)] == OPPOSITE.get(gt_dir):
                perm_correct += 1
    permissive = perm_correct / len(gt_edges)
    return {
        "edge_recall": edge_recall,
        "edge_direction_accuracy": edge_direction_accuracy,
        "strict_direction_match": strict,
        "permissive_direction_match": permissive,
    }


def _run_one(game: str, mode: str, model: str, edges_dir: Path) -> Dict:
    edges_path = edges_dir / f"{game}_edges.json"
    if not edges_path.exists():
        return {"game": game, "mode": mode, "model": model, "error": "edges missing"}
    broken = load_legacy_edges(edges_path)
    gt = ground_truth_graph(game)
    pre = _evaluate(gt, broken)
    pre_conflicts = detect_all(broken)
    if not pre_conflicts:
        # nothing to repair
        return {
            "game": game, "mode": mode, "model": model,
            "num_conflicts": 0, "iterations": 0, "success": True,
            "pre_edge_recall": pre["edge_recall"],
            "pre_strict_direction_match": pre["strict_direction_match"],
            "post_edge_recall": pre["edge_recall"],
            "post_strict_direction_match": pre["strict_direction_match"],
            "post_permissive_direction_match": pre["permissive_direction_match"],
            "edge_recall_delta": 0.0,
            "strict_direction_delta": 0.0,
        }
    agent = LLMRepairAgent(model=model, mode=mode, max_attempts_per_conflict=10)
    result = agent.repair(broken, max_iterations=80)
    post = _evaluate(gt, result.graph_after)
    return {
        "game": game,
        "mode": mode,
        "model": model,
        "num_conflicts": len(pre_conflicts),
        "iterations": result.iterations,
        "success": result.success,
        "remaining_conflicts": len(result.conflicts_after),
        "pre_edge_recall": pre["edge_recall"],
        "pre_strict_direction_match": pre["strict_direction_match"],
        "pre_edge_direction_accuracy": pre["edge_direction_accuracy"],
        "post_edge_recall": post["edge_recall"],
        "post_strict_direction_match": post["strict_direction_match"],
        "post_permissive_direction_match": post["permissive_direction_match"],
        "post_edge_direction_accuracy": post["edge_direction_accuracy"],
        "edge_recall_delta": post["edge_recall"] - pre["edge_recall"],
        "strict_direction_delta": post["strict_direction_match"] - pre["strict_direction_match"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges-dir", type=Path,
                    default=Path("/Users/puzhen/Downloads/maprepair/llm_experiments/results/exp1_mapping/gpt-4.1"),
                    help="Directory of pre-computed legacy {game}_edges.json files.")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games", default="",
                    help="comma-separated list of games (default: all in edges-dir)")
    ap.add_argument("--out-root", type=Path, default=Path("results/exp05"))
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if args.games:
        games = [g.strip() for g in args.games.split(",")]
    else:
        games = sorted(p.stem.replace("_edges", "") for p in args.edges_dir.glob("*_edges.json"))

    work = [(g, m) for g in games for m in modes]
    print(f"Total runs: {len(work)} ({len(games)} games x {len(modes)} modes), model={args.model}")

    out_dir = args.out_root / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, g, m, args.model, args.edges_dir): (g, m) for (g, m) in work}
        for fut in as_completed(futs):
            g, m = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"game": g, "mode": m, "model": args.model, "error": str(e)}
            rows.append(row)
            (out_dir / m).mkdir(parents=True, exist_ok=True)
            (out_dir / m / f"{g}.json").write_text(json.dumps(row, indent=2))
            print(f"  {g}/{m}: conflicts={row.get('num_conflicts','?')} "
                  f"success={row.get('success','?')} "
                  f"strict={row.get('post_strict_direction_match', '?')}")

    # aggregate per mode
    by_mode: Dict[str, List[Dict]] = {}
    for r in rows:
        if "mode" in r:
            by_mode.setdefault(r["mode"], []).append(r)

    md = [
        f"# Experiment 5 - MANGO LLM repair, refined accuracy (model={args.model})\n",
        f"Games: {len(games)} | Modes: {modes} | Runs: {len(rows)}",
        "\nMetrics:",
        "- pre_*  -- baseline (broken graph) vs GT",
        "- post_* -- repaired graph vs GT",
        "- delta  -- post - pre",
        "- strict_direction_match: |E_repaired_correct_direction| / |E_gt|",
        "- permissive_direction_match: same but also counts cases where the reverse edge captures the GT edge\n",
        "| mode | n | conflicts_total | success% | mean_iters | pre_recall | pre_strict | post_recall | post_strict | post_permissive | mean_delta_strict |",
        "|------|--:|-----------------|---------:|----------:|-----------:|-----------:|------------:|------------:|----------------:|------------------:|",
    ]
    for mode in modes:
        rs = [r for r in by_mode.get(mode, []) if "error" not in r]
        if not rs:
            continue
        def avg(field): return statistics.mean(r.get(field, 0) or 0 for r in rs)
        def pct(field): return 100 * sum(1 for r in rs if r.get(field)) / len(rs)
        conflicts_total = sum(r.get("num_conflicts", 0) for r in rs)
        md.append(f"| {mode} | {len(rs)} | {conflicts_total} | "
                  f"{pct('success'):.1f}% | {avg('iterations'):.2f} | "
                  f"{100*avg('pre_edge_recall'):.2f}% | "
                  f"{100*avg('pre_strict_direction_match'):.2f}% | "
                  f"{100*avg('post_edge_recall'):.2f}% | "
                  f"{100*avg('post_strict_direction_match'):.2f}% | "
                  f"{100*avg('post_permissive_direction_match'):.2f}% | "
                  f"{100*avg('strict_direction_delta'):.2f}pp |")
    (args.out_root / f"summary_{args.model}.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out_root}/summary_{args.model}.md")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
