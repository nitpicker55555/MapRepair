"""Experiment 7: heuristic (no-LLM) repair on MANGO graphs.

Same MANGO inputs as Exp 5, but the repair agent is the algorithmic
HeuristicRepairAgent. If the heuristic does as well as or better than the
LLM agents on real LLM-noisy graphs, that is the strongest possible
evidence that the *algorithmic* contribution carries the framework.

Reports the same metrics as Exp 5 (pre/post edge_recall,
strict_direction_match, permissive_direction_match) so the two
experiments are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from maprepair.conflict import detect_all
from maprepair.mango import ground_truth_graph
from maprepair.mapping import load_legacy_edges
from maprepair.agents.heuristic import HeuristicRepairAgent

from experiments.exp05_mango_llm import _evaluate


def _run(game: str, edges_dir: Path, max_iterations: int) -> Dict:
    edges_path = edges_dir / f"{game}_edges.json"
    if not edges_path.exists():
        return {"game": game, "error": "edges missing"}
    broken = load_legacy_edges(edges_path)
    gt = ground_truth_graph(game)
    pre = _evaluate(gt, broken)
    pre_conflicts = detect_all(broken)
    if not pre_conflicts:
        return {
            "game": game,
            "num_conflicts": 0, "iterations": 0, "success": True,
            "pre_edge_recall": pre["edge_recall"],
            "pre_strict_direction_match": pre["strict_direction_match"],
            "post_edge_recall": pre["edge_recall"],
            "post_strict_direction_match": pre["strict_direction_match"],
            "post_permissive_direction_match": pre["permissive_direction_match"],
            "strict_direction_delta": 0.0,
            "edge_recall_delta": 0.0,
        }
    agent = HeuristicRepairAgent()
    result = agent.repair(broken, max_iterations=max_iterations)
    post = _evaluate(gt, result.graph_after)
    return {
        "game": game,
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
                    default=Path("/Users/puzhen/Downloads/maprepair/llm_experiments/results/exp1_mapping/gpt-4.1"))
    ap.add_argument("--out-root", type=Path, default=Path("results/exp07"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-iterations", type=int, default=200)
    args = ap.parse_args()

    games = sorted(p.stem.replace("_edges", "") for p in args.edges_dir.glob("*_edges.json"))
    print(f"Heuristic on MANGO: {len(games)} games, workers={args.workers}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run, g, args.edges_dir, args.max_iterations): g for g in games}
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception as e:
                row = {"error": str(e)}
            rows.append(row)
            (args.out_root / f"{row['game']}.json").write_text(json.dumps(row, indent=2))
            if "error" not in row:
                print(f"  {row['game']:<15} conflicts={row['num_conflicts']:>3} success={row['success']!s:<5} "
                      f"strict_delta={row['strict_direction_delta']:+.3f}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    rs = [r for r in rows if "error" not in r]
    if rs:
        success_pct = 100 * sum(1 for r in rs if r["success"]) / len(rs)
        non_trivial = [r for r in rs if r["num_conflicts"] > 0]
        nt_success_pct = 100 * sum(1 for r in non_trivial if r["success"]) / max(1, len(non_trivial))
        avg_delta = statistics.mean(r["strict_direction_delta"] for r in rs)
        avg_pre = statistics.mean(r["pre_strict_direction_match"] for r in rs)
        avg_post = statistics.mean(r["post_strict_direction_match"] for r in rs)
        md = [
            "# Experiment 7 - HeuristicRepairAgent on MANGO graphs\n",
            f"Games: {len(rs)} | non-trivial (with conflicts): {len(non_trivial)}",
            f"Conflict-free rate (all): {success_pct:.1f}%",
            f"Conflict-free rate (non-trivial): {nt_success_pct:.1f}%",
            f"Mean pre  strict_direction_match: {avg_pre*100:.2f}%",
            f"Mean post strict_direction_match: {avg_post*100:.2f}%",
            f"Mean delta (post-pre): {avg_delta*100:+.2f}pp",
            "",
            "## Per-game",
            "| game | conflicts | success | iters | pre_strict | post_strict | delta |",
            "|------|----------:|---------|------:|-----------:|------------:|------:|",
        ]
        for r in sorted(rs, key=lambda x: -x.get("num_conflicts", 0)):
            md.append(f"| {r['game']} | {r['num_conflicts']} | {r['success']!s} | "
                      f"{r['iterations']} | "
                      f"{r['pre_strict_direction_match']*100:.1f}% | "
                      f"{r['post_strict_direction_match']*100:.1f}% | "
                      f"{r['strict_direction_delta']*100:+.2f}pp |")
        (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
        print("\n" + "\n".join(md[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
