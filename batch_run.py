"""
Batch runner for MapRepair on the MANGO benchmark.

Processes one or more games end-to-end using the pipeline described in the
paper, optionally in parallel via a process pool, and saves per-game JSON
reports plus an aggregated summary.

Usage examples:
    # Run a single game
    python batch_run.py --games zork1

    # Run a list of games
    python batch_run.py --games zork1 zork2 cutthroat

    # Run all games in the dataset
    python batch_run.py --all

    # Limit walkthrough length and run in parallel
    python batch_run.py --all --max-steps 50 --workers 4

    # Use a different LLM backbone
    python batch_run.py --games zork1 --model gpt-4o-mini
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from map_slam_system import MapSLAMSystem


def _convert(obj: Any, depth: int = 0) -> Any:
    """Recursively convert custom objects into JSON-serializable form."""
    if depth > 10:
        return f"<depth_limit:{type(obj).__name__}>"
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_convert(x, depth + 1) for x in obj]
    if isinstance(obj, set):
        return [_convert(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _convert(v, depth + 1) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        out = {"__class__": type(obj).__name__}
        for k, v in obj.__dict__.items():
            try:
                out[k] = _convert(v, depth + 1)
            except Exception as e:
                out[k] = f"<error:{e}>"
        return out
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return f"<{type(obj).__name__}:{obj}>"


def run_one(game: str, data_dir: str, model: str,
            max_steps: Optional[int], output_dir: str) -> Dict:
    """Process one game and return a serializable summary."""
    slam = MapSLAMSystem(data_dir=data_dir, model=model)
    results = slam.process_game(game, max_steps=max_steps, verbose=False)

    # Persist full per-game artifacts
    game_dir = os.path.join(output_dir, game)
    slam.save_results(output_dir=output_dir)

    summary = {
        "game": game,
        "processed_steps": results["processed_steps"],
        "final_graph_stats": _convert(results["final_graph_stats"]),
        "conflicts_summary": _convert(results["conflicts_summary"]),
    }
    eis = results.get("edge_impact_analysis") or {}
    if eis.get("ranked_candidates"):
        summary["top_eis_candidate"] = eis["ranked_candidates"][0]
        summary["total_candidates"] = eis.get("total_candidates", 0)

    with open(os.path.join(game_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="/Users/puzhen/Desktop/mango/data",
                        help="Path to the (refined) MANGO dataset directory")
    parser.add_argument("--games", nargs="+", default=None,
                        help="Specific games to run")
    parser.add_argument("--all", action="store_true",
                        help="Run on every game in the dataset")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap walkthrough length per game")
    parser.add_argument("--model", default="gpt-4o",
                        help="LLM model name passed to NavigationGraph")
    parser.add_argument("--output-dir", default="./output",
                        help="Where to write per-game JSON reports")
    parser.add_argument("--workers", type=int, default=1,
                        help="Process pool size for parallel runs")
    args = parser.parse_args()

    # Resolve game list
    if args.all:
        slam = MapSLAMSystem(data_dir=args.data_dir, model=args.model)
        games = slam.dataset.games
    elif args.games:
        games = args.games
    else:
        parser.error("specify --games <name> [...] or --all")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Running MapRepair on {len(games)} game(s) "
          f"with model={args.model}, workers={args.workers}")

    summaries: List[Dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_one, g, args.data_dir, args.model,
                                args.max_steps, args.output_dir): g
                    for g in games}
            for fut in as_completed(futs):
                game = futs[fut]
                try:
                    s = fut.result()
                    summaries.append(s)
                    print(f"  done: {game} "
                          f"({s['processed_steps']} steps, "
                          f"{s['conflicts_summary']['total_conflicts']} conflicts)")
                except Exception as e:
                    print(f"  failed: {game}: {e}")
    else:
        for game in games:
            try:
                s = run_one(game, args.data_dir, args.model,
                            args.max_steps, args.output_dir)
                summaries.append(s)
                print(f"  done: {game} "
                      f"({s['processed_steps']} steps, "
                      f"{s['conflicts_summary']['total_conflicts']} conflicts)")
            except Exception as e:
                print(f"  failed: {game}: {e}")

    # Aggregate summary
    aggregate = {
        "total_games": len(summaries),
        "total_processed_steps": sum(s["processed_steps"] for s in summaries),
        "total_conflicts": sum(s["conflicts_summary"]["total_conflicts"]
                               for s in summaries),
        "by_type": {
            "topological": sum(s["conflicts_summary"]["by_type"].get("topological", 0)
                               for s in summaries),
            "directional": sum(s["conflicts_summary"]["by_type"].get("directional", 0)
                               for s in summaries),
            "naming":      sum(s["conflicts_summary"]["by_type"].get("naming", 0)
                               for s in summaries),
        },
        "summaries": summaries,
    }
    with open(os.path.join(args.output_dir, "aggregate_summary.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate summary saved to "
          f"{os.path.join(args.output_dir, 'aggregate_summary.json')}")
    print(f"  total games: {aggregate['total_games']}")
    print(f"  total conflicts: {aggregate['total_conflicts']}")
    print(f"  by type: {aggregate['by_type']}")


if __name__ == "__main__":
    main()
