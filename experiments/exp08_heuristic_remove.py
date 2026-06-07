"""Experiment 8: variant of HeuristicRepairAgent that prefers EDGE REMOVAL.

Motivation: On LLM-generated graphs, conflicts often arise because the
LLM hallucinated an edge that should not exist. Rotating that edge's
direction only papers over the issue. This variant tries removing the
high-impact candidate first, and only falls back to rotation if
remove-then-detect introduces *more* conflicts than it resolves.

Runs on both MANGO (the noisy LLM-generated graphs) and the synthetic
exp03 cohort, so we can see which strategy wins on each.
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
from maprepair.synth import make_synthetic

from experiments.exp03_heuristic_repair import evaluate as eval_synth
from experiments.exp05_mango_llm import _evaluate as eval_mango


# --------------------------------------------------------------------
# MANGO portion
# --------------------------------------------------------------------

def _mango_one(game: str, edges_dir: Path, max_iters: int) -> Dict:
    edges_path = edges_dir / f"{game}_edges.json"
    if not edges_path.exists():
        return {"game": game, "error": "edges missing"}
    broken = load_legacy_edges(edges_path)
    gt = ground_truth_graph(game)
    pre = eval_mango(gt, broken)
    pre_conflicts = detect_all(broken)
    if not pre_conflicts:
        return {
            "game": game,
            "num_conflicts": 0, "iterations": 0, "success": True,
            "pre_strict": pre["strict_direction_match"],
            "post_strict": pre["strict_direction_match"],
            "pre_recall": pre["edge_recall"],
            "post_recall": pre["edge_recall"],
            "delta_strict": 0.0,
            "delta_recall": 0.0,
        }
    agent = HeuristicRepairAgent(prefer_remove=True)
    result = agent.repair(broken, max_iterations=max_iters)
    post = eval_mango(gt, result.graph_after)
    return {
        "game": game,
        "num_conflicts": len(pre_conflicts),
        "iterations": result.iterations,
        "success": result.success,
        "pre_strict": pre["strict_direction_match"],
        "post_strict": post["strict_direction_match"],
        "pre_recall": pre["edge_recall"],
        "post_recall": post["edge_recall"],
        "delta_strict": post["strict_direction_match"] - pre["strict_direction_match"],
        "delta_recall": post["edge_recall"] - pre["edge_recall"],
    }


# --------------------------------------------------------------------
# Synthetic portion (re-uses Exp 3's planning + evaluate)
# --------------------------------------------------------------------

def _synth_one(family: str, size: int, err_type: str, num_err: int, seed: int,
               prefer_remove: bool) -> Dict:
    if family == "grid":
        size = max(size, 3)
        spec = make_synthetic("grid", rows=size, cols=size,
                              error_mix={err_type: num_err}, seed=seed)
    else:
        spec = make_synthetic(family, size=size,
                              error_mix={err_type: num_err}, seed=seed)
    if not detect_all(spec.graph):
        return {}
    agent = HeuristicRepairAgent(prefer_remove=prefer_remove)
    res = agent.repair(spec.graph.copy(), max_iterations=30)
    ev = eval_synth(spec, res.graph_after)
    return {
        "family": family, "size": size, "err_type": err_type,
        "num_err": num_err, "seed": seed, "prefer_remove": prefer_remove,
        "conflict_free": res.success,
        "iterations": res.iterations,
        "dir_acc": ev["gt_direction_accuracy"],
        "recovered": ev["recovered_errors"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges-dir", type=Path,
                    default=Path("/Users/puzhen/Downloads/maprepair/llm_experiments/results/exp1_mapping/gpt-4.1"))
    ap.add_argument("--out-root", type=Path, default=Path("results/exp08"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-iters", type=int, default=300)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    # ----- MANGO portion -----
    games = sorted(p.stem.replace("_edges", "") for p in args.edges_dir.glob("*_edges.json"))
    mango_rows: List[Dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_mango_one, g, args.edges_dir, args.max_iters): g for g in games}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"error": str(e)}
            mango_rows.append(r)
    (args.out_root / "mango.json").write_text(json.dumps(mango_rows, indent=2))

    # ----- Synthetic portion (same plan as Exp 3) -----
    plan = [
        ("tree",   4, "direction", 1),
        ("tree",   4, "topology",  1),
        ("random", 30, "direction", 1),
        ("random", 30, "topology",  1),
        ("grid",   4, "direction", 1),
        ("grid",   4, "topology",  1),
    ]
    synth_rows: List[Dict] = []
    for (family, size, err, n_err) in plan:
        for seed in range(args.seeds):
            for prefer in (False, True):
                r = _synth_one(family, size, err, n_err, seed, prefer)
                if r:
                    synth_rows.append(r)
    (args.out_root / "synth.json").write_text(json.dumps(synth_rows, indent=2))

    # ----- Summary -----
    mango_ok = [r for r in mango_rows if "error" not in r and r["num_conflicts"] > 0]
    if mango_ok:
        m_success = 100 * sum(1 for r in mango_ok if r["success"]) / len(mango_ok)
        m_delta = statistics.mean(r["delta_strict"] for r in mango_ok)
        m_recall_delta = statistics.mean(r["delta_recall"] for r in mango_ok)
    else:
        m_success = m_delta = m_recall_delta = float("nan")

    md = [
        "# Experiment 8 - HeuristicRepairAgent with prefer_remove=True",
        "",
        "## MANGO (non-trivial games)",
        f"games: {len(mango_ok)}",
        f"conflict_free_rate: {m_success:.1f}%",
        f"mean delta strict_direction_match: {m_delta*100:+.2f}pp",
        f"mean delta edge_recall: {m_recall_delta*100:+.2f}pp",
        "",
        "## Synthetic (paired rotate vs remove)",
        "| family | err | n | rotate cf | remove cf | rotate dir | remove dir |",
        "|--------|-----|--:|----------:|----------:|-----------:|-----------:|",
    ]
    by_cohort: Dict = {}
    for r in synth_rows:
        by_cohort.setdefault((r["family"], r["err_type"], r["prefer_remove"]), []).append(r)
    cohorts = sorted({(f, e) for (f, e, _) in by_cohort.keys()})
    for f, e in cohorts:
        rot = by_cohort.get((f, e, False), [])
        rem = by_cohort.get((f, e, True), [])
        if not rot or not rem: continue
        def avg(rs, k): return statistics.mean(x.get(k, 0) or 0 for x in rs)
        def pct(rs, k): return 100 * sum(1 for x in rs if x.get(k)) / len(rs)
        md.append(f"| {f} | {e} | {len(rot)} | "
                  f"{pct(rot,'conflict_free'):.1f}% | {pct(rem,'conflict_free'):.1f}% | "
                  f"{100*avg(rot,'dir_acc'):.1f}% | {100*avg(rem,'dir_acc'):.1f}% |")
    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
