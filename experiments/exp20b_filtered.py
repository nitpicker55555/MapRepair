"""Experiment 20b: same as exp20 but restricts the conflict detector to
direction + topology only.

Rationale: exp10's analysis found ~36% of naming-conflict detections
on MANGO are false positives caused by the position-inference BFS
assuming Euclidean geometry on deliberately non-Euclidean game maps.
The original paper's evaluation also skipped spatial-overlap detection
on MANGO for this reason.

If we run repair only on the algorithm-target conflict set
(direction + topology, which IS Euclidean-clean), we test whether
the algorithm helps on the conflicts it was designed to fix.
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

import maprepair.conflict as _conflict_mod
from maprepair.conflict import (
    detect_direction_conflicts, detect_topology_conflicts,
    Conflict,
)
from maprepair.agents.heuristic import HeuristicRepairAgent
from maprepair.agents.llm_agent import LLMRepairAgent
from maprepair.graph import NavGraph
from maprepair.mango import ground_truth_graph


# Monkey-patch detect_all to skip naming (the false-positive detector on
# non-Euclidean MANGO maps). Restores after experiment.
def _filtered_detect_all(graph: NavGraph) -> List[Conflict]:
    seen: Set[str] = set()
    out: List[Conflict] = []
    for c in (*detect_direction_conflicts(graph),
              *detect_topology_conflicts(graph)):
        cid = c.conflict_id()
        if cid in seen:
            continue
        seen.add(cid)
        out.append(c)
    return out

_conflict_mod.detect_all = _filtered_detect_all


# Now import the run_one machinery from exp20 — it will use the patched
# detector via maprepair.conflict.detect_all.
from experiments.exp20_mango_repair_e2e import (
    load_llm_map, measure, run_one as _orig_run_one, MODES,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp14-dir", type=Path,
                    default=Path("results/exp14/gpt-4.1"))
    ap.add_argument("--out-root", type=Path, default=Path("results/exp20b"))
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-iterations", type=int, default=40)
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    games = sorted(
        p.stem.replace("_edges", "")
        for p in args.exp14_dir.glob("*_edges.json")
    )
    print(f"Games: {len(games)}; modes: {MODES} (filtered: direction+topology only)")

    jobs: List[Tuple[str, str]] = [(g, m) for g in games for m in MODES]
    print(f"Total runs: {len(jobs)}", flush=True)

    rows: List[Dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_orig_run_one, g, m, args.model, args.exp14_dir,
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
        if not rs: return {}
        return {
            "n": len(rs),
            "edge_recall_before": 100 * statistics.mean(r["edge_recall_before"] for r in rs),
            "edge_recall_after": 100 * statistics.mean(r["edge_recall_after"] for r in rs),
            "edge_recall_delta_pp": 100 * statistics.mean(r["edge_recall_delta"] for r in rs),
            "edge_precision_after": 100 * statistics.mean(r["edge_precision_after"] for r in rs),
            "direction_accuracy_after": 100 * statistics.mean(r["direction_accuracy_after"] for r in rs),
            "conflict_reduction_pct": statistics.mean(r["conflict_reduction_pct"] for r in rs),
            "conflicts_before": statistics.mean(r["n_conflicts_before"] for r in rs),
            "conflicts_after": statistics.mean(r["n_conflicts_after"] for r in rs),
            "iterations": statistics.mean(r["iterations"] for r in rs),
            "actions": statistics.mean(r["n_actions"] for r in rs),
            "remove_per_run": statistics.mean(r["n_remove"] for r in rs),
        }

    md = [
        f"# Experiment 20b — End-to-end MANGO repair, FILTERED detector ({args.model})\n",
        "Same setup as exp20 but `detect_all` is restricted to direction + "
        "topology conflicts only (skips the naming detector, which has a known "
        "36% false-positive rate on non-Euclidean MANGO maps — original paper "
        "also skipped this).\n",
        f"Total runs: {len(rows)} (valid {len(ok)}).\n",
        "## Aggregate by mode (macro across 53 games)\n",
        "| mode | n | conf reduction % | conflicts before | conflicts after | edge recall before | edge recall after | Δ edge recall | edge prec after | dir acc after | iters | actions | remove |",
        "|------|--:|-----------------:|----------------:|---------------:|-------------------:|------------------:|--------------:|----------------:|--------------:|------:|--------:|-------:|",
    ]
    for mode in MODES:
        s = stats(by_mode[mode])
        if not s: continue
        md.append(
            f"| {mode} | {s['n']} | {s['conflict_reduction_pct']:.1f} | "
            f"{s['conflicts_before']:.1f} | {s['conflicts_after']:.1f} | "
            f"{s['edge_recall_before']:.1f} | {s['edge_recall_after']:.1f} | "
            f"{s['edge_recall_delta_pp']:+.2f}pp | {s['edge_precision_after']:.1f} | "
            f"{s['direction_accuracy_after']:.1f} | {s['iterations']:.1f} | "
            f"{s['actions']:.1f} | {s['remove_per_run']:.1f} |"
        )

    md.append("\n## Per-game lift (llm_edge_impact, sorted by lift desc)\n")
    md.append("| game | edges before | conflicts before | edges after | conflicts after | edge recall before | after | Δ | actions | remove |")
    md.append("|------|-------------:|----------------:|------------:|---------------:|-------------------:|------:|--:|--------:|-------:|")
    rows_ei = by_mode.get("llm_edge_impact", [])
    for r in sorted(rows_ei, key=lambda x: -x["edge_recall_delta"]):
        md.append(
            f"| {r['game']} | {r['n_pred_edges_before']} | {r['n_conflicts_before']} | "
            f"{r['n_pred_edges_after']} | {r['n_conflicts_after']} | "
            f"{100*r['edge_recall_before']:.1f} | {100*r['edge_recall_after']:.1f} | "
            f"{100*r['edge_recall_delta']:+.2f}pp | {r['n_actions']} | {r['n_remove']} |"
        )

    lifters = [r for r in rows_ei if r["edge_recall_delta"] >= 0.01]
    flat    = [r for r in rows_ei if -0.01 < r["edge_recall_delta"] < 0.01]
    hurt    = [r for r in rows_ei if r["edge_recall_delta"] <= -0.01]
    md.append("\n## Subset summary (llm_edge_impact)\n")
    md.append(f"- Games with **>= +1pp** edge-recall lift: **{len(lifters)}**/{len(rows_ei)}")
    md.append(f"- Games **flat** (|Δ| < 1pp): {len(flat)}/{len(rows_ei)}")
    md.append(f"- Games **hurt** (<= -1pp): {len(hurt)}/{len(rows_ei)}")
    if lifters:
        avg = 100 * statistics.mean(r["edge_recall_delta"] for r in lifters)
        md.append(f"- Mean lift on the lifter subset: **+{avg:.2f}pp** edge recall")
    if hurt:
        avg_h = 100 * statistics.mean(r["edge_recall_delta"] for r in hurt)
        md.append(f"- Mean drop on hurt subset: {avg_h:.2f}pp")

    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out_root}/summary.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
