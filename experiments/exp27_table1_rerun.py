"""Reproduce the paper's Table 1 (4-method ablation on MANGO) numbers
with current code.

Original Table 1 (GPT-4o October 2024 snapshot):
  Edge-Impact Only:    6.39 loops, 75.21% repair, 44.69% acc
  Version Control:     7.44 loops, 63.03% repair, 54.00% acc
  VC+Edge-Impact:      8.20 loops, 68.91% repair, 54.88% acc
  Baseline (GPT-4o):   9.52 loops, 21.85% repair,  5.77% acc

Approach:
  - Load exp14 V3 bootstrap-fixed LLM-built maps for a representative
    subset of MANGO games (5 mid-conflict games).
  - For each game, detect direction+topology conflicts only (skipping
    naming, which has the known 36% FP rate; matches the original
    paper's evaluator).
  - Run 4 LLMRepairAgent modes: baseline / edge_impact / vc_only / vc_ei.
  - For each: count avg loops, repair rate (conflicts before - after / before),
    and accuracy (gt edge_with_dir preserved).
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

from maprepair.agents.llm_agent import LLMRepairAgent
from maprepair.conflict import detect_direction_conflicts, detect_topology_conflicts
from maprepair.graph import NavGraph
from maprepair.mango import ground_truth_graph
import maprepair.conflict as _conflict_mod


# Restrict the detector to direction + topology (matches the original
# paper's evaluation -- naming conflicts had high FP rate on
# non-Euclidean MANGO maps)
def _filtered_detect_all(graph: NavGraph):
    seen = set(); out = []
    for c in (*detect_direction_conflicts(graph), *detect_topology_conflicts(graph)):
        cid = c.conflict_id()
        if cid in seen: continue
        seen.add(cid); out.append(c)
    return out

_conflict_mod.detect_all = _filtered_detect_all


def _norm(s): return (s or "").strip().lower()


def load_llm_map(edges_path: Path) -> NavGraph:
    g = NavGraph()
    for e in json.loads(edges_path.read_text()):
        u = _norm(e.get("src_node")); v = _norm(e.get("dst_node"))
        a = _norm(e.get("action"))
        if not u or not v or u == v or not a or g.has_edge(u, v):
            continue
        try:
            g.add_edge(u, v, a, add_auto_reverse=True)
        except Exception:
            pass
    return g


def gt_dir_correct_count(work: NavGraph, gt: NavGraph) -> int:
    """Count edges in work that match GT direction exactly."""
    gt_dir = {(e.source, e.target): e.direction for e in gt.primary_edges()}
    cnt = 0
    for e in work.primary_edges():
        if (e.source, e.target) in gt_dir and gt_dir[(e.source, e.target)] == e.direction:
            cnt += 1
    return cnt


def run_one(game: str, mode: str, model: str, exp14_dir: Path,
             max_iter: int, max_att: int) -> Dict:
    edges_path = exp14_dir / f"{game}_edges.json"
    if not edges_path.exists():
        return {"game": game, "mode": mode, "error": "no edges"}
    work = load_llm_map(edges_path)
    gt = ground_truth_graph(game)
    conflicts_before = _filtered_detect_all(work)
    n_cb = len(conflicts_before)
    if n_cb == 0:
        return {"game": game, "mode": mode, "n_conflicts_before": 0, "skip": True}

    gt_dir_pre = gt_dir_correct_count(work, gt)

    t0 = time.time()
    agent = LLMRepairAgent(model=model, mode=mode,
                             max_attempts_per_conflict=max_att,
                             lookahead=False)
    r = agent.repair(work.copy(), max_iterations=max_iter)
    elapsed = time.time() - t0
    n_ca = len(r.conflicts_after)
    repaired = n_cb - n_ca
    gt_dir_post = gt_dir_correct_count(r.graph_after, gt)
    # Accuracy: number of actions that result in a GT-direction-correct edge
    correct = gt_dir_post - gt_dir_pre  # net gain in GT-direction-correct edges
    return {
        "game": game, "mode": mode, "model": model,
        "n_conflicts_before": n_cb,
        "n_conflicts_after": n_ca,
        "repaired": repaired,
        "correct_dir_edges_delta": correct,
        "n_actions": len(r.actions),
        "iterations": r.iterations,
        "elapsed_sec": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp14-dir", type=Path,
                    default=Path("results/exp14/gpt-4.1"))
    ap.add_argument("--out-root", type=Path, default=Path("results/exp27_table1"))
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--games", default="cutthroat,detective,inhumane,zork1,zork2,murdac,advent,sherlock,wishbringer,deephome",
                    help="Comma-separated list of MANGO games to run.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-iterations", type=int, default=30)
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    modes = ["baseline", "edge_impact", "vc_only", "vc_ei"]
    print(f"Games: {games}\nModes: {modes}", flush=True)
    jobs = []
    for g in games:
        for m in modes:
            jobs.append((g, m))
    print(f"Total runs: {len(jobs)}", flush=True)

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, g, m, args.model, args.exp14_dir,
                          args.max_iterations, args.max_attempts): (g, m) for g, m in jobs}
        done = 0
        for fut in as_completed(futs):
            g, m = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"game": g, "mode": m, "error": str(e)}
            rows.append(r)
            done += 1
            print(f"  [{done}/{len(jobs)}] {g}/{m}: cf {r.get('n_conflicts_before','?')}->{r.get('n_conflicts_after','?')} "
                  f"acts={r.get('n_actions','?')} iters={r.get('iterations','?')} t={time.time()-t0:.0f}s", flush=True)
            (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if "error" not in r and "skip" not in r]

    md = [f"# exp27 — Table 1 rerun (model={args.model})\n",
          f"Games: {games}",
          f"Modes: {modes}\n",
          "## Aggregate by mode\n",
          "| Mode | Avg Loops | Repair Rate (%) | Acc.Δ (correct-direction edges gained) |",
          "|------|----------:|----------------:|---------------------------------------:|"]
    for mode in modes:
        rs = [r for r in ok if r["mode"] == mode]
        if not rs: continue
        avg_loops = statistics.mean(r["iterations"] for r in rs)
        sum_cb = sum(r["n_conflicts_before"] for r in rs)
        sum_repaired = sum(r["repaired"] for r in rs)
        repair_rate = 100 * sum_repaired / max(1, sum_cb)
        sum_correct_delta = sum(r["correct_dir_edges_delta"] for r in rs)
        acc_delta = sum_correct_delta
        md.append(f"| {mode} | {avg_loops:.2f} | {repair_rate:.2f} | {acc_delta:+d} |")

    md.append("\n## Per-game per-mode\n")
    md.append("| game | mode | conf_before | conf_after | repaired | actions | iters | dir_delta | elapsed |")
    md.append("|------|------|-----------:|----------:|---------:|--------:|------:|----------:|--------:|")
    for r in sorted(ok, key=lambda x: (x["game"], x["mode"])):
        md.append(f"| {r['game']} | {r['mode']} | {r['n_conflicts_before']} | "
                  f"{r['n_conflicts_after']} | {r['repaired']} | {r['n_actions']} | "
                  f"{r['iterations']} | {r['correct_dir_edges_delta']:+d} | "
                  f"{r['elapsed_sec']:.1f}s |")
    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out_root}/summary.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
