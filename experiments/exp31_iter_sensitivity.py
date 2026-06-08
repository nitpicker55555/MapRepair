"""Experiment 31: max-iteration sensitivity analysis.

Addresses a reviewer concern that the choice of max_iterations=20 and
max_attempts_per_conflict=3 is unjustified.  We sweep max_iterations
across {5, 10, 20, 40} on a 5-game subset with gpt-5.5 + Edge-Impact
(the strongest LLM cell in Table 3) to characterise how repair quality
scales with the iteration budget.

Total runs: 5 games x 4 iter values x 1 mode x 1 model = 20.
Wall-clock estimate: ~10-15 minutes at workers=6.
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Dict

from maprepair.agents.llm_agent import LLMRepairAgent
from maprepair.conflict import detect_direction_conflicts, detect_topology_conflicts
from maprepair.graph import NavGraph
from maprepair.llm_client_proxy import chat_json as proxy_chat_json
from maprepair.mango import ground_truth_graph
import maprepair.conflict as _conflict_mod


def _filtered_detect_all(graph: NavGraph):
    seen = set(); out = []
    for c in (*detect_direction_conflicts(graph), *detect_topology_conflicts(graph)):
        cid = c.conflict_id()
        if cid in seen: continue
        seen.add(cid); out.append(c)
    return out


_conflict_mod.detect_all = _filtered_detect_all


GAMES = ("cutthroat", "zork1", "murdac", "wishbringer", "adventureland")
ITER_VALUES = (5, 10, 20, 40)
MODEL = "gpt-5.5"
MODE = "edge_impact"


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
    gt_dir = {(e.source, e.target): e.direction for e in gt.primary_edges()}
    cnt = 0
    for e in work.primary_edges():
        if (e.source, e.target) in gt_dir and gt_dir[(e.source, e.target)] == e.direction:
            cnt += 1
    return cnt


def run_one(game: str, max_iter: int, exp14_dir: Path, max_att: int) -> Dict:
    edges_path = exp14_dir / f"{game}_edges.json"
    if not edges_path.exists():
        return {"game": game, "max_iter": max_iter, "error": "no edges"}
    work = load_llm_map(edges_path)
    gt = ground_truth_graph(game)
    conflicts_before = _filtered_detect_all(work)
    n_cb = len(conflicts_before)
    if n_cb == 0:
        return {"game": game, "max_iter": max_iter,
                "n_conflicts_before": 0, "skip": True}
    gt_dir_pre = gt_dir_correct_count(work, gt)
    t0 = time.time()
    agent = LLMRepairAgent(model=MODEL, mode=MODE,
                            max_attempts_per_conflict=max_att,
                            lookahead=False,
                            chat_json_fn=proxy_chat_json)
    try:
        r = agent.repair(work.copy(), max_iterations=max_iter)
    except Exception as e:
        return {"game": game, "max_iter": max_iter, "error": str(e)[:200]}
    elapsed = time.time() - t0
    n_ca = len(r.conflicts_after)
    return {
        "game": game, "max_iter": max_iter,
        "n_conflicts_before": n_cb, "n_conflicts_after": n_ca,
        "repaired": n_cb - n_ca,
        "correct_dir_edges_delta": gt_dir_correct_count(r.graph_after, gt) - gt_dir_pre,
        "n_actions": len(r.actions),
        "iterations": r.iterations,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp14-dir", type=Path, default=Path("results/exp14/gpt-4.1"))
    ap.add_argument("--out-root", type=Path, default=Path("results/exp31_iter_sensitivity"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    jobs = [(g, it) for g in GAMES for it in ITER_VALUES]
    print(f"Total runs: {len(jobs)} ({len(GAMES)} games x {len(ITER_VALUES)} iter values)", flush=True)

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, g, it, args.exp14_dir, args.max_attempts): (g, it)
                 for g, it in jobs}
        done = 0
        for fut in as_completed(futs):
            g, it = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"game": g, "max_iter": it, "error": str(e)[:200]}
            rows.append(r)
            done += 1
            print(f"  [{done}/{len(jobs)}] {g}/iter={it}: cf "
                  f"{r.get('n_conflicts_before','?')}->{r.get('n_conflicts_after','?')} "
                  f"acts={r.get('n_actions','?')} iters={r.get('iterations','?')} "
                  f"t={time.time()-t0:.0f}s", flush=True)
            (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if "error" not in r and "skip" not in r]
    by_iter = defaultdict(list)
    for r in ok:
        by_iter[r["max_iter"]].append(r)

    md = [f"# exp31 - max_iterations sensitivity\n",
          f"Model: {MODEL}  Mode: {MODE}  Games: {list(GAMES)}\n",
          "## Aggregate by max_iter\n",
          "| max_iter | n | cf_before | cf_after | repaired | repair % | avg iters used | avg elapsed |",
          "|---------:|--:|---------:|---------:|---------:|--------:|---------------:|------------:|"]
    for it in sorted(by_iter.keys()):
        rs = by_iter[it]
        sum_cb = sum(r["n_conflicts_before"] for r in rs)
        sum_ca = sum(r["n_conflicts_after"] for r in rs)
        sum_rep = sum(r["repaired"] for r in rs)
        avg_iters = statistics.mean(r["iterations"] for r in rs)
        avg_elapsed = statistics.mean(r["elapsed_sec"] for r in rs)
        rate = 100 * sum_rep / max(1, sum_cb)
        md.append(f"| {it} | {len(rs)} | {sum_cb} | {sum_ca} | {sum_rep} | {rate:.1f} | {avg_iters:.1f} | {avg_elapsed:.1f}s |")

    md.append("\n## Per-game per-max_iter\n")
    md.append("| game | max_iter | cb | ca | repaired | iters used | elapsed |")
    md.append("|------|---------:|---:|---:|--------:|----------:|--------:|")
    for r in sorted(ok, key=lambda x: (x["game"], x["max_iter"])):
        md.append(f"| {r['game']} | {r['max_iter']} | {r['n_conflicts_before']} | "
                  f"{r['n_conflicts_after']} | {r['repaired']} | "
                  f"{r['iterations']} | {r['elapsed_sec']:.1f}s |")

    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out_root}/summary.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
