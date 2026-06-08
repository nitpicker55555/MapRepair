"""Experiment 30b: complete the per-mode matrix for the real IF-map setting.

exp30 reported baseline vs Edge-Impact for three June-2026 frontier models.
exp30b adds the two missing rows that reviewers expect: (i) the full
VC+EI configuration on the same three models, so the matrix matches
Table 2; (ii) the non-LLM heuristic_remove baseline, which establishes
that an LLM is necessary in the first place.
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

from maprepair.agents.heuristic import HeuristicRepairAgent
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


MODELS = ("gpt-5.5", "gemini-3.5-flash", "claude-haiku-4-5-20251001")
DEFAULT_GAMES = ("cutthroat", "detective", "inhumane", "zork1", "zork2",
                 "murdac", "advent", "sherlock", "wishbringer", "deephome")


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


def run_one(game: str, mode: str, model: str, exp14_dir: Path,
            max_iter: int, max_att: int) -> Dict:
    edges_path = exp14_dir / f"{game}_edges.json"
    if not edges_path.exists():
        return {"game": game, "mode": mode, "model": model, "error": "no edges"}
    work = load_llm_map(edges_path)
    gt = ground_truth_graph(game)
    conflicts_before = _filtered_detect_all(work)
    n_cb = len(conflicts_before)
    if n_cb == 0:
        return {"game": game, "mode": mode, "model": model,
                "n_conflicts_before": 0, "skip": True}
    gt_dir_pre = gt_dir_correct_count(work, gt)

    t0 = time.time()
    if mode == "heuristic_remove":
        try:
            r = HeuristicRepairAgent(prefer_remove=True).repair(
                work.copy(), max_iterations=max_iter)
        except Exception as e:
            return {"game": game, "mode": mode, "model": model, "error": str(e)[:200]}
    else:
        agent = LLMRepairAgent(model=model, mode=mode,
                                max_attempts_per_conflict=max_att,
                                lookahead=False,
                                chat_json_fn=proxy_chat_json)
        try:
            r = agent.repair(work.copy(), max_iterations=max_iter)
        except Exception as e:
            return {"game": game, "mode": mode, "model": model, "error": str(e)[:200]}
    elapsed = time.time() - t0
    n_ca = len(r.conflicts_after)
    repaired = n_cb - n_ca
    gt_dir_post = gt_dir_correct_count(r.graph_after, gt)
    return {
        "game": game, "mode": mode, "model": model,
        "n_conflicts_before": n_cb, "n_conflicts_after": n_ca,
        "repaired": repaired,
        "correct_dir_edges_delta": gt_dir_post - gt_dir_pre,
        "n_actions": len(r.actions),
        "iterations": r.iterations,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp14-dir", type=Path, default=Path("results/exp14/gpt-4.1"))
    ap.add_argument("--out-root", type=Path, default=Path("results/exp30b"))
    ap.add_argument("--games", default=",".join(DEFAULT_GAMES))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-iterations", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    games = [g.strip() for g in args.games.split(",") if g.strip()]

    # Build job list: VC+EI for each model + heuristic_remove (once)
    jobs = []
    for g in games:
        for mdl in MODELS:
            jobs.append((g, "vc_ei", mdl))
        jobs.append((g, "heuristic_remove", "n/a"))
    print(f"Total runs: {len(jobs)}", flush=True)

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, g, m, mdl, args.exp14_dir,
                          args.max_iterations, args.max_attempts): (g, m, mdl)
                 for g, m, mdl in jobs}
        done = 0
        for fut in as_completed(futs):
            g, m, mdl = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"game": g, "mode": m, "model": mdl, "error": str(e)[:200]}
            rows.append(r)
            done += 1
            print(f"  [{done}/{len(jobs)}] {mdl}/{g}/{m}: cf "
                  f"{r.get('n_conflicts_before','?')}->{r.get('n_conflicts_after','?')} "
                  f"acts={r.get('n_actions','?')} iters={r.get('iterations','?')} "
                  f"t={time.time()-t0:.0f}s", flush=True)
            (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if "error" not in r and "skip" not in r]
    by = defaultdict(list)
    for r in ok:
        by[(r["model"], r["mode"])].append(r)

    md = [f"# exp30b - VC+EI and heuristic baseline on MANGO IF maps\n",
          f"Games: {games}\n",
          "## Aggregate by (model, mode)\n",
          "| Model | Mode | n | cf_before | cf_after | repaired | repair % | dir_delta |",
          "|-------|------|--:|---------:|---------:|---------:|--------:|----------:|"]
    for (mdl, m), rs in sorted(by.items()):
        sum_cb = sum(r["n_conflicts_before"] for r in rs)
        sum_repaired = sum(r["repaired"] for r in rs)
        sum_delta = sum(r["correct_dir_edges_delta"] for r in rs)
        sum_ca = sum(r["n_conflicts_after"] for r in rs)
        rate = 100 * sum_repaired / max(1, sum_cb)
        md.append(f"| {mdl} | {m} | {len(rs)} | {sum_cb} | {sum_ca} | {sum_repaired} | {rate:.1f} | {sum_delta:+d} |")

    md.append("\n## Per-game per-(model, mode)\n")
    md.append("| game | model | mode | cb | ca | repaired | actions | iters | dir_delta |")
    md.append("|------|-------|------|---:|---:|--------:|--------:|------:|----------:|")
    for r in sorted(ok, key=lambda x: (x["game"], x["model"], x["mode"])):
        md.append(f"| {r['game']} | {r['model']} | {r['mode']} | "
                  f"{r['n_conflicts_before']} | {r['n_conflicts_after']} | "
                  f"{r['repaired']} | {r['n_actions']} | {r['iterations']} | "
                  f"{r['correct_dir_edges_delta']:+d} |")
    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out_root}/summary.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
