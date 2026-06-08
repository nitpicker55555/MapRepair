"""Experiment 30c: extend exp30 + exp30b from 10 hand-picked games to the
full set of cleaned-MANGO games with non-zero residual conflicts (30 games).

This addresses two reviewer concerns:
  (i)  cherry-picked game selection -- by covering ALL conflicting games
       on the cleaned MANGO benchmark, the result becomes a population
       measurement rather than a curated subset;
  (ii) graph-repair baseline -- we add the model-independent
       heuristic_modify reference (relabel direction rather than delete)
       alongside heuristic_remove, so the LLM modes are bracketed by
       both an aggressive-deletion and a structure-preserving non-LLM
       baseline.

The 10 games already covered by exp30/exp30b are skipped for repair
modes already computed; this script then aggregates everything.
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
from maprepair.mango import ground_truth_graph, list_games
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
LLM_MODES = ("baseline", "edge_impact", "vc_ei")
HEURISTIC_MODES = ("heuristic_remove", "heuristic_modify")

# Games already covered by exp30 + exp30b for LLM modes
ALREADY_COVERED_LLM = {"cutthroat", "detective", "inhumane", "zork1", "zork2",
                        "murdac", "advent", "sherlock", "wishbringer", "deephome"}


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
    try:
        gt = ground_truth_graph(game)
    except Exception:
        gt = NavGraph()
    conflicts_before = _filtered_detect_all(work)
    n_cb = len(conflicts_before)
    if n_cb == 0:
        return {"game": game, "mode": mode, "model": model,
                "n_conflicts_before": 0, "skip": True}
    gt_dir_pre = gt_dir_correct_count(work, gt)

    t0 = time.time()
    if mode == "heuristic_remove":
        r = HeuristicRepairAgent(prefer_remove=True).repair(work.copy(), max_iterations=max_iter)
    elif mode == "heuristic_modify":
        r = HeuristicRepairAgent(prefer_remove=False).repair(work.copy(), max_iterations=max_iter)
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
    ap.add_argument("--out-root", type=Path, default=Path("results/exp30c"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-iterations", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    # Discover all games with non-zero residual conflicts on cleaned MANGO
    all_games = list_games()
    candidate_games = []
    for g in all_games:
        ep = args.exp14_dir / f"{g}_edges.json"
        if not ep.exists():
            continue
        gr = load_llm_map(ep)
        cf = _filtered_detect_all(gr)
        if len(cf) > 0:
            candidate_games.append(g)
    print(f"Cleaned MANGO games with non-zero residual conflicts: {len(candidate_games)}", flush=True)
    print(f"Already covered (LLM modes): {sorted(ALREADY_COVERED_LLM)}", flush=True)
    new_games = [g for g in candidate_games if g not in ALREADY_COVERED_LLM]
    print(f"New games this run: {sorted(new_games)}", flush=True)

    jobs = []
    # All candidate games: heuristic_modify
    for g in candidate_games:
        jobs.append((g, "heuristic_modify", "n/a"))
    # New games for heuristic_remove (already have 10)
    for g in new_games:
        jobs.append((g, "heuristic_remove", "n/a"))
    # New games for LLM modes (3 models x 3 modes)
    for g in new_games:
        for mdl in MODELS:
            for m in LLM_MODES:
                jobs.append((g, m, mdl))
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

    # Merge with exp30 and exp30b raws
    merged = list(rows)
    for prior_path in [Path("results/exp30_mango_frontier/raw.json"),
                        Path("results/exp30b/raw.json")]:
        if prior_path.exists():
            merged.extend(json.loads(prior_path.read_text()))
    ok = [r for r in merged if "error" not in r and "skip" not in r]

    by = defaultdict(list)
    for r in ok:
        by[(r["model"], r["mode"])].append(r)

    md = [f"# exp30c - Full-coverage MANGO IF maps (30 games with non-zero residual conflicts)\n",
          f"Candidate games: {candidate_games}\n",
          "## Aggregate by (model, mode), merged with exp30/exp30b\n",
          "| Model | Mode | n | cf_before | cf_after | repaired | repair % | dir_delta |",
          "|-------|------|--:|---------:|---------:|---------:|--------:|----------:|"]
    for (mdl, m), rs in sorted(by.items()):
        if not rs: continue
        sum_cb = sum(r["n_conflicts_before"] for r in rs)
        sum_repaired = sum(r["repaired"] for r in rs)
        sum_delta = sum(r["correct_dir_edges_delta"] for r in rs)
        sum_ca = sum(r["n_conflicts_after"] for r in rs)
        rate = 100 * sum_repaired / max(1, sum_cb)
        md.append(f"| {mdl} | {m} | {len(rs)} | {sum_cb} | {sum_ca} | {sum_repaired} | {rate:.1f} | {sum_delta:+d} |")

    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out_root}/summary.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
