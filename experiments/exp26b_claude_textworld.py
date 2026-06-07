"""Experiment 26b: targeted re-run of claude-sonnet-4-6 + gemini-2.5-flash
on TextWorld mango_like.

The original exp26 hit a proxy-credit pre-charge limit on the 3 expensive
models (claude / gemini / o4-mini) with max_tokens=1500. Reducing the
budget to max_tokens=600 brings the pre-charge under the threshold and
lets those models run.

We deliberately skip o4-mini (reasoning models still need ~4000 tokens
for hidden reasoning, which is well over the pre-charge limit).

Setup:
  - 10 TextWorld games (same as exp23/exp26)
  - 1 noise regime: mango_like
  - 3 seeds per cell
  - 2 LLM modes: baseline + edge_impact
  - 2 models: claude-sonnet-4-6, gemini-2.5-flash
  - max_tokens=600 to keep pre-charge under proxy limit
  - Total: 10 × 1 × 2 × 2 × 3 = 120 LLM runs, ~$3-5

Output: results/exp26b/raw.json + summary.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from maprepair.agents.llm_agent import LLMRepairAgent
from maprepair.conflict import detect_all
from maprepair.graph import NavGraph
from maprepair.llm_client_proxy import chat_json as proxy_chat_json, chat
from maprepair.llm_client_proxy import message as proxy_message

from experiments.exp23_textworld_noise import gt_to_navgraph
from experiments.exp16_noise import apply_regime, regime_by_name
from experiments.exp26_frontier_textworld import gt_recall_precision, gt_dir_acc


REGIME_NAME = "mango_like"
DEFAULT_MODELS = ("claude-sonnet-4-6", "gemini-2.5-flash")
LLM_MODES = ("baseline", "edge_impact")


def proxy_chat_json_small(messages, **kw):
    # Cap max_tokens at 600 so the proxy's pre-charge estimate stays
    # under the per-call budget threshold for opus/sonnet/gemini.
    kw["max_tokens"] = min(kw.get("max_tokens", 600), 600)
    return proxy_chat_json(messages, **kw)


def run_one(game_id: str, gt: NavGraph, mode: str, model: str,
             seed: int, max_iter: int, max_att: int) -> Dict:
    work = gt.copy()
    recs = apply_regime(work, regime_by_name(REGIME_NAME), seed=seed)
    n_cb = len(detect_all(work))
    er_b, _ = gt_recall_precision(work, gt)
    t0 = time.time()
    agent = LLMRepairAgent(model=model, mode=mode,
                             max_attempts_per_conflict=max_att,
                             lookahead=False,
                             chat_json_fn=proxy_chat_json_small)
    r = agent.repair(work.copy(), max_iterations=max_iter)
    elapsed = time.time() - t0
    er_a, ep_a = gt_recall_precision(r.graph_after, gt)
    dir_acc = gt_dir_acc(r.graph_after, gt)
    action_kinds = [a.kind for a in r.actions]
    return {
        "game": game_id, "regime": REGIME_NAME, "mode": mode,
        "model": model, "seed": seed,
        "n_conflicts_before": n_cb,
        "n_conflicts_after": len(r.conflicts_after),
        "conflict_free": r.success,
        "conflict_reduction_pct": 100 * (n_cb - len(r.conflicts_after)) / max(1, n_cb),
        "edge_recall_before_repair": er_b,
        "edge_recall_after": er_a,
        "edge_recall_delta": er_a - er_b,
        "edge_precision_after": ep_a,
        "direction_accuracy_after": dir_acc,
        "n_actions": len(r.actions),
        "n_modify": action_kinds.count("modify_edge"),
        "n_remove": action_kinds.count("remove_edge"),
        "iterations": r.iterations,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-iterations", type=int, default=15)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--out-root", type=Path, default=Path("results/exp26b"))
    ap.add_argument("--games-root", type=Path,
                    default=Path("results/exp23/games"))
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    models = args.models.split(",")

    # Pre-load TextWorld GTs sequentially
    game_dirs = sorted(args.games_root.iterdir())
    game_gts: Dict[str, NavGraph] = {}
    print(f"Pre-loading TextWorld GTs ...", flush=True)
    for gdir in game_dirs:
        gt_path = gdir / "game.json"
        if not gt_path.exists(): continue
        try:
            game_gts[gdir.name] = gt_to_navgraph(gt_path)
        except Exception as e:
            print(f"  failed {gdir.name}: {e}", flush=True)
    print(f"  loaded {len(game_gts)}", flush=True)

    jobs: List[Tuple] = []
    for model in models:
        for gid in game_gts:
            for mode in LLM_MODES:
                for seed in range(args.seeds):
                    jobs.append((gid, game_gts[gid], mode, model, seed))

    print(f"Models: {models}", flush=True)
    print(f"Total runs: {len(jobs)}", flush=True)

    rows: List[Dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_one, *job,
                       max_iter=args.max_iterations,
                       max_att=args.max_attempts): job
            for job in jobs
        }
        done = 0
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"game": j[0], "mode": j[2], "model": j[3],
                        "seed": j[4], "regime": REGIME_NAME,
                        "error": f"{type(e).__name__}: {str(e)[:200]}"}
            rows.append(row)
            done += 1
            if done % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))
                print(f"  [{done}/{len(jobs)}] {el:.0f}s", flush=True)

    (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if "error" not in r]

    def stats(rs):
        if not rs: return {"n": 0}
        return {
            "n": len(rs),
            "cf": 100 * sum(1 for r in rs if r["conflict_free"]) / len(rs),
            "gtr_after": 100 * statistics.mean(r["edge_recall_after"] for r in rs),
            "gtr_delta": 100 * statistics.mean(r["edge_recall_delta"] for r in rs),
            "gtd_after": 100 * statistics.mean(r["direction_accuracy_after"] for r in rs),
            "cred": statistics.mean(r["conflict_reduction_pct"] for r in rs),
            "iters": statistics.mean(r["iterations"] for r in rs),
            "actions": statistics.mean(r["n_actions"] for r in rs),
            "remove": statistics.mean(r.get("n_remove", 0) for r in rs),
        }

    by_mm = defaultdict(list)
    for r in ok:
        by_mm[(r["mode"], r["model"])].append(r)

    md = [
        f"# Experiment 26b — claude + gemini on TextWorld mango_like (retry)\n",
        f"Total runs: {len(rows)} (valid {len(ok)}, errors {sum(1 for r in rows if 'error' in r)})\n",
        f"Models: {models}\n",
        f"max_tokens=600 (proxy pre-charge limit workaround)\n",
        "## Headline: CF % per model\n",
        "| Model | baseline | edge_impact | Δ | conf reduce (edge_imp) | edge recall after | dir acc |",
        "|-------|---------:|------------:|--:|----------------------:|-------------------:|--------:|",
    ]
    for model in models:
        sb = stats(by_mm[("baseline", model)])
        se = stats(by_mm[("edge_impact", model)])
        if sb["n"] == 0 or se["n"] == 0: continue
        delta = se["cf"] - sb["cf"]
        md.append(f"| {model} | {sb['cf']:.1f}% (n={sb['n']}) | "
                  f"**{se['cf']:.1f}%** (n={se['n']}) | "
                  f"**{delta:+.1f}pp** | {se['cred']:.1f}% | "
                  f"{se['gtr_after']:.1f}% | {se['gtd_after']:.1f}% |")

    md.append("\n## Detailed\n")
    md.append("| mode | model | n | CF % | conf reduce | recall after | recall Δ | dir acc | actions | iters |")
    md.append("|------|-------|--:|-----:|------------:|-------------:|---------:|--------:|--------:|------:|")
    for mode in LLM_MODES:
        for model in models:
            s = stats(by_mm[(mode, model)])
            if s["n"] == 0: continue
            md.append(f"| {mode} | {model} | {s['n']} | {s['cf']:.1f} | "
                      f"{s['cred']:.1f} | {s['gtr_after']:.1f} | "
                      f"{s['gtr_delta']:+.2f}pp | {s['gtd_after']:.1f} | "
                      f"{s['actions']:.1f} | {s['iters']:.1f} |")

    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out_root}/summary.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
