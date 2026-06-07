"""Experiment 9: head-to-head heuristic vs LLM repair on the SAME synthetic
scenarios.

For each (family, err_type, num_err, seed) we generate one broken graph
and feed it to four agents in turn:
  1. HeuristicRepairAgent(prefer_remove=True)   -- no LLM
  2. LLMRepairAgent(mode='baseline')             -- LLM only
  3. LLMRepairAgent(mode='edge_impact')          -- LLM + impact scoring
  4. LLMRepairAgent(mode='vc_ei')                -- LLM + impact + version
                                                    history (paper's full
                                                    method)

We report the same metrics for all four agents so the comparison is paired.

The hypothesis: the heuristic agent (zero LLM cost, deterministic) achieves
results at least as good as the LLM agents on this task. If true, this is
the strongest possible argument that the algorithmic primitives carry the
framework and the LLM is optional.
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from maprepair.conflict import detect_all
from maprepair.synth import make_synthetic
from maprepair.agents.heuristic import HeuristicRepairAgent
from maprepair.agents.llm_agent import LLMRepairAgent

from experiments.exp03_heuristic_repair import evaluate as eval_synth


AGENTS = [
    ("heuristic_remove", lambda: HeuristicRepairAgent(prefer_remove=True)),
    ("llm_baseline",     lambda model: LLMRepairAgent(model=model, mode="baseline",
                                                        max_attempts_per_conflict=5)),
    ("llm_edge_impact",  lambda model: LLMRepairAgent(model=model, mode="edge_impact",
                                                        max_attempts_per_conflict=5)),
    ("llm_vc_ei",        lambda model: LLMRepairAgent(model=model, mode="vc_ei",
                                                        max_attempts_per_conflict=5)),
]


def _make_spec(family: str, size: int, err_type: str, num_err: int, seed: int):
    if family == "grid":
        return make_synthetic("grid", rows=max(size, 3), cols=max(size, 3),
                              error_mix={err_type: num_err}, seed=seed)
    return make_synthetic(family, size=size,
                          error_mix={err_type: num_err}, seed=seed)


def _run_pair(family: str, size: int, err_type: str, num_err: int,
               seed: int, model: str) -> Dict:
    spec = _make_spec(family, size, err_type, num_err, seed)
    if not detect_all(spec.graph):
        return {}
    out: Dict = {"family": family, "size": size, "err_type": err_type,
                 "num_err": num_err, "seed": seed,
                 "num_conflicts": len(detect_all(spec.graph))}
    for tag, builder in AGENTS:
        if tag.startswith("llm_"):
            agent = builder(model)
        else:
            agent = builder()
        result = agent.repair(spec.graph.copy(), max_iterations=15)
        ev = eval_synth(spec, result.graph_after)
        out[f"{tag}_conflict_free"] = result.success
        out[f"{tag}_iters"] = result.iterations
        out[f"{tag}_dir_acc"] = ev["gt_direction_accuracy"]
        out[f"{tag}_recovered"] = ev["recovered_errors"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("results/exp09"))
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cases: List[Tuple[str, int, str, int]] = [
        ("tree",   4, "direction", 1),
        ("tree",   4, "topology",  1),
        ("random", 30, "direction", 1),
        ("random", 30, "topology",  1),
        ("grid",   4, "direction", 1),
        ("grid",   4, "topology",  1),
    ]
    work = [(f, s, e, n, sd, args.model) for (f, s, e, n) in cases for sd in range(args.seeds)]
    print(f"Total scenarios: {len(work)} (each runs 4 agents) with workers={args.workers}")

    rows: List[Dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_pair, *args_t): args_t for args_t in work}
        completed = 0
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"error": str(e), "args": futs[fut]}
            rows.append(r)
            completed += 1
            if completed % 5 == 0:
                print(f"  {completed}/{len(work)}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "raw.json").write_text(json.dumps(rows, indent=2))

    # ---- summary
    agents = [a[0] for a in AGENTS]
    md = [
        f"# Experiment 9 - Heuristic vs LLM head-to-head (model={args.model})",
        "",
        "| family | err | n | "
        + " | ".join([f"{a} cf | {a} dir" for a in agents])
        + " |",
        "|--------|-----|--:|"
        + "|".join([":---:|:---:" for _ in agents])
        + "|",
    ]
    cohort: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        if "family" not in r: continue
        cohort.setdefault((r["family"], r["err_type"]), []).append(r)
    for (family, err), rs in sorted(cohort.items()):
        if not rs: continue
        def pct(field): return 100 * sum(1 for r in rs if r.get(field)) / len(rs)
        def avg(field): return statistics.mean(r.get(field, 0) or 0 for r in rs)
        cells = []
        for a in agents:
            cells.append(f"{pct(a+'_conflict_free'):.1f}%")
            cells.append(f"{100*avg(a+'_dir_acc'):.1f}%")
        md.append(f"| {family} | {err} | {len(rs)} | " + " | ".join(cells) + " |")
    (args.out_root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
