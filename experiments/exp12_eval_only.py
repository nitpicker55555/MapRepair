"""Run only the quality-evaluation pass of exp12 (mapping is already done).

Reads the per-game `<game>_edges.json` files produced by exp12 and
computes node/edge recall, direction accuracy, and conflict counts
against the refined GT. Writes `quality.json` and `summary.md` next to
the edges.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiments.exp12_remap_on_v3_clean import evaluate_quality


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path,
                    default=Path("results/exp12/gpt-4.1"))
    ap.add_argument("--gt-root", type=Path,
                    default=Path("/Users/puzhen/Downloads/maprepair/data_fixed"))
    args = ap.parse_args()

    games = sorted(p.stem.replace("_edges", "")
                   for p in args.model_dir.glob("*_edges.json"))
    print(f"Evaluating {len(games)} games...")

    rows = []
    for g in games:
        q = evaluate_quality(g, args.model_dir / f"{g}_edges.json", args.gt_root)
        rows.append(q)
        if "error" in q:
            print(f"  {g}: ERROR {q['error']}")
        else:
            print(f"  {g}: ER={q['edge_recall']*100:.1f}% "
                  f"NR={q['node_recall']*100:.1f}% "
                  f"DA={q['direction_accuracy']*100:.1f}% "
                  f"conf={q['conflicts_total']}")

    (args.model_dir / "quality.json").write_text(json.dumps(rows, indent=2))

    ok = [q for q in rows if "error" not in q]
    if not ok:
        print("No quality rows.")
        return 1
    macro = {
        "node_recall": statistics.mean(q["node_recall"] for q in ok),
        "edge_recall": statistics.mean(q["edge_recall"] for q in ok),
        "direction_accuracy": statistics.mean(q["direction_accuracy"] for q in ok),
        "edge_with_dir_recall": statistics.mean(q["edge_with_dir_recall"] for q in ok),
        "conflicts_per_game": statistics.mean(q["conflicts_total"] for q in ok),
    }

    old_path = Path("/Users/puzhen/Downloads/maprepair/llm_experiments/results/"
                    "exp1_mapping_quality/gpt-4.1.json")
    old_macro = {}
    if old_path.exists():
        od = json.loads(old_path.read_text())
        old_macro = {
            "node_recall": od.get("macro_node_recall"),
            "edge_recall": od.get("macro_edge_recall"),
            "direction_accuracy": od.get("macro_dir_accuracy"),
        }

    md = [
        "# Experiment 12 — LLM mapping on V3 clean walkthroughs (model=gpt-4.1)\n",
        f"Games: {len(ok)}",
        f"Total edges produced: {sum(q['pred_pairs'] for q in ok)}",
        f"Total GT edges: {sum(q['gt_pairs'] for q in ok)}",
        f"Total conflicts in resulting graphs: {sum(q['conflicts_total'] for q in ok)}",
        "\n## Macro metrics vs the old repaired_walkthroughs run\n",
        "| Metric | V3 clean (this run) | Old run | Δ |",
        "|--------|--------------------:|--------:|---:|",
    ]
    for k in ("node_recall", "edge_recall", "direction_accuracy"):
        new_v = macro[k] * 100
        old_v = (old_macro.get(k) or 0) * 100
        delta = new_v - old_v
        md.append(f"| {k} | {new_v:.2f}% | {old_v:.2f}% | {delta:+.2f}pp |")
    md.append(f"| conflicts / game | {macro['conflicts_per_game']:.2f} | (n/a) | — |")

    md.append("\n## Per-game (sorted by conflict count desc)\n")
    md.append("| game | gt_edges | pred_edges | edge_recall % | dir_acc % | conflicts |")
    md.append("|------|---------:|-----------:|--------------:|----------:|----------:|")
    for q in sorted(ok, key=lambda x: -x["conflicts_total"]):
        md.append(f"| {q['game']} | {q['gt_pairs']} | {q['pred_pairs']} | "
                  f"{q['edge_recall']*100:.1f} | {q['direction_accuracy']*100:.1f} | "
                  f"{q['conflicts_total']} |")
    (args.model_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.model_dir}/summary.md")
    for k, v in macro.items():
        if k == "conflicts_per_game":
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
