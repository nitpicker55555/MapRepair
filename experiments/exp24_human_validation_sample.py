"""Experiment 24: build a human-validation sample for the F1-F9
failure-mode taxonomy.

The audit identified the C6 claim ("MANGO failures are F3/F4/F5
dominated") as weak because it rests entirely on a subagent's LLM
classification with no inter-rater agreement.

This script:
  1. Loads the 53-game per-edge classification from exp13.
  2. Selects a stratified sample of 100 conflicts/edges covering
     each predicted bucket (CORRECT, WRONG_DIRECTION, WRONG_DST,
     SPURIOUS_PAIR, HALLUCINATED_DST, HALLUCINATED_SRC, SELF_LOOP,
     WRONG_SRC_UNKNOWN).
  3. Renders each sample in a markdown form with:
       - the predicted bucket (subagent label)
       - the actual edge tuple
       - the observation that produced the edge
       - the GT context (room list, what GT says at this src/action)
       - a blank line for the human label
  4. Saves to results/exp24/human_validation_sheet.md for annotation.

After human annotation, run `python compute_kappa.py` to compute
Cohen's kappa between subagent and human labels.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


BUCKETS_OF_INTEREST = (
    "CORRECT", "WRONG_DIRECTION", "WRONG_DST", "SPURIOUS_PAIR",
    "HALLUCINATED_DST", "HALLUCINATED_SRC",
)


def select_stratified(per_game: List[Dict], target_per_bucket: int = 12,
                       seed: int = 42) -> List[Dict]:
    """Pull up to target_per_bucket examples from each bucket."""
    rng = random.Random(seed)
    pool: Dict[str, List[Dict]] = defaultdict(list)
    for game_data in per_game:
        gname = game_data["game"]
        samples = game_data.get("pred_samples", {})
        for bucket, items in samples.items():
            for item in items:
                pool[bucket].append({**item, "_game": gname,
                                       "_predicted_bucket": bucket})
    out = []
    for bucket in BUCKETS_OF_INTEREST:
        items = pool.get(bucket, [])
        rng.shuffle(items)
        out.extend(items[:target_per_bucket])
    return out


def render_sample(idx: int, item: Dict) -> str:
    """Format one sample for human annotation."""
    src = item.get("src", "<?>")
    dst = item.get("dst", "<?>")
    action = item.get("action", "<?>")
    obs = (item.get("obs_snip") or "").strip()
    walk_act = item.get("walk_action", "<?>")
    gt_hint = item.get("gt_dst_for_src_dir", None)

    lines = [
        f"## Sample {idx:03d} — game = {item['_game']}\n",
        f"**Predicted bucket (subagent):** `{item['_predicted_bucket']}`\n",
        f"- step_num: {item.get('step_num', '?')}",
        f"- walkthrough action: `{walk_act}`",
        f"- predicted edge: `{src!r} --[{action}]--> {dst!r}`",
    ]
    if gt_hint:
        lines.append(f"- GT destination from same (src, dir): `{gt_hint!r}`")
    if obs:
        obs_display = obs.replace("\n", " ").strip()[:400]
        lines.append(f"- observation: `{obs_display!r}`")
    lines.append("")
    lines.append("**Human label:** _____  (choose one):")
    lines.append("- `CORRECT` — predicted edge exactly matches GT")
    lines.append("- `WRONG_DIRECTION` — (src, dst) right, direction wrong")
    lines.append("- `WRONG_DST` — (src, dir) right but dst is a sibling room (F3)")
    lines.append("- `SPURIOUS_PAIR` — both nodes real but no GT edge exists")
    lines.append("- `HALLUCINATED_DST` — dst is not a GT room (F4 — object as room)")
    lines.append("- `HALLUCINATED_SRC` — src is not a GT room")
    lines.append("- `UNCERTAIN` — observation too ambiguous to decide")
    lines.append("")
    lines.append("**Notes:** ")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("results/exp13/error_analysis.json"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("results/exp24"))
    ap.add_argument("--per-bucket", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    per_game = json.loads(args.input.read_text())
    samples = select_stratified(per_game, target_per_bucket=args.per_bucket,
                                  seed=args.seed)
    # also drop the raw indices for later kappa computation
    (args.out_root / "sample_index.json").write_text(json.dumps(
        [{"idx": i, "predicted_bucket": s["_predicted_bucket"], "game": s["_game"],
          "src": s.get("src"), "dst": s.get("dst"), "action": s.get("action")}
         for i, s in enumerate(samples)], indent=2))

    md_lines = [
        "# Human validation sheet — F1-F9 taxonomy verification\n",
        f"Total samples: **{len(samples)}** "
        f"(stratified: up to {args.per_bucket} per bucket from {len(BUCKETS_OF_INTEREST)} buckets).\n",
        "## Instructions",
        "",
        "For each sample below, read the observation + predicted edge + GT hint, "
        "and assign a label from the same vocabulary as the subagent. Then save "
        "the file and run `experiments/exp24_compute_kappa.py` to compute "
        "Cohen's kappa between subagent and human labels.",
        "",
        "Annotate by filling in the `Human label: _____` line. You can use any "
        "of the six bucket labels listed below each sample, or `UNCERTAIN`.",
        "",
        "---",
        "",
    ]
    for i, item in enumerate(samples):
        md_lines.append(render_sample(i, item))
        md_lines.append("\n---\n")

    (args.out_root / "human_validation_sheet.md").write_text(
        "\n".join(md_lines) + "\n"
    )
    print(f"Wrote {args.out_root}/human_validation_sheet.md")
    print(f"  {len(samples)} samples, sample_index.json saved alongside")
    print(f"\nNext: open the .md, label each sample, then run "
          f"experiments/exp24_compute_kappa.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
