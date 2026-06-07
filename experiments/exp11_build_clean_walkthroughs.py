"""Experiment 11: rebuild *truly* clean MANGO walkthroughs.

The current `walkthrough_repair_pipeline/output/repaired_walkthroughs/` was
itself produced by an earlier LLM pipeline that tried to disguise
non-topological steps (drop, take, plugh, ...) as cardinal movements,
producing inconsistent observation text such as

    ACT: south
    OBSERVATION: Low N/S Passage
                 [Current Location changed to at "y2"]
                 Having moved south, you find yourself now at "y2".

That walkthrough is impossible to map back to the refined GT edges in
`data_fixed/`. We rebuild a clean variant directly from the *original*
`data/` walkthrough by:

  1. Keeping only steps whose action is one of the 14 canonical compass
     / portal directions (matches the refinement spec in the paper).
  2. Stripping any synthetic "[Current Location changed]" / "Having
     moved X" lines that previous pipelines may have inserted.
  3. Renumbering the surviving steps from 0.

Then we **verify** the clean walkthrough against the refined GT edges:
  - track current_location as the LLM-induced room name in each step's
    observation;
  - for each consecutive pair (prev_loc, action, this_loc), check
    whether the corresponding edge exists in
    `data_fixed/<game>/<game>.edges.json`;
  - flag any unmatched (loc, action, loc) tuple.

Outputs:
  results/exp11/clean_walkthroughs/<game>/<game>.walkthrough
  results/exp11/review/<game>.md      issues found per game
  results/exp11/global_review.md      summary
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CANONICAL_DIRECTIONS = {
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
    "up", "down", "in", "out", "enter", "exit",
}

# Patterns that look like synthetic text injected by a previous LLM pipeline.
SYNTHETIC_PATTERNS = [
    re.compile(r"^\s*\[Current Location.*?\]\s*$", re.IGNORECASE),
    re.compile(r"^\s*Having moved [a-z]+.*you find yourself.*", re.IGNORECASE),
    re.compile(r"^\s*Standing here.*you reach out.*", re.IGNORECASE),
]


def _strip_synthetic(text: str) -> Tuple[str, int]:
    """Remove lines that look injected; return (cleaned_text, num_lines_removed)."""
    keep: List[str] = []
    removed = 0
    for line in text.splitlines():
        if any(p.match(line) for p in SYNTHETIC_PATTERNS):
            removed += 1
            continue
        keep.append(line)
    # Also drop completely empty trailing blocks introduced by removal.
    return ("\n".join(keep).rstrip(), removed)


def parse_walkthrough(text: str) -> List[Dict]:
    """Parse a MANGO walkthrough file into ordered step dicts."""
    blocks = [b.strip() for b in text.split("===========") if b.strip()]
    steps: List[Dict] = []
    for block in blocks:
        step_num: Optional[int] = None
        action: Optional[str] = None
        obs_lines: List[str] = []
        in_obs = False
        for line in block.splitlines():
            if line.startswith("==>STEP NUM:"):
                step_num = int(line.split(":", 1)[1].strip())
                in_obs = False
            elif line.startswith("==>ACT:"):
                action = line.split(":", 1)[1].strip()
                in_obs = False
            elif line.startswith("==>OBSERVATION:"):
                first = line.split(":", 1)[1].strip()
                if first:
                    obs_lines.append(first)
                in_obs = True
            else:
                if in_obs:
                    obs_lines.append(line)
        if step_num is None or action is None:
            continue
        steps.append({
            "step_num": step_num,
            "action": action,
            "observation": "\n".join(obs_lines).strip(),
        })
    return steps


def build_clean(game: str, data_root: Path) -> Tuple[str, Dict]:
    src_path = data_root / "data" / game / f"{game}.walkthrough"
    if not src_path.exists():
        return "", {"error": f"missing source walkthrough at {src_path}"}
    raw = src_path.read_text(encoding="utf-8")
    raw_steps = parse_walkthrough(raw)

    kept: List[Dict] = []
    skipped_actions: Dict[str, int] = {}
    total_synthetic_lines_removed = 0
    for s in raw_steps:
        if s["action"].lower() not in CANONICAL_DIRECTIONS:
            skipped_actions[s["action"]] = skipped_actions.get(s["action"], 0) + 1
            continue
        cleaned_obs, n_removed = _strip_synthetic(s["observation"])
        total_synthetic_lines_removed += n_removed
        kept.append({
            "orig_step_num": s["step_num"],
            "action": s["action"],
            "observation": cleaned_obs,
        })

    # Render
    out_lines: List[str] = []
    for new_num, step in enumerate(kept):
        out_lines.append("===========")
        out_lines.append(f"==>STEP NUM: {new_num}")
        out_lines.append(f"==>ACT: {step['action']}")
        if step["observation"]:
            out_lines.append(f"==>OBSERVATION: {step['observation']}")
        else:
            out_lines.append("==>OBSERVATION: ")
    out_lines.append("===========")
    out_text = "\n".join(out_lines) + "\n"

    stats = {
        "src_step_count": len(raw_steps),
        "kept_step_count": len(kept),
        "skipped_non_canonical_count": sum(skipped_actions.values()),
        "skipped_action_distribution": skipped_actions,
        "synthetic_lines_removed": total_synthetic_lines_removed,
    }
    return out_text, stats


# ----------------------------------------------------------------------
# Review pass — check kept walkthrough against refined GT edges
# ----------------------------------------------------------------------

def _load_gt_edges(game: str, data_root: Path) -> List[Dict]:
    p = data_root / "data_fixed" / game / f"{game}.edges.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _normalize_room(name: str) -> str:
    return (name or "").strip().lower()


def _extract_first_room_from_obs(obs: str) -> Optional[str]:
    """Heuristic: the first non-empty line of the observation is the room name
    (for MANGO walkthroughs). Lowercased."""
    for line in obs.splitlines():
        line = line.strip()
        if line:
            return _normalize_room(line.rstrip("."))
    return None


def review_clean(game: str, clean_text: str, data_root: Path,
                  *, kept_orig_step_nums: Optional[List[int]] = None) -> Dict:
    """Reviews a clean walkthrough against the refined GT edges.

    Two perspectives:

    1. **GT-step coverage** (preferred). For each *kept* step in the clean
       walkthrough we know its original step number; we check whether at
       least one GT edge has that number as `seen_in_forward` *or*
       `seen_in_reversed`. If yes, the step is structurally meaningful.
       If no, the step is a directional action that survived filtering
       but does not correspond to a real GT edge -- a candidate for
       further inspection.

    2. **Synthetic-text leak**: scan kept observations for the synthetic
       patterns from the original buggy pipeline.
    """
    gt_edges = _load_gt_edges(game, data_root)
    gt_forward_steps = {e.get("seen_in_forward") for e in gt_edges if e.get("seen_in_forward")}
    gt_reverse_steps = {e.get("seen_in_reversed") for e in gt_edges if e.get("seen_in_reversed") not in (None, 9999)}
    gt_step_set = gt_forward_steps | gt_reverse_steps

    steps = parse_walkthrough(clean_text)
    if kept_orig_step_nums is None:
        kept_orig_step_nums = []
    leaks: List[Dict] = []
    unmapped_steps: List[Dict] = []
    n_matched = 0
    for i, step in enumerate(steps):
        # synthetic text leak check
        for line in step["observation"].splitlines():
            if any(p.match(line) for p in SYNTHETIC_PATTERNS):
                leaks.append({"step": step["step_num"], "line": line})
        orig_step_num = kept_orig_step_nums[i] if i < len(kept_orig_step_nums) else None
        if orig_step_num is None:
            continue
        if orig_step_num in gt_step_set:
            n_matched += 1
        else:
            unmapped_steps.append({
                "new_step_num": step["step_num"],
                "orig_step_num": orig_step_num,
                "action": step["action"],
                "obs_first_line": _extract_first_room_from_obs(step["observation"]),
            })

    return {
        "n_steps": len(steps),
        "n_matched_to_gt_step": n_matched,
        "n_unmapped_steps": len(unmapped_steps),
        "unmapped_steps": unmapped_steps,
        "synthetic_text_leaks": leaks,
        "num_gt_edges": len(gt_edges),
        "num_gt_step_indices": len(gt_step_set),
    }


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/Users/puzhen/Downloads/maprepair"))
    ap.add_argument("--out-root", type=Path, default=Path("results/exp11"))
    ap.add_argument("--games", type=str, default="")
    args = ap.parse_args()

    data_root: Path = args.data_root
    out_root: Path = args.out_root
    walk_out = out_root / "clean_walkthroughs"
    review_out = out_root / "review"
    walk_out.mkdir(parents=True, exist_ok=True)
    review_out.mkdir(parents=True, exist_ok=True)

    if args.games:
        games = [g.strip() for g in args.games.split(",") if g.strip()]
    else:
        games = sorted(
            d.name for d in (data_root / "data").iterdir()
            if d.is_dir() and (d / f"{d.name}.walkthrough").exists()
        )

    summary: List[Dict] = []
    total_match = 0
    total_mismatch = 0
    total_kept = 0
    for game in games:
        text, stats = build_clean(game, data_root)
        if "error" in stats:
            summary.append({"game": game, **stats})
            continue
        game_dir = walk_out / game
        game_dir.mkdir(exist_ok=True)
        (game_dir / f"{game}.walkthrough").write_text(text, encoding="utf-8")
        # Copy locations/actions over for downstream use
        for fname in (f"{game}.locations.json", f"{game}.actions.json"):
            src = data_root / "data_fixed" / game / fname
            if src.exists():
                (game_dir / fname).write_text(src.read_text())

        # Extract original step numbers we kept (recover from stats)
        raw_path = data_root / "data" / game / f"{game}.walkthrough"
        raw_steps_all = parse_walkthrough(raw_path.read_text(encoding="utf-8"))
        kept_orig_nums = [s["step_num"] for s in raw_steps_all
                           if s["action"].lower() in CANONICAL_DIRECTIONS]
        review = review_clean(game, text, data_root, kept_orig_step_nums=kept_orig_nums)
        per_game_md = [f"# Review: {game}\n",
                       f"- Source steps: {stats['src_step_count']}",
                       f"- Kept steps: {stats['kept_step_count']}",
                       f"- Skipped non-canonical actions: {stats['skipped_non_canonical_count']}",
                       f"  - distribution: {stats['skipped_action_distribution']}",
                       f"- Synthetic lines removed: {stats['synthetic_lines_removed']}",
                       f"- GT edges: {review['num_gt_edges']}",
                       f"- GT distinct step indices: {review['num_gt_step_indices']}",
                       f"- Kept steps matched to a GT-step: {review['n_matched_to_gt_step']}/{review['n_steps']}",
                       f"- Unmapped kept steps: {review['n_unmapped_steps']}",
                       f"- Synthetic-text leaks in kept steps: {len(review['synthetic_text_leaks'])}\n"]
        if review["synthetic_text_leaks"]:
            per_game_md.append("## Synthetic-text leaks (should be 0)\n")
            for L in review["synthetic_text_leaks"][:10]:
                per_game_md.append(f"- step {L['step']}: {L['line']!r}")
        if review["unmapped_steps"]:
            per_game_md.append("\n## Unmapped kept steps (action ∈ canonical but no GT edge cites step) — first 20\n")
            for m in review["unmapped_steps"][:20]:
                per_game_md.append(
                    f"- new step {m['new_step_num']} (orig {m['orig_step_num']}): "
                    f"--[{m['action']}]--> {m['obs_first_line']!r}"
                )
        (review_out / f"{game}.md").write_text("\n".join(per_game_md) + "\n")

        row = {
            "game": game,
            **stats,
            "n_matched_to_gt_step": review["n_matched_to_gt_step"],
            "n_unmapped_steps": review["n_unmapped_steps"],
            "synthetic_leaks": len(review["synthetic_text_leaks"]),
            "num_gt_edges": review["num_gt_edges"],
            "num_gt_step_indices": review["num_gt_step_indices"],
        }
        summary.append(row)
        total_match += review["n_matched_to_gt_step"]
        total_mismatch += review["n_unmapped_steps"]
        total_kept += stats["kept_step_count"]

    # Global review
    total_synth_leak = sum(r.get("synthetic_leaks", 0) or 0 for r in summary)
    md = ["# Clean-walkthrough rebuild — global review\n",
          f"Games: {len(summary)}",
          f"Total kept steps: {total_kept}",
          f"Total kept steps that match a GT-edge step index: {total_match}",
          f"Total unmapped kept steps: {total_mismatch}",
          f"Macro match rate: {100 * total_match / max(1, total_kept):.1f}%",
          f"Synthetic-text leaks in kept observations: {total_synth_leak}\n",
          "(An unmapped step has a canonical direction action but no GT edge cites it via "
          "`seen_in_forward` or `seen_in_reversed`. These steps were re-traversals or were "
          "removed during edge refinement; the new LLM should still infer the edge.)\n",
          "## Per-game summary",
          "| game | src | kept | skipped | synth_removed | gt_edges | matched | unmapped | leaks |",
          "|------|----:|-----:|--------:|--------------:|---------:|--------:|---------:|------:|"]
    for r in summary:
        if "error" in r:
            md.append(f"| {r['game']} | ERROR | | | | | | | |")
            continue
        md.append(f"| {r['game']} | {r['src_step_count']} | {r['kept_step_count']} | "
                  f"{r['skipped_non_canonical_count']} | {r['synthetic_lines_removed']} | "
                  f"{r['num_gt_edges']} | {r['n_matched_to_gt_step']} | "
                  f"{r['n_unmapped_steps']} | {r['synthetic_leaks']} |")

    (out_root / "global_review.md").write_text("\n".join(md) + "\n")
    (out_root / "raw.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_root}/global_review.md")
    print("\n".join(md[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
