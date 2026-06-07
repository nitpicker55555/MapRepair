"""Experiment 12: re-run LLM incremental mapping on the V3 GT-aligned
clean walkthroughs (from exp11c) and compare against the old run on the
polluted `repaired_walkthroughs/`.

For each game we:
  1. Read the V3 clean walkthrough.
  2. Use the canonical `locations.json` + `actions.json` from data_fixed.
  3. Step-by-step incremental mapping with gpt-4.1 (matches the paper
     setup but on cleaner input).
  4. Save the resulting edge list.
  5. Compute:
     - GT node recall / precision / F1
     - GT edge recall / precision / F1
     - direction-aware edge accuracy
     - conflict count from our detector
  6. Compare with the corresponding numbers from the old
     repaired_walkthroughs run (results/exp10/raw.json).

Output:
  results/exp12/<model>/<game>_edges.json     LLM-generated edges
  results/exp12/<model>/<game>.log            per-step LLM decisions
  results/exp12/<model>/quality.json          per-game metrics
  results/exp12/<model>/summary.md            human-readable comparison
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from maprepair.conflict import detect_all
from maprepair.graph import NavGraph
from maprepair.llm_client import chat_json, message
from maprepair.mango import ground_truth_graph

CANONICAL_DIRECTIONS = (
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
    "up", "down", "in", "out", "enter", "exit",
)

SYS = (
    "You are a text-adventure map analyst. You analyse one walkthrough step "
    "at a time and decide whether the player moved to a new location. Only "
    "output strict JSON. Be precise: emit a movement edge when the action is "
    "a compass direction and the observation indicates arrival in a (named) "
    "room. The action is always one of the canonical compass directions; "
    "your job is to identify the destination room name."
)


def parse_walkthrough(text: str) -> List[Dict]:
    blocks = [b.strip() for b in text.split("===========") if b.strip()]
    out: List[Dict] = []
    for block in blocks:
        sn: Optional[int] = None
        act: Optional[str] = None
        obs: List[str] = []
        in_obs = False
        for L in block.splitlines():
            if L.startswith("==>STEP NUM:"):
                sn = int(L.split(":", 1)[1].strip())
                in_obs = False
            elif L.startswith("==>ACT:"):
                act = L.split(":", 1)[1].strip()
                in_obs = False
            elif L.startswith("==>OBSERVATION:"):
                first = L.split(":", 1)[1].strip()
                if first:
                    obs.append(first)
                in_obs = True
            else:
                if in_obs:
                    obs.append(L)
        if sn is not None and act is not None:
            out.append({"step_num": sn, "action": act, "observation": "\n".join(obs).strip()})
    return out


def _user_prompt(step: Dict, prev: List[Dict], current_loc: Optional[str],
                  locations: List[str], actions: List[str]) -> str:
    ctx = ""
    if prev:
        ctx = "Recent steps (most recent last):\n"
        for p in prev:
            obs_short = (p["observation"] or "")[:160]
            ctx += f"  step {p['step_num']}: action={p['action']!r}; obs={obs_short!r}\n"
    locs_short = ", ".join(locations[:60]) if locations else "<unknown>"
    return f"""{ctx}
Known canonical locations (lowercased, exhaustive list): {locs_short}
Known actions: {', '.join(actions) if actions else '<unknown>'}
Current location BEFORE this step: {current_loc or '<unknown>'}

ANALYZE THIS STEP:
  step_num: {step['step_num']}
  action: {step['action']!r}
  observation: {step['observation']!r}

The action is in the canonical compass set (north/south/east/west/up/down/
northeast/.../in/out/enter/exit). Your job is to identify the destination
room from the observation. Some walks land in dark or NPC-occupied rooms
where the observation text doesn't directly state the room name — in
those cases infer the destination from the prior context (which room is
adjacent to current_location via this direction, given the canonical
list).

Output strict JSON:
{{
  "is_movement": <bool>,
  "src_node": <string|null>,        // copy of current_location
  "dst_node": <string|null>,        // canonical destination room name
  "action": <string>,               // the same compass direction
  "seen_in_forward": <int>,         // = step_num
  "current_location": <string>,     // destination if movement, otherwise unchanged
  "reasoning": <short string>
}}

Always set is_movement=true for canonical compass actions; the V3 input
guarantees the step corresponds to a real GT edge.
Match the destination name to the canonical list when possible.
"""


def map_one_step(step: Dict, prev: List[Dict], current_loc: Optional[str],
                  locations: List[str], actions: List[str], model: str) -> Dict:
    try:
        resp = chat_json([
            message("system", SYS),
            message("user", _user_prompt(step, prev, current_loc, locations, actions)),
        ], model=model, temperature=0.0, max_tokens=400)
    except Exception as e:
        return {"is_movement": False, "current_location": current_loc, "error": str(e)}
    resp.setdefault("is_movement", False)
    resp.setdefault("current_location", current_loc)
    return resp


def map_game(game: str, walk_root: Path, gt_root: Path, out_dir: Path,
              model: str, context_size: int = 2) -> Dict:
    walk_path = walk_root / game / f"{game}.walkthrough"
    if not walk_path.exists():
        return {"game": game, "error": "walkthrough missing"}
    locations_path = gt_root / game / f"{game}.locations.json"
    actions_path = gt_root / game / f"{game}.actions.json"
    locations = json.loads(locations_path.read_text()) if locations_path.exists() else []
    actions = json.loads(actions_path.read_text()) if actions_path.exists() else []
    steps = parse_walkthrough(walk_path.read_text(encoding="utf-8"))

    edges_out: List[Dict] = []
    log_lines: List[str] = [f"# {game}  model={model}  n_steps={len(steps)}"]
    current_loc: Optional[str] = None
    start = time.time()
    for i, step in enumerate(steps):
        prev = steps[max(0, i - context_size): i]
        r = map_one_step(step, prev, current_loc, locations, actions, model)
        new_loc = r.get("current_location") or current_loc
        if r.get("is_movement"):
            src = (r.get("src_node") or current_loc or "").strip().lower() or None
            dst = (r.get("dst_node") or "").strip().lower() or None
            action = (r.get("action") or step["action"]).strip().lower()
            seen = r.get("seen_in_forward") or step["step_num"]
            if src and dst and action in CANONICAL_DIRECTIONS:
                edges_out.append({
                    "src_node": src, "dst_node": dst, "action": action,
                    "seen_in_forward": int(seen),
                })
                log_lines.append(f"step {step['step_num']:>3} MOVE {src!r} --[{action}]--> {dst!r}")
                current_loc = dst
            else:
                log_lines.append(f"step {step['step_num']:>3} bad movement output: {r}")
        else:
            if new_loc and new_loc.lower() != (current_loc or "").lower():
                current_loc = new_loc.lower()
            log_lines.append(f"step {step['step_num']:>3} non-movement (loc={current_loc!r}) action={step['action']!r}")
        if "error" in r:
            log_lines.append(f"  WARN: {r['error']}")

    elapsed = time.time() - start
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{game}_edges.json").write_text(json.dumps(edges_out, indent=2, ensure_ascii=False))
    (out_dir / f"{game}.log").write_text("\n".join(log_lines))
    return {"game": game, "n_steps": len(steps), "n_edges": len(edges_out),
            "elapsed_sec": elapsed}


# ----------------------------------------------------------------------
# Quality evaluation
# ----------------------------------------------------------------------

def _norm(s: str) -> str:
    return (s or "").strip().lower()


def evaluate_quality(game: str, edges_path: Path, gt_root: Path) -> Dict:
    if not edges_path.exists():
        return {"game": game, "error": "edges missing"}
    pred_edges = json.loads(edges_path.read_text())
    gt = ground_truth_graph(game)
    pred_nodes: Set[str] = set()
    pred_pairs: Set[Tuple[str, str]] = set()
    pred_with_dir: Set[Tuple[str, str, str]] = set()
    for e in pred_edges:
        u = _norm(e.get("src_node")); v = _norm(e.get("dst_node")); a = _norm(e.get("action"))
        if not u or not v or u == v:
            continue
        pred_nodes.add(u); pred_nodes.add(v)
        pred_pairs.add((u, v))
        pred_with_dir.add((u, v, a))
    gt_nodes: Set[str] = set(_norm(n) for n in gt.nodes())
    gt_pairs: Set[Tuple[str, str]] = set((e.source, e.target) for e in gt.primary_edges())
    gt_with_dir: Set[Tuple[str, str, str]] = set((e.source, e.target, e.direction) for e in gt.primary_edges())

    def prf(pred: Set, gold: Set) -> Tuple[float, float, float]:
        if not gold:
            return 0.0, 0.0, 0.0
        tp = len(pred & gold)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(gold)
        f = (2 * p * r) / (p + r) if (p + r) else 0.0
        return p, r, f

    np_, nr_, nf_ = prf(pred_nodes, gt_nodes)
    ep_, er_, ef_ = prf(pred_pairs, gt_pairs)
    dp_, dr_, df_ = prf(pred_with_dir, gt_with_dir)
    # Direction accuracy on edge-pair overlap
    overlap = pred_pairs & gt_pairs
    gt_dir = {(u, v): _norm(e.direction)
              for u, v, e in [(e.source, e.target, e) for e in gt.primary_edges()]}
    pred_dir = {(u, v): a for (u, v, a) in pred_with_dir}
    dir_correct = sum(1 for e in overlap if pred_dir.get(e) == gt_dir.get(e))
    dir_acc = dir_correct / len(overlap) if overlap else 0.0
    # Conflict count from our detector (drop self-loops + missing fields)
    safe_edges = []
    for e in pred_edges:
        u = _norm(e.get("src_node")); v = _norm(e.get("dst_node"))
        a = _norm(e.get("action"))
        if not u or not v or not a or u == v:
            continue
        safe_edges.append((u, v, a))
    try:
        g = NavGraph.from_edges(safe_edges, add_auto_reverse=True)
        conflicts = detect_all(g)
    except Exception:
        conflicts = []

    return {
        "game": game,
        "pred_nodes": len(pred_nodes), "gt_nodes": len(gt_nodes),
        "pred_pairs": len(pred_pairs), "gt_pairs": len(gt_pairs),
        "node_recall": nr_, "node_precision": np_, "node_f1": nf_,
        "edge_recall": er_, "edge_precision": ep_, "edge_f1": ef_,
        "edge_with_dir_recall": dr_, "edge_with_dir_precision": dp_, "edge_with_dir_f1": df_,
        "direction_accuracy": dir_acc,
        "conflicts_total": len(conflicts),
        "conflicts_by_type": {
            "direction": sum(1 for c in conflicts if c.type == "direction"),
            "topology": sum(1 for c in conflicts if c.type == "topology"),
            "naming": sum(1 for c in conflicts if c.type == "naming"),
        },
    }


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk-root", type=Path,
                    default=Path("results/exp11c/clean_walkthroughs"),
                    help="Directory of V3 clean walkthroughs (one subdir per game).")
    ap.add_argument("--gt-root", type=Path,
                    default=Path("/Users/puzhen/Downloads/maprepair/data_fixed"),
                    help="Directory of refined GT (locations.json, actions.json, edges.json).")
    ap.add_argument("--out-root", type=Path, default=Path("results/exp12"))
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games", default="")
    args = ap.parse_args()

    model_dir = args.out_root / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.games:
        games = [g.strip() for g in args.games.split(",") if g.strip()]
    else:
        games = sorted(d.name for d in args.walk_root.iterdir() if d.is_dir())
    print(f"Mapping {len(games)} games with {args.model}, workers={args.workers}")

    map_results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(map_game, g, args.walk_root, args.gt_root, model_dir, args.model): g
            for g in games
        }
        completed = 0
        for fut in as_completed(futs):
            g = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"game": g, "error": str(e)}
            map_results.append(r)
            completed += 1
            print(f"  [{completed}/{len(games)}] {g}: edges={r.get('n_edges','?')} "
                  f"t={r.get('elapsed_sec', 0):.1f}s")

    (model_dir / "mapping_summary.json").write_text(json.dumps(map_results, indent=2))

    # ---- Quality
    quality_rows: List[Dict] = []
    for g in games:
        q = evaluate_quality(g, model_dir / f"{g}_edges.json", args.gt_root)
        quality_rows.append(q)

    (model_dir / "quality.json").write_text(json.dumps(quality_rows, indent=2))

    ok = [q for q in quality_rows if "error" not in q]
    if not ok:
        print("No quality rows computed.")
        return 1
    macro = {
        "node_recall": statistics.mean(q["node_recall"] for q in ok),
        "edge_recall": statistics.mean(q["edge_recall"] for q in ok),
        "direction_accuracy": statistics.mean(q["direction_accuracy"] for q in ok),
        "edge_with_dir_recall": statistics.mean(q["edge_with_dir_recall"] for q in ok),
        "conflicts_per_game": statistics.mean(q["conflicts_total"] for q in ok),
    }

    # Compare against old (repaired_walkthroughs) gpt-4.1 numbers in
    # /Users/puzhen/Downloads/maprepair/llm_experiments/results/exp1_mapping_quality/gpt-4.1.json
    old_path = Path("/Users/puzhen/Downloads/maprepair/llm_experiments/results/exp1_mapping_quality/gpt-4.1.json")
    old_macro: Dict = {}
    if old_path.exists():
        od = json.loads(old_path.read_text())
        old_macro = {
            "node_recall": od.get("macro_node_recall"),
            "edge_recall": od.get("macro_edge_recall"),
            "direction_accuracy": od.get("macro_dir_accuracy"),
        }

    md = [
        f"# Experiment 12 — LLM mapping on V3 clean walkthroughs (model={args.model})\n",
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

    md.append("\n## Per-game (sorted by conflict count)\n")
    md.append("| game | gt_edges | pred_edges | edge_recall % | dir_acc % | conflicts |")
    md.append("|------|---------:|-----------:|--------------:|----------:|----------:|")
    for q in sorted(ok, key=lambda x: -x["conflicts_total"]):
        md.append(f"| {q['game']} | {q['gt_pairs']} | {q['pred_pairs']} | "
                  f"{q['edge_recall']*100:.1f} | {q['direction_accuracy']*100:.1f} | "
                  f"{q['conflicts_total']} |")
    (model_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {model_dir}/summary.md")
    print("\nMacro numbers:")
    for k, v in macro.items():
        if k == "conflicts_per_game":
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
