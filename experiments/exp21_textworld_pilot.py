"""Experiment 21 (Phase 1): TextWorld diagnostic pilot.

Generate 10 small TextWorld games with varied size/seed, extract GT
navigation graphs, play a complete BFS walkthrough collecting
(action, observation) pairs, run LLM incremental mapping, and check
the error distribution.

Goal: validate that TextWorld's natural error distribution lands in
the algorithm's target class (edge-level: F2/F6 dominant) rather
than node-level (F3/F4/F5 dominant like MANGO).

Outputs:
  results/exp21/games/<game_id>/{game.z8, walkthrough.txt, gt.json}
  results/exp21/maps/<game_id>_edges.json
  results/exp21/diagnostic.md  -- pilot summary
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import textworld

from maprepair.llm_client import chat_json, message


COMPASS_GO = {
    "north": "go north", "south": "go south", "east": "go east", "west": "go west",
    "up": "go up", "down": "go down", "northeast": "go northeast",
    "northwest": "go northwest", "southeast": "go southeast",
    "southwest": "go southwest",
}
CANONICAL_DIRS = tuple(COMPASS_GO.keys())


def make_game(out_path: Path, world_size: int, seed: int) -> bool:
    """Generate one TextWorld game via tw-make CLI."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "./.venv/bin/tw-make", "custom",
        "--world-size", str(world_size),
        "--nb-objects", "4",
        "--quest-length", "3",
        "--seed", str(seed),
        "--output", str(out_path),
        "--silent", "-f",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/puzhen/Downloads/spatial_paper_polish")
    return r.returncode == 0


def extract_gt(json_path: Path) -> Dict:
    """Read the GT navigation graph (room name -> {direction: room name})."""
    game = textworld.Game.load(str(json_path))
    rooms = {}
    for r in game.world.rooms:
        info = game.infos.get(r.id, None)
        name = (info.name if info else r.name).strip().lower()
        rooms[r.id] = {"id": r.id, "name": name}
    edges = []
    for r in game.world.rooms:
        info_r = game.infos.get(r.id, None)
        src_name = (info_r.name if info_r else r.name).strip().lower()
        for dirn, dst_room in r.exits.items():
            info_d = game.infos.get(dst_room.id, None)
            dst_name = (info_d.name if info_d else dst_room.name).strip().lower()
            edges.append({"src_name": src_name, "dst_name": dst_name,
                          "src_id": r.id, "dst_id": dst_room.id,
                          "direction": dirn})
    return {
        "rooms": rooms,
        "edges": edges,
        "n_rooms": len(rooms),
        "n_edges": len(edges),
        "walkthrough": game.metadata.get("walkthrough", []),
    }


def bfs_tour(gt: Dict, start_id: str) -> List[Tuple[str, str, str]]:
    """BFS the GT graph and produce a tour visiting every room.
    Returns a list of (src_id, direction, dst_id) edges to traverse."""
    visited: Set[str] = {start_id}
    queue: deque[str] = deque([start_id])
    plan: List[Tuple[str, str, str]] = []
    # Build adjacency for quick lookup
    adj: Dict[str, List[Tuple[str, str]]] = {}
    for e in gt["edges"]:
        adj.setdefault(e["src_id"], []).append((e["direction"], e["dst_id"]))
    # DFS-like: walk to nearest unvisited, then walk back when stuck
    path = [start_id]
    while True:
        cur = path[-1]
        outgoing = adj.get(cur, [])
        next_step = None
        for dirn, dst in outgoing:
            if dst not in visited:
                next_step = (dirn, dst)
                break
        if next_step is not None:
            dirn, dst = next_step
            plan.append((cur, dirn, dst))
            visited.add(dst)
            path.append(dst)
        else:
            # backtrack one step
            if len(path) <= 1:
                break  # full tour done
            prev = path[-2]
            # find the direction from cur back to prev
            back_dir = None
            for dirn, dst in outgoing:
                if dst == prev:
                    back_dir = dirn
                    break
            if back_dir is None:
                # no edge back; jump (this shouldn't happen in a well-formed game)
                break
            plan.append((cur, back_dir, prev))
            path.pop()
    return plan


def play_tour(game_z8: Path, plan: List[Tuple[str, str, str]],
              gt: Dict) -> List[Dict]:
    """Replay the BFS plan in the TextWorld env and capture observations."""
    env = textworld.start(str(game_z8))
    state = env.reset()
    # Issue a 'look' to get the starting room name
    state, _, _ = env.step("look")
    # Try to find the start room name from the obs header
    steps: List[Dict] = []
    for i, (src_id, dirn, dst_id) in enumerate(plan):
        cmd = COMPASS_GO.get(dirn, f"go {dirn}")
        state, reward, done = env.step(cmd)
        obs = state.feedback.strip()
        # strip ANSI/prompt
        obs = re.sub(r"^>\s*", "", obs, flags=re.M).strip()
        steps.append({
            "step_num": i,
            "action": dirn,
            "command": cmd,
            "observation": obs,
            "gt_src": gt["rooms"][src_id]["name"],
            "gt_dst": gt["rooms"][dst_id]["name"],
        })
    return steps


# ----------------------------------------------------------------------
# LLM mapping (same prompt skeleton as exp12/exp14)
# ----------------------------------------------------------------------

SYS = (
    "You are a text-adventure map analyst. You analyse one walkthrough step "
    "at a time and decide whether the player moved to a new location. Only "
    "output strict JSON. The action is one of the canonical compass "
    "directions; identify the destination room name from the observation."
)


def _user_prompt(step: Dict, prev: List[Dict], current_loc: Optional[str],
                  locations: List[str]) -> str:
    ctx = ""
    if prev:
        ctx = "Recent steps (most recent last):\n"
        for p in prev:
            obs_short = (p["observation"] or "")[:160]
            ctx += f"  step {p['step_num']}: action={p['action']!r}; obs={obs_short!r}\n"
    locs_short = ", ".join(locations[:60]) if locations else "<unknown>"
    return f"""{ctx}
Known canonical rooms (lowercased): {locs_short}
Current location BEFORE this step: {current_loc or '<unknown>'}

ANALYZE THIS STEP:
  step_num: {step['step_num']}
  action: {step['action']!r}
  observation: {step['observation']!r}

Output strict JSON:
{{
  "is_movement": <bool>,
  "src_node": <string|null>,
  "dst_node": <string|null>,
  "action": <string>,
  "current_location": <string>,
  "reasoning": <short string>
}}
"""


def llm_map_one(step: Dict, prev: List[Dict], current_loc: Optional[str],
                 locations: List[str], model: str) -> Dict:
    try:
        return chat_json(
            [message("system", SYS),
             message("user", _user_prompt(step, prev, current_loc, locations))],
            model=model, temperature=0.0, max_tokens=400,
        )
    except Exception as e:
        return {"is_movement": False, "current_location": current_loc, "error": str(e)}


def map_game(game_id: str, steps: List[Dict], locations: List[str], model: str,
              context_size: int = 2) -> List[Dict]:
    edges_out: List[Dict] = []
    current_loc: Optional[str] = None
    for i, step in enumerate(steps):
        prev = steps[max(0, i - context_size):i]
        r = llm_map_one(step, prev, current_loc, locations, model)
        llm_src = (r.get("src_node") or "").strip().lower() or None
        dst = (r.get("dst_node") or "").strip().lower() or None
        action = (r.get("action") or step["action"]).strip().lower()
        # Apply the exp14 bootstrap fix
        if action not in CANONICAL_DIRS:
            continue
        if r.get("is_movement"):
            src = llm_src or current_loc
            if src and dst and src != dst:
                edges_out.append({"src_node": src, "dst_node": dst, "action": action,
                                  "step_num": step["step_num"],
                                  "gt_src": step["gt_src"], "gt_dst": step["gt_dst"]})
                current_loc = dst
            elif dst:
                current_loc = dst  # bootstrap seed
        else:
            if r.get("current_location"):
                current_loc = (r["current_location"] or "").strip().lower() or current_loc
    return edges_out


def classify_errors(pred_edges: List[Dict], gt: Dict) -> Dict:
    """Bucket pred edges into CORRECT / WRONG_DST / WRONG_DIR / SPURIOUS / HALL."""
    gt_nodes = {r["name"] for r in gt["rooms"].values()}
    gt_with_dir = {(e["src_name"], e["dst_name"], e["direction"]) for e in gt["edges"]}
    gt_pair = {(e["src_name"], e["dst_name"]) for e in gt["edges"]}
    gt_outgoing = {(e["src_name"], e["direction"]): e["dst_name"] for e in gt["edges"]}

    buckets = {"CORRECT": 0, "WRONG_DIRECTION": 0, "WRONG_DST": 0,
               "SPURIOUS_PAIR": 0, "HALLUCINATED_DST": 0, "HALLUCINATED_SRC": 0}
    samples = {k: [] for k in buckets}

    for e in pred_edges:
        u, v, a = e["src_node"], e["dst_node"], e["action"]
        if u not in gt_nodes:
            buckets["HALLUCINATED_SRC"] += 1
            samples["HALLUCINATED_SRC"].append(e); continue
        if v not in gt_nodes:
            buckets["HALLUCINATED_DST"] += 1
            samples["HALLUCINATED_DST"].append(e); continue
        if (u, v, a) in gt_with_dir:
            buckets["CORRECT"] += 1
            samples["CORRECT"].append(e); continue
        if (u, v) in gt_pair:
            buckets["WRONG_DIRECTION"] += 1
            samples["WRONG_DIRECTION"].append(e); continue
        if (u, a) in gt_outgoing:
            buckets["WRONG_DST"] += 1
            samples["WRONG_DST"].append({**e, "gt_dst_for_src_dir": gt_outgoing[(u, a)]}); continue
        buckets["SPURIOUS_PAIR"] += 1
        samples["SPURIOUS_PAIR"].append(e)

    # Recall: how many GT edges are matched exactly
    pred_set = {(e["src_node"], e["dst_node"], e["action"]) for e in pred_edges}
    pred_pair = {(e["src_node"], e["dst_node"]) for e in pred_edges}
    correct = pred_set & gt_with_dir
    recall = len(correct) / len(gt_with_dir) if gt_with_dir else 0.0
    pair_recall = len(pred_pair & gt_pair) / len(gt_pair) if gt_pair else 0.0
    return {"buckets": buckets, "samples": samples,
            "n_gt_edges": len(gt_with_dir), "n_pred_edges": len(pred_edges),
            "gt_edge_recall": recall, "gt_pair_recall": pair_recall}


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("results/exp21"))
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--num-games", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=1)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    games_dir = args.out_root / "games"
    maps_dir = args.out_root / "maps"
    games_dir.mkdir(exist_ok=True)
    maps_dir.mkdir(exist_ok=True)

    # Plan: vary sizes 5,7,9,11 across the games
    sizes = [5, 7, 9, 11]
    plans = []
    for i in range(args.num_games):
        plans.append({"id": f"tw_{i:02d}", "size": sizes[i % len(sizes)],
                      "seed": args.seed_base + i})

    summary_rows = []
    for cfg in plans:
        gid = cfg["id"]
        gdir = games_dir / gid
        gdir.mkdir(exist_ok=True)
        z8_path = gdir / "game.z8"
        json_path = gdir / "game.json"
        if not z8_path.exists():
            print(f"Generating {gid} (size={cfg['size']}, seed={cfg['seed']})", flush=True)
            ok = make_game(z8_path, cfg["size"], cfg["seed"])
            if not ok:
                print(f"  FAILED to generate {gid}"); continue

        # Extract GT
        gt = extract_gt(json_path)
        (gdir / "gt.json").write_text(json.dumps(gt, indent=2))

        # BFS tour
        start_id = list(gt["rooms"].keys())[0]
        plan = bfs_tour(gt, start_id)
        # Save the tour
        tour_rows = [{"src_id": s, "direction": d, "dst_id": t} for (s, d, t) in plan]
        (gdir / "tour.json").write_text(json.dumps(tour_rows, indent=2))

        # Play
        print(f"  Playing tour ({len(plan)} steps)...", flush=True)
        steps = play_tour(z8_path, plan, gt)
        (gdir / "walkthrough.json").write_text(json.dumps(steps, indent=2))
        # Also a human-readable .txt
        wt_text = []
        for s in steps:
            wt_text.append(f"==>STEP NUM: {s['step_num']}\n==>ACT: {s['action']}\n"
                            f"==>OBSERVATION: {s['observation']}\n===========")
        (gdir / "walkthrough.txt").write_text("\n".join(wt_text))

        # LLM mapping
        locations = sorted({r["name"] for r in gt["rooms"].values()})
        print(f"  LLM mapping ({len(steps)} steps)...", flush=True)
        t0 = time.time()
        pred_edges = map_game(gid, steps, locations, args.model)
        elapsed = time.time() - t0
        (maps_dir / f"{gid}_edges.json").write_text(json.dumps(pred_edges, indent=2))

        # Diagnose
        diag = classify_errors(pred_edges, gt)
        summary_rows.append({
            "game": gid, "size": cfg["size"], "seed": cfg["seed"],
            "n_rooms": gt["n_rooms"], "n_gt_edges": gt["n_edges"],
            "tour_steps": len(plan), "n_pred_edges": diag["n_pred_edges"],
            "gt_edge_recall": diag["gt_edge_recall"],
            "gt_pair_recall": diag["gt_pair_recall"],
            "buckets": diag["buckets"],
            "elapsed_sec": elapsed,
        })
        # Save per-game diagnostic
        (maps_dir / f"{gid}_diag.json").write_text(json.dumps({
            **diag, "samples": {k: v[:5] for k, v in diag["samples"].items()},
        }, indent=2))
        print(f"  {gid}: rooms={gt['n_rooms']} gt_edges={gt['n_edges']} "
              f"pred={diag['n_pred_edges']} recall={diag['gt_edge_recall']*100:.1f}% "
              f"correct={diag['buckets']['CORRECT']} buckets={diag['buckets']}", flush=True)

    (args.out_root / "pilot_summary.json").write_text(json.dumps(summary_rows, indent=2))

    # Aggregate
    from collections import Counter
    total = Counter()
    for r in summary_rows:
        for k, v in r["buckets"].items():
            total[k] += v
    total_pred = sum(r["n_pred_edges"] for r in summary_rows)
    total_gt = sum(r["n_gt_edges"] for r in summary_rows)
    correct = total.get("CORRECT", 0)
    recall = correct / total_gt if total_gt else 0

    md = [
        "# Experiment 21 (Phase 1) — TextWorld diagnostic pilot\n",
        f"Generated {len(summary_rows)} games (sizes {sorted(set(r['size'] for r in summary_rows))}).",
        f"Model: {args.model}\n",
        "## Aggregate error distribution\n",
        f"Total pred edges: {total_pred}",
        f"Total GT edges:   {total_gt}",
        f"Micro edge recall: {recall*100:.1f}%",
        f"Macro edge recall: {100*sum(r['gt_edge_recall'] for r in summary_rows)/len(summary_rows):.1f}%\n",
        "| bucket | count | % of pred |",
        "|--------|------:|----------:|",
    ]
    for k, v in total.most_common():
        pct = 100 * v / total_pred if total_pred else 0
        md.append(f"| {k} | {v} | {pct:.1f}% |")

    md.append("\n## Per-game\n")
    md.append("| game | rooms | gt_edges | tour | pred | recall % | CORRECT | WRONG_DIR | WRONG_DST | SPURIOUS | HALL_DST |")
    md.append("|------|-----:|--------:|-----:|-----:|---------:|--------:|----------:|----------:|--------:|---------:|")
    for r in summary_rows:
        b = r["buckets"]
        md.append(f"| {r['game']} | {r['n_rooms']} | {r['n_gt_edges']} | {r['tour_steps']} | "
                  f"{r['n_pred_edges']} | {r['gt_edge_recall']*100:.1f} | "
                  f"{b.get('CORRECT',0)} | {b.get('WRONG_DIRECTION',0)} | "
                  f"{b.get('WRONG_DST',0)} | {b.get('SPURIOUS_PAIR',0)} | {b.get('HALLUCINATED_DST',0)} |")

    md.append("\n## Reading guide\n")
    md.append("- **CORRECT + WRONG_DIRECTION** dominate ⇒ TextWorld errors are *edge-level* — algorithm target ✅")
    md.append("- **WRONG_DST + SPURIOUS_PAIR + HALLUCINATED_***** dominate ⇒ TextWorld errors are *node-level* — same problem as MANGO ❌")

    (args.out_root / "diagnostic.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.out_root}/diagnostic.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
