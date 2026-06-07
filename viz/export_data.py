"""Export per-game JSON for the interactive viewer.

For each game we emit a single JSON containing:
  - meta:    name, dataset, n_rooms, n_edges, walkthrough text
  - rooms:   [{id, label, description}]
  - states:
      gt:        {edges: [{src, dst, dir, kind: 'gt'}]}
      noised:    {edges, noise_records, conflicts, noise_summary}
                 (TextWorld only — MANGO uses 'llm_built' instead)
      llm_built: {edges, n_correct, n_wrong_dir, n_spurious}
                 (MANGO only)
      repaired:  {edges, actions: [{kind, target, new_dir, reason}]}

TextWorld games are sourced from results/exp23/games/.
MANGO games are sourced from data_fixed/.
LLM-built MANGO maps are sourced from results/exp14/gpt-4.1/.

We then run heuristic_remove on each noised / llm_built input to
generate the 'repaired' state without any LLM calls.

Output: viz/data/<dataset>/<game>.json + viz/data/games_index.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import textworld

from maprepair.agents.heuristic import HeuristicRepairAgent
from maprepair.conflict import detect_all
from maprepair.graph import NavGraph
from experiments.exp16_noise import apply_regime, regime_by_name


ROOT = Path("/Users/puzhen/Downloads/spatial_paper_polish")
OUT = ROOT / "viz/data"
TW_GAMES = ROOT / "results/exp23/games"
MANGO_GT = Path("/Users/puzhen/Downloads/maprepair/data_fixed")
MANGO_LLM = ROOT / "results/exp14/gpt-4.1"


def graph_to_edges(g: NavGraph) -> List[Dict]:
    out = []
    for e in g.primary_edges():
        out.append({"src": e.source, "dst": e.target,
                    "dir": e.direction})
    return out


def conflicts_to_dict(g: NavGraph) -> List[Dict]:
    out = []
    for c in detect_all(g):
        out.append({
            "type": c.type,
            "severity": c.severity,
            "description": c.description,
            "edges": [list(e) for e in c.involved_edges],
        })
    return out


def navgraph_from_edges(edges: List[Dict], add_reverse: bool = True) -> NavGraph:
    g = NavGraph()
    seen: Set[Tuple[str, str]] = set()
    for e in edges:
        u = (e.get("src") or e.get("src_node") or "").strip().lower()
        v = (e.get("dst") or e.get("dst_node") or "").strip().lower()
        d = (e.get("dir") or e.get("action") or "").strip().lower()
        if not u or not v or u == v or not d or (u, v) in seen:
            continue
        try:
            g.add_edge(u, v, d, add_auto_reverse=add_reverse)
            seen.add((u, v))
        except Exception:
            pass
    return g


# ----------------------------------------------------------------------
# TextWorld
# ----------------------------------------------------------------------

def tw_to_nav(gt_path: Path) -> Tuple[NavGraph, Dict[str, str], Dict[str, str]]:
    """Returns (NavGraph, name->id, id->description)"""
    game = textworld.Game.load(str(gt_path))
    name_of: Dict[str, str] = {}
    desc_of: Dict[str, str] = {}
    for r in game.world.rooms:
        info = game.infos.get(r.id, None)
        name = (info.name if info else r.name).strip().lower()
        name_of[r.id] = name
        if info and info.desc:
            # strip tags like [if c_0 is open]...[end if]
            desc = re.sub(r"\[[^\]]*\]", "", info.desc).strip()
            desc = re.sub(r"\s+", " ", desc)
            desc_of[name] = desc
    g = NavGraph()
    seen: Set[Tuple[str, str]] = set()
    for r in game.world.rooms:
        u = name_of[r.id]
        for d, dst_room in r.exits.items():
            v = name_of[dst_room.id]
            if u == v or (u, v) in seen:
                continue
            try:
                g.add_edge(u, v, d, add_auto_reverse=False)
                seen.add((u, v))
            except Exception:
                pass
    return g, name_of, desc_of


def export_textworld() -> List[Dict]:
    """One JSON per TextWorld game."""
    out_dir = OUT / "textworld"
    out_dir.mkdir(parents=True, exist_ok=True)
    index: List[Dict] = []
    for gdir in sorted(TW_GAMES.iterdir()):
        gt_path = gdir / "game.json"
        if not gt_path.exists():
            continue
        gid = gdir.name
        gt, name_of, desc_of = tw_to_nav(gt_path)
        rooms = []
        for name in sorted(gt.nodes()):
            rooms.append({
                "id": name, "label": name,
                "description": desc_of.get(name, "")[:300],
            })
        gt_edges_dict = graph_to_edges(gt)

        # Apply mango_like noise (seed 0)
        noised = gt.copy()
        recs = apply_regime(noised, regime_by_name("mango_like"), seed=0)
        noise_records = [r.to_dict() for r in recs]
        from collections import Counter
        noise_summary = Counter(r["type"] for r in noise_records)
        noised_edges = graph_to_edges(noised)
        conflicts = conflicts_to_dict(noised)

        # Heuristic repair
        agent = HeuristicRepairAgent(prefer_remove=True)
        result = agent.repair(noised.copy(), max_iterations=30)
        repaired_edges = graph_to_edges(result.graph_after)
        actions = []
        for a in result.actions:
            actions.append({
                "kind": a.kind,
                "target": list(a.target) if a.target else None,
                "new_dir": a.new_direction,
                "reason": (a.reason or "")[:160],
            })

        # Load walkthrough text (per-step LLM input)
        wt_path = gdir / "walkthrough.json"
        walkthrough = []
        if wt_path.exists():
            for step in json.loads(wt_path.read_text()):
                walkthrough.append({
                    "step": step["step_num"],
                    "action": step["action"],
                    "obs": (step["observation"] or "").strip()[:300],
                    "gt_src": step.get("gt_src", ""),
                    "gt_dst": step.get("gt_dst", ""),
                })

        # Game metadata (size, # nodes/edges, etc.)
        data = {
            "id": gid,
            "dataset": "textworld",
            "n_rooms": gt.num_nodes(),
            "n_edges": len(gt_edges_dict),
            "rooms": rooms,
            "walkthrough": walkthrough,
            "states": {
                "gt": {"edges": gt_edges_dict},
                "noised": {
                    "edges": noised_edges,
                    "noise_records": noise_records,
                    "noise_summary": dict(noise_summary),
                    "conflicts": conflicts,
                },
                "repaired": {
                    "edges": repaired_edges,
                    "actions": actions,
                    "n_conflicts_before": len(conflicts),
                    "n_conflicts_after": len(detect_all(result.graph_after)),
                },
            },
        }
        (out_dir / f"{gid}.json").write_text(json.dumps(data, indent=2))
        index.append({
            "id": gid, "dataset": "textworld",
            "label": f"{gid} · {gt.num_nodes()} rooms",
            "n_rooms": gt.num_nodes(),
            "n_edges": len(gt_edges_dict),
            "n_noise": len(noise_records),
            "n_conflicts": len(conflicts),
            "n_actions": len(actions),
            "n_conflicts_after": len(detect_all(result.graph_after)),
        })
        print(f"  TextWorld {gid}: {gt.num_nodes()} rooms, {len(gt_edges_dict)} GT edges, "
              f"{len(recs)} noise → {len(conflicts)} conflicts → "
              f"{len(actions)} actions → {len(detect_all(result.graph_after))} remaining")
    return index


# ----------------------------------------------------------------------
# MANGO
# ----------------------------------------------------------------------

def export_mango(games: List[str]) -> List[Dict]:
    out_dir = OUT / "mango"
    out_dir.mkdir(parents=True, exist_ok=True)
    index: List[Dict] = []
    for gid in games:
        gt_root = MANGO_GT / gid
        locs_path = gt_root / f"{gid}.locations.json"
        edges_path = gt_root / f"{gid}.edges.json"
        llm_path = MANGO_LLM / f"{gid}_edges.json"
        if not (locs_path.exists() and edges_path.exists() and llm_path.exists()):
            print(f"  MANGO {gid}: skip (missing files)")
            continue
        locations = [l.strip().lower() for l in json.loads(locs_path.read_text())]
        gt_raw = json.loads(edges_path.read_text())
        # Build the GT graph (just primary forward edges, no auto-reverse since
        # MANGO data is already explicit and includes both directions)
        gt = NavGraph()
        seen: Set[Tuple[str, str]] = set()
        for e in gt_raw:
            u = (e.get("src_node") or "").strip().lower()
            v = (e.get("dst_node") or "").strip().lower()
            d = (e.get("action") or "").strip().lower()
            if not u or not v or u == v or not d or (u, v) in seen:
                continue
            try:
                gt.add_edge(u, v, d, add_auto_reverse=False)
                seen.add((u, v))
            except Exception:
                pass

        # LLM-built map
        llm_raw = json.loads(llm_path.read_text())
        llm = navgraph_from_edges(llm_raw, add_reverse=True)

        # Compute label kinds for LLM edges vs GT
        gt_with_dir: Set[Tuple[str, str, str]] = {
            (e.source, e.target, e.direction) for e in gt.primary_edges()
        }
        gt_pairs: Set[Tuple[str, str]] = {(s, t) for (s, t, _d) in gt_with_dir}
        gt_outgoing: Dict[Tuple[str, str], str] = {
            (s, d): t for (s, t, d) in gt_with_dir
        }
        gt_nodes: Set[str] = {n for n in gt.nodes()}
        llm_edge_labels = []
        for e in llm.primary_edges():
            u, v, d = e.source, e.target, e.direction
            if u not in gt_nodes:
                kind = "hallucinated_src"
            elif v not in gt_nodes:
                kind = "hallucinated_dst"
            elif (u, v, d) in gt_with_dir:
                kind = "correct"
            elif (u, v) in gt_pairs:
                kind = "wrong_direction"
            elif (u, d) in gt_outgoing:
                kind = "wrong_dst"
            else:
                kind = "spurious"
            llm_edge_labels.append({"src": u, "dst": v, "dir": d, "kind": kind})

        # Run heuristic repair on the LLM-built map
        agent = HeuristicRepairAgent(prefer_remove=True)
        result = agent.repair(llm.copy(), max_iterations=40)
        repaired_edges = graph_to_edges(result.graph_after)
        actions = []
        for a in result.actions:
            actions.append({
                "kind": a.kind,
                "target": list(a.target) if a.target else None,
                "new_dir": a.new_direction,
                "reason": (a.reason or "")[:160],
            })

        all_nodes = sorted(set(gt.nodes()) | set(llm.nodes()))
        rooms = [{"id": n, "label": n, "description": ""} for n in all_nodes]

        data = {
            "id": gid,
            "dataset": "mango",
            "n_rooms": gt.num_nodes(),
            "n_gt_edges": len(gt_with_dir),
            "n_llm_edges": len(llm.primary_edges()),
            "rooms": rooms,
            "walkthrough": [],  # MANGO walkthroughs are large; load separately if needed
            "states": {
                "gt": {"edges": graph_to_edges(gt)},
                "llm_built": {
                    "edges": llm_edge_labels,
                    "n_correct": sum(1 for e in llm_edge_labels if e["kind"] == "correct"),
                    "n_wrong_direction": sum(1 for e in llm_edge_labels if e["kind"] == "wrong_direction"),
                    "n_wrong_dst": sum(1 for e in llm_edge_labels if e["kind"] == "wrong_dst"),
                    "n_spurious": sum(1 for e in llm_edge_labels if e["kind"] == "spurious"),
                    "n_hallucinated_dst": sum(1 for e in llm_edge_labels if e["kind"] == "hallucinated_dst"),
                    "n_hallucinated_src": sum(1 for e in llm_edge_labels if e["kind"] == "hallucinated_src"),
                    "conflicts": conflicts_to_dict(llm),
                },
                "repaired": {
                    "edges": repaired_edges,
                    "actions": actions,
                    "n_conflicts_before": len(detect_all(llm)),
                    "n_conflicts_after": len(detect_all(result.graph_after)),
                },
            },
        }
        (out_dir / f"{gid}.json").write_text(json.dumps(data, indent=2))
        index.append({
            "id": gid, "dataset": "mango",
            "label": f"{gid} · {gt.num_nodes()} rooms · {len(gt_with_dir)} GT edges",
            "n_rooms": gt.num_nodes(),
            "n_edges": len(gt_with_dir),
            "n_llm_edges": len(llm.primary_edges()),
            "n_correct": data["states"]["llm_built"]["n_correct"],
            "n_actions": len(actions),
        })
        print(f"  MANGO {gid}: GT {gt.num_nodes()} rooms / {len(gt_with_dir)} edges; "
              f"LLM {len(llm.primary_edges())} edges "
              f"(correct={data['states']['llm_built']['n_correct']}, "
              f"spurious={data['states']['llm_built']['n_spurious']}); "
              f"repair {len(actions)} actions → "
              f"{len(detect_all(result.graph_after))} conflicts remaining")
    return index


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Exporting TextWorld games...")
    tw_index = export_textworld()
    print("\nExporting MANGO games (illustrative subset)...")
    mango_index = export_mango([
        "zork1",       # canonical clean
        "cutthroat",   # F3: 5x 'back alley (...)' hierarchical-name confusion
        "advent",      # F1: magic words / non-Euclidean
        "night",       # F2: dark passages
        "zork2",       # F6: vertical inverse-direction confusion
        "ludicorp",    # mid-size modern office floors
    ])
    index = {"games": tw_index + mango_index}
    (OUT / "games_index.json").write_text(json.dumps(index, indent=2))
    print(f"\nWrote {OUT}/games_index.json with {len(index['games'])} games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
