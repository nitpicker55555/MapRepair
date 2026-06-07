"""Experiment 10: programmatically classify every conflict in the LLM-mapped
graphs against ground truth.

For each game and each detected conflict we identify:

  * the conflict type (direction / topology / naming)
  * the involved edges
  * for each involved edge: did it appear in GT? If yes, was the direction
    right? If no, is the source/target present in GT (in which case it
    might be a mis-pairing) or is the room name itself hallucinated?
  * the walkthrough step number where the edge was introduced (from
    seen_in_forward in the LLM output)
  * a one-line root-cause label

Then we aggregate the root causes across all 42 games and produce:

  results/exp10/per_game/<game>.json   detail
  results/exp10/per_game/<game>.md     human-readable per-game report
  results/exp10/global_summary.md      cross-game distribution + examples
  results/exp10/raw.json               machine-readable global aggregate
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from maprepair.conflict import detect_all, Conflict
from maprepair.graph import OPPOSITE, NavGraph
from maprepair.mango import ground_truth_graph, load_repaired_walkthrough, parse_walkthrough
from maprepair.mapping import load_legacy_edges


# ----------------------------------------------------------------------
# Root-cause classification
# ----------------------------------------------------------------------

def classify_edge(edge: Tuple[str, str], llm_dir: str,
                   gt: NavGraph) -> Dict:
    """Classify a single (LLM-emitted) edge by comparing against GT.

    Returns a dict like::
        {
          "edge": (u, v),
          "llm_dir": "north",
          "in_gt": True,
          "gt_dir": "south",
          "case": "wrong_direction",  # or hallucinated_edge, name_hallucinated, ...
          "source_in_gt": True,
          "target_in_gt": True,
        }
    """
    u, v = edge
    u_in = u in gt.nx
    v_in = v in gt.nx
    if gt.has_edge(u, v):
        gt_dir = gt.direction(u, v)
        if gt_dir == llm_dir:
            case = "correct"
        else:
            case = "wrong_direction"
        return {
            "edge": list(edge), "llm_dir": llm_dir,
            "in_gt": True, "gt_dir": gt_dir, "case": case,
            "source_in_gt": True, "target_in_gt": True,
        }
    # edge not in GT — figure out why
    if u_in and v_in:
        # both nodes exist in GT, but the edge between them doesn't.
        # could be wrong direction in disguise (LLM swapped src/dst) or hallucinated.
        if gt.has_edge(v, u):
            return {
                "edge": list(edge), "llm_dir": llm_dir,
                "in_gt": False, "gt_dir": gt.direction(v, u),
                "case": "swapped_src_dst",
                "source_in_gt": True, "target_in_gt": True,
            }
        return {
            "edge": list(edge), "llm_dir": llm_dir,
            "in_gt": False, "gt_dir": None,
            "case": "hallucinated_edge",
            "source_in_gt": True, "target_in_gt": True,
        }
    if not u_in and not v_in:
        return {
            "edge": list(edge), "llm_dir": llm_dir,
            "in_gt": False, "gt_dir": None,
            "case": "both_names_hallucinated",
            "source_in_gt": False, "target_in_gt": False,
        }
    return {
        "edge": list(edge), "llm_dir": llm_dir,
        "in_gt": False, "gt_dir": None,
        "case": "src_hallucinated" if not u_in else "dst_hallucinated",
        "source_in_gt": u_in, "target_in_gt": v_in,
    }


def _edges_incident(graph: NavGraph, node: str) -> List[Tuple[str, str]]:
    g = graph.nx
    out: List[Tuple[str, str]] = []
    if node not in g:
        return out
    for u, v in g.in_edges(node):
        if not g[u][v].get("is_auto_reverse"):
            out.append((u, v))
    for u, v in g.out_edges(node):
        if not g[u][v].get("is_auto_reverse"):
            out.append((u, v))
    return out


def classify_conflict(conflict: Conflict, llm_graph: NavGraph,
                       gt: NavGraph,
                       step_lookup: Dict[Tuple[str, str], int]) -> Dict:
    """Attach an edge-level classification + a top-level cause label.

    For naming / position-overlap conflicts (where `involved_edges` is empty)
    we expand to the edges *incident* to the conflicting nodes — those are
    the structural candidates for the conflict.
    """
    edge_records: List[Dict] = []
    edges_to_classify: List[Tuple[str, str]] = list(conflict.involved_edges)
    if not edges_to_classify and conflict.involved_nodes:
        seen: Set[Tuple[str, str]] = set()
        for node in conflict.involved_nodes:
            for edge in _edges_incident(llm_graph, node):
                if edge not in seen:
                    seen.add(edge)
                    edges_to_classify.append(edge)

    for edge in edges_to_classify:
        u, v = edge
        if not llm_graph.has_edge(u, v):
            continue
        llm_dir = llm_graph.direction(u, v) or ""
        rec = classify_edge((u, v), llm_dir, gt)
        rec["step"] = step_lookup.get((u, v))
        edge_records.append(rec)

    # Roll up to a single root-cause label
    cases = [r["case"] for r in edge_records]
    case_counts = Counter(cases)
    name_cases = {"src_hallucinated", "dst_hallucinated", "both_names_hallucinated"}

    if not cases:
        root = "empty"
    elif conflict.type == "naming":
        # naming conflict: the same name reached at two coordinates. Almost
        # always means LLM gave the same name to genuinely distinct rooms.
        if any(c == "correct" for c in cases) and any(c in {"wrong_direction", "hallucinated_edge"} or c in name_cases for c in cases):
            root = "real_name_corrupted_by_neighbour_error"
        elif all(c == "correct" for c in cases):
            root = "naming_collision_on_correct_subgraph"
        elif any(c in name_cases for c in cases):
            root = "name_hallucination"
        else:
            root = "naming_mixed"
    elif conflict.type == "topology" and ("position" in conflict.description and "occupied" in conflict.description):
        # spatial overlap: two distinct rooms inferred at same coord.
        if any(c == "wrong_direction" for c in cases) and any(c == "correct" for c in cases):
            root = "wrong_direction_caused_overlap"
        elif any(c == "wrong_direction" for c in cases):
            root = "wrong_direction_all"
        elif any(c in name_cases for c in cases):
            root = "name_hallucination_caused_overlap"
        elif all(c == "correct" for c in cases):
            root = "false_positive_overlap"   # all involved edges match GT
        else:
            root = "overlap_mixed"
    elif conflict.type == "topology":
        # reverse-direction mismatch / self-loop / disconnected
        if any(c == "wrong_direction" for c in cases):
            root = "reverse_mismatch_real"
        elif any(c == "hallucinated_edge" for c in cases):
            root = "reverse_mismatch_hallucinated"
        else:
            root = "topology_mixed"
    elif conflict.type == "direction":
        if all(c == "hallucinated_edge" for c in cases):
            root = "all_hallucinated_edges"
        elif "swapped_src_dst" in cases:
            root = "src_dst_swap"
        elif all(c == "wrong_direction" for c in cases):
            root = "wrong_direction_all"
        elif any(c == "wrong_direction" for c in cases) and any(c == "correct" for c in cases):
            root = "real_vs_hallucinated"
        elif any(c == "hallucinated_edge" for c in cases) and any(c == "correct" for c in cases):
            root = "real_vs_hallucinated"
        elif any(c in name_cases for c in cases):
            root = "name_hallucination"
        else:
            root = "direction_mixed"
    else:
        root = "uncategorised"

    return {
        "type": conflict.type,
        "description": conflict.description,
        "involved_edges": [list(e) for e in conflict.involved_edges],
        "involved_nodes": list(conflict.involved_nodes),
        "edge_records": edge_records,
        "case_counts": dict(case_counts),
        "root_cause": root,
    }


# ----------------------------------------------------------------------
# Step lookup
# ----------------------------------------------------------------------

def step_lookup_from_graph(g: NavGraph) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    for e in g.primary_edges():
        out[(e.source, e.target)] = e.step_num
    return out


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def analyse_game(game: str, edges_path: Path) -> Dict:
    if not edges_path.exists():
        return {"game": game, "error": "edges file missing"}
    llm = load_legacy_edges(edges_path)
    gt = ground_truth_graph(game)
    walkthrough_text = load_repaired_walkthrough(game)
    if walkthrough_text is None:
        walkthrough_text = ""
    steps = parse_walkthrough(walkthrough_text, max_steps=70) if walkthrough_text else []
    step_obs = {s.step_num: s for s in steps}

    step_lookup = step_lookup_from_graph(llm)
    raw_conflicts = detect_all(llm)
    classified = [classify_conflict(c, llm, gt, step_lookup) for c in raw_conflicts]

    root_counter = Counter(c["root_cause"] for c in classified)
    by_type = Counter(c["type"] for c in classified)

    return {
        "game": game,
        "num_llm_edges": len(llm.primary_edges()),
        "num_gt_edges": len(gt.primary_edges()),
        "num_conflicts": len(classified),
        "type_distribution": dict(by_type),
        "root_cause_distribution": dict(root_counter),
        "conflicts": classified,
        "walkthrough_steps_available": len(step_obs),
    }


def write_game_md(report: Dict, out_path: Path) -> None:
    lines = [f"# Conflict analysis: {report['game']}\n"]
    if "error" in report:
        lines.append(f"_(error: {report['error']})_")
        out_path.write_text("\n".join(lines) + "\n")
        return
    lines.append(f"- LLM edges: {report['num_llm_edges']}")
    lines.append(f"- GT edges: {report['num_gt_edges']}")
    lines.append(f"- Conflicts: {report['num_conflicts']}")
    lines.append(f"- Type distribution: {report['type_distribution']}")
    lines.append(f"- Root-cause distribution: {report['root_cause_distribution']}")
    lines.append("")
    for i, c in enumerate(report["conflicts"]):
        lines.append(f"## Conflict {i+1} — {c['type']} ({c['root_cause']})")
        lines.append(f"- description: {c['description']}")
        for r in c["edge_records"]:
            edge = r["edge"]
            step = r.get("step")
            note = r["case"]
            if r["case"] == "wrong_direction":
                note += f" (LLM: {r['llm_dir']!r}, GT: {r['gt_dir']!r})"
            elif r["case"] == "swapped_src_dst":
                note += f" (LLM: {edge[0]}--[{r['llm_dir']}]-->{edge[1]} but GT has {edge[1]}--[{r['gt_dir']}]-->{edge[0]})"
            lines.append(f"  - step {step}: {edge[0]} --[{r['llm_dir']}]--> {edge[1]} — {note}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges-dir", type=Path,
                    default=Path("/Users/puzhen/Downloads/maprepair/llm_experiments/results/exp1_mapping/gpt-4.1"))
    ap.add_argument("--out-root", type=Path, default=Path("results/exp10"))
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "per_game").mkdir(exist_ok=True)

    edge_files = sorted(args.edges_dir.glob("*_edges.json"))
    print(f"Analysing {len(edge_files)} games...")

    all_reports: List[Dict] = []
    global_root_counter: Counter = Counter()
    global_type_counter: Counter = Counter()
    examples_per_cause: Dict[str, List[Dict]] = defaultdict(list)

    for f in edge_files:
        game = f.stem.replace("_edges", "")
        r = analyse_game(game, f)
        all_reports.append(r)
        if "error" in r:
            continue
        global_root_counter.update(r["root_cause_distribution"])
        global_type_counter.update(r["type_distribution"])
        # Save per-game artefacts
        (args.out_root / "per_game" / f"{game}.json").write_text(json.dumps(r, indent=2))
        write_game_md(r, args.out_root / "per_game" / f"{game}.md")
        # collect up to 3 examples per root cause
        for c in r["conflicts"]:
            cause = c["root_cause"]
            if len(examples_per_cause[cause]) < 3:
                examples_per_cause[cause].append({
                    "game": game,
                    "type": c["type"],
                    "description": c["description"],
                    "edge_records": c["edge_records"],
                })

    # Global summary
    total_conflicts = sum(r.get("num_conflicts", 0) for r in all_reports if "error" not in r)
    md = [
        "# Global conflict-cause analysis (LLM-mapped gpt-4.1 on MANGO)\n",
        f"Games analysed: {sum(1 for r in all_reports if 'error' not in r)}",
        f"Total conflicts: {total_conflicts}",
        f"Type distribution: {dict(global_type_counter)}",
        "\n## Root-cause distribution",
        "| Root cause | Count | % of total |",
        "|------------|------:|-----------:|",
    ]
    for cause, count in global_root_counter.most_common():
        pct = 100 * count / total_conflicts if total_conflicts else 0
        md.append(f"| {cause} | {count} | {pct:.1f} |")
    md.append("\n## Examples per root cause\n")
    for cause, examples in examples_per_cause.items():
        md.append(f"### {cause}\n")
        for ex in examples[:3]:
            md.append(f"- **{ex['game']}** ({ex['type']}): _{ex['description']}_")
            for r in ex["edge_records"][:3]:
                edge = r["edge"]
                note = r["case"]
                if r["case"] == "wrong_direction":
                    note += f" (LLM dir={r['llm_dir']!r}, GT dir={r['gt_dir']!r})"
                md.append(f"  - step {r.get('step','?')}: `{edge[0]} --[{r['llm_dir']}]--> {edge[1]}` ({note})")
            md.append("")

    (args.out_root / "global_summary.md").write_text("\n".join(md) + "\n")
    (args.out_root / "raw.json").write_text(json.dumps({
        "total_conflicts": total_conflicts,
        "type_distribution": dict(global_type_counter),
        "root_cause_distribution": dict(global_root_counter),
        "per_game": [
            {k: v for k, v in r.items() if k != "conflicts"}
            for r in all_reports
        ],
    }, indent=2))
    print(f"Wrote {args.out_root}/global_summary.md")
    print("\n".join(md[:25]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
