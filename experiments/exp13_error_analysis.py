"""Experiment 13: per-edge error classification for the V3 LLM mapping
results (exp12). For every game we compare the LLM-emitted edges
against the refined GT and bucket each pred / GT edge into one of:

PRED edges:
  - CORRECT                : (src, dst, dir) exactly matches GT
  - WRONG_DIRECTION        : (src, dst) matches a GT pair but dir differs
  - WRONG_DST              : (src, dir) matches some GT (s,?,d) but dst differs
  - WRONG_SRC              : src is '<unknown>' / not in GT, rest could be right
  - SPURIOUS_PAIR          : both nodes real but no GT edge between them
  - HALLUCINATED_DST       : dst not present in GT locations
  - HALLUCINATED_SRC       : src not present in GT locations (and not <unknown>)
  - SELF_LOOP              : src == dst

GT edges:
  - RECALLED               : matched by some CORRECT pred
  - RECALLED_WRONG_DIR     : same pair recalled but with wrong direction
  - MISSED                 : no pred edge has this src,dst pair at all

The script also samples 5-15 concrete examples per bucket per game so a
human reviewer can read the actual walkthrough context that caused the
error.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from maprepair.mango import ground_truth_graph
from maprepair.graph import OPPOSITE


SPP = Path("/Users/puzhen/Downloads/spatial_paper_polish")
DATA_FIXED = Path("/Users/puzhen/Downloads/maprepair/data_fixed")


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def parse_walk(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    blocks = [b.strip() for b in path.read_text(encoding="utf-8").split("===========") if b.strip()]
    out = []
    for blk in blocks:
        sn = act = None
        obs = []
        in_obs = False
        for L in blk.splitlines():
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
            out.append({"step_num": sn, "action": act,
                         "observation": "\n".join(obs).strip()})
    return out


def parse_log(path: Path) -> Dict[int, Dict]:
    """Returns step_num -> {kind, action, src, dst, raw_line}"""
    out: Dict[int, Dict] = {}
    if not path.exists():
        return out
    for L in path.read_text().splitlines():
        m = re.match(r"step\s+(\d+)\s+MOVE\s+'([^']*)'\s+--\[(\w+)\]-->\s+'([^']*)'", L)
        if m:
            sn = int(m.group(1))
            out[sn] = {"kind": "MOVE", "src": m.group(2),
                       "action": m.group(3), "dst": m.group(4), "raw": L}
            continue
        m = re.match(r"step\s+(\d+)\s+non-movement.*", L)
        if m:
            out[int(m.group(1))] = {"kind": "NON_MOVE", "raw": L}
            continue
        m = re.match(r"step\s+(\d+)\s+bad movement output:\s+(.*)", L)
        if m:
            sn = int(m.group(1))
            try:
                payload = eval(m.group(2), {"__builtins__": {}}, {"True": True, "False": False, "None": None})
            except Exception:
                payload = {}
            out[sn] = {
                "kind": "BAD_MOVE",
                "src": _norm(payload.get("src_node") or "<unknown>"),
                "dst": _norm(payload.get("dst_node")),
                "action": _norm(payload.get("action")),
                "raw": L,
                "reasoning": payload.get("reasoning", ""),
            }
    return out


def classify_game(game: str) -> Dict:
    walk_path = SPP / "results/exp11c/clean_walkthroughs" / game / f"{game}.walkthrough"
    edges_path = SPP / "results/exp12/gpt-4.1" / f"{game}_edges.json"
    log_path = SPP / "results/exp12/gpt-4.1" / f"{game}.log"
    if not edges_path.exists():
        return {"game": game, "error": "no edges"}

    walk = parse_walk(walk_path)
    log = parse_log(log_path)
    pred_raw = json.loads(edges_path.read_text())

    gt = ground_truth_graph(game)
    gt_nodes: Set[str] = set(_norm(n) for n in gt.nodes())
    gt_edges: Set[Tuple[str, str, str]] = set(
        (_norm(e.source), _norm(e.target), _norm(e.direction)) for e in gt.primary_edges()
    )
    gt_pairs: Set[Tuple[str, str]] = {(s, t) for (s, t, _d) in gt_edges}
    gt_outgoing: Dict[Tuple[str, str], str] = {}  # (src, dir) -> dst
    for (s, t, d) in gt_edges:
        gt_outgoing[(s, d)] = t

    # ---- classify PRED edges
    pred_buckets: Dict[str, List[Dict]] = defaultdict(list)
    recalled_pairs: Set[Tuple[str, str]] = set()
    recalled_correct_full: Set[Tuple[str, str, str]] = set()

    for e in pred_raw:
        src = _norm(e.get("src_node"))
        dst = _norm(e.get("dst_node"))
        act = _norm(e.get("action"))
        sn = e.get("seen_in_forward", -1)
        log_entry = log.get(sn, {})
        walk_step = next((s for s in walk if s["step_num"] == sn), {})
        item = {
            "step_num": sn,
            "src": src, "dst": dst, "action": act,
            "walk_action": walk_step.get("action"),
            "obs_snip": (walk_step.get("observation") or "")[:200],
            "log_raw": log_entry.get("raw", ""),
        }
        if src == dst and src:
            pred_buckets["SELF_LOOP"].append(item); continue
        if src == "<unknown>" or not src:
            pred_buckets["WRONG_SRC_UNKNOWN"].append(item); continue
        if src not in gt_nodes:
            pred_buckets["HALLUCINATED_SRC"].append(item); continue
        if dst not in gt_nodes:
            pred_buckets["HALLUCINATED_DST"].append(item); continue
        if (src, dst, act) in gt_edges:
            pred_buckets["CORRECT"].append(item)
            recalled_pairs.add((src, dst))
            recalled_correct_full.add((src, dst, act))
            continue
        # both nodes real, not exact match
        if (src, dst) in gt_pairs:
            pred_buckets["WRONG_DIRECTION"].append(item)
            recalled_pairs.add((src, dst))
            continue
        # check if (src, act) points to some gt dst that isn't ours
        if (src, act) in gt_outgoing:
            item["gt_dst_for_src_dir"] = gt_outgoing[(src, act)]
            pred_buckets["WRONG_DST"].append(item)
            continue
        pred_buckets["SPURIOUS_PAIR"].append(item)

    # ---- classify GT edges
    gt_buckets: Dict[str, List[Dict]] = defaultdict(list)
    for (s, t, d) in gt_edges:
        if (s, t, d) in recalled_correct_full:
            gt_buckets["RECALLED"].append({"src": s, "dst": t, "dir": d})
        elif (s, t) in recalled_pairs:
            gt_buckets["RECALLED_WRONG_DIR"].append({"src": s, "dst": t, "dir": d})
        else:
            gt_buckets["MISSED"].append({"src": s, "dst": t, "dir": d})

    # Try to pin each MISSED to a walkthrough step. Simple heuristic:
    # walk the V3 walkthrough as a path and at each transition check which
    # GT edge was attempted but mis-classified.
    miss_examples = []
    # build the player path from log/pred to identify where misses happen
    cur = None
    for step in walk:
        sn = step["step_num"]
        entry = log.get(sn, {})
        if entry.get("kind") == "MOVE":
            src = _norm(entry["src"])
            dst = _norm(entry["dst"])
            act = _norm(entry["action"])
            if cur is None:
                cur = src
            # Is there a GT edge from cur with this action?
            gt_dst = gt_outgoing.get((cur, act))
            if gt_dst is not None and gt_dst != dst:
                miss_examples.append({
                    "step_num": sn,
                    "cur": cur, "predicted_dst": dst, "gt_dst": gt_dst,
                    "action": act,
                    "action_in_walk": step["action"],
                    "obs": step["observation"][:300],
                })
            cur = dst
        elif entry.get("kind") == "BAD_MOVE":
            miss_examples.append({
                "step_num": sn,
                "cur": cur, "predicted_dst": entry.get("dst"),
                "gt_dst": gt_outgoing.get((cur or "", entry.get("action") or "")),
                "action": entry.get("action"),
                "action_in_walk": step["action"],
                "obs": step["observation"][:300],
                "reason": entry.get("reasoning", ""),
                "bucket_hint": "BAD_MOVE_BOOTSTRAP",
            })
            # leave cur unchanged (we don't know the real source)

    return {
        "game": game,
        "n_walk_steps": len(walk),
        "n_pred": len(pred_raw),
        "n_gt": len(gt_edges),
        "pred_buckets": {k: len(v) for k, v in pred_buckets.items()},
        "gt_buckets": {k: len(v) for k, v in gt_buckets.items()},
        "pred_samples": {k: v[:8] for k, v in pred_buckets.items()},
        "gt_samples": {k: v[:8] for k, v in gt_buckets.items()},
        "miss_examples": miss_examples[:15],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=SPP / "results/exp13/error_analysis.json")
    ap.add_argument("--md", type=Path,
                    default=SPP / "results/exp13/global_summary.md")
    ap.add_argument("--games-md-dir", type=Path,
                    default=SPP / "results/exp13/per_game")
    args = ap.parse_args()

    games = sorted(p.stem.replace("_edges", "")
                   for p in (SPP / "results/exp12/gpt-4.1").glob("*_edges.json"))
    all_results = []
    global_pred = Counter()
    global_gt = Counter()
    for g in games:
        r = classify_game(g)
        if "error" in r:
            print(f"  {g}: ERROR {r['error']}")
            continue
        all_results.append(r)
        for k, v in r["pred_buckets"].items():
            global_pred[k] += v
        for k, v in r["gt_buckets"].items():
            global_gt[k] += v
        print(f"  {g}: pred={r['n_pred']} buckets={r['pred_buckets']} "
              f"gt_miss={r['gt_buckets'].get('MISSED', 0)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    # global markdown
    md = ["# Experiment 13 — V3 mapping error classification\n",
          "Per-edge buckets for the exp12 gpt-4.1 run on V3 clean walkthroughs.\n"]

    total_pred = sum(global_pred.values())
    total_gt = sum(global_gt.values())
    md.append(f"Total predicted edges: {total_pred}")
    md.append(f"Total GT edges:        {total_gt}\n")

    md.append("## Predicted-edge buckets\n")
    md.append("| Bucket | Count | % of pred |")
    md.append("|--------|------:|----------:|")
    for k, c in global_pred.most_common():
        md.append(f"| {k} | {c} | {100*c/max(1,total_pred):.1f}% |")

    md.append("\n## GT-edge buckets (recall view)\n")
    md.append("| Bucket | Count | % of GT |")
    md.append("|--------|------:|--------:|")
    for k, c in global_gt.most_common():
        md.append(f"| {k} | {c} | {100*c/max(1,total_gt):.1f}% |")

    md.append("\n## Per-game error breakdown\n")
    md.append("| game | n_pred | CORRECT | WRONG_DIR | WRONG_DST | WRONG_SRC | "
              "SPURIOUS | HALL_DST | n_gt | MISSED |")
    md.append("|------|------:|--------:|----------:|----------:|----------:|--------:|---------:|----:|------:|")
    for r in sorted(all_results, key=lambda x: -(x['gt_buckets'].get('MISSED', 0))):
        pb = r["pred_buckets"]; gb = r["gt_buckets"]
        md.append(f"| {r['game']} | {r['n_pred']} | "
                  f"{pb.get('CORRECT', 0)} | {pb.get('WRONG_DIRECTION', 0)} | "
                  f"{pb.get('WRONG_DST', 0)} | {pb.get('WRONG_SRC_UNKNOWN', 0)} | "
                  f"{pb.get('SPURIOUS_PAIR', 0)} | {pb.get('HALLUCINATED_DST', 0)} | "
                  f"{r['n_gt']} | {gb.get('MISSED', 0)} |")

    args.md.write_text("\n".join(md) + "\n")
    print(f"\nWrote {args.md}")

    # per-game detail markdowns
    args.games_md_dir.mkdir(parents=True, exist_ok=True)
    for r in all_results:
        lines = [f"# {r['game']} — per-edge error analysis\n",
                 f"V3 walkthrough steps: {r['n_walk_steps']}",
                 f"Predicted edges:      {r['n_pred']}",
                 f"GT edges:             {r['n_gt']}\n",
                 "## Pred buckets",
                 "| bucket | n |", "|--------|--:|"]
        for k, c in sorted(r["pred_buckets"].items(), key=lambda x: -x[1]):
            lines.append(f"| {k} | {c} |")
        lines.append("\n## GT buckets")
        lines.append("| bucket | n |"); lines.append("|--------|--:|")
        for k, c in sorted(r["gt_buckets"].items(), key=lambda x: -x[1]):
            lines.append(f"| {k} | {c} |")
        for bucket in ("WRONG_DIRECTION", "WRONG_DST", "WRONG_SRC_UNKNOWN",
                        "SPURIOUS_PAIR", "HALLUCINATED_DST"):
            if not r["pred_samples"].get(bucket):
                continue
            lines.append(f"\n## Samples — {bucket}\n")
            for s in r["pred_samples"][bucket][:6]:
                lines.append(f"- step {s['step_num']} walk_action={s.get('walk_action')!r} "
                             f"PRED: {s['src']!r} --[{s['action']}]--> {s['dst']!r}"
                             f"{'  (GT here: '+s['gt_dst_for_src_dir']+')' if 'gt_dst_for_src_dir' in s else ''}")
                if s.get("obs_snip"):
                    lines.append(f"  > obs: {s['obs_snip']!r}")
        if r["miss_examples"]:
            lines.append("\n## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)\n")
            for m in r["miss_examples"][:8]:
                lines.append(f"- step {m['step_num']} cur={m['cur']!r} "
                             f"action={m['action']!r}  pred_dst={m['predicted_dst']!r} "
                             f"GT_dst={m['gt_dst']!r}")
                if m.get("reason"):
                    lines.append(f"  - LLM reasoning: {m['reason']}")
                lines.append(f"  - obs: {m['obs']!r}")
        (args.games_md_dir / f"{r['game']}.md").write_text("\n".join(lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
