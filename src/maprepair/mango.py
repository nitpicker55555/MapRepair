"""MANGO dataset loader.

Resolves walkthroughs and ground-truth edges from the original maprepair
checkout. We deliberately keep the dataset *outside* this repo (it's hundreds
of MB) and resolve it via env var ``MAPREPAIR_DATA_ROOT`` or a sensible
default.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .graph import NavGraph


@dataclass
class WalkthroughStep:
    step_num: int
    action: str
    observation: str


def data_root() -> Path:
    p = os.environ.get("MAPREPAIR_DATA_ROOT")
    if p:
        return Path(p)
    # default: assume the original maprepair checkout sits next to this one
    return Path("/Users/puzhen/Downloads/maprepair")


def list_games(refined: bool = True) -> List[str]:
    root = data_root()
    fixed_root = root / ("data_fixed" if refined else "data")
    if not fixed_root.exists():
        return []
    return sorted(d.name for d in fixed_root.iterdir()
                  if d.is_dir() and not d.name.startswith("."))


def load_ground_truth_edges(game: str) -> List[Dict]:
    p = data_root() / "data_fixed" / game / f"{game}.edges.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def load_repaired_walkthrough(game: str) -> Optional[str]:
    p = data_root() / "walkthrough_repair_pipeline" / "output" / \
        "repaired_walkthroughs" / game / f"{game}.walkthrough"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def parse_walkthrough(text: str, max_steps: int = 70) -> List[WalkthroughStep]:
    steps: List[WalkthroughStep] = []
    blocks = [b.strip() for b in text.split("===========") if b.strip()]
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
        if step_num is not None and action is not None:
            steps.append(WalkthroughStep(step_num, action, "\n".join(obs_lines).strip()))
        if step_num is not None and step_num >= max_steps:
            break
    return [s for s in steps if s.step_num <= max_steps]


def ground_truth_graph(game: str) -> NavGraph:
    edges = load_ground_truth_edges(game)
    g = NavGraph()
    for e in edges:
        src = (e.get("src_node") or "").strip().lower()
        dst = (e.get("dst_node") or "").strip().lower()
        action = (e.get("action") or "").strip().lower()
        if not src or not dst or src == dst:
            continue
        if g.has_edge(src, dst):
            continue
        try:
            g.add_edge(src, dst, action, step_num=int(e.get("seen_in_forward", 0) or 0),
                        add_auto_reverse=False)
        except Exception:
            continue
    return g
