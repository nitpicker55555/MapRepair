"""Incremental LLM-driven mapping from MANGO walkthroughs.

For each step in a walkthrough the LLM decides whether the player moved to a
new location; if yes it emits (src, dst, direction, seen_in_forward). We
accumulate these into a NavGraph (with auto-reverse) and return it.

This is the "exp1" construction stage; the result is a *broken* graph that
Experiment 5 then feeds into the repair agents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .graph import DIRECTIONS, NavGraph
from .llm_client import chat_json, message
from .mango import WalkthroughStep, parse_walkthrough


SYS_PROMPT = (
    "You are a text-adventure map analyst. You analyze one step at a time and "
    "decide whether the player just moved to a new location. Only output strict "
    "JSON, no commentary. Be conservative: only emit a movement when the "
    "observation clearly shows transition into a new room (typically following "
    "one of the cardinal direction actions). 'open <obj>', 'take', 'examine', "
    "'look', 'wait' are NOT movement actions."
)


def _user_prompt(step: WalkthroughStep, prev: List[WalkthroughStep],
                  current_location: Optional[str], locations: List[str],
                  actions: List[str]) -> str:
    ctx = "Recent steps (most recent last):\n" if prev else ""
    for p in prev:
        ctx += f"  step {p.step_num}: action={p.action!r}; obs={p.observation[:120]!r}\n"
    locs = ", ".join(locations[:60]) if locations else "<unknown>"
    return f"""{ctx}Known location list (lowercased): {locs}
Known actions: {', '.join(actions) if actions else '<unknown>'}
Current location BEFORE this step: {current_location or '<unknown>'}

ANALYZE THIS STEP:
  step_num: {step.step_num}
  action: {step.action!r}
  observation: {step.observation!r}

Output strict JSON:
{{
  "is_movement": <bool>,
  "src_node": <string|null>,
  "dst_node": <string|null>,
  "action": <string|null>,        // canonical direction word if movement
  "seen_in_forward": <int|null>,
  "current_location": <string>,    // player position AFTER this step
  "reasoning": <short string>
}}

Canonical directions: {', '.join(DIRECTIONS)}.
Match location names to the known list when possible (lowercase). Never invent
room names unsupported by the observation."""


def build_graph_from_walkthrough(
    walkthrough_text: str,
    locations: List[str],
    actions: List[str],
    *,
    model: str = "gpt-4.1-mini",
    max_steps: int = 70,
    context_size: int = 2,
) -> NavGraph:
    steps = parse_walkthrough(walkthrough_text, max_steps=max_steps)
    g = NavGraph()
    current_location: Optional[str] = None
    for i, step in enumerate(steps):
        prev = steps[max(0, i - context_size): i]
        try:
            resp = chat_json(
                [message("system", SYS_PROMPT),
                 message("user", _user_prompt(step, prev, current_location, locations, actions))],
                model=model, temperature=0.0, max_tokens=512,
            )
        except Exception as e:
            continue
        if resp.get("is_movement"):
            src = (resp.get("src_node") or current_location or "").strip().lower()
            dst = (resp.get("dst_node") or "").strip().lower()
            action = (resp.get("action") or step.action).strip().lower()
            if src and dst and src != dst and action in DIRECTIONS:
                if not g.has_edge(src, dst):
                    try:
                        g.add_edge(src, dst, action, step_num=step.step_num,
                                    add_auto_reverse=True)
                    except Exception:
                        pass
                current_location = dst
        else:
            new_loc = (resp.get("current_location") or current_location or "").strip().lower()
            if new_loc:
                current_location = new_loc
    return g


# ----------------------------------------------------------------------
# Bridge for previously-generated edges
# ----------------------------------------------------------------------

def load_legacy_edges(path: Path | str) -> NavGraph:
    """Load the older `{game}_edges.json` files produced by my earlier mapping
    runs into a clean NavGraph. Edges are added with auto-reverse so the
    resulting graph matches the original repair-experiment inputs.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    g = NavGraph()
    for e in data:
        src = (e.get("src_node") or "").strip().lower()
        dst = (e.get("dst_node") or "").strip().lower()
        action = (e.get("action") or "").strip().lower()
        if not src or not dst or src == dst:
            continue
        if src in {"<unknown>", "unknown"} or dst in {"<unknown>", "unknown"}:
            continue
        if action not in DIRECTIONS:
            continue
        if not g.has_edge(src, dst):
            try:
                g.add_edge(src, dst, action,
                            step_num=int(e.get("seen_in_forward", 0) or 0),
                            add_auto_reverse=True)
            except Exception:
                continue
    return g
