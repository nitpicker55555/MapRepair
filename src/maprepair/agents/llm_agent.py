"""LLM-driven repair agent (configurable per ablation mode).

Mode controls what the LLM sees in each prompt:

  * 'baseline'    : only the conflict descriptor + raw involved edges
  * 'edge_impact' : candidate edges (LCA-filtered) with their impact scores
  * 'vc_only'     : candidate edges + a compact version-history summary
  * 'vc_ei'       : both signals (paper's full method)

The agent loops up to `max_iterations`. Each iteration the LLM emits a JSON
action {modify_edge | rollback_to_version | skip_conflict}. We validate the
action, apply it, re-detect conflicts, and decide whether the target conflict
was resolved.

The agent is *deterministic w.r.t. mode* (same broken graph + same LLM
sample order => same trace) so ablation comparisons are meaningful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..conflict import Conflict, detect_all
from ..graph import DIRECTIONS, NavGraph, OPPOSITE
from ..history import VersionHistory
from ..llm_client import chat_json, message
from ..localizer import Localizer
from ..scoring import score_edges
from .base import RepairAction, RepairAgent, RepairResult


_SCHEMA_MODIFY = (
    '{"action":"modify_edge","edge":["<src>","<dst>"],'
    '"new_direction":"<compass>","reason":"..."}'
)
_SCHEMA_REMOVE = (
    '{"action":"remove_edge","edge":["<src>","<dst>"],"reason":"..."}'
)
_SCHEMA_ROLLBACK = (
    '{"action":"rollback_to_version","version_id":<int>,"reason":"..."}'
)
_SCHEMA_SKIP = (
    '{"action":"skip_conflict","reason":"..."}'
)
_DIRECTIONS_LIST = (
    "north, south, east, west, up, down, in, out, enter, exit, "
    "northeast, northwest, southeast, southwest"
)

SYS = {
    "baseline": (
        "You are a graph repair agent. Resolve one structural conflict at a "
        "time. Only act on edges that appear in 'involved_edges'.\n\n"
        f"Output strict JSON, EXACTLY one of these shapes:\n"
        f"  {_SCHEMA_MODIFY}\n  {_SCHEMA_REMOVE}\n  {_SCHEMA_SKIP}\n\n"
        f"`new_direction` must be one of: {_DIRECTIONS_LIST}.\n"
        "Use remove_edge when an edge looks spurious / contradicts the rest "
        "of the graph; modify_edge when the source-destination pair is real "
        "but the direction is wrong."
    ),
    "edge_impact": (
        "You are a graph repair agent with access to candidate edges ranked by "
        "impact score (higher = more likely the root cause). Prefer acting on "
        "the highest-scored candidate from the 'Candidate edges' block.\n\n"
        f"Output strict JSON, EXACTLY one of these shapes:\n"
        f"  {_SCHEMA_MODIFY}\n  {_SCHEMA_REMOVE}\n  {_SCHEMA_SKIP}\n\n"
        f"`new_direction` must be one of: {_DIRECTIONS_LIST}.\n"
        "Use remove_edge when an edge looks spurious; modify_edge when only "
        "the direction is wrong. Pick the edge from the candidates list."
    ),
    "vc_only": (
        "You are a graph repair agent with access to a version history of how "
        "the graph was constructed. You may modify or remove an involved edge, "
        "or rollback to an earlier version if the conflict stems from a "
        "clearly bad early edit.\n\n"
        f"Output strict JSON, EXACTLY one of these shapes:\n"
        f"  {_SCHEMA_MODIFY}\n  {_SCHEMA_REMOVE}\n  {_SCHEMA_ROLLBACK}\n  {_SCHEMA_SKIP}\n\n"
        f"`new_direction` must be one of: {_DIRECTIONS_LIST}.\n"
        "`version_id` must be one of the integers shown in 'Recent version "
        "history'. Use rollback sparingly — only when modify/remove cannot fix it."
    ),
    "vc_ei": (
        "You are a graph repair agent. You see BOTH impact-ranked candidate "
        "edges AND a version history. Prefer the smallest edit (modify the "
        "highest-scored candidate). Remove the edge if it looks spurious. "
        "Rollback only when the conflict clearly stems from an early bad "
        "edit visible in the history.\n\n"
        f"Output strict JSON, EXACTLY one of these shapes:\n"
        f"  {_SCHEMA_MODIFY}\n  {_SCHEMA_REMOVE}\n  {_SCHEMA_ROLLBACK}\n  {_SCHEMA_SKIP}\n\n"
        f"`new_direction` must be one of: {_DIRECTIONS_LIST}.\n"
        "`version_id` must be one of the integers shown in 'Recent version "
        "history'. Use rollback sparingly."
    ),
}


def _edge_summary(g: NavGraph, edge: Tuple[str, str]) -> str:
    if not g.has_edge(*edge):
        return f"{edge[0]}--[MISSING]-->{edge[1]}"
    return f"{edge[0]}--[{g.direction(*edge)}]-->{edge[1]}"


def _candidate_block(g: NavGraph, scored, k: int = 5) -> str:
    if not scored:
        return "  (no candidates)"
    out = []
    for i, s in enumerate(scored[:k]):
        out.append(f"  [{i+1}] {_edge_summary(g, s.edge)} score={s.score:.3f} "
                   f"(reach={s.reach:.2f}, conflict={s.conflict:.2f}, usage={s.usage:.2f})")
    return "\n".join(out)


def _history_block(history: VersionHistory, k: int = 5) -> str:
    if len(history) == 0:
        return "  (no history)"
    versions = list(history)[-k:]
    return "\n".join(f"  v{v.version_id} step={v.step_num} trigger={v.trigger!r}"
                      for v in versions)


def _build_prompt(conflict: Conflict, graph: NavGraph,
                   scored, history: Optional[VersionHistory],
                   mode: str, repair_log: List[RepairAction]) -> str:
    parts = [
        "Conflict:",
        f"  type={conflict.type} severity={conflict.severity}",
        f"  description={conflict.description}",
        f"  involved_edges={list(conflict.involved_edges)}",
    ]
    if conflict.details:
        parts.append("  details=" + json.dumps(conflict.details))

    if mode in ("edge_impact", "vc_ei") and scored:
        parts.append("\nCandidate edges (ranked by impact score):")
        parts.append(_candidate_block(graph, scored))
    if mode in ("vc_only", "vc_ei") and history is not None:
        parts.append("\nRecent version history:")
        parts.append(_history_block(history))
    if repair_log:
        parts.append("\nRepair attempts in this loop so far:")
        for i, a in enumerate(repair_log[-4:]):
            parts.append(f"  {i+1}. {a.kind} {a.target} -> {a.new_direction or '-'} ({a.reason})")
    parts.append(
        "\nDecide on ONE action. Output strict JSON. new_direction must be a "
        "lowercase compass direction (north, south, east, west, up, down, "
        "northeast, ...). rollback_to_version is only valid when version "
        "history is shown."
    )
    return "\n".join(parts)


class LLMRepairAgent(RepairAgent):
    def __init__(self, *, model: str, mode: str, name: Optional[str] = None,
                 max_attempts_per_conflict: int = 10,
                 lookahead: bool = False,
                 lookahead_retries: int = 3,
                 chat_json_fn=None) -> None:
        """If ``chat_json_fn`` is provided it overrides the default Azure
        client (``maprepair.llm_client.chat_json``). Used for exp25+ to
        dispatch to the openai-hub proxy for non-Azure models
        (gpt-5*, claude-*, gemini-*, o3*, o4*)."""
        if mode not in SYS:
            raise ValueError(f"unknown mode: {mode}")
        self.model = model
        self.mode = mode
        self.lookahead = lookahead
        self.lookahead_retries = lookahead_retries
        suffix = "+lookahead" if lookahead else ""
        self.name = name or f"llm_{mode}{suffix}_{model}"
        self.max_attempts = max_attempts_per_conflict
        self.localizer = Localizer()
        self._chat_json_fn = chat_json_fn  # None = use default Azure client

    # ------------------------------------------------------------------
    def repair(self, graph: NavGraph, *, max_iterations: int = 10,
                max_rollbacks_per_conflict: int = 1) -> RepairResult:
        """Iterative LLM-driven repair.

        Outer loop now picks the *first remaining conflict* dynamically
        each round (a snapshot-based outer loop was bugged: every edit
        reshapes the conflict graph, so most original `conflict_id`s
        stop appearing in `current` and would have been silently
        skipped).

        Oscillation guards:
          - hard cap on `iter_total` across the whole loop (max_iterations);
          - per-conflict cap on rollbacks (default 1);
          - per-conflict memo of (action_kind, target, new_direction)
            signatures: repeating one increments `no_progress`;
          - `no_progress >= 3` aborts the conflict and moves on;
          - global stop after 5 consecutive no-action-applied rounds.
        """
        before = graph.copy()
        work = graph.copy()
        history = VersionHistory()
        history.commit(work, step_num=0, trigger="initial")
        conflicts_before = detect_all(work)
        repair_log: List[RepairAction] = []

        all_actions: List[RepairAction] = []
        iter_total = 0
        global_stalls = 0
        # conflicts we've already given up on (by id) so we don't retry
        abandoned: set = set()
        while iter_total < max_iterations:
            cs = detect_all(work)
            if not cs:
                break
            target = next((c for c in cs if c.conflict_id() not in abandoned), None)
            if target is None:
                # all remaining conflicts have been abandoned
                break
            cid = target.conflict_id()
            tried_signatures: set = set()
            n_rollback = 0
            no_progress = 0
            last_conflict_count = len(cs)
            for attempt in range(self.max_attempts):
                if iter_total >= max_iterations:
                    break
                iter_total += 1
                cs2 = detect_all(work)
                tgt = next((c for c in cs2 if c.conflict_id() == cid), None)
                if tgt is None:
                    break
                cand_edges = self.localizer.localize(work, tgt)
                cand_edges = [e for e in cand_edges if work.has_edge(*e)]
                scored_all = score_edges(work, conflicts=cs2)
                scored = [s for s in scored_all if s.edge in set(cand_edges)] if cand_edges else []
                show_history = (self.mode in ("vc_only", "vc_ei")
                                  and n_rollback < max_rollbacks_per_conflict)
                prompt = _build_prompt(tgt, work, scored,
                                        history if show_history else None,
                                        self.mode, repair_log)
                # ---- Action selection (with optional lookahead) ----
                applied = None
                rejected_actions: List[str] = []
                last_conflict_count_local = len(cs2)
                attempts_for_this_round = self.lookahead_retries if self.lookahead else 1
                for trial in range(attempts_for_this_round):
                    trial_prompt = prompt
                    if rejected_actions:
                        trial_prompt = prompt + (
                            "\n\nThe following actions were tried and did not "
                            "reduce the total conflict count; pick a different "
                            "edge or direction:\n  - " + "\n  - ".join(rejected_actions[-3:])
                        )
                    action = self._call_llm(trial_prompt)
                    if not show_history and action.get("action") == "rollback_to_version":
                        action = {"action": "skip_conflict",
                                   "reason": "rollback budget exhausted"}
                    if not self.lookahead:
                        applied = self._apply_action(action, work, history, iter_total)
                        break
                    # Lookahead: apply to a test graph, accept only on strict decrease
                    test_graph = work.copy()
                    test_history = VersionHistory()
                    for v in list(history):
                        # shallow re-commit so version_ids match (only used for rollback)
                        pass
                    test_applied = self._apply_action(action, test_graph, history, iter_total)
                    if test_applied is None or test_applied.kind in ("skip",):
                        applied = test_applied
                        break
                    new_count = len(detect_all(test_graph))
                    if new_count < last_conflict_count_local:
                        # accept: re-apply to the actual work graph
                        applied = self._apply_action(action, work, history, iter_total)
                        break
                    else:
                        rejected_actions.append(
                            f"{test_applied.kind} {test_applied.target} -> "
                            f"{test_applied.new_direction or '-'} "
                            f"(conflicts {last_conflict_count_local} -> {new_count})"
                        )
                        applied = RepairAction(
                            kind="skip", reason=f"lookahead rejected ({new_count}>={last_conflict_count_local})"
                        )
                # ---- end action selection ----
                sig = (getattr(applied, "kind", "skip"),
                        getattr(applied, "target", None),
                        getattr(applied, "new_direction", None))
                action_applied = applied is not None and applied.kind not in ("skip",)
                if action_applied:
                    if applied.kind == "rollback":
                        n_rollback += 1
                    tried_signatures.add(sig)
                    repair_log.append(applied)
                    all_actions.append(applied)
                    history.commit(work, step_num=iter_total,
                                    trigger=applied.reason or applied.kind)

                new_cs = detect_all(work)
                # Conflict count is the ONLY signal for progress on this conflict
                if len(new_cs) < last_conflict_count:
                    no_progress = 0
                    global_stalls = 0
                    last_conflict_count = len(new_cs)
                else:
                    no_progress += 1
                    if not action_applied:
                        global_stalls += 1

                # If the targeted conflict is resolved, we're done with it
                if not any(c.conflict_id() == cid for c in new_cs):
                    break
                # Repeated action signature: also bumps no_progress
                if sig in tried_signatures and not action_applied:
                    no_progress += 1
                if no_progress >= 3:
                    break  # abandon this conflict
            else:
                # Used all max_attempts without resolving
                abandoned.add(cid)
                continue
            # If we broke without resolving and no_progress hit threshold:
            if any(c.conflict_id() == cid for c in detect_all(work)):
                abandoned.add(cid)
            if global_stalls >= 5:
                break

        return RepairResult(
            agent=self.name,
            graph_before=before,
            graph_after=work,
            conflicts_before=conflicts_before,
            conflicts_after=detect_all(work),
            actions=all_actions,
            iterations=iter_total,
            success=not detect_all(work),
        )

    # ------------------------------------------------------------------
    def _call_llm(self, user_prompt: str) -> Dict[str, Any]:
        try:
            fn = self._chat_json_fn or chat_json
            # 1500 tokens: enough for verbose Gemini / Claude `reason` fields
            # without breaking gpt-4.x (which only used ~80 tokens previously).
            return fn([
                message("system", SYS[self.mode]),
                message("user", user_prompt),
            ], model=self.model, temperature=0.2, max_tokens=1500)
        except Exception as e:
            return {"action": "skip_conflict", "reason": f"llm_error: {e}"}

    def _apply_action(self, action: Dict[str, Any], graph: NavGraph,
                       history: VersionHistory, step: int) -> Optional[RepairAction]:
        kind = action.get("action")
        reason = action.get("reason", "")
        if kind == "modify_edge":
            edge = action.get("edge") or []
            if len(edge) != 2:
                return RepairAction(kind="skip", reason="bad edge spec")
            u, v = edge[0].strip().lower(), edge[1].strip().lower()
            new_dir = (action.get("new_direction") or "").strip().lower()
            if not graph.has_edge(u, v):
                return RepairAction(kind="skip", reason=f"edge missing: {u}->{v}")
            if new_dir not in DIRECTIONS:
                return RepairAction(kind="skip", reason=f"bad direction: {new_dir}")
            graph.set_direction(u, v, new_dir, sync_reverse=True)
            return RepairAction(kind="modify_edge", target=(u, v),
                                 new_direction=new_dir, reason=reason)
        if kind == "remove_edge":
            edge = action.get("edge") or []
            if len(edge) != 2:
                return RepairAction(kind="skip", reason="bad edge spec")
            u, v = edge[0].strip().lower(), edge[1].strip().lower()
            if not graph.has_edge(u, v):
                return RepairAction(kind="skip", reason=f"edge missing: {u}->{v}")
            graph.remove_edge(u, v)
            # Drop the auto-reverse twin too (consistent with HeuristicRepairAgent)
            if graph.has_edge(v, u) and graph.nx[v][u].get("is_auto_reverse"):
                graph.remove_edge(v, u)
            return RepairAction(kind="remove_edge", target=(u, v),
                                 reason=reason)
        if kind == "rollback_to_version":
            vid = action.get("version_id")
            if not isinstance(vid, int):
                return RepairAction(kind="skip", reason="bad version id")
            rolled = history.rollback_to(vid)
            if rolled is None:
                return RepairAction(kind="skip", reason=f"version missing: {vid}")
            # Replace the graph contents in place.
            graph._g.clear()
            graph._g.update(rolled._g)
            return RepairAction(kind="rollback", rolled_back_to=vid, reason=reason)
        return RepairAction(kind="skip", reason=reason or "skip")
