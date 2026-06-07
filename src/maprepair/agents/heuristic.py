"""Non-LLM heuristic repair agent.

This is the standalone algorithmic validation of the paper's claims. The
agent has no LLM call. Its policy is:

  1. Detect all conflicts.
  2. Use the Localizer to obtain a candidate-edge set.
  3. Use the EdgeImpactScorer to rank the candidates.
  4. Inspect the top-ranked candidate and pick the *cheapest* repair that
     removes the conflict it participates in. Choices, in priority order:
       a. If the candidate's reverse direction matches an existing reverse
          edge's expected direction (i.e. flipping its direction restores
          symmetry), flip the candidate's direction.
       b. Otherwise, swap the candidate to an unused compass direction at
          the same source.
       c. If none of the above resolves the local conflict, remove the
          candidate edge entirely.
  5. Re-detect conflicts; if any remain, loop.

The agent keeps a small memory of "recently touched edges" to avoid undoing
its own work.

If the agent succeeds in producing a conflict-free graph, that is direct
evidence that the algorithmic contribution (LCA + scoring) is sufficient to
recover correctness without any LLM in the loop.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from ..conflict import detect_all, Conflict
from ..graph import DIRECTIONS, NavGraph, OPPOSITE
from ..localizer import Localizer
from ..scoring import score_edges
from .base import RepairAction, RepairAgent, RepairResult


class HeuristicRepairAgent(RepairAgent):
    name = "heuristic"

    def __init__(self, *, prefer_remove: bool = False) -> None:
        """If ``prefer_remove`` is True, the agent will try removing an edge
        before rotating its direction. Useful on graphs with many
        hallucinated edges (e.g. LLM-generated MANGO maps), where the
        "right" fix is often to delete an edge that should not exist.
        """
        self.localizer = Localizer()
        self.prefer_remove = prefer_remove
        if prefer_remove:
            self.name = "heuristic_remove"

    # ------------------------------------------------------------------
    def repair(self, graph: NavGraph, *, max_iterations: int = 10) -> RepairResult:
        before = graph.copy()
        work = graph.copy()
        actions: List[RepairAction] = []
        conflicts_before = detect_all(work)
        iter_ = 0
        touched_recently: List[Tuple[str, str]] = []

        while iter_ < max_iterations:
            conflicts = detect_all(work)
            if not conflicts:
                break

            # 1. localize -> candidate set
            candidates_per_conflict = [
                (c, [(u, v) for (u, v) in self.localizer.localize(work, c)
                     if work.has_edge(u, v)])
                for c in conflicts
            ]
            primary = {(e.source, e.target) for e in work.primary_edges()}
            cand_set = {
                e for _c, edges in candidates_per_conflict for e in edges if e in primary
            }
            if not cand_set:
                actions.append(RepairAction(kind="skip", reason="no candidates"))
                break

            # 2. rank by impact score
            scored = score_edges(work, conflicts=conflicts)
            cand_scored = [s for s in scored if s.edge in cand_set]
            if not cand_scored:
                actions.append(RepairAction(kind="skip", reason="no scored candidates"))
                break

            # 3. greedy fix: try the highest-impact candidate first
            done = False
            for s in cand_scored:
                edge = s.edge
                if edge in touched_recently[-3:]:
                    continue
                act = self._try_fix(work, edge, conflicts)
                if act is not None:
                    actions.append(act)
                    touched_recently.append(edge)
                    done = True
                    break

            if not done:
                actions.append(RepairAction(kind="skip", reason="exhausted candidates"))
                break
            iter_ += 1

        conflicts_after = detect_all(work)
        return RepairResult(
            agent=self.name,
            graph_before=before,
            graph_after=work,
            conflicts_before=conflicts_before,
            conflicts_after=conflicts_after,
            actions=actions,
            iterations=iter_,
            success=not conflicts_after,
        )

    # ------------------------------------------------------------------
    def _try_fix(self, graph: NavGraph, edge: Tuple[str, str],
                  conflicts: List[Conflict]) -> Optional[RepairAction]:
        """Attempt the cheapest fix that strictly reduces conflict count.

        We enumerate candidate actions in this priority order:
          1. flip direction to match an existing reverse (topology fix)
          2. rotate to each free compass direction (direction fix)
          3. flip direction to opposite then remove the auto-reverse
          4. remove the edge entirely (with its auto-reverse)
        After each action we re-detect conflicts. If the new conflict count
        is *less than* before, we keep the action; otherwise we revert and
        try the next option. This greedy lookahead prevents introducing new
        position overlaps via direction rotation.
        """
        u, v = edge
        if not graph.has_edge(u, v):
            return None
        current_dir = graph.direction(u, v)
        if current_dir is None:
            return None

        before_count = len(conflicts)
        original_state = graph.copy()

        candidates: List[RepairAction] = []
        my_conflicts = [c for c in conflicts if edge in c.involved_edges]

        # 1. Topology-aware fix (match existing reverse)
        for c in my_conflicts:
            if c.type == "topology" and "reverse direction mismatch" in c.description:
                actual_rev = c.details.get("actual_reverse")
                if actual_rev and actual_rev in OPPOSITE:
                    new_dir = OPPOSITE[actual_rev]
                    if new_dir != current_dir:
                        candidates.append(RepairAction(
                            kind="modify_edge", target=edge,
                            new_direction=new_dir,
                            reason=f"sync to existing reverse {actual_rev}"))

        # 2. Rotate to each currently-free direction at the source
        used = {e.direction for e in graph.outgoing(u) if (e.source, e.target) != edge}
        for d in DIRECTIONS:
            if d in used or d == current_dir:
                continue
            candidates.append(RepairAction(
                kind="modify_edge", target=edge,
                new_direction=d,
                reason=f"rotate to free direction {d}"))

        # 3. Remove fallback (promoted to position 1 if prefer_remove is set).
        remove_action = RepairAction(kind="remove_edge", target=edge,
                                       reason="remove fallback")
        if self.prefer_remove:
            candidates.insert(0, remove_action)
        else:
            candidates.append(remove_action)

        # Try candidates in order, accept the first that strictly reduces conflicts.
        for cand in candidates:
            test = original_state.copy()
            if cand.kind == "modify_edge":
                test.set_direction(u, v, cand.new_direction, sync_reverse=True)
            elif cand.kind == "remove_edge":
                test.remove_edge(u, v)
                if test.has_edge(v, u) and test.nx[v][u].get("is_auto_reverse"):
                    test.remove_edge(v, u)
            new_count = len(detect_all(test))
            if new_count < before_count:
                # apply the action to the real graph
                if cand.kind == "modify_edge":
                    graph.set_direction(u, v, cand.new_direction, sync_reverse=True)
                elif cand.kind == "remove_edge":
                    graph.remove_edge(u, v)
                    if graph.has_edge(v, u) and graph.nx[v][u].get("is_auto_reverse"):
                        graph.remove_edge(v, u)
                return cand
        return None  # no candidate improves things
