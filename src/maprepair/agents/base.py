"""Repair agent abstraction.

A `RepairAgent` is anything that, given a broken `NavGraph`, mutates it toward
a conflict-free state. The harness records (agent_name, before -> after,
actions, success) and computes recovery metrics.

Repair history is intentionally kept on the agent: this lets the LLM agent
maintain context across iterations without polluting the broken graph
representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..conflict import Conflict
from ..graph import NavGraph


@dataclass
class RepairAction:
    """One concrete edit performed during a repair loop."""

    kind: str                       # 'modify_edge' | 'remove_edge' | 'rollback' | 'skip'
    target: Optional[Tuple[str, str]] = None
    new_direction: Optional[str] = None
    rolled_back_to: Optional[int] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "new_direction": self.new_direction,
            "rolled_back_to": self.rolled_back_to,
            "reason": self.reason,
        }


@dataclass
class RepairResult:
    """Outcome of a single repair attempt."""

    agent: str
    graph_before: NavGraph
    graph_after: NavGraph
    conflicts_before: List[Conflict]
    conflicts_after: List[Conflict]
    actions: List[RepairAction] = field(default_factory=list)
    iterations: int = 0
    success: bool = False  # True iff conflicts_after == []

    def summary(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "iterations": self.iterations,
            "success": self.success,
            "conflicts_before": len(self.conflicts_before),
            "conflicts_after": len(self.conflicts_after),
            "actions": [a.to_dict() for a in self.actions],
        }


class RepairAgent:
    """Interface every repair agent implements."""

    name: str = "abstract"

    def repair(self, graph: NavGraph, *, max_iterations: int = 10) -> RepairResult:  # noqa: D401
        raise NotImplementedError
