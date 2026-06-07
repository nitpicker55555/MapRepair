"""Versioned graph history.

Each commit captures:

  * `step_num` (the walkthrough step that triggered the change),
  * the *snapshot* of the graph after the change,
  * `changes` (a structured dict describing what changed),
  * `trigger` (a short description, e.g. an observation or repair-reason),
  * the list of conflicts detected immediately after the commit.

Supported operations:

  * `commit(...)` - add a new version
  * `rollback_to(version_id)` - return the snapshot of an earlier version
  * `diff(a, b)` - structural difference between two versions
  * `lookup_step(step_num)` - find the version whose step_num matches

The store is in-memory; persistence is the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .conflict import Conflict
from .graph import NavGraph


@dataclass
class Version:
    version_id: int
    step_num: int
    snapshot: NavGraph
    changes: Dict[str, Any]
    trigger: str
    conflicts: Tuple[Conflict, ...] = field(default_factory=tuple)


class VersionHistory:
    def __init__(self) -> None:
        self._versions: List[Version] = []

    # ------------------------------------------------------------------
    def commit(
        self,
        graph: NavGraph,
        *,
        step_num: int,
        changes: Optional[Dict[str, Any]] = None,
        trigger: str = "",
        conflicts: Optional[List[Conflict]] = None,
    ) -> int:
        v = Version(
            version_id=len(self._versions),
            step_num=step_num,
            snapshot=graph.copy(),
            changes=dict(changes or {}),
            trigger=trigger,
            conflicts=tuple(conflicts or []),
        )
        self._versions.append(v)
        return v.version_id

    # ------------------------------------------------------------------
    def rollback_to(self, version_id: int) -> Optional[NavGraph]:
        if not (0 <= version_id < len(self._versions)):
            return None
        return self._versions[version_id].snapshot.copy()

    def latest(self) -> Optional[Version]:
        return self._versions[-1] if self._versions else None

    def get(self, version_id: int) -> Optional[Version]:
        if 0 <= version_id < len(self._versions):
            return self._versions[version_id]
        return None

    def __len__(self) -> int:
        return len(self._versions)

    def __iter__(self):
        return iter(self._versions)

    # ------------------------------------------------------------------
    def diff(self, a: int, b: int) -> Dict[str, Any]:
        va = self.get(a); vb = self.get(b)
        if va is None or vb is None:
            return {}
        ga = va.snapshot.nx; gb = vb.snapshot.nx
        return {
            "added_nodes": sorted(set(gb.nodes()) - set(ga.nodes())),
            "removed_nodes": sorted(set(ga.nodes()) - set(gb.nodes())),
            "added_edges": sorted(set(gb.edges()) - set(ga.edges())),
            "removed_edges": sorted(set(ga.edges()) - set(gb.edges())),
            "version_range": (a, b),
            "step_range": (va.step_num, vb.step_num),
        }

    def lookup_step(self, step_num: int) -> Optional[int]:
        if not self._versions:
            return None
        # exact match preferred
        for v in self._versions:
            if v.step_num == step_num:
                return v.version_id
        # otherwise return the closest
        return min(self._versions, key=lambda v: abs(v.step_num - step_num)).version_id

    def to_dict(self) -> List[Dict[str, Any]]:
        return [
            {
                "version_id": v.version_id,
                "step_num": v.step_num,
                "trigger": v.trigger,
                "changes": v.changes,
                "num_nodes": v.snapshot.num_nodes(),
                "num_edges": v.snapshot.num_edges(),
                "conflicts": len(v.conflicts),
            }
            for v in self._versions
        ]
