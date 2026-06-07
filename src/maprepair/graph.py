"""NavGraph: a directed multigraph with direction-labelled edges.

Matches the structure of the LLM-MapRepair paper:
  - Nodes are room/location names (lowercased strings).
  - Edges carry a compass `direction` label (north, south, east, west, up, down,
    in, out, enter, exit, northeast, ...) and a `step_num` (when the edge was
    introduced).
  - Edges are added with a forward direction. The graph also tracks an optional
    automatically-derived reverse edge.

We use a thin wrapper around `networkx.DiGraph` so other modules (localizer,
scorer, repair agents) can rely on familiar NX algorithms while still going
through a single API for mutations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import networkx as nx

# Canonical direction set we treat as valid spatial transitions.
DIRECTIONS: Tuple[str, ...] = (
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
    "up", "down", "in", "out", "enter", "exit",
)

OPPOSITE: Dict[str, str] = {
    "north": "south", "south": "north",
    "east": "west", "west": "east",
    "northeast": "southwest", "southwest": "northeast",
    "northwest": "southeast", "southeast": "northwest",
    "up": "down", "down": "up",
    "in": "out", "out": "in",
    "enter": "exit", "exit": "enter",
}

ORTHOGONAL: Dict[str, Tuple[str, ...]] = {
    "north": ("east", "west"),
    "south": ("east", "west"),
    "east": ("north", "south"),
    "west": ("north", "south"),
}


@dataclass(frozen=True)
class Edge:
    """A directed edge with its compass label."""

    source: str
    target: str
    direction: str
    step_num: int = 0
    is_auto_reverse: bool = False

    def reversed(self) -> "Edge":
        rev = OPPOSITE.get(self.direction)
        if rev is None:
            raise ValueError(f"Direction '{self.direction}' has no defined opposite")
        return Edge(
            source=self.target,
            target=self.source,
            direction=rev,
            step_num=self.step_num,
            is_auto_reverse=True,
        )


class NavGraph:
    """Mutable, version-aware navigation graph."""

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_edges(
        cls,
        edges: Iterable[Tuple[str, str, str]] | Iterable[Edge],
        *,
        add_auto_reverse: bool = True,
    ) -> "NavGraph":
        g = cls()
        for e in edges:
            if isinstance(e, Edge):
                g.add_edge(e.source, e.target, e.direction,
                            step_num=e.step_num,
                            add_auto_reverse=add_auto_reverse and not e.is_auto_reverse)
            else:
                src, dst, direction = e
                g.add_edge(src, dst, direction, add_auto_reverse=add_auto_reverse)
        return g

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_edge(
        self,
        src: str,
        dst: str,
        direction: str,
        *,
        step_num: Optional[int] = None,
        add_auto_reverse: bool = True,
        is_auto_reverse: bool = False,
    ) -> Edge:
        src = src.strip().lower()
        dst = dst.strip().lower()
        direction = direction.strip().lower()
        if src == dst:
            raise ValueError(f"Self-loops are not allowed: {src!r}")
        if step_num is None:
            self._step_counter += 1
            step_num = self._step_counter
        self._g.add_edge(
            src, dst,
            direction=direction,
            step_num=step_num,
            is_auto_reverse=is_auto_reverse,
        )
        edge = Edge(src, dst, direction, step_num, is_auto_reverse)
        if add_auto_reverse and direction in OPPOSITE and not self._g.has_edge(dst, src):
            rev_dir = OPPOSITE[direction]
            self._g.add_edge(
                dst, src,
                direction=rev_dir,
                step_num=step_num,
                is_auto_reverse=True,
            )
        return edge

    def remove_edge(self, src: str, dst: str) -> None:
        src = src.strip().lower(); dst = dst.strip().lower()
        if self._g.has_edge(src, dst):
            self._g.remove_edge(src, dst)

    def set_direction(self, src: str, dst: str, new_direction: str,
                       *, sync_reverse: bool = True) -> None:
        src = src.strip().lower(); dst = dst.strip().lower()
        new_direction = new_direction.strip().lower()
        if not self._g.has_edge(src, dst):
            raise KeyError(f"No edge {src!r} -> {dst!r}")
        self._g[src][dst]["direction"] = new_direction
        if sync_reverse and self._g.has_edge(dst, src) and self._g[dst][src].get("is_auto_reverse"):
            rev = OPPOSITE.get(new_direction)
            if rev:
                self._g[dst][src]["direction"] = rev

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    @property
    def nx(self) -> nx.DiGraph:
        return self._g

    def has_edge(self, src: str, dst: str) -> bool:
        return self._g.has_edge(src.strip().lower(), dst.strip().lower())

    def direction(self, src: str, dst: str) -> Optional[str]:
        if not self.has_edge(src, dst):
            return None
        return self._g[src.strip().lower()][dst.strip().lower()].get("direction")

    def step_num(self, src: str, dst: str) -> Optional[int]:
        if not self.has_edge(src, dst):
            return None
        return self._g[src.strip().lower()][dst.strip().lower()].get("step_num")

    def edges(self) -> List[Edge]:
        out: List[Edge] = []
        for u, v, d in self._g.edges(data=True):
            out.append(Edge(u, v, d.get("direction", ""), d.get("step_num", 0),
                            d.get("is_auto_reverse", False)))
        return out

    def primary_edges(self) -> List[Edge]:
        """All edges that are NOT auto-reverse (i.e. originally added)."""
        return [e for e in self.edges() if not e.is_auto_reverse]

    def nodes(self) -> List[str]:
        return list(self._g.nodes())

    def outgoing(self, node: str) -> List[Edge]:
        node = node.strip().lower()
        return [Edge(node, t, d.get("direction", ""), d.get("step_num", 0),
                     d.get("is_auto_reverse", False))
                for _, t, d in self._g.out_edges(node, data=True)]

    def num_nodes(self) -> int:
        return self._g.number_of_nodes()

    def num_edges(self) -> int:
        return self._g.number_of_edges()

    def copy(self) -> "NavGraph":
        new = NavGraph()
        new._g = copy.deepcopy(self._g)
        new._step_counter = self._step_counter
        return new

    def __repr__(self) -> str:
        return (f"NavGraph(nodes={self.num_nodes()}, edges={self.num_edges()}, "
                f"primary={len(self.primary_edges())})")
