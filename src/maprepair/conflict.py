"""Conflict types and detectors.

The paper recognises three structural-conflict classes:

  * **Directional conflict** - a single source node has multiple outgoing edges
    with the same direction label, violating spatial uniqueness in deterministic
    environments (e.g. two distinct rooms reachable by `north` from the same
    room).
  * **Topological conflict** - a bidirectional pair where the reverse edge's
    direction does not match the expected opposite (e.g. forward = `down`,
    reverse = `north` instead of `up`). Also includes self-loops and
    overlapping-position conflicts produced by hostile direction sequences.
  * **Naming conflict** - the same node name is used for two physically distinct
    rooms; surfaces as conflicting spatial inferences from different paths.

The detectors operate purely on `NavGraph` state (no LLM input), so they can be
reused for both algorithmic validation and LLM repair loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .graph import DIRECTIONS, NavGraph, OPPOSITE


@dataclass(frozen=True)
class Conflict:
    """A detected structural conflict."""

    type: str                 # 'direction' | 'topology' | 'naming'
    severity: str             # 'high' | 'medium' | 'low'
    description: str
    involved_nodes: Tuple[str, ...]
    involved_edges: Tuple[Tuple[str, str], ...]
    details: Dict[str, str] = field(default_factory=dict)

    def conflict_id(self) -> str:
        edges = sorted(self.involved_edges)
        nodes = sorted(self.involved_nodes)
        return f"{self.type}|{self.description}|{nodes}|{edges}"


def detect_direction_conflicts(graph: NavGraph) -> List[Conflict]:
    """A node with >1 outgoing edge sharing the same direction."""
    out: List[Conflict] = []
    for node in graph.nodes():
        per_dir: Dict[str, List[str]] = {}
        for e in graph.outgoing(node):
            per_dir.setdefault(e.direction, []).append(e.target)
        for direction, targets in per_dir.items():
            if len(targets) > 1:
                out.append(Conflict(
                    type="direction",
                    severity="high",
                    description=f"node {node!r} has multiple outgoing edges labelled {direction!r}",
                    involved_nodes=tuple([node, *targets]),
                    involved_edges=tuple((node, t) for t in targets),
                    details={"direction": direction, "targets": ",".join(targets)},
                ))
    return out


def detect_topology_conflicts(graph: NavGraph) -> List[Conflict]:
    """Reverse-direction mismatch + self-loops + overlapping-position checks."""
    out: List[Conflict] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    for e in graph.primary_edges():
        if (e.target, e.source) in seen_pairs or (e.source, e.target) in seen_pairs:
            continue
        seen_pairs.add((e.source, e.target))

        if e.source == e.target:
            out.append(Conflict(
                type="topology",
                severity="medium",
                description=f"self-loop on {e.source!r}",
                involved_nodes=(e.source,),
                involved_edges=((e.source, e.target),),
                details={"direction": e.direction},
            ))
            continue

        expected_rev = OPPOSITE.get(e.direction)
        if not expected_rev:
            continue
        if graph.has_edge(e.target, e.source):
            actual_rev = graph.direction(e.target, e.source)
            if actual_rev and actual_rev != expected_rev and actual_rev in OPPOSITE:
                out.append(Conflict(
                    type="topology",
                    severity="high",
                    description=(
                        f"reverse direction mismatch: {e.source}--[{e.direction}]-->{e.target} "
                        f"but {e.target}--[{actual_rev}]-->{e.source} (expected {expected_rev})"
                    ),
                    involved_nodes=(e.source, e.target),
                    involved_edges=((e.source, e.target), (e.target, e.source)),
                    details={
                        "forward_direction": e.direction,
                        "actual_reverse": actual_rev,
                        "expected_reverse": expected_rev,
                    },
                ))

    out.extend(_detect_spatial_overlap(graph))
    return out


def detect_naming_conflicts(graph: NavGraph) -> List[Conflict]:
    """Different physical positions assigned to a single node name.

    We infer (x, y, z) coordinates for each node by treating cardinal directions
    as unit moves and walking the graph along shortest paths from an arbitrary
    root. If the same node receives inconsistent coordinates from two different
    paths, that is a naming conflict.
    """
    out: List[Conflict] = []
    pos = _infer_positions(graph)
    # Two nodes with different names but identical position is a separate
    # "overlap" issue handled by detect_topology_conflicts. Two paths giving the
    # same node name different positions is what we capture here.
    for node, candidates in pos.items():
        if len(candidates) > 1:
            sorted_cands = sorted(candidates)
            out.append(Conflict(
                type="naming",
                severity="high",
                description=f"node {node!r} reachable at conflicting positions {sorted_cands}",
                involved_nodes=(node,),
                involved_edges=tuple(),
                details={"positions": str(sorted_cands)},
            ))
    return out


def _detect_spatial_overlap(graph: NavGraph) -> List[Conflict]:
    """Two distinct nodes inferred to occupy the same coordinate."""
    pos = _infer_positions(graph)
    flat: Dict[Tuple[int, int, int], List[str]] = {}
    for node, candidates in pos.items():
        for p in candidates:
            flat.setdefault(p, []).append(node)
    out: List[Conflict] = []
    for p, nodes in flat.items():
        if len(set(nodes)) > 1:
            distinct = sorted(set(nodes))
            out.append(Conflict(
                type="topology",
                severity="high",
                description=f"position {p} occupied by multiple rooms {distinct}",
                involved_nodes=tuple(distinct),
                involved_edges=tuple(),
                details={"position": str(p)},
            ))
    return out


_DIRECTION_VECTORS: Dict[str, Tuple[int, int, int]] = {
    "north":     (0,  1, 0),
    "south":     (0, -1, 0),
    "east":      (1,  0, 0),
    "west":      (-1, 0, 0),
    "northeast": (1,  1, 0),
    "northwest": (-1, 1, 0),
    "southeast": (1, -1, 0),
    "southwest": (-1, -1, 0),
    "up":        (0,  0, 1),
    "down":      (0,  0, -1),
}


def _infer_positions(graph: NavGraph, max_positions_per_node: int = 4) -> Dict[str, Set[Tuple[int, int, int]]]:
    """BFS from each connected component's root using direction vectors.

    Cycles with mismatched directions can in principle generate unbounded
    distinct coordinates for a node; we cap each node's coordinate set to
    ``max_positions_per_node`` to keep this bounded. The cap is large enough to
    surface a conflict (which only needs 2 distinct positions) but small enough
    to keep BFS polynomial.
    """
    positions: Dict[str, Set[Tuple[int, int, int]]] = {}
    visited_roots: Set[str] = set()
    g = graph.nx
    nodes = list(g.nodes())
    if not nodes:
        return positions
    import networkx as nx
    for component in nx.weakly_connected_components(g):
        root = sorted(component)[0]
        if root in visited_roots:
            continue
        visited_roots.add(root)
        frontier = [(root, (0, 0, 0))]
        seen: Dict[str, Set[Tuple[int, int, int]]] = {root: {(0, 0, 0)}}
        while frontier:
            node, p = frontier.pop()
            for _, target, data in g.out_edges(node, data=True):
                direction = data.get("direction", "")
                vec = _DIRECTION_VECTORS.get(direction)
                if vec is None:
                    continue
                np_ = (p[0] + vec[0], p[1] + vec[1], p[2] + vec[2])
                bucket = seen.setdefault(target, set())
                if np_ in bucket:
                    continue
                if len(bucket) >= max_positions_per_node:
                    # Record the conflicting position but stop expanding from it.
                    bucket.add(np_)
                    continue
                bucket.add(np_)
                frontier.append((target, np_))
        for n, pset in seen.items():
            positions.setdefault(n, set()).update(pset)
    return positions


def detect_all(graph: NavGraph) -> List[Conflict]:
    """All conflict detectors combined, deduplicated by conflict_id."""
    seen: Set[str] = set()
    out: List[Conflict] = []
    for c in (*detect_direction_conflicts(graph),
              *detect_topology_conflicts(graph),
              *detect_naming_conflicts(graph)):
        cid = c.conflict_id()
        if cid in seen:
            continue
        seen.add(cid)
        out.append(c)
    return out
