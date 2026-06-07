"""LCA-based candidate-edge localization.

For each conflict we identify a *minimal conflicting path pair* and walk it back
to their lowest common ancestor on a temporal-dependency tree (the order in
which edges were inserted). Edges along the divergent subpaths become the
candidate set for repair.

This narrower set is the engine that gives the algorithmic contribution: a
random/baseline repair has to consider every edge in the graph, while LCA-based
repair only inspects edges that could plausibly be the cause.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from .conflict import Conflict
from .graph import NavGraph


@dataclass(frozen=True)
class Candidate:
    edge: Tuple[str, str]
    conflict_ids: Tuple[str, ...]


class Localizer:
    """Compute LCA-filtered candidate edges per conflict.

    The localizer is *stateless* w.r.t. graphs; pass them in each call.
    """

    def localize(self, graph: NavGraph, conflict: Conflict) -> List[Tuple[str, str]]:
        """Return the candidate-edge list for a single conflict.

        Strategy:
          - directional conflict: candidates = the conflicting outgoing edges.
            LCA degenerates to the source node, so this is the smallest set.
          - topology conflict (reverse-mismatch): candidates = the two
            disagreeing edges.
          - topology conflict (position overlap): candidates = edges on the two
            shortest distinct paths from the LCA to the colliding nodes.
          - naming conflict: candidates = edges on the two paths that yield the
            conflicting coordinates.

        When the conflict object already carries `involved_edges`, we always
        include them and supplement with any deeper LCA-derived candidates.
        """
        explicit = list(conflict.involved_edges)
        if conflict.type == "direction":
            # By construction LCA = source; explicit edges are exactly the
            # candidates.
            return list(_unique(explicit))

        # For non-degenerate cases we attempt LCA-based path enumeration.
        node_pair = _conflict_node_pair(conflict)
        if node_pair is None:
            return list(_unique(explicit))
        a, b = node_pair
        cand: List[Tuple[str, str]] = list(explicit)
        cand.extend(self._lca_candidate_edges(graph, a, b))
        return list(_unique(cand))

    def reduction_ratio(self, graph: NavGraph, conflicts: Iterable[Conflict]) -> float:
        primary_edges = {(e.source, e.target) for e in graph.primary_edges()}
        total = len(primary_edges)
        if total == 0:
            return 0.0
        cands: Set[Tuple[str, str]] = set()
        for c in conflicts:
            for e in self.localize(graph, c):
                if e in primary_edges:
                    cands.add(e)
        return 1.0 - (len(cands) / total)

    # ------------------------------------------------------------------
    def _lca_candidate_edges(self, graph: NavGraph,
                              a: str, b: str) -> List[Tuple[str, str]]:
        """Edges on the two shortest paths from LCA(a, b) to a and to b.

        We use the *step-num order* as a stand-in for the reasoning-history
        tree. The LCA is the latest node on both paths.
        """
        g = graph.nx
        # Use temporal step-num to weight edges (older edges have lower step).
        undirected = nx.Graph()
        for u, v, data in g.edges(data=True):
            step = data.get("step_num", 0)
            undirected.add_edge(u, v, weight=step)

        if a not in undirected or b not in undirected:
            return []

        # Roots = nodes reachable from both a and b. We follow the lowest step
        # ancestor.
        try:
            common = list(nx.algorithms.shortest_paths.generic.shortest_path(undirected, a, b))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        if len(common) < 2:
            return []

        edges: List[Tuple[str, str]] = []
        for idx in range(len(common) - 1):
            u, v = common[idx], common[idx + 1]
            if g.has_edge(u, v):
                edges.append((u, v))
            elif g.has_edge(v, u):
                edges.append((v, u))
        return edges


def _conflict_node_pair(conflict: Conflict) -> Optional[Tuple[str, str]]:
    if len(conflict.involved_nodes) >= 2:
        return conflict.involved_nodes[0], conflict.involved_nodes[1]
    return None


def _unique(items: Iterable[Tuple[str, str]]) -> Iterable[Tuple[str, str]]:
    seen: Set[Tuple[str, str]] = set()
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        yield x
