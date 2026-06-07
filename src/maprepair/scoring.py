"""Edge impact scoring.

Formal definition from the paper:

    score(e) = Reach_hat(e) + Conflict_hat(e) + Usage_hat(e)

where each component is min-max normalized. Concretely:

  * **Reach**: number of nodes downstream of e's target (i.e. nodes that become
    unreachable from the rest of the graph if e is severed). Approximated by
    `len(nx.descendants(g - e, target))`.
  * **Conflict**: number of currently-detected conflicts in which e participates.
  * **Usage**: how often e appears on observed walkthrough paths (or, if no
    walkthrough is given, how many shortest paths between random node pairs go
    through e — i.e. edge betweenness).

The output ranks edges by descending total score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .conflict import Conflict
from .graph import NavGraph


@dataclass(frozen=True)
class ScoredEdge:
    edge: Tuple[str, str]
    score: float
    reach: float
    conflict: float
    usage: float


def _min_max(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mn = min(values); mx = max(values)
    if mx <= mn:
        return [0.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


def reach_components(graph: NavGraph) -> Dict[Tuple[str, str], float]:
    g = graph.nx
    out: Dict[Tuple[str, str], float] = {}
    for (u, v) in [(u, v) for u, v in g.edges() if not g[u][v].get("is_auto_reverse")]:
        # how many nodes does v reach (excluding u and the reverse edge)?
        try:
            descendants = nx.descendants(g, v)
        except nx.NetworkXError:
            descendants = set()
        out[(u, v)] = float(len(descendants))
    return out


def conflict_components(graph: NavGraph,
                         conflicts: Iterable[Conflict]) -> Dict[Tuple[str, str], float]:
    counts: Dict[Tuple[str, str], int] = {(u, v): 0
                                          for u, v in graph.nx.edges()
                                          if not graph.nx[u][v].get("is_auto_reverse")}
    for c in conflicts:
        for e in c.involved_edges:
            if e in counts:
                counts[e] += 1
    return {k: float(v) for k, v in counts.items()}


def usage_components(graph: NavGraph,
                     walkthrough_edges: Optional[Sequence[Tuple[str, str]]] = None
                     ) -> Dict[Tuple[str, str], float]:
    g = graph.nx
    primary = [(u, v) for u, v in g.edges() if not g[u][v].get("is_auto_reverse")]
    out: Dict[Tuple[str, str], float] = {e: 0.0 for e in primary}
    if walkthrough_edges:
        for e in walkthrough_edges:
            if e in out:
                out[e] += 1.0
        return out
    # Fall back to edge betweenness on the undirected projection.
    if g.number_of_edges() == 0:
        return out
    try:
        bet = nx.edge_betweenness_centrality(g.to_undirected(), normalized=True)
    except Exception:
        return out
    for (u, v), val in bet.items():
        if (u, v) in out:
            out[(u, v)] = float(val)
        elif (v, u) in out:
            out[(v, u)] = float(val)
    return out


def score_edges(
    graph: NavGraph,
    conflicts: Optional[Iterable[Conflict]] = None,
    walkthrough_edges: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[ScoredEdge]:
    """Return scored primary edges in descending order of total score."""
    reach = reach_components(graph)
    conflict = conflict_components(graph, conflicts or [])
    usage = usage_components(graph, walkthrough_edges)
    edges = list(reach.keys())
    if not edges:
        return []
    reach_arr = _min_max([reach[e] for e in edges])
    conflict_arr = _min_max([conflict.get(e, 0.0) for e in edges])
    usage_arr = _min_max([usage.get(e, 0.0) for e in edges])
    scored: List[ScoredEdge] = []
    for i, e in enumerate(edges):
        s = reach_arr[i] + conflict_arr[i] + usage_arr[i]
        scored.append(ScoredEdge(
            edge=e,
            score=float(s),
            reach=reach_arr[i],
            conflict=conflict_arr[i],
            usage=usage_arr[i],
        ))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored
