"""
Edge Impact Scorer

Implements the Edge Impact Score (EIS) defined in the paper, Section 2.3,
Equation 1:

    score(e) = R_hat(e) + C_hat(e) + U_hat(e)

where each of the three factors is min-max normalized to [0, 1] across all
candidate edges:

    Reachability R(e):    number of nodes downstream-reachable from e (i.e.
                          |descendants(target(e))| + 1).
    Conflict count C(e):  number of distinct conflicts in which e is involved.
    Usage U(e):           number of conflict-related paths (or walkthrough
                          traversals) that include e.

Inspired by PageRank, the score is an unweighted sum of three normalized
factors, deliberately avoiding hand-tuned weights.
"""

import networkx as nx
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict


@dataclass
class EdgeScore:
    """Per-edge scoring information.

    Attributes mirror the paper notation:
        reach:           raw reachability count R(e)
        conflict_count:  raw conflict count C(e)
        usage:           raw usage count U(e)
        r_hat / c_hat / u_hat: min-max normalized values in [0, 1]
        score:           R_hat + C_hat + U_hat (final EIS)
    """
    edge: Tuple[str, str]
    reach: int = 0
    conflict_count: int = 0
    usage: int = 0
    r_hat: float = 0.0
    c_hat: float = 0.0
    u_hat: float = 0.0
    score: float = 0.0


class EdgeImpactScorer:
    """Computes the Edge Impact Score for a set of candidate edges.

    Usage:
        scorer = EdgeImpactScorer()
        scores = scorer.score_edges(graph, conflicts, walkthrough_path,
                                    candidate_edges=cand)
        ranked = scorer.rank()
    """

    def __init__(self):
        self.edge_scores: Dict[Tuple[str, str], EdgeScore] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def score_edges(self,
                    graph: nx.DiGraph,
                    conflicts: List,
                    walkthrough_path: List[Tuple[str, str]],
                    candidate_edges: List[Tuple[str, str]] = None
                    ) -> Dict[Tuple[str, str], EdgeScore]:
        """Score the supplied candidate edges (or all graph edges if None).

        Args:
            graph: the current navigation graph
            conflicts: list of detected Conflict objects
            walkthrough_path: ordered list of (src, dst) edges traversed by the
                              agent so far. Used to compute U(e).
            candidate_edges: edges to score. If None, all edges in the graph
                             are scored.

        Returns:
            mapping edge -> EdgeScore
        """
        self.edge_scores.clear()

        if candidate_edges is None:
            candidate_edges = list(graph.edges())

        # 1. Raw factor computation
        conflict_edge_counter = self._count_conflict_edges(conflicts)
        usage_counter = Counter(walkthrough_path)

        for edge in candidate_edges:
            if edge not in graph.edges():
                continue
            r = self._reachability(graph, edge)
            c = conflict_edge_counter.get(edge, 0)
            u = usage_counter.get(edge, 0)
            self.edge_scores[edge] = EdgeScore(
                edge=edge, reach=r, conflict_count=c, usage=u
            )

        # 2. Min-max normalization
        self._min_max_normalize()

        # 3. Final EIS = unweighted sum of the three normalized factors
        for s in self.edge_scores.values():
            s.score = s.r_hat + s.c_hat + s.u_hat

        return self.edge_scores

    def rank(self) -> List[EdgeScore]:
        """Return EdgeScore objects sorted by descending EIS score."""
        return sorted(self.edge_scores.values(),
                      key=lambda x: x.score, reverse=True)

    def get_top_impact_edges(self, n: int = 10) -> List[EdgeScore]:
        """Top n edges by EIS score."""
        return self.rank()[:n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _reachability(graph: nx.DiGraph, edge: Tuple[str, str]) -> int:
        """R(e): number of nodes reachable downstream of edge e.

        Defined as |descendants(target(e))| + 1 (the target itself is counted
        because it would be displaced if e were wrong).
        """
        _, dst = edge
        if dst not in graph:
            return 0
        return len(nx.descendants(graph, dst)) + 1

    @staticmethod
    def _count_conflict_edges(conflicts: List) -> Dict[Tuple[str, str], int]:
        """C(e): number of distinct conflicts involving edge e."""
        counter: Dict[Tuple[str, str], int] = defaultdict(int)
        for conflict in conflicts:
            seen_in_this_conflict: Set[Tuple[str, str]] = set()
            for edge in getattr(conflict, "involved_edges", []):
                if isinstance(edge, tuple) and len(edge) == 2:
                    e = (edge[0], edge[1])
                    if e not in seen_in_this_conflict:
                        counter[e] += 1
                        seen_in_this_conflict.add(e)
        return counter

    def _min_max_normalize(self):
        """Apply min-max normalization to each of the three factors."""
        if not self.edge_scores:
            return

        def normalize(values: List[float]) -> List[float]:
            mn, mx = min(values), max(values)
            if mx == mn:
                # Degenerate case: every candidate has the same raw value;
                # assign 0.0 so it does not bias the ranking.
                return [0.0] * len(values)
            return [(v - mn) / (mx - mn) for v in values]

        edges = list(self.edge_scores.keys())
        reach_norm = normalize([self.edge_scores[e].reach for e in edges])
        conflict_norm = normalize([self.edge_scores[e].conflict_count for e in edges])
        usage_norm = normalize([self.edge_scores[e].usage for e in edges])

        for e, r_n, c_n, u_n in zip(edges, reach_norm, conflict_norm, usage_norm):
            s = self.edge_scores[e]
            s.r_hat = r_n
            s.c_hat = c_n
            s.u_hat = u_n

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def visualize_edge_rankings(self, top_n: int = 10) -> str:
        """Plain-text visualization of the EIS ranking."""
        lines = ["Edge Impact Score (EIS) ranking:"]
        lines.append("-" * 88)
        lines.append(f"{'Rank':<5}{'Edge':<40}{'Reach':>7}{'Conf':>7}"
                     f"{'Usage':>7}{'R_hat':>7}{'C_hat':>7}{'U_hat':>7}{'EIS':>8}")
        lines.append("-" * 88)
        for i, s in enumerate(self.get_top_impact_edges(top_n), 1):
            edge_str = f"{s.edge[0]} -> {s.edge[1]}"
            if len(edge_str) > 38:
                edge_str = edge_str[:35] + "..."
            lines.append(
                f"{i:<5}{edge_str:<40}{s.reach:>7}{s.conflict_count:>7}"
                f"{s.usage:>7}{s.r_hat:>7.2f}{s.c_hat:>7.2f}{s.u_hat:>7.2f}"
                f"{s.score:>8.3f}"
            )
        return "\n".join(lines)
