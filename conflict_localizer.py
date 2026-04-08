"""
Conflict Localizer

Implements the four-stage error localization pipeline from the paper,
Section 2.3:

  (1) Minimal conflicting path pair: identify two distinct paths leading to
      the conflicting nodes.
  (2) LCA on the Reasoning History Tree T (paper Eq. 2):

          LCA(pi_1, pi_2) = argmax_{v in pi_1 ∩ pi_2} tau(v)

      where T is a DAG built from version control commits and tau(v) is
      each node's construction timestamp. LCA is computed on T - not on the
      spatial graph G - so it remains well-defined when G contains cycles.
  (3) Candidate edge extraction: collect every edge along the divergent
      subpaths from LCA to each conflict node.
  (4) Pass the candidates to the Edge Impact Scorer for ranking.
"""

import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from conflict_detector import Conflict


@dataclass
class CandidateEdge:
    """One candidate error edge produced by LCA-based localization."""
    edge: Tuple[str, str]
    direction: str = ""
    step_id: int = -1
    conflict_count: int = 0
    conflict_types: List[str] = field(default_factory=list)


@dataclass
class LocalizationResult:
    """Per-conflict localization output."""
    conflict_type: str
    path1: List[str]
    path2: List[str]
    lca: Optional[str]
    candidate_edges: List[Tuple[str, str]]


class ConflictLocalizer:
    """LCA-based error localization on the Reasoning History Tree T."""

    def __init__(self):
        self.results: List[LocalizationResult] = []
        self.candidate_edges: Dict[Tuple[str, str], CandidateEdge] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def localize_conflicts(self,
                           graph: nx.DiGraph,
                           conflicts: List[Conflict],
                           reasoning_tree: nx.DiGraph,
                           tau: Dict[str, int]) -> Dict:
        """Localize all detected conflicts on the Reasoning History Tree.

        Args:
            graph: current spatial navigation graph (used to look up
                   candidate edge directions)
            conflicts: list of detected Conflict objects
            reasoning_tree: T, a DAG over spatial graph nodes whose edges
                            were laid in commit order
            tau: mapping node -> construction timestamp

        Returns:
            dict with `conflict_localizations` (per conflict),
            `candidate_edges` (deduplicated union), and basic statistics.
        """
        self.results = []
        self.candidate_edges = {}

        for conflict in conflicts:
            if conflict.type == "directional":
                self._localize_directional(graph, conflict, tau)
            else:
                # topological / naming conflicts use full path tracing
                self._localize_path_based(graph, conflict, reasoning_tree, tau)

        return {
            "conflict_localizations": [self._result_to_dict(r) for r in self.results],
            "candidate_edges":        [self._cand_to_dict(c)   for c in self.candidate_edges.values()],
            "stats": {
                "total_conflicts": len(conflicts),
                "total_candidates": len(self.candidate_edges),
            }
        }

    # ------------------------------------------------------------------
    # Directional conflicts: degenerate LCA = source node
    # ------------------------------------------------------------------
    def _localize_directional(self,
                              graph: nx.DiGraph,
                              conflict: Conflict,
                              tau: Dict[str, int]):
        """LCA = Source for directional conflicts (paper Section 2.3)."""
        if not conflict.involved_nodes:
            return
        source = conflict.involved_nodes[0]
        targets = conflict.details.get("targets", conflict.involved_nodes[1:])
        candidate_edges = [(source, t) for t in targets]

        for e in candidate_edges:
            self._record_candidate(graph, e, conflict)

        # Build degenerate path pair (length 0 from LCA)
        if len(targets) >= 2:
            self.results.append(LocalizationResult(
                conflict_type="directional",
                path1=[source, targets[0]],
                path2=[source, targets[1]],
                lca=source,
                candidate_edges=candidate_edges,
            ))

    # ------------------------------------------------------------------
    # Topological / naming conflicts: full LCA on the reasoning tree
    # ------------------------------------------------------------------
    def _localize_path_based(self,
                             graph: nx.DiGraph,
                             conflict: Conflict,
                             reasoning_tree: nx.DiGraph,
                             tau: Dict[str, int]):
        """Compute paths from a tree root to each conflict node, then LCA."""
        if len(conflict.involved_nodes) < 2:
            return

        node_a = conflict.involved_nodes[0]
        node_b = conflict.involved_nodes[1]

        # Pick a root for tracing: the node with the smallest tau in the tree
        # (typically the agent's starting location).
        if not tau:
            return
        roots = [n for n in reasoning_tree.nodes()
                 if reasoning_tree.in_degree(n) == 0]
        if not roots:
            roots = [min(tau, key=tau.get)]
        root = min(roots, key=lambda n: tau.get(n, float("inf")))

        path1 = self._shortest_path(reasoning_tree, root, node_a)
        path2 = self._shortest_path(reasoning_tree, root, node_b)
        if not path1 or not path2:
            return

        lca = self._lca_on_tree(path1, path2, tau)
        candidate_edges = self._divergent_edges(path1, path2, lca)

        for e in candidate_edges:
            self._record_candidate(graph, e, conflict)

        self.results.append(LocalizationResult(
            conflict_type=conflict.type,
            path1=path1,
            path2=path2,
            lca=lca,
            candidate_edges=candidate_edges,
        ))

    # ------------------------------------------------------------------
    # LCA on the Reasoning History Tree (paper Eq. 2)
    # ------------------------------------------------------------------
    @staticmethod
    def _lca_on_tree(path1: List[str],
                     path2: List[str],
                     tau: Dict[str, int]) -> Optional[str]:
        """LCA(pi_1, pi_2) = argmax_{v in pi_1 ∩ pi_2} tau(v)."""
        common = set(path1) & set(path2)
        if not common:
            return None
        return max(common, key=lambda v: tau.get(v, -1))

    @staticmethod
    def _divergent_edges(path1: List[str],
                         path2: List[str],
                         lca: Optional[str]) -> List[Tuple[str, str]]:
        """Edges along the divergent subpaths from LCA to each conflict node."""
        if lca is None:
            return []

        def subpath(path: List[str]) -> List[str]:
            if lca not in path:
                return []
            return path[path.index(lca):]

        edges: List[Tuple[str, str]] = []
        for sub in (subpath(path1), subpath(path2)):
            for i in range(len(sub) - 1):
                e = (sub[i], sub[i + 1])
                if e not in edges:
                    edges.append(e)
        return edges

    @staticmethod
    def _shortest_path(tree: nx.DiGraph,
                       source: str,
                       target: str) -> List[str]:
        if source not in tree or target not in tree:
            return []
        try:
            return nx.shortest_path(tree, source=source, target=target)
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []

    # ------------------------------------------------------------------
    # Candidate accumulation
    # ------------------------------------------------------------------
    def _record_candidate(self,
                          graph: nx.DiGraph,
                          edge: Tuple[str, str],
                          conflict: Conflict):
        if not graph.has_edge(*edge):
            return
        data = graph.get_edge_data(*edge) or {}
        existing = self.candidate_edges.get(edge)
        if existing is None:
            existing = CandidateEdge(
                edge=edge,
                direction=data.get("direction", ""),
                step_id=data.get("step_num", -1),
            )
            self.candidate_edges[edge] = existing
        existing.conflict_count += 1
        if conflict.type not in existing.conflict_types:
            existing.conflict_types.append(conflict.type)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _result_to_dict(result: LocalizationResult) -> Dict:
        return {
            "conflict_type": result.conflict_type,
            "path1": result.path1,
            "path2": result.path2,
            "lca": result.lca,
            "candidate_edges": result.candidate_edges,
        }

    @staticmethod
    def _cand_to_dict(cand: CandidateEdge) -> Dict:
        return {
            "edge": list(cand.edge),
            "direction": cand.direction,
            "step_id": cand.step_id,
            "conflict_count": cand.conflict_count,
            "conflict_types": cand.conflict_types,
        }

    def get_candidate_edges(self) -> List[Tuple[str, str]]:
        return list(self.candidate_edges.keys())
