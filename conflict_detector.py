"""
Conflict Detector

Implements the three structural conflict types defined in the paper,
Section 2.2:

  1. Topological conflict: two distinct rooms occupy the same physical
     position (physical exclusivity violation), or invalid topology such as
     unreachable nodes.

  2. Directional conflict: a single source node has multiple outgoing edges
     with the same direction label (directional uniqueness violation).

  3. Naming conflict: identical names assigned to two structurally distinct
     nodes (identity uniqueness violation).

The detector is intentionally lightweight and stateless: it takes a graph
plus auxiliary metadata (per-node positions for the topological check) and
returns a list of Conflict objects. All three checks share the same Conflict
data type, so downstream localization and repair are conflict-type-agnostic.
"""

import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Conflict:
    """A detected structural conflict.

    Attributes:
        type:            'topological' | 'directional' | 'naming'
        description:     human-readable description
        involved_nodes:  the nodes that are in tension
        involved_edges:  the edges (src, dst) that participated in producing
                         the conflict; downstream localization treats these as
                         seed candidates
        step_num:        the construction step at which the conflict surfaced
        details:         arbitrary structured detail (positions, directions...)
    """
    type: str
    description: str
    involved_nodes: List[str]
    involved_edges: List[Tuple[str, str]]
    step_num: int
    details: Dict = field(default_factory=dict)


class ConflictDetector:
    """Detects the three structural conflicts from paper Section 2.2."""

    def __init__(self):
        self.conflicts: List[Conflict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_all_conflicts(self,
                             graph: nx.DiGraph,
                             step_num: int,
                             node_positions: Optional[Dict[str, Tuple[int, int]]] = None
                             ) -> List[Conflict]:
        """Run all three conflict checks and return the union.

        Args:
            graph: the navigation graph
            step_num: current construction step number
            node_positions: optional mapping node -> (x, y) integer coords used
                            for the topological exclusivity check. If absent,
                            the topological check falls back to a structural
                            unreachable-node test.
        """
        self.conflicts = []
        self.conflicts.extend(self._detect_directional(graph, step_num))
        self.conflicts.extend(self._detect_topological(graph, step_num, node_positions))
        self.conflicts.extend(self._detect_naming(graph, step_num, node_positions))
        return self.conflicts

    def get_conflict_summary(self) -> Dict:
        """Counts of detected conflicts by type."""
        return {
            "total_conflicts": len(self.conflicts),
            "by_type": {
                "topological": sum(1 for c in self.conflicts if c.type == "topological"),
                "directional": sum(1 for c in self.conflicts if c.type == "directional"),
                "naming":      sum(1 for c in self.conflicts if c.type == "naming"),
            }
        }

    # ------------------------------------------------------------------
    # 1. Directional conflict
    # ------------------------------------------------------------------
    def _detect_directional(self,
                            graph: nx.DiGraph,
                            step_num: int) -> List[Conflict]:
        """A single node has multiple outgoing edges with the same direction."""
        conflicts: List[Conflict] = []
        for node in graph.nodes():
            buckets: Dict[str, List[str]] = defaultdict(list)
            for _, dst, data in graph.out_edges(node, data=True):
                direction = (data.get("direction") or "").lower().strip()
                if direction:
                    buckets[direction].append(dst)

            for direction, targets in buckets.items():
                if len(targets) > 1:
                    conflicts.append(Conflict(
                        type="directional",
                        description=(f"Directional conflict: node '{node}' has "
                                     f"{len(targets)} outgoing '{direction}' edges"),
                        involved_nodes=[node] + targets,
                        involved_edges=[(node, t) for t in targets],
                        step_num=step_num,
                        details={"direction": direction, "targets": targets}
                    ))
        return conflicts

    # ------------------------------------------------------------------
    # 2. Topological conflict (physical exclusivity)
    # ------------------------------------------------------------------
    def _detect_topological(self,
                            graph: nx.DiGraph,
                            step_num: int,
                            node_positions: Optional[Dict[str, Tuple[int, int]]]
                            ) -> List[Conflict]:
        """Detect topological conflicts.

        Primary check (paper challenge example, rebuttal C1): two distinct
        nodes are inferred to occupy the same unit-distance position. The
        framework assumes each cardinal action moves the agent one spatial
        unit, so any two nodes mapping to the same coordinate violate
        physical exclusivity.

        Fallback check (when no positions are provided): structural sanity
        such as multiple weakly connected components, which corresponds to
        the "unreachable nodes" wording in the paper definition.
        """
        conflicts: List[Conflict] = []

        if node_positions:
            position_map: Dict[Tuple[int, int], List[str]] = defaultdict(list)
            for node, pos in node_positions.items():
                if pos is not None:
                    position_map[tuple(pos)].append(node)

            for pos, nodes_here in position_map.items():
                if len(nodes_here) > 1:
                    involved_edges: List[Tuple[str, str]] = []
                    for n in nodes_here:
                        for pred in graph.predecessors(n):
                            involved_edges.append((pred, n))
                    conflicts.append(Conflict(
                        type="topological",
                        description=(f"Topological conflict: {len(nodes_here)} "
                                     f"distinct nodes occupy position {pos}: "
                                     f"{nodes_here}"),
                        involved_nodes=nodes_here,
                        involved_edges=involved_edges,
                        step_num=step_num,
                        details={"position": list(pos), "nodes": nodes_here}
                    ))
        else:
            # Fallback: report unreachable / disconnected nodes.
            if graph.number_of_nodes() > 1:
                components = list(nx.weakly_connected_components(graph))
                if len(components) > 1:
                    main = max(components, key=len)
                    for comp in components:
                        if comp is main:
                            continue
                        nodes_here = sorted(comp)
                        conflicts.append(Conflict(
                            type="topological",
                            description=(f"Topological conflict: unreachable "
                                         f"component {nodes_here}"),
                            involved_nodes=nodes_here,
                            involved_edges=[],
                            step_num=step_num,
                            details={"component_size": len(comp)}
                        ))
        return conflicts

    # ------------------------------------------------------------------
    # 3. Naming conflict (identity uniqueness)
    # ------------------------------------------------------------------
    def _detect_naming(self,
                       graph: nx.DiGraph,
                       step_num: int,
                       node_positions: Optional[Dict[str, Tuple[int, int]]]
                       ) -> List[Conflict]:
        """Detect naming conflicts.

        We canonicalize node names (lowercase, stripped) and look for
        clusters of structurally distinct nodes that share the same
        canonical name.

        "Structurally distinct" is determined by:
          (a) different positions, when positions are available; otherwise
          (b) the presence of multiple distinct nodes whose canonical name
              collides (the canonical-name set is built from node ids that
              the graph already separated as different nodes, e.g., when the
              graph stores variants like "Wharf Road (1)" / "Wharf Road (2)").

        Either way, identical canonical names attached to >1 distinct node
        identifies a naming conflict.
        """
        conflicts: List[Conflict] = []

        def canonical(node_id: str) -> str:
            """Strip parenthetical disambiguation suffixes."""
            base = node_id.lower().strip()
            # Strip patterns like "wharf road (1)" -> "wharf road"
            paren_idx = base.find(" (")
            if paren_idx > 0 and base.endswith(")"):
                base = base[:paren_idx].strip()
            return base

        groups: Dict[str, List[str]] = defaultdict(list)
        for node in graph.nodes():
            groups[canonical(node)].append(node)

        for canon, members in groups.items():
            if len(members) <= 1:
                continue

            # If we have positions, only flag groups whose members occupy
            # different positions (different physical places sharing a name).
            if node_positions:
                positions = {tuple(node_positions[n]) for n in members
                             if node_positions.get(n) is not None}
                if len(positions) <= 1:
                    continue

            involved_edges: List[Tuple[str, str]] = []
            for n in members:
                for pred in graph.predecessors(n):
                    involved_edges.append((pred, n))
                for succ in graph.successors(n):
                    involved_edges.append((n, succ))

            conflicts.append(Conflict(
                type="naming",
                description=(f"Naming conflict: {len(members)} distinct nodes "
                             f"share the canonical name '{canon}': {members}"),
                involved_nodes=members,
                involved_edges=involved_edges,
                step_num=step_num,
                details={"canonical_name": canon, "members": members}
            ))
        return conflicts
