"""
Version Control

Implements the Version Control module described in the paper, Section 2.4.

A Version Control instance maintains a directed chain of commits

    [G_0, G_1, ..., G_t]

where each commit G_i records a step-wise modification to the navigation
graph plus the originating observation and analysis. The commit schema
follows the paper:

    G_i = { Step_id, Commit, Trigger_event, Observation_id, Analysis }

Commits store edge-level diffs only (additions and removals during
replacements), not full graph snapshots. The fully reconstructed graph at
any version can be obtained by replaying the commits up to that version.

Three operations are exposed, matching the paper:

    rollback_to(version)  : restore the graph to a prior version
    recall_step(version)  : retrieve the reasoning history (observation +
                            analysis) recorded for a given commit
    diff(G_i, G_j)        : compute edge-level differences between two
                            versions

In addition, the version chain can be exposed as a Reasoning History Tree
T (a DAG with a unique timestamp tau(v) per node) that the conflict
localizer uses for LCA computation.
"""

import networkx as nx
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


# Trigger event labels (paper Section 2.4)
TRIGGER_OBSERVATION = "observation"
TRIGGER_CONFLICT_REPAIR = "conflict_repair"
TRIGGER_INIT = "init"


@dataclass
class Commit:
    """A single commit G_i in the Version Control chain.

    Fields mirror the paper's commit schema:
        step_id:        construction step at which the commit was made
        added_edges:    edges added in this commit, as (src, dst, direction)
        removed_edges:  edges removed in this commit, as (src, dst, direction)
        added_nodes:    nodes added in this commit
        removed_nodes:  nodes removed in this commit
        trigger_event:  why this commit was made (observation / conflict_repair / init)
        observation_id: pointer to the originating observation (typically the
                        step number, identical to step_id for ordinary additions)
        observation:    raw observation text recorded for recall_step()
        analysis:       LLM analysis attached to this commit
    """
    version_id: str
    step_id: int
    added_edges: List[Tuple[str, str, str]] = field(default_factory=list)
    removed_edges: List[Tuple[str, str, str]] = field(default_factory=list)
    added_nodes: List[str] = field(default_factory=list)
    removed_nodes: List[str] = field(default_factory=list)
    trigger_event: str = TRIGGER_OBSERVATION
    observation_id: Optional[int] = None
    observation: str = ""
    analysis: str = ""


class VersionControl:
    """Versioned history of the navigation graph (paper Section 2.4)."""

    def __init__(self):
        self._commits: Dict[str, Commit] = {}
        self._order: List[str] = []
        self._current: Optional[str] = None

    # ------------------------------------------------------------------
    # Commit creation
    # ------------------------------------------------------------------
    def commit(self,
               step_id: int,
               added_edges: List[Tuple[str, str, str]] = None,
               removed_edges: List[Tuple[str, str, str]] = None,
               added_nodes: List[str] = None,
               removed_nodes: List[str] = None,
               trigger_event: str = TRIGGER_OBSERVATION,
               observation: str = "",
               analysis: str = "",
               observation_id: Optional[int] = None) -> str:
        """Append a new commit and return its version id."""
        version_id = f"v{step_id}"
        # Disambiguate if we already have a commit at this step (e.g.,
        # observation commit followed by a conflict_repair commit).
        suffix = 1
        while version_id in self._commits:
            suffix += 1
            version_id = f"v{step_id}.{suffix}"

        commit_obj = Commit(
            version_id=version_id,
            step_id=step_id,
            added_edges=list(added_edges or []),
            removed_edges=list(removed_edges or []),
            added_nodes=list(added_nodes or []),
            removed_nodes=list(removed_nodes or []),
            trigger_event=trigger_event,
            observation_id=observation_id if observation_id is not None else step_id,
            observation=observation,
            analysis=analysis,
        )
        self._commits[version_id] = commit_obj
        self._order.append(version_id)
        self._current = version_id
        return version_id

    # ------------------------------------------------------------------
    # Paper-defined operations
    # ------------------------------------------------------------------
    def rollback_to(self, version_id: str) -> nx.DiGraph:
        """Restore the navigation graph to the state at `version_id`.

        The graph is reconstructed by replaying every commit in order from
        the start of the chain up to and including `version_id`. Returns the
        reconstructed graph; the caller is responsible for swapping it back
        into the active NavigationGraph.
        """
        if version_id not in self._commits:
            raise ValueError(f"Unknown version {version_id}")

        graph = nx.DiGraph()
        for vid in self._order:
            commit_obj = self._commits[vid]
            self._apply(graph, commit_obj)
            if vid == version_id:
                break

        # Truncate the chain so future commits start from this point.
        idx = self._order.index(version_id)
        for stale in self._order[idx + 1:]:
            self._commits.pop(stale, None)
        self._order = self._order[:idx + 1]
        self._current = version_id
        return graph

    def recall_step(self, version_id: str) -> Dict[str, Any]:
        """Return the reasoning history recorded for the given commit.

        Output dictionary fields match the paper's commit schema and are
        suitable for direct injection into the LLM repair prompt.
        """
        if version_id not in self._commits:
            raise ValueError(f"Unknown version {version_id}")
        c = self._commits[version_id]
        return {
            "version_id":     c.version_id,
            "step_id":        c.step_id,
            "trigger_event":  c.trigger_event,
            "observation_id": c.observation_id,
            "observation":    c.observation,
            "analysis":       c.analysis,
            "added_edges":    c.added_edges,
            "removed_edges":  c.removed_edges,
        }

    def diff(self, version_id_a: str, version_id_b: str) -> Dict[str, List]:
        """Edge-level differences between two versions.

        Returns the set of edges that exist in version B but not A
        (`added`) and the set that exist in A but not B (`removed`).
        """
        graph_a = self._reconstruct_at(version_id_a)
        graph_b = self._reconstruct_at(version_id_b)
        edges_a: Set[Tuple[str, str]] = set(graph_a.edges())
        edges_b: Set[Tuple[str, str]] = set(graph_b.edges())
        return {
            "added":   sorted(edges_b - edges_a),
            "removed": sorted(edges_a - edges_b),
        }

    # ------------------------------------------------------------------
    # Reasoning History Tree (used by ConflictLocalizer for LCA)
    # ------------------------------------------------------------------
    def reasoning_history_tree(self) -> Tuple[nx.DiGraph, Dict[str, int]]:
        """Build the Reasoning History Tree T from the commit chain.

        The tree's nodes are spatial-graph nodes; an edge u -> v exists if a
        commit added the spatial edge u -> v. Each node v carries a unique
        timestamp tau(v) equal to the construction step at which it was first
        added (the earliest commit that introduced v as a destination).

        Returns:
            (T, tau) where T is a nx.DiGraph and tau maps node -> step number.
        """
        T = nx.DiGraph()
        tau: Dict[str, int] = {}

        for vid in self._order:
            c = self._commits[vid]
            for node in c.added_nodes:
                if node not in tau:
                    tau[node] = c.step_id
                    T.add_node(node)
            for src, dst, _direction in c.added_edges:
                if src not in tau:
                    tau[src] = c.step_id
                    T.add_node(src)
                if dst not in tau:
                    tau[dst] = c.step_id
                    T.add_node(dst)
                T.add_edge(src, dst, step_id=c.step_id, version_id=vid)
            for src, dst, _direction in c.removed_edges:
                if T.has_edge(src, dst):
                    T.remove_edge(src, dst)
        return T, tau

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def list_versions(self) -> List[str]:
        return list(self._order)

    def get_commit(self, version_id: str) -> Optional[Commit]:
        return self._commits.get(version_id)

    def current_version(self) -> Optional[str]:
        return self._current

    def export_timeline(self) -> List[Dict]:
        """JSON-serializable list of commits in chronological order."""
        out = []
        for vid in self._order:
            c = self._commits[vid]
            out.append({
                "version_id":     c.version_id,
                "step_id":        c.step_id,
                "trigger_event":  c.trigger_event,
                "observation_id": c.observation_id,
                "added_edges":    c.added_edges,
                "removed_edges":  c.removed_edges,
                "added_nodes":    c.added_nodes,
                "removed_nodes":  c.removed_nodes,
                "observation":    c.observation[:200],
                "analysis":       c.analysis[:200],
            })
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reconstruct_at(self, version_id: str) -> nx.DiGraph:
        """Replay commits up to (and including) `version_id` and return the graph."""
        if version_id not in self._commits:
            raise ValueError(f"Unknown version {version_id}")
        graph = nx.DiGraph()
        for vid in self._order:
            self._apply(graph, self._commits[vid])
            if vid == version_id:
                break
        return graph

    @staticmethod
    def _apply(graph: nx.DiGraph, commit_obj: Commit):
        """Apply a single commit to a graph in place."""
        for node in commit_obj.added_nodes:
            if node not in graph:
                graph.add_node(node)
        for src, dst, direction in commit_obj.added_edges:
            graph.add_edge(src, dst, direction=direction,
                           step_num=commit_obj.step_id)
        for src, dst, _direction in commit_obj.removed_edges:
            if graph.has_edge(src, dst):
                graph.remove_edge(src, dst)
        for node in commit_obj.removed_nodes:
            if node in graph:
                graph.remove_node(node)
