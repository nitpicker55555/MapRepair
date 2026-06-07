"""Synthetic graph generator + controllable error injection.

This module produces fully ground-truthed navigation graphs that we can use to
evaluate the LCA filtering, edge scoring, and repair algorithms without having
to rely on an LLM. Each generated graph has

  * a deterministic *correct* set of `(src, dst, direction)` triples,
  * an optional set of *injected error edges* with type labels,
  * a list of conflicts that should be detectable from the resulting graph.

Three graph families are supported:

  - `tree`: a directed tree rooted at "r" where each child is reached by a
    distinct cardinal direction. Children at each level alternate to avoid
    immediate position overlaps.
  - `grid`: an m x n rectangular grid; edges connect 4-neighbours by
    north/south/east/west.
  - `random`: a connected random spatial graph built by walking from a root and
    growing branches with random compass directions, ensuring no spurious
    conflicts in the *ground-truth* version.

Error injection supports three modes mirroring the paper:

  - `direction`: pick a node N and add an extra outgoing edge with the same
    direction as an existing one but pointing at a fresh target.
  - `topology`: pick a real edge (u --[d]--> v) and replace its reverse
    auto-edge direction with something other than OPPOSITE[d] (creates a
    topology conflict).
  - `naming`: collapse two distinct nodes by renaming one to the other; this
    causes the same node name to appear at two different inferred positions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .graph import NavGraph, OPPOSITE


@dataclass
class SyntheticGraph:
    graph: NavGraph
    ground_truth: NavGraph
    injected_errors: List[Tuple[str, str, str]] = field(default_factory=list)
    error_types: List[str] = field(default_factory=list)
    description: str = ""

    def error_edge_keys(self) -> Set[Tuple[str, str]]:
        return {(u, v) for (u, v, _d) in self.injected_errors}


_DIR_4: Tuple[str, ...] = ("north", "south", "east", "west")
_DIR_8: Tuple[str, ...] = ("north", "south", "east", "west",
                           "northeast", "northwest", "southeast", "southwest")


# ----------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------

def gen_tree(depth: int, branching: int, *, seed: int = 0,
             include_reverse: bool = True) -> NavGraph:
    """Tree of given depth & branching factor with non-colliding positions.

    Each node has up to `branching` children. Children at each level get
    different compass directions assigned in turn (north, east, south, west,
    NE, ...). The generator only emits configurations that do NOT collide on
    positions, so the resulting graph is conflict-free by construction.
    """
    rng = random.Random(seed)
    g = NavGraph()
    # BFS from root, assigning unique compass directions per parent.
    root = "r0"
    queue: List[Tuple[str, int, Tuple[int, int, int]]] = [(root, 0, (0, 0, 0))]
    used_positions: Set[Tuple[int, int, int]] = {(0, 0, 0)}
    next_id = 1
    while queue:
        node, lvl, pos = queue.pop(0)
        if lvl >= depth:
            continue
        directions = list(_DIR_4)
        rng.shuffle(directions)
        added = 0
        for d in directions:
            if added >= branching:
                break
            vec = _DIRECTION_VECTORS[d]
            np_ = (pos[0] + vec[0], pos[1] + vec[1], pos[2] + vec[2])
            if np_ in used_positions:
                continue
            child = f"r{next_id}"
            next_id += 1
            used_positions.add(np_)
            g.add_edge(node, child, d, add_auto_reverse=include_reverse)
            queue.append((child, lvl + 1, np_))
            added += 1
    return g


def gen_grid(rows: int, cols: int) -> NavGraph:
    """`rows`x`cols` grid graph with north/south/east/west connectivity."""
    g = NavGraph()
    for r in range(rows):
        for c in range(cols):
            name = f"({r},{c})"
            if c + 1 < cols:
                g.add_edge(name, f"({r},{c+1})", "east", add_auto_reverse=False)
            if r + 1 < rows:
                g.add_edge(name, f"({r+1},{c})", "south", add_auto_reverse=False)
    # add the reverse direction edges so the graph is fully symmetric
    primary = g.primary_edges()
    for e in primary:
        rev_dir = OPPOSITE[e.direction]
        if not g.has_edge(e.target, e.source):
            g.add_edge(e.target, e.source, rev_dir, add_auto_reverse=False)
    return g


def gen_random(num_nodes: int, *, branching: int = 3, seed: int = 0) -> NavGraph:
    """A random connected nav graph grown by directional random walks.

    Starts with a root and repeatedly picks a frontier node + an unused compass
    direction, adding a fresh node at the resulting position. Guarantees no
    position collisions on construction.
    """
    rng = random.Random(seed)
    g = NavGraph()
    root = "n0"
    positions: Dict[str, Tuple[int, int, int]] = {root: (0, 0, 0)}
    used_positions: Set[Tuple[int, int, int]] = {(0, 0, 0)}
    queue: List[str] = [root]
    next_id = 1
    while len(positions) < num_nodes and queue:
        # pick a node from the frontier (FIFO with random tie-break)
        idx = rng.randrange(len(queue))
        node = queue.pop(idx)
        children_added = 0
        directions = list(_DIR_4)
        rng.shuffle(directions)
        for d in directions:
            if children_added >= branching:
                break
            if len(positions) >= num_nodes:
                break
            base = positions[node]
            vec = _DIRECTION_VECTORS[d]
            np_ = (base[0] + vec[0], base[1] + vec[1], base[2] + vec[2])
            if np_ in used_positions:
                continue
            child = f"n{next_id}"
            next_id += 1
            positions[child] = np_
            used_positions.add(np_)
            g.add_edge(node, child, d, add_auto_reverse=True)
            queue.append(child)
            children_added += 1
    return g


# ----------------------------------------------------------------------
# Error injection
# ----------------------------------------------------------------------

def inject_direction_errors(graph: NavGraph, n: int, *, seed: int = 0,
                              ) -> List[Tuple[str, str, str]]:
    """Add n new edges that share a direction with an existing edge from the
    same source. Returns the list of injected (src, dst, direction).
    """
    rng = random.Random(seed)
    injected: List[Tuple[str, str, str]] = []
    nodes = graph.nodes()
    rng.shuffle(nodes)
    for src in nodes:
        if len(injected) >= n:
            break
        existing = graph.outgoing(src)
        if not existing:
            continue
        # pick a random existing edge and create a sibling with the same dir
        anchor = rng.choice(existing)
        # destination: a real node that is NOT already targeted by this dir
        candidates = [v for v in graph.nodes()
                      if v != src and v != anchor.target
                      and not graph.has_edge(src, v)]
        if not candidates:
            continue
        target = rng.choice(candidates)
        graph.add_edge(src, target, anchor.direction, add_auto_reverse=False)
        injected.append((src, target, anchor.direction))
    return injected


def inject_topology_errors(graph: NavGraph, n: int, *, seed: int = 0,
                            ) -> List[Tuple[str, str, str]]:
    """Replace reverse-direction labels with non-opposite ones.

    Returns the list of *touched* primary edges (their reverse was mangled).
    """
    rng = random.Random(seed)
    injected: List[Tuple[str, str, str]] = []
    primaries = graph.primary_edges()
    rng.shuffle(primaries)
    for e in primaries:
        if len(injected) >= n:
            break
        if not graph.has_edge(e.target, e.source):
            continue
        # change the reverse direction to a wrong (non-opposite, non-equal) direction
        choices = [d for d in _DIR_4 if d != OPPOSITE.get(e.direction) and d != e.direction]
        if not choices:
            continue
        wrong = rng.choice(choices)
        # patch directly through nx since we deliberately want desync
        graph.nx[e.target][e.source]["direction"] = wrong
        graph.nx[e.target][e.source]["is_auto_reverse"] = False  # mark as deliberate
        injected.append((e.target, e.source, wrong))
    return injected


def inject_naming_errors(graph: NavGraph, n: int, *, seed: int = 0,
                          ) -> List[Tuple[str, str, str]]:
    """Pick two distant nodes and merge them by re-using one of the names.

    The resulting graph then has edges incident to the surviving name `a` that
    used to be incident to `b`. We report each *relabeled* edge as an
    "injected error edge" so downstream metrics know which edges are
    candidates for correction.
    """
    rng = random.Random(seed)
    injected: List[Tuple[str, str, str]] = []
    nodes = list(graph.nodes())
    rng.shuffle(nodes)
    used: Set[str] = set()
    while len(used) < n * 2 and len(nodes) >= 2:
        a, b = nodes.pop(), nodes.pop()
        if a in used or b in used:
            continue
        used.add(a); used.add(b)
        # capture the edges incident to b BEFORE renaming so we can record them
        # under the new (post-rename) name `a`.
        incidence: List[Tuple[str, str, str]] = []
        nx = graph.nx
        for u, v, data in list(nx.edges(b, data=True)):
            incidence.append((a, v, data.get("direction", "")))
        for u, v, data in list(nx.in_edges(b, data=True)):
            incidence.append((u, a, data.get("direction", "")))
        _rename_node(graph, b, a)
        injected.extend(incidence)
    return injected


def _rename_node(graph: NavGraph, old: str, new: str) -> None:
    if old not in graph.nx:
        return
    edges = list(graph.nx.edges(data=True))
    incoming = [(u, v, d) for u, v, d in edges if v == old]
    outgoing = [(u, v, d) for u, v, d in edges if u == old]
    for u, _v, d in incoming:
        graph.nx.add_edge(u, new, **d)
    for _u, v, d in outgoing:
        graph.nx.add_edge(new, v, **d)
    graph.nx.remove_node(old)


# ----------------------------------------------------------------------
# High-level convenience
# ----------------------------------------------------------------------

def make_synthetic(
    family: str,
    *,
    size: int = 12,
    error_mix: Optional[Dict[str, int]] = None,
    seed: int = 0,
    branching: int = 3,
    rows: int = 4,
    cols: int = 4,
) -> SyntheticGraph:
    """One-shot synthetic graph builder.

    `family` is one of {tree, grid, random}.
    `size` controls the dominant axis (depth for tree, total nodes for random).
    `error_mix` is a dict like {"direction": 2, "topology": 1, "naming": 0}.
    """
    if family == "tree":
        gt = gen_tree(depth=size, branching=branching, seed=seed)
    elif family == "grid":
        gt = gen_grid(rows, cols)
    elif family == "random":
        gt = gen_random(num_nodes=size, branching=branching, seed=seed)
    else:
        raise ValueError(f"unknown family: {family}")
    # clone ground-truth before injecting
    broken = gt.copy()
    mix = dict(error_mix or {"direction": 1})
    inj: List[Tuple[str, str, str]] = []
    types: List[str] = []
    if mix.get("direction"):
        new = inject_direction_errors(broken, mix["direction"], seed=seed + 1)
        inj.extend(new); types.extend("direction" for _ in new)
    if mix.get("topology"):
        new = inject_topology_errors(broken, mix["topology"], seed=seed + 2)
        inj.extend(new); types.extend("topology" for _ in new)
    if mix.get("naming"):
        new = inject_naming_errors(broken, mix["naming"], seed=seed + 3)
        inj.extend(new); types.extend("naming" for _ in new)
    return SyntheticGraph(
        graph=broken,
        ground_truth=gt,
        injected_errors=inj,
        error_types=types,
        description=f"{family} size={size} mix={mix} seed={seed}",
    )


# direction vectors (mirrors conflict.py's table; duplicated here for
# generation purposes to avoid an import cycle)
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
