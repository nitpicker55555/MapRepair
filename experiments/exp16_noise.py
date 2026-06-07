"""Experiment 16 noise library: extends synth.py with four new injectors
that model the MANGO node-level failure modes (F2/F3/F4/F5/F6) the
repair pipeline cannot fix by design. Each injector returns a list of
labelled error records so downstream metrics can attribute drops to
specific noise types.

  N3 endpoint_swap   (F2/F6-style: wrong dst at same source)
  N4 spurious_pair   (F2-style: extra edge between real rooms)
  N5 node_collapse   (F3: prefix-sharing rooms merged into one)
  N7 hallucinated    (F4: edge to a fake room not in GT)

Existing primitives we reuse:
  N1 direction      = synth.inject_direction_errors
  N2 topology       = synth.inject_topology_errors
  N6 node_duplicate = synth.inject_naming_errors  (already there, inverse of F5)

All injectors are *seeded* and operate on a NavGraph in-place. Each
returns `List[Dict]` of `{type, src, dst, dir, note}` for the
attribution log.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from maprepair.graph import DIRECTIONS, NavGraph, OPPOSITE


HALLUCINATED_ROOM_PREFIXES = (
    "viewing room", "old plaque", "the gallery", "rear annex",
    "back chamber", "western alcove", "store room", "trophy room",
    "the office", "abandoned cellar",
)


# ----------------------------------------------------------------------
# Synthetic-graph decoration: give nodes shared prefixes so N5 has
# realistic structure to collapse.  Used for exp16 where we want a
# conflict-free baseline.
# ----------------------------------------------------------------------

def decorate_with_prefixes(graph: NavGraph, *, n_groups: int = 4,
                              members_per_group: int = 3,
                              seed: int = 0) -> Dict[str, str]:
    """Rename a subset of nodes so they share prefixes like
    'wing 1 (room a)', 'wing 1 (room b)', 'wing 2 (room a)', ...

    Returns the rename map old_name -> new_name.
    """
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    rng.shuffle(nodes)
    rename: Dict[str, str] = {}
    suffixes = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
                 "golf", "hotel", "india", "juliet"]
    cursor = 0
    for g_idx in range(n_groups):
        new_prefix = f"wing {g_idx + 1}"
        # take up to members_per_group nodes
        rng.shuffle(suffixes)
        for m_idx in range(members_per_group):
            if cursor >= len(nodes):
                break
            old = nodes[cursor]; cursor += 1
            new = f"{new_prefix} ({suffixes[m_idx]})"
            if new in graph.nx:
                continue
            rename[old] = new
    # apply renames atomically (build a tmp graph)
    if rename:
        nx = graph.nx
        # gather edges first
        edges = list(nx.edges(data=True))
        # add new nodes
        for new in rename.values():
            if not nx.has_node(new):
                nx.add_node(new)
        # rewire edges
        for u, v, data in edges:
            new_u = rename.get(u, u); new_v = rename.get(v, v)
            if new_u == new_v:
                continue
            if not nx.has_edge(new_u, new_v):
                nx.add_edge(new_u, new_v, **data)
        # remove old nodes
        for old in rename.keys():
            if nx.has_node(old):
                nx.remove_node(old)
    return rename


@dataclass
class NoiseRecord:
    """One labelled error injection."""
    type: str                       # N1..N7
    src: Optional[str]
    dst: Optional[str]
    direction: Optional[str]
    note: str = ""

    def to_dict(self) -> Dict:
        return {"type": self.type, "src": self.src, "dst": self.dst,
                "direction": self.direction, "note": self.note}


# ----------------------------------------------------------------------
# N3: endpoint-swap  (F2/F6-style)
# ----------------------------------------------------------------------

def inject_endpoint_swap(graph: NavGraph, n: int, *, seed: int = 0) -> List[NoiseRecord]:
    """Pick edges (u, v, d) and rewrite v to a *different* real GT node v'.

    Models the LLM's F2 case ("dark room, wrong destination") and the F6
    case where it correctly identifies the action but mislabels which
    adjacent room the player landed in.
    """
    rng = random.Random(seed)
    out: List[NoiseRecord] = []
    primaries = graph.primary_edges()
    rng.shuffle(primaries)
    nodes = graph.nodes()
    for e in primaries:
        if len(out) >= n:
            break
        u, v, d = e.source, e.target, e.direction
        candidates = [w for w in nodes if w != u and w != v and not graph.has_edge(u, w)]
        if not candidates:
            continue
        new_v = rng.choice(candidates)
        try:
            graph.remove_edge(u, v)
            # also drop the auto-reverse if it exists
            graph.remove_edge(v, u)
            graph.add_edge(u, new_v, d, add_auto_reverse=True)
        except Exception:
            continue
        out.append(NoiseRecord("N3_endpoint_swap", u, new_v, d,
                                note=f"orig dst={v!r}"))
    return out


# ----------------------------------------------------------------------
# N4: spurious-pair  (F2-style)
# ----------------------------------------------------------------------

def inject_spurious_pair(graph: NavGraph, n: int, *, seed: int = 0) -> List[NoiseRecord]:
    """Add (u, x, d) where u, x are real GT nodes but no GT edge exists.

    Models the LLM's habit of inventing an edge between real rooms after
    a src-drift cascade.
    """
    rng = random.Random(seed)
    out: List[NoiseRecord] = []
    nodes = graph.nodes()
    rng.shuffle(nodes)
    pairs_seen: Set[Tuple[str, str]] = set()
    for u in nodes:
        if len(out) >= n:
            break
        candidates = [w for w in nodes if w != u and not graph.has_edge(u, w)]
        rng.shuffle(candidates)
        for x in candidates:
            if (u, x) in pairs_seen:
                continue
            pairs_seen.add((u, x))
            used = {e.direction for e in graph.outgoing(u)}
            free = [d for d in DIRECTIONS if d not in used]
            if not free:
                continue
            d = rng.choice(free)
            try:
                graph.add_edge(u, x, d, add_auto_reverse=False)
            except Exception:
                continue
            out.append(NoiseRecord("N4_spurious_pair", u, x, d))
            break
    return out


# ----------------------------------------------------------------------
# N5: node-collapse  (F3, the key one)
# ----------------------------------------------------------------------

_PREFIX_PATTERN = re.compile(r"^(.+?)\s*\(")


def _split_prefix(name: str) -> Optional[str]:
    """Return the substring before the first ' (' if present, else None."""
    m = _PREFIX_PATTERN.match(name)
    return m.group(1).strip() if m else None


def inject_node_collapse(graph: NavGraph, n: int, *, seed: int = 0) -> List[NoiseRecord]:
    """Pick two GT nodes sharing a name prefix (e.g., 'back alley (...)') and
    rename one to match the other, merging their edge sets.

    If no prefix-sharing pair exists in the graph, fall back to merging
    two random GT nodes (still a node-level error, attributes as N5_fallback).
    """
    rng = random.Random(seed)
    out: List[NoiseRecord] = []
    # group nodes by prefix
    groups: Dict[str, List[str]] = {}
    for name in graph.nodes():
        pref = _split_prefix(name)
        if pref:
            groups.setdefault(pref, []).append(name)
    eligible = [(p, names) for p, names in groups.items() if len(names) >= 2]
    rng.shuffle(eligible)

    used: Set[str] = set()
    for prefix, names in eligible:
        if len(out) >= n:
            break
        rng.shuffle(names)
        if len(names) < 2:
            continue
        # take pairs from this group
        pairs_in_group = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs_in_group.append((names[i], names[j]))
        rng.shuffle(pairs_in_group)
        for a, b in pairs_in_group:
            if len(out) >= n:
                break
            if a in used or b in used or a == b:
                continue
            try:
                _rename_node_inplace(graph, b, a)
                used.update({a, b})
                out.append(NoiseRecord("N5_node_collapse", b, a, None,
                                        note=f"collapsed under prefix={prefix!r}"))
            except Exception:
                continue

    # fallback: if we still don't have enough, randomly merge any two
    if len(out) < n:
        all_nodes = [x for x in graph.nodes() if x not in used]
        rng.shuffle(all_nodes)
        while len(out) < n and len(all_nodes) >= 2:
            a = all_nodes.pop(); b = all_nodes.pop()
            if a in used or b in used:
                continue
            try:
                _rename_node_inplace(graph, b, a)
                used.update({a, b})
                out.append(NoiseRecord("N5_collapse_fallback", b, a, None,
                                        note="no prefix match available"))
            except Exception:
                continue
    return out


def _rename_node_inplace(graph: NavGraph, old: str, new: str) -> None:
    """Redirect every edge incident to `old` to `new`, then drop `old`."""
    nx = graph.nx
    if old not in nx or new not in nx or old == new:
        raise ValueError(f"bad rename {old!r} -> {new!r}")
    for u, v, data in list(nx.edges(old, data=True)):
        # outgoing from old (u==old) -> new
        if v == new:
            continue
        if not nx.has_edge(new, v):
            nx.add_edge(new, v, **data)
    for u, v, data in list(nx.in_edges(old, data=True)):
        if u == new:
            continue
        if not nx.has_edge(u, new):
            nx.add_edge(u, new, **data)
    nx.remove_node(old)


# ----------------------------------------------------------------------
# N7: hallucinated-node  (F4)
# ----------------------------------------------------------------------

def inject_hallucinated_node(graph: NavGraph, n: int, *, seed: int = 0) -> List[NoiseRecord]:
    """Add (u, fake_name, d) where fake_name is not in the GT node set.

    Models F4: LLM read an object/decoration string out of an
    observation ('viewing room' on a plaque) and added it as a room.
    """
    rng = random.Random(seed)
    out: List[NoiseRecord] = []
    real_nodes = set(graph.nodes())
    nodes = list(graph.nodes())
    rng.shuffle(nodes)
    fakes_used: Set[str] = set()
    for u in nodes:
        if len(out) >= n:
            break
        base = rng.choice(HALLUCINATED_ROOM_PREFIXES)
        fake = base
        suf = 0
        while fake in real_nodes or fake in fakes_used:
            suf += 1
            fake = f"{base} ({suf})"
        fakes_used.add(fake)
        used = {e.direction for e in graph.outgoing(u)}
        free = [d for d in DIRECTIONS if d not in used]
        if not free:
            continue
        d = rng.choice(free)
        try:
            graph.add_edge(u, fake, d, add_auto_reverse=False)
        except Exception:
            continue
        out.append(NoiseRecord("N7_hallucinated", u, fake, d,
                                note="dst not in GT"))
    return out


# ----------------------------------------------------------------------
# Regime config
# ----------------------------------------------------------------------

@dataclass
class Regime:
    """Each rate is a fraction of |primary_edges| (or |nodes| for N5/N6)."""
    name: str
    n1_direction: float = 0.0
    n2_topology: float = 0.0
    n3_endpoint: float = 0.0
    n4_spurious: float = 0.0
    n5_collapse: float = 0.0
    n6_duplicate: float = 0.0
    n7_hallucinated: float = 0.0


REGIMES = [
    Regime("edge_minimal", n1_direction=0.05, n2_topology=0.05,
           n3_endpoint=0.0, n4_spurious=0.0),     # ~exp03 conditions
    Regime("edge_clean",   n1_direction=0.10, n2_topology=0.10,
           n3_endpoint=0.05, n4_spurious=0.05),
    Regime("edge_heavy",   n1_direction=0.25, n2_topology=0.25,
           n3_endpoint=0.10, n4_spurious=0.10),
    Regime("node_only",    n5_collapse=0.25, n6_duplicate=0.10,
           n7_hallucinated=0.05),
    Regime("node_heavy",   n5_collapse=0.40, n6_duplicate=0.15,
           n7_hallucinated=0.10),
    Regime("mango_like",   n1_direction=0.15, n2_topology=0.10,
           n3_endpoint=0.05, n4_spurious=0.05,
           n5_collapse=0.25, n6_duplicate=0.10, n7_hallucinated=0.05),
]


def apply_regime(graph: NavGraph, regime: Regime, *, seed: int = 0) -> List[NoiseRecord]:
    """Apply all noise primitives in a deterministic order.

    Order matters: we do node-level edits first (N5 collapse + N6
    duplicate), then edge-level (N1-N4, N7). Reason: collapsing nodes
    AFTER injecting edge-level noise would also delete the noise.
    """
    n_primary = len(graph.primary_edges())
    n_nodes = len(graph.nodes())
    recs: List[NoiseRecord] = []
    if regime.n5_collapse > 0:
        recs += inject_node_collapse(graph, int(regime.n5_collapse * n_nodes),
                                       seed=seed)
    if regime.n6_duplicate > 0:
        # use existing synth.inject_naming_errors
        from maprepair.synth import inject_naming_errors
        injected = inject_naming_errors(graph, int(regime.n6_duplicate * n_nodes),
                                         seed=seed + 1)
        for (s, t, d) in injected:
            recs.append(NoiseRecord("N6_duplicate", s, t, d))
    # recount after node-level merges
    n_primary = len(graph.primary_edges())
    if regime.n1_direction > 0:
        from maprepair.synth import inject_direction_errors
        injected = inject_direction_errors(graph, int(regime.n1_direction * n_primary),
                                              seed=seed + 2)
        for (s, t, d) in injected:
            recs.append(NoiseRecord("N1_direction", s, t, d))
    if regime.n2_topology > 0:
        from maprepair.synth import inject_topology_errors
        injected = inject_topology_errors(graph, int(regime.n2_topology * n_primary),
                                             seed=seed + 3)
        for (s, t, d) in injected:
            recs.append(NoiseRecord("N2_topology", s, t, d))
    if regime.n3_endpoint > 0:
        recs += inject_endpoint_swap(graph, int(regime.n3_endpoint * n_primary),
                                       seed=seed + 4)
    if regime.n4_spurious > 0:
        recs += inject_spurious_pair(graph, int(regime.n4_spurious * n_primary),
                                       seed=seed + 5)
    if regime.n7_hallucinated > 0:
        recs += inject_hallucinated_node(graph, int(regime.n7_hallucinated * n_nodes),
                                            seed=seed + 6)
    return recs


def regime_by_name(name: str) -> Regime:
    for r in REGIMES:
        if r.name == name:
            return r
    raise KeyError(f"Unknown regime: {name}")
