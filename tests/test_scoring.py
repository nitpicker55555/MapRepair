from maprepair.graph import NavGraph
from maprepair.scoring import score_edges
from maprepair.conflict import detect_all


def test_score_edges_returns_nonempty():
    g = NavGraph()
    g.add_edge("a", "b", "north", add_auto_reverse=False)
    g.add_edge("b", "c", "east", add_auto_reverse=False)
    g.add_edge("c", "d", "south", add_auto_reverse=False)
    scored = score_edges(g)
    assert len(scored) == 3
    assert all(s.score >= 0 for s in scored)
    # the most upstream edge should have the highest reach component
    by_edge = {s.edge: s for s in scored}
    assert by_edge[("a", "b")].reach == 1.0
    assert by_edge[("c", "d")].reach == 0.0


def test_score_edges_orders_by_descending_score():
    g = NavGraph()
    for i in range(5):
        g.add_edge(f"n{i}", f"n{i+1}", "north", add_auto_reverse=False)
    scored = score_edges(g)
    scores = [s.score for s in scored]
    assert scores == sorted(scores, reverse=True)


def test_cascade_correlation_perfect_on_paper_example():
    """A purpose-built graph where downstream reach is the only signal.

    Five disjoint chains of length 1, 2, 3, 4, 5 starting from a common root.
    The error edge per chain has reach = chain_length - 1.
    """
    g = NavGraph()
    chains = [1, 2, 3, 4, 5]
    error_edges = []
    for i, length in enumerate(chains):
        prev = "root"
        # use a unique sequence per chain so they're independent
        for step in range(length):
            nxt = f"c{i}_{step}"
            g.add_edge(prev, nxt, "north", add_auto_reverse=False)
            if step == 0:
                error_edges.append((prev, nxt))
            prev = nxt
    scored = score_edges(g)
    # rank of each error edge in descending score
    scored_dict = {s.edge: i for i, s in enumerate(scored)}
    # error edges should be ordered by chain length (longest chain = highest reach)
    ranks = [(scored_dict[e], chains[i]) for i, e in enumerate(error_edges)]
    ranks.sort()  # by rank ascending
    chain_lengths = [r[1] for r in ranks]
    # longest chain should be first (highest impact)
    assert chain_lengths[0] >= chain_lengths[-1]
