from maprepair.graph import NavGraph
from maprepair.history import VersionHistory


def test_commit_and_rollback():
    g = NavGraph()
    g.add_edge("a", "b", "north")
    history = VersionHistory()
    v0 = history.commit(g, step_num=1, trigger="initial")
    g.add_edge("b", "c", "east")
    v1 = history.commit(g, step_num=2, trigger="second")
    # mutate graph after commit; rollback should give frozen state
    g.add_edge("c", "d", "south")
    rolled = history.rollback_to(v0)
    assert rolled is not None
    assert rolled.num_edges() == g.copy().num_edges() - 4  # original 2 + auto-reverse
    assert rolled.has_edge("a", "b")
    assert not rolled.has_edge("b", "c")


def test_diff_between_versions():
    g = NavGraph()
    history = VersionHistory()
    g.add_edge("a", "b", "north")
    v0 = history.commit(g, step_num=1)
    g.add_edge("b", "c", "east")
    v1 = history.commit(g, step_num=2)
    diff = history.diff(v0, v1)
    assert "added_edges" in diff
    assert ("b", "c") in diff["added_edges"]


def test_lookup_step_returns_closest():
    g = NavGraph()
    history = VersionHistory()
    g.add_edge("a", "b", "north")
    v0 = history.commit(g, step_num=1)
    g.add_edge("b", "c", "east")
    v1 = history.commit(g, step_num=5)
    assert history.lookup_step(1) == v0
    assert history.lookup_step(3) in (v0, v1)
    assert history.lookup_step(5) == v1
