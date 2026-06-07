from maprepair.graph import NavGraph
from maprepair.conflict import detect_all
from maprepair.localizer import Localizer


def test_direction_conflict_yields_two_candidates():
    g = NavGraph()
    g.add_edge("a", "b", "north", add_auto_reverse=False)
    g.add_edge("a", "c", "north", add_auto_reverse=False)
    cs = detect_all(g)
    assert cs
    direction_cs = [c for c in cs if c.type == "direction"]
    assert direction_cs
    cands = Localizer().localize(g, direction_cs[0])
    assert set(cands) == {("a", "b"), ("a", "c")}


def test_reduction_ratio_smaller_for_local_conflicts():
    g = NavGraph()
    for i in range(10):
        g.add_edge(f"r{i}", f"r{i+1}", "north", add_auto_reverse=False)
    # Inject a direction conflict at r5 -> r6 and r5 -> r6'
    g.add_edge("r5", "r6_alt", "north", add_auto_reverse=False)
    cs = detect_all(g)
    direction_cs = [c for c in cs if c.type == "direction"]
    assert direction_cs
    loc = Localizer()
    ratio = loc.reduction_ratio(g, direction_cs)
    # Should be substantial because direction conflicts are extremely localized
    assert 0.5 <= ratio <= 1.0
