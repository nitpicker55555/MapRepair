from maprepair.graph import NavGraph, Edge, OPPOSITE


def test_add_edge_auto_reverse():
    g = NavGraph()
    g.add_edge("a", "b", "north")
    assert g.has_edge("a", "b")
    assert g.has_edge("b", "a")
    assert g.direction("a", "b") == "north"
    assert g.direction("b", "a") == "south"


def test_add_edge_no_auto_reverse():
    g = NavGraph()
    g.add_edge("a", "b", "in", add_auto_reverse=False)
    assert g.has_edge("a", "b")
    assert not g.has_edge("b", "a")


def test_set_direction_syncs_auto_reverse():
    g = NavGraph()
    g.add_edge("a", "b", "north")  # creates reverse with direction=south
    g.set_direction("a", "b", "east")
    assert g.direction("a", "b") == "east"
    assert g.direction("b", "a") == "west"


def test_set_direction_does_not_touch_manual_reverse():
    g = NavGraph()
    g.add_edge("a", "b", "north")
    g.add_edge("b", "a", "south", add_auto_reverse=False, is_auto_reverse=False)
    g.set_direction("a", "b", "east")
    # because the (b,a) edge was not auto, we leave it alone
    assert g.direction("b", "a") == "south"


def test_no_self_loop():
    g = NavGraph()
    try:
        g.add_edge("a", "a", "north")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError on self-loop")


def test_primary_edges_excludes_auto_reverse():
    g = NavGraph()
    g.add_edge("a", "b", "north")
    primary = g.primary_edges()
    assert len(primary) == 1
    assert primary[0].source == "a" and primary[0].target == "b"
    assert primary[0].is_auto_reverse is False


def test_copy_isolates():
    g1 = NavGraph()
    g1.add_edge("a", "b", "north")
    g2 = g1.copy()
    g2.add_edge("b", "c", "east")
    assert not g1.has_edge("b", "c")
    assert g2.has_edge("b", "c")


def test_opposite_table_symmetric():
    for k, v in OPPOSITE.items():
        assert OPPOSITE[v] == k
