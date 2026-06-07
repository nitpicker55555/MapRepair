from maprepair.graph import NavGraph
from maprepair.conflict import (
    detect_direction_conflicts,
    detect_topology_conflicts,
    detect_naming_conflicts,
    detect_all,
)


def test_no_conflict_clean_graph():
    g = NavGraph()
    g.add_edge("a", "b", "north")
    g.add_edge("b", "c", "east")
    assert detect_direction_conflicts(g) == []
    assert detect_topology_conflicts(g) == []
    assert detect_naming_conflicts(g) == []


def test_direction_conflict_detected():
    g = NavGraph()
    g.add_edge("a", "b", "north", add_auto_reverse=False)
    g.add_edge("a", "c", "north", add_auto_reverse=False)
    cs = detect_direction_conflicts(g)
    assert len(cs) == 1
    c = cs[0]
    assert c.type == "direction"
    assert c.details["direction"] == "north"
    assert {"b", "c"}.issubset(set(c.details["targets"].split(",")))


def test_topology_reverse_mismatch():
    g = NavGraph()
    # Manually create a mismatched reverse
    g.add_edge("a", "b", "north", add_auto_reverse=False)
    g.add_edge("b", "a", "east", add_auto_reverse=False)  # should be 'south'
    cs = detect_topology_conflicts(g)
    assert any(c.type == "topology" and "reverse direction mismatch" in c.description for c in cs)


def test_topology_position_overlap():
    g = NavGraph()
    # a --north--> b and b --east--> c and a --east--> b' and b' --north--> c'
    # produces c and c' at same coordinate (1, 1, 0) if they were the same node;
    # to actually trigger a position-overlap we need two distinct nodes at same coord:
    g.add_edge("a", "b", "north", add_auto_reverse=False)
    g.add_edge("b", "c", "east", add_auto_reverse=False)
    g.add_edge("a", "d", "east", add_auto_reverse=False)
    g.add_edge("d", "e", "north", add_auto_reverse=False)
    # b: (0,1,0); c: (1,1,0); d: (1,0,0); e: (1,1,0) -> c & e collide
    cs = detect_topology_conflicts(g)
    assert any(c.type == "topology" and "position" in c.description and "occupied" in c.description
               for c in cs)


def test_naming_conflict_detected():
    g = NavGraph()
    # Loop with mismatched directions -> same name reached at two coords
    g.add_edge("a", "b", "north", add_auto_reverse=False)
    g.add_edge("b", "c", "north", add_auto_reverse=False)
    # alternate path to c that puts it east of a -> conflict
    g.add_edge("a", "d", "east", add_auto_reverse=False)
    g.add_edge("d", "c", "east", add_auto_reverse=False)
    cs = detect_naming_conflicts(g)
    assert any(c.type == "naming" for c in cs)


def test_detect_all_dedupes():
    g = NavGraph()
    g.add_edge("a", "b", "north", add_auto_reverse=False)
    g.add_edge("a", "c", "north", add_auto_reverse=False)
    all_ = detect_all(g)
    ids = [c.conflict_id() for c in all_]
    assert len(ids) == len(set(ids))
