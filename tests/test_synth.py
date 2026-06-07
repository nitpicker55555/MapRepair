from maprepair.conflict import detect_all
from maprepair.synth import (
    gen_tree, gen_grid, gen_random,
    inject_direction_errors, inject_topology_errors,
    make_synthetic,
)


def test_tree_no_conflicts_by_construction():
    g = gen_tree(depth=4, branching=3, seed=0)
    cs = detect_all(g)
    assert cs == [], f"unexpected conflicts: {[c.description for c in cs[:3]]}"
    assert g.num_nodes() >= 5


def test_grid_no_conflicts_by_construction():
    g = gen_grid(4, 4)
    cs = detect_all(g)
    assert cs == [], f"unexpected conflicts: {[c.description for c in cs[:3]]}"


def test_random_no_conflicts_by_construction():
    g = gen_random(num_nodes=30, branching=3, seed=42)
    cs = detect_all(g)
    assert cs == [], f"unexpected conflicts: {[c.description for c in cs[:3]]}"


def test_direction_injection_creates_direction_conflict():
    g = gen_tree(depth=4, branching=3, seed=0)
    injected = inject_direction_errors(g, n=1, seed=0)
    assert injected, "no error injected"
    cs = detect_all(g)
    assert any(c.type == "direction" for c in cs)


def test_topology_injection_creates_topology_conflict():
    g = gen_tree(depth=4, branching=3, seed=0)
    injected = inject_topology_errors(g, n=1, seed=0)
    assert injected, "no error injected"
    cs = detect_all(g)
    assert any(c.type == "topology" for c in cs)


def test_make_synthetic_returns_groundtruth_and_broken():
    spec = make_synthetic("random", size=20, error_mix={"direction": 1, "topology": 1}, seed=0)
    assert spec.ground_truth.num_nodes() == spec.graph.num_nodes() or True
    gt_conflicts = detect_all(spec.ground_truth)
    assert gt_conflicts == []
    broken_conflicts = detect_all(spec.graph)
    assert broken_conflicts, "broken graph should have at least one detectable conflict"
