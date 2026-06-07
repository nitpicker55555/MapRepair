from maprepair.conflict import detect_all
from maprepair.synth import gen_tree, inject_direction_errors, inject_topology_errors
from maprepair.agents.heuristic import HeuristicRepairAgent


def test_heuristic_fixes_direction_conflict():
    g = gen_tree(depth=4, branching=3, seed=0)
    inject_direction_errors(g, n=1, seed=0)
    pre = detect_all(g)
    assert pre, "expected at least one conflict"
    agent = HeuristicRepairAgent()
    result = agent.repair(g, max_iterations=10)
    assert result.iterations >= 1
    # the agent must mutate something
    assert len(result.actions) >= 1


def test_heuristic_terminates_on_clean_graph():
    g = gen_tree(depth=4, branching=3, seed=1)
    agent = HeuristicRepairAgent()
    result = agent.repair(g, max_iterations=10)
    assert result.iterations == 0
    assert result.success is True
    assert len(result.actions) == 0


def test_heuristic_reports_actions():
    g = gen_tree(depth=4, branching=3, seed=2)
    inject_direction_errors(g, n=1, seed=2)
    agent = HeuristicRepairAgent()
    result = agent.repair(g, max_iterations=10)
    assert result.summary()["iterations"] >= 1
    assert isinstance(result.summary()["actions"], list)
