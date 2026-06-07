"""Smoke tests for the LLM repair agent.

We do NOT call the real LLM here. Instead we monkey-patch the chat_json
function to return a deterministic JSON action, then verify that the agent
correctly applies it.
"""

import maprepair.agents.llm_agent as llm_agent_mod
from maprepair.agents.llm_agent import LLMRepairAgent
from maprepair.conflict import detect_all
from maprepair.synth import gen_tree, inject_direction_errors


def _stub_chat_json_reply(reply):
    def f(messages, **kwargs):
        return reply
    return f


def test_llm_agent_applies_modify_edge_action(monkeypatch):
    g = gen_tree(depth=4, branching=3, seed=0)
    injected = inject_direction_errors(g, n=1, seed=0)
    bad_src, bad_dst, _ = injected[0]
    # Force the LLM stub to always rotate the bad edge to "south"
    reply = {
        "action": "modify_edge",
        "edge": [bad_src, bad_dst],
        "new_direction": "south",
        "reason": "stub",
    }
    monkeypatch.setattr(llm_agent_mod, "chat_json", _stub_chat_json_reply(reply))
    agent = LLMRepairAgent(model="gpt-4.1", mode="baseline")
    result = agent.repair(g, max_iterations=3)
    assert result.iterations >= 1
    assert any(a.kind == "modify_edge" for a in result.actions)


def test_llm_agent_handles_skip(monkeypatch):
    g = gen_tree(depth=4, branching=3, seed=1)
    inject_direction_errors(g, n=1, seed=1)
    reply = {"action": "skip_conflict", "reason": "stub-skip"}
    monkeypatch.setattr(llm_agent_mod, "chat_json", _stub_chat_json_reply(reply))
    agent = LLMRepairAgent(model="gpt-4.1", mode="baseline",
                             max_attempts_per_conflict=2)
    result = agent.repair(g, max_iterations=3)
    # graph should be unchanged (skips don't mutate)
    assert result.graph_after.num_edges() == g.num_edges()
