import os
import pytest
from pathlib import Path

from maprepair import mango


def _has_data() -> bool:
    return (mango.data_root() / "data_fixed").exists()


@pytest.mark.skipif(not _has_data(), reason="data_fixed not available")
def test_list_games_includes_zork1():
    games = mango.list_games()
    assert "zork1" in games


@pytest.mark.skipif(not _has_data(), reason="data_fixed not available")
def test_ground_truth_graph_zork1():
    g = mango.ground_truth_graph("zork1")
    assert g.num_nodes() > 0
    assert g.num_edges() > 0


@pytest.mark.skipif(not _has_data(), reason="data_fixed not available")
def test_parse_walkthrough_zork1():
    text = mango.load_repaired_walkthrough("zork1")
    if text is None:
        pytest.skip("zork1 walkthrough missing")
    steps = mango.parse_walkthrough(text, max_steps=20)
    assert any(s.step_num == 0 for s in steps)
    assert any(s.action.lower() == "north" for s in steps)
