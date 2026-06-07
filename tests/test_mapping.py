import json
from pathlib import Path

import pytest

from maprepair.mapping import load_legacy_edges


def test_load_legacy_edges(tmp_path: Path):
    sample = [
        {"src_node": "a", "dst_node": "b", "action": "north", "seen_in_forward": 1},
        {"src_node": "b", "dst_node": "c", "action": "east", "seen_in_forward": 2},
        {"src_node": "<unknown>", "dst_node": "x", "action": "north", "seen_in_forward": 3},
    ]
    p = tmp_path / "g.json"
    p.write_text(json.dumps(sample))
    g = load_legacy_edges(p)
    assert g.num_nodes() == 3
    primary = g.primary_edges()
    assert any(e.source == "a" and e.target == "b" for e in primary)
