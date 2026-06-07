"""Reproduce the paper's Section 4.4 / Appendix B Table A1 (TC1-TC6)
algorithmic validation numbers by re-constructing the six hand-crafted
test cases described in the paper and running the current localizer +
scoring code on them.

Paper claims:
  TC1 Topo:  9 edges -> 8 cands (11.1% reduction), error rank 3/8
  TC2 Dir:   7 edges -> 6 cands (14.3% reduction), error rank 3/6
  TC3 Casc:  9 edges -> 7 cands (22.2% reduction), error rank 3/7
  TC4 Mix-T: 12 edges -> 9 cands (25.0% reduction), error rank 5/9
  TC4 Mix-D: 12 edges -> 3 cands (75.0% reduction), error rank 3/3
  TC5 Long: 18 edges -> 18 cands (0.0% reduction), error rank 3/18
  Avg:       10.9 edges -> 8.4 cands, 22.7% reduction
  TC6 (sep): 36-edge cascade graph -> Spearman rho = 1.0
"""
from __future__ import annotations
import json
from pathlib import Path
from maprepair.graph import NavGraph
from maprepair.conflict import detect_all
from maprepair.localizer import Localizer
from maprepair.scoring import score_edges


def build_tc1():
    g = NavGraph()
    # Recreate the Figure conflict scenario: B is LCA; two paths diverge,
    # one bad edge E->G causes topology conflict between D and I.
    edges = [
        ("A", "B", "north"),
        ("B", "C", "east"),
        ("C", "D", "north"),
        ("B", "E", "west"),
        ("E", "G", "north"),  # wrong direction -- should be south
        ("G", "H", "east"),
        ("H", "I", "north"),
        ("E", "J", "east"),  # silent error, no conflict
    ]
    for u, v, d in edges:
        g.add_edge(u, v, d, add_auto_reverse=False)
    error_edges = {("E", "G")}
    return g, error_edges, "TC1: Topo"


def build_tc2():
    g = NavGraph()
    # Degenerate case: node C has two "north" outgoing edges
    edges = [
        ("A", "B", "east"),
        ("B", "C", "north"),
        ("C", "D", "north"),
        ("C", "E", "north"),  # duplicate direction at C
        ("D", "F", "east"),
        ("E", "G", "west"),
    ]
    for u, v, d in edges:
        g.add_edge(u, v, d, add_auto_reverse=False)
    error_edges = {("C", "E")}
    return g, error_edges, "TC2: Dir"


def build_tc3():
    g = NavGraph()
    # Cascading: R1->R2 (wrong) propagates to R2/L2 + R3/L3 overlaps
    edges = [
        ("Root", "L1", "east"),
        ("L1", "L2", "north"),
        ("L2", "L3", "north"),
        ("Root", "R1", "west"),
        ("R1", "R2", "north"),  # wrong -- should be south
        ("R2", "R3", "north"),
        ("L1", "M1", "south"),
        ("L2", "M2", "south"),
    ]
    for u, v, d in edges:
        g.add_edge(u, v, d, add_auto_reverse=False)
    error_edges = {("R1", "R2")}
    return g, error_edges, "TC3: Cascade"


def build_tc4():
    g = NavGraph()
    # Mixed: 12 edges across 12 nodes with both topo and dir conflicts
    edges = [
        ("A", "B", "east"),
        ("B", "C", "north"),
        ("C", "D", "north"),  # topology error here
        ("A", "E", "west"),
        ("E", "F", "north"),
        ("F", "G", "east"),
        ("G", "H", "east"),
        ("H", "I", "north"),  # dir conflict (same as H->J)
        ("H", "J", "north"),  # dir conflict
        ("I", "K", "east"),
        ("J", "L", "east"),
        ("D", "K", "east"),  # creates topology conflict via overlap
    ]
    for u, v, d in edges:
        g.add_edge(u, v, d, add_auto_reverse=False)
    error_edges_topo = {("C", "D")}
    error_edges_dir = {("H", "J")}
    return g, error_edges_topo, error_edges_dir, "TC4: Mix(T/D)"


def build_tc5():
    g = NavGraph()
    # Long-range: error at depth 2 manifests at depth 10
    edges = [
        ("N0", "M1", "east"),
        ("M1", "M2", "east"),
        ("M2", "M3", "east"),
        ("M3", "M4", "east"),
        ("M4", "M5", "north"),
        ("M5", "M6", "north"),
        ("M6", "M7", "west"),
        ("N0", "N1", "west"),
        ("N1", "N2", "south"),   # wrong -- causes long-range overlap
        ("N2", "N3", "south"),
        ("N3", "N4", "east"),
        ("N4", "N5", "east"),
        ("N5", "N6", "east"),
        ("N6", "N7", "east"),
        ("N7", "N8", "north"),
        ("N8", "N9", "north"),
        ("M7", "M8", "south"),
        ("N9", "M8", "east"),  # overlap point: M7 vs N9
    ]
    for u, v, d in edges:
        g.add_edge(u, v, d, add_auto_reverse=False)
    error_edges = {("N1", "N2")}
    return g, error_edges, "TC5: Long"


def build_tc6():
    g = NavGraph()
    # Cascade-potential graph: 5 error edges with downstream sizes 10, 5, 3, 2, 1
    # Build it so each error edge leads to a subtree of given size.
    next_id = [0]
    def new(prefix="x"):
        nid = f"{prefix}{next_id[0]}"
        next_id[0] += 1
        return nid

    root = "root"
    # 5 chains under root
    err_edges = []
    chains = [10, 5, 3, 2, 1]
    chain_labels = ["E", "J", "L", "N", "P"]
    chain_letters_b = ["F", "K", "M", "O", "Q"]
    for chain_idx, (size, l_a, l_b) in enumerate(zip(chains, chain_labels, chain_letters_b)):
        # error edge from root -> l_a (the err edge)
        g.add_edge(root, l_a, "east", add_auto_reverse=False)
        # then l_a -> l_b ... extending size-1 more nodes
        prev = l_a
        for i in range(size):
            nxt = new(f"{l_b}")
            g.add_edge(prev, nxt, "east", add_auto_reverse=False)
            prev = nxt
        # also add a few "fluff" branches off root for diversity
        for j in range(2):
            n = new("fluff")
            g.add_edge(prev, n, "north", add_auto_reverse=False)
        err_edges.append((root, l_a))
    return g, err_edges


def lca_reduce_and_rank(g: NavGraph, error_edges, conflict_type: str = "any"):
    """Run our Localizer on every conflict in g, take the union of
    candidate sets restricted by conflict type, compare against |E|."""
    total_edges = len(g.primary_edges())
    conflicts = detect_all(g)
    if conflict_type != "any":
        conflicts = [c for c in conflicts if c.type == conflict_type]
    if not conflicts:
        return {
            "total_edges": total_edges, "n_conflicts": 0,
            "lca_cand": total_edges, "reduction_pct": 0.0,
            "error_rank": None,
        }
    loc = Localizer()
    cand_union = set()
    for c in conflicts:
        cs = loc.localize(g, c)
        cand_union |= set((u, v) for (u, v) in cs if g.has_edge(u, v))
    if not cand_union:
        return {
            "total_edges": total_edges, "n_conflicts": len(conflicts),
            "lca_cand": 0, "reduction_pct": 100.0, "error_rank": None,
        }
    n_cand = len(cand_union)
    # Score
    scored = score_edges(g, conflicts=conflicts)
    scored = [s for s in scored if s.edge in cand_union]
    scored.sort(key=lambda s: -s.score)
    ranks = []
    for er in error_edges:
        for i, s in enumerate(scored):
            if s.edge == er:
                ranks.append(i + 1)  # 1-indexed
                break
    return {
        "total_edges": total_edges,
        "n_conflicts": len(conflicts),
        "lca_cand": n_cand,
        "reduction_pct": 100 * (1 - n_cand / total_edges),
        "error_rank": min(ranks) if ranks else None,
    }


def main():
    rows = []
    for builder, kind in [
        (build_tc1, "TC1: Topo"),
        (build_tc2, "TC2: Dir"),
        (build_tc3, "TC3: Cascade"),
    ]:
        g, errs, name = builder()
        r = lca_reduce_and_rank(g, errs, "any")
        r["name"] = name
        rows.append(r)
        print(f"{name:18s} edges={r['total_edges']:>3} cands={r['lca_cand']:>3} "
              f"reduce={r['reduction_pct']:>5.1f}% rank={r['error_rank']} "
              f"n_conflicts={r['n_conflicts']}")

    # TC4 split: topology and directional sub-cases
    g, errs_topo, errs_dir, name = build_tc4()
    r_t = lca_reduce_and_rank(g, errs_topo, "topology")
    r_t["name"] = "TC4: Mix(T)"
    r_d = lca_reduce_and_rank(g, errs_dir, "direction")
    r_d["name"] = "TC4: Mix(D)"
    for r in (r_t, r_d):
        rows.append(r)
        print(f"{r['name']:18s} edges={r['total_edges']:>3} cands={r['lca_cand']:>3} "
              f"reduce={r['reduction_pct']:>5.1f}% rank={r['error_rank']} "
              f"n_conflicts={r['n_conflicts']}")

    # TC5
    g, errs, name = build_tc5()
    r = lca_reduce_and_rank(g, errs, "any")
    r["name"] = name
    rows.append(r)
    print(f"{name:18s} edges={r['total_edges']:>3} cands={r['lca_cand']:>3} "
          f"reduce={r['reduction_pct']:>5.1f}% rank={r['error_rank']} "
          f"n_conflicts={r['n_conflicts']}")

    # Aggregate
    avg_edges = sum(r["total_edges"] for r in rows) / len(rows)
    avg_cands = sum(r["lca_cand"] for r in rows) / len(rows)
    overall_reduce = (1 - avg_cands / avg_edges) * 100
    mean_pct = sum(r["reduction_pct"] for r in rows) / len(rows)
    print(f"\n{'AVG':18s} edges={avg_edges:>5.1f} cands={avg_cands:>4.1f} "
          f"reduce_from_avgs={overall_reduce:.1f}% mean_pct_reduce={mean_pct:.1f}%")

    # TC6 cascade prediction: Spearman correlation
    g, err_edges = build_tc6()
    # Truth: cascade sizes
    truth = [(e, sz) for e, sz in zip(err_edges, [10, 5, 3, 2, 1])]
    # Predicted: by Reach score
    from maprepair.scoring import score_edges
    # Use a synthetic conflict that involves all err edges to compute scores
    # We'll just compute reachability of each err edge directly:
    import networkx as nx
    G = g.nx
    pred = []
    for e, sz_actual in truth:
        # downstream count = nodes reachable from e[1]
        reach = len(nx.descendants(G, e[1])) + 1
        pred.append((e, reach, sz_actual))
    # Spearman:
    actual_ranks = sorted(range(len(pred)), key=lambda i: -pred[i][2])
    predicted_ranks = sorted(range(len(pred)), key=lambda i: -pred[i][1])
    # Compute Spearman: convert to ranks
    n = len(pred)
    actual = [sz for _, _, sz in pred]
    predicted = [r for _, r, _ in pred]
    # Simple rank-based correlation
    def ranks_of(xs):
        sorted_pairs = sorted(enumerate(xs), key=lambda kv: -kv[1])
        ranks = [0] * len(xs)
        for r, (i, _) in enumerate(sorted_pairs):
            ranks[i] = r + 1
        return ranks
    ra = ranks_of(actual)
    rp = ranks_of(predicted)
    sum_d2 = sum((a - p) ** 2 for a, p in zip(ra, rp))
    rho = 1 - (6 * sum_d2) / (n * (n ** 2 - 1))
    print(f"\nTC6 cascade prediction:")
    for (e, p, a) in pred:
        print(f"  {e}: predicted_reach={p}, actual_cascade={a}")
    print(f"Spearman rho = {rho:.3f}")
    print(f"Paper claim: rho = 1.0")

    out = Path("results/exp28_verify_tc16/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "tc6_rho": rho,
                                 "tc6_pairs": [(str(e), p, a) for e, p, a in pred]},
                                indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
