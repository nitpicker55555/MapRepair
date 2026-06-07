# Experiment 16 — controlled-noise benchmark

Graphs:  ['grid_4x4', 'grid_5x6', 'tree_d3b3', 'random_24', 'random_40']
Regimes: ['edge_minimal', 'edge_clean', 'edge_heavy', 'node_only', 'node_heavy', 'mango_like']
Methods: ['no_repair', 'random', 'heuristic_remove', 'heuristic_modify']
Seeds:   [0, 1, 2]
Total runs: 360 (errors: 0)

## Aggregate by regime × method

| regime | method | n | conflict-free % | conf reduction % | GT edge recall after | GT edge prec after | GT node recall after | actions/run | iters/run |
|--------|--------|--:|----------------:|-----------------:|---------------------:|-------------------:|---------------------:|------------:|----------:|
| edge_minimal | no_repair | 15 | 0.0 | 0.0 | 100.0 | 94.2 | 100.0 | 0.0 | 0.0 |
| edge_minimal | random | 15 | 0.0 | 78.8 | 30.4 | 83.3 | 100.0 | 0.0 | 49.2 |
| edge_minimal | heuristic_remove | 15 | 60.0 | 91.9 | 92.1 | 97.1 | 100.0 | 5.7 | 5.3 |
| edge_minimal | heuristic_modify | 15 | 6.7 | 67.4 | 98.5 | 94.1 | 100.0 | 11.0 | 10.1 |
| edge_clean | no_repair | 15 | 0.0 | 0.0 | 94.7 | 81.8 | 100.0 | 0.0 | 0.0 |
| edge_clean | random | 15 | 0.0 | 75.3 | 37.8 | 81.8 | 100.0 | 0.0 | 50.0 |
| edge_clean | heuristic_remove | 15 | 0.0 | 68.1 | 78.7 | 87.0 | 100.0 | 14.5 | 13.5 |
| edge_clean | heuristic_modify | 15 | 6.7 | 63.2 | 91.2 | 81.4 | 100.0 | 21.6 | 20.7 |
| edge_heavy | no_repair | 15 | 0.0 | 0.0 | 91.4 | 65.1 | 100.0 | 0.0 | 0.0 |
| edge_heavy | random | 15 | 0.0 | 76.0 | 36.7 | 64.4 | 100.0 | 0.0 | 50.0 |
| edge_heavy | heuristic_remove | 15 | 13.3 | 77.3 | 68.2 | 66.7 | 100.0 | 23.7 | 22.9 |
| edge_heavy | heuristic_modify | 15 | 6.7 | 60.9 | 83.7 | 63.2 | 100.0 | 30.5 | 29.7 |
| node_only | no_repair | 15 | 0.0 | 0.0 | 39.5 | 42.7 | 67.3 | 0.0 | 0.0 |
| node_only | random | 15 | 0.0 | 83.4 | 8.8 | 22.1 | 67.3 | 0.0 | 47.6 |
| node_only | heuristic_remove | 15 | 0.0 | 66.5 | 27.0 | 41.6 | 67.3 | 15.6 | 14.6 |
| node_only | heuristic_modify | 15 | 0.0 | 49.6 | 38.2 | 42.7 | 67.3 | 17.1 | 16.1 |
| node_heavy | no_repair | 15 | 0.0 | 0.0 | 20.8 | 25.2 | 47.9 | 0.0 | 0.0 |
| node_heavy | random | 15 | 0.0 | 81.6 | 3.6 | 8.9 | 47.9 | 0.0 | 46.5 |
| node_heavy | heuristic_remove | 15 | 6.7 | 70.8 | 13.6 | 24.8 | 47.9 | 15.5 | 14.5 |
| node_heavy | heuristic_modify | 15 | 6.7 | 56.7 | 18.9 | 24.9 | 47.9 | 16.3 | 15.3 |
| mango_like | no_repair | 15 | 0.0 | 0.0 | 38.2 | 34.5 | 67.3 | 0.0 | 0.0 |
| mango_like | random | 15 | 0.0 | 65.6 | 12.8 | 27.9 | 67.3 | 0.0 | 50.0 |
| mango_like | heuristic_remove | 15 | 0.0 | 67.0 | 25.9 | 34.8 | 67.3 | 19.3 | 18.3 |
| mango_like | heuristic_modify | 15 | 0.0 | 55.6 | 34.7 | 33.8 | 67.3 | 22.3 | 21.3 |

## Method lift over no_repair (per regime, edge-recall delta)

| regime | no_repair | random | heur_remove | heur_modify | best lift |
|--------|----------:|-------:|------------:|------------:|----------:|
| edge_minimal | 100.0 | 30.4 | 92.1 | 98.5 | -1.5pp (heuristic_modify) |
| edge_clean | 94.7 | 37.8 | 78.7 | 91.2 | -3.5pp (heuristic_modify) |
| edge_heavy | 91.4 | 36.7 | 68.2 | 83.7 | -7.7pp (heuristic_modify) |
| node_only | 39.5 | 8.8 | 27.0 | 38.2 | -1.3pp (heuristic_modify) |
| node_heavy | 20.8 | 3.6 | 13.6 | 18.9 | -1.9pp (heuristic_modify) |
| mango_like | 38.2 | 12.8 | 25.9 | 34.7 | -3.5pp (heuristic_modify) |

## Mean injection counts per regime

| regime | total | N1_direction | N2_topology | N3_endpoint_swap | N4_spurious_pair | N5_node_collapse | N5_collapse_fallback | N6_duplicate | N7_hallucinated |
|--------|------:|---:|---:|---:|---:|---:|---:|---:|---:|
| edge_minimal | 3.6 | 1.8 | 1.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| edge_clean | 11.6 | 4.0 | 4.0 | 1.8 | 1.8 | 0.0 | 0.0 | 0.0 | 0.0 |
| edge_heavy | 30.0 | 11.0 | 11.0 | 4.0 | 4.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| node_only | 24.1 | 0.0 | 0.0 | 0.0 | 0.0 | 5.0 | 1.4 | 16.7 | 1.0 |
| node_heavy | 36.9 | 0.0 | 0.0 | 0.0 | 0.0 | 5.0 | 5.4 | 24.1 | 2.4 |
| mango_like | 36.2 | 5.5 | 3.5 | 1.5 | 1.5 | 5.0 | 1.4 | 16.7 | 1.0 |

## Per-graph aggregate (edge-recall after repair, per method)

| graph | regime | no_repair | random | heur_remove | heur_modify |
|-------|--------|----------:|-------:|------------:|------------:|
| grid_4x4 | edge_minimal | 100.0 | 23.6 | 91.7 | 94.4 |
| grid_4x4 | edge_clean | 93.1 | 30.6 | 79.9 | 87.5 |
| grid_4x4 | edge_heavy | 86.8 | 25.7 | 59.7 | 74.3 |
| grid_4x4 | node_only | 44.4 | 4.2 | 34.7 | 42.4 |
| grid_4x4 | node_heavy | 29.2 | 0.0 | 18.1 | 25.0 |
| grid_4x4 | mango_like | 41.0 | 10.4 | 31.9 | 34.7 |
| grid_5x6 | edge_minimal | 100.0 | 59.9 | 95.6 | 98.0 |
| grid_5x6 | edge_clean | 91.8 | 60.5 | 78.6 | 82.7 |
| grid_5x6 | edge_heavy | 85.7 | 57.1 | 63.9 | 76.5 |
| grid_5x6 | node_only | 42.2 | 21.1 | 30.6 | 37.8 |
| grid_5x6 | node_heavy | 19.7 | 9.5 | 13.9 | 14.6 |
| grid_5x6 | mango_like | 42.2 | 27.9 | 31.3 | 35.7 |
| tree_d3b3 | edge_minimal | 100.0 | 9.1 | 83.3 | 100.0 |
| tree_d3b3 | edge_clean | 95.5 | 18.2 | 74.2 | 95.5 |
| tree_d3b3 | edge_heavy | 95.5 | 25.8 | 66.7 | 93.9 |
| tree_d3b3 | node_only | 33.3 | 0.0 | 18.2 | 33.3 |
| tree_d3b3 | node_heavy | 18.2 | 0.0 | 6.1 | 18.2 |
| tree_d3b3 | mango_like | 31.8 | 6.1 | 15.2 | 30.3 |
| random_24 | edge_minimal | 100.0 | 13.0 | 98.6 | 100.0 |
| random_24 | edge_clean | 95.7 | 26.1 | 79.7 | 95.7 |
| random_24 | edge_heavy | 94.2 | 24.6 | 78.3 | 84.1 |
| random_24 | node_only | 39.1 | 0.0 | 23.2 | 39.1 |
| random_24 | node_heavy | 23.2 | 0.0 | 18.8 | 23.2 |
| random_24 | mango_like | 39.1 | 1.4 | 24.6 | 37.7 |
| random_40 | edge_minimal | 100.0 | 46.2 | 91.5 | 100.0 |
| random_40 | edge_clean | 97.4 | 53.8 | 81.2 | 94.9 |
| random_40 | edge_heavy | 94.9 | 50.4 | 72.6 | 89.7 |
| random_40 | node_only | 38.5 | 18.8 | 28.2 | 38.5 |
| random_40 | node_heavy | 13.7 | 8.5 | 11.1 | 13.7 |
| random_40 | mango_like | 36.8 | 17.9 | 26.5 | 35.0 |
