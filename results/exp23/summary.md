# Experiment 23 — Controlled noise on TextWorld GT (model=gpt-4.1)

Total runs: 600 (valid 600)

Games: 10 TextWorld games (sizes 5–11, real room names)
Regimes: ('edge_minimal', 'edge_clean', 'edge_heavy', 'node_only', 'mango_like')
Methods: ('no_repair', 'heuristic_remove', 'llm_baseline', 'llm_edge_impact')
Seeds per cell: 3

## Aggregate by (regime × method)

| regime | method | n | CF % | conf reduction % | edge recall after | edge recall Δ | dir acc | node recall | iters | rm/run |
|--------|--------|--:|-----:|----------------:|------------------:|--------------:|--------:|------------:|------:|-------:|
| edge_minimal | no_repair | 30 | 80.0 | 0.0 | 100.0 | +0.00pp | 99.1 | 100.0 | 0.0 | 0.0 |
| edge_minimal | heuristic_remove | 30 | 96.7 | 18.7 | 98.4 | -1.58pp | 99.7 | 100.0 | 0.6 | 0.5 |
| edge_minimal | llm_baseline | 30 | 93.3 | 15.2 | 99.7 | -0.29pp | 99.1 | 100.0 | 1.2 | 0.2 |
| edge_minimal | llm_edge_impact | 30 | 96.7 | 19.9 | 99.3 | -0.71pp | 99.2 | 100.0 | 0.6 | 0.4 |
| edge_clean | no_repair | 30 | 20.0 | 0.0 | 98.8 | +0.00pp | 94.0 | 100.0 | 0.0 | 0.0 |
| edge_clean | heuristic_remove | 30 | 80.0 | 74.2 | 86.6 | -12.21pp | 96.3 | 100.0 | 2.9 | 2.7 |
| edge_clean | llm_baseline | 30 | 20.0 | 24.2 | 97.0 | -1.80pp | 90.4 | 100.0 | 8.7 | 0.4 |
| edge_clean | llm_edge_impact | 30 | 70.0 | 71.7 | 87.9 | -10.97pp | 91.5 | 100.0 | 6.8 | 2.7 |
| edge_heavy | no_repair | 30 | 0.0 | 0.0 | 90.7 | +0.00pp | 75.4 | 100.0 | 0.0 | 0.0 |
| edge_heavy | heuristic_remove | 30 | 43.3 | 88.8 | 61.3 | -29.45pp | 84.4 | 100.0 | 7.2 | 6.7 |
| edge_heavy | llm_baseline | 30 | 10.0 | 44.6 | 82.9 | -7.82pp | 65.0 | 100.0 | 15.4 | 1.8 |
| edge_heavy | llm_edge_impact | 30 | 36.7 | 86.3 | 64.4 | -26.30pp | 73.6 | 100.0 | 13.3 | 6.4 |
| node_only | no_repair | 30 | 26.7 | 0.0 | 58.4 | +0.00pp | 96.3 | 79.8 | 0.0 | 0.0 |
| node_only | heuristic_remove | 30 | 60.0 | 50.4 | 53.3 | -5.04pp | 96.1 | 79.8 | 1.8 | 1.5 |
| node_only | llm_baseline | 30 | 33.3 | 20.4 | 57.7 | -0.68pp | 93.0 | 79.8 | 5.8 | 0.4 |
| node_only | llm_edge_impact | 30 | 63.3 | 54.3 | 49.9 | -8.51pp | 94.0 | 79.8 | 5.0 | 2.3 |
| mango_like | no_repair | 30 | 16.7 | 0.0 | 58.4 | +0.00pp | 91.4 | 79.8 | 0.0 | 0.0 |
| mango_like | heuristic_remove | 30 | 56.7 | 69.0 | 49.2 | -9.15pp | 93.4 | 79.8 | 3.9 | 3.5 |
| mango_like | llm_baseline | 30 | 20.0 | 37.4 | 57.0 | -1.33pp | 84.8 | 79.8 | 9.5 | 0.7 |
| mango_like | llm_edge_impact | 30 | 66.7 | 77.2 | 47.6 | -10.75pp | 92.1 | 79.8 | 6.5 | 3.8 |

## Method lift over no_repair (per regime, CF %)

| regime | no_repair | heur_remove | llm_baseline | llm_edge_impact | best lift |
|--------|----------:|------------:|-------------:|----------------:|----------:|
| edge_minimal | 80.0 | 96.7 | 93.3 | 96.7 | **+16.7pp (heuristic_remove)** |
| edge_clean | 20.0 | 80.0 | 20.0 | 70.0 | **+60.0pp (heuristic_remove)** |
| edge_heavy | 0.0 | 43.3 | 10.0 | 36.7 | **+43.3pp (heuristic_remove)** |
| node_only | 26.7 | 60.0 | 33.3 | 63.3 | **+36.7pp (llm_edge_impact)** |
| mango_like | 16.7 | 56.7 | 20.0 | 66.7 | **+50.0pp (llm_edge_impact)** |
