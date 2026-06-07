# Experiment 18 — LLM + heuristic ablation, depth study (model=gpt-4.1)

Total runs: 1632  (valid 1632, errors 0, skipped 0)

Core configs: 16 × 15 seeds.
Large configs: 4 × 8 seeds.

## Aggregate by mode

| mode | n | conflict-free % | GT edge recall % | GT dir acc % | mean iters | mean actions |
|------|--:|----------------:|-----------------:|-------------:|-----------:|-------------:|
| baseline | 272 | 36.8 | 100.0 | 99.2 | 6.4 | 2.8 |
| edge_impact | 272 | 37.9 | 100.0 | 93.9 | 12.2 | 9.9 |
| vc_only | 272 | 36.8 | 100.0 | 99.0 | 7.4 | 3.5 |
| vc_ei | 272 | 36.8 | 100.0 | 96.0 | 11.3 | 8.6 |
| heuristic_remove | 272 | 68.4 | 96.0 | 99.1 | 3.2 | 3.5 |
| heuristic_modify | 272 | 43.0 | 99.3 | 90.6 | 7.1 | 7.6 |

## By mode × err_type

| mode | err_type | n | CF % | GT recall | dir acc | iters |
|------|----------|--:|-----:|----------:|--------:|------:|
| baseline | direction | 136 | 5.9 | 100.0 | 99.9 | 7.9 |
| baseline | topology | 136 | 67.6 | 100.0 | 98.4 | 5.0 |
| edge_impact | direction | 136 | 2.2 | 100.0 | 89.9 | 17.3 |
| edge_impact | topology | 136 | 73.5 | 100.0 | 97.8 | 7.1 |
| vc_only | direction | 136 | 5.1 | 100.0 | 99.9 | 9.2 |
| vc_only | topology | 136 | 68.4 | 100.0 | 98.1 | 5.6 |
| vc_ei | direction | 136 | 2.9 | 100.0 | 94.3 | 15.4 |
| vc_ei | topology | 136 | 70.6 | 100.0 | 97.6 | 7.3 |
| heuristic_remove | direction | 136 | 62.5 | 97.5 | 99.4 | 3.1 |
| heuristic_remove | topology | 136 | 74.3 | 94.6 | 98.7 | 3.2 |
| heuristic_modify | direction | 136 | 22.1 | 99.6 | 89.8 | 7.8 |
| heuristic_modify | topology | 136 | 64.0 | 99.0 | 91.4 | 6.4 |

## Detailed: family × err_type × size × num_errors × mode

| family | err | size | n_err | mode | n | CF % | GT recall | dir acc | iters |
|--------|-----|-----:|------:|------|--:|-----:|----------:|--------:|------:|
| grid | direction | 3 | 1 | baseline | 15 | 26.7 | 100.0 | 100.0 | 5.5 |
| grid | direction | 3 | 1 | edge_impact | 15 | 6.7 | 100.0 | 89.2 | 15.0 |
| grid | direction | 3 | 1 | heuristic_modify | 15 | 0.0 | 98.6 | 94.9 | 2.5 |
| grid | direction | 3 | 1 | heuristic_remove | 15 | 66.7 | 97.2 | 98.2 | 1.9 |
| grid | direction | 3 | 1 | vc_ei | 15 | 13.3 | 100.0 | 92.5 | 12.1 |
| grid | direction | 3 | 1 | vc_only | 15 | 20.0 | 100.0 | 100.0 | 6.3 |
| grid | direction | 4 | 1 | baseline | 15 | 20.0 | 100.0 | 99.7 | 6.5 |
| grid | direction | 4 | 1 | edge_impact | 15 | 13.3 | 100.0 | 92.5 | 16.9 |
| grid | direction | 4 | 1 | heuristic_modify | 15 | 6.7 | 99.3 | 95.1 | 4.4 |
| grid | direction | 4 | 1 | heuristic_remove | 15 | 100.0 | 100.0 | 100.0 | 1.0 |
| grid | direction | 4 | 1 | vc_ei | 15 | 13.3 | 100.0 | 97.6 | 14.1 |
| grid | direction | 4 | 1 | vc_only | 15 | 26.7 | 100.0 | 99.7 | 6.5 |
| grid | direction | 5 | 2 | baseline | 15 | 0.0 | 100.0 | 99.3 | 9.3 |
| grid | direction | 5 | 2 | edge_impact | 15 | 0.0 | 100.0 | 97.9 | 17.9 |
| grid | direction | 5 | 2 | heuristic_modify | 15 | 0.0 | 98.4 | 95.1 | 8.6 |
| grid | direction | 5 | 2 | heuristic_remove | 15 | 60.0 | 99.2 | 99.5 | 3.1 |
| grid | direction | 5 | 2 | vc_ei | 15 | 0.0 | 100.0 | 97.7 | 17.1 |
| grid | direction | 5 | 2 | vc_only | 15 | 0.0 | 100.0 | 99.8 | 8.9 |
| grid | topology | 3 | 1 | baseline | 15 | 60.0 | 100.0 | 96.7 | 5.7 |
| grid | topology | 3 | 1 | edge_impact | 15 | 60.0 | 100.0 | 95.6 | 8.2 |
| grid | topology | 3 | 1 | heuristic_modify | 15 | 53.3 | 98.1 | 93.9 | 2.1 |
| grid | topology | 3 | 1 | heuristic_remove | 15 | 86.7 | 93.3 | 98.4 | 1.7 |
| grid | topology | 3 | 1 | vc_ei | 15 | 53.3 | 100.0 | 93.1 | 10.0 |
| grid | topology | 3 | 1 | vc_only | 15 | 53.3 | 100.0 | 94.7 | 7.4 |
| grid | topology | 4 | 1 | baseline | 15 | 73.3 | 100.0 | 98.9 | 4.5 |
| grid | topology | 4 | 1 | edge_impact | 15 | 66.7 | 100.0 | 97.4 | 7.5 |
| grid | topology | 4 | 1 | heuristic_modify | 15 | 40.0 | 97.8 | 95.6 | 4.0 |
| grid | topology | 4 | 1 | heuristic_remove | 15 | 60.0 | 95.7 | 98.2 | 2.7 |
| grid | topology | 4 | 1 | vc_ei | 15 | 73.3 | 100.0 | 98.3 | 7.1 |
| grid | topology | 4 | 1 | vc_only | 15 | 66.7 | 100.0 | 97.9 | 6.7 |
| grid | topology | 5 | 2 | baseline | 15 | 53.3 | 100.0 | 97.8 | 10.6 |
| grid | topology | 5 | 2 | edge_impact | 15 | 26.7 | 100.0 | 96.7 | 14.7 |
| grid | topology | 5 | 2 | heuristic_modify | 15 | 26.7 | 98.2 | 95.3 | 8.1 |
| grid | topology | 5 | 2 | heuristic_remove | 15 | 60.0 | 94.8 | 98.4 | 5.3 |
| grid | topology | 5 | 2 | vc_ei | 15 | 40.0 | 100.0 | 97.3 | 14.4 |
| grid | topology | 5 | 2 | vc_only | 15 | 46.7 | 100.0 | 96.2 | 12.0 |
| random | direction | 24 | 1 | baseline | 15 | 6.7 | 100.0 | 100.0 | 6.8 |
| random | direction | 24 | 1 | edge_impact | 15 | 0.0 | 100.0 | 86.1 | 15.4 |
| random | direction | 24 | 1 | heuristic_modify | 15 | 40.0 | 100.0 | 84.6 | 5.9 |
| random | direction | 24 | 1 | heuristic_remove | 15 | 86.7 | 99.4 | 100.0 | 1.0 |
| random | direction | 24 | 1 | vc_ei | 15 | 0.0 | 100.0 | 93.3 | 13.7 |
| random | direction | 24 | 1 | vc_only | 15 | 0.0 | 100.0 | 100.0 | 8.0 |
| random | direction | 36 | 2 | baseline | 15 | 0.0 | 100.0 | 99.8 | 8.3 |
| random | direction | 36 | 2 | edge_impact | 15 | 0.0 | 100.0 | 87.4 | 18.7 |
| random | direction | 36 | 2 | heuristic_modify | 15 | 13.3 | 100.0 | 86.1 | 8.5 |
| random | direction | 36 | 2 | heuristic_remove | 15 | 53.3 | 97.3 | 98.8 | 3.1 |
| random | direction | 36 | 2 | vc_ei | 15 | 0.0 | 100.0 | 93.5 | 17.1 |
| random | direction | 36 | 2 | vc_only | 15 | 0.0 | 100.0 | 99.8 | 12.7 |
| random | direction | 60 | 4 | baseline | 8 | 0.0 | 100.0 | 100.0 | 10.0 |
| random | direction | 60 | 4 | edge_impact | 8 | 0.0 | 100.0 | 94.9 | 20.0 |
| random | direction | 60 | 4 | heuristic_modify | 8 | 12.5 | 100.0 | 88.3 | 14.4 |
| random | direction | 60 | 4 | heuristic_remove | 8 | 12.5 | 94.3 | 99.6 | 7.6 |
| random | direction | 60 | 4 | vc_ei | 8 | 0.0 | 100.0 | 95.8 | 16.9 |
| random | direction | 60 | 4 | vc_only | 8 | 0.0 | 100.0 | 99.8 | 13.1 |
| random | direction | 60 | 8 | baseline | 8 | 0.0 | 100.0 | 100.0 | 14.0 |
| random | direction | 60 | 8 | edge_impact | 8 | 0.0 | 100.0 | 93.6 | 20.0 |
| random | direction | 60 | 8 | heuristic_modify | 8 | 0.0 | 100.0 | 86.0 | 20.0 |
| random | direction | 60 | 8 | heuristic_remove | 8 | 25.0 | 94.5 | 98.9 | 11.6 |
| random | direction | 60 | 8 | vc_ei | 8 | 0.0 | 100.0 | 97.0 | 19.5 |
| random | direction | 60 | 8 | vc_only | 8 | 0.0 | 100.0 | 99.2 | 16.2 |
| random | topology | 24 | 1 | baseline | 15 | 93.3 | 100.0 | 98.6 | 1.5 |
| random | topology | 24 | 1 | edge_impact | 15 | 93.3 | 100.0 | 98.0 | 3.5 |
| random | topology | 24 | 1 | heuristic_modify | 15 | 66.7 | 99.7 | 89.8 | 2.5 |
| random | topology | 24 | 1 | heuristic_remove | 15 | 100.0 | 95.9 | 99.1 | 1.2 |
| random | topology | 24 | 1 | vc_ei | 15 | 93.3 | 100.0 | 99.4 | 3.3 |
| random | topology | 24 | 1 | vc_only | 15 | 93.3 | 100.0 | 99.1 | 1.5 |
| random | topology | 36 | 2 | baseline | 15 | 60.0 | 100.0 | 98.7 | 3.7 |
| random | topology | 36 | 2 | edge_impact | 15 | 86.7 | 100.0 | 97.9 | 5.5 |
| random | topology | 36 | 2 | heuristic_modify | 15 | 100.0 | 100.0 | 88.8 | 7.5 |
| random | topology | 36 | 2 | heuristic_remove | 15 | 73.3 | 93.5 | 98.2 | 3.1 |
| random | topology | 36 | 2 | vc_ei | 15 | 73.3 | 100.0 | 98.9 | 5.3 |
| random | topology | 36 | 2 | vc_only | 15 | 66.7 | 100.0 | 99.0 | 4.3 |
| random | topology | 60 | 4 | baseline | 8 | 37.5 | 100.0 | 98.1 | 7.8 |
| random | topology | 60 | 4 | edge_impact | 8 | 62.5 | 100.0 | 97.5 | 11.6 |
| random | topology | 60 | 4 | heuristic_modify | 8 | 75.0 | 98.7 | 89.5 | 16.2 |
| random | topology | 60 | 4 | heuristic_remove | 8 | 12.5 | 92.6 | 98.7 | 7.5 |
| random | topology | 60 | 4 | vc_ei | 8 | 75.0 | 100.0 | 98.7 | 9.8 |
| random | topology | 60 | 4 | vc_only | 8 | 50.0 | 100.0 | 98.3 | 7.4 |
| random | topology | 60 | 8 | baseline | 8 | 12.5 | 100.0 | 96.4 | 15.0 |
| random | topology | 60 | 8 | edge_impact | 8 | 50.0 | 100.0 | 96.0 | 16.6 |
| random | topology | 60 | 8 | heuristic_modify | 8 | 12.5 | 97.0 | 83.4 | 20.0 |
| random | topology | 60 | 8 | heuristic_remove | 8 | 75.0 | 87.1 | 96.6 | 11.6 |
| random | topology | 60 | 8 | vc_ei | 8 | 25.0 | 100.0 | 95.8 | 17.2 |
| random | topology | 60 | 8 | vc_only | 8 | 25.0 | 100.0 | 96.8 | 14.6 |
| tree | direction | 3 | 1 | baseline | 15 | 0.0 | 100.0 | 100.0 | 7.2 |
| tree | direction | 3 | 1 | edge_impact | 15 | 0.0 | 100.0 | 83.9 | 15.3 |
| tree | direction | 3 | 1 | heuristic_modify | 15 | 26.7 | 100.0 | 85.2 | 5.5 |
| tree | direction | 3 | 1 | heuristic_remove | 15 | 53.3 | 94.8 | 99.7 | 2.1 |
| tree | direction | 3 | 1 | vc_ei | 15 | 0.0 | 100.0 | 93.6 | 12.3 |
| tree | direction | 3 | 1 | vc_only | 15 | 0.0 | 100.0 | 100.0 | 7.9 |
| tree | direction | 4 | 1 | baseline | 15 | 0.0 | 100.0 | 100.0 | 7.3 |
| tree | direction | 4 | 1 | edge_impact | 15 | 0.0 | 100.0 | 86.0 | 17.3 |
| tree | direction | 4 | 1 | heuristic_modify | 15 | 60.0 | 100.0 | 90.2 | 5.9 |
| tree | direction | 4 | 1 | heuristic_remove | 15 | 66.7 | 97.2 | 99.5 | 2.1 |
| tree | direction | 4 | 1 | vc_ei | 15 | 0.0 | 100.0 | 90.7 | 16.3 |
| tree | direction | 4 | 1 | vc_only | 15 | 0.0 | 100.0 | 100.0 | 8.4 |
| tree | direction | 5 | 2 | baseline | 15 | 0.0 | 100.0 | 100.0 | 8.0 |
| tree | direction | 5 | 2 | edge_impact | 15 | 0.0 | 100.0 | 91.8 | 18.9 |
| tree | direction | 5 | 2 | heuristic_modify | 15 | 46.7 | 100.0 | 90.0 | 10.6 |
| tree | direction | 5 | 2 | heuristic_remove | 15 | 60.0 | 97.8 | 99.6 | 3.3 |
| tree | direction | 5 | 2 | vc_ei | 15 | 0.0 | 100.0 | 93.0 | 17.3 |
| tree | direction | 5 | 2 | vc_only | 15 | 0.0 | 100.0 | 100.0 | 8.8 |
| tree | topology | 3 | 1 | baseline | 15 | 73.3 | 100.0 | 98.8 | 2.3 |
| tree | topology | 3 | 1 | edge_impact | 15 | 100.0 | 100.0 | 100.0 | 1.4 |
| tree | topology | 3 | 1 | heuristic_modify | 15 | 86.7 | 99.7 | 88.5 | 3.1 |
| tree | topology | 3 | 1 | heuristic_remove | 15 | 86.7 | 94.8 | 99.4 | 1.3 |
| tree | topology | 3 | 1 | vc_ei | 15 | 86.7 | 100.0 | 96.7 | 4.0 |
| tree | topology | 3 | 1 | vc_only | 15 | 73.3 | 100.0 | 98.8 | 2.5 |
| tree | topology | 4 | 1 | baseline | 15 | 93.3 | 100.0 | 99.6 | 1.4 |
| tree | topology | 4 | 1 | edge_impact | 15 | 86.7 | 100.0 | 98.2 | 3.7 |
| tree | topology | 4 | 1 | heuristic_modify | 15 | 73.3 | 100.0 | 92.8 | 3.6 |
| tree | topology | 4 | 1 | heuristic_remove | 15 | 86.7 | 97.2 | 99.8 | 1.3 |
| tree | topology | 4 | 1 | vc_ei | 15 | 86.7 | 100.0 | 99.6 | 1.7 |
| tree | topology | 4 | 1 | vc_only | 15 | 93.3 | 100.0 | 99.8 | 1.7 |
| tree | topology | 5 | 2 | baseline | 15 | 80.0 | 100.0 | 99.7 | 3.3 |
| tree | topology | 5 | 2 | edge_impact | 15 | 86.7 | 100.0 | 99.5 | 4.8 |
| tree | topology | 5 | 2 | heuristic_modify | 15 | 86.7 | 100.0 | 91.8 | 7.4 |
| tree | topology | 5 | 2 | heuristic_remove | 15 | 73.3 | 96.4 | 99.6 | 2.7 |
| tree | topology | 5 | 2 | vc_ei | 15 | 80.0 | 100.0 | 98.3 | 5.7 |
| tree | topology | 5 | 2 | vc_only | 15 | 86.7 | 100.0 | 99.7 | 3.0 |
