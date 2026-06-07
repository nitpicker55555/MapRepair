# Experiment 19 — LLM with remove_edge + optional lookahead (model=gpt-4.1)

Total runs: 2720  (valid 2720, errors 0, skipped 0)

Core configs: 16 × 15 seeds  |  Large configs: 4 × 8 seeds.

## Aggregate by mode

| mode | n | CF % | GT recall % | dir acc % | iters | modify/run | remove/run |
|------|--:|-----:|------------:|----------:|------:|-----------:|-----------:|
| llm_baseline | 272 | 65.1 | 99.5 | 99.5 | 4.7 | 2.2 | 1.0 |
| llm_baseline+lookahead | 272 | 62.9 | 99.4 | 99.8 | 4.4 | 0.9 | 0.9 |
| llm_edge_impact | 272 | 85.3 | 98.2 | 99.4 | 4.0 | 1.4 | 1.7 |
| llm_edge_impact+lookahead | 272 | 76.5 | 98.7 | 99.8 | 3.9 | 0.7 | 1.4 |
| llm_vc_only | 272 | 66.2 | 99.5 | 99.2 | 5.3 | 2.2 | 1.1 |
| llm_vc_only+lookahead | 272 | 58.1 | 99.5 | 99.7 | 4.6 | 1.0 | 0.9 |
| llm_vc_ei | 272 | 65.8 | 98.2 | 99.3 | 5.7 | 1.9 | 1.8 |
| llm_vc_ei+lookahead | 272 | 57.7 | 98.6 | 99.7 | 5.1 | 0.9 | 1.3 |
| heuristic_remove | 272 | 68.4 | 96.0 | 99.1 | 3.2 | 0.6 | 2.5 |
| heuristic_modify | 272 | 43.0 | 99.3 | 90.6 | 7.1 | 6.7 | 0.4 |

## By mode × err_type

| mode | err_type | n | CF % | GT recall | dir acc | iters | modify | remove |
|------|----------|--:|-----:|----------:|--------:|------:|-------:|-------:|
| llm_baseline | direction | 136 | 61.0 | 99.9 | 100.0 | 4.6 | 0.9 | 1.5 |
| llm_baseline | topology | 136 | 69.1 | 99.2 | 99.1 | 4.9 | 3.5 | 0.4 |
| llm_baseline+lookahead | direction | 136 | 62.5 | 99.9 | 100.0 | 4.3 | 0.5 | 1.4 |
| llm_baseline+lookahead | topology | 136 | 63.2 | 99.0 | 99.6 | 4.4 | 1.4 | 0.5 |
| llm_edge_impact | direction | 136 | 81.6 | 98.5 | 99.7 | 4.0 | 0.4 | 2.3 |
| llm_edge_impact | topology | 136 | 89.0 | 98.0 | 99.0 | 4.0 | 2.5 | 1.1 |
| llm_edge_impact+lookahead | direction | 136 | 74.3 | 98.8 | 99.9 | 4.2 | 0.3 | 2.0 |
| llm_edge_impact+lookahead | topology | 136 | 78.7 | 98.6 | 99.7 | 3.5 | 1.1 | 0.9 |
| llm_vc_only | direction | 136 | 66.2 | 99.9 | 99.9 | 4.5 | 0.7 | 1.7 |
| llm_vc_only | topology | 136 | 66.2 | 99.0 | 98.4 | 6.0 | 3.7 | 0.6 |
| llm_vc_only+lookahead | direction | 136 | 57.4 | 99.8 | 100.0 | 4.6 | 0.5 | 1.4 |
| llm_vc_only+lookahead | topology | 136 | 58.8 | 99.2 | 99.5 | 4.7 | 1.5 | 0.4 |
| llm_vc_ei | direction | 136 | 64.0 | 98.3 | 99.3 | 6.4 | 1.2 | 2.6 |
| llm_vc_ei | topology | 136 | 67.6 | 98.2 | 99.2 | 5.1 | 2.6 | 1.0 |
| llm_vc_ei+lookahead | direction | 136 | 53.7 | 98.5 | 99.9 | 5.7 | 0.5 | 1.8 |
| llm_vc_ei+lookahead | topology | 136 | 61.8 | 98.6 | 99.5 | 4.6 | 1.4 | 0.7 |
| heuristic_remove | direction | 136 | 62.5 | 97.5 | 99.4 | 3.1 | 0.6 | 2.5 |
| heuristic_remove | topology | 136 | 74.3 | 94.6 | 98.7 | 3.2 | 0.7 | 2.6 |
| heuristic_modify | direction | 136 | 22.1 | 99.6 | 89.8 | 7.8 | 7.5 | 0.2 |
| heuristic_modify | topology | 136 | 64.0 | 99.0 | 91.4 | 6.4 | 5.9 | 0.5 |

## Detailed: family × err_type × size × num_errors × mode

| family | err | size | n_err | mode | n | CF % | GT recall | dir acc | iters |
|--------|-----|-----:|------:|------|--:|-----:|----------:|--------:|------:|
| grid | direction | 3 | 1 | heuristic_modify | 15 | 0.0 | 98.6 | 94.9 | 2.5 |
| grid | direction | 3 | 1 | heuristic_remove | 15 | 66.7 | 97.2 | 98.2 | 1.9 |
| grid | direction | 3 | 1 | llm_baseline | 15 | 80.0 | 100.0 | 100.0 | 2.8 |
| grid | direction | 3 | 1 | llm_baseline+lookahead | 15 | 93.3 | 100.0 | 100.0 | 1.4 |
| grid | direction | 3 | 1 | llm_edge_impact | 15 | 93.3 | 97.8 | 99.2 | 2.2 |
| grid | direction | 3 | 1 | llm_edge_impact+lookahead | 15 | 100.0 | 99.7 | 100.0 | 1.1 |
| grid | direction | 3 | 1 | llm_vc_ei | 15 | 46.7 | 96.4 | 97.6 | 7.1 |
| grid | direction | 3 | 1 | llm_vc_ei+lookahead | 15 | 60.0 | 98.6 | 100.0 | 3.5 |
| grid | direction | 3 | 1 | llm_vc_only | 15 | 33.3 | 99.7 | 99.4 | 5.9 |
| grid | direction | 3 | 1 | llm_vc_only+lookahead | 15 | 53.3 | 100.0 | 100.0 | 3.8 |
| grid | direction | 4 | 1 | heuristic_modify | 15 | 6.7 | 99.3 | 95.1 | 4.4 |
| grid | direction | 4 | 1 | heuristic_remove | 15 | 100.0 | 100.0 | 100.0 | 1.0 |
| grid | direction | 4 | 1 | llm_baseline | 15 | 66.7 | 99.9 | 100.0 | 3.1 |
| grid | direction | 4 | 1 | llm_baseline+lookahead | 15 | 93.3 | 100.0 | 100.0 | 1.4 |
| grid | direction | 4 | 1 | llm_edge_impact | 15 | 100.0 | 100.0 | 100.0 | 1.0 |
| grid | direction | 4 | 1 | llm_edge_impact+lookahead | 15 | 100.0 | 99.9 | 100.0 | 1.1 |
| grid | direction | 4 | 1 | llm_vc_ei | 15 | 40.0 | 97.6 | 99.0 | 8.5 |
| grid | direction | 4 | 1 | llm_vc_ei+lookahead | 15 | 40.0 | 99.0 | 99.9 | 4.8 |
| grid | direction | 4 | 1 | llm_vc_only | 15 | 46.7 | 100.0 | 100.0 | 4.2 |
| grid | direction | 4 | 1 | llm_vc_only+lookahead | 15 | 46.7 | 100.0 | 100.0 | 4.2 |
| grid | direction | 5 | 2 | heuristic_modify | 15 | 0.0 | 98.4 | 95.1 | 8.6 |
| grid | direction | 5 | 2 | heuristic_remove | 15 | 60.0 | 99.2 | 99.5 | 3.1 |
| grid | direction | 5 | 2 | llm_baseline | 15 | 46.7 | 99.8 | 99.7 | 7.3 |
| grid | direction | 5 | 2 | llm_baseline+lookahead | 15 | 53.3 | 99.9 | 100.0 | 5.2 |
| grid | direction | 5 | 2 | llm_edge_impact | 15 | 93.3 | 99.4 | 99.7 | 3.2 |
| grid | direction | 5 | 2 | llm_edge_impact+lookahead | 15 | 80.0 | 99.8 | 100.0 | 3.5 |
| grid | direction | 5 | 2 | llm_vc_ei | 15 | 40.0 | 98.0 | 98.1 | 10.5 |
| grid | direction | 5 | 2 | llm_vc_ei+lookahead | 15 | 33.3 | 99.3 | 99.7 | 7.5 |
| grid | direction | 5 | 2 | llm_vc_only | 15 | 26.7 | 100.0 | 99.8 | 7.1 |
| grid | direction | 5 | 2 | llm_vc_only+lookahead | 15 | 26.7 | 100.0 | 100.0 | 6.7 |
| grid | topology | 3 | 1 | heuristic_modify | 15 | 53.3 | 98.1 | 93.9 | 2.1 |
| grid | topology | 3 | 1 | heuristic_remove | 15 | 86.7 | 93.3 | 98.4 | 1.7 |
| grid | topology | 3 | 1 | llm_baseline | 15 | 66.7 | 98.9 | 96.6 | 5.0 |
| grid | topology | 3 | 1 | llm_baseline+lookahead | 15 | 66.7 | 98.1 | 98.3 | 3.2 |
| grid | topology | 3 | 1 | llm_edge_impact | 15 | 73.3 | 90.6 | 93.3 | 8.3 |
| grid | topology | 3 | 1 | llm_edge_impact+lookahead | 15 | 80.0 | 95.8 | 99.1 | 2.5 |
| grid | topology | 3 | 1 | llm_vc_ei | 15 | 53.3 | 95.0 | 96.0 | 7.1 |
| grid | topology | 3 | 1 | llm_vc_ei+lookahead | 15 | 66.7 | 97.5 | 98.3 | 3.3 |
| grid | topology | 3 | 1 | llm_vc_only | 15 | 46.7 | 97.5 | 92.4 | 6.9 |
| grid | topology | 3 | 1 | llm_vc_only+lookahead | 15 | 73.3 | 99.7 | 98.6 | 2.9 |
| grid | topology | 4 | 1 | heuristic_modify | 15 | 40.0 | 97.8 | 95.6 | 4.0 |
| grid | topology | 4 | 1 | heuristic_remove | 15 | 60.0 | 95.7 | 98.2 | 2.7 |
| grid | topology | 4 | 1 | llm_baseline | 15 | 73.3 | 99.7 | 98.2 | 5.3 |
| grid | topology | 4 | 1 | llm_baseline+lookahead | 15 | 80.0 | 99.4 | 99.6 | 2.7 |
| grid | topology | 4 | 1 | llm_edge_impact | 15 | 80.0 | 96.0 | 99.1 | 4.8 |
| grid | topology | 4 | 1 | llm_edge_impact+lookahead | 15 | 86.7 | 97.9 | 99.7 | 2.1 |
| grid | topology | 4 | 1 | llm_vc_ei | 15 | 80.0 | 97.9 | 99.6 | 3.9 |
| grid | topology | 4 | 1 | llm_vc_ei+lookahead | 15 | 80.0 | 98.8 | 99.6 | 2.7 |
| grid | topology | 4 | 1 | llm_vc_only | 15 | 73.3 | 99.3 | 98.2 | 6.1 |
| grid | topology | 4 | 1 | llm_vc_only+lookahead | 15 | 86.7 | 99.7 | 99.7 | 2.3 |
| grid | topology | 5 | 2 | heuristic_modify | 15 | 26.7 | 98.2 | 95.3 | 8.1 |
| grid | topology | 5 | 2 | heuristic_remove | 15 | 60.0 | 94.8 | 98.4 | 5.3 |
| grid | topology | 5 | 2 | llm_baseline | 15 | 40.0 | 99.3 | 97.0 | 12.7 |
| grid | topology | 5 | 2 | llm_baseline+lookahead | 15 | 60.0 | 99.3 | 98.7 | 5.9 |
| grid | topology | 5 | 2 | llm_edge_impact | 15 | 86.7 | 96.8 | 99.5 | 5.2 |
| grid | topology | 5 | 2 | llm_edge_impact+lookahead | 15 | 93.3 | 97.9 | 99.9 | 2.7 |
| grid | topology | 5 | 2 | llm_vc_ei | 15 | 53.3 | 97.0 | 97.8 | 9.1 |
| grid | topology | 5 | 2 | llm_vc_ei+lookahead | 15 | 73.3 | 98.2 | 99.2 | 4.7 |
| grid | topology | 5 | 2 | llm_vc_only | 15 | 40.0 | 98.8 | 95.4 | 12.8 |
| grid | topology | 5 | 2 | llm_vc_only+lookahead | 15 | 46.7 | 99.5 | 98.5 | 6.6 |
| random | direction | 24 | 1 | heuristic_modify | 15 | 40.0 | 100.0 | 84.6 | 5.9 |
| random | direction | 24 | 1 | heuristic_remove | 15 | 86.7 | 99.4 | 100.0 | 1.0 |
| random | direction | 24 | 1 | llm_baseline | 15 | 73.3 | 100.0 | 100.0 | 2.6 |
| random | direction | 24 | 1 | llm_baseline+lookahead | 15 | 86.7 | 99.7 | 100.0 | 1.9 |
| random | direction | 24 | 1 | llm_edge_impact | 15 | 93.3 | 99.7 | 100.0 | 1.4 |
| random | direction | 24 | 1 | llm_edge_impact+lookahead | 15 | 93.3 | 99.7 | 100.0 | 1.5 |
| random | direction | 24 | 1 | llm_vc_ei | 15 | 86.7 | 98.3 | 99.6 | 2.9 |
| random | direction | 24 | 1 | llm_vc_ei+lookahead | 15 | 86.7 | 98.8 | 100.0 | 2.1 |
| random | direction | 24 | 1 | llm_vc_only | 15 | 86.7 | 100.0 | 100.0 | 1.8 |
| random | direction | 24 | 1 | llm_vc_only+lookahead | 15 | 93.3 | 100.0 | 100.0 | 1.4 |
| random | direction | 36 | 2 | heuristic_modify | 15 | 13.3 | 100.0 | 86.1 | 8.5 |
| random | direction | 36 | 2 | heuristic_remove | 15 | 53.3 | 97.3 | 98.8 | 3.1 |
| random | direction | 36 | 2 | llm_baseline | 15 | 53.3 | 100.0 | 100.0 | 4.8 |
| random | direction | 36 | 2 | llm_baseline+lookahead | 15 | 46.7 | 99.8 | 100.0 | 5.5 |
| random | direction | 36 | 2 | llm_edge_impact | 15 | 73.3 | 97.3 | 100.0 | 5.2 |
| random | direction | 36 | 2 | llm_edge_impact+lookahead | 15 | 60.0 | 97.7 | 100.0 | 5.6 |
| random | direction | 36 | 2 | llm_vc_ei | 15 | 66.7 | 98.5 | 100.0 | 6.3 |
| random | direction | 36 | 2 | llm_vc_ei+lookahead | 15 | 33.3 | 97.5 | 99.6 | 8.2 |
| random | direction | 36 | 2 | llm_vc_only | 15 | 66.7 | 99.8 | 99.8 | 5.3 |
| random | direction | 36 | 2 | llm_vc_only+lookahead | 15 | 60.0 | 99.6 | 100.0 | 4.7 |
| random | direction | 60 | 4 | heuristic_modify | 8 | 12.5 | 100.0 | 88.3 | 14.4 |
| random | direction | 60 | 4 | heuristic_remove | 8 | 12.5 | 94.3 | 99.6 | 7.6 |
| random | direction | 60 | 4 | llm_baseline | 8 | 62.5 | 99.8 | 100.0 | 6.2 |
| random | direction | 60 | 4 | llm_baseline+lookahead | 8 | 25.0 | 100.0 | 99.6 | 9.0 |
| random | direction | 60 | 4 | llm_edge_impact | 8 | 62.5 | 98.7 | 100.0 | 7.5 |
| random | direction | 60 | 4 | llm_edge_impact+lookahead | 8 | 25.0 | 98.3 | 99.8 | 11.0 |
| random | direction | 60 | 4 | llm_vc_ei | 8 | 50.0 | 98.5 | 100.0 | 10.8 |
| random | direction | 60 | 4 | llm_vc_ei+lookahead | 8 | 0.0 | 97.0 | 99.8 | 13.8 |
| random | direction | 60 | 4 | llm_vc_only | 8 | 62.5 | 100.0 | 99.8 | 8.8 |
| random | direction | 60 | 4 | llm_vc_only+lookahead | 8 | 25.0 | 99.6 | 99.8 | 9.0 |
| random | direction | 60 | 8 | heuristic_modify | 8 | 0.0 | 100.0 | 86.0 | 20.0 |
| random | direction | 60 | 8 | heuristic_remove | 8 | 25.0 | 94.5 | 98.9 | 11.6 |
| random | direction | 60 | 8 | llm_baseline | 8 | 50.0 | 99.8 | 99.8 | 11.0 |
| random | direction | 60 | 8 | llm_baseline+lookahead | 8 | 0.0 | 99.6 | 99.8 | 14.8 |
| random | direction | 60 | 8 | llm_edge_impact | 8 | 50.0 | 97.5 | 99.5 | 12.9 |
| random | direction | 60 | 8 | llm_edge_impact+lookahead | 8 | 12.5 | 97.2 | 99.8 | 13.5 |
| random | direction | 60 | 8 | llm_vc_ei | 8 | 50.0 | 98.1 | 99.6 | 12.6 |
| random | direction | 60 | 8 | llm_vc_ei+lookahead | 8 | 12.5 | 96.6 | 100.0 | 15.0 |
| random | direction | 60 | 8 | llm_vc_only | 8 | 87.5 | 99.8 | 100.0 | 10.2 |
| random | direction | 60 | 8 | llm_vc_only+lookahead | 8 | 0.0 | 98.7 | 100.0 | 14.8 |
| random | topology | 24 | 1 | heuristic_modify | 15 | 66.7 | 99.7 | 89.8 | 2.5 |
| random | topology | 24 | 1 | heuristic_remove | 15 | 100.0 | 95.9 | 99.1 | 1.2 |
| random | topology | 24 | 1 | llm_baseline | 15 | 80.0 | 99.1 | 100.0 | 1.8 |
| random | topology | 24 | 1 | llm_baseline+lookahead | 15 | 86.7 | 99.4 | 100.0 | 1.6 |
| random | topology | 24 | 1 | llm_edge_impact | 15 | 100.0 | 100.0 | 100.0 | 1.0 |
| random | topology | 24 | 1 | llm_edge_impact+lookahead | 15 | 100.0 | 100.0 | 100.0 | 1.0 |
| random | topology | 24 | 1 | llm_vc_ei | 15 | 86.7 | 99.4 | 100.0 | 2.2 |
| random | topology | 24 | 1 | llm_vc_ei+lookahead | 15 | 73.3 | 98.8 | 100.0 | 2.1 |
| random | topology | 24 | 1 | llm_vc_only | 15 | 86.7 | 99.4 | 100.0 | 2.1 |
| random | topology | 24 | 1 | llm_vc_only+lookahead | 15 | 73.3 | 98.8 | 100.0 | 2.1 |
| random | topology | 36 | 2 | heuristic_modify | 15 | 100.0 | 100.0 | 88.8 | 7.5 |
| random | topology | 36 | 2 | heuristic_remove | 15 | 73.3 | 93.5 | 98.2 | 3.1 |
| random | topology | 36 | 2 | llm_baseline | 15 | 66.7 | 99.0 | 100.0 | 3.3 |
| random | topology | 36 | 2 | llm_baseline+lookahead | 15 | 40.0 | 98.7 | 99.8 | 5.3 |
| random | topology | 36 | 2 | llm_edge_impact | 15 | 93.3 | 99.6 | 99.8 | 2.5 |
| random | topology | 36 | 2 | llm_edge_impact+lookahead | 15 | 66.7 | 98.9 | 100.0 | 4.1 |
| random | topology | 36 | 2 | llm_vc_ei | 15 | 66.7 | 99.0 | 100.0 | 5.3 |
| random | topology | 36 | 2 | llm_vc_ei+lookahead | 15 | 33.3 | 98.1 | 99.6 | 5.9 |
| random | topology | 36 | 2 | llm_vc_only | 15 | 66.7 | 99.0 | 100.0 | 5.9 |
| random | topology | 36 | 2 | llm_vc_only+lookahead | 15 | 33.3 | 98.5 | 100.0 | 5.5 |
| random | topology | 60 | 4 | heuristic_modify | 8 | 75.0 | 98.7 | 89.5 | 16.2 |
| random | topology | 60 | 4 | heuristic_remove | 8 | 12.5 | 92.6 | 98.7 | 7.5 |
| random | topology | 60 | 4 | llm_baseline | 8 | 50.0 | 98.5 | 100.0 | 6.4 |
| random | topology | 60 | 4 | llm_baseline+lookahead | 8 | 0.0 | 97.7 | 99.6 | 10.8 |
| random | topology | 60 | 4 | llm_edge_impact | 8 | 87.5 | 99.6 | 99.8 | 5.4 |
| random | topology | 60 | 4 | llm_edge_impact+lookahead | 8 | 12.5 | 97.2 | 98.2 | 12.8 |
| random | topology | 60 | 4 | llm_vc_ei | 8 | 25.0 | 97.2 | 100.0 | 8.1 |
| random | topology | 60 | 4 | llm_vc_ei+lookahead | 8 | 0.0 | 97.5 | 99.6 | 10.8 |
| random | topology | 60 | 4 | llm_vc_only | 8 | 37.5 | 98.5 | 100.0 | 9.8 |
| random | topology | 60 | 4 | llm_vc_only+lookahead | 8 | 0.0 | 98.7 | 99.1 | 11.1 |
| random | topology | 60 | 8 | heuristic_modify | 8 | 12.5 | 97.0 | 83.4 | 20.0 |
| random | topology | 60 | 8 | heuristic_remove | 8 | 75.0 | 87.1 | 96.6 | 11.6 |
| random | topology | 60 | 8 | llm_baseline | 8 | 25.0 | 97.0 | 100.0 | 13.0 |
| random | topology | 60 | 8 | llm_baseline+lookahead | 8 | 0.0 | 96.8 | 100.0 | 15.6 |
| random | topology | 60 | 8 | llm_edge_impact | 8 | 62.5 | 98.7 | 99.3 | 12.8 |
| random | topology | 60 | 8 | llm_edge_impact+lookahead | 8 | 12.5 | 97.5 | 98.5 | 14.1 |
| random | topology | 60 | 8 | llm_vc_ei | 8 | 0.0 | 96.6 | 99.8 | 13.9 |
| random | topology | 60 | 8 | llm_vc_ei+lookahead | 8 | 0.0 | 97.0 | 98.9 | 18.0 |
| random | topology | 60 | 8 | llm_vc_only | 8 | 25.0 | 97.5 | 99.4 | 14.9 |
| random | topology | 60 | 8 | llm_vc_only+lookahead | 8 | 0.0 | 97.5 | 98.9 | 15.4 |
| tree | direction | 3 | 1 | heuristic_modify | 15 | 26.7 | 100.0 | 85.2 | 5.5 |
| tree | direction | 3 | 1 | heuristic_remove | 15 | 53.3 | 94.8 | 99.7 | 2.1 |
| tree | direction | 3 | 1 | llm_baseline | 15 | 73.3 | 100.0 | 100.0 | 2.6 |
| tree | direction | 3 | 1 | llm_baseline+lookahead | 15 | 73.3 | 100.0 | 100.0 | 2.6 |
| tree | direction | 3 | 1 | llm_edge_impact | 15 | 80.0 | 97.0 | 99.3 | 3.5 |
| tree | direction | 3 | 1 | llm_edge_impact+lookahead | 15 | 80.0 | 98.2 | 99.3 | 2.9 |
| tree | direction | 3 | 1 | llm_vc_ei | 15 | 100.0 | 100.0 | 100.0 | 1.0 |
| tree | direction | 3 | 1 | llm_vc_ei+lookahead | 15 | 100.0 | 100.0 | 100.0 | 1.0 |
| tree | direction | 3 | 1 | llm_vc_only | 15 | 100.0 | 100.0 | 100.0 | 1.0 |
| tree | direction | 3 | 1 | llm_vc_only+lookahead | 15 | 100.0 | 100.0 | 100.0 | 1.0 |
| tree | direction | 4 | 1 | heuristic_modify | 15 | 60.0 | 100.0 | 90.2 | 5.9 |
| tree | direction | 4 | 1 | heuristic_remove | 15 | 66.7 | 97.2 | 99.5 | 2.1 |
| tree | direction | 4 | 1 | llm_baseline | 15 | 46.7 | 99.6 | 100.0 | 4.2 |
| tree | direction | 4 | 1 | llm_baseline+lookahead | 15 | 66.7 | 99.8 | 100.0 | 3.0 |
| tree | direction | 4 | 1 | llm_edge_impact | 15 | 73.3 | 98.4 | 99.8 | 3.8 |
| tree | direction | 4 | 1 | llm_edge_impact+lookahead | 15 | 80.0 | 98.8 | 99.8 | 3.0 |
| tree | direction | 4 | 1 | llm_vc_ei | 15 | 73.3 | 98.2 | 100.0 | 3.7 |
| tree | direction | 4 | 1 | llm_vc_ei+lookahead | 15 | 73.3 | 98.1 | 100.0 | 3.5 |
| tree | direction | 4 | 1 | llm_vc_only | 15 | 80.0 | 99.8 | 100.0 | 2.2 |
| tree | direction | 4 | 1 | llm_vc_only+lookahead | 15 | 73.3 | 99.8 | 100.0 | 2.6 |
| tree | direction | 5 | 2 | heuristic_modify | 15 | 46.7 | 100.0 | 90.0 | 10.6 |
| tree | direction | 5 | 2 | heuristic_remove | 15 | 60.0 | 97.8 | 99.6 | 3.3 |
| tree | direction | 5 | 2 | llm_baseline | 15 | 53.3 | 100.0 | 100.0 | 4.9 |
| tree | direction | 5 | 2 | llm_baseline+lookahead | 15 | 40.0 | 99.9 | 100.0 | 5.5 |
| tree | direction | 5 | 2 | llm_edge_impact | 15 | 73.3 | 98.6 | 99.9 | 5.1 |
| tree | direction | 5 | 2 | llm_edge_impact+lookahead | 15 | 60.0 | 98.0 | 99.8 | 6.8 |
| tree | direction | 5 | 2 | llm_vc_ei | 15 | 73.3 | 99.0 | 99.7 | 5.4 |
| tree | direction | 5 | 2 | llm_vc_ei+lookahead | 15 | 53.3 | 98.6 | 99.8 | 5.8 |
| tree | direction | 5 | 2 | llm_vc_only | 15 | 80.0 | 100.0 | 100.0 | 3.2 |
| tree | direction | 5 | 2 | llm_vc_only+lookahead | 15 | 53.3 | 100.0 | 99.8 | 4.8 |
| tree | topology | 3 | 1 | heuristic_modify | 15 | 86.7 | 99.7 | 88.5 | 3.1 |
| tree | topology | 3 | 1 | heuristic_remove | 15 | 86.7 | 94.8 | 99.4 | 1.3 |
| tree | topology | 3 | 1 | llm_baseline | 15 | 86.7 | 99.4 | 100.0 | 1.6 |
| tree | topology | 3 | 1 | llm_baseline+lookahead | 15 | 86.7 | 99.4 | 100.0 | 1.5 |
| tree | topology | 3 | 1 | llm_edge_impact | 15 | 100.0 | 100.0 | 100.0 | 1.0 |
| tree | topology | 3 | 1 | llm_edge_impact+lookahead | 15 | 100.0 | 100.0 | 100.0 | 1.0 |
| tree | topology | 3 | 1 | llm_vc_ei | 15 | 86.7 | 99.4 | 100.0 | 1.9 |
| tree | topology | 3 | 1 | llm_vc_ei+lookahead | 15 | 86.7 | 99.4 | 100.0 | 1.7 |
| tree | topology | 3 | 1 | llm_vc_only | 15 | 86.7 | 99.7 | 100.0 | 2.3 |
| tree | topology | 3 | 1 | llm_vc_only+lookahead | 15 | 80.0 | 99.1 | 100.0 | 2.0 |
| tree | topology | 4 | 1 | heuristic_modify | 15 | 73.3 | 100.0 | 92.8 | 3.6 |
| tree | topology | 4 | 1 | heuristic_remove | 15 | 86.7 | 97.2 | 99.8 | 1.3 |
| tree | topology | 4 | 1 | llm_baseline | 15 | 93.3 | 99.8 | 100.0 | 1.3 |
| tree | topology | 4 | 1 | llm_baseline+lookahead | 15 | 93.3 | 99.8 | 100.0 | 1.3 |
| tree | topology | 4 | 1 | llm_edge_impact | 15 | 93.3 | 99.8 | 100.0 | 1.5 |
| tree | topology | 4 | 1 | llm_edge_impact+lookahead | 15 | 93.3 | 99.8 | 100.0 | 1.5 |
| tree | topology | 4 | 1 | llm_vc_ei | 15 | 86.7 | 99.6 | 100.0 | 1.5 |
| tree | topology | 4 | 1 | llm_vc_ei+lookahead | 15 | 86.7 | 99.6 | 100.0 | 1.5 |
| tree | topology | 4 | 1 | llm_vc_only | 15 | 86.7 | 99.6 | 100.0 | 1.5 |
| tree | topology | 4 | 1 | llm_vc_only+lookahead | 15 | 86.7 | 99.6 | 100.0 | 1.5 |
| tree | topology | 5 | 2 | heuristic_modify | 15 | 86.7 | 100.0 | 91.8 | 7.4 |
| tree | topology | 5 | 2 | heuristic_remove | 15 | 73.3 | 96.4 | 99.6 | 2.7 |
| tree | topology | 5 | 2 | llm_baseline | 15 | 80.0 | 99.7 | 100.0 | 3.2 |
| tree | topology | 5 | 2 | llm_baseline+lookahead | 15 | 60.0 | 99.7 | 100.0 | 4.1 |
| tree | topology | 5 | 2 | llm_edge_impact | 15 | 100.0 | 100.0 | 100.0 | 2.1 |
| tree | topology | 5 | 2 | llm_edge_impact+lookahead | 15 | 80.0 | 100.0 | 100.0 | 2.8 |
| tree | topology | 5 | 2 | llm_vc_ei | 15 | 86.7 | 99.8 | 100.0 | 3.1 |
| tree | topology | 5 | 2 | llm_vc_ei+lookahead | 15 | 60.0 | 99.5 | 99.9 | 4.4 |
| tree | topology | 5 | 2 | llm_vc_only | 15 | 80.0 | 99.7 | 100.0 | 4.0 |
| tree | topology | 5 | 2 | llm_vc_only+lookahead | 15 | 53.3 | 99.5 | 99.8 | 5.4 |
