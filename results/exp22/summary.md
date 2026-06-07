# Experiment 22 — Multi-model LLM-MapRepair ablation

Total runs: 1204 (valid 1204)

Models: ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4o']
LLM modes: ('baseline', 'edge_impact') (lookahead=off)
Reference baseline: ('heuristic_remove',)

## Aggregate by (mode × model)

| mode | model | n | CF % | GT recall % | dir acc % | iters | mod/run | rm/run |
|------|-------|--:|-----:|------------:|----------:|------:|--------:|-------:|
| baseline | gpt-4.1 | 172 | 73.3 | 99.6 | 99.6 | 3.8 | 2.0 | 0.8 |
| baseline | gpt-4.1-mini | 172 | 57.0 | 99.6 | 99.4 | 4.6 | 1.7 | 0.5 |
| baseline | gpt-4o | 172 | 83.1 | 99.3 | 99.6 | 2.6 | 0.7 | 1.0 |
| edge_impact | gpt-4.1 | 172 | 90.1 | 98.4 | 99.5 | 3.3 | 1.1 | 1.5 |
| edge_impact | gpt-4.1-mini | 172 | 76.2 | 96.8 | 98.6 | 5.8 | 2.8 | 1.7 |
| edge_impact | gpt-4o | 172 | 78.5 | 97.5 | 99.2 | 4.3 | 1.2 | 1.6 |
| heuristic_remove | n/a | 172 | 68.6 | 96.3 | 98.9 | 2.8 | 0.6 | 2.3 |

## Cross-model headline (CF % — main contribution)

| model | baseline LLM | edge_impact (ours) | Δ (lift) |
|-------|-------------:|-------------------:|---------:|
| gpt-4.1 | 73.3 | 90.1 | **+16.9pp** |
| gpt-4.1-mini | 57.0 | 76.2 | **+19.2pp** |
| gpt-4o | 83.1 | 78.5 | **-4.7pp** |
| heuristic_remove (ref) | — | — | 68.6% (any-model reference) |

## By model × error type

| mode | model | err_type | n | CF % | GT recall | dir acc |
|------|-------|----------|--:|-----:|----------:|--------:|
| baseline | gpt-4.1 | direction | 86 | 75.6 | 99.9 | 100.0 |
| baseline | gpt-4.1 | topology | 86 | 70.9 | 99.2 | 99.2 |
| baseline | gpt-4.1-mini | direction | 86 | 41.9 | 100.0 | 100.0 |
| baseline | gpt-4.1-mini | topology | 86 | 72.1 | 99.2 | 98.8 |
| baseline | gpt-4o | direction | 86 | 88.4 | 100.0 | 100.0 |
| baseline | gpt-4o | topology | 86 | 77.9 | 98.7 | 99.3 |
| edge_impact | gpt-4.1 | direction | 86 | 86.0 | 99.0 | 99.7 |
| edge_impact | gpt-4.1 | topology | 86 | 94.2 | 97.7 | 99.3 |
| edge_impact | gpt-4.1-mini | direction | 86 | 62.8 | 96.5 | 98.5 |
| edge_impact | gpt-4.1-mini | topology | 86 | 89.5 | 97.1 | 98.6 |
| edge_impact | gpt-4o | direction | 86 | 79.1 | 97.7 | 99.5 |
| edge_impact | gpt-4o | topology | 86 | 77.9 | 97.3 | 98.9 |
| heuristic_remove | n/a | direction | 86 | 65.1 | 97.5 | 99.3 |
| heuristic_remove | n/a | topology | 86 | 72.1 | 95.0 | 98.6 |
