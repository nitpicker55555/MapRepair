# exp29 — Complementary roles (model=gpt-4.1)

Cells: [('random', 'direction', 60, 4), ('random', 'direction', 60, 8), ('random', 'topology', 60, 4), ('random', 'topology', 60, 8)]
Modes: ['llm_baseline', 'llm_edge_impact', 'llm_vc_only', 'llm_vc_ei']
Seeds per cell: 20

## Aggregate by err_type x num_errors x mode

| err_type | n_err | mode | n | CF % | GT recall | dir acc | iters | actions |
|----------|------:|------|--:|-----:|----------:|--------:|------:|--------:|
| direction | 4 | llm_baseline | 20 | 70.0 | 99.83 | 100.00 | 5.8 | 4.0 |
| direction | 4 | llm_edge_impact | 20 | 75.0 | 98.98 | 100.00 | 6.5 | 4.5 |
| direction | 4 | llm_vc_ei | 20 | 55.0 | 99.15 | 99.83 | 8.3 | 6.0 |
| direction | 4 | llm_vc_only | 20 | 55.0 | 99.92 | 100.00 | 7.7 | 5.5 |
| direction | 8 | llm_baseline | 20 | 70.0 | 99.75 | 99.92 | 9.8 | 8.0 |
| direction | 8 | llm_edge_impact | 20 | 50.0 | 97.63 | 99.82 | 12.2 | 8.9 |
| direction | 8 | llm_vc_ei | 20 | 25.0 | 96.61 | 99.82 | 15.1 | 10.8 |
| direction | 8 | llm_vc_only | 20 | 60.0 | 99.75 | 99.92 | 12.2 | 10.9 |
| topology | 4 | llm_baseline | 20 | 50.0 | 98.64 | 100.00 | 7.0 | 4.8 |
| topology | 4 | llm_edge_impact | 20 | 95.0 | 99.32 | 99.83 | 5.0 | 4.8 |
| topology | 4 | llm_vc_ei | 20 | 50.0 | 98.47 | 100.00 | 8.0 | 6.5 |
| topology | 4 | llm_vc_only | 20 | 50.0 | 98.90 | 99.83 | 8.4 | 6.7 |
| topology | 8 | llm_baseline | 20 | 30.0 | 97.88 | 99.15 | 13.2 | 9.6 |
| topology | 8 | llm_edge_impact | 20 | 60.0 | 98.31 | 99.57 | 12.6 | 10.9 |
| topology | 8 | llm_vc_ei | 20 | 40.0 | 98.05 | 100.00 | 11.8 | 9.4 |
| topology | 8 | llm_vc_only | 20 | 25.0 | 97.03 | 99.75 | 15.2 | 13.2 |
