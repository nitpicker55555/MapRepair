# Experiment 17 — LLM ablation on synthetic GT (model=gpt-4.1)

Total runs: 320  (320 valid, 0 skipped, 0 errors)

## Aggregate by mode

| mode | n | conflict-free % | GT edge recall % | GT dir acc % | mean iters | mean actions |
|------|--:|----------------:|-----------------:|-------------:|-----------:|-------------:|
| baseline | 80 | 46.2 | 100.0 | 99.3 | 5.0 | 1.9 |
| edge_impact | 80 | 45.0 | 100.0 | 96.1 | 8.2 | 6.2 |
| vc_only | 80 | 45.0 | 100.0 | 99.3 | 6.0 | 2.8 |
| vc_ei | 80 | 42.5 | 100.0 | 96.7 | 8.2 | 6.0 |

## By mode × error type

| mode | err_type | n | conflict-free % | GT recall % | dir acc % | iters |
|------|----------|--:|----------------:|------------:|----------:|------:|
| baseline | direction | 40 | 10.0 | 100.0 | 99.5 | 7.2 |
| baseline | topology | 40 | 82.5 | 100.0 | 99.1 | 2.8 |
| edge_impact | direction | 40 | 7.5 | 100.0 | 94.5 | 12.4 |
| edge_impact | topology | 40 | 82.5 | 100.0 | 97.8 | 4.1 |
| vc_only | direction | 40 | 10.0 | 100.0 | 99.7 | 8.4 |
| vc_only | topology | 40 | 80.0 | 100.0 | 99.0 | 3.6 |
| vc_ei | direction | 40 | 7.5 | 100.0 | 95.4 | 11.6 |
| vc_ei | topology | 40 | 77.5 | 100.0 | 98.0 | 4.7 |

## By mode × family × error type × num_errors

| family | err_type | n_err | mode | CF % | GT recall | iters |
|--------|----------|------:|------|-----:|----------:|------:|
| grid | direction | 1 | baseline | 30.0 | 100.0 | 5.7 |
| grid | direction | 1 | edge_impact | 30.0 | 100.0 | 9.9 |
| grid | direction | 1 | vc_ei | 30.0 | 100.0 | 9.4 |
| grid | direction | 1 | vc_only | 30.0 | 100.0 | 5.5 |
| grid | direction | 2 | baseline | 0.0 | 100.0 | 9.6 |
| grid | direction | 2 | edge_impact | 0.0 | 100.0 | 14.6 |
| grid | direction | 2 | vc_ei | 0.0 | 100.0 | 14.4 |
| grid | direction | 2 | vc_only | 0.0 | 100.0 | 12.2 |
| grid | topology | 1 | baseline | 70.0 | 100.0 | 3.5 |
| grid | topology | 1 | edge_impact | 50.0 | 100.0 | 7.8 |
| grid | topology | 1 | vc_ei | 60.0 | 100.0 | 6.9 |
| grid | topology | 1 | vc_only | 70.0 | 100.0 | 4.7 |
| grid | topology | 2 | baseline | 100.0 | 100.0 | 3.8 |
| grid | topology | 2 | edge_impact | 80.0 | 100.0 | 5.8 |
| grid | topology | 2 | vc_ei | 40.0 | 100.0 | 10.2 |
| grid | topology | 2 | vc_only | 80.0 | 100.0 | 6.6 |
| random | direction | 1 | baseline | 20.0 | 100.0 | 6.2 |
| random | direction | 1 | edge_impact | 0.0 | 100.0 | 12.4 |
| random | direction | 1 | vc_ei | 0.0 | 100.0 | 11.8 |
| random | direction | 1 | vc_only | 20.0 | 100.0 | 6.6 |
| random | direction | 2 | baseline | 0.0 | 100.0 | 8.6 |
| random | direction | 2 | edge_impact | 0.0 | 100.0 | 15.0 |
| random | direction | 2 | vc_ei | 0.0 | 100.0 | 12.8 |
| random | direction | 2 | vc_only | 0.0 | 100.0 | 9.4 |
| random | topology | 1 | baseline | 100.0 | 100.0 | 1.2 |
| random | topology | 1 | edge_impact | 100.0 | 100.0 | 1.4 |
| random | topology | 1 | vc_ei | 100.0 | 100.0 | 1.4 |
| random | topology | 1 | vc_only | 100.0 | 100.0 | 1.2 |
| random | topology | 2 | baseline | 40.0 | 100.0 | 4.4 |
| random | topology | 2 | edge_impact | 80.0 | 100.0 | 4.6 |
| random | topology | 2 | vc_ei | 60.0 | 100.0 | 6.8 |
| random | topology | 2 | vc_only | 40.0 | 100.0 | 4.4 |
| tree | direction | 1 | baseline | 0.0 | 100.0 | 7.1 |
| tree | direction | 1 | edge_impact | 0.0 | 100.0 | 11.6 |
| tree | direction | 1 | vc_ei | 0.0 | 100.0 | 10.2 |
| tree | direction | 1 | vc_only | 0.0 | 100.0 | 8.3 |
| tree | direction | 2 | baseline | 0.0 | 100.0 | 8.0 |
| tree | direction | 2 | edge_impact | 0.0 | 100.0 | 14.4 |
| tree | direction | 2 | vc_ei | 0.0 | 100.0 | 14.4 |
| tree | direction | 2 | vc_only | 0.0 | 100.0 | 11.6 |
| tree | topology | 1 | baseline | 100.0 | 100.0 | 1.1 |
| tree | topology | 1 | edge_impact | 100.0 | 100.0 | 1.4 |
| tree | topology | 1 | vc_ei | 100.0 | 100.0 | 1.2 |
| tree | topology | 1 | vc_only | 90.0 | 100.0 | 2.0 |
| tree | topology | 2 | baseline | 80.0 | 100.0 | 3.4 |
| tree | topology | 2 | edge_impact | 100.0 | 100.0 | 2.4 |
| tree | topology | 2 | vc_ei | 100.0 | 100.0 | 3.2 |
| tree | topology | 2 | vc_only | 100.0 | 100.0 | 3.4 |
