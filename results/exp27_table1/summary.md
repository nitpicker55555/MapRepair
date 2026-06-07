# exp27 — Table 1 rerun (model=gpt-4.1)

Games: ['cutthroat', 'detective', 'inhumane', 'zork1', 'zork2', 'murdac', 'advent', 'sherlock', 'wishbringer', 'deephome']
Modes: ['baseline', 'edge_impact', 'vc_only', 'vc_ei']

## Aggregate by mode

| Mode | Avg Loops | Repair Rate (%) | Acc.Δ (correct-direction edges gained) |
|------|----------:|----------------:|---------------------------------------:|
| baseline | 12.10 | -83.78 | -11 |
| edge_impact | 8.80 | 77.03 | -15 |
| vc_only | 15.60 | -93.92 | -12 |
| vc_ei | 10.60 | -15.54 | -14 |

## Per-game per-mode

| game | mode | conf_before | conf_after | repaired | actions | iters | dir_delta | elapsed |
|------|------|-----------:|----------:|---------:|--------:|------:|----------:|--------:|
| advent | baseline | 3 | 49 | -46 | 2 | 8 | -1 | 10.2s |
| advent | edge_impact | 3 | 1 | 2 | 2 | 5 | -2 | 7.2s |
| advent | vc_ei | 3 | 1 | 2 | 2 | 5 | -2 | 7.0s |
| advent | vc_only | 3 | 0 | 3 | 13 | 16 | -1 | 26.0s |
| cutthroat | baseline | 35 | 55 | -20 | 30 | 30 | -1 | 54.9s |
| cutthroat | edge_impact | 35 | 4 | 31 | 13 | 22 | -3 | 31.6s |
| cutthroat | vc_ei | 35 | 35 | 0 | 9 | 15 | -3 | 23.2s |
| cutthroat | vc_only | 35 | 60 | -25 | 30 | 30 | -1 | 44.8s |
| deephome | baseline | 1 | 1 | 0 | 0 | 3 | +0 | 2.3s |
| deephome | edge_impact | 1 | 1 | 0 | 1 | 3 | -1 | 3.5s |
| deephome | vc_ei | 1 | 1 | 0 | 1 | 3 | -1 | 5.2s |
| deephome | vc_only | 1 | 1 | 0 | 0 | 3 | +0 | 2.7s |
| detective | baseline | 40 | 17 | 23 | 10 | 16 | -2 | 24.2s |
| detective | edge_impact | 40 | 15 | 25 | 8 | 13 | -2 | 16.8s |
| detective | vc_ei | 40 | 67 | -27 | 29 | 30 | -2 | 56.3s |
| detective | vc_only | 40 | 70 | -30 | 27 | 30 | -2 | 55.4s |
| inhumane | baseline | 9 | 35 | -26 | 8 | 14 | +0 | 21.9s |
| inhumane | edge_impact | 9 | 1 | 8 | 3 | 6 | +0 | 9.4s |
| inhumane | vc_ei | 9 | 30 | -21 | 2 | 8 | +0 | 12.3s |
| inhumane | vc_only | 9 | 33 | -24 | 6 | 15 | +0 | 19.5s |
| murdac | baseline | 29 | 4 | 25 | 11 | 17 | -2 | 22.4s |
| murdac | edge_impact | 29 | 5 | 24 | 5 | 11 | -2 | 14.6s |
| murdac | vc_ei | 29 | 4 | 25 | 5 | 11 | -2 | 20.2s |
| murdac | vc_only | 29 | 50 | -21 | 27 | 30 | -2 | 45.0s |
| sherlock | baseline | 3 | 0 | 3 | 1 | 1 | -1 | 1.2s |
| sherlock | edge_impact | 3 | 2 | 1 | 1 | 7 | -1 | 6.3s |
| sherlock | vc_ei | 3 | 3 | 0 | 2 | 7 | +0 | 9.9s |
| sherlock | vc_only | 3 | 0 | 3 | 1 | 1 | -1 | 1.5s |
| wishbringer | baseline | 5 | 41 | -36 | 5 | 11 | -2 | 15.9s |
| wishbringer | edge_impact | 5 | 2 | 3 | 3 | 9 | -2 | 11.1s |
| wishbringer | vc_ei | 5 | 2 | 3 | 3 | 9 | -2 | 11.9s |
| wishbringer | vc_only | 5 | 42 | -37 | 12 | 18 | -3 | 26.7s |
| zork1 | baseline | 17 | 31 | -14 | 3 | 9 | +0 | 10.9s |
| zork1 | edge_impact | 17 | 0 | 17 | 3 | 3 | +0 | 5.1s |
| zork1 | vc_ei | 17 | 28 | -11 | 7 | 15 | +0 | 20.7s |
| zork1 | vc_only | 17 | 31 | -14 | 4 | 10 | +0 | 12.6s |
| zork2 | baseline | 6 | 39 | -33 | 6 | 12 | -2 | 16.1s |
| zork2 | edge_impact | 6 | 3 | 3 | 3 | 9 | -2 | 8.6s |
| zork2 | vc_ei | 6 | 0 | 6 | 3 | 3 | -2 | 4.3s |
| zork2 | vc_only | 6 | 0 | 6 | 3 | 3 | -2 | 4.5s |
