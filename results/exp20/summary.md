# Experiment 20 — End-to-end MANGO repair (model=gpt-4.1)

Real-benchmark validation. Input: exp14 V3 gpt-4.1 LLM-built maps on the 53 refined MANGO games. Compared 4 repair conditions per game.

Total runs: 212 (valid 212, errors 0).

## Aggregate by mode (macro across 53 games)

| mode | n | conf reduction % | edge recall before | edge recall after | Δ edge recall | edge prec after | dir acc after | iters | actions/run | remove/run |
|------|--:|-----------------:|-------------------:|------------------:|--------------:|----------------:|--------------:|------:|------------:|-----------:|
| no_repair | 53 | 0.0 | 42.5 | 42.5 | +0.00pp | 79.5 | 96.3 | 0.0 | 0.0 | 0.0 |
| heuristic_remove | 53 | 65.0 | 42.5 | 39.8 | -2.75pp | 81.5 | 94.1 | 2.0 | 2.6 | 1.5 |
| llm_baseline | 53 | -22.3 | 42.5 | 42.0 | -0.50pp | 79.9 | 91.5 | 8.2 | 4.5 | 0.8 |
| llm_edge_impact | 53 | 49.4 | 42.5 | 39.3 | -3.19pp | 81.5 | 96.1 | 5.4 | 2.2 | 1.9 |

## Per-game lift (llm_edge_impact)

| game | edges before | conflicts before | edges after | conflicts after | edge recall before | after | Δ | actions | remove |
|------|-------------:|-----------------:|------------:|----------------:|-------------------:|------:|--:|--------:|-------:|
| 905 | 3 | 0 | 3 | 0 | 50.0 | 50.0 | +0.00pp | 0 | 0 |
| anchor | 22 | 2 | 21 | 1 | 47.6 | 47.6 | +0.00pp | 1 | 1 |
| enchanter | 20 | 0 | 20 | 0 | 51.3 | 51.3 | +0.00pp | 0 | 0 |
| hhgg | 2 | 0 | 2 | 0 | 25.0 | 25.0 | +0.00pp | 0 | 0 |
| enter | 13 | 3 | 12 | 4 | 46.2 | 46.2 | +0.00pp | 1 | 1 |
| huntdark | 3 | 0 | 3 | 0 | 0.0 | 0.0 | +0.00pp | 0 | 0 |
| inhumane | 26 | 35 | 25 | 1 | 40.0 | 40.0 | +0.00pp | 2 | 2 |
| jewel | 16 | 36 | 14 | 2 | 45.2 | 45.2 | +0.00pp | 2 | 2 |
| lostpig | 4 | 0 | 4 | 0 | 27.3 | 27.3 | +0.00pp | 0 | 0 |
| loose | 11 | 1 | 10 | 1 | 47.6 | 47.6 | +0.00pp | 2 | 1 |
| moonlit | 4 | 0 | 4 | 0 | 37.5 | 37.5 | +0.00pp | 0 | 0 |
| karn | 18 | 21 | 18 | 18 | 42.4 | 42.4 | +0.00pp | 5 | 0 |
| ludicorp | 21 | 4 | 20 | 4 | 41.9 | 41.9 | +0.00pp | 1 | 1 |
| partyfoul | 4 | 0 | 4 | 0 | 57.1 | 57.1 | +0.00pp | 0 | 0 |
| seastalker | 6 | 0 | 6 | 0 | 50.0 | 50.0 | +0.00pp | 0 | 0 |
| snacktime | 3 | 0 | 3 | 0 | 40.0 | 40.0 | +0.00pp | 0 | 0 |
| sorcerer | 23 | 1 | 23 | 1 | 51.2 | 51.2 | +0.00pp | 0 | 0 |
| trinity | 10 | 0 | 10 | 0 | 40.0 | 40.0 | +0.00pp | 0 | 0 |
| tryst205 | 4 | 0 | 4 | 0 | 44.4 | 44.4 | +0.00pp | 0 | 0 |
| temple | 16 | 4 | 14 | 3 | 43.8 | 43.8 | +0.00pp | 2 | 2 |
| zork1 | 16 | 32 | 14 | 0 | 41.4 | 41.4 | +0.00pp | 3 | 3 |
| ztuu | 14 | 32 | 13 | 1 | 50.0 | 50.0 | +0.00pp | 1 | 1 |
| omniquest | 28 | 50 | 27 | 50 | 45.6 | 43.9 | -1.75pp | 8 | 1 |
| infidel | 25 | 49 | 24 | 1 | 50.0 | 47.9 | -2.08pp | 1 | 1 |
| deephome | 23 | 1 | 22 | 1 | 45.7 | 43.5 | -2.17pp | 1 | 1 |
| planetfall | 20 | 2 | 19 | 1 | 47.5 | 45.0 | -2.50pp | 1 | 1 |
| spirit | 20 | 43 | 19 | 0 | 45.0 | 42.5 | -2.50pp | 1 | 1 |
| sherlock | 15 | 3 | 14 | 2 | 48.3 | 44.8 | -3.45pp | 1 | 1 |
| lurking | 13 | 27 | 11 | 1 | 42.3 | 38.5 | -3.85pp | 2 | 2 |
| pentari | 13 | 2 | 12 | 2 | 34.6 | 30.8 | -3.85pp | 1 | 1 |
| advent | 31 | 3 | 29 | 1 | 54.9 | 51.0 | -3.92pp | 2 | 2 |
| awaken | 15 | 31 | 12 | 1 | 45.8 | 41.7 | -4.17pp | 3 | 3 |
| murdac | 27 | 56 | 22 | 5 | 43.8 | 39.6 | -4.17pp | 5 | 5 |
| hollywood | 10 | 1 | 9 | 1 | 45.5 | 40.9 | -4.55pp | 1 | 1 |
| gold | 13 | 7 | 11 | 1 | 50.0 | 45.5 | -4.55pp | 2 | 2 |
| dragon | 21 | 38 | 18 | 1 | 42.9 | 38.1 | -4.76pp | 5 | 3 |
| detective | 30 | 68 | 24 | 15 | 43.6 | 38.5 | -5.13pp | 7 | 7 |
| yomomma | 10 | 22 | 9 | 0 | 47.4 | 42.1 | -5.26pp | 2 | 1 |
| zork3 | 22 | 37 | 19 | 22 | 44.7 | 39.5 | -5.26pp | 4 | 3 |
| wishbringer | 21 | 5 | 18 | 2 | 51.4 | 45.9 | -5.41pp | 3 | 3 |
| zork2 | 22 | 6 | 19 | 3 | 44.4 | 38.9 | -5.56pp | 3 | 3 |
| ballyhoo | 19 | 35 | 15 | 3 | 45.5 | 39.4 | -6.06pp | 4 | 4 |
| cutthroat | 29 | 58 | 20 | 5 | 30.6 | 24.5 | -6.12pp | 11 | 10 |
| night | 20 | 43 | 16 | 19 | 40.6 | 34.4 | -6.25pp | 4 | 4 |
| plundered | 15 | 26 | 13 | 1 | 43.3 | 36.7 | -6.67pp | 2 | 2 |
| zenon | 13 | 29 | 11 | 0 | 26.9 | 19.2 | -7.69pp | 2 | 2 |
| balances | 7 | 2 | 6 | 1 | 38.5 | 30.8 | -7.69pp | 1 | 1 |
| curses | 15 | 34 | 13 | 1 | 50.0 | 42.3 | -7.69pp | 2 | 2 |
| reverb | 16 | 37 | 12 | 2 | 32.0 | 24.0 | -8.00pp | 4 | 4 |
| library | 7 | 15 | 6 | 0 | 50.0 | 41.7 | -8.33pp | 1 | 1 |
| spellbrkr | 13 | 3 | 11 | 1 | 29.2 | 20.8 | -8.33pp | 2 | 2 |
| afflicted | 13 | 25 | 8 | 3 | 30.0 | 20.0 | -10.00pp | 6 | 6 |
| adventureland | 17 | 37 | 13 | 1 | 44.4 | 33.3 | -11.11pp | 5 | 5 |

## Subset summary (llm_edge_impact)

- Games with **>= +1pp** edge-recall lift: **0**/53
- Games **flat** (|Δ| < 1pp): 22/53
- Games **hurt** (<= -1pp): 31/53
