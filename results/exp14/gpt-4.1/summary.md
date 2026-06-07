# Experiment 14 — V3 LLM mapping with bootstrap+drift fixes (gpt-4.1)

Games: 53
Total edges produced: 1129
Total GT edges: 1510
Total conflicts in resulting graphs: 1081

## Macro metrics — exp14 (fixed) vs exp12 (V3 unfixed)

| Metric | exp14 fixed | exp12 V3 | Δ |
|--------|-----------:|---------:|---:|
| node_recall | 89.72% | 85.01% | +4.71pp |
| edge_recall | 61.73% | 56.07% | +5.67pp |
| direction_accuracy | 96.56% | 94.80% | +1.75pp |
| conflicts / game | 20.40 | 20.51 | -0.11 |

## Per-game

| game | gt_edges | pred_edges | edge_recall % | dir_acc % | conflicts |
|------|---------:|-----------:|--------------:|----------:|----------:|
| detective | 39 | 34 | 51.3 | 100.0 | 74 |
| cutthroat | 49 | 35 | 38.8 | 94.7 | 61 |
| murdac | 48 | 37 | 64.6 | 100.0 | 56 |
| omniquest | 57 | 39 | 64.9 | 100.0 | 50 |
| infidel | 48 | 28 | 56.2 | 100.0 | 49 |
| night | 32 | 28 | 62.5 | 90.0 | 49 |
| spirit | 40 | 36 | 82.5 | 100.0 | 43 |
| adventureland | 27 | 21 | 51.9 | 92.9 | 41 |
| zork2 | 36 | 32 | 69.4 | 100.0 | 40 |
| dragon | 42 | 31 | 66.7 | 100.0 | 38 |
| reverb | 25 | 22 | 56.0 | 100.0 | 37 |
| zork3 | 38 | 25 | 52.6 | 100.0 | 37 |
| jewel | 31 | 27 | 80.6 | 100.0 | 36 |
| ballyhoo | 33 | 25 | 63.6 | 100.0 | 35 |
| inhumane | 50 | 29 | 46.0 | 95.7 | 35 |
| curses | 26 | 24 | 84.6 | 100.0 | 34 |
| zork1 | 29 | 21 | 58.6 | 100.0 | 32 |
| ztuu | 26 | 25 | 92.3 | 100.0 | 32 |
| karn | 33 | 23 | 57.6 | 100.0 | 30 |
| wishbringer | 37 | 26 | 64.9 | 100.0 | 30 |
| awaken | 24 | 21 | 70.8 | 100.0 | 29 |
| zenon | 26 | 15 | 30.8 | 87.5 | 29 |
| lurking | 26 | 15 | 50.0 | 84.6 | 28 |
| spellbrkr | 24 | 15 | 37.5 | 88.9 | 27 |
| plundered | 30 | 23 | 70.0 | 100.0 | 26 |
| yomomma | 19 | 13 | 63.2 | 83.3 | 26 |
| afflicted | 20 | 17 | 45.0 | 100.0 | 25 |
| library | 12 | 11 | 83.3 | 100.0 | 15 |
| gold | 22 | 21 | 86.4 | 100.0 | 7 |
| ludicorp | 43 | 35 | 74.4 | 100.0 | 4 |
| temple | 32 | 20 | 56.2 | 100.0 | 4 |
| advent | 51 | 40 | 72.5 | 100.0 | 3 |
| enter | 26 | 22 | 80.8 | 100.0 | 3 |
| sherlock | 29 | 22 | 72.4 | 100.0 | 3 |
| anchor | 42 | 29 | 64.3 | 100.0 | 2 |
| balances | 13 | 10 | 61.5 | 100.0 | 2 |
| pentari | 26 | 20 | 57.7 | 100.0 | 2 |
| planetfall | 40 | 28 | 67.5 | 100.0 | 2 |
| deephome | 46 | 41 | 84.8 | 100.0 | 1 |
| hollywood | 22 | 17 | 77.3 | 100.0 | 1 |
| loose | 21 | 17 | 76.2 | 100.0 | 1 |
| seastalker | 12 | 9 | 66.7 | 100.0 | 1 |
| sorcerer | 43 | 28 | 62.8 | 100.0 | 1 |
| 905 | 6 | 3 | 50.0 | 100.0 | 0 |
| enchanter | 39 | 27 | 69.2 | 100.0 | 0 |
| hhgg | 4 | 2 | 25.0 | 100.0 | 0 |
| huntdark | 6 | 3 | 0.0 | 0.0 | 0 |
| lostpig | 11 | 5 | 36.4 | 100.0 | 0 |
| moonlit | 8 | 5 | 50.0 | 100.0 | 0 |
| partyfoul | 7 | 6 | 85.7 | 100.0 | 0 |
| snacktime | 5 | 4 | 60.0 | 100.0 | 0 |
| trinity | 20 | 10 | 40.0 | 100.0 | 0 |
| tryst205 | 9 | 7 | 77.8 | 100.0 | 0 |
