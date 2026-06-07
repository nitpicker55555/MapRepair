# Experiment 12 — LLM mapping on V3 clean walkthroughs (model=gpt-4.1)

Games: 53
Total edges produced: 1054
Total GT edges: 1510
Total conflicts in resulting graphs: 1087

## Macro metrics vs the old repaired_walkthroughs run

| Metric | V3 clean (this run) | Old run | Δ |
|--------|--------------------:|--------:|---:|
| node_recall | 85.01% | 95.58% | -10.57pp |
| edge_recall | 56.07% | 71.43% | -15.36pp |
| direction_accuracy | 94.80% | 73.64% | +21.16pp |
| conflicts / game | 20.51 | (n/a) | — |

## Per-game (sorted by conflict count desc)

| game | gt_edges | pred_edges | edge_recall % | dir_acc % | conflicts |
|------|---------:|-----------:|--------------:|----------:|----------:|
| detective | 39 | 30 | 43.6 | 100.0 | 63 |
| cutthroat | 49 | 30 | 34.7 | 94.1 | 57 |
| night | 32 | 26 | 53.1 | 88.2 | 50 |
| omniquest | 57 | 38 | 63.2 | 100.0 | 50 |
| infidel | 48 | 27 | 54.2 | 100.0 | 49 |
| murdac | 48 | 29 | 47.9 | 100.0 | 48 |
| karn | 33 | 23 | 54.5 | 83.3 | 45 |
| adventureland | 27 | 21 | 48.1 | 100.0 | 43 |
| spirit | 40 | 36 | 82.5 | 100.0 | 43 |
| sherlock | 29 | 22 | 69.0 | 100.0 | 39 |
| zork2 | 36 | 31 | 69.4 | 100.0 | 39 |
| dragon | 42 | 31 | 66.7 | 100.0 | 38 |
| reverb | 25 | 20 | 48.0 | 100.0 | 37 |
| jewel | 31 | 26 | 77.4 | 100.0 | 36 |
| inhumane | 50 | 28 | 44.0 | 95.5 | 35 |
| ballyhoo | 33 | 23 | 57.6 | 100.0 | 34 |
| zork1 | 29 | 20 | 51.7 | 100.0 | 34 |
| curses | 26 | 23 | 80.8 | 100.0 | 33 |
| zork3 | 38 | 23 | 50.0 | 100.0 | 31 |
| afflicted | 20 | 17 | 40.0 | 100.0 | 29 |
| wishbringer | 37 | 25 | 62.2 | 100.0 | 29 |
| awaken | 24 | 20 | 66.7 | 100.0 | 28 |
| lurking | 26 | 14 | 46.2 | 100.0 | 28 |
| ztuu | 26 | 21 | 76.9 | 100.0 | 28 |
| plundered | 30 | 22 | 66.7 | 100.0 | 26 |
| spellbrkr | 24 | 14 | 33.3 | 87.5 | 26 |
| yomomma | 19 | 12 | 57.9 | 90.9 | 26 |
| zenon | 26 | 15 | 42.3 | 90.9 | 23 |
| gold | 22 | 18 | 72.7 | 100.0 | 7 |
| ludicorp | 43 | 34 | 72.1 | 100.0 | 4 |
| sorcerer | 43 | 24 | 53.5 | 100.0 | 4 |
| temple | 32 | 19 | 53.1 | 94.1 | 4 |
| advent | 51 | 38 | 68.6 | 100.0 | 3 |
| anchor | 42 | 27 | 61.9 | 100.0 | 2 |
| balances | 13 | 9 | 53.8 | 100.0 | 2 |
| enter | 26 | 21 | 76.9 | 100.0 | 2 |
| library | 12 | 10 | 75.0 | 100.0 | 2 |
| lostpig | 11 | 4 | 18.2 | 100.0 | 2 |
| pentari | 26 | 18 | 50.0 | 100.0 | 2 |
| planetfall | 40 | 27 | 65.0 | 100.0 | 2 |
| deephome | 46 | 40 | 82.6 | 100.0 | 1 |
| hollywood | 22 | 17 | 77.3 | 100.0 | 1 |
| loose | 21 | 15 | 66.7 | 100.0 | 1 |
| seastalker | 12 | 8 | 58.3 | 100.0 | 1 |
| 905 | 6 | 2 | 33.3 | 100.0 | 0 |
| enchanter | 39 | 23 | 59.0 | 100.0 | 0 |
| hhgg | 4 | 1 | 0.0 | 0.0 | 0 |
| huntdark | 6 | 3 | 0.0 | 0.0 | 0 |
| moonlit | 8 | 4 | 37.5 | 100.0 | 0 |
| partyfoul | 7 | 6 | 85.7 | 100.0 | 0 |
| snacktime | 5 | 4 | 60.0 | 100.0 | 0 |
| trinity | 20 | 9 | 35.0 | 100.0 | 0 |
| tryst205 | 9 | 6 | 66.7 | 100.0 | 0 |
