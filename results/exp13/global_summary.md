# Experiment 13 — V3 mapping error classification

Per-edge buckets for the exp12 gpt-4.1 run on V3 clean walkthroughs.

Total predicted edges: 1076
Total GT edges:        1510

## Predicted-edge buckets

| Bucket | Count | % of pred |
|--------|------:|----------:|
| CORRECT | 877 | 81.5% |
| SPURIOUS_PAIR | 94 | 8.7% |
| WRONG_DST | 56 | 5.2% |
| WRONG_DIRECTION | 16 | 1.5% |
| SELF_LOOP | 12 | 1.1% |
| HALLUCINATED_DST | 9 | 0.8% |
| HALLUCINATED_SRC | 8 | 0.7% |
| WRONG_SRC_UNKNOWN | 4 | 0.4% |

## GT-edge buckets (recall view)

| Bucket | Count | % of GT |
|--------|------:|--------:|
| RECALLED | 877 | 58.1% |
| MISSED | 625 | 41.4% |
| RECALLED_WRONG_DIR | 8 | 0.5% |

## Per-game error breakdown

| game | n_pred | CORRECT | WRONG_DIR | WRONG_DST | WRONG_SRC | SPURIOUS | HALL_DST | n_gt | MISSED |
|------|------:|--------:|----------:|----------:|----------:|--------:|---------:|----:|------:|
| cutthroat | 33 | 16 | 1 | 6 | 0 | 6 | 1 | 49 | 32 |
| inhumane | 28 | 21 | 1 | 0 | 0 | 6 | 0 | 50 | 28 |
| murdac | 31 | 23 | 0 | 4 | 0 | 2 | 0 | 48 | 25 |
| detective | 31 | 17 | 0 | 10 | 0 | 1 | 1 | 39 | 22 |
| infidel | 27 | 26 | 0 | 0 | 0 | 1 | 0 | 48 | 22 |
| omniquest | 38 | 36 | 0 | 0 | 0 | 2 | 0 | 57 | 21 |
| sorcerer | 24 | 23 | 0 | 0 | 0 | 1 | 0 | 43 | 20 |
| zork3 | 24 | 19 | 0 | 1 | 0 | 3 | 0 | 38 | 19 |
| advent | 38 | 35 | 0 | 0 | 0 | 3 | 0 | 51 | 16 |
| anchor | 27 | 26 | 0 | 1 | 0 | 0 | 0 | 42 | 16 |
| enchanter | 23 | 23 | 0 | 0 | 0 | 0 | 0 | 39 | 16 |
| spellbrkr | 14 | 7 | 1 | 1 | 0 | 5 | 0 | 24 | 16 |
| karn | 24 | 16 | 3 | 2 | 0 | 3 | 0 | 33 | 15 |
| night | 28 | 16 | 2 | 3 | 0 | 7 | 0 | 32 | 15 |
| temple | 20 | 17 | 1 | 2 | 0 | 0 | 0 | 32 | 15 |
| zenon | 15 | 10 | 1 | 3 | 0 | 1 | 0 | 26 | 15 |
| adventureland | 22 | 13 | 0 | 3 | 0 | 5 | 0 | 27 | 14 |
| ballyhoo | 23 | 19 | 0 | 2 | 0 | 2 | 0 | 33 | 14 |
| dragon | 31 | 28 | 0 | 1 | 0 | 2 | 0 | 42 | 14 |
| lurking | 16 | 12 | 2 | 0 | 0 | 2 | 0 | 26 | 14 |
| planetfall | 27 | 26 | 0 | 1 | 0 | 0 | 0 | 40 | 14 |
| wishbringer | 26 | 23 | 1 | 1 | 0 | 1 | 0 | 37 | 14 |
| zork1 | 20 | 15 | 0 | 0 | 1 | 4 | 0 | 29 | 14 |
| pentari | 19 | 13 | 0 | 0 | 0 | 3 | 1 | 26 | 13 |
| reverb | 20 | 12 | 0 | 2 | 0 | 4 | 1 | 25 | 13 |
| trinity | 9 | 7 | 0 | 0 | 0 | 2 | 0 | 20 | 13 |
| afflicted | 18 | 8 | 0 | 2 | 1 | 2 | 2 | 20 | 12 |
| ludicorp | 35 | 31 | 0 | 1 | 0 | 0 | 1 | 43 | 12 |
| zork2 | 32 | 25 | 0 | 1 | 0 | 5 | 0 | 36 | 11 |
| plundered | 22 | 20 | 0 | 0 | 0 | 2 | 0 | 30 | 10 |
| lostpig | 4 | 2 | 0 | 1 | 1 | 0 | 0 | 11 | 9 |
| sherlock | 22 | 20 | 0 | 1 | 0 | 1 | 0 | 29 | 9 |
| awaken | 20 | 16 | 0 | 2 | 0 | 2 | 0 | 24 | 8 |
| deephome | 40 | 38 | 0 | 0 | 0 | 2 | 0 | 46 | 8 |
| yomomma | 14 | 10 | 3 | 1 | 0 | 0 | 0 | 19 | 8 |
| jewel | 26 | 24 | 0 | 1 | 0 | 1 | 0 | 31 | 7 |
| loose | 15 | 14 | 0 | 0 | 0 | 1 | 0 | 21 | 7 |
| spirit | 36 | 33 | 0 | 0 | 0 | 1 | 1 | 40 | 7 |
| balances | 9 | 7 | 0 | 0 | 0 | 1 | 1 | 13 | 6 |
| enter | 21 | 20 | 0 | 1 | 0 | 0 | 0 | 26 | 6 |
| gold | 18 | 16 | 0 | 1 | 0 | 1 | 0 | 22 | 6 |
| huntdark | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 6 | 6 |
| ztuu | 21 | 20 | 0 | 1 | 0 | 0 | 0 | 26 | 6 |
| curses | 23 | 21 | 0 | 0 | 0 | 2 | 0 | 26 | 5 |
| hollywood | 17 | 17 | 0 | 0 | 0 | 0 | 0 | 22 | 5 |
| moonlit | 4 | 3 | 0 | 0 | 0 | 1 | 0 | 8 | 5 |
| seastalker | 8 | 7 | 0 | 0 | 0 | 1 | 0 | 12 | 5 |
| 905 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 6 | 4 |
| hhgg | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 4 | 4 |
| library | 10 | 9 | 0 | 0 | 0 | 1 | 0 | 12 | 3 |
| tryst205 | 7 | 6 | 0 | 0 | 0 | 0 | 0 | 9 | 3 |
| snacktime | 4 | 3 | 0 | 0 | 1 | 0 | 0 | 5 | 2 |
| partyfoul | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 7 | 1 |
