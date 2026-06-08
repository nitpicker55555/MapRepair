# exp31 - max_iterations sensitivity

Model: gpt-5.5  Mode: edge_impact  Games: ['cutthroat', 'zork1', 'murdac', 'wishbringer', 'adventureland']

## Aggregate by max_iter

| max_iter | n | cf_before | cf_after | repaired | repair % | avg iters used | avg elapsed |
|---------:|--:|---------:|---------:|---------:|--------:|---------------:|------------:|
| 5 | 5 | 108 | 141 | -33 | -30.6 | 5.0 | 88.7s |
| 10 | 5 | 108 | 143 | -35 | -32.4 | 8.8 | 125.4s |
| 20 | 5 | 108 | 53 | 55 | 50.9 | 12.2 | 144.3s |
| 40 | 5 | 108 | 127 | -19 | -17.6 | 17.6 | 267.8s |

## Per-game per-max_iter

| game | max_iter | cb | ca | repaired | iters used | elapsed |
|------|---------:|---:|---:|--------:|----------:|--------:|
| adventureland | 5 | 22 | 23 | -1 | 5 | 71.7s |
| adventureland | 10 | 22 | 23 | -1 | 10 | 136.9s |
| adventureland | 20 | 22 | 1 | 21 | 8 | 79.0s |
| adventureland | 40 | 22 | 1 | 21 | 8 | 80.3s |
| cutthroat | 5 | 35 | 58 | -23 | 5 | 50.1s |
| cutthroat | 10 | 35 | 57 | -22 | 10 | 193.8s |
| cutthroat | 20 | 35 | 30 | 5 | 20 | 300.3s |
| cutthroat | 40 | 35 | 43 | -8 | 40 | 602.4s |
| murdac | 5 | 29 | 32 | -3 | 5 | 196.8s |
| murdac | 10 | 29 | 60 | -31 | 10 | 119.6s |
| murdac | 20 | 29 | 4 | 25 | 20 | 234.1s |
| murdac | 40 | 29 | 52 | -23 | 14 | 154.1s |
| wishbringer | 5 | 5 | 28 | -23 | 5 | 63.3s |
| wishbringer | 10 | 5 | 3 | 2 | 10 | 125.5s |
| wishbringer | 20 | 5 | 18 | -13 | 10 | 83.5s |
| wishbringer | 40 | 5 | 16 | -11 | 17 | 296.0s |
| zork1 | 5 | 17 | 0 | 17 | 5 | 61.9s |
| zork1 | 10 | 17 | 0 | 17 | 4 | 51.5s |
| zork1 | 20 | 17 | 0 | 17 | 3 | 24.8s |
| zork1 | 40 | 17 | 15 | 2 | 9 | 206.1s |
