# exp30 - Cross-vendor MANGO repair (input maps: gpt-4.1 V3)

Games: ['cutthroat', 'detective', 'inhumane', 'zork1', 'zork2', 'murdac', 'advent', 'sherlock', 'wishbringer', 'deephome']
Models: ('gpt-5.5', 'gemini-3.5-flash', 'claude-haiku-4-5-20251001')
Modes: ('baseline', 'edge_impact')

## Aggregate by model x mode

| Model | Mode | n | conf_before | conf_after | repaired | repair % | dir_delta | avg iters |
|------|------|--:|-----------:|----------:|---------:|--------:|----------:|----------:|
| claude-haiku-4-5-20251001 | baseline | 10 | 148 | 339 | -191 | -129.1 | -8 | 10.6 |
| claude-haiku-4-5-20251001 | edge_impact | 10 | 148 | 254 | -106 | -71.6 | -19 | 11.5 |
| gemini-3.5-flash | baseline | 10 | 148 | 347 | -199 | -134.5 | -11 | 11.3 |
| gemini-3.5-flash | edge_impact | 10 | 148 | 265 | -117 | -79.1 | -16 | 10.6 |
| gpt-5.5 | baseline | 10 | 148 | 183 | -35 | -23.6 | -14 | 11.1 |
| gpt-5.5 | edge_impact | 10 | 148 | 125 | 23 | 15.5 | -23 | 11.7 |

## Per-game per-(model, mode)

| game | model | mode | cb | ca | repaired | actions | iters | dir_delta | elapsed |
|------|------|------|---:|---:|--------:|--------:|------:|----------:|--------:|
| advent | claude-haiku-4-5-20251001 | baseline | 3 | 53 | -50 | 4 | 11 | -1 | 38.1s |
| advent | claude-haiku-4-5-20251001 | edge_impact | 3 | 1 | 2 | 3 | 6 | -1 | 18.0s |
| advent | gemini-3.5-flash | baseline | 3 | 37 | -34 | 3 | 9 | -1 | 34.3s |
| advent | gemini-3.5-flash | edge_impact | 3 | 60 | -57 | 4 | 10 | -2 | 82.3s |
| advent | gpt-5.5 | baseline | 3 | 1 | 2 | 3 | 6 | +0 | 79.5s |
| advent | gpt-5.5 | edge_impact | 3 | 0 | 3 | 2 | 2 | -1 | 62.5s |
| cutthroat | claude-haiku-4-5-20251001 | baseline | 35 | 68 | -33 | 16 | 20 | -1 | 75.1s |
| cutthroat | claude-haiku-4-5-20251001 | edge_impact | 35 | 62 | -27 | 8 | 17 | -1 | 55.2s |
| cutthroat | gemini-3.5-flash | baseline | 35 | 60 | -25 | 18 | 20 | -2 | 121.0s |
| cutthroat | gemini-3.5-flash | edge_impact | 35 | 61 | -26 | 16 | 20 | -3 | 128.6s |
| cutthroat | gpt-5.5 | baseline | 35 | 53 | -18 | 17 | 20 | -2 | 306.6s |
| cutthroat | gpt-5.5 | edge_impact | 35 | 53 | -18 | 19 | 20 | -3 | 263.9s |
| deephome | claude-haiku-4-5-20251001 | baseline | 1 | 1 | 0 | 0 | 3 | +0 | 6.4s |
| deephome | claude-haiku-4-5-20251001 | edge_impact | 1 | 1 | 0 | 1 | 3 | -1 | 10.3s |
| deephome | gemini-3.5-flash | baseline | 1 | 1 | 0 | 0 | 3 | +0 | 7.9s |
| deephome | gemini-3.5-flash | edge_impact | 1 | 0 | 1 | 2 | 2 | -1 | 11.8s |
| deephome | gpt-5.5 | baseline | 1 | 1 | 0 | 0 | 3 | +0 | 11.1s |
| deephome | gpt-5.5 | edge_impact | 1 | 1 | 0 | 3 | 6 | -1 | 60.0s |
| detective | claude-haiku-4-5-20251001 | baseline | 40 | 53 | -13 | 18 | 20 | -3 | 67.7s |
| detective | claude-haiku-4-5-20251001 | edge_impact | 40 | 71 | -31 | 14 | 20 | -2 | 68.3s |
| detective | gemini-3.5-flash | baseline | 40 | 73 | -33 | 5 | 11 | +0 | 88.7s |
| detective | gemini-3.5-flash | edge_impact | 40 | 43 | -3 | 11 | 19 | -2 | 128.6s |
| detective | gpt-5.5 | baseline | 40 | 2 | 38 | 8 | 14 | -1 | 100.9s |
| detective | gpt-5.5 | edge_impact | 40 | 2 | 38 | 16 | 20 | -1 | 181.4s |
| inhumane | claude-haiku-4-5-20251001 | baseline | 9 | 36 | -27 | 3 | 9 | +0 | 28.4s |
| inhumane | claude-haiku-4-5-20251001 | edge_impact | 9 | 33 | -24 | 3 | 14 | +0 | 43.6s |
| inhumane | gemini-3.5-flash | baseline | 9 | 22 | -13 | 3 | 9 | +0 | 29.9s |
| inhumane | gemini-3.5-flash | edge_impact | 9 | 22 | -13 | 3 | 9 | +0 | 36.6s |
| inhumane | gpt-5.5 | baseline | 9 | 13 | -4 | 9 | 15 | -1 | 126.2s |
| inhumane | gpt-5.5 | edge_impact | 9 | 22 | -13 | 3 | 9 | +0 | 45.8s |
| murdac | claude-haiku-4-5-20251001 | baseline | 29 | 43 | -14 | 4 | 13 | -1 | 41.9s |
| murdac | claude-haiku-4-5-20251001 | edge_impact | 29 | 61 | -32 | 6 | 12 | -3 | 36.5s |
| murdac | gemini-3.5-flash | baseline | 29 | 64 | -35 | 14 | 20 | -2 | 122.5s |
| murdac | gemini-3.5-flash | edge_impact | 29 | 2 | 27 | 9 | 17 | -3 | 120.3s |
| murdac | gpt-5.5 | baseline | 29 | 5 | 24 | 15 | 20 | -3 | 231.4s |
| murdac | gpt-5.5 | edge_impact | 29 | 4 | 25 | 11 | 17 | -3 | 160.1s |
| sherlock | claude-haiku-4-5-20251001 | baseline | 3 | 0 | 3 | 1 | 1 | -1 | 2.7s |
| sherlock | claude-haiku-4-5-20251001 | edge_impact | 3 | 0 | 3 | 1 | 1 | -1 | 2.3s |
| sherlock | gemini-3.5-flash | baseline | 3 | 1 | 2 | 2 | 5 | +0 | 20.3s |
| sherlock | gemini-3.5-flash | edge_impact | 3 | 0 | 3 | 1 | 1 | -1 | 9.2s |
| sherlock | gpt-5.5 | baseline | 3 | 0 | 3 | 1 | 1 | -1 | 7.6s |
| sherlock | gpt-5.5 | edge_impact | 3 | 0 | 3 | 1 | 1 | -1 | 15.9s |
| wishbringer | claude-haiku-4-5-20251001 | baseline | 5 | 47 | -42 | 7 | 17 | -1 | 60.1s |
| wishbringer | claude-haiku-4-5-20251001 | edge_impact | 5 | 1 | 4 | 7 | 13 | -3 | 49.9s |
| wishbringer | gemini-3.5-flash | baseline | 5 | 27 | -22 | 4 | 10 | -2 | 46.3s |
| wishbringer | gemini-3.5-flash | edge_impact | 5 | 20 | -15 | 4 | 10 | -3 | 57.5s |
| wishbringer | gpt-5.5 | baseline | 5 | 46 | -41 | 6 | 12 | -3 | 108.1s |
| wishbringer | gpt-5.5 | edge_impact | 5 | 16 | -11 | 8 | 19 | -4 | 220.5s |
| zork1 | claude-haiku-4-5-20251001 | baseline | 17 | 32 | -15 | 0 | 6 | +0 | 15.6s |
| zork1 | claude-haiku-4-5-20251001 | edge_impact | 17 | 23 | -6 | 4 | 11 | -2 | 30.5s |
| zork1 | gemini-3.5-flash | baseline | 17 | 19 | -2 | 4 | 10 | +0 | 49.6s |
| zork1 | gemini-3.5-flash | edge_impact | 17 | 15 | 2 | 3 | 10 | +0 | 75.2s |
| zork1 | gpt-5.5 | baseline | 17 | 31 | -14 | 1 | 7 | +0 | 32.1s |
| zork1 | gpt-5.5 | edge_impact | 17 | 0 | 17 | 3 | 3 | +0 | 24.9s |
| zork2 | claude-haiku-4-5-20251001 | baseline | 6 | 6 | 0 | 0 | 6 | +0 | 26.2s |
| zork2 | claude-haiku-4-5-20251001 | edge_impact | 6 | 1 | 5 | 12 | 18 | -5 | 68.1s |
| zork2 | gemini-3.5-flash | baseline | 6 | 43 | -37 | 10 | 16 | -4 | 93.6s |
| zork2 | gemini-3.5-flash | edge_impact | 6 | 42 | -36 | 3 | 8 | -1 | 63.8s |
| zork2 | gpt-5.5 | baseline | 6 | 31 | -25 | 7 | 13 | -3 | 127.5s |
| zork2 | gpt-5.5 | edge_impact | 6 | 27 | -21 | 20 | 20 | -9 | 283.6s |
