# exp30b - VC+EI and heuristic baseline on MANGO IF maps

Games: ['cutthroat', 'detective', 'inhumane', 'zork1', 'zork2', 'murdac', 'advent', 'sherlock', 'wishbringer', 'deephome']

## Aggregate by (model, mode)

| Model | Mode | n | cf_before | cf_after | repaired | repair % | dir_delta |
|-------|------|--:|---------:|---------:|---------:|--------:|----------:|
| claude-haiku-4-5-20251001 | vc_ei | 10 | 148 | 175 | -27 | -18.2 | -7 |
| gemini-3.5-flash | vc_ei | 10 | 148 | 319 | -171 | -115.5 | -18 |
| gpt-5.5 | vc_ei | 10 | 148 | 139 | 9 | 6.1 | -16 |
| n/a | heuristic_remove | 10 | 148 | 27 | 121 | 81.8 | -18 |

## Per-game per-(model, mode)

| game | model | mode | cb | ca | repaired | actions | iters | dir_delta |
|------|-------|------|---:|---:|--------:|--------:|------:|----------:|
| advent | claude-haiku-4-5-20251001 | vc_ei | 3 | 2 | 1 | 4 | 8 | -1 |
| advent | gemini-3.5-flash | vc_ei | 3 | 59 | -56 | 3 | 14 | -1 |
| advent | gpt-5.5 | vc_ei | 3 | 0 | 3 | 3 | 3 | -1 |
| advent | n/a | heuristic_remove | 3 | 1 | 2 | 3 | 2 | -2 |
| cutthroat | claude-haiku-4-5-20251001 | vc_ei | 35 | 58 | -23 | 14 | 20 | +0 |
| cutthroat | gemini-3.5-flash | vc_ei | 35 | 62 | -27 | 15 | 20 | -3 |
| cutthroat | gpt-5.5 | vc_ei | 35 | 53 | -18 | 20 | 20 | -2 |
| cutthroat | n/a | heuristic_remove | 35 | 1 | 34 | 12 | 11 | -5 |
| deephome | claude-haiku-4-5-20251001 | vc_ei | 1 | 1 | 0 | 0 | 3 | +0 |
| deephome | gemini-3.5-flash | vc_ei | 1 | 0 | 1 | 2 | 4 | -1 |
| deephome | gpt-5.5 | vc_ei | 1 | 0 | 1 | 3 | 3 | -2 |
| deephome | n/a | heuristic_remove | 1 | 0 | 1 | 1 | 1 | -1 |
| detective | claude-haiku-4-5-20251001 | vc_ei | 40 | 13 | 27 | 12 | 20 | -2 |
| detective | gemini-3.5-flash | vc_ei | 40 | 57 | -17 | 10 | 17 | -3 |
| detective | gpt-5.5 | vc_ei | 40 | 3 | 37 | 13 | 20 | -1 |
| detective | n/a | heuristic_remove | 40 | 15 | 25 | 7 | 6 | -2 |
| inhumane | claude-haiku-4-5-20251001 | vc_ei | 9 | 35 | -26 | 0 | 6 | +0 |
| inhumane | gemini-3.5-flash | vc_ei | 9 | 35 | -26 | 6 | 13 | +0 |
| inhumane | gpt-5.5 | vc_ei | 9 | 22 | -13 | 3 | 9 | +0 |
| inhumane | n/a | heuristic_remove | 9 | 1 | 8 | 2 | 1 | +0 |
| murdac | claude-haiku-4-5-20251001 | vc_ei | 29 | 24 | 5 | 16 | 20 | -2 |
| murdac | gemini-3.5-flash | vc_ei | 29 | 32 | -3 | 8 | 20 | -2 |
| murdac | gpt-5.5 | vc_ei | 29 | 0 | 29 | 13 | 16 | -2 |
| murdac | n/a | heuristic_remove | 29 | 3 | 26 | 6 | 5 | -2 |
| sherlock | claude-haiku-4-5-20251001 | vc_ei | 3 | 2 | 1 | 6 | 11 | +0 |
| sherlock | gemini-3.5-flash | vc_ei | 3 | 0 | 3 | 1 | 1 | -1 |
| sherlock | gpt-5.5 | vc_ei | 3 | 0 | 3 | 1 | 1 | -1 |
| sherlock | n/a | heuristic_remove | 3 | 2 | 1 | 2 | 1 | -1 |
| wishbringer | claude-haiku-4-5-20251001 | vc_ei | 5 | 5 | 0 | 20 | 20 | +0 |
| wishbringer | gemini-3.5-flash | vc_ei | 5 | 25 | -20 | 7 | 14 | -1 |
| wishbringer | gpt-5.5 | vc_ei | 5 | 30 | -25 | 13 | 20 | -1 |
| wishbringer | n/a | heuristic_remove | 5 | 0 | 5 | 3 | 3 | -2 |
| zork1 | claude-haiku-4-5-20251001 | vc_ei | 17 | 31 | -14 | 1 | 9 | +0 |
| zork1 | gemini-3.5-flash | vc_ei | 17 | 20 | -3 | 16 | 20 | -1 |
| zork1 | gpt-5.5 | vc_ei | 17 | 0 | 17 | 6 | 6 | -1 |
| zork1 | n/a | heuristic_remove | 17 | 1 | 16 | 5 | 4 | -1 |
| zork2 | claude-haiku-4-5-20251001 | vc_ei | 6 | 4 | 2 | 18 | 20 | -2 |
| zork2 | gemini-3.5-flash | vc_ei | 6 | 29 | -23 | 11 | 20 | -5 |
| zork2 | gpt-5.5 | vc_ei | 6 | 31 | -25 | 19 | 20 | -5 |
| zork2 | n/a | heuristic_remove | 6 | 3 | 3 | 4 | 3 | -2 |
