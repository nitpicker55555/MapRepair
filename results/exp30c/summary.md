# exp30c - Full-coverage MANGO IF maps (30 games with non-zero residual conflicts)

Candidate games: ['advent', 'adventureland', 'afflicted', 'anchor', 'awaken', 'balances', 'ballyhoo', 'curses', 'cutthroat', 'deephome', 'detective', 'dragon', 'enter', 'gold', 'hollywood', 'infidel', 'inhumane', 'jewel', 'karn', 'library', 'loose', 'ludicorp', 'lurking', 'murdac', 'night', 'omniquest', 'pentari', 'planetfall', 'plundered', 'reverb', 'sherlock', 'sorcerer', 'spellbrkr', 'spirit', 'temple', 'wishbringer', 'yomomma', 'zenon', 'zork1', 'zork2', 'zork3', 'ztuu']

## Aggregate by (model, mode), merged with exp30/exp30b

| Model | Mode | n | cf_before | cf_after | repaired | repair % | dir_delta |
|-------|------|--:|---------:|---------:|---------:|--------:|----------:|
| claude-haiku-4-5-20251001 | baseline | 42 | 534 | 874 | -340 | -63.7 | -32 |
| claude-haiku-4-5-20251001 | edge_impact | 42 | 534 | 827 | -293 | -54.9 | -67 |
| claude-haiku-4-5-20251001 | vc_ei | 42 | 534 | 625 | -91 | -17.0 | -31 |
| gemini-3.5-flash | baseline | 42 | 534 | 841 | -307 | -57.5 | -32 |
| gemini-3.5-flash | edge_impact | 42 | 534 | 572 | -38 | -7.1 | -69 |
| gemini-3.5-flash | vc_ei | 42 | 534 | 837 | -303 | -56.7 | -51 |
| gpt-5.5 | baseline | 42 | 534 | 609 | -75 | -14.0 | -33 |
| gpt-5.5 | edge_impact | 42 | 534 | 396 | 138 | 25.8 | -86 |
| gpt-5.5 | vc_ei | 42 | 534 | 458 | 76 | 14.2 | -89 |
| n/a | heuristic_modify | 42 | 534 | 438 | 96 | 18.0 | -97 |
| n/a | heuristic_remove | 42 | 534 | 98 | 436 | 81.6 | -56 |
