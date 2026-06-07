# Clean-walkthrough rebuild — global review

Games: 53
Total kept steps: 5949
Total kept steps that match a GT-edge step index: 1210
Total unmapped kept steps: 4739
Macro match rate: 20.3%
Synthetic-text leaks in kept observations: 0

(An unmapped step has a canonical direction action but no GT edge cites it via `seen_in_forward` or `seen_in_reversed`. These steps were re-traversals or were removed during edge refinement; the new LLM should still infer the edge.)

## Per-game summary
| game | src | kept | skipped | synth_removed | gt_edges | matched | unmapped | leaks |
|------|----:|-----:|--------:|--------------:|---------:|--------:|---------:|------:|
| 905 | 23 | 4 | 19 | 0 | 6 | 4 | 0 | 0 |
| advent | 278 | 165 | 113 | 0 | 51 | 41 | 124 | 0 |
| adventureland | 171 | 89 | 82 | 0 | 28 | 23 | 66 | 0 |
| afflicted | 99 | 49 | 50 | 0 | 20 | 19 | 30 | 0 |
| anchor | 532 | 251 | 281 | 0 | 42 | 30 | 221 | 0 |
| awaken | 58 | 27 | 31 | 0 | 24 | 22 | 5 | 0 |
| balances | 123 | 17 | 106 | 0 | 13 | 11 | 6 | 0 |
| ballyhoo | 417 | 223 | 194 | 0 | 33 | 26 | 197 | 0 |
| curses | 817 | 359 | 458 | 0 | 26 | 25 | 334 | 0 |
| cutthroat | 337 | 181 | 156 | 0 | 49 | 41 | 140 | 0 |
| deephome | 328 | 196 | 132 | 0 | 46 | 42 | 154 | 0 |
| detective | 52 | 42 | 10 | 0 | 39 | 36 | 6 | 0 |
| dragon | 102 | 52 | 50 | 0 | 42 | 32 | 20 | 0 |
| enchanter | 266 | 132 | 134 | 0 | 39 | 28 | 104 | 0 |
| enter | 103 | 42 | 61 | 0 | 26 | 23 | 19 | 0 |
| gold | 346 | 128 | 218 | 0 | 22 | 22 | 106 | 0 |
| hhgg | 362 | 45 | 317 | 0 | 4 | 3 | 42 | 0 |
| hollywood | 398 | 231 | 167 | 0 | 22 | 18 | 213 | 0 |
| huntdark | 68 | 5 | 63 | 0 | 6 | 4 | 1 | 0 |
| infidel | 251 | 106 | 145 | 0 | 48 | 29 | 77 | 0 |
| inhumane | 123 | 71 | 52 | 0 | 50 | 30 | 41 | 0 |
| jewel | 224 | 66 | 158 | 0 | 31 | 28 | 38 | 0 |
| karn | 363 | 130 | 233 | 0 | 33 | 26 | 104 | 0 |
| library | 53 | 15 | 38 | 0 | 12 | 12 | 3 | 0 |
| loose | 51 | 26 | 25 | 0 | 21 | 18 | 8 | 0 |
| lostpig | 147 | 26 | 121 | 0 | 11 | 7 | 19 | 0 |
| ludicorp | 365 | 225 | 140 | 0 | 43 | 37 | 188 | 0 |
| lurking | 295 | 138 | 157 | 0 | 26 | 18 | 120 | 0 |
| moonlit | 60 | 12 | 48 | 0 | 8 | 6 | 6 | 0 |
| murdac | 305 | 213 | 92 | 0 | 48 | 39 | 174 | 0 |
| night | 91 | 51 | 40 | 0 | 32 | 31 | 20 | 0 |
| omniquest | 79 | 47 | 32 | 0 | 57 | 40 | 7 | 0 |
| partyfoul | 57 | 15 | 42 | 0 | 7 | 7 | 8 | 0 |
| pentari | 50 | 28 | 22 | 0 | 26 | 22 | 6 | 0 |
| planetfall | 400 | 189 | 211 | 0 | 40 | 29 | 160 | 0 |
| plundered | 190 | 86 | 104 | 0 | 30 | 24 | 62 | 0 |
| reverb | 75 | 35 | 40 | 0 | 25 | 23 | 12 | 0 |
| seastalker | 205 | 90 | 115 | 0 | 12 | 10 | 80 | 0 |
| sherlock | 340 | 114 | 226 | 0 | 29 | 24 | 90 | 0 |
| snacktime | 35 | 8 | 27 | 0 | 5 | 5 | 3 | 0 |
| sorcerer | 255 | 132 | 123 | 0 | 43 | 29 | 103 | 0 |
| spellbrkr | 413 | 80 | 333 | 0 | 24 | 16 | 64 | 0 |
| spirit | 1265 | 582 | 683 | 0 | 40 | 37 | 545 | 0 |
| temple | 182 | 70 | 112 | 0 | 32 | 22 | 48 | 0 |
| trinity | 611 | 247 | 364 | 0 | 20 | 11 | 236 | 0 |
| tryst205 | 519 | 209 | 310 | 0 | 9 | 9 | 200 | 0 |
| wishbringer | 185 | 82 | 103 | 0 | 37 | 28 | 54 | 0 |
| yomomma | 99 | 30 | 69 | 0 | 19 | 16 | 14 | 0 |
| zenon | 84 | 26 | 58 | 0 | 26 | 17 | 9 | 0 |
| zork1 | 397 | 236 | 161 | 0 | 29 | 22 | 214 | 0 |
| zork2 | 297 | 158 | 139 | 0 | 38 | 35 | 123 | 0 |
| zork3 | 274 | 134 | 140 | 0 | 38 | 27 | 107 | 0 |
| ztuu | 85 | 34 | 51 | 0 | 26 | 26 | 8 | 0 |
