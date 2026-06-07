# yomomma — per-edge error analysis

V3 walkthrough steps: 16
Predicted edges:      14
GT edges:             19

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 10 |
| WRONG_DIRECTION | 3 |
| WRONG_DST | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 10 |
| MISSED | 8 |
| RECALLED_WRONG_DIR | 1 |

## Samples — WRONG_DIRECTION

- step 10 walk_action='north' PRED: 'dance floor (center of the club)' --[north]--> 'vip lounge (nw corner)'
  > obs: "VIP lounge (NW corner)\nThis area is reserved to those with VIP passes. There are cushy red sofas to sit on and you don't have to worry about drunken idiots crashing your table.\n\nA guard is making sure"
- step 12 walk_action='southeast' PRED: 'bar (western side)' --[southeast]--> 'dance floor (center of the club)'
  > obs: "Dance floor (center of the club)\nThis is the center of the Compass Club. Most people don't orient themselves in relation to compass directions, but here it's more than natural. A huge compass rose is "
- step 15 walk_action='northwest' PRED: 'artist hangout (ne corner)' --[northwest]--> 'dance floor (center of the club)'
  > obs: "Dance floor (center of the club)\nThis is the center of the Compass Club. Most people don't orient themselves in relation to compass directions, but here it's more than natural. A huge compass rose is "

## Samples — WRONG_DST

- step 13 walk_action='south' PRED: 'dance floor (center of the club)' --[south]--> 'hall of fame (eastern side)'  (GT here: entrance (southern side))
  > obs: "Hall of Fame (eastern side)\nJust being here makes your heart beat faster. The walls are adorned by portraits of past champions, each a part of great history.\n\nRalph, one of Gus's goons, is sitting her"

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='down'  pred_dst='dance floor (center of the club)' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Dance floor (center of the club)' as the location after the 'down' action; matches canonical list.
  - obs: "Dance floor (center of the club)\nThis is the center of the Compass Club. Most people don't orient themselves in relation to compass directions, but here it's more than natural. A huge compass rose is painted to the ceiling and the abbreviations of cardinal directions are painted on the walls.\n\nTo no"
- step 1 cur=None action='southeast'  pred_dst='dark corner (se corner)' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Dark corner (SE corner)', matching a canonical location.
  - obs: 'Dark corner (SE corner)\nThe sharp spotlights in this southeastern corner of the club create dark and gloomy shadows around the tables.\n\nA mysterious stranger sits behind a table, gesturing you to sit on the chair opposite to him.'
- step 13 cur='dance floor (center of the club)' action='south'  pred_dst='hall of fame (eastern side)' GT_dst='entrance (southern side)'
  - obs: "Hall of Fame (eastern side)\nJust being here makes your heart beat faster. The walls are adorned by portraits of past champions, each a part of great history.\n\nRalph, one of Gus's goons, is sitting here with a satchel.\n\nAn old jukebox sits under the pictures."
