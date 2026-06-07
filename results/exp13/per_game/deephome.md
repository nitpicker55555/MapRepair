# deephome — per-edge error analysis

V3 walkthrough steps: 42
Predicted edges:      40
GT edges:             46

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 38 |
| SPURIOUS_PAIR | 2 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 38 |
| MISSED | 8 |

## Samples — SPURIOUS_PAIR

- step 19 walk_action='out' PRED: 'railway station (main hall)' --[out]--> 'railway station (smithy court)'
  > obs: 'You get out of the rail car.\n\nRailway Station\nHere hangs, from thick cables, the major means of transportation over the great chasms that separate the locals of the city. You can hear the hum of the m'
- step 34 walk_action='west' PRED: 'railway station (smithy court)' --[west]--> "soldiers' barracks"
  > obs: "Soldiers' Barracks\nScores of plain mounds stand before you, all with closed and locked doors.  Inside each one will be accommodations for twenty dwarves.  However, as is customary, they can not be ent"

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='southwest'  pred_dst='northern guard post' GT_dst=None
  - LLM reasoning: The observation begins with 'Northern Guard Post', matching a canonical location, indicating arrival after moving southwest.
  - obs: 'Northern Guard Post\nThis guard post is small and inconspicuous, but positioned so that no one unwanted takes advantage of the secret entrance to the northeast, which is disguised as a common fireplace.  On the northwest wall is a cabinet.  To the southwest is a small door that leads out to the Main '
- step 1 cur=None action='southwest'  pred_dst='main hall' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Main Hall', which matches the canonical location list. The action 'southwest' is a compass direction, so this is a movement to 'main hall'.
  - obs: 'You are met by an imposing and ominous dark phantom that fills most of the room.\n\nMain Hall\nThe main hall is quite large, and is lit by magical torches that line the walls all around, in a pattern that spirals up the grandiose room.  Elevators hang in mid air, no longer powered.  To the northeast is'
