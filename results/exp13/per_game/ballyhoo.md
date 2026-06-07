# ballyhoo — per-edge error analysis

V3 walkthrough steps: 26
Predicted edges:      23
GT edges:             33

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 19 |
| WRONG_DST | 2 |
| SPURIOUS_PAIR | 2 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 19 |
| MISSED | 14 |

## Samples — WRONG_DST

- step 16 walk_action='south' PRED: 'in the wings' --[south]--> 'camp, east'  (GT here: connection)
  > obs: "You pass through the creaky turnstile.\n\nCamp, East\nYou're in the east end of the performers' camp. It is unevenly lighted by a number of glaring spotlights which ring the camp, creating a stage-like p"
- step 19 walk_action='north' PRED: 'clown alley' --[north]--> 'beside the big top'  (GT here: camp, west)
  > obs: 'Beside the Big Top\n\nAbout waist-high, a barred passage for wild animals extends several yards out from the big top.'

## Samples — SPURIOUS_PAIR

- step 21 walk_action='northeast' PRED: 'connection' --[northeast]--> 'under the bleachers'
  > obs: "Under the Bleachers\nYou're standing between the underside of the grandstand and the side wall of the big top. Not unexpectedly, the ground here is strewn with garbage. The only exit is southwest."
- step 23 walk_action='east' PRED: 'in the wings' --[east]--> 'midway entrance'
  > obs: 'You pass through the creaky turnstile.\n\nMidway Entrance\nTo the west is the turnstile exit from the midway, which continues east. The side of the big top borders on the north. A soiled, sagging banner '

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='connection' GT_dst=None
  - LLM reasoning: Observation names the room as 'Connection', matching the canonical list; action is 'south', indicating movement.
  - obs: 'You emerge into the warm night air of summer.\n\nConnection\nThis area of matted-down crabgrass lies between the vaulted big top entrance to the north and the enticements of the midway to the east, where a sagging banner hangs crookedly above a turnstile. There is a drinking fountain near the side wall'
- step 1 cur=None action='west'  pred_dst='beside the big top' GT_dst=None
  - LLM reasoning: Observation heading matches canonical location 'beside the big top'.
  - obs: "Beside the Big Top\nYou're standing next to the big top, which is gently flapping in the warm breeze.\n\nA cheap plastic clown mask, having been carelessly dropped here, is lying face-up on the ground.\n\nAbout waist-high, a barred passage for wild animals extends several yards out from the big top."
- step 2 cur=None action='south'  pred_dst='back yard' GT_dst=None
  - LLM reasoning: The observation begins with 'Back Yard', which matches the canonical location 'back yard', indicating arrival in that room after moving south.
  - obs: "Back Yard\nYou're standing in front of the turnstile entrance to a fenced-in area, which is south. To the west stands a droopy tent, and the field continues north and east.\n\nA tall and lanky clown in baggy pants trudges out of the tent and toward the turnstile. Comrade Thumb follows behind.\n\nAdjacent"
- step 16 cur='in the wings' action='south'  pred_dst='camp, east' GT_dst='connection'
  - obs: "You pass through the creaky turnstile.\n\nCamp, East\nYou're in the east end of the performers' camp. It is unevenly lighted by a number of glaring spotlights which ring the camp, creating a stage-like patchwork of light and dark.\n\nAt the eastern end of the camp sits one lone trailer."
- step 19 cur='clown alley' action='north'  pred_dst='beside the big top' GT_dst='camp, west'
  - obs: 'Beside the Big Top\n\nAbout waist-high, a barred passage for wild animals extends several yards out from the big top.'
