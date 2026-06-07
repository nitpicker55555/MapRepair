# karn — per-edge error analysis

V3 walkthrough steps: 26
Predicted edges:      24
GT edges:             33

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 16 |
| SPURIOUS_PAIR | 3 |
| WRONG_DIRECTION | 3 |
| WRONG_DST | 2 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 16 |
| MISSED | 15 |
| RECALLED_WRONG_DIR | 2 |

## Samples — WRONG_DIRECTION

- step 14 walk_action='southwest' PRED: 'mountain trail (carved steps to the south)' --[southwest]--> 'mountain trail (cleft to northwest, down southwest, down east)'
  > obs: 'Mountain Trail\n\nTo the south some steps are carved into the mountain.  They lead upwards out of sight.'
- step 15 walk_action='west' PRED: 'mountain trail (cleft to northwest, down southwest, down east)' --[west]--> 'mountain trail (up west, down east)'
  > obs: 'Mountain Trail'
- step 16 walk_action='southwest' PRED: 'mountain trail (up west, down east)' --[southwest]--> 'mountain trail (cleft to northwest, down southwest, down east)'
  > obs: 'Mountain Trail'

## Samples — WRONG_DST

- step 13 walk_action='southeast' PRED: 'narrow cleft' --[southeast]--> 'mountain trail (carved steps to the south)'  (GT here: mountain trail (cleft to northwest, down southwest, down east))
  > obs: 'Mountain Trail'
- step 25 walk_action='west' PRED: 'mountain trail (up west, down east)' --[west]--> 'mountain trail (carved steps to the south)'  (GT here: mountain trail (cleft to northwest, down southwest, down east))
  > obs: 'Mountain Trail'

## Samples — SPURIOUS_PAIR

- step 4 walk_action='northwest' PRED: 'console room' --[northwest]--> 'mountain trail (up northeast, clearing to southeast, stoney trail leads west)'
  > obs: 'Mountain Trail\nYou are on a grim trail that winds itself upwards along the mountain to the northeast.  A clearing lies to the southeast, and a stoney trail leads upwards to the west.'
- step 17 walk_action='southeast' PRED: 'mountain trail (cleft to northwest, down southwest, down east)' --[southeast]--> 'rocky clearing'
  > obs: 'Rocky Clearing\n\nThe TARDIS sits reassuringly nearby.\n\nAgainst the western rockface stands a massive boulder.'
- step 22 walk_action='down' PRED: 'ledge' --[down]--> 'plain'
  > obs: "You slide down the smooth, polished sides of the saucer...\n\n\nPlain\nYou're in a large area fairly clear of rocks and debris.  Unclimbable walls of rock surround this area, but a path does lead between "

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='east'  pred_dst='corridor' GT_dst=None
  - LLM reasoning: The observation names the destination as 'Corridor', which matches the canonical location 'corridor'.
  - obs: "Corridor\nThis corridor, like almost all the corridors of the TARDIS, seems to stretch on forever.  The console room lies to the west, and the corridor continues eastward into the distance.  There's a closed door on the south wall."
- step 1 cur=None action='south'  pred_dst='workshop' GT_dst=None
  - LLM reasoning: The observation begins with 'Workshop', which matches the canonical location 'workshop'.
  - obs: "Workshop\nThis is the room where you do much of your tinkering with electronic gadgetry.  Many a Rutan invasion or Dalek plot has been foiled by an ingenius invention you've thrown together at last minute in this room.  In fitting with your work habits, it's usually a rather untidy place.  However, i"
- step 13 cur='narrow cleft' action='southeast'  pred_dst='mountain trail (carved steps to the south)' GT_dst='mountain trail (cleft to northwest, down southwest, down east)'
  - obs: 'Mountain Trail'
- step 25 cur='mountain trail (up west, down east)' action='west'  pred_dst='mountain trail (carved steps to the south)' GT_dst='mountain trail (cleft to northwest, down southwest, down east)'
  - obs: 'Mountain Trail'
