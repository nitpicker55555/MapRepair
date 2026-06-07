# library — per-edge error analysis

V3 walkthrough steps: 12
Predicted edges:      10
GT edges:             12

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 9 |
| SPURIOUS_PAIR | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 9 |
| MISSED | 3 |

## Samples — SPURIOUS_PAIR

- step 2 walk_action='north' PRED: 'lobby' --[north]--> 'ground floor stairwell'
  > obs: 'Ground Floor Stairwell\nGrungy stairs lead upwards, to the second level. A damaged picture hangs on the wall. The door to the ground floor shelving area is to the south.'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='west'  pred_dst='ground floor stacks' GT_dst=None
  - LLM reasoning: The observation explicitly names 'Ground Floor Stacks' as the room entered after moving west.
  - obs: 'You push the glass doors open. Their hydraulics hiss faintly.\n\nGround Floor Stacks\nThis cavernous room is lit with flickering flourescent lighting which makes the ceiling look even lower than it really is. It is filled with half-empty shelves, upon which rest the dusty boxes of many early games. To '
- step 1 cur=None action='east'  pred_dst='lobby' GT_dst=None
  - LLM reasoning: The observation begins with 'Lobby', which matches a canonical location, so the player has moved east into the Lobby.
  - obs: 'Lobby\nThis is the entrance to the library. The circulation desk dominates the room, seemingly cobbled together over the course of several generations from an assortment of desks and tables. A pair of security gates stands before the front doors to prevent people from stealing books. A card catalog l'
