# awaken — per-edge error analysis

V3 walkthrough steps: 22
Predicted edges:      20
GT edges:             24

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 16 |
| SPURIOUS_PAIR | 2 |
| WRONG_DST | 2 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 16 |
| MISSED | 8 |

## Samples — WRONG_DST

- step 18 walk_action='north' PRED: 'front of the church' --[north]--> 'belfry'  (GT here: church porch)
  > obs: 'Belfry\n\nA massive bell hangs here, from a damaged mount.\nA heavy rope is attached to the bell and hangs through the beams to the room below.'
- step 21 walk_action='north' PRED: 'front of the church' --[north]--> 'inner chamber'  (GT here: church porch)
  > obs: 'Inner Chamber\nEven with the door open and the wind from the storm blowing in this room is rank with a fetid odor. The floor is piled high with trash and half eaten scraps of food. There are no windows'

## Samples — SPURIOUS_PAIR

- step 6 walk_action='north' PRED: 'east side of the church' --[north]--> 'church porch'
  > obs: 'Church Porch\nThis wide, covered porch spans the width of the church front. A white picket railing surrounds the porch on three sides, except to the south where a short flight of steps leads down to th'
- step 14 walk_action='down' PRED: 'vestibule' --[down]--> 'steeple chamber'
  > obs: 'Steeple Chamber\n\nA frayed but heavy rope hangs from somewhere overhead. There is a tree branch hanging from the end of the rope.\n\nThe heavy wooden door to the north is closed.\n\nThe top of the broken s'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='up'  pred_dst='graveyard' GT_dst=None
  - LLM reasoning: The observation explicitly states arrival in the 'Graveyard', which matches the canonical location list.
  - obs: 'As you scramble out of the hole the mud collapses downward, filling in the hole behind you.\n\nGraveyard\nDozens of stone markers stand here, oblivious to the wind and the rain. Huge, twisted oaks, their limbs tossed by the wind, draw an unmentionable sustenance from the moss-covered, moldy ground. An '
- step 1 cur=None action='east'  pred_dst='west side of the church' GT_dst=None
  - LLM reasoning: Observation explicitly names 'West Side of the Church', which matches the canonical location list.
  - obs: 'West Side of the Church\nThis side of the church is untended and grasses and weeds have run rampant. A muddy trail leads back to the graveyard to the west and on to the southeast and a faint path continues around the church to the northeast.\n\nTo the east, the wall of the church breaks some of the for'
- step 18 cur='front of the church' action='north'  pred_dst='belfry' GT_dst='church porch'
  - obs: 'Belfry\n\nA massive bell hangs here, from a damaged mount.\nA heavy rope is attached to the bell and hangs through the beams to the room below.'
- step 21 cur='front of the church' action='north'  pred_dst='inner chamber' GT_dst='church porch'
  - obs: 'Inner Chamber\nEven with the door open and the wind from the storm blowing in this room is rank with a fetid odor. The floor is piled high with trash and half eaten scraps of food. There are no windows, only the broken door frame to the south. An old sofa is pushed against one wall and a heavy table '
