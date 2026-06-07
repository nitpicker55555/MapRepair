# trinity — per-edge error analysis

V3 walkthrough steps: 11
Predicted edges:      9
GT edges:             20

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 7 |
| SPURIOUS_PAIR | 2 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 13 |
| RECALLED | 7 |

## Samples — SPURIOUS_PAIR

- step 5 walk_action='east' PRED: 'black lion gate' --[east]--> 'wading'
  > obs: "You wade into the cool, dark water.\n\nWading\n\nYou're standing knee-deep in the Long Water, not far from the western shore. Looking east, you can see a white door hovering just above the surface.\n\nSwans"
- step 6 walk_action='north' PRED: 'wading' --[north]--> 'summit'
  > obs: "You ascend the gentle hill.\n\nSummit\n\nThe hill you've climbed lies at the southwest edge of a vast wilderness. Towering forests are broken by long tracts of wasteland, rugged plateaus and marshes shrou"

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='east'  pred_dst='flower walk' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Flower Walk' as the destination room after moving east.
  - obs: 'The east wind makes it difficult to walk straight.\n\nFlower Walk\n\nGaily colored flower beds line the walks bending north and west, filling the air with a gentle fragrance. A little path leads northwest, between the trees.\n\nYou can see a soccer ball half-hidden among the blossoms.\n\nThe spires of the A'
- step 1 cur=None action='northwest'  pred_dst='the wabe' GT_dst=None
  - LLM reasoning: Observation explicitly names 'The Wabe', which matches a canonical location, after a canonical compass action.
  - obs: 'The Wabe\n\nThis grassy clearing is only twenty feet across, and perfectly circular. Paths wander off in many directions through the surrounding thicket.\n\nA shaft of golden sunlight falls across a handsome antique sundial, erected at the exact center of the clearing.\n           The Wabe'
