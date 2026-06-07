# ludicorp — per-edge error analysis

V3 walkthrough steps: 37
Predicted edges:      35
GT edges:             43

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 31 |
| SELF_LOOP | 1 |
| WRONG_DST | 1 |
| HALLUCINATED_DST | 1 |
| HALLUCINATED_SRC | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 31 |
| MISSED | 12 |

## Samples — WRONG_DST

- step 31 walk_action='west' PRED: 'front of the office' --[west]--> 'the delivery bay gateway'  (GT here: west of office)
  > obs: 'The Delivery Bay Gateway\nA large gate blocks the delivery bay to the west. There is some sort of control panel mounted by the side of the gate. The carpark lies to the east.'

## Samples — HALLUCINATED_DST

- step 34 walk_action='north' PRED: 'storeroom' --[north]--> 'west end'
  > obs: 'Long Corridor, West End\nYou are at the west end of a very long east-west corridor. Doors lead north and south.'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='the main carpark' GT_dst=None
  - LLM reasoning: Observation explicitly names 'The Main Carpark', matching the canonical list; action is a compass direction.
  - obs: 'The Main Carpark\nThe carpark is large and mostly empty - perhaps the size of a football field. Surely there should be more cars here. You parked over in the south east corner. The office building lies to the north. Signs point at a delivery bay to the west.'
- step 1 cur=None action='southeast'  pred_dst='corner of the carpark' GT_dst=None
  - LLM reasoning: Observation states arrival at 'Corner Of the Carpark', which matches the canonical location 'corner of the carpark'.
  - obs: "Corner Of the Carpark\nYou're standing in the south east corner of the carpark.\n\nYou can see your Car (which is closed) here."
- step 31 cur='front of the office' action='west'  pred_dst='the delivery bay gateway' GT_dst='west of office'
  - obs: 'The Delivery Bay Gateway\nA large gate blocks the delivery bay to the west. There is some sort of control panel mounted by the side of the gate. The carpark lies to the east.'
- step 34 cur='storeroom' action='north'  pred_dst='west end' GT_dst='long corridor, west end'
  - obs: 'Long Corridor, West End\nYou are at the west end of a very long east-west corridor. Doors lead north and south.'
