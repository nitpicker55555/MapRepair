# advent — per-edge error analysis

V3 walkthrough steps: 41
Predicted edges:      38
GT edges:             51

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 35 |
| SPURIOUS_PAIR | 3 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 35 |
| MISSED | 16 |

## Samples — SPURIOUS_PAIR

- step 31 walk_action='northwest' PRED: 'in swiss cheese room' --[northwest]--> 'oriental room'
  > obs: 'Oriental Room\nThis is the oriental room. Ancient oriental cave drawings cover the walls. A gently sloping passage leads upward to the north, another passage leads se, and a hands and knees crawl leads'
- step 33 walk_action='east' PRED: 'misty cavern' --[east]--> 'plover room'
  > obs: "Plover Room\nYou're in a small chamber lit by an eerie green light. An extremely narrow tunnel exits to the west. A dark corridor leads northeast.\n\nThere is an emerald here the size of a plover's egg!"
- step 35 walk_action='south' PRED: 'alcove' --[south]--> 'oriental room'
  > obs: 'Oriental Room\n\nYou can see a ming vase here.'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='east'  pred_dst='inside building' GT_dst=None
  - LLM reasoning: Observation describes 'Inside Building', which matches the canonical location 'inside building'.
  - obs: 'Inside Building\nYou are inside a building, a well house for a large spring.\n\nThere are some keys on the ground here.\n\nThere is tasty food here.\n\nThere is a shiny brass lamp nearby.\n\nThere is an empty bottle here.'
- step 1 cur=None action='west'  pred_dst='end of road' GT_dst=None
  - LLM reasoning: Observation matches canonical room 'end of road' after moving west from unknown location.
  - obs: 'At End Of Road'
- step 2 cur=None action='south'  pred_dst='in a valley' GT_dst=None
  - LLM reasoning: Observation matches canonical room 'in a valley', so player moved there via 'south'.
  - obs: 'In A Valley\nYou are in a valley in the forest beside a stream tumbling along a rocky bed.'
