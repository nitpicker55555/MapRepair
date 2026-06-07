# sherlock — per-edge error analysis

V3 walkthrough steps: 24
Predicted edges:      22
GT edges:             29

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 20 |
| WRONG_DST | 1 |
| SPURIOUS_PAIR | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 20 |
| MISSED | 9 |

## Samples — WRONG_DST

- step 15 walk_action='east' PRED: 'marylebone road' --[east]--> 'parliament square'  (GT here: tottenham court road)
  > obs: '>\nSaturday  6:00:00 a.m.\nIn the distance, you hear Big Ben strike the hour.\n[Press any key to continue.]'

## Samples — SPURIOUS_PAIR

- step 16 walk_action='east' PRED: 'parliament square' --[east]--> 'scotland yard'
  > obs: '>\nScotland Yard\n\nYou walk up the steps and into the building. Inside, it looks just like any other government building - drab and worn. The door to the street is to the west, and you see some steps le'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='up'  pred_dst='vestibule' GT_dst=None
  - LLM reasoning: The observation states 'Vestibule' at the top, indicating arrival in the vestibule after going up.
  - obs: ">\nVestibule\n\nYou climb the stairs and pause outside the door to Holmes's study, nervous about what you might find inside. \n\nHolmes's study is to the north.\n                                       VestibuleSaturday  5:02:00 a.m."
- step 1 cur=None action='north'  pred_dst="holmes's study" GT_dst=None
  - LLM reasoning: The action is 'north' and the observation begins with 'Holmes's Study', which matches a canonical location.
  - obs: ">\nHolmes's Study\n\nYou step back through time to your bachelor days, when you shared these cluttered digs with the world's only consulting detective. Despite the mess, you can see that the room is unchanged. The fireplace to the north has a pile of letters transfixed to its mantelpiece by a jackknife"
- step 15 cur='marylebone road' action='east'  pred_dst='parliament square' GT_dst='tottenham court road'
  - obs: '>\nSaturday  6:00:00 a.m.\nIn the distance, you hear Big Ben strike the hour.\n[Press any key to continue.]'
