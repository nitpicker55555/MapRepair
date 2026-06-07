# snacktime — per-edge error analysis

V3 walkthrough steps: 5
Predicted edges:      4
GT edges:             5

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 3 |
| WRONG_SRC_UNKNOWN | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 3 |
| MISSED | 2 |

## Samples — WRONG_SRC_UNKNOWN

- step 1 walk_action='east' PRED: '<unknown>' --[east]--> 'sitting room (on the long soft thing)'
  > obs: 'Sitting Room\nYour pet is here, all stretched out on the long soft thing and snoring.'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='west'  pred_dst='food room' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Food Room', matching the canonical 'food room'.
  - obs: "Food Room\nFood room, glorious food room!\n\nYour bowl for food and water is here, but that's only the beginning. There's also a lot of secret places where your pet keeps food and things that go with food and things that make food better. There's a thing that gets hot and a box that stays cold, and tha"
