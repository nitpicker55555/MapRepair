# partyfoul — per-edge error analysis

V3 walkthrough steps: 7
Predicted edges:      6
GT edges:             7

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 6 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 6 |
| MISSED | 1 |

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='west'  pred_dst='guest room' GT_dst=None
  - LLM reasoning: The observation explicitly states arrival in the guest room, which matches the canonical list.
  - obs: 'You step into the guest room to find Frank wobbling around the room. Ooh, perhaps you shouldn\'t have given him that last glass of wine. "Frank," you ask, "Are you doing okay? Do you want me to get you some water?"\n\n"Nah, I\'m fine," he says, clearly lying. He slumps over to the bed and begins to craw'
