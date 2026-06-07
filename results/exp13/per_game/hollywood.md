# hollywood — per-edge error analysis

V3 walkthrough steps: 18
Predicted edges:      17
GT edges:             22

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 17 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 17 |
| MISSED | 5 |

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='north'  pred_dst='front porch' GT_dst=None
  - LLM reasoning: The observation explicitly states arrival at 'Front Porch', which matches the canonical location list.
  - obs: "As you walk toward the house, a large black cat scurries across the path heading toward Johnny Carson's house.\n\nFront Porch\nThis is the front porch of the house with its towering white pillars. Next to the front door is a regulation mailbox endorsed by Buck Palace. Beneath the mailbox is a doorbell "
