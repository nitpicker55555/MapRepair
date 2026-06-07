# hhgg — per-edge error analysis

V3 walkthrough steps: 3
Predicted edges:      1
GT edges:             4

## Pred buckets
| bucket | n |
|--------|--:|
| SPURIOUS_PAIR | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 4 |

## Samples — SPURIOUS_PAIR

- step 2 walk_action='exit' PRED: 'front of house' --[exit]--> 'country lane'
  > obs: 'Country Lane\nYou see the huge bulldozer heaving itself among the cloud of brick dust which is all that remains of your home. As you start up the lane, a small dog runs up to you, yapping.\n\nFord hurrie'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='front porch' GT_dst=None
  - LLM reasoning: Observation explicitly states arrival at 'Front Porch', which matches the canonical location list.
  - obs: 'You make your way down to the front porch.\n\nFront Porch\nThis is the enclosed front porch of your home. Your front garden lies to the south, and you can re-enter your home to the north.\nOn the doormat is a pile of junk mail.'
- step 1 cur=None action='south'  pred_dst='front of house' GT_dst=None
  - LLM reasoning: The observation explicitly names 'Front of House', which matches the canonical location list.
  - obs: 'Front of House\nYou can enter your home to the north. A path leads around it to the northeast and northwest, and a country lane is visible to the south. All that lies between your home and the huge yellow bulldozer bearing down on it is a few yards of mud.\n\nMr. Prosser, from the local council, is sta'
