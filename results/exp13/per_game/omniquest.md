# omniquest — per-edge error analysis

V3 walkthrough steps: 40
Predicted edges:      38
GT edges:             57

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 36 |
| SPURIOUS_PAIR | 2 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 36 |
| MISSED | 21 |

## Samples — SPURIOUS_PAIR

- step 17 walk_action='southeast' PRED: 'island coast' --[southeast]--> 'giant cavern'
  > obs: 'Giant Cavern\nThis is a large room which has corridors leaving the room in four different directions:  northwest, northeast, west, and south.'
- step 24 walk_action='southeast' PRED: 'twisting passageway' --[southeast]--> 'east path'
  > obs: 'East Path\nThis is a path that came from the northwest and continues to the southwest.\n\nThere is an old man here.\n"I haven\'t had a smoke in 12 years," he says.  "Please, PLEASE give me a match!"\n\nYou c'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='east'  pred_dst='small path' GT_dst=None
  - LLM reasoning: The observation describes 'Small Path', which matches the canonical location 'small path'. The action is 'east', indicating movement to this room.
  - obs: 'Small Path\nThe path branches here.  There are paths to the west and south that continue into the forest.  The path to the east appears to lead into some rather rocky territory.'
- step 1 cur=None action='south'  pred_dst='fork in path' GT_dst=None
  - LLM reasoning: Observation begins with 'Fork in Path', which matches the canonical room name 'fork in path'.
  - obs: 'Fork in Path\nThis path forks to the southwest and southeast.'
