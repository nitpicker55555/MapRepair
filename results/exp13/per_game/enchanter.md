# enchanter — per-edge error analysis

V3 walkthrough steps: 28
Predicted edges:      23
GT edges:             39

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 23 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 23 |
| MISSED | 16 |

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='northeast'  pred_dst='outside of shack' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Outside of Shack', which matches a canonical location; action is a compass direction.
  - obs: 'Outside of Shack\nYou are south of an old wooden shack, apparently deserted. The trail extends from northeast to southwest here, circling the Lonely Mountain, which lies to the southeast.\nThe sun is rising over the lands to the east.'
- step 1 cur=None action='northeast'  pred_dst='trail' GT_dst=None
  - LLM reasoning: The observation describes a trail and matches the canonical location 'trail'; the action 'northeast' is a compass direction, so movement occurred.
  - obs: 'Trail\nThe trail turns a corner here, bending out of sight to the southeast and southwest. The ground is higher here than to the south, allowing you to make out the top of the Lonely Mountain, shrouded in clouds. No path from here leads up the mountain, however.'
- step 2 cur=None action='southeast'  pred_dst='trail head' GT_dst=None
  - LLM reasoning: Observation begins with 'Trail Head', which matches the canonical location 'trail head'.
  - obs: 'Trail Head\nYou are on the northeast side of the Lonely Mountain, walking a narrow path through rolling hills. On one side of the path is a sign and behind that, to the northeast, a winding path through the thick undergrowth. The trail continues to the southeast.'
- step 3 cur=None action='northeast'  pred_dst='shady brook' GT_dst=None
  - LLM reasoning: Observation begins with 'Shady Brook', which matches the canonical location 'shady brook'.
  - obs: 'Shady Brook\nThe trail ends here. A ruined stone foundation marks the site of the Old Lingolf House, which presumably was a monument of some sort. The trail head is off to the southwest, and a small brook flows slowly by.\nThe sun has now risen above the hills.'
- step 4 cur=None action='southwest'  pred_dst='trail head' GT_dst=None
  - LLM reasoning: Observation matches canonical room name 'trail head', so player moved there.
  - obs: 'Trail Head'
