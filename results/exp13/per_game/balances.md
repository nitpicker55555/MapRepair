# balances — per-edge error analysis

V3 walkthrough steps: 11
Predicted edges:      9
GT edges:             13

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 7 |
| SPURIOUS_PAIR | 1 |
| HALLUCINATED_DST | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 7 |
| MISSED | 6 |

## Samples — SPURIOUS_PAIR

- step 2 walk_action='north' PRED: 'pocket valley' --[north]--> 'crest of hill'
  > obs: 'Crest of Hill\nThe road crosses the top of a ridge here, sloping downwards to the south and the northwest. A track diverges to east.\n\nA tortoise ambles along the road, extremely slowly.'

## Samples — HALLUCINATED_DST

- step 10 walk_action='east' PRED: 'edge of chasm' --[east]--> 'outside temple'
  > obs: 'Track, outside Temple\nThis is the end of a long track winding through desolate hills, which runs back west up to the ridge.\n\nYou stand outside an enormous temple in the shape of a perfect, featureless'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='out'  pred_dst='grasslands, near hut' GT_dst=None
  - LLM reasoning: The observation explicitly names 'Grasslands, near Hut', which matches the canonical location list; action 'out' indicates movement.
  - obs: 'Grasslands, near Hut\nThe grasslands sway over low hills in all directions: it is a peaceful wilderness, broken only by this hut and a faint path to the north.'
- step 1 cur=None action='north'  pred_dst='pocket valley' GT_dst=None
  - LLM reasoning: The action is 'north', a canonical compass direction, and the observation explicitly names the destination as 'Pocket Valley', matching the canonical list.
  - obs: 'Pocket Valley\nA pleasant pocket valley in the grassy hills, through which a trail runs north-to-south.\nThere is a chestnut horse here, munching on a pile of oats.'
