# zork3 — per-edge error analysis

V3 walkthrough steps: 27
Predicted edges:      24
GT edges:             38

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 19 |
| SPURIOUS_PAIR | 3 |
| SELF_LOOP | 1 |
| WRONG_DST | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 19 |
| MISSED | 19 |

## Samples — WRONG_DST

- step 18 walk_action='south' PRED: 'western shore' --[south]--> 'southern shore'  (GT here: scenic vista)
  > obs: 'Southern Shore\nYou are on the south shore of the lake. Rock formations prevent movement to the west and thickening swamp to the east makes the going all but impossible. To the south, where the beach m'

## Samples — SPURIOUS_PAIR

- step 14 walk_action='down' PRED: 'lake shore' --[down]--> 'underwater'
  > obs: 'Underwater\nYou are below the surface of the lake. It turns out that the lake is quite shallow and the bottom is only a few feet below you. Considering the frigid temperature of the water, you should p'
- step 15 walk_action='west' PRED: 'underwater' --[west]--> 'western shore'
  > obs: "Western Shore\nYou are on the western shore of the lake. The ground here is quite hard, but a few sickly reeds manage to grow near the water's edge. The only path leads into the rock to the south."
- step 26 walk_action='down' PRED: 'damp passage' --[down]--> 'cliff ledge'
  > obs: 'Cliff Ledge\nThis is a rock-strewn ledge near the base of a tall cliff. The bottom of the cliff is another fifteen feet below. You have little hope of climbing up the cliff face, but you might be able '

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='pitch black 1' GT_dst=None
  - LLM reasoning: Compass movement south leads to a pitch black room; canonical list includes 'pitch black 1', so this is the destination.
  - obs: 'It is pitch black. You are likely to be eaten by a grue.'
- step 1 cur=None action='west'  pred_dst='barren area' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Barren Area', which matches the canonical location list.
  - obs: 'Barren Area\nYou are west of the junction, where the rock-bound passage widens out into a large, flat area. Although the land here is barren, you can see vegetation to the west. South of here is a mighty wall of stone, ancient and crumbling.  To the southwest the wall has decayed enough to form an op'
- step 2 cur=None action='west'  pred_dst='cliff' GT_dst=None
  - LLM reasoning: Observation begins with 'Cliff', which matches the canonical room 'cliff'.
  - obs: "Cliff\nThis is a remarkable spot in the dungeon. Perhaps two hundred feet above you is a gaping hole in the earth's surface through which pours bright sunshine! A few seedlings from the world above, nurtured by the sunlight and occasional rains, have grown into giant trees, making this a virtual oasi"
- step 18 cur='western shore' action='south'  pred_dst='southern shore' GT_dst='scenic vista'
  - obs: 'Southern Shore\nYou are on the south shore of the lake. Rock formations prevent movement to the west and thickening swamp to the east makes the going all but impossible. To the south, where the beach meets a rock formation, you can make out a dark passage sloping steeply upward into the rock.'
