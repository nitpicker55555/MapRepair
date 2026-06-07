# moonlit — per-edge error analysis

V3 walkthrough steps: 6
Predicted edges:      4
GT edges:             8

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 3 |
| SPURIOUS_PAIR | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 5 |
| RECALLED | 3 |

## Samples — SPURIOUS_PAIR

- step 4 walk_action='west' PRED: 'moonlit chamber' --[west]--> 'skeletal banquet'
  > obs: 'Skeletal Banquet\nIt seems this is a banquet room, if a banquet had no especial place for people to dine. Glassware and stoneware are arranged in deliberate shapes, as though they are constellations. \n'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='down'  pred_dst='shadow armory' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Shadow Armory' as the destination room, matching the canonical list.
  - obs: 'Shadow Armory\nShadows gather around you. For a moment, dregs of light defined by the negative space take on the shapes of maple leaf, kite, comb, and other shapes that flicker across the walls too rapidly for you to interpret.\n\nCuriously, the only weapons here live amid light and cannot be worn: the'
- step 1 cur=None action='north'  pred_dst='undying garden' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Undying Garden' after moving north, matching the canonical location list.
  - obs: "The shapes of light you saw as you entered fade irrevocably.\n\n\nUndying Garden\nBarren branches and pines suggest winter's quiet. Metal and wood, painted and sculpted, the false garden is heedless of harvest, drought or rain.\n\nA jade compass rests at the garden's metaphysical center, between transluce"
