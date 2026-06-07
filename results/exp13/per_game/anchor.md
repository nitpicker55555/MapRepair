# anchor — per-edge error analysis

V3 walkthrough steps: 30
Predicted edges:      27
GT edges:             42

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 26 |
| WRONG_DST | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 26 |
| MISSED | 16 |

## Samples — WRONG_DST

- step 4 walk_action='west' PRED: 'file room' --[west]--> 'outside the real estate office'  (GT here: office)
  > obs: '(opening the real estate office door first)\nThe rain is still coming down, so you open your umbrella.\n\nOutside the Real Estate Office\nA grim little cul-de-sac, tucked away in a corner of the claustrop'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='southeast'  pred_dst='alley' GT_dst=None
  - LLM reasoning: Observation begins with 'Alley', which matches a canonical location; action is a compass direction.
  - obs: 'Alley\nThis narrow aperture between two buildings is nearly blocked with piles of rotting cardboard boxes and overstuffed garbage cans. Ugly, half-crumbling brick walls to either side totter oppressively over you. The alley ends here at a tall, wooden fence.\n\nHigh up on the wall of the northern build'
- step 1 cur=None action='up'  pred_dst='file room' GT_dst=None
  - LLM reasoning: The observation explicitly states arrival in 'File Room', which matches the canonical location list.
  - obs: "(opening the transom window first)\n(closing the umbrella first)\nIt's a tight squeeze, but you just manage to wriggle through, dropping quietly to the floor inside.\n\nFile Room\nPeering through the murk, you can make out the blocky outlines of filing cabinets lining the walls and a doorway to the west."
- step 2 cur=None action='west'  pred_dst='office' GT_dst=None
  - LLM reasoning: Observation begins with 'Office', which matches the canonical room name 'office'.
  - obs: 'Office\nPallid gray light trickles in through the drawn blinds. The office is deserted, papers still scattered across the top of the desk. The front door lies west, and the file room lies east.\n\nSitting on the corner of the paper-strewn desk are a telephone and an answering machine.\n\nSomeone seems to'
- step 4 cur='file room' action='west'  pred_dst='outside the real estate office' GT_dst='office'
  - obs: '(opening the real estate office door first)\nThe rain is still coming down, so you open your umbrella.\n\nOutside the Real Estate Office\nA grim little cul-de-sac, tucked away in a corner of the claustrophobic tangle of narrow, twisting avenues that largely constitute the older portion of Anchorhead. Li'
