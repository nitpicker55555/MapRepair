# curses — per-edge error analysis

V3 walkthrough steps: 25
Predicted edges:      23
GT edges:             26

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 21 |
| SPURIOUS_PAIR | 2 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 21 |
| MISSED | 5 |

## Samples — SPURIOUS_PAIR

- step 5 walk_action='west' PRED: 'airing cupboard' --[west]--> 'potting room'
  > obs: 'Potting Room\nThis light room is full of pot plants, flowers, seeds, ornamental trowels and other miscellaneous garden implements.\n\nA pair of yellow rubber gloves hangs from a hook on one wall.\n\nAunt J'
- step 11 walk_action='southeast' PRED: 'attic' --[southeast]--> 'over the east wing'
  > obs: 'Over the East Wing\nThe air is dusty and warm, almost making you choke in this rather empty area. The attic turns from northwest to east, and there is also a tight doorway leading west. A short flight '

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='old furniture' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Old Furniture', matching canonical location list.
  - obs: 'Old Furniture\nScruffy old furniture is piled up here: armchairs with springs coming out, umbrella stands, a badly scratched cupboard, a table with one leg missing... You try to remember why you keep all this rubbish, and fail. Anyway the attic continues to the southeast.'
- step 1 cur=None action='north'  pred_dst='attic' GT_dst=None
  - LLM reasoning: Observation begins with 'Attic', which matches a canonical location; action is a compass direction.
  - obs: 'Attic\n\nA hinged trapdoor in the floor stands open, and light streams in from below.\n\nYou can also see a crumpled piece of paper and a chocolate biscuit here.'
- step 5 cur='old winery' action='west'  pred_dst='potting room' GT_dst="aunt jemima's lair"
  - obs: 'Potting Room\nThis light room is full of pot plants, flowers, seeds, ornamental trowels and other miscellaneous garden implements.\n\nA pair of yellow rubber gloves hangs from a hook on one wall.\n\nAunt Jemima, who has for years collected varieties of daisy, is engaged in her regular annual pastime of d'
- step 8 cur='potting room' action='east'  pred_dst='storage room' GT_dst="aunt jemima's lair"
  - obs: 'Storage Room\nA disused storage room off the winery. In one wall is an opening onto an ominous dark shaft, and beside it is a big Victorian-steam-engine style wheel with a handle on.\n\nInside the dark shaft is the old dumbwaiter.\n\nA steel wrench gathers dust in the corner.'
