# spellbrkr — per-edge error analysis

V3 walkthrough steps: 16
Predicted edges:      14
GT edges:             24

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 7 |
| SPURIOUS_PAIR | 5 |
| WRONG_DST | 1 |
| WRONG_DIRECTION | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 16 |
| RECALLED | 7 |
| RECALLED_WRONG_DIR | 1 |

## Samples — WRONG_DIRECTION

- step 8 walk_action='up' PRED: 'boulder(a long oblong boulder below you)' --[up]--> 'boulder(a nice oblong boulder above you and another large one below you)'
  > obs: 'Boulder\nYou are clinging to a boulder that is floating in midair. There are many other boulders around, also floating, and lots of dust and dirt, also not moving. A nice oblong boulder is above, and a'

## Samples — WRONG_DST

- step 7 walk_action='up' PRED: 'cliff top' --[up]--> 'boulder(a long oblong boulder below you)'  (GT here: boulder(large boulder above you and cliff below you))
  > obs: 'Boulder\nYou are clinging to a boulder that is floating in midair. There are many other boulders around, also floating, and lots of dust and dirt, also not moving. One particularly large boulder with g'

## Samples — SPURIOUS_PAIR

- step 2 walk_action='down' PRED: 'belwit square' --[down]--> 'midair'
  > obs: 'As you leave, the "1" cube reappears in your hand.\n\nMidair\nYou are in midair. You are dizzyingly high above a trackless wilderness to the east of a range of jagged mountains.\nYou have begun to fall.\n\n'
- step 3 walk_action='east' PRED: 'midair' --[east]--> 'hall of stone'
  > obs: 'As you leave, the "1" cube reappears in your hand.\n\nHall of Stone\nThis is a long hall built of crudely dressed stone. The blocks are as tall as you and the ceiling invisible in the gloom above. Dirt t'
- step 5 walk_action='south' PRED: 'ruins room' --[south]--> 'cliff middle'
  > obs: 'As you leave, the "1" cube reappears in your hand.\n\nCliff Middle\nA narrow ledge, barely wide enough to stand on, interrupts the cliff here.\nThere is a dirty scroll here.'
- step 13 walk_action='south' PRED: 'mountain top' --[south]--> 'meadow'
  > obs: 'As you leave, the "2" cube reappears in your hand.\n\nMeadow\nThis is a warm, sunny meadow nestled among low hills. Wildflowers abound, and insects buzz lazily through the air. The grass is soft and thic'
- step 14 walk_action='west' PRED: 'meadow' --[west]--> 'cave entrance'
  > obs: 'As you leave, the "1" cube reappears in your hand.\n\nCave Entrance\nA well-worn trail terminates here where a cave enters the cliff-side to the north. Outside the cave is a litter of bones, old hides an'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='guild hall' GT_dst=None
  - LLM reasoning: Observation begins with 'Guild Hall', matching a canonical location, indicating arrival after moving south.
  - obs: 'Guild Hall\nThis is the entrance to the Guild Hall in Borphee. To the north is the Council Chamber, and to the south is an exit leading outside. Little is left of the sumptuous buffet lunch. Only a loaf of bread and some smoked fish remains.\nA shadowy figure in a dark cloak is running out the door.\n\n'
- step 1 cur=None action='south'  pred_dst='belwit square' GT_dst=None
  - LLM reasoning: The observation explicitly names 'Belwit Square', which matches a canonical location, after moving south.
  - obs: 'Belwit Square\nThis is Belwit Square. Its many historic and picturesque buildings are obscured by a cloud of orange smoke.'
- step 7 cur='cliff top' action='up'  pred_dst='boulder(a long oblong boulder below you)' GT_dst='boulder(large boulder above you and cliff below you)'
  - obs: 'Boulder\nYou are clinging to a boulder that is floating in midair. There are many other boulders around, also floating, and lots of dust and dirt, also not moving. One particularly large boulder with good handholds is above you. Below you is the cliff face.'
- step 8 cur='boulder(a long oblong boulder below you)' action='up'  pred_dst='boulder(a nice oblong boulder above you and another large one below you)' GT_dst='mountain top'
  - obs: 'Boulder\nYou are clinging to a boulder that is floating in midair. There are many other boulders around, also floating, and lots of dust and dirt, also not moving. A nice oblong boulder is above, and another large one is below you.'
