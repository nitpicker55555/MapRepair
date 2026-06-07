# zenon — per-edge error analysis

V3 walkthrough steps: 17
Predicted edges:      15
GT edges:             26

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 10 |
| WRONG_DST | 3 |
| WRONG_DIRECTION | 1 |
| SPURIOUS_PAIR | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 15 |
| RECALLED | 10 |
| RECALLED_WRONG_DIR | 1 |

## Samples — WRONG_DIRECTION

- step 11 walk_action='down' PRED: 'lift(up button and a door to north)' --[down]--> 'lift(down button and a door to south)'
  > obs: 'The lift shoots downwards and gradually comes to a halt...\n\nLift\nThe bottom of the lift. There\'s a button with "Up" written on it and also a door to the north.'

## Samples — WRONG_DST

- step 6 walk_action='west' PRED: 'corridor(east old cell and stretches further west)' --[west]--> 'corridor(south into a room and west to old cell)'  (GT here: corridor(east to old cell and north to hall))
  > obs: 'Corridor\nA section of corridor that stretches east back toward your old cell and north into some sort of hall. It also leads south to another room.'
- step 7 walk_action='south' PRED: 'corridor(south into a room and west to old cell)' --[south]--> 'control room'  (GT here: armoury)
  > obs: 'Control Room\nSome kind of control room. Unfortunatly you fail to see a big button marked "Disable security devices"...Might have to be a bit more subtle.\n\nA cupboard is beneath the control panels.'
- step 10 walk_action='north' PRED: 'hall' --[north]--> 'lift(up button and a door to north)'  (GT here: lift(down button and a door to south))
  > obs: 'Lift\nA lift. There\'s a button that says "Down" on it and also a door to the south.'

## Samples — SPURIOUS_PAIR

- step 12 walk_action='north' PRED: 'lift(down button and a door to south)' --[north]--> 'hallway'
  > obs: 'Hallway\nA hallway leading south into a lift of some kind and northwards into a fairly large room.'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='landing' GT_dst=None
  - LLM reasoning: The observation begins with 'Landing', which matches the canonical location 'landing', indicating arrival in that room after moving south.
  - obs: 'Landing\nThis is a long corridor stretching west and east. The cell you where once held in is to the north, though you have no wish to return there.\n\nThe door is open'
- step 1 cur=None action='east'  pred_dst='corridor(south into a room and west to old cell)' GT_dst=None
  - LLM reasoning: Observation describes a corridor leading south into a room and west to old cell; matches canonical 'corridor(south into a room and west to old cell)'.
  - obs: 'Corridor\nAnother section of corridor that leads south into a room of some sort and west back towards your old cell.\n\nYou can see a Janitor here.'
- step 6 cur='corridor(east old cell and stretches further west)' action='west'  pred_dst='corridor(south into a room and west to old cell)' GT_dst='corridor(east to old cell and north to hall)'
  - obs: 'Corridor\nA section of corridor that stretches east back toward your old cell and north into some sort of hall. It also leads south to another room.'
- step 7 cur='corridor(south into a room and west to old cell)' action='south'  pred_dst='control room' GT_dst='armoury'
  - obs: 'Control Room\nSome kind of control room. Unfortunatly you fail to see a big button marked "Disable security devices"...Might have to be a bit more subtle.\n\nA cupboard is beneath the control panels.'
- step 10 cur='hall' action='north'  pred_dst='lift(up button and a door to north)' GT_dst='lift(down button and a door to south)'
  - obs: 'Lift\nA lift. There\'s a button that says "Down" on it and also a door to the south.'
