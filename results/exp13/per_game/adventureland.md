# adventureland — per-edge error analysis

V3 walkthrough steps: 23
Predicted edges:      22
GT edges:             27

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 13 |
| SPURIOUS_PAIR | 5 |
| WRONG_DST | 3 |
| SELF_LOOP | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 14 |
| RECALLED | 13 |

## Samples — WRONG_DST

- step 8 walk_action='down' PRED: 'inside stump' --[down]--> 'hole'  (GT here: root chamber)
  > obs: "Hole\nYou're in a semi-dark hole by the root chamber.\n\nObvious exits: Up, Down.\n\nYou can also see: locked door"
- step 9 walk_action='down' PRED: 'hole' --[down]--> 'maze (exit north, south, east, west, up, down; arrow down)'  (GT here: hall)
  > obs: "Darkness\nIt is pitch dark, and you can't see a thing."
- step 10 walk_action='down' PRED: 'maze (exit north, south, east, west, up, down; arrow down)' --[down]--> 'cavern'  (GT here: chasm)
  > obs: "Cavern\nYou're in a large cavern.\n\nObvious exits: North, South, West, Up, Down."

## Samples — SPURIOUS_PAIR

- step 3 walk_action='west' PRED: 'bog' --[west]--> 'grove'
  > obs: "Grove\nYou're in a hidden grove.\n\nObvious exits: North, East, West.\n\nYou can also see: rusty axe (magic word- BUNYON -on it) - *SMALL STATUE OF A BLUE OX* - *JEWELLED FRUIT* - sign reads- PAUL'S PLACE"
- step 6 walk_action='down' PRED: 'dismal swamp' --[down]--> 'root chamber'
  > obs: "Root Chamber\nYou're in a root chamber under the stump.\n\nObvious exits: Up, Down.\n\nYou can also see: dark hole - *POT OF RUBIES*"
- step 19 walk_action='up' PRED: 'maze (exit north, south, east, west, up, down; arrow down)' --[up]--> 'maze (exit north, south, east, west, up, down; rock aladdin)'
  > obs: "Maze\nYou're in a maze of pits.\n\nObvious exits: North, South, East, West, Up, Down.\n\nYou can also see: strange scratchings on rock reveals- ALADDIN WAS HERE"
- step 20 walk_action='south' PRED: 'maze (exit north, south, east, west, up, down; rock aladdin)' --[south]--> 'dismal swamp'
  > obs: "Dismal Swamp\nYou're in a dismal swamp.\n\nObvious exits: North, South, East, West.\n\nYou can also see: rusty axe (magic word- BUNYON -on it) - hollow stump and remains of a felled tree - evil smelling mu"
- step 22 walk_action='up' PRED: 'meadow' --[up]--> 'dismal swamp'
  > obs: "Dismal Swamp\nYou're in a dismal swamp.\n\nObvious exits: North, South, East, West.\n\nYou can also see: empty wine bladder - flint and steel - rusty axe (magic word- BUNYON -on it) - hollow stump and rema"

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='east'  pred_dst='meadow' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Meadow', which matches the canonical location list; action is a compass direction.
  - obs: "Meadow\nYou're in a sunny meadow.\n\nObvious exits: North, South, East, West.\n\nYou can also see: sleeping Dragon - sign reads- IN SOME CASES MUD IS GOOD, IN OTHERS..."
- step 8 cur='inside stump' action='down'  pred_dst='hole' GT_dst='root chamber'
  - obs: "Hole\nYou're in a semi-dark hole by the root chamber.\n\nObvious exits: Up, Down.\n\nYou can also see: locked door"
- step 9 cur='hole' action='down'  pred_dst='maze (exit north, south, east, west, up, down; arrow down)' GT_dst='hall'
  - obs: "Darkness\nIt is pitch dark, and you can't see a thing."
- step 10 cur='maze (exit north, south, east, west, up, down; arrow down)' action='down'  pred_dst='cavern' GT_dst='chasm'
  - obs: "Cavern\nYou're in a large cavern.\n\nObvious exits: North, South, West, Up, Down."
