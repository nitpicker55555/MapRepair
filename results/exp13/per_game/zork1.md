# zork1 — per-edge error analysis

V3 walkthrough steps: 22
Predicted edges:      20
GT edges:             29

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 15 |
| SPURIOUS_PAIR | 4 |
| WRONG_SRC_UNKNOWN | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 15 |
| MISSED | 14 |

## Samples — WRONG_SRC_UNKNOWN

- step 2 walk_action='up' PRED: '<unknown>' --[up]--> 'up a tree'
  > obs: "Up a Tree\nYou are about 10 feet above the ground nestled among some large branches. The nearest branch above you is above your reach.\nBeside you on the branch is a small bird's nest.\nIn the bird's nes"

## Samples — SPURIOUS_PAIR

- step 5 walk_action='west' PRED: 'north of house' --[west]--> 'kitchen'
  > obs: 'Kitchen\nYou are in the kitchen of the white house. A table seems to have been used recently for the preparation of food. A passage leads to the west and a dark staircase can be seen leading upward. A '
- step 10 walk_action='down' PRED: 'kitchen' --[down]--> 'cellar'
  > obs: 'You have moved into a dark place.\nThe trap door crashes shut, and you hear someone barring it.\n\nIt is pitch black. You are likely to be eaten by a grue.\nYour sword is glowing with a faint blue glow.'
- step 20 walk_action='east' PRED: 'temple' --[east]--> 'forest path'
  > obs: 'Forest Path'
- step 21 walk_action='down' PRED: 'forest path' --[down]--> 'cellar'
  > obs: 'Cellar\nThere is a bloody axe here.'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='north'  pred_dst='north of house' GT_dst=None
  - LLM reasoning: Observation explicitly names 'North of House', which matches the canonical location list.
  - obs: 'North of House\nYou are facing the north side of a white house. There is no door here, and all the windows are boarded up. To the north a narrow path winds through the trees.'
- step 1 cur=None action='north'  pred_dst='forest path' GT_dst=None
  - LLM reasoning: Observation states 'Forest Path', which matches the canonical location 'forest path'. The action is 'north', a compass direction, so this is a movement to 'forest path'.
  - obs: 'Forest Path\nThis is a path winding through a dimly lit forest. The path heads north-south here. One particularly large tree with some low branches stands at the edge of the path.'
