# night — per-edge error analysis

V3 walkthrough steps: 31
Predicted edges:      28
GT edges:             32

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 16 |
| SPURIOUS_PAIR | 7 |
| WRONG_DST | 3 |
| WRONG_DIRECTION | 2 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 16 |
| MISSED | 15 |
| RECALLED_WRONG_DIR | 1 |

## Samples — WRONG_DIRECTION

- step 14 walk_action='north' PRED: 'hall (1st floor, north end of north/south hall)' --[north]--> 'hall (1st floor, middle of north/south hall)'
  > obs: 'Hall\n\nTo one side, a water fountain hums quietly.'
- step 19 walk_action='east' PRED: 'hall (2nd floor, middle of north/south hall)' --[east]--> 'maze of twisty passages (stop 1)'
  > obs: 'Maze of Twisty Passages\nYou are in a maze of twisty little passages, all alike.'

## Samples — WRONG_DST

- step 4 walk_action='north' PRED: 'hall (1st floor, middle of north/south hall)' --[north]--> 'hall outside computer site'  (GT here: hall (1st floor, north end of north/south hall))
  > obs: 'Hall Outside Computer Site\n\nTo one side, a water fountain hums quietly.'
- step 13 walk_action='west' PRED: 'hall outside elevator (1st floor)' --[west]--> 'hall (1st floor, north end of north/south hall)'  (GT here: hall (1st floor, middle of north/south hall))
  > obs: 'Hall'
- step 22 walk_action='east' PRED: 'maze of twisty passages (stop 4)' --[east]--> 'maze of twisty passages (stop 5)'  (GT here: maze of twisty passages (stop 3))
  > obs: 'Maze of Twisty Passages'

## Samples — SPURIOUS_PAIR

- step 15 walk_action='west' PRED: 'hall (1st floor, middle of north/south hall)' --[west]--> 'stairwell (first floor)'
  > obs: 'Stairwell (First Floor)'
- step 20 walk_action='west' PRED: 'maze of twisty passages (stop 1)' --[west]--> 'maze of twisty passages (stop 5)'
  > obs: 'Maze of Twisty Passages'
- step 21 walk_action='west' PRED: 'maze of twisty passages (stop 5)' --[west]--> 'maze of twisty passages (stop 4)'
  > obs: 'Maze of Twisty Passages'
- step 23 walk_action='west' PRED: 'maze of twisty passages (stop 5)' --[west]--> 'maze of twisty passages (stop 4)'
  > obs: 'Maze of Twisty Passages'
- step 24 walk_action='up' PRED: 'maze of twisty passages (stop 4)' --[up]--> 'hall (1st floor, north end of north/south hall)'
  > obs: 'Hall\n\nTo one side, a buzzing water fountain has been moved, revealing an open panel and a dark, sinister hole leading down.'
- step 25 walk_action='north' PRED: 'hall (1st floor, north end of north/south hall)' --[north]--> 'outside physics office'
  > obs: 'Outside Physics Office'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='northeast'  pred_dst='hall outside computer site' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Hall Outside Computer Site', matching a canonical location.
  - obs: 'Hall Outside Computer Site\nThis is the north end of the third floor of Hodges Hall.  To the southwest is the entrance to the Computer Site, and to the east are stairs up to the fourth floor.  Stairs down to lower floors are to the west.\n\nTo one side, a water fountain hums quietly.'
- step 1 cur=None action='south'  pred_dst='hall (3rd floor, middle of north/south hall)' GT_dst=None
  - LLM reasoning: Observation describes being in the middle of a long north-south hallway, matching 'hall (3rd floor, middle of north/south hall)' from the canonical list.
  - obs: "Hall\nYou're in the middle of a long north-south hallway in Hodges Hall.  You can go east here as well as north or south."
- step 2 cur=None action='east'  pred_dst='hall outside elevator (3rd floor)' GT_dst=None
  - LLM reasoning: Observation states 'Hall Outside Elevator', which matches the canonical 'hall outside elevator (3rd floor)'.
  - obs: "Hall Outside Elevator\nYou're in a short east-west hall off the main hall.  You can see an elevator here.\n\nThere's a sign on the elevator.\n\nYou see a box mounted on the wall in one corner."
- step 4 cur='hall (1st floor, middle of north/south hall)' action='north'  pred_dst='hall outside computer site' GT_dst='hall (1st floor, north end of north/south hall)'
  - obs: 'Hall Outside Computer Site\n\nTo one side, a water fountain hums quietly.'
- step 13 cur='hall outside elevator (1st floor)' action='west'  pred_dst='hall (1st floor, north end of north/south hall)' GT_dst='hall (1st floor, middle of north/south hall)'
  - obs: 'Hall'
- step 22 cur='maze of twisty passages (stop 4)' action='east'  pred_dst='maze of twisty passages (stop 5)' GT_dst='maze of twisty passages (stop 3)'
  - obs: 'Maze of Twisty Passages'
