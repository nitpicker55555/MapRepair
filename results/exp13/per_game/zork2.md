# zork2 — per-edge error analysis

V3 walkthrough steps: 35
Predicted edges:      32
GT edges:             36

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 25 |
| SPURIOUS_PAIR | 5 |
| SELF_LOOP | 1 |
| WRONG_DST | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 25 |
| MISSED | 11 |

## Samples — WRONG_DST

- step 11 walk_action='north' PRED: 'carousel room' --[north]--> 'marble hall'  (GT here: topiary)
  > obs: "You're not sure which direction is which. This room is very disorienting.\n\nMarble Hall\nThis is an arched hall of fine marble. The hall stops abruptly to the north at a ford across a stream, where the "

## Samples — SPURIOUS_PAIR

- step 10 walk_action='west' PRED: 'topiary' --[west]--> 'carousel room'
  > obs: 'Carousel Room'
- step 13 walk_action='north' PRED: 'deep ford' --[north]--> 'ledge in ravine'
  > obs: 'Ledge in Ravine\nYou are on a narrow ledge near the bottom of a deep ravine. The ledge continues to the west. A precarious climb up to another tiny ledge is possible. A short scramble down the rock fac'
- step 19 walk_action='south' PRED: 'end of ledge' --[south]--> 'stone bridge'
  > obs: 'Stone Bridge\nThis is the middle of a ruined but still impressive stone bridge spanning a deep chasm. Water flows far beneath. A paved path leads north into a large open space. To the south, the path l'
- step 27 walk_action='east' PRED: 'dragon room' --[east]--> 'ledge in ravine'
  > obs: 'Ledge in Ravine\nThe princess enters from the west. She seems surprised to see you.'
- step 34 walk_action='south' PRED: 'gazebo' --[south]--> 'topiary'
  > obs: 'Topiary'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='narrow tunnel' GT_dst=None
  - LLM reasoning: Observation states 'Narrow Tunnel' as the location, matching the canonical list; action is 'south', a compass direction.
  - obs: 'Narrow Tunnel\nYou are standing at the southern end of a narrow tunnel where it opens into a wide cavern. The cavern is dimly illuminated by phosphorescent mosses clinging to its high ceiling. A deep ravine winds through the cavern, with a small stream at the bottom. The walls of the ravine are steep'
- step 1 cur=None action='south'  pred_dst='foot bridge' GT_dst=None
  - LLM reasoning: The observation after moving south names 'Foot Bridge', which matches the canonical room 'foot bridge'.
  - obs: 'Foot Bridge\nYou are standing on a crude but sturdy wooden foot bridge crossing a deep ravine. The path runs north and south from here.'
- step 2 cur=None action='south'  pred_dst='great cavern' GT_dst=None
  - LLM reasoning: The observation begins with 'Great Cavern' and describes the center of the great cavern, matching the canonical room name 'great cavern'.
  - obs: 'Great Cavern\nThis is the center of the great cavern, carved out of the limestone. Stalactites and stalagmites of many sizes are everywhere. The room glows with dim light provided by phosphorescent moss, and weird shadows move all around you. A narrow path winds southwest among the stalagmites, and a'
- step 11 cur='carousel room' action='north'  pred_dst='marble hall' GT_dst='topiary'
  - obs: "You're not sure which direction is which. This room is very disorienting.\n\nMarble Hall\nThis is an arched hall of fine marble. The hall stops abruptly to the north at a ford across a stream, where the marble is cracked and broken. Perhaps a flood or collapse of the cave was responsible. To the south "
