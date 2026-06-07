# enter — per-edge error analysis

V3 walkthrough steps: 23
Predicted edges:      21
GT edges:             26

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 20 |
| WRONG_DST | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 20 |
| MISSED | 6 |

## Samples — WRONG_DST

- step 15 walk_action='north' PRED: 'north-south hall--mid way' --[north]--> 'north-south hall--north end'  (GT here: hall near room 7)
  > obs: "North-South Hall--North End\nThis is the end of a rather plain hall, running north to south.  To the north is the building's main entranceway, which opens automatically whenever one approaches it.  Sch"

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='north'  pred_dst='the enterprise closet' GT_dst=None
  - LLM reasoning: Observation begins with 'The Enterprise Closet', matching a canonical location; action is a compass direction.
  - obs: 'The Enterprise Closet\nThis rather dark, slightly smelly closet is home to the Enterprise Store, a service of the Enterprise Program, an educational project in which about a dozen underachieving eighth graders run their own business.  The closet is packed with piles of soft drinks for the school vend'
- step 1 cur=None action='south'  pred_dst='the cafe' GT_dst=None
  - LLM reasoning: Observation explicitly names 'The Cafe', which matches the canonical location list.
  - obs: "The Cafe\nThe Neil Armstrong Middle and Elementary School Cafeteria is a big, unadorned room that echoes like Carlsbad Caverns whenever there's any noise at all, though it's quiet here now.  There's an exit to the west; and a door to the north leads to the Enterprise closet, where materials for your "
- step 15 cur='north-south hall--mid way' action='north'  pred_dst='north-south hall--north end' GT_dst='hall near room 7'
  - obs: "North-South Hall--North End\nThis is the end of a rather plain hall, running north to south.  To the north is the building's main entranceway, which opens automatically whenever one approaches it.  School rules clearly prohibit you from going outside during the school day. Ms. Empirious' room, Room 8"
