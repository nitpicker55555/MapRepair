# tryst205 — per-edge error analysis

V3 walkthrough steps: 9
Predicted edges:      7
GT edges:             9

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 6 |
| SELF_LOOP | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 6 |
| MISSED | 3 |

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='north'  pred_dst='master bedroom' GT_dst=None
  - LLM reasoning: The observation begins with 'Master Bedroom', which matches a canonical location, indicating arrival in that room after moving north.
  - obs: "Master Bedroom\nWhile considerably better than your children's rooms, your room is also a little bit on the messy side. There is a king-size bed here with mountains of assorted pillows. A masterfully finished dresser graces the south wall. To the south is the hallway and a bathroom is off to the west"
- step 1 cur=None action='west'  pred_dst='master bathroom' GT_dst=None
  - LLM reasoning: Observation begins with 'Master Bathroom', which matches the canonical location 'master bathroom'.
  - obs: 'Master Bathroom\nBeautiful plants hang down from the ceiling and ceramic cherubims watch your every move in this small area off of the master bedroom to the east. To the south is a spacious walk-in closet with mirrored doors.\n\nThe closet door is closed.\n\nYou can also see a little gnome here.'
