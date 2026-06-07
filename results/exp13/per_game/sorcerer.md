# sorcerer — per-edge error analysis

V3 walkthrough steps: 29
Predicted edges:      24
GT edges:             43

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 23 |
| SPURIOUS_PAIR | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 23 |
| MISSED | 20 |

## Samples — SPURIOUS_PAIR

- step 13 walk_action='northeast' PRED: "helistar's quarters" --[northeast]--> 'forest edge'
  > obs: 'Forest Edge\nTo the west, a path enters the blighted woods, which stretch out of sight. A signpost stands beside another path leading north, and to the east is a wide meadow. At the base of the signpos'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='west'  pred_dst='hallway (wooden door leading north)' GT_dst=None
  - LLM reasoning: The observation describes a hallway with a wooden door to the north, matching the canonical 'hallway (wooden door leading north)'.
  - obs: 'Hallway\nRooms lie to the east and west from this north-south corridor. A heavy wooden door, currently closed, leads north.\nTacked to the doorframe of your room is a note, hurriedly scribbled on parchment.'
- step 1 cur=None action='west'  pred_dst="belboz's quarters" GT_dst=None
  - LLM reasoning: Observation begins with 'Belboz's Quarters', which matches the canonical location name.
  - obs: "Belboz's Quarters\nThe personal chamber of Belboz, who presides over the Circle of Enchanters, is appointed with a beautiful woven wall hanging, an exotic morgia plant, and a wide darkwood desk crafted by the artisans of Gurth. The hallway lies to the east.\nPacing back and forth on a perch in the cor"
