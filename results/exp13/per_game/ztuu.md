# ztuu — per-edge error analysis

V3 walkthrough steps: 26
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

- step 18 walk_action='south' PRED: 'cultural complex' --[south]--> 'the mud forum'  (GT here: hall of science lobby)
  > obs: 'The Mud Forum\nThis large, square chamber is certainly messy enough for a mud forum (whatever that might be), but there is no mud in sight (for purposes of argument, it could be postulated that the wat'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='northeast'  pred_dst='tunnel' GT_dst=None
  - LLM reasoning: The observation states 'Tunnel' as the room name after moving northeast, matching the canonical list.
  - obs: 'As you enter the tunnel, you feel a slight rumble beneath your crepe-soled boots. As you turn to look back up to the exit, giant, translucent boulders begin to fall from the walls and ceiling. You duck and cover your head -- when the dust clears, you feel your heart sink in your chest. You are trapp'
- step 1 cur=None action='northeast'  pred_dst='cultural complex' GT_dst=None
  - LLM reasoning: Observation begins with 'Cultural Complex', which matches the canonical location 'cultural complex'.
  - obs: 'Cultural Complex\nThis imposing ante-room, the center of what was apparently the cultural center of the GUE, is adorned in the ghastly style of the GUE\'s "Grotesque Period."  With leering gargoyles, cartoonish friezes depicting long-forgotten scenes of GUE history, and primitive statuary of pointy-he'
- step 2 cur=None action='north'  pred_dst='convention center lobby' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Convention Center Lobby', which matches the canonical list; action is a compass direction.
  - obs: 'Convention Center Lobby\nYou are standing at the entrance of the GUE Convention Center. On the floor, you can see occasional splotches of paint through the grime and dust. Although the whole cultural center appears to have been long-abandoned, loud, low-pitched sounds emanate from the door standing o'
- step 3 cur=None action='northwest'  pred_dst='souvenir stand' GT_dst=None
  - LLM reasoning: Observation begins with 'Souvenir Stand', which matches a canonical location; action is a compass direction.
  - obs: 'Souvenir Stand\nYou find yourself standing in front of a souvenir stand that is clearly affiliated with the Convention Center. An odd-looking salesman sits behind the counter, his head resting on his hand and his eyes glued to a spot somewhere above and behind you (a quick, frightened glance over you'
- step 4 cur=None action='north'  pred_dst='changing room' GT_dst=None
  - LLM reasoning: The observation begins with 'Changing Room', which matches the canonical location 'changing room'. The action is a canonical compass direction ('north'), so this is a movement to a new room.
  - obs: 'Changing Room\nIt is obvious from the array of body suits, masks, and gloves heaped on the ground, that this is a changing room, and not a very comfortable one at that. No chairs, tables, hooks, hangers, or shelves here: just the costumes themselves, each piece more hideous than the last. A mirror ha'
- step 18 cur='cultural complex' action='south'  pred_dst='the mud forum' GT_dst='hall of science lobby'
  - obs: 'The Mud Forum\nThis large, square chamber is certainly messy enough for a mud forum (whatever that might be), but there is no mud in sight (for purposes of argument, it could be postulated that the water source for the mud has dried up, or that mud was brought in for some scientific purpose, or that '
