# detective — per-edge error analysis

V3 walkthrough steps: 36
Predicted edges:      31
GT edges:             39

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 17 |
| WRONG_DST | 10 |
| HALLUCINATED_DST | 1 |
| HALLUCINATED_SRC | 1 |
| SPURIOUS_PAIR | 1 |
| SELF_LOOP | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 22 |
| RECALLED | 17 |

## Samples — WRONG_DST

- step 8 walk_action='north' PRED: "upstairs hallway (mayer's house)" --[north]--> "hallway (mayer's house, east/west intersect, exit to north)"  (GT here: hallway (mayer's house, east end, room to north))
  > obs: '<< Hallway >>\nYou are at the end of the hallway. To the north is a room, while to the west is the rest of the hallway.'
- step 9 walk_action='west' PRED: "hallway (mayer's house, east/west intersect, exit to north)" --[west]--> 'hallway (mayer\'s house, "guests" door to east)'  (GT here: bedroom)
  > obs: '<< Hallway >>\nYou are in the hallway. To the north is more hallway, and to the east is a door marked "Guests".'
- step 10 walk_action='north' PRED: 'hallway (mayer\'s house, "guests" door to east)' --[north]--> "hallway (mayer's house, east/west intersect, exit to north)"  (GT here: hallway (mayer's house, room to west, room to east))
  > obs: '<< Hallway >>\nYou are STILL in the hallway. There is EVEN MORE hallway to the north, and a room to the west and a room to the east of you.'
- step 13 walk_action='east' PRED: 'bedroom' --[east]--> 'hallway (15th floor, room 19-22)'  (GT here: hallway (mayer's house, east/west intersect, exit to north))
  > obs: '<< Hallway >>\nYou are still in the hallway. You can go north to where there is a police officer who will let you outside, or you can go east or west.'
- step 14 walk_action='north' PRED: 'hallway (15th floor, room 19-22)' --[north]--> 'outside (holiday inn to north, doughnut king to east, the wall to west)'  (GT here: hallway (sauna to west, pool a to east))
  > obs: '<< Outside >>\nYou pass the guard. He nods at you. You are now outside standing on the street. You can go north and east, your choice. To the north is more of the street, and to the east is a video sto'
- step 15 walk_action='east' PRED: 'outside (holiday inn to north, doughnut king to east, the wall to west)' --[east]--> 'video store'  (GT here: doughnut king)
  > obs: '<< Video Store >>\nYou are in a video store called Brickbuster Video. There are about 3,000 videos here. You can go north, or east.'

## Samples — SPURIOUS_PAIR

- step 17 walk_action='south' PRED: 'outside (video store to east)' --[south]--> 'mcdonalds'
  > obs: '<< McDonalds >>\nYou are in a McDonalds. You pay the guy behind the counter. Now there is a hamburger there. When you have picked it up, go north.\n\nYou can see a food hamburger here.'

## Samples — HALLUCINATED_DST

- step 11 walk_action='north' PRED: "hallway (mayer's house, east/west intersect, exit to north)" --[north]--> 'outside (restaurant to north, mayer home to east)'
  > obs: '<< Hallway >>\nYou are still in the hallway. You can go north to where there is a police officer who will let you outside, or you can go east or west.'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='west'  pred_dst='closet' GT_dst=None
  - LLM reasoning: Observation header is 'Closet', matching canonical location; action is compass direction.
  - obs: '<< Closet >>\nYou are in a closet. There is a gun on the floor. Better get it. To exit, go east.\n\nYou can see a small black pistol here.'
- step 1 cur=None action='east'  pred_dst="chief's office" GT_dst=None
  - LLM reasoning: Observation explicitly states arrival in 'Chief's office', which matches a canonical location.
  - obs: '<< Chief\'s office >>\nYou are standing in the Chief\'s office. He is telling you "The Mayor was murdered yeaterday night at 12:03 am. I want you to solve it before we get any bad publicity or the FBI has to come in. "Yessir!" You reply. He hands you a sheet of paper. Once you have read it, go north or'
- step 2 cur=None action='north'  pred_dst='outside (dead end to east)' GT_dst=None
  - LLM reasoning: Observation indicates arrival 'outside' with a dead end to the east, matching canonical 'outside (dead end to east)'.
  - obs: "<< Outside >>\nYou are outside in the cold. To the east is a dead end. To the west is the rest of the street. Papers are blowing around. It's amazingly cold for this time of year."
- step 3 cur=None action='east'  pred_dst='living room' GT_dst=None
  - LLM reasoning: The observation explicitly states arrival in the 'living room', which matches a canonical location. The action 'east' is a compass direction, so this is a movement edge.
  - obs: '<< Living room >>\nYou are standing in the living room. You see a battered piece of wood. You wonder "Should I pick this thing up?". Well, whether you do or don\'t the only way out of the room is back west.\n\nYou can see a wooden wood here.'
- step 4 cur=None action='west'  pred_dst="mayor's house (scene of crime)" GT_dst=None
  - LLM reasoning: Observation header is '<< Mayor's house >>' and description matches the canonical 'mayor's house (scene of crime)'.
  - obs: "<< Mayor's house >>\nYou are in the house, at the scene of the crime. You enter and flash your badge before a cop. He admits you. To the north is the upstairs. To the east is the living room and to the west is the dining room."
- step 13 cur='bedroom' action='east'  pred_dst='hallway (15th floor, room 19-22)' GT_dst="hallway (mayer's house, east/west intersect, exit to north)"
  - obs: '<< Hallway >>\nYou are still in the hallway. You can go north to where there is a police officer who will let you outside, or you can go east or west.'
- step 14 cur='hallway (15th floor, room 19-22)' action='north'  pred_dst='outside (holiday inn to north, doughnut king to east, the wall to west)' GT_dst='hallway (sauna to west, pool a to east)'
  - obs: '<< Outside >>\nYou pass the guard. He nods at you. You are now outside standing on the street. You can go north and east, your choice. To the north is more of the street, and to the east is a video store.\n\n[Your score has just gone up by ten points.]'
- step 15 cur='outside (holiday inn to north, doughnut king to east, the wall to west)' action='east'  pred_dst='video store' GT_dst='doughnut king'
  - obs: '<< Video Store >>\nYou are in a video store called Brickbuster Video. There are about 3,000 videos here. You can go north, or east.'
