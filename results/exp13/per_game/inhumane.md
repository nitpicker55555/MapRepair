# inhumane — per-edge error analysis

V3 walkthrough steps: 30
Predicted edges:      28
GT edges:             50

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 21 |
| SPURIOUS_PAIR | 6 |
| WRONG_DIRECTION | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 28 |
| RECALLED | 21 |
| RECALLED_WRONG_DIR | 1 |

## Samples — WRONG_DIRECTION

- step 7 walk_action='west' PRED: 'round room' --[west]--> 't-intersection (east/west/south, east to round room)'
  > obs: 'T-Intersection\nThe east/west corridor branches to the south here.'

## Samples — SPURIOUS_PAIR

- step 8 walk_action='south' PRED: 't-intersection (east/west/south, east to round room)' --[south]--> 'south branch'
  > obs: 'South Branch\nThe hall ends in an alcove to the south. The intersection in north. Odd... The floor feels a bit shaky here. Nothing to worry about though.'
- step 11 walk_action='east' PRED: 'south branch' --[east]--> 't intersection'
  > obs: 'T-Intersection\nTo the hall goes east from here. To the west is the stairway to the round room. A narrow passage leads south.'
- step 12 walk_action='east' PRED: 't intersection' --[east]--> 'corridor at doorway'
  > obs: 'Corridor at Doorway\nThe hall goes east and west from here and there is a doorway to the north.'
- step 18 walk_action='north' PRED: 'end of glass hall' --[north]--> 'landing'
  > obs: 'Landing\nThis is a landing on the stairway. You can go down to the north or up to the south. A passage leads east.'
- step 21 walk_action='east' PRED: 'exercise-wheel room' --[east]--> 't intersection'
  > obs: 'T Intersection\nThe hallway goes east and west here with a branch north.'
- step 25 walk_action='west' PRED: 'north stalagmite room' --[west]--> 'corridor near pit'
  > obs: 'Corridor Near Pit\nWest is a large round pit about 65 feet deep. Running down the center of the pit is an axle with a platform attached at the level of this hall. The platform is swinging around the wa'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='center of camp' GT_dst=None
  - LLM reasoning: Observation heading matches canonical location 'center of camp', so player arrived there.
  - obs: "Center of Camp\nThis is the center of your camp. The entire place seems to be completely deserted. You can see Roboff's tent to the south and the storage tent west. East is open desert."
- step 1 cur=None action='west'  pred_dst='storage tent' GT_dst=None
  - LLM reasoning: Observation heading is 'Storage Tent', which matches the canonical location 'storage tent'. The action is a compass direction ('west'), so this is a movement to a new room.
  - obs: 'Storage Tent\nThis is where the party kept all its supplies. However Roboff seems to have cleaned everything out that was of any use. Several arrows of Da-Glo paint on the wall attract your attention to the empty food and water chests.\nLying on the ground is a brass torch of the "self-igniting" type.'
- step 7 cur='center of camp' action='west'  pred_dst='t-intersection (east/west/south, east to round room)' GT_dst='storage tent'
  - obs: 'T-Intersection\nThe east/west corridor branches to the south here.'
