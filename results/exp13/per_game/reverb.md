# reverb — per-edge error analysis

V3 walkthrough steps: 23
Predicted edges:      20
GT edges:             25

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 12 |
| SPURIOUS_PAIR | 4 |
| WRONG_DST | 2 |
| HALLUCINATED_DST | 1 |
| HALLUCINATED_SRC | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 13 |
| RECALLED | 12 |

## Samples — WRONG_DST

- step 7 walk_action='east' PRED: 'office building' --[east]--> 'law office'  (GT here: downtown)
  > obs: 'Law Office\nYou are in a bustling law office. The east wall is dominated by a large window.\n\nJill stands here.\n\nAs you walk into the office, Jill spots you. "Hi!" she says. "Listen, Stan. I have reason'
- step 22 walk_action='north' PRED: 'street, by pizza parlor' --[north]--> 'roof of office building'  (GT here: pizza parlor)
  > obs: 'The stairs angle up and back, and you emerge south.\n\nRoof of Office Building\nYou are on the roof of the government building. You can go back down the stairs to the north. There is a lightning rod stic'

## Samples — SPURIOUS_PAIR

- step 10 walk_action='east' PRED: "mayor's office" --[east]--> 'downtown'
  > obs: 'Downtown\n\nA police car drives by. You press yourself against a wall until it goes by.'
- step 14 walk_action='east' PRED: 'pizza parlor' --[east]--> 'street, by department store'
  > obs: "Street, by Department Store\nOn this end of the street, a department store called Barkley's sits to the south."
- step 19 walk_action='southwest' PRED: 'street, by department store' --[southwest]--> 'hardware department'
  > obs: "Hardware Department\nYou are in the hardware department at Barkley's. There are tools all over the place. You can go northeast to the clothing department, or east to the cosmetics department.\n \nThe pla"
- step 21 walk_action='west' PRED: 'clothing department' --[west]--> 'street, by pizza parlor'
  > obs: 'Street, by Pizza Parlor\n\nYou lurch for a moment as the ground makes loud rumbling noises. Sounds like an earthquake is about to hit. A bad one.'

## Samples — HALLUCINATED_DST

- step 8 walk_action='west' PRED: 'law office' --[west]--> 'second floor'
  > obs: 'Office Building, Second Floor'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='southwest'  pred_dst='pizza parlor' GT_dst=None
  - LLM reasoning: Observation states arrival in 'Mr. Tasty's Pizza Parlor', matching canonical 'pizza parlor'.
  - obs: 'You walk around to the front of the pizza parlor.\n\nPizza Parlor\nYou are standing in the middle of "Mr. Tasty\'s Pizza Parlor". The counter separating you from the back of the parlor stretches in front of you on the northeast corner of the room. To the south is an exit to the street.\n\nOn the counter i'
- step 1 cur=None action='south'  pred_dst='street, by pizza parlor' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Street, by Pizza Parlor', matching the canonical location list.
  - obs: "Street, by Pizza Parlor\nSan Doppleton is where you live, work and play -- at least until summer. It has the typical California culture; which is to say, none. But at least it's just forty minutes' drive from Santa Barbara and the beach, so you can't complain much.\n\nYou stand in the heart of San Dopp"
- step 2 cur=None action='west'  pred_dst='street, near courthouse' GT_dst=None
  - LLM reasoning: Observation heading matches canonical room 'street, near courthouse'; action is compass direction.
  - obs: 'Street, Near Courthouse\nYou are approaching the center of downtown San Doppleton. There is a courthouse to the south. The street runs east and west.'
- step 7 cur='office building' action='east'  pred_dst='law office' GT_dst='downtown'
  - obs: 'Law Office\nYou are in a bustling law office. The east wall is dominated by a large window.\n\nJill stands here.\n\nAs you walk into the office, Jill spots you. "Hi!" she says. "Listen, Stan. I have reason to believe that the Mayor is tangled up in the mafia. I managed to use some connections to get this'
- step 8 cur='law office' action='west'  pred_dst='second floor' GT_dst='office building, second floor'
  - obs: 'Office Building, Second Floor'
- step 22 cur='street, by pizza parlor' action='north'  pred_dst='roof of office building' GT_dst='pizza parlor'
  - obs: 'The stairs angle up and back, and you emerge south.\n\nRoof of Office Building\nYou are on the roof of the government building. You can go back down the stairs to the north. There is a lightning rod sticking up.'
