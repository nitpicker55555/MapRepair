# cutthroat — per-edge error analysis

V3 walkthrough steps: 41
Predicted edges:      33
GT edges:             49

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 16 |
| WRONG_DST | 6 |
| SPURIOUS_PAIR | 6 |
| SELF_LOOP | 2 |
| WRONG_DIRECTION | 1 |
| HALLUCINATED_DST | 1 |
| HALLUCINATED_SRC | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 32 |
| RECALLED | 16 |
| RECALLED_WRONG_DIR | 1 |

## Samples — WRONG_DIRECTION

- step 16 walk_action='east' PRED: 'back alley (east end of east/west alley)' --[east]--> 'back alley (behind outfitters international)'
  > obs: 'Back Alley'

## Samples — WRONG_DST

- step 9 walk_action='west' PRED: 'wharf road (east end of east/west road)' --[west]--> 'wharf road (end of east/west road)'  (GT here: wharf road (north to outfitters international))
  > obs: 'Wharf Road'
- step 11 walk_action='south' PRED: 'wharf road (east end of east/west road)' --[south]--> 'vacant lot'  (GT here: the shanty)
  > obs: 'Vacant Lot\nYou are standing on the former site of the Outfitters International warehouse, which burned down a few months back. To the north is the Wharf Road and an alley is to the south.'
- step 13 walk_action='west' PRED: 'back alley (south to vacant lot)' --[west]--> 'back alley (west end of east/west alley)'  (GT here: back alley (back door of mcginty salvage))
  > obs: 'Back Alley\nYou are in a narrow alley. To the north is the back door of McGinty Salvage, as well as a small window. An overgrown field lies to the south.'
- step 15 walk_action='east' PRED: 'back alley (west end of east/west alley)' --[east]--> 'back alley (east end of east/west alley)'  (GT here: back alley (back door of mcginty salvage))
  > obs: 'Back Alley'
- step 33 walk_action='west' PRED: 'back alley (behind outfitters international)' --[west]--> 'back alley (west end of east/west alley)'  (GT here: back alley (south to vacant lot))
  > obs: 'Back Alley'
- step 35 walk_action='north' PRED: 'vacant lot' --[north]--> 'wharf road (north to mcginty salvage)'  (GT here: wharf road (north to outfitters international former warehouse))
  > obs: 'Wharf Road'

## Samples — SPURIOUS_PAIR

- step 10 walk_action='west' PRED: 'wharf road (end of east/west road)' --[west]--> 'wharf road (east end of east/west road)'
  > obs: 'Wharf Road'
- step 31 walk_action='northwest' PRED: 'ocean road (north end of north/south road)' --[northwest]--> 'back alley (west end of east/west alley)'
  > obs: 'Back Alley'
- step 32 walk_action='west' PRED: 'back alley (west end of east/west alley)' --[west]--> 'back alley (behind outfitters international)'
  > obs: 'Back Alley'
- step 34 walk_action='north' PRED: 'back alley (west end of east/west alley)' --[north]--> 'vacant lot'
  > obs: 'Vacant Lot'
- step 37 walk_action='west' PRED: 'wharf road (end of east/west road)' --[west]--> 'wharf road (east end of east/west road)'
  > obs: 'Wharf Road'
- step 38 walk_action='southwest' PRED: 'wharf road (east end of east/west road)' --[southwest]--> 'winding road (start of southwest-ward road)'
  > obs: 'Winding Road\nA winding road starts up here to the southwest. You can see the top of the lighthouse off to the northwest. The Wharf Road lies to the northeast.'

## Samples — HALLUCINATED_DST

- step 28 walk_action='south' PRED: "mariners' trust" --[south]--> 'shore road'
  > obs: 'Shore Road'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='north'  pred_dst='upstairs hallway' GT_dst=None
  - LLM reasoning: Observation states 'Upstairs Hallway', which matches a canonical location; action is compass direction.
  - obs: 'Upstairs Hallway\nYou are at the top of the stairs in the Red Boar. Your room is to the south, and other rooms line the hall. The lighting in the hallway is dim and drab like the wallpaper and carpeting.'
- step 1 cur=None action='north'  pred_dst='red boar inn' GT_dst=None
  - LLM reasoning: The observation begins with 'Red Boar Inn', which matches the canonical location 'red boar inn'.
  - obs: 'Red Boar Inn\nThis is the lobby of the Red Boar Inn, a place with the charm of a two-dollar flophouse. The Wharf Road is outside, to the north, and a stairway leads up to the rooms to the south.\nA desk clerk sits behind the counter.\nThe Weasel walks past you.'
- step 2 cur=None action='north'  pred_dst='wharf road (end of east/west road)' GT_dst=None
  - LLM reasoning: Observation heading matches 'Wharf Road' and describes it as the end of the road, which matches the canonical 'wharf road (end of east/west road)'.
  - obs: "Wharf Road\nThis is the end of the Wharf Road, an east/west, two-lane road that's the island's main street. Off to the east, small businesses line the south side of the street. A winding road starts to the southwest, heading toward the lighthouse. The Red Boar Inn is to the south."
- step 3 cur=None action='east'  pred_dst='wharf road (north to mcginty salvage)' GT_dst=None
  - LLM reasoning: Observation describes being on Wharf Road with McGinty Salvage to the south, matching the canonical location 'wharf road (north to mcginty salvage)'.
  - obs: 'Wharf Road\nYou are on the Wharf Road, with the McGinty Salvage office to the south. The ocean lies to the north, its brine smell strong and refreshing.\nYour throat starts to feel dry.'
- step 4 cur=None action='east'  pred_dst='wharf road (north to outfitters international former warehouse)' GT_dst=None
  - LLM reasoning: Observation mentions the wharf starting to the north and the former site of Outfitters International's warehouse to the south, matching the canonical room 'wharf road (north to outfitters international former warehouse)'.
  - obs: "Wharf Road\nYou are on the Wharf Road where the wharf starts up to the north. The former site of Outfitters International's warehouse fronts the south side of the road."
- step 5 cur=None action='east'  pred_dst='wharf road (east end of east/west road)' GT_dst=None
  - LLM reasoning: Observation describes the east/west Wharf Road with Outfitters International to the south, matching 'wharf road (east end of east/west road)' in the canonical list.
  - obs: 'Wharf Road\nThe east/west Wharf Road continues, with Outfitters International, a store which supplies ocean-going vessels, to the south.\nTo the west the Weasel comes into view from the south.'
- step 6 cur=None action='east'  pred_dst='wharf road (east end of east/west road)' GT_dst=None
  - LLM reasoning: Observation states 'You're on the east end of Wharf Road', matching the canonical location 'wharf road (east end of east/west road)'.
  - obs: "Wharf Road\nYou're on the east end of Wharf Road. The Ocean Road starts here and parallels the island's east edge, heading to the southeast. The Shanty is to the south.\nThe Weasel is to the west, heading toward the east."
- step 7 cur=None action='south'  pred_dst='the shanty' GT_dst=None
  - LLM reasoning: Observation begins with 'The Shanty', which matches a canonical location; action is a compass direction.
  - obs: 'The Shanty\nThis is The Shanty, a tavern which serves the locals. A wooden bar travels the length of the place, and behind it is a mirror so dirty, it offers no reflection. Round tables occupy the floor. The smell of cooking food permeates the place.\nA bartender behind the bar polishes glasses with a'
