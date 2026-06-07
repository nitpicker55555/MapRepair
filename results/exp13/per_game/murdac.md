# murdac — per-edge error analysis

V3 walkthrough steps: 39
Predicted edges:      31
GT edges:             48

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 23 |
| WRONG_DST | 4 |
| SPURIOUS_PAIR | 2 |
| SELF_LOOP | 2 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 25 |
| RECALLED | 23 |

## Samples — WRONG_DST

- step 28 walk_action='east' PRED: 'high tunnel (east/west, path to west)' --[east]--> 'wooden plank in east/west tunnel'  (GT here: high tunnel (east/west, alcove off to north))
  > obs: 'You are in the east-west tunnel by the alcove, with the wiring\nto your east.\nThere is a plank lying across the wires'
- step 30 walk_action='east' PRED: 'wooden plank in east/west tunnel' --[east]--> "mad scientist's laboratory"  (GT here: high tunnel (to the west of lab))
  > obs: 'You are in the tunnel between the laboratory and the wiring.\nThere is a plank lying across the wires'
- step 35 walk_action='west' PRED: 'entrance hall to haunted house' --[west]--> 'small bedroom in haunted house'  (GT here: complicated junction of passages)
  > obs: 'A large item of furniture flies across the room, hurled by\na poltergeist. With the aid of the pillow you fend it off,\nand it narrowly misses your face.\nYou are in a small bedroom in the haunted house.'
- step 38 walk_action='south' PRED: 'kitchen of haunted house' --[south]--> 'living room of haunted house'  (GT here: study of haunted house)
  > obs: 'A large item of furniture flies across the room, hurled by\na poltergeist. With the aid of the pillow you fend it off,\nand it narrowly misses your face.\nYou are in the living-room (!) of the haunted ho'

## Samples — SPURIOUS_PAIR

- step 9 walk_action='west' PRED: 'flower garden' --[west]--> 'old untended grave'
  > obs: 'You are at an old, untended grave. The tombstone is worn but\nyou can make out some of the writing on it. There are paths\nto the east and north'
- step 37 walk_action='north' PRED: 'small bedroom in haunted house' --[north]--> 'kitchen of haunted house'
  > obs: 'A large item of furniture flies across the room, hurled by\na poltergeist. With the aid of the pillow you fend it off,\nand it narrowly misses your face.\nYou are in the kitchen of the haunted house.\nThe'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='north'  pred_dst='brick wall (destroyed at some point)' GT_dst=None
  - LLM reasoning: Observation describes arriving at a brick wall with a gap, matching the canonical location 'brick wall (destroyed at some point)'.
  - obs: 'The south-north path ends at a nearly-completed brick wall.\nThere is a still a gap through which you can pass.\nTwo ogres here are busily engaged in building activities.\nThey take no notice of you'
- step 1 cur=None action='north'  pred_dst='alley (north/south)' GT_dst=None
  - LLM reasoning: The observation describes a long north-south alley between high walls, matching the canonical location 'alley (north/south)'.
  - obs: 'The ogres finish the wall behind you, cutting off your retreat.\nYou are in a long north-south alley that runs between two\nextremely high sheer walls.\nThe way south is blocked by a newly-completed brick wall'
- step 2 cur=None action='north'  pred_dst='alley (north end)' GT_dst=None
  - LLM reasoning: Observation states 'You are at the north end of the alley', which matches the canonical location 'alley (north end)'.
  - obs: 'You are at the north end of the alley. It is impossible to\nproceed further as there is a very high wall to the north.\nThere is an antique shawm here'
- step 3 cur=None action='south'  pred_dst='alley (north/south)' GT_dst=None
  - LLM reasoning: Observation states 'You are in the alley', matching the canonical 'alley (north/south)'.
  - obs: 'You are in the alley.\nThe way south is blocked by a newly-completed brick wall'
- step 4 cur=None action='south'  pred_dst='brick wall (destroyed at some point)' GT_dst=None
  - LLM reasoning: Observation describes passing through the ruins of a brick wall into the garden; the canonical location matching this is 'brick wall (destroyed at some point)'.
  - obs: 'You are on a north-south path which passes through the ruins of\na brick wall into the garden'
- step 5 cur=None action='south'  pred_dst='vegetable garden' GT_dst=None
  - LLM reasoning: Observation explicitly states arrival in 'vegetable garden', which matches the canonical location.
  - obs: 'You are in the vegetable garden'
- step 6 cur=None action='south'  pred_dst='shrubbery' GT_dst=None
  - LLM reasoning: Observation explicitly states arrival in 'shrubbery', which matches the canonical location list.
  - obs: 'You are in a shrubbery. There are paths to the north and east.\nThere is a long plank here'
- step 19 cur='large quadrangular cellar' action='west'  pred_dst='wooden plank in east/west tunnel' GT_dst="mad scientist's laboratory"
  - obs: 'You are standing on a wooden plank which is resting on\nsome wires in an east-west tunnel'
