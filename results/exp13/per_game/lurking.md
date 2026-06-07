# lurking — per-edge error analysis

V3 walkthrough steps: 18
Predicted edges:      16
GT edges:             26

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 12 |
| SPURIOUS_PAIR | 2 |
| WRONG_DIRECTION | 2 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 14 |
| RECALLED | 12 |

## Samples — WRONG_DIRECTION

- step 10 walk_action='up' PRED: 'temporary basement' --[up]--> 'basement'
  > obs: 'It is pitch black.'
- step 11 walk_action='down' PRED: 'basement' --[down]--> 'temporary basement'
  > obs: 'Temporary Basement'

## Samples — SPURIOUS_PAIR

- step 4 walk_action='down' PRED: 'terminal room' --[down]--> 'basalt bowl'
  > obs: 'Basalt Bowl\nYou are at the bottom of a deeply cut, smooth basalt bowl. Dimly seen shapes crowd you on all sides. Ahead, in the focus of the movement, is a rock platform.'
- step 5 walk_action='south' PRED: 'basalt bowl' --[south]--> 'elevator'
  > obs: 'Elevator\nThis is a battered, rather dirty elevator. The fake wood walls are scratched and marred with graffiti. The elevator doors are open. To the right of the doors is an area with floor buttons (B '

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='second floor' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Second Floor', matching the canonical location list.
  - obs: 'Second Floor\nThis is the second floor of the Computer Center. An elevator and call buttons are on the south side of the hallway. A large, noisy room is to the north. Stairs also lead up and down, for the energetic. To the west a corridor leads into a smaller room.'
- step 1 cur=None action='west'  pred_dst='kitchen' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Kitchen', which matches the canonical location list.
  - obs: 'Kitchen\nThis is a filthy kitchen. The exit is to the east. On the wall near a counter are a refrigerator and a microwave.\n\nSitting on the kitchen counter is a package of Funny Bones.'
- step 10 cur='temporary basement' action='up'  pred_dst='basement' GT_dst='temporary lab'
  - obs: 'It is pitch black.'
