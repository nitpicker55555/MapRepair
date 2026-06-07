# lostpig — per-edge error analysis

V3 walkthrough steps: 7
Predicted edges:      4
GT edges:             11

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 2 |
| WRONG_SRC_UNKNOWN | 1 |
| WRONG_DST | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 9 |
| RECALLED | 2 |

## Samples — WRONG_DST

- step 6 walk_action='east' PRED: 'table room' --[east]--> 'shelf room'  (GT here: gnome room)
  > obs: 'Shelf Room\nThere lots of shelfs in this room. Them on every wall. This room probably have lots and lots of thing in it before. But shelfs all empty now. Grunk not see any thing there at all. Maybe Gru'

## Samples — WRONG_SRC_UNKNOWN

- step 2 walk_action='north' PRED: '<unknown>' --[north]--> 'statue room'
  > obs: 'Pig see Grunk leave. But not seem to care much.\n\nStatue Room\nThis room have two big picture that face each other. West wall have picture on it and east wall have different picture on it. Grunk think b'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='northeast'  pred_dst='hole' GT_dst=None
  - LLM reasoning: Observation describes arrival at the bottom of a deep, dark hole, matching the canonical room 'hole'.
  - obs: 'If that where pig go, that where Grunk go. It hard to see where Grunk going to, because there so much dark and leaves and branches here. Then suddenly, there not ground here any more!\n\nOof! Ow! Umf! Ugh!\n\nGrunk fall down deep hole! Drop torch. Torch go out when Grunk land on it, so now it dark, too.'
- step 1 cur=None action='east'  pred_dst='fountain room' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Fountain Room', which matches the canonical list; action is 'east', a compass direction.
  - obs: 'Fountain Room\nAll wall in this room glow! It bright, just like day time. Except that instead of sun, it wall that glow. And instead of grass and tree, it square room with four wall. And instead of outside, it all under ground. But beside those thing, it just like it! Tunnel keep going out to east an'
- step 5 cur='table room' action='east'  pred_dst='' GT_dst='gnome room'
  - LLM reasoning: The observation gives the room name as 'Closet', which is not in the canonical list. The canonical destination is unknown from the list, so dst_node and current_location are set to null.
  - obs: 'Closet\nIt dark. Grunk see lots of shadow. Grunk see doorway to east and west, too. But mostly shadow.\n\nChhhkkkrrcht! What that strange noise?'
- step 6 cur='table room' action='east'  pred_dst='shelf room' GT_dst='gnome room'
  - obs: 'Shelf Room\nThere lots of shelfs in this room. Them on every wall. This room probably have lots and lots of thing in it before. But shelfs all empty now. Grunk not see any thing there at all. Maybe Grunk see more if Grunk have torch that on fire. But probably not, because lots of light come from nort'
