# temple — per-edge error analysis

V3 walkthrough steps: 22
Predicted edges:      20
GT edges:             32

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 17 |
| WRONG_DST | 2 |
| WRONG_DIRECTION | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 17 |
| MISSED | 15 |

## Samples — WRONG_DIRECTION

- step 12 walk_action='west' PRED: 'road (north/south)' --[west]--> 'crossroads'
  > obs: 'Crossroads\nThe road to the public square, way back east, forks into a north-south road here. To the west looms yet another of the stone buildings.\n\nCharles comes walking behind you.'

## Samples — WRONG_DST

- step 5 walk_action='south' PRED: 'study' --[south]--> 'before a dark tower'  (GT here: storage)
  > obs: 'Before A Dark Tower\nThis area in front of an old tower offers a nightmarish view over a monstrous tangle of dark stone buildings. Most buildings are elliptical, built of irregular-sized basalt blocks '
- step 11 walk_action='south' PRED: 'a dark hallway' --[south]--> 'road (north/south)'  (GT here: road (east/west))
  > obs: 'Road\n\nCharles comes walking behind you.'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='down'  pred_dst='study' GT_dst=None
  - LLM reasoning: The observation describes the player arriving in the 'Study', which matches a canonical location.
  - obs: 'The winding staircase leads to a spacious chamber inside the tower.\n\nStudy\nDust and spider webs cover all surfaces in this room in a soft silver gleam, blurring the outlines of the furnishings into fantastic, dreamy forms. The walls are lined with bookshelves and a lone desk lies at one corner. Clos'
- step 1 cur=None action='down'  pred_dst='hall' GT_dst=None
  - LLM reasoning: The observation begins with 'Hall', which matches the canonical location 'hall'. The action 'down' indicates movement to this room.
  - obs: 'Hall\nThis damp hall occupies the whole ground level of the tower. A staircase provides access to the higher levels, while a massive oak door, set in the southern wall, leads out.\n\nCharles comes walking behind you.'
- step 5 cur='study' action='south'  pred_dst='before a dark tower' GT_dst='storage'
  - obs: 'Before A Dark Tower\nThis area in front of an old tower offers a nightmarish view over a monstrous tangle of dark stone buildings. Most buildings are elliptical, built of irregular-sized basalt blocks of irregular size. None of them seem to have any doors or windows. There is a square further down to'
- step 11 cur='a dark hallway' action='south'  pred_dst='road (north/south)' GT_dst='road (east/west)'
  - obs: 'Road\n\nCharles comes walking behind you.'
