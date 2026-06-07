# afflicted — per-edge error analysis

V3 walkthrough steps: 19
Predicted edges:      18
GT edges:             20

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 8 |
| WRONG_DST | 2 |
| HALLUCINATED_DST | 2 |
| HALLUCINATED_SRC | 2 |
| SPURIOUS_PAIR | 2 |
| WRONG_SRC_UNKNOWN | 1 |
| SELF_LOOP | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 12 |
| RECALLED | 8 |

## Samples — WRONG_DST

- step 8 walk_action='north' PRED: 'galley kitchen' --[north]--> 'refrigerator'  (GT here: winding hall)
  > obs: "Darkness\nIt is pitch dark, and you can't see a thing."
- step 17 walk_action='south' PRED: 'dish room' --[south]--> '19th street'  (GT here: winding hall)
  > obs: '19th Street\nAn aluminum placard is fixed to the front door.'

## Samples — WRONG_SRC_UNKNOWN

- step 1 walk_action='west' PRED: '<unknown>' --[west]--> '19th street'
  > obs: '19th Street\nAn aluminum placard is fixed to the front door.\n\nAngela arrives from the bar and unlocks the door.'

## Samples — SPURIOUS_PAIR

- step 15 walk_action='down' PRED: 'dining room' --[down]--> 'crypt'
  > obs: '(first opening the drain cover)\n\nCrypt\nThe edges and corners of this brown room are rounded by accumulated cobweb nets, making it look more like a small cave than an urban underground.  Moldy pylons a'
- step 18 walk_action='up' PRED: '19th street' --[up]--> "nikolai's office"
  > obs: 'Nikolai pursues you up the stairs, huffing and wheezing from the effort.  You double your speed just in time to escape his reach.  Nikolai is left stranded and out of breath back on the fifth step.\n\nN'

## Samples — HALLUCINATED_DST

- step 10 walk_action='east' PRED: 'refrigerator' --[east]--> 'darkness'
  > obs: 'Darkness'
- step 12 walk_action='south' PRED: 'dish room' --[south]--> 'darkness'
  > obs: 'Darkness'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='east'  pred_dst='19th street east' GT_dst=None
  - LLM reasoning: The observation begins with '19th Street East', which matches a canonical location, indicating arrival after moving east.
  - obs: "19th Street East\nA street-front window looks into Nikolai's bar.  If you want into the bar, you'll have to go back to the restaurant entrance, which is west of here.  There are no cars on this end of 19th Street.  The sidewalk is uneven.  A curb-side storm gutter empties here into a grated storm dra"
- step 8 cur='galley kitchen' action='north'  pred_dst='refrigerator' GT_dst='winding hall'
  - obs: "Darkness\nIt is pitch dark, and you can't see a thing."
- step 10 cur='refrigerator' action='east'  pred_dst='darkness' GT_dst='winding hall'
  - obs: 'Darkness'
- step 12 cur='dish room' action='south'  pred_dst='darkness' GT_dst='winding hall'
  - obs: 'Darkness'
- step 17 cur='dish room' action='south'  pred_dst='19th street' GT_dst='winding hall'
  - obs: '19th Street\nAn aluminum placard is fixed to the front door.'
