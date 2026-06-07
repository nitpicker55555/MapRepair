# pentari — per-edge error analysis

V3 walkthrough steps: 22
Predicted edges:      19
GT edges:             26

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 13 |
| SPURIOUS_PAIR | 3 |
| HALLUCINATED_DST | 1 |
| HALLUCINATED_SRC | 1 |
| SELF_LOOP | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 13 |
| MISSED | 13 |

## Samples — SPURIOUS_PAIR

- step 3 walk_action='east' PRED: 'the charlie company barracks (in the pentarian transporter)' --[east]--> 'library'
  > obs: "Library\nThis is a vast chamber lined floor to ceiling with bookcases filled to capacity with books.  You haven't seen such a collection of knowledge since your academy days.  A cursory glance at some "
- step 4 walk_action='south' PRED: 'library' --[south]--> 'castle entrance'
  > obs: "Castle Entrance\nThis large open area stands just inside the castle's gate and would be the place all visitors would first go before being allowed entrance to the castle proper.  A guard station, long "
- step 21 walk_action='enter' PRED: 'castle' --[enter]--> 'the main hall, by the fireplace (in the large fireplace)'
  > obs: 'The Main Hall, by the Fireplace (in the Large Fireplace)\n\nYou can see a dirty scroll here.'

## Samples — HALLUCINATED_DST

- step 7 walk_action='north' PRED: 'main hall' --[north]--> 'by the fireplace'
  > obs: 'Main Hall, by the Fireplace\nYou are at the northern end of the main hall taking in the panorama of what was once a majestic area where perhaps treaties were signed, strategies laid and plans foiled.\n\n'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='north'  pred_dst='charlie company compound' GT_dst=None
  - LLM reasoning: The observation begins with 'Charlie Company Compound', which matches a canonical location, indicating arrival after moving north.
  - obs: "Charlie Company Compound\nYour company compound consists of three large buildings bordering a large, open area where company formations precede the orders of the day.  To the west the Company's supply building can be seen while directly ahead of you lies your company's barracks which seem quieter tha"
- step 1 cur=None action='north'  pred_dst='charlie company barracks' GT_dst=None
  - LLM reasoning: The observation explicitly names 'Charlie Company Barracks', which matches the canonical list; action 'north' indicates movement.
  - obs: "Charlie Company Barracks\nIf the Army of Bostwin were to publish a new edition of the soldier's handbook, a picture of this barracks should be included as the example other companies should follow.  The floors are polished to an almost blinding shine while each of the bunks lining both sides of this "
- step 2 cur=None action='in'  pred_dst='the charlie company barracks (in the pentarian transporter)' GT_dst=None
  - LLM reasoning: The observation explicitly names 'The Charlie Company Barracks (in the Pentarian Transporter)', which matches a canonical location.
  - obs: 'The Charlie Company Barracks (in the Pentarian Transporter)'
- step 7 cur='main hall' action='north'  pred_dst='by the fireplace' GT_dst='main hall, by the fireplace'
  - obs: 'Main Hall, by the Fireplace\nYou are at the northern end of the main hall taking in the panorama of what was once a majestic area where perhaps treaties were signed, strategies laid and plans foiled.\n\nYou can see a Large Fireplace here.'
