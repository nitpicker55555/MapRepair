# wishbringer — per-edge error analysis

V3 walkthrough steps: 28
Predicted edges:      26
GT edges:             37

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 23 |
| WRONG_DST | 1 |
| WRONG_DIRECTION | 1 |
| SPURIOUS_PAIR | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 23 |
| MISSED | 14 |

## Samples — WRONG_DIRECTION

- step 4 walk_action='north' PRED: 'open grave' --[north]--> 'spooky copse'
  > obs: "Spooky Copse\nA copse of willow trees makes this part of the cemetery look really spooky. Narrow lanes wander south and west.\n\nThere's an open grave nearby, freshly dug, with a tombstone erected next t"

## Samples — WRONG_DST

- step 3 walk_action='west' PRED: 'outside cemetery' --[west]--> 'open grave'  (GT here: creepy corner)
  > obs: 'The cemetery is a fearful place. Are you sure you want to go in there?\n\n(Please type YES or NO.)'

## Samples — SPURIOUS_PAIR

- step 23 walk_action='west' PRED: 'magick shoppe' --[west]--> 'north of bridge'
  > obs: 'North of Bridge\nThis is the north side of the Witchville Bridge. Sheer cliff walls rise abruptly to the north and west. A path wanders east along the shore of the river.\n\nA rickety toll gate has been '

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='post office' GT_dst=None
  - LLM reasoning: The observation describes the lobby of the Festeron Post Office, which matches the canonical location 'post office'.
  - obs: "Post Office\nThis is the lobby of the Festeron Post Office. The walls are lined with small, private mailboxes and wanted posters. A service counter runs along the entire length of the room.\n\nYour boss, Mr. Crisp, is behind the service counter reading other people's postcards.\n\nMr. Crisp hides the pos"
- step 1 cur=None action='north'  pred_dst='hilltop' GT_dst=None
  - LLM reasoning: Observation explicitly states arrival at 'Hilltop', which matches a canonical location.
  - obs: 'As you leave the Post Office, somebody slams and locks the door behind you.\n\nHilltop\n\nWhat next?'
- step 3 cur='outside cemetery' action='west'  pred_dst='open grave' GT_dst='creepy corner'
  - obs: 'The cemetery is a fearful place. Are you sure you want to go in there?\n\n(Please type YES or NO.)'
