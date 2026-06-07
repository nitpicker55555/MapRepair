# infidel — per-edge error analysis

V3 walkthrough steps: 29
Predicted edges:      27
GT edges:             48

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 26 |
| SPURIOUS_PAIR | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 26 |
| MISSED | 22 |

## Samples — SPURIOUS_PAIR

- step 10 walk_action='west' PRED: 'your tent' --[west]--> 'near the nile'
  > obs: 'Near the Nile\nYou are on an east/west path on the north side of the encampment. A path to the south starts here, and you can see the riverbank clearly to the west. A warm, light breeze reaches your fa'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='outside your tent' GT_dst=None
  - LLM reasoning: Observation heading matches canonical room 'outside your tent', confirming arrival after moving south.
  - obs: 'Outside Your Tent\nYou are on an east/west path on the north side of the encampment. To the south you can see a firepit and to the north is the entrance to your tent. Everything is oddly quiet, unsettling, creating a feeling of floating anxiety. The stillness seems to enhance the eerie quality of the'
- step 1 cur=None action='southwest'  pred_dst='outside supply tent' GT_dst=None
  - LLM reasoning: The observation begins with 'Outside Supply Tent', which matches the canonical location 'outside supply tent'.
  - obs: "Outside Supply Tent\nYou're on a north/south path on the west edge of the encampment. Directly to the west is the supply tent, its flaps open, still in the hot, quiet air. To the east you can see the central firepit, a reminder of your being alone. A row of thickets, impossible to make any progress t"
