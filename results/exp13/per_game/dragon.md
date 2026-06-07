# dragon — per-edge error analysis

V3 walkthrough steps: 32
Predicted edges:      31
GT edges:             42

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 28 |
| SPURIOUS_PAIR | 2 |
| WRONG_DST | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 28 |
| MISSED | 14 |

## Samples — WRONG_DST

- step 26 walk_action='north' PRED: 'wasteland by a castle' --[north]--> 'forbidding castle'  (GT here: castle forge)
  > obs: 'Forbidding Castle\nThe hulk of a massive, stone castle towers into the sky above you. The walls are covered in grey lichen and moss, and seem to have stood here since the beginning of time itself. Set '

## Samples — SPURIOUS_PAIR

- step 17 walk_action='south' PRED: 'inside a cottage' --[south]--> 'forest pathway'
  > obs: 'Forest Pathway\nGrey-green ferns brush your ankles as you move along the path, and the light filters dimly through the branches and leaves overhead. You hear the soft calls of woodland birds. The pathw'
- step 29 walk_action='south' PRED: 'top of a tower' --[south]--> 'lake in the forest'
  > obs: 'Lake in the Forest\nDragonflies hum and hover, and you hear the occasional splash of feeding fish. You have a feeling of tranquility instilled by the placid, lazy waters. There is a little path winding'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='abandoned inn' GT_dst=None
  - LLM reasoning: Observation describes the abandoned inn, matching the canonical location; action is 'south', so player arrived at 'abandoned inn'.
  - obs: "Abandoned Inn\nAn old, thatched building stands near the path. It seems to be an abandoned inn or tavern, as a faded sign reads 'Troll's Arms'. The walls are a dirty white and the paintwork is starting to peel and blister. The windows are covered in cobwebs and there is the faint odour of mildew and "
- step 26 cur='wasteland by a castle' action='north'  pred_dst='forbidding castle' GT_dst='castle forge'
  - obs: 'Forbidding Castle\nThe hulk of a massive, stone castle towers into the sky above you. The walls are covered in grey lichen and moss, and seem to have stood here since the beginning of time itself. Set into the wall to the north you can see a large oak door, dark and weathered. There is a keyhole surr'
