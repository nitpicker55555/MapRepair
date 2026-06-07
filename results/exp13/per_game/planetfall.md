# planetfall — per-edge error analysis

V3 walkthrough steps: 29
Predicted edges:      27
GT edges:             40

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 26 |
| WRONG_DST | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 26 |
| MISSED | 14 |

## Samples — WRONG_DST

- step 25 walk_action='north' PRED: 'mess corridor' --[north]--> 'admin corridor'  (GT here: storage west)
  > obs: 'Admin Corridor\nThe hallway, in fact the entire building, has been rent apart here, presumably by seismic upheaval. You can see the sky through the severed roof above, and the ground is thick with rubb'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='west'  pred_dst='escape pod' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Escape Pod', matching canonical location; action is compass direction.
  - obs: "Escape Pod\nThis is one of the Feinstein's primary escape pods, for use in extreme emergencies. A mass of safety webbing, large enough to hold several dozen people, fills half the pod. The controls are entirely automated. The bulkhead leading out is open.\nThe ship shakes again. You hear, from close b"
- step 1 cur=None action='out'  pred_dst='underwater' GT_dst=None
  - LLM reasoning: Observation begins with 'Underwater', which matches the canonical location 'underwater'.
  - obs: 'Underwater\nYou are momentarily disoriented as you enter the turbulent waters. Currents buffet you against the sharp rocks of an underwater cliff. A dim light filters down from above.'
- step 25 cur='mess corridor' action='north'  pred_dst='admin corridor' GT_dst='storage west'
  - obs: 'Admin Corridor\nThe hallway, in fact the entire building, has been rent apart here, presumably by seismic upheaval. You can see the sky through the severed roof above, and the ground is thick with rubble. To the north is a gaping rift, at least eight meters across and thirty meters deep. A wide doorw'
