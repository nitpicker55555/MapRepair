# spirit — per-edge error analysis

V3 walkthrough steps: 37
Predicted edges:      36
GT edges:             40

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 33 |
| HALLUCINATED_DST | 1 |
| HALLUCINATED_SRC | 1 |
| SPURIOUS_PAIR | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 33 |
| MISSED | 7 |

## Samples — SPURIOUS_PAIR

- step 23 walk_action='up' PRED: 'hallway (near your quarters)' --[up]--> 'hallway (second floor, west)'
  > obs: 'Hallway (second floor, west)\nYou are in a hallway on the second floor of the Monastery. The hallway continues east, while stairs lead downwards.'

## Samples — HALLUCINATED_DST

- step 5 walk_action='north' PRED: 'main hall' --[north]--> 'monastery chapel'
  > obs: 'Monastery Chapel\nYou are in the Monastery Chapel, the site of the recent appearance of the legendary Anabais. Signs of the recent visit include the slightly scorched walls, and the reek of brimstone. '

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='north'  pred_dst='hallway (near your quarters)' GT_dst=None
  - LLM reasoning: Observation explicitly names 'Hallway (near your quarters)', which matches a canonical location.
  - obs: 'Hallway (near your quarters)\nYou are in a hallway running west and east. The vaulted ceilings are a nice touch. Your quarters are to the south.'
- step 5 cur='main hall' action='north'  pred_dst='monastery chapel' GT_dst='monastery chapel(exit on the south side)'
  - obs: 'Monastery Chapel\nYou are in the Monastery Chapel, the site of the recent appearance of the legendary Anabais. Signs of the recent visit include the slightly scorched walls, and the reek of brimstone. Wood pews are arranged in a circle around the central altar. The exit is south.\nBrother Joseph is he'
