# seastalker — per-edge error analysis

V3 walkthrough steps: 10
Predicted edges:      8
GT edges:             12

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 7 |
| SPURIOUS_PAIR | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 7 |
| MISSED | 5 |

## Samples — SPURIOUS_PAIR

- step 9 walk_action='exit' PRED: 'scimitar' --[exit]--> 'north walkway'
  > obs: "There's a sharp pain in your right arm! A flood of wet warmth and a spreading red stain mean you've seriously gashed your arm on a sharp metal corner!\n(SCIMITAR)\nTip is sitting in the back seat, readi"

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='north'  pred_dst='north part' GT_dst=None
  - LLM reasoning: Observation explicitly states arrival in 'north part', which matches a canonical location.
  - obs: "(north part)\nYou're in the north part of your lab. A Computestor stands here to help test your inventions.\nTip is off to the south."
- step 1 cur=None action='east'  pred_dst='east part' GT_dst=None
  - LLM reasoning: Observation begins with '(east part)', matching a canonical location, so the player moved to 'east part'.
  - obs: "(east part)\nYou're in the east part of your lab. A doorway leads out past the Electrical Panel through a short corridor to the office of your lab assistant, Sharon Kemp. A Microwave Security Scanner stands against the wall. An intercom sits on your desk.\nTip is off to the west."
