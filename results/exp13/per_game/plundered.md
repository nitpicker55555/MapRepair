# plundered — per-edge error analysis

V3 walkthrough steps: 24
Predicted edges:      22
GT edges:             30

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 20 |
| SPURIOUS_PAIR | 2 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 20 |
| MISSED | 10 |

## Samples — SPURIOUS_PAIR

- step 11 walk_action='up' PRED: 'the ledge' --[up]--> 'poop deck'
  > obs: 'You heave yourself over the stern rail and sink breathless to the deck of the Helena Louise.\n\nPoop\n   From this platform deck you can see the whole of the two-masted ship, shadowy forms moving about h'
- step 21 walk_action='west' PRED: 'quarterdeck' --[west]--> 'beach'
  > obs: 'Beach\n   This is a tiny beach, at the base of vine-covered cliffs. To the east, the Helena Louise floats serenely on the lagoon.\n   Immediately north of you, a rocky staircase is hewn into the cliff.'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='north'  pred_dst='landing' GT_dst=None
  - LLM reasoning: The observation explicitly states arrival at 'Landing', which matches the canonical location 'landing'.
  - obs: "The large cupboard is not flush against the doorsill. You squeeze through the opening between cupboard and wall.\n\nLanding\n   You are on the landing of a steep stairway heading up and down. A large cupboard against the aft wall blocks the doorway to the captain's quarters."
- step 1 cur=None action='in'  pred_dst='sleeping cupboard' GT_dst=None
  - LLM reasoning: The observation begins with 'Sleeping Cupboard', which matches the canonical location 'sleeping cupboard'. The action 'in' leads directly to this room.
  - obs: 'Sleeping Cupboard\n   The bed is softer than it looks, plumped up with fresh straw. A soft, if somewhat cramped, bunk for the tall bulk of Jamison. The cupboard opens to the fore.\n   Kicked into a far corner is a pile of clothes.'
