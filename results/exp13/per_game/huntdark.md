# huntdark — per-edge error analysis

V3 walkthrough steps: 4
Predicted edges:      3
GT edges:             6

## Pred buckets
| bucket | n |
|--------|--:|
| SPURIOUS_PAIR | 3 |

## GT buckets
| bucket | n |
|--------|--:|
| MISSED | 6 |

## Samples — SPURIOUS_PAIR

- step 1 walk_action='up' PRED: 'tight crawl' --[up]--> 'cramped cavern'
  > obs: 'Cramped Cavern\nYou are in a cramped cavern, with a high roof. Stalactites hang low overhead, and you can hear water rushing in the distance. The ground is worn into polished flowing curves. A pit lead'
- step 2 walk_action='up' PRED: 'cramped cavern' --[up]--> 'small cave'
  > obs: 'Small Cave\nThis is a small cave. Ropy pillars rise from floor to roof, and water trickles audibly, somewhere nearby. The floor is crusted with wet sand. A low tunnel leads ahead, a path goes right, an'
- step 3 walk_action='up' PRED: 'small cave' --[up]--> 'base of canyon'
  > obs: 'The passage opens abruptly into vast space.\n\nBase of Canyon\nYou are in an alcove, a side chamber at the base of a vaulting canyon. The walls spread upward around you. And the distant roof is hung with'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='down'  pred_dst='tight crawl' GT_dst=None
  - LLM reasoning: The observation explicitly names 'Tight Crawl', which matches a canonical location, after the 'down' action.
  - obs: "You kneel, stiffly, and poke your head into the opening. It's tight, and it looks like it gets tighter ahead. No help for it -- you crawl inside.\n\nTight Crawl\nYou're on your hands and knees in a throat of stone. The roof dips even lower ahead. If you advance, it'll be squirming forwards on your bell"
