# jewel — per-edge error analysis

V3 walkthrough steps: 28
Predicted edges:      26
GT edges:             31

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 24 |
| WRONG_DST | 1 |
| SPURIOUS_PAIR | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 24 |
| MISSED | 7 |

## Samples — WRONG_DST

- step 17 walk_action='west' PRED: 'cool cavern' --[west]--> 'narrow passage'  (GT here: east-west passage)
  > obs: "Narrow Passage\nThe passage is quite tight here. You wish now that you hadn't eaten so much in the weeks preceding the expedition. A narrow crack opens to the west and a larger opening heads east. You "

## Samples — SPURIOUS_PAIR

- step 19 walk_action='up' PRED: 'gaseous geyser' --[up]--> 'middle shaft'
  > obs: "Middle Shaft\nYou can no longer see the ground beneath you. The wall you are clinging to is still climbable although you're not sure for how long."

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='west'  pred_dst='cool cavern' GT_dst=None
  - LLM reasoning: Observation names 'Cool Cavern', which matches a canonical location; action is 'west', so player arrived here.
  - obs: 'Cool Cavern\nThis cavern features a change in rock type from molten to granite. You can feel a cool breeze coming from the northwest tunnel and flowing to the east. In the dim light, you can also see passages heading west and southwest.'
- step 1 cur=None action='northwest'  pred_dst='mineralized corridor' GT_dst=None
  - LLM reasoning: The observation explicitly names 'Mineralized Corridor', which matches the canonical list, indicating arrival in this room after moving northwest.
  - obs: "Mineralized Corridor\nThe granite walls in this corridor are peppered with outcroppings of different minerals. The minerals protrude like jagged teeth from the cave's walls and you get chills down your spine as your glowing hands reflect off their surface. A gentle breeze flows from the west to the s"
- step 17 cur='cool cavern' action='west'  pred_dst='narrow passage' GT_dst='east-west passage'
  - obs: "Narrow Passage\nThe passage is quite tight here. You wish now that you hadn't eaten so much in the weeks preceding the expedition. A narrow crack opens to the west and a larger opening heads east. You can feel a cool breeze coming from the west."
