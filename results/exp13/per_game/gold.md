# gold — per-edge error analysis

V3 walkthrough steps: 22
Predicted edges:      18
GT edges:             22

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 16 |
| SPURIOUS_PAIR | 1 |
| WRONG_DST | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 16 |
| MISSED | 6 |

## Samples — WRONG_DST

- step 14 walk_action='north' PRED: 'in the garden' --[north]--> 'in the pantry'  (GT here: in a small meadow)
  > obs: "In the pantry\nI'm in the pantry, a small claustrophobic room which smells faintly of cinammon, honey and the mouse droppings that are liberally sprinkled across the tiled floor. There is an upright fr"

## Samples — SPURIOUS_PAIR

- step 12 walk_action='in' PRED: 'in the enchanted forest' --[in]--> 'in the potting shed'
  > obs: "In the potting shed\nI'm in the potting shed, a ramshackle structure that was obviously cobbled together in five minutes some idle Sunday afternoon.  I didn't know bears had green fingers. In fact I di"

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='south'  pred_dst='on a path in the forest' GT_dst=None
  - LLM reasoning: The observation describes being on a path in the forest, which matches the canonical location 'on a path in the forest'.
  - obs: "On a path in the forest\nI'm on a path winding through the Enchanted Forest. The path winds its way from the north towards the west. In every other direction there are dark, sinister looking trees that wave their branches at me in an anthropomorphic kind of way. Ooh, I'm soooo scared!"
- step 1 cur=None action='west'  pred_dst='at a fork in the path' GT_dst=None
  - LLM reasoning: The observation explicitly states arrival at 'a fork in the path', which matches a canonical location.
  - obs: "As I head off along the path a pathetic, mangey looking wolf jumps out from behind a bush and slobbers at me in what he probably hopes is a menacing way.\n   'I am the Big Bad Wolf!' says he. Then, after an awkward pause he adds 'well, what are you waiting for? Aren't you going to ask me the way to G"
- step 2 cur=None action='north'  pred_dst='in a clearing in the forest' GT_dst=None
  - LLM reasoning: Observation begins with 'In a clearing in the forest', which matches the canonical location name.
  - obs: "In a clearing in the forest\nI'm in a clearing. There is a teeny tiny cottage here complete with a thatched roof and leaded windows. It's just like something out of a fairy tale, funnily enough, except for the large hand-painted sign designed to deter trespassers. Just to the north is the front porch"
- step 3 cur=None action='north'  pred_dst='on the porch' GT_dst=None
  - LLM reasoning: Observation begins with 'On the porch', which matches the canonical location 'on the porch'.
  - obs: "On the porch\nI'm standing on the porch of the Three Bears' cottage. It would be quite pretty in a rustic sort of way had someone not seen fit to paint it gold. The front door is closed.  Just south of me is the clearing in the forest.\nA battered old dustbin sits to one side of the porch.\n\nI can also"
- step 14 cur='in the garden' action='north'  pred_dst='in the pantry' GT_dst='in a small meadow'
  - obs: "In the pantry\nI'm in the pantry, a small claustrophobic room which smells faintly of cinammon, honey and the mouse droppings that are liberally sprinkled across the tiled floor. There is an upright freezer in the corner and some sturdy shelves stacked high with box after box of McScottie's porridge "
