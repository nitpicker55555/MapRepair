# loose — per-edge error analysis

V3 walkthrough steps: 18
Predicted edges:      15
GT edges:             21

## Pred buckets
| bucket | n |
|--------|--:|
| CORRECT | 14 |
| SPURIOUS_PAIR | 1 |

## GT buckets
| bucket | n |
|--------|--:|
| RECALLED | 14 |
| MISSED | 7 |

## Samples — SPURIOUS_PAIR

- step 14 walk_action='northwest' PRED: 'rickety fence' --[northwest]--> 'schoolyard'
  > obs: 'Schoolyard\nBoth the playground and its adjacent schoolhouse to the south lie vacant and deserted. The bare, hard-packed rectangle of earth, despite a few faint chalk marks, shows no evidence that chil'

## Miss inspection (player at cur, took action, LLM said dst, GT says gt_dst)

- step 0 cur=None action='north'  pred_dst='forest' GT_dst=None
  - LLM reasoning: The observation explicitly states arrival in 'Forest', which matches a canonical location.
  - obs: 'Bravely, the egg calls after you: "My life is in your hands! I trust you will return shortly to rescue me!"\n\nForest\nThe sun has disappeared. Only a narrow north-south passage winds through the forest here; trees and gnarled undergrowth barricade all other exits. Chill air seeps up from the ground an'
- step 1 cur=None action='north'  pred_dst='deep forest' GT_dst=None
  - LLM reasoning: Observation begins with 'Deep Forest', which matches a canonical location. The action is 'north', so the player moved from an unknown location to 'deep forest'.
  - obs: 'Deep Forest\nWet branches slap your face, their icy needles stinging. Dense underbrush creeps in from all sides. With each step, mud oozes into your shoes. Insects drop from above and crawl through your hair; knuckles scrape against rough bark. The path hooks from south to southwest here, slithering '
- step 2 cur=None action='southwest'  pred_dst='country lane' GT_dst=None
  - LLM reasoning: The observation begins with 'Country Lane', which matches the canonical location 'country lane', indicating arrival there after moving southwest.
  - obs: 'Country Lane\nThe footpath snaking down from the northeastern forest has widened here into a country lane. Gravel crunches beneath your feet and a warm breeze riffles your hair as you walk in the sun. Some distance to the west, a few small cottages flock together like children telling secrets. Smoke '
