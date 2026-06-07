# Conflict analysis: huntdark

- LLM edges: 5
- GT edges: 6
- Conflicts: 3
- Type distribution: {'topology': 3}
- Root-cause distribution: {'name_hallucination_caused_overlap': 2, 'false_positive_overlap': 1}

## Conflict 1 — topology (name_hallucination_caused_overlap)
- description: position (0, 0, 0) occupied by multiple rooms ['base of canyon', 'bottom of pit', 'canyon wall', 'cramped cavern', 'largish dome']
  - step 49: cramped room --[up]--> base of canyon — correct
  - step 6: bottom of pit --[down]--> tight crawl — correct
  - step 60: canyon wall --[up]--> rising path — both_names_hallucinated
  - step 25: smallish cave --[up]--> cramped cavern — correct
  - step 42: largish dome --[up]--> small cave — correct

## Conflict 2 — topology (false_positive_overlap)
- description: position (0, 0, -1) occupied by multiple rooms ['cramped room', 'smallish cave', 'tight crawl']
  - step 49: cramped room --[up]--> base of canyon — correct
  - step 25: smallish cave --[up]--> cramped cavern — correct
  - step 6: bottom of pit --[down]--> tight crawl — correct

## Conflict 3 — topology (name_hallucination_caused_overlap)
- description: position (0, 0, 1) occupied by multiple rooms ['rising path', 'small cave']
  - step 60: canyon wall --[up]--> rising path — both_names_hallucinated
  - step 42: largish dome --[up]--> small cave — correct
