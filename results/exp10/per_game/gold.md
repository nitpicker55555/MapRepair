# Conflict analysis: gold

- LLM edges: 14
- GT edges: 22
- Conflicts: 4
- Type distribution: {'direction': 2, 'topology': 2}
- Root-cause distribution: {'name_hallucination': 2, 'name_hallucination_caused_overlap': 2}

## Conflict 1 — direction (name_hallucination)
- description: node 'in the garden' has multiple outgoing edges labelled 'north'
  - step 14: in the garden --[north]--> in a small meadow — correct
  - step 45: in the garden --[north]--> on the roof — dst_hallucinated

## Conflict 2 — direction (name_hallucination)
- description: node 'in the kitchen' has multiple outgoing edges labelled 'north'
  - step None: in the kitchen --[north]--> at the top of the beanstalk — dst_hallucinated
  - step 61: in the kitchen --[north]--> in the pantry — correct

## Conflict 3 — topology (name_hallucination_caused_overlap)
- description: position (-1, 2, 0) occupied by multiple rooms ['in a small meadow', 'on the roof']
  - step 14: in the garden --[north]--> in a small meadow — correct
  - step 45: in the garden --[north]--> on the roof — dst_hallucinated

## Conflict 4 — topology (name_hallucination_caused_overlap)
- description: position (-1, 1, 1) occupied by multiple rooms ['at the top of the beanstalk', 'in the pantry']
  - step 40: in the garden --[up]--> at the top of the beanstalk — dst_hallucinated
  - step 50: at the top of the beanstalk --[south]--> in the kitchen — src_hallucinated
  - step 61: in the kitchen --[north]--> in the pantry — correct
