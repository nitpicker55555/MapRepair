# Conflict analysis: partyfoul

- LLM edges: 6
- GT edges: 7
- Conflicts: 3
- Type distribution: {'direction': 2, 'topology': 1}
- Root-cause distribution: {'name_hallucination': 2, 'name_hallucination_caused_overlap': 1}

## Conflict 1 — direction (name_hallucination)
- description: node 'living room' has multiple outgoing edges labelled 'north'
  - step 9: living room --[north]--> den — correct
  - step None: living room --[north]--> darkness — dst_hallucinated

## Conflict 2 — direction (name_hallucination)
- description: node 'kitchen' has multiple outgoing edges labelled 'east'
  - step None: kitchen --[east]--> den — correct
  - step 36: kitchen --[east]--> darkness — dst_hallucinated

## Conflict 3 — topology (name_hallucination_caused_overlap)
- description: position (0, 0, 0) occupied by multiple rooms ['darkness', 'den']
  - step 36: kitchen --[east]--> darkness — dst_hallucinated
  - step 41: darkness --[south]--> living room — src_hallucinated
  - step 9: living room --[north]--> den — correct
  - step 12: den --[west]--> kitchen — correct
