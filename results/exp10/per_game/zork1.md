# Conflict analysis: zork1

- LLM edges: 19
- GT edges: 29
- Conflicts: 2
- Type distribution: {'direction': 1, 'topology': 1}
- Root-cause distribution: {'real_vs_hallucinated': 1, 'name_hallucination_caused_overlap': 1}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'forest path' has multiple outgoing edges labelled 'west'
  - step None: forest path --[west]--> temple — hallucinated_edge
  - step None: forest path --[west]--> forest — correct

## Conflict 2 — topology (name_hallucination_caused_overlap)
- description: position (0, 1, 0) occupied by multiple rooms ['forest', 'temple']
  - step 70: forest --[east]--> forest path — correct
  - step 37: torch room --[down]--> temple — correct
  - step 38: temple --[south]--> altar — correct
  - step 45: temple --[in]--> egyptian room — dst_hallucinated
  - step 48: temple --[east]--> forest path — hallucinated_edge
