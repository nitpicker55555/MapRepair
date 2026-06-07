# Conflict analysis: planetfall

- LLM edges: 22
- GT edges: 40
- Conflicts: 1
- Type distribution: {'direction': 1}
- Root-cause distribution: {'name_hallucination': 1}

## Conflict 1 — direction (name_hallucination)
- description: node 'escape pod' has multiple outgoing edges labelled 'out'
  - step None: escape pod --[out]--> safety web — dst_hallucinated
  - step 23: escape pod --[out]--> underwater — correct
