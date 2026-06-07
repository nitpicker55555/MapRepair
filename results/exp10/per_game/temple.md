# Conflict analysis: temple

- LLM edges: 18
- GT edges: 32
- Conflicts: 2
- Type distribution: {'direction': 1, 'topology': 1}
- Root-cause distribution: {'name_hallucination': 1, 'name_hallucination_caused_overlap': 1}

## Conflict 1 — direction (name_hallucination)
- description: node 'a dark hallway' has multiple outgoing edges labelled 'south'
  - step None: a dark hallway --[south]--> road (east/west) — correct
  - step 50: a dark hallway --[south]--> road — dst_hallucinated

## Conflict 2 — topology (name_hallucination_caused_overlap)
- description: position (0, -1, 0) occupied by multiple rooms ['road', 'road (east/west)']
  - step 50: a dark hallway --[south]--> road — dst_hallucinated
  - step 51: road --[west]--> crossroads — src_hallucinated
  - step 33: public square --[west]--> road (east/west) — correct
  - step 35: road (east/west) --[north]--> a dark hallway — correct
