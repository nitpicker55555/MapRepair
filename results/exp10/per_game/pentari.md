# Conflict analysis: pentari

- LLM edges: 16
- GT edges: 26
- Conflicts: 5
- Type distribution: {'direction': 2, 'topology': 3}
- Root-cause distribution: {'real_vs_hallucinated': 2, 'name_hallucination_caused_overlap': 2, 'overlap_mixed': 1}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'castle entrance' has multiple outgoing edges labelled 'north'
  - step None: castle entrance --[north]--> library — hallucinated_edge
  - step 7: castle entrance --[north]--> castle — correct

## Conflict 2 — direction (real_vs_hallucinated)
- description: node 'armory' has multiple outgoing edges labelled 'east'
  - step None: armory --[east]--> main hall — correct
  - step None: armory --[east]--> main hall, by the fireplace — hallucinated_edge

## Conflict 3 — topology (name_hallucination_caused_overlap)
- description: position (1, 0, 0) occupied by multiple rooms ['main hall', 'main hall, by the fireplace']
  - step 8: castle --[north]--> main hall — correct
  - step 9: main hall --[north]--> by the fireplace — dst_hallucinated
  - step 11: main hall --[east]--> spiral stairway — correct
  - step 19: main hall --[west]--> armory — correct
  - step 36: by the fireplace (in the large fireplace) --[south]--> main hall, by the fireplace — src_hallucinated
  - step 40: main hall, by the fireplace --[west]--> armory — hallucinated_edge

## Conflict 4 — topology (name_hallucination_caused_overlap)
- description: position (1, 1, 0) occupied by multiple rooms ['by the fireplace', 'by the fireplace (in the large fireplace)']
  - step 9: main hall --[north]--> by the fireplace — dst_hallucinated
  - step 34: by the fireplace --[enter]--> by the fireplace (in the large fireplace) — both_names_hallucinated
  - step 36: by the fireplace (in the large fireplace) --[south]--> main hall, by the fireplace — src_hallucinated

## Conflict 5 — topology (overlap_mixed)
- description: position (1, -1, 0) occupied by multiple rooms ['castle', 'library']
  - step 7: castle entrance --[north]--> castle — correct
  - step 8: castle --[north]--> main hall — correct
  - step 4: the charlie company barracks (in the pentarian transporter) --[east]--> library — hallucinated_edge
  - step 5: library --[south]--> castle entrance — hallucinated_edge
