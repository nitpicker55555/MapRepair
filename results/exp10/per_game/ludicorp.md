# Conflict analysis: ludicorp

- LLM edges: 22
- GT edges: 43
- Conflicts: 2
- Type distribution: {'topology': 2}
- Root-cause distribution: {'overlap_mixed': 1, 'name_hallucination_caused_overlap': 1}

## Conflict 1 — topology (overlap_mixed)
- description: position (-1, 0, 0) occupied by multiple rooms ['corner of the carpark', 'maintanence stairwell (top)']
  - step 3: the main carpark --[southeast]--> corner of the carpark — correct
  - step 6: corner of the carpark --[in]--> car — swapped_src_dst (LLM: corner of the carpark--[in]-->car but GT has car--[out]-->corner of the carpark)
  - step 37: some bushes --[south]--> maintanence stairwell (top) — correct
  - step 38: maintanence stairwell (top) --[down]--> maintanence stairwell (bottom) — correct

## Conflict 2 — topology (name_hallucination_caused_overlap)
- description: position (-4, 2, 0) occupied by multiple rooms ['bike shed', 'storeroom']
  - step 53: west of office --[west]--> bike shed — correct
  - step 64: delivery bay --[north]--> storeroom — correct
  - step 67: storeroom --[north]--> west end — dst_hallucinated
