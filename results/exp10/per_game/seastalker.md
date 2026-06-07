# Conflict analysis: seastalker

- LLM edges: 8
- GT edges: 12
- Conflicts: 1
- Type distribution: {'topology': 1}
- Root-cause distribution: {'false_positive_overlap': 1}

## Conflict 1 — topology (false_positive_overlap)
- description: position (0, 0, 0) occupied by multiple rooms ['corridor', 'crawl space']
  - step 15: east part --[east]--> corridor — correct
  - step 18: corridor --[east]--> kemp's office — correct
  - step 31: crawl space --[exit]--> scimitar — correct
