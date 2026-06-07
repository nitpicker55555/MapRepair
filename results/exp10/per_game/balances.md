# Conflict analysis: balances

- LLM edges: 11
- GT edges: 13
- Conflicts: 25
- Type distribution: {'direction': 3, 'topology': 12, 'naming': 10}
- Root-cause distribution: {'real_vs_hallucinated': 1, 'name_hallucination': 5, 'name_hallucination_caused_overlap': 11, 'overlap_mixed': 1, 'real_name_corrupted_by_neighbour_error': 4, 'naming_collision_on_correct_subgraph': 3}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'cave mouth' has multiple outgoing edges labelled 'southeast'
  - step None: cave mouth --[southeast]--> edge of chasm — hallucinated_edge
  - step 42: cave mouth --[southeast]--> crest of hill — correct

## Conflict 2 — direction (name_hallucination)
- description: node 'cave mouth' has multiple outgoing edges labelled 'west'
  - step 30: cave mouth --[west]--> darkness — dst_hallucinated
  - step None: cave mouth --[west]--> inside cave — correct

## Conflict 3 — direction (name_hallucination)
- description: node 'outside temple' has multiple outgoing edges labelled 'west'
  - step None: outside temple --[west]--> on the beautiful red carpet — both_names_hallucinated
  - step 69: outside temple --[west]--> crest of hill — src_hallucinated

## Conflict 4 — topology (name_hallucination_caused_overlap)
- description: position (0, -2, 0) occupied by multiple rooms ['cave mouth', 'gorse bushes']
  - step 19: edge of chasm --[northwest]--> cave mouth — hallucinated_edge
  - step 41: inside cave --[east]--> cave mouth — correct
  - step 25: cave mouth --[north]--> gorse bushes — correct
  - step 30: cave mouth --[west]--> darkness — dst_hallucinated
  - step 42: cave mouth --[southeast]--> crest of hill — correct

## Conflict 5 — topology (name_hallucination_caused_overlap)
- description: position (0, 0, 0) occupied by multiple rooms ['cave mouth', 'gorse bushes']
  - step 19: edge of chasm --[northwest]--> cave mouth — hallucinated_edge
  - step 41: inside cave --[east]--> cave mouth — correct
  - step 25: cave mouth --[north]--> gorse bushes — correct
  - step 30: cave mouth --[west]--> darkness — dst_hallucinated
  - step 42: cave mouth --[southeast]--> crest of hill — correct

## Conflict 6 — topology (name_hallucination_caused_overlap)
- description: position (0, -1, 0) occupied by multiple rooms ['cave mouth', 'gorse bushes']
  - step 19: edge of chasm --[northwest]--> cave mouth — hallucinated_edge
  - step 41: inside cave --[east]--> cave mouth — correct
  - step 25: cave mouth --[north]--> gorse bushes — correct
  - step 30: cave mouth --[west]--> darkness — dst_hallucinated
  - step 42: cave mouth --[southeast]--> crest of hill — correct

## Conflict 7 — topology (overlap_mixed)
- description: position (1, -5, 0) occupied by multiple rooms ['edge of chasm', 'grasslands, near hut', 'pocket valley']
  - step 13: pocket valley --[north]--> edge of chasm — hallucinated_edge
  - step 43: crest of hill --[south]--> edge of chasm — correct
  - step 19: edge of chasm --[northwest]--> cave mouth — hallucinated_edge
  - step 6: ramshackle hut --[out]--> grasslands, near hut — correct
  - step 7: grasslands, near hut --[north]--> pocket valley — correct

## Conflict 8 — topology (name_hallucination_caused_overlap)
- description: position (1, -4, 0) occupied by multiple rooms ['crest of hill', 'edge of chasm', 'grasslands, near hut', 'on the beautiful red carpet', 'pocket valley']
  - step 42: cave mouth --[southeast]--> crest of hill — correct
  - step 69: outside temple --[west]--> crest of hill — src_hallucinated
  - step 43: crest of hill --[south]--> edge of chasm — correct
  - step 13: pocket valley --[north]--> edge of chasm — hallucinated_edge
  - step 19: edge of chasm --[northwest]--> cave mouth — hallucinated_edge
  - step 6: ramshackle hut --[out]--> grasslands, near hut — correct
  - step 7: grasslands, near hut --[north]--> pocket valley — correct
  - step 50: on the beautiful red carpet --[east]--> outside temple — both_names_hallucinated

## Conflict 9 — topology (name_hallucination_caused_overlap)
- description: position (1, -1, 0) occupied by multiple rooms ['crest of hill', 'edge of chasm', 'on the beautiful red carpet']
  - step 42: cave mouth --[southeast]--> crest of hill — correct
  - step 69: outside temple --[west]--> crest of hill — src_hallucinated
  - step 43: crest of hill --[south]--> edge of chasm — correct
  - step 13: pocket valley --[north]--> edge of chasm — hallucinated_edge
  - step 19: edge of chasm --[northwest]--> cave mouth — hallucinated_edge
  - step 50: on the beautiful red carpet --[east]--> outside temple — both_names_hallucinated

## Conflict 10 — topology (name_hallucination_caused_overlap)
- description: position (1, -3, 0) occupied by multiple rooms ['crest of hill', 'edge of chasm', 'grasslands, near hut', 'on the beautiful red carpet', 'pocket valley']
  - step 42: cave mouth --[southeast]--> crest of hill — correct
  - step 69: outside temple --[west]--> crest of hill — src_hallucinated
  - step 43: crest of hill --[south]--> edge of chasm — correct
  - step 13: pocket valley --[north]--> edge of chasm — hallucinated_edge
  - step 19: edge of chasm --[northwest]--> cave mouth — hallucinated_edge
  - step 6: ramshackle hut --[out]--> grasslands, near hut — correct
  - step 7: grasslands, near hut --[north]--> pocket valley — correct
  - step 50: on the beautiful red carpet --[east]--> outside temple — both_names_hallucinated

## Conflict 11 — topology (name_hallucination_caused_overlap)
- description: position (1, -2, 0) occupied by multiple rooms ['crest of hill', 'edge of chasm', 'on the beautiful red carpet', 'pocket valley']
  - step 42: cave mouth --[southeast]--> crest of hill — correct
  - step 69: outside temple --[west]--> crest of hill — src_hallucinated
  - step 43: crest of hill --[south]--> edge of chasm — correct
  - step 13: pocket valley --[north]--> edge of chasm — hallucinated_edge
  - step 19: edge of chasm --[northwest]--> cave mouth — hallucinated_edge
  - step 50: on the beautiful red carpet --[east]--> outside temple — both_names_hallucinated
  - step 7: grasslands, near hut --[north]--> pocket valley — correct

## Conflict 12 — topology (name_hallucination_caused_overlap)
- description: position (-1, -2, 0) occupied by multiple rooms ['darkness', 'inside cave']
  - step 30: cave mouth --[west]--> darkness — dst_hallucinated
  - step 41: inside cave --[east]--> cave mouth — correct

## Conflict 13 — topology (name_hallucination_caused_overlap)
- description: position (-1, 0, 0) occupied by multiple rooms ['darkness', 'inside cave']
  - step 30: cave mouth --[west]--> darkness — dst_hallucinated
  - step 41: inside cave --[east]--> cave mouth — correct

## Conflict 14 — topology (name_hallucination_caused_overlap)
- description: position (-1, -1, 0) occupied by multiple rooms ['darkness', 'inside cave']
  - step 30: cave mouth --[west]--> darkness — dst_hallucinated
  - step 41: inside cave --[east]--> cave mouth — correct

## Conflict 15 — topology (name_hallucination_caused_overlap)
- description: position (-1, -3, 0) occupied by multiple rooms ['darkness', 'inside cave']
  - step 30: cave mouth --[west]--> darkness — dst_hallucinated
  - step 41: inside cave --[east]--> cave mouth — correct

## Conflict 16 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'cave mouth' reachable at conflicting positions [(0, -3, 0), (0, -2, 0), (0, -1, 0), (0, 0, 0)]
  - step 19: edge of chasm --[northwest]--> cave mouth — hallucinated_edge
  - step 41: inside cave --[east]--> cave mouth — correct
  - step 25: cave mouth --[north]--> gorse bushes — correct
  - step 30: cave mouth --[west]--> darkness — dst_hallucinated
  - step 42: cave mouth --[southeast]--> crest of hill — correct

## Conflict 17 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'edge of chasm' reachable at conflicting positions [(1, -5, 0), (1, -4, 0), (1, -3, 0), (1, -2, 0), (1, -1, 0)]
  - step 13: pocket valley --[north]--> edge of chasm — hallucinated_edge
  - step 43: crest of hill --[south]--> edge of chasm — correct
  - step 19: edge of chasm --[northwest]--> cave mouth — hallucinated_edge

## Conflict 18 — naming (naming_collision_on_correct_subgraph)
- description: node 'gorse bushes' reachable at conflicting positions [(0, -2, 0), (0, -1, 0), (0, 0, 0), (0, 1, 0)]
  - step 25: cave mouth --[north]--> gorse bushes — correct

## Conflict 19 — naming (name_hallucination)
- description: node 'darkness' reachable at conflicting positions [(-1, -3, 0), (-1, -2, 0), (-1, -1, 0), (-1, 0, 0)]
  - step 30: cave mouth --[west]--> darkness — dst_hallucinated

## Conflict 20 — naming (naming_collision_on_correct_subgraph)
- description: node 'inside cave' reachable at conflicting positions [(-1, -3, 0), (-1, -2, 0), (-1, -1, 0), (-1, 0, 0)]
  - step 41: inside cave --[east]--> cave mouth — correct

## Conflict 21 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'crest of hill' reachable at conflicting positions [(1, -4, 0), (1, -3, 0), (1, -2, 0), (1, -1, 0), (1, 0, 0)]
  - step 42: cave mouth --[southeast]--> crest of hill — correct
  - step 69: outside temple --[west]--> crest of hill — src_hallucinated
  - step 43: crest of hill --[south]--> edge of chasm — correct

## Conflict 22 — naming (name_hallucination)
- description: node 'outside temple' reachable at conflicting positions [(2, -4, 0), (2, -3, 0), (2, -2, 0), (2, -1, 0)]
  - step 50: on the beautiful red carpet --[east]--> outside temple — both_names_hallucinated
  - step 69: outside temple --[west]--> crest of hill — src_hallucinated

## Conflict 23 — naming (name_hallucination)
- description: node 'on the beautiful red carpet' reachable at conflicting positions [(1, -4, 0), (1, -3, 0), (1, -2, 0), (1, -1, 0)]
  - step 50: on the beautiful red carpet --[east]--> outside temple — both_names_hallucinated

## Conflict 24 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'pocket valley' reachable at conflicting positions [(1, -5, 0), (1, -4, 0), (1, -3, 0), (1, -2, 0)]
  - step 7: grasslands, near hut --[north]--> pocket valley — correct
  - step 13: pocket valley --[north]--> edge of chasm — hallucinated_edge

## Conflict 25 — naming (naming_collision_on_correct_subgraph)
- description: node 'grasslands, near hut' reachable at conflicting positions [(1, -6, 0), (1, -5, 0), (1, -4, 0), (1, -3, 0)]
  - step 6: ramshackle hut --[out]--> grasslands, near hut — correct
  - step 7: grasslands, near hut --[north]--> pocket valley — correct
