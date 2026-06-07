# Conflict analysis: anchor

- LLM edges: 24
- GT edges: 42
- Conflicts: 38
- Type distribution: {'direction': 1, 'topology': 13, 'naming': 24}
- Root-cause distribution: {'real_vs_hallucinated': 1, 'overlap_mixed': 4, 'false_positive_overlap': 9, 'real_name_corrupted_by_neighbour_error': 2, 'naming_collision_on_correct_subgraph': 21, 'naming_mixed': 1}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'office' has multiple outgoing edges labelled 'east'
  - step None: office --[east]--> the lower edge of the window — hallucinated_edge
  - step 6: office --[east]--> file room — correct

## Conflict 2 — topology (overlap_mixed)
- description: position (-2, -2, 3) occupied by multiple rooms ['file room', 'the lower edge of the window']
  - step 6: office --[east]--> file room — correct
  - step 3: alley --[up]--> the lower edge of the window — hallucinated_edge
  - step 4: the lower edge of the window --[west]--> office — hallucinated_edge

## Conflict 3 — topology (overlap_mixed)
- description: position (0, 0, 1) occupied by multiple rooms ['file room', 'the lower edge of the window']
  - step 6: office --[east]--> file room — correct
  - step 3: alley --[up]--> the lower edge of the window — hallucinated_edge
  - step 4: the lower edge of the window --[west]--> office — hallucinated_edge

## Conflict 4 — topology (overlap_mixed)
- description: position (-3, -3, 4) occupied by multiple rooms ['file room', 'the lower edge of the window']
  - step 6: office --[east]--> file room — correct
  - step 3: alley --[up]--> the lower edge of the window — hallucinated_edge
  - step 4: the lower edge of the window --[west]--> office — hallucinated_edge

## Conflict 5 — topology (overlap_mixed)
- description: position (-1, -1, 2) occupied by multiple rooms ['file room', 'the lower edge of the window']
  - step 6: office --[east]--> file room — correct
  - step 3: alley --[up]--> the lower edge of the window — hallucinated_edge
  - step 4: the lower edge of the window --[west]--> office — hallucinated_edge

## Conflict 6 — topology (false_positive_overlap)
- description: position (-3, 0, 1) occupied by multiple rooms ['master bedroom', 'narrow street']
  - step 31: upstairs landing --[north]--> master bedroom — correct
  - step 34: master bedroom --[west]--> bathroom — correct
  - step 11: outside the real estate office --[west]--> narrow street — correct
  - step 12: narrow street --[west]--> junction — correct
  - step 20: narrow street --[south]--> whateley bridge — correct

## Conflict 7 — topology (false_positive_overlap)
- description: position (-4, -1, 2) occupied by multiple rooms ['master bedroom', 'narrow street']
  - step 31: upstairs landing --[north]--> master bedroom — correct
  - step 34: master bedroom --[west]--> bathroom — correct
  - step 11: outside the real estate office --[west]--> narrow street — correct
  - step 12: narrow street --[west]--> junction — correct
  - step 20: narrow street --[south]--> whateley bridge — correct

## Conflict 8 — topology (false_positive_overlap)
- description: position (-5, -2, 3) occupied by multiple rooms ['master bedroom', 'narrow street']
  - step 31: upstairs landing --[north]--> master bedroom — correct
  - step 34: master bedroom --[west]--> bathroom — correct
  - step 11: outside the real estate office --[west]--> narrow street — correct
  - step 12: narrow street --[west]--> junction — correct
  - step 20: narrow street --[south]--> whateley bridge — correct

## Conflict 9 — topology (false_positive_overlap)
- description: position (-5, -1, 2) occupied by multiple rooms ['bathroom', 'junction']
  - step 34: master bedroom --[west]--> bathroom — correct
  - step 12: narrow street --[west]--> junction — correct
  - step 13: junction --[northwest]--> university court — correct

## Conflict 10 — topology (false_positive_overlap)
- description: position (-6, -2, 3) occupied by multiple rooms ['bathroom', 'junction']
  - step 34: master bedroom --[west]--> bathroom — correct
  - step 12: narrow street --[west]--> junction — correct
  - step 13: junction --[northwest]--> university court — correct

## Conflict 11 — topology (false_positive_overlap)
- description: position (-4, 0, 1) occupied by multiple rooms ['bathroom', 'junction']
  - step 34: master bedroom --[west]--> bathroom — correct
  - step 12: narrow street --[west]--> junction — correct
  - step 13: junction --[northwest]--> university court — correct

## Conflict 12 — topology (false_positive_overlap)
- description: position (-4, -2, 2) occupied by multiple rooms ['upstairs landing', 'whateley bridge']
  - step 30: foyer --[up]--> upstairs landing — correct
  - step 31: upstairs landing --[north]--> master bedroom — correct
  - step 20: narrow street --[south]--> whateley bridge — correct
  - step 21: whateley bridge --[south]--> town square — correct

## Conflict 13 — topology (false_positive_overlap)
- description: position (-3, -1, 1) occupied by multiple rooms ['upstairs landing', 'whateley bridge']
  - step 30: foyer --[up]--> upstairs landing — correct
  - step 31: upstairs landing --[north]--> master bedroom — correct
  - step 20: narrow street --[south]--> whateley bridge — correct
  - step 21: whateley bridge --[south]--> town square — correct

## Conflict 14 — topology (false_positive_overlap)
- description: position (-5, -3, 3) occupied by multiple rooms ['upstairs landing', 'whateley bridge']
  - step 30: foyer --[up]--> upstairs landing — correct
  - step 31: upstairs landing --[north]--> master bedroom — correct
  - step 20: narrow street --[south]--> whateley bridge — correct
  - step 21: whateley bridge --[south]--> town square — correct

## Conflict 15 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'alley' reachable at conflicting positions [(-3, -3, 3), (-2, -2, 2), (-1, -1, 1), (0, 0, 0)]
  - step 1: outside the real estate office --[southeast]--> alley — correct
  - step 3: alley --[up]--> the lower edge of the window — hallucinated_edge

## Conflict 16 — naming (naming_collision_on_correct_subgraph)
- description: node 'outside the real estate office' reachable at conflicting positions [(-5, -3, 4), (-4, -2, 3), (-3, -1, 2), (-2, 0, 1), (-1, 1, 0)]
  - step 10: office --[west]--> outside the real estate office — correct
  - step 1: outside the real estate office --[southeast]--> alley — correct
  - step 11: outside the real estate office --[west]--> narrow street — correct

## Conflict 17 — naming (naming_mixed)
- description: node 'the lower edge of the window' reachable at conflicting positions [(-3, -3, 4), (-2, -2, 3), (-1, -1, 2), (0, 0, 1)]
  - step 3: alley --[up]--> the lower edge of the window — hallucinated_edge
  - step 4: the lower edge of the window --[west]--> office — hallucinated_edge

## Conflict 18 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'office' reachable at conflicting positions [(-4, -3, 4), (-3, -2, 3), (-2, -1, 2), (-1, 0, 1), (0, 1, 0)]
  - step 4: the lower edge of the window --[west]--> office — hallucinated_edge
  - step 6: office --[east]--> file room — correct
  - step 10: office --[west]--> outside the real estate office — correct

## Conflict 19 — naming (naming_collision_on_correct_subgraph)
- description: node 'file room' reachable at conflicting positions [(-3, -3, 4), (-2, -2, 3), (-1, -1, 2), (0, 0, 1)]
  - step 6: office --[east]--> file room — correct

## Conflict 20 — naming (naming_collision_on_correct_subgraph)
- description: node 'narrow street' reachable at conflicting positions [(-5, -2, 3), (-4, -1, 2), (-3, 0, 1), (-2, 1, 0)]
  - step 11: outside the real estate office --[west]--> narrow street — correct
  - step 12: narrow street --[west]--> junction — correct
  - step 20: narrow street --[south]--> whateley bridge — correct

## Conflict 21 — naming (naming_collision_on_correct_subgraph)
- description: node 'junction' reachable at conflicting positions [(-6, -2, 3), (-5, -1, 2), (-4, 0, 1), (-3, 1, 0)]
  - step 12: narrow street --[west]--> junction — correct
  - step 13: junction --[northwest]--> university court — correct

## Conflict 22 — naming (naming_collision_on_correct_subgraph)
- description: node 'whateley bridge' reachable at conflicting positions [(-5, -3, 3), (-4, -2, 2), (-3, -1, 1), (-2, 0, 0)]
  - step 20: narrow street --[south]--> whateley bridge — correct
  - step 21: whateley bridge --[south]--> town square — correct

## Conflict 23 — naming (naming_collision_on_correct_subgraph)
- description: node 'town square' reachable at conflicting positions [(-5, -4, 3), (-4, -3, 2), (-3, -2, 1), (-2, -1, 0)]
  - step 21: whateley bridge --[south]--> town square — correct
  - step 22: town square --[east]--> riverwalk — correct

## Conflict 24 — naming (naming_collision_on_correct_subgraph)
- description: node 'riverwalk' reachable at conflicting positions [(-4, -4, 3), (-3, -3, 2), (-2, -2, 1), (-1, -1, 0)]
  - step 22: town square --[east]--> riverwalk — correct
  - step 23: riverwalk --[south]--> chilly avenue — correct

## Conflict 25 — naming (naming_collision_on_correct_subgraph)
- description: node 'chilly avenue' reachable at conflicting positions [(-4, -5, 3), (-3, -4, 2), (-2, -3, 1), (-1, -2, 0)]
  - step 23: riverwalk --[south]--> chilly avenue — correct
  - step 24: chilly avenue --[southwest]--> scenic view — correct

## Conflict 26 — naming (naming_collision_on_correct_subgraph)
- description: node 'scenic view' reachable at conflicting positions [(-5, -6, 3), (-4, -5, 2), (-3, -4, 1), (-2, -3, 0)]
  - step 24: chilly avenue --[southwest]--> scenic view — correct
  - step 25: scenic view --[northwest]--> outside the house — correct

## Conflict 27 — naming (naming_collision_on_correct_subgraph)
- description: node 'outside the house' reachable at conflicting positions [(-6, -5, 3), (-5, -4, 2), (-4, -3, 1), (-3, -2, 0)]
  - step 25: scenic view --[northwest]--> outside the house — correct
  - step 27: outside the house --[north]--> foyer — correct

## Conflict 28 — naming (naming_collision_on_correct_subgraph)
- description: node 'foyer' reachable at conflicting positions [(-6, -4, 3), (-5, -3, 2), (-4, -2, 1), (-3, -1, 0)]
  - step 27: outside the house --[north]--> foyer — correct
  - step 30: foyer --[up]--> upstairs landing — correct
  - step 49: foyer --[west]--> dining room — correct

## Conflict 29 — naming (naming_collision_on_correct_subgraph)
- description: node 'upstairs landing' reachable at conflicting positions [(-6, -4, 4), (-5, -3, 3), (-4, -2, 2), (-3, -1, 1)]
  - step 30: foyer --[up]--> upstairs landing — correct
  - step 31: upstairs landing --[north]--> master bedroom — correct

## Conflict 30 — naming (naming_collision_on_correct_subgraph)
- description: node 'dining room' reachable at conflicting positions [(-7, -4, 3), (-6, -3, 2), (-5, -2, 1), (-4, -1, 0)]
  - step 49: foyer --[west]--> dining room — correct
  - step 53: dining room --[north]--> kitchen — correct

## Conflict 31 — naming (naming_collision_on_correct_subgraph)
- description: node 'kitchen' reachable at conflicting positions [(-7, -3, 3), (-6, -2, 2), (-5, -1, 1), (-4, 0, 0)]
  - step 53: dining room --[north]--> kitchen — correct
  - step 58: kitchen --[northwest]--> pantry — correct

## Conflict 32 — naming (naming_collision_on_correct_subgraph)
- description: node 'pantry' reachable at conflicting positions [(-8, -2, 3), (-7, -1, 2), (-6, 0, 1), (-5, 1, 0)]
  - step 58: kitchen --[northwest]--> pantry — correct
  - step 61: pantry --[down]--> cellar — correct

## Conflict 33 — naming (naming_collision_on_correct_subgraph)
- description: node 'cellar' reachable at conflicting positions [(-8, -2, 2), (-7, -1, 1), (-6, 0, 0), (-5, 1, -1)]
  - step 61: pantry --[down]--> cellar — correct
  - step 62: cellar --[south]--> storage — correct

## Conflict 34 — naming (naming_collision_on_correct_subgraph)
- description: node 'storage' reachable at conflicting positions [(-8, -3, 2), (-7, -2, 1), (-6, -1, 0), (-5, 0, -1)]
  - step 62: cellar --[south]--> storage — correct

## Conflict 35 — naming (naming_collision_on_correct_subgraph)
- description: node 'master bedroom' reachable at conflicting positions [(-6, -3, 4), (-5, -2, 3), (-4, -1, 2), (-3, 0, 1)]
  - step 31: upstairs landing --[north]--> master bedroom — correct
  - step 34: master bedroom --[west]--> bathroom — correct

## Conflict 36 — naming (naming_collision_on_correct_subgraph)
- description: node 'bathroom' reachable at conflicting positions [(-7, -3, 4), (-6, -2, 3), (-5, -1, 2), (-4, 0, 1)]
  - step 34: master bedroom --[west]--> bathroom — correct

## Conflict 37 — naming (naming_collision_on_correct_subgraph)
- description: node 'university court' reachable at conflicting positions [(-7, -1, 3), (-6, 0, 2), (-5, 1, 1), (-4, 2, 0)]
  - step 13: junction --[northwest]--> university court — correct
  - step 14: university court --[west]--> library — correct

## Conflict 38 — naming (naming_collision_on_correct_subgraph)
- description: node 'library' reachable at conflicting positions [(-8, -1, 3), (-7, 0, 2), (-6, 1, 1), (-5, 2, 0)]
  - step 14: university court --[west]--> library — correct
