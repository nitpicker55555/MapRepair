# Conflict analysis: sherlock

- LLM edges: 18
- GT edges: 29
- Conflicts: 21
- Type distribution: {'direction': 3, 'topology': 8, 'naming': 10}
- Root-cause distribution: {'real_vs_hallucinated': 3, 'wrong_direction_caused_overlap': 4, 'overlap_mixed': 4, 'real_name_corrupted_by_neighbour_error': 4, 'naming_collision_on_correct_subgraph': 6}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'entry hall' has multiple outgoing edges labelled 'north'
  - step 2: entry hall --[north]--> holmes's study — hallucinated_edge
  - step 17: entry hall --[north]--> parlour — correct

## Conflict 2 — direction (real_vs_hallucinated)
- description: node "holmes's study" has multiple outgoing edges labelled 'south'
  - step None: holmes's study --[south]--> entry hall — hallucinated_edge
  - step 15: holmes's study --[south]--> vestibule — correct

## Conflict 3 — direction (real_vs_hallucinated)
- description: node 'the black museum' has multiple outgoing edges labelled 'up'
  - step None: the black museum --[up]--> whitehall — hallucinated_edge
  - step 58: the black museum --[up]--> scotland yard — correct

## Conflict 4 — topology (wrong_direction_caused_overlap)
- description: position (0, 0, 1) occupied by multiple rooms ['221-b baker street', 'entry hall', 'vestibule']
  - step 1: 221-b baker street --[up]--> entry hall — wrong_direction (LLM: 'up', GT: 'west')
  - step 23: 221-b baker street --[north]--> york place — correct
  - step 16: vestibule --[down]--> entry hall — correct
  - step 2: entry hall --[north]--> holmes's study — hallucinated_edge
  - step 17: entry hall --[north]--> parlour — correct
  - step 15: holmes's study --[south]--> vestibule — correct

## Conflict 5 — topology (wrong_direction_caused_overlap)
- description: position (0, 0, 2) occupied by multiple rooms ['221-b baker street', 'entry hall', 'vestibule']
  - step 1: 221-b baker street --[up]--> entry hall — wrong_direction (LLM: 'up', GT: 'west')
  - step 23: 221-b baker street --[north]--> york place — correct
  - step 16: vestibule --[down]--> entry hall — correct
  - step 2: entry hall --[north]--> holmes's study — hallucinated_edge
  - step 17: entry hall --[north]--> parlour — correct
  - step 15: holmes's study --[south]--> vestibule — correct

## Conflict 6 — topology (wrong_direction_caused_overlap)
- description: position (0, 0, 3) occupied by multiple rooms ['221-b baker street', 'entry hall', 'vestibule']
  - step 1: 221-b baker street --[up]--> entry hall — wrong_direction (LLM: 'up', GT: 'west')
  - step 23: 221-b baker street --[north]--> york place — correct
  - step 16: vestibule --[down]--> entry hall — correct
  - step 2: entry hall --[north]--> holmes's study — hallucinated_edge
  - step 17: entry hall --[north]--> parlour — correct
  - step 15: holmes's study --[south]--> vestibule — correct

## Conflict 7 — topology (wrong_direction_caused_overlap)
- description: position (0, 0, 4) occupied by multiple rooms ['entry hall', 'vestibule']
  - step 1: 221-b baker street --[up]--> entry hall — wrong_direction (LLM: 'up', GT: 'west')
  - step 16: vestibule --[down]--> entry hall — correct
  - step 2: entry hall --[north]--> holmes's study — hallucinated_edge
  - step 17: entry hall --[north]--> parlour — correct
  - step 15: holmes's study --[south]--> vestibule — correct

## Conflict 8 — topology (overlap_mixed)
- description: position (0, 1, 2) occupied by multiple rooms ["holmes's study", 'parlour', 'york place']
  - step 2: entry hall --[north]--> holmes's study — hallucinated_edge
  - step 12: holmes's study --[west]--> holmes's bedroom — correct
  - step 15: holmes's study --[south]--> vestibule — correct
  - step 17: entry hall --[north]--> parlour — correct
  - step 23: 221-b baker street --[north]--> york place — correct
  - step 24: york place --[east]--> marylebone road — correct

## Conflict 9 — topology (overlap_mixed)
- description: position (0, 1, 3) occupied by multiple rooms ["holmes's study", 'parlour', 'york place']
  - step 2: entry hall --[north]--> holmes's study — hallucinated_edge
  - step 12: holmes's study --[west]--> holmes's bedroom — correct
  - step 15: holmes's study --[south]--> vestibule — correct
  - step 17: entry hall --[north]--> parlour — correct
  - step 23: 221-b baker street --[north]--> york place — correct
  - step 24: york place --[east]--> marylebone road — correct

## Conflict 10 — topology (overlap_mixed)
- description: position (0, 1, 1) occupied by multiple rooms ["holmes's study", 'parlour', 'york place']
  - step 2: entry hall --[north]--> holmes's study — hallucinated_edge
  - step 12: holmes's study --[west]--> holmes's bedroom — correct
  - step 15: holmes's study --[south]--> vestibule — correct
  - step 17: entry hall --[north]--> parlour — correct
  - step 23: 221-b baker street --[north]--> york place — correct
  - step 24: york place --[east]--> marylebone road — correct

## Conflict 11 — topology (overlap_mixed)
- description: position (0, 1, 4) occupied by multiple rooms ["holmes's study", 'parlour']
  - step 2: entry hall --[north]--> holmes's study — hallucinated_edge
  - step 12: holmes's study --[west]--> holmes's bedroom — correct
  - step 15: holmes's study --[south]--> vestibule — correct
  - step 17: entry hall --[north]--> parlour — correct

## Conflict 12 — naming (real_name_corrupted_by_neighbour_error)
- description: node '221-b baker street' reachable at conflicting positions [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)]
  - step 1: 221-b baker street --[up]--> entry hall — wrong_direction (LLM: 'up', GT: 'west')
  - step 23: 221-b baker street --[north]--> york place — correct

## Conflict 13 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'entry hall' reachable at conflicting positions [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)]
  - step 1: 221-b baker street --[up]--> entry hall — wrong_direction (LLM: 'up', GT: 'west')
  - step 16: vestibule --[down]--> entry hall — correct
  - step 2: entry hall --[north]--> holmes's study — hallucinated_edge
  - step 17: entry hall --[north]--> parlour — correct

## Conflict 14 — naming (naming_collision_on_correct_subgraph)
- description: node 'york place' reachable at conflicting positions [(0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3)]
  - step 23: 221-b baker street --[north]--> york place — correct
  - step 24: york place --[east]--> marylebone road — correct

## Conflict 15 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'marylebone road' reachable at conflicting positions [(1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3)]
  - step 24: york place --[east]--> marylebone road — correct
  - step 33: marylebone road --[north]--> madame tussaud's — correct
  - step 53: marylebone road --[in]--> whitehall — hallucinated_edge

## Conflict 16 — naming (naming_collision_on_correct_subgraph)
- description: node "madame tussaud's" reachable at conflicting positions [(1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3)]
  - step 33: marylebone road --[north]--> madame tussaud's — correct
  - step 35: madame tussaud's --[west]--> chamber of horrors — correct

## Conflict 17 — naming (naming_collision_on_correct_subgraph)
- description: node 'chamber of horrors' reachable at conflicting positions [(0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3)]
  - step 35: madame tussaud's --[west]--> chamber of horrors — correct

## Conflict 18 — naming (real_name_corrupted_by_neighbour_error)
- description: node "holmes's study" reachable at conflicting positions [(0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5)]
  - step 2: entry hall --[north]--> holmes's study — hallucinated_edge
  - step 12: holmes's study --[west]--> holmes's bedroom — correct
  - step 15: holmes's study --[south]--> vestibule — correct

## Conflict 19 — naming (naming_collision_on_correct_subgraph)
- description: node 'vestibule' reachable at conflicting positions [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4), (0, 0, 5)]
  - step 15: holmes's study --[south]--> vestibule — correct
  - step 16: vestibule --[down]--> entry hall — correct

## Conflict 20 — naming (naming_collision_on_correct_subgraph)
- description: node 'parlour' reachable at conflicting positions [(0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4)]
  - step 17: entry hall --[north]--> parlour — correct

## Conflict 21 — naming (naming_collision_on_correct_subgraph)
- description: node "holmes's bedroom" reachable at conflicting positions [(-1, 1, 1), (-1, 1, 2), (-1, 1, 3), (-1, 1, 4)]
  - step 12: holmes's study --[west]--> holmes's bedroom — correct
