# Conflict analysis: plundered

- LLM edges: 20
- GT edges: 30
- Conflicts: 24
- Type distribution: {'direction': 1, 'topology': 11, 'naming': 12}
- Root-cause distribution: {'name_hallucination': 2, 'false_positive_overlap': 2, 'name_hallucination_caused_overlap': 9, 'naming_collision_on_correct_subgraph': 8, 'real_name_corrupted_by_neighbour_error': 3}

## Conflict 1 — direction (name_hallucination)
- description: node 'landing' has multiple outgoing edges labelled 'south'
  - step None: landing --[south]--> on the bed — dst_hallucinated
  - step 33: landing --[south]--> captain's quarters — correct

## Conflict 2 — topology (false_positive_overlap)
- description: position (0, 0, 0) occupied by multiple rooms ['aft hold', 'beach']
  - step 24: landing --[down]--> aft hold — correct
  - step 25: aft hold --[north]--> hold — correct
  - step 61: shallows --[west]--> beach — correct
  - step 62: beach --[north]--> lawn — correct

## Conflict 3 — topology (name_hallucination_caused_overlap)
- description: position (0, 0, 1) occupied by multiple rooms ['landing', 'quarterdeck']
  - step 14: on the bed --[north]--> landing — src_hallucinated
  - step 16: landing --[in]--> sleeping cupboard — correct
  - step 24: landing --[down]--> aft hold — correct
  - step 33: landing --[south]--> captain's quarters — correct
  - step 40: poop deck --[north]--> quarterdeck — correct
  - step 41: quarterdeck --[north]--> main deck — correct

## Conflict 4 — topology (name_hallucination_caused_overlap)
- description: position (0, 2, -3) occupied by multiple rooms ['landing', 'quarterdeck']
  - step 14: on the bed --[north]--> landing — src_hallucinated
  - step 16: landing --[in]--> sleeping cupboard — correct
  - step 24: landing --[down]--> aft hold — correct
  - step 33: landing --[south]--> captain's quarters — correct
  - step 40: poop deck --[north]--> quarterdeck — correct
  - step 41: quarterdeck --[north]--> main deck — correct

## Conflict 5 — topology (name_hallucination_caused_overlap)
- description: position (0, 3, -5) occupied by multiple rooms ['landing', 'quarterdeck']
  - step 14: on the bed --[north]--> landing — src_hallucinated
  - step 16: landing --[in]--> sleeping cupboard — correct
  - step 24: landing --[down]--> aft hold — correct
  - step 33: landing --[south]--> captain's quarters — correct
  - step 40: poop deck --[north]--> quarterdeck — correct
  - step 41: quarterdeck --[north]--> main deck — correct

## Conflict 6 — topology (name_hallucination_caused_overlap)
- description: position (0, 1, -1) occupied by multiple rooms ['landing', 'quarterdeck']
  - step 14: on the bed --[north]--> landing — src_hallucinated
  - step 16: landing --[in]--> sleeping cupboard — correct
  - step 24: landing --[down]--> aft hold — correct
  - step 33: landing --[south]--> captain's quarters — correct
  - step 40: poop deck --[north]--> quarterdeck — correct
  - step 41: quarterdeck --[north]--> main deck — correct

## Conflict 7 — topology (false_positive_overlap)
- description: position (0, 1, 0) occupied by multiple rooms ['hold', 'lawn']
  - step 25: aft hold --[north]--> hold — correct
  - step 45: main deck --[down]--> hold — correct
  - step 26: hold --[north]--> crew's quarters — correct
  - step 62: beach --[north]--> lawn — correct
  - step 63: lawn --[west]--> forest — correct

## Conflict 8 — topology (name_hallucination_caused_overlap)
- description: position (0, 2, 0) occupied by multiple rooms ['clearing', "crew's quarters", 'kitchen']
  - step 70: forest --[northeast]--> clearing — dst_hallucinated
  - step 26: hold --[north]--> crew's quarters — correct
  - step 65: trade entrance --[east]--> kitchen — both_names_hallucinated

## Conflict 9 — topology (name_hallucination_caused_overlap)
- description: position (0, 0, -1) occupied by multiple rooms ["captain's quarters", 'on the bed', 'poop deck']
  - step 33: landing --[south]--> captain's quarters — correct
  - step 35: captain's quarters --[south]--> the ledge — correct
  - step 14: on the bed --[north]--> landing — src_hallucinated
  - step 39: on the ladder --[up]--> poop deck — correct
  - step 40: poop deck --[north]--> quarterdeck — correct

## Conflict 10 — topology (name_hallucination_caused_overlap)
- description: position (0, 1, -3) occupied by multiple rooms ["captain's quarters", 'on the bed', 'poop deck']
  - step 33: landing --[south]--> captain's quarters — correct
  - step 35: captain's quarters --[south]--> the ledge — correct
  - step 14: on the bed --[north]--> landing — src_hallucinated
  - step 39: on the ladder --[up]--> poop deck — correct
  - step 40: poop deck --[north]--> quarterdeck — correct

## Conflict 11 — topology (name_hallucination_caused_overlap)
- description: position (0, 2, -5) occupied by multiple rooms ["captain's quarters", 'on the bed', 'poop deck']
  - step 33: landing --[south]--> captain's quarters — correct
  - step 35: captain's quarters --[south]--> the ledge — correct
  - step 14: on the bed --[north]--> landing — src_hallucinated
  - step 39: on the ladder --[up]--> poop deck — correct
  - step 40: poop deck --[north]--> quarterdeck — correct

## Conflict 12 — topology (name_hallucination_caused_overlap)
- description: position (0, -1, 1) occupied by multiple rooms ["captain's quarters", 'on the bed', 'poop deck']
  - step 33: landing --[south]--> captain's quarters — correct
  - step 35: captain's quarters --[south]--> the ledge — correct
  - step 14: on the bed --[north]--> landing — src_hallucinated
  - step 39: on the ladder --[up]--> poop deck — correct
  - step 40: poop deck --[north]--> quarterdeck — correct

## Conflict 13 — naming (naming_collision_on_correct_subgraph)
- description: node 'aft hold' reachable at conflicting positions [(0, 0, 0), (0, 1, -2), (0, 2, -4), (0, 3, -6)]
  - step 24: landing --[down]--> aft hold — correct
  - step 25: aft hold --[north]--> hold — correct

## Conflict 14 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'landing' reachable at conflicting positions [(0, 0, 1), (0, 1, -1), (0, 2, -3), (0, 3, -5), (0, 4, -7)]
  - step 14: on the bed --[north]--> landing — src_hallucinated
  - step 16: landing --[in]--> sleeping cupboard — correct
  - step 24: landing --[down]--> aft hold — correct
  - step 33: landing --[south]--> captain's quarters — correct

## Conflict 15 — naming (naming_collision_on_correct_subgraph)
- description: node 'hold' reachable at conflicting positions [(0, 1, 0), (0, 2, -2), (0, 3, -4), (0, 4, -6)]
  - step 25: aft hold --[north]--> hold — correct
  - step 45: main deck --[down]--> hold — correct
  - step 26: hold --[north]--> crew's quarters — correct

## Conflict 16 — naming (naming_collision_on_correct_subgraph)
- description: node "crew's quarters" reachable at conflicting positions [(0, 2, 0), (0, 3, -2), (0, 4, -4), (0, 5, -6)]
  - step 26: hold --[north]--> crew's quarters — correct

## Conflict 17 — naming (naming_collision_on_correct_subgraph)
- description: node 'main deck' reachable at conflicting positions [(0, 1, 1), (0, 2, -1), (0, 3, -3), (0, 4, -5)]
  - step 41: quarterdeck --[north]--> main deck — correct
  - step 45: main deck --[down]--> hold — correct
  - step 48: main deck --[north]--> forecastle — correct

## Conflict 18 — naming (naming_collision_on_correct_subgraph)
- description: node 'quarterdeck' reachable at conflicting positions [(0, 0, 1), (0, 1, -1), (0, 2, -3), (0, 3, -5)]
  - step 40: poop deck --[north]--> quarterdeck — correct
  - step 41: quarterdeck --[north]--> main deck — correct

## Conflict 19 — naming (naming_collision_on_correct_subgraph)
- description: node 'forecastle' reachable at conflicting positions [(0, 2, 1), (0, 3, -1), (0, 4, -3), (0, 5, -5)]
  - step 48: main deck --[north]--> forecastle — correct
  - step 51: forecastle --[in]--> galley — correct

## Conflict 20 — naming (naming_collision_on_correct_subgraph)
- description: node 'poop deck' reachable at conflicting positions [(0, -1, 1), (0, 0, -1), (0, 1, -3), (0, 2, -5)]
  - step 39: on the ladder --[up]--> poop deck — correct
  - step 40: poop deck --[north]--> quarterdeck — correct

## Conflict 21 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'on the ladder' reachable at conflicting positions [(0, -1, 0), (0, 0, -2), (0, 1, -4), (0, 2, -6)]
  - step 36: the ledge --[up]--> on the ladder — hallucinated_edge
  - step 39: on the ladder --[up]--> poop deck — correct

## Conflict 22 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'the ledge' reachable at conflicting positions [(0, -1, -1), (0, 0, -3), (0, 1, -5), (0, 2, -7)]
  - step 35: captain's quarters --[south]--> the ledge — correct
  - step 36: the ledge --[up]--> on the ladder — hallucinated_edge

## Conflict 23 — naming (naming_collision_on_correct_subgraph)
- description: node "captain's quarters" reachable at conflicting positions [(0, -1, 1), (0, 0, -1), (0, 1, -3), (0, 2, -5), (0, 3, -7)]
  - step 33: landing --[south]--> captain's quarters — correct
  - step 35: captain's quarters --[south]--> the ledge — correct

## Conflict 24 — naming (name_hallucination)
- description: node 'on the bed' reachable at conflicting positions [(0, -1, 1), (0, 0, -1), (0, 1, -3), (0, 2, -5)]
  - step 14: on the bed --[north]--> landing — src_hallucinated
