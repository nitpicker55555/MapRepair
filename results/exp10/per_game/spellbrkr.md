# Conflict analysis: spellbrkr

- LLM edges: 14
- GT edges: 24
- Conflicts: 22
- Type distribution: {'direction': 1, 'topology': 9, 'naming': 12}
- Root-cause distribution: {'real_vs_hallucinated': 1, 'wrong_direction_caused_overlap': 5, 'name_hallucination_caused_overlap': 4, 'real_name_corrupted_by_neighbour_error': 3, 'naming_mixed': 4, 'naming_collision_on_correct_subgraph': 2, 'name_hallucination': 3}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'boulder(a long oblong boulder below you)' has multiple outgoing edges labelled 'up'
  - step 43: boulder(a long oblong boulder below you) --[up]--> boulder(a nice oblong boulder above you and another large one below you) — wrong_direction (LLM: 'up', GT: 'down')
  - step 45: boulder(a long oblong boulder below you) --[up]--> mountain top — correct

## Conflict 2 — topology (wrong_direction_caused_overlap)
- description: position (0, 0, 0) occupied by multiple rooms ['belwit square', 'boulder(a long oblong boulder below you)']
  - step 7: guild hall --[south]--> belwit square — correct
  - step 42: cliff top --[up]--> boulder(a long oblong boulder below you) — hallucinated_edge
  - step 43: boulder(a long oblong boulder below you) --[up]--> boulder(a nice oblong boulder above you and another large one below you) — wrong_direction (LLM: 'up', GT: 'down')
  - step 45: boulder(a long oblong boulder below you) --[up]--> mountain top — correct

## Conflict 3 — topology (wrong_direction_caused_overlap)
- description: position (-1, -1, 4) occupied by multiple rooms ['boulder(a nice oblong boulder above you and another large one below you)', 'mountain top']
  - step 43: boulder(a long oblong boulder below you) --[up]--> boulder(a nice oblong boulder above you and another large one below you) — wrong_direction (LLM: 'up', GT: 'down')
  - step 45: boulder(a long oblong boulder below you) --[up]--> mountain top — correct
  - step 48: mountain top --[west]--> stone hut — correct
  - step 58: mountain top --[south]--> meadow — hallucinated_edge

## Conflict 4 — topology (wrong_direction_caused_overlap)
- description: position (-3, -3, 10) occupied by multiple rooms ['boulder(a nice oblong boulder above you and another large one below you)', 'mountain top']
  - step 43: boulder(a long oblong boulder below you) --[up]--> boulder(a nice oblong boulder above you and another large one below you) — wrong_direction (LLM: 'up', GT: 'down')
  - step 45: boulder(a long oblong boulder below you) --[up]--> mountain top — correct
  - step 48: mountain top --[west]--> stone hut — correct
  - step 58: mountain top --[south]--> meadow — hallucinated_edge

## Conflict 5 — topology (wrong_direction_caused_overlap)
- description: position (0, 0, 1) occupied by multiple rooms ['boulder(a nice oblong boulder above you and another large one below you)', 'mountain top']
  - step 43: boulder(a long oblong boulder below you) --[up]--> boulder(a nice oblong boulder above you and another large one below you) — wrong_direction (LLM: 'up', GT: 'down')
  - step 45: boulder(a long oblong boulder below you) --[up]--> mountain top — correct
  - step 48: mountain top --[west]--> stone hut — correct
  - step 58: mountain top --[south]--> meadow — hallucinated_edge

## Conflict 6 — topology (wrong_direction_caused_overlap)
- description: position (-2, -2, 7) occupied by multiple rooms ['boulder(a nice oblong boulder above you and another large one below you)', 'mountain top']
  - step 43: boulder(a long oblong boulder below you) --[up]--> boulder(a nice oblong boulder above you and another large one below you) — wrong_direction (LLM: 'up', GT: 'down')
  - step 45: boulder(a long oblong boulder below you) --[up]--> mountain top — correct
  - step 48: mountain top --[west]--> stone hut — correct
  - step 58: mountain top --[south]--> meadow — hallucinated_edge

## Conflict 7 — topology (name_hallucination_caused_overlap)
- description: position (-3, -2, 7) occupied by multiple rooms ['cave', 'stone hut']
  - step 63: packed earth --[north]--> cave — hallucinated_edge
  - step 69: cave --[down]--> ogre lair — dst_hallucinated
  - step 48: mountain top --[west]--> stone hut — correct

## Conflict 8 — topology (name_hallucination_caused_overlap)
- description: position (-2, -1, 4) occupied by multiple rooms ['cave', 'stone hut']
  - step 63: packed earth --[north]--> cave — hallucinated_edge
  - step 69: cave --[down]--> ogre lair — dst_hallucinated
  - step 48: mountain top --[west]--> stone hut — correct

## Conflict 9 — topology (name_hallucination_caused_overlap)
- description: position (-1, 0, 1) occupied by multiple rooms ['cave', 'stone hut']
  - step 63: packed earth --[north]--> cave — hallucinated_edge
  - step 69: cave --[down]--> ogre lair — dst_hallucinated
  - step 48: mountain top --[west]--> stone hut — correct

## Conflict 10 — topology (name_hallucination_caused_overlap)
- description: position (-4, -3, 10) occupied by multiple rooms ['cave', 'stone hut']
  - step 63: packed earth --[north]--> cave — hallucinated_edge
  - step 69: cave --[down]--> ogre lair — dst_hallucinated
  - step 48: mountain top --[west]--> stone hut — correct

## Conflict 11 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'boulder(a long oblong boulder below you)' reachable at conflicting positions [(-3, -3, 9), (-2, -2, 6), (-1, -1, 3), (0, 0, 0)]
  - step 42: cliff top --[up]--> boulder(a long oblong boulder below you) — hallucinated_edge
  - step 43: boulder(a long oblong boulder below you) --[up]--> boulder(a nice oblong boulder above you and another large one below you) — wrong_direction (LLM: 'up', GT: 'down')
  - step 45: boulder(a long oblong boulder below you) --[up]--> mountain top — correct

## Conflict 12 — naming (naming_mixed)
- description: node 'cliff top' reachable at conflicting positions [(-4, -4, 11), (-3, -3, 8), (-2, -2, 5), (-1, -1, 2), (0, 0, -1)]
  - step 37: packed earth --[up]--> cliff top — hallucinated_edge
  - step 42: cliff top --[up]--> boulder(a long oblong boulder below you) — hallucinated_edge

## Conflict 13 — naming (naming_mixed)
- description: node 'boulder(a nice oblong boulder above you and another large one below you)' reachable at conflicting positions [(-3, -3, 10), (-2, -2, 7), (-1, -1, 4), (0, 0, 1)]
  - step 43: boulder(a long oblong boulder below you) --[up]--> boulder(a nice oblong boulder above you and another large one below you) — wrong_direction (LLM: 'up', GT: 'down')

## Conflict 14 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'mountain top' reachable at conflicting positions [(-3, -3, 10), (-2, -2, 7), (-1, -1, 4), (0, 0, 1)]
  - step 45: boulder(a long oblong boulder below you) --[up]--> mountain top — correct
  - step 48: mountain top --[west]--> stone hut — correct
  - step 58: mountain top --[south]--> meadow — hallucinated_edge

## Conflict 15 — naming (naming_collision_on_correct_subgraph)
- description: node 'stone hut' reachable at conflicting positions [(-4, -3, 10), (-3, -2, 7), (-2, -1, 4), (-1, 0, 1)]
  - step 48: mountain top --[west]--> stone hut — correct

## Conflict 16 — naming (naming_mixed)
- description: node 'meadow' reachable at conflicting positions [(-3, -4, 10), (-2, -3, 7), (-1, -2, 4), (0, -1, 1)]
  - step 58: mountain top --[south]--> meadow — hallucinated_edge
  - step 62: meadow --[west]--> packed earth — hallucinated_edge

## Conflict 17 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'packed earth' reachable at conflicting positions [(-4, -4, 10), (-3, -3, 7), (-2, -2, 4), (-1, -1, 1), (0, 0, -2)]
  - step 24: in roc nest --[east]--> packed earth — src_hallucinated
  - step 62: meadow --[west]--> packed earth — hallucinated_edge
  - step 15: packed earth --[down]--> midair — correct
  - step 25: packed earth --[south]--> ruins room — hallucinated_edge
  - step 37: packed earth --[up]--> cliff top — hallucinated_edge
  - step 63: packed earth --[north]--> cave — hallucinated_edge

## Conflict 18 — naming (naming_collision_on_correct_subgraph)
- description: node 'midair' reachable at conflicting positions [(-4, -4, 9), (-3, -3, 6), (-2, -2, 3), (-1, -1, 0)]
  - step 15: packed earth --[down]--> midair — correct

## Conflict 19 — naming (name_hallucination)
- description: node 'in roc nest' reachable at conflicting positions [(-5, -4, 10), (-4, -3, 7), (-3, -2, 4), (-2, -1, 1)]
  - step 24: in roc nest --[east]--> packed earth — src_hallucinated

## Conflict 20 — naming (naming_mixed)
- description: node 'ruins room' reachable at conflicting positions [(-4, -5, 10), (-3, -4, 7), (-2, -3, 4), (-1, -2, 1)]
  - step 25: packed earth --[south]--> ruins room — hallucinated_edge

## Conflict 21 — naming (name_hallucination)
- description: node 'cave' reachable at conflicting positions [(-4, -3, 10), (-3, -2, 7), (-2, -1, 4), (-1, 0, 1)]
  - step 63: packed earth --[north]--> cave — hallucinated_edge
  - step 69: cave --[down]--> ogre lair — dst_hallucinated

## Conflict 22 — naming (name_hallucination)
- description: node 'ogre lair' reachable at conflicting positions [(-4, -3, 9), (-3, -2, 6), (-2, -1, 3), (-1, 0, 0)]
  - step 69: cave --[down]--> ogre lair — dst_hallucinated
