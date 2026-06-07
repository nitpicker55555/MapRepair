# Conflict analysis: awaken

- LLM edges: 18
- GT edges: 24
- Conflicts: 35
- Type distribution: {'direction': 3, 'topology': 17, 'naming': 15}
- Root-cause distribution: {'real_vs_hallucinated': 3, 'overlap_mixed': 14, 'false_positive_overlap': 3, 'real_name_corrupted_by_neighbour_error': 5, 'naming_collision_on_correct_subgraph': 9, 'naming_mixed': 1}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'front of the church' has multiple outgoing edges labelled 'north'
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step None: front of the church --[north]--> church porch — correct
  - step 40: front of the church --[north]--> belfry — hallucinated_edge

## Conflict 2 — direction (real_vs_hallucinated)
- description: node 'vestibule' has multiple outgoing edges labelled 'south'
  - step None: vestibule --[south]--> front of the church — hallucinated_edge
  - step 37: vestibule --[south]--> church porch — correct

## Conflict 3 — direction (real_vs_hallucinated)
- description: node 'belfry' has multiple outgoing edges labelled 'south'
  - step None: belfry --[south]--> front of the church — hallucinated_edge
  - step 42: belfry --[south]--> top of oak tree — correct

## Conflict 4 — topology (overlap_mixed)
- description: position (0, 0, -1) occupied by multiple rooms ['belfry', 'church porch', 'steeple chamber', 'vestibule']
  - step 40: front of the church --[north]--> belfry — hallucinated_edge
  - step 42: belfry --[south]--> top of oak tree — correct
  - step 37: vestibule --[south]--> church porch — correct
  - step 38: church porch --[south]--> front of the church — correct
  - step 32: vestibule --[up]--> steeple chamber — swapped_src_dst (LLM: vestibule--[up]-->steeple chamber but GT has steeple chamber--[down]-->vestibule)
  - step 49: steeple chamber --[north]--> inner chamber — correct
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 14: vestibule --[north]--> sanctuary — correct

## Conflict 5 — topology (overlap_mixed)
- description: position (0, 0, 0) occupied by multiple rooms ['belfry', 'church porch', 'steeple chamber', 'vestibule']
  - step 40: front of the church --[north]--> belfry — hallucinated_edge
  - step 42: belfry --[south]--> top of oak tree — correct
  - step 37: vestibule --[south]--> church porch — correct
  - step 38: church porch --[south]--> front of the church — correct
  - step 32: vestibule --[up]--> steeple chamber — swapped_src_dst (LLM: vestibule--[up]-->steeple chamber but GT has steeple chamber--[down]-->vestibule)
  - step 49: steeple chamber --[north]--> inner chamber — correct
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 14: vestibule --[north]--> sanctuary — correct

## Conflict 6 — topology (overlap_mixed)
- description: position (0, 0, -2) occupied by multiple rooms ['belfry', 'church porch', 'steeple chamber', 'vestibule']
  - step 40: front of the church --[north]--> belfry — hallucinated_edge
  - step 42: belfry --[south]--> top of oak tree — correct
  - step 37: vestibule --[south]--> church porch — correct
  - step 38: church porch --[south]--> front of the church — correct
  - step 32: vestibule --[up]--> steeple chamber — swapped_src_dst (LLM: vestibule--[up]-->steeple chamber but GT has steeple chamber--[down]-->vestibule)
  - step 49: steeple chamber --[north]--> inner chamber — correct
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 14: vestibule --[north]--> sanctuary — correct

## Conflict 7 — topology (overlap_mixed)
- description: position (0, 0, -3) occupied by multiple rooms ['belfry', 'church porch', 'front of the church', 'vestibule']
  - step 40: front of the church --[north]--> belfry — hallucinated_edge
  - step 42: belfry --[south]--> top of oak tree — correct
  - step 37: vestibule --[south]--> church porch — correct
  - step 38: church porch --[south]--> front of the church — correct
  - step 4: west side of the church --[southeast]--> front of the church — correct
  - step 10: east side of the church --[north]--> front of the church — hallucinated_edge
  - step 43: top of oak tree --[down]--> front of the church — correct
  - step 53: inner chamber --[out]--> front of the church — hallucinated_edge
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 14: vestibule --[north]--> sanctuary — correct
  - step 32: vestibule --[up]--> steeple chamber — swapped_src_dst (LLM: vestibule--[up]-->steeple chamber but GT has steeple chamber--[down]-->vestibule)

## Conflict 8 — topology (overlap_mixed)
- description: position (0, -1, -1) occupied by multiple rooms ['church porch', 'front of the church', 'top of oak tree']
  - step 37: vestibule --[south]--> church porch — correct
  - step 38: church porch --[south]--> front of the church — correct
  - step 4: west side of the church --[southeast]--> front of the church — correct
  - step 10: east side of the church --[north]--> front of the church — hallucinated_edge
  - step 43: top of oak tree --[down]--> front of the church — correct
  - step 53: inner chamber --[out]--> front of the church — hallucinated_edge
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 40: front of the church --[north]--> belfry — hallucinated_edge
  - step 42: belfry --[south]--> top of oak tree — correct

## Conflict 9 — topology (overlap_mixed)
- description: position (0, -1, -2) occupied by multiple rooms ['church porch', 'front of the church', 'top of oak tree']
  - step 37: vestibule --[south]--> church porch — correct
  - step 38: church porch --[south]--> front of the church — correct
  - step 4: west side of the church --[southeast]--> front of the church — correct
  - step 10: east side of the church --[north]--> front of the church — hallucinated_edge
  - step 43: top of oak tree --[down]--> front of the church — correct
  - step 53: inner chamber --[out]--> front of the church — hallucinated_edge
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 40: front of the church --[north]--> belfry — hallucinated_edge
  - step 42: belfry --[south]--> top of oak tree — correct

## Conflict 10 — topology (overlap_mixed)
- description: position (0, 2, -3) occupied by multiple rooms ['front of the church', 'pulpit', 'sanctuary']
  - step 4: west side of the church --[southeast]--> front of the church — correct
  - step 10: east side of the church --[north]--> front of the church — hallucinated_edge
  - step 38: church porch --[south]--> front of the church — correct
  - step 43: top of oak tree --[down]--> front of the church — correct
  - step 53: inner chamber --[out]--> front of the church — hallucinated_edge
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 40: front of the church --[north]--> belfry — hallucinated_edge
  - step 15: sanctuary --[north]--> pulpit — correct
  - step 18: pulpit --[north]--> small office — correct
  - step 14: vestibule --[north]--> sanctuary — correct

## Conflict 11 — topology (overlap_mixed)
- description: position (0, -1, 0) occupied by multiple rooms ['front of the church', 'top of oak tree']
  - step 4: west side of the church --[southeast]--> front of the church — correct
  - step 10: east side of the church --[north]--> front of the church — hallucinated_edge
  - step 38: church porch --[south]--> front of the church — correct
  - step 43: top of oak tree --[down]--> front of the church — correct
  - step 53: inner chamber --[out]--> front of the church — hallucinated_edge
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 40: front of the church --[north]--> belfry — hallucinated_edge
  - step 42: belfry --[south]--> top of oak tree — correct

## Conflict 12 — topology (overlap_mixed)
- description: position (0, -1, -3) occupied by multiple rooms ['church porch', 'front of the church', 'top of oak tree']
  - step 37: vestibule --[south]--> church porch — correct
  - step 38: church porch --[south]--> front of the church — correct
  - step 4: west side of the church --[southeast]--> front of the church — correct
  - step 10: east side of the church --[north]--> front of the church — hallucinated_edge
  - step 43: top of oak tree --[down]--> front of the church — correct
  - step 53: inner chamber --[out]--> front of the church — hallucinated_edge
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 40: front of the church --[north]--> belfry — hallucinated_edge
  - step 42: belfry --[south]--> top of oak tree — correct

## Conflict 13 — topology (overlap_mixed)
- description: position (0, -2, -3) occupied by multiple rooms ['east side of the church', 'front of the church']
  - step 9: north side of the church --[east]--> east side of the church — correct
  - step 10: east side of the church --[north]--> front of the church — hallucinated_edge
  - step 4: west side of the church --[southeast]--> front of the church — correct
  - step 38: church porch --[south]--> front of the church — correct
  - step 43: top of oak tree --[down]--> front of the church — correct
  - step 53: inner chamber --[out]--> front of the church — hallucinated_edge
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 40: front of the church --[north]--> belfry — hallucinated_edge

## Conflict 14 — topology (overlap_mixed)
- description: position (0, 1, -3) occupied by multiple rooms ['east side of the church', 'sanctuary', 'vestibule']
  - step 9: north side of the church --[east]--> east side of the church — correct
  - step 10: east side of the church --[north]--> front of the church — hallucinated_edge
  - step 14: vestibule --[north]--> sanctuary — correct
  - step 15: sanctuary --[north]--> pulpit — correct
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 32: vestibule --[up]--> steeple chamber — swapped_src_dst (LLM: vestibule--[up]-->steeple chamber but GT has steeple chamber--[down]-->vestibule)
  - step 37: vestibule --[south]--> church porch — correct

## Conflict 15 — topology (overlap_mixed)
- description: position (0, 1, -2) occupied by multiple rooms ['east side of the church', 'inner chamber', 'sanctuary', 'steeple chamber', 'vestibule']
  - step 9: north side of the church --[east]--> east side of the church — correct
  - step 10: east side of the church --[north]--> front of the church — hallucinated_edge
  - step 49: steeple chamber --[north]--> inner chamber — correct
  - step 53: inner chamber --[out]--> front of the church — hallucinated_edge
  - step 14: vestibule --[north]--> sanctuary — correct
  - step 15: sanctuary --[north]--> pulpit — correct
  - step 32: vestibule --[up]--> steeple chamber — swapped_src_dst (LLM: vestibule--[up]-->steeple chamber but GT has steeple chamber--[down]-->vestibule)
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 37: vestibule --[south]--> church porch — correct

## Conflict 16 — topology (overlap_mixed)
- description: position (0, 1, -1) occupied by multiple rooms ['inner chamber', 'sanctuary', 'vestibule']
  - step 49: steeple chamber --[north]--> inner chamber — correct
  - step 53: inner chamber --[out]--> front of the church — hallucinated_edge
  - step 14: vestibule --[north]--> sanctuary — correct
  - step 15: sanctuary --[north]--> pulpit — correct
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 32: vestibule --[up]--> steeple chamber — swapped_src_dst (LLM: vestibule--[up]-->steeple chamber but GT has steeple chamber--[down]-->vestibule)
  - step 37: vestibule --[south]--> church porch — correct

## Conflict 17 — topology (overlap_mixed)
- description: position (0, 2, -2) occupied by multiple rooms ['inner chamber', 'pulpit']
  - step 49: steeple chamber --[north]--> inner chamber — correct
  - step 53: inner chamber --[out]--> front of the church — hallucinated_edge
  - step 15: sanctuary --[north]--> pulpit — correct
  - step 18: pulpit --[north]--> small office — correct

## Conflict 18 — topology (false_positive_overlap)
- description: position (0, 3, -3) occupied by multiple rooms ['pulpit', 'small office']
  - step 15: sanctuary --[north]--> pulpit — correct
  - step 18: pulpit --[north]--> small office — correct

## Conflict 19 — topology (false_positive_overlap)
- description: position (-2, 0, -3) occupied by multiple rooms ['graveyard', 'in the mud']
  - step 1: in the mud --[up]--> graveyard — correct
  - step 3: graveyard --[east]--> west side of the church — correct

## Conflict 20 — topology (false_positive_overlap)
- description: position (-2, 0, -2) occupied by multiple rooms ['graveyard', 'in the mud']
  - step 1: in the mud --[up]--> graveyard — correct
  - step 3: graveyard --[east]--> west side of the church — correct

## Conflict 21 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'belfry' reachable at conflicting positions [(0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0)]
  - step 40: front of the church --[north]--> belfry — hallucinated_edge
  - step 42: belfry --[south]--> top of oak tree — correct

## Conflict 22 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'front of the church' reachable at conflicting positions [(0, -4, -3), (0, -2, -3), (0, -1, -4), (0, -1, -3), (0, -1, -2), (0, -1, -1), (0, -1, 0), (0, 0, -3), (0, 2, -3)]
  - step 4: west side of the church --[southeast]--> front of the church — correct
  - step 10: east side of the church --[north]--> front of the church — hallucinated_edge
  - step 38: church porch --[south]--> front of the church — correct
  - step 43: top of oak tree --[down]--> front of the church — correct
  - step 53: inner chamber --[out]--> front of the church — hallucinated_edge
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 40: front of the church --[north]--> belfry — hallucinated_edge

## Conflict 23 — naming (naming_collision_on_correct_subgraph)
- description: node 'top of oak tree' reachable at conflicting positions [(0, -1, -3), (0, -1, -2), (0, -1, -1), (0, -1, 0), (0, -1, 1)]
  - step 42: belfry --[south]--> top of oak tree — correct
  - step 43: top of oak tree --[down]--> front of the church — correct

## Conflict 24 — naming (naming_collision_on_correct_subgraph)
- description: node 'west side of the church' reachable at conflicting positions [(-1, -3, -3), (-1, -3, -2), (-1, 0, -3), (-1, 0, -2), (-1, 0, -1), (-1, 0, 0)]
  - step 3: graveyard --[east]--> west side of the church — correct
  - step 4: west side of the church --[southeast]--> front of the church — correct
  - step 8: west side of the church --[north]--> north side of the church — correct

## Conflict 25 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'east side of the church' reachable at conflicting positions [(0, -2, -3), (0, -2, -2), (0, -2, -1), (0, -2, 0), (0, 1, -3), (0, 1, -2)]
  - step 9: north side of the church --[east]--> east side of the church — correct
  - step 10: east side of the church --[north]--> front of the church — hallucinated_edge

## Conflict 26 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'vestibule' reachable at conflicting positions [(0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 1, -3), (0, 1, -2), (0, 1, -1)]
  - step 13: front of the church --[north]--> vestibule — hallucinated_edge
  - step 14: vestibule --[north]--> sanctuary — correct
  - step 32: vestibule --[up]--> steeple chamber — swapped_src_dst (LLM: vestibule--[up]-->steeple chamber but GT has steeple chamber--[down]-->vestibule)
  - step 37: vestibule --[south]--> church porch — correct

## Conflict 27 — naming (naming_collision_on_correct_subgraph)
- description: node 'church porch' reachable at conflicting positions [(0, -1, -3), (0, -1, -2), (0, -1, -1), (0, 0, -3), (0, 0, -2), (0, 0, -1), (0, 0, 0)]
  - step 37: vestibule --[south]--> church porch — correct
  - step 38: church porch --[south]--> front of the church — correct

## Conflict 28 — naming (naming_collision_on_correct_subgraph)
- description: node 'sanctuary' reachable at conflicting positions [(0, 1, -3), (0, 1, -2), (0, 1, -1), (0, 2, -3)]
  - step 14: vestibule --[north]--> sanctuary — correct
  - step 15: sanctuary --[north]--> pulpit — correct

## Conflict 29 — naming (naming_mixed)
- description: node 'steeple chamber' reachable at conflicting positions [(0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 1, -2)]
  - step 32: vestibule --[up]--> steeple chamber — swapped_src_dst (LLM: vestibule--[up]-->steeple chamber but GT has steeple chamber--[down]-->vestibule)
  - step 49: steeple chamber --[north]--> inner chamber — correct

## Conflict 30 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'inner chamber' reachable at conflicting positions [(0, 1, -2), (0, 1, -1), (0, 1, 0), (0, 2, -2)]
  - step 49: steeple chamber --[north]--> inner chamber — correct
  - step 53: inner chamber --[out]--> front of the church — hallucinated_edge

## Conflict 31 — naming (naming_collision_on_correct_subgraph)
- description: node 'pulpit' reachable at conflicting positions [(0, 2, -3), (0, 2, -2), (0, 2, -1), (0, 3, -3)]
  - step 15: sanctuary --[north]--> pulpit — correct
  - step 18: pulpit --[north]--> small office — correct

## Conflict 32 — naming (naming_collision_on_correct_subgraph)
- description: node 'small office' reachable at conflicting positions [(0, 3, -3), (0, 3, -2), (0, 3, -1), (0, 4, -3)]
  - step 18: pulpit --[north]--> small office — correct

## Conflict 33 — naming (naming_collision_on_correct_subgraph)
- description: node 'north side of the church' reachable at conflicting positions [(-1, -2, -3), (-1, -2, -2), (-1, -2, -1), (-1, 1, -3), (-1, 1, -2), (-1, 1, -1)]
  - step 8: west side of the church --[north]--> north side of the church — correct
  - step 9: north side of the church --[east]--> east side of the church — correct

## Conflict 34 — naming (naming_collision_on_correct_subgraph)
- description: node 'graveyard' reachable at conflicting positions [(-2, -3, -3), (-2, 0, -3), (-2, 0, -2), (-2, 0, -1)]
  - step 1: in the mud --[up]--> graveyard — correct
  - step 3: graveyard --[east]--> west side of the church — correct

## Conflict 35 — naming (naming_collision_on_correct_subgraph)
- description: node 'in the mud' reachable at conflicting positions [(-2, -3, -4), (-2, 0, -4), (-2, 0, -3), (-2, 0, -2)]
  - step 1: in the mud --[up]--> graveyard — correct
