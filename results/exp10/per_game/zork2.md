# Conflict analysis: zork2

- LLM edges: 26
- GT edges: 36
- Conflicts: 43
- Type distribution: {'direction': 4, 'topology': 21, 'naming': 18}
- Root-cause distribution: {'src_dst_swap': 1, 'real_vs_hallucinated': 3, 'name_hallucination_caused_overlap': 4, 'overlap_mixed': 14, 'false_positive_overlap': 3, 'real_name_corrupted_by_neighbour_error': 5, 'naming_collision_on_correct_subgraph': 12, 'name_hallucination': 1}

## Conflict 1 — direction (src_dst_swap)
- description: node 'carousel room' has multiple outgoing edges labelled 'north'
  - step 13: carousel room --[north]--> topiary — correct
  - step 18: carousel room --[north]--> marble hall — swapped_src_dst (LLM: carousel room--[north]-->marble hall but GT has marble hall--[south]-->carousel room)

## Conflict 2 — direction (real_vs_hallucinated)
- description: node 'ledge in ravine' has multiple outgoing edges labelled 'down'
  - step None: ledge in ravine --[down]--> marble hall — hallucinated_edge
  - step 51: ledge in ravine --[down]--> deep ford — correct

## Conflict 3 — direction (real_vs_hallucinated)
- description: node 'ledge in ravine' has multiple outgoing edges labelled 'west'
  - step 34: ledge in ravine --[west]--> end of ledge — correct
  - step None: ledge in ravine --[west]--> dragon room — hallucinated_edge

## Conflict 4 — direction (real_vs_hallucinated)
- description: node 'dreary room' has multiple outgoing edges labelled 'south'
  - step None: dreary room --[south]--> ledge in ravine — hallucinated_edge
  - step 32: dreary room --[south]--> tiny room — correct

## Conflict 5 — topology (name_hallucination_caused_overlap)
- description: position (0, 0, 0) occupied by multiple rooms ['carousel room', 'dark place']
  - step 12: path near stream --[southwest]--> carousel room — correct
  - step 13: carousel room --[north]--> topiary — correct
  - step 18: carousel room --[north]--> marble hall — swapped_src_dst (LLM: carousel room--[north]-->marble hall but GT has marble hall--[south]-->carousel room)
  - step 67: carousel room --[southeast]--> menhir room — dst_hallucinated
  - step 5: great cavern --[south]--> dark place — dst_hallucinated

## Conflict 6 — topology (name_hallucination_caused_overlap)
- description: position (0, 1, 0) occupied by multiple rooms ['carousel room', 'deep ford', 'great cavern', 'marble hall', 'topiary']
  - step 12: path near stream --[southwest]--> carousel room — correct
  - step 13: carousel room --[north]--> topiary — correct
  - step 18: carousel room --[north]--> marble hall — swapped_src_dst (LLM: carousel room--[north]-->marble hall but GT has marble hall--[south]-->carousel room)
  - step 67: carousel room --[southeast]--> menhir room — dst_hallucinated
  - step 51: ledge in ravine --[down]--> deep ford — correct
  - step 52: deep ford --[south]--> marble hall — correct
  - step 4: foot bridge --[south]--> great cavern — correct
  - step 5: great cavern --[south]--> dark place — dst_hallucinated
  - step 20: marble hall --[up]--> ledge in ravine — hallucinated_edge
  - step 55: marble hall --[east]--> path near stream — correct
  - step 63: formal garden --[south]--> topiary — correct

## Conflict 7 — topology (overlap_mixed)
- description: position (0, 2, 0) occupied by multiple rooms ['deep ford', 'foot bridge', 'formal garden', 'marble hall']
  - step 51: ledge in ravine --[down]--> deep ford — correct
  - step 52: deep ford --[south]--> marble hall — correct
  - step 3: narrow tunnel --[south]--> foot bridge — correct
  - step 4: foot bridge --[south]--> great cavern — correct
  - step 10: north end of garden --[south]--> formal garden — correct
  - step 11: formal garden --[west]--> path near stream — correct
  - step 63: formal garden --[south]--> topiary — correct
  - step 18: carousel room --[north]--> marble hall — swapped_src_dst (LLM: carousel room--[north]-->marble hall but GT has marble hall--[south]-->carousel room)
  - step 20: marble hall --[up]--> ledge in ravine — hallucinated_edge
  - step 55: marble hall --[east]--> path near stream — correct

## Conflict 8 — topology (false_positive_overlap)
- description: position (0, 3, 0) occupied by multiple rooms ['deep ford', 'narrow tunnel', 'north end of garden']
  - step 51: ledge in ravine --[down]--> deep ford — correct
  - step 52: deep ford --[south]--> marble hall — correct
  - step 2: inside the barrow --[south]--> narrow tunnel — correct
  - step 3: narrow tunnel --[south]--> foot bridge — correct
  - step 7: dark tunnel --[southeast]--> north end of garden — correct
  - step 9: gazebo --[out]--> north end of garden — correct
  - step 10: north end of garden --[south]--> formal garden — correct

## Conflict 9 — topology (name_hallucination_caused_overlap)
- description: position (0, 0, 1) occupied by multiple rooms ['carousel room', 'marble hall']
  - step 12: path near stream --[southwest]--> carousel room — correct
  - step 13: carousel room --[north]--> topiary — correct
  - step 18: carousel room --[north]--> marble hall — swapped_src_dst (LLM: carousel room--[north]-->marble hall but GT has marble hall--[south]-->carousel room)
  - step 67: carousel room --[southeast]--> menhir room — dst_hallucinated
  - step 52: deep ford --[south]--> marble hall — correct
  - step 20: marble hall --[up]--> ledge in ravine — hallucinated_edge
  - step 55: marble hall --[east]--> path near stream — correct

## Conflict 10 — topology (name_hallucination_caused_overlap)
- description: position (0, 1, 1) occupied by multiple rooms ['carousel room', 'deep ford', 'ledge in ravine', 'marble hall', 'topiary']
  - step 12: path near stream --[southwest]--> carousel room — correct
  - step 13: carousel room --[north]--> topiary — correct
  - step 18: carousel room --[north]--> marble hall — swapped_src_dst (LLM: carousel room--[north]-->marble hall but GT has marble hall--[south]-->carousel room)
  - step 67: carousel room --[southeast]--> menhir room — dst_hallucinated
  - step 51: ledge in ravine --[down]--> deep ford — correct
  - step 52: deep ford --[south]--> marble hall — correct
  - step 20: marble hall --[up]--> ledge in ravine — hallucinated_edge
  - step 33: tiny room --[down]--> ledge in ravine — correct
  - step 50: dragon room --[east]--> ledge in ravine — hallucinated_edge
  - step 30: ledge in ravine --[north]--> dreary room — hallucinated_edge
  - step 34: ledge in ravine --[west]--> end of ledge — correct
  - step 55: marble hall --[east]--> path near stream — correct
  - step 63: formal garden --[south]--> topiary — correct

## Conflict 11 — topology (overlap_mixed)
- description: position (2, 0, 1) occupied by multiple rooms ['marble hall', 'topiary']
  - step 18: carousel room --[north]--> marble hall — swapped_src_dst (LLM: carousel room--[north]-->marble hall but GT has marble hall--[south]-->carousel room)
  - step 52: deep ford --[south]--> marble hall — correct
  - step 20: marble hall --[up]--> ledge in ravine — hallucinated_edge
  - step 55: marble hall --[east]--> path near stream — correct
  - step 13: carousel room --[north]--> topiary — correct
  - step 63: formal garden --[south]--> topiary — correct

## Conflict 12 — topology (overlap_mixed)
- description: position (4, -1, 1) occupied by multiple rooms ['marble hall', 'topiary']
  - step 18: carousel room --[north]--> marble hall — swapped_src_dst (LLM: carousel room--[north]-->marble hall but GT has marble hall--[south]-->carousel room)
  - step 52: deep ford --[south]--> marble hall — correct
  - step 20: marble hall --[up]--> ledge in ravine — hallucinated_edge
  - step 55: marble hall --[east]--> path near stream — correct
  - step 13: carousel room --[north]--> topiary — correct
  - step 63: formal garden --[south]--> topiary — correct

## Conflict 13 — topology (overlap_mixed)
- description: position (0, 2, 1) occupied by multiple rooms ['deep ford', 'dreary room', 'ledge in ravine', 'marble hall', 'tiny room']
  - step 51: ledge in ravine --[down]--> deep ford — correct
  - step 52: deep ford --[south]--> marble hall — correct
  - step 30: ledge in ravine --[north]--> dreary room — hallucinated_edge
  - step 32: dreary room --[south]--> tiny room — correct
  - step 20: marble hall --[up]--> ledge in ravine — hallucinated_edge
  - step 33: tiny room --[down]--> ledge in ravine — correct
  - step 50: dragon room --[east]--> ledge in ravine — hallucinated_edge
  - step 34: ledge in ravine --[west]--> end of ledge — correct
  - step 18: carousel room --[north]--> marble hall — swapped_src_dst (LLM: carousel room--[north]-->marble hall but GT has marble hall--[south]-->carousel room)
  - step 55: marble hall --[east]--> path near stream — correct

## Conflict 14 — topology (overlap_mixed)
- description: position (0, 2, 2) occupied by multiple rooms ['dreary room', 'ledge in ravine', 'tiny room']
  - step 30: ledge in ravine --[north]--> dreary room — hallucinated_edge
  - step 32: dreary room --[south]--> tiny room — correct
  - step 20: marble hall --[up]--> ledge in ravine — hallucinated_edge
  - step 33: tiny room --[down]--> ledge in ravine — correct
  - step 50: dragon room --[east]--> ledge in ravine — hallucinated_edge
  - step 34: ledge in ravine --[west]--> end of ledge — correct
  - step 51: ledge in ravine --[down]--> deep ford — correct

## Conflict 15 — topology (overlap_mixed)
- description: position (0, 1, 2) occupied by multiple rooms ['ledge in ravine', 'tiny room']
  - step 20: marble hall --[up]--> ledge in ravine — hallucinated_edge
  - step 33: tiny room --[down]--> ledge in ravine — correct
  - step 50: dragon room --[east]--> ledge in ravine — hallucinated_edge
  - step 30: ledge in ravine --[north]--> dreary room — hallucinated_edge
  - step 34: ledge in ravine --[west]--> end of ledge — correct
  - step 51: ledge in ravine --[down]--> deep ford — correct
  - step 32: dreary room --[south]--> tiny room — correct

## Conflict 16 — topology (overlap_mixed)
- description: position (0, 1, 3) occupied by multiple rooms ['ledge in ravine', 'tiny room']
  - step 20: marble hall --[up]--> ledge in ravine — hallucinated_edge
  - step 33: tiny room --[down]--> ledge in ravine — correct
  - step 50: dragon room --[east]--> ledge in ravine — hallucinated_edge
  - step 30: ledge in ravine --[north]--> dreary room — hallucinated_edge
  - step 34: ledge in ravine --[west]--> end of ledge — correct
  - step 51: ledge in ravine --[down]--> deep ford — correct
  - step 32: dreary room --[south]--> tiny room — correct

## Conflict 17 — topology (overlap_mixed)
- description: position (0, 3, 2) occupied by multiple rooms ['dreary room', 'ledge in ravine']
  - step 30: ledge in ravine --[north]--> dreary room — hallucinated_edge
  - step 32: dreary room --[south]--> tiny room — correct
  - step 20: marble hall --[up]--> ledge in ravine — hallucinated_edge
  - step 33: tiny room --[down]--> ledge in ravine — correct
  - step 50: dragon room --[east]--> ledge in ravine — hallucinated_edge
  - step 34: ledge in ravine --[west]--> end of ledge — correct
  - step 51: ledge in ravine --[down]--> deep ford — correct

## Conflict 18 — topology (overlap_mixed)
- description: position (0, 3, 1) occupied by multiple rooms ['deep ford', 'dreary room']
  - step 51: ledge in ravine --[down]--> deep ford — correct
  - step 52: deep ford --[south]--> marble hall — correct
  - step 30: ledge in ravine --[north]--> dreary room — hallucinated_edge
  - step 32: dreary room --[south]--> tiny room — correct

## Conflict 19 — topology (overlap_mixed)
- description: position (0, 2, 3) occupied by multiple rooms ['dreary room', 'tiny room']
  - step 30: ledge in ravine --[north]--> dreary room — hallucinated_edge
  - step 32: dreary room --[south]--> tiny room — correct
  - step 33: tiny room --[down]--> ledge in ravine — correct

## Conflict 20 — topology (overlap_mixed)
- description: position (-1, 1, 2) occupied by multiple rooms ['dragon room', 'end of ledge', 'stone bridge']
  - step 36: dragon room --[south]--> stone bridge — correct
  - step 45: dragon room --[north]--> dragon's lair — correct
  - step 50: dragon room --[east]--> ledge in ravine — hallucinated_edge
  - step 34: ledge in ravine --[west]--> end of ledge — correct
  - step 38: stone bridge --[south]--> cool room — correct

## Conflict 21 — topology (overlap_mixed)
- description: position (-1, 2, 1) occupied by multiple rooms ['dragon room', "dragon's lair", 'end of ledge']
  - step 36: dragon room --[south]--> stone bridge — correct
  - step 45: dragon room --[north]--> dragon's lair — correct
  - step 50: dragon room --[east]--> ledge in ravine — hallucinated_edge
  - step 34: ledge in ravine --[west]--> end of ledge — correct

## Conflict 22 — topology (overlap_mixed)
- description: position (-1, 2, 2) occupied by multiple rooms ['dragon room', "dragon's lair", 'end of ledge']
  - step 36: dragon room --[south]--> stone bridge — correct
  - step 45: dragon room --[north]--> dragon's lair — correct
  - step 50: dragon room --[east]--> ledge in ravine — hallucinated_edge
  - step 34: ledge in ravine --[west]--> end of ledge — correct

## Conflict 23 — topology (overlap_mixed)
- description: position (-1, 1, 1) occupied by multiple rooms ['dragon room', 'end of ledge', 'stone bridge']
  - step 36: dragon room --[south]--> stone bridge — correct
  - step 45: dragon room --[north]--> dragon's lair — correct
  - step 50: dragon room --[east]--> ledge in ravine — hallucinated_edge
  - step 34: ledge in ravine --[west]--> end of ledge — correct
  - step 38: stone bridge --[south]--> cool room — correct

## Conflict 24 — topology (false_positive_overlap)
- description: position (-1, 0, 1) occupied by multiple rooms ['cool room', 'stone bridge']
  - step 38: stone bridge --[south]--> cool room — correct
  - step 40: cool room --[west]--> ice room — correct
  - step 36: dragon room --[south]--> stone bridge — correct

## Conflict 25 — topology (false_positive_overlap)
- description: position (-1, 0, 2) occupied by multiple rooms ['cool room', 'stone bridge']
  - step 38: stone bridge --[south]--> cool room — correct
  - step 40: cool room --[west]--> ice room — correct
  - step 36: dragon room --[south]--> stone bridge — correct

## Conflict 26 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'carousel room' reachable at conflicting positions [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (2, -1, 1), (4, -2, 1), (6, -3, 1)]
  - step 12: path near stream --[southwest]--> carousel room — correct
  - step 13: carousel room --[north]--> topiary — correct
  - step 18: carousel room --[north]--> marble hall — swapped_src_dst (LLM: carousel room--[north]-->marble hall but GT has marble hall--[south]-->carousel room)
  - step 67: carousel room --[southeast]--> menhir room — dst_hallucinated

## Conflict 27 — naming (naming_collision_on_correct_subgraph)
- description: node 'path near stream' reachable at conflicting positions [(-1, 2, 0), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1), (3, 0, 1), (5, -1, 1)]
  - step 11: formal garden --[west]--> path near stream — correct
  - step 55: marble hall --[east]--> path near stream — correct
  - step 12: path near stream --[southwest]--> carousel room — correct

## Conflict 28 — naming (naming_collision_on_correct_subgraph)
- description: node 'topiary' reachable at conflicting positions [(0, 1, 0), (0, 1, 1), (2, 0, 1), (4, -1, 1), (6, -2, 1)]
  - step 13: carousel room --[north]--> topiary — correct
  - step 63: formal garden --[south]--> topiary — correct

## Conflict 29 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'marble hall' reachable at conflicting positions [(0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (2, 0, 1), (4, -1, 1)]
  - step 18: carousel room --[north]--> marble hall — swapped_src_dst (LLM: carousel room--[north]-->marble hall but GT has marble hall--[south]-->carousel room)
  - step 52: deep ford --[south]--> marble hall — correct
  - step 20: marble hall --[up]--> ledge in ravine — hallucinated_edge
  - step 55: marble hall --[east]--> path near stream — correct

## Conflict 30 — naming (name_hallucination)
- description: node 'menhir room' reachable at conflicting positions [(1, -1, 0), (1, -1, 1), (3, -2, 1), (5, -3, 1)]
  - step 67: carousel room --[southeast]--> menhir room — dst_hallucinated

## Conflict 31 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'ledge in ravine' reachable at conflicting positions [(0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 2), (0, 3, 2)]
  - step 20: marble hall --[up]--> ledge in ravine — hallucinated_edge
  - step 33: tiny room --[down]--> ledge in ravine — correct
  - step 50: dragon room --[east]--> ledge in ravine — hallucinated_edge
  - step 30: ledge in ravine --[north]--> dreary room — hallucinated_edge
  - step 34: ledge in ravine --[west]--> end of ledge — correct
  - step 51: ledge in ravine --[down]--> deep ford — correct

## Conflict 32 — naming (naming_collision_on_correct_subgraph)
- description: node 'deep ford' reachable at conflicting positions [(0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1), (0, 3, 0), (0, 3, 1)]
  - step 51: ledge in ravine --[down]--> deep ford — correct
  - step 52: deep ford --[south]--> marble hall — correct

## Conflict 33 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'dreary room' reachable at conflicting positions [(0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 3, 1), (0, 3, 2), (0, 3, 3)]
  - step 30: ledge in ravine --[north]--> dreary room — hallucinated_edge
  - step 32: dreary room --[south]--> tiny room — correct

## Conflict 34 — naming (naming_collision_on_correct_subgraph)
- description: node 'tiny room' reachable at conflicting positions [(0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 2), (0, 2, 3)]
  - step 32: dreary room --[south]--> tiny room — correct
  - step 33: tiny room --[down]--> ledge in ravine — correct

## Conflict 35 — naming (naming_collision_on_correct_subgraph)
- description: node 'end of ledge' reachable at conflicting positions [(-1, 1, 1), (-1, 1, 2), (-1, 2, 1), (-1, 2, 2)]
  - step 34: ledge in ravine --[west]--> end of ledge — correct

## Conflict 36 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'dragon room' reachable at conflicting positions [(-1, 1, 1), (-1, 1, 2), (-1, 2, 1), (-1, 2, 2)]
  - step 36: dragon room --[south]--> stone bridge — correct
  - step 45: dragon room --[north]--> dragon's lair — correct
  - step 50: dragon room --[east]--> ledge in ravine — hallucinated_edge

## Conflict 37 — naming (naming_collision_on_correct_subgraph)
- description: node 'stone bridge' reachable at conflicting positions [(-1, 0, 1), (-1, 0, 2), (-1, 1, 1), (-1, 1, 2)]
  - step 36: dragon room --[south]--> stone bridge — correct
  - step 38: stone bridge --[south]--> cool room — correct

## Conflict 38 — naming (naming_collision_on_correct_subgraph)
- description: node "dragon's lair" reachable at conflicting positions [(-1, 2, 1), (-1, 2, 2), (-1, 3, 1), (-1, 3, 2)]
  - step 45: dragon room --[north]--> dragon's lair — correct

## Conflict 39 — naming (naming_collision_on_correct_subgraph)
- description: node 'cool room' reachable at conflicting positions [(-1, -1, 1), (-1, -1, 2), (-1, 0, 1), (-1, 0, 2)]
  - step 38: stone bridge --[south]--> cool room — correct
  - step 40: cool room --[west]--> ice room — correct

## Conflict 40 — naming (naming_collision_on_correct_subgraph)
- description: node 'ice room' reachable at conflicting positions [(-2, -1, 1), (-2, -1, 2), (-2, 0, 1), (-2, 0, 2)]
  - step 40: cool room --[west]--> ice room — correct

## Conflict 41 — naming (naming_collision_on_correct_subgraph)
- description: node 'formal garden' reachable at conflicting positions [(0, 2, 0), (2, 1, 0), (2, 1, 1), (4, 0, 1), (6, -1, 1)]
  - step 10: north end of garden --[south]--> formal garden — correct
  - step 11: formal garden --[west]--> path near stream — correct
  - step 63: formal garden --[south]--> topiary — correct

## Conflict 42 — naming (naming_collision_on_correct_subgraph)
- description: node 'north end of garden' reachable at conflicting positions [(0, 3, 0), (2, 2, 1), (4, 1, 1), (6, 0, 1)]
  - step 7: dark tunnel --[southeast]--> north end of garden — correct
  - step 9: gazebo --[out]--> north end of garden — correct
  - step 10: north end of garden --[south]--> formal garden — correct

## Conflict 43 — naming (naming_collision_on_correct_subgraph)
- description: node 'dark tunnel' reachable at conflicting positions [(-1, 4, 0), (1, 3, 1), (3, 2, 1), (5, 1, 1)]
  - step 7: dark tunnel --[southeast]--> north end of garden — correct
