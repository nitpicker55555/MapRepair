# Conflict analysis: ballyhoo

- LLM edges: 19
- GT edges: 33
- Conflicts: 34
- Type distribution: {'direction': 1, 'topology': 15, 'naming': 18}
- Root-cause distribution: {'real_vs_hallucinated': 1, 'overlap_mixed': 9, 'false_positive_overlap': 6, 'naming_collision_on_correct_subgraph': 14, 'real_name_corrupted_by_neighbour_error': 3, 'name_hallucination': 1}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'beside the big top' has multiple outgoing edges labelled 'south'
  - step 5: beside the big top --[south]--> back yard — correct
  - step None: beside the big top --[south]--> camp, west — hallucinated_edge

## Conflict 2 — topology (overlap_mixed)
- description: position (-2, -2, 0) occupied by multiple rooms ['back yard', 'camp, west', 'connection']
  - step 5: beside the big top --[south]--> back yard — correct
  - step 6: back yard --[west]--> inside prop tent — correct
  - step 11: back yard --[east]--> near white wagon — correct
  - step 41: back yard --[south]--> camp, east — correct
  - step 42: camp, east --[west]--> camp, west — correct
  - step 45: camp, west --[south]--> clown alley — correct
  - step 51: camp, west --[north]--> beside the big top — hallucinated_edge
  - step 1: in the wings --[south]--> connection — correct
  - step 13: near white wagon --[north]--> connection — correct
  - step 3: connection --[west]--> beside the big top — correct
  - step 61: connection --[east]--> midway entrance — correct

## Conflict 3 — topology (overlap_mixed)
- description: position (0, 0, 0) occupied by multiple rooms ['back yard', 'camp, west', 'connection']
  - step 5: beside the big top --[south]--> back yard — correct
  - step 6: back yard --[west]--> inside prop tent — correct
  - step 11: back yard --[east]--> near white wagon — correct
  - step 41: back yard --[south]--> camp, east — correct
  - step 42: camp, east --[west]--> camp, west — correct
  - step 45: camp, west --[south]--> clown alley — correct
  - step 51: camp, west --[north]--> beside the big top — hallucinated_edge
  - step 1: in the wings --[south]--> connection — correct
  - step 13: near white wagon --[north]--> connection — correct
  - step 3: connection --[west]--> beside the big top — correct
  - step 61: connection --[east]--> midway entrance — correct

## Conflict 4 — topology (overlap_mixed)
- description: position (-1, -1, 0) occupied by multiple rooms ['back yard', 'camp, west', 'connection']
  - step 5: beside the big top --[south]--> back yard — correct
  - step 6: back yard --[west]--> inside prop tent — correct
  - step 11: back yard --[east]--> near white wagon — correct
  - step 41: back yard --[south]--> camp, east — correct
  - step 42: camp, east --[west]--> camp, west — correct
  - step 45: camp, west --[south]--> clown alley — correct
  - step 51: camp, west --[north]--> beside the big top — hallucinated_edge
  - step 1: in the wings --[south]--> connection — correct
  - step 13: near white wagon --[north]--> connection — correct
  - step 3: connection --[west]--> beside the big top — correct
  - step 61: connection --[east]--> midway entrance — correct

## Conflict 5 — topology (overlap_mixed)
- description: position (-3, -3, 0) occupied by multiple rooms ['back yard', 'camp, west']
  - step 5: beside the big top --[south]--> back yard — correct
  - step 6: back yard --[west]--> inside prop tent — correct
  - step 11: back yard --[east]--> near white wagon — correct
  - step 41: back yard --[south]--> camp, east — correct
  - step 42: camp, east --[west]--> camp, west — correct
  - step 45: camp, west --[south]--> clown alley — correct
  - step 51: camp, west --[north]--> beside the big top — hallucinated_edge

## Conflict 6 — topology (overlap_mixed)
- description: position (-1, 0, 0) occupied by multiple rooms ['beside the big top', 'in the wings', 'inside prop tent', 'under the bleachers']
  - step 3: connection --[west]--> beside the big top — correct
  - step 51: camp, west --[north]--> beside the big top — hallucinated_edge
  - step 5: beside the big top --[south]--> back yard — correct
  - step 1: in the wings --[south]--> connection — correct
  - step 15: in the wings --[north]--> performance ring — correct
  - step 54: in the wings --[northeast]--> under the bleachers — correct
  - step 6: back yard --[west]--> inside prop tent — correct

## Conflict 7 — topology (overlap_mixed)
- description: position (0, 1, 0) occupied by multiple rooms ['beside the big top', 'in the wings', 'under the bleachers']
  - step 3: connection --[west]--> beside the big top — correct
  - step 51: camp, west --[north]--> beside the big top — hallucinated_edge
  - step 5: beside the big top --[south]--> back yard — correct
  - step 1: in the wings --[south]--> connection — correct
  - step 15: in the wings --[north]--> performance ring — correct
  - step 54: in the wings --[northeast]--> under the bleachers — correct

## Conflict 8 — topology (overlap_mixed)
- description: position (-4, -3, 0) occupied by multiple rooms ['beside the big top', 'inside prop tent']
  - step 3: connection --[west]--> beside the big top — correct
  - step 51: camp, west --[north]--> beside the big top — hallucinated_edge
  - step 5: beside the big top --[south]--> back yard — correct
  - step 6: back yard --[west]--> inside prop tent — correct

## Conflict 9 — topology (overlap_mixed)
- description: position (-3, -2, 0) occupied by multiple rooms ['beside the big top', 'inside prop tent']
  - step 3: connection --[west]--> beside the big top — correct
  - step 51: camp, west --[north]--> beside the big top — hallucinated_edge
  - step 5: beside the big top --[south]--> back yard — correct
  - step 6: back yard --[west]--> inside prop tent — correct

## Conflict 10 — topology (overlap_mixed)
- description: position (-2, -1, 0) occupied by multiple rooms ['beside the big top', 'in the wings', 'inside prop tent']
  - step 3: connection --[west]--> beside the big top — correct
  - step 51: camp, west --[north]--> beside the big top — hallucinated_edge
  - step 5: beside the big top --[south]--> back yard — correct
  - step 1: in the wings --[south]--> connection — correct
  - step 15: in the wings --[north]--> performance ring — correct
  - step 54: in the wings --[northeast]--> under the bleachers — correct
  - step 6: back yard --[west]--> inside prop tent — correct

## Conflict 11 — topology (false_positive_overlap)
- description: position (1, 0, 0) occupied by multiple rooms ['midway entrance', 'near white wagon']
  - step 61: connection --[east]--> midway entrance — correct
  - step 63: midway entrance --[south]--> menagerie — correct
  - step 11: back yard --[east]--> near white wagon — correct
  - step 13: near white wagon --[north]--> connection — correct

## Conflict 12 — topology (false_positive_overlap)
- description: position (-2, -3, 0) occupied by multiple rooms ['camp, east', 'clown alley', 'near white wagon']
  - step 41: back yard --[south]--> camp, east — correct
  - step 42: camp, east --[west]--> camp, west — correct
  - step 45: camp, west --[south]--> clown alley — correct
  - step 11: back yard --[east]--> near white wagon — correct
  - step 13: near white wagon --[north]--> connection — correct

## Conflict 13 — topology (false_positive_overlap)
- description: position (-1, -2, 0) occupied by multiple rooms ['camp, east', 'clown alley', 'midway entrance', 'near white wagon']
  - step 41: back yard --[south]--> camp, east — correct
  - step 42: camp, east --[west]--> camp, west — correct
  - step 45: camp, west --[south]--> clown alley — correct
  - step 61: connection --[east]--> midway entrance — correct
  - step 63: midway entrance --[south]--> menagerie — correct
  - step 11: back yard --[east]--> near white wagon — correct
  - step 13: near white wagon --[north]--> connection — correct

## Conflict 14 — topology (false_positive_overlap)
- description: position (0, -1, 0) occupied by multiple rooms ['camp, east', 'midway entrance', 'near white wagon']
  - step 41: back yard --[south]--> camp, east — correct
  - step 42: camp, east --[west]--> camp, west — correct
  - step 61: connection --[east]--> midway entrance — correct
  - step 63: midway entrance --[south]--> menagerie — correct
  - step 11: back yard --[east]--> near white wagon — correct
  - step 13: near white wagon --[north]--> connection — correct

## Conflict 15 — topology (false_positive_overlap)
- description: position (-3, -4, 0) occupied by multiple rooms ['camp, east', 'clown alley']
  - step 41: back yard --[south]--> camp, east — correct
  - step 42: camp, east --[west]--> camp, west — correct
  - step 45: camp, west --[south]--> clown alley — correct

## Conflict 16 — topology (false_positive_overlap)
- description: position (1, 2, 0) occupied by multiple rooms ['in the wings', 'under the bleachers']
  - step 1: in the wings --[south]--> connection — correct
  - step 15: in the wings --[north]--> performance ring — correct
  - step 54: in the wings --[northeast]--> under the bleachers — correct

## Conflict 17 — naming (naming_collision_on_correct_subgraph)
- description: node 'back yard' reachable at conflicting positions [(-3, -3, 0), (-2, -2, 0), (-1, -1, 0), (0, 0, 0)]
  - step 5: beside the big top --[south]--> back yard — correct
  - step 6: back yard --[west]--> inside prop tent — correct
  - step 11: back yard --[east]--> near white wagon — correct
  - step 41: back yard --[south]--> camp, east — correct

## Conflict 18 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'beside the big top' reachable at conflicting positions [(-4, -3, 0), (-3, -2, 0), (-2, -1, 0), (-1, 0, 0), (0, 1, 0)]
  - step 3: connection --[west]--> beside the big top — correct
  - step 51: camp, west --[north]--> beside the big top — hallucinated_edge
  - step 5: beside the big top --[south]--> back yard — correct

## Conflict 19 — naming (naming_collision_on_correct_subgraph)
- description: node 'inside prop tent' reachable at conflicting positions [(-4, -3, 0), (-3, -2, 0), (-2, -1, 0), (-1, 0, 0)]
  - step 6: back yard --[west]--> inside prop tent — correct

## Conflict 20 — naming (naming_collision_on_correct_subgraph)
- description: node 'near white wagon' reachable at conflicting positions [(-2, -3, 0), (-1, -2, 0), (0, -1, 0), (1, 0, 0)]
  - step 11: back yard --[east]--> near white wagon — correct
  - step 13: near white wagon --[north]--> connection — correct

## Conflict 21 — naming (naming_collision_on_correct_subgraph)
- description: node 'camp, east' reachable at conflicting positions [(-3, -4, 0), (-2, -3, 0), (-1, -2, 0), (0, -1, 0)]
  - step 41: back yard --[south]--> camp, east — correct
  - step 42: camp, east --[west]--> camp, west — correct

## Conflict 22 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'camp, west' reachable at conflicting positions [(-4, -4, 0), (-3, -3, 0), (-2, -2, 0), (-1, -1, 0), (0, 0, 0)]
  - step 42: camp, east --[west]--> camp, west — correct
  - step 45: camp, west --[south]--> clown alley — correct
  - step 51: camp, west --[north]--> beside the big top — hallucinated_edge

## Conflict 23 — naming (naming_collision_on_correct_subgraph)
- description: node 'clown alley' reachable at conflicting positions [(-4, -5, 0), (-3, -4, 0), (-2, -3, 0), (-1, -2, 0)]
  - step 45: camp, west --[south]--> clown alley — correct

## Conflict 24 — naming (naming_collision_on_correct_subgraph)
- description: node 'connection' reachable at conflicting positions [(-2, -2, 0), (-1, -1, 0), (0, 0, 0), (1, 1, 0)]
  - step 1: in the wings --[south]--> connection — correct
  - step 13: near white wagon --[north]--> connection — correct
  - step 3: connection --[west]--> beside the big top — correct
  - step 61: connection --[east]--> midway entrance — correct

## Conflict 25 — naming (naming_collision_on_correct_subgraph)
- description: node 'in the wings' reachable at conflicting positions [(-2, -1, 0), (-1, 0, 0), (0, 1, 0), (1, 2, 0)]
  - step 1: in the wings --[south]--> connection — correct
  - step 15: in the wings --[north]--> performance ring — correct
  - step 54: in the wings --[northeast]--> under the bleachers — correct

## Conflict 26 — naming (naming_collision_on_correct_subgraph)
- description: node 'midway entrance' reachable at conflicting positions [(-1, -2, 0), (0, -1, 0), (1, 0, 0), (2, 1, 0)]
  - step 61: connection --[east]--> midway entrance — correct
  - step 63: midway entrance --[south]--> menagerie — correct

## Conflict 27 — naming (naming_collision_on_correct_subgraph)
- description: node 'menagerie' reachable at conflicting positions [(-1, -3, 0), (0, -2, 0), (1, -1, 0), (2, 0, 0)]
  - step 63: midway entrance --[south]--> menagerie — correct
  - step 64: menagerie --[southeast]--> menagerie nook — correct

## Conflict 28 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'menagerie nook' reachable at conflicting positions [(0, -4, 0), (1, -3, 0), (2, -2, 0), (3, -1, 0)]
  - step 64: menagerie --[southeast]--> menagerie nook — correct
  - step 69: menagerie nook --[north]--> inside cage — dst_hallucinated

## Conflict 29 — naming (name_hallucination)
- description: node 'inside cage' reachable at conflicting positions [(0, -3, 0), (1, -2, 0), (2, -1, 0), (3, 0, 0)]
  - step 69: menagerie nook --[north]--> inside cage — dst_hallucinated

## Conflict 30 — naming (naming_collision_on_correct_subgraph)
- description: node 'performance ring' reachable at conflicting positions [(-2, 0, 0), (-1, 1, 0), (0, 2, 0), (1, 3, 0)]
  - step 15: in the wings --[north]--> performance ring — correct
  - step 17: performance ring --[up]--> platform (west side) — correct

## Conflict 31 — naming (naming_collision_on_correct_subgraph)
- description: node 'under the bleachers' reachable at conflicting positions [(-1, 0, 0), (0, 1, 0), (1, 2, 0), (2, 3, 0)]
  - step 54: in the wings --[northeast]--> under the bleachers — correct

## Conflict 32 — naming (naming_collision_on_correct_subgraph)
- description: node 'platform (west side)' reachable at conflicting positions [(-2, 0, 1), (-1, 1, 1), (0, 2, 1), (1, 3, 1)]
  - step 17: performance ring --[up]--> platform (west side) — correct
  - step 18: platform (west side) --[east]--> on the tightrope — correct

## Conflict 33 — naming (naming_collision_on_correct_subgraph)
- description: node 'on the tightrope' reachable at conflicting positions [(-1, 0, 1), (0, 1, 1), (1, 2, 1), (2, 3, 1)]
  - step 18: platform (west side) --[east]--> on the tightrope — correct
  - step 23: on the tightrope --[east]--> platform (east side) — correct

## Conflict 34 — naming (naming_collision_on_correct_subgraph)
- description: node 'platform (east side)' reachable at conflicting positions [(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)]
  - step 23: on the tightrope --[east]--> platform (east side) — correct
