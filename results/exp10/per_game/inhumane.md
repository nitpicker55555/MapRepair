# Conflict analysis: inhumane

- LLM edges: 32
- GT edges: 50
- Conflicts: 55
- Type distribution: {'direction': 3, 'topology': 23, 'naming': 29}
- Root-cause distribution: {'direction_mixed': 1, 'all_hallucinated_edges': 1, 'real_vs_hallucinated': 1, 'overlap_mixed': 10, 'wrong_direction_caused_overlap': 9, 'false_positive_overlap': 2, 'name_hallucination_caused_overlap': 2, 'naming_collision_on_correct_subgraph': 16, 'real_name_corrupted_by_neighbour_error': 10, 'naming_mixed': 1, 'name_hallucination': 2}

## Conflict 1 — direction (direction_mixed)
- description: node 'round room' has multiple outgoing edges labelled 'west'
  - step 21: round room --[west]--> t-intersection (east/west/south, east to round room) — wrong_direction (LLM: 'west', GT: 'east')
  - step None: round room --[west]--> south branch — hallucinated_edge
  - step 53: round room --[west]--> corridor near pit — hallucinated_edge

## Conflict 2 — direction (all_hallucinated_edges)
- description: node 'round room' has multiple outgoing edges labelled 'east'
  - step 29: round room --[east]--> corridor at doorway — hallucinated_edge
  - step None: round room --[east]--> north stalagmite room — hallucinated_edge

## Conflict 3 — direction (real_vs_hallucinated)
- description: node 'landing' has multiple outgoing edges labelled 'south'
  - step None: landing --[south]--> end of glass hall — hallucinated_edge
  - step None: landing --[south]--> round room — correct

## Conflict 4 — topology (overlap_mixed)
- description: position (0, 0, 0) occupied by multiple rooms ['alcove at end of corridor', 'south branch']
  - step 23: south branch --[south]--> alcove at end of corridor — correct
  - step 22: t-intersection (east/west/south, east to round room) --[south]--> south branch — hallucinated_edge
  - step 28: south branch --[east]--> round room — hallucinated_edge

## Conflict 5 — topology (overlap_mixed)
- description: position (0, -8, 0) occupied by multiple rooms ['alcove at end of corridor', 'south branch']
  - step 23: south branch --[south]--> alcove at end of corridor — correct
  - step 22: t-intersection (east/west/south, east to round room) --[south]--> south branch — hallucinated_edge
  - step 28: south branch --[east]--> round room — hallucinated_edge

## Conflict 6 — topology (overlap_mixed)
- description: position (0, -4, 0) occupied by multiple rooms ['alcove at end of corridor', 'south branch']
  - step 23: south branch --[south]--> alcove at end of corridor — correct
  - step 22: t-intersection (east/west/south, east to round room) --[south]--> south branch — hallucinated_edge
  - step 28: south branch --[east]--> round room — hallucinated_edge

## Conflict 7 — topology (wrong_direction_caused_overlap)
- description: position (0, -3, 0) occupied by multiple rooms ['corridor near pit', 'glass hall', 'south branch', 't-intersection (east/west/south, east to round room)']
  - step 53: round room --[west]--> corridor near pit — hallucinated_edge
  - step 57: corridor near pit --[west]--> on the platform — correct
  - step 34: hall full of fur --[east]--> glass hall — correct
  - step 35: glass hall --[east]--> end of glass hall — correct
  - step 22: t-intersection (east/west/south, east to round room) --[south]--> south branch — hallucinated_edge
  - step 23: south branch --[south]--> alcove at end of corridor — correct
  - step 28: south branch --[east]--> round room — hallucinated_edge
  - step 21: round room --[west]--> t-intersection (east/west/south, east to round room) — wrong_direction (LLM: 'west', GT: 'east')

## Conflict 8 — topology (wrong_direction_caused_overlap)
- description: position (0, -7, 0) occupied by multiple rooms ['corridor near pit', 'glass hall', 'south branch', 't-intersection (east/west/south, east to round room)']
  - step 53: round room --[west]--> corridor near pit — hallucinated_edge
  - step 57: corridor near pit --[west]--> on the platform — correct
  - step 34: hall full of fur --[east]--> glass hall — correct
  - step 35: glass hall --[east]--> end of glass hall — correct
  - step 22: t-intersection (east/west/south, east to round room) --[south]--> south branch — hallucinated_edge
  - step 23: south branch --[south]--> alcove at end of corridor — correct
  - step 28: south branch --[east]--> round room — hallucinated_edge
  - step 21: round room --[west]--> t-intersection (east/west/south, east to round room) — wrong_direction (LLM: 'west', GT: 'east')

## Conflict 9 — topology (wrong_direction_caused_overlap)
- description: position (0, 1, 0) occupied by multiple rooms ['corridor near pit', 'south branch', 't-intersection (east/west/south, east to round room)']
  - step 53: round room --[west]--> corridor near pit — hallucinated_edge
  - step 57: corridor near pit --[west]--> on the platform — correct
  - step 22: t-intersection (east/west/south, east to round room) --[south]--> south branch — hallucinated_edge
  - step 23: south branch --[south]--> alcove at end of corridor — correct
  - step 28: south branch --[east]--> round room — hallucinated_edge
  - step 21: round room --[west]--> t-intersection (east/west/south, east to round room) — wrong_direction (LLM: 'west', GT: 'east')

## Conflict 10 — topology (wrong_direction_caused_overlap)
- description: position (0, -11, 0) occupied by multiple rooms ['corridor near pit', 'glass hall', 'south branch', 't-intersection (east/west/south, east to round room)']
  - step 53: round room --[west]--> corridor near pit — hallucinated_edge
  - step 57: corridor near pit --[west]--> on the platform — correct
  - step 34: hall full of fur --[east]--> glass hall — correct
  - step 35: glass hall --[east]--> end of glass hall — correct
  - step 22: t-intersection (east/west/south, east to round room) --[south]--> south branch — hallucinated_edge
  - step 23: south branch --[south]--> alcove at end of corridor — correct
  - step 28: south branch --[east]--> round room — hallucinated_edge
  - step 21: round room --[west]--> t-intersection (east/west/south, east to round room) — wrong_direction (LLM: 'west', GT: 'east')

## Conflict 11 — topology (wrong_direction_caused_overlap)
- description: position (1, -11, 0) occupied by multiple rooms ['end of glass hall', 'round room']
  - step 35: glass hall --[east]--> end of glass hall — correct
  - step 36: end of glass hall --[north]--> landing — hallucinated_edge
  - step 19: roboff's tent --[down]--> round room — correct
  - step 28: south branch --[east]--> round room — hallucinated_edge
  - step 45: exercise-wheel room --[north]--> round room — hallucinated_edge
  - step 52: north stalagmite room --[west]--> round room — hallucinated_edge
  - step 21: round room --[west]--> t-intersection (east/west/south, east to round room) — wrong_direction (LLM: 'west', GT: 'east')
  - step 29: round room --[east]--> corridor at doorway — hallucinated_edge
  - step 46: round room --[north]--> landing — correct
  - step 53: round room --[west]--> corridor near pit — hallucinated_edge

## Conflict 12 — topology (wrong_direction_caused_overlap)
- description: position (1, 1, 0) occupied by multiple rooms ['end of glass hall', 'round room']
  - step 35: glass hall --[east]--> end of glass hall — correct
  - step 36: end of glass hall --[north]--> landing — hallucinated_edge
  - step 19: roboff's tent --[down]--> round room — correct
  - step 28: south branch --[east]--> round room — hallucinated_edge
  - step 45: exercise-wheel room --[north]--> round room — hallucinated_edge
  - step 52: north stalagmite room --[west]--> round room — hallucinated_edge
  - step 21: round room --[west]--> t-intersection (east/west/south, east to round room) — wrong_direction (LLM: 'west', GT: 'east')
  - step 29: round room --[east]--> corridor at doorway — hallucinated_edge
  - step 46: round room --[north]--> landing — correct
  - step 53: round room --[west]--> corridor near pit — hallucinated_edge

## Conflict 13 — topology (wrong_direction_caused_overlap)
- description: position (1, 2, 0) occupied by multiple rooms ['landing', 'round room']
  - step 36: end of glass hall --[north]--> landing — hallucinated_edge
  - step 46: round room --[north]--> landing — correct
  - step 37: landing --[north]--> bottom of shaft — correct
  - step 47: landing --[east]--> t intersection — correct
  - step 19: roboff's tent --[down]--> round room — correct
  - step 28: south branch --[east]--> round room — hallucinated_edge
  - step 45: exercise-wheel room --[north]--> round room — hallucinated_edge
  - step 52: north stalagmite room --[west]--> round room — hallucinated_edge
  - step 21: round room --[west]--> t-intersection (east/west/south, east to round room) — wrong_direction (LLM: 'west', GT: 'east')
  - step 29: round room --[east]--> corridor at doorway — hallucinated_edge
  - step 53: round room --[west]--> corridor near pit — hallucinated_edge

## Conflict 14 — topology (wrong_direction_caused_overlap)
- description: position (1, -3, 0) occupied by multiple rooms ['end of glass hall', 'round room']
  - step 35: glass hall --[east]--> end of glass hall — correct
  - step 36: end of glass hall --[north]--> landing — hallucinated_edge
  - step 19: roboff's tent --[down]--> round room — correct
  - step 28: south branch --[east]--> round room — hallucinated_edge
  - step 45: exercise-wheel room --[north]--> round room — hallucinated_edge
  - step 52: north stalagmite room --[west]--> round room — hallucinated_edge
  - step 21: round room --[west]--> t-intersection (east/west/south, east to round room) — wrong_direction (LLM: 'west', GT: 'east')
  - step 29: round room --[east]--> corridor at doorway — hallucinated_edge
  - step 46: round room --[north]--> landing — correct
  - step 53: round room --[west]--> corridor near pit — hallucinated_edge

## Conflict 15 — topology (wrong_direction_caused_overlap)
- description: position (1, -7, 0) occupied by multiple rooms ['end of glass hall', 'round room']
  - step 35: glass hall --[east]--> end of glass hall — correct
  - step 36: end of glass hall --[north]--> landing — hallucinated_edge
  - step 19: roboff's tent --[down]--> round room — correct
  - step 28: south branch --[east]--> round room — hallucinated_edge
  - step 45: exercise-wheel room --[north]--> round room — hallucinated_edge
  - step 52: north stalagmite room --[west]--> round room — hallucinated_edge
  - step 21: round room --[west]--> t-intersection (east/west/south, east to round room) — wrong_direction (LLM: 'west', GT: 'east')
  - step 29: round room --[east]--> corridor at doorway — hallucinated_edge
  - step 46: round room --[north]--> landing — correct
  - step 53: round room --[west]--> corridor near pit — hallucinated_edge

## Conflict 16 — topology (overlap_mixed)
- description: position (2, -7, 0) occupied by multiple rooms ['corridor at doorway', 'north stalagmite room']
  - step 29: round room --[east]--> corridor at doorway — hallucinated_edge
  - step 30: corridor at doorway --[east]--> glue pit — correct
  - step 51: stalagmite room --[north]--> north stalagmite room — correct
  - step 52: north stalagmite room --[west]--> round room — hallucinated_edge

## Conflict 17 — topology (overlap_mixed)
- description: position (2, -11, 0) occupied by multiple rooms ['corridor at doorway', 'north stalagmite room']
  - step 29: round room --[east]--> corridor at doorway — hallucinated_edge
  - step 30: corridor at doorway --[east]--> glue pit — correct
  - step 51: stalagmite room --[north]--> north stalagmite room — correct
  - step 52: north stalagmite room --[west]--> round room — hallucinated_edge

## Conflict 18 — topology (overlap_mixed)
- description: position (2, 1, 0) occupied by multiple rooms ['corridor at doorway', 'north stalagmite room']
  - step 29: round room --[east]--> corridor at doorway — hallucinated_edge
  - step 30: corridor at doorway --[east]--> glue pit — correct
  - step 51: stalagmite room --[north]--> north stalagmite room — correct
  - step 52: north stalagmite room --[west]--> round room — hallucinated_edge

## Conflict 19 — topology (overlap_mixed)
- description: position (2, -3, 0) occupied by multiple rooms ['corridor at doorway', 'north stalagmite room']
  - step 29: round room --[east]--> corridor at doorway — hallucinated_edge
  - step 30: corridor at doorway --[east]--> glue pit — correct
  - step 51: stalagmite room --[north]--> north stalagmite room — correct
  - step 52: north stalagmite room --[west]--> round room — hallucinated_edge

## Conflict 20 — topology (overlap_mixed)
- description: position (1, -5, 1) occupied by multiple rooms ['exercise-wheel room', 'in your tent']
  - step 38: bottom of shaft --[up]--> exercise-wheel room — correct
  - step 45: exercise-wheel room --[north]--> round room — hallucinated_edge
  - step 2: in your tent --[south]--> center of camp — correct

## Conflict 21 — topology (overlap_mixed)
- description: position (1, -9, 1) occupied by multiple rooms ['exercise-wheel room', 'in your tent']
  - step 38: bottom of shaft --[up]--> exercise-wheel room — correct
  - step 45: exercise-wheel room --[north]--> round room — hallucinated_edge
  - step 2: in your tent --[south]--> center of camp — correct

## Conflict 22 — topology (overlap_mixed)
- description: position (1, -1, 1) occupied by multiple rooms ['exercise-wheel room', 'in your tent']
  - step 38: bottom of shaft --[up]--> exercise-wheel room — correct
  - step 45: exercise-wheel room --[north]--> round room — hallucinated_edge
  - step 2: in your tent --[south]--> center of camp — correct

## Conflict 23 — topology (false_positive_overlap)
- description: position (-1, -11, 0) occupied by multiple rooms ['hall full of fur', 'on the platform']
  - step 33: outside door --[east]--> hall full of fur — correct
  - step 34: hall full of fur --[east]--> glass hall — correct
  - step 57: corridor near pit --[west]--> on the platform — correct
  - step 61: on the platform --[west]--> west of the pit — correct

## Conflict 24 — topology (false_positive_overlap)
- description: position (-1, -7, 0) occupied by multiple rooms ['hall full of fur', 'on the platform']
  - step 33: outside door --[east]--> hall full of fur — correct
  - step 34: hall full of fur --[east]--> glass hall — correct
  - step 57: corridor near pit --[west]--> on the platform — correct
  - step 61: on the platform --[west]--> west of the pit — correct

## Conflict 25 — topology (name_hallucination_caused_overlap)
- description: position (-2, -11, 0) occupied by multiple rooms ['outside door', 'west of the pit']
  - step 31: glue pit --[east]--> outside door — correct
  - step 33: outside door --[east]--> hall full of fur — correct
  - step 61: on the platform --[west]--> west of the pit — correct
  - step 62: west of the pit --[north]--> key room — correct
  - step 68: west of the pit --[south]--> black room — dst_hallucinated

## Conflict 26 — topology (name_hallucination_caused_overlap)
- description: position (-2, -7, 0) occupied by multiple rooms ['outside door', 'west of the pit']
  - step 31: glue pit --[east]--> outside door — correct
  - step 33: outside door --[east]--> hall full of fur — correct
  - step 61: on the platform --[west]--> west of the pit — correct
  - step 62: west of the pit --[north]--> key room — correct
  - step 68: west of the pit --[south]--> black room — dst_hallucinated

## Conflict 27 — naming (naming_collision_on_correct_subgraph)
- description: node 'alcove at end of corridor' reachable at conflicting positions [(0, -12, 0), (0, -8, 0), (0, -4, 0), (0, 0, 0)]
  - step 23: south branch --[south]--> alcove at end of corridor — correct

## Conflict 28 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'south branch' reachable at conflicting positions [(0, -11, 0), (0, -8, 0), (0, -7, 0), (0, -4, 0), (0, -3, 0), (0, 0, 0), (0, 1, 0)]
  - step 22: t-intersection (east/west/south, east to round room) --[south]--> south branch — hallucinated_edge
  - step 23: south branch --[south]--> alcove at end of corridor — correct
  - step 28: south branch --[east]--> round room — hallucinated_edge

## Conflict 29 — naming (naming_mixed)
- description: node 't-intersection (east/west/south, east to round room)' reachable at conflicting positions [(0, -11, 0), (0, -10, 0), (0, -7, 0), (0, -6, 0), (0, -3, 0), (0, -2, 0), (0, 1, 0), (0, 2, 0)]
  - step 21: round room --[west]--> t-intersection (east/west/south, east to round room) — wrong_direction (LLM: 'west', GT: 'east')
  - step 22: t-intersection (east/west/south, east to round room) --[south]--> south branch — hallucinated_edge

## Conflict 30 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'round room' reachable at conflicting positions [(1, -11, 0), (1, -7, 0), (1, -3, 0), (1, 1, 0), (1, 2, 0)]
  - step 19: roboff's tent --[down]--> round room — correct
  - step 28: south branch --[east]--> round room — hallucinated_edge
  - step 45: exercise-wheel room --[north]--> round room — hallucinated_edge
  - step 52: north stalagmite room --[west]--> round room — hallucinated_edge
  - step 21: round room --[west]--> t-intersection (east/west/south, east to round room) — wrong_direction (LLM: 'west', GT: 'east')
  - step 29: round room --[east]--> corridor at doorway — hallucinated_edge
  - step 46: round room --[north]--> landing — correct
  - step 53: round room --[west]--> corridor near pit — hallucinated_edge

## Conflict 31 — naming (naming_collision_on_correct_subgraph)
- description: node "roboff's tent" reachable at conflicting positions [(1, -11, 1), (1, -7, 1), (1, -3, 1), (1, 1, 1)]
  - step 17: center of camp --[south]--> roboff's tent — correct
  - step 19: roboff's tent --[down]--> round room — correct

## Conflict 32 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'corridor at doorway' reachable at conflicting positions [(-4, -11, 0), (-4, -7, 0), (2, -11, 0), (2, -7, 0), (2, -3, 0), (2, 1, 0)]
  - step 29: round room --[east]--> corridor at doorway — hallucinated_edge
  - step 30: corridor at doorway --[east]--> glue pit — correct

## Conflict 33 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'exercise-wheel room' reachable at conflicting positions [(1, -12, 0), (1, -9, 1), (1, -8, 0), (1, -5, 1), (1, -4, 0), (1, -1, 1), (1, 0, 0)]
  - step 38: bottom of shaft --[up]--> exercise-wheel room — correct
  - step 45: exercise-wheel room --[north]--> round room — hallucinated_edge

## Conflict 34 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'landing' reachable at conflicting positions [(1, -14, 0), (1, -13, -1), (1, -10, 0), (1, -6, 0), (1, -2, 0), (1, 2, 0), (7, -10, 0)]
  - step 36: end of glass hall --[north]--> landing — hallucinated_edge
  - step 46: round room --[north]--> landing — correct
  - step 37: landing --[north]--> bottom of shaft — correct
  - step 47: landing --[east]--> t intersection — correct

## Conflict 35 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'north stalagmite room' reachable at conflicting positions [(2, -11, 0), (2, -7, 0), (2, -3, 0), (2, 1, 0)]
  - step 51: stalagmite room --[north]--> north stalagmite room — correct
  - step 52: north stalagmite room --[west]--> round room — hallucinated_edge

## Conflict 36 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'corridor near pit' reachable at conflicting positions [(0, -11, 0), (0, -7, 0), (0, -3, 0), (0, 1, 0)]
  - step 53: round room --[west]--> corridor near pit — hallucinated_edge
  - step 57: corridor near pit --[west]--> on the platform — correct

## Conflict 37 — naming (naming_collision_on_correct_subgraph)
- description: node 'on the platform' reachable at conflicting positions [(-1, -11, 0), (-1, -7, 0), (-1, -3, 0), (-1, 1, 0)]
  - step 57: corridor near pit --[west]--> on the platform — correct
  - step 61: on the platform --[west]--> west of the pit — correct

## Conflict 38 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'west of the pit' reachable at conflicting positions [(-2, -11, 0), (-2, -7, 0), (-2, -3, 0), (-2, 1, 0)]
  - step 61: on the platform --[west]--> west of the pit — correct
  - step 62: west of the pit --[north]--> key room — correct
  - step 68: west of the pit --[south]--> black room — dst_hallucinated

## Conflict 39 — naming (naming_collision_on_correct_subgraph)
- description: node 'key room' reachable at conflicting positions [(-2, -10, 0), (-2, -6, 0), (-2, -2, 0), (-2, 2, 0)]
  - step 62: west of the pit --[north]--> key room — correct
  - step 64: key room --[north]--> demon room — correct

## Conflict 40 — naming (name_hallucination)
- description: node 'black room' reachable at conflicting positions [(-2, -12, 0), (-2, -8, 0), (-2, -4, 0), (-2, 0, 0)]
  - step 68: west of the pit --[south]--> black room — dst_hallucinated

## Conflict 41 — naming (naming_collision_on_correct_subgraph)
- description: node 'demon room' reachable at conflicting positions [(-2, -9, 0), (-2, -5, 0), (-2, -1, 0), (-2, 3, 0)]
  - step 64: key room --[north]--> demon room — correct

## Conflict 42 — naming (naming_collision_on_correct_subgraph)
- description: node 'stalagmite room' reachable at conflicting positions [(2, -12, 0), (2, -8, 0), (2, -4, 0), (2, 0, 0)]
  - step 50: door near blades --[north]--> stalagmite room — correct
  - step 51: stalagmite room --[north]--> north stalagmite room — correct

## Conflict 43 — naming (naming_collision_on_correct_subgraph)
- description: node 'door near blades' reachable at conflicting positions [(2, -13, 0), (2, -9, 0), (2, -5, 0), (2, -1, 0)]
  - step 48: t intersection --[north]--> door near blades — correct
  - step 50: door near blades --[north]--> stalagmite room — correct

## Conflict 44 — naming (naming_collision_on_correct_subgraph)
- description: node 't intersection' reachable at conflicting positions [(2, -14, 0), (2, -10, 0), (2, -6, 0), (2, -2, 0), (2, 2, 0)]
  - step 47: landing --[east]--> t intersection — correct
  - step 48: t intersection --[north]--> door near blades — correct

## Conflict 45 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'end of glass hall' reachable at conflicting positions [(1, -11, 0), (1, -7, 0), (1, -3, 0), (1, 1, 0), (7, -11, 0), (7, -7, 0)]
  - step 35: glass hall --[east]--> end of glass hall — correct
  - step 36: end of glass hall --[north]--> landing — hallucinated_edge

## Conflict 46 — naming (naming_collision_on_correct_subgraph)
- description: node 'bottom of shaft' reachable at conflicting positions [(1, -12, -1), (1, -9, 0), (1, -8, -1), (1, -5, 0), (1, -4, -1), (1, -1, 0), (1, 0, -1), (1, 3, 0)]
  - step 37: landing --[north]--> bottom of shaft — correct
  - step 38: bottom of shaft --[up]--> exercise-wheel room — correct

## Conflict 47 — naming (naming_collision_on_correct_subgraph)
- description: node 'glue pit' reachable at conflicting positions [(-3, -11, 0), (-3, -7, 0), (3, -11, 0), (3, -7, 0), (3, -3, 0), (3, 1, 0)]
  - step 30: corridor at doorway --[east]--> glue pit — correct
  - step 31: glue pit --[east]--> outside door — correct

## Conflict 48 — naming (naming_collision_on_correct_subgraph)
- description: node 'outside door' reachable at conflicting positions [(-2, -11, 0), (-2, -7, 0), (4, -11, 0), (4, -7, 0)]
  - step 31: glue pit --[east]--> outside door — correct
  - step 33: outside door --[east]--> hall full of fur — correct

## Conflict 49 — naming (naming_collision_on_correct_subgraph)
- description: node 'hall full of fur' reachable at conflicting positions [(-1, -11, 0), (-1, -7, 0), (5, -11, 0), (5, -7, 0)]
  - step 33: outside door --[east]--> hall full of fur — correct
  - step 34: hall full of fur --[east]--> glass hall — correct

## Conflict 50 — naming (naming_collision_on_correct_subgraph)
- description: node 'glass hall' reachable at conflicting positions [(0, -11, 0), (0, -7, 0), (0, -3, 0), (6, -11, 0), (6, -7, 0)]
  - step 34: hall full of fur --[east]--> glass hall — correct
  - step 35: glass hall --[east]--> end of glass hall — correct

## Conflict 51 — naming (naming_collision_on_correct_subgraph)
- description: node 'center of camp' reachable at conflicting positions [(1, -10, 1), (1, -6, 1), (1, -2, 1), (1, 2, 1)]
  - step 2: in your tent --[south]--> center of camp — correct
  - step 3: center of camp --[west]--> storage tent — correct
  - step 7: center of camp --[east]--> in the desert — correct
  - step 17: center of camp --[south]--> roboff's tent — correct

## Conflict 52 — naming (naming_collision_on_correct_subgraph)
- description: node 'in your tent' reachable at conflicting positions [(1, -9, 1), (1, -5, 1), (1, -1, 1), (1, 3, 1)]
  - step 2: in your tent --[south]--> center of camp — correct

## Conflict 53 — naming (naming_collision_on_correct_subgraph)
- description: node 'storage tent' reachable at conflicting positions [(0, -10, 1), (0, -6, 1), (0, -2, 1), (0, 2, 1)]
  - step 3: center of camp --[west]--> storage tent — correct

## Conflict 54 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'in the desert' reachable at conflicting positions [(2, -10, 1), (2, -6, 1), (2, -2, 1), (2, 2, 1)]
  - step 7: center of camp --[east]--> in the desert — correct
  - step 12: in the desert --[south]--> bottom of a hole — dst_hallucinated

## Conflict 55 — naming (name_hallucination)
- description: node 'bottom of a hole' reachable at conflicting positions [(2, -11, 1), (2, -7, 1), (2, -3, 1), (2, 1, 1)]
  - step 12: in the desert --[south]--> bottom of a hole — dst_hallucinated
