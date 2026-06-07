# Conflict analysis: adventureland

- LLM edges: 20
- GT edges: 27
- Conflicts: 36
- Type distribution: {'direction': 3, 'topology': 14, 'naming': 19}
- Root-cause distribution: {'name_hallucination': 3, 'src_dst_swap': 1, 'real_vs_hallucinated': 1, 'name_hallucination_caused_overlap': 8, 'overlap_mixed': 5, 'false_positive_overlap': 1, 'real_name_corrupted_by_neighbour_error': 6, 'naming_collision_on_correct_subgraph': 11}

## Conflict 1 — direction (name_hallucination)
- description: node 'hole' has multiple outgoing edges labelled 'down'
  - step 30: hole --[down]--> maze (exit north, south, east, down) — hallucinated_edge
  - step 64: hole --[down]--> darkness — dst_hallucinated

## Conflict 2 — direction (src_dst_swap)
- description: node 'maze (exit north, south, east, down)' has multiple outgoing edges labelled 'up'
  - step None: maze (exit north, south, east, down) --[up]--> hole — hallucinated_edge
  - step None: maze (exit north, south, east, down) --[up]--> cavern — swapped_src_dst (LLM: maze (exit north, south, east, down)--[up]-->cavern but GT has cavern--[down]-->maze (exit north, south, east, down))

## Conflict 3 — direction (real_vs_hallucinated)
- description: node 'maze (exit north, south, east, down)' has multiple outgoing edges labelled 'down'
  - step 31: maze (exit north, south, east, down) --[down]--> hall — hallucinated_edge
  - step 37: maze (exit north, south, east, down) --[down]--> maze (exit west, up) — correct

## Conflict 4 — topology (name_hallucination_caused_overlap)
- description: position (0, 1, -9) occupied by multiple rooms ['cavern', 'hole']
  - step 32: hall --[down]--> cavern — correct
  - step 33: cavern --[south]--> anteroom — correct
  - step 36: cavern --[down]--> maze (exit north, south, east, down) — correct
  - step 26: root chamber --[down]--> hole — correct
  - step 30: hole --[down]--> maze (exit north, south, east, down) — hallucinated_edge
  - step 64: hole --[down]--> darkness — dst_hallucinated

## Conflict 5 — topology (name_hallucination_caused_overlap)
- description: position (0, 1, -3) occupied by multiple rooms ['cavern', 'hole']
  - step 32: hall --[down]--> cavern — correct
  - step 33: cavern --[south]--> anteroom — correct
  - step 36: cavern --[down]--> maze (exit north, south, east, down) — correct
  - step 26: root chamber --[down]--> hole — correct
  - step 30: hole --[down]--> maze (exit north, south, east, down) — hallucinated_edge
  - step 64: hole --[down]--> darkness — dst_hallucinated

## Conflict 6 — topology (name_hallucination_caused_overlap)
- description: position (0, 1, -6) occupied by multiple rooms ['cavern', 'dismal swamp', 'hole']
  - step 32: hall --[down]--> cavern — correct
  - step 33: cavern --[south]--> anteroom — correct
  - step 36: cavern --[down]--> maze (exit north, south, east, down) — correct
  - step 8: lakeside --[west]--> dismal swamp — hallucinated_edge
  - step 14: tree top --[down]--> dismal swamp — correct
  - step 56: inside stump --[up]--> dismal swamp — correct
  - step 49: dismal swamp --[north]--> meadow — correct
  - step 26: root chamber --[down]--> hole — correct
  - step 30: hole --[down]--> maze (exit north, south, east, down) — hallucinated_edge
  - step 64: hole --[down]--> darkness — dst_hallucinated

## Conflict 7 — topology (name_hallucination_caused_overlap)
- description: position (0, 1, 0) occupied by multiple rooms ['cavern', 'hole']
  - step 32: hall --[down]--> cavern — correct
  - step 33: cavern --[south]--> anteroom — correct
  - step 36: cavern --[down]--> maze (exit north, south, east, down) — correct
  - step 26: root chamber --[down]--> hole — correct
  - step 30: hole --[down]--> maze (exit north, south, east, down) — hallucinated_edge
  - step 64: hole --[down]--> darkness — dst_hallucinated

## Conflict 8 — topology (overlap_mixed)
- description: position (0, 1, -11) occupied by multiple rooms ['hall', 'maze (exit west, up)']
  - step 31: maze (exit north, south, east, down) --[down]--> hall — hallucinated_edge
  - step 32: hall --[down]--> cavern — correct
  - step 37: maze (exit north, south, east, down) --[down]--> maze (exit west, up) — correct
  - step 39: maze (exit west, up) --[west]--> maze (exit north, south, east, west, up, down; rock aladdin) — correct

## Conflict 9 — topology (overlap_mixed)
- description: position (0, 1, -8) occupied by multiple rooms ['hall', 'maze (exit west, up)', 'root chamber']
  - step 31: maze (exit north, south, east, down) --[down]--> hall — hallucinated_edge
  - step 32: hall --[down]--> cavern — correct
  - step 37: maze (exit north, south, east, down) --[down]--> maze (exit west, up) — correct
  - step 39: maze (exit west, up) --[west]--> maze (exit north, south, east, west, up, down; rock aladdin) — correct
  - step 21: inside stump --[down]--> root chamber — correct
  - step 26: root chamber --[down]--> hole — correct

## Conflict 10 — topology (overlap_mixed)
- description: position (0, 1, -2) occupied by multiple rooms ['hall', 'maze (exit west, up)', 'root chamber']
  - step 31: maze (exit north, south, east, down) --[down]--> hall — hallucinated_edge
  - step 32: hall --[down]--> cavern — correct
  - step 37: maze (exit north, south, east, down) --[down]--> maze (exit west, up) — correct
  - step 39: maze (exit west, up) --[west]--> maze (exit north, south, east, west, up, down; rock aladdin) — correct
  - step 21: inside stump --[down]--> root chamber — correct
  - step 26: root chamber --[down]--> hole — correct

## Conflict 11 — topology (overlap_mixed)
- description: position (0, 1, -5) occupied by multiple rooms ['hall', 'maze (exit west, up)', 'root chamber', 'tree top']
  - step 31: maze (exit north, south, east, down) --[down]--> hall — hallucinated_edge
  - step 32: hall --[down]--> cavern — correct
  - step 37: maze (exit north, south, east, down) --[down]--> maze (exit west, up) — correct
  - step 39: maze (exit west, up) --[west]--> maze (exit north, south, east, west, up, down; rock aladdin) — correct
  - step 21: inside stump --[down]--> root chamber — correct
  - step 26: root chamber --[down]--> hole — correct
  - step 14: tree top --[down]--> dismal swamp — correct

## Conflict 12 — topology (overlap_mixed)
- description: position (0, 1, 1) occupied by multiple rooms ['hall', 'root chamber']
  - step 31: maze (exit north, south, east, down) --[down]--> hall — hallucinated_edge
  - step 32: hall --[down]--> cavern — correct
  - step 21: inside stump --[down]--> root chamber — correct
  - step 26: root chamber --[down]--> hole — correct

## Conflict 13 — topology (name_hallucination_caused_overlap)
- description: position (0, 1, -4) occupied by multiple rooms ['darkness', 'maze (exit north, south, east, down)']
  - step 64: hole --[down]--> darkness — dst_hallucinated
  - step 30: hole --[down]--> maze (exit north, south, east, down) — hallucinated_edge
  - step 36: cavern --[down]--> maze (exit north, south, east, down) — correct
  - step 31: maze (exit north, south, east, down) --[down]--> hall — hallucinated_edge
  - step 37: maze (exit north, south, east, down) --[down]--> maze (exit west, up) — correct

## Conflict 14 — topology (name_hallucination_caused_overlap)
- description: position (0, 1, -1) occupied by multiple rooms ['darkness', 'maze (exit north, south, east, down)']
  - step 64: hole --[down]--> darkness — dst_hallucinated
  - step 30: hole --[down]--> maze (exit north, south, east, down) — hallucinated_edge
  - step 36: cavern --[down]--> maze (exit north, south, east, down) — correct
  - step 31: maze (exit north, south, east, down) --[down]--> hall — hallucinated_edge
  - step 37: maze (exit north, south, east, down) --[down]--> maze (exit west, up) — correct

## Conflict 15 — topology (name_hallucination_caused_overlap)
- description: position (0, 1, -7) occupied by multiple rooms ['darkness', 'inside stump', 'maze (exit north, south, east, down)']
  - step 64: hole --[down]--> darkness — dst_hallucinated
  - step 21: inside stump --[down]--> root chamber — correct
  - step 56: inside stump --[up]--> dismal swamp — correct
  - step 30: hole --[down]--> maze (exit north, south, east, down) — hallucinated_edge
  - step 36: cavern --[down]--> maze (exit north, south, east, down) — correct
  - step 31: maze (exit north, south, east, down) --[down]--> hall — hallucinated_edge
  - step 37: maze (exit north, south, east, down) --[down]--> maze (exit west, up) — correct

## Conflict 16 — topology (name_hallucination_caused_overlap)
- description: position (0, 1, -10) occupied by multiple rooms ['darkness', 'maze (exit north, south, east, down)']
  - step 64: hole --[down]--> darkness — dst_hallucinated
  - step 30: hole --[down]--> maze (exit north, south, east, down) — hallucinated_edge
  - step 36: cavern --[down]--> maze (exit north, south, east, down) — correct
  - step 31: maze (exit north, south, east, down) --[down]--> hall — hallucinated_edge
  - step 37: maze (exit north, south, east, down) --[down]--> maze (exit west, up) — correct

## Conflict 17 — topology (false_positive_overlap)
- description: position (-1, 2, -6) occupied by multiple rooms ['chasm', 'forest']
  - step 42: maze (exit north, south, east, west, up, down; arrow down) --[down]--> chasm — correct
  - step 1: forest --[east]--> meadow — correct

## Conflict 18 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'anteroom' reachable at conflicting positions [(0, 0, -9), (0, 0, -6), (0, 0, -3), (0, 0, 0)]
  - step 33: cavern --[south]--> anteroom — correct
  - step 68: anteroom --[up]--> royal chamber — dst_hallucinated

## Conflict 19 — naming (naming_collision_on_correct_subgraph)
- description: node 'cavern' reachable at conflicting positions [(0, 1, -9), (0, 1, -6), (0, 1, -3), (0, 1, 0)]
  - step 32: hall --[down]--> cavern — correct
  - step 33: cavern --[south]--> anteroom — correct
  - step 36: cavern --[down]--> maze (exit north, south, east, down) — correct

## Conflict 20 — naming (name_hallucination)
- description: node 'royal chamber' reachable at conflicting positions [(0, 0, -8), (0, 0, -5), (0, 0, -2), (0, 0, 1)]
  - step 68: anteroom --[up]--> royal chamber — dst_hallucinated

## Conflict 21 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'hall' reachable at conflicting positions [(0, 1, -11), (0, 1, -8), (0, 1, -5), (0, 1, -2), (0, 1, 1)]
  - step 31: maze (exit north, south, east, down) --[down]--> hall — hallucinated_edge
  - step 32: hall --[down]--> cavern — correct

## Conflict 22 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'maze (exit north, south, east, down)' reachable at conflicting positions [(0, 1, -10), (0, 1, -7), (0, 1, -4), (0, 1, -1), (0, 1, 2)]
  - step 30: hole --[down]--> maze (exit north, south, east, down) — hallucinated_edge
  - step 36: cavern --[down]--> maze (exit north, south, east, down) — correct
  - step 31: maze (exit north, south, east, down) --[down]--> hall — hallucinated_edge
  - step 37: maze (exit north, south, east, down) --[down]--> maze (exit west, up) — correct

## Conflict 23 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'hole' reachable at conflicting positions [(-3, 7, -9), (-2, 5, -9), (-1, 3, -9), (0, 1, -9), (0, 1, -6), (0, 1, -3), (0, 1, 0)]
  - step 26: root chamber --[down]--> hole — correct
  - step 30: hole --[down]--> maze (exit north, south, east, down) — hallucinated_edge
  - step 64: hole --[down]--> darkness — dst_hallucinated

## Conflict 24 — naming (naming_collision_on_correct_subgraph)
- description: node 'maze (exit west, up)' reachable at conflicting positions [(0, 1, -11), (0, 1, -8), (0, 1, -5), (0, 1, -2)]
  - step 37: maze (exit north, south, east, down) --[down]--> maze (exit west, up) — correct
  - step 39: maze (exit west, up) --[west]--> maze (exit north, south, east, west, up, down; rock aladdin) — correct

## Conflict 25 — naming (naming_collision_on_correct_subgraph)
- description: node 'maze (exit north, south, east, west, up, down; rock aladdin)' reachable at conflicting positions [(-1, 1, -11), (-1, 1, -8), (-1, 1, -5), (-1, 1, -2)]
  - step 39: maze (exit west, up) --[west]--> maze (exit north, south, east, west, up, down; rock aladdin) — correct
  - step 40: maze (exit north, south, east, west, up, down; rock aladdin) --[north]--> maze (exit north, south, east, west, up, down; arrow down) — correct

## Conflict 26 — naming (naming_collision_on_correct_subgraph)
- description: node 'maze (exit north, south, east, west, up, down; arrow down)' reachable at conflicting positions [(-1, 2, -11), (-1, 2, -8), (-1, 2, -5), (-1, 2, -2)]
  - step 40: maze (exit north, south, east, west, up, down; rock aladdin) --[north]--> maze (exit north, south, east, west, up, down; arrow down) — correct
  - step 42: maze (exit north, south, east, west, up, down; arrow down) --[down]--> chasm — correct

## Conflict 27 — naming (naming_collision_on_correct_subgraph)
- description: node 'chasm' reachable at conflicting positions [(-1, 2, -12), (-1, 2, -9), (-1, 2, -6), (-1, 2, -3)]
  - step 42: maze (exit north, south, east, west, up, down; arrow down) --[down]--> chasm — correct

## Conflict 28 — naming (naming_collision_on_correct_subgraph)
- description: node 'root chamber' reachable at conflicting positions [(-3, 7, -8), (-2, 5, -8), (-1, 3, -8), (0, 1, -8), (0, 1, -5), (0, 1, -2), (0, 1, 1)]
  - step 21: inside stump --[down]--> root chamber — correct
  - step 26: root chamber --[down]--> hole — correct

## Conflict 29 — naming (name_hallucination)
- description: node 'darkness' reachable at conflicting positions [(0, 1, -10), (0, 1, -7), (0, 1, -4), (0, 1, -1)]
  - step 64: hole --[down]--> darkness — dst_hallucinated

## Conflict 30 — naming (naming_collision_on_correct_subgraph)
- description: node 'inside stump' reachable at conflicting positions [(-3, 7, -7), (-2, 5, -7), (-1, 3, -7), (0, 1, -7)]
  - step 21: inside stump --[down]--> root chamber — correct
  - step 56: inside stump --[up]--> dismal swamp — correct

## Conflict 31 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'dismal swamp' reachable at conflicting positions [(-3, 7, -6), (-2, 5, -6), (-1, 3, -6), (0, 1, -6)]
  - step 8: lakeside --[west]--> dismal swamp — hallucinated_edge
  - step 14: tree top --[down]--> dismal swamp — correct
  - step 56: inside stump --[up]--> dismal swamp — correct
  - step 49: dismal swamp --[north]--> meadow — correct

## Conflict 32 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'lakeside' reachable at conflicting positions [(-3, 9, -6), (-2, 7, -6), (-1, 5, -6), (0, 3, -6), (1, 1, -6)]
  - step 2: meadow --[north]--> lakeside — correct
  - step 4: lakeside --[down]--> bog — correct
  - step 8: lakeside --[west]--> dismal swamp — hallucinated_edge

## Conflict 33 — naming (naming_collision_on_correct_subgraph)
- description: node 'tree top' reachable at conflicting positions [(-3, 7, -5), (-2, 5, -5), (-1, 3, -5), (0, 1, -5)]
  - step 14: tree top --[down]--> dismal swamp — correct

## Conflict 34 — naming (naming_collision_on_correct_subgraph)
- description: node 'meadow' reachable at conflicting positions [(-3, 8, -6), (-2, 6, -6), (-1, 4, -6), (0, 2, -6), (1, 0, -6)]
  - step 1: forest --[east]--> meadow — correct
  - step 49: dismal swamp --[north]--> meadow — correct
  - step 2: meadow --[north]--> lakeside — correct

## Conflict 35 — naming (naming_collision_on_correct_subgraph)
- description: node 'forest' reachable at conflicting positions [(-4, 8, -6), (-3, 6, -6), (-2, 4, -6), (-1, 2, -6)]
  - step 1: forest --[east]--> meadow — correct

## Conflict 36 — naming (naming_collision_on_correct_subgraph)
- description: node 'bog' reachable at conflicting positions [(-2, 7, -7), (-1, 5, -7), (0, 3, -7), (1, 1, -7)]
  - step 4: lakeside --[down]--> bog — correct
