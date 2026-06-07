# Conflict analysis: omniquest

- LLM edges: 31
- GT edges: 57
- Conflicts: 65
- Type distribution: {'direction': 2, 'topology': 34, 'naming': 29}
- Root-cause distribution: {'real_vs_hallucinated': 2, 'false_positive_overlap': 12, 'overlap_mixed': 15, 'name_hallucination_caused_overlap': 7, 'naming_collision_on_correct_subgraph': 22, 'real_name_corrupted_by_neighbour_error': 6, 'name_hallucination': 1}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'intersection' has multiple outgoing edges labelled 'northwest'
  - step None: intersection --[northwest]--> twisting passageway — hallucinated_edge
  - step 66: intersection --[northwest]--> slimy passageway — correct

## Conflict 2 — direction (real_vs_hallucinated)
- description: node 'intersection' has multiple outgoing edges labelled 'southwest'
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step None: intersection --[southwest]--> west path — correct

## Conflict 3 — topology (false_positive_overlap)
- description: position (2, 1, 0) occupied by multiple rooms ['blue ridge mountain sports', 'giant cavern', 'small clearing', 'small path']
  - step 33: twisting passageway --[north]--> blue ridge mountain sports — correct
  - step 35: blue ridge mountain sports --[east]--> dirt cave — correct
  - step 29: cylindrical room --[southeast]--> giant cavern — correct
  - step 67: slimy passageway --[north]--> giant cavern — correct
  - step 30: giant cavern --[northeast]--> small cavern — correct
  - step 16: canyon path --[north]--> small clearing — correct
  - step 2: large clearing --[east]--> small path — correct
  - step 3: small path --[south]--> fork in path — correct
  - step 14: small path --[east]--> canyon entrance — correct

## Conflict 4 — topology (false_positive_overlap)
- description: position (6, 3, 0) occupied by multiple rooms ['blue ridge mountain sports', 'giant cavern', 'small clearing']
  - step 33: twisting passageway --[north]--> blue ridge mountain sports — correct
  - step 35: blue ridge mountain sports --[east]--> dirt cave — correct
  - step 29: cylindrical room --[southeast]--> giant cavern — correct
  - step 67: slimy passageway --[north]--> giant cavern — correct
  - step 30: giant cavern --[northeast]--> small cavern — correct
  - step 16: canyon path --[north]--> small clearing — correct

## Conflict 5 — topology (false_positive_overlap)
- description: position (0, 0, 0) occupied by multiple rooms ['blue ridge mountain sports', 'giant cavern', 'small clearing', 'small path']
  - step 33: twisting passageway --[north]--> blue ridge mountain sports — correct
  - step 35: blue ridge mountain sports --[east]--> dirt cave — correct
  - step 29: cylindrical room --[southeast]--> giant cavern — correct
  - step 67: slimy passageway --[north]--> giant cavern — correct
  - step 30: giant cavern --[northeast]--> small cavern — correct
  - step 16: canyon path --[north]--> small clearing — correct
  - step 2: large clearing --[east]--> small path — correct
  - step 3: small path --[south]--> fork in path — correct
  - step 14: small path --[east]--> canyon entrance — correct

## Conflict 6 — topology (false_positive_overlap)
- description: position (4, 2, 0) occupied by multiple rooms ['blue ridge mountain sports', 'giant cavern', 'small clearing', 'small path']
  - step 33: twisting passageway --[north]--> blue ridge mountain sports — correct
  - step 35: blue ridge mountain sports --[east]--> dirt cave — correct
  - step 29: cylindrical room --[southeast]--> giant cavern — correct
  - step 67: slimy passageway --[north]--> giant cavern — correct
  - step 30: giant cavern --[northeast]--> small cavern — correct
  - step 16: canyon path --[north]--> small clearing — correct
  - step 2: large clearing --[east]--> small path — correct
  - step 3: small path --[south]--> fork in path — correct
  - step 14: small path --[east]--> canyon entrance — correct

## Conflict 7 — topology (overlap_mixed)
- description: position (4, 1, 0) occupied by multiple rooms ['canyon path', 'fork in path', 'slimy passageway', 'twisting passageway']
  - step 15: canyon entrance --[east]--> canyon path — correct
  - step 16: canyon path --[north]--> small clearing — correct
  - step 3: small path --[south]--> fork in path — correct
  - step 4: fork in path --[southwest]--> end of path(an exit to the northeast) — correct
  - step 7: fork in path --[southeast]--> end of path(an exit to the northeast and a raincoat here) — correct
  - step 66: intersection --[northwest]--> slimy passageway — correct
  - step 67: slimy passageway --[north]--> giant cavern — correct
  - step 31: small cavern --[southeast]--> twisting passageway — correct
  - step 33: twisting passageway --[north]--> blue ridge mountain sports — correct
  - step 39: twisting passageway --[southeast]--> intersection — hallucinated_edge

## Conflict 8 — topology (overlap_mixed)
- description: position (0, -1, 0) occupied by multiple rooms ['canyon path', 'fork in path', 'slimy passageway', 'twisting passageway']
  - step 15: canyon entrance --[east]--> canyon path — correct
  - step 16: canyon path --[north]--> small clearing — correct
  - step 3: small path --[south]--> fork in path — correct
  - step 4: fork in path --[southwest]--> end of path(an exit to the northeast) — correct
  - step 7: fork in path --[southeast]--> end of path(an exit to the northeast and a raincoat here) — correct
  - step 66: intersection --[northwest]--> slimy passageway — correct
  - step 67: slimy passageway --[north]--> giant cavern — correct
  - step 31: small cavern --[southeast]--> twisting passageway — correct
  - step 33: twisting passageway --[north]--> blue ridge mountain sports — correct
  - step 39: twisting passageway --[southeast]--> intersection — hallucinated_edge

## Conflict 9 — topology (overlap_mixed)
- description: position (6, 2, 0) occupied by multiple rooms ['canyon path', 'slimy passageway', 'twisting passageway']
  - step 15: canyon entrance --[east]--> canyon path — correct
  - step 16: canyon path --[north]--> small clearing — correct
  - step 66: intersection --[northwest]--> slimy passageway — correct
  - step 67: slimy passageway --[north]--> giant cavern — correct
  - step 31: small cavern --[southeast]--> twisting passageway — correct
  - step 33: twisting passageway --[north]--> blue ridge mountain sports — correct
  - step 39: twisting passageway --[southeast]--> intersection — hallucinated_edge

## Conflict 10 — topology (overlap_mixed)
- description: position (2, 0, 0) occupied by multiple rooms ['canyon path', 'fork in path', 'slimy passageway', 'twisting passageway']
  - step 15: canyon entrance --[east]--> canyon path — correct
  - step 16: canyon path --[north]--> small clearing — correct
  - step 3: small path --[south]--> fork in path — correct
  - step 4: fork in path --[southwest]--> end of path(an exit to the northeast) — correct
  - step 7: fork in path --[southeast]--> end of path(an exit to the northeast and a raincoat here) — correct
  - step 66: intersection --[northwest]--> slimy passageway — correct
  - step 67: slimy passageway --[north]--> giant cavern — correct
  - step 31: small cavern --[southeast]--> twisting passageway — correct
  - step 33: twisting passageway --[north]--> blue ridge mountain sports — correct
  - step 39: twisting passageway --[southeast]--> intersection — hallucinated_edge

## Conflict 11 — topology (false_positive_overlap)
- description: position (1, 0, 0) occupied by multiple rooms ['canyon entrance', 'dirt cave']
  - step 14: small path --[east]--> canyon entrance — correct
  - step 15: canyon entrance --[east]--> canyon path — correct
  - step 20: canyon entrance --[northwest]--> endless beach — correct
  - step 35: blue ridge mountain sports --[east]--> dirt cave — correct

## Conflict 12 — topology (false_positive_overlap)
- description: position (5, 2, 0) occupied by multiple rooms ['canyon entrance', 'dirt cave']
  - step 14: small path --[east]--> canyon entrance — correct
  - step 15: canyon entrance --[east]--> canyon path — correct
  - step 20: canyon entrance --[northwest]--> endless beach — correct
  - step 35: blue ridge mountain sports --[east]--> dirt cave — correct

## Conflict 13 — topology (false_positive_overlap)
- description: position (3, 1, 0) occupied by multiple rooms ['canyon entrance', 'dirt cave']
  - step 14: small path --[east]--> canyon entrance — correct
  - step 15: canyon entrance --[east]--> canyon path — correct
  - step 20: canyon entrance --[northwest]--> endless beach — correct
  - step 35: blue ridge mountain sports --[east]--> dirt cave — correct

## Conflict 14 — topology (name_hallucination_caused_overlap)
- description: position (-1, 0, 0) occupied by multiple rooms ['large clearing', 'small cavern']
  - step 2: large clearing --[east]--> small path — correct
  - step 30: giant cavern --[northeast]--> small cavern — correct
  - step 31: small cavern --[southeast]--> twisting passageway — correct
  - step 69: small cavern --[north]--> organ room — dst_hallucinated

## Conflict 15 — topology (name_hallucination_caused_overlap)
- description: position (1, 1, 0) occupied by multiple rooms ['large clearing', 'small cavern']
  - step 2: large clearing --[east]--> small path — correct
  - step 30: giant cavern --[northeast]--> small cavern — correct
  - step 31: small cavern --[southeast]--> twisting passageway — correct
  - step 69: small cavern --[north]--> organ room — dst_hallucinated

## Conflict 16 — topology (name_hallucination_caused_overlap)
- description: position (3, 2, 0) occupied by multiple rooms ['large clearing', 'small cavern']
  - step 2: large clearing --[east]--> small path — correct
  - step 30: giant cavern --[northeast]--> small cavern — correct
  - step 31: small cavern --[southeast]--> twisting passageway — correct
  - step 69: small cavern --[north]--> organ room — dst_hallucinated

## Conflict 17 — topology (overlap_mixed)
- description: position (1, -2, 0) occupied by multiple rooms ['end of path(an exit to the northeast and a raincoat here)', 'guardian chamber', 'intersection']
  - step 7: fork in path --[southeast]--> end of path(an exit to the northeast and a raincoat here) — correct
  - step 57: west path --[west]--> guardian chamber — correct
  - step 59: guardian chamber --[west]--> jeweled room — correct
  - step 39: twisting passageway --[southeast]--> intersection — hallucinated_edge
  - step 65: west path --[northeast]--> intersection — correct
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step 66: intersection --[northwest]--> slimy passageway — correct

## Conflict 18 — topology (overlap_mixed)
- description: position (3, -1, 0) occupied by multiple rooms ['end of path(an exit to the northeast and a raincoat here)', 'guardian chamber', 'intersection']
  - step 7: fork in path --[southeast]--> end of path(an exit to the northeast and a raincoat here) — correct
  - step 57: west path --[west]--> guardian chamber — correct
  - step 59: guardian chamber --[west]--> jeweled room — correct
  - step 39: twisting passageway --[southeast]--> intersection — hallucinated_edge
  - step 65: west path --[northeast]--> intersection — correct
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step 66: intersection --[northwest]--> slimy passageway — correct

## Conflict 19 — topology (overlap_mixed)
- description: position (5, 0, 0) occupied by multiple rooms ['end of path(an exit to the northeast and a raincoat here)', 'guardian chamber', 'intersection']
  - step 7: fork in path --[southeast]--> end of path(an exit to the northeast and a raincoat here) — correct
  - step 57: west path --[west]--> guardian chamber — correct
  - step 59: guardian chamber --[west]--> jeweled room — correct
  - step 39: twisting passageway --[southeast]--> intersection — hallucinated_edge
  - step 65: west path --[northeast]--> intersection — correct
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step 66: intersection --[northwest]--> slimy passageway — correct

## Conflict 20 — topology (overlap_mixed)
- description: position (2, -2, 0) occupied by multiple rooms ['south lake', 'west path']
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step 55: musty corridor --[north]--> south lake — correct
  - step 44: south lake --[southeast]--> chilly corridor — correct
  - step 56: south lake --[northwest]--> west path — correct
  - step 57: west path --[west]--> guardian chamber — correct
  - step 65: west path --[northeast]--> intersection — correct

## Conflict 21 — topology (overlap_mixed)
- description: position (3, -3, 0) occupied by multiple rooms ['chilly corridor', 'south lake']
  - step 44: south lake --[southeast]--> chilly corridor — correct
  - step 46: chilly corridor --[southeast]--> ice room — correct
  - step 51: chilly corridor --[southwest]--> rice room — correct
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step 55: musty corridor --[north]--> south lake — correct
  - step 56: south lake --[northwest]--> west path — correct

## Conflict 22 — topology (overlap_mixed)
- description: position (0, -3, 0) occupied by multiple rooms ['south lake', 'west path']
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step 55: musty corridor --[north]--> south lake — correct
  - step 44: south lake --[southeast]--> chilly corridor — correct
  - step 56: south lake --[northwest]--> west path — correct
  - step 57: west path --[west]--> guardian chamber — correct
  - step 65: west path --[northeast]--> intersection — correct

## Conflict 23 — topology (overlap_mixed)
- description: position (6, 0, 0) occupied by multiple rooms ['south lake', 'west path']
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step 55: musty corridor --[north]--> south lake — correct
  - step 44: south lake --[southeast]--> chilly corridor — correct
  - step 56: south lake --[northwest]--> west path — correct
  - step 57: west path --[west]--> guardian chamber — correct
  - step 65: west path --[northeast]--> intersection — correct

## Conflict 24 — topology (overlap_mixed)
- description: position (1, -4, 0) occupied by multiple rooms ['chilly corridor', 'south lake']
  - step 44: south lake --[southeast]--> chilly corridor — correct
  - step 46: chilly corridor --[southeast]--> ice room — correct
  - step 51: chilly corridor --[southwest]--> rice room — correct
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step 55: musty corridor --[north]--> south lake — correct
  - step 56: south lake --[northwest]--> west path — correct

## Conflict 25 — topology (overlap_mixed)
- description: position (7, -1, 0) occupied by multiple rooms ['chilly corridor', 'south lake']
  - step 44: south lake --[southeast]--> chilly corridor — correct
  - step 46: chilly corridor --[southeast]--> ice room — correct
  - step 51: chilly corridor --[southwest]--> rice room — correct
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step 55: musty corridor --[north]--> south lake — correct
  - step 56: south lake --[northwest]--> west path — correct

## Conflict 26 — topology (overlap_mixed)
- description: position (4, -1, 0) occupied by multiple rooms ['south lake', 'west path']
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step 55: musty corridor --[north]--> south lake — correct
  - step 44: south lake --[southeast]--> chilly corridor — correct
  - step 56: south lake --[northwest]--> west path — correct
  - step 57: west path --[west]--> guardian chamber — correct
  - step 65: west path --[northeast]--> intersection — correct

## Conflict 27 — topology (overlap_mixed)
- description: position (5, -2, 0) occupied by multiple rooms ['chilly corridor', 'south lake']
  - step 44: south lake --[southeast]--> chilly corridor — correct
  - step 46: chilly corridor --[southeast]--> ice room — correct
  - step 51: chilly corridor --[southwest]--> rice room — correct
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step 55: musty corridor --[north]--> south lake — correct
  - step 56: south lake --[northwest]--> west path — correct

## Conflict 28 — topology (false_positive_overlap)
- description: position (3, 0, 0) occupied by multiple rooms ['end of path(an exit to the northeast)', 'west path']
  - step 4: fork in path --[southwest]--> end of path(an exit to the northeast) — correct
  - step 56: south lake --[northwest]--> west path — correct
  - step 57: west path --[west]--> guardian chamber — correct
  - step 65: west path --[northeast]--> intersection — correct

## Conflict 29 — topology (false_positive_overlap)
- description: position (1, -1, 0) occupied by multiple rooms ['end of path(an exit to the northeast)', 'west path']
  - step 4: fork in path --[southwest]--> end of path(an exit to the northeast) — correct
  - step 56: south lake --[northwest]--> west path — correct
  - step 57: west path --[west]--> guardian chamber — correct
  - step 65: west path --[northeast]--> intersection — correct

## Conflict 30 — topology (false_positive_overlap)
- description: position (-1, -2, 0) occupied by multiple rooms ['end of path(an exit to the northeast)', 'west path']
  - step 4: fork in path --[southwest]--> end of path(an exit to the northeast) — correct
  - step 56: south lake --[northwest]--> west path — correct
  - step 57: west path --[west]--> guardian chamber — correct
  - step 65: west path --[northeast]--> intersection — correct

## Conflict 31 — topology (false_positive_overlap)
- description: position (-2, -1, 0) occupied by multiple rooms ['giant cavern', 'small path']
  - step 29: cylindrical room --[southeast]--> giant cavern — correct
  - step 67: slimy passageway --[north]--> giant cavern — correct
  - step 30: giant cavern --[northeast]--> small cavern — correct
  - step 2: large clearing --[east]--> small path — correct
  - step 3: small path --[south]--> fork in path — correct
  - step 14: small path --[east]--> canyon entrance — correct

## Conflict 32 — topology (name_hallucination_caused_overlap)
- description: position (3, 3, 0) occupied by multiple rooms ['cylindrical room', 'organ room']
  - step 27: island coast --[southeast]--> cylindrical room — hallucinated_edge
  - step 29: cylindrical room --[southeast]--> giant cavern — correct
  - step 69: small cavern --[north]--> organ room — dst_hallucinated

## Conflict 33 — topology (name_hallucination_caused_overlap)
- description: position (5, 4, 0) occupied by multiple rooms ['cylindrical room', 'organ room']
  - step 27: island coast --[southeast]--> cylindrical room — hallucinated_edge
  - step 29: cylindrical room --[southeast]--> giant cavern — correct
  - step 69: small cavern --[north]--> organ room — dst_hallucinated

## Conflict 34 — topology (name_hallucination_caused_overlap)
- description: position (-1, 1, 0) occupied by multiple rooms ['cylindrical room', 'organ room']
  - step 27: island coast --[southeast]--> cylindrical room — hallucinated_edge
  - step 29: cylindrical room --[southeast]--> giant cavern — correct
  - step 69: small cavern --[north]--> organ room — dst_hallucinated

## Conflict 35 — topology (name_hallucination_caused_overlap)
- description: position (1, 2, 0) occupied by multiple rooms ['cylindrical room', 'organ room']
  - step 27: island coast --[southeast]--> cylindrical room — hallucinated_edge
  - step 29: cylindrical room --[southeast]--> giant cavern — correct
  - step 69: small cavern --[north]--> organ room — dst_hallucinated

## Conflict 36 — topology (false_positive_overlap)
- description: position (-1, -3, 0) occupied by multiple rooms ['end of path(an exit to the northeast and a raincoat here)', 'guardian chamber']
  - step 7: fork in path --[southeast]--> end of path(an exit to the northeast and a raincoat here) — correct
  - step 57: west path --[west]--> guardian chamber — correct
  - step 59: guardian chamber --[west]--> jeweled room — correct

## Conflict 37 — naming (naming_collision_on_correct_subgraph)
- description: node 'blue ridge mountain sports' reachable at conflicting positions [(0, 0, 0), (2, 1, 0), (4, 2, 0), (6, 3, 0)]
  - step 33: twisting passageway --[north]--> blue ridge mountain sports — correct
  - step 35: blue ridge mountain sports --[east]--> dirt cave — correct

## Conflict 38 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'twisting passageway' reachable at conflicting positions [(0, -1, 0), (2, 0, 0), (4, 1, 0), (6, 2, 0)]
  - step 31: small cavern --[southeast]--> twisting passageway — correct
  - step 33: twisting passageway --[north]--> blue ridge mountain sports — correct
  - step 39: twisting passageway --[southeast]--> intersection — hallucinated_edge

## Conflict 39 — naming (naming_collision_on_correct_subgraph)
- description: node 'dirt cave' reachable at conflicting positions [(1, 0, 0), (3, 1, 0), (5, 2, 0), (7, 3, 0)]
  - step 35: blue ridge mountain sports --[east]--> dirt cave — correct

## Conflict 40 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'small cavern' reachable at conflicting positions [(-1, 0, 0), (1, 1, 0), (3, 2, 0), (5, 3, 0), (7, 4, 0)]
  - step 30: giant cavern --[northeast]--> small cavern — correct
  - step 31: small cavern --[southeast]--> twisting passageway — correct
  - step 69: small cavern --[north]--> organ room — dst_hallucinated

## Conflict 41 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'intersection' reachable at conflicting positions [(1, -2, 0), (3, -1, 0), (5, 0, 0), (7, 1, 0)]
  - step 39: twisting passageway --[southeast]--> intersection — hallucinated_edge
  - step 65: west path --[northeast]--> intersection — correct
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step 66: intersection --[northwest]--> slimy passageway — correct

## Conflict 42 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'south lake' reachable at conflicting positions [(0, -3, 0), (1, -4, 0), (2, -2, 0), (3, -3, 0), (4, -1, 0), (5, -2, 0), (6, 0, 0), (7, -1, 0)]
  - step 43: intersection --[southwest]--> south lake — hallucinated_edge
  - step 55: musty corridor --[north]--> south lake — correct
  - step 44: south lake --[southeast]--> chilly corridor — correct
  - step 56: south lake --[northwest]--> west path — correct

## Conflict 43 — naming (naming_collision_on_correct_subgraph)
- description: node 'west path' reachable at conflicting positions [(-1, -2, 0), (0, -3, 0), (1, -1, 0), (2, -2, 0), (3, 0, 0), (4, -1, 0), (5, 1, 0), (6, 0, 0)]
  - step 56: south lake --[northwest]--> west path — correct
  - step 57: west path --[west]--> guardian chamber — correct
  - step 65: west path --[northeast]--> intersection — correct

## Conflict 44 — naming (naming_collision_on_correct_subgraph)
- description: node 'slimy passageway' reachable at conflicting positions [(0, -1, 0), (2, 0, 0), (4, 1, 0), (6, 2, 0)]
  - step 66: intersection --[northwest]--> slimy passageway — correct
  - step 67: slimy passageway --[north]--> giant cavern — correct

## Conflict 45 — naming (naming_collision_on_correct_subgraph)
- description: node 'giant cavern' reachable at conflicting positions [(-2, -1, 0), (0, 0, 0), (2, 1, 0), (4, 2, 0), (6, 3, 0)]
  - step 29: cylindrical room --[southeast]--> giant cavern — correct
  - step 67: slimy passageway --[north]--> giant cavern — correct
  - step 30: giant cavern --[northeast]--> small cavern — correct

## Conflict 46 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'cylindrical room' reachable at conflicting positions [(-1, 1, 0), (1, 2, 0), (3, 3, 0), (5, 4, 0)]
  - step 27: island coast --[southeast]--> cylindrical room — hallucinated_edge
  - step 29: cylindrical room --[southeast]--> giant cavern — correct

## Conflict 47 — naming (name_hallucination)
- description: node 'organ room' reachable at conflicting positions [(-1, 1, 0), (1, 2, 0), (3, 3, 0), (5, 4, 0)]
  - step 69: small cavern --[north]--> organ room — dst_hallucinated

## Conflict 48 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'island coast' reachable at conflicting positions [(-2, 2, 0), (0, 3, 0), (2, 4, 0), (4, 5, 0)]
  - step 23: ocean --[north]--> island coast — correct
  - step 24: island coast --[north]--> sandy beach — correct
  - step 27: island coast --[southeast]--> cylindrical room — hallucinated_edge

## Conflict 49 — naming (naming_collision_on_correct_subgraph)
- description: node 'ocean' reachable at conflicting positions [(-2, 1, 0), (0, 2, 0), (2, 3, 0), (4, 4, 0)]
  - step 22: endless beach --[north]--> ocean — correct
  - step 23: ocean --[north]--> island coast — correct

## Conflict 50 — naming (naming_collision_on_correct_subgraph)
- description: node 'sandy beach' reachable at conflicting positions [(-2, 3, 0), (0, 4, 0), (2, 5, 0), (4, 6, 0)]
  - step 24: island coast --[north]--> sandy beach — correct

## Conflict 51 — naming (naming_collision_on_correct_subgraph)
- description: node 'endless beach' reachable at conflicting positions [(-2, 0, 0), (0, 1, 0), (2, 2, 0), (4, 3, 0)]
  - step 20: canyon entrance --[northwest]--> endless beach — correct
  - step 22: endless beach --[north]--> ocean — correct

## Conflict 52 — naming (naming_collision_on_correct_subgraph)
- description: node 'canyon entrance' reachable at conflicting positions [(-1, -1, 0), (1, 0, 0), (3, 1, 0), (5, 2, 0)]
  - step 14: small path --[east]--> canyon entrance — correct
  - step 15: canyon entrance --[east]--> canyon path — correct
  - step 20: canyon entrance --[northwest]--> endless beach — correct

## Conflict 53 — naming (naming_collision_on_correct_subgraph)
- description: node 'small path' reachable at conflicting positions [(-2, -1, 0), (0, 0, 0), (2, 1, 0), (4, 2, 0)]
  - step 2: large clearing --[east]--> small path — correct
  - step 3: small path --[south]--> fork in path — correct
  - step 14: small path --[east]--> canyon entrance — correct

## Conflict 54 — naming (naming_collision_on_correct_subgraph)
- description: node 'canyon path' reachable at conflicting positions [(0, -1, 0), (2, 0, 0), (4, 1, 0), (6, 2, 0)]
  - step 15: canyon entrance --[east]--> canyon path — correct
  - step 16: canyon path --[north]--> small clearing — correct

## Conflict 55 — naming (naming_collision_on_correct_subgraph)
- description: node 'small clearing' reachable at conflicting positions [(0, 0, 0), (2, 1, 0), (4, 2, 0), (6, 3, 0)]
  - step 16: canyon path --[north]--> small clearing — correct

## Conflict 56 — naming (naming_collision_on_correct_subgraph)
- description: node 'large clearing' reachable at conflicting positions [(-3, -1, 0), (-1, 0, 0), (1, 1, 0), (3, 2, 0)]
  - step 2: large clearing --[east]--> small path — correct

## Conflict 57 — naming (naming_collision_on_correct_subgraph)
- description: node 'fork in path' reachable at conflicting positions [(-2, -2, 0), (0, -1, 0), (2, 0, 0), (4, 1, 0)]
  - step 3: small path --[south]--> fork in path — correct
  - step 4: fork in path --[southwest]--> end of path(an exit to the northeast) — correct
  - step 7: fork in path --[southeast]--> end of path(an exit to the northeast and a raincoat here) — correct

## Conflict 58 — naming (naming_collision_on_correct_subgraph)
- description: node 'end of path(an exit to the northeast)' reachable at conflicting positions [(-3, -3, 0), (-1, -2, 0), (1, -1, 0), (3, 0, 0)]
  - step 4: fork in path --[southwest]--> end of path(an exit to the northeast) — correct

## Conflict 59 — naming (naming_collision_on_correct_subgraph)
- description: node 'end of path(an exit to the northeast and a raincoat here)' reachable at conflicting positions [(-1, -3, 0), (1, -2, 0), (3, -1, 0), (5, 0, 0)]
  - step 7: fork in path --[southeast]--> end of path(an exit to the northeast and a raincoat here) — correct

## Conflict 60 — naming (naming_collision_on_correct_subgraph)
- description: node 'guardian chamber' reachable at conflicting positions [(-1, -3, 0), (1, -2, 0), (3, -1, 0), (5, 0, 0)]
  - step 57: west path --[west]--> guardian chamber — correct
  - step 59: guardian chamber --[west]--> jeweled room — correct

## Conflict 61 — naming (naming_collision_on_correct_subgraph)
- description: node 'jeweled room' reachable at conflicting positions [(-2, -3, 0), (0, -2, 0), (2, -1, 0), (4, 0, 0)]
  - step 59: guardian chamber --[west]--> jeweled room — correct

## Conflict 62 — naming (naming_collision_on_correct_subgraph)
- description: node 'chilly corridor' reachable at conflicting positions [(1, -4, 0), (3, -3, 0), (5, -2, 0), (7, -1, 0)]
  - step 44: south lake --[southeast]--> chilly corridor — correct
  - step 46: chilly corridor --[southeast]--> ice room — correct
  - step 51: chilly corridor --[southwest]--> rice room — correct

## Conflict 63 — naming (naming_collision_on_correct_subgraph)
- description: node 'musty corridor' reachable at conflicting positions [(0, -4, 0), (2, -3, 0), (4, -2, 0), (6, -1, 0)]
  - step 54: rice room --[north]--> musty corridor — correct
  - step 55: musty corridor --[north]--> south lake — correct

## Conflict 64 — naming (naming_collision_on_correct_subgraph)
- description: node 'rice room' reachable at conflicting positions [(0, -5, 0), (2, -4, 0), (4, -3, 0), (6, -2, 0)]
  - step 51: chilly corridor --[southwest]--> rice room — correct
  - step 54: rice room --[north]--> musty corridor — correct

## Conflict 65 — naming (naming_collision_on_correct_subgraph)
- description: node 'ice room' reachable at conflicting positions [(2, -5, 0), (4, -4, 0), (6, -3, 0), (8, -2, 0)]
  - step 46: chilly corridor --[southeast]--> ice room — correct
