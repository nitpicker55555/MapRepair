# Conflict analysis: zork3

- LLM edges: 24
- GT edges: 38
- Conflicts: 50
- Type distribution: {'direction': 3, 'topology': 25, 'naming': 22}
- Root-cause distribution: {'real_vs_hallucinated': 1, 'name_hallucination': 3, 'src_dst_swap': 1, 'overlap_mixed': 3, 'false_positive_overlap': 6, 'name_hallucination_caused_overlap': 16, 'naming_collision_on_correct_subgraph': 12, 'real_name_corrupted_by_neighbour_error': 7, 'naming_mixed': 1}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'pitch black 1' has multiple outgoing edges labelled 'north'
  - step None: pitch black 1 --[north]--> endless stair — hallucinated_edge
  - step None: pitch black 1 --[north]--> southern shore — correct

## Conflict 2 — direction (name_hallucination)
- description: node 'western shore' has multiple outgoing edges labelled 'south'
  - step 26: western shore --[south]--> scenic vista — correct
  - step None: western shore --[south]--> viewing room — dst_hallucinated

## Conflict 3 — direction (src_dst_swap)
- description: node 'southern shore' has multiple outgoing edges labelled 'north'
  - step None: southern shore --[north]--> on the lake — swapped_src_dst (LLM: southern shore--[north]-->on the lake but GT has on the lake--[south]-->southern shore)
  - step None: southern shore --[north]--> cliff base — dst_hallucinated

## Conflict 4 — topology (overlap_mixed)
- description: position (-1, 0, 1) occupied by multiple rooms ['key room', 'pitch black 1']
  - step 43: pitch black 1 --[east]--> key room — hallucinated_edge
  - step 46: key room --[down]--> aqueduct — correct
  - step 2: endless stair --[south]--> pitch black 1 — hallucinated_edge
  - step 40: southern shore --[south]--> pitch black 1 — correct

## Conflict 5 — topology (overlap_mixed)
- description: position (3, -4, 5) occupied by multiple rooms ['barren area', 'key room']
  - step 4: junction --[west]--> barren area — correct
  - step 5: barren area --[west]--> cliff — correct
  - step 43: pitch black 1 --[east]--> key room — hallucinated_edge
  - step 46: key room --[down]--> aqueduct — correct

## Conflict 6 — topology (false_positive_overlap)
- description: position (-1, 2, 0) occupied by multiple rooms ['creepy crawl', 'water slide']
  - step 18: junction --[south]--> creepy crawl — correct
  - step 19: creepy crawl --[south]--> foggy room — correct
  - step 48: high arch --[north]--> water slide — correct
  - step 49: water slide --[north]--> damp passage — correct

## Conflict 7 — topology (false_positive_overlap)
- description: position (2, -1, 4) occupied by multiple rooms ['damp passage', 'junction']
  - step 9: junction --[east]--> damp passage — correct
  - step 12: dead end --[west]--> damp passage — correct
  - step 49: water slide --[north]--> damp passage — correct
  - step 10: damp passage --[northeast]--> engravings room — correct
  - step 4: junction --[west]--> barren area — correct
  - step 18: junction --[south]--> creepy crawl — correct

## Conflict 8 — topology (false_positive_overlap)
- description: position (-1, 3, 0) occupied by multiple rooms ['damp passage', 'junction']
  - step 9: junction --[east]--> damp passage — correct
  - step 12: dead end --[west]--> damp passage — correct
  - step 49: water slide --[north]--> damp passage — correct
  - step 10: damp passage --[northeast]--> engravings room — correct
  - step 4: junction --[west]--> barren area — correct
  - step 18: junction --[south]--> creepy crawl — correct

## Conflict 9 — topology (false_positive_overlap)
- description: position (0, 3, 0) occupied by multiple rooms ['damp passage', 'dead end']
  - step 9: junction --[east]--> damp passage — correct
  - step 12: dead end --[west]--> damp passage — correct
  - step 49: water slide --[north]--> damp passage — correct
  - step 10: damp passage --[northeast]--> engravings room — correct
  - step 11: engravings room --[southeast]--> dead end — correct

## Conflict 10 — topology (false_positive_overlap)
- description: position (4, -1, 4) occupied by multiple rooms ['damp passage', 'dead end']
  - step 9: junction --[east]--> damp passage — correct
  - step 12: dead end --[west]--> damp passage — correct
  - step 49: water slide --[north]--> damp passage — correct
  - step 10: damp passage --[northeast]--> engravings room — correct
  - step 11: engravings room --[southeast]--> dead end — correct

## Conflict 11 — topology (false_positive_overlap)
- description: position (-2, 3, 0) occupied by multiple rooms ['barren area', 'junction']
  - step 4: junction --[west]--> barren area — correct
  - step 5: barren area --[west]--> cliff — correct
  - step 9: junction --[east]--> damp passage — correct
  - step 18: junction --[south]--> creepy crawl — correct

## Conflict 12 — topology (overlap_mixed)
- description: position (2, -4, 5) occupied by multiple rooms ['cliff', 'pitch black 1']
  - step 5: barren area --[west]--> cliff — correct
  - step 54: cliff --[down]--> cliff ledge — correct
  - step 2: endless stair --[south]--> pitch black 1 — hallucinated_edge
  - step 40: southern shore --[south]--> pitch black 1 — correct
  - step 43: pitch black 1 --[east]--> key room — hallucinated_edge

## Conflict 13 — topology (name_hallucination_caused_overlap)
- description: position (0, -2, 2) occupied by multiple rooms ['endless stair', 'southern shore']
  - step 2: endless stair --[south]--> pitch black 1 — hallucinated_edge
  - step 39: on the lake --[south]--> southern shore — correct
  - step 65: cliff base --[south]--> southern shore — src_hallucinated
  - step 40: southern shore --[south]--> pitch black 1 — correct

## Conflict 14 — topology (name_hallucination_caused_overlap)
- description: position (2, -5, 3) occupied by multiple rooms ['endless stair', 'southern shore']
  - step 2: endless stair --[south]--> pitch black 1 — hallucinated_edge
  - step 39: on the lake --[south]--> southern shore — correct
  - step 65: cliff base --[south]--> southern shore — src_hallucinated
  - step 40: southern shore --[south]--> pitch black 1 — correct

## Conflict 15 — topology (name_hallucination_caused_overlap)
- description: position (2, -3, 5) occupied by multiple rooms ['endless stair', 'southern shore']
  - step 2: endless stair --[south]--> pitch black 1 — hallucinated_edge
  - step 39: on the lake --[south]--> southern shore — correct
  - step 65: cliff base --[south]--> southern shore — src_hallucinated
  - step 40: southern shore --[south]--> pitch black 1 — correct

## Conflict 16 — topology (name_hallucination_caused_overlap)
- description: position (-2, 1, 1) occupied by multiple rooms ['endless stair', 'southern shore']
  - step 2: endless stair --[south]--> pitch black 1 — hallucinated_edge
  - step 39: on the lake --[south]--> southern shore — correct
  - step 65: cliff base --[south]--> southern shore — src_hallucinated
  - step 40: southern shore --[south]--> pitch black 1 — correct

## Conflict 17 — topology (name_hallucination_caused_overlap)
- description: position (-2, 2, 1) occupied by multiple rooms ['cliff base', 'on the lake']
  - step 64: cliff ledge --[down]--> cliff base — dst_hallucinated
  - step 65: cliff base --[south]--> southern shore — src_hallucinated
  - step 23: lake shore --[down]--> on the lake — hallucinated_edge
  - step 25: on the lake --[west]--> western shore — correct
  - step 39: on the lake --[south]--> southern shore — correct

## Conflict 18 — topology (name_hallucination_caused_overlap)
- description: position (2, -2, 5) occupied by multiple rooms ['cliff base', 'on the lake']
  - step 64: cliff ledge --[down]--> cliff base — dst_hallucinated
  - step 65: cliff base --[south]--> southern shore — src_hallucinated
  - step 23: lake shore --[down]--> on the lake — hallucinated_edge
  - step 25: on the lake --[west]--> western shore — correct
  - step 39: on the lake --[south]--> southern shore — correct

## Conflict 19 — topology (name_hallucination_caused_overlap)
- description: position (0, -1, 2) occupied by multiple rooms ['cliff base', 'on the lake']
  - step 64: cliff ledge --[down]--> cliff base — dst_hallucinated
  - step 65: cliff base --[south]--> southern shore — src_hallucinated
  - step 23: lake shore --[down]--> on the lake — hallucinated_edge
  - step 25: on the lake --[west]--> western shore — correct
  - step 39: on the lake --[south]--> southern shore — correct

## Conflict 20 — topology (name_hallucination_caused_overlap)
- description: position (2, -4, 3) occupied by multiple rooms ['cliff base', 'on the lake']
  - step 64: cliff ledge --[down]--> cliff base — dst_hallucinated
  - step 65: cliff base --[south]--> southern shore — src_hallucinated
  - step 23: lake shore --[down]--> on the lake — hallucinated_edge
  - step 25: on the lake --[west]--> western shore — correct
  - step 39: on the lake --[south]--> southern shore — correct

## Conflict 21 — topology (name_hallucination_caused_overlap)
- description: position (0, -1, 3) occupied by multiple rooms ['cliff ledge', 'lake shore']
  - step 54: cliff --[down]--> cliff ledge — correct
  - step 64: cliff ledge --[down]--> cliff base — dst_hallucinated
  - step 20: foggy room --[south]--> lake shore — correct
  - step 23: lake shore --[down]--> on the lake — hallucinated_edge

## Conflict 22 — topology (name_hallucination_caused_overlap)
- description: position (2, -4, 4) occupied by multiple rooms ['cliff ledge', 'lake shore']
  - step 54: cliff --[down]--> cliff ledge — correct
  - step 64: cliff ledge --[down]--> cliff base — dst_hallucinated
  - step 20: foggy room --[south]--> lake shore — correct
  - step 23: lake shore --[down]--> on the lake — hallucinated_edge

## Conflict 23 — topology (name_hallucination_caused_overlap)
- description: position (-2, 2, 2) occupied by multiple rooms ['cliff ledge', 'lake shore']
  - step 54: cliff --[down]--> cliff ledge — correct
  - step 64: cliff ledge --[down]--> cliff base — dst_hallucinated
  - step 20: foggy room --[south]--> lake shore — correct
  - step 23: lake shore --[down]--> on the lake — hallucinated_edge

## Conflict 24 — topology (name_hallucination_caused_overlap)
- description: position (2, -2, 6) occupied by multiple rooms ['cliff ledge', 'lake shore']
  - step 54: cliff --[down]--> cliff ledge — correct
  - step 64: cliff ledge --[down]--> cliff base — dst_hallucinated
  - step 20: foggy room --[south]--> lake shore — correct
  - step 23: lake shore --[down]--> on the lake — hallucinated_edge

## Conflict 25 — topology (name_hallucination_caused_overlap)
- description: position (-3, 1, 1) occupied by multiple rooms ['scenic vista', 'viewing room']
  - step 26: western shore --[south]--> scenic vista — correct
  - step 36: viewing room --[north]--> western shore — src_hallucinated

## Conflict 26 — topology (name_hallucination_caused_overlap)
- description: position (1, -5, 3) occupied by multiple rooms ['scenic vista', 'viewing room']
  - step 26: western shore --[south]--> scenic vista — correct
  - step 36: viewing room --[north]--> western shore — src_hallucinated

## Conflict 27 — topology (name_hallucination_caused_overlap)
- description: position (-1, -2, 2) occupied by multiple rooms ['scenic vista', 'viewing room']
  - step 26: western shore --[south]--> scenic vista — correct
  - step 36: viewing room --[north]--> western shore — src_hallucinated

## Conflict 28 — topology (name_hallucination_caused_overlap)
- description: position (1, -3, 5) occupied by multiple rooms ['scenic vista', 'viewing room']
  - step 26: western shore --[south]--> scenic vista — correct
  - step 36: viewing room --[north]--> western shore — src_hallucinated

## Conflict 29 — naming (naming_collision_on_correct_subgraph)
- description: node 'aqueduct' reachable at conflicting positions [(-1, 0, 0), (0, 0, 0), (3, -6, 2), (3, -4, 4)]
  - step 46: key room --[down]--> aqueduct — correct
  - step 47: aqueduct --[north]--> high arch — correct

## Conflict 30 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'key room' reachable at conflicting positions [(-1, 0, 1), (0, 0, 1), (1, -3, 2), (3, -6, 3), (3, -4, 5)]
  - step 43: pitch black 1 --[east]--> key room — hallucinated_edge
  - step 46: key room --[down]--> aqueduct — correct

## Conflict 31 — naming (naming_collision_on_correct_subgraph)
- description: node 'high arch' reachable at conflicting positions [(-1, 1, 0), (0, 1, 0), (3, -5, 2), (3, -3, 4)]
  - step 47: aqueduct --[north]--> high arch — correct
  - step 48: high arch --[north]--> water slide — correct

## Conflict 32 — naming (naming_collision_on_correct_subgraph)
- description: node 'water slide' reachable at conflicting positions [(-1, 2, 0), (0, 2, 0), (1, 1, 3), (3, -4, 2), (3, -2, 4)]
  - step 48: high arch --[north]--> water slide — correct
  - step 49: water slide --[north]--> damp passage — correct

## Conflict 33 — naming (naming_collision_on_correct_subgraph)
- description: node 'damp passage' reachable at conflicting positions [(-1, 3, 0), (0, 3, 0), (1, 2, 3), (2, -1, 4), (3, -3, 2), (3, -1, 4), (4, -1, 4)]
  - step 9: junction --[east]--> damp passage — correct
  - step 12: dead end --[west]--> damp passage — correct
  - step 49: water slide --[north]--> damp passage — correct
  - step 10: damp passage --[northeast]--> engravings room — correct

## Conflict 34 — naming (naming_collision_on_correct_subgraph)
- description: node 'junction' reachable at conflicting positions [(-2, 3, 0), (-2, 5, 2), (-1, 3, 0), (0, 2, 3), (2, -1, 4), (2, 1, 6), (4, -4, 5), (4, -2, 7)]
  - step 4: junction --[west]--> barren area — correct
  - step 9: junction --[east]--> damp passage — correct
  - step 18: junction --[south]--> creepy crawl — correct

## Conflict 35 — naming (naming_collision_on_correct_subgraph)
- description: node 'engravings room' reachable at conflicting positions [(-1, 4, 0), (0, 4, 0), (1, 4, 0), (2, 3, 3), (3, 0, 4), (4, 0, 4)]
  - step 10: damp passage --[northeast]--> engravings room — correct
  - step 11: engravings room --[southeast]--> dead end — correct

## Conflict 36 — naming (naming_collision_on_correct_subgraph)
- description: node 'dead end' reachable at conflicting positions [(0, 3, 0), (1, 3, 0), (2, 2, 3), (2, 3, 0), (4, -1, 4), (5, -1, 4)]
  - step 11: engravings room --[southeast]--> dead end — correct
  - step 12: dead end --[west]--> damp passage — correct

## Conflict 37 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'pitch black 1' reachable at conflicting positions [(-2, 0, 1), (-1, 0, 1), (0, -3, 2), (2, -6, 3), (2, -4, 5)]
  - step 2: endless stair --[south]--> pitch black 1 — hallucinated_edge
  - step 40: southern shore --[south]--> pitch black 1 — correct
  - step 43: pitch black 1 --[east]--> key room — hallucinated_edge

## Conflict 38 — naming (naming_mixed)
- description: node 'endless stair' reachable at conflicting positions [(-2, 1, 1), (0, -2, 2), (2, -5, 3), (2, -3, 5)]
  - step 2: endless stair --[south]--> pitch black 1 — hallucinated_edge

## Conflict 39 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'southern shore' reachable at conflicting positions [(-2, 1, 1), (0, -2, 2), (2, -5, 3), (2, -3, 5)]
  - step 39: on the lake --[south]--> southern shore — correct
  - step 65: cliff base --[south]--> southern shore — src_hallucinated
  - step 40: southern shore --[south]--> pitch black 1 — correct

## Conflict 40 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'on the lake' reachable at conflicting positions [(-2, 2, 1), (0, -1, 2), (2, -4, 3), (2, -2, 5)]
  - step 23: lake shore --[down]--> on the lake — hallucinated_edge
  - step 25: on the lake --[west]--> western shore — correct
  - step 39: on the lake --[south]--> southern shore — correct

## Conflict 41 — naming (name_hallucination)
- description: node 'cliff base' reachable at conflicting positions [(-2, 2, 1), (0, -1, 2), (2, -4, 3), (2, -2, 5)]
  - step 64: cliff ledge --[down]--> cliff base — dst_hallucinated
  - step 65: cliff base --[south]--> southern shore — src_hallucinated

## Conflict 42 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'cliff ledge' reachable at conflicting positions [(-2, 2, 2), (0, -1, 3), (2, -4, 4), (2, -2, 6)]
  - step 54: cliff --[down]--> cliff ledge — correct
  - step 64: cliff ledge --[down]--> cliff base — dst_hallucinated

## Conflict 43 — naming (naming_collision_on_correct_subgraph)
- description: node 'cliff' reachable at conflicting positions [(-2, 2, 3), (0, -1, 4), (2, -4, 5), (2, -2, 7)]
  - step 5: barren area --[west]--> cliff — correct
  - step 54: cliff --[down]--> cliff ledge — correct

## Conflict 44 — naming (naming_collision_on_correct_subgraph)
- description: node 'barren area' reachable at conflicting positions [(-3, 3, 0), (-2, 3, 0), (-1, 2, 3), (1, -1, 4), (3, -4, 5), (3, -2, 7)]
  - step 4: junction --[west]--> barren area — correct
  - step 5: barren area --[west]--> cliff — correct

## Conflict 45 — naming (naming_collision_on_correct_subgraph)
- description: node 'creepy crawl' reachable at conflicting positions [(-2, 2, 0), (-2, 4, 2), (-1, 2, 0), (0, 1, 3), (2, -2, 4), (2, 0, 6)]
  - step 18: junction --[south]--> creepy crawl — correct
  - step 19: creepy crawl --[south]--> foggy room — correct

## Conflict 46 — naming (naming_collision_on_correct_subgraph)
- description: node 'foggy room' reachable at conflicting positions [(-2, 3, 2), (0, 0, 3), (2, -3, 4), (2, -1, 6)]
  - step 19: creepy crawl --[south]--> foggy room — correct
  - step 20: foggy room --[south]--> lake shore — correct

## Conflict 47 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'lake shore' reachable at conflicting positions [(-2, 2, 2), (0, -1, 3), (2, -4, 4), (2, -2, 6)]
  - step 20: foggy room --[south]--> lake shore — correct
  - step 23: lake shore --[down]--> on the lake — hallucinated_edge

## Conflict 48 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'western shore' reachable at conflicting positions [(-3, 2, 1), (-1, -1, 2), (1, -4, 3), (1, -2, 5)]
  - step 25: on the lake --[west]--> western shore — correct
  - step 36: viewing room --[north]--> western shore — src_hallucinated
  - step 26: western shore --[south]--> scenic vista — correct

## Conflict 49 — naming (naming_collision_on_correct_subgraph)
- description: node 'scenic vista' reachable at conflicting positions [(-3, 1, 1), (-1, -2, 2), (1, -5, 3), (1, -3, 5)]
  - step 26: western shore --[south]--> scenic vista — correct

## Conflict 50 — naming (name_hallucination)
- description: node 'viewing room' reachable at conflicting positions [(-3, 1, 1), (-1, -2, 2), (1, -5, 3), (1, -3, 5)]
  - step 36: viewing room --[north]--> western shore — src_hallucinated
