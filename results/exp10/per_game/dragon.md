# Conflict analysis: dragon

- LLM edges: 24
- GT edges: 42
- Conflicts: 45
- Type distribution: {'direction': 2, 'topology': 22, 'naming': 21}
- Root-cause distribution: {'real_vs_hallucinated': 2, 'false_positive_overlap': 7, 'overlap_mixed': 15, 'naming_collision_on_correct_subgraph': 16, 'real_name_corrupted_by_neighbour_error': 5}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'north end of a road' has multiple outgoing edges labelled 'south'
  - step 13: north end of a road --[south]--> south end of a road — correct
  - step None: north end of a road --[south]--> wasteland by a castle — hallucinated_edge

## Conflict 2 — direction (real_vs_hallucinated)
- description: node 'wasteland by a castle' has multiple outgoing edges labelled 'north'
  - step 15: wasteland by a castle --[north]--> castle forge — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge

## Conflict 3 — topology (false_positive_overlap)
- description: position (0, 0, 0) occupied by multiple rooms ['abandoned inn', 'swampy ground']
  - step 1: rocky mountains --[south]--> abandoned inn — correct
  - step 2: abandoned inn --[east]--> inside the inn — correct
  - step 9: abandoned inn --[west]--> swampy ground — correct
  - step 12: swampy ground --[west]--> north end of a road — correct
  - step 21: swampy ground --[south]--> cave entrance — correct

## Conflict 4 — topology (overlap_mixed)
- description: position (-3, -3, -1) occupied by multiple rooms ['abandoned inn', 'castle forge', 'north end of a road', 'swampy ground']
  - step 1: rocky mountains --[south]--> abandoned inn — correct
  - step 2: abandoned inn --[east]--> inside the inn — correct
  - step 9: abandoned inn --[west]--> swampy ground — correct
  - step 15: wasteland by a castle --[north]--> castle forge — correct
  - step 39: castle forge --[north]--> forbidding castle — correct
  - step 12: swampy ground --[west]--> north end of a road — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge
  - step 13: north end of a road --[south]--> south end of a road — correct
  - step 21: swampy ground --[south]--> cave entrance — correct

## Conflict 5 — topology (false_positive_overlap)
- description: position (-1, -3, -1) occupied by multiple rooms ['abandoned inn', 'inside the inn']
  - step 1: rocky mountains --[south]--> abandoned inn — correct
  - step 2: abandoned inn --[east]--> inside the inn — correct
  - step 9: abandoned inn --[west]--> swampy ground — correct
  - step 3: inside the inn --[down]--> inn cellar — correct

## Conflict 6 — topology (overlap_mixed)
- description: position (-2, -3, -1) occupied by multiple rooms ['abandoned inn', 'inside the inn', 'north end of a road', 'swampy ground']
  - step 1: rocky mountains --[south]--> abandoned inn — correct
  - step 2: abandoned inn --[east]--> inside the inn — correct
  - step 9: abandoned inn --[west]--> swampy ground — correct
  - step 3: inside the inn --[down]--> inn cellar — correct
  - step 12: swampy ground --[west]--> north end of a road — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge
  - step 13: north end of a road --[south]--> south end of a road — correct
  - step 21: swampy ground --[south]--> cave entrance — correct

## Conflict 7 — topology (false_positive_overlap)
- description: position (-3, -2, -1) occupied by multiple rooms ['forbidding castle', 'rocky mountains']
  - step 39: castle forge --[north]--> forbidding castle — correct
  - step 43: forbidding castle --[north]--> castle hallway — correct
  - step 1: rocky mountains --[south]--> abandoned inn — correct

## Conflict 8 — topology (overlap_mixed)
- description: position (-1, 0, 0) occupied by multiple rooms ['north end of a road', 'swampy ground']
  - step 12: swampy ground --[west]--> north end of a road — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge
  - step 13: north end of a road --[south]--> south end of a road — correct
  - step 9: abandoned inn --[west]--> swampy ground — correct
  - step 21: swampy ground --[south]--> cave entrance — correct

## Conflict 9 — topology (overlap_mixed)
- description: position (-4, -3, -1) occupied by multiple rooms ['castle forge', 'north end of a road', 'swampy ground']
  - step 15: wasteland by a castle --[north]--> castle forge — correct
  - step 39: castle forge --[north]--> forbidding castle — correct
  - step 12: swampy ground --[west]--> north end of a road — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge
  - step 13: north end of a road --[south]--> south end of a road — correct
  - step 9: abandoned inn --[west]--> swampy ground — correct
  - step 21: swampy ground --[south]--> cave entrance — correct

## Conflict 10 — topology (overlap_mixed)
- description: position (-2, 0, 0) occupied by multiple rooms ['castle forge', 'north end of a road']
  - step 15: wasteland by a castle --[north]--> castle forge — correct
  - step 39: castle forge --[north]--> forbidding castle — correct
  - step 12: swampy ground --[west]--> north end of a road — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge
  - step 13: north end of a road --[south]--> south end of a road — correct

## Conflict 11 — topology (overlap_mixed)
- description: position (-5, -3, -1) occupied by multiple rooms ['castle forge', 'north end of a road']
  - step 15: wasteland by a castle --[north]--> castle forge — correct
  - step 39: castle forge --[north]--> forbidding castle — correct
  - step 12: swampy ground --[west]--> north end of a road — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge
  - step 13: north end of a road --[south]--> south end of a road — correct

## Conflict 12 — topology (overlap_mixed)
- description: position (-3, -4, -1) occupied by multiple rooms ['cave entrance', 'south end of a road', 'wasteland by a castle']
  - step 21: swampy ground --[south]--> cave entrance — correct
  - step 22: cave entrance --[south]--> cottage — correct
  - step 13: north end of a road --[south]--> south end of a road — correct
  - step 14: south end of a road --[west]--> wasteland by a castle — correct
  - step 58: south end of a road --[south]--> lake in the forest — correct
  - step 37: pebbled beach --[north]--> wasteland by a castle — correct
  - step 51: top of a tower --[east]--> wasteland by a castle — hallucinated_edge
  - step 15: wasteland by a castle --[north]--> castle forge — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge

## Conflict 13 — topology (overlap_mixed)
- description: position (-1, -1, 0) occupied by multiple rooms ['cave entrance', 'south end of a road', 'wasteland by a castle']
  - step 21: swampy ground --[south]--> cave entrance — correct
  - step 22: cave entrance --[south]--> cottage — correct
  - step 13: north end of a road --[south]--> south end of a road — correct
  - step 14: south end of a road --[west]--> wasteland by a castle — correct
  - step 58: south end of a road --[south]--> lake in the forest — correct
  - step 37: pebbled beach --[north]--> wasteland by a castle — correct
  - step 51: top of a tower --[east]--> wasteland by a castle — hallucinated_edge
  - step 15: wasteland by a castle --[north]--> castle forge — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge

## Conflict 14 — topology (false_positive_overlap)
- description: position (-2, -4, -1) occupied by multiple rooms ['cave entrance', 'south end of a road']
  - step 21: swampy ground --[south]--> cave entrance — correct
  - step 22: cave entrance --[south]--> cottage — correct
  - step 13: north end of a road --[south]--> south end of a road — correct
  - step 14: south end of a road --[west]--> wasteland by a castle — correct
  - step 58: south end of a road --[south]--> lake in the forest — correct

## Conflict 15 — topology (overlap_mixed)
- description: position (-4, -4, -1) occupied by multiple rooms ['cave entrance', 'south end of a road', 'top of a tower', 'wasteland by a castle']
  - step 21: swampy ground --[south]--> cave entrance — correct
  - step 22: cave entrance --[south]--> cottage — correct
  - step 13: north end of a road --[south]--> south end of a road — correct
  - step 14: south end of a road --[west]--> wasteland by a castle — correct
  - step 58: south end of a road --[south]--> lake in the forest — correct
  - step 44: castle hallway --[up]--> top of a tower — correct
  - step 51: top of a tower --[east]--> wasteland by a castle — hallucinated_edge
  - step 37: pebbled beach --[north]--> wasteland by a castle — correct
  - step 15: wasteland by a castle --[north]--> castle forge — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge

## Conflict 16 — topology (overlap_mixed)
- description: position (-2, -5, -1) occupied by multiple rooms ['cottage', 'inside a cottage', 'lake in the forest']
  - step 22: cave entrance --[south]--> cottage — correct
  - step 23: cottage --[east]--> inside a cottage — correct
  - step 27: inside a cottage --[south]--> forest pathway — hallucinated_edge
  - step 29: tree stump --[north]--> lake in the forest — correct
  - step 58: south end of a road --[south]--> lake in the forest — correct
  - step 30: lake in the forest --[west]--> pebbled beach — correct

## Conflict 17 — topology (false_positive_overlap)
- description: position (-1, -2, 0) occupied by multiple rooms ['cottage', 'lake in the forest']
  - step 22: cave entrance --[south]--> cottage — correct
  - step 23: cottage --[east]--> inside a cottage — correct
  - step 29: tree stump --[north]--> lake in the forest — correct
  - step 58: south end of a road --[south]--> lake in the forest — correct
  - step 30: lake in the forest --[west]--> pebbled beach — correct

## Conflict 18 — topology (false_positive_overlap)
- description: position (-4, -5, -1) occupied by multiple rooms ['cottage', 'lake in the forest', 'pebbled beach']
  - step 22: cave entrance --[south]--> cottage — correct
  - step 23: cottage --[east]--> inside a cottage — correct
  - step 29: tree stump --[north]--> lake in the forest — correct
  - step 58: south end of a road --[south]--> lake in the forest — correct
  - step 30: lake in the forest --[west]--> pebbled beach — correct
  - step 31: pebbled beach --[south]--> beside a ruined lighthouse — correct
  - step 37: pebbled beach --[north]--> wasteland by a castle — correct

## Conflict 19 — topology (overlap_mixed)
- description: position (-3, -5, -1) occupied by multiple rooms ['cottage', 'inside a cottage', 'lake in the forest', 'pebbled beach']
  - step 22: cave entrance --[south]--> cottage — correct
  - step 23: cottage --[east]--> inside a cottage — correct
  - step 27: inside a cottage --[south]--> forest pathway — hallucinated_edge
  - step 29: tree stump --[north]--> lake in the forest — correct
  - step 58: south end of a road --[south]--> lake in the forest — correct
  - step 30: lake in the forest --[west]--> pebbled beach — correct
  - step 31: pebbled beach --[south]--> beside a ruined lighthouse — correct
  - step 37: pebbled beach --[north]--> wasteland by a castle — correct

## Conflict 20 — topology (overlap_mixed)
- description: position (-3, -6, -1) occupied by multiple rooms ['beside a ruined lighthouse', 'forest pathway', 'tree stump']
  - step 31: pebbled beach --[south]--> beside a ruined lighthouse — correct
  - step 32: beside a ruined lighthouse --[south]--> inside the lighthouse ruins — correct
  - step 27: inside a cottage --[south]--> forest pathway — hallucinated_edge
  - step 28: forest pathway --[west]--> tree stump — correct
  - step 29: tree stump --[north]--> lake in the forest — correct

## Conflict 21 — topology (overlap_mixed)
- description: position (-2, -6, -1) occupied by multiple rooms ['forest pathway', 'tree stump']
  - step 27: inside a cottage --[south]--> forest pathway — hallucinated_edge
  - step 28: forest pathway --[west]--> tree stump — correct
  - step 29: tree stump --[north]--> lake in the forest — correct

## Conflict 22 — topology (false_positive_overlap)
- description: position (-4, -6, -1) occupied by multiple rooms ['beside a ruined lighthouse', 'tree stump']
  - step 31: pebbled beach --[south]--> beside a ruined lighthouse — correct
  - step 32: beside a ruined lighthouse --[south]--> inside the lighthouse ruins — correct
  - step 28: forest pathway --[west]--> tree stump — correct
  - step 29: tree stump --[north]--> lake in the forest — correct

## Conflict 23 — topology (overlap_mixed)
- description: position (-2, -1, 0) occupied by multiple rooms ['south end of a road', 'wasteland by a castle']
  - step 13: north end of a road --[south]--> south end of a road — correct
  - step 14: south end of a road --[west]--> wasteland by a castle — correct
  - step 58: south end of a road --[south]--> lake in the forest — correct
  - step 37: pebbled beach --[north]--> wasteland by a castle — correct
  - step 51: top of a tower --[east]--> wasteland by a castle — hallucinated_edge
  - step 15: wasteland by a castle --[north]--> castle forge — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge

## Conflict 24 — topology (overlap_mixed)
- description: position (-5, -4, -1) occupied by multiple rooms ['top of a tower', 'wasteland by a castle']
  - step 44: castle hallway --[up]--> top of a tower — correct
  - step 51: top of a tower --[east]--> wasteland by a castle — hallucinated_edge
  - step 14: south end of a road --[west]--> wasteland by a castle — correct
  - step 37: pebbled beach --[north]--> wasteland by a castle — correct
  - step 15: wasteland by a castle --[north]--> castle forge — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge

## Conflict 25 — naming (naming_collision_on_correct_subgraph)
- description: node 'abandoned inn' reachable at conflicting positions [(-3, -3, -1), (-2, -3, -1), (-1, -3, -1), (0, 0, 0)]
  - step 1: rocky mountains --[south]--> abandoned inn — correct
  - step 2: abandoned inn --[east]--> inside the inn — correct
  - step 9: abandoned inn --[west]--> swampy ground — correct

## Conflict 26 — naming (naming_collision_on_correct_subgraph)
- description: node 'rocky mountains' reachable at conflicting positions [(-3, -2, -1), (-2, -2, -1), (-1, -2, -1), (0, 1, 0)]
  - step 1: rocky mountains --[south]--> abandoned inn — correct

## Conflict 27 — naming (naming_collision_on_correct_subgraph)
- description: node 'inside the inn' reachable at conflicting positions [(-2, -3, -1), (-1, -3, -1), (0, -3, -1), (1, 0, 0)]
  - step 2: abandoned inn --[east]--> inside the inn — correct
  - step 3: inside the inn --[down]--> inn cellar — correct

## Conflict 28 — naming (naming_collision_on_correct_subgraph)
- description: node 'swampy ground' reachable at conflicting positions [(-4, -3, -1), (-3, -3, -1), (-2, -3, -1), (-1, 0, 0), (0, 0, 0)]
  - step 9: abandoned inn --[west]--> swampy ground — correct
  - step 12: swampy ground --[west]--> north end of a road — correct
  - step 21: swampy ground --[south]--> cave entrance — correct

## Conflict 29 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'north end of a road' reachable at conflicting positions [(-5, -3, -1), (-4, -3, -1), (-3, -3, -1), (-2, -3, -1), (-2, 0, 0), (-1, 0, 0)]
  - step 12: swampy ground --[west]--> north end of a road — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge
  - step 13: north end of a road --[south]--> south end of a road — correct

## Conflict 30 — naming (naming_collision_on_correct_subgraph)
- description: node 'cave entrance' reachable at conflicting positions [(-4, -4, -1), (-3, -4, -1), (-2, -4, -1), (-1, -1, 0)]
  - step 21: swampy ground --[south]--> cave entrance — correct
  - step 22: cave entrance --[south]--> cottage — correct

## Conflict 31 — naming (naming_collision_on_correct_subgraph)
- description: node 'cottage' reachable at conflicting positions [(-4, -5, -1), (-3, -5, -1), (-2, -5, -1), (-1, -2, 0)]
  - step 22: cave entrance --[south]--> cottage — correct
  - step 23: cottage --[east]--> inside a cottage — correct

## Conflict 32 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'inside a cottage' reachable at conflicting positions [(-3, -5, -1), (-2, -5, -1), (-1, -5, -1), (0, -2, 0)]
  - step 23: cottage --[east]--> inside a cottage — correct
  - step 27: inside a cottage --[south]--> forest pathway — hallucinated_edge

## Conflict 33 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'forest pathway' reachable at conflicting positions [(-3, -6, -1), (-2, -6, -1), (-1, -6, -1), (0, -3, 0)]
  - step 27: inside a cottage --[south]--> forest pathway — hallucinated_edge
  - step 28: forest pathway --[west]--> tree stump — correct

## Conflict 34 — naming (naming_collision_on_correct_subgraph)
- description: node 'tree stump' reachable at conflicting positions [(-4, -6, -1), (-3, -6, -1), (-2, -6, -1), (-1, -3, 0)]
  - step 28: forest pathway --[west]--> tree stump — correct
  - step 29: tree stump --[north]--> lake in the forest — correct

## Conflict 35 — naming (naming_collision_on_correct_subgraph)
- description: node 'lake in the forest' reachable at conflicting positions [(-4, -5, -1), (-3, -5, -1), (-2, -5, -1), (-1, -2, 0)]
  - step 29: tree stump --[north]--> lake in the forest — correct
  - step 58: south end of a road --[south]--> lake in the forest — correct
  - step 30: lake in the forest --[west]--> pebbled beach — correct

## Conflict 36 — naming (naming_collision_on_correct_subgraph)
- description: node 'pebbled beach' reachable at conflicting positions [(-5, -5, -1), (-4, -5, -1), (-3, -5, -1), (-2, -2, 0)]
  - step 30: lake in the forest --[west]--> pebbled beach — correct
  - step 31: pebbled beach --[south]--> beside a ruined lighthouse — correct
  - step 37: pebbled beach --[north]--> wasteland by a castle — correct

## Conflict 37 — naming (naming_collision_on_correct_subgraph)
- description: node 'south end of a road' reachable at conflicting positions [(-4, -4, -1), (-3, -4, -1), (-2, -4, -1), (-2, -1, 0), (-1, -1, 0)]
  - step 13: north end of a road --[south]--> south end of a road — correct
  - step 14: south end of a road --[west]--> wasteland by a castle — correct
  - step 58: south end of a road --[south]--> lake in the forest — correct

## Conflict 38 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'wasteland by a castle' reachable at conflicting positions [(-6, -7, -2), (-5, -4, -1), (-4, -4, -1), (-4, -1, 0), (-3, -4, -1), (-2, -1, 0), (-1, -1, 0)]
  - step 14: south end of a road --[west]--> wasteland by a castle — correct
  - step 37: pebbled beach --[north]--> wasteland by a castle — correct
  - step 51: top of a tower --[east]--> wasteland by a castle — hallucinated_edge
  - step 15: wasteland by a castle --[north]--> castle forge — correct
  - step 53: wasteland by a castle --[north]--> north end of a road — hallucinated_edge

## Conflict 39 — naming (naming_collision_on_correct_subgraph)
- description: node 'castle forge' reachable at conflicting positions [(-6, -6, -2), (-5, -3, -1), (-4, -6, -2), (-4, -3, -1), (-3, -3, -1), (-2, 0, 0)]
  - step 15: wasteland by a castle --[north]--> castle forge — correct
  - step 39: castle forge --[north]--> forbidding castle — correct

## Conflict 40 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'top of a tower' reachable at conflicting positions [(-6, -4, -1), (-5, -4, -1), (-5, -1, 0), (-4, -4, -1), (-3, -1, 0)]
  - step 44: castle hallway --[up]--> top of a tower — correct
  - step 51: top of a tower --[east]--> wasteland by a castle — hallucinated_edge

## Conflict 41 — naming (naming_collision_on_correct_subgraph)
- description: node 'castle hallway' reachable at conflicting positions [(-6, -4, -2), (-5, -1, -1), (-4, -4, -2), (-3, -1, -1)]
  - step 43: forbidding castle --[north]--> castle hallway — correct
  - step 44: castle hallway --[up]--> top of a tower — correct

## Conflict 42 — naming (naming_collision_on_correct_subgraph)
- description: node 'forbidding castle' reachable at conflicting positions [(-6, -5, -2), (-5, -2, -1), (-4, -5, -2), (-3, -2, -1), (-2, 1, 0)]
  - step 39: castle forge --[north]--> forbidding castle — correct
  - step 43: forbidding castle --[north]--> castle hallway — correct

## Conflict 43 — naming (naming_collision_on_correct_subgraph)
- description: node 'beside a ruined lighthouse' reachable at conflicting positions [(-5, -6, -1), (-4, -6, -1), (-3, -6, -1), (-2, -3, 0)]
  - step 31: pebbled beach --[south]--> beside a ruined lighthouse — correct
  - step 32: beside a ruined lighthouse --[south]--> inside the lighthouse ruins — correct

## Conflict 44 — naming (naming_collision_on_correct_subgraph)
- description: node 'inside the lighthouse ruins' reachable at conflicting positions [(-5, -7, -1), (-4, -7, -1), (-3, -7, -1), (-2, -4, 0)]
  - step 32: beside a ruined lighthouse --[south]--> inside the lighthouse ruins — correct

## Conflict 45 — naming (naming_collision_on_correct_subgraph)
- description: node 'inn cellar' reachable at conflicting positions [(-2, -3, -2), (-1, -3, -2), (0, -3, -2), (1, 0, -1)]
  - step 3: inside the inn --[down]--> inn cellar — correct
