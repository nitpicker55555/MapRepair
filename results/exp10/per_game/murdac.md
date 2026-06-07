# Conflict analysis: murdac

- LLM edges: 34
- GT edges: 48
- Conflicts: 76
- Type distribution: {'direction': 13, 'topology': 34, 'naming': 29}
- Root-cause distribution: {'real_vs_hallucinated': 7, 'all_hallucinated_edges': 2, 'name_hallucination': 11, 'name_hallucination_caused_overlap': 24, 'overlap_mixed': 9, 'false_positive_overlap': 1, 'real_name_corrupted_by_neighbour_error': 11, 'naming_collision_on_correct_subgraph': 11}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'outside the house' has multiple outgoing edges labelled 'north'
  - step 1: outside the house --[north]--> flower garden — hallucinated_edge
  - step 16: outside the house --[north]--> sandpit — hallucinated_edge
  - step 28: outside the house --[north]--> inside the hut — correct

## Conflict 2 — direction (real_vs_hallucinated)
- description: node 'flower garden' has multiple outgoing edges labelled 'south'
  - step None: flower garden --[south]--> outside the house — hallucinated_edge
  - step None: flower garden --[south]--> remnants of a bonfire — correct

## Conflict 3 — direction (all_hallucinated_edges)
- description: node 'alley (north/south)' has multiple outgoing edges labelled 'south'
  - step None: alley (north/south) --[south]--> flower garden — hallucinated_edge
  - step 7: alley (north/south) --[south]--> vegetable garden — hallucinated_edge

## Conflict 4 — direction (real_vs_hallucinated)
- description: node 'sandpit' has multiple outgoing edges labelled 'south'
  - step None: sandpit --[south]--> outside the house — hallucinated_edge
  - step 19: sandpit --[south]--> old untended grave — correct

## Conflict 5 — direction (name_hallucination)
- description: node "mad scientist's laboratory" has multiple outgoing edges labelled 'west'
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct
  - step None: mad scientist's laboratory --[west]--> tunnel between the laboratory and the wiring — dst_hallucinated

## Conflict 6 — direction (name_hallucination)
- description: node 'wooden plank in east/west tunnel' has multiple outgoing edges labelled 'east'
  - step None: wooden plank in east/west tunnel --[east]--> high tunnel (to the west of lab) — correct
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated

## Conflict 7 — direction (real_vs_hallucinated)
- description: node 'wooden plank in east/west tunnel' has multiple outgoing edges labelled 'west'
  - step 37: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, alcove off to north) — correct
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge

## Conflict 8 — direction (real_vs_hallucinated)
- description: node 'alcove' has multiple outgoing edges labelled 'south'
  - step None: alcove --[south]--> high tunnel (east/west, alcove off to north) — correct
  - step 42: alcove --[south]--> wooden plank in east/west tunnel — hallucinated_edge

## Conflict 9 — direction (all_hallucinated_edges)
- description: node 'high tunnel (east/west, path to west)' has multiple outgoing edges labelled 'east'
  - step None: high tunnel (east/west, path to west) --[east]--> wooden plank in east/west tunnel — hallucinated_edge
  - step 69: high tunnel (east/west, path to west) --[east]--> complicated junction of passages — hallucinated_edge

## Conflict 10 — direction (real_vs_hallucinated)
- description: node 'complicated junction of passages' has multiple outgoing edges labelled 'west'
  - step None: complicated junction of passages --[west]--> large quadrangular cellar — correct
  - step None: complicated junction of passages --[west]--> high tunnel (east/west, path to west) — hallucinated_edge

## Conflict 11 — direction (real_vs_hallucinated)
- description: node 'entrance hall to haunted house' has multiple outgoing edges labelled 'west'
  - step None: entrance hall to haunted house --[west]--> complicated junction of passages — correct
  - step 60: entrance hall to haunted house --[west]--> large bedroom in haunted house — hallucinated_edge

## Conflict 12 — direction (name_hallucination)
- description: node 'large bedroom in haunted house' has multiple outgoing edges labelled 'east'
  - step None: large bedroom in haunted house --[east]--> entrance hall to haunted house — hallucinated_edge
  - step 61: large bedroom in haunted house --[east]--> billiard room of the haunted house — dst_hallucinated

## Conflict 13 — direction (name_hallucination)
- description: node 'pantry of haunted house' has multiple outgoing edges labelled 'south'
  - step None: pantry of haunted house --[south]--> billiard room of the haunted house — both_names_hallucinated
  - step 63: pantry of haunted house --[south]--> living room of haunted house — src_hallucinated

## Conflict 14 — topology (name_hallucination_caused_overlap)
- description: position (1, -1, 0) occupied by multiple rooms ['alcove', 'high tunnel (to the west of lab)', 'sentry-post', 'tunnel between the laboratory and the wiring', 'wooden plank in east/west tunnel']
  - step 41: high tunnel (east/west, alcove off to north) --[north]--> alcove — correct
  - step 42: alcove --[south]--> wooden plank in east/west tunnel — hallucinated_edge
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct
  - step 36: high tunnel (to the west of lab) --[west]--> wooden plank in east/west tunnel — correct
  - step 70: complicated junction of passages --[north]--> sentry-post — dst_hallucinated
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated
  - step 55: tunnel between the laboratory and the wiring --[east]--> mad scientist's laboratory — src_hallucinated
  - step 37: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, alcove off to north) — correct
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge

## Conflict 15 — topology (name_hallucination_caused_overlap)
- description: position (-2, -2, 0) occupied by multiple rooms ['alcove', 'low damp chamber', 'sentry-post']
  - step 41: high tunnel (east/west, alcove off to north) --[north]--> alcove — correct
  - step 42: alcove --[south]--> wooden plank in east/west tunnel — hallucinated_edge
  - step 46: secret cavern --[south]--> low damp chamber — correct
  - step 70: complicated junction of passages --[north]--> sentry-post — dst_hallucinated

## Conflict 16 — topology (name_hallucination_caused_overlap)
- description: position (4, 0, 0) occupied by multiple rooms ['alcove', 'sentry-post']
  - step 41: high tunnel (east/west, alcove off to north) --[north]--> alcove — correct
  - step 42: alcove --[south]--> wooden plank in east/west tunnel — hallucinated_edge
  - step 70: complicated junction of passages --[north]--> sentry-post — dst_hallucinated

## Conflict 17 — topology (name_hallucination_caused_overlap)
- description: position (3, -1, 0) occupied by multiple rooms ['high tunnel (east/west, alcove off to north)', 'high tunnel (east/west, path to west)', 'large quadrangular cellar']
  - step 37: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, alcove off to north) — correct
  - step 41: high tunnel (east/west, alcove off to north) --[north]--> alcove — correct
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge
  - step 68: deserted railway platform --[north]--> high tunnel (east/west, path to west) — src_hallucinated
  - step 44: high tunnel (east/west, path to west) --[west]--> secret cavern — correct
  - step 69: high tunnel (east/west, path to west) --[east]--> complicated junction of passages — hallucinated_edge
  - step 32: inside the hut --[down]--> large quadrangular cellar — correct
  - step 33: large quadrangular cellar --[west]--> mad scientist's laboratory — correct
  - step 58: large quadrangular cellar --[east]--> complicated junction of passages — correct

## Conflict 18 — topology (name_hallucination_caused_overlap)
- description: position (-3, -3, 0) occupied by multiple rooms ['high tunnel (east/west, alcove off to north)', 'high tunnel (east/west, path to west)', 'large quadrangular cellar', 'wooden plank in east/west tunnel']
  - step 37: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, alcove off to north) — correct
  - step 41: high tunnel (east/west, alcove off to north) --[north]--> alcove — correct
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge
  - step 68: deserted railway platform --[north]--> high tunnel (east/west, path to west) — src_hallucinated
  - step 44: high tunnel (east/west, path to west) --[west]--> secret cavern — correct
  - step 69: high tunnel (east/west, path to west) --[east]--> complicated junction of passages — hallucinated_edge
  - step 32: inside the hut --[down]--> large quadrangular cellar — correct
  - step 33: large quadrangular cellar --[west]--> mad scientist's laboratory — correct
  - step 58: large quadrangular cellar --[east]--> complicated junction of passages — correct
  - step 36: high tunnel (to the west of lab) --[west]--> wooden plank in east/west tunnel — correct
  - step 42: alcove --[south]--> wooden plank in east/west tunnel — hallucinated_edge
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated

## Conflict 19 — topology (name_hallucination_caused_overlap)
- description: position (0, -1, 0) occupied by multiple rooms ['complicated junction of passages', 'high tunnel (east/west, alcove off to north)', 'wooden plank in east/west tunnel']
  - step 58: large quadrangular cellar --[east]--> complicated junction of passages — correct
  - step 69: high tunnel (east/west, path to west) --[east]--> complicated junction of passages — hallucinated_edge
  - step 59: complicated junction of passages --[east]--> entrance hall to haunted house — correct
  - step 70: complicated junction of passages --[north]--> sentry-post — dst_hallucinated
  - step 37: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, alcove off to north) — correct
  - step 41: high tunnel (east/west, alcove off to north) --[north]--> alcove — correct
  - step 36: high tunnel (to the west of lab) --[west]--> wooden plank in east/west tunnel — correct
  - step 42: alcove --[south]--> wooden plank in east/west tunnel — hallucinated_edge
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated

## Conflict 20 — topology (name_hallucination_caused_overlap)
- description: position (1, -2, 0) occupied by multiple rooms ['complicated junction of passages', 'high tunnel (east/west, alcove off to north)', 'large bedroom in haunted house', 'wooden plank in east/west tunnel']
  - step 58: large quadrangular cellar --[east]--> complicated junction of passages — correct
  - step 69: high tunnel (east/west, path to west) --[east]--> complicated junction of passages — hallucinated_edge
  - step 59: complicated junction of passages --[east]--> entrance hall to haunted house — correct
  - step 70: complicated junction of passages --[north]--> sentry-post — dst_hallucinated
  - step 37: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, alcove off to north) — correct
  - step 41: high tunnel (east/west, alcove off to north) --[north]--> alcove — correct
  - step 60: entrance hall to haunted house --[west]--> large bedroom in haunted house — hallucinated_edge
  - step 61: large bedroom in haunted house --[east]--> billiard room of the haunted house — dst_hallucinated
  - step 36: high tunnel (to the west of lab) --[west]--> wooden plank in east/west tunnel — correct
  - step 42: alcove --[south]--> wooden plank in east/west tunnel — hallucinated_edge
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated

## Conflict 21 — topology (name_hallucination_caused_overlap)
- description: position (-2, -3, 0) occupied by multiple rooms ['complicated junction of passages', 'high tunnel (east/west, alcove off to north)', 'large bedroom in haunted house', 'wooden plank in east/west tunnel']
  - step 58: large quadrangular cellar --[east]--> complicated junction of passages — correct
  - step 69: high tunnel (east/west, path to west) --[east]--> complicated junction of passages — hallucinated_edge
  - step 59: complicated junction of passages --[east]--> entrance hall to haunted house — correct
  - step 70: complicated junction of passages --[north]--> sentry-post — dst_hallucinated
  - step 37: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, alcove off to north) — correct
  - step 41: high tunnel (east/west, alcove off to north) --[north]--> alcove — correct
  - step 60: entrance hall to haunted house --[west]--> large bedroom in haunted house — hallucinated_edge
  - step 61: large bedroom in haunted house --[east]--> billiard room of the haunted house — dst_hallucinated
  - step 36: high tunnel (to the west of lab) --[west]--> wooden plank in east/west tunnel — correct
  - step 42: alcove --[south]--> wooden plank in east/west tunnel — hallucinated_edge
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated

## Conflict 22 — topology (name_hallucination_caused_overlap)
- description: position (-1, -1, 0) occupied by multiple rooms ['high tunnel (east/west, alcove off to north)', 'high tunnel (east/west, path to west)']
  - step 37: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, alcove off to north) — correct
  - step 41: high tunnel (east/west, alcove off to north) --[north]--> alcove — correct
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge
  - step 68: deserted railway platform --[north]--> high tunnel (east/west, path to west) — src_hallucinated
  - step 44: high tunnel (east/west, path to west) --[west]--> secret cavern — correct
  - step 69: high tunnel (east/west, path to west) --[east]--> complicated junction of passages — hallucinated_edge

## Conflict 23 — topology (name_hallucination_caused_overlap)
- description: position (0, -2, 0) occupied by multiple rooms ['high tunnel (east/west, alcove off to north)', 'high tunnel (east/west, path to west)', 'large quadrangular cellar']
  - step 37: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, alcove off to north) — correct
  - step 41: high tunnel (east/west, alcove off to north) --[north]--> alcove — correct
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge
  - step 68: deserted railway platform --[north]--> high tunnel (east/west, path to west) — src_hallucinated
  - step 44: high tunnel (east/west, path to west) --[west]--> secret cavern — correct
  - step 69: high tunnel (east/west, path to west) --[east]--> complicated junction of passages — hallucinated_edge
  - step 32: inside the hut --[down]--> large quadrangular cellar — correct
  - step 33: large quadrangular cellar --[west]--> mad scientist's laboratory — correct
  - step 58: large quadrangular cellar --[east]--> complicated junction of passages — correct

## Conflict 24 — topology (name_hallucination_caused_overlap)
- description: position (-1, -3, 0) occupied by multiple rooms ['billiard room of the haunted house', 'entrance hall to haunted house', 'high tunnel (to the west of lab)', 'living room of haunted house', 'low damp chamber', 'tunnel between the laboratory and the wiring', 'wooden plank in east/west tunnel']
  - step 61: large bedroom in haunted house --[east]--> billiard room of the haunted house — dst_hallucinated
  - step 62: billiard room of the haunted house --[north]--> pantry of haunted house — both_names_hallucinated
  - step 59: complicated junction of passages --[east]--> entrance hall to haunted house — correct
  - step 60: entrance hall to haunted house --[west]--> large bedroom in haunted house — hallucinated_edge
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct
  - step 36: high tunnel (to the west of lab) --[west]--> wooden plank in east/west tunnel — correct
  - step 63: pantry of haunted house --[south]--> living room of haunted house — src_hallucinated
  - step 65: living room of haunted house --[east]--> deserted railway platform — dst_hallucinated
  - step 46: secret cavern --[south]--> low damp chamber — correct
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated
  - step 55: tunnel between the laboratory and the wiring --[east]--> mad scientist's laboratory — src_hallucinated
  - step 42: alcove --[south]--> wooden plank in east/west tunnel — hallucinated_edge
  - step 37: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, alcove off to north) — correct
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge

## Conflict 25 — topology (name_hallucination_caused_overlap)
- description: position (4, -1, 0) occupied by multiple rooms ['complicated junction of passages', 'large bedroom in haunted house', 'wooden plank in east/west tunnel']
  - step 58: large quadrangular cellar --[east]--> complicated junction of passages — correct
  - step 69: high tunnel (east/west, path to west) --[east]--> complicated junction of passages — hallucinated_edge
  - step 59: complicated junction of passages --[east]--> entrance hall to haunted house — correct
  - step 70: complicated junction of passages --[north]--> sentry-post — dst_hallucinated
  - step 60: entrance hall to haunted house --[west]--> large bedroom in haunted house — hallucinated_edge
  - step 61: large bedroom in haunted house --[east]--> billiard room of the haunted house — dst_hallucinated
  - step 36: high tunnel (to the west of lab) --[west]--> wooden plank in east/west tunnel — correct
  - step 42: alcove --[south]--> wooden plank in east/west tunnel — hallucinated_edge
  - step 37: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, alcove off to north) — correct
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated

## Conflict 26 — topology (name_hallucination_caused_overlap)
- description: position (2, -2, 0) occupied by multiple rooms ['billiard room of the haunted house', 'entrance hall to haunted house', 'high tunnel (to the west of lab)', 'living room of haunted house', 'low damp chamber', 'tunnel between the laboratory and the wiring']
  - step 61: large bedroom in haunted house --[east]--> billiard room of the haunted house — dst_hallucinated
  - step 62: billiard room of the haunted house --[north]--> pantry of haunted house — both_names_hallucinated
  - step 59: complicated junction of passages --[east]--> entrance hall to haunted house — correct
  - step 60: entrance hall to haunted house --[west]--> large bedroom in haunted house — hallucinated_edge
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct
  - step 36: high tunnel (to the west of lab) --[west]--> wooden plank in east/west tunnel — correct
  - step 63: pantry of haunted house --[south]--> living room of haunted house — src_hallucinated
  - step 65: living room of haunted house --[east]--> deserted railway platform — dst_hallucinated
  - step 46: secret cavern --[south]--> low damp chamber — correct
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated
  - step 55: tunnel between the laboratory and the wiring --[east]--> mad scientist's laboratory — src_hallucinated

## Conflict 27 — topology (name_hallucination_caused_overlap)
- description: position (-8, -4, 0) occupied by multiple rooms ['high tunnel (to the west of lab)', 'large quadrangular cellar', 'tunnel between the laboratory and the wiring']
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct
  - step 36: high tunnel (to the west of lab) --[west]--> wooden plank in east/west tunnel — correct
  - step 32: inside the hut --[down]--> large quadrangular cellar — correct
  - step 33: large quadrangular cellar --[west]--> mad scientist's laboratory — correct
  - step 58: large quadrangular cellar --[east]--> complicated junction of passages — correct
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated
  - step 55: tunnel between the laboratory and the wiring --[east]--> mad scientist's laboratory — src_hallucinated

## Conflict 28 — topology (name_hallucination_caused_overlap)
- description: position (-5, -3, 0) occupied by multiple rooms ['high tunnel (to the west of lab)', 'sentry-post', 'tunnel between the laboratory and the wiring']
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct
  - step 36: high tunnel (to the west of lab) --[west]--> wooden plank in east/west tunnel — correct
  - step 70: complicated junction of passages --[north]--> sentry-post — dst_hallucinated
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated
  - step 55: tunnel between the laboratory and the wiring --[east]--> mad scientist's laboratory — src_hallucinated

## Conflict 29 — topology (name_hallucination_caused_overlap)
- description: position (5, -1, 0) occupied by multiple rooms ['billiard room of the haunted house', 'entrance hall to haunted house', 'high tunnel (to the west of lab)', 'living room of haunted house', 'tunnel between the laboratory and the wiring']
  - step 61: large bedroom in haunted house --[east]--> billiard room of the haunted house — dst_hallucinated
  - step 62: billiard room of the haunted house --[north]--> pantry of haunted house — both_names_hallucinated
  - step 59: complicated junction of passages --[east]--> entrance hall to haunted house — correct
  - step 60: entrance hall to haunted house --[west]--> large bedroom in haunted house — hallucinated_edge
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct
  - step 36: high tunnel (to the west of lab) --[west]--> wooden plank in east/west tunnel — correct
  - step 63: pantry of haunted house --[south]--> living room of haunted house — src_hallucinated
  - step 65: living room of haunted house --[east]--> deserted railway platform — dst_hallucinated
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated
  - step 55: tunnel between the laboratory and the wiring --[east]--> mad scientist's laboratory — src_hallucinated

## Conflict 30 — topology (name_hallucination_caused_overlap)
- description: position (-6, -4, 0) occupied by multiple rooms ['high tunnel (east/west, path to west)', 'large quadrangular cellar']
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge
  - step 68: deserted railway platform --[north]--> high tunnel (east/west, path to west) — src_hallucinated
  - step 44: high tunnel (east/west, path to west) --[west]--> secret cavern — correct
  - step 69: high tunnel (east/west, path to west) --[east]--> complicated junction of passages — hallucinated_edge
  - step 32: inside the hut --[down]--> large quadrangular cellar — correct
  - step 33: large quadrangular cellar --[west]--> mad scientist's laboratory — correct
  - step 58: large quadrangular cellar --[east]--> complicated junction of passages — correct

## Conflict 31 — topology (name_hallucination_caused_overlap)
- description: position (2, -1, 0) occupied by multiple rooms ["mad scientist's laboratory", 'pantry of haunted house', 'secret cavern']
  - step 33: large quadrangular cellar --[west]--> mad scientist's laboratory — correct
  - step 55: tunnel between the laboratory and the wiring --[east]--> mad scientist's laboratory — src_hallucinated
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct
  - step 62: billiard room of the haunted house --[north]--> pantry of haunted house — both_names_hallucinated
  - step 63: pantry of haunted house --[south]--> living room of haunted house — src_hallucinated
  - step 44: high tunnel (east/west, path to west) --[west]--> secret cavern — correct
  - step 46: secret cavern --[south]--> low damp chamber — correct

## Conflict 32 — topology (name_hallucination_caused_overlap)
- description: position (0, -3, 0) occupied by multiple rooms ['deserted railway platform', "mad scientist's laboratory"]
  - step 65: living room of haunted house --[east]--> deserted railway platform — dst_hallucinated
  - step 68: deserted railway platform --[north]--> high tunnel (east/west, path to west) — src_hallucinated
  - step 33: large quadrangular cellar --[west]--> mad scientist's laboratory — correct
  - step 55: tunnel between the laboratory and the wiring --[east]--> mad scientist's laboratory — src_hallucinated
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct

## Conflict 33 — topology (name_hallucination_caused_overlap)
- description: position (-4, -3, 0) occupied by multiple rooms ["mad scientist's laboratory", 'pantry of haunted house', 'secret cavern']
  - step 33: large quadrangular cellar --[west]--> mad scientist's laboratory — correct
  - step 55: tunnel between the laboratory and the wiring --[east]--> mad scientist's laboratory — src_hallucinated
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct
  - step 62: billiard room of the haunted house --[north]--> pantry of haunted house — both_names_hallucinated
  - step 63: pantry of haunted house --[south]--> living room of haunted house — src_hallucinated
  - step 44: high tunnel (east/west, path to west) --[west]--> secret cavern — correct
  - step 46: secret cavern --[south]--> low damp chamber — correct

## Conflict 34 — topology (name_hallucination_caused_overlap)
- description: position (-7, -4, 0) occupied by multiple rooms ['large quadrangular cellar', "mad scientist's laboratory"]
  - step 32: inside the hut --[down]--> large quadrangular cellar — correct
  - step 33: large quadrangular cellar --[west]--> mad scientist's laboratory — correct
  - step 58: large quadrangular cellar --[east]--> complicated junction of passages — correct
  - step 55: tunnel between the laboratory and the wiring --[east]--> mad scientist's laboratory — src_hallucinated
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct

## Conflict 35 — topology (name_hallucination_caused_overlap)
- description: position (-1, -2, 0) occupied by multiple rooms ['deserted railway platform', "mad scientist's laboratory", 'pantry of haunted house', 'secret cavern']
  - step 65: living room of haunted house --[east]--> deserted railway platform — dst_hallucinated
  - step 68: deserted railway platform --[north]--> high tunnel (east/west, path to west) — src_hallucinated
  - step 33: large quadrangular cellar --[west]--> mad scientist's laboratory — correct
  - step 55: tunnel between the laboratory and the wiring --[east]--> mad scientist's laboratory — src_hallucinated
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct
  - step 62: billiard room of the haunted house --[north]--> pantry of haunted house — both_names_hallucinated
  - step 63: pantry of haunted house --[south]--> living room of haunted house — src_hallucinated
  - step 44: high tunnel (east/west, path to west) --[west]--> secret cavern — correct
  - step 46: secret cavern --[south]--> low damp chamber — correct

## Conflict 36 — topology (overlap_mixed)
- description: position (-7, -4, 1) occupied by multiple rooms ['flower garden', 'inside the hut', 'sandpit', 'vegetable garden']
  - step 1: outside the house --[north]--> flower garden — hallucinated_edge
  - step 14: remnants of a bonfire --[north]--> flower garden — correct
  - step 2: flower garden --[north]--> alley (north/south) — hallucinated_edge
  - step 28: outside the house --[north]--> inside the hut — correct
  - step 32: inside the hut --[down]--> large quadrangular cellar — correct
  - step 16: outside the house --[north]--> sandpit — hallucinated_edge
  - step 19: sandpit --[south]--> old untended grave — correct
  - step 7: alley (north/south) --[south]--> vegetable garden — hallucinated_edge
  - step 9: vegetable garden --[south]--> shrubbery — correct

## Conflict 37 — topology (overlap_mixed)
- description: position (-6, -4, 1) occupied by multiple rooms ['flower garden', 'inside the hut', 'sandpit']
  - step 1: outside the house --[north]--> flower garden — hallucinated_edge
  - step 14: remnants of a bonfire --[north]--> flower garden — correct
  - step 2: flower garden --[north]--> alley (north/south) — hallucinated_edge
  - step 28: outside the house --[north]--> inside the hut — correct
  - step 32: inside the hut --[down]--> large quadrangular cellar — correct
  - step 16: outside the house --[north]--> sandpit — hallucinated_edge
  - step 19: sandpit --[south]--> old untended grave — correct

## Conflict 38 — topology (overlap_mixed)
- description: position (-8, -4, 1) occupied by multiple rooms ['flower garden', 'inside the hut', 'sandpit', 'vegetable garden']
  - step 1: outside the house --[north]--> flower garden — hallucinated_edge
  - step 14: remnants of a bonfire --[north]--> flower garden — correct
  - step 2: flower garden --[north]--> alley (north/south) — hallucinated_edge
  - step 28: outside the house --[north]--> inside the hut — correct
  - step 32: inside the hut --[down]--> large quadrangular cellar — correct
  - step 16: outside the house --[north]--> sandpit — hallucinated_edge
  - step 19: sandpit --[south]--> old untended grave — correct
  - step 7: alley (north/south) --[south]--> vegetable garden — hallucinated_edge
  - step 9: vegetable garden --[south]--> shrubbery — correct

## Conflict 39 — topology (overlap_mixed)
- description: position (-9, -4, 1) occupied by multiple rooms ['flower garden', 'inside the hut', 'sandpit', 'vegetable garden']
  - step 1: outside the house --[north]--> flower garden — hallucinated_edge
  - step 14: remnants of a bonfire --[north]--> flower garden — correct
  - step 2: flower garden --[north]--> alley (north/south) — hallucinated_edge
  - step 28: outside the house --[north]--> inside the hut — correct
  - step 32: inside the hut --[down]--> large quadrangular cellar — correct
  - step 16: outside the house --[north]--> sandpit — hallucinated_edge
  - step 19: sandpit --[south]--> old untended grave — correct
  - step 7: alley (north/south) --[south]--> vegetable garden — hallucinated_edge
  - step 9: vegetable garden --[south]--> shrubbery — correct

## Conflict 40 — topology (name_hallucination_caused_overlap)
- description: position (-5, -4, 0) occupied by multiple rooms ['complicated junction of passages', 'large bedroom in haunted house']
  - step 58: large quadrangular cellar --[east]--> complicated junction of passages — correct
  - step 69: high tunnel (east/west, path to west) --[east]--> complicated junction of passages — hallucinated_edge
  - step 59: complicated junction of passages --[east]--> entrance hall to haunted house — correct
  - step 70: complicated junction of passages --[north]--> sentry-post — dst_hallucinated
  - step 60: entrance hall to haunted house --[west]--> large bedroom in haunted house — hallucinated_edge
  - step 61: large bedroom in haunted house --[east]--> billiard room of the haunted house — dst_hallucinated

## Conflict 41 — topology (name_hallucination_caused_overlap)
- description: position (-4, -4, 0) occupied by multiple rooms ['billiard room of the haunted house', 'entrance hall to haunted house', 'living room of haunted house', 'low damp chamber']
  - step 61: large bedroom in haunted house --[east]--> billiard room of the haunted house — dst_hallucinated
  - step 62: billiard room of the haunted house --[north]--> pantry of haunted house — both_names_hallucinated
  - step 59: complicated junction of passages --[east]--> entrance hall to haunted house — correct
  - step 60: entrance hall to haunted house --[west]--> large bedroom in haunted house — hallucinated_edge
  - step 63: pantry of haunted house --[south]--> living room of haunted house — src_hallucinated
  - step 65: living room of haunted house --[east]--> deserted railway platform — dst_hallucinated
  - step 46: secret cavern --[south]--> low damp chamber — correct

## Conflict 42 — topology (overlap_mixed)
- description: position (-6, -5, 1) occupied by multiple rooms ['edge of a large calm lake', 'old untended grave', 'outside the house', 'remnants of a bonfire']
  - step 22: outside the house --[east]--> edge of a large calm lake — correct
  - step 19: sandpit --[south]--> old untended grave — correct
  - step 21: old untended grave --[east]--> outside the house — correct
  - step 1: outside the house --[north]--> flower garden — hallucinated_edge
  - step 16: outside the house --[north]--> sandpit — hallucinated_edge
  - step 28: outside the house --[north]--> inside the hut — correct
  - step 12: shrubbery --[east]--> remnants of a bonfire — correct
  - step 14: remnants of a bonfire --[north]--> flower garden — correct

## Conflict 43 — topology (overlap_mixed)
- description: position (-9, -5, 1) occupied by multiple rooms ['old untended grave', 'outside the house', 'remnants of a bonfire', 'shrubbery']
  - step 19: sandpit --[south]--> old untended grave — correct
  - step 21: old untended grave --[east]--> outside the house — correct
  - step 1: outside the house --[north]--> flower garden — hallucinated_edge
  - step 16: outside the house --[north]--> sandpit — hallucinated_edge
  - step 22: outside the house --[east]--> edge of a large calm lake — correct
  - step 28: outside the house --[north]--> inside the hut — correct
  - step 12: shrubbery --[east]--> remnants of a bonfire — correct
  - step 14: remnants of a bonfire --[north]--> flower garden — correct
  - step 9: vegetable garden --[south]--> shrubbery — correct

## Conflict 44 — topology (overlap_mixed)
- description: position (-8, -5, 1) occupied by multiple rooms ['edge of a large calm lake', 'old untended grave', 'outside the house', 'remnants of a bonfire', 'shrubbery']
  - step 22: outside the house --[east]--> edge of a large calm lake — correct
  - step 19: sandpit --[south]--> old untended grave — correct
  - step 21: old untended grave --[east]--> outside the house — correct
  - step 1: outside the house --[north]--> flower garden — hallucinated_edge
  - step 16: outside the house --[north]--> sandpit — hallucinated_edge
  - step 28: outside the house --[north]--> inside the hut — correct
  - step 12: shrubbery --[east]--> remnants of a bonfire — correct
  - step 14: remnants of a bonfire --[north]--> flower garden — correct
  - step 9: vegetable garden --[south]--> shrubbery — correct

## Conflict 45 — topology (overlap_mixed)
- description: position (-7, -5, 1) occupied by multiple rooms ['edge of a large calm lake', 'old untended grave', 'outside the house', 'remnants of a bonfire', 'shrubbery']
  - step 22: outside the house --[east]--> edge of a large calm lake — correct
  - step 19: sandpit --[south]--> old untended grave — correct
  - step 21: old untended grave --[east]--> outside the house — correct
  - step 1: outside the house --[north]--> flower garden — hallucinated_edge
  - step 16: outside the house --[north]--> sandpit — hallucinated_edge
  - step 28: outside the house --[north]--> inside the hut — correct
  - step 12: shrubbery --[east]--> remnants of a bonfire — correct
  - step 14: remnants of a bonfire --[north]--> flower garden — correct
  - step 9: vegetable garden --[south]--> shrubbery — correct

## Conflict 46 — topology (overlap_mixed)
- description: position (-10, -4, 1) occupied by multiple rooms ['flower garden', 'sandpit', 'vegetable garden']
  - step 1: outside the house --[north]--> flower garden — hallucinated_edge
  - step 14: remnants of a bonfire --[north]--> flower garden — correct
  - step 2: flower garden --[north]--> alley (north/south) — hallucinated_edge
  - step 16: outside the house --[north]--> sandpit — hallucinated_edge
  - step 19: sandpit --[south]--> old untended grave — correct
  - step 7: alley (north/south) --[south]--> vegetable garden — hallucinated_edge
  - step 9: vegetable garden --[south]--> shrubbery — correct

## Conflict 47 — topology (false_positive_overlap)
- description: position (-10, -5, 1) occupied by multiple rooms ['old untended grave', 'shrubbery']
  - step 19: sandpit --[south]--> old untended grave — correct
  - step 21: old untended grave --[east]--> outside the house — correct
  - step 9: vegetable garden --[south]--> shrubbery — correct
  - step 12: shrubbery --[east]--> remnants of a bonfire — correct

## Conflict 48 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'alcove' reachable at conflicting positions [(-3, -2, 0), (-2, -2, 0), (-1, 0, 0), (0, 0, 0), (1, -1, 0), (4, 0, 0)]
  - step 41: high tunnel (east/west, alcove off to north) --[north]--> alcove — correct
  - step 42: alcove --[south]--> wooden plank in east/west tunnel — hallucinated_edge

## Conflict 49 — naming (naming_collision_on_correct_subgraph)
- description: node 'high tunnel (east/west, alcove off to north)' reachable at conflicting positions [(-3, -3, 0), (-2, -3, 0), (-1, -1, 0), (0, -2, 0), (0, -1, 0), (1, -2, 0), (3, -1, 0)]
  - step 37: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, alcove off to north) — correct
  - step 41: high tunnel (east/west, alcove off to north) --[north]--> alcove — correct

## Conflict 50 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'wooden plank in east/west tunnel' reachable at conflicting positions [(-9, -4, 0), (-6, -3, 0), (-3, -3, 0), (-2, -3, 0), (-1, -3, 0), (0, -1, 0), (1, -2, 0), (1, -1, 0), (4, -1, 0)]
  - step 36: high tunnel (to the west of lab) --[west]--> wooden plank in east/west tunnel — correct
  - step 42: alcove --[south]--> wooden plank in east/west tunnel — hallucinated_edge
  - step 37: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, alcove off to north) — correct
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated

## Conflict 51 — naming (naming_collision_on_correct_subgraph)
- description: node 'high tunnel (to the west of lab)' reachable at conflicting positions [(-8, -4, 0), (-5, -3, 0), (-1, -3, 0), (1, -1, 0), (2, -2, 0), (5, -1, 0)]
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct
  - step 36: high tunnel (to the west of lab) --[west]--> wooden plank in east/west tunnel — correct

## Conflict 52 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'high tunnel (east/west, path to west)' reachable at conflicting positions [(-6, -4, 0), (-3, -3, 0), (-1, -1, 0), (0, -2, 0), (3, -1, 0), (6, 0, 0)]
  - step 43: wooden plank in east/west tunnel --[west]--> high tunnel (east/west, path to west) — hallucinated_edge
  - step 68: deserted railway platform --[north]--> high tunnel (east/west, path to west) — src_hallucinated
  - step 44: high tunnel (east/west, path to west) --[west]--> secret cavern — correct
  - step 69: high tunnel (east/west, path to west) --[east]--> complicated junction of passages — hallucinated_edge

## Conflict 53 — naming (name_hallucination)
- description: node 'tunnel between the laboratory and the wiring' reachable at conflicting positions [(-8, -4, 0), (-5, -3, 0), (-1, -3, 0), (1, -1, 0), (2, -2, 0), (5, -1, 0)]
  - step 54: wooden plank in east/west tunnel --[east]--> tunnel between the laboratory and the wiring — dst_hallucinated
  - step 55: tunnel between the laboratory and the wiring --[east]--> mad scientist's laboratory — src_hallucinated

## Conflict 54 — naming (real_name_corrupted_by_neighbour_error)
- description: node "mad scientist's laboratory" reachable at conflicting positions [(-7, -4, 0), (-4, -3, 0), (-1, -2, 0), (0, -3, 0), (2, -1, 0)]
  - step 33: large quadrangular cellar --[west]--> mad scientist's laboratory — correct
  - step 55: tunnel between the laboratory and the wiring --[east]--> mad scientist's laboratory — src_hallucinated
  - step 34: mad scientist's laboratory --[west]--> high tunnel (to the west of lab) — correct

## Conflict 55 — naming (naming_collision_on_correct_subgraph)
- description: node 'large quadrangular cellar' reachable at conflicting positions [(-8, -4, 0), (-7, -4, 0), (-6, -4, 0), (-3, -3, 0), (0, -2, 0), (1, -3, 0), (3, -1, 0)]
  - step 32: inside the hut --[down]--> large quadrangular cellar — correct
  - step 33: large quadrangular cellar --[west]--> mad scientist's laboratory — correct
  - step 58: large quadrangular cellar --[east]--> complicated junction of passages — correct

## Conflict 56 — naming (naming_collision_on_correct_subgraph)
- description: node 'inside the hut' reachable at conflicting positions [(-9, -4, 1), (-8, -4, 1), (-7, -4, 1), (-6, -4, 1), (-3, -3, 1), (0, -2, 1), (3, -1, 1)]
  - step 28: outside the house --[north]--> inside the hut — correct
  - step 32: inside the hut --[down]--> large quadrangular cellar — correct

## Conflict 57 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'complicated junction of passages' reachable at conflicting positions [(-5, -4, 0), (-2, -3, 0), (0, -1, 0), (1, -2, 0), (4, -1, 0)]
  - step 58: large quadrangular cellar --[east]--> complicated junction of passages — correct
  - step 69: high tunnel (east/west, path to west) --[east]--> complicated junction of passages — hallucinated_edge
  - step 59: complicated junction of passages --[east]--> entrance hall to haunted house — correct
  - step 70: complicated junction of passages --[north]--> sentry-post — dst_hallucinated

## Conflict 58 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'entrance hall to haunted house' reachable at conflicting positions [(-4, -4, 0), (-1, -3, 0), (2, -2, 0), (5, -1, 0)]
  - step 59: complicated junction of passages --[east]--> entrance hall to haunted house — correct
  - step 60: entrance hall to haunted house --[west]--> large bedroom in haunted house — hallucinated_edge

## Conflict 59 — naming (name_hallucination)
- description: node 'sentry-post' reachable at conflicting positions [(-5, -3, 0), (-2, -2, 0), (1, -1, 0), (4, 0, 0)]
  - step 70: complicated junction of passages --[north]--> sentry-post — dst_hallucinated

## Conflict 60 — naming (naming_collision_on_correct_subgraph)
- description: node 'secret cavern' reachable at conflicting positions [(-4, -3, 0), (-2, -1, 0), (-1, -2, 0), (2, -1, 0)]
  - step 44: high tunnel (east/west, path to west) --[west]--> secret cavern — correct
  - step 46: secret cavern --[south]--> low damp chamber — correct

## Conflict 61 — naming (name_hallucination)
- description: node 'deserted railway platform' reachable at conflicting positions [(-3, -4, 0), (-1, -2, 0), (0, -3, 0), (3, -2, 0), (6, -1, 0)]
  - step 65: living room of haunted house --[east]--> deserted railway platform — dst_hallucinated
  - step 68: deserted railway platform --[north]--> high tunnel (east/west, path to west) — src_hallucinated

## Conflict 62 — naming (name_hallucination)
- description: node 'living room of haunted house' reachable at conflicting positions [(-4, -4, 0), (-1, -3, 0), (2, -2, 0), (5, -1, 0)]
  - step 63: pantry of haunted house --[south]--> living room of haunted house — src_hallucinated
  - step 65: living room of haunted house --[east]--> deserted railway platform — dst_hallucinated

## Conflict 63 — naming (name_hallucination)
- description: node 'pantry of haunted house' reachable at conflicting positions [(-4, -3, 0), (-1, -2, 0), (2, -1, 0), (5, 0, 0)]
  - step 62: billiard room of the haunted house --[north]--> pantry of haunted house — both_names_hallucinated
  - step 63: pantry of haunted house --[south]--> living room of haunted house — src_hallucinated

## Conflict 64 — naming (name_hallucination)
- description: node 'billiard room of the haunted house' reachable at conflicting positions [(-4, -4, 0), (-1, -3, 0), (2, -2, 0), (5, -1, 0)]
  - step 61: large bedroom in haunted house --[east]--> billiard room of the haunted house — dst_hallucinated
  - step 62: billiard room of the haunted house --[north]--> pantry of haunted house — both_names_hallucinated

## Conflict 65 — naming (name_hallucination)
- description: node 'large bedroom in haunted house' reachable at conflicting positions [(-5, -4, 0), (-2, -3, 0), (1, -2, 0), (4, -1, 0)]
  - step 60: entrance hall to haunted house --[west]--> large bedroom in haunted house — hallucinated_edge
  - step 61: large bedroom in haunted house --[east]--> billiard room of the haunted house — dst_hallucinated

## Conflict 66 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'outside the house' reachable at conflicting positions [(-9, -5, 1), (-8, -5, 1), (-7, -5, 1), (-6, -5, 1), (3, -2, 1)]
  - step 21: old untended grave --[east]--> outside the house — correct
  - step 1: outside the house --[north]--> flower garden — hallucinated_edge
  - step 16: outside the house --[north]--> sandpit — hallucinated_edge
  - step 22: outside the house --[east]--> edge of a large calm lake — correct
  - step 28: outside the house --[north]--> inside the hut — correct

## Conflict 67 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'flower garden' reachable at conflicting positions [(-10, -4, 1), (-9, -4, 1), (-8, -4, 1), (-7, -4, 1), (-6, -4, 1)]
  - step 1: outside the house --[north]--> flower garden — hallucinated_edge
  - step 14: remnants of a bonfire --[north]--> flower garden — correct
  - step 2: flower garden --[north]--> alley (north/south) — hallucinated_edge

## Conflict 68 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'sandpit' reachable at conflicting positions [(-10, -4, 1), (-9, -4, 1), (-8, -4, 1), (-7, -4, 1), (-6, -4, 1)]
  - step 16: outside the house --[north]--> sandpit — hallucinated_edge
  - step 19: sandpit --[south]--> old untended grave — correct

## Conflict 69 — naming (naming_collision_on_correct_subgraph)
- description: node 'old untended grave' reachable at conflicting positions [(-10, -5, 1), (-9, -5, 1), (-8, -5, 1), (-7, -5, 1), (-6, -5, 1)]
  - step 19: sandpit --[south]--> old untended grave — correct
  - step 21: old untended grave --[east]--> outside the house — correct

## Conflict 70 — naming (naming_collision_on_correct_subgraph)
- description: node 'edge of a large calm lake' reachable at conflicting positions [(-8, -5, 1), (-7, -5, 1), (-6, -5, 1), (-5, -5, 1)]
  - step 22: outside the house --[east]--> edge of a large calm lake — correct

## Conflict 71 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'alley (north/south)' reachable at conflicting positions [(-10, -3, 1), (-9, -3, 1), (-8, -3, 1), (-7, -3, 1), (-6, -3, 1)]
  - step 2: flower garden --[north]--> alley (north/south) — hallucinated_edge
  - step 3: alley (north/south) --[north]--> alley (north end) — correct
  - step 7: alley (north/south) --[south]--> vegetable garden — hallucinated_edge

## Conflict 72 — naming (naming_collision_on_correct_subgraph)
- description: node 'remnants of a bonfire' reachable at conflicting positions [(-9, -5, 1), (-8, -5, 1), (-7, -5, 1), (-6, -5, 1)]
  - step 12: shrubbery --[east]--> remnants of a bonfire — correct
  - step 14: remnants of a bonfire --[north]--> flower garden — correct

## Conflict 73 — naming (naming_collision_on_correct_subgraph)
- description: node 'shrubbery' reachable at conflicting positions [(-10, -5, 1), (-9, -5, 1), (-8, -5, 1), (-7, -5, 1)]
  - step 9: vegetable garden --[south]--> shrubbery — correct
  - step 12: shrubbery --[east]--> remnants of a bonfire — correct

## Conflict 74 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'vegetable garden' reachable at conflicting positions [(-10, -4, 1), (-9, -4, 1), (-8, -4, 1), (-7, -4, 1)]
  - step 7: alley (north/south) --[south]--> vegetable garden — hallucinated_edge
  - step 9: vegetable garden --[south]--> shrubbery — correct

## Conflict 75 — naming (naming_collision_on_correct_subgraph)
- description: node 'alley (north end)' reachable at conflicting positions [(-10, -2, 1), (-9, -2, 1), (-8, -2, 1), (-7, -2, 1)]
  - step 3: alley (north/south) --[north]--> alley (north end) — correct

## Conflict 76 — naming (naming_collision_on_correct_subgraph)
- description: node 'low damp chamber' reachable at conflicting positions [(-4, -4, 0), (-2, -2, 0), (-1, -3, 0), (2, -2, 0)]
  - step 46: secret cavern --[south]--> low damp chamber — correct
