# Conflict analysis: advent

- LLM edges: 34
- GT edges: 51
- Conflicts: 69
- Type distribution: {'direction': 2, 'topology': 34, 'naming': 33}
- Root-cause distribution: {'real_vs_hallucinated': 2, 'overlap_mixed': 5, 'false_positive_overlap': 15, 'name_hallucination_caused_overlap': 14, 'real_name_corrupted_by_neighbour_error': 8, 'naming_collision_on_correct_subgraph': 21, 'naming_mixed': 1, 'name_hallucination': 3}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'hall of the mountain king' has multiple outgoing edges labelled 'north'
  - step 25: hall of the mountain king --[north]--> low n/s passage — correct
  - step None: hall of the mountain king --[north]--> at "y2" — hallucinated_edge

## Conflict 2 — direction (real_vs_hallucinated)
- description: node 'at "y2"' has multiple outgoing edges labelled 'south'
  - step None: at "y2" --[south]--> low n/s passage — correct
  - step 34: at "y2" --[south]--> hall of the mountain king — hallucinated_edge

## Conflict 3 — topology (overlap_mixed)
- description: position (2, 2, 1) occupied by multiple rooms ['alcove', 'misty cavern']
  - step 53: alcove --[east]--> plover room — correct
  - step 57: alcove --[south]--> misty cavern — hallucinated_edge
  - step 59: misty cavern --[southeast]--> in swiss cheese room — hallucinated_edge

## Conflict 4 — topology (false_positive_overlap)
- description: position (1, 0, 0) occupied by multiple rooms ['in west pit', 'plover room']
  - step 46: at west end of twopit room --[down]--> in west pit — correct
  - step 53: alcove --[east]--> plover room — correct

## Conflict 5 — topology (false_positive_overlap)
- description: position (3, 2, 1) occupied by multiple rooms ['in west pit', 'plover room']
  - step 46: at west end of twopit room --[down]--> in west pit — correct
  - step 53: alcove --[east]--> plover room — correct

## Conflict 6 — topology (overlap_mixed)
- description: position (2, 1, 1) occupied by multiple rooms ['at east end of twopit room', 'misty cavern']
  - step 44: in swiss cheese room --[west]--> at east end of twopit room — correct
  - step 45: at east end of twopit room --[west]--> at west end of twopit room — correct
  - step 57: alcove --[south]--> misty cavern — hallucinated_edge
  - step 59: misty cavern --[southeast]--> in swiss cheese room — hallucinated_edge

## Conflict 7 — topology (overlap_mixed)
- description: position (3, 0, 1) occupied by multiple rooms ['in swiss cheese room', 'in tall e/w canyon']
  - step 43: in tall e/w canyon --[north]--> in swiss cheese room — correct
  - step 59: misty cavern --[southeast]--> in swiss cheese room — hallucinated_edge
  - step 44: in swiss cheese room --[west]--> at east end of twopit room — correct
  - step 60: in swiss cheese room --[east]--> in soft room — correct
  - step 63: in swiss cheese room --[northeast]--> bedquilt — correct
  - step 42: n/s canyon --[north]--> in tall e/w canyon — correct

## Conflict 8 — topology (name_hallucination_caused_overlap)
- description: position (5, 2, 2) occupied by multiple rooms ['in dusty rock room', 'in swiss cheese room']
  - step 65: at complex junction --[up]--> in dusty rock room — dst_hallucinated
  - step 66: in dusty rock room --[east]--> dirty passage — both_names_hallucinated
  - step 43: in tall e/w canyon --[north]--> in swiss cheese room — correct
  - step 59: misty cavern --[southeast]--> in swiss cheese room — hallucinated_edge
  - step 44: in swiss cheese room --[west]--> at east end of twopit room — correct
  - step 60: in swiss cheese room --[east]--> in soft room — correct
  - step 63: in swiss cheese room --[northeast]--> bedquilt — correct

## Conflict 9 — topology (name_hallucination_caused_overlap)
- description: position (3, -1, 1) occupied by multiple rooms ['in dusty rock room', 'in tall e/w canyon', 'n/s canyon']
  - step 65: at complex junction --[up]--> in dusty rock room — dst_hallucinated
  - step 66: in dusty rock room --[east]--> dirty passage — both_names_hallucinated
  - step 42: n/s canyon --[north]--> in tall e/w canyon — correct
  - step 43: in tall e/w canyon --[north]--> in swiss cheese room — correct
  - step 41: secret e/w canyon above tight canyon --[down]--> n/s canyon — correct

## Conflict 10 — topology (false_positive_overlap)
- description: position (5, 0, 2) occupied by multiple rooms ['below the grate', 'in tall e/w canyon', 'n/s canyon']
  - step 9: outside grate --[down]--> below the grate — correct
  - step 10: below the grate --[west]--> in cobble — correct
  - step 42: n/s canyon --[north]--> in tall e/w canyon — correct
  - step 43: in tall e/w canyon --[north]--> in swiss cheese room — correct
  - step 41: secret e/w canyon above tight canyon --[down]--> n/s canyon — correct

## Conflict 11 — topology (name_hallucination_caused_overlap)
- description: position (5, 1, 2) occupied by multiple rooms ['below the grate', 'in dusty rock room', 'in tall e/w canyon']
  - step 9: outside grate --[down]--> below the grate — correct
  - step 10: below the grate --[west]--> in cobble — correct
  - step 65: at complex junction --[up]--> in dusty rock room — dst_hallucinated
  - step 66: in dusty rock room --[east]--> dirty passage — both_names_hallucinated
  - step 42: n/s canyon --[north]--> in tall e/w canyon — correct
  - step 43: in tall e/w canyon --[north]--> in swiss cheese room — correct

## Conflict 12 — topology (name_hallucination_caused_overlap)
- description: position (4, 0, 1) occupied by multiple rooms ['dirty passage', 'in soft room']
  - step 66: in dusty rock room --[east]--> dirty passage — both_names_hallucinated
  - step 67: dirty passage --[up]--> low n/s passage — src_hallucinated
  - step 60: in swiss cheese room --[east]--> in soft room — correct

## Conflict 13 — topology (false_positive_overlap)
- description: position (4, 1, 1) occupied by multiple rooms ['bedquilt', 'in soft room']
  - step 63: in swiss cheese room --[northeast]--> bedquilt — correct
  - step 64: bedquilt --[east]--> at complex junction — correct
  - step 60: in swiss cheese room --[east]--> in soft room — correct

## Conflict 14 — topology (name_hallucination_caused_overlap)
- description: position (6, 2, 2) occupied by multiple rooms ['dirty passage', 'in soft room']
  - step 66: in dusty rock room --[east]--> dirty passage — both_names_hallucinated
  - step 67: dirty passage --[up]--> low n/s passage — src_hallucinated
  - step 60: in swiss cheese room --[east]--> in soft room — correct

## Conflict 15 — topology (name_hallucination_caused_overlap)
- description: position (6, 2, 3) occupied by multiple rooms ['at "y2"', 'hall of the mountain king', 'inside building', 'low n/s passage']
  - step 28: low n/s passage --[north]--> at "y2" — correct
  - step 33: inside building --[south]--> at "y2" — hallucinated_edge
  - step 34: at "y2" --[south]--> hall of the mountain king — hallucinated_edge
  - step 22: in hall of mists --[north]--> hall of the mountain king — correct
  - step 25: hall of the mountain king --[north]--> low n/s passage — correct
  - step 35: hall of the mountain king --[southwest]--> secret e/w canyon above tight canyon — correct
  - step 1: end of road --[east]--> inside building — correct
  - step 67: dirty passage --[up]--> low n/s passage — src_hallucinated

## Conflict 16 — topology (name_hallucination_caused_overlap)
- description: position (4, -1, 2) occupied by multiple rooms ['at "y2"', 'hall of the mountain king', 'in cobble', 'low n/s passage']
  - step 28: low n/s passage --[north]--> at "y2" — correct
  - step 33: inside building --[south]--> at "y2" — hallucinated_edge
  - step 34: at "y2" --[south]--> hall of the mountain king — hallucinated_edge
  - step 22: in hall of mists --[north]--> hall of the mountain king — correct
  - step 25: hall of the mountain king --[north]--> low n/s passage — correct
  - step 35: hall of the mountain king --[southwest]--> secret e/w canyon above tight canyon — correct
  - step 10: below the grate --[west]--> in cobble — correct
  - step 12: in cobble --[west]--> darkness — dst_hallucinated
  - step 67: dirty passage --[up]--> low n/s passage — src_hallucinated

## Conflict 17 — topology (name_hallucination_caused_overlap)
- description: position (6, 1, 3) occupied by multiple rooms ['at "y2"', 'hall of the mountain king', 'low n/s passage']
  - step 28: low n/s passage --[north]--> at "y2" — correct
  - step 33: inside building --[south]--> at "y2" — hallucinated_edge
  - step 34: at "y2" --[south]--> hall of the mountain king — hallucinated_edge
  - step 22: in hall of mists --[north]--> hall of the mountain king — correct
  - step 25: hall of the mountain king --[north]--> low n/s passage — correct
  - step 35: hall of the mountain king --[southwest]--> secret e/w canyon above tight canyon — correct
  - step 67: dirty passage --[up]--> low n/s passage — src_hallucinated

## Conflict 18 — topology (name_hallucination_caused_overlap)
- description: position (4, 0, 2) occupied by multiple rooms ['at "y2"', 'hall of the mountain king', 'in cobble', 'low n/s passage']
  - step 28: low n/s passage --[north]--> at "y2" — correct
  - step 33: inside building --[south]--> at "y2" — hallucinated_edge
  - step 34: at "y2" --[south]--> hall of the mountain king — hallucinated_edge
  - step 22: in hall of mists --[north]--> hall of the mountain king — correct
  - step 25: hall of the mountain king --[north]--> low n/s passage — correct
  - step 35: hall of the mountain king --[southwest]--> secret e/w canyon above tight canyon — correct
  - step 10: below the grate --[west]--> in cobble — correct
  - step 12: in cobble --[west]--> darkness — dst_hallucinated
  - step 67: dirty passage --[up]--> low n/s passage — src_hallucinated

## Conflict 19 — topology (name_hallucination_caused_overlap)
- description: position (6, 0, 3) occupied by multiple rooms ['hall of the mountain king', 'in hall of mists', 'low n/s passage']
  - step 22: in hall of mists --[north]--> hall of the mountain king — correct
  - step 34: at "y2" --[south]--> hall of the mountain king — hallucinated_edge
  - step 25: hall of the mountain king --[north]--> low n/s passage — correct
  - step 35: hall of the mountain king --[southwest]--> secret e/w canyon above tight canyon — correct
  - step 18: at top of small pit --[down]--> in hall of mists — correct
  - step 19: in hall of mists --[south]--> low room — correct
  - step 67: dirty passage --[up]--> low n/s passage — src_hallucinated
  - step 28: low n/s passage --[north]--> at "y2" — correct

## Conflict 20 — topology (overlap_mixed)
- description: position (4, -2, 2) occupied by multiple rooms ['hall of the mountain king', 'in hall of mists']
  - step 22: in hall of mists --[north]--> hall of the mountain king — correct
  - step 34: at "y2" --[south]--> hall of the mountain king — hallucinated_edge
  - step 25: hall of the mountain king --[north]--> low n/s passage — correct
  - step 35: hall of the mountain king --[southwest]--> secret e/w canyon above tight canyon — correct
  - step 18: at top of small pit --[down]--> in hall of mists — correct
  - step 19: in hall of mists --[south]--> low room — correct

## Conflict 21 — topology (overlap_mixed)
- description: position (6, 3, 3) occupied by multiple rooms ['at "y2"', 'inside building']
  - step 28: low n/s passage --[north]--> at "y2" — correct
  - step 33: inside building --[south]--> at "y2" — hallucinated_edge
  - step 34: at "y2" --[south]--> hall of the mountain king — hallucinated_edge
  - step 1: end of road --[east]--> inside building — correct

## Conflict 22 — topology (name_hallucination_caused_overlap)
- description: position (4, 1, 2) occupied by multiple rooms ['at "y2"', 'in cobble', 'inside building']
  - step 28: low n/s passage --[north]--> at "y2" — correct
  - step 33: inside building --[south]--> at "y2" — hallucinated_edge
  - step 34: at "y2" --[south]--> hall of the mountain king — hallucinated_edge
  - step 10: below the grate --[west]--> in cobble — correct
  - step 12: in cobble --[west]--> darkness — dst_hallucinated
  - step 1: end of road --[east]--> inside building — correct

## Conflict 23 — topology (false_positive_overlap)
- description: position (4, -3, 2) occupied by multiple rooms ['in hall of mists', 'low room']
  - step 18: at top of small pit --[down]--> in hall of mists — correct
  - step 19: in hall of mists --[south]--> low room — correct
  - step 22: in hall of mists --[north]--> hall of the mountain king — correct

## Conflict 24 — topology (false_positive_overlap)
- description: position (6, -1, 3) occupied by multiple rooms ['in hall of mists', 'low room']
  - step 18: at top of small pit --[down]--> in hall of mists — correct
  - step 19: in hall of mists --[south]--> low room — correct
  - step 22: in hall of mists --[north]--> hall of the mountain king — correct

## Conflict 25 — topology (false_positive_overlap)
- description: position (5, -1, 3) occupied by multiple rooms ['outside grate', 'secret e/w canyon above tight canyon']
  - step 6: at slit in streambed --[south]--> outside grate — correct
  - step 9: outside grate --[down]--> below the grate — correct
  - step 35: hall of the mountain king --[southwest]--> secret e/w canyon above tight canyon — correct
  - step 36: secret e/w canyon above tight canyon --[west]--> secret canyon — correct
  - step 41: secret e/w canyon above tight canyon --[down]--> n/s canyon — correct

## Conflict 26 — topology (false_positive_overlap)
- description: position (3, -2, 2) occupied by multiple rooms ['outside grate', 'secret e/w canyon above tight canyon']
  - step 6: at slit in streambed --[south]--> outside grate — correct
  - step 9: outside grate --[down]--> below the grate — correct
  - step 35: hall of the mountain king --[southwest]--> secret e/w canyon above tight canyon — correct
  - step 36: secret e/w canyon above tight canyon --[west]--> secret canyon — correct
  - step 41: secret e/w canyon above tight canyon --[down]--> n/s canyon — correct

## Conflict 27 — topology (false_positive_overlap)
- description: position (5, 0, 3) occupied by multiple rooms ['at slit in streambed', 'outside grate', 'secret e/w canyon above tight canyon']
  - step 5: in a valley --[south]--> at slit in streambed — correct
  - step 6: at slit in streambed --[south]--> outside grate — correct
  - step 9: outside grate --[down]--> below the grate — correct
  - step 35: hall of the mountain king --[southwest]--> secret e/w canyon above tight canyon — correct
  - step 36: secret e/w canyon above tight canyon --[west]--> secret canyon — correct
  - step 41: secret e/w canyon above tight canyon --[down]--> n/s canyon — correct

## Conflict 28 — topology (name_hallucination_caused_overlap)
- description: position (3, -1, 2) occupied by multiple rooms ['at slit in streambed', 'darkness', 'secret e/w canyon above tight canyon']
  - step 5: in a valley --[south]--> at slit in streambed — correct
  - step 6: at slit in streambed --[south]--> outside grate — correct
  - step 12: in cobble --[west]--> darkness — dst_hallucinated
  - step 35: hall of the mountain king --[southwest]--> secret e/w canyon above tight canyon — correct
  - step 36: secret e/w canyon above tight canyon --[west]--> secret canyon — correct
  - step 41: secret e/w canyon above tight canyon --[down]--> n/s canyon — correct

## Conflict 29 — topology (false_positive_overlap)
- description: position (3, -2, 1) occupied by multiple rooms ['below the grate', 'n/s canyon']
  - step 9: outside grate --[down]--> below the grate — correct
  - step 10: below the grate --[west]--> in cobble — correct
  - step 41: secret e/w canyon above tight canyon --[down]--> n/s canyon — correct
  - step 42: n/s canyon --[north]--> in tall e/w canyon — correct

## Conflict 30 — topology (false_positive_overlap)
- description: position (5, -1, 2) occupied by multiple rooms ['below the grate', 'n/s canyon']
  - step 9: outside grate --[down]--> below the grate — correct
  - step 10: below the grate --[west]--> in cobble — correct
  - step 41: secret e/w canyon above tight canyon --[down]--> n/s canyon — correct
  - step 42: n/s canyon --[north]--> in tall e/w canyon — correct

## Conflict 31 — topology (false_positive_overlap)
- description: position (5, 3, 3) occupied by multiple rooms ['end of road', 'in a valley']
  - step 1: end of road --[east]--> inside building — correct
  - step 4: end of road --[south]--> in a valley — correct
  - step 5: in a valley --[south]--> at slit in streambed — correct

## Conflict 32 — topology (name_hallucination_caused_overlap)
- description: position (3, 1, 2) occupied by multiple rooms ['darkness', 'end of road']
  - step 12: in cobble --[west]--> darkness — dst_hallucinated
  - step 1: end of road --[east]--> inside building — correct
  - step 4: end of road --[south]--> in a valley — correct

## Conflict 33 — topology (false_positive_overlap)
- description: position (5, 2, 3) occupied by multiple rooms ['at slit in streambed', 'end of road', 'in a valley']
  - step 5: in a valley --[south]--> at slit in streambed — correct
  - step 6: at slit in streambed --[south]--> outside grate — correct
  - step 1: end of road --[east]--> inside building — correct
  - step 4: end of road --[south]--> in a valley — correct

## Conflict 34 — topology (name_hallucination_caused_overlap)
- description: position (3, 0, 2) occupied by multiple rooms ['darkness', 'in a valley']
  - step 12: in cobble --[west]--> darkness — dst_hallucinated
  - step 4: end of road --[south]--> in a valley — correct
  - step 5: in a valley --[south]--> at slit in streambed — correct

## Conflict 35 — topology (false_positive_overlap)
- description: position (5, 1, 3) occupied by multiple rooms ['at slit in streambed', 'in a valley', 'outside grate']
  - step 5: in a valley --[south]--> at slit in streambed — correct
  - step 6: at slit in streambed --[south]--> outside grate — correct
  - step 4: end of road --[south]--> in a valley — correct
  - step 9: outside grate --[down]--> below the grate — correct

## Conflict 36 — topology (false_positive_overlap)
- description: position (6, -2, 3) occupied by multiple rooms ['low room', 'sloping e/w canyon']
  - step 19: in hall of mists --[south]--> low room — correct
  - step 14: in debris room --[west]--> sloping e/w canyon — correct
  - step 15: sloping e/w canyon --[west]--> orange river chamber — correct

## Conflict 37 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'alcove' reachable at conflicting positions [(0, 0, 0), (2, 2, 1), (2, 3, 1), (4, 4, 2)]
  - step 53: alcove --[east]--> plover room — correct
  - step 57: alcove --[south]--> misty cavern — hallucinated_edge

## Conflict 38 — naming (naming_collision_on_correct_subgraph)
- description: node 'plover room' reachable at conflicting positions [(1, 0, 0), (3, 2, 1), (3, 3, 1), (5, 4, 2)]
  - step 53: alcove --[east]--> plover room — correct

## Conflict 39 — naming (naming_mixed)
- description: node 'misty cavern' reachable at conflicting positions [(0, -1, 0), (2, 1, 1), (2, 2, 1), (4, 3, 2)]
  - step 57: alcove --[south]--> misty cavern — hallucinated_edge
  - step 59: misty cavern --[southeast]--> in swiss cheese room — hallucinated_edge

## Conflict 40 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'in swiss cheese room' reachable at conflicting positions [(1, -2, 0), (3, 0, 1), (3, 1, 1), (5, 2, 2)]
  - step 43: in tall e/w canyon --[north]--> in swiss cheese room — correct
  - step 59: misty cavern --[southeast]--> in swiss cheese room — hallucinated_edge
  - step 44: in swiss cheese room --[west]--> at east end of twopit room — correct
  - step 60: in swiss cheese room --[east]--> in soft room — correct
  - step 63: in swiss cheese room --[northeast]--> bedquilt — correct

## Conflict 41 — naming (naming_collision_on_correct_subgraph)
- description: node 'in tall e/w canyon' reachable at conflicting positions [(1, -3, 0), (3, -1, 1), (3, 0, 1), (5, 0, 2), (5, 1, 2)]
  - step 42: n/s canyon --[north]--> in tall e/w canyon — correct
  - step 43: in tall e/w canyon --[north]--> in swiss cheese room — correct

## Conflict 42 — naming (naming_collision_on_correct_subgraph)
- description: node 'at east end of twopit room' reachable at conflicting positions [(0, -2, 0), (2, 0, 1), (2, 1, 1), (4, 2, 2)]
  - step 44: in swiss cheese room --[west]--> at east end of twopit room — correct
  - step 45: at east end of twopit room --[west]--> at west end of twopit room — correct

## Conflict 43 — naming (naming_collision_on_correct_subgraph)
- description: node 'in soft room' reachable at conflicting positions [(2, -2, 0), (4, 0, 1), (4, 1, 1), (6, 2, 2)]
  - step 60: in swiss cheese room --[east]--> in soft room — correct

## Conflict 44 — naming (naming_collision_on_correct_subgraph)
- description: node 'bedquilt' reachable at conflicting positions [(2, -1, 0), (4, 1, 1), (4, 2, 1), (6, 3, 2)]
  - step 63: in swiss cheese room --[northeast]--> bedquilt — correct
  - step 64: bedquilt --[east]--> at complex junction — correct

## Conflict 45 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'at complex junction' reachable at conflicting positions [(3, -1, 0), (5, 1, 1), (5, 2, 1), (7, 3, 2)]
  - step 64: bedquilt --[east]--> at complex junction — correct
  - step 65: at complex junction --[up]--> in dusty rock room — dst_hallucinated

## Conflict 46 — naming (name_hallucination)
- description: node 'in dusty rock room' reachable at conflicting positions [(3, -1, 1), (5, 1, 2), (5, 2, 2), (7, 3, 3)]
  - step 65: at complex junction --[up]--> in dusty rock room — dst_hallucinated
  - step 66: in dusty rock room --[east]--> dirty passage — both_names_hallucinated

## Conflict 47 — naming (name_hallucination)
- description: node 'dirty passage' reachable at conflicting positions [(4, -1, 1), (4, 0, 1), (6, 1, 2), (6, 2, 2), (8, 3, 3)]
  - step 66: in dusty rock room --[east]--> dirty passage — both_names_hallucinated
  - step 67: dirty passage --[up]--> low n/s passage — src_hallucinated

## Conflict 48 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'low n/s passage' reachable at conflicting positions [(4, -1, 2), (4, 0, 2), (6, 0, 3), (6, 1, 3), (6, 2, 3), (8, 3, 4)]
  - step 25: hall of the mountain king --[north]--> low n/s passage — correct
  - step 67: dirty passage --[up]--> low n/s passage — src_hallucinated
  - step 28: low n/s passage --[north]--> at "y2" — correct

## Conflict 49 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'hall of the mountain king' reachable at conflicting positions [(4, -2, 2), (4, -1, 2), (4, 0, 2), (6, 0, 3), (6, 1, 3), (6, 2, 3)]
  - step 22: in hall of mists --[north]--> hall of the mountain king — correct
  - step 34: at "y2" --[south]--> hall of the mountain king — hallucinated_edge
  - step 25: hall of the mountain king --[north]--> low n/s passage — correct
  - step 35: hall of the mountain king --[southwest]--> secret e/w canyon above tight canyon — correct

## Conflict 50 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'at "y2"' reachable at conflicting positions [(4, -1, 2), (4, 0, 2), (4, 1, 2), (6, 1, 3), (6, 2, 3), (6, 3, 3)]
  - step 28: low n/s passage --[north]--> at "y2" — correct
  - step 33: inside building --[south]--> at "y2" — hallucinated_edge
  - step 34: at "y2" --[south]--> hall of the mountain king — hallucinated_edge

## Conflict 51 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'inside building' reachable at conflicting positions [(4, 1, 2), (6, 2, 3), (6, 3, 3), (6, 4, 3)]
  - step 1: end of road --[east]--> inside building — correct
  - step 33: inside building --[south]--> at "y2" — hallucinated_edge

## Conflict 52 — naming (naming_collision_on_correct_subgraph)
- description: node 'in hall of mists' reachable at conflicting positions [(4, -3, 2), (4, -2, 2), (6, -1, 3), (6, 0, 3)]
  - step 18: at top of small pit --[down]--> in hall of mists — correct
  - step 19: in hall of mists --[south]--> low room — correct
  - step 22: in hall of mists --[north]--> hall of the mountain king — correct

## Conflict 53 — naming (naming_collision_on_correct_subgraph)
- description: node 'secret e/w canyon above tight canyon' reachable at conflicting positions [(3, -3, 2), (3, -2, 2), (3, -1, 2), (5, -1, 3), (5, 0, 3)]
  - step 35: hall of the mountain king --[southwest]--> secret e/w canyon above tight canyon — correct
  - step 36: secret e/w canyon above tight canyon --[west]--> secret canyon — correct
  - step 41: secret e/w canyon above tight canyon --[down]--> n/s canyon — correct

## Conflict 54 — naming (naming_collision_on_correct_subgraph)
- description: node 'secret canyon' reachable at conflicting positions [(2, -2, 2), (2, -1, 2), (4, -1, 3), (4, 0, 3)]
  - step 36: secret e/w canyon above tight canyon --[west]--> secret canyon — correct

## Conflict 55 — naming (naming_collision_on_correct_subgraph)
- description: node 'n/s canyon' reachable at conflicting positions [(1, -4, 0), (3, -2, 1), (3, -1, 1), (5, -1, 2), (5, 0, 2)]
  - step 41: secret e/w canyon above tight canyon --[down]--> n/s canyon — correct
  - step 42: n/s canyon --[north]--> in tall e/w canyon — correct

## Conflict 56 — naming (naming_collision_on_correct_subgraph)
- description: node 'at west end of twopit room' reachable at conflicting positions [(-1, -2, 0), (1, 0, 1), (1, 1, 1), (3, 2, 2)]
  - step 45: at east end of twopit room --[west]--> at west end of twopit room — correct
  - step 46: at west end of twopit room --[down]--> in west pit — correct

## Conflict 57 — naming (naming_collision_on_correct_subgraph)
- description: node 'in west pit' reachable at conflicting positions [(-1, -2, -1), (1, 0, 0), (1, 1, 0), (3, 2, 1)]
  - step 46: at west end of twopit room --[down]--> in west pit — correct

## Conflict 58 — naming (naming_collision_on_correct_subgraph)
- description: node 'end of road' reachable at conflicting positions [(3, 1, 2), (5, 2, 3), (5, 3, 3), (5, 4, 3)]
  - step 1: end of road --[east]--> inside building — correct
  - step 4: end of road --[south]--> in a valley — correct

## Conflict 59 — naming (naming_collision_on_correct_subgraph)
- description: node 'in a valley' reachable at conflicting positions [(3, 0, 2), (5, 1, 3), (5, 2, 3), (5, 3, 3)]
  - step 4: end of road --[south]--> in a valley — correct
  - step 5: in a valley --[south]--> at slit in streambed — correct

## Conflict 60 — naming (naming_collision_on_correct_subgraph)
- description: node 'at slit in streambed' reachable at conflicting positions [(3, -1, 2), (5, 0, 3), (5, 1, 3), (5, 2, 3)]
  - step 5: in a valley --[south]--> at slit in streambed — correct
  - step 6: at slit in streambed --[south]--> outside grate — correct

## Conflict 61 — naming (naming_collision_on_correct_subgraph)
- description: node 'outside grate' reachable at conflicting positions [(3, -2, 2), (5, -1, 3), (5, 0, 3), (5, 1, 3)]
  - step 6: at slit in streambed --[south]--> outside grate — correct
  - step 9: outside grate --[down]--> below the grate — correct

## Conflict 62 — naming (naming_collision_on_correct_subgraph)
- description: node 'below the grate' reachable at conflicting positions [(3, -2, 1), (5, -1, 2), (5, 0, 2), (5, 1, 2)]
  - step 9: outside grate --[down]--> below the grate — correct
  - step 10: below the grate --[west]--> in cobble — correct

## Conflict 63 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'in cobble' reachable at conflicting positions [(2, -2, 1), (4, -1, 2), (4, 0, 2), (4, 1, 2)]
  - step 10: below the grate --[west]--> in cobble — correct
  - step 12: in cobble --[west]--> darkness — dst_hallucinated

## Conflict 64 — naming (name_hallucination)
- description: node 'darkness' reachable at conflicting positions [(1, -2, 1), (3, -1, 2), (3, 0, 2), (3, 1, 2)]
  - step 12: in cobble --[west]--> darkness — dst_hallucinated

## Conflict 65 — naming (naming_collision_on_correct_subgraph)
- description: node 'at top of small pit' reachable at conflicting positions [(4, -3, 3), (4, -2, 3), (6, -1, 4), (6, 0, 4)]
  - step 17: orange river chamber --[west]--> at top of small pit — correct
  - step 18: at top of small pit --[down]--> in hall of mists — correct

## Conflict 66 — naming (naming_collision_on_correct_subgraph)
- description: node 'low room' reachable at conflicting positions [(4, -4, 2), (4, -3, 2), (6, -2, 3), (6, -1, 3)]
  - step 19: in hall of mists --[south]--> low room — correct

## Conflict 67 — naming (naming_collision_on_correct_subgraph)
- description: node 'orange river chamber' reachable at conflicting positions [(5, -3, 3), (5, -2, 3), (7, -1, 4), (7, 0, 4)]
  - step 15: sloping e/w canyon --[west]--> orange river chamber — correct
  - step 17: orange river chamber --[west]--> at top of small pit — correct

## Conflict 68 — naming (naming_collision_on_correct_subgraph)
- description: node 'sloping e/w canyon' reachable at conflicting positions [(6, -3, 3), (6, -2, 3), (8, -1, 4), (8, 0, 4)]
  - step 14: in debris room --[west]--> sloping e/w canyon — correct
  - step 15: sloping e/w canyon --[west]--> orange river chamber — correct

## Conflict 69 — naming (naming_collision_on_correct_subgraph)
- description: node 'in debris room' reachable at conflicting positions [(7, -3, 3), (7, -2, 3), (9, -1, 4), (9, 0, 4)]
  - step 14: in debris room --[west]--> sloping e/w canyon — correct
