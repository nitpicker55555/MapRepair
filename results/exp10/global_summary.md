# Global conflict-cause analysis (LLM-mapped gpt-4.1 on MANGO)

Games analysed: 42
Total conflicts: 928
Type distribution: {'direction': 86, 'topology': 433, 'naming': 409}

## Root-cause distribution
| Root cause | Count | % of total |
|------------|------:|-----------:|
| naming_collision_on_correct_subgraph | 242 | 26.1 |
| overlap_mixed | 159 | 17.1 |
| name_hallucination_caused_overlap | 141 | 15.2 |
| real_name_corrupted_by_neighbour_error | 113 | 12.2 |
| false_positive_overlap | 96 | 10.3 |
| name_hallucination | 57 | 6.1 |
| real_vs_hallucinated | 46 | 5.0 |
| wrong_direction_caused_overlap | 37 | 4.0 |
| naming_mixed | 22 | 2.4 |
| all_hallucinated_edges | 7 | 0.8 |
| src_dst_swap | 5 | 0.5 |
| direction_mixed | 3 | 0.3 |

## Examples per root cause

### real_vs_hallucinated

- **advent** (direction): _node 'hall of the mountain king' has multiple outgoing edges labelled 'north'_
  - step 25: `hall of the mountain king --[north]--> low n/s passage` (correct)
  - step None: `hall of the mountain king --[north]--> at "y2"` (hallucinated_edge)

- **advent** (direction): _node 'at "y2"' has multiple outgoing edges labelled 'south'_
  - step None: `at "y2" --[south]--> low n/s passage` (correct)
  - step 34: `at "y2" --[south]--> hall of the mountain king` (hallucinated_edge)

- **adventureland** (direction): _node 'maze (exit north, south, east, down)' has multiple outgoing edges labelled 'down'_
  - step 31: `maze (exit north, south, east, down) --[down]--> hall` (hallucinated_edge)
  - step 37: `maze (exit north, south, east, down) --[down]--> maze (exit west, up)` (correct)

### overlap_mixed

- **advent** (topology): _position (2, 2, 1) occupied by multiple rooms ['alcove', 'misty cavern']_
  - step 53: `alcove --[east]--> plover room` (correct)
  - step 57: `alcove --[south]--> misty cavern` (hallucinated_edge)
  - step 59: `misty cavern --[southeast]--> in swiss cheese room` (hallucinated_edge)

- **advent** (topology): _position (2, 1, 1) occupied by multiple rooms ['at east end of twopit room', 'misty cavern']_
  - step 44: `in swiss cheese room --[west]--> at east end of twopit room` (correct)
  - step 45: `at east end of twopit room --[west]--> at west end of twopit room` (correct)
  - step 57: `alcove --[south]--> misty cavern` (hallucinated_edge)

- **advent** (topology): _position (3, 0, 1) occupied by multiple rooms ['in swiss cheese room', 'in tall e/w canyon']_
  - step 43: `in tall e/w canyon --[north]--> in swiss cheese room` (correct)
  - step 59: `misty cavern --[southeast]--> in swiss cheese room` (hallucinated_edge)
  - step 44: `in swiss cheese room --[west]--> at east end of twopit room` (correct)

### false_positive_overlap

- **advent** (topology): _position (1, 0, 0) occupied by multiple rooms ['in west pit', 'plover room']_
  - step 46: `at west end of twopit room --[down]--> in west pit` (correct)
  - step 53: `alcove --[east]--> plover room` (correct)

- **advent** (topology): _position (3, 2, 1) occupied by multiple rooms ['in west pit', 'plover room']_
  - step 46: `at west end of twopit room --[down]--> in west pit` (correct)
  - step 53: `alcove --[east]--> plover room` (correct)

- **advent** (topology): _position (5, 0, 2) occupied by multiple rooms ['below the grate', 'in tall e/w canyon', 'n/s canyon']_
  - step 9: `outside grate --[down]--> below the grate` (correct)
  - step 10: `below the grate --[west]--> in cobble` (correct)
  - step 42: `n/s canyon --[north]--> in tall e/w canyon` (correct)

### name_hallucination_caused_overlap

- **advent** (topology): _position (5, 2, 2) occupied by multiple rooms ['in dusty rock room', 'in swiss cheese room']_
  - step 65: `at complex junction --[up]--> in dusty rock room` (dst_hallucinated)
  - step 66: `in dusty rock room --[east]--> dirty passage` (both_names_hallucinated)
  - step 43: `in tall e/w canyon --[north]--> in swiss cheese room` (correct)

- **advent** (topology): _position (3, -1, 1) occupied by multiple rooms ['in dusty rock room', 'in tall e/w canyon', 'n/s canyon']_
  - step 65: `at complex junction --[up]--> in dusty rock room` (dst_hallucinated)
  - step 66: `in dusty rock room --[east]--> dirty passage` (both_names_hallucinated)
  - step 42: `n/s canyon --[north]--> in tall e/w canyon` (correct)

- **advent** (topology): _position (5, 1, 2) occupied by multiple rooms ['below the grate', 'in dusty rock room', 'in tall e/w canyon']_
  - step 9: `outside grate --[down]--> below the grate` (correct)
  - step 10: `below the grate --[west]--> in cobble` (correct)
  - step 65: `at complex junction --[up]--> in dusty rock room` (dst_hallucinated)

### real_name_corrupted_by_neighbour_error

- **advent** (naming): _node 'alcove' reachable at conflicting positions [(0, 0, 0), (2, 2, 1), (2, 3, 1), (4, 4, 2)]_
  - step 53: `alcove --[east]--> plover room` (correct)
  - step 57: `alcove --[south]--> misty cavern` (hallucinated_edge)

- **advent** (naming): _node 'in swiss cheese room' reachable at conflicting positions [(1, -2, 0), (3, 0, 1), (3, 1, 1), (5, 2, 2)]_
  - step 43: `in tall e/w canyon --[north]--> in swiss cheese room` (correct)
  - step 59: `misty cavern --[southeast]--> in swiss cheese room` (hallucinated_edge)
  - step 44: `in swiss cheese room --[west]--> at east end of twopit room` (correct)

- **advent** (naming): _node 'at complex junction' reachable at conflicting positions [(3, -1, 0), (5, 1, 1), (5, 2, 1), (7, 3, 2)]_
  - step 64: `bedquilt --[east]--> at complex junction` (correct)
  - step 65: `at complex junction --[up]--> in dusty rock room` (dst_hallucinated)

### naming_collision_on_correct_subgraph

- **advent** (naming): _node 'plover room' reachable at conflicting positions [(1, 0, 0), (3, 2, 1), (3, 3, 1), (5, 4, 2)]_
  - step 53: `alcove --[east]--> plover room` (correct)

- **advent** (naming): _node 'in tall e/w canyon' reachable at conflicting positions [(1, -3, 0), (3, -1, 1), (3, 0, 1), (5, 0, 2), (5, 1, 2)]_
  - step 42: `n/s canyon --[north]--> in tall e/w canyon` (correct)
  - step 43: `in tall e/w canyon --[north]--> in swiss cheese room` (correct)

- **advent** (naming): _node 'at east end of twopit room' reachable at conflicting positions [(0, -2, 0), (2, 0, 1), (2, 1, 1), (4, 2, 2)]_
  - step 44: `in swiss cheese room --[west]--> at east end of twopit room` (correct)
  - step 45: `at east end of twopit room --[west]--> at west end of twopit room` (correct)

### naming_mixed

- **advent** (naming): _node 'misty cavern' reachable at conflicting positions [(0, -1, 0), (2, 1, 1), (2, 2, 1), (4, 3, 2)]_
  - step 57: `alcove --[south]--> misty cavern` (hallucinated_edge)
  - step 59: `misty cavern --[southeast]--> in swiss cheese room` (hallucinated_edge)

- **anchor** (naming): _node 'the lower edge of the window' reachable at conflicting positions [(-3, -3, 4), (-2, -2, 3), (-1, -1, 2), (0, 0, 1)]_
  - step 3: `alley --[up]--> the lower edge of the window` (hallucinated_edge)
  - step 4: `the lower edge of the window --[west]--> office` (hallucinated_edge)

- **awaken** (naming): _node 'steeple chamber' reachable at conflicting positions [(0, 0, -2), (0, 0, -1), (0, 0, 0), (0, 1, -2)]_
  - step 32: `vestibule --[up]--> steeple chamber` (swapped_src_dst)
  - step 49: `steeple chamber --[north]--> inner chamber` (correct)

### name_hallucination

- **advent** (naming): _node 'in dusty rock room' reachable at conflicting positions [(3, -1, 1), (5, 1, 2), (5, 2, 2), (7, 3, 3)]_
  - step 65: `at complex junction --[up]--> in dusty rock room` (dst_hallucinated)
  - step 66: `in dusty rock room --[east]--> dirty passage` (both_names_hallucinated)

- **advent** (naming): _node 'dirty passage' reachable at conflicting positions [(4, -1, 1), (4, 0, 1), (6, 1, 2), (6, 2, 2), (8, 3, 3)]_
  - step 66: `in dusty rock room --[east]--> dirty passage` (both_names_hallucinated)
  - step 67: `dirty passage --[up]--> low n/s passage` (src_hallucinated)

- **advent** (naming): _node 'darkness' reachable at conflicting positions [(1, -2, 1), (3, -1, 2), (3, 0, 2), (3, 1, 2)]_
  - step 12: `in cobble --[west]--> darkness` (dst_hallucinated)

### src_dst_swap

- **adventureland** (direction): _node 'maze (exit north, south, east, down)' has multiple outgoing edges labelled 'up'_
  - step None: `maze (exit north, south, east, down) --[up]--> hole` (hallucinated_edge)
  - step None: `maze (exit north, south, east, down) --[up]--> cavern` (swapped_src_dst)

- **detective** (direction): _node 'outside (holiday inn to north, doughnut king to east, the wall to west)' has multiple outgoing edges labelled 'south'_
  - step None: `outside (holiday inn to north, doughnut king to east, the wall to west) --[south]--> hallway (15th floor, room 19-22)` (hallucinated_edge)
  - step None: `outside (holiday inn to north, doughnut king to east, the wall to west) --[south]--> mcdonalds` (hallucinated_edge)
  - step None: `outside (holiday inn to north, doughnut king to east, the wall to west) --[south]--> police station` (swapped_src_dst)

- **night** (direction): _node 'maze of twisty passages (stop 3)' has multiple outgoing edges labelled 'west'_
  - step 34: `maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2)` (hallucinated_edge)
  - step 37: `maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4)` (swapped_src_dst)

### all_hallucinated_edges

- **detective** (direction): _node 'bedroom' has multiple outgoing edges labelled 'east'_
  - step None: `bedroom --[east]--> hallway (mayer's house, "guests" door to east)` (hallucinated_edge)
  - step 22: `bedroom --[east]--> hallway (15th floor, room 19-22)` (hallucinated_edge)

- **detective** (direction): _node 'mcdonalds' has multiple outgoing edges labelled 'north'_
  - step None: `mcdonalds --[north]--> outside (video store to east)` (hallucinated_edge)
  - step 29: `mcdonalds --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west)` (hallucinated_edge)

- **inhumane** (direction): _node 'round room' has multiple outgoing edges labelled 'east'_
  - step 29: `round room --[east]--> corridor at doorway` (hallucinated_edge)
  - step None: `round room --[east]--> north stalagmite room` (hallucinated_edge)

### direction_mixed

- **inhumane** (direction): _node 'round room' has multiple outgoing edges labelled 'west'_
  - step 21: `round room --[west]--> t-intersection (east/west/south, east to round room)` (wrong_direction (LLM dir='west', GT dir='east'))
  - step None: `round room --[west]--> south branch` (hallucinated_edge)
  - step 53: `round room --[west]--> corridor near pit` (hallucinated_edge)

- **night** (direction): _node 'maze of twisty passages (stop 1)' has multiple outgoing edges labelled 'west'_
  - step None: `maze of twisty passages (stop 1) --[west]--> hall (1st floor, middle of north/south hall)` (hallucinated_edge)
  - step 70: `maze of twisty passages (stop 1) --[west]--> maze of twisty passages (stop 2)` (wrong_direction (LLM dir='west', GT dir='east'))

- **night** (direction): _node 'maze of twisty passages (stop 2)' has multiple outgoing edges labelled 'east'_
  - step None: `maze of twisty passages (stop 2) --[east]--> maze of twisty passages (stop 3)` (hallucinated_edge)
  - step None: `maze of twisty passages (stop 2) --[east]--> maze of twisty passages (stop 4)` (hallucinated_edge)
  - step None: `maze of twisty passages (stop 2) --[east]--> maze of twisty passages (stop 1)` (wrong_direction (LLM dir='east', GT dir='west'))

### wrong_direction_caused_overlap

- **inhumane** (topology): _position (0, -3, 0) occupied by multiple rooms ['corridor near pit', 'glass hall', 'south branch', 't-intersection (east/west/south, east to round room)']_
  - step 53: `round room --[west]--> corridor near pit` (hallucinated_edge)
  - step 57: `corridor near pit --[west]--> on the platform` (correct)
  - step 34: `hall full of fur --[east]--> glass hall` (correct)

- **inhumane** (topology): _position (0, -7, 0) occupied by multiple rooms ['corridor near pit', 'glass hall', 'south branch', 't-intersection (east/west/south, east to round room)']_
  - step 53: `round room --[west]--> corridor near pit` (hallucinated_edge)
  - step 57: `corridor near pit --[west]--> on the platform` (correct)
  - step 34: `hall full of fur --[east]--> glass hall` (correct)

- **inhumane** (topology): _position (0, 1, 0) occupied by multiple rooms ['corridor near pit', 'south branch', 't-intersection (east/west/south, east to round room)']_
  - step 53: `round room --[west]--> corridor near pit` (hallucinated_edge)
  - step 57: `corridor near pit --[west]--> on the platform` (correct)
  - step 22: `t-intersection (east/west/south, east to round room) --[south]--> south branch` (hallucinated_edge)

