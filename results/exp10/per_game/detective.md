# Conflict analysis: detective

- LLM edges: 30
- GT edges: 39
- Conflicts: 72
- Type distribution: {'direction': 7, 'topology': 37, 'naming': 28}
- Root-cause distribution: {'name_hallucination': 2, 'all_hallucinated_edges': 2, 'real_vs_hallucinated': 3, 'src_dst_swap': 1, 'false_positive_overlap': 8, 'overlap_mixed': 22, 'name_hallucination_caused_overlap': 7, 'naming_collision_on_correct_subgraph': 15, 'real_name_corrupted_by_neighbour_error': 6, 'naming_mixed': 6}

## Conflict 1 — direction (name_hallucination)
- description: node 'outside (restaurant to north, mayer home to east)' has multiple outgoing edges labelled 'west'
  - step None: outside (restaurant to north, mayer home to east) --[west]--> outside (dead end to east) — src_hallucinated
  - step 11: outside (restaurant to north, mayer home to east) --[west]--> mayor's house (scene of crime) — src_hallucinated

## Conflict 2 — direction (all_hallucinated_edges)
- description: node 'bedroom' has multiple outgoing edges labelled 'east'
  - step None: bedroom --[east]--> hallway (mayer's house, "guests" door to east) — hallucinated_edge
  - step 22: bedroom --[east]--> hallway (15th floor, room 19-22) — hallucinated_edge

## Conflict 3 — direction (real_vs_hallucinated)
- description: node 'hallway (15th floor, room 19-22)' has multiple outgoing edges labelled 'north'
  - step 23: hallway (15th floor, room 19-22) --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 47: hallway (15th floor, room 19-22) --[north]--> hallway (sauna to west, pool a to east) — correct

## Conflict 4 — direction (src_dst_swap)
- description: node 'outside (holiday inn to north, doughnut king to east, the wall to west)' has multiple outgoing edges labelled 'south'
  - step None: outside (holiday inn to north, doughnut king to east, the wall to west) --[south]--> hallway (15th floor, room 19-22) — hallucinated_edge
  - step None: outside (holiday inn to north, doughnut king to east, the wall to west) --[south]--> mcdonalds — hallucinated_edge
  - step None: outside (holiday inn to north, doughnut king to east, the wall to west) --[south]--> police station — swapped_src_dst (LLM: outside (holiday inn to north, doughnut king to east, the wall to west)--[south]-->police station but GT has police station--[north]-->outside (holiday inn to north, doughnut king to east, the wall to west))

## Conflict 5 — direction (real_vs_hallucinated)
- description: node 'outside (holiday inn to north, doughnut king to east, the wall to west)' has multiple outgoing edges labelled 'east'
  - step 24: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> video store — hallucinated_edge
  - step 36: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> doughnut king — correct

## Conflict 6 — direction (real_vs_hallucinated)
- description: node 'outside (holiday inn to north, doughnut king to east, the wall to west)' has multiple outgoing edges labelled 'north'
  - step 30: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> outside (north end, east only available) — hallucinated_edge
  - step 40: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> holiday inn — correct

## Conflict 7 — direction (all_hallucinated_edges)
- description: node 'mcdonalds' has multiple outgoing edges labelled 'north'
  - step None: mcdonalds --[north]--> outside (video store to east) — hallucinated_edge
  - step 29: mcdonalds --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge

## Conflict 8 — topology (false_positive_overlap)
- description: position (-4, 7, 0) occupied by multiple rooms ['back of music store', 'dining room']
  - step 32: music store --[north]--> back of music store — correct
  - step 33: back of music store --[north]--> alley — correct
  - step 12: mayor's house (scene of crime) --[west]--> dining room — correct

## Conflict 9 — topology (false_positive_overlap)
- description: position (-6, 11, 0) occupied by multiple rooms ['back of music store', 'dining room']
  - step 32: music store --[north]--> back of music store — correct
  - step 33: back of music store --[north]--> alley — correct
  - step 12: mayor's house (scene of crime) --[west]--> dining room — correct

## Conflict 10 — topology (false_positive_overlap)
- description: position (0, -1, 0) occupied by multiple rooms ['back of music store', 'dining room']
  - step 32: music store --[north]--> back of music store — correct
  - step 33: back of music store --[north]--> alley — correct
  - step 12: mayor's house (scene of crime) --[west]--> dining room — correct

## Conflict 11 — topology (overlap_mixed)
- description: position (-2, 5, 0) occupied by multiple rooms ['hallway (15th floor, room 19-22)', 'hallway (mayer\'s house, "guests" door to east)', 'mcdonalds', 'police station']
  - step 22: bedroom --[east]--> hallway (15th floor, room 19-22) — hallucinated_edge
  - step 46: hallway (15th floor, east/west intersection) --[west]--> hallway (15th floor, room 19-22) — correct
  - step 23: hallway (15th floor, room 19-22) --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 47: hallway (15th floor, room 19-22) --[north]--> hallway (sauna to west, pool a to east) — correct
  - step 18: hallway (mayer's house, east/west intersect, exit to north) --[west]--> hallway (mayer's house, "guests" door to east) — hallucinated_edge
  - step 21: hallway (mayer's house, "guests" door to east) --[west]--> bedroom — hallucinated_edge
  - step 27: outside (video store to east) --[south]--> mcdonalds — hallucinated_edge
  - step 29: mcdonalds --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 34: alley --[north]--> police station — correct
  - step 35: police station --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — correct

## Conflict 12 — topology (overlap_mixed)
- description: position (0, 1, 0) occupied by multiple rooms ['hallway (15th floor, room 19-22)', 'hallway (mayer\'s house, "guests" door to east)', 'mcdonalds', 'police station']
  - step 22: bedroom --[east]--> hallway (15th floor, room 19-22) — hallucinated_edge
  - step 46: hallway (15th floor, east/west intersection) --[west]--> hallway (15th floor, room 19-22) — correct
  - step 23: hallway (15th floor, room 19-22) --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 47: hallway (15th floor, room 19-22) --[north]--> hallway (sauna to west, pool a to east) — correct
  - step 18: hallway (mayer's house, east/west intersect, exit to north) --[west]--> hallway (mayer's house, "guests" door to east) — hallucinated_edge
  - step 21: hallway (mayer's house, "guests" door to east) --[west]--> bedroom — hallucinated_edge
  - step 27: outside (video store to east) --[south]--> mcdonalds — hallucinated_edge
  - step 29: mcdonalds --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 34: alley --[north]--> police station — correct
  - step 35: police station --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — correct

## Conflict 13 — topology (overlap_mixed)
- description: position (-4, 9, 0) occupied by multiple rooms ['hallway (15th floor, room 19-22)', 'hallway (mayer\'s house, "guests" door to east)', 'mcdonalds', 'police station']
  - step 22: bedroom --[east]--> hallway (15th floor, room 19-22) — hallucinated_edge
  - step 46: hallway (15th floor, east/west intersection) --[west]--> hallway (15th floor, room 19-22) — correct
  - step 23: hallway (15th floor, room 19-22) --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 47: hallway (15th floor, room 19-22) --[north]--> hallway (sauna to west, pool a to east) — correct
  - step 18: hallway (mayer's house, east/west intersect, exit to north) --[west]--> hallway (mayer's house, "guests" door to east) — hallucinated_edge
  - step 21: hallway (mayer's house, "guests" door to east) --[west]--> bedroom — hallucinated_edge
  - step 27: outside (video store to east) --[south]--> mcdonalds — hallucinated_edge
  - step 29: mcdonalds --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 34: alley --[north]--> police station — correct
  - step 35: police station --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — correct

## Conflict 14 — topology (overlap_mixed)
- description: position (-6, 13, 0) occupied by multiple rooms ['hallway (15th floor, room 19-22)', 'hallway (mayer\'s house, "guests" door to east)', 'mcdonalds', 'police station']
  - step 22: bedroom --[east]--> hallway (15th floor, room 19-22) — hallucinated_edge
  - step 46: hallway (15th floor, east/west intersection) --[west]--> hallway (15th floor, room 19-22) — correct
  - step 23: hallway (15th floor, room 19-22) --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 47: hallway (15th floor, room 19-22) --[north]--> hallway (sauna to west, pool a to east) — correct
  - step 18: hallway (mayer's house, east/west intersect, exit to north) --[west]--> hallway (mayer's house, "guests" door to east) — hallucinated_edge
  - step 21: hallway (mayer's house, "guests" door to east) --[west]--> bedroom — hallucinated_edge
  - step 27: outside (video store to east) --[south]--> mcdonalds — hallucinated_edge
  - step 29: mcdonalds --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 34: alley --[north]--> police station — correct
  - step 35: police station --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — correct

## Conflict 15 — topology (overlap_mixed)
- description: position (0, 2, 0) occupied by multiple rooms ['hallway (sauna to west, pool a to east)', 'outside (holiday inn to north, doughnut king to east, the wall to west)', 'outside (video store to east)']
  - step 47: hallway (15th floor, room 19-22) --[north]--> hallway (sauna to west, pool a to east) — correct
  - step 48: hallway (sauna to west, pool a to east) --[north]--> room # 30 — correct
  - step 23: hallway (15th floor, room 19-22) --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 29: mcdonalds --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 35: police station --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — correct
  - step 24: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> video store — hallucinated_edge
  - step 30: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> outside (north end, east only available) — hallucinated_edge
  - step 36: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> doughnut king — correct
  - step 38: outside (holiday inn to north, doughnut king to east, the wall to west) --[west]--> the wall — correct
  - step 40: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> holiday inn — correct
  - step 26: video store --[east]--> outside (video store to east) — swapped_src_dst (LLM: video store--[east]-->outside (video store to east) but GT has outside (video store to east)--[east]-->video store)
  - step 27: outside (video store to east) --[south]--> mcdonalds — hallucinated_edge

## Conflict 16 — topology (overlap_mixed)
- description: position (-2, 6, 0) occupied by multiple rooms ['hallway (sauna to west, pool a to east)', 'outside (holiday inn to north, doughnut king to east, the wall to west)', 'outside (video store to east)']
  - step 47: hallway (15th floor, room 19-22) --[north]--> hallway (sauna to west, pool a to east) — correct
  - step 48: hallway (sauna to west, pool a to east) --[north]--> room # 30 — correct
  - step 23: hallway (15th floor, room 19-22) --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 29: mcdonalds --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 35: police station --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — correct
  - step 24: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> video store — hallucinated_edge
  - step 30: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> outside (north end, east only available) — hallucinated_edge
  - step 36: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> doughnut king — correct
  - step 38: outside (holiday inn to north, doughnut king to east, the wall to west) --[west]--> the wall — correct
  - step 40: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> holiday inn — correct
  - step 26: video store --[east]--> outside (video store to east) — swapped_src_dst (LLM: video store--[east]-->outside (video store to east) but GT has outside (video store to east)--[east]-->video store)
  - step 27: outside (video store to east) --[south]--> mcdonalds — hallucinated_edge

## Conflict 17 — topology (overlap_mixed)
- description: position (-4, 10, 0) occupied by multiple rooms ['hallway (sauna to west, pool a to east)', 'outside (holiday inn to north, doughnut king to east, the wall to west)', 'outside (video store to east)']
  - step 47: hallway (15th floor, room 19-22) --[north]--> hallway (sauna to west, pool a to east) — correct
  - step 48: hallway (sauna to west, pool a to east) --[north]--> room # 30 — correct
  - step 23: hallway (15th floor, room 19-22) --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 29: mcdonalds --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 35: police station --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — correct
  - step 24: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> video store — hallucinated_edge
  - step 30: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> outside (north end, east only available) — hallucinated_edge
  - step 36: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> doughnut king — correct
  - step 38: outside (holiday inn to north, doughnut king to east, the wall to west) --[west]--> the wall — correct
  - step 40: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> holiday inn — correct
  - step 26: video store --[east]--> outside (video store to east) — swapped_src_dst (LLM: video store--[east]-->outside (video store to east) but GT has outside (video store to east)--[east]-->video store)
  - step 27: outside (video store to east) --[south]--> mcdonalds — hallucinated_edge

## Conflict 18 — topology (overlap_mixed)
- description: position (-6, 14, 0) occupied by multiple rooms ['hallway (sauna to west, pool a to east)', 'outside (holiday inn to north, doughnut king to east, the wall to west)', 'outside (video store to east)']
  - step 47: hallway (15th floor, room 19-22) --[north]--> hallway (sauna to west, pool a to east) — correct
  - step 48: hallway (sauna to west, pool a to east) --[north]--> room # 30 — correct
  - step 23: hallway (15th floor, room 19-22) --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 29: mcdonalds --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 35: police station --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — correct
  - step 24: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> video store — hallucinated_edge
  - step 30: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> outside (north end, east only available) — hallucinated_edge
  - step 36: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> doughnut king — correct
  - step 38: outside (holiday inn to north, doughnut king to east, the wall to west) --[west]--> the wall — correct
  - step 40: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> holiday inn — correct
  - step 26: video store --[east]--> outside (video store to east) — swapped_src_dst (LLM: video store--[east]-->outside (video store to east) but GT has outside (video store to east)--[east]-->video store)
  - step 27: outside (video store to east) --[south]--> mcdonalds — hallucinated_edge

## Conflict 19 — topology (overlap_mixed)
- description: position (-7, 14, 0) occupied by multiple rooms ['the wall', 'video store']
  - step 38: outside (holiday inn to north, doughnut king to east, the wall to west) --[west]--> the wall — correct
  - step 24: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> video store — hallucinated_edge
  - step 26: video store --[east]--> outside (video store to east) — swapped_src_dst (LLM: video store--[east]-->outside (video store to east) but GT has outside (video store to east)--[east]-->video store)

## Conflict 20 — topology (overlap_mixed)
- description: position (-5, 14, 0) occupied by multiple rooms ['doughnut king', 'video store']
  - step 36: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> doughnut king — correct
  - step 24: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> video store — hallucinated_edge
  - step 26: video store --[east]--> outside (video store to east) — swapped_src_dst (LLM: video store--[east]-->outside (video store to east) but GT has outside (video store to east)--[east]-->video store)

## Conflict 21 — topology (overlap_mixed)
- description: position (1, 2, 0) occupied by multiple rooms ['doughnut king', 'video store']
  - step 36: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> doughnut king — correct
  - step 24: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> video store — hallucinated_edge
  - step 26: video store --[east]--> outside (video store to east) — swapped_src_dst (LLM: video store--[east]-->outside (video store to east) but GT has outside (video store to east)--[east]-->video store)

## Conflict 22 — topology (overlap_mixed)
- description: position (-5, 10, 0) occupied by multiple rooms ["chief's office", 'the wall', 'video store']
  - step 5: chief's office --[west]--> closet — correct
  - step 8: chief's office --[north]--> outside (dead end to east) — correct
  - step 38: outside (holiday inn to north, doughnut king to east, the wall to west) --[west]--> the wall — correct
  - step 24: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> video store — hallucinated_edge
  - step 26: video store --[east]--> outside (video store to east) — swapped_src_dst (LLM: video store--[east]-->outside (video store to east) but GT has outside (video store to east)--[east]-->video store)

## Conflict 23 — topology (overlap_mixed)
- description: position (-1, 6, 0) occupied by multiple rooms ['doughnut king', 'video store']
  - step 36: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> doughnut king — correct
  - step 24: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> video store — hallucinated_edge
  - step 26: video store --[east]--> outside (video store to east) — swapped_src_dst (LLM: video store--[east]-->outside (video store to east) but GT has outside (video store to east)--[east]-->video store)

## Conflict 24 — topology (overlap_mixed)
- description: position (-3, 10, 0) occupied by multiple rooms ['doughnut king', 'video store']
  - step 36: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> doughnut king — correct
  - step 24: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> video store — hallucinated_edge
  - step 26: video store --[east]--> outside (video store to east) — swapped_src_dst (LLM: video store--[east]-->outside (video store to east) but GT has outside (video store to east)--[east]-->video store)

## Conflict 25 — topology (overlap_mixed)
- description: position (-6, 15, 0) occupied by multiple rooms ['holiday inn', 'outside (north end, east only available)', 'room # 30']
  - step 40: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> holiday inn — correct
  - step 42: holiday inn --[north]--> holiday inn 15th floor — correct
  - step 30: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> outside (north end, east only available) — hallucinated_edge
  - step 31: outside (north end, east only available) --[east]--> music store — correct
  - step 48: hallway (sauna to west, pool a to east) --[north]--> room # 30 — correct

## Conflict 26 — topology (name_hallucination_caused_overlap)
- description: position (-2, 7, 0) occupied by multiple rooms ['holiday inn', 'outside (north end, east only available)', 'outside (restaurant to north, mayer home to east)', 'room # 30']
  - step 40: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> holiday inn — correct
  - step 42: holiday inn --[north]--> holiday inn 15th floor — correct
  - step 30: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> outside (north end, east only available) — hallucinated_edge
  - step 31: outside (north end, east only available) --[east]--> music store — correct
  - step 9: outside (dead end to east) --[east]--> outside (restaurant to north, mayer home to east) — dst_hallucinated
  - step 11: outside (restaurant to north, mayer home to east) --[west]--> mayor's house (scene of crime) — src_hallucinated
  - step 48: hallway (sauna to west, pool a to east) --[north]--> room # 30 — correct

## Conflict 27 — topology (name_hallucination_caused_overlap)
- description: position (0, 3, 0) occupied by multiple rooms ['holiday inn', 'outside (north end, east only available)', 'outside (restaurant to north, mayer home to east)', 'room # 30']
  - step 40: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> holiday inn — correct
  - step 42: holiday inn --[north]--> holiday inn 15th floor — correct
  - step 30: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> outside (north end, east only available) — hallucinated_edge
  - step 31: outside (north end, east only available) --[east]--> music store — correct
  - step 9: outside (dead end to east) --[east]--> outside (restaurant to north, mayer home to east) — dst_hallucinated
  - step 11: outside (restaurant to north, mayer home to east) --[west]--> mayor's house (scene of crime) — src_hallucinated
  - step 48: hallway (sauna to west, pool a to east) --[north]--> room # 30 — correct

## Conflict 28 — topology (name_hallucination_caused_overlap)
- description: position (-4, 11, 0) occupied by multiple rooms ['holiday inn', 'outside (north end, east only available)', 'outside (restaurant to north, mayer home to east)', 'room # 30']
  - step 40: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> holiday inn — correct
  - step 42: holiday inn --[north]--> holiday inn 15th floor — correct
  - step 30: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> outside (north end, east only available) — hallucinated_edge
  - step 31: outside (north end, east only available) --[east]--> music store — correct
  - step 9: outside (dead end to east) --[east]--> outside (restaurant to north, mayer home to east) — dst_hallucinated
  - step 11: outside (restaurant to north, mayer home to east) --[west]--> mayor's house (scene of crime) — src_hallucinated
  - step 48: hallway (sauna to west, pool a to east) --[north]--> room # 30 — correct

## Conflict 29 — topology (false_positive_overlap)
- description: position (-3, 6, 0) occupied by multiple rooms ["chief's office", 'the wall']
  - step 5: chief's office --[west]--> closet — correct
  - step 8: chief's office --[north]--> outside (dead end to east) — correct
  - step 38: outside (holiday inn to north, doughnut king to east, the wall to west) --[west]--> the wall — correct

## Conflict 30 — topology (false_positive_overlap)
- description: position (-1, 2, 0) occupied by multiple rooms ["chief's office", 'the wall']
  - step 5: chief's office --[west]--> closet — correct
  - step 8: chief's office --[north]--> outside (dead end to east) — correct
  - step 38: outside (holiday inn to north, doughnut king to east, the wall to west) --[west]--> the wall — correct

## Conflict 31 — topology (overlap_mixed)
- description: position (-3, 8, 0) occupied by multiple rooms ['hallway (15th floor, room 1-7)', "upstairs hallway (mayer's house)"]
  - step 43: holiday inn 15th floor --[west]--> hallway (15th floor, room 1-7) — correct
  - step 44: hallway (15th floor, room 1-7) --[north]--> hallway (15th floor, east/west intersection) — correct
  - step 16: mayor's house (scene of crime) --[north]--> upstairs hallway (mayer's house) — correct
  - step 17: upstairs hallway (mayer's house) --[north]--> hallway (mayer's house, east/west intersect, exit to north) — hallucinated_edge

## Conflict 32 — topology (overlap_mixed)
- description: position (-5, 12, 0) occupied by multiple rooms ['hallway (15th floor, room 1-7)', "upstairs hallway (mayer's house)"]
  - step 43: holiday inn 15th floor --[west]--> hallway (15th floor, room 1-7) — correct
  - step 44: hallway (15th floor, room 1-7) --[north]--> hallway (15th floor, east/west intersection) — correct
  - step 16: mayor's house (scene of crime) --[north]--> upstairs hallway (mayer's house) — correct
  - step 17: upstairs hallway (mayer's house) --[north]--> hallway (mayer's house, east/west intersect, exit to north) — hallucinated_edge

## Conflict 33 — topology (overlap_mixed)
- description: position (-1, 4, 0) occupied by multiple rooms ['hallway (15th floor, room 1-7)', "upstairs hallway (mayer's house)"]
  - step 43: holiday inn 15th floor --[west]--> hallway (15th floor, room 1-7) — correct
  - step 44: hallway (15th floor, room 1-7) --[north]--> hallway (15th floor, east/west intersection) — correct
  - step 16: mayor's house (scene of crime) --[north]--> upstairs hallway (mayer's house) — correct
  - step 17: upstairs hallway (mayer's house) --[north]--> hallway (mayer's house, east/west intersect, exit to north) — hallucinated_edge

## Conflict 34 — topology (overlap_mixed)
- description: position (-5, 13, 0) occupied by multiple rooms ['hallway (15th floor, east/west intersection)', "hallway (mayer's house, east/west intersect, exit to north)"]
  - step 44: hallway (15th floor, room 1-7) --[north]--> hallway (15th floor, east/west intersection) — correct
  - step 46: hallway (15th floor, east/west intersection) --[west]--> hallway (15th floor, room 19-22) — correct
  - step 17: upstairs hallway (mayer's house) --[north]--> hallway (mayer's house, east/west intersect, exit to north) — hallucinated_edge
  - step 18: hallway (mayer's house, east/west intersect, exit to north) --[west]--> hallway (mayer's house, "guests" door to east) — hallucinated_edge

## Conflict 35 — topology (overlap_mixed)
- description: position (-1, 5, 0) occupied by multiple rooms ['hallway (15th floor, east/west intersection)', "hallway (mayer's house, east/west intersect, exit to north)"]
  - step 44: hallway (15th floor, room 1-7) --[north]--> hallway (15th floor, east/west intersection) — correct
  - step 46: hallway (15th floor, east/west intersection) --[west]--> hallway (15th floor, room 19-22) — correct
  - step 17: upstairs hallway (mayer's house) --[north]--> hallway (mayer's house, east/west intersect, exit to north) — hallucinated_edge
  - step 18: hallway (mayer's house, east/west intersect, exit to north) --[west]--> hallway (mayer's house, "guests" door to east) — hallucinated_edge

## Conflict 36 — topology (overlap_mixed)
- description: position (1, 1, 0) occupied by multiple rooms ['hallway (15th floor, east/west intersection)', "hallway (mayer's house, east/west intersect, exit to north)"]
  - step 44: hallway (15th floor, room 1-7) --[north]--> hallway (15th floor, east/west intersection) — correct
  - step 46: hallway (15th floor, east/west intersection) --[west]--> hallway (15th floor, room 19-22) — correct
  - step 17: upstairs hallway (mayer's house) --[north]--> hallway (mayer's house, east/west intersect, exit to north) — hallucinated_edge
  - step 18: hallway (mayer's house, east/west intersect, exit to north) --[west]--> hallway (mayer's house, "guests" door to east) — hallucinated_edge

## Conflict 37 — topology (overlap_mixed)
- description: position (-3, 9, 0) occupied by multiple rooms ['hallway (15th floor, east/west intersection)', "hallway (mayer's house, east/west intersect, exit to north)"]
  - step 44: hallway (15th floor, room 1-7) --[north]--> hallway (15th floor, east/west intersection) — correct
  - step 46: hallway (15th floor, east/west intersection) --[west]--> hallway (15th floor, room 19-22) — correct
  - step 17: upstairs hallway (mayer's house) --[north]--> hallway (mayer's house, east/west intersect, exit to north) — hallucinated_edge
  - step 18: hallway (mayer's house, east/west intersect, exit to north) --[west]--> hallway (mayer's house, "guests" door to east) — hallucinated_edge

## Conflict 38 — topology (false_positive_overlap)
- description: position (-6, 10, 0) occupied by multiple rooms ['closet', 'music store']
  - step 5: chief's office --[west]--> closet — correct
  - step 31: outside (north end, east only available) --[east]--> music store — correct
  - step 32: music store --[north]--> back of music store — correct

## Conflict 39 — topology (false_positive_overlap)
- description: position (0, -2, 0) occupied by multiple rooms ['closet', 'music store']
  - step 5: chief's office --[west]--> closet — correct
  - step 31: outside (north end, east only available) --[east]--> music store — correct
  - step 32: music store --[north]--> back of music store — correct

## Conflict 40 — topology (false_positive_overlap)
- description: position (-4, 6, 0) occupied by multiple rooms ['closet', 'music store']
  - step 5: chief's office --[west]--> closet — correct
  - step 31: outside (north end, east only available) --[east]--> music store — correct
  - step 32: music store --[north]--> back of music store — correct

## Conflict 41 — topology (name_hallucination_caused_overlap)
- description: position (1, -1, 0) occupied by multiple rooms ["mayor's house (scene of crime)", 'outside (dead end to east)']
  - step 11: outside (restaurant to north, mayer home to east) --[west]--> mayor's house (scene of crime) — src_hallucinated
  - step 12: mayor's house (scene of crime) --[west]--> dining room — correct
  - step 16: mayor's house (scene of crime) --[north]--> upstairs hallway (mayer's house) — correct
  - step 8: chief's office --[north]--> outside (dead end to east) — correct
  - step 9: outside (dead end to east) --[east]--> outside (restaurant to north, mayer home to east) — dst_hallucinated

## Conflict 42 — topology (name_hallucination_caused_overlap)
- description: position (-1, 3, 0) occupied by multiple rooms ["mayor's house (scene of crime)", 'outside (dead end to east)']
  - step 11: outside (restaurant to north, mayer home to east) --[west]--> mayor's house (scene of crime) — src_hallucinated
  - step 12: mayor's house (scene of crime) --[west]--> dining room — correct
  - step 16: mayor's house (scene of crime) --[north]--> upstairs hallway (mayer's house) — correct
  - step 8: chief's office --[north]--> outside (dead end to east) — correct
  - step 9: outside (dead end to east) --[east]--> outside (restaurant to north, mayer home to east) — dst_hallucinated

## Conflict 43 — topology (name_hallucination_caused_overlap)
- description: position (-5, 11, 0) occupied by multiple rooms ["mayor's house (scene of crime)", 'outside (dead end to east)']
  - step 11: outside (restaurant to north, mayer home to east) --[west]--> mayor's house (scene of crime) — src_hallucinated
  - step 12: mayor's house (scene of crime) --[west]--> dining room — correct
  - step 16: mayor's house (scene of crime) --[north]--> upstairs hallway (mayer's house) — correct
  - step 8: chief's office --[north]--> outside (dead end to east) — correct
  - step 9: outside (dead end to east) --[east]--> outside (restaurant to north, mayer home to east) — dst_hallucinated

## Conflict 44 — topology (name_hallucination_caused_overlap)
- description: position (-3, 7, 0) occupied by multiple rooms ["mayor's house (scene of crime)", 'outside (dead end to east)']
  - step 11: outside (restaurant to north, mayer home to east) --[west]--> mayor's house (scene of crime) — src_hallucinated
  - step 12: mayor's house (scene of crime) --[west]--> dining room — correct
  - step 16: mayor's house (scene of crime) --[north]--> upstairs hallway (mayer's house) — correct
  - step 8: chief's office --[north]--> outside (dead end to east) — correct
  - step 9: outside (dead end to east) --[east]--> outside (restaurant to north, mayer home to east) — dst_hallucinated

## Conflict 45 — naming (naming_collision_on_correct_subgraph)
- description: node 'alley' reachable at conflicting positions [(-6, 12, 0), (-5, 17, 0), (-4, 8, 0), (-2, 4, 0), (0, 0, 0)]
  - step 33: back of music store --[north]--> alley — correct
  - step 34: alley --[north]--> police station — correct

## Conflict 46 — naming (naming_collision_on_correct_subgraph)
- description: node 'back of music store' reachable at conflicting positions [(-6, 11, 0), (-5, 16, 0), (-4, 7, 0), (-3, 12, 0), (0, -1, 0)]
  - step 32: music store --[north]--> back of music store — correct
  - step 33: back of music store --[north]--> alley — correct

## Conflict 47 — naming (naming_collision_on_correct_subgraph)
- description: node 'police station' reachable at conflicting positions [(-6, 13, 0), (-5, 18, 0), (-4, 9, 0), (-2, 5, 0), (0, 1, 0)]
  - step 34: alley --[north]--> police station — correct
  - step 35: police station --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — correct

## Conflict 48 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'outside (holiday inn to north, doughnut king to east, the wall to west)' reachable at conflicting positions [(-6, 14, 0), (-4, 10, 0), (-2, 6, 0), (0, 2, 0)]
  - step 23: hallway (15th floor, room 19-22) --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 29: mcdonalds --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 35: police station --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — correct
  - step 24: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> video store — hallucinated_edge
  - step 30: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> outside (north end, east only available) — hallucinated_edge
  - step 36: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> doughnut king — correct
  - step 38: outside (holiday inn to north, doughnut king to east, the wall to west) --[west]--> the wall — correct
  - step 40: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> holiday inn — correct

## Conflict 49 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'hallway (15th floor, room 19-22)' reachable at conflicting positions [(-8, 17, 0), (-6, 13, 0), (-4, 9, 0), (-2, 5, 0), (0, 1, 0)]
  - step 22: bedroom --[east]--> hallway (15th floor, room 19-22) — hallucinated_edge
  - step 46: hallway (15th floor, east/west intersection) --[west]--> hallway (15th floor, room 19-22) — correct
  - step 23: hallway (15th floor, room 19-22) --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge
  - step 47: hallway (15th floor, room 19-22) --[north]--> hallway (sauna to west, pool a to east) — correct

## Conflict 50 — naming (naming_mixed)
- description: node 'video store' reachable at conflicting positions [(-7, 14, 0), (-5, 10, 0), (-5, 14, 0), (-3, 10, 0), (-1, 6, 0), (1, 2, 0)]
  - step 24: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> video store — hallucinated_edge
  - step 26: video store --[east]--> outside (video store to east) — swapped_src_dst (LLM: video store--[east]-->outside (video store to east) but GT has outside (video store to east)--[east]-->video store)

## Conflict 51 — naming (naming_mixed)
- description: node 'mcdonalds' reachable at conflicting positions [(-6, 13, 0), (-4, 9, 0), (-4, 13, 0), (-2, 5, 0), (-2, 9, 0), (0, 1, 0)]
  - step 27: outside (video store to east) --[south]--> mcdonalds — hallucinated_edge
  - step 29: mcdonalds --[north]--> outside (holiday inn to north, doughnut king to east, the wall to west) — hallucinated_edge

## Conflict 52 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'outside (north end, east only available)' reachable at conflicting positions [(-7, 10, 0), (-6, 15, 0), (-5, 6, 0), (-4, 11, 0), (-2, 7, 0), (0, 3, 0)]
  - step 30: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> outside (north end, east only available) — hallucinated_edge
  - step 31: outside (north end, east only available) --[east]--> music store — correct

## Conflict 53 — naming (naming_collision_on_correct_subgraph)
- description: node 'doughnut king' reachable at conflicting positions [(-5, 14, 0), (-3, 10, 0), (-1, 6, 0), (1, 2, 0)]
  - step 36: outside (holiday inn to north, doughnut king to east, the wall to west) --[east]--> doughnut king — correct

## Conflict 54 — naming (naming_collision_on_correct_subgraph)
- description: node 'the wall' reachable at conflicting positions [(-7, 14, 0), (-5, 10, 0), (-3, 6, 0), (-1, 2, 0)]
  - step 38: outside (holiday inn to north, doughnut king to east, the wall to west) --[west]--> the wall — correct

## Conflict 55 — naming (naming_collision_on_correct_subgraph)
- description: node 'holiday inn' reachable at conflicting positions [(-6, 15, 0), (-4, 11, 0), (-2, 7, 0), (0, 3, 0)]
  - step 40: outside (holiday inn to north, doughnut king to east, the wall to west) --[north]--> holiday inn — correct
  - step 42: holiday inn --[north]--> holiday inn 15th floor — correct

## Conflict 56 — naming (naming_collision_on_correct_subgraph)
- description: node 'holiday inn 15th floor' reachable at conflicting positions [(-6, 16, 0), (-4, 12, 0), (-2, 8, 0), (0, 4, 0)]
  - step 42: holiday inn --[north]--> holiday inn 15th floor — correct
  - step 43: holiday inn 15th floor --[west]--> hallway (15th floor, room 1-7) — correct

## Conflict 57 — naming (naming_collision_on_correct_subgraph)
- description: node 'hallway (15th floor, room 1-7)' reachable at conflicting positions [(-7, 16, 0), (-5, 12, 0), (-3, 8, 0), (-1, 4, 0)]
  - step 43: holiday inn 15th floor --[west]--> hallway (15th floor, room 1-7) — correct
  - step 44: hallway (15th floor, room 1-7) --[north]--> hallway (15th floor, east/west intersection) — correct

## Conflict 58 — naming (naming_collision_on_correct_subgraph)
- description: node 'hallway (15th floor, east/west intersection)' reachable at conflicting positions [(-7, 17, 0), (-5, 13, 0), (-3, 9, 0), (-1, 5, 0), (1, 1, 0)]
  - step 44: hallway (15th floor, room 1-7) --[north]--> hallway (15th floor, east/west intersection) — correct
  - step 46: hallway (15th floor, east/west intersection) --[west]--> hallway (15th floor, room 19-22) — correct

## Conflict 59 — naming (naming_mixed)
- description: node 'bedroom' reachable at conflicting positions [(-7, 13, 0), (-5, 9, 0), (-3, 5, 0), (-1, 1, 0)]
  - step 21: hallway (mayer's house, "guests" door to east) --[west]--> bedroom — hallucinated_edge
  - step 22: bedroom --[east]--> hallway (15th floor, room 19-22) — hallucinated_edge

## Conflict 60 — naming (naming_collision_on_correct_subgraph)
- description: node 'hallway (sauna to west, pool a to east)' reachable at conflicting positions [(-6, 14, 0), (-4, 10, 0), (-2, 6, 0), (0, 2, 0)]
  - step 47: hallway (15th floor, room 19-22) --[north]--> hallway (sauna to west, pool a to east) — correct
  - step 48: hallway (sauna to west, pool a to east) --[north]--> room # 30 — correct

## Conflict 61 — naming (naming_collision_on_correct_subgraph)
- description: node 'room # 30' reachable at conflicting positions [(-6, 15, 0), (-4, 11, 0), (-2, 7, 0), (0, 3, 0)]
  - step 48: hallway (sauna to west, pool a to east) --[north]--> room # 30 — correct

## Conflict 62 — naming (naming_collision_on_correct_subgraph)
- description: node 'music store' reachable at conflicting positions [(-6, 10, 0), (-5, 15, 0), (-4, 6, 0), (-3, 11, 0), (-1, 7, 0), (0, -2, 0), (1, 3, 0)]
  - step 31: outside (north end, east only available) --[east]--> music store — correct
  - step 32: music store --[north]--> back of music store — correct

## Conflict 63 — naming (naming_mixed)
- description: node 'outside (video store to east)' reachable at conflicting positions [(-6, 14, 0), (-4, 10, 0), (-4, 14, 0), (-2, 6, 0), (-2, 10, 0), (0, 2, 0), (0, 6, 0), (2, 2, 0)]
  - step 26: video store --[east]--> outside (video store to east) — swapped_src_dst (LLM: video store--[east]-->outside (video store to east) but GT has outside (video store to east)--[east]-->video store)
  - step 27: outside (video store to east) --[south]--> mcdonalds — hallucinated_edge

## Conflict 64 — naming (naming_mixed)
- description: node 'hallway (mayer\'s house, "guests" door to east)' reachable at conflicting positions [(-6, 13, 0), (-4, 9, 0), (-2, 5, 0), (0, 1, 0)]
  - step 18: hallway (mayer's house, east/west intersect, exit to north) --[west]--> hallway (mayer's house, "guests" door to east) — hallucinated_edge
  - step 21: hallway (mayer's house, "guests" door to east) --[west]--> bedroom — hallucinated_edge

## Conflict 65 — naming (naming_mixed)
- description: node "hallway (mayer's house, east/west intersect, exit to north)" reachable at conflicting positions [(-5, 13, 0), (-3, 9, 0), (-1, 5, 0), (1, 1, 0)]
  - step 17: upstairs hallway (mayer's house) --[north]--> hallway (mayer's house, east/west intersect, exit to north) — hallucinated_edge
  - step 18: hallway (mayer's house, east/west intersect, exit to north) --[west]--> hallway (mayer's house, "guests" door to east) — hallucinated_edge

## Conflict 66 — naming (real_name_corrupted_by_neighbour_error)
- description: node "upstairs hallway (mayer's house)" reachable at conflicting positions [(-5, 12, 0), (-3, 8, 0), (-1, 4, 0), (1, 0, 0)]
  - step 16: mayor's house (scene of crime) --[north]--> upstairs hallway (mayer's house) — correct
  - step 17: upstairs hallway (mayer's house) --[north]--> hallway (mayer's house, east/west intersect, exit to north) — hallucinated_edge

## Conflict 67 — naming (real_name_corrupted_by_neighbour_error)
- description: node "mayor's house (scene of crime)" reachable at conflicting positions [(-5, 11, 0), (-3, 7, 0), (-1, 3, 0), (1, -1, 0)]
  - step 11: outside (restaurant to north, mayer home to east) --[west]--> mayor's house (scene of crime) — src_hallucinated
  - step 12: mayor's house (scene of crime) --[west]--> dining room — correct
  - step 16: mayor's house (scene of crime) --[north]--> upstairs hallway (mayer's house) — correct

## Conflict 68 — naming (name_hallucination)
- description: node 'outside (restaurant to north, mayer home to east)' reachable at conflicting positions [(-4, 11, 0), (-2, 7, 0), (0, 3, 0), (2, -1, 0)]
  - step 9: outside (dead end to east) --[east]--> outside (restaurant to north, mayer home to east) — dst_hallucinated
  - step 11: outside (restaurant to north, mayer home to east) --[west]--> mayor's house (scene of crime) — src_hallucinated

## Conflict 69 — naming (naming_collision_on_correct_subgraph)
- description: node 'dining room' reachable at conflicting positions [(-6, 11, 0), (-4, 7, 0), (-2, 3, 0), (0, -1, 0)]
  - step 12: mayor's house (scene of crime) --[west]--> dining room — correct

## Conflict 70 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'outside (dead end to east)' reachable at conflicting positions [(-5, 11, 0), (-3, 7, 0), (-1, 3, 0), (1, -1, 0)]
  - step 8: chief's office --[north]--> outside (dead end to east) — correct
  - step 9: outside (dead end to east) --[east]--> outside (restaurant to north, mayer home to east) — dst_hallucinated

## Conflict 71 — naming (naming_collision_on_correct_subgraph)
- description: node "chief's office" reachable at conflicting positions [(-5, 10, 0), (-3, 6, 0), (-1, 2, 0), (1, -2, 0)]
  - step 5: chief's office --[west]--> closet — correct
  - step 8: chief's office --[north]--> outside (dead end to east) — correct

## Conflict 72 — naming (naming_collision_on_correct_subgraph)
- description: node 'closet' reachable at conflicting positions [(-6, 10, 0), (-4, 6, 0), (-2, 2, 0), (0, -2, 0)]
  - step 5: chief's office --[west]--> closet — correct
