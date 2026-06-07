# Conflict analysis: night

- LLM edges: 21
- GT edges: 32
- Conflicts: 45
- Type distribution: {'direction': 5, 'topology': 22, 'naming': 18}
- Root-cause distribution: {'real_vs_hallucinated': 2, 'direction_mixed': 2, 'src_dst_swap': 1, 'false_positive_overlap': 6, 'overlap_mixed': 8, 'wrong_direction_caused_overlap': 8, 'naming_collision_on_correct_subgraph': 11, 'real_name_corrupted_by_neighbour_error': 3, 'naming_mixed': 4}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'hall (1st floor, middle of north/south hall)' has multiple outgoing edges labelled 'east'
  - step 14: hall (1st floor, middle of north/south hall) --[east]--> hall outside elevator (1st floor) — correct
  - step 29: hall (1st floor, middle of north/south hall) --[east]--> maze of twisty passages (stop 1) — hallucinated_edge

## Conflict 2 — direction (real_vs_hallucinated)
- description: node 'outside physics office' has multiple outgoing edges labelled 'south'
  - step 24: outside physics office --[south]--> hall (2nd floor, middle of north/south hall) — correct
  - step None: outside physics office --[south]--> maze of twisty passages (stop 2) — hallucinated_edge

## Conflict 3 — direction (direction_mixed)
- description: node 'maze of twisty passages (stop 1)' has multiple outgoing edges labelled 'west'
  - step None: maze of twisty passages (stop 1) --[west]--> hall (1st floor, middle of north/south hall) — hallucinated_edge
  - step 70: maze of twisty passages (stop 1) --[west]--> maze of twisty passages (stop 2) — wrong_direction (LLM: 'west', GT: 'east')

## Conflict 4 — direction (src_dst_swap)
- description: node 'maze of twisty passages (stop 3)' has multiple outgoing edges labelled 'west'
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))

## Conflict 5 — direction (direction_mixed)
- description: node 'maze of twisty passages (stop 2)' has multiple outgoing edges labelled 'east'
  - step None: maze of twisty passages (stop 2) --[east]--> maze of twisty passages (stop 3) — hallucinated_edge
  - step None: maze of twisty passages (stop 2) --[east]--> maze of twisty passages (stop 4) — hallucinated_edge
  - step None: maze of twisty passages (stop 2) --[east]--> maze of twisty passages (stop 1) — wrong_direction (LLM: 'east', GT: 'west')

## Conflict 6 — topology (false_positive_overlap)
- description: position (2, 1, 1) occupied by multiple rooms ['hall outside computer site', 'stairwell (first floor)', 'stairwell (second floor)']
  - step 1: computer site --[northeast]--> hall outside computer site — correct
  - step 2: hall outside computer site --[south]--> hall (3rd floor, middle of north/south hall) — correct
  - step 9: hall outside computer site --[west]--> stairwell (third floor) — correct
  - step 11: stairwell (second floor) --[down]--> stairwell (first floor) — correct
  - step 12: stairwell (first floor) --[east]--> hall (1st floor, north end of north/south hall) — correct
  - step 10: stairwell (third floor) --[down]--> stairwell (second floor) — correct
  - step 23: stairwell (second floor) --[east]--> outside physics office — correct

## Conflict 7 — topology (false_positive_overlap)
- description: position (3, 1, 2) occupied by multiple rooms ['hall outside computer site', 'stairwell (first floor)', 'stairwell (second floor)']
  - step 1: computer site --[northeast]--> hall outside computer site — correct
  - step 2: hall outside computer site --[south]--> hall (3rd floor, middle of north/south hall) — correct
  - step 9: hall outside computer site --[west]--> stairwell (third floor) — correct
  - step 11: stairwell (second floor) --[down]--> stairwell (first floor) — correct
  - step 12: stairwell (first floor) --[east]--> hall (1st floor, north end of north/south hall) — correct
  - step 10: stairwell (third floor) --[down]--> stairwell (second floor) — correct
  - step 23: stairwell (second floor) --[east]--> outside physics office — correct

## Conflict 8 — topology (false_positive_overlap)
- description: position (1, 1, 0) occupied by multiple rooms ['hall outside computer site', 'stairwell (first floor)', 'stairwell (second floor)']
  - step 1: computer site --[northeast]--> hall outside computer site — correct
  - step 2: hall outside computer site --[south]--> hall (3rd floor, middle of north/south hall) — correct
  - step 9: hall outside computer site --[west]--> stairwell (third floor) — correct
  - step 11: stairwell (second floor) --[down]--> stairwell (first floor) — correct
  - step 12: stairwell (first floor) --[east]--> hall (1st floor, north end of north/south hall) — correct
  - step 10: stairwell (third floor) --[down]--> stairwell (second floor) — correct
  - step 23: stairwell (second floor) --[east]--> outside physics office — correct

## Conflict 9 — topology (false_positive_overlap)
- description: position (3, 1, 3) occupied by multiple rooms ['stairwell (second floor)', 'stairwell (third floor)']
  - step 10: stairwell (third floor) --[down]--> stairwell (second floor) — correct
  - step 11: stairwell (second floor) --[down]--> stairwell (first floor) — correct
  - step 23: stairwell (second floor) --[east]--> outside physics office — correct
  - step 9: hall outside computer site --[west]--> stairwell (third floor) — correct

## Conflict 10 — topology (false_positive_overlap)
- description: position (2, 1, 2) occupied by multiple rooms ['stairwell (second floor)', 'stairwell (third floor)']
  - step 10: stairwell (third floor) --[down]--> stairwell (second floor) — correct
  - step 11: stairwell (second floor) --[down]--> stairwell (first floor) — correct
  - step 23: stairwell (second floor) --[east]--> outside physics office — correct
  - step 9: hall outside computer site --[west]--> stairwell (third floor) — correct

## Conflict 11 — topology (overlap_mixed)
- description: position (2, 1, 0) occupied by multiple rooms ['hall (1st floor, north end of north/south hall)', 'outside physics office', 'stairwell (first floor)']
  - step 12: stairwell (first floor) --[east]--> hall (1st floor, north end of north/south hall) — correct
  - step 13: hall (1st floor, north end of north/south hall) --[south]--> hall (1st floor, middle of north/south hall) — correct
  - step 23: stairwell (second floor) --[east]--> outside physics office — correct
  - step 41: maze of twisty passages (stop 2) --[north]--> outside physics office — hallucinated_edge
  - step 24: outside physics office --[south]--> hall (2nd floor, middle of north/south hall) — correct
  - step 11: stairwell (second floor) --[down]--> stairwell (first floor) — correct

## Conflict 12 — topology (overlap_mixed)
- description: position (1, 1, -1) occupied by multiple rooms ['hall (1st floor, north end of north/south hall)', 'outside physics office', 'stairwell (first floor)']
  - step 12: stairwell (first floor) --[east]--> hall (1st floor, north end of north/south hall) — correct
  - step 13: hall (1st floor, north end of north/south hall) --[south]--> hall (1st floor, middle of north/south hall) — correct
  - step 23: stairwell (second floor) --[east]--> outside physics office — correct
  - step 41: maze of twisty passages (stop 2) --[north]--> outside physics office — hallucinated_edge
  - step 24: outside physics office --[south]--> hall (2nd floor, middle of north/south hall) — correct
  - step 11: stairwell (second floor) --[down]--> stairwell (first floor) — correct

## Conflict 13 — topology (overlap_mixed)
- description: position (3, 1, 1) occupied by multiple rooms ['hall (1st floor, north end of north/south hall)', 'outside physics office', 'stairwell (first floor)']
  - step 12: stairwell (first floor) --[east]--> hall (1st floor, north end of north/south hall) — correct
  - step 13: hall (1st floor, north end of north/south hall) --[south]--> hall (1st floor, middle of north/south hall) — correct
  - step 23: stairwell (second floor) --[east]--> outside physics office — correct
  - step 41: maze of twisty passages (stop 2) --[north]--> outside physics office — hallucinated_edge
  - step 24: outside physics office --[south]--> hall (2nd floor, middle of north/south hall) — correct
  - step 11: stairwell (second floor) --[down]--> stairwell (first floor) — correct

## Conflict 14 — topology (overlap_mixed)
- description: position (4, 1, 2) occupied by multiple rooms ['hall (1st floor, north end of north/south hall)', 'outside physics office']
  - step 12: stairwell (first floor) --[east]--> hall (1st floor, north end of north/south hall) — correct
  - step 13: hall (1st floor, north end of north/south hall) --[south]--> hall (1st floor, middle of north/south hall) — correct
  - step 23: stairwell (second floor) --[east]--> outside physics office — correct
  - step 41: maze of twisty passages (stop 2) --[north]--> outside physics office — hallucinated_edge
  - step 24: outside physics office --[south]--> hall (2nd floor, middle of north/south hall) — correct

## Conflict 15 — topology (wrong_direction_caused_overlap)
- description: position (1, 0, -1) occupied by multiple rooms ['hall (1st floor, middle of north/south hall)', 'hall (2nd floor, middle of north/south hall)', 'maze of twisty passages (stop 2)', 'maze of twisty passages (stop 4)']
  - step 13: hall (1st floor, north end of north/south hall) --[south]--> hall (1st floor, middle of north/south hall) — correct
  - step 14: hall (1st floor, middle of north/south hall) --[east]--> hall outside elevator (1st floor) — correct
  - step 29: hall (1st floor, middle of north/south hall) --[east]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 24: outside physics office --[south]--> hall (2nd floor, middle of north/south hall) — correct
  - step 56: hall (2nd floor, middle of north/south hall) --[down]--> maze of twisty passages (stop 1) — correct
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 39: maze of twisty passages (stop 4) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 70: maze of twisty passages (stop 1) --[west]--> maze of twisty passages (stop 2) — wrong_direction (LLM: 'west', GT: 'east')
  - step 41: maze of twisty passages (stop 2) --[north]--> outside physics office — hallucinated_edge
  - step 66: maze of twisty passages (stop 2) --[out]--> gnome's lair — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))

## Conflict 16 — topology (wrong_direction_caused_overlap)
- description: position (2, 0, 0) occupied by multiple rooms ['hall (1st floor, middle of north/south hall)', 'hall (2nd floor, middle of north/south hall)', 'hall outside elevator (3rd floor)', 'maze of twisty passages (stop 2)', 'maze of twisty passages (stop 4)']
  - step 13: hall (1st floor, north end of north/south hall) --[south]--> hall (1st floor, middle of north/south hall) — correct
  - step 14: hall (1st floor, middle of north/south hall) --[east]--> hall outside elevator (1st floor) — correct
  - step 29: hall (1st floor, middle of north/south hall) --[east]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 24: outside physics office --[south]--> hall (2nd floor, middle of north/south hall) — correct
  - step 56: hall (2nd floor, middle of north/south hall) --[down]--> maze of twisty passages (stop 1) — correct
  - step 3: hall (3rd floor, middle of north/south hall) --[east]--> hall outside elevator (3rd floor) — correct
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 39: maze of twisty passages (stop 4) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 70: maze of twisty passages (stop 1) --[west]--> maze of twisty passages (stop 2) — wrong_direction (LLM: 'west', GT: 'east')
  - step 41: maze of twisty passages (stop 2) --[north]--> outside physics office — hallucinated_edge
  - step 66: maze of twisty passages (stop 2) --[out]--> gnome's lair — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))

## Conflict 17 — topology (false_positive_overlap)
- description: position (5, 0, 3) occupied by multiple rooms ['hall (2nd floor, middle of north/south hall)', 'hall outside elevator (3rd floor)']
  - step 24: outside physics office --[south]--> hall (2nd floor, middle of north/south hall) — correct
  - step 56: hall (2nd floor, middle of north/south hall) --[down]--> maze of twisty passages (stop 1) — correct
  - step 3: hall (3rd floor, middle of north/south hall) --[east]--> hall outside elevator (3rd floor) — correct

## Conflict 18 — topology (wrong_direction_caused_overlap)
- description: position (3, 0, 1) occupied by multiple rooms ['hall (1st floor, middle of north/south hall)', 'hall (2nd floor, middle of north/south hall)', 'hall outside elevator (3rd floor)', 'maze of twisty passages (stop 2)', 'maze of twisty passages (stop 4)']
  - step 13: hall (1st floor, north end of north/south hall) --[south]--> hall (1st floor, middle of north/south hall) — correct
  - step 14: hall (1st floor, middle of north/south hall) --[east]--> hall outside elevator (1st floor) — correct
  - step 29: hall (1st floor, middle of north/south hall) --[east]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 24: outside physics office --[south]--> hall (2nd floor, middle of north/south hall) — correct
  - step 56: hall (2nd floor, middle of north/south hall) --[down]--> maze of twisty passages (stop 1) — correct
  - step 3: hall (3rd floor, middle of north/south hall) --[east]--> hall outside elevator (3rd floor) — correct
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 39: maze of twisty passages (stop 4) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 70: maze of twisty passages (stop 1) --[west]--> maze of twisty passages (stop 2) — wrong_direction (LLM: 'west', GT: 'east')
  - step 41: maze of twisty passages (stop 2) --[north]--> outside physics office — hallucinated_edge
  - step 66: maze of twisty passages (stop 2) --[out]--> gnome's lair — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))

## Conflict 19 — topology (wrong_direction_caused_overlap)
- description: position (4, 0, 2) occupied by multiple rooms ['hall (1st floor, middle of north/south hall)', 'hall (2nd floor, middle of north/south hall)', 'hall outside elevator (3rd floor)', 'maze of twisty passages (stop 2)', 'maze of twisty passages (stop 4)']
  - step 13: hall (1st floor, north end of north/south hall) --[south]--> hall (1st floor, middle of north/south hall) — correct
  - step 14: hall (1st floor, middle of north/south hall) --[east]--> hall outside elevator (1st floor) — correct
  - step 29: hall (1st floor, middle of north/south hall) --[east]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 24: outside physics office --[south]--> hall (2nd floor, middle of north/south hall) — correct
  - step 56: hall (2nd floor, middle of north/south hall) --[down]--> maze of twisty passages (stop 1) — correct
  - step 3: hall (3rd floor, middle of north/south hall) --[east]--> hall outside elevator (3rd floor) — correct
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 39: maze of twisty passages (stop 4) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 70: maze of twisty passages (stop 1) --[west]--> maze of twisty passages (stop 2) — wrong_direction (LLM: 'west', GT: 'east')
  - step 41: maze of twisty passages (stop 2) --[north]--> outside physics office — hallucinated_edge
  - step 66: maze of twisty passages (stop 2) --[out]--> gnome's lair — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))

## Conflict 20 — topology (wrong_direction_caused_overlap)
- description: position (5, 0, 2) occupied by multiple rooms ['hall outside elevator (1st floor)', 'maze of twisty passages (stop 1)', 'maze of twisty passages (stop 3)', 'maze of twisty passages (stop 4)']
  - step 14: hall (1st floor, middle of north/south hall) --[east]--> hall outside elevator (1st floor) — correct
  - step 16: hall outside elevator (1st floor) --[south]--> janitor's closet — correct
  - step 29: hall (1st floor, middle of north/south hall) --[east]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 56: hall (2nd floor, middle of north/south hall) --[down]--> maze of twisty passages (stop 1) — correct
  - step 69: gnome's lair --[west]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 70: maze of twisty passages (stop 1) --[west]--> maze of twisty passages (stop 2) — wrong_direction (LLM: 'west', GT: 'east')
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))
  - step 39: maze of twisty passages (stop 4) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge

## Conflict 21 — topology (wrong_direction_caused_overlap)
- description: position (3, 0, 0) occupied by multiple rooms ['hall outside elevator (1st floor)', 'maze of twisty passages (stop 1)', 'maze of twisty passages (stop 3)', 'maze of twisty passages (stop 4)']
  - step 14: hall (1st floor, middle of north/south hall) --[east]--> hall outside elevator (1st floor) — correct
  - step 16: hall outside elevator (1st floor) --[south]--> janitor's closet — correct
  - step 29: hall (1st floor, middle of north/south hall) --[east]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 56: hall (2nd floor, middle of north/south hall) --[down]--> maze of twisty passages (stop 1) — correct
  - step 69: gnome's lair --[west]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 70: maze of twisty passages (stop 1) --[west]--> maze of twisty passages (stop 2) — wrong_direction (LLM: 'west', GT: 'east')
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))
  - step 39: maze of twisty passages (stop 4) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge

## Conflict 22 — topology (wrong_direction_caused_overlap)
- description: position (4, 0, 1) occupied by multiple rooms ['hall (1st floor, middle of north/south hall)', 'hall outside elevator (1st floor)', 'maze of twisty passages (stop 1)', 'maze of twisty passages (stop 3)', 'maze of twisty passages (stop 4)']
  - step 13: hall (1st floor, north end of north/south hall) --[south]--> hall (1st floor, middle of north/south hall) — correct
  - step 14: hall (1st floor, middle of north/south hall) --[east]--> hall outside elevator (1st floor) — correct
  - step 29: hall (1st floor, middle of north/south hall) --[east]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 16: hall outside elevator (1st floor) --[south]--> janitor's closet — correct
  - step 56: hall (2nd floor, middle of north/south hall) --[down]--> maze of twisty passages (stop 1) — correct
  - step 69: gnome's lair --[west]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 70: maze of twisty passages (stop 1) --[west]--> maze of twisty passages (stop 2) — wrong_direction (LLM: 'west', GT: 'east')
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))
  - step 39: maze of twisty passages (stop 4) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge

## Conflict 23 — topology (wrong_direction_caused_overlap)
- description: position (2, 0, -1) occupied by multiple rooms ['hall outside elevator (1st floor)', 'maze of twisty passages (stop 1)', 'maze of twisty passages (stop 3)', 'maze of twisty passages (stop 4)']
  - step 14: hall (1st floor, middle of north/south hall) --[east]--> hall outside elevator (1st floor) — correct
  - step 16: hall outside elevator (1st floor) --[south]--> janitor's closet — correct
  - step 29: hall (1st floor, middle of north/south hall) --[east]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 56: hall (2nd floor, middle of north/south hall) --[down]--> maze of twisty passages (stop 1) — correct
  - step 69: gnome's lair --[west]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 70: maze of twisty passages (stop 1) --[west]--> maze of twisty passages (stop 2) — wrong_direction (LLM: 'west', GT: 'east')
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))
  - step 39: maze of twisty passages (stop 4) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge

## Conflict 24 — topology (overlap_mixed)
- description: position (5, 0, 1) occupied by multiple rooms ["gnome's lair", 'maze of twisty passages (stop 3)']
  - step 66: maze of twisty passages (stop 2) --[out]--> gnome's lair — hallucinated_edge
  - step 69: gnome's lair --[west]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))

## Conflict 25 — topology (overlap_mixed)
- description: position (4, 0, 0) occupied by multiple rooms ["gnome's lair", 'maze of twisty passages (stop 3)']
  - step 66: maze of twisty passages (stop 2) --[out]--> gnome's lair — hallucinated_edge
  - step 69: gnome's lair --[west]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))

## Conflict 26 — topology (overlap_mixed)
- description: position (6, 0, 2) occupied by multiple rooms ["gnome's lair", 'maze of twisty passages (stop 3)']
  - step 66: maze of twisty passages (stop 2) --[out]--> gnome's lair — hallucinated_edge
  - step 69: gnome's lair --[west]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))

## Conflict 27 — topology (overlap_mixed)
- description: position (3, 0, -1) occupied by multiple rooms ["gnome's lair", 'maze of twisty passages (stop 3)']
  - step 66: maze of twisty passages (stop 2) --[out]--> gnome's lair — hallucinated_edge
  - step 69: gnome's lair --[west]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))

## Conflict 28 — naming (naming_collision_on_correct_subgraph)
- description: node 'computer site' reachable at conflicting positions [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3)]
  - step 1: computer site --[northeast]--> hall outside computer site — correct

## Conflict 29 — naming (naming_collision_on_correct_subgraph)
- description: node 'hall outside computer site' reachable at conflicting positions [(1, 1, 0), (2, 1, 1), (3, 1, 2), (4, 1, 3)]
  - step 1: computer site --[northeast]--> hall outside computer site — correct
  - step 2: hall outside computer site --[south]--> hall (3rd floor, middle of north/south hall) — correct
  - step 9: hall outside computer site --[west]--> stairwell (third floor) — correct

## Conflict 30 — naming (naming_collision_on_correct_subgraph)
- description: node 'hall (3rd floor, middle of north/south hall)' reachable at conflicting positions [(1, 0, 0), (2, 0, 1), (3, 0, 2), (4, 0, 3)]
  - step 2: hall outside computer site --[south]--> hall (3rd floor, middle of north/south hall) — correct
  - step 3: hall (3rd floor, middle of north/south hall) --[east]--> hall outside elevator (3rd floor) — correct

## Conflict 31 — naming (naming_collision_on_correct_subgraph)
- description: node 'stairwell (third floor)' reachable at conflicting positions [(0, 1, 0), (1, 1, 1), (2, 1, 2), (3, 1, 3)]
  - step 9: hall outside computer site --[west]--> stairwell (third floor) — correct
  - step 10: stairwell (third floor) --[down]--> stairwell (second floor) — correct

## Conflict 32 — naming (naming_collision_on_correct_subgraph)
- description: node 'stairwell (second floor)' reachable at conflicting positions [(0, 1, -1), (1, 1, 0), (2, 1, 1), (2, 1, 2), (3, 1, 2), (3, 1, 3)]
  - step 10: stairwell (third floor) --[down]--> stairwell (second floor) — correct
  - step 11: stairwell (second floor) --[down]--> stairwell (first floor) — correct
  - step 23: stairwell (second floor) --[east]--> outside physics office — correct

## Conflict 33 — naming (naming_collision_on_correct_subgraph)
- description: node 'stairwell (first floor)' reachable at conflicting positions [(0, 1, -2), (1, 1, -1), (1, 1, 0), (2, 1, 0), (2, 1, 1), (3, 1, 1), (3, 1, 2)]
  - step 11: stairwell (second floor) --[down]--> stairwell (first floor) — correct
  - step 12: stairwell (first floor) --[east]--> hall (1st floor, north end of north/south hall) — correct

## Conflict 34 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'outside physics office' reachable at conflicting positions [(1, 1, -1), (2, 1, 0), (3, 1, 1), (4, 1, 2)]
  - step 23: stairwell (second floor) --[east]--> outside physics office — correct
  - step 41: maze of twisty passages (stop 2) --[north]--> outside physics office — hallucinated_edge
  - step 24: outside physics office --[south]--> hall (2nd floor, middle of north/south hall) — correct

## Conflict 35 — naming (naming_collision_on_correct_subgraph)
- description: node 'hall (2nd floor, middle of north/south hall)' reachable at conflicting positions [(1, 0, -1), (2, 0, 0), (3, 0, 1), (4, 0, 2), (5, 0, 3)]
  - step 24: outside physics office --[south]--> hall (2nd floor, middle of north/south hall) — correct
  - step 56: hall (2nd floor, middle of north/south hall) --[down]--> maze of twisty passages (stop 1) — correct

## Conflict 36 — naming (naming_mixed)
- description: node 'maze of twisty passages (stop 2)' reachable at conflicting positions [(1, 0, -1), (2, 0, 0), (3, 0, 1), (4, 0, 2)]
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 39: maze of twisty passages (stop 4) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 70: maze of twisty passages (stop 1) --[west]--> maze of twisty passages (stop 2) — wrong_direction (LLM: 'west', GT: 'east')
  - step 41: maze of twisty passages (stop 2) --[north]--> outside physics office — hallucinated_edge
  - step 66: maze of twisty passages (stop 2) --[out]--> gnome's lair — hallucinated_edge

## Conflict 37 — naming (naming_mixed)
- description: node 'maze of twisty passages (stop 3)' reachable at conflicting positions [(2, 0, -1), (3, 0, -1), (3, 0, 0), (4, 0, 0), (4, 0, 1), (5, 0, 1), (5, 0, 2), (6, 0, 2)]
  - step 34: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))

## Conflict 38 — naming (naming_mixed)
- description: node 'maze of twisty passages (stop 4)' reachable at conflicting positions [(1, 0, -1), (2, 0, -1), (2, 0, 0), (3, 0, 0), (3, 0, 1), (4, 0, 1), (4, 0, 2), (5, 0, 2)]
  - step 37: maze of twisty passages (stop 3) --[west]--> maze of twisty passages (stop 4) — swapped_src_dst (LLM: maze of twisty passages (stop 3)--[west]-->maze of twisty passages (stop 4) but GT has maze of twisty passages (stop 4)--[east]-->maze of twisty passages (stop 3))
  - step 39: maze of twisty passages (stop 4) --[west]--> maze of twisty passages (stop 2) — hallucinated_edge

## Conflict 39 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'maze of twisty passages (stop 1)' reachable at conflicting positions [(1, 0, -2), (2, 0, -1), (3, 0, 0), (4, 0, 1), (5, 0, 2)]
  - step 29: hall (1st floor, middle of north/south hall) --[east]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 56: hall (2nd floor, middle of north/south hall) --[down]--> maze of twisty passages (stop 1) — correct
  - step 69: gnome's lair --[west]--> maze of twisty passages (stop 1) — hallucinated_edge
  - step 70: maze of twisty passages (stop 1) --[west]--> maze of twisty passages (stop 2) — wrong_direction (LLM: 'west', GT: 'east')

## Conflict 40 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'hall (1st floor, middle of north/south hall)' reachable at conflicting positions [(1, 0, -1), (2, 0, 0), (3, 0, 1), (4, 0, 1), (4, 0, 2)]
  - step 13: hall (1st floor, north end of north/south hall) --[south]--> hall (1st floor, middle of north/south hall) — correct
  - step 14: hall (1st floor, middle of north/south hall) --[east]--> hall outside elevator (1st floor) — correct
  - step 29: hall (1st floor, middle of north/south hall) --[east]--> maze of twisty passages (stop 1) — hallucinated_edge

## Conflict 41 — naming (naming_mixed)
- description: node "gnome's lair" reachable at conflicting positions [(3, 0, -1), (4, 0, 0), (5, 0, 1), (6, 0, 2)]
  - step 66: maze of twisty passages (stop 2) --[out]--> gnome's lair — hallucinated_edge
  - step 69: gnome's lair --[west]--> maze of twisty passages (stop 1) — hallucinated_edge

## Conflict 42 — naming (naming_collision_on_correct_subgraph)
- description: node 'hall (1st floor, north end of north/south hall)' reachable at conflicting positions [(1, 1, -2), (1, 1, -1), (2, 1, 0), (3, 1, 1), (4, 1, 1), (4, 1, 2)]
  - step 12: stairwell (first floor) --[east]--> hall (1st floor, north end of north/south hall) — correct
  - step 13: hall (1st floor, north end of north/south hall) --[south]--> hall (1st floor, middle of north/south hall) — correct

## Conflict 43 — naming (naming_collision_on_correct_subgraph)
- description: node 'hall outside elevator (1st floor)' reachable at conflicting positions [(2, 0, -1), (3, 0, 0), (4, 0, 1), (5, 0, 2)]
  - step 14: hall (1st floor, middle of north/south hall) --[east]--> hall outside elevator (1st floor) — correct
  - step 16: hall outside elevator (1st floor) --[south]--> janitor's closet — correct

## Conflict 44 — naming (naming_collision_on_correct_subgraph)
- description: node "janitor's closet" reachable at conflicting positions [(2, -1, -1), (3, -1, 0), (4, -1, 1), (5, -1, 2)]
  - step 16: hall outside elevator (1st floor) --[south]--> janitor's closet — correct

## Conflict 45 — naming (naming_collision_on_correct_subgraph)
- description: node 'hall outside elevator (3rd floor)' reachable at conflicting positions [(2, 0, 0), (3, 0, 1), (4, 0, 2), (5, 0, 3)]
  - step 3: hall (3rd floor, middle of north/south hall) --[east]--> hall outside elevator (3rd floor) — correct
