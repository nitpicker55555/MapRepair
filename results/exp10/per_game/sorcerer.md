# Conflict analysis: sorcerer

- LLM edges: 25
- GT edges: 43
- Conflicts: 4
- Type distribution: {'direction': 2, 'topology': 2}
- Root-cause distribution: {'all_hallucinated_edges': 1, 'real_vs_hallucinated': 1, 'overlap_mixed': 1, 'name_hallucination_caused_overlap': 1}

## Conflict 1 — direction (all_hallucinated_edges)
- description: node "belboz's quarters" has multiple outgoing edges labelled 'east'
  - step None: belboz's quarters --[east]--> quarters — hallucinated_edge
  - step 14: belboz's quarters --[east]--> hallway (marble archway leading south) — hallucinated_edge

## Conflict 2 — direction (real_vs_hallucinated)
- description: node 'lobby' has multiple outgoing edges labelled 'north'
  - step None: lobby --[north]--> hallway (marble archway leading south) — correct
  - step 38: lobby --[north]--> hallway (wooden door leading north) — hallucinated_edge

## Conflict 3 — topology (overlap_mixed)
- description: position (2, 0, 3) occupied by multiple rooms ["belboz's quarters", "helistar's quarters"]
  - step 6: quarters --[west]--> belboz's quarters — hallucinated_edge
  - step 14: belboz's quarters --[east]--> hallway (marble archway leading south) — hallucinated_edge
  - step 39: hallway (wooden door leading north) --[west]--> helistar's quarters — hallucinated_edge
  - step 43: helistar's quarters --[northeast]--> twisted forest — hallucinated_edge

## Conflict 4 — topology (name_hallucination_caused_overlap)
- description: position (3, 0, 3) occupied by multiple rooms ['hallway (marble archway leading south)', 'hallway (wooden door leading north)', 'quarters']
  - step 14: belboz's quarters --[east]--> hallway (marble archway leading south) — hallucinated_edge
  - step 16: hallway (marble archway leading south) --[south]--> lobby — correct
  - step 38: lobby --[north]--> hallway (wooden door leading north) — hallucinated_edge
  - step 39: hallway (wooden door leading north) --[west]--> helistar's quarters — hallucinated_edge
  - step 3: in the bed --[west]--> quarters — src_hallucinated
  - step 6: quarters --[west]--> belboz's quarters — hallucinated_edge
