# Conflict analysis: jewel

- LLM edges: 17
- GT edges: 31
- Conflicts: 29
- Type distribution: {'direction': 2, 'topology': 11, 'naming': 16}
- Root-cause distribution: {'real_vs_hallucinated': 2, 'overlap_mixed': 6, 'false_positive_overlap': 5, 'naming_collision_on_correct_subgraph': 11, 'real_name_corrupted_by_neighbour_error': 4, 'naming_mixed': 1}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'shaft base' has multiple outgoing edges labelled 'up'
  - step 45: shaft base --[up]--> top of the shaft — hallucinated_edge
  - step None: shaft base --[up]--> middle shaft — correct

## Conflict 2 — direction (real_vs_hallucinated)
- description: node 'top of the shaft' has multiple outgoing edges labelled 'down'
  - step None: top of the shaft --[down]--> shaft base — hallucinated_edge
  - step 54: top of the shaft --[down]--> middle shaft — correct

## Conflict 3 — topology (overlap_mixed)
- description: position (-1, 1, 5) occupied by multiple rooms ['east-west passage', 'middle shaft', 'top of the shaft']
  - step 17: cool cavern --[west]--> east-west passage — correct
  - step 18: east-west passage --[west]--> carved steps — correct
  - step 54: top of the shaft --[down]--> middle shaft — correct
  - step 55: middle shaft --[down]--> shaft base — correct
  - step 45: shaft base --[up]--> top of the shaft — hallucinated_edge
  - step 46: top of the shaft --[east]--> on the ledge — correct

## Conflict 4 — topology (false_positive_overlap)
- description: position (-4, 3, 7) occupied by multiple rooms ['moss-filled cave', 'narrow passage']
  - step 19: carved steps --[north]--> moss-filled cave — correct
  - step 26: moss-filled cave --[down]--> cleaning closet — correct
  - step 40: mineralized corridor --[west]--> narrow passage — correct
  - step 41: narrow passage --[west]--> gaseous geyser — correct

## Conflict 5 — topology (false_positive_overlap)
- description: position (0, 1, 3) occupied by multiple rooms ['moss-filled cave', 'narrow passage', 'on the ledge']
  - step 19: carved steps --[north]--> moss-filled cave — correct
  - step 26: moss-filled cave --[down]--> cleaning closet — correct
  - step 40: mineralized corridor --[west]--> narrow passage — correct
  - step 41: narrow passage --[west]--> gaseous geyser — correct
  - step 46: top of the shaft --[east]--> on the ledge — correct

## Conflict 6 — topology (false_positive_overlap)
- description: position (0, 1, 0) occupied by multiple rooms ['moss-filled cave', 'narrow passage']
  - step 19: carved steps --[north]--> moss-filled cave — correct
  - step 26: moss-filled cave --[down]--> cleaning closet — correct
  - step 40: mineralized corridor --[west]--> narrow passage — correct
  - step 41: narrow passage --[west]--> gaseous geyser — correct

## Conflict 7 — topology (false_positive_overlap)
- description: position (-2, 2, 5) occupied by multiple rooms ['moss-filled cave', 'narrow passage']
  - step 19: carved steps --[north]--> moss-filled cave — correct
  - step 26: moss-filled cave --[down]--> cleaning closet — correct
  - step 40: mineralized corridor --[west]--> narrow passage — correct
  - step 41: narrow passage --[west]--> gaseous geyser — correct

## Conflict 8 — topology (false_positive_overlap)
- description: position (0, 1, 2) occupied by multiple rooms ['cleaning closet', 'narrow passage', 'on the ledge']
  - step 26: moss-filled cave --[down]--> cleaning closet — correct
  - step 40: mineralized corridor --[west]--> narrow passage — correct
  - step 41: narrow passage --[west]--> gaseous geyser — correct
  - step 46: top of the shaft --[east]--> on the ledge — correct

## Conflict 9 — topology (overlap_mixed)
- description: position (0, 1, 5) occupied by multiple rooms ['cool cavern', 'on the ledge']
  - step 3: fifth layer dropoff --[west]--> cool cavern — hallucinated_edge
  - step 4: cool cavern --[northwest]--> mineralized corridor — correct
  - step 17: cool cavern --[west]--> east-west passage — correct
  - step 46: top of the shaft --[east]--> on the ledge — correct

## Conflict 10 — topology (overlap_mixed)
- description: position (-1, 1, 2) occupied by multiple rooms ['gaseous geyser', 'middle shaft', 'shaft base', 'top of the shaft']
  - step 41: narrow passage --[west]--> gaseous geyser — correct
  - step 44: gaseous geyser --[up]--> shaft base — swapped_src_dst (LLM: gaseous geyser--[up]-->shaft base but GT has shaft base--[down]-->gaseous geyser)
  - step 66: gaseous geyser --[west]--> quarry — correct
  - step 54: top of the shaft --[down]--> middle shaft — correct
  - step 55: middle shaft --[down]--> shaft base — correct
  - step 45: shaft base --[up]--> top of the shaft — hallucinated_edge
  - step 46: top of the shaft --[east]--> on the ledge — correct

## Conflict 11 — topology (overlap_mixed)
- description: position (-1, 1, 3) occupied by multiple rooms ['gaseous geyser', 'middle shaft', 'shaft base', 'top of the shaft']
  - step 41: narrow passage --[west]--> gaseous geyser — correct
  - step 44: gaseous geyser --[up]--> shaft base — swapped_src_dst (LLM: gaseous geyser--[up]-->shaft base but GT has shaft base--[down]-->gaseous geyser)
  - step 66: gaseous geyser --[west]--> quarry — correct
  - step 54: top of the shaft --[down]--> middle shaft — correct
  - step 55: middle shaft --[down]--> shaft base — correct
  - step 45: shaft base --[up]--> top of the shaft — hallucinated_edge
  - step 46: top of the shaft --[east]--> on the ledge — correct

## Conflict 12 — topology (overlap_mixed)
- description: position (-1, 1, 1) occupied by multiple rooms ['gaseous geyser', 'middle shaft', 'shaft base']
  - step 41: narrow passage --[west]--> gaseous geyser — correct
  - step 44: gaseous geyser --[up]--> shaft base — swapped_src_dst (LLM: gaseous geyser--[up]-->shaft base but GT has shaft base--[down]-->gaseous geyser)
  - step 66: gaseous geyser --[west]--> quarry — correct
  - step 54: top of the shaft --[down]--> middle shaft — correct
  - step 55: middle shaft --[down]--> shaft base — correct
  - step 45: shaft base --[up]--> top of the shaft — hallucinated_edge

## Conflict 13 — topology (overlap_mixed)
- description: position (-1, 1, 4) occupied by multiple rooms ['middle shaft', 'shaft base', 'top of the shaft']
  - step 54: top of the shaft --[down]--> middle shaft — correct
  - step 55: middle shaft --[down]--> shaft base — correct
  - step 44: gaseous geyser --[up]--> shaft base — swapped_src_dst (LLM: gaseous geyser--[up]-->shaft base but GT has shaft base--[down]-->gaseous geyser)
  - step 45: shaft base --[up]--> top of the shaft — hallucinated_edge
  - step 46: top of the shaft --[east]--> on the ledge — correct

## Conflict 14 — naming (naming_collision_on_correct_subgraph)
- description: node 'carved steps' reachable at conflicting positions [(-4, 2, 7), (-2, 1, 5), (0, 0, 0), (0, 0, 3)]
  - step 18: east-west passage --[west]--> carved steps — correct
  - step 19: carved steps --[north]--> moss-filled cave — correct
  - step 30: carved steps --[south]--> lonely burial ground — correct

## Conflict 15 — naming (naming_collision_on_correct_subgraph)
- description: node 'east-west passage' reachable at conflicting positions [(-3, 2, 7), (-1, 1, 5), (1, 0, 0), (1, 0, 3)]
  - step 17: cool cavern --[west]--> east-west passage — correct
  - step 18: east-west passage --[west]--> carved steps — correct

## Conflict 16 — naming (naming_collision_on_correct_subgraph)
- description: node 'moss-filled cave' reachable at conflicting positions [(-4, 3, 7), (-2, 2, 5), (0, 1, 0), (0, 1, 3)]
  - step 19: carved steps --[north]--> moss-filled cave — correct
  - step 26: moss-filled cave --[down]--> cleaning closet — correct

## Conflict 17 — naming (naming_collision_on_correct_subgraph)
- description: node 'lonely burial ground' reachable at conflicting positions [(-4, 1, 7), (-2, 0, 5), (0, -1, 0), (0, -1, 3)]
  - step 30: carved steps --[south]--> lonely burial ground — correct

## Conflict 18 — naming (naming_collision_on_correct_subgraph)
- description: node 'cleaning closet' reachable at conflicting positions [(-4, 3, 6), (-2, 2, 4), (0, 1, -1), (0, 1, 2)]
  - step 26: moss-filled cave --[down]--> cleaning closet — correct

## Conflict 19 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'cool cavern' reachable at conflicting positions [(-4, 3, 9), (-2, 2, 7), (0, 1, 5), (2, 0, 0), (2, 0, 3)]
  - step 3: fifth layer dropoff --[west]--> cool cavern — hallucinated_edge
  - step 4: cool cavern --[northwest]--> mineralized corridor — correct
  - step 17: cool cavern --[west]--> east-west passage — correct

## Conflict 20 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'fifth layer dropoff' reachable at conflicting positions [(-3, 3, 9), (-1, 2, 7), (1, 1, 2), (1, 1, 5), (3, 0, 0), (3, 0, 3)]
  - step 8: tight fit --[up]--> fifth layer dropoff — correct
  - step 3: fifth layer dropoff --[west]--> cool cavern — hallucinated_edge

## Conflict 21 — naming (naming_collision_on_correct_subgraph)
- description: node 'mineralized corridor' reachable at conflicting positions [(-3, 3, 7), (-1, 2, 5), (1, 1, 0), (1, 1, 3)]
  - step 4: cool cavern --[northwest]--> mineralized corridor — correct
  - step 7: mineralized corridor --[up]--> tight fit — correct
  - step 40: mineralized corridor --[west]--> narrow passage — correct

## Conflict 22 — naming (naming_collision_on_correct_subgraph)
- description: node 'tight fit' reachable at conflicting positions [(-3, 3, 8), (-1, 2, 6), (1, 1, 1), (1, 1, 4), (3, 0, -1)]
  - step 7: mineralized corridor --[up]--> tight fit — correct
  - step 8: tight fit --[up]--> fifth layer dropoff — correct

## Conflict 23 — naming (naming_collision_on_correct_subgraph)
- description: node 'narrow passage' reachable at conflicting positions [(-4, 3, 7), (-2, 2, 5), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3)]
  - step 40: mineralized corridor --[west]--> narrow passage — correct
  - step 41: narrow passage --[west]--> gaseous geyser — correct

## Conflict 24 — naming (naming_mixed)
- description: node 'gaseous geyser' reachable at conflicting positions [(-5, 3, 7), (-3, 2, 5), (-1, 1, 0), (-1, 1, 1), (-1, 1, 2), (-1, 1, 3)]
  - step 41: narrow passage --[west]--> gaseous geyser — correct
  - step 44: gaseous geyser --[up]--> shaft base — swapped_src_dst (LLM: gaseous geyser--[up]-->shaft base but GT has shaft base--[down]-->gaseous geyser)
  - step 66: gaseous geyser --[west]--> quarry — correct

## Conflict 25 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'shaft base' reachable at conflicting positions [(-1, 1, 1), (-1, 1, 2), (-1, 1, 3), (-1, 1, 4)]
  - step 44: gaseous geyser --[up]--> shaft base — swapped_src_dst (LLM: gaseous geyser--[up]-->shaft base but GT has shaft base--[down]-->gaseous geyser)
  - step 55: middle shaft --[down]--> shaft base — correct
  - step 45: shaft base --[up]--> top of the shaft — hallucinated_edge

## Conflict 26 — naming (naming_collision_on_correct_subgraph)
- description: node 'quarry' reachable at conflicting positions [(-2, 1, 0), (-2, 1, 1), (-2, 1, 2), (-2, 1, 3)]
  - step 66: gaseous geyser --[west]--> quarry — correct

## Conflict 27 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'top of the shaft' reachable at conflicting positions [(-1, 1, 2), (-1, 1, 3), (-1, 1, 4), (-1, 1, 5), (-1, 1, 6)]
  - step 45: shaft base --[up]--> top of the shaft — hallucinated_edge
  - step 46: top of the shaft --[east]--> on the ledge — correct
  - step 54: top of the shaft --[down]--> middle shaft — correct

## Conflict 28 — naming (naming_collision_on_correct_subgraph)
- description: node 'middle shaft' reachable at conflicting positions [(-1, 1, 1), (-1, 1, 2), (-1, 1, 3), (-1, 1, 4), (-1, 1, 5)]
  - step 54: top of the shaft --[down]--> middle shaft — correct
  - step 55: middle shaft --[down]--> shaft base — correct

## Conflict 29 — naming (naming_collision_on_correct_subgraph)
- description: node 'on the ledge' reachable at conflicting positions [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5)]
  - step 46: top of the shaft --[east]--> on the ledge — correct
