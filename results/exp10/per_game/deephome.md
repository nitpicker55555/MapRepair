# Conflict analysis: deephome

- LLM edges: 26
- GT edges: 46
- Conflicts: 20
- Type distribution: {'direction': 1, 'topology': 12, 'naming': 7}
- Root-cause distribution: {'real_vs_hallucinated': 1, 'false_positive_overlap': 2, 'overlap_mixed': 10, 'naming_collision_on_correct_subgraph': 4, 'real_name_corrupted_by_neighbour_error': 3}

## Conflict 1 — direction (real_vs_hallucinated)
- description: node 'training grounds' has multiple outgoing edges labelled 'northeast'
  - step None: training grounds --[northeast]--> railway station (soldiers' barracks) — hallucinated_edge
  - step 64: training grounds --[northeast]--> soldiers' barracks — correct

## Conflict 2 — topology (false_positive_overlap)
- description: position (0, 0, 0) occupied by multiple rooms ['armory', 'city generator']
  - step 65: soldiers' barracks --[west]--> armory — correct
  - step 29: extreme south main street --[east]--> city generator — correct

## Conflict 3 — topology (false_positive_overlap)
- description: position (-1, -2, 0) occupied by multiple rooms ['ore mines', 'waterfall']
  - step 21: mining center --[south]--> ore mines — correct
  - step 61: training grounds --[southwest]--> waterfall — correct

## Conflict 4 — topology (overlap_mixed)
- description: position (0, -1, 0) occupied by multiple rooms ['coal mines', 'training grounds']
  - step 24: mining center --[east]--> coal mines — correct
  - step 60: railway station (soldiers' barracks) --[southwest]--> training grounds — hallucinated_edge
  - step 61: training grounds --[southwest]--> waterfall — correct
  - step 64: training grounds --[northeast]--> soldiers' barracks — correct

## Conflict 5 — topology (overlap_mixed)
- description: position (1, 0, 0) occupied by multiple rooms ['armory', "railway station (soldiers' barracks)", "soldiers' barracks"]
  - step 65: soldiers' barracks --[west]--> armory — correct
  - step 59: railcar --[west]--> railway station (soldiers' barracks) — hallucinated_edge
  - step 69: soldiers' barracks --[east]--> railway station (soldiers' barracks) — correct
  - step 60: railway station (soldiers' barracks) --[southwest]--> training grounds — hallucinated_edge
  - step 64: training grounds --[northeast]--> soldiers' barracks — correct
  - step 67: soldiers' barracks --[south]--> mess hall — correct

## Conflict 6 — topology (overlap_mixed)
- description: position (3, 0, 0) occupied by multiple rooms ['armory', 'railcar', "railway station (soldiers' barracks)", "soldiers' barracks"]
  - step 65: soldiers' barracks --[west]--> armory — correct
  - step 42: railcar --[out]--> railway station (smithy court) — correct
  - step 59: railcar --[west]--> railway station (soldiers' barracks) — hallucinated_edge
  - step 69: soldiers' barracks --[east]--> railway station (soldiers' barracks) — correct
  - step 60: railway station (soldiers' barracks) --[southwest]--> training grounds — hallucinated_edge
  - step 64: training grounds --[northeast]--> soldiers' barracks — correct
  - step 67: soldiers' barracks --[south]--> mess hall — correct

## Conflict 7 — topology (overlap_mixed)
- description: position (2, 0, 0) occupied by multiple rooms ['armory', "railway station (soldiers' barracks)", "soldiers' barracks"]
  - step 65: soldiers' barracks --[west]--> armory — correct
  - step 59: railcar --[west]--> railway station (soldiers' barracks) — hallucinated_edge
  - step 69: soldiers' barracks --[east]--> railway station (soldiers' barracks) — correct
  - step 60: railway station (soldiers' barracks) --[southwest]--> training grounds — hallucinated_edge
  - step 64: training grounds --[northeast]--> soldiers' barracks — correct
  - step 67: soldiers' barracks --[south]--> mess hall — correct

## Conflict 8 — topology (overlap_mixed)
- description: position (4, 0, 0) occupied by multiple rooms ['railcar', "railway station (soldiers' barracks)", "soldiers' barracks"]
  - step 42: railcar --[out]--> railway station (smithy court) — correct
  - step 59: railcar --[west]--> railway station (soldiers' barracks) — hallucinated_edge
  - step 69: soldiers' barracks --[east]--> railway station (soldiers' barracks) — correct
  - step 60: railway station (soldiers' barracks) --[southwest]--> training grounds — hallucinated_edge
  - step 64: training grounds --[northeast]--> soldiers' barracks — correct
  - step 65: soldiers' barracks --[west]--> armory — correct
  - step 67: soldiers' barracks --[south]--> mess hall — correct

## Conflict 9 — topology (overlap_mixed)
- description: position (2, -1, 0) occupied by multiple rooms ['mess hall', 'training grounds']
  - step 67: soldiers' barracks --[south]--> mess hall — correct
  - step 60: railway station (soldiers' barracks) --[southwest]--> training grounds — hallucinated_edge
  - step 61: training grounds --[southwest]--> waterfall — correct
  - step 64: training grounds --[northeast]--> soldiers' barracks — correct

## Conflict 10 — topology (overlap_mixed)
- description: position (3, -1, 0) occupied by multiple rooms ['mess hall', 'training grounds']
  - step 67: soldiers' barracks --[south]--> mess hall — correct
  - step 60: railway station (soldiers' barracks) --[southwest]--> training grounds — hallucinated_edge
  - step 61: training grounds --[southwest]--> waterfall — correct
  - step 64: training grounds --[northeast]--> soldiers' barracks — correct

## Conflict 11 — topology (overlap_mixed)
- description: position (4, -1, 0) occupied by multiple rooms ['mess hall', 'training grounds']
  - step 67: soldiers' barracks --[south]--> mess hall — correct
  - step 60: railway station (soldiers' barracks) --[southwest]--> training grounds — hallucinated_edge
  - step 61: training grounds --[southwest]--> waterfall — correct
  - step 64: training grounds --[northeast]--> soldiers' barracks — correct

## Conflict 12 — topology (overlap_mixed)
- description: position (1, -1, 0) occupied by multiple rooms ['mess hall', 'training grounds']
  - step 67: soldiers' barracks --[south]--> mess hall — correct
  - step 60: railway station (soldiers' barracks) --[southwest]--> training grounds — hallucinated_edge
  - step 61: training grounds --[southwest]--> waterfall — correct
  - step 64: training grounds --[northeast]--> soldiers' barracks — correct

## Conflict 13 — topology (overlap_mixed)
- description: position (5, 0, 0) occupied by multiple rooms ['railcar', "railway station (soldiers' barracks)"]
  - step 42: railcar --[out]--> railway station (smithy court) — correct
  - step 59: railcar --[west]--> railway station (soldiers' barracks) — hallucinated_edge
  - step 69: soldiers' barracks --[east]--> railway station (soldiers' barracks) — correct
  - step 60: railway station (soldiers' barracks) --[southwest]--> training grounds — hallucinated_edge

## Conflict 14 — naming (naming_collision_on_correct_subgraph)
- description: node 'armory' reachable at conflicting positions [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
  - step 65: soldiers' barracks --[west]--> armory — correct

## Conflict 15 — naming (naming_collision_on_correct_subgraph)
- description: node "soldiers' barracks" reachable at conflicting positions [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]
  - step 64: training grounds --[northeast]--> soldiers' barracks — correct
  - step 65: soldiers' barracks --[west]--> armory — correct
  - step 67: soldiers' barracks --[south]--> mess hall — correct
  - step 69: soldiers' barracks --[east]--> railway station (soldiers' barracks) — correct

## Conflict 16 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'training grounds' reachable at conflicting positions [(0, -1, 0), (1, -1, 0), (2, -1, 0), (3, -1, 0), (4, -1, 0)]
  - step 60: railway station (soldiers' barracks) --[southwest]--> training grounds — hallucinated_edge
  - step 61: training grounds --[southwest]--> waterfall — correct
  - step 64: training grounds --[northeast]--> soldiers' barracks — correct

## Conflict 17 — naming (naming_collision_on_correct_subgraph)
- description: node 'mess hall' reachable at conflicting positions [(1, -1, 0), (2, -1, 0), (3, -1, 0), (4, -1, 0)]
  - step 67: soldiers' barracks --[south]--> mess hall — correct

## Conflict 18 — naming (real_name_corrupted_by_neighbour_error)
- description: node "railway station (soldiers' barracks)" reachable at conflicting positions [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0)]
  - step 59: railcar --[west]--> railway station (soldiers' barracks) — hallucinated_edge
  - step 69: soldiers' barracks --[east]--> railway station (soldiers' barracks) — correct
  - step 60: railway station (soldiers' barracks) --[southwest]--> training grounds — hallucinated_edge

## Conflict 19 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'railcar' reachable at conflicting positions [(3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0)]
  - step 42: railcar --[out]--> railway station (smithy court) — correct
  - step 59: railcar --[west]--> railway station (soldiers' barracks) — hallucinated_edge

## Conflict 20 — naming (naming_collision_on_correct_subgraph)
- description: node 'waterfall' reachable at conflicting positions [(-1, -2, 0), (0, -2, 0), (1, -2, 0), (2, -2, 0)]
  - step 61: training grounds --[southwest]--> waterfall — correct
