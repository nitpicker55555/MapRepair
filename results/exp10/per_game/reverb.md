# Conflict analysis: reverb

- LLM edges: 23
- GT edges: 25
- Conflicts: 45
- Type distribution: {'direction': 10, 'topology': 16, 'naming': 19}
- Root-cause distribution: {'name_hallucination': 10, 'real_vs_hallucinated': 4, 'name_hallucination_caused_overlap': 16, 'naming_collision_on_correct_subgraph': 6, 'real_name_corrupted_by_neighbour_error': 9}

## Conflict 1 — direction (name_hallucination)
- description: node 'pizza parlor' has multiple outgoing edges labelled 'south'
  - step 6: pizza parlor --[south]--> street, by pizza parlor — correct
  - step 36: pizza parlor --[south]--> by pizza parlor — dst_hallucinated

## Conflict 2 — direction (real_vs_hallucinated)
- description: node 'street, near courthouse' has multiple outgoing edges labelled 'west'
  - step 15: street, near courthouse --[west]--> downtown — correct
  - step None: street, near courthouse --[west]--> office building — hallucinated_edge

## Conflict 3 — direction (real_vs_hallucinated)
- description: node 'office building' has multiple outgoing edges labelled 'east'
  - step None: office building --[east]--> downtown — correct
  - step 17: office building --[east]--> law office — hallucinated_edge
  - step 30: office building --[east]--> street, near courthouse — hallucinated_edge

## Conflict 4 — direction (real_vs_hallucinated)
- description: node 'law office' has multiple outgoing edges labelled 'west'
  - step None: law office --[west]--> office building — hallucinated_edge
  - step 18: law office --[west]--> office building, second floor — correct

## Conflict 5 — direction (real_vs_hallucinated)
- description: node 'office building, second floor' has multiple outgoing edges labelled 'east'
  - step None: office building, second floor --[east]--> law office — correct
  - step 19: office building, second floor --[east]--> office building, third floor — hallucinated_edge

## Conflict 6 — direction (name_hallucination)
- description: node 'office building, third floor' has multiple outgoing edges labelled 'north'
  - step None: office building, third floor --[north]--> second floor — dst_hallucinated
  - step 62: office building, third floor --[north]--> roof of office building — correct

## Conflict 7 — direction (name_hallucination)
- description: node 'street, by department store' has multiple outgoing edges labelled 'west'
  - step None: street, by department store --[west]--> by pizza parlor — dst_hallucinated
  - step 54: street, by department store --[west]--> street, by pizza parlor — correct

## Conflict 8 — direction (name_hallucination)
- description: node 'street, by department store' has multiple outgoing edges labelled 'south'
  - step 39: street, by department store --[south]--> clothing department — correct
  - step None: street, by department store --[south]--> clothing department, office building — dst_hallucinated

## Conflict 9 — direction (name_hallucination)
- description: node 'hardware department' has multiple outgoing edges labelled 'northeast'
  - step None: hardware department --[northeast]--> clothing department — correct
  - step 52: hardware department --[northeast]--> clothing department, office building — dst_hallucinated

## Conflict 10 — direction (name_hallucination)
- description: node 'second floor' has multiple outgoing edges labelled 'south'
  - step None: second floor --[south]--> office building — src_hallucinated
  - step 61: second floor --[south]--> office building, third floor — src_hallucinated

## Conflict 11 — topology (name_hallucination_caused_overlap)
- description: position (-3, -1, 0) occupied by multiple rooms ['pizza parlor', 'roof of office building', 'second floor']
  - step 4: behind the counter --[southwest]--> pizza parlor — correct
  - step 6: pizza parlor --[south]--> street, by pizza parlor — correct
  - step 36: pizza parlor --[south]--> by pizza parlor — dst_hallucinated
  - step 62: office building, third floor --[north]--> roof of office building — correct
  - step 65: roof of office building --[in]--> mayor's office — hallucinated_edge
  - step 60: office building --[north]--> second floor — dst_hallucinated
  - step 61: second floor --[south]--> office building, third floor — src_hallucinated

## Conflict 12 — topology (name_hallucination_caused_overlap)
- description: position (-4, -1, 0) occupied by multiple rooms ['pizza parlor', 'roof of office building', 'second floor']
  - step 4: behind the counter --[southwest]--> pizza parlor — correct
  - step 6: pizza parlor --[south]--> street, by pizza parlor — correct
  - step 36: pizza parlor --[south]--> by pizza parlor — dst_hallucinated
  - step 62: office building, third floor --[north]--> roof of office building — correct
  - step 65: roof of office building --[in]--> mayor's office — hallucinated_edge
  - step 60: office building --[north]--> second floor — dst_hallucinated
  - step 61: second floor --[south]--> office building, third floor — src_hallucinated

## Conflict 13 — topology (name_hallucination_caused_overlap)
- description: position (-2, -2, 0) occupied by multiple rooms ['by pizza parlor', 'downtown', "hangin' out", 'law office', 'street, by department store', 'street, by pizza parlor', 'street, near courthouse']
  - step 36: pizza parlor --[south]--> by pizza parlor — dst_hallucinated
  - step 38: by pizza parlor --[east]--> street, by department store — src_hallucinated
  - step 15: street, near courthouse --[west]--> downtown — correct
  - step 16: downtown --[west]--> office building — correct
  - step 28: hangin' out --[west]--> law office — src_hallucinated
  - step 17: office building --[east]--> law office — hallucinated_edge
  - step 18: law office --[west]--> office building, second floor — correct
  - step 53: clothing department, office building --[north]--> street, by department store — src_hallucinated
  - step 39: street, by department store --[south]--> clothing department — correct
  - step 54: street, by department store --[west]--> street, by pizza parlor — correct
  - step 6: pizza parlor --[south]--> street, by pizza parlor — correct
  - step 7: street, by pizza parlor --[west]--> street, near courthouse — correct
  - step 30: office building --[east]--> street, near courthouse — hallucinated_edge
  - step 8: street, near courthouse --[south]--> courthouse — correct

## Conflict 14 — topology (name_hallucination_caused_overlap)
- description: position (-3, -2, 0) occupied by multiple rooms ['by pizza parlor', 'downtown', "hangin' out", 'law office', 'office building', 'office building, second floor', 'office building, third floor', 'street, by department store', 'street, by pizza parlor', 'street, near courthouse']
  - step 36: pizza parlor --[south]--> by pizza parlor — dst_hallucinated
  - step 38: by pizza parlor --[east]--> street, by department store — src_hallucinated
  - step 15: street, near courthouse --[west]--> downtown — correct
  - step 16: downtown --[west]--> office building — correct
  - step 28: hangin' out --[west]--> law office — src_hallucinated
  - step 17: office building --[east]--> law office — hallucinated_edge
  - step 18: law office --[west]--> office building, second floor — correct
  - step 30: office building --[east]--> street, near courthouse — hallucinated_edge
  - step 60: office building --[north]--> second floor — dst_hallucinated
  - step 19: office building, second floor --[east]--> office building, third floor — hallucinated_edge
  - step 61: second floor --[south]--> office building, third floor — src_hallucinated
  - step 62: office building, third floor --[north]--> roof of office building — correct
  - step 53: clothing department, office building --[north]--> street, by department store — src_hallucinated
  - step 39: street, by department store --[south]--> clothing department — correct
  - step 54: street, by department store --[west]--> street, by pizza parlor — correct
  - step 6: pizza parlor --[south]--> street, by pizza parlor — correct
  - step 7: street, by pizza parlor --[west]--> street, near courthouse — correct
  - step 8: street, near courthouse --[south]--> courthouse — correct

## Conflict 15 — topology (name_hallucination_caused_overlap)
- description: position (-1, -2, 0) occupied by multiple rooms ['by pizza parlor', "hangin' out", 'street, by department store', 'street, by pizza parlor', 'street, near courthouse']
  - step 36: pizza parlor --[south]--> by pizza parlor — dst_hallucinated
  - step 38: by pizza parlor --[east]--> street, by department store — src_hallucinated
  - step 28: hangin' out --[west]--> law office — src_hallucinated
  - step 53: clothing department, office building --[north]--> street, by department store — src_hallucinated
  - step 39: street, by department store --[south]--> clothing department — correct
  - step 54: street, by department store --[west]--> street, by pizza parlor — correct
  - step 6: pizza parlor --[south]--> street, by pizza parlor — correct
  - step 7: street, by pizza parlor --[west]--> street, near courthouse — correct
  - step 30: office building --[east]--> street, near courthouse — hallucinated_edge
  - step 8: street, near courthouse --[south]--> courthouse — correct
  - step 15: street, near courthouse --[west]--> downtown — correct

## Conflict 16 — topology (name_hallucination_caused_overlap)
- description: position (-4, -2, 0) occupied by multiple rooms ['by pizza parlor', 'downtown', "hangin' out", 'law office', 'office building', 'office building, second floor', 'office building, third floor', 'street, by pizza parlor', 'street, near courthouse']
  - step 36: pizza parlor --[south]--> by pizza parlor — dst_hallucinated
  - step 38: by pizza parlor --[east]--> street, by department store — src_hallucinated
  - step 15: street, near courthouse --[west]--> downtown — correct
  - step 16: downtown --[west]--> office building — correct
  - step 28: hangin' out --[west]--> law office — src_hallucinated
  - step 17: office building --[east]--> law office — hallucinated_edge
  - step 18: law office --[west]--> office building, second floor — correct
  - step 30: office building --[east]--> street, near courthouse — hallucinated_edge
  - step 60: office building --[north]--> second floor — dst_hallucinated
  - step 19: office building, second floor --[east]--> office building, third floor — hallucinated_edge
  - step 61: second floor --[south]--> office building, third floor — src_hallucinated
  - step 62: office building, third floor --[north]--> roof of office building — correct
  - step 6: pizza parlor --[south]--> street, by pizza parlor — correct
  - step 54: street, by department store --[west]--> street, by pizza parlor — correct
  - step 7: street, by pizza parlor --[west]--> street, near courthouse — correct
  - step 8: street, near courthouse --[south]--> courthouse — correct

## Conflict 17 — topology (name_hallucination_caused_overlap)
- description: position (-2, -3, 0) occupied by multiple rooms ['clothing department', 'clothing department, office building', 'courthouse']
  - step 39: street, by department store --[south]--> clothing department — correct
  - step 40: clothing department --[southeast]--> cosmetics department — correct
  - step 49: clothing department --[southwest]--> hardware department — correct
  - step 52: hardware department --[northeast]--> clothing department, office building — dst_hallucinated
  - step 53: clothing department, office building --[north]--> street, by department store — src_hallucinated
  - step 8: street, near courthouse --[south]--> courthouse — correct

## Conflict 18 — topology (name_hallucination_caused_overlap)
- description: position (0, -3, 0) occupied by multiple rooms ['clothing department', 'clothing department, office building']
  - step 39: street, by department store --[south]--> clothing department — correct
  - step 40: clothing department --[southeast]--> cosmetics department — correct
  - step 49: clothing department --[southwest]--> hardware department — correct
  - step 52: hardware department --[northeast]--> clothing department, office building — dst_hallucinated
  - step 53: clothing department, office building --[north]--> street, by department store — src_hallucinated

## Conflict 19 — topology (name_hallucination_caused_overlap)
- description: position (-1, -3, 0) occupied by multiple rooms ['clothing department', 'clothing department, office building']
  - step 39: street, by department store --[south]--> clothing department — correct
  - step 40: clothing department --[southeast]--> cosmetics department — correct
  - step 49: clothing department --[southwest]--> hardware department — correct
  - step 52: hardware department --[northeast]--> clothing department, office building — dst_hallucinated
  - step 53: clothing department, office building --[north]--> street, by department store — src_hallucinated

## Conflict 20 — topology (name_hallucination_caused_overlap)
- description: position (-3, -3, 0) occupied by multiple rooms ['clothing department', 'clothing department, office building', 'courthouse']
  - step 39: street, by department store --[south]--> clothing department — correct
  - step 40: clothing department --[southeast]--> cosmetics department — correct
  - step 49: clothing department --[southwest]--> hardware department — correct
  - step 52: hardware department --[northeast]--> clothing department, office building — dst_hallucinated
  - step 53: clothing department, office building --[north]--> street, by department store — src_hallucinated
  - step 8: street, near courthouse --[south]--> courthouse — correct

## Conflict 21 — topology (name_hallucination_caused_overlap)
- description: position (-2, -4, 0) occupied by multiple rooms ['cosmetics department', 'hardware department']
  - step 40: clothing department --[southeast]--> cosmetics department — correct
  - step 49: clothing department --[southwest]--> hardware department — correct
  - step 52: hardware department --[northeast]--> clothing department, office building — dst_hallucinated

## Conflict 22 — topology (name_hallucination_caused_overlap)
- description: position (-1, -4, 0) occupied by multiple rooms ['cosmetics department', 'hardware department']
  - step 40: clothing department --[southeast]--> cosmetics department — correct
  - step 49: clothing department --[southwest]--> hardware department — correct
  - step 52: hardware department --[northeast]--> clothing department, office building — dst_hallucinated

## Conflict 23 — topology (name_hallucination_caused_overlap)
- description: position (-5, -2, 0) occupied by multiple rooms ['downtown', 'law office', 'office building', 'office building, second floor', 'office building, third floor', 'street, near courthouse']
  - step 15: street, near courthouse --[west]--> downtown — correct
  - step 16: downtown --[west]--> office building — correct
  - step 17: office building --[east]--> law office — hallucinated_edge
  - step 28: hangin' out --[west]--> law office — src_hallucinated
  - step 18: law office --[west]--> office building, second floor — correct
  - step 30: office building --[east]--> street, near courthouse — hallucinated_edge
  - step 60: office building --[north]--> second floor — dst_hallucinated
  - step 19: office building, second floor --[east]--> office building, third floor — hallucinated_edge
  - step 61: second floor --[south]--> office building, third floor — src_hallucinated
  - step 62: office building, third floor --[north]--> roof of office building — correct
  - step 7: street, by pizza parlor --[west]--> street, near courthouse — correct
  - step 8: street, near courthouse --[south]--> courthouse — correct

## Conflict 24 — topology (name_hallucination_caused_overlap)
- description: position (-6, -2, 0) occupied by multiple rooms ['downtown', 'law office', 'office building', 'office building, second floor', 'office building, third floor']
  - step 15: street, near courthouse --[west]--> downtown — correct
  - step 16: downtown --[west]--> office building — correct
  - step 17: office building --[east]--> law office — hallucinated_edge
  - step 28: hangin' out --[west]--> law office — src_hallucinated
  - step 18: law office --[west]--> office building, second floor — correct
  - step 30: office building --[east]--> street, near courthouse — hallucinated_edge
  - step 60: office building --[north]--> second floor — dst_hallucinated
  - step 19: office building, second floor --[east]--> office building, third floor — hallucinated_edge
  - step 61: second floor --[south]--> office building, third floor — src_hallucinated
  - step 62: office building, third floor --[north]--> roof of office building — correct

## Conflict 25 — topology (name_hallucination_caused_overlap)
- description: position (-6, -1, 0) occupied by multiple rooms ['roof of office building', 'second floor']
  - step 62: office building, third floor --[north]--> roof of office building — correct
  - step 65: roof of office building --[in]--> mayor's office — hallucinated_edge
  - step 60: office building --[north]--> second floor — dst_hallucinated
  - step 61: second floor --[south]--> office building, third floor — src_hallucinated

## Conflict 26 — topology (name_hallucination_caused_overlap)
- description: position (-5, -1, 0) occupied by multiple rooms ['roof of office building', 'second floor']
  - step 62: office building, third floor --[north]--> roof of office building — correct
  - step 65: roof of office building --[in]--> mayor's office — hallucinated_edge
  - step 60: office building --[north]--> second floor — dst_hallucinated
  - step 61: second floor --[south]--> office building, third floor — src_hallucinated

## Conflict 27 — naming (naming_collision_on_correct_subgraph)
- description: node 'behind the counter' reachable at conflicting positions [(-3, 0, 0), (-2, 0, 0), (-1, 0, 0), (0, 0, 0)]
  - step 4: behind the counter --[southwest]--> pizza parlor — correct

## Conflict 28 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'pizza parlor' reachable at conflicting positions [(-4, -1, 0), (-3, -1, 0), (-2, -1, 0), (-1, -1, 0)]
  - step 4: behind the counter --[southwest]--> pizza parlor — correct
  - step 6: pizza parlor --[south]--> street, by pizza parlor — correct
  - step 36: pizza parlor --[south]--> by pizza parlor — dst_hallucinated

## Conflict 29 — naming (naming_collision_on_correct_subgraph)
- description: node 'street, by pizza parlor' reachable at conflicting positions [(-4, -2, 0), (-3, -2, 0), (-2, -2, 0), (-1, -2, 0)]
  - step 6: pizza parlor --[south]--> street, by pizza parlor — correct
  - step 54: street, by department store --[west]--> street, by pizza parlor — correct
  - step 7: street, by pizza parlor --[west]--> street, near courthouse — correct

## Conflict 30 — naming (name_hallucination)
- description: node 'by pizza parlor' reachable at conflicting positions [(-4, -2, 0), (-3, -2, 0), (-2, -2, 0), (-1, -2, 0)]
  - step 36: pizza parlor --[south]--> by pizza parlor — dst_hallucinated
  - step 38: by pizza parlor --[east]--> street, by department store — src_hallucinated

## Conflict 31 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'street, by department store' reachable at conflicting positions [(-3, -2, 0), (-2, -2, 0), (-1, -2, 0), (0, -2, 0)]
  - step 38: by pizza parlor --[east]--> street, by department store — src_hallucinated
  - step 53: clothing department, office building --[north]--> street, by department store — src_hallucinated
  - step 39: street, by department store --[south]--> clothing department — correct
  - step 54: street, by department store --[west]--> street, by pizza parlor — correct

## Conflict 32 — naming (naming_collision_on_correct_subgraph)
- description: node 'clothing department' reachable at conflicting positions [(-3, -3, 0), (-2, -3, 0), (-1, -3, 0), (0, -3, 0)]
  - step 39: street, by department store --[south]--> clothing department — correct
  - step 40: clothing department --[southeast]--> cosmetics department — correct
  - step 49: clothing department --[southwest]--> hardware department — correct

## Conflict 33 — naming (name_hallucination)
- description: node 'clothing department, office building' reachable at conflicting positions [(-3, -3, 0), (-2, -3, 0), (-1, -3, 0), (0, -3, 0)]
  - step 52: hardware department --[northeast]--> clothing department, office building — dst_hallucinated
  - step 53: clothing department, office building --[north]--> street, by department store — src_hallucinated

## Conflict 34 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'hardware department' reachable at conflicting positions [(-4, -4, 0), (-3, -4, 0), (-2, -4, 0), (-1, -4, 0)]
  - step 49: clothing department --[southwest]--> hardware department — correct
  - step 52: hardware department --[northeast]--> clothing department, office building — dst_hallucinated

## Conflict 35 — naming (naming_collision_on_correct_subgraph)
- description: node 'cosmetics department' reachable at conflicting positions [(-2, -4, 0), (-1, -4, 0), (0, -4, 0), (1, -4, 0)]
  - step 40: clothing department --[southeast]--> cosmetics department — correct

## Conflict 36 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'street, near courthouse' reachable at conflicting positions [(-5, -2, 0), (-4, -2, 0), (-3, -2, 0), (-2, -2, 0), (-1, -2, 0)]
  - step 7: street, by pizza parlor --[west]--> street, near courthouse — correct
  - step 30: office building --[east]--> street, near courthouse — hallucinated_edge
  - step 8: street, near courthouse --[south]--> courthouse — correct
  - step 15: street, near courthouse --[west]--> downtown — correct

## Conflict 37 — naming (naming_collision_on_correct_subgraph)
- description: node 'courthouse' reachable at conflicting positions [(-5, -3, 0), (-4, -3, 0), (-3, -3, 0), (-2, -3, 0)]
  - step 8: street, near courthouse --[south]--> courthouse — correct

## Conflict 38 — naming (naming_collision_on_correct_subgraph)
- description: node 'downtown' reachable at conflicting positions [(-6, -2, 0), (-5, -2, 0), (-4, -2, 0), (-3, -2, 0), (-2, -2, 0)]
  - step 15: street, near courthouse --[west]--> downtown — correct
  - step 16: downtown --[west]--> office building — correct

## Conflict 39 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'office building' reachable at conflicting positions [(-6, -2, 0), (-5, -2, 0), (-4, -2, 0), (-3, -2, 0)]
  - step 16: downtown --[west]--> office building — correct
  - step 17: office building --[east]--> law office — hallucinated_edge
  - step 30: office building --[east]--> street, near courthouse — hallucinated_edge
  - step 60: office building --[north]--> second floor — dst_hallucinated

## Conflict 40 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'law office' reachable at conflicting positions [(-6, -2, 0), (-5, -2, 0), (-4, -2, 0), (-3, -2, 0), (-2, -2, 0)]
  - step 17: office building --[east]--> law office — hallucinated_edge
  - step 28: hangin' out --[west]--> law office — src_hallucinated
  - step 18: law office --[west]--> office building, second floor — correct

## Conflict 41 — naming (name_hallucination)
- description: node 'second floor' reachable at conflicting positions [(-6, -1, 0), (-5, -1, 0), (-4, -1, 0), (-3, -1, 0)]
  - step 60: office building --[north]--> second floor — dst_hallucinated
  - step 61: second floor --[south]--> office building, third floor — src_hallucinated

## Conflict 42 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'office building, third floor' reachable at conflicting positions [(-6, -2, 0), (-5, -2, 0), (-4, -2, 0), (-3, -2, 0)]
  - step 19: office building, second floor --[east]--> office building, third floor — hallucinated_edge
  - step 61: second floor --[south]--> office building, third floor — src_hallucinated
  - step 62: office building, third floor --[north]--> roof of office building — correct

## Conflict 43 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'office building, second floor' reachable at conflicting positions [(-7, -2, 0), (-6, -2, 0), (-5, -2, 0), (-4, -2, 0), (-3, -2, 0)]
  - step 18: law office --[west]--> office building, second floor — correct
  - step 19: office building, second floor --[east]--> office building, third floor — hallucinated_edge

## Conflict 44 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'roof of office building' reachable at conflicting positions [(-6, -1, 0), (-5, -1, 0), (-4, -1, 0), (-3, -1, 0)]
  - step 62: office building, third floor --[north]--> roof of office building — correct
  - step 65: roof of office building --[in]--> mayor's office — hallucinated_edge

## Conflict 45 — naming (name_hallucination)
- description: node "hangin' out" reachable at conflicting positions [(-4, -2, 0), (-3, -2, 0), (-2, -2, 0), (-1, -2, 0)]
  - step 28: hangin' out --[west]--> law office — src_hallucinated
