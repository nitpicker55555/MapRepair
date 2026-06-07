# Conflict analysis: karn

- LLM edges: 24
- GT edges: 33
- Conflicts: 51
- Type distribution: {'direction': 5, 'topology': 26, 'naming': 20}
- Root-cause distribution: {'name_hallucination': 4, 'real_vs_hallucinated': 1, 'all_hallucinated_edges': 1, 'name_hallucination_caused_overlap': 8, 'wrong_direction_caused_overlap': 11, 'overlap_mixed': 2, 'false_positive_overlap': 5, 'naming_collision_on_correct_subgraph': 11, 'real_name_corrupted_by_neighbour_error': 6, 'naming_mixed': 2}

## Conflict 1 — direction (name_hallucination)
- description: node 'rocky clearing' has multiple outgoing edges labelled 'northwest'
  - step None: rocky clearing --[northwest]--> mountain trail — dst_hallucinated
  - step None: rocky clearing --[northwest]--> mountain trail (carved steps to the south) — hallucinated_edge

## Conflict 2 — direction (real_vs_hallucinated)
- description: node 'ruins' has multiple outgoing edges labelled 'down'
  - step None: ruins --[down]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 30: ruins --[down]--> mountain trail (carved steps to the south) — correct

## Conflict 3 — direction (name_hallucination)
- description: node 'mountain trail (carved steps to the south)' has multiple outgoing edges labelled 'northeast'
  - step 31: mountain trail (carved steps to the south) --[northeast]--> mountain trail (cleft to northwest, down southwest, down east) — correct
  - step None: mountain trail (carved steps to the south) --[northeast]--> mountain trail — dst_hallucinated

## Conflict 4 — direction (all_hallucinated_edges)
- description: node 'mountain trail (carved steps to the south)' has multiple outgoing edges labelled 'west'
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge

## Conflict 5 — direction (name_hallucination)
- description: node 'narrow cleft' has multiple outgoing edges labelled 'southeast'
  - step None: narrow cleft --[southeast]--> mountain trail (cleft to northwest, down southwest, down east) — correct
  - step 37: narrow cleft --[southeast]--> mountain trail — dst_hallucinated

## Conflict 6 — topology (name_hallucination_caused_overlap)
- description: position (0, 0, 0) occupied by multiple rooms ['bottom of cliff', 'narrow cleft']
  - step 33: narrow cleft --[west]--> bottom of cliff — correct
  - step 32: mountain trail (cleft to northwest, down southwest, down east) --[northwest]--> narrow cleft — correct
  - step 37: narrow cleft --[southeast]--> mountain trail — dst_hallucinated

## Conflict 7 — topology (name_hallucination_caused_overlap)
- description: position (1, 1, 0) occupied by multiple rooms ['bottom of cliff', 'narrow cleft']
  - step 33: narrow cleft --[west]--> bottom of cliff — correct
  - step 32: mountain trail (cleft to northwest, down southwest, down east) --[northwest]--> narrow cleft — correct
  - step 37: narrow cleft --[southeast]--> mountain trail — dst_hallucinated

## Conflict 8 — topology (name_hallucination_caused_overlap)
- description: position (1, 0, 0) occupied by multiple rooms ['mountain trail (up west, down east)', 'narrow cleft']
  - step 60: mountain trail (blackened tree beside trail) --[west]--> mountain trail (up west, down east) — correct
  - step 32: mountain trail (cleft to northwest, down southwest, down east) --[northwest]--> narrow cleft — correct
  - step 33: narrow cleft --[west]--> bottom of cliff — correct
  - step 37: narrow cleft --[southeast]--> mountain trail — dst_hallucinated

## Conflict 9 — topology (wrong_direction_caused_overlap)
- description: position (2, -1, 0) occupied by multiple rooms ['mountain trail', 'mountain trail (carved steps to the south)', 'mountain trail (cleft to northwest, down southwest, down east)', 'mountain trail (up northeast, clearing to southeast, stoney trail leads west)']
  - step 37: narrow cleft --[southeast]--> mountain trail — dst_hallucinated
  - step 40: mountain trail (blackened tree beside trail) --[southwest]--> mountain trail — dst_hallucinated
  - step 38: mountain trail --[southwest]--> mountain trail (carved steps to the south) — src_hallucinated
  - step 41: mountain trail --[southeast]--> rocky clearing — src_hallucinated
  - step 30: ruins --[down]--> mountain trail (carved steps to the south) — correct
  - step 31: mountain trail (carved steps to the south) --[northeast]--> mountain trail (cleft to northwest, down southwest, down east) — correct
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 32: mountain trail (cleft to northwest, down southwest, down east) --[northwest]--> narrow cleft — correct
  - step 25: mountain trail (up east, down southwest) --[east]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — wrong_direction (LLM: 'east', GT: 'southwest')
  - step 26: mountain trail (up northeast, clearing to southeast, stoney trail leads west) --[up]--> ruins — hallucinated_edge

## Conflict 10 — topology (wrong_direction_caused_overlap)
- description: position (1, -1, 0) occupied by multiple rooms ['mountain trail', 'mountain trail (blackened tree beside trail)', 'mountain trail (carved steps to the south)', 'mountain trail (cleft to northwest, down southwest, down east)', 'mountain trail (up northeast, clearing to southeast, stoney trail leads west)']
  - step 37: narrow cleft --[southeast]--> mountain trail — dst_hallucinated
  - step 40: mountain trail (blackened tree beside trail) --[southwest]--> mountain trail — dst_hallucinated
  - step 38: mountain trail --[southwest]--> mountain trail (carved steps to the south) — src_hallucinated
  - step 41: mountain trail --[southeast]--> rocky clearing — src_hallucinated
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 56: plain --[northwest]--> mountain trail (blackened tree beside trail) — correct
  - step 60: mountain trail (blackened tree beside trail) --[west]--> mountain trail (up west, down east) — correct
  - step 30: ruins --[down]--> mountain trail (carved steps to the south) — correct
  - step 31: mountain trail (carved steps to the south) --[northeast]--> mountain trail (cleft to northwest, down southwest, down east) — correct
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 32: mountain trail (cleft to northwest, down southwest, down east) --[northwest]--> narrow cleft — correct
  - step 25: mountain trail (up east, down southwest) --[east]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — wrong_direction (LLM: 'east', GT: 'southwest')
  - step 26: mountain trail (up northeast, clearing to southeast, stoney trail leads west) --[up]--> ruins — hallucinated_edge

## Conflict 11 — topology (name_hallucination_caused_overlap)
- description: position (3, 0, 0) occupied by multiple rooms ['mountain trail', 'mountain trail (blackened tree beside trail)', 'mountain trail (carved steps to the south)', 'mountain trail (cleft to northwest, down southwest, down east)']
  - step 37: narrow cleft --[southeast]--> mountain trail — dst_hallucinated
  - step 40: mountain trail (blackened tree beside trail) --[southwest]--> mountain trail — dst_hallucinated
  - step 38: mountain trail --[southwest]--> mountain trail (carved steps to the south) — src_hallucinated
  - step 41: mountain trail --[southeast]--> rocky clearing — src_hallucinated
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 56: plain --[northwest]--> mountain trail (blackened tree beside trail) — correct
  - step 60: mountain trail (blackened tree beside trail) --[west]--> mountain trail (up west, down east) — correct
  - step 30: ruins --[down]--> mountain trail (carved steps to the south) — correct
  - step 31: mountain trail (carved steps to the south) --[northeast]--> mountain trail (cleft to northwest, down southwest, down east) — correct
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 32: mountain trail (cleft to northwest, down southwest, down east) --[northwest]--> narrow cleft — correct

## Conflict 12 — topology (name_hallucination_caused_overlap)
- description: position (2, 0, 0) occupied by multiple rooms ['mountain trail', 'mountain trail (blackened tree beside trail)', 'mountain trail (carved steps to the south)', 'mountain trail (cleft to northwest, down southwest, down east)', 'mountain trail (up west, down east)']
  - step 37: narrow cleft --[southeast]--> mountain trail — dst_hallucinated
  - step 40: mountain trail (blackened tree beside trail) --[southwest]--> mountain trail — dst_hallucinated
  - step 38: mountain trail --[southwest]--> mountain trail (carved steps to the south) — src_hallucinated
  - step 41: mountain trail --[southeast]--> rocky clearing — src_hallucinated
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 56: plain --[northwest]--> mountain trail (blackened tree beside trail) — correct
  - step 60: mountain trail (blackened tree beside trail) --[west]--> mountain trail (up west, down east) — correct
  - step 30: ruins --[down]--> mountain trail (carved steps to the south) — correct
  - step 31: mountain trail (carved steps to the south) --[northeast]--> mountain trail (cleft to northwest, down southwest, down east) — correct
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 32: mountain trail (cleft to northwest, down southwest, down east) --[northwest]--> narrow cleft — correct

## Conflict 13 — topology (wrong_direction_caused_overlap)
- description: position (-1, -2, 0) occupied by multiple rooms ['mountain trail', 'mountain trail (blackened tree beside trail)', 'mountain trail (up northeast, clearing to southeast, stoney trail leads west)', 'rocky clearing']
  - step 37: narrow cleft --[southeast]--> mountain trail — dst_hallucinated
  - step 40: mountain trail (blackened tree beside trail) --[southwest]--> mountain trail — dst_hallucinated
  - step 38: mountain trail --[southwest]--> mountain trail (carved steps to the south) — src_hallucinated
  - step 41: mountain trail --[southeast]--> rocky clearing — src_hallucinated
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 56: plain --[northwest]--> mountain trail (blackened tree beside trail) — correct
  - step 60: mountain trail (blackened tree beside trail) --[west]--> mountain trail (up west, down east) — correct
  - step 25: mountain trail (up east, down southwest) --[east]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — wrong_direction (LLM: 'east', GT: 'southwest')
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 26: mountain trail (up northeast, clearing to southeast, stoney trail leads west) --[up]--> ruins — hallucinated_edge
  - step 23: console room --[northwest]--> rocky clearing — hallucinated_edge
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 24: rocky clearing --[northeast]--> mountain trail (up east, down southwest) — hallucinated_edge
  - step 42: rocky clearing --[east]--> rocky trail (clearing to west) — correct

## Conflict 14 — topology (wrong_direction_caused_overlap)
- description: position (0, -2, 0) occupied by multiple rooms ['mountain trail', 'mountain trail (blackened tree beside trail)', 'mountain trail (carved steps to the south)', 'mountain trail (up northeast, clearing to southeast, stoney trail leads west)']
  - step 37: narrow cleft --[southeast]--> mountain trail — dst_hallucinated
  - step 40: mountain trail (blackened tree beside trail) --[southwest]--> mountain trail — dst_hallucinated
  - step 38: mountain trail --[southwest]--> mountain trail (carved steps to the south) — src_hallucinated
  - step 41: mountain trail --[southeast]--> rocky clearing — src_hallucinated
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 56: plain --[northwest]--> mountain trail (blackened tree beside trail) — correct
  - step 60: mountain trail (blackened tree beside trail) --[west]--> mountain trail (up west, down east) — correct
  - step 30: ruins --[down]--> mountain trail (carved steps to the south) — correct
  - step 31: mountain trail (carved steps to the south) --[northeast]--> mountain trail (cleft to northwest, down southwest, down east) — correct
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 25: mountain trail (up east, down southwest) --[east]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — wrong_direction (LLM: 'east', GT: 'southwest')
  - step 26: mountain trail (up northeast, clearing to southeast, stoney trail leads west) --[up]--> ruins — hallucinated_edge

## Conflict 15 — topology (wrong_direction_caused_overlap)
- description: position (3, -2, 0) occupied by multiple rooms ['mountain trail (carved steps to the south)', 'mountain trail (up northeast, clearing to southeast, stoney trail leads west)', 'rocky clearing', 'rocky trail (clearing to west)']
  - step 30: ruins --[down]--> mountain trail (carved steps to the south) — correct
  - step 38: mountain trail --[southwest]--> mountain trail (carved steps to the south) — src_hallucinated
  - step 31: mountain trail (carved steps to the south) --[northeast]--> mountain trail (cleft to northwest, down southwest, down east) — correct
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 25: mountain trail (up east, down southwest) --[east]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — wrong_direction (LLM: 'east', GT: 'southwest')
  - step 26: mountain trail (up northeast, clearing to southeast, stoney trail leads west) --[up]--> ruins — hallucinated_edge
  - step 23: console room --[northwest]--> rocky clearing — hallucinated_edge
  - step 41: mountain trail --[southeast]--> rocky clearing — src_hallucinated
  - step 24: rocky clearing --[northeast]--> mountain trail (up east, down southwest) — hallucinated_edge
  - step 42: rocky clearing --[east]--> rocky trail (clearing to west) — correct
  - step 43: rocky trail (clearing to west) --[southeast]--> rocky trail (trail widens eastward and narrows northwest) — correct

## Conflict 16 — topology (wrong_direction_caused_overlap)
- description: position (4, 0, 0) occupied by multiple rooms ['mountain trail (carved steps to the south)', 'mountain trail (up east, down southwest)']
  - step 30: ruins --[down]--> mountain trail (carved steps to the south) — correct
  - step 38: mountain trail --[southwest]--> mountain trail (carved steps to the south) — src_hallucinated
  - step 31: mountain trail (carved steps to the south) --[northeast]--> mountain trail (cleft to northwest, down southwest, down east) — correct
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 24: rocky clearing --[northeast]--> mountain trail (up east, down southwest) — hallucinated_edge
  - step 25: mountain trail (up east, down southwest) --[east]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — wrong_direction (LLM: 'east', GT: 'southwest')

## Conflict 17 — topology (name_hallucination_caused_overlap)
- description: position (4, -2, 0) occupied by multiple rooms ['console room', 'mountain trail (carved steps to the south)', 'rocky trail (clearing to west)']
  - step 6: console room --[east]--> corridor — correct
  - step 23: console room --[northwest]--> rocky clearing — hallucinated_edge
  - step 30: ruins --[down]--> mountain trail (carved steps to the south) — correct
  - step 38: mountain trail --[southwest]--> mountain trail (carved steps to the south) — src_hallucinated
  - step 31: mountain trail (carved steps to the south) --[northeast]--> mountain trail (cleft to northwest, down southwest, down east) — correct
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 42: rocky clearing --[east]--> rocky trail (clearing to west) — correct
  - step 43: rocky trail (clearing to west) --[southeast]--> rocky trail (trail widens eastward and narrows northwest) — correct

## Conflict 18 — topology (name_hallucination_caused_overlap)
- description: position (1, -2, 0) occupied by multiple rooms ['mountain trail (carved steps to the south)', 'plain']
  - step 30: ruins --[down]--> mountain trail (carved steps to the south) — correct
  - step 38: mountain trail --[southwest]--> mountain trail (carved steps to the south) — src_hallucinated
  - step 31: mountain trail (carved steps to the south) --[northeast]--> mountain trail (cleft to northwest, down southwest, down east) — correct
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 53: on top of the saucer --[down]--> plain — correct
  - step 56: plain --[northwest]--> mountain trail (blackened tree beside trail) — correct

## Conflict 19 — topology (wrong_direction_caused_overlap)
- description: position (0, -1, 0) occupied by multiple rooms ['mountain trail (blackened tree beside trail)', 'mountain trail (up east, down southwest)', 'mountain trail (up northeast, clearing to southeast, stoney trail leads west)', 'mountain trail (up west, down east)']
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 56: plain --[northwest]--> mountain trail (blackened tree beside trail) — correct
  - step 40: mountain trail (blackened tree beside trail) --[southwest]--> mountain trail — dst_hallucinated
  - step 60: mountain trail (blackened tree beside trail) --[west]--> mountain trail (up west, down east) — correct
  - step 24: rocky clearing --[northeast]--> mountain trail (up east, down southwest) — hallucinated_edge
  - step 25: mountain trail (up east, down southwest) --[east]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — wrong_direction (LLM: 'east', GT: 'southwest')
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 26: mountain trail (up northeast, clearing to southeast, stoney trail leads west) --[up]--> ruins — hallucinated_edge

## Conflict 20 — topology (wrong_direction_caused_overlap)
- description: position (2, -2, 0) occupied by multiple rooms ['mountain trail (up east, down southwest)', 'plain', 'rocky clearing']
  - step 24: rocky clearing --[northeast]--> mountain trail (up east, down southwest) — hallucinated_edge
  - step 25: mountain trail (up east, down southwest) --[east]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — wrong_direction (LLM: 'east', GT: 'southwest')
  - step 53: on top of the saucer --[down]--> plain — correct
  - step 56: plain --[northwest]--> mountain trail (blackened tree beside trail) — correct
  - step 23: console room --[northwest]--> rocky clearing — hallucinated_edge
  - step 41: mountain trail --[southeast]--> rocky clearing — src_hallucinated
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 42: rocky clearing --[east]--> rocky trail (clearing to west) — correct

## Conflict 21 — topology (name_hallucination_caused_overlap)
- description: position (2, -3, 0) occupied by multiple rooms ['rocky clearing', 'rocky trail (clearing to west)']
  - step 23: console room --[northwest]--> rocky clearing — hallucinated_edge
  - step 41: mountain trail --[southeast]--> rocky clearing — src_hallucinated
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 24: rocky clearing --[northeast]--> mountain trail (up east, down southwest) — hallucinated_edge
  - step 42: rocky clearing --[east]--> rocky trail (clearing to west) — correct
  - step 43: rocky trail (clearing to west) --[southeast]--> rocky trail (trail widens eastward and narrows northwest) — correct

## Conflict 22 — topology (wrong_direction_caused_overlap)
- description: position (3, -1, 0) occupied by multiple rooms ['mountain trail (up east, down southwest)', 'plain', 'rocky clearing']
  - step 24: rocky clearing --[northeast]--> mountain trail (up east, down southwest) — hallucinated_edge
  - step 25: mountain trail (up east, down southwest) --[east]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — wrong_direction (LLM: 'east', GT: 'southwest')
  - step 53: on top of the saucer --[down]--> plain — correct
  - step 56: plain --[northwest]--> mountain trail (blackened tree beside trail) — correct
  - step 23: console room --[northwest]--> rocky clearing — hallucinated_edge
  - step 41: mountain trail --[southeast]--> rocky clearing — src_hallucinated
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 42: rocky clearing --[east]--> rocky trail (clearing to west) — correct

## Conflict 23 — topology (wrong_direction_caused_overlap)
- description: position (4, -1, 0) occupied by multiple rooms ['mountain trail (up east, down southwest)', 'mountain trail (up northeast, clearing to southeast, stoney trail leads west)', 'plain', 'rocky clearing', 'rocky trail (clearing to west)']
  - step 24: rocky clearing --[northeast]--> mountain trail (up east, down southwest) — hallucinated_edge
  - step 25: mountain trail (up east, down southwest) --[east]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — wrong_direction (LLM: 'east', GT: 'southwest')
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 26: mountain trail (up northeast, clearing to southeast, stoney trail leads west) --[up]--> ruins — hallucinated_edge
  - step 53: on top of the saucer --[down]--> plain — correct
  - step 56: plain --[northwest]--> mountain trail (blackened tree beside trail) — correct
  - step 23: console room --[northwest]--> rocky clearing — hallucinated_edge
  - step 41: mountain trail --[southeast]--> rocky clearing — src_hallucinated
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 42: rocky clearing --[east]--> rocky trail (clearing to west) — correct
  - step 43: rocky trail (clearing to west) --[southeast]--> rocky trail (trail widens eastward and narrows northwest) — correct

## Conflict 24 — topology (overlap_mixed)
- description: position (4, -3, 0) occupied by multiple rooms ['console room', 'corridor', 'ledge', 'rocky trail (trail widens eastward and narrows northwest)']
  - step 6: console room --[east]--> corridor — correct
  - step 23: console room --[northwest]--> rocky clearing — hallucinated_edge
  - step 8: corridor --[south]--> workshop — correct
  - step 45: lakeside --[north]--> ledge — correct
  - step 43: rocky trail (clearing to west) --[southeast]--> rocky trail (trail widens eastward and narrows northwest) — correct
  - step 44: rocky trail (trail widens eastward and narrows northwest) --[east]--> lakeside — correct

## Conflict 25 — topology (wrong_direction_caused_overlap)
- description: position (-1, -1, 0) occupied by multiple rooms ['mountain trail (up east, down southwest)', 'mountain trail (up west, down east)']
  - step 24: rocky clearing --[northeast]--> mountain trail (up east, down southwest) — hallucinated_edge
  - step 25: mountain trail (up east, down southwest) --[east]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — wrong_direction (LLM: 'east', GT: 'southwest')
  - step 60: mountain trail (blackened tree beside trail) --[west]--> mountain trail (up west, down east) — correct

## Conflict 26 — topology (overlap_mixed)
- description: position (1, -2, 1) occupied by multiple rooms ['on top of the saucer', 'ruins']
  - step 53: on top of the saucer --[down]--> plain — correct
  - step 26: mountain trail (up northeast, clearing to southeast, stoney trail leads west) --[up]--> ruins — hallucinated_edge
  - step 30: ruins --[down]--> mountain trail (carved steps to the south) — correct

## Conflict 27 — topology (false_positive_overlap)
- description: position (5, -3, 0) occupied by multiple rooms ['corridor', 'lakeside', 'rocky trail (trail widens eastward and narrows northwest)', 'workshop']
  - step 6: console room --[east]--> corridor — correct
  - step 8: corridor --[south]--> workshop — correct
  - step 44: rocky trail (trail widens eastward and narrows northwest) --[east]--> lakeside — correct
  - step 45: lakeside --[north]--> ledge — correct
  - step 43: rocky trail (clearing to west) --[southeast]--> rocky trail (trail widens eastward and narrows northwest) — correct

## Conflict 28 — topology (false_positive_overlap)
- description: position (5, -2, 0) occupied by multiple rooms ['corridor', 'ledge', 'rocky trail (trail widens eastward and narrows northwest)']
  - step 6: console room --[east]--> corridor — correct
  - step 8: corridor --[south]--> workshop — correct
  - step 45: lakeside --[north]--> ledge — correct
  - step 43: rocky trail (clearing to west) --[southeast]--> rocky trail (trail widens eastward and narrows northwest) — correct
  - step 44: rocky trail (trail widens eastward and narrows northwest) --[east]--> lakeside — correct

## Conflict 29 — topology (false_positive_overlap)
- description: position (3, -4, 0) occupied by multiple rooms ['corridor', 'rocky trail (trail widens eastward and narrows northwest)']
  - step 6: console room --[east]--> corridor — correct
  - step 8: corridor --[south]--> workshop — correct
  - step 43: rocky trail (clearing to west) --[southeast]--> rocky trail (trail widens eastward and narrows northwest) — correct
  - step 44: rocky trail (trail widens eastward and narrows northwest) --[east]--> lakeside — correct

## Conflict 30 — topology (false_positive_overlap)
- description: position (6, -2, 0) occupied by multiple rooms ['lakeside', 'ledge']
  - step 44: rocky trail (trail widens eastward and narrows northwest) --[east]--> lakeside — correct
  - step 45: lakeside --[north]--> ledge — correct

## Conflict 31 — topology (false_positive_overlap)
- description: position (4, -4, 0) occupied by multiple rooms ['lakeside', 'workshop']
  - step 44: rocky trail (trail widens eastward and narrows northwest) --[east]--> lakeside — correct
  - step 45: lakeside --[north]--> ledge — correct
  - step 8: corridor --[south]--> workshop — correct

## Conflict 32 — naming (naming_collision_on_correct_subgraph)
- description: node 'bottom of cliff' reachable at conflicting positions [(-1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)]
  - step 33: narrow cleft --[west]--> bottom of cliff — correct

## Conflict 33 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'narrow cleft' reachable at conflicting positions [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]
  - step 32: mountain trail (cleft to northwest, down southwest, down east) --[northwest]--> narrow cleft — correct
  - step 33: narrow cleft --[west]--> bottom of cliff — correct
  - step 37: narrow cleft --[southeast]--> mountain trail — dst_hallucinated

## Conflict 34 — naming (naming_collision_on_correct_subgraph)
- description: node 'mountain trail (cleft to northwest, down southwest, down east)' reachable at conflicting positions [(1, -1, 0), (2, -1, 0), (2, 0, 0), (3, 0, 0)]
  - step 31: mountain trail (carved steps to the south) --[northeast]--> mountain trail (cleft to northwest, down southwest, down east) — correct
  - step 32: mountain trail (cleft to northwest, down southwest, down east) --[northwest]--> narrow cleft — correct

## Conflict 35 — naming (name_hallucination)
- description: node 'mountain trail' reachable at conflicting positions [(-1, -2, 0), (0, -2, 0), (1, -1, 0), (2, -1, 0), (2, 0, 0), (3, 0, 0)]
  - step 37: narrow cleft --[southeast]--> mountain trail — dst_hallucinated
  - step 40: mountain trail (blackened tree beside trail) --[southwest]--> mountain trail — dst_hallucinated
  - step 38: mountain trail --[southwest]--> mountain trail (carved steps to the south) — src_hallucinated
  - step 41: mountain trail --[southeast]--> rocky clearing — src_hallucinated

## Conflict 36 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'mountain trail (carved steps to the south)' reachable at conflicting positions [(0, -2, 0), (1, -2, 0), (1, -1, 0), (2, -1, 0), (2, 0, 0), (3, -2, 0), (3, 0, 0), (4, -2, 0), (4, 0, 0)]
  - step 30: ruins --[down]--> mountain trail (carved steps to the south) — correct
  - step 38: mountain trail --[southwest]--> mountain trail (carved steps to the south) — src_hallucinated
  - step 31: mountain trail (carved steps to the south) --[northeast]--> mountain trail (cleft to northwest, down southwest, down east) — correct
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge

## Conflict 37 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'mountain trail (blackened tree beside trail)' reachable at conflicting positions [(-1, -2, 0), (0, -2, 0), (0, -1, 0), (1, -1, 0), (2, 0, 0), (3, 0, 0), (3, 1, 0), (4, 1, 0)]
  - step 39: mountain trail (carved steps to the south) --[west]--> mountain trail (blackened tree beside trail) — hallucinated_edge
  - step 56: plain --[northwest]--> mountain trail (blackened tree beside trail) — correct
  - step 40: mountain trail (blackened tree beside trail) --[southwest]--> mountain trail — dst_hallucinated
  - step 60: mountain trail (blackened tree beside trail) --[west]--> mountain trail (up west, down east) — correct

## Conflict 38 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'rocky clearing' reachable at conflicting positions [(-1, -2, 0), (1, -3, 0), (2, -3, 0), (2, -2, 0), (3, -2, 0), (3, -1, 0), (4, -1, 0)]
  - step 23: console room --[northwest]--> rocky clearing — hallucinated_edge
  - step 41: mountain trail --[southeast]--> rocky clearing — src_hallucinated
  - step 65: mountain trail (carved steps to the south) --[southeast]--> rocky clearing — hallucinated_edge
  - step 24: rocky clearing --[northeast]--> mountain trail (up east, down southwest) — hallucinated_edge
  - step 42: rocky clearing --[east]--> rocky trail (clearing to west) — correct

## Conflict 39 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'console room' reachable at conflicting positions [(2, -4, 0), (3, -3, 0), (4, -3, 0), (4, -2, 0)]
  - step 6: console room --[east]--> corridor — correct
  - step 23: console room --[northwest]--> rocky clearing — hallucinated_edge

## Conflict 40 — naming (naming_mixed)
- description: node 'mountain trail (up east, down southwest)' reachable at conflicting positions [(-2, -2, 0), (-1, -1, 0), (0, -1, 0), (2, -2, 0), (3, -1, 0), (4, -1, 0), (4, 0, 0)]
  - step 24: rocky clearing --[northeast]--> mountain trail (up east, down southwest) — hallucinated_edge
  - step 25: mountain trail (up east, down southwest) --[east]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — wrong_direction (LLM: 'east', GT: 'southwest')

## Conflict 41 — naming (naming_collision_on_correct_subgraph)
- description: node 'rocky trail (clearing to west)' reachable at conflicting positions [(2, -3, 0), (3, -2, 0), (4, -2, 0), (4, -1, 0)]
  - step 42: rocky clearing --[east]--> rocky trail (clearing to west) — correct
  - step 43: rocky trail (clearing to west) --[southeast]--> rocky trail (trail widens eastward and narrows northwest) — correct

## Conflict 42 — naming (real_name_corrupted_by_neighbour_error)
- description: node 'ruins' reachable at conflicting positions [(-1, -2, 1), (0, -2, 1), (0, -1, 1), (1, -2, 1), (1, -1, 1), (2, -1, 1), (3, -2, 1)]
  - step 26: mountain trail (up northeast, clearing to southeast, stoney trail leads west) --[up]--> ruins — hallucinated_edge
  - step 30: ruins --[down]--> mountain trail (carved steps to the south) — correct

## Conflict 43 — naming (naming_mixed)
- description: node 'mountain trail (up northeast, clearing to southeast, stoney trail leads west)' reachable at conflicting positions [(-1, -2, 0), (0, -2, 0), (0, -1, 0), (1, -1, 0), (2, -1, 0), (3, -2, 0), (4, -1, 0), (5, -1, 0)]
  - step 25: mountain trail (up east, down southwest) --[east]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — wrong_direction (LLM: 'east', GT: 'southwest')
  - step 63: mountain trail (carved steps to the south) --[west]--> mountain trail (up northeast, clearing to southeast, stoney trail leads west) — hallucinated_edge
  - step 26: mountain trail (up northeast, clearing to southeast, stoney trail leads west) --[up]--> ruins — hallucinated_edge

## Conflict 44 — naming (naming_collision_on_correct_subgraph)
- description: node 'rocky trail (trail widens eastward and narrows northwest)' reachable at conflicting positions [(3, -4, 0), (4, -3, 0), (5, -3, 0), (5, -2, 0)]
  - step 43: rocky trail (clearing to west) --[southeast]--> rocky trail (trail widens eastward and narrows northwest) — correct
  - step 44: rocky trail (trail widens eastward and narrows northwest) --[east]--> lakeside — correct

## Conflict 45 — naming (naming_collision_on_correct_subgraph)
- description: node 'lakeside' reachable at conflicting positions [(4, -4, 0), (5, -3, 0), (6, -3, 0), (6, -2, 0)]
  - step 44: rocky trail (trail widens eastward and narrows northwest) --[east]--> lakeside — correct
  - step 45: lakeside --[north]--> ledge — correct

## Conflict 46 — naming (naming_collision_on_correct_subgraph)
- description: node 'ledge' reachable at conflicting positions [(4, -3, 0), (5, -2, 0), (6, -2, 0), (6, -1, 0)]
  - step 45: lakeside --[north]--> ledge — correct

## Conflict 47 — naming (naming_collision_on_correct_subgraph)
- description: node 'plain' reachable at conflicting positions [(1, -2, 0), (2, -2, 0), (3, -1, 0), (4, -1, 0)]
  - step 53: on top of the saucer --[down]--> plain — correct
  - step 56: plain --[northwest]--> mountain trail (blackened tree beside trail) — correct

## Conflict 48 — naming (naming_collision_on_correct_subgraph)
- description: node 'mountain trail (up west, down east)' reachable at conflicting positions [(-1, -1, 0), (0, -1, 0), (1, 0, 0), (2, 0, 0)]
  - step 60: mountain trail (blackened tree beside trail) --[west]--> mountain trail (up west, down east) — correct

## Conflict 49 — naming (naming_collision_on_correct_subgraph)
- description: node 'on top of the saucer' reachable at conflicting positions [(1, -2, 1), (2, -2, 1), (3, -1, 1), (4, -1, 1)]
  - step 53: on top of the saucer --[down]--> plain — correct

## Conflict 50 — naming (naming_collision_on_correct_subgraph)
- description: node 'corridor' reachable at conflicting positions [(3, -4, 0), (4, -3, 0), (5, -3, 0), (5, -2, 0)]
  - step 6: console room --[east]--> corridor — correct
  - step 8: corridor --[south]--> workshop — correct

## Conflict 51 — naming (naming_collision_on_correct_subgraph)
- description: node 'workshop' reachable at conflicting positions [(3, -5, 0), (4, -4, 0), (5, -4, 0), (5, -3, 0)]
  - step 8: corridor --[south]--> workshop — correct
