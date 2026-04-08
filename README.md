# Constructing Coherent Spatial Memory in LLM Agents through Graph Rectification


## Problem Statement

Current LLM approaches to spatial reasoning face significant limitations:
- **Lack of interpretability**: LLMs using context windows for text reading and spatial reasoning operate as black boxes
- **Context limitations**: Small context windows cannot handle long texts effectively
- **Need for external memory**: These limitations necessitate incremental graph construction to build spatial memory externally

However, incremental graph construction introduces new challenges:
- **Construction errors**: The graph building process may introduce errors
- **Delayed conflict manifestation**: Detectable structural conflicts in graphs may not indicate the root cause of errors
- **Error masking**: Errors can mask each other, causing conflicts to trigger with delays
<img width="1156" height="421" alt="image" src="https://github.com/user-attachments/assets/2f5e55fb-21c6-412e-ade9-ac2726a8aa67" />

## Our Solution

We propose a framework that enables LLMs to resolve conflicts and potential errors more efficiently through:

1. **Version Control System**: Tracks the evolution of the navigation graph, enabling backtracking to previous states when conflicts are detected
2. **Edge Dependency Detection**: Evaluates the structural importance and impact of edges, helping LLMs prioritize which edges to examine when resolving conflicts

This dual approach allows for both temporal analysis (when errors were introduced) and structural analysis (which edges are most critical to graph coherence).
<img width="1165" height="717" alt="image" src="https://github.com/user-attachments/assets/f1a036c2-b88a-4a80-9ad1-16ac626fbc7e" />

## Interactive map construction process

<img width="1240" height="687" alt="image" src="https://github.com/user-attachments/assets/817bd230-ad4d-4a90-a65b-34b892257d32" />

https://nitpicker55555.github.io/text_maze.github.io/

## Refined mango dataset without structural conflicts
https://huggingface.co/datasets/boboIloveyou/spatial_refined_mango/tree/main

## File Descriptions

The implementation is split into the same five modules as the paper:

### Core Pipeline

- **`map_slam_system.py`** — End-to-end pipeline. For each walkthrough step it runs (1) `NavigationGraph.process_step`, (2) `VersionControl.commit`, (3) `ConflictDetector.detect_all_conflicts`, and on detected conflicts (4) `ConflictLocalizer.localize_conflicts` followed by (5) `EdgeImpactScorer.score_edges`.

- **`navigation_graph.py`** — LLM-driven incremental graph construction. Tracks unit-distance positions for every node so the topological exclusivity check has the coordinates it needs, and auto-inserts reverse edges for cardinal directions.

- **`conflict_detector.py`** — Implements the three structural conflict types from paper Section 2.2:
  1. **Topological conflict** — two distinct nodes inferred to occupy the same unit-distance position (physical exclusivity violation).
  2. **Directional conflict** — a single source node has multiple outgoing edges with the same direction label.
  3. **Naming conflict** — the same canonical name is attached to two structurally distinct nodes.

- **`conflict_localizer.py`** — Implements the four-stage error localization pipeline from paper Section 2.3. Crucially, the LCA is computed on the **Reasoning History Tree** `T` (a DAG indexed by construction timestamps `tau(v)`), not on the spatial graph, so it remains well-defined when the spatial graph contains cycles. Uses paper Equation 2:

  ```
  LCA(pi_1, pi_2) = argmax_{v in pi_1 ∩ pi_2} tau(v)
  ```

- **`edge_impact_scorer.py`** — Implements the Edge Impact Score from paper Equation 1:

  ```
  score(e) = R_hat(e) + C_hat(e) + U_hat(e)
  ```

  where each factor is min-max normalized to [0, 1] across the candidate edges. The three factors are:
  - **Reachability** `R(e)`: number of nodes downstream-reachable from `e`
  - **Conflict count** `C(e)`: distinct conflicts involving `e`
  - **Usage** `U(e)`: walkthrough traversals containing `e`

- **`version_control.py`** — Versioned reasoning history (paper Section 2.4). Each commit `G_i` records `{Step_id, Commit, Trigger_event, Observation_id, Analysis}`. Exposes the three operations from the paper: `rollback_to(version)`, `recall_step(version)`, and `diff(G_i, G_j)`. Also exports the Reasoning History Tree `T` and timestamps `tau` for the localizer.

### Utilities

- **`mango_dataset.py`** — Loader for the refined MANGO walkthrough format.
- **`llm_agent.py`** — Thin OpenAI wrapper (`chat_single`).
- **`batch_run.py`** — CLI that runs the full pipeline on one or many games and writes per-game JSON reports plus an aggregate summary. Supports `--all` and `--workers N`.
- **`test_pipeline.py`** — End-to-end smoke test that replays the paper's Case Study B (long-range conflict, 9-node toy environment) without invoking any LLM and verifies the EIS ranking matches the paper.

## Experimental Results

### Main Results (GPT-4o as Model)

| Method | Avg. Loops | Repair Rate (%) | Accuracy (%) |
|--------|------------|-----------------|--------------|
| **Edge-Impact Ranking Only** | **6.39** | **75.21** | 44.69 |
| Version Control Only | 7.44 | 63.03 | 54.00 |
| **Version Control+Edge-Impact Ranking** | 8.20 | 68.91 | **54.88** |
| Baseline(GPT-4o) | 9.52 | 21.85 | 5.77 |

### Performance Across Different Models

| Model | Our Method ||| Baseline |||
|-------|---------|-------------|---------|---------|-------------|---------|
|       | **Loops** | **Repair (%)** | **Acc. (%)** | **Loops** | **Repair (%)** | **Acc. (%)** |
| GPT-4o | 8.20 | 68.91 | 54.88 | 9.52 | 21.85 | 5.77 |
| GPT-4.1 | 8.28 | 64.71 | 56.49 | 8.98 | 23.05 | 7.32 |
| GPT-4o-mini | 9.08 | 58.40 | 56.12 | 9.52 | 15.55 | 5.60 |
| Claude-Haiku | 6.98 | 44.31 | 61.76 | 9.33 | 17.15 | 6.67 |

## Usage

```python
from map_slam_system import MapSLAMSystem

# Initialize system
slam = MapSLAMSystem(data_dir="/path/to/refined_mango", model="gpt-4o")

# Process a game end-to-end
results = slam.process_game("zork1", max_steps=100)

# Save per-game JSON artifacts (graph, conflicts, version history,
# EIS-ranked candidate edges, and a text report)
slam.save_results("./output")
```

### Batch experiments

```bash
# Single game
python batch_run.py --games zork1 --data-dir /path/to/refined_mango

# All 53 games in parallel
python batch_run.py --all --workers 4 --data-dir /path/to/refined_mango

# Different LLM backbone, capped at 50 steps each
python batch_run.py --all --model gpt-4o-mini --max-steps 50
```

### End-to-end smoke test (no LLM required)

```bash
python test_pipeline.py
```

`test_pipeline.py` replays the paper's Case Study B in plain Python: it
introduces a directional error at step 5, watches the topological conflict
fire 15 steps later when Lab and Meeting Room land on the same coordinate,
runs LCA on the Reasoning History Tree, and prints the EIS ranking that
puts the true error at the top.
## Citation
```
@misc{zhang2025constructingcoherentspatialmemory,
      title={Constructing coherent spatial memory in LLM agents through graph rectification}, 
      author={Puzhen Zhang and Xuyang Chen and Yu Feng and Yuhan Jiang and Liqiu Meng},
      year={2025},
      eprint={2510.04195},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.04195}, 
}

```
