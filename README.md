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

## Our Solution

We propose a framework that enables LLMs to resolve conflicts and potential errors more efficiently through:

1. **Version Control System**: Tracks the evolution of the navigation graph, enabling backtracking to previous states when conflicts are detected
2. **Edge Dependency Detection**: Evaluates the structural importance and impact of edges, helping LLMs prioritize which edges to examine when resolving conflicts

This dual approach allows for both temporal analysis (when errors were introduced) and structural analysis (which edges are most critical to graph coherence).

## Interactive map construction process

<img width="1240" height="687" alt="image" src="https://github.com/user-attachments/assets/817bd230-ad4d-4a90-a65b-34b892257d32" />

https://nitpicker55555.github.io/text_maze.github.io/

## File Descriptions

### Core System Components

- **`map_slam_system.py`**: Main system integrating all components. Orchestrates the entire pipeline from data loading through graph construction, conflict detection, localization, and repair.

- **`navigation_graph.py`**: Handles incremental navigation graph construction from walkthrough steps. Uses LLM-based analysis to extract location nodes and movement edges from action-observation sequences.

- **`conflict_detector.py`**: Detects various types of conflicts in the navigation graph:
  - Direction conflicts (same direction from a node leads to multiple locations)
  - Topology conflicts (unreachable nodes, disconnected components)
  - Reverse edge conflicts (bidirectional edges with inconsistent directions)
  - Spatial consistency conflicts

### Analysis and Repair Components

- **`version_control.py`**: Implements the version control system that tracks the evolution history of navigation graphs. Maintains version snapshots at each step, enabling backtracking and analysis of when conflicts were introduced.

- **`edge_impact_scorer.py`**: Evaluates the importance and impact of edges based on:
  - Structural importance (bridge edges, connectivity)
  - Usage frequency in walkthroughs
  - Conflict involvement
  - Centrality measures (betweenness, PageRank)

- **`conflict_localizer.py`**: Localizes conflicts by analyzing paths between conflicting nodes and identifying candidate edges that may be erroneous.

### Utilities

- **`mango_dataset.py`**: Interface for loading and parsing MANGO dataset walkthrough files. Handles step-by-step action and observation extraction.

- **`llm_agent.py`**: Wrapper for LLM API calls (OpenAI/Claude) used in navigation information extraction.

- **`batch_run.py`**: Batch processing script for running experiments across multiple games with different repair strategies and models.

## Experimental Results

### Main Results (GPT-4o as Navigation Model)

| Method | Avg. Loops | Repair Rate (%) | Accuracy (%) |
|--------|------------|-----------------|--------------|
| Edge-Impact Ranking Only | **6.39** | **75.21** | 44.69 |
| Version Control Only | 7.44 | 63.03 | 54.00 |
| **Version Control + Edge-Impact Ranking** | 8.20 | 68.91 | **54.88** |
| Baseline (GPT-4o) | 9.52 | 21.85 | 5.77 |

### Performance Across Different Models

#### Our Method (Version Control + Edge-Impact Ranking)

| Model | Avg. Loops | Repair Rate (%) | Accuracy (%) |
|-------|------------|-----------------|--------------|
| GPT-4.1 | 7.88 | 64.89 | 36.64 |
| GPT-4o | 8.20 | 68.91 | 54.88 |
| GPT-4o-mini | 9.02 | 58.40 | 32.82 |
| Claude-Haiku | 6.98 | 44.31 | 24.76 |

#### Baseline (No Repair)

| Model | Avg. Loops | Repair Rate (%) | Accuracy (%) |
|-------|------------|-----------------|--------------|
| GPT-4.1 | 8.98 | 23.05 | 7.32 |
| GPT-4o | 9.52 | 21.85 | 5.77 |
| GPT-4o-mini | 9.52 | 15.55 | 5.60 |
| Claude-Haiku | 9.33 | 17.15 | 6.67 |


## Usage

```python
from map_slam_system import MapSLAMSystem

# Initialize system
slam = MapSLAMSystem(data_dir="/path/to/mango/data", model="gpt-4o")

# Process a game
results = slam.process_game("game_name", max_steps=100)

# Save results
slam.save_results("./output")
```

