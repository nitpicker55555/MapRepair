# MapRepair: Navigation Graph Construction and Conflict Resolution

This repository contains the implementation of MapRepair, a system for constructing navigation graphs from text-based game walkthroughs and automatically detecting/repairing conflicts using version control and edge impact ranking techniques.

## Overview

MapRepair processes walkthrough data from the MANGO dataset to incrementally build navigation graphs, detect various types of conflicts (direction conflicts, topology conflicts, spatial inconsistencies), and apply repair strategies to improve graph quality.

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

- **`temporal_dependency_graph.py`**: Tracks the evolution history of navigation graphs. Maintains version snapshots at each step, enabling analysis of when and how conflicts were introduced.

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

### Key Findings

1. **Edge-Impact Ranking** achieves the highest repair rate (75.21%) and fastest convergence (6.39 loops)
2. **Version Control + Edge-Impact Ranking** achieves the best accuracy (54.88%) while maintaining good repair rate
3. All repair methods significantly outperform the baseline across all metrics
4. Model performance varies, with GPT-4o showing the best results overall

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

## Requirements

- Python 3.8+
- networkx
- openai
- tenacity
- Other dependencies in requirements.txt

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{maprepair2024,
  title={MapRepair: Navigation Graph Construction and Conflict Resolution in Text-Based Games},
  author={[Authors]},
  year={2024}
}
```