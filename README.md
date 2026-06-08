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

## Refined MANGO dataset without structural conflicts
https://huggingface.co/datasets/boboIloveyou/spatial_refined_mango/tree/main

## File Descriptions

The root-level files are a minimal reference implementation, organized as the same five modules as in the paper:

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

## Paper Data Sources

Every quantitative claim in the paper is produced by a script under `experiments/` and saved as a raw JSON under `results/`. This repo is self-contained for all paper data except (i) the *Dream of the Red Chamber* deployment (Table 4) and (ii) the hand-crafted TC1–TC6 scenarios (Section 4.4 / Appendix B); both live in the companion repo https://github.com/nitpicker55555/spatial_memory.

Setup once:

```bash
pip install -e .        # or: pip install -r requirements.txt
cp .env.example .env    # add OPENAI_API_KEY / proxy keys
export PYTHONPATH=src
```

### Paper Table 1 — Per-component synthetic ablation

> gpt-4.1 as repair LLM. Random graphs of size 60 with directly injected topology or direction errors at densities 4 and 8. Each cell aggregates n=20 independent seeds; 95% Wilson confidence intervals. **Base.** = unscaffolded LLM, **EI** = Edge-Impact Ranking, **VC** = Version Control, **VC+EI** = combined.

| Conflict | Errors | Base. | EI | VC | VC+EI |
|----------|-------:|------:|---:|---:|------:|
| Topology  | 4 | 50.0 | **95.0** | 50.0 | 50.0 |
| Topology  | 8 | 30.0 | **60.0** | 25.0 | 40.0 |
| Direction | 4 | 70.0 | **75.0** | 55.0 | 55.0 |
| Direction | 8 | **70.0** | 50.0 | 60.0 | 25.0 |

| Script | Raw output |
|---|---|
| `experiments/exp29_complementary_roles.py` | `results/exp29/raw.json` — 320 runs (4 cells × 4 modes × 20 seeds) |

```bash
python -m experiments.exp29_complementary_roles --seeds 20
```

### Paper Table 2 — Cross-vendor generalization

> Seven LLMs from OpenAI, Anthropic, and Google. (i) Synthetic graphs with direction-conflict noise (1–3 conflicts per graph, n=20 seeds per cell). (ii) TextWorld procedurally-generated text-adventure games with mango-like noise mixture (room-name collapses, duplicate-direction edges, hallucinated rooms; n=30 seeds per cell). Bold entries denote cells where VC+EI outperforms the baseline LLM.

| Model | Synthetic Base | Synthetic Ours | TextWorld Base | TextWorld Ours |
|-------|---------------:|---------------:|---------------:|---------------:|
| GPT-5.5           | 25.0 | **75.0** | 20.0 | 20.0          |
| GPT-5-mini        | 20.0 | **35.0** | 16.7 | 16.7          |
| o4-mini           | 30.0 | 20.0     | 20.0 | **26.7**      |
| Claude-Sonnet 4.6 | 30.0 | **40.0** | 20.0 | **33.3**      |
| Claude-Haiku 4.5  | 10.0 | **30.0** | 16.7 | **33.3**      |
| Gemini 2.5-Flash  | 25.0 | 10.0     | 20.0 | **26.7**      |
| Gemini 3.5-Flash  | 20.0 | **50.0** | 20.0 | **33.3**      |

| Script | Raw output |
|---|---|
| `experiments/exp25_frontier_2026.py` | `results/exp25/raw.json` + `results/exp25_extra/raw.json` (synthetic, n=20 per cell) |
| `experiments/exp26_frontier_textworld.py`, `experiments/exp26b_claude_textworld.py` | `results/exp26/raw.json` + `results/exp26b/raw.json` + `results/exp26c/raw.json` + `results/exp26_extra/raw.json` (TextWorld, n=30 per cell) |

```bash
python -m experiments.exp25_frontier_2026
python -m experiments.exp26_frontier_textworld
```

### Paper Table 3 — Real IF maps from MANGO walkthroughs

> Repair on all 42 cleaned-MANGO games whose gpt-4.1-built input graphs contain ≥1 residual conflict (534 conflicts in aggregate). Three vendors × three modes plus two non-LLM references. Counts above 534 indicate that the repair mode introduced additional conflicts; lower is better.

Headline cells (residual conflicts after repair):

- **GPT-5.5 EI**: 609 → 396 (Δ = −213, 35% relative improvement)
- **Claude-Haiku 4.5 VC+EI**: 874 → 625 (Δ = −249, 28% relative improvement)
- **Gemini 3.5-Flash EI**: 841 → 572 (Δ = −269, 32% relative improvement)
- **heuristic_remove**: 98 (strongest absolute reducer)
- **heuristic_modify**: 438

The Table 3 data pipeline (each stage's output feeds the next):

```
data_fixed/<game>/<game>.walkthrough         ← MANGO raw walkthrough
        ↓ experiments/exp11c_gt_aligned_clean.py
results/exp11c/clean_walkthroughs/           ← 53 GT-aligned clean walkthroughs
        ↓ experiments/exp14_remap_v3_fixed.py  (gpt-4.1 LLM mapping)
results/exp14/gpt-4.1/<game>_edges.json      ← 53 LLM-built input graphs (42 of them have ≥1 conflict, aggregating to 534)
        ↓ experiments/exp30c_full_mango_sweep.py  (EI / VC+EI / heuristic repair)
results/exp30c/raw.json                      ← 362 per-(game, mode, model) repair runs that produce Table 3
```

```bash
python -m experiments.exp11c_gt_aligned_clean
python -m experiments.exp14_remap_v3_fixed
python -m experiments.exp30c_full_mango_sweep
```

### Paper Table A3 — Structure preservation across repair modes

> Aggregate change in ground-truth-direction-correct edges across the same 42 cleaned-MANGO games. All values are non-positive; values closer to zero (less negative) indicate better preservation.

| Repair LLM | Base. | EI | VC+EI |
|---|---:|---:|---:|
| GPT-5.5            | −33 | −86 | −89 |
| Claude-Haiku 4.5   | −32 | −67 | −31 |
| Gemini 3.5-Flash   | −32 | −69 | −51 |

Non-LLM references (model-independent):

| Reference | Edge loss |
|---|---:|
| `heuristic_modify` | −97 (largest GT edge loss in the table) |
| `heuristic_remove` | −56 |

| Script | Raw output |
|---|---|
| `experiments/exp30c_full_mango_sweep.py` | `results/exp30c/raw.json` — the `correct_dir_edges_delta` field of each row |

(Table A3 shares its raw source with Table 3; the two tables are different aggregations of the same 362-run experiment.)

### Hyperparameters — max_iter sensitivity sweep

> 5 representative cleaned-MANGO games × GPT-5.5 + Edge-Impact × max_iter ∈ {5, 10, 20, 40}.

Aggregate net resolution rate (repaired minus newly-introduced, normalized by input conflicts; positive = net repair):

| max_iter | 5 | 10 | **20** | 40 |
|---|---:|---:|---:|---:|
| Net resolution | −30.6% | −32.4% | **+50.9%** | −17.6% |

| Script | Raw output |
|---|---|
| `experiments/exp31_iter_sensitivity.py` | `results/exp31_iter_sensitivity/raw.json` — 20 runs (5 games × 4 max_iter values) |

```bash
python -m experiments.exp31_iter_sensitivity
```

### Paper Table 4 — *Dream of the Red Chamber* (DRC) deployment

> End-to-end LLM-MapRepair on natural text. Chapters 16–17 against a human-authored ground-truth map (35 unique locations, 34 spatial relation pairs evaluated as undirected pairs). Both methods use gpt-4.1.

| Method | Predicted #N | Predicted #E | Node recall | Edge recall |
|---|---:|---:|---:|---:|
| Baseline LLM        | 47  | 49  | 85.7% | 32.4% |
| **LLM-MapRepair**   | 143 | 144 | **94.3%** | **88.2%** |
| Δ                   | +96 | +95 | +8.6 pp | +55.8 pp |

The DRC pipeline data files live in the companion repository https://github.com/nitpicker55555/spatial_memory:

| File | What it represents |
|---|---|
| `honglou_ground_truth_fixed.json` | 35-node / 34-edge human-authored ground truth |
| `honglou_llm_incremental.json`    | Baseline LLM output: 47 nodes / 49 edges (85.7% / 32.4% recall) |
| `honglou_llm_rule_fixed.json`     | LLM-MapRepair output: 143 nodes / 144 edges (94.3% / 88.2% recall) |

### Section 4.4 / Appendix B — Algorithmic Validation

**TC1–TC6 hand-crafted scenarios:**

| Metric | Value |
|---|---:|
| Average LCA candidate-edge reduction across 6 TCs | **24.6%** |
| Per-TC reduction (TC1 / TC2 / TC3 / TC4-T / TC4-D / TC5) | 11.1% / 14.3% / 22.2% / 25.0% / 75.0% / 0.0% |
| TC6 cascade-prediction Spearman ρ (5-edge sanity check) | **1.000** |
| TC6 priority-inspection speedup vs random | **2.3×** (10 vs 23 edges) |
| TC6 inspection reduction | 56.5% fewer edges examined |
| TC6 80%-impact acceleration | 1.82× (17 vs 31 edges) |

These numbers reproduce bit-exactly from the companion repository https://github.com/nitpicker55555/spatial_memory, under `lca_algorithm_validation/`:

| Source script | Raw output |
|---|---|
| `test_lca_error_localization.py`        | `lca_test_results.json` |
| `test_secondary_conflict_acceleration.py` | `secondary_conflict_test_results.json` |

**1,160-graph programmatically-generated scale-up:**

| Metric | Value |
|---|---:|
| Mean LCA candidate-edge reduction | **47.80%** (direction 54.64% / topology 32.27% / naming 56.72%) |
| True-error retention in LCA candidate set | **81.12%** (100% on direction+topology, 42.4% on naming) |

| Script | Raw output |
|---|---|
| `experiments/exp01_localization.py` | `results/exp01/raw.json` — 1,160 entries (~390 per conflict type) |

```bash
python -m experiments.exp01_localization
```

### Appendix A — Refined MANGO dataset

> 53 environments, 1,673 → 1,513 edges (160 removed) via the 6-step refinement pipeline.

| Item | Path |
|---|---|
| Original MANGO | external (Ding et al. 2024) |
| Refined dataset | `data_fixed/` — also mirrored to https://huggingface.co/datasets/boboIloveyou/spatial_refined_mango |
| 6-step pipeline description | Paper Appendix A |

## Usage

### Batch experiments (reference implementation)

```bash
# Single game
python batch_run.py --games zork1 --data-dir /path/to/data_fixed

# All 53 games in parallel
python batch_run.py --all --workers 4 --data-dir /path/to/data_fixed

# Different LLM backbone, capped at 50 steps each
python batch_run.py --all --model gpt-4o-mini --max-steps 50
```

### End-to-end smoke test (no LLM required)

```bash
python test_pipeline.py
```

`test_pipeline.py` replays the paper's Case Study B in plain Python: it introduces a directional error at step 5, watches the topological conflict fire 15 steps later when Lab and Meeting Room land on the same coordinate, runs LCA on the Reasoning History Tree, and prints the EIS ranking that puts the true error at the top.

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
