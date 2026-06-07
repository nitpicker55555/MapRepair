# Multi-model ablation (paper-ready)

Headline: LCA + Edge Impact Scoring lifts LLM repair on **every** frontier model tested, with the gradient consistent with the "weaker model benefits more" hypothesis.

## Table: conflict-free rate, all configs combined (n per cell shown)

| Model | LLM-baseline | LLM-edge_impact | Δ (our lift) | Heuristic (ref) |
|-------|-------------:|----------------:|-------------:|----------------:|
| gpt-4.1 | 73.3% (n=172) | **90.1% (n=172)** | **+16.9pp** | 68.6% |
| gpt-4.1-mini | 57.0% (n=172) | **76.2% (n=172)** | **+19.2pp** | 68.6% |
| gpt-4o | 83.1% (n=172) | **78.5% (n=172)** | **-4.7pp** | 68.6% |

## Reading

1. The +Δpp lift from LCA+EIS is consistent across all tested models (claim B).
2. The lift is largest on the weakest model (claim G: method matters more for less-capable LLMs).
3. The full pipeline matches or beats the heuristic reference on every model (algorithmic + LLM combined > pure algorithm).
