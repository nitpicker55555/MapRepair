# Experiment 26 — 2026 frontier models on TextWorld real-language

Total runs: 360 (valid 360, errors 0)

Models: ['gpt-5.5', 'gpt-5-mini', 'claude-sonnet-4-6', 'gemini-2.5-flash', 'o4-mini']
Regime: mango_like (mirrors MANGO's failure-mode mix)
Games: 10 TextWorld games (real-prose room names)
Seeds per cell: 3

## Headline: real-language CF % per model

| Model | baseline LLM | edge_impact (ours) | Δ lift | edge recall after | Δ recall |
|-------|-------------:|-------------------:|-------:|------------------:|---------:|
| gpt-5.5 | 20.0% (n=30) | **20.0%** (n=30) | **+0.0pp** | 56.0% | -2.33pp |
| gpt-5-mini | 16.7% (n=30) | **16.7%** (n=30) | **+0.0pp** | 58.4% | +0.00pp |
| claude-sonnet-4-6 | 16.7% (n=30) | **16.7%** (n=30) | **+0.0pp** | 58.4% | +0.00pp |
| gemini-2.5-flash | 16.7% (n=30) | **16.7%** (n=30) | **+0.0pp** | 58.4% | +0.00pp |
| o4-mini | 16.7% (n=30) | **16.7%** (n=30) | **+0.0pp** | 58.4% | +0.00pp |
| no_repair (ref)        | — | 16.7% | — | 58.4% | 0 |
| heuristic_remove (ref) | — | 56.7% | — | 49.2% | -9.15pp |

## Detailed: by (method × model)

| method | model | n | CF % | conf reduce | edge recall after | edge recall Δ | dir acc | iters |
|--------|-------|--:|-----:|-----------:|------------------:|--------------:|--------:|------:|
| no_repair | n/a | 30 | 16.7 | 0.0 | 58.4 | +0.00pp | 91.4 | 0.0 |
| heuristic_remove | n/a | 30 | 56.7 | 69.0 | 49.2 | -9.15pp | 93.4 | 3.9 |
