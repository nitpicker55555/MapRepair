# Experiment 26 — 2026 frontier models on TextWorld real-language

Total runs: 240 (valid 240, errors 0)

Models: ['claude-sonnet-4-6', 'gemini-2.5-flash', 'o4-mini']
Regime: mango_like (mirrors MANGO's failure-mode mix)
Games: 10 TextWorld games (real-prose room names)
Seeds per cell: 3

## Headline: real-language CF % per model

| Model | baseline LLM | edge_impact (ours) | Δ lift | edge recall after | Δ recall |
|-------|-------------:|-------------------:|-------:|------------------:|---------:|
| claude-sonnet-4-6 | 20.0% (n=30) | **33.3%** (n=30) | **+13.3pp** | 53.7% | -4.64pp |
| gemini-2.5-flash | 20.0% (n=30) | **26.7%** (n=30) | **+6.7pp** | 57.3% | -1.10pp |
| o4-mini | 20.0% (n=30) | **26.7%** (n=30) | **+6.7pp** | 56.3% | -2.03pp |
| no_repair (ref)        | — | 16.7% | — | 58.4% | 0 |
| heuristic_remove (ref) | — | 56.7% | — | 49.2% | -9.15pp |

## Detailed: by (method × model)

| method | model | n | CF % | conf reduce | edge recall after | edge recall Δ | dir acc | iters |
|--------|-------|--:|-----:|-----------:|------------------:|--------------:|--------:|------:|
| no_repair | n/a | 30 | 16.7 | 0.0 | 58.4 | +0.00pp | 91.4 | 0.0 |
| heuristic_remove | n/a | 30 | 56.7 | 69.0 | 49.2 | -9.15pp | 93.4 | 3.9 |
