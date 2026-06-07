# Experiment 26 — 2026 frontier models on TextWorld real-language

Total runs: 180 (valid 180, errors 0)

Models: ['gemini-3.5-flash', 'claude-haiku-4-5-20251001']
Regime: mango_like (mirrors MANGO's failure-mode mix)
Games: 10 TextWorld games (real-prose room names)
Seeds per cell: 3

## Headline: real-language CF % per model

| Model | baseline LLM | edge_impact (ours) | Δ lift | edge recall after | Δ recall |
|-------|-------------:|-------------------:|-------:|------------------:|---------:|
| gemini-3.5-flash | 20.0% (n=30) | **33.3%** (n=30) | **+13.3pp** | 54.0% | -4.39pp |
| claude-haiku-4-5-20251001 | 16.7% (n=30) | **33.3%** (n=30) | **+16.7pp** | 53.0% | -5.34pp |
| no_repair (ref)        | — | 16.7% | — | 58.4% | 0 |
| heuristic_remove (ref) | — | 56.7% | — | 49.2% | -9.15pp |

## Detailed: by (method × model)

| method | model | n | CF % | conf reduce | edge recall after | edge recall Δ | dir acc | iters |
|--------|-------|--:|-----:|-----------:|------------------:|--------------:|--------:|------:|
| no_repair | n/a | 30 | 16.7 | 0.0 | 58.4 | +0.00pp | 91.4 | 0.0 |
| heuristic_remove | n/a | 30 | 56.7 | 69.0 | 49.2 | -9.15pp | 93.4 | 3.9 |
