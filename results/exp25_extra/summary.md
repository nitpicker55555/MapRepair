# Experiment 25 — 2026 frontier-model robustness check

Total runs: 200 (valid 200, errors 0, skipped 0)

Models: ['gemini-3.5-flash', 'claude-haiku-4-5-20251001']

## Headline: CF % per model (n per cell shown)

| Model | baseline LLM | edge_impact (ours) | Δ lift | Δ vs heuristic |
|-------|-------------:|-------------------:|-------:|---------------:|
| gemini-3.5-flash | 50.0% (n=40) | **50.0%** (n=40) | **+0.0pp** | **-30.0pp** |
| claude-haiku-4-5-20251001 | 47.5% (n=40) | **55.0%** (n=40) | **+7.5pp** | **-25.0pp** |
| heuristic_remove (ref) | — | — | — | 80.0% baseline |

## Detailed: by (mode × model)

| mode | model | n | CF % | GT recall | dir acc | iters | elapsed |
|------|-------|--:|-----:|----------:|--------:|------:|--------:|
| baseline | gemini-3.5-flash | 40 | 50.0 | 100.0 | 98.8 | 5.8 | 24s |
| baseline | claude-haiku-4-5-20251001 | 40 | 47.5 | 100.0 | 99.6 | 5.3 | 18s |
| edge_impact | gemini-3.5-flash | 40 | 50.0 | 99.8 | 93.6 | 8.9 | 52s |
| edge_impact | claude-haiku-4-5-20251001 | 40 | 55.0 | 98.0 | 96.5 | 7.5 | 30s |
| heuristic_remove | n/a | 40 | 80.0 | 97.2 | 99.1 | 2.4 | 0.0s |

## By model × err_type

| model | err_type | mode | n | CF % | dir acc |
|-------|----------|------|--:|-----:|--------:|
| gemini-3.5-flash | direction | baseline | 20 | 20.0 | 98.9 |
| gemini-3.5-flash | direction | edge_impact | 20 | 50.0 | 95.8 |
| gemini-3.5-flash | topology | baseline | 20 | 80.0 | 98.8 |
| gemini-3.5-flash | topology | edge_impact | 20 | 50.0 | 91.4 |
| claude-haiku-4-5-20251001 | direction | baseline | 20 | 10.0 | 99.7 |
| claude-haiku-4-5-20251001 | direction | edge_impact | 20 | 30.0 | 94.6 |
| claude-haiku-4-5-20251001 | topology | baseline | 20 | 85.0 | 99.5 |
| claude-haiku-4-5-20251001 | topology | edge_impact | 20 | 80.0 | 98.5 |
