# Experiment 25 — 2026 frontier-model robustness check

Total runs: 440 (valid 440, errors 0, skipped 0)

Models: ['gpt-5.5', 'gpt-5-mini', 'claude-sonnet-4-6', 'gemini-2.5-flash', 'o4-mini']

## Headline: CF % per model (n per cell shown)

| Model | baseline LLM | edge_impact (ours) | Δ lift | Δ vs heuristic |
|-------|-------------:|-------------------:|-------:|---------------:|
| gpt-5.5 | 60.0% (n=40) | **60.0%** (n=40) | **+0.0pp** | **-20.0pp** |
| gpt-5-mini | 50.0% (n=40) | **37.5%** (n=40) | **-12.5pp** | **-42.5pp** |
| claude-sonnet-4-6 | 55.0% (n=40) | **67.5%** (n=40) | **+12.5pp** | **-12.5pp** |
| gemini-2.5-flash | 47.5% (n=40) | **10.0%** (n=40) | **-37.5pp** | **-70.0pp** |
| o4-mini | 50.0% (n=40) | **32.5%** (n=40) | **-17.5pp** | **-47.5pp** |
| heuristic_remove (ref) | — | — | — | 80.0% baseline |

## Detailed: by (mode × model)

| mode | model | n | CF % | GT recall | dir acc | iters | elapsed |
|------|-------|--:|-----:|----------:|--------:|------:|--------:|
| baseline | gpt-5.5 | 40 | 60.0 | 99.9 | 99.9 | 4.2 | 39s |
| baseline | gpt-5-mini | 40 | 50.0 | 100.0 | 99.6 | 5.1 | 29s |
| baseline | claude-sonnet-4-6 | 40 | 55.0 | 99.7 | 99.3 | 5.4 | 32s |
| baseline | gemini-2.5-flash | 40 | 47.5 | 100.0 | 98.6 | 6.9 | 28s |
| baseline | o4-mini | 40 | 50.0 | 100.0 | 99.1 | 5.3 | 21s |
| edge_impact | gpt-5.5 | 40 | 60.0 | 99.3 | 94.3 | 7.5 | 79s |
| edge_impact | gpt-5-mini | 40 | 37.5 | 99.6 | 94.2 | 13.9 | 111s |
| edge_impact | claude-sonnet-4-6 | 40 | 67.5 | 98.0 | 95.9 | 7.5 | 45s |
| edge_impact | gemini-2.5-flash | 40 | 10.0 | 99.8 | 98.3 | 8.9 | 57s |
| edge_impact | o4-mini | 40 | 32.5 | 98.6 | 90.5 | 12.8 | 90s |
| heuristic_remove | n/a | 40 | 80.0 | 97.2 | 99.1 | 2.4 | 0.0s |

## By model × err_type

| model | err_type | mode | n | CF % | dir acc |
|-------|----------|------|--:|-----:|--------:|
| gpt-5.5 | direction | baseline | 20 | 25.0 | 100.0 |
| gpt-5.5 | direction | edge_impact | 20 | 75.0 | 98.2 |
| gpt-5.5 | topology | baseline | 20 | 95.0 | 99.7 |
| gpt-5.5 | topology | edge_impact | 20 | 45.0 | 90.4 |
| gpt-5-mini | direction | baseline | 20 | 20.0 | 100.0 |
| gpt-5-mini | direction | edge_impact | 20 | 35.0 | 94.0 |
| gpt-5-mini | topology | baseline | 20 | 80.0 | 99.3 |
| gpt-5-mini | topology | edge_impact | 20 | 40.0 | 94.3 |
| claude-sonnet-4-6 | direction | baseline | 20 | 30.0 | 99.7 |
| claude-sonnet-4-6 | direction | edge_impact | 20 | 40.0 | 92.0 |
| claude-sonnet-4-6 | topology | baseline | 20 | 80.0 | 98.9 |
| claude-sonnet-4-6 | topology | edge_impact | 20 | 95.0 | 99.9 |
| gemini-2.5-flash | direction | baseline | 20 | 25.0 | 99.4 |
| gemini-2.5-flash | direction | edge_impact | 20 | 10.0 | 98.1 |
| gemini-2.5-flash | topology | baseline | 20 | 70.0 | 97.8 |
| gemini-2.5-flash | topology | edge_impact | 20 | 10.0 | 98.4 |
| o4-mini | direction | baseline | 20 | 30.0 | 100.0 |
| o4-mini | direction | edge_impact | 20 | 20.0 | 88.6 |
| o4-mini | topology | baseline | 20 | 70.0 | 98.3 |
| o4-mini | topology | edge_impact | 20 | 45.0 | 92.5 |
