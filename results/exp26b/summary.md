# Experiment 26b — claude + gemini on TextWorld mango_like (retry)

Total runs: 120 (valid 120, errors 0)

Models: ['claude-sonnet-4-6', 'gemini-2.5-flash']

max_tokens=600 (proxy pre-charge limit workaround)

## Headline: CF % per model

| Model | baseline | edge_impact | Δ | conf reduce (edge_imp) | edge recall after | dir acc |
|-------|---------:|------------:|--:|----------------------:|-------------------:|--------:|
| claude-sonnet-4-6 | 16.7% (n=30) | **16.7%** (n=30) | **+0.0pp** | -0.3% | 58.4% | 91.4% |
| gemini-2.5-flash | 16.7% (n=30) | **16.7%** (n=30) | **+0.0pp** | 0.0% | 58.4% | 91.4% |

## Detailed

| mode | model | n | CF % | conf reduce | recall after | recall Δ | dir acc | actions | iters |
|------|-------|--:|-----:|------------:|-------------:|---------:|--------:|--------:|------:|
| baseline | claude-sonnet-4-6 | 30 | 16.7 | 0.5 | 58.4 | +0.00pp | 91.4 | 0.0 | 5.1 |
| baseline | gemini-2.5-flash | 30 | 16.7 | 0.0 | 58.4 | +0.00pp | 91.4 | 0.0 | 5.0 |
| edge_impact | claude-sonnet-4-6 | 30 | 16.7 | -0.3 | 58.4 | +0.00pp | 91.4 | 0.1 | 5.1 |
| edge_impact | gemini-2.5-flash | 30 | 16.7 | 0.0 | 58.4 | +0.00pp | 91.4 | 0.0 | 5.0 |
