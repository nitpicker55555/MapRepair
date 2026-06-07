# LLM-MapRepair ablation (model=gpt-4.1)

Each mode adds one capability vs the previous:

  - baseline:    raw LLM, sees only involved edges of the current conflict
  - edge_impact: + LCA-filtered candidates ranked by impact score
  - vc_only:     + version history (can rollback to a prior commit)
  - vc_ei:       both (= full LLM-MapRepair pipeline)

## Headline

| mode | CF % | GT recall % | GT dir acc % | mean iters | Δ CF vs baseline |
|------|-----:|------------:|-------------:|-----------:|-----------------:|
| baseline | 46.2 | 100.0 | 99.3 | 5.0 | +0.0pp |
| edge_impact | 45.0 | 100.0 | 96.1 | 8.2 | -1.2pp |
| vc_only | 45.0 | 100.0 | 99.3 | 6.0 | -1.2pp |
| vc_ei | 42.5 | 100.0 | 96.7 | 8.2 | -3.8pp |
