# Experiment 21 (Phase 1) — TextWorld diagnostic pilot

Generated 10 games (sizes [5, 7, 9, 11]).
Model: gpt-4.1

## Aggregate error distribution

Total pred edges: 72
Total GT edges:   148
Micro edge recall: 48.6%
Macro edge recall: 43.2%

| bucket | count | % of pred |
|--------|------:|----------:|
| CORRECT | 72 | 100.0% |
| WRONG_DIRECTION | 0 | 0.0% |
| WRONG_DST | 0 | 0.0% |
| SPURIOUS_PAIR | 0 | 0.0% |
| HALLUCINATED_DST | 0 | 0.0% |
| HALLUCINATED_SRC | 0 | 0.0% |

## Per-game

| game | rooms | gt_edges | tour | pred | recall % | CORRECT | WRONG_DIR | WRONG_DST | SPURIOUS | HALL_DST |
|------|-----:|--------:|-----:|-----:|---------:|--------:|----------:|----------:|--------:|---------:|
| tw_00 | 5 | 10 | 8 | 6 | 60.0 | 6 | 0 | 0 | 0 | 0 |
| tw_01 | 7 | 16 | 12 | 7 | 43.8 | 7 | 0 | 0 | 0 | 0 |
| tw_02 | 9 | 16 | 16 | 7 | 31.2 | 7 | 0 | 0 | 0 | 0 |
| tw_03 | 11 | 22 | 20 | 8 | 31.8 | 8 | 0 | 0 | 0 | 0 |
| tw_04 | 5 | 8 | 8 | 7 | 87.5 | 7 | 0 | 0 | 0 | 0 |
| tw_05 | 7 | 14 | 12 | 9 | 42.9 | 9 | 0 | 0 | 0 | 0 |
| tw_06 | 9 | 16 | 16 | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 |
| tw_07 | 11 | 24 | 20 | 16 | 54.2 | 16 | 0 | 0 | 0 | 0 |
| tw_08 | 5 | 8 | 8 | 4 | 37.5 | 4 | 0 | 0 | 0 | 0 |
| tw_09 | 7 | 14 | 12 | 8 | 42.9 | 8 | 0 | 0 | 0 | 0 |

## Reading guide

- **CORRECT + WRONG_DIRECTION** dominate ⇒ TextWorld errors are *edge-level* — algorithm target ✅
- **WRONG_DST + SPURIOUS_PAIR + HALLUCINATED_***** dominate ⇒ TextWorld errors are *node-level* — same problem as MANGO ❌
