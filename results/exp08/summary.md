# Experiment 8 - HeuristicRepairAgent with prefer_remove=True

## MANGO (non-trivial games)
games: 32
conflict_free_rate: 25.0%
mean delta strict_direction_match: -6.20pp
mean delta edge_recall: -5.35pp

## Synthetic (paired rotate vs remove)
| family | err | n | rotate cf | remove cf | rotate dir | remove dir |
|--------|-----|--:|----------:|----------:|-----------:|-----------:|
| grid | direction | 20 | 10.0% | 95.0% | 94.4% | 99.9% |
| grid | topology | 20 | 55.0% | 70.0% | 96.7% | 98.7% |
| random | direction | 20 | 40.0% | 95.0% | 87.8% | 100.0% |
| random | topology | 20 | 75.0% | 95.0% | 90.9% | 99.6% |
| tree | direction | 20 | 65.0% | 60.0% | 90.0% | 99.6% |
| tree | topology | 20 | 75.0% | 85.0% | 92.9% | 99.9% |
