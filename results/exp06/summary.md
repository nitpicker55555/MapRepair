# Experiment 6 - Difficulty curves (heuristic vs random vs num_errors)

Total scenarios: 900

Cell values are mean over `seeds`. Use the CSV / raw JSON for distributions.

| family | err_type | num_err | n | heur_cf% | heur_dir% | heur_rec | rand_cf% | rand_dir% | rand_rec |
|--------|----------|--------:|--:|---------:|----------:|---------:|---------:|----------:|---------:|
| grid | direction | 1 | 25 | 4.0 | 96.1 | 0.92 | 0.0 | 69.1 | 0.44 |
| grid | direction | 2 | 25 | 0.0 | 95.6 | 1.92 | 0.0 | 68.9 | 0.68 |
| grid | direction | 3 | 25 | 0.0 | 94.4 | 2.88 | 0.0 | 69.7 | 1.28 |
| grid | direction | 4 | 25 | 0.0 | 95.2 | 3.84 | 0.0 | 69.5 | 1.04 |
| grid | direction | 5 | 25 | 0.0 | 94.8 | 4.72 | 0.0 | 69.9 | 1.64 |
| grid | topology | 1 | 25 | 56.0 | 97.8 | 0.88 | 0.0 | 68.0 | 0.52 |
| grid | topology | 2 | 25 | 24.0 | 95.3 | 1.72 | 0.0 | 67.2 | 0.80 |
| grid | topology | 3 | 25 | 20.0 | 94.6 | 2.56 | 0.0 | 66.5 | 1.20 |
| grid | topology | 4 | 25 | 8.0 | 93.0 | 3.12 | 0.0 | 65.9 | 1.56 |
| grid | topology | 5 | 25 | 4.0 | 91.8 | 3.64 | 0.0 | 64.8 | 1.68 |
| random | direction | 1 | 25 | 48.0 | 91.1 | 0.92 | 12.0 | 63.7 | 0.28 |
| random | direction | 2 | 25 | 36.0 | 87.6 | 1.88 | 4.0 | 62.6 | 0.68 |
| random | direction | 3 | 25 | 52.0 | 87.2 | 2.80 | 0.0 | 62.2 | 1.24 |
| random | direction | 4 | 25 | 4.0 | 85.6 | 3.64 | 0.0 | 62.6 | 1.48 |
| random | direction | 5 | 25 | 28.0 | 82.3 | 4.64 | 0.0 | 62.4 | 1.92 |
| random | direction | 6 | 25 | 4.0 | 82.2 | 5.56 | 0.0 | 63.0 | 2.08 |
| random | direction | 7 | 25 | 4.0 | 82.0 | 6.48 | 0.0 | 63.2 | 2.40 |
| random | direction | 8 | 25 | 8.0 | 80.9 | 7.28 | 0.0 | 63.9 | 2.68 |
| random | topology | 1 | 25 | 84.0 | 95.5 | 0.20 | 0.0 | 62.6 | 0.52 |
| random | topology | 2 | 25 | 88.0 | 93.3 | 1.00 | 0.0 | 62.0 | 0.72 |
| random | topology | 3 | 25 | 96.0 | 91.5 | 1.32 | 0.0 | 62.2 | 1.20 |
| random | topology | 4 | 25 | 92.0 | 89.4 | 1.68 | 0.0 | 62.6 | 1.48 |
| random | topology | 5 | 25 | 92.0 | 87.2 | 1.56 | 0.0 | 62.2 | 1.76 |
| random | topology | 6 | 25 | 92.0 | 86.0 | 2.12 | 0.0 | 63.1 | 2.28 |
| random | topology | 7 | 25 | 92.0 | 84.7 | 2.08 | 0.0 | 64.1 | 2.60 |
| random | topology | 8 | 25 | 84.0 | 83.2 | 2.04 | 0.0 | 64.5 | 3.00 |
| tree | direction | 1 | 25 | 84.0 | 91.6 | 0.88 | 0.0 | 61.9 | 0.36 |
| tree | direction | 2 | 25 | 44.0 | 88.7 | 1.76 | 0.0 | 62.5 | 0.76 |
| tree | direction | 3 | 25 | 40.0 | 86.5 | 2.60 | 0.0 | 62.6 | 1.08 |
| tree | direction | 4 | 25 | 36.0 | 83.8 | 3.56 | 0.0 | 62.5 | 1.56 |
| tree | direction | 5 | 25 | 32.0 | 80.8 | 4.28 | 0.0 | 63.1 | 1.88 |
| tree | topology | 1 | 25 | 76.0 | 95.6 | 0.04 | 0.0 | 61.5 | 0.36 |
| tree | topology | 2 | 25 | 80.0 | 92.1 | 0.40 | 0.0 | 62.6 | 0.76 |
| tree | topology | 3 | 25 | 92.0 | 89.1 | 0.72 | 0.0 | 63.1 | 1.36 |
| tree | topology | 4 | 25 | 100.0 | 88.6 | 1.36 | 0.0 | 63.2 | 1.76 |
| tree | topology | 5 | 25 | 92.0 | 85.5 | 1.32 | 0.0 | 63.0 | 2.00 |
