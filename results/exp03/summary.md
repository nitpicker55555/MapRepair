# Experiment 3 — End-to-end heuristic repair vs random / none baselines

Total scenarios: 540

Each row aggregates over seeds and graph sizes for a (family, err_type).
Columns: conflict-free rate %, mean GT edge recall %, mean GT direction acc %,
mean iterations, mean recovered errors per scenario.

| family | err_type | n | heuristic_cf | heuristic_dir_acc | heuristic_rec_err | random_cf | random_dir_acc | random_rec_err | none_cf |
|--------|----------|--:|-------------:|-------------------:|-------------------:|----------:|---------------:|---------------:|--------:|
| grid | direction | 90 | 77.8% | 99.2% | 1.23 | 0.0% | 71.2% | 0.33 | 0.0% |
| grid | topology | 90 | 71.1% | 98.7% | 1.09 | 0.0% | 69.2% | 0.39 | 0.0% |
| random | direction | 90 | 60.0% | 99.5% | 2.21 | 3.3% | 76.4% | 0.39 | 0.0% |
| random | topology | 90 | 62.2% | 99.5% | 0.46 | 2.2% | 76.6% | 0.42 | 0.0% |
| tree | direction | 90 | 56.7% | 99.7% | 1.16 | 7.8% | 67.9% | 0.44 | 0.0% |
| tree | topology | 90 | 85.6% | 99.6% | 0.11 | 1.1% | 66.5% | 0.41 | 0.0% |
