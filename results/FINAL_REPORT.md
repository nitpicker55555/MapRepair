# Final consolidated report

(Numbers below are aggregated from each experiment's `raw.json` / per-game files. Missing entries indicate the experiment has not finished yet.)


## exp01_localization
```json
{
  "n": 1160,
  "mean_reduction": 0.477979672828166,
  "mean_truth_in_lca": 0.811248973727422,
  "per_err": {
    "direction": {
      "n": 390,
      "mean_reduction": 0.5463626843866864,
      "mean_truth_in_lca": 1.0
    },
    "topology": {
      "n": 390,
      "mean_reduction": 0.3227048217669301,
      "mean_truth_in_lca": 1.0
    },
    "naming": {
      "n": 380,
      "mean_reduction": 0.5671581396862161,
      "mean_truth_in_lca": 0.423812656641604
    }
  }
}
```

## exp02_scoring
```json
{
  "n": 140,
  "mean_spearman": 0.999173524728481,
  "stdev_spearman": 0.0028123742402089164,
  "mean_kendall": 0.9962108286726173
}
```

## exp03_heuristic_repair
```json
{
  "n": 540,
  "per_cohort": {
    "tree__direction": {
      "n": 90,
      "heuristic_conflict_free_pct": 56.666666666666664,
      "heuristic_dir_acc_pct": 99.71104142068504,
      "random_conflict_free_pct": 7.777777777777778,
      "random_dir_acc_pct": 67.87017177216813
    },
    "tree__topology": {
      "n": 90,
      "heuristic_conflict_free_pct": 85.55555555555556,
      "heuristic_dir_acc_pct": 99.64961817593397,
      "random_conflict_free_pct": 1.1111111111111112,
      "random_dir_acc_pct": 66.47518744614933
    },
    "grid__direction": {
      "n": 90,
      "heuristic_conflict_free_pct": 77.77777777777777,
      "heuristic_dir_acc_pct": 99.18822507068977,
      "random_conflict_free_pct": 0.0,
      "random_dir_acc_pct": 71.15277777777777
    },
    "grid__topology": {
      "n": 90,
      "heuristic_conflict_free_pct": 71.11111111111111,
      "heuristic_dir_acc_pct": 98.66598241992469,
      "random_conflict_free_pct": 0.0,
      "random_dir_acc_pct": 69.16666666666667
    },
    "random__direction": {
      "n": 90,
      "heuristic_conflict_free_pct": 60.0,
      "heuristic_dir_acc_pct": 99.47796583370796,
      "random_conflict_free_pct": 3.3333333333333335,
      "random_dir_acc_pct": 76.36377675948619
    },
    "random__topology": {
      "n": 90,
      "heuristic_conflict_free_pct": 62.22222222222222,
      "heuristic_dir_acc_pct": 99.52340980958087,
      "random_conflict_free_pct": 2.2222222222222223,
      "random_dir_acc_pct": 76.57498768063626
    }
  }
}
```

## exp04_llm_synth
```json
{
  "n": 160,
  "per_mode": {
    "vc_only": {
      "n": 40,
      "conflict_free_pct": 35.0,
      "mean_edge_recall_pct": 100.0,
      "mean_dir_acc_pct": 99.89583333333333,
      "strict_correct_pct": 35.0,
      "mean_iterations": 1.4
    },
    "edge_impact": {
      "n": 40,
      "conflict_free_pct": 12.5,
      "mean_edge_recall_pct": 100.0,
      "mean_dir_acc_pct": 98.75337726988424,
      "strict_correct_pct": 12.5,
      "mean_iterations": 4.075
    },
    "baseline": {
      "n": 40,
      "conflict_free_pct": 32.5,
      "mean_edge_recall_pct": 100.0,
      "mean_dir_acc_pct": 99.84375,
      "strict_correct_pct": 32.5,
      "mean_iterations": 1.375
    },
    "vc_ei": {
      "n": 40,
      "conflict_free_pct": 0.0,
      "mean_edge_recall_pct": 100.0,
      "mean_dir_acc_pct": 99.73958333333334,
      "strict_correct_pct": 0.0,
      "mean_iterations": 376.125
    }
  }
}
```

## exp05_mango_llm
```json
{
  "per_mode": {
    "baseline": {
      "n": 42,
      "n_non_trivial": 32,
      "conflict_free_pct_all": 26.19047619047619,
      "conflict_free_pct_non_trivial": 3.125,
      "mean_pre_strict": 48.772794311326436,
      "mean_post_strict": 46.96300386945797,
      "mean_delta_strict": -1.8097904418684727
    },
    "edge_impact": {
      "n": 42,
      "n_non_trivial": 32,
      "conflict_free_pct_all": 26.19047619047619,
      "conflict_free_pct_non_trivial": 3.125,
      "mean_pre_strict": 48.772794311326436,
      "mean_post_strict": 45.58715433291665,
      "mean_delta_strict": -3.1856399784097955
    },
    "vc_only": {
      "n": 42,
      "n_non_trivial": 32,
      "conflict_free_pct_all": 23.80952380952381,
      "conflict_free_pct_non_trivial": 0.0,
      "mean_pre_strict": 48.772794311326436,
      "mean_post_strict": 47.18291788046261,
      "mean_delta_strict": -1.5898764308638322
    },
    "vc_ei": {
      "n": 40,
      "n_non_trivial": 30,
      "conflict_free_pct_all": 25.0,
      "conflict_free_pct_non_trivial": 0.0,
      "mean_pre_strict": 48.63833461168808,
      "mean_post_strict": 48.63833461168808,
      "mean_delta_strict": 0
    }
  }
}
```

## exp06_difficulty
```json
{
  "n": 900,
  "per_cohort": {
    "tree__direction__1": {
      "n": 25,
      "heuristic_cf_pct": 84.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 91.58620689655173,
      "random_dir_acc_pct": 61.862068965517246
    },
    "tree__direction__2": {
      "n": 25,
      "heuristic_cf_pct": 44.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 88.6896551724138,
      "random_dir_acc_pct": 62.48275862068966
    },
    "tree__direction__3": {
      "n": 25,
      "heuristic_cf_pct": 40.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 86.48275862068967,
      "random_dir_acc_pct": 62.62068965517241
    },
    "tree__direction__4": {
      "n": 25,
      "heuristic_cf_pct": 36.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 83.79310344827586,
      "random_dir_acc_pct": 62.48275862068966
    },
    "tree__direction__5": {
      "n": 25,
      "heuristic_cf_pct": 32.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 80.82758620689656,
      "random_dir_acc_pct": 63.10344827586207
    },
    "tree__topology__1": {
      "n": 25,
      "heuristic_cf_pct": 76.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 95.58620689655173,
      "random_dir_acc_pct": 61.51724137931035
    },
    "tree__topology__2": {
      "n": 25,
      "heuristic_cf_pct": 80.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 92.06533575317604,
      "random_dir_acc_pct": 62.55172413793103
    },
    "tree__topology__3": {
      "n": 25,
      "heuristic_cf_pct": 92.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 89.13490623109497,
      "random_dir_acc_pct": 63.10344827586207
    },
    "tree__topology__4": {
      "n": 25,
      "heuristic_cf_pct": 100.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 88.60573848414138,
      "random_dir_acc_pct": 63.17241379310345
    },
    "tree__topology__5": {
      "n": 25,
      "heuristic_cf_pct": 92.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 85.52721432630689,
      "random_dir_acc_pct": 63.03448275862069
    },
    "random__direction__1": {
      "n": 25,
      "heuristic_cf_pct": 48.0,
      "random_cf_pct": 12.0,
      "heuristic_dir_acc_pct": 91.11864406779662,
      "random_dir_acc_pct": 63.66101694915254
    },
    "random__direction__2": {
      "n": 25,
      "heuristic_cf_pct": 36.0,
      "random_cf_pct": 4.0,
      "heuristic_dir_acc_pct": 87.59322033898304,
      "random_dir_acc_pct": 62.64406779661017
    },
    "random__direction__3": {
      "n": 25,
      "heuristic_cf_pct": 52.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 87.1864406779661,
      "random_dir_acc_pct": 62.23728813559322
    },
    "random__direction__4": {
      "n": 25,
      "heuristic_cf_pct": 4.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 85.6271186440678,
      "random_dir_acc_pct": 62.64406779661017
    },
    "random__direction__5": {
      "n": 25,
      "heuristic_cf_pct": 28.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 82.30508474576271,
      "random_dir_acc_pct": 62.37288135593219
    },
    "random__direction__6": {
      "n": 25,
      "heuristic_cf_pct": 4.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 82.16949152542374,
      "random_dir_acc_pct": 62.98305084745762
    },
    "random__direction__7": {
      "n": 25,
      "heuristic_cf_pct": 4.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 81.96610169491525,
      "random_dir_acc_pct": 63.186440677966104
    },
    "random__direction__8": {
      "n": 25,
      "heuristic_cf_pct": 8.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 80.88135593220339,
      "random_dir_acc_pct": 63.93220338983051
    },
    "random__topology__1": {
      "n": 25,
      "heuristic_cf_pct": 84.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 95.52542372881356,
      "random_dir_acc_pct": 62.64406779661017
    },
    "random__topology__2": {
      "n": 25,
      "heuristic_cf_pct": 88.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 93.31969608416131,
      "random_dir_acc_pct": 61.96610169491525
    },
    "random__topology__3": {
      "n": 25,
      "heuristic_cf_pct": 96.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 91.54642304182433,
      "random_dir_acc_pct": 62.23728813559322
    },
    "random__topology__4": {
      "n": 25,
      "heuristic_cf_pct": 92.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 89.37506316932014,
      "random_dir_acc_pct": 62.57627118644068
    },
    "random__topology__5": {
      "n": 25,
      "heuristic_cf_pct": 92.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 87.16908806738701,
      "random_dir_acc_pct": 62.23728813559322
    },
    "random__topology__6": {
      "n": 25,
      "heuristic_cf_pct": 92.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 86.02536662840886,
      "random_dir_acc_pct": 63.118644067796616
    },
    "random__topology__7": {
      "n": 25,
      "heuristic_cf_pct": 92.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 84.7164221959417,
      "random_dir_acc_pct": 64.0677966101695
    },
    "random__topology__8": {
      "n": 25,
      "heuristic_cf_pct": 84.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 83.22568341787942,
      "random_dir_acc_pct": 64.54237288135593
    },
    "grid__direction__1": {
      "n": 25,
      "heuristic_cf_pct": 4.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 96.11159831729452,
      "random_dir_acc_pct": 69.1
    },
    "grid__direction__2": {
      "n": 25,
      "heuristic_cf_pct": 0.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 95.60339158509711,
      "random_dir_acc_pct": 68.9
    },
    "grid__direction__3": {
      "n": 25,
      "heuristic_cf_pct": 0.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 94.42759481025304,
      "random_dir_acc_pct": 69.69999999999999
    },
    "grid__direction__4": {
      "n": 25,
      "heuristic_cf_pct": 0.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 95.17763666491182,
      "random_dir_acc_pct": 69.55
    },
    "grid__direction__5": {
      "n": 25,
      "heuristic_cf_pct": 0.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 94.7677744906659,
      "random_dir_acc_pct": 69.89999999999999
    },
    "grid__topology__1": {
      "n": 25,
      "heuristic_cf_pct": 56.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 97.8165940654948,
      "random_dir_acc_pct": 68.0
    },
    "grid__topology__2": {
      "n": 25,
      "heuristic_cf_pct": 24.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 95.26399273611132,
      "random_dir_acc_pct": 67.25
    },
    "grid__topology__3": {
      "n": 25,
      "heuristic_cf_pct": 20.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 94.64656224764774,
      "random_dir_acc_pct": 66.55
    },
    "grid__topology__4": {
      "n": 25,
      "heuristic_cf_pct": 8.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 92.96063995819091,
      "random_dir_acc_pct": 65.9
    },
    "grid__topology__5": {
      "n": 25,
      "heuristic_cf_pct": 4.0,
      "random_cf_pct": 0.0,
      "heuristic_dir_acc_pct": 91.82001874060145,
      "random_dir_acc_pct": 64.85
    }
  }
}
```

## exp07_heur_on_mango
```json
{
  "n": 42,
  "n_non_trivial": 32,
  "conflict_free_pct_non_trivial": 28.125,
  "mean_pre_strict": 48.772794311326436,
  "mean_post_strict": 42.65216685557149,
  "mean_delta_strict": -6.120627455754952
}
```

## exp08_remove_first
```json
{
  "mango": {
    "n_non_trivial": 32,
    "conflict_free_pct": 25.0,
    "mean_delta_strict": -6.198452302393828
  },
  "synth_compare": {
    "tree__direction__rotate": {
      "n": 20,
      "conflict_free_pct": 65.0,
      "dir_acc_pct": 90.0
    },
    "tree__direction__remove": {
      "n": 20,
      "conflict_free_pct": 60.0,
      "dir_acc_pct": 99.5945945945946
    },
    "tree__topology__rotate": {
      "n": 20,
      "conflict_free_pct": 75.0,
      "dir_acc_pct": 92.89473684210526
    },
    "tree__topology__remove": {
      "n": 20,
      "conflict_free_pct": 85.0,
      "dir_acc_pct": 99.86486486486487
    },
    "random__direction__rotate": {
      "n": 20,
      "conflict_free_pct": 40.0,
      "dir_acc_pct": 87.75862068965517
    },
    "random__direction__remove": {
      "n": 20,
      "conflict_free_pct": 95.0,
      "dir_acc_pct": 100.0
    },
    "random__topology__rotate": {
      "n": 20,
      "conflict_free_pct": 75.0,
      "dir_acc_pct": 90.86206896551724
    },
    "random__topology__remove": {
      "n": 20,
      "conflict_free_pct": 95.0,
      "dir_acc_pct": 99.64285714285714
    },
    "grid__direction__rotate": {
      "n": 20,
      "conflict_free_pct": 10.0,
      "dir_acc_pct": 94.37557816836262
    },
    "grid__direction__remove": {
      "n": 20,
      "conflict_free_pct": 95.0,
      "dir_acc_pct": 99.89130434782608
    },
    "grid__topology__rotate": {
      "n": 20,
      "conflict_free_pct": 55.0,
      "dir_acc_pct": 96.73221460675208
    },
    "grid__topology__remove": {
      "n": 20,
      "conflict_free_pct": 70.0,
      "dir_acc_pct": 98.66465705673588
    }
  }
}
```

## exp09_head_to_head
_(no data yet)_
