# Cluster QCSE sweep — short report

**Scope:** 45 production causal configurations in `sweep.zip`. **44/45 completed**; experiment 039 was still running at archive capture and is excluded from rankings.

## Result

Select by mean full validation word top-1. In this binary decoder, word top-1 means exact agreement of all 14 thresholded bits with the target word ID. The winner is **experiment 35**: alpha=1, learning rate=0.0003, window=2, 64 layers, batch 64; validation word top-1 **1.121% ± 0.053%**, held-out test word top-1 **1.068%**. Secondary validation BCE is **0.661860**.

| ID | setting | validation word top-1 ± SD | test word top-1 | validation BCE |
| ---: | --- | ---: | ---: | ---: |
| 35 | a=1, lr=0.0003, w=2, L=64, B=64 | 1.121% ± 0.053% | 1.068% | 0.661860 |
| 36 | a=1, lr=0.0003, w=8, L=64, B=64 | 1.065% ± 0.021% | 0.913% | 0.662819 |
| 31 | a=1, lr=0.0003, w=2, L=8, B=16 | 0.479% ± 0.123% | 0.593% | 0.545285 |
| 24 | a=3, lr=0.001, w=2, L=8, B=64 | 0.420% ± 0.102% | 0.483% | 0.566082 |
| 06 | a=0.1, lr=0.001, w=2, L=8, B=64 | 0.406% ± 0.009% | 0.475% | 0.566024 |
| 15 | a=1, lr=0.001, w=2, L=8, B=64 | 0.388% ± 0.051% | 0.484% | 0.566056 |
| 25 | a=3, lr=0.001, w=4, L=8, B=64 | 0.322% ± 0.062% | 0.336% | 0.566663 |
| 07 | a=0.1, lr=0.001, w=4, L=8, B=64 | 0.320% ± 0.040% | 0.386% | 0.566566 |

## What the sweep says

- Learning rate 0.001 gives the highest average word top-1 in the 8-layer/batch-64 grid; alpha changes are small and window effects are weak/non-monotonic.
- Experiment 035 is the top-1 winner, but depth/batch comparisons are confounded: batches 16, 64 and 256 receive 313, 79 and 20 optimizer updates per epoch.
- Full-pool and capped-pool word top-1 is zero or near zero after ten sampled epochs; the 5,000-sentence uniform condition is incomplete.
- Sampling is with replacement inside training IDs. Repeated sentences/examples are allowed, while sentence groups remain isolated between train, validation and test.

## Quality gate

All 44 completed archives passed checks for split isolation, sentence-group isolation, fold aggregation, fixed-monitoring histories, and recomputed held-out metrics. The detailed report and 22 plots are in `detailed-report.md` and `plots/`; the machine-readable result is `aggregate.json`.

The 30→50 epoch extension rule in commit `50cff5e` belongs to the separate semantic-decoder sweep and cannot be applied to this 10-epoch production archive.
