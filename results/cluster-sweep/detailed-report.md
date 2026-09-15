# Cluster QCSE sweep — detailed report

Date generated: 2026-09-15  
Source archive: `/Users/ethan/Documents/Ecole/3A/Filière recherche/Papers/sweep.zip`  
Scope: the 45-configuration production QCSE sweep, not the separate ten-configuration semantic-decoder sweep.

## Executive summary

The archive contains 45 experiment IDs. **44 configurations completed** the intended two-fold cross-validation and final refit; experiment 039 was still running at archive capture, with only its epoch-0 first-fold checkpoint present. The archive also contains 54 older non-CV or legacy attempt directories; none is included in the rankings. All 44 completed runs passed the independent archive validator described below.

For this reanalysis, configurations are ranked by mean full validation **word top-1**, never by the held-out test. In this binary QCSE decoder, word top-1 is `exact_word_accuracy`: all 14 thresholded output bits must match the target word ID. The top configuration is experiment **35**: alpha=1, learning rate=0.0003, window=2, 64 layers, batch 64. It reached validation word top-1 **1.121% ± 0.053%**, then test word top-1 **1.068%**. Its secondary validation BCE is 0.661860 ± 0.001027. The previous BCE winner is experiment 31; it is retained as a diagnostic, not the primary selection.

The top-1 result is driven by the depth/batch comparison: experiment 035 (64 layers, batch 64) is the word-top-1 winner, while experiment 031 (8 layers, batch 16) is the BCE winner. Every epoch draws exactly 5,000 examples with replacement: batch 16 receives 313 Adam updates, batch 64 receives 79, and batch 256 receives 20. Therefore neither the batch nor depth effect is compute-matched. In the main grid, learning rate 0.001 gives the highest average word top-1, while alpha changes are small and the word-top-1 advantage of context windows is not monotonic.

## Protocol and configuration

- Both active sources were used: `phrases.csv` and `cleaned_sentences.csv`; Tatoeba was excluded. Dedupe cleaning retained 202,172 sentence rows, 1,582,956 causal examples, the 10,864-word vocabulary and 14 qubits.
- Every run used causal next-token prediction, seed 42, exponential context encoding, 5,000 replacement draws per epoch, fixed monitoring samples of 2,048 train and validation examples, two independent 20% validation shuffle-splits inside the 80% development partition, and one final held-out test evaluation after refitting on all development examples.
- Because `samples_per_epoch=5000`, the refit optimizer work is 5,000 draws per epoch rather than a full pass over the roughly 1.58 million eligible causal examples. The expensive exhaustive scoring is reserved for the two full validation folds and the final held-out test: for a full-pool run this is about 505,665 validation examples plus 316,538 test examples, not all examples on every epoch.
- A sampled epoch draws 5,000 example IDs from the training partition with replacement. Repeated IDs are therefore allowed, and a sentence can occur several times through repeated windows or repeated draws; this is training resampling, not leakage. The fixed train/validation monitoring subsets are sampled without replacement and independently from training draws.
- IDs 0–38 use the full 202,172-sentence pool. IDs 39–44 are the sentence-pool group; five are completed in this archive and ID 39 is incomplete. The completed pool runs use 5,000/20,000/50,000 sentences with uniform or balanced source sampling.
- Completed artifacts report CUDA execution with `torch 2.14.0+cu130` and CUDA `13.0`. The archive does not retain scheduler stdout/stderr, the exact Slurm allocation metadata, or a Git commit SHA; those provenance limits are kept explicit here.

## Validation and archive integrity

The validator checked, for every completed ID:

- the intended arguments, 10-epoch target, 5,000-draw budget, two-fold/fixed-monitoring protocol, and required final artifacts;
- unique and disjoint outer development/test IDs, fold train/validation disjointness, no fold/test overlap, and all sentence groups kept on one side of every split;
- the two full validation JSON files against the CV summary, including mean and sample standard deviation;
- all 11 epoch rows in both fold histories and the aggregate `cv_history.json`, recomputing every mean and sample standard deviation;
- the final test export shape, finiteness and `[0, 1]` probability range, exact example-ID correspondence to `splits.npz`, and all four test metrics recomputed from exported probabilities and target IDs.

Result: **44/44 completed runs passed; 0 validation issues**. The complete machine-readable validation result is `archive-validation.json`.

The sentence-level leakage check is explicit: the validator maps every causal
example back to its normalized sentence row and verifies that no sentence group
appears in both outer development and test, or in both train and validation
within either fold. It found **zero group overlaps** across all 44 completed
runs. Repeated sampled examples remain confined to the training IDs.

## Results

### Ranked configuration table

| ID | alpha | LR | window | layers | batch | pool | sampling | val word top-1 ± SD | test word top-1 | val BCE ± SD | test BCE | minutes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 35 | 1 | 0.0003 | 2 | 64 | 64 | full | uniform | 1.121% ± 0.053% | 1.068% | 0.661860 ± 0.001027 | 0.662792 | 129.0 |
| 36 | 1 | 0.0003 | 8 | 64 | 64 | full | uniform | 1.065% ± 0.021% | 0.913% | 0.662819 ± 0.001121 | 0.663688 | 243.7 |
| 31 | 1 | 0.0003 | 2 | 8 | 16 | full | uniform | 0.479% ± 0.123% | 0.593% | 0.545285 ± 0.001124 | 0.541256 | 49.2 |
| 24 | 3 | 0.001 | 2 | 8 | 64 | full | uniform | 0.420% ± 0.102% | 0.483% | 0.566082 ± 0.004149 | 0.569048 | 39.0 |
| 06 | 0.1 | 0.001 | 2 | 8 | 64 | full | uniform | 0.406% ± 0.009% | 0.475% | 0.566024 ± 0.004113 | 0.569096 | 39.4 |
| 15 | 1 | 0.001 | 2 | 8 | 64 | full | uniform | 0.388% ± 0.051% | 0.484% | 0.566056 ± 0.004132 | 0.569065 | 40.8 |
| 25 | 3 | 0.001 | 4 | 8 | 64 | full | uniform | 0.322% ± 0.062% | 0.336% | 0.566663 ± 0.003842 | 0.569793 | 70.9 |
| 07 | 0.1 | 0.001 | 4 | 8 | 64 | full | uniform | 0.320% ± 0.040% | 0.386% | 0.566566 ± 0.003778 | 0.569840 | 69.3 |
| 16 | 1 | 0.001 | 4 | 8 | 64 | full | uniform | 0.304% ± 0.045% | 0.346% | 0.566635 ± 0.003803 | 0.569795 | 72.0 |
| 32 | 1 | 0.0003 | 8 | 8 | 16 | full | uniform | 0.298% ± 0.116% | 0.353% | 0.548596 ± 0.001066 | 0.544316 | 140.3 |
| 26 | 3 | 0.001 | 8 | 8 | 64 | full | uniform | 0.297% ± 0.089% | 0.337% | 0.568450 ± 0.003957 | 0.571258 | 124.2 |
| 17 | 1 | 0.001 | 8 | 8 | 64 | full | uniform | 0.263% ± 0.040% | 0.334% | 0.568657 ± 0.003907 | 0.571473 | 123.4 |
| 08 | 0.1 | 0.001 | 8 | 8 | 64 | full | uniform | 0.262% ± 0.052% | 0.337% | 0.570008 ± 0.003769 | 0.572833 | 122.4 |
| 28 | 1 | 0.0003 | 8 | 2 | 16 | full | uniform | 0.082% ± 0.001% | 0.097% | 0.614825 ± 0.000631 | 0.617615 | 109.5 |
| 27 | 1 | 0.0003 | 2 | 2 | 16 | full | uniform | 0.078% ± 0.002% | 0.091% | 0.614730 ± 0.000503 | 0.617701 | 31.1 |
| 38 | 1 | 0.0003 | 8 | 64 | 256 | full | uniform | 0.013% ± 0.000% | 0.017% | 0.676868 ± 0.000376 | 0.676543 | 233.9 |
| 37 | 1 | 0.0003 | 2 | 64 | 256 | full | uniform | 0.010% ± 0.002% | 0.009% | 0.676340 ± 0.000385 | 0.676004 | 111.9 |
| 14 | 1 | 0.0003 | 8 | 8 | 64 | full | uniform | 0.002% ± 0.001% | 0.001% | 0.655330 ± 0.000622 | 0.655132 | 120.8 |
| 05 | 0.1 | 0.0003 | 8 | 8 | 64 | full | uniform | 0.002% ± 0.001% | 0.002% | 0.655719 ± 0.000589 | 0.655565 | 126.3 |
| 43 | 1 | 0.0003 | 4 | 8 | 64 | 50000 | uniform | 0.001% ± 0.001% | 0.000% | 0.652881 ± 0.000484 | 0.654681 | 42.2 |
| 23 | 3 | 0.0003 | 8 | 8 | 64 | full | uniform | 0.001% ± 0.000% | 0.001% | 0.655385 ± 0.000631 | 0.655179 | 122.7 |
| 02 | 0.1 | 0.0001 | 8 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.681963 ± 0.000180 | 0.681742 | 123.0 |
| 29 | 1 | 0.0003 | 2 | 2 | 64 | full | uniform | 0.000% ± 0.001% | 0.000% | 0.680410 ± 0.000680 | 0.680037 | 28.8 |
| 30 | 1 | 0.0003 | 8 | 2 | 64 | full | uniform | 0.000% ± 0.001% | 0.000% | 0.678728 ± 0.000736 | 0.678431 | 108.9 |
| 00 | 0.1 | 0.0001 | 2 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.681964 ± 0.000223 | 0.681694 | 40.7 |
| 01 | 0.1 | 0.0001 | 4 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.681341 ± 0.000176 | 0.681113 | 72.6 |
| 03 | 0.1 | 0.0003 | 2 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.001% | 0.654663 ± 0.000693 | 0.654501 | 39.3 |
| 04 | 0.1 | 0.0003 | 4 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.654308 ± 0.000534 | 0.654240 | 69.2 |
| 09 | 1 | 0.0001 | 2 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.681993 ± 0.000226 | 0.681717 | 39.2 |
| 10 | 1 | 0.0001 | 4 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.681505 ± 0.000180 | 0.681257 | 72.2 |
| 11 | 1 | 0.0001 | 8 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.681896 ± 0.000188 | 0.681650 | 124.6 |
| 12 | 1 | 0.0003 | 2 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.654686 ± 0.000706 | 0.654506 | 39.6 |
| 13 | 1 | 0.0003 | 4 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.654452 ± 0.000551 | 0.654343 | 69.9 |
| 18 | 3 | 0.0001 | 2 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.682017 ± 0.000226 | 0.681737 | 40.0 |
| 19 | 3 | 0.0001 | 4 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.681602 ± 0.000186 | 0.681349 | 69.6 |
| 20 | 3 | 0.0001 | 8 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.682029 ± 0.000188 | 0.681777 | 126.1 |
| 21 | 3 | 0.0003 | 2 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.654705 ± 0.000713 | 0.654513 | 39.6 |
| 22 | 3 | 0.0003 | 4 | 8 | 64 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.654534 ± 0.000568 | 0.654414 | 69.8 |
| 33 | 1 | 0.0003 | 2 | 8 | 256 | full | uniform | 0.000% ± 0.000% | 0.000% | 0.684855 ± 0.000901 | 0.683879 | 36.2 |
| 34 | 1 | 0.0003 | 8 | 8 | 256 | full | uniform | 0.000% ± 0.000% | 0.001% | 0.684683 ± 0.000863 | 0.683756 | 114.7 |
| 40 | 1 | 0.0003 | 4 | 8 | 64 | 5000 | balanced | 0.000% ± 0.000% | 0.000% | 0.654565 ± 0.000967 | 0.656801 | 33.9 |
| 41 | 1 | 0.0003 | 4 | 8 | 64 | 20000 | uniform | 0.000% ± 0.000% | 0.000% | 0.653174 ± 0.001881 | 0.654569 | 36.8 |
| 42 | 1 | 0.0003 | 4 | 8 | 64 | 20000 | balanced | 0.000% ± 0.000% | 0.000% | 0.653567 ± 0.002088 | 0.652451 | 36.5 |
| 44 | 1 | 0.0003 | 4 | 8 | 64 | 50000 | balanced | 0.000% ± 0.000% | 0.000% | 0.654484 ± 0.000711 | 0.655012 | 42.1 |

### Main 27-configuration grid

The reference family varies context alpha in {0.1, 1, 3}, learning rate in {0.0001, 0.0003, 0.001}, and window in {2, 4, 8}, with 8 layers and batch 64. The best member of this family by word top-1 is experiment 24: validation word top-1 0.420%. Across the main grid, learning rate 0.001 gives the largest improvement in word top-1; alpha changes are small, and window 8 has the highest descriptive average despite overlapping low-accuracy results.

### Depth and batch comparison

Experiments 27–38 hold alpha=1 and learning rate=0.0003 while comparing depths 2/8/64 and batch sizes 16/64/256 at windows 2/8. The best overall validation word top-1 is experiment 35 (64 layers, batch 64, window 2), at 1.121%. The batch/depth comparison is confounded by update count: batch 16 receives 313 updates per epoch, batch 64 receives 79, and batch 256 receives 20. The result therefore does not isolate architecture quality from optimization budget.

### Sentence-pool comparison

The completed pool results are 5 runs; experiment 039 is the missing uniform 5,000-sentence result. The fixed full-pool reference is experiment 13 (alpha=1, learning rate=0.0003, window 4, 8 layers, batch 64), with validation word top-1 0.000% and secondary BCE 0.654452. The pool comparisons are:

| ID | pool | sampling | validation word top-1 ± SD | test word top-1 | validation BCE | test BCE |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 40 | 5000 | balanced | 0.000% ± 0.000% | 0.000% | 0.654565 | 0.656801 |
| 41 | 20000 | uniform | 0.000% ± 0.000% | 0.000% | 0.653174 | 0.654569 |
| 42 | 20000 | balanced | 0.000% ± 0.000% | 0.000% | 0.653567 | 0.652451 |
| 43 | 50000 | uniform | 0.001% ± 0.001% | 0.000% | 0.652881 | 0.654681 |
| 44 | 50000 | balanced | 0.000% ± 0.000% | 0.000% | 0.654484 | 0.655012 |

At 20,000 and 50,000 sentences, uniform and balanced sampling are close and word top-1 is zero or near zero in the completed pool runs. No monotonic pool-size improvement is established after ten sampled epochs.

### Main-grid marginal summaries

These summaries average over the other two main-grid factors; they are descriptive, not a factorial ANOVA.

Learning rate (word top-1):

| learning rate | mean validation word top-1 | minimum | maximum |
| ---: | ---: | ---: | ---: |
| 0.0001 | 0.000219% | 0.000197% | 0.000396% |
| 0.0003 | 0.000659% | 0.000197% | 0.002372% |
| 0.001 | 0.331266% | 0.261696% | 0.420160% |

Context alpha:

| alpha | mean validation word top-1 | minimum | maximum |
| ---: | ---: | ---: | ---: |
| 0.1 | 0.110077% | 0.000197% | 0.406208% |
| 1 | 0.106412% | 0.000197% | 0.387668% |
| 3 | 0.115656% | 0.000197% | 0.420160% |

Window:

| window | mean validation word top-1 | minimum | maximum |
| ---: | ---: | ---: | ---: |
| 2 | 0.135024% | 0.000197% | 0.420160% |
| 4 | 0.105227% | 0.000197% | 0.322222% |
| 8 | 0.091893% | 0.000197% | 0.296940% |

BCE remains available as a secondary loss-oriented view in `aggregate.csv`,
`aggregate.json`, and the BCE-specific figures.

### Test interpretation

The held-out test is appropriately used once after the final refit and is not a selection signal. Experiment 35 has the highest held-out word top-1 (1.068%), but this descriptive test ranking must not replace validation selection. Experiment 31 has the lowest test BCE (0.541256). Word top-1 is a strict metric here: all 14 predicted bits must be correct, so it is sparse; the ≥half-bit metric can be high even when exact words are rarely recovered. BCE is retained only as a secondary calibration/optimization diagnostic.

## Epoch-extension criterion and scope boundary

Commit `50cff5e` (“docs: make epoch extension criterion explicit”) defines a rule for the **semantic decoder** sweep: continue a configuration from 30 to 50 epochs only when the median of its three seeded `best_epoch` values equals 30. That rule is not applicable to this archive: `sweep.zip` contains the 45 production QCSE `experiment-000`…`experiment-044` folders, not the semantic `experience-*` folders, and its completed runs have a 10-epoch target. No semantic 30→50 extension decision can be inferred from these artifacts. The incomplete ID 039 should be resumed or rerun before any complete pool-size comparison is claimed.

## Limitations and recommended follow-up

The sweep uses one seed and two validation repeats, so uncertainty is only split-to-split variation, not a robust estimate across random initializations. The batch-size comparison changes the number of optimizer updates per epoch. The archive has no scheduler logs or exact commit metadata, and ID 039 is incomplete. For a publication-grade follow-up, repeat the top configurations across several seeds, equalize optimizer updates or total examples processed when comparing batch sizes, complete ID 039, and reserve an independent test set for the final locked comparison.

## Figures

All figures were generated from the same validated JSON/NPZ artifacts:

- [validation-top1-ranking](plots/validation-top1-ranking.png)
- [test-top1-ranking](plots/test-top1-ranking.png)
- [validation-bce-ranking](plots/validation-bce-ranking.png)
- [test-bce-ranking](plots/test-bce-ranking.png)
- [validation-metrics-heatmap](plots/validation-metrics-heatmap.png)
- [main-grid-validation-top1](plots/main-grid-validation-top1.png)
- [main-grid-validation-bce](plots/main-grid-validation-bce.png)
- [learning-rate-effect-top1](plots/learning-rate-effect-top1.png)
- [learning-rate-effect](plots/learning-rate-effect.png)
- [alpha-effect-top1](plots/alpha-effect-top1.png)
- [alpha-effect](plots/alpha-effect.png)
- [window-effect-top1](plots/window-effect-top1.png)
- [window-effect](plots/window-effect.png)
- [depth-batch-validation-top1](plots/depth-batch-validation-top1.png)
- [depth-batch-validation-bce](plots/depth-batch-validation-bce.png)
- [sentence-pool-validation-top1](plots/sentence-pool-validation-top1.png)
- [sentence-pool-validation-bce](plots/sentence-pool-validation-bce.png)
- [validation-test-top1-gap](plots/validation-test-top1-gap.png)
- [validation-test-gap](plots/validation-test-gap.png)
- [epoch-curves-top1](plots/epoch-curves-top1.png)
- [epoch-curves](plots/epoch-curves.png)
- [runtime-by-configuration](plots/runtime-by-configuration.png)
