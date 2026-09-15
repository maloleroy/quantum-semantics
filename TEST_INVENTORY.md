# QCSE project test inventory

Updated 2026-09-15 on branch `qcse-classical-attention` at commit `01665e9`.
The documentation commit `50cff5e` (“make epoch extension criterion explicit”) is
an ancestor of this checkout. Counts below distinguish tests rerun now from
historical validation recorded in `PROGRESS.md`; historical suite counts are not
added together because later suites supersede earlier ones.

The active production launchers now target **50 epochs** for the QCSE causal sweep,
semantic encoding sweep, and quantum-attention runs. The archived cluster handoff
reviewed below is an older 10-epoch run and remains labelled as historical.

## Current local rerun

Command:

```text
.venv/bin/python -m pytest -q -rs
```

Result: **84 passed, 46 skipped in 29.89 s**. The 46 skips are the CUDA and MPS
parameterizations that cannot run on this Mac: 23 CUDA cases and 23 MPS cases.
There were no failures. The current environment reports Python 3.12.12, NumPy
2.5.3, Qiskit 2.5.2, PyTorch 2.14.0 without CUDA, pytest 9.1.1, Ruff 0.16.6,
and Pyright 1.1.411; MPS is built but unavailable.

| Test module | Collected | Coverage |
| --- | ---: | --- |
| [`tests/test_qcse.py`](tests/test_qcse.py) | 19 | CSV parsing, causal/CBOW boundaries, all context equations, angle padding, Qiskit circuit parity, bit order/marginals, persistence, training reproducibility, continuation and CLI end-to-end behavior |
| [`tests/test_data.py`](tests/test_data.py) | 7 | headers/Unicode, fixed global vocabulary, seeded uniform/balanced sampling, duplicate and sentence-group leakage, source-quota redistribution |
| [`tests/test_simulation.py`](tests/test_simulation.py) | 44 | batched tensor/Qiskit parity at 1/2/5/14 qubits and forward/reverse directions, invalid settings, runtime batch options, parallel SPSA, ten-epoch training parity, CLI resume |
| [`tests/test_cross_validation.py`](tests/test_cross_validation.py) | 19 | grouped outer holdout, five-fold and repeated shuffle-split semantics, fresh initialization, no test scoring in folds, interrupted resume, exact resume, bounded state cache, capped final validation/test export |
| [`tests/test_evaluation.py`](tests/test_evaluation.py) | 3 | deterministic seeded no-replacement evaluation subsets, uncapped/all behavior, invalid limits |
| [`tests/test_sampling.py`](tests/test_sampling.py) | 15 | replacement draw budget, partial batches, train-only sampling, stable monitoring subsets, sampled resume, invalid budgets |
| [`tests/test_outputs.py`](tests/test_outputs.py) | 3 | unique run folders, checkpoint-write failure preservation, interruption status |
| [`tests/test_semantic.py`](tests/test_semantic.py) | 4 | semantic prototype state/XYZ Qiskit parity, finite gradients, token-ID permutation equivariance, decoder/pathway controls |
| [`tests/test_attention.py`](tests/test_attention.py) | 6 | QCSE/classical attention encoding and postselection parity, gradients, causality, padding, checkpoint roundtrip and no-circuit control |
| [`tests/test_attention_pipeline.py`](tests/test_attention_pipeline.py) | 3 | attention resume equivalence for QCSE, classical and legacy semantic models, capped test exports |
| [`tests/test_sweep.py`](tests/test_sweep.py) | 7 | 45-config manifest, smoke extremes, nine five-run groups, top-1-first selection, evaluation defaults, fail-fast behavior, independent DGX array submission |
| **Total** | **130** | **84 passed locally; 46 unavailable-device skips** |

Additional checks rerun now:

- `.venv/bin/ruff check .`: passed.
- `.venv/bin/pyright`: 0 errors, 0 warnings, 0 informations.
- `bash -n scripts/*.sh slurm-*.sbatch`: passed.
- `git diff --check`: passed.
- `scripts/check_backend.py --device cpu`: passed at 14 qubits and 64 layers;
  maximum absolute error versus Qiskit and sequential evaluation was
  `3.3306690738754696e-15`.
- `sweep.zip` ZIP integrity: passed for all 2,022 archive entries.

The bounded-inference smoke used all five attention-CV setups with 1 epoch, 16
sentences, 3 replacement draws, and `--final-eval-examples 2`. It recorded 2 of
30 held-out test examples and 2 evaluation examples in each validation fold, with
the saved source/evaluation counts consistent. The CV resume tests exercised the
same cap for QCSE and the semantic pipeline tests exercised it for QCSE, classical,
and legacy semantic runs. After this change, the complete local suite remains
**84 passed, 46 skipped**; Ruff, Pyright, shell syntax, compilation, and diff
checks also pass.

## Actual 50-epoch local execution audit

To verify that the new target is executed rather than merely present in help text,
the following capped CPU runs were completed in `/private/tmp`:

- one causal QCSE sweep configuration: two validation-fold fits plus one development
  refit, **3 fits × 50 epochs**, 16 sentences and 17 sampled draws per epoch;
- all ten semantic configurations with seeds 42/43/44: **30 fits × 50 epochs**,
  16 sentences and the configured 5,000 replacement draws per epoch;
- all five attention-CV ablation setups: two folds plus one refit each,
  **15 fits × 50 epochs**, 16 sentences and 17 sampled draws per epoch.

All 48 capped fits reached histories `0..50`; losses/validation metrics were finite,
test scoring completed at the final target, and the causal artifact passed an explicit
sentence-group split check. These are execution and protocol audits, not full-corpus
performance claims: the QCSE production matrix still has 45 configurations and its
full 50-epoch CUDA execution remains unmeasured here.

The top-1-first selector was tested directly: a higher validation cosine/word top-1
beats a lower cross-entropy epoch or setup, with top-5, then CE, then earliest epoch as
deterministic tie-breakers. BCE/CE remain optimization diagnostics; validation retrieval
top-1 is the selection metric.

## Cluster-artifact validation added in this review

The archive validator is [`scripts/cluster_sweep_report.py`](scripts/cluster_sweep_report.py).
It reads `/Users/ethan/Documents/Ecole/3A/Filière recherche/Papers/sweep.zip`
without extracting the raw run folders and performs checks not covered by the
unit suite:

- exactly 44 completed target CV runs are separated from 54 legacy/non-CV attempt
  directories and the still-running experiment 039;
- all target arguments, output files, CUDA metadata, epoch targets and sampling
  budgets are consistent;
- development/test and fold IDs are unique and disjoint, and causal sentence
  groups do not cross outer or inner split boundaries;
- both complete validation reports match the CV summaries, including sample SDs;
- every fold history and aggregate history contains epochs 0–10 and its means/SDs
  are recomputed from the two folds;
- held-out probabilities are finite and in `[0, 1]`, exported example IDs match
  `splits.npz`, and all four test metrics are recomputed from the exported arrays.

Result: **44/44 completed runs passed, zero artifact-validation issues**. The
machine-readable result is [`results/cluster-sweep/archive-validation.json`](results/cluster-sweep/archive-validation.json).
This validates the saved cluster evidence; it is not a substitute for rerunning
the CUDA code on the original allocation, because the local machine has no CUDA.

For this QCSE archive, “top-1” is reported as `exact_word_accuracy`: the
thresholded 14-bit prediction must match the target word ID on every bit. The
reports now rank configurations by mean full-validation word top-1; BCE remains
available as a secondary diagnostic. This is distinct from the cosine top-1
metric used by the separate semantic/attention experiments.

## Historical validation recorded in the project ledger

The following milestones are documented in [`PROGRESS.md`](PROGRESS.md):

- Initial focused data/sweep checks: **12 passed** for data and manifest behavior.
- Earlier CPU/MPS/scheduling suite: **63 passed, 14 CUDA skipped**; included
  real-data MPS runs across objectives, datasets, depths, batch sizes, cleaning
  modes and sampling modes, plus finite archives, model/archive equality,
  standalone CPU inference and sentence-text split isolation.
- Longer grouped-CV validation: **72 passed, 17 CUDA skipped**; checked grouped
  isolation, validation coverage, fresh fold initialization, one-time test scoring,
  partial CV resume, streaming/resident cache parity and fold/refit artifacts.
- Sampled-epoch validation: **88 passed, 23 CUDA skipped**; checked 5,000-draw
  budgets, partial batches, train-partition bounds, fixed monitoring, no hidden
  full-corpus prediction, exact sampled resume, full validation recomputation and
  complete held-out exports.
- Semantic prototype: focused exact-state/Qiskit, gradient and permutation tests;
  CPU checkpoint reload, continuation, test scoring and word exports completed.
- Quantum attention: focused encoding/Qiskit, postselected block-encoding,
  gradient, causality/padding and resume checks; targeted local comparison runs
  completed. CUDA/MPS and Slurm execution were not locally available.

Static checks historically recorded alongside these milestones include Ruff,
Pyright, shell syntax, diff checks and the QCSE training-skill validator.

## Gaps that were made explicit or rechecked

The weak points are now called out rather than hidden behind a green unit-test
summary:

- Local CPU/MPS checks do not establish CUDA behavior. The saved cluster runs
  report CUDA 13.0 and PyTorch `2.14.0+cu130`, while this rerun cannot exercise
  those device branches.
- A completed marker alone is insufficient for a sweep result. The new validator
  requires the CV summary, two fold histories and full-validation files, refit,
  splits and held-out export, then recomputes metrics from arrays.
- Legacy and partial attempts in `sweep.zip` are excluded from rankings. Experiment
  039 remains incomplete and must be resumed/rerun before the 5,000-sentence
  uniform pool comparison is considered complete.
- The batch-size comparison is not compute-matched: with 5,000 draws per epoch,
  batches 16, 64 and 256 receive 313, 79 and 20 optimizer updates respectively.
  This is reported as a methodological confound, not as a robust batch-size claim.
- Sampled training epochs use `rng.choice(train_ids, 5000, replace=True)`. Exact
  example IDs may recur, and a sentence may contribute several sampled windows;
  all such repeats remain inside the training partition. Validation and test are
  not sampled for training, and the archive-level sentence-group check found no
  train/validation or development/test sentence overlap.
- The 30→50 extension rule from commit `50cff5e` applies only to the separate
  semantic-decoder sweep with three seeds and `best_epoch`; the 45-run production
  archive is a 10-epoch QCSE sweep and has no valid extension decision.
- The local 50-epoch audit uses small sentence caps and is therefore not evidence that
  the complete 202,172-sentence corpus trains or generalizes identically at epoch 50.
- QCSE production evidence still has one random seed and two validation repeats; this
  is weaker than a multi-seed publication-grade estimate. Batch-size comparisons also
  change optimizer-update counts for a fixed 5,000-draw epoch.

The new sweep reports and figures are collected under
[`results/cluster-sweep/`](results/cluster-sweep/).
