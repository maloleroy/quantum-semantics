---
name: qcse-training
description: Maintain and validate the quantum-semantics QCSE corpus, training, checkpoint, and Slurm pipeline. Use for this repository's experiments and backend debugging.
---

Read [PROGRESS.md](../../../PROGRESS.md) for completed work and measured limitations,
and [README.md](../../../README.md) for current commands. Prefer a small fix supported
by a reproducer; leave working simulation and optimization code alone.

Preserve these project invariants:

- Named dataset subsets select training sentences; vocabulary and frequency-based
  IDs use both complete active files, phrases and cleaned sentences, before filtering
  or sampling. Exclude Tatoeba: cleaned sentences are its curated version. `--data`
  instead defines an explicit custom corpus. Known CSV headers are not sentences.
- Keep duplicates and overlapping sentence windows in the same partition. The
  cluster protocol holds out 20% of sentence groups, runs five-fold CV within the
  remaining 80%, then refits on that full 80% and scores test once. Reset weights
  and optimizer for each fit; use validation metrics to compare configurations.
- Full-data profiles keep every curated sentence/token example eligible. The
  cluster draws 5,000 training examples with replacement per epoch, independent
  of batch size; retain the final partial batch. Do not replace the full pool with
  a permanent tiny subset. Source balancing must not select a shared sentence twice.
- Sampled epochs monitor fixed, independently seeded subsets (2,048 per split by
  default), then score each complete validation fold once and the full held-out
  test after refit. Distinguish sampled curves from full final scores. Keep epoch
  saves free of full-corpus inference; sampled checkpoints omit embeddings while
  retaining the sampler RNG. Old runs preserve their saved full-pass schedule.
- Adam batch size changes training; simulation batch size only bounds simultaneous
  contexts. GPU float32 results are compared with tolerances against Qiskit/main.
  The bounded state cache preserves encoding and predictions; it does not bound
  host corpus/embedding arrays or guarantee full-data runtime within a Slurm job.
- New train/prepare invocations create unique output folders. `resume-cv RUN_DIR`
  resumes unfinished fits to the saved epoch target and skips completed ones.
  Single-fit `continue` adds epochs, defaults to the source run, and forks when
  `--output` is explicit. Preserve split/evaluation semantics, vocabulary, RNG,
  optimizer, and the last valid archive if saving fails.
- Validate CUDA inside an actual GPU allocation. CPU/MPS passes cannot establish
  CUDA success. Preserve Slurm's device mask, including MIG UUIDs; use the locked
  checkout environment and the existing backend preflight before changing kernels.
- The cluster sweep has 75 causal configurations: 45 alpha/LR/window combinations,
  24 depth/batch comparisons, and six sentence-pool sampling comparisons. Keep 150
  epochs per fold/refit and five configurations per job; submit ten jobs then five
  dependent jobs. Preserve
  `aftercorr` dependencies and the ten-job ceiling. Defaults are 12 hours,
  `prod10`, `gpu:nvidia_a100_1g.10gb:1`, and output/error files under `logs/`.

Use the existing tests in [tests/](../../../tests/) for changed behavior and
`scripts/check_backend.py` on the available device. Record actual commands/results
and remaining hardware limitations in PROGRESS; keep generated run outputs out of Git.
