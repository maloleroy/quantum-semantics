---
name: qcse-training
description: Maintain and validate the quantum-semantics QCSE corpus, training, checkpoint, and Slurm pipeline. Use for this repository's experiments and backend debugging.
---

Read [PROGRESS.md](../../../PROGRESS.md) for completed work and measured limitations,
and [README.md](../../../README.md) for current commands. Prefer a small fix supported
by a reproducer; leave working simulation and optimization code alone.

Preserve these project invariants:

- Named dataset subsets select training sentences; vocabulary and frequency-based
  IDs use all three complete supplied files before filtering or sampling. `--data`
  instead defines an explicit custom corpus. Known CSV headers are not sentences.
- Keep duplicates and overlapping sentence windows in the same partition. The
  cluster protocol holds out 20% of sentence groups, runs five-fold CV within the
  remaining 80%, then refits on that full 80% and scores test once. Reset weights
  and optimizer for each fit; use validation metrics to compare configurations.
- Full-data profiles retain every curated sentence and token example. Sampling
  profiles are explicit. Source balancing must not select a shared sentence twice.
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
- The cluster sweep has 150 configurations, 150 epochs per fold/refit, in ten
  chains of three jobs, five configurations sequentially per job. Preserve
  `aftercorr` dependencies and the ten-job ceiling. Defaults are 12 hours,
  `prod10`, `gpu:nvidia_a100_1g.10gb:1`, and output/error files under `logs/`.

Use the existing tests in [tests/](../../../tests/) for changed behavior and
`scripts/check_backend.py` on the available device. Record actual commands/results
and remaining hardware limitations in PROGRESS; keep generated run outputs out of Git.
