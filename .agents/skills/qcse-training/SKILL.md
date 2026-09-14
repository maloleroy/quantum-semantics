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
- Keep duplicates and overlapping sentence windows on the same side of the split.
  Source-balanced sampling must not select a deduplicated shared sentence twice.
- Adam batch size changes training; simulation batch size only bounds simultaneous
  contexts. GPU float32 results are compared with tolerances against Qiskit/main.
- New train/prepare invocations create unique output folders. Continuation defaults
  to the source run; explicit `--output` forks its artifacts. Preserve the last valid
  archive if saving fails, and keep vocabulary, split, RNG, and optimizer resumable.
- Validate CUDA inside an actual GPU allocation. CPU/MPS passes cannot establish
  CUDA success. Preserve Slurm's device mask, including MIG UUIDs; use the locked
  checkout environment and the existing backend preflight before changing kernels.
- The cluster sweep has 150 experiments in ten chains of three jobs, five runs
  sequentially per job. Preserve `aftercorr` dependencies and the ten-job ceiling.

Use the existing tests in [tests/](../../../tests/) for changed behavior and
`scripts/check_backend.py` on the available device. Record actual commands/results
and remaining hardware limitations in PROGRESS; keep generated run outputs out of Git.
