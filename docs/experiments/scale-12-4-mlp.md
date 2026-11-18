# Matched MLP control for the 12+4 quantum instance

Declared 18 November 2026 at the user's request, after the quantum result was
known. Compare the existing `adaptive_mlp` control with the locked
[12+4 adaptive quantum fit](scale-12-4.md#completed-results--audited-18-november-2026).
Do not change hyperparameters in response to this control's validation scores.

## Fixed comparison

- Use the exact prepared 4,096-token, 128-context WikiText-2 byte-BPE NPZ from
  quantum job 11727, SHA-256
  `9b499cfde14ff1e83f3ffcf7fa819516703b1896dd60313361ca8a8427f38544`.
- Reuse training-only tokenizer/statistics, official article partitions, UNK
  policy and the identical 524,288 training / 4,096 validation / 8,192 test rows.
  The test remains unscored.
- Model `adaptive_mlp`, configuration size 4,096, window 128, layers 2,
  ancilla-size feature setting 16, rank 128, recent window 16, phase scale 1.
  All other configuration fields match the locked quantum fit.
- The MLP uses the same types of prepared real amplitudes, local controls and
  spectral shifts, with independently learned preparation/controllers. It uses
  a 65,680 → 480 → 4,096 network with Tanh and softmax. The feature
  setting does not imply that the classical network contains qubits.
- **35,463,168 trainable parameters**, versus quantum **35,520,800**: 57,632 fewer,
  a **0.1623%** difference from integer hidden-width rounding. Encoders/controllers
  count toward both totals. No padding parameters are added to equalize counts.
- Seed **11**, float64/complex128 feature preparation, Adam constant .003,
  gradient clipping 5, no dropout and no factor normalization.
- Thirty epochs, batch **512**, **30,720 updates / 15,728,640 row exposures**.
  With the identical source code, seed and dataset, NumPy's training-row shuffle
  schedule matches quantum. Initialization draws differ between architectures.
- NLL loss; choose minimum validation NLL, then maximum sampling success, retaining
  the earliest epoch on an exact tie. There are the same 31 validation checks.
  Report selected most-likely-token accuracy, sampling success, NLL and bits per
  byte separately. Do not select by accuracy after looking at the results.

This is a single-seed, equal-exposure comparison with almost equal parameter
count and matching input features and optimizer settings. It is not a three-seed
comparison or a claim that the chosen hyperparameters are optimal for either model.

## Execution and checks

Use an isolated checkout `~/quantum-semantics-scale-12-4-mlp-20261118`, copying
the quantum run's source snapshot and locked CUDA dependencies. Preserve the
original quantum checkpoint and run. Generated tools are in
`output/scale_12_4_mlp/tools/`; results go to `output/scale-12-4-mlp/`.

Start with one `prod10` A100 `nvidia_a100_1g.10gb` slice and four CPU support
threads. A short verification job must pass the full GPU/project suite and six
full-batch Adam updates on the actual MLP, checking finite CUDA probabilities,
CUDA gradients, changed weights and exact parameter count. Keep resident data,
Adam moments and a best-model copy in the memory measurement. Require projected
training and scoring below 22 hours before submitting a 24-hour training job.
Preserve the batch/update budget if a larger allocation is needed.

The runner checks model and training metadata against the original quantum
record, as well as the exact dataset hash. Discard all preflight updates and
reinitialize seed 11 before training. Record live validation CSV, backward-pass
progress, device and peak-memory metadata, source/dataset hashes, NPZ weights
and final CSV history. Activate the laptop `dgx start` completion notifier.
Stop work after confirming training is ongoing; if allocation remains queued,
the user's earlier instruction allows leaving it pending without waiting an hour.

## Verified and launched — 18 November 2026

Short verification job **11816** completed with exit 0:0 in 24 seconds. All
**160 GPU/project tests passed**. Six full-batch CUDA Adam updates verified
finite probabilities and gradients, changed weights and exactly 35,463,168
parameters. The runner's checks confirm the dataset hash, quantum source hashes
and matching model/training metadata. Mean measured update time was **0.1665 s**,
projecting **1.44 hours** including scoring. Peak allocated/reserved memory was
**2.988 / 3.781 GiB** on one 9.5 GiB usable A100 MIG slice. This is a simulator
resource estimate, not a result about prediction quality or hardware speedup.

Full training job **11817**, `qcse-12plus4-mlp`, is **RUNNING** on `prod10`,
one 10 GB MIG slice, four support CPUs, 15 GiB host memory, 24-hour allocation.
It passed all 160 tests and the preflight again, discarded those updates, and
started the fresh seed-11 fit. At handoff, live progress confirms at least
**64 scientific backward passes**, finite CUDA gradients and peak reserved
memory **3.781 GiB**. Initial validation NLL is **8.30112** before training;
this is not a selected predictive result. The completion notifier is active.

Logs, progress and metadata are in the isolated remote checkout under
`output/slurm/` and `output/scale-12-4-mlp/`. Verification logs and preflight
metadata are also retained locally under `output/scale-12-4-mlp/retrieved/`.
Training results and the paired most-likely-token comparison remain pending.
