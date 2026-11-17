# 12 vocabulary qubits and 4 ancilla qubits

Declared 17 November 2026 at the user's request. This is a single-seed scaling
and training feasibility experiment, not a controlled demonstration of a benefit
from more vocabulary or ancilla qubits. Predictive results are pending.

## Fixed protocol

- Model: `adaptive`, direct Born word measurement, no classical vocabulary head.
- Vocabulary: 4,096 byte-BPE outcomes (12 qubits); ancilla dimension 16 (4 qubits).
- Two Hamiltonian layers, rank 128, phase scale 1, recent window 16, context 128.
- Seed 11; float64/complex128; Adam at constant .003; gradient clipping at 5.
- Thirty epochs, 524,288 training examples per epoch, batch 512: 30,720 updates.
- NLL loss and validation NLL checkpoint selection; factor normalization disabled.
- Pinned WikiText-2 raw snapshot from `data/wikitext-2-raw-v1/manifest.csv`.
  Train tokenizer/statistics on training articles only. Preserve official article
  partitions, normalized overlap guard, lossless byte encoding and UNK policy.
- Prepare 524,288 train, 4,096 validation and 8,192 test targets using the existing
  deterministic preparation. Test is stored but never scored by this job.
- Report token NLL, bits per byte, sampling success and argmax separately. Different
  vocabulary token NLL is not directly comparable to previous 512-token runs.

## Execution

Isolated DGX checkout: `~/quantum-semantics-scale-12-4-20261117`.
Generated runner: `output/scale_12_4/tools/run.py`; execute with
`uv run python -m output.scale_12_4.tools.run` from the checkout root.

Use one `prod10` A100 `nvidia_a100_1g.10gb` slice, four CPU support threads,
24-hour allocation. An actual CUDA preflight uses the full model and batch,
resident training data, Adam moments and a saved-best model copy. Six optimizer
steps verify CUDA probabilities, finite CUDA gradients and changed weights;
measure the last four steps and a validation forward pass. Discard the preflight
model and optimizer and reinitialize seed 11 before the scientific fit.

Record peak allocated/reserved VRAM and project all 30,720 updates, per-epoch
validation and final scoring. Require a projection below 22 hours before training.
If this allocation fails memory or time requirements, retain the scientific
configuration and document a justified `prod80` full-A100 allocation instead.
No automatic batch, rank or update-budget reduction.

Record source/dataset hashes, GPU metadata, preflight JSON, live metrics CSV and
backward-pass progress JSON. Final weights and metric arrays use NPZ; final
histories use CSV. Selection is locked before any future held-out evaluation.
Start the local `dgx start` notifier after submission and stop work after
verification that CUDA training is ongoing, as requested.

## Submission

Slurm job **11711**, `qcse-12plus4`, submitted on 17 November at 15:35 DGX
time to `prod10` with one 10 GB slice, four CPUs and 15 GiB host memory.
Initially queued for resources; all seven 10 GB slices were occupied.
The local completion notifier is active. Source corpus hashes are verified
against the pinned manifest during data preparation. Project tests run inside
the GPU allocation before preflight. Data preparation can run ahead of the
allocation under a shared setup lock; model computation runs only on CUDA.

The user subsequently requested stopping once preparation is complete even if
the job remains queued. At handoff, job 11711 is pending resources; GPU tests,
preflight and scientific training have not yet run. The queued launcher performs
these gates automatically. Data-only tokenizer preparation is already in progress;
the shared lock prevents duplicate preparation if the GPU job starts meanwhile.

## Setup failure and resubmission

Job 11711 started at 15:53:24 DGX time on 17 November and failed after 15 seconds
with exit 1:0. The GPU test suite reported 96 passes and 62 setup errors because
`data/phrases.csv`, required by its fixtures, had not been copied to the isolated
checkout. The experiment's GPU preflight and scientific training never started;
this failure provides no evidence about model memory or throughput.

Data preparation separately completed. The 4,096-token dataset SHA-256 is
`9b499cfde14ff1e83f3ffcf7fa819516703b1896dd60313361ca8a8427f38544`.
Copied the missing fixture, refreshed the input hash manifest, and resubmitted
the unchanged protocol as job **11716**. It is pending resources at handoff;
the same test and resource gates will run automatically. The local notifier is
active. No test scoring or scientific optimizer updates have occurred.

## Remove the sentence corpus dependency

At the user's request, delete the old sentence CSV from the repository and this
experiment's checkout. Tests now generate a tiny temporary text corpus through
a shared fixture and no longer require a research dataset. `prepare` requires
an explicit `--data` argument; usage examples select WikiText.
Temporarily hold queued job 11716 while synchronizing these changes, then release
the same job with its original WikiText dataset, model and training budget.

## CUDA test failure, verified correction and relaunch

Job 11716 started at 16:16:26 DGX time and failed after 13 seconds, exit 1:0.
The suite passed 157 tests and failed the `prior` and `prior_position` CUDA
checkpoint tests on argmax accuracy arrays. Their probability, gradient and
checkpoint comparisons passed. The generated corpus produces tied priors;
identity dynamics and device roundoff can split those ties, changing the exact
tie-count accuracy from 1/2 to 1. Scientific training and its resource preflight
had not started.

The cross-device test now exercises nonzero learned dynamics for every prior
model and explicitly checks a margin above 1e-10 before comparing argmax scores.
A separate CUDA test checks analytically specified exact ties. Production models,
scoring rules, tolerance thresholds and experiment settings are unchanged.

Before submitting another training job, verification job **11726** ran the full
suite and actual 12+4 preflight on one `prod10` 10 GB slice. It completed with
exit 0:0 in 23 seconds: **160 tests passed**, all six full-size Adam updates
completed with finite CUDA gradients and changed weights. The measured model has
35,520,800 parameters; peak allocated/reserved memory was **6.123/6.811 GiB**
on the 9.5 GiB usable slice. Mean measured update time was **0.6845 seconds**;
projected fit time including scoring was **6.04 hours**, an estimate rather than
a completion guarantee. Preflight weights and optimizer are discarded.

The unchanged scientific fit is resubmitted as **11727**. Its launcher repeats
all CUDA tests and resource checks before fresh seed-11 training. Original
logs and verified preflight metadata are retained locally under
`output/scale-12-4/retrieved/`. The completion notifier is active.
