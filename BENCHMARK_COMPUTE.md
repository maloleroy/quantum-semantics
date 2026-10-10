# DGX A100 duration estimate

Updated **2026-10-11** for the optional trainability matrix: **1,530 CV fits +
33 selected refits**, full 10,864-word vocabulary and both complete curated pools.
The user now reports a successful DGX focused-study pilot (`9115_0`), allocation
elapsed **7:46**, exit `0:0`, and one completed CV fit. That total includes
checks/profile/estimation/training; its phase timings and CUDA profile have not
been inspected here. The hypothetical full-matrix budgets below remain
uncalibrated and must not be inferred from this one pilot.
The next focused study adds detailed timing and a measured compute report;
see [BENCHMARK_TIMING.md](BENCHMARK_TIMING.md).

**Launch recommendation:** calibrate on the allocated device and run a focused
causal comparison first. The 1,530-fit matrix is an available exploratory grid,
not a justified minimum or the recommended first production submission. No
power calculation established a need for all its configurations and repetitions.

**Planning budget at four slots: about 56–115 hours at the 50-epoch cap, conditional on each
allocated device running the measured workload 2–5 times faster than this CPU.**
At CPU-equivalent speed, the same planning budget is about **214 hours / 8.9 days**.
These are scenarios, not measured GPU speedups or confidence bounds; a GPU can
also be slower than the CPU for these small, sequential operations.

## Allocation and scope

The current Slurm script requests **one 10 GB A100 MIG instance per fit** and the
submitter permits **four concurrent jobs**, split into two arrays of 765 tasks
with two slots each. Refits run after validation-only selection, with four slots.
This estimates that allocation, assuming slots remain available.

A complete [DGX A100 contains eight A100 GPUs](https://docs.nvidia.com/dgx/dgxa100-user-guide/introduction-to-dgxa100.html).
The current launcher fixes one 10 GB MIG per job and rejects resource or
concurrency overrides that would exceed this four-instance protocol. Full-GPU
allocations are outside this launcher and need separate timing/protocol decisions.
The code assigns each fit to one device; it does not distribute one fit over eight GPUs.

Each trainable epoch draws 5,000 examples with replacement from the complete
eligible pool, giving 79 optimizer updates at batch 64 and microbatch 8. Fifty
epochs are 250,000 draws per fit, not fifty exhaustive corpus passes. Evaluation
uses the prepared 2,048-example monitor and final 2,000 whole sentence groups.
The epoch scenarios assume every trainable fit and selected refit reaches the
stated count; early stopping and validation-selected refit budgets can reduce it.

## Fresh measurements

Remeasured **all 53 causal configurations** on current source, with four Torch
threads, one warmup and three timed repeats per training/evaluation batch.
Profile, prepared-manifest and current-source hashes match. Four CBOW checks
had training times within 4% of their causal equivalents; inference ratios were
0.98–1.11. Using causal timings for both tracks remains an approximation.

| Configuration | Training seconds / 64 | Evaluation seconds / 64 |
| --- | ---: | ---: |
| QCSE, 14 qubits, 2 layers | 0.749 | 0.305 |
| QCSE, 14 qubits, 8 layers | 2.627 | 0.826 |
| Hybrid, 4 qubits, 2 layers | 0.019 | 0.007 |
| Hybrid, 8 qubits, 8 layers, std 1 | 0.143 | 0.053 |
| BERT, width 16, 2 layers | 0.018 | 0.008 |

The old all-arm CPU profile was substantially slower: e.g. 3.29/11.10 seconds
for QCSE training at two/eight layers. Some historical timings overlapped tests;
the cause of the full discrepancy is not established. **Do not combine the old
1230-fit projection with the fresh speed factors or treat it as the current ETA.**
This profile measures batch work, not an end-to-end training job. It excludes
the production gradient logger, dataset/source hashing, checkpoints, preflights
and job startup. Peak CPU RSS does not establish CUDA memory feasibility.

## Duration scenarios

Training, monitoring, final inference and conservative refits together require
**136 / 317 / 619 four-thread CPU-reference job-hours** at 10 / 25 / 50 epochs.
Four ideal slots at 50 epochs yield about **155 hours before overhead**, or 77/31
hours at assumed 2×/5× speed. Refits use the costliest potential winner in each
family; the actual winners are unknown.

The following budgets additionally simulate task-order scheduling with the two
two-slot arrays and subsequent four-slot refits, assume **two minutes of fixed overhead
per job**, and add **25% allowance**. Both overhead choices are planning
assumptions, not measurements. Queue waits and human handoff delays are excluded.

| Completed epochs | CPU-equivalent device speed | Assumed 2× speed | Assumed 5× speed |
| --- | ---: | ---: | ---: |
| 10 | 60 h | 38 h | 25 h |
| 25 | 118 h | 67 h | 36 h |
| 50 | 214 h | 115 h | 56 h |

At four slots, every additional minute of fixed overhead across 1,563 jobs adds
roughly **6.5 wall hours** before scheduling tails and allowance. The repeated
allocation preflights therefore matter, especially for cheap models. An actual
Slurm task order can differ from the greedy scheduling simulation.

At 50 epochs, QCSE accounts for approximately **94% of CV batch work**. The ten
new hybrid controls add 300 fits but only **20.4 CPU-reference CV hours**, about
3.4% of the extended CV workload. Fit count alone exaggerates their cost.
The longest projected CV fit is **3.32 CPU-reference hours**, before overhead;
the 12-hour job limit still needs an allocated-device pilot.

Bootstrap resamples saved predictions and does not retrain models. A synthetic
timing check using the actual 2,000 test sentence groups, three seeds and three
metrics took **0.25 seconds for 2,000 draws**, approximately five seconds across
20 comparisons. This is execution timing only; report loading/validation is extra.
The separate larger-register barren-plateau scaling study is outside this batch.

## Historical cluster export and scope review

There is an existing cluster export: `../sweep.zip` (1,934,043,155 bytes).
Its previously validated aggregates and reports are tracked under
`results/cluster-sweep/`. A fresh metadata review confirmed **44 completed
CUDA configurations**, each with two validation fits and a development refit
at ten epochs: **132 completed fits**. Experiment 039 was incomplete; the 54
older/legacy directories are excluded. Summed configuration durations are
**60.59 hours**, and the median is **69.84 minutes per three-fit configuration**.
The sum is job-duration evidence, not the elapsed wall time of the parallel sweep.

For example, experiment 013 used the full corpus, window 4, eight layers and
batch 64 and took **69.90 minutes**. Experiments 035/036 used 64 layers at
windows 2/8 and took **129.02 / 243.70 minutes**, respectively. These durations
include exhaustive historical validation/test inference. The completed run
metadata records Torch 2.14.0+cu130, CUDA 13.0, simulation batch 64 and SPSA
perturbation 0.1. It does not record the GPU model/MIG identity, source commit
or scheduler logs, so exact DGX allocation attribution is unavailable.

This is actual CUDA runtime evidence for the historical pipeline. Its SPSA
gradient estimates, bit-BCE decoder, microbatching and evaluation schedule differ
from the current autograd word-loss benchmark. It neither supplies a measured
2–5× speedup for the new pipeline nor validates its conditional duration scenarios.

The full matrix repeats every diagnostic arm on five folds and three seeds in
both objectives. These fits answer different questions, and overlapping training
folds do not provide 15 independent repetitions. Broader searches also need care
with selection bias; see
[Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html).
Bootstrap operates on the saved predictions; it does not require extra training.

A more efficient sequence is:

1. Measure current allocated-device batch costs and end-to-end pilots for the
   expensive QCSE reference/depth arm and a learned hybrid/BERT reference.
2. Screen a small, hypothesis-driven causal set using validation only, equal
   tuning budgets and the complete corpus pool. Compare quantum models with
   appropriate identity/frozen and classical controls. Retain the held-out test.
3. Freeze 4–6 relevant configurations, then evaluate each on three seeds and
   five grouped folds: **60–90 CV fits**, followed by the selected family refits.
   This is a proposed scope, not a power-based guarantee or a prepared study.
4. Keep direct gradient-variance probes separate from model-quality comparisons;
   add bidirectional prediction and other ablations only for explicit questions.

Prioritize the expensive arms when reducing computation: QCSE uses approximately
94% of the current CV workload; removing cheap hybrid controls alone saves little.
Reusing the fixed outer test for adaptive redesign makes later claims exploratory;
an independent test is needed for confirmatory evaluation after such reuse.

The runtime defaults and existing prepared studies have not been altered by
this review. Use a fresh study with explicit arm/fold/seed/objective settings
when adopting a smaller protocol.

## Evidence and next measurement

Ignored local evidence: `outputs/compute-estimate-2026-10-09/`, containing raw
batch repeats, source hashes, four CBOW spot checks, `compute.json` and
`planning.json`. Prepared manifest:
`676beeec9b250267f5b95276191796c684f7fba9d6e6495a2c4a9dd35746efbd`.

That directory retains the historical ten-slot estimates. Four-slot projections
are saved separately under `outputs/compute-estimate-2026-10-11/`, reusing the
2026-10-09 CPU timings of unchanged model code. The old 22–47-hour recommendation
does not apply to the four-instance limit. The estimator now defaults to four
slots and records `concurrency`; slot-related JSON keys omit the old `ten` label.
Launcher/Slurm source changes require a freshly prepared production study.

Before relying on this ETA, prepare in the frozen DGX checkout, profile all
manifest arms inside the actual CUDA allocation using the commands in
[BENCHMARK.md](BENCHMARK.md), and run an end-to-end QCSE depth-8 pilot including
preflights, hashing, epoch saves and resume. Measure GPU memory and fixed job
overhead. Replace hypothetical speedups with allocated-device timings.
No runtime code changed for this estimate; no scheduler job was submitted.
