# Detailed timing for the focused QCSE/BERT study

The completed timed study kept the same six causal arms, five grouped development folds,
three seeds, full corpus/vocabulary, 5,000 draws per epoch and 50-epoch cap:
**90 CV fits, then 12 selected refits**. The user chose timing and diagnostics
before expanding the model grid. Data, losses, Adam updates, early stopping and
test selection rules are unchanged.

Its [measured DGX results](results/robust-timed-cv/README.md) are now available.
The next [staged long-budget study](BENCHMARK_SCREEN.md) changes the training
budget, scoring policy and discovery repetitions; the commands below document
the earlier timed protocol. Do not merge those two protocols' results.

## Measurements

Every new fit attempt writes `timing-attempts/ATTEMPT.json`. It records manifest
verification, model/data setup, resume, initial validation, each training epoch,
validation, checkpoint selection, checkpoint I/O, final evaluation and artifact
serialization separately. Compute phases synchronize CUDA/MPS at their
boundaries; launch latency is not reported as completed GPU computation. Timing
uses a monotonic clock, records failures/interruption, and does not sample RNG.
An abruptly killed process may leave a running phase with no duration. Each
resume retains a separate attempt instead of overwriting previous measurements.

The logger writes at phase boundaries, not per optimizer step. Its metadata I/O,
unclassified Python work and initial device startup contribute to total job/fit
duration; phase sums do not exhaust the total. Epoch checkpoint contents retain
the original optimizer/sampler/dropout RNG semantics. Portable inference
checkpoint writing is now included in `result.json` fit seconds.

`history.json` adds mean/max unclipped optimizer-batch gradient norm and the
fraction of batches clipped at the existing threshold of five. Existing per-layer
circuit RMS/near-zero diagnostics still exclude known structurally dead QCSE
parameters. A flat validation curve or small gradient alone does not establish
a barren plateau; this six-arm study does not vary register size.

`runtime.json` records Torch thread count and CUDA-visible device name, UUID
when exposed, memory capacity, multiprocessor count and compute capability.
Fit results retain peak allocated/reserved CUDA memory; attempt timing records
retain process peak RSS. These are process/allocator measurements, not device-wide
utilization or total GPU memory consumption.

`job-timings/` records device inventory, both CUDA preflights, pilot profiling and
projection, and the fit subprocess separately, including failed attempts and
whether a completed result was reused. This measures job work after the wrapper
starts. Slurm accounting supplies full allocation elapsed time, including batch
startup/cleanup outside the wrapper. Queue waiting is separate.
[Slurm documents `ElapsedRaw` as allocation elapsed seconds](https://slurm.schedmd.com/sacct.html).

The batch profiler now honors the prepared optimizer batch size and executes
the circuit gradient diagnostic used during training. Profiler/estimator/job
wrapper source is included in new source snapshots. No packages install inside
allocations, and Slurm's CUDA/MIG mask is preserved.

## Switch to the new study

Keep the old checkout frozen while its jobs are active. If stopping it, cancel
only its actual array ID and wait until no jobs are running, queued, configuring
or completing; `COMPLETED` entries may remain and do not block the corrected
launcher. Keep the old study outputs. Pull the updated branch only after stopping
or finishing that study. Do not edit its manifest or mix old checkpoints into
the new study.

```bash
git pull --ff-only
uv sync --locked
PREPARED_STUDY=$(
  .venv/bin/python scripts/robust_benchmark.py prepare \
    --output outputs/robust-timed \
    --objectives causal --folds 5 --seeds 42 43 44 --epochs 50 \
    --arms qcse-crz-chain qcse-xx-chain \
           hybrid-crz-chain hybrid-identity mlp bert-16
) &&
STUDY="${PREPARED_STUDY##* CV tasks at }" &&
test -f "$STUDY/manifest.json" &&
printf '%s\n' "$STUDY" | tee outputs/robust-timed-study.txt
```

Run the above blocks separately and stop on any error. Submit the one-instance
batch pilot; the launcher validates source/artifact hashes and blocks overlapping
unfinished `prod10` jobs:

```bash
.venv/bin/python scripts/submit_robust_benchmark.py "$STUDY" --stage pilot --dry-run
.venv/bin/python scripts/submit_robust_benchmark.py "$STUDY" --stage pilot
```

After its actual job ID reports `COMPLETED` with exit `0:0`, submit CV:

```bash
.venv/bin/python scripts/submit_robust_benchmark.py "$STUDY" --concurrency 4
```

There is still exactly one 10 GB MIG per job and at most four concurrent jobs.
The pilot fit is reused. Selection/refit/bootstrap commands remain as documented
in `BENCHMARK.md`; they require every planned CV/refit result respectively.

## Report and another batch

### Parameter-free controls in an already prepared study

The 156-fit screen includes parameter-free identity and frozen QCSE controls.
These have zero optimizer steps and do not write `history.json`.
The original timing reporter incorrectly required that file for every fit;
successful CV selection still validates their predictions and budget ledgers.
The corrected reporter permits missing history only for zero-parameter,
zero-epoch results, reports clipping as unavailable and retains their measured
evaluation cost without extrapolating training epochs.

Keep the study's checkout and source snapshot frozen. Fetch the corrected
analysis module into ignored `outputs/` and run it there, using the original
package to validate the unchanged manifest/artifacts:

```bash
git fetch origin &&
git show origin/codex/qcse-bert-robust-benchmark:src/qcse/benchmark_compute_report.py \
  > outputs/report_benchmark_timing.py &&
.venv/bin/python outputs/report_benchmark_timing.py "$STUDY" \
  --concurrency 4 --slurm-accounting "$STUDY/slurm-accounting.psv"
```

The report records the executed reporter's SHA-256 alongside the original
study fingerprint. No prepared source hashes, histories, predictions or
metrics need editing, and no training is repeated. Newly prepared studies
can use the normal `scripts/report_benchmark_timing.py` entry point.

### Allocation accounting and projections

Read the allocation start date from the receipts and replace `YYYY-MM-DD` below.
Only accounting records whose job IDs match this study's ledgers are used;
`.batch` and `.extern` steps are not counted again:

```bash
sacct --starttime YYYY-MM-DD --name qcse-bert --parsable2 \
  --format=JobID,State,ExitCode,ElapsedRaw \
  > "$STUDY/slurm-accounting.psv"
.venv/bin/python scripts/report_benchmark_timing.py "$STUDY" \
  --slurm-accounting "$STUDY/slurm-accounting.psv"
# Prospective 30 CV fits per arm: 180 additional CV fits of the same protocol.
.venv/bin/python scripts/report_benchmark_timing.py "$STUDY" \
  --slurm-accounting "$STUDY/slurm-accounting.psv" --fits-per-arm 30
```

`timing-report/` contains JSON, per-fit CSV and a Markdown report. Reports can run
on partial results, but projections wait until each arm has a completed fit and
use one device type. They show observed-length median/P90 scenarios and a
per-epoch extrapolation to the configured epoch cap. P10/P90 are descriptive
quantiles, not confidence or prediction bounds; a single run offers weak
calibration. The scheduler replay uses the launcher's balanced array chunks and
four-slot ceiling and assumes available slots and task-order scheduling.

Pilot profile/estimate time is excluded from repeat-fit forecasts. Failed
recorded job attempts are retained; reused/skipped tasks are excluded from
per-fit calibration. Without accounting, wrapper time omits allocation
startup/cleanup; without job ledgers, fit time also omits preflights. Incomplete
records can miss lost work. Selected refits, queue waits and human handoff are
outside these **additional CV** forecasts. Read the scope in `summary.json`.

Use phase costs to prioritize a short profile: compute-heavy training,
validation overhead, checkpoint I/O, or available memory for a different
simulation microbatch. Microbatch changes need a new pilot and study; preserving
optimizer batch does not guarantee identical BERT dropout draws. New circuits,
qubit counts, corpus sizes or GPU partition types need fresh calibration.
Choose further experiments using development results and keep the held-out test
out of tuning.
