# Reproducible QCSE versus BERT batch

Research findings, exact generator definitions and comparison limits are in
[BENCHMARK_RESEARCH.md](BENCHMARK_RESEARCH.md). Implementation progress and final
verification are tracked in [BENCHMARK_PROGRESS.md](BENCHMARK_PROGRESS.md).
[BENCHMARK_TIMING.md](BENCHMARK_TIMING.md) gives the focused six-arm timed study,
per-phase diagnostics, allocation accounting and same-arm compute forecasts.
[BENCHMARK_SCREEN.md](BENCHMARK_SCREEN.md) gives the longer fixed-budget,
three-fold/one-seed discovery blocks and a smaller confirmation stage.
This batch is independent of the earlier sweep outputs and checkpoint formats.

## Common protocol

Prepare the full deduplicated `phrases` + `cleaned` corpus once. Vocabulary and
frequency-ranked IDs use both complete files before filtering or sampling, as
in the existing project. This is a declared transductive vocabulary convention:
test text influences the vocabulary/ID ranking, but never gradient updates or
checkpoint selection. Input hashes and this convention are recorded. A fully
inductive vocabulary comparison would be a different experiment.

The outer sentence-grouped 80/20 split is fixed at seed 2026. Within development,
five exhaustive grouped folds each hold out one fifth of development groups.
Every arm trains every fold with seeds 42/43/44. Seeds vary initialization and
sampling, not the corpus or partitions. Causal and bidirectional tracks share
sentence partitions; their prediction examples differ, so scores across tracks
are not paired as if they predicted identical targets. Each fold starts from
fresh weights and Adam state. There is no selection of a favorable fold.

Every trainable fit targets at most 50 epochs, with 5,000 replacement draws
from its complete training pool per epoch, batch 64 and the final partial batch
retained: 79 optimizer steps per completed epoch. Simulation microbatches of
eight accumulate a batch-mean gradient before one optimizer update, bounding
statevector/softmax working memory without changing the optimizer batch.
Dropout draws can differ with microbatch shape, so stochastic BERT trajectories
are not promised to match a single full-batch forward pass.
No permanent training subset is imposed. Parameter-free controls do not train.

Checkpoint selection uses fixed validation-monitor samples of up to 2,048
examples: vocabulary logit top-1, then top-5, lower NLL and earlier epoch.
Early stopping begins after epoch 10, with eight epochs lacking retrieval
improvement. NLL-only tie-breaks do not reset patience. Epoch zero is a recorded
baseline, not a selectable trained checkpoint. All final fold scores use the
selected state and a fixed sample of up to 2,000 **complete sentence groups**;
`--final-eval-sentences 0` scores the entire split. The final cap is in sentences,
not the historical 10,000-token cap. Save exact evaluated IDs and split sizes.

`select` requires every planned CV result. For each model family and track, it
chooses a configuration using the mean across all seeds/folds, with the same
metric tie-breaks. It freezes `selection.json` and sets the refit epoch budget
to the integer median of that winner's selected epochs. Refit one fresh model
per seed on the complete development pool. Score held-out test once per refit,
only after selection; no test-driven tuning. Re-running completed tasks skips
them. Reports are descriptive if the same test is subsequently used to redesign
the study; use a new independent test before making confirmatory claims.

## Models and experiment matrix

List the executable manifest definitions:

```bash
.venv/bin/python scripts/robust_benchmark.py list
```

Both tracks include paper QCSE (Z marginals), a four-qubit learned semantic
hybrid (XYZ and tied word head), ordered classical MLP, mean-embedding control
and randomly initialized BERT MLM. BERT uses actual multihead softmax attention,
FFN/residual/LayerNorm blocks and a tied MLM head; width 16 or 32 and depths two
or four. The absent target is represented by a MASK slot, and PAD/MASK are
excluded from the prediction vocabulary. Bidirectional contexts retain left
and right slots, including boundary padding. Causal BERT sees only the prefix;
it is a controlled prediction baseline rather than the original BERT task.

QCSE and hybrid circuits compare CRZ chain/ring, ZZ chain, XX chain, RY/CNOT,
local rotations and identity, plus frozen-random CRZ and depths four/eight.
QCSE additionally tests all five existing context formulas, a fixed encoder-ID
permutation and the original-style bit BCE objective. The principal QCSE,
hybrid, MLP and width-16 BERT references each get LR 0.0003/0.003/0.01. Different
parameter counts and tuning search sizes are reported explicitly; this is not
an exactly parameter/FLOP-matched comparison. Alternative ansatzes may have
different counts and effective degrees of freedom.

The existing QCSE/classical quantum-attention × circuit/no-circuit arms join
the causal track with this same data, objective, selection and evaluation
protocol. Their original attention mask is causal; no bidirectional result is
invented for that architecture. Legacy results trained with other budgets or
cosine selection must remain historical evidence.

The generated matrix contains 43 causal and 39 bidirectional configurations:
82 × three seeds × five folds = **1,230 CV fits**. Six causal and five
bidirectional family winners need **33 selected refits**. Compute budgets:
The four parameter-free track/config pairs account for 60 zero-epoch fits.
The remaining 1,170 CV fits permit at most 4,621,500 optimizer updates /
292,500,000 draws, with
early stopping reducing that maximum. The exact array and fit counts are
printed from the manifest; use those if changing the arm subset or fold count.
Wall time remains a cluster measurement, particularly for 14-qubit/deep runs.

The optional `prepare --trainability` extension compares hybrid registers of
2/4/8 qubits, depths 2/8 and initialization standard deviations 0.05/1.0. It adds
28 configurations per track: 2,070 CV fits and the same 33 family refits. Four
of the new controls match identity-circuit learning rates and latent sizes. The
remaining additions include the focused width-32/q8 LR grid and controls in
[BENCHMARK_FOLLOWUP.md](BENCHMARK_FOLLOWUP.md). The standard matrix is unchanged.
[TRAINABILITY.md](TRAINABILITY.md) explains the
barren-plateau literature, diagnostics, comparison limits and mechanism probe.

## DGX A100 10 GB MIG handoff

Prepare on the **cluster checkout after the final commit**. Do not move a study
prepared on this Mac: provenance and input paths would then describe the wrong
host. The launcher verifies code, lockfile and prepared-artifact hashes, rejecting
mixed checkouts or altered splits. Source code must stay frozen until the study
finishes. The manifest records dirty state for transparency; a clean checkout is
preferred for the production run.

```bash
uv sync --locked
.venv/bin/python scripts/robust_benchmark.py prepare --output outputs/robust-benchmark
# Copy the unique directory printed above into STUDY:
STUDY=outputs/robust-benchmark/qcse-bert-<printed-run-id>
# prod10 accepts sbatch scripts only: validate/time one job before the batch.
.venv/bin/python scripts/submit_robust_benchmark.py "$STUDY" --stage pilot --dry-run
.venv/bin/python scripts/submit_robust_benchmark.py "$STUDY" --stage pilot
# Wait for the pilot to complete successfully before the following commands:
.venv/bin/python scripts/submit_robust_benchmark.py "$STUDY" --dry-run
.venv/bin/python scripts/submit_robust_benchmark.py "$STUDY"
```

Independent arrays have a combined **four-task concurrency ceiling**, one
configuration/seed/fold per job, no scheduler dependencies, `prod10`, four CPUs,
12 hours and `gpu:nvidia_a100_1g.10gb:1`. It preserves Slurm's CUDA mask, runs
`nvidia-smi`, the existing CUDA backend preflight and a forward/gradient CPU–device
parity preflight for all 71 standard/optional configurations in each allocation, and uses
the locked `.venv` directly. It does not install packages inside a job. Internal
threads default to the four allocated CPUs. Array-size/submit-count site limits
can be handled with `--array-size` (default maximum 1,000). The standard CV batch is two
balanced arrays of 615 tasks, each capped at **two** concurrent jobs. The optional
trainability batch uses two arrays of 765 tasks with the same two-job caps. Equal-sized
chunks prevent the smaller array from finishing early and halving utilization.
The 33 refits use one
array capped at **four**. `--concurrency` permits only 1–4; requests above four
are rejected. The partition and one 10 GB MIG per single-node job are fixed;
resource overrides cannot request extra GPU instances. Global task offsets preserve the manifest IDs. If more
chunks are requested than the concurrency budget, the launcher stops with an
explicit error; it never silently exceeds the ceiling. Verify site submit limits.
Fixed resources are passed directly to `sbatch`, and inherited `SBATCH_*`
environment overrides are removed from submission subprocesses.

Submit from one shared checkout. A local file lock prevents simultaneous
submissions there, and the launcher refuses a new batch while any of your
unfinished `prod10` jobs remain in `squeue`. This includes batch pilots and prior
arrays, preventing sequential duplicate submissions from multiplying the cap.
Wait for the pilot job to finish before submitting CV, and wait for CV jobs to finish
before submitting refits or retrying. Finished entries retained by the controller
do not block submission. Pending, running, configuring, completing, suspended,
requeued and unknown states still block. Failed scheduler queries stop
submission. [Slurm's queue filters](https://slurm.schedmd.com/squeue.html) are
used for this check. Dry runs do not contact Slurm.
Manual GPU jobs or simultaneous submissions from separate checkouts are outside
this guard; keep them separate from this workflow. An account-wide scheduler
limit would require a site-managed QoS/association rule.

For a study already prepared with the earlier launcher, keep its checkout
frozen. Fetch the deployment fix into the ignored `outputs/` directory, which
is outside the study's source-hash inputs:

```bash
git fetch origin
git show origin/codex/qcse-bert-robust-benchmark:scripts/submit_robust_benchmark.py \
  > outputs/submit_robust_benchmark.py
.venv/bin/python outputs/submit_robust_benchmark.py "$STUDY" --concurrency 4
```

This copied launcher resolves the same repository root, validates the unchanged
manifest/artifacts, applies the queue/lock/resource guards and submits the
original frozen Slurm/training files. Each receipt records the executed launcher
hash and saves its source under `STUDY/submissions/launcher-TIMESTAMP.py`.
Use the copied launcher for later refits too. Pull the branch after the study
finishes; newly prepared studies can use the updated `scripts/` launcher directly.

The site's `prod10` submission policy rejects interactive allocations. Use
`--stage pilot`, which submits one array task (`0-0%1`) requesting one 10 GB MIG,
through the same queue check, lock and receipt path. After both CUDA preflights,
it profiles all prepared arms for task 0's objective, writes `STUDY/profile-OBJECTIVE.json` and
`STUDY/compute.json` for four concurrent jobs, then runs manifest task 0 with
checkpoint/resume enabled. Its profile must cover every arm needed by the
estimate. The follow-up prepares causal and CBOW in separate studies so each
gets its own objective-specific calibration. Its completed task 0 is reused
when the CV batch starts. Check the pilot's scheduler exit status and both
`logs/qcse-bert-JOBID_0.out` and `.err`; an empty queue alone does not prove success.
Run `uv sync --locked` on the login node before submission. The batch job uses
that environment without installing packages; CUDA checks/training run only
inside the allocation.

Monitor and resume unfinished fits by resubmitting the same study; completed
tasks skip, and epoch checkpoints restore Adam, sampler RNG, Torch/CUDA/MPS RNG,
patience and the selected state. Interrupted partial epochs are replayed from
the preceding atomic checkpoint. CUDA reproducibility is limited to the same
software/hardware settings; no cross-device bit equality is promised.

```bash
.venv/bin/python scripts/robust_benchmark.py status "$STUDY"
.venv/bin/python scripts/robust_benchmark.py select "$STUDY"
.venv/bin/python scripts/submit_robust_benchmark.py "$STUDY" --stage refit --dry-run
.venv/bin/python scripts/submit_robust_benchmark.py "$STUDY" --stage refit
# After every refit completes:
.venv/bin/python scripts/robust_benchmark.py report "$STUDY" --bootstrap-draws 2000
```

Resource overrides use structured options, for example
`--sbatch-option=--time=24:00:00`. Array IDs, exports and dependencies are fixed.
GPU count and partition overrides are rejected.
Submission receipts are retained under `STUDY/submissions`, and Slurm output/error
under `logs/qcse-bert-%A_%a.*`. If a deep arm exceeds 10 GB, prepare a new study
with a smaller `--microbatch-size`; do not change training batch or silently
continue an incompatible manifest.

## Saved metrics and uncertainty

Each fit retains runtime/scheduler/code provenance, epoch metrics, gradient norm
from the final batch, sampled draws/update counts, selected/stopped epochs,
online training loss and circuit-only per-layer gradients averaged before clipping,
parameter counts, wall seconds and CUDA peak allocated memory. `checkpoint.pt`
is the last resumable epoch; `selected.pt` is the chosen portable inference state.
`predictions.npz` stores IDs, sentence groups, positions, targets, predicted top-5
IDs, per-example NLL/top-1/top-5 and the training-frequency baseline. Eight decoded
examples support inspection. The full word softmax is used for CE; the paper
adapter reports valid-word normalized independent-bit likelihood explicitly.
These logits differ from cosine retrieval, Hamming similarity and true joint
Born probabilities. No metric is renamed to hide the difference.

Selection, completed-fit reuse and reporting validate saved prediction IDs,
groups, targets, ranks, aggregates, train-only frequency scores and update/draw
ledgers against the prepared splits. Each CLI attempt retains its status under
`attempts/` and each started fit retains runtime provenance under
`runtime-attempts/`. Result seconds include setup and saved completed work;
failed partial-epoch work and scheduler preflights require attempt/Slurm records.
CUDA reserved peak memory is saved alongside allocated peak memory.

The report exports per-refit CSV, JSON aggregates and Markdown with paired
differences to BERT and the frequency baseline. Bootstrap draws resample seed
pairs and complete sentence groups with replacement; tokens within a sentence
stay together and token-weighted denominators retain group size. Predicted
examples/targets must align exactly across compared models. The 95% percentile
intervals are exploratory and conditional on the fixed split, chosen models
and evaluated groups. Fold/seed SD is descriptive: overlapping training folds
are not independent replicates. Three seeds give limited information about
optimization variability. No multiplicity correction or significance claim is
attached to the large exploratory grid.

Keep the entire study folder and scheduler logs backed up on cluster storage.
Generated fits/reports live under ignored `outputs/`; code, manifest definition,
research/protocol and validation records are versioned. Publish curated reports
with their manifest fingerprint, source commit and explicit sample budgets.
Every prepared study also retains a verified snapshot of the effective source
and lockfile under `source/`, including any declared uncommitted changes.

## Small end-to-end protocol check

This retains the full shared vocabulary but caps the sentence pool and budgets.
It is pipeline validation, not an NLP result:

```bash
.venv/bin/python scripts/robust_benchmark.py prepare --output outputs/robust-smoke \
  --arms qcse-crz-chain hybrid-identity bert-16 mlp \
  --max-sentences 32 --folds 2 --seeds 42 --epochs 1 --samples-per-epoch 9 \
  --batch-size 4 --microbatch-size 2 --eval-examples 8 \
  --final-eval-sentences 4 --min-epochs 1 --threads 1
# Use the printed unique STUDY directory. Run every task listed by status:
.venv/bin/python scripts/robust_benchmark.py fit "$STUDY" --task-id 0 --device cpu
```

Use `select`, all printed refit IDs and `report` exactly as for the full study.
Local CPU checks cannot establish CUDA success or full-corpus completion time.

## Audit and runtime calibration

`BENCHMARK_STATE_REPORT.md` records the independent full-data audit, verified
limitations and measured CPU budget scenarios. These tools retain JSON evidence:

```bash
.venv/bin/python scripts/audit_prepared_benchmark.py "$STUDY"
.venv/bin/python scripts/profile_robust_benchmark.py "$STUDY" --device cuda \
  --output "$STUDY/profile-causal.json"
.venv/bin/python scripts/profile_robust_benchmark.py "$STUDY" --device cuda \
  --objective cbow --arms qcse-crz-chain qcse-depth-8 hybrid-crz-chain bert-16 \
  --output "$STUDY/profile-cbow.json"
.venv/bin/python scripts/estimate_benchmark_compute.py "$STUDY" "$STUDY/profile-causal.json" \
  --output "$STUDY/compute.json"
```

Run CUDA profiling inside an actual allocation. The estimate script uses the
same tensor dimensions for both tracks and the costliest potential family
winner for refits. It emits explicitly unmeasured speed-factor scenarios when
given a CPU profile; startup, preflight, I/O, queueing and retries are excluded.
If given a CUDA profile, its base costs describe that allocated device; do not
interpret additional speed-factor scenarios as another measured acceleration.
