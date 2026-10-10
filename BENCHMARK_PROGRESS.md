# QCSE/BERT benchmark progress — 2026-10-08

Branch: `codex/qcse-bert-robust-benchmark`, based on `2e037ca`.
Existing uncommitted focused-benchmark code, reports and images are preserved;
they are not part of these commits. No cluster job has been submitted.

## Completed review

- Read README, PROGRESS, FOCUSED_BENCHMARK, SEMANTIC_PROTOTYPE, data/split,
  context, circuit, simulator and attention implementations and DGX launchers.
- Read earlier chats “Analyze sweep results and plots”, “Build robust training
  pipeline” and “Implement attention configurations”. Preserve full curated pool,
  shared vocabulary, sampled epochs, all-fold reporting, independent array tasks,
  prod10 / A100 10 GB MIG allocation and CUDA preflight.
- Read `../QCSE_ideas.md` graph/Hamiltonian proposals and the XX interaction
  discussion in `../french/paper1.md`. These are hypotheses, not measured NLP models.
- Checked primary QCSE, BERT, QNLP/lambeq, ansatz/trainability and NLP statistics
  literature. Findings and protocol go in BENCHMARK_RESEARCH.md / BENCHMARK.md.

## Implementation ledger

- [x] Exact QCSE adapter, BERT architecture, classical controls, circuit variants.
- [x] Shared prepared corpus/splits, training, resume, per-example metrics.
- [x] Validation-only selection, development refits, paired sentence bootstrap.
- [x] Frozen large manifest, independent DGX arrays and retained job provenance.
- [x] Final validation and measured limitations.

## Final validation

Checks were deferred until implementation was complete, as requested.

- `.venv/bin/python -m pytest -q -rs`: **102 passed, 46 skipped**, 87.19 seconds.
  Skips are CUDA/MPS cases; neither backend is available in this environment.
- After adding Slurm chunking/source snapshots, `OMP_NUM_THREADS=1
  .venv/bin/python -m pytest -q tests/test_robust_benchmark.py`: **14 passed**,
  24.39 seconds (includes the new array-chunk test). The changed protocol tests
  verify Qiskit parity for all seven gate families, exact paper marginals and
  dead final diagonal gradients, BERT target masking and tied head, grouped
  splits, paired bootstrap, full CV/refit/report execution and corrupted-input
  rejection. Interrupted BERT training resumes with bit-identical CPU weights
  and Adam state, including dropout/sampling RNG and the final partial batch.
- `.venv/bin/ruff check .`, `.venv/bin/pyright`, shell syntax for all existing
  launchers/batch files and `git diff --check` pass after final corrections.
- `scripts/check_backend.py --device cpu`: 14 qubits / 64 layers, maximum
  Qiskit error `3.219646771412954e-15`, sequential error `3.3306690738754696e-15`.
- `scripts/check_benchmark_backend.py --device cpu`: all **43 adapter
  configurations** passed forward/gradient parity and finite-gradient checks.
  The same check executes on CUDA in every actual DGX allocation.
- Real-corpus end-to-end smoke: **16 CV fits + eight refits**, both prediction
  tracks, 32 curated sentences, full **10,864-word vocabulary / 14-qubit QCSE**,
  one epoch, nine draws, batch four, simulation microbatch two, eight validation
  monitoring examples and four final sentence groups. Selection and 100-draw
  paired-bootstrap report completed with finite scores. These tiny budgets
  validate execution, not comparative NLP performance.
  Artifacts and their verified effective-source snapshot are retained at
  `outputs/robust-smoke/qcse-bert-20261007T225825Z-fe_jdj9s/`.
  This run predates the final launcher chunking/snapshot convenience changes;
  use its saved `source/` to reproduce that exact manifest.
- Initial launch chunking used arrays 0–999 and 0–229, each capped
  at five simultaneous jobs, with task offset 1,000 in the second array.
  The deeper audit below supersedes this uneven split with 615/615 tasks.
  No scheduler dependency or CUDA device-mask override is introduced.
- A clean `git archive HEAD` snapshot independently passed all **43** adapter
  preflights. New benchmark execution does not depend on the earlier uncommitted
  vectorized context or focused-training edits.

## Full-corpus prepared plan

Prepared the actual default matrix locally after source commit `1c263d2`, without
training or submitting it:

`outputs/robust-dgx-plan/qcse-bert-20261007T230441Z-saiitlwg/manifest.json`

Manifest SHA-256 fingerprint:
`81768b652d32fa01d61ba38f29d8e8009b6583502fd9e477edd8c2e7e94efe58`.
It retains **202,172 curated sentences**, **10,864 vocabulary words** and
**1,230 CV tasks**, together with source/lockfile snapshots and prepared splits.

| Track | All examples | Development | Held-out test | Final evaluated test |
| --- | ---: | ---: | ---: | ---: |
| Causal | 1,582,956 | 1,266,205 | 316,751 | 15,944 |
| Bidirectional | 1,785,128 | 1,427,943 | 357,185 | 17,944 |

Final test samples each contain the same 2,000 complete sentence groups; token
counts differ between prediction objectives. First-fold training/validation/test
sentence sets are disjoint in both prepared tracks. Tests cover every fold's
group isolation. The full-manifest submitter dry-run verifies exactly two
independent CV arrays, `0-999%5` and `0-229%5`, with the latter using global task
offset 1,000. The result is a reviewable local plan, not a submitted cluster run.
Prepare again on DGX so recorded input paths and provenance describe its checkout.
This historical plan predates the integrity fixes below.

## Remaining measurements

No job was submitted, no GPU fit ran locally, and there is no new production
quality result. Runtime, 10 GB peak-memory feasibility and CUDA gradients at
the full vocabulary/deepest ansatz must be measured on DGX. All-family local
preflight uses a small register; the existing production preflight uses the
full 14-qubit/64-layer simulator. Exact autograd in the new deep adapter still
needs actual allocated-device execution. Use the prepared protocol and
BENCHMARK.md commands on the cluster's locked checkout; retain all outputs/logs.

## Requested second audit and compute estimate — 2026-10-08

Integrity fixes and independent tools committed atomically as `2949842`.
The complete report is [BENCHMARK_STATE_REPORT.md](BENCHMARK_STATE_REPORT.md).

- Independently reconstructed every one of 3,368,084 contexts/targets, vocabulary
  order and full deduplicated pool; checked all five folds and both outer splits,
  input/artifact/source hashes and paired final sentence samples.
- Added saved prediction/aggregate/baseline/budget validation for completed fits,
  selection and reports; preserve runtime/status attempts; include setup in fit
  time and CUDA reserved peak memory. Reject nonfinite bootstrap arrays.
- Balanced CV arrays to 615/615, each capped at five jobs; blocked abbreviated
  Slurm options from bypassing fixed array/export/dependency controls. Tested
  submission receipts with a stub; no scheduler submission occurred.
- All six trainable gate families pass finite-difference gradcheck. Weighted
  microbatch gradients agree for QCSE/hybrid/MLP/BERT without dropout; exact
  interrupted BERT resume and attempt-provenance retention pass.
- Revalidated all 24 historical full-vocabulary smoke fits and reloaded every
  selected checkpoint: 192 replayed predictions match metrics and ranks exactly.
- Whole repository: **115 passed / 46 CUDA/MPS skips**, 129.11 seconds. Final
  focused rerun: **26 passed**, 25.25 seconds. Ruff/Pyright/shell/diff checks pass.
  Clean `git archive 2949842` passes all 43 CPU adapter preflights.
- Measured full-vocabulary optimizer/evaluation batches for all 43 adapters;
  retained raw repeats, four CBOW spot checks and three further causal checks.
  CPU-equivalent projections for 10/25/50 epochs plus conservative refits:
  **619 / 1,431 / 2,784 hours**. GPU speed factors are expressly unmeasured.
- Audited the bit-threshold metric analytically and on development targets:
  constant-zero predictions pass the ≥0.5 bit agreement criterion 98.68% of
  the time, while exact word accuracy is 7.86%. This is our corpus diagnostic,
  not a reproduction of the QCSE paper's dataset.
- Prepared and independently audited a fresh full plan after `2949842`:
  `outputs/robust-dgx-audited-plan/qcse-bert-20261007T233022Z-83f4xxt4/`,
  fingerprint `ad263266423e0f91fcb577078b77aa4b4bbf0c7ed7b85fedbb5cf3251f2b6798`.
  Data artifacts exactly match the historical plan; current dry-run uses two
  `0-614%5` arrays, global offsets 0/615. Original dirty work is preserved and
  recorded; actual cluster preparation must describe its frozen checkout.

Remaining: allocated 10 GB MIG validation/runtime and save/resume pilots;
1,230 CV fits and 33 refits; complete results/paired-bootstrap analysis; stronger
classical/pretrained references and independent downstream semantics research.
No production run or comparative quality conclusion is claimed.

## Trainability investigation — 2026-10-09

- Reviewed primary barren-plateau, cost-locality, initialization, noise-induced
  and gradient-free-optimization papers. The name is established; its diagnosis
  for this NLP objective remains unproved. See [TRAINABILITY.md](TRAINABILITY.md).
- Added opt-in hybrid 2/4/8-qubit × 2/8-layer × std 0.05/1.0 controls, reusing
  two standard references and adding 10 arms. Standard/extended matrices remain
  1,230/1,530 CV fits and 33 family refits. Selection uses actual manifest arms.
- Added circuit-only pre-clipping gradient traces and online loss to each epoch,
  excluding known final diagonal zeros for QCSE's Z readout. No extra training
  forward/backward pass. Profiler and compute projection cover both matrices;
  allocation preflight includes all 53 configurations.
- Added paired float64 local/global initial-gradient probes with fixed synthetic
  inputs, retained per-coordinate arrays and verified effective-source snapshots.
  One-thread local run: 512 initializations / 32 paired summary rows, 4.96 seconds.
  At depth 8/std 1.0, global first-RX variance falls about 266-fold from q=2 to
  q=8; local variance stays similar. This is not an NLP plateau diagnosis.
- Current full-vocabulary 14-qubit microbatch: layer RMS 0.0188/0.0356, 55 eligible
  parameters and 27 structural zeros. No universally tiny gradient in this one
  instance; distributions and trajectories require further measurements.
- Validation: 127 passed / 46 CUDA/MPS skips (38.32 seconds), then final focused
  rerun 38 passed (10.42 seconds); all 53 CPU backend preflights; Ruff/Pyright,
  shell syntax and diff checks pass.
- Commits: `a2f13c8` keeps exports ignored; `dabfe83` adds tested controls/probes.
  Refreshed all remote branch refs and fetched filename history: no tracked
  conversation exports/transcripts to delete. No history rewrite or push needed.
- Prepared full optional plan after `dabfe83`:
  `outputs/robust-trainability-plan/qcse-bert-20261009T010146Z-vzr8545d/`.
  Source snapshots verified, 1,530 unique tasks; data artifacts exactly match
  the prior independent full-data audit. Committed-source probe retained at
  `outputs/trainability-verified/gradient-probe-20261009T010147Z-my46q9e0/` with
  every raw array bit-identical to the initial probe.
  Manifest fingerprint:
  `676beeec9b250267f5b95276191796c684f7fba9d6e6495a2c4a9dd35746efbd`.
  Dry run verified two `0-764%5` arrays and offsets 0/765; no submission occurred.

Remaining: larger-register/depth variance scaling with uncertainty, all NLP
control cells and trajectories, actual MIG performance/memory/save-resume pilots,
and the previously documented production computation. No GPU or Slurm run started.

## Updated compute estimate — 2026-10-09

- Remeasured all 53 configurations at full vocabulary, four CPU threads, one
  warmup and three timed repeats. Effective source hashes match the current
  prepared trainability manifest. Four CBOW checks agree within 4% for training.
- Found substantial CPU timing variation: fresh QCSE two/eight-layer training
  batches take 0.749/2.627 seconds versus historical 3.29/11.10. Cause is not
  established; historical estimates are superseded, not silently rescaled.
- Existing estimator completes on the matching 1,530-task manifest: 136/317/619
  CPU-reference job-hours for 10/25/50 epochs including conservative refits.
  At 50 epochs, conditional ten-MIG planning budgets are 47 h at 2× CPU speed,
  22 h at 5×, or 87 h at parity, including assumed startup and 25% allowance.
- Saved raw timings, compute projection, ordered-slot scheduling scenarios and
  measured synthetic bootstrap cost under ignored
  `outputs/compute-estimate-2026-10-09/`. See
  [BENCHMARK_COMPUTE.md](BENCHMARK_COMPUTE.md) for scope, formulas and assumptions.
  No runtime changes, dependency changes, production training or cluster launch.

Remaining: actual CUDA batch timings, end-to-end overhead/memory/resume pilot,
then the previously documented production matrix and separate scaling study.

## Scope and historical CUDA evidence review — 2026-10-09

- Reopened `../sweep.zip` metadata and tracked cluster aggregates: 44 completed
  CUDA configurations, each two ten-epoch validation fits plus a refit (132
  fits), total recorded configuration duration 60.59 h, median 69.84 minutes.
  Confirmed full-corpus experiment 013 takes 69.90 minutes for all three fits.
  Exact GPU/MIG allocation, source SHA and scheduler logs are absent.
- Corrected the launch recommendation: the 1,530-fit grid is optional exploratory
  breadth, not a justified minimum. Proposed allocated-device pilots, a small
  causal screening set and 4–6 frozen configurations with 60–90 CV fits before
  adding further questions. This proposal has no power-based robustness guarantee.
- Reaffirmed that 22–47 hours relies on unmeasured speedup/overhead assumptions;
  the historical SPSA/bit-BCE runtime does not calibrate current autograd/word-loss
  runs. Full CUDA calibration remains necessary. See BENCHMARK_COMPUTE.md.
- Metadata/count/duration review only; runtime code, prepared studies and
  experiment defaults remain unchanged. No training or cluster job launched.

## Four-instance resource ceiling — 2026-10-11

- User limit: at most four 10 GB MIG instances concurrently; no benchmark jobs
  have been submitted yet. Robust CV arrays now cap at two jobs each, and the
  refit array caps at four. CLI and command construction reject concurrency >4.
- Fixed `prod10`, one GPU, one task and one node per job; GPU/partition overrides
  are rejected and inherited `SBATCH_*` environment settings are excluded from
  submission. A checkout-local submission lock and an own-user `prod10` queue
  check prevent overlapping sequential batches/pilot allocations. Scheduler
  query failures stop submission. Use one shared checkout; manual jobs or
  simultaneous submissions from separate checkouts require separate coordination
  or a site-managed account limit.
- Compute estimator now defaults to four slots and records the actual concurrency
  with generic slot JSON keys. Reused measured CPU batch timings, without any new
  training, to save four-slot projections under
  `outputs/compute-estimate-2026-10-11/`. Full 50-epoch hypothetical planning
  budgets are 56–115 hours at 5×/2× CPU speed; these remain uncalibrated GPU scenarios.
- Focused verification: **5 passed / 25 deselected**, 1.37 seconds. Covers both CV
  matrix sizes, refits, upper-limit rejection, resource/environment overrides,
  busy/failed queue checks, concurrent-submission locking, receipt retention and
  estimator default/override arithmetic. Independent command checks also verify
  a four-array matrix stays within four slots. Ruff, Pyright, shell syntax and
  diff checks pass.
- Launcher/Slurm source hashes changed: pull the updated branch and prepare a
  fresh study before production. Existing submitted jobs would retain their old
  settings, but the user confirmed none have been submitted. No scheduler or
  training job was launched from this workspace.

## Batch-only CUDA pilot — 2026-10-11

- The user reported the site's rejection: `partition prod10 does not accept
  interactive jobs, sbatch script only`. The suggested interactive `srun` pilot
  was incompatible with this policy. Checked all three available local Markdown
  conversation exports: none contains that earlier error; existing cluster
  documentation already uses batch submission.
- Added `--stage pilot` to the existing guarded launcher and Slurm script. It
  submits exactly one task on one 10 GB MIG, retains submission receipts and
  applies the same queue/lock/environment guards as CV/refit. CUDA preflights,
  causal-arm profiling, a four-slot compute projection and task 0 run in the
  batch allocation. Later CV submission reuses the completed pilot fit.
- Updated the operational documentation to require a successful batch pilot
  before CV submission. Source hashes changed: prepare a fresh study after
  pulling; do not patch a previous manifest to bypass provenance checks.
- Focused validation: **7 passed / 25 deselected**, 2.29 seconds; Ruff, Pyright
  (zero errors), shell syntax and diff checks pass. Mocked submission checks
  confirm the one-task pilot, receipt, causal-track guard, busy/failed queue and
  submission lock. Running the real shell script against stub executables
  verifies device-mask preservation, success ordering and that a profiling
  failure prevents estimation/training. These are orchestration checks, not a
  CUDA hardware validation. No local training or DGX submission performed.
