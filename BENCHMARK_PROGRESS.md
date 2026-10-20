# QCSE/BERT benchmark progress — 2026-10-08

Branch: `codex/qcse-bert-robust-benchmark`, based on `2e037ca`.
Existing uncommitted focused-benchmark code, reports and images are preserved;
they are not part of these commits. No cluster job was submitted at branch
creation; the user's completed DGX timed study is audited below.

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

## Completed-job queue guard correction — 2026-10-11

- User evidence: DGX pilot `9115_0` completed in **7:46**, with `COMPLETED`,
  exit `0:0`, and one validated completed fit out of 90. This duration includes
  the batch checks/profile/estimate/fit; it is not a per-fit average or a
  calibrated duration for the other five model arms.
- The controller still exposes this completed pilot in `squeue --states=all`.
  The launcher incorrectly treated every listed job as unfinished. It now queries
  job IDs and full states, ignores explicitly finished states, and retains the
  guard for pending/running/configuring/completing/suspended/requeued/preempted,
  unknown and malformed entries. Scheduler failures still stop submission.
- Verified state semantics against official Slurm `squeue` and job-state
  documentation. No change to the four-instance ceiling or training code.
- Existing prepared study/pilot can be reused: fetch only the fixed launcher
  into ignored `outputs/`, leaving the experiment checkout and manifest intact.
  Submission receipts retain the actual executed launcher hash and source.
  Pulling the updated branch is appropriate after this study finishes.
- Focused validation: **7 passed / 25 deselected**, 2.38 seconds. Covers the
  retained completed-pilot reproducer, mixed terminal/active queues, ten
  nonterminal/unknown states, malformed rows, query failure and lock rejection,
  fixed resources/concurrency, copied-launcher root resolution, receipts and
  saved launcher source/hash. Ruff, Pyright (zero errors), shell syntax and diff
  checks pass. No training or scheduler job launched from this workspace.

## Detailed timing and diagnostics — 2026-10-11

- User chose detailed measurements before broadening the experiment matrix.
  Keep the focused six causal arms, five folds, three seeds: 90 CV fits and
  12 selected refits, with the existing full-corpus protocol and four-MIG cap.
- Added synchronized per-attempt/per-epoch phase timing and failure records;
  mean/max preclip gradient norm and clipping fraction; device capacity/UUID/SM
  metadata and process RSS. Retain checkpoints, RNG and circuit diagnostics.
- Added a timed job wrapper separating inventory/preflights, pilot profiling,
  projection and fit subprocess costs. New source snapshots include the wrapper,
  profiler and estimator. The profiler uses the prepared batch size and includes
  the production circuit gradient diagnostic.
- Added measured CV cost reports and same-arm budget scenarios with optional
  Slurm allocation accounting. Exclude calibration profile time, avoid counting
  batch/extern steps again, retain failed job attempt costs, and distinguish
  empirical quantiles from uncertainty bounds. Refits/queue waits are outside
  additional-CV forecasts; new arms require new pilots.
- Documented a fresh timed study after stopping/finishing old jobs, keeping old
  outputs and manifests intact. Updated compute notes with the user-reported
  successful pilot, without claiming its unknown per-phase costs as measured.
- Final bounded validation: **49 passed**, 11.19 seconds in the working checkout
  and **49 passed**, 9.80 seconds in a clean `git archive` snapshot with only
  this change overlaid. The latter excludes unrelated local model/simulator
  changes. Ruff, Pyright (zero errors), shell syntax and diff checks pass.
- Independent CPU comparison with the pre-instrumentation `dcfd292` fit code:
  QCSE, MLP and dropout-enabled BERT have **bit-identical model and Adam state**
  and equal validation metrics after two synthetic nine-draw epochs. Existing
  interruption/resume checks also preserve exact BERT state. Retained evidence:
  `outputs/timing-review/equivalence.json` and the synthetic timing report.
- The real profiler honors a prepared batch of four, includes gradient
  diagnostics, and emits three adapter rows; the real timing report handles
  partial synthetic results. These CPU pipeline checks do not validate CUDA
  phase synchronization/hardware metadata. The new DGX pilot remains necessary.
- No DGX job submitted or cancelled from this workspace; no full-corpus local
  training ran. Existing DGX study outputs stay intact and separate.

## Measured DGX CV audit — 2026-10-11

- Received `robust-timed-diagnostics.zip`, inspected data only. Verified 90 task
  ledgers, manifest/selection fingerprints, 30 source hashes against `18d4ef6`,
  included JSON artifacts, candidate aggregates and CSV/allocation costs.
- Pilot `9215_0` and array `9216` have 91 successful allocations. The CV array
  completed in 2:30:56 / 9.9525 job-hours; accounting peaks at four concurrent
  instances. No timeout or failed attempt in this timed study.
- Hybrid CRZ vs identity: 30.369% vs 30.136% top-1, 2.69× median cost;
  fold-average differences change sign. QCSE underperforms the frequency
  baseline. MLP beats this tiny BERT configuration in all 15 paired fits.
- Active circuit gradients are measurable; these fixed-qubit fits do not prove
  a barren plateau. Final CRZ Z-readout has 27 structural zero-gradient gates.
- Saved metrics, plot, study-only accounting and limitations in
  `results/robust-timed-cv/`. Binary split/prediction/checkpoint files were not
  exported, so independent per-example/split/bootstrap validation remains open.
- User requested 100 × 20,000-draw fixed-budget runs, broader ablations and
  fewer repetitions during discovery. Prepare a fresh staged study; retain the
  completed outputs, keep test data out of screening and cap concurrency at four.

## Long-budget staged experiments — 2026-10-11

- Added `--fixed-epochs`: every trainable CV fit reaches its declared budget,
  scores the final state and gives selected refits the same epoch count.
  The default historical early-stop policy is unchanged. Reject incomplete
  fixed-budget results; interruption/resume preserves model/Adam/RNG state.
- User-approved production screen is 100 × 20,000 replacement draws: 31,300
  updates/two million draws per fit. Three folds/one seed discover candidates;
  five folds/three seeds confirm only a small development-selected set.
- Added complete-source dataset selection with the existing shared vocabulary,
  full final fold/test scoring, identity LR/latent-size controls and retained
  prespecified identity refits. This prevents family selection from dropping
  the circuit ablation required for the final held-out comparison.
- Added ten executable preparation blocks: 89 settings/configurations, 267
  optional screen fits. No automatic submission. Cover hyperparameters, BERT
  capacity, circuit generators/depth, qubit/init, context, source/window and
  bidirectional tasks. Graph/Laplacian semantic Hamiltonians remain research.
- CBOW-only pilots profile their actual objective. Dimension pilots retain a
  separate 32-seed local/global synthetic gradient probe, excluded from
  repeated-fit timing costs. No synthetic plateau is relabelled as NLP evidence.
- Estimator uses the requested fixed budget, full prepared evaluation counts,
  per-configuration costs, measured-anchor job overhead and scheduling tails.
  Require matching profile source/microbatch/threads/objective and every arm.
  Deep QCSE needs fresh device/time-limit calibration before array submission.
- Final clean-source snapshot: **59 passed in 11.41 seconds**, excluding the
  unrelated local model/context/simulator edits. Includes flat-monitor default
  stopping vs fixed final scoring, budget integrity, retained controls, exact
  BERT resume under both policies, all block definitions, source/profile
  mismatch guards, scheduling tails and objective-specific pilot orchestration.
  Ruff, Pyright (zero errors), Bash syntax and diff checks pass. Corrected an
  empty-array portability issue found by the Bash helper test and refreshed
  older estimator fixtures for the stricter metadata contract.
- No full-corpus local training and no new DGX submission/cancellation.
  Expanded controls/long budgets still require actual CUDA pilots. Retained
  analysis and clean-checkout evidence under ignored `outputs/dgx-timed-review/`.

## Shared-screen baseline deduplication — 2026-10-11

- Added a `main` block merging the first five research blocks: 52 unique arms /
  156 CV fits. Repeated anchors use the same source, cohort, split, seed and
  budget, so repeating those fits would add no independent evidence.
- Combined main plus the five optional data/task cohorts is 79 distinct setups /
  237 fits, avoiding 30 redundant fits and 60 million training draws compared
  with executing every standalone block. Standalone blocks remain available
  for a narrow question; do not run them in addition to the same main study.
- Final focused check: the executable `main` arm set equals the union of all
  five component blocks, with no duplicate names; **1 passed in 1.22 seconds**
  in the clean-source snapshot. Ruff, Bash syntax and diff checks pass.

## Parameter-free screen reporting fix — 2026-10-16

- User reports all 156 CV fits complete and validation-only selection succeeds
  (five proposed refits; none submitted by selection). The timing report then
  fails reading `cv/task-0015/history.json`.
- Root cause: zero-parameter controls exit before checkpoint saving, so their
  valid results have no persisted optimizer history. The report wrongly
  required it for all fits. Missing history is now allowed only with zero
  trainable parameters and zero stopped epochs; trainable missing histories
  still fail. Costs remain measured and clipping stays unavailable.
- The report module now owns the existing CLI and supports execution as a
  copied analysis file under ignored `outputs/`. Its imports retain the
  frozen checkout's strict manifest/artifact/source checks. The report records
  the executed analyzer hash. Deployment uses `git fetch`/`git show`, preserving
  the prepared source and saved results without another fit.
- Clean-source final checks: **16 passed in 3.92 seconds**. The regression fits
  identity/frozen QCSE, trainable mean and MLP controls on a tiny corpus, checks
  unavailable clipping and unchanged zero-epoch costs, executes the copied CLI
  with matching source hashes, and rejects missing trainable history. Ruff,
  Pyright (zero errors) and diff checks pass. No full-corpus training or DGX
  jobs were run locally.

## Completed main-screen audit and focused follow-up — 2026-10-16

- Inspected the downloaded `robust-screen-main-diagnostics.zip` as data only.
  Its SHA-256, original `dd105d5` source hashes, both corpus hashes, manifest /
  selection fingerprints and included prepared JSON artifacts agree. Audited
  all 156 task ledgers, fixed budgets, histories, phases and CUDA runtime data.
  Independently reconstructed grouped fold counts and training-frequency scores
  from saved sentences; every result agrees. Binary splits/predictions/checkpoints
  were omitted, so their actual contents remain an explicitly unverified layer
  of this local export. No held-out results are present.
- All 157 Slurm allocations succeeded, peak concurrency four. The main array
  took 94h 19m 33s; measured per-fit allocation costs total 370.506 job-hours
  after pilot calibration is removed. The original 350.980-hour projection was
  5.56% low. Future profiles apply an explicit 1.10 engineering margin; this is
  not a confidence bound. The longest old fit was 13.522 hours.
- BERT-32 reaches 34.537% full-fold top-1; q8 identity / CRZ reach 33.416% /
  33.400%. CRZ costs 3.41 times as much, with mixed-sign fold differences.
  Angular QCSE reaches 8.099%, slightly above frequency top-1 but worse NLL /
  top-5. Deeper QCSE adds cost without a top-1 gain. Three correlated folds /
  one training seed are discovery evidence, not an independent-fold interval.
- Published all-arm metrics, fit costs, study-only accounting, two figures,
  audit scripts, literature notes, limitations and hypotheses under
  `results/robust-screen-main/`. Barren-plateau literature distinguishes
  cost/circuit assumptions from a flat training curve. The synthetic global-cost
  probe concentrates; active NLP gradients do not establish an NLP barren plateau.
  XX/ZZ gates remain generator ablations, not learned semantic Hamiltonians.
- Implemented exact analytic identity XYZ features. Tiny one-thread CPU checks
  match statevector logits and all parameter gradients (float64 errors below
  4e-16); the recorded q8 microbenchmark is 11.3 times faster. This is not a
  full-vocabulary or DGX speed claim. Production CUDA pilots remain required.
- Added the focused protocol in `BENCHMARK_FOLLOWUP.md`: two objective-specific
  18-arm / three-fold / seed-43 LR screens, then five fixed architectures per
  objective at five folds / seeds 44,45,46. All retain 100 x 20,000 draws, full
  final scoring and complete eligible pools. Total 258 CV fits plus 30 refits;
  four pilots reuse task 0. Approximate pre-pilot expectation is 4–5 days plus
  queue waits. Actual per-objective CUDA profiles replace those timing proxies.
- Near-capacity MLP-32 controls BERT-32; matched q8 circuit/identity use one common
  validation-selected LR. Small-std mean embeddings rescale the same random
  draws. Encoder/head initialization matches across same-depth q8 controls.
  Old depth changes consumed different RNG before the head: disclose that
  confound rather than treating old depth results as a pure depth intervention.
- Fixed held-out reports overwriting retained configurations from the same
  family. Reports now key by arm, retain every pair and enforce example alignment.
  Prespecified CRZ-minus-identity top-1 uses 97.5% paired seed/sentence intervals
  across the two planned objectives; secondary comparisons are exploratory.
  Both confirmation selections freeze before either test stage begins.
- Added actual train-fold CE gradient probes at q2/4/8/12, depths 2/8, std
  0.05/1 and 32 circuit seeds, fixing non-circuit weights within each register
  size. Raw inputs/gradients and common-state hashes are saved. Across register
  sizes encoder/head dimensions also change; the probe cannot prove asymptotic
  barren plateaus. Only a small toy probe ran locally; CUDA data are pending.
- Added a restartable sequential `sbatch --wait` driver. It stops on failed jobs,
  guards a 22-hour predicted fit within a 24-hour allocation, preserves strict
  source validation, freezes selections, retains accounting and caps arrays at
  four one-MIG tasks. Submission and wait-completion timestamps are distinct.
  No refinement refits or automatic optional data-cohort expansion are run.
- Final clean-source snapshot, excluding unrelated local research edits:
  **74 tests passed in 18.18 seconds** across robust benchmark, trainability,
  timing, screening, follow-up and driver suites. Includes analytic forward/all-
  parameter gradients, common initialization, retained same-family refits,
  paired intervals, full driver failure/resume/order, fixed budgets, BERT resume,
  source guards, planning margin and refit-count checks. After final report /
  accounting edits, two targeted tests passed (6.59s); after timestamp correction,
  the submission/resource regression passed (1.61s). Ruff passes all owned Python
  files; Pyright reports zero errors/warnings, Bash syntax and diff checks pass.
- No local full-corpus training, DGX submission or cancellation. Remaining work:
  run the new CUDA pilots and batch, inspect actual CE gradient scaling, audit
  exported outcomes and paired held-out intervals. Train-only vocabulary,
  out-of-domain semantics and graph-Hamiltonian designs remain separate research.

## Follow-up completed: held-out audit and numerical repair — 2026-10-20

- Reviewed `robust-followup-diagnostics.zip` as data only (SHA-256
  `bfb71bdb8035851723decf54a4a7f3829530fda48bed95b7f2c2f3279bd818d6`).
  All 34 recorded source hashes match `483bea6`; manifests, parent selections,
  source CSVs and included prepared JSON artifacts agree. Independently
  reconstructed fold/test counts and training-frequency metrics for both tasks.
- All 258 CV fits plus 30 refits completed 100 x 20,000 draws / 31,300 updates.
  All 292 allocations succeeded, peak concurrency four. Allocation span
  97h 23m 41s, total 348.443 allocation-hours. CUDA CV projections with the
  explicit 1.10 factor were 5.7–9.3% above measured costs.
- MLP-32 beats near-capacity BERT-32 in all 15 paired CV fits and all three
  held-out seeds for each task: held-out top-1 35.537% vs 33.842% causal,
  56.586% vs 54.006% CBOW. Cost is roughly 5.5 times lower. This corrects the
  old smaller-MLP comparison, not a claim about pretrained/full-context BERT.
- Primary CRZ-minus-identity differences are −0.0423 / +0.0853 pp, with 97.5%
  intervals spanning zero and leave-one-seed-out means changing sign in both
  tasks. Secondary CBOW NLL/top-5 signals remain exploratory. Identity costs
  about 7.4 minutes per CV fit versus 132–133 minutes for CRZ.
- Three seeds and one outer split limit robustness; folds overlap, vocabulary
  preprocessing is transductive and semantic/source independence is untested.
  CBOW also adds sentence-initial targets: 357,185 test targets vs causal
  316,751. Cross-objective accuracy gains are not a pure right-context effect.
- Four actual-CE gradient probes retain nonzero active gradients through q12;
  larger-register variance patterns are not consistently exponential. One
  head/batch per condition, q-dependent head shape and omitted raw gradients
  prevent a broad barren-plateau diagnosis. QCSE representation/independent-bit
  decoding is a more promising hypothesis than more identical deep fits.
- Found a real float32 aggregation issue in held-out summaries: CBOW QCSE NLL
  5.532076 vs the three-ledger mean 5.542458. Top-k agrees within 1e-7. Fixed
  reporting means/SDs and paired differences in float64, without changing any
  training, selection, saved fit or original archive. A copied CLI retains the
  frozen checkout's original strict binary validators and optionally exports
  hash-bound seed/sentence sums, including common noninitial targets.
- Published the audit, all 288 fit records, filtered accounting, two figures,
  conclusions, literature and prioritized research/compute decisions under
  `results/robust-followup/`. Binary indices/predictions/checkpoints and raw
  gradients were omitted; actual contents and exported bootstrap intervals
  remain independently unverified in this local review. Next step is the small
  corrected-prediction export, not a new grid. A future confirmatory population
  must be new because the present held-out scores have now been observed.
- Final focused clean-source check: **76 passed in 24.71 seconds**, one CPU
  thread. Includes a million-row precision reproducer, cluster/token weighting,
  common-position export and exact copied-reporter parity, plus existing fixed
  budgets, model parity, resume, source guards and scheduler cap regressions.
  Ruff passes, Pyright has zero errors/warnings, command blocks pass Bash/ASCII
  checks; figures visually inspected. The first snapshot test invocation lacked
  Git provenance metadata; correcting the test harness resolved its setup errors.
- No full-corpus local fit, new DGX job, or automatic next batch was run.
