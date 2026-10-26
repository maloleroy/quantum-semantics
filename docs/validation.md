# Validation and evidence boundaries

## Hamiltonian-only language follow-up (2026-10-26)

Post-hoc operational readout clarification: `LANGUAGE_BORN.md` now distinguishes
argmax accuracy from one-shot success `mean(p(correct word))`. The latter was
recovered as `mean(exp(-individual NLL))` from existing locked test arrays, with
no new inference or fitting. Coherent H improves 1.91%→6.76%; retrained dephasing
gives 5.71%, matched MLP sampling 24.77% and subset-bigram sampling 18.95%.
The coherent-minus-dephased one-shot group interval is [+0.988, +1.111] percentage
points. `one_shot_summary.json` retains raw values. These are ideal simulated
sampling expectations, not hardware measurements. Existing checkpoint selection
was by NLL; changing the operational endpoint does not retroactively reselect fits.

Executed **24 fits** (12 LR-screen / 12 fresh-seed confirmation) on real Tatoeba
causal next-word targets, with 256 training-vocabulary entries including UNK, eight
qubits, a four-word fixed phase/recency encoding, 65,536 full-H parameters and direct
Born word measurement. No learned classical encoder/head, bigram prior or smoothing
is in the Hamiltonian predictor. New grouped split seed 20261026; vocabulary and
statistics are training-only. Fixed sample sizes are 32,768 train / 4,096 validation /
8,192 test targets. Test has 7,225 sentence groups and 16.98% UNK targets.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python -m experiments.hamiltonian_study.language_born --epochs 30 --workers 8
```

Completed run: `outputs/language-born/tatoeba-20261026T072646Z-wc27z5h5/`.
Preparation, screening, confirmation, locked test scoring, audit and plots completed
in **260.493s** on eight one-thread workers on the local Ryzen 7 7840S. Observed
aggregate worker resident memory was approximately 4.5 GB. A prior validation-only
pilot informed the budgets; no test scores were read before the selection lock.

Across seeds 61/62/63, H test NLL improves **6.5861→3.5974** and word accuracy
**2.57%→24.21%**. Known-target accuracy improves **0.12%→22.49%**. Retrained
dephasing gives NLL 3.6683 / accuracy 19.51%, with coherent-minus-dephased
sentence-group 95% interval **[-0.08301, -0.05937]** NLL. Matched-input MLP gives
2.6205 / 38.29%; a bigram fitted on just the same optimization examples gives
2.8837 / 34.09%. The [report](../experiments/hamiltonian_study/LANGUAGE_BORN.md)
retains all controls, known-target metrics, paired intervals and interpretation.
H training and a coherence contribution transfer to language; superiority over
these classical predictors is not established. H selects the final epoch for all
three seeds and the lowest grid LR; the experiment does not establish convergence.

Audit verified frozen source/dependency/data/checkpoint/recipe hashes, selected
epoch minima, identical coherent/dephased starting tensors, saved per-example
metrics and split-group separation. Largest test probability-mass error was
**2.665e-15**, without renormalization or conditioning. `known_comparison.json`
adds post-hoc intervals from the saved arrays; it involves no refitting/inference.
The executed source SHA-256 remains
`a851d4ebde8fc9062abfb0f730206fd987b81ec2649d22eab85b973dcf755623`.
Only the report and one inspected figure are retained; raw fits/arrays stay ignored.

The exact Hermitian spectral forward and gap-safe first-order Frechet adjoint
avoid generic exponential-backward cost without changing the model. Six new
scientific tests pass: independent exponentials/adjoint/finite differences at
zero, near-degenerate and random matrices; repeat/padding/order/permutation-safe
encoding; density-matrix/Born readout and parameter attribution; streamed metrics
and fitting without test arrays. Full checks: **279 passed, 46 hardware skips**
in 62.26s while training was active; two existing SciPy sparse-format warnings.
Ruff format/check pass (120 files unchanged), Pyright has zero errors and nine
existing optional ReportLab source warnings. CPU q14/depth64 preflight passes
against Qiskit at **5.8842e-15** max error. No GPU, hardware, shot/noise or
near-20-qubit training behavior is established.

## Hamiltonian-only learning with fixed encoding/readout (2026-10-26)

Executed the new `born_evidence.py` pipeline: 30 three-LR screen fits at seed 11,
then 50 confirmation fits across seeds 47–51, two tasks and five model families.
All data preparation and scoring were local CPU; the UCI digits archive was
downloaded from its primary repository. Eight one-thread spawned workers on the
Ryzen 7 7840S completed the pipeline in **27.664s**, excluding pilots/download/checks.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python -m experiments.hamiltonian_study.born_evidence \
  --digits-archive outputs/hamiltonian-evidence-pilot/optdigits.zip --workers 8
uv run python -m experiments.hamiltonian_study.analyze_born \
  outputs/born-evidence/fixed-encoding-20261025T233448Z-q_806f9r \
  --figure outputs/born-evidence/fixed-encoding-20261025T233448Z-q_806f9r/learning.png
```

The [report](../experiments/hamiltonian_study/BORN_EVIDENCE.md) retains the complete
comparisons, equations, budget, analytic limitations and commands. The symbolic
split is 160/48/48 unique contexts. Digits use 3,062 training and 761 validation
images from the official training file; all 1,797 official test images are kept.
Exact image groups are disjoint across partitions. Inputs and measurement rules
are fixed; every coherent-model weight is a Hermitian-matrix coordinate.

Mean coherent held-out accuracy improves **11.67%→100%** for symbols and
**5.84%→91.96%** for digits. Independently retrained dephased controls reach
8.33% / 61.88%; the 4,066-parameter digit MLP reaches **95.43%**. Coherent digit
NLL is 1.04751, below the post-hoc 1.33008 lower bound for *every* completely
dephased model using this rank-four readout. This establishes Hamiltonian
trainability and an architecture-specific coherence contribution. It does not
establish language transfer, general quantum advantage or hardware feasibility.

Audit checked source/dependency/data/weight/lock hashes, all 50 selected epochs
against full validation histories, paired coherent/dephased initialization,
saved test probabilities against reported NLL, and probability mass (no output
renormalization or role conditioning). Recipe selection was locked before fresh
seed fits; every checkpoint was locked before the first test evaluation. The
analytic bound was derived afterwards, with no further fitting/test inference.
The run contains executed source, dependency lock, all 80 fit records/weights,
selection manifests, raw data fingerprints and test arrays. Generated/downloaded
data remain ignored; only the report and one summary figure are retained in Git.

Six new focused scientific checks pass: independent SciPy exponentials and
finite-difference gradients, the exact 256-pair Fourier solution, dephasing's
loss of the second symbol, partial-trace readout, fitting/selection without any
test file, and a tight dephased bound with a coherent counterexample. The saved
figure was visually inspected. CPU backend preflight at q14/depth64 also passes
against Qiskit, max absolute error **5.8842e-15**. CUDA/MPS were not exercised.

Final repository checks: `uv run ruff format .` leaves all 118 files unchanged;
`uv run ruff check .` passes; `uv run pyright .` reports zero errors and nine
pre-existing optional ReportLab source warnings; `uv run pytest -q` gives
**273 passed, 46 hardware skips** in 30.76s, with two existing SciPy sparse-format
warnings. Executed model source still matches its frozen run SHA-256
`6f094e9340fec7c915f94d929ee9a3405260c5eeafb5c97b030c363215dd5c35`.

## Short CPU scaling profiles (2026-10-23)

User planning limit: 24 **physical** cores / about 10 GB RAM / a few hours.
Executed eight inline one-process profiles with `OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -`, setting Torch threads
per case. Each used `ContextResidual`, random 128-target batches, AdamW at
0.003 / decay 0.01, clipping norm 5, two warmups and five measured forward/
backward/update steps. Local output: `/tmp/qcse-scaling-pilot.json`.

Cases were vocabulary/window/embedding/hidden/features =
4096/16/64/128/16 at 1 and 4 threads; 8192/32/128/256/32 at 1, 4 and 8;
8192/32/128/256/64 at 1 and 4; and 8192/32/2048/1024/16 at 4.
The 4,096-vocabulary case measured **13,775 targets/s at four threads** and
7,973 at one thread. The 5-qubit model
measured **3,241 / 5,206 targets/s at 4 / 8 threads**; the 6-qubit model measured
**821 targets/s at 4 threads**. The 84,044,816-parameter, 4-qubit example measured
**230 targets/s at 4 threads**. The [compute notes](optimizations.md) retain the
useful architecture/size projections and actual timing scope.

These are synthetic execution probes, not corpus training, allocation-wide
throughput, sustained thermal measurements or a 24-core host benchmark. No
learning/generalization or full-fit time is established by these short steps.
No runtime/model code changed, so the existing scientific tests were not rerun.
Streaming metrics, bounded batch/worker memory, tokenized official dataset
splits and a profile inside the actual 10-GB allocation remain necessary before
starting a large study. No dataset download or production run was performed.

## Matched contextual-model tuning (2026-10-23)

The full staged search completed successfully in **250.976s wall time** on the
Ryzen 7 7840S, using 16 spawned workers with one numerical thread each. The
parent thread count and BLAS/OpenMP limits were explicit; no GPU was used.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python -m experiments.hamiltonian_study.tune_contextual \
  --dataset experiments/hamiltonian_study/results/tatoeba_full_data \
  --train-limit 32768 --validation-limit 4096 --epochs 60 --batch-size 128 \
  --lrs 0.001 0.003 0.01 --decays 0 0.001 0.01 --screen-seed 11 \
  --confirmation-seeds 47 48 49 --keep 2 --workers 16 --threads 1
```

Artifacts: `outputs/contextual-tuning/paired-search-20261023T071728Z-0w2vlf8d/`.
All **18 screen + 12 confirmation fits** have complete records and selected
inference weights. Independent review checked the 61 history rows per fit,
optimizer settings and budgets, minimum-NLL selection including epoch zero,
fresh seed IDs, seven frozen source/lock hashes and every confirmation's
selection-lock hash. The prior POC's training/validation contexts, targets,
groups and initial weights match exactly; the first 12 epochs at the shared
original recipe reproduce its NLL trajectory with maximum difference zero.
Neither the tuning loader nor fitting code accesses the test arrays.

The full [report](../experiments/hamiltonian_study/CONTEXTUAL_POC.md#matched-tuning-follow-up)
retains the grid, per-recipe means, seed SDs and selected epochs. By fresh-seed
mean NLL, both choose LR 0.01 / decay 0.01. Hamiltonian / classical validation
NLL is 2.794025 / 2.781565, top-1 36.1735% / 36.0270%. These are exploratory
selection results on reused validation data; fresh seeds do not supply a new
evaluation population or prove an optimum beyond the tested boundaries.

The runner's bounded parallel integration test verifies locked recipe selection,
paired initial tensors, fresh seeds and source snapshots. The selection test
checks that NLL takes priority over accuracy and ties are deterministic.
`uv run pytest -q --tb=short` with one numerical thread passes **267 tests,
46 hardware skips**, in 32.84s. Ruff format/check pass. Explicit Pyright on the
new runner/module/tests has zero errors or warnings. Repository-wide Pyright
excludes generated `outputs/` source snapshots and logs; these are provenance
copies, not importable source packages. The retained runtime sources remain
checked. The nine existing optional ReportLab import warnings remain.

The only later change to the run's snapshotted project configuration is the
type-checker's generated-output exclusion. Executed model/optimizer code and
the archived source snapshot are preserved. Previous CPU/CUDA dependency edits
are retained. No new CUDA certification, optimizer-resume implementation or
held-out test result is claimed.

## Context-conditioned Hamiltonian proof of concept (2026-10-23)

Implemented the isolated `experiments/hamiltonian_study/contextual.py` runner,
with a latent Hermitian generator, learned ordered context, full Born readout
and an initially zero correction to a fixed bigram. See the complete
[model, protocol, results and commands](../experiments/hamiltonian_study/CONTEXTUAL_POC.md).

Executed on one CPU thread:

```bash
uv run python -m experiments.hamiltonian_study.contextual \
  --toy --epochs 60 --batch-size 64 --lr 0.01 --seeds 11 22 33
uv run python -m experiments.hamiltonian_study.contextual \
  --dataset experiments/hamiltonian_study/results/tatoeba_full_data \
  --train-limit 32768 --validation-limit 4096 --epochs 12 \
  --batch-size 128 --lr 0.003 --seeds 11 22 33
```

Nine fits per task compare Hamiltonian, classical context and last-word-only
controls with paired initial trainable tensors and identical optimizer/draw
budgets. Initial predictions exactly match the fixed bigram. Real-data mean
validation NLL is 2.805048 / 2.828817 / 2.949332; mean top-1 is 35.5632% /
35.2458% / 32.9102%. Both full-context models solve the controlled first-cue
task at 100% accuracy in all seeds; last-word-only remains at 18.75%. Real
test arrays were not accessed, and no hardware or quantum speedup was tested.

`uv run ruff format .` and `uv run ruff check .` pass. Explicit Pyright on the
new module and tests has zero errors/warnings; `uv run pyright .` has zero
errors and the nine existing optional ReportLab warnings. With one numerical
thread, `uv run pytest -q --tb=short` passes **265 tests, 46 hardware skips**
in 29.17s, including seven new scientific regression checks. These establish
independent exponential/gradient agreement, norm preservation, paired controls,
learning on unseen toy combinations, nontrivial last-word invariance, vocabulary
relabeling and grouped loading without test arrays. Previous working models
and local edits are preserved.

Raw source/lock snapshots, hash manifests, selected weights, split identities
and curves remain in ignored `outputs/contextual-hamiltonian/`. The concise
POC report retains aggregate and per-seed numbers. The executed contextual
source hash matches its retained snapshot; tuning subsequently exposed weight
decay while preserving its default value and original training trajectory.
All gains are exploratory validation
results on reused data; component attribution, tuning/convergence parity and
fresh held-out confirmation remain open.

## Directional model mechanism check (2026-10-23)

Executed a one-thread inline `uv run python -` diagnostic with NumPy, Torch,
`HamiltonianModel(method="directed", layers=16)`, the saved screen dataset and
`refinement_screen/directed_count_L16_s11.pt`. Only the first 32 validation
examples were used for forward/gradient checks; no test examples were scored.
The diagnostic loaded the saved selected parameters, evaluated original,
half-time, double-time and zero-context-phase variants, then differentiated
the original NLL. Full train/validation arrays were used only to count target
occurrences in context. The local output is `/tmp/qcse-directional-diagnosis.json`.

At original / half / double time, mean target-role success was 0.00061402 /
0.00015529 / 0.00234730; raw conditional total-variation distance to the bigram
was 0.003756 / 0.001326 / 0.015315. Original subset gradient norms were 0.17040
for beta, 0.01554 for gamma and 0.02378 for the mixture logit. These float32
subset values are mechanism checks, not scaling/precision or convergence
claims. Validation targets appeared in their input context 7.7148% of the time.
No model was trained, no hardware execution was attempted, and no full-set
performance comparison is inferred from these 32 examples. The diagnosis and
untested model proposal are recorded in [findings](findings.md).

## Hamiltonian focus cleanup (2026-10-23)

The cleanup retired 186 tracked files: 113 historical result exports, 27 archive
notes, three superseded overview/planning documents, 30 one-off scripts, 11
cohort-specific test files, and two loose training artifacts. The circuit
coordinate/Schmidt checks were retained in `tests/test_gate_coordinates.py`.
The Hamiltonian implementation, records, figures and PDF were preserved.

Executed checks:

- `uv run ruff format .`: 110 files formatted; a second pass left all unchanged.
- `uv run ruff check .`: passed.
- `uv run pyright .`: zero errors; nine pre-existing warnings for unavailable
  optional ReportLab PDF-export imports. Five existing encoding/readout type
  errors were resolved with explicit non-None invariants and a complex-array
  annotation; the simulation formulas were preserved.
- With one numerical thread, `uv run pytest -q --tb=short`: **258 passed,
  46 hardware skips**, 30.24 seconds. This includes the Hamiltonian and retained
  generic preparation/source-snapshot tests. After the gate test's type narrowing,
  its targeted rerun passed all seven tests in 2.29 seconds; these are included
  in the suite count, not seven extra independent checks.
- `uv run python scripts/check_backend.py --device cpu`: passed at 14 qubits /
  64 layers; maximum error versus Qiskit 5.8842e-15 and versus sequential
  simulation zero. No CUDA/MPS allocation was available.
- Production CLI, Hamiltonian runner and robust runner help commands passed.
- All 60 retained local Markdown file links resolve. Retained Bash/Slurm files
  pass `bash -n`, with no references to retired scripts. `git diff --check` passes.
- All 48 numbers in the retained eight-model quality table match the historical
  `fb71a4a:results/robust-causal-controls/mean-scores.csv` at printed precision.

Fresh robust source snapshots now exclude three retired cohort drivers. Frozen
manifests retain their original hashes and require their historical checkout;
no guard was bypassed. Pre-existing dependency, backend-check and Slurm edits
were preserved byte-for-byte. Their initial patch and files were backed up to
`/tmp/qcse-before-cleanup-dhx_bg26/`; edited overview docs retain their CPU/CUDA
setup and validation information.

No new training or DGX submission was performed. Retained findings transcribe
prior audited reports; they do not claim to reproduce omitted raw predictions,
trained checkpoints or DGX draw streams. Ignored local run artifacts were kept.

## Opt-in CUDA dependencies (2026-10-22)

`uv sync --locked` now selects the default `cpu` and `dev` groups. The actual
sync completed with PyTorch 2.14.0+cpu; `torch.version.cuda` is `None`, and installed
package metadata contains no `cuda-*`, `nvidia-*` or Triton packages.
`uv sync --locked --check` and `uv pip check` pass.

Fresh-environment `uv sync --locked --dry-run` plans 47 packages with no GPU
support packages. `uv sync --locked --no-group cpu --group cuda --dry-run`
plans 66 packages, including the original PyPI PyTorch 2.14.0 and 19
CUDA/NVIDIA/Triton packages. Enabling both groups is rejected as incompatible.
The GPU environment was not installed or executed: this machine has no NVIDIA
GPU. GPU jobs must retain `uv run --no-sync` after opting into CUDA.

With one numerical thread, `uv run --no-sync pytest -q --tb=short` passed
**308 tests with 46 hardware skips** in 47.44s. The CPU
`scripts/check_backend.py --device cpu` preflight passed at 14 qubits and 64
layers, with max absolute error 5.89e-15 against Qiskit. The CLI help,
`uv run --no-sync ruff check .`, configured `uv run --no-sync pyright`, modified
backend-script formatting, Slurm Bash syntax and `git diff --check` pass.

Broader baseline checks still report five Pyright errors in the encoding and
measurement-basis experiments when running `pyright .`, and six existing files
needing formatting with `ruff format --check .`. Those files were not changed
by the dependency setup.

## Hamiltonian study recovery (2026-10-22)

The tracked Hamiltonian study was restored from
`codex/hamiltonian-encoder-research`: 160 files under
`experiments/hamiltonian_study/`, its test, versioned Tatoeba source and PDF.
The source dataset SHA-256 is
`3636ce3f85e9729e1af44ca004d54613a3a738f4a48df725a4d06073035b5706`,
matching the study manifests and the existing root `tatoeba.csv`. All 237 local
run files from the stash remain ignored in that study directory.

`uv run --no-sync pytest -q tests/test_hamiltonian_study.py` passed 21 tests;
the full one-thread suite passed 308 tests with 46 hardware skips. Ruff on
`src scripts tests experiments/hamiltonian_study` and project Pyright passed.
These checks establish local CPU behavior and archive integrity; no new
Hamiltonian training or CUDA run was performed.

## Historical checks

Earlier merge/benchmark validation records remain recoverable at
`git show fb71a4a:docs/validation.md`. The historical benchmark source and
report provenance are summarized in [findings](findings.md#evidence-and-recovery).

CPU checks cannot establish CUDA/MPS behavior. Full-trajectory microbatch-64
parity and full-fit speedup remain unvalidated. New trained-state plateau probes
and a fresh external confirmation population remain future work. The separate
Hamiltonian study already implements joint-Born and graph controls, under its
own protocol; it does not validate those alternatives for every robust adapter.
