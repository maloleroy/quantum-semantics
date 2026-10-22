# Validation and evidence boundaries

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

## Main merge checks (2026-10-22)

The merge of `codex/qcse-bert-robust-benchmark` into `main` retained both
focused studies from main and added learned local readout to the production
tensor simulator. The simulator now applies the last ansatz diagonal layer
before a learned readout; it still skips that layer for Z readout, where it
cannot affect the marginals. New parity tests compare both learned bases,
forward/reverse circuits and two simultaneous weight vectors with Qiskit.

With one numerical thread, `uv run --no-sync pytest -q --tb=short` passed:
**287 passed, 46 hardware skips**. `uv run --no-sync ruff check src scripts tests`
and `uv run --no-sync pyright` passed. The CPU
`scripts/check_backend.py --device cpu` preflight passed at 14 qubits and 64
layers, with max absolute error 5.89e-15 against Qiskit. The available PyTorch
is 2.14.0+cpu; no CUDA allocation was tested. The local environment was
completed with the lockfile's Transformers 4.57.6 after a full `uv sync --locked`
would have downloaded CUDA packages beyond the available disk space.

The latest causal audit checks 131 exported hashes, 34 frozen source copies
against 483bea6, three copied extensions against 6d8ca19, parent anchors, manifest/
split/target fingerprints, reconstructed pilot inputs, all eight new fit ledgers,
full validation sums and resource overlap. See
[the exact scope and reproducer](../results/robust-causal-controls/README.md).
It cannot verify omitted raw predictions, checkpoints or actual draw streams.

Earlier modern cohorts have their own hash/audit records under `results/`.
The legacy focused reports newly committed during cleanup are historical
summaries, not new independently retrained or hash-certified DGX experiments.
Historical test counts are in the [old inventory](archive/TEST_INVENTORY.md).

## Current cleanup checks

2026-10-22 cleanup: **245 passed, 46 skipped in 57.79s**, one numerical thread.
The skipped backend tests require unavailable CUDA/MPS. Full `src`/`scripts`/
`tests` Ruff checks and the project Pyright check passed; changed scripts also
pass an explicit Pyright run because the default scope is `src`/`tests`.
Bash syntax and whitespace checks passed.

The documentation audit covers 57 Markdown files and 309 local links, with no
missing in-repository file/anchor targets. One archived reference points to a
local PDF outside this repository; the current README links the official paper.
Both historical focused report collectors reproduce their retained 16/14 CSV
rows from completed source metadata, and both complete report generators render
successfully into ignored outputs. All 30 revised local completion markers are
present. This verifies summaries/status, not a new training reproduction.
Representative rendered figures were inspected; causal exact-accuracy axes now
retain sub-percent precision instead of rounding all labels to zero.

| Changed behavior | Regression coverage |
| --- | --- |
| Batched fixed encoding | All five scalar context formulas, lengths 1/3/5, forward/reverse Qiskit state parity, mixed context lengths, per-call batch bounds and empty axes. |
| Streamed states | Cache eviction/order/duplicate IDs and independent CPU allocations, plus existing training/SPSA/simulation/resume tests. |
| Legacy focused selection | Retrieval/BCE tie handling, selected full embeddings, explicit unsupported SPSA early-stop continuation, chronological semantic checkpoint versus selected inference state. |
| Semantic/attention resume | Existing uninterrupted/resumed parity and new saved-patience/selection-state regression. |
| Focused Slurm driver | Mocked sequential waited arrays, concurrency/resource caps, inherited override cleanup, active/unknown queue rejection and failure stop. No real job submission. |
| Research probes | Existing fixed-code decoder, CRZ/ZZ state/gradient conversion, chain Schmidt bounds, GPT causal/PAD/tied-head behavior and classical-feature gradients. |
| Documentation | Internal file/anchor links, archive relocation, result/report references, usable CLI help and no conversation files in Git. |

One-thread local check commands:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --no-sync pytest -q
uv run --no-sync ruff check src scripts tests
uv run --no-sync pyright
git diff --check
```

No full-corpus training belongs in these checks. GPU tests skip when the backend
is unavailable. CPU results do not establish new CUDA speed/parity. Received
DGX preflights certify their frozen source/conditions, not every subsequent edit.
For a changed backend, run `check_backend.py` or `check_benchmark_backend.py`
inside an actual `sbatch` allocation before using the change in a study.

## What has not been validated

- Full-trajectory micro 64 behavior or full-fit speedup; one-update summaries
  and single-batch timing are the currently received evidence.
- New GPT/classical-feature factory/resume/CUDA integration.
- Joint-Born/learned-head alternatives, fixed-head trained-state plateau probes,
  new samplers, graph evolution or exact-MPS backend.
- A fresh external population or a semantic downstream task. Reused validation,
  transductive IDs and small seed counts remain scientific limits.
- Safety of inherited pre-branch launchers under the four-MIG policy; supported
  drivers and remaining legacy exceptions are listed in [DGX operations](dgx-a100.md).
