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
- Launch chunking is tested at 1,230 tasks: arrays 0–999 and 0–229, each capped
  at five simultaneous jobs, with task offset 1,000 in the second array.
  No scheduler dependency or CUDA device-mask override is introduced.

## Remaining measurements

No job was submitted, no GPU fit ran locally, and there is no new production
quality result. Runtime, 10 GB peak-memory feasibility and CUDA gradients at
the full vocabulary/deepest ansatz must be measured on DGX. All-family local
preflight uses a small register; the existing production preflight uses the
full 14-qubit/64-layer simulator. Exact autograd in the new deep adapter still
needs actual allocated-device execution. Use the prepared protocol and
BENCHMARK.md commands on the cluster's locked checkout; retain all outputs/logs.
