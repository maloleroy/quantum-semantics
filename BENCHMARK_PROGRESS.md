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
- [ ] Shared prepared corpus/splits, training, resume, per-example metrics.
- [ ] Validation-only selection, development refits, paired sentence bootstrap.
- [ ] Frozen large manifest, independent DGX array and retained job provenance.
- [ ] Final validation and measured limitations.

Validation is deferred until the implementation is complete, as requested.
