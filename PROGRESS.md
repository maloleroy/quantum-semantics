# QCSE research progress

## Scope and rules — 2026-09-09

Audit the existing Qiskit implementation against `../paper4.pdf`, then run a small,
predeclared selection from `../QCSE_ideas.md` (the actual filename). Keep simple
working code, real simulation, local atomic commits, and no pushes. Every experiment
gets a report; this file keeps the detailed decisions, commands, failures, and results.
Branch: `research/qcse-ordering`, starting from the clean `main` checkout.

## Initial inspection

- Repository is `quantum-semantics`, not its parent Papers directory.
- Source modules: context, circuit, model, data, training, CLI; existing tests include
  independent dense matrices, bit order, checkpoint roundtrip and complete CLI runs.
- Paper is arXiv:2509.05729v2, 10 March 2026. Existing extracted text is available
  at `../.work/paper4.txt`; extraction will be checked against the supplied PDF.
- Default corpus is different from the paper: 1,000 phrases rather than its 110
  English sentences. Paper accuracy must not be treated as a reproducible target.
- Default window 4 and vocabulary 729 imply 10 qubits and only ONE encoding layer.
  Between-layer experiments must use a longer context and report the actual layer counts.
- Final RZ/CRZ gates are diagonal, so they cannot affect Z probabilities. First RX
  rotations act on |+>, so their angles affect only global phase.
- Paper's half-bits-match metric is weak. BCE is primary; bit and exact word accuracy,
  invalid decoded IDs, initial scores, and train-only classical controls are required.
- Frequency-ranked vocabulary uses the complete corpus: this is transductive and
  test frequencies affect IDs. Research protocol will instead use alphabetical IDs,
  still transparently exposing the full vocabulary, and freeze output IDs when only
  input numerical IDs change. Joint input/target relabeling is a separate condition.

## Bounded experiment plan (before outcomes)

1. Paper audit: formula/gate checks, inactive parameter diagnostics, metric null
   baselines, original-default short end-to-end training on sentence-held-out data.
2. Input word IDs: canonical vs three fixed random bijections, plus a clearly
   separate joint input/target relabeling control. Keep every run, never select the best.
3. Between encoding layers: same layout vs fixed random vs layer-specific pair
   layouts and reverse upload order, with a window that actually makes multiple layers.
   Also compare fixed input IDs against a different frozen ID map at each upload layer.
4. Gate order: RX/RZ vs RZ/RX execution, reverse CNOT execution order (same edges),
   and commuting-CRZ order as a numerical invariance check.
5. Bit mean family: train-target bit prior, raw mean of context bits, fitted bit-mean
   predictor; include shuffled training labels as a negative control.
6. Local Qwen: embed vocabulary once with Qwen3-Embedding-0.6B, compare semantic
   layout with shuffled-vector layout, then direct projected Qwen representation
   with a classical predictor using exactly the same features.

Small pilot: paired seeds, fixed split and examples, fixed training budget; no test
selection, large sweeps, or quantum-advantage claims. Concrete counts and settings
will be recorded before the runner starts. Preserve raw predictions and provenance.

## Environment log

- `uv sync` could not write the default user cache. Redirected cache to
  `/private/tmp/qcse-uv-cache`.
- Checkout requests Python 3.12, but local Python 3.11.14 is already installed and
  satisfies pyproject's >=3.11. Use that explicit interpreter instead of downloading
  another Python. Sandbox networking blocked package downloads; authorized escalated
  dependency installation is in progress.
