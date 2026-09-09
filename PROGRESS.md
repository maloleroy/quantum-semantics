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

## User clarification: semantic bit families

The user clarified that “bit mean-family” means semantic bit codes: related meanings
should differ by a few bits, potentially with attribute-like bit roles. Revised
experiment 5 accordingly: balanced recursive partitioning of Qwen vocabulary vectors
assigns unique 10-bit codes; compare alphabetical IDs and a shuffled assignment of
exactly the same semantic code set. Preserve all input features, change targets only.
Report Qwen-neighbor Hamming distances and context prediction separately. These are
basis-state labels: different bitstrings are orthogonal quantum states, even when
Hamming-close. A claim of close full quantum states requires fidelity measurements.
Classical bit-mean methods remain controls, not the requested semantic representation.
No manually assigned attribute meaning will be claimed for an unsupervised bit.

## Implementation and environment completed

- All original 15 tests passed before edits (4.65 s including first imports).
- Extracted all 15 pages directly from paper4.pdf using pypdf. SHA256:
  `e01a1d2cb98c2ba74081597e0d5131ca977a0fe15cb1d1a37df805fd93160768`.
  Direct extraction confirms equations 10–28 and the original setup.
- `research/variants.py` builds actual Qiskit circuits for every variant. A small
  NumPy batch evaluator applies the bound Qiskit RX/RZ/CRZ matrices to full complex
  statevectors. It is not a surrogate. Gate/order variants and canonical results are
  compared to Qiskit, including full amplitudes and normalization, at ~1e-14 tolerance.
- Batched 64-example, 10-qubit, 2-layer objective benchmark: 0.0183 seconds on this
  machine. This is batching overhead reduction, not a quantum speedup.
- Dependency installation temporarily let `uv add` download its requested Python 3.12
  and recreate the environment. Final `uv sync --extra embeddings --python ...3.11...`
  restored Python 3.11.14; lockfile captures both supported resolution branches.
- Current 26 tests pass, including train-only projection and held-out label isolation.
- Research models deliberately refuse base-model `save()`: that format would lose
  ordering configuration. The runner saves variant config, initial/final weights,
  codebooks, splits, projections and raw probabilities instead.

## Local vocabulary embedding completed

- Actual model: `Qwen/Qwen3-Embedding-0.6B`, immutable revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- Local CPU, float32, 4 threads, batch 16, no task prompt; isolated vocabulary words,
  left padding, last-token pooling, L2 normalization. No inference API used.
- 729 x 1024 vectors; longest vocabulary tokenization 4 tokens.
- Download/load: 49.53 s; vocabulary inference: 7.91 s; peak process RSS 2,021,179,392
  bytes (~1.88 GiB). Fits within the requested M2 16GB budget on CPU.
- Artifacts: `artifacts/qwen/vocabulary.npz`, `metadata.json`; model weights stay in
  ignored `outputs/huggingface`. Exact vector/vocabulary hashes are in metadata.
- Followed the official model card's pooling convention:
  https://huggingface.co/Qwen/Qwen3-Embedding-0.6B .

## Frozen pilot settings, recorded before training

- 19 conditions x 3 initialization/SPSA seeds (11, 23, 37) = 57 short runs.
  This is a small directional pilot, not 57 independent datasets or an architecture search.
- Entire examples from 24 train and 8 test sentence-text groups. Split seed 31415,
  subset selection seed 2718. All conditions share these exact examples.
- Window 8, 10 qubits, 2 ansatz layers, 20 epochs, batch 32, Adam-SPSA,
  learning rate 0.003, L2 0.001. Rate/epoch budget intentionally differ from paper
  for the bounded pilot; a separate audit uses original rate 0.0003 and 50 epochs.
- Three fixed ID map seeds 101, 202, 303, with all outcomes retained. No test-based
  map choice, checkpoint choice, learning-rate selection, or early stopping.
- No-entanglement control retains 58 parameter slots to pair initialization, but
  only 40 rotation parameters occur in the circuit. Reports distinguish both counts.
- Direct Qwen features use PCA fitted only on train context means, dimension 20,
  scaled by training standard deviation to 0.5 radians. A classical logistic head
  receives exactly the same 20 context features, with its own disclosed budget.
- Shuffled training labels and shuffled vocabulary embeddings are explicit negative
  controls. The semantic code control permutes the same set of valid codewords.
- Primary outcome is test BCE; all bit/exact/valid-code accuracies and initialization
  metrics will also be preserved. Valid-code decoding uses factorized marginal
  likelihood, not an asserted joint quantum probability distribution.

## Geometry follow-up (diagnostic, no new training sweep)

The user’s phrase “similar quantum state” motivated an additional invariant check.
For any shared ansatz U, <Uψ|Uφ>=<ψ|φ>. Thus training this QCSE ansatz cannot change
full-state fidelity across contexts; only measurement-space geometry can change.
Added an explicit numerical test and a diagnostic that measures full-state fidelity
and marginal distance on held-out contexts for canonical/semantic/random targets.

A semantic binary code does not itself make computational basis states close: any
unequal basis labels are orthogonal. A separate explicit nonorthogonal construction
uses tensor products of Ry((pi/3)*bit)|0>. Its fidelity is (3/4)^Hamming distance.
This is checked against Qiskit and reported solely as a representation diagnostic,
not a new trained QCSE model. It shows how to implement actual Hamming-related
state proximity without pretending that basis encoding already has it.

Initial codebook geometry (before training outcomes): semantic recursive codes improve
on shuffled assignments but do not beat alphabetical IDs on average neighbor Hamming
distance. Retain this negative finding; do not search another codebook after seeing it.
Alphabetical order can retain morphology, which is a useful baseline rather than a
meaningless arbitrary control. No gender/species/quality bit semantics were assumed.
