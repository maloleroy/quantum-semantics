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

## Final validation and deliverables

- Final suite: 28 tests pass; Ruff passes; `git diff --check` passes.
- Recomputed every reported train/test metric directly from all 57 saved prediction
  archives; every value agrees within 1e-12. All 21 history records per run are present.
- Verified report/progress/skill local links resolve; visually inspected the scientific
  plot. Error bars are sample SD across three seeds, not confidence intervals.
- Both reusable skill files pass the skill-creator validator. Updated them with the
  semantic-code clarification, fidelity invariance and small-data limitations.
- Full original-evaluator audit and every planned pilot condition finished; no runs
  were omitted, stopped early, or promoted using test metrics.
- Source import issue for research tests was fixed by explicitly adding the repository
  root to pytest's path. Report generation's Python 3.11 syntax and string formatting
  issues were fixed before outputs were produced. Matplotlib's initial unwritable
  user font-cache warning was resolved through a workspace-local cache configuration.
- Deliverables: seven experiment reports, one short summary, PNG/PDF scientific
  figure, all numerical artifacts (~5.5 MB including vocabulary vectors), two skills,
  simple research code, and this complete training ledger. Downloaded ~1.1 GB model
  weights remain local and ignored; no remote writes or pushes.
- The post-run variant validation change only rejects invalid configurations and
  does not alter any of the 19 valid configurations used in the committed pilot.

## Completed experiment ledger (generated from saved results)

All 57 pilot runs completed; total measured model-run time 213.97 s. Separate original-evaluator audit: 36.78 s. Projection, embedding and reporting time are not included.

The sample has 177 training and 57 test occurrences. Of 57 test targets, 26 were
unseen in selected training labels. No exact context tuple crosses the split.
Full protocol and source commit are stored in artifacts/pilot/protocol.json.

### Condition overview

| Condition | Initial test BCE | Final test BCE (mean ± SD) | Bit accuracy | Exact words | Valid-code words |
|---|---:|---:|---:|---:|---:|
| canonical | 0.7256 | 0.7214 ± 0.0095 | 51.6% | 0.0% | 0.0% |
| id-map-101 | 0.7170 | 0.7160 ± 0.0118 | 51.2% | 0.0% | 0.0% |
| id-map-202 | 0.7068 | 0.7032 ± 0.0051 | 53.6% | 0.0% | 1.2% |
| id-map-303 | 0.7119 | 0.7093 ± 0.0085 | 51.2% | 0.0% | 0.0% |
| joint-id-map | 0.7207 | 0.7171 ± 0.0031 | 51.2% | 0.0% | 0.0% |
| fixed-pairs | 0.7204 | 0.7165 ± 0.0088 | 49.9% | 0.0% | 0.6% |
| layer-pairs | 0.7198 | 0.7163 ± 0.0097 | 49.7% | 0.0% | 0.6% |
| reverse-uploads | 0.7163 | 0.7128 ± 0.0052 | 52.7% | 0.0% | 0.0% |
| layer-id-maps | 0.7147 | 0.7137 ± 0.0091 | 53.0% | 0.0% | 0.0% |
| rz-before-rx | 0.7457 | 0.7308 ± 0.0265 | 50.4% | 0.0% | 0.0% |
| reverse-cnot | 0.6990 | 0.6973 ± 0.0033 | 50.3% | 0.0% | 0.0% |
| shuffled-labels | 0.7256 | 0.7203 ± 0.0101 | 51.6% | 0.0% | 0.0% |
| semantic-codes | 0.7285 | 0.7235 ± 0.0127 | 48.8% | 0.0% | 0.0% |
| random-codes | 0.7342 | 0.7288 ± 0.0184 | 50.2% | 0.0% | 0.0% |
| qwen-layout | 0.7201 | 0.7172 ± 0.0097 | 49.6% | 0.0% | 0.0% |
| shuffled-qwen-layout | 0.7236 | 0.7186 ± 0.0187 | 50.2% | 0.0% | 0.6% |
| qwen-layout-no-entanglement | 1.0465 | 1.0065 ± 0.0486 | 49.5% | 0.0% | 0.0% |
| shuffled-layout-no-entanglement | 1.0043 | 0.9705 ± 0.0199 | 50.2% | 0.0% | 0.0% |
| qwen-direct | 0.7197 | 0.7061 ± 0.0205 | 53.3% | 0.0% | 0.0% |

### Audit evidence

| Model | Test BCE | Bit accuracy | Exact words | Paper half-bits metric |
|---|---:|---:|---:|---:|
| QCSE initial | 0.7283 | 47.4% | 0.0% | 54.4% |
| QCSE epoch 50 | 0.7200 | 47.9% | 0.0% | 56.1% |
| train_bit_prior | 0.5029 | 74.7% | 22.8% | 96.5% |
| context_bit_mean | 2.6731 | 64.4% | 0.0% | 93.0% |
| linear_bit_mean | 0.4964 | 73.5% | 15.8% | 94.7% |

### Decisions and limits

- No quantum condition beats its canonical constant-prior baseline in the input-only comparisons.
- Every quantum run has zero threshold exact-word accuracy; valid-code ranking occasionally
  recovers a word and is reported separately, never substituted silently.
- Much of the apparent order advantage is already present at initialization.
- Shuffled labels match real-label test performance: contextual learning remains unproven.
- No new codebook or architecture was selected after outcomes.
- Semantic lexical geometry uses the same Qwen space that constructed the codes; independent
  attribute tests are still needed before labeling individual bits.
- The direct quantum Qwen map discards first RX-channel information; this is an architectural
  limitation to test explicitly before an enlarged experiment.
- Next useful work would be a small controlled learnability task and a context-dependent
  trainable encoder if full-state metric learning is desired. It is not another broad sweep.

### Complete per-run training record

The following tables come from saved history files, including epoch zero. Test scores
were observed for reporting only. They never selected weights or changed the schedule.
For shuffled-label runs, training BCE is against shuffled labels; all other training
BCE and every test BCE use that run’s actual target codebook. Final real-label training
scores for the shuffle control are stored separately in result.json.


#### canonical-s11

Runtime 3.88 s; weights L2 change 0.484937; Qiskit maximum marginal error 6.66e-16. [Saved run](artifacts/pilot/canonical-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.728761 | 0.736462 | 0.5228 | 0.0000 |
| 1 | 0.728527 | 0.735777 | 0.5263 | 0.0000 |
| 2 | 0.728496 | 0.735797 | 0.5263 | 0.0000 |
| 3 | 0.728351 | 0.735614 | 0.5246 | 0.0000 |
| 4 | 0.728127 | 0.735518 | 0.5228 | 0.0000 |
| 5 | 0.727866 | 0.735427 | 0.5211 | 0.0000 |
| 6 | 0.727558 | 0.735264 | 0.5211 | 0.0000 |
| 7 | 0.727160 | 0.735016 | 0.5193 | 0.0000 |
| 8 | 0.726794 | 0.734692 | 0.5193 | 0.0000 |
| 9 | 0.726524 | 0.734526 | 0.5193 | 0.0000 |
| 10 | 0.726179 | 0.734261 | 0.5193 | 0.0000 |
| 11 | 0.725729 | 0.733726 | 0.5175 | 0.0000 |
| 12 | 0.725332 | 0.733363 | 0.5175 | 0.0000 |
| 13 | 0.724974 | 0.733227 | 0.5158 | 0.0000 |
| 14 | 0.724697 | 0.733061 | 0.5158 | 0.0000 |
| 15 | 0.724313 | 0.732749 | 0.5175 | 0.0000 |
| 16 | 0.723890 | 0.732552 | 0.5193 | 0.0000 |
| 17 | 0.723552 | 0.732478 | 0.5193 | 0.0000 |
| 18 | 0.723304 | 0.732440 | 0.5175 | 0.0000 |
| 19 | 0.723124 | 0.732427 | 0.5193 | 0.0000 |
| 20 | 0.722979 | 0.732352 | 0.5193 | 0.0000 |

#### canonical-s23

Runtime 3.88 s; weights L2 change 0.429862; Qiskit maximum marginal error 3.89e-16. [Saved run](artifacts/pilot/canonical-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.726767 | 0.722413 | 0.5175 | 0.0000 |
| 1 | 0.726451 | 0.721914 | 0.5175 | 0.0000 |
| 2 | 0.726236 | 0.721633 | 0.5193 | 0.0000 |
| 3 | 0.726088 | 0.721426 | 0.5193 | 0.0000 |
| 4 | 0.725943 | 0.721225 | 0.5175 | 0.0000 |
| 5 | 0.725690 | 0.720954 | 0.5175 | 0.0000 |
| 6 | 0.725244 | 0.720472 | 0.5175 | 0.0000 |
| 7 | 0.724854 | 0.720043 | 0.5175 | 0.0000 |
| 8 | 0.724484 | 0.719664 | 0.5193 | 0.0000 |
| 9 | 0.724105 | 0.719351 | 0.5193 | 0.0000 |
| 10 | 0.723750 | 0.718995 | 0.5211 | 0.0000 |
| 11 | 0.723502 | 0.718649 | 0.5211 | 0.0000 |
| 12 | 0.723295 | 0.718328 | 0.5211 | 0.0000 |
| 13 | 0.723091 | 0.718094 | 0.5228 | 0.0000 |
| 14 | 0.722548 | 0.717541 | 0.5228 | 0.0000 |
| 15 | 0.721920 | 0.717027 | 0.5211 | 0.0000 |
| 16 | 0.721522 | 0.716705 | 0.5211 | 0.0000 |
| 17 | 0.721242 | 0.716449 | 0.5211 | 0.0000 |
| 18 | 0.721087 | 0.716239 | 0.5211 | 0.0000 |
| 19 | 0.720981 | 0.716077 | 0.5211 | 0.0000 |
| 20 | 0.720769 | 0.715870 | 0.5211 | 0.0000 |

#### canonical-s37

Runtime 3.91 s; weights L2 change 0.606779; Qiskit maximum marginal error 7.77e-16. [Saved run](artifacts/pilot/canonical-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.718172 | 0.717881 | 0.5105 | 0.0000 |
| 1 | 0.718043 | 0.717538 | 0.5105 | 0.0000 |
| 2 | 0.717912 | 0.717438 | 0.5123 | 0.0000 |
| 3 | 0.717756 | 0.717343 | 0.5140 | 0.0000 |
| 4 | 0.717563 | 0.717167 | 0.5140 | 0.0000 |
| 5 | 0.717396 | 0.717050 | 0.5158 | 0.0000 |
| 6 | 0.717160 | 0.716878 | 0.5158 | 0.0000 |
| 7 | 0.716914 | 0.716703 | 0.5158 | 0.0000 |
| 8 | 0.716685 | 0.716546 | 0.5140 | 0.0000 |
| 9 | 0.716478 | 0.716486 | 0.5158 | 0.0000 |
| 10 | 0.716299 | 0.716443 | 0.5140 | 0.0000 |
| 11 | 0.716137 | 0.716423 | 0.5123 | 0.0000 |
| 12 | 0.716022 | 0.716410 | 0.5158 | 0.0000 |
| 13 | 0.715884 | 0.716368 | 0.5158 | 0.0000 |
| 14 | 0.715731 | 0.716290 | 0.5140 | 0.0000 |
| 15 | 0.715555 | 0.716214 | 0.5158 | 0.0000 |
| 16 | 0.715385 | 0.716177 | 0.5158 | 0.0000 |
| 17 | 0.715213 | 0.716122 | 0.5123 | 0.0000 |
| 18 | 0.715105 | 0.716043 | 0.5088 | 0.0000 |
| 19 | 0.714937 | 0.715925 | 0.5070 | 0.0000 |
| 20 | 0.714806 | 0.715839 | 0.5070 | 0.0000 |

#### id-map-101-s11

Runtime 3.87 s; weights L2 change 0.492979; Qiskit maximum marginal error 8.88e-16. [Saved run](artifacts/pilot/id-map-101-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.729654 | 0.705073 | 0.5053 | 0.0000 |
| 1 | 0.729553 | 0.705293 | 0.5070 | 0.0000 |
| 2 | 0.729368 | 0.705453 | 0.5070 | 0.0000 |
| 3 | 0.729093 | 0.705630 | 0.5053 | 0.0000 |
| 4 | 0.728780 | 0.705837 | 0.5070 | 0.0000 |
| 5 | 0.728482 | 0.705838 | 0.5053 | 0.0000 |
| 6 | 0.728132 | 0.705699 | 0.5035 | 0.0000 |
| 7 | 0.727947 | 0.705565 | 0.5035 | 0.0000 |
| 8 | 0.727830 | 0.705530 | 0.5035 | 0.0000 |
| 9 | 0.727711 | 0.705439 | 0.5035 | 0.0000 |
| 10 | 0.727524 | 0.705263 | 0.5035 | 0.0000 |
| 11 | 0.727044 | 0.705006 | 0.5035 | 0.0000 |
| 12 | 0.726632 | 0.704884 | 0.5035 | 0.0000 |
| 13 | 0.726381 | 0.704964 | 0.5053 | 0.0000 |
| 14 | 0.726085 | 0.705006 | 0.5018 | 0.0000 |
| 15 | 0.725686 | 0.704953 | 0.5018 | 0.0000 |
| 16 | 0.725406 | 0.704910 | 0.5018 | 0.0000 |
| 17 | 0.725164 | 0.704861 | 0.5018 | 0.0000 |
| 18 | 0.724978 | 0.704878 | 0.5018 | 0.0000 |
| 19 | 0.724768 | 0.704839 | 0.5035 | 0.0000 |
| 20 | 0.724420 | 0.704567 | 0.5035 | 0.0000 |

#### id-map-101-s23

Runtime 3.88 s; weights L2 change 0.399798; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/id-map-101-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.709780 | 0.731020 | 0.4947 | 0.0000 |
| 1 | 0.709735 | 0.731026 | 0.4930 | 0.0000 |
| 2 | 0.709465 | 0.730908 | 0.4982 | 0.0000 |
| 3 | 0.709302 | 0.730803 | 0.4965 | 0.0000 |
| 4 | 0.709189 | 0.730684 | 0.4965 | 0.0000 |
| 5 | 0.709030 | 0.730497 | 0.4965 | 0.0000 |
| 6 | 0.708874 | 0.730353 | 0.4947 | 0.0000 |
| 7 | 0.708783 | 0.730266 | 0.4895 | 0.0000 |
| 8 | 0.708723 | 0.730162 | 0.4895 | 0.0000 |
| 9 | 0.708666 | 0.730130 | 0.4947 | 0.0000 |
| 10 | 0.708616 | 0.730123 | 0.4965 | 0.0000 |
| 11 | 0.708545 | 0.729976 | 0.4965 | 0.0000 |
| 12 | 0.708507 | 0.729867 | 0.5000 | 0.0000 |
| 13 | 0.708405 | 0.729772 | 0.4982 | 0.0000 |
| 14 | 0.708190 | 0.729409 | 0.5000 | 0.0000 |
| 15 | 0.707987 | 0.729099 | 0.4965 | 0.0000 |
| 16 | 0.707804 | 0.728729 | 0.4965 | 0.0000 |
| 17 | 0.707678 | 0.728469 | 0.4965 | 0.0000 |
| 18 | 0.707634 | 0.728306 | 0.4965 | 0.0000 |
| 19 | 0.707584 | 0.728210 | 0.4965 | 0.0000 |
| 20 | 0.707538 | 0.728164 | 0.4965 | 0.0000 |

#### id-map-101-s37

Runtime 3.92 s; weights L2 change 0.577840; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/id-map-101-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.709109 | 0.714965 | 0.5456 | 0.0000 |
| 1 | 0.709055 | 0.714961 | 0.5456 | 0.0000 |
| 2 | 0.708961 | 0.715004 | 0.5439 | 0.0000 |
| 3 | 0.708894 | 0.715016 | 0.5474 | 0.0000 |
| 4 | 0.708830 | 0.715010 | 0.5421 | 0.0000 |
| 5 | 0.708592 | 0.715043 | 0.5439 | 0.0000 |
| 6 | 0.708307 | 0.715106 | 0.5439 | 0.0000 |
| 7 | 0.708139 | 0.715108 | 0.5439 | 0.0000 |
| 8 | 0.708018 | 0.715084 | 0.5421 | 0.0000 |
| 9 | 0.707908 | 0.715130 | 0.5368 | 0.0000 |
| 10 | 0.707842 | 0.715174 | 0.5368 | 0.0000 |
| 11 | 0.707759 | 0.715215 | 0.5351 | 0.0000 |
| 12 | 0.707645 | 0.715214 | 0.5351 | 0.0000 |
| 13 | 0.707519 | 0.715233 | 0.5351 | 0.0000 |
| 14 | 0.707266 | 0.715321 | 0.5333 | 0.0000 |
| 15 | 0.707073 | 0.715363 | 0.5333 | 0.0000 |
| 16 | 0.706921 | 0.715395 | 0.5351 | 0.0000 |
| 17 | 0.706823 | 0.715406 | 0.5368 | 0.0000 |
| 18 | 0.706696 | 0.715393 | 0.5351 | 0.0000 |
| 19 | 0.706552 | 0.715374 | 0.5351 | 0.0000 |
| 20 | 0.706399 | 0.715371 | 0.5351 | 0.0000 |

#### id-map-202-s11

Runtime 3.89 s; weights L2 change 0.386208; Qiskit maximum marginal error 6.11e-16. [Saved run](artifacts/pilot/id-map-202-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.734368 | 0.712385 | 0.5456 | 0.0000 |
| 1 | 0.733908 | 0.711967 | 0.5456 | 0.0000 |
| 2 | 0.733602 | 0.711780 | 0.5421 | 0.0000 |
| 3 | 0.733383 | 0.711598 | 0.5421 | 0.0000 |
| 4 | 0.733238 | 0.711446 | 0.5421 | 0.0000 |
| 5 | 0.733068 | 0.711279 | 0.5421 | 0.0000 |
| 6 | 0.732912 | 0.711083 | 0.5439 | 0.0000 |
| 7 | 0.732765 | 0.710932 | 0.5439 | 0.0000 |
| 8 | 0.732567 | 0.710817 | 0.5439 | 0.0000 |
| 9 | 0.732407 | 0.710696 | 0.5421 | 0.0000 |
| 10 | 0.732011 | 0.710510 | 0.5439 | 0.0000 |
| 11 | 0.731323 | 0.710215 | 0.5439 | 0.0000 |
| 12 | 0.730731 | 0.709953 | 0.5439 | 0.0000 |
| 13 | 0.730335 | 0.709781 | 0.5439 | 0.0000 |
| 14 | 0.730022 | 0.709626 | 0.5439 | 0.0000 |
| 15 | 0.729537 | 0.709331 | 0.5439 | 0.0000 |
| 16 | 0.729127 | 0.709102 | 0.5439 | 0.0000 |
| 17 | 0.728833 | 0.708933 | 0.5439 | 0.0000 |
| 18 | 0.728597 | 0.708811 | 0.5439 | 0.0000 |
| 19 | 0.728361 | 0.708708 | 0.5439 | 0.0000 |
| 20 | 0.728106 | 0.708637 | 0.5439 | 0.0000 |

#### id-map-202-s23

Runtime 3.88 s; weights L2 change 0.398636; Qiskit maximum marginal error 7.77e-16. [Saved run](artifacts/pilot/id-map-202-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.718640 | 0.708856 | 0.5544 | 0.0000 |
| 1 | 0.718512 | 0.708594 | 0.5509 | 0.0000 |
| 2 | 0.718045 | 0.708041 | 0.5368 | 0.0000 |
| 3 | 0.717666 | 0.707538 | 0.5404 | 0.0000 |
| 4 | 0.717276 | 0.707081 | 0.5491 | 0.0000 |
| 5 | 0.716921 | 0.706663 | 0.5526 | 0.0000 |
| 6 | 0.716396 | 0.706302 | 0.5526 | 0.0000 |
| 7 | 0.715953 | 0.705984 | 0.5491 | 0.0000 |
| 8 | 0.715506 | 0.705661 | 0.5404 | 0.0000 |
| 9 | 0.715034 | 0.705297 | 0.5368 | 0.0000 |
| 10 | 0.714709 | 0.705044 | 0.5368 | 0.0000 |
| 11 | 0.714366 | 0.704824 | 0.5404 | 0.0000 |
| 12 | 0.714082 | 0.704703 | 0.5368 | 0.0000 |
| 13 | 0.713781 | 0.704499 | 0.5368 | 0.0000 |
| 14 | 0.713206 | 0.703999 | 0.5368 | 0.0000 |
| 15 | 0.712712 | 0.703537 | 0.5368 | 0.0000 |
| 16 | 0.712382 | 0.703188 | 0.5368 | 0.0000 |
| 17 | 0.712146 | 0.702907 | 0.5333 | 0.0000 |
| 18 | 0.712023 | 0.702758 | 0.5333 | 0.0000 |
| 19 | 0.711838 | 0.702560 | 0.5333 | 0.0000 |
| 20 | 0.711662 | 0.702445 | 0.5316 | 0.0000 |

#### id-map-202-s37

Runtime 3.92 s; weights L2 change 0.510625; Qiskit maximum marginal error 6.66e-16. [Saved run](artifacts/pilot/id-map-202-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.714133 | 0.699291 | 0.5140 | 0.0000 |
| 1 | 0.714091 | 0.699265 | 0.5140 | 0.0000 |
| 2 | 0.714043 | 0.699230 | 0.5123 | 0.0000 |
| 3 | 0.713912 | 0.699150 | 0.5105 | 0.0000 |
| 4 | 0.713717 | 0.699027 | 0.5140 | 0.0000 |
| 5 | 0.713594 | 0.698957 | 0.5140 | 0.0000 |
| 6 | 0.713433 | 0.698875 | 0.5140 | 0.0000 |
| 7 | 0.713240 | 0.698809 | 0.5140 | 0.0000 |
| 8 | 0.712942 | 0.698672 | 0.5175 | 0.0000 |
| 9 | 0.712716 | 0.698632 | 0.5158 | 0.0000 |
| 10 | 0.712564 | 0.698643 | 0.5211 | 0.0000 |
| 11 | 0.712419 | 0.698697 | 0.5263 | 0.0000 |
| 12 | 0.712311 | 0.698719 | 0.5298 | 0.0000 |
| 13 | 0.712229 | 0.698811 | 0.5316 | 0.0000 |
| 14 | 0.712139 | 0.698795 | 0.5333 | 0.0000 |
| 15 | 0.711979 | 0.698641 | 0.5333 | 0.0000 |
| 16 | 0.711821 | 0.698482 | 0.5316 | 0.0000 |
| 17 | 0.711705 | 0.698319 | 0.5316 | 0.0000 |
| 18 | 0.711627 | 0.698264 | 0.5316 | 0.0000 |
| 19 | 0.711574 | 0.698351 | 0.5333 | 0.0000 |
| 20 | 0.711522 | 0.698475 | 0.5316 | 0.0000 |

#### id-map-303-s11

Runtime 3.83 s; weights L2 change 0.431762; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/id-map-303-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.734396 | 0.721202 | 0.5105 | 0.0000 |
| 1 | 0.733883 | 0.720893 | 0.5088 | 0.0000 |
| 2 | 0.733476 | 0.720651 | 0.5105 | 0.0000 |
| 3 | 0.733152 | 0.720459 | 0.5088 | 0.0000 |
| 4 | 0.732817 | 0.720322 | 0.5088 | 0.0000 |
| 5 | 0.732483 | 0.720214 | 0.5070 | 0.0000 |
| 6 | 0.732154 | 0.720044 | 0.5070 | 0.0000 |
| 7 | 0.731930 | 0.719954 | 0.5088 | 0.0000 |
| 8 | 0.731737 | 0.719890 | 0.5088 | 0.0000 |
| 9 | 0.731527 | 0.719796 | 0.5088 | 0.0000 |
| 10 | 0.731211 | 0.719691 | 0.5088 | 0.0000 |
| 11 | 0.730560 | 0.719502 | 0.5088 | 0.0000 |
| 12 | 0.730019 | 0.719345 | 0.5070 | 0.0000 |
| 13 | 0.729484 | 0.719170 | 0.5070 | 0.0000 |
| 14 | 0.729012 | 0.719017 | 0.5158 | 0.0000 |
| 15 | 0.728556 | 0.718809 | 0.5175 | 0.0000 |
| 16 | 0.728147 | 0.718586 | 0.5175 | 0.0000 |
| 17 | 0.727853 | 0.718423 | 0.5158 | 0.0000 |
| 18 | 0.727525 | 0.718306 | 0.5123 | 0.0000 |
| 19 | 0.727270 | 0.718217 | 0.5123 | 0.0000 |
| 20 | 0.726941 | 0.718014 | 0.5140 | 0.0000 |

#### id-map-303-s23

Runtime 3.78 s; weights L2 change 0.403516; Qiskit maximum marginal error 1.22e-15. [Saved run](artifacts/pilot/id-map-303-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.706727 | 0.704914 | 0.5070 | 0.0000 |
| 1 | 0.706602 | 0.704874 | 0.5053 | 0.0000 |
| 2 | 0.706344 | 0.704702 | 0.4965 | 0.0000 |
| 3 | 0.706163 | 0.704529 | 0.4982 | 0.0000 |
| 4 | 0.705979 | 0.704293 | 0.4982 | 0.0000 |
| 5 | 0.705737 | 0.704097 | 0.4965 | 0.0000 |
| 6 | 0.705319 | 0.703805 | 0.4965 | 0.0000 |
| 7 | 0.704974 | 0.703553 | 0.4965 | 0.0000 |
| 8 | 0.704700 | 0.703314 | 0.4912 | 0.0000 |
| 9 | 0.704465 | 0.703213 | 0.4860 | 0.0000 |
| 10 | 0.704272 | 0.703125 | 0.4895 | 0.0000 |
| 11 | 0.704047 | 0.702933 | 0.4895 | 0.0000 |
| 12 | 0.703824 | 0.702719 | 0.4895 | 0.0000 |
| 13 | 0.703599 | 0.702519 | 0.4895 | 0.0000 |
| 14 | 0.703124 | 0.702179 | 0.4912 | 0.0000 |
| 15 | 0.702642 | 0.701833 | 0.4912 | 0.0000 |
| 16 | 0.702387 | 0.701628 | 0.4912 | 0.0000 |
| 17 | 0.702171 | 0.701419 | 0.4877 | 0.0000 |
| 18 | 0.702031 | 0.701193 | 0.4877 | 0.0000 |
| 19 | 0.701925 | 0.701044 | 0.4877 | 0.0000 |
| 20 | 0.701660 | 0.700948 | 0.4860 | 0.0000 |

#### id-map-303-s37

Runtime 3.74 s; weights L2 change 0.514685; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/id-map-303-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.724622 | 0.709722 | 0.5316 | 0.0000 |
| 1 | 0.724531 | 0.709775 | 0.5351 | 0.0000 |
| 2 | 0.724380 | 0.709849 | 0.5316 | 0.0000 |
| 3 | 0.724280 | 0.709846 | 0.5298 | 0.0000 |
| 4 | 0.724197 | 0.709853 | 0.5263 | 0.0000 |
| 5 | 0.724096 | 0.709836 | 0.5298 | 0.0000 |
| 6 | 0.724023 | 0.709825 | 0.5298 | 0.0000 |
| 7 | 0.723933 | 0.709787 | 0.5316 | 0.0000 |
| 8 | 0.723852 | 0.709720 | 0.5298 | 0.0000 |
| 9 | 0.723755 | 0.709657 | 0.5281 | 0.0000 |
| 10 | 0.723576 | 0.709568 | 0.5298 | 0.0000 |
| 11 | 0.723431 | 0.709533 | 0.5298 | 0.0000 |
| 12 | 0.723300 | 0.709494 | 0.5316 | 0.0000 |
| 13 | 0.723202 | 0.709484 | 0.5316 | 0.0000 |
| 14 | 0.723073 | 0.709475 | 0.5316 | 0.0000 |
| 15 | 0.722893 | 0.709352 | 0.5333 | 0.0000 |
| 16 | 0.722694 | 0.709207 | 0.5368 | 0.0000 |
| 17 | 0.722533 | 0.709105 | 0.5351 | 0.0000 |
| 18 | 0.722365 | 0.708994 | 0.5351 | 0.0000 |
| 19 | 0.722263 | 0.709004 | 0.5351 | 0.0000 |
| 20 | 0.722185 | 0.708968 | 0.5368 | 0.0000 |

#### joint-id-map-s11

Runtime 3.80 s; weights L2 change 0.398193; Qiskit maximum marginal error 9.99e-16. [Saved run](artifacts/pilot/joint-id-map-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.722152 | 0.726259 | 0.5088 | 0.0000 |
| 1 | 0.721535 | 0.725783 | 0.5088 | 0.0000 |
| 2 | 0.721146 | 0.725482 | 0.5088 | 0.0000 |
| 3 | 0.720829 | 0.725230 | 0.5105 | 0.0000 |
| 4 | 0.720582 | 0.724909 | 0.5105 | 0.0000 |
| 5 | 0.720336 | 0.724602 | 0.5105 | 0.0000 |
| 6 | 0.720087 | 0.724363 | 0.5088 | 0.0000 |
| 7 | 0.719824 | 0.723988 | 0.5070 | 0.0000 |
| 8 | 0.719442 | 0.723470 | 0.5070 | 0.0000 |
| 9 | 0.719152 | 0.723125 | 0.5070 | 0.0000 |
| 10 | 0.718979 | 0.722897 | 0.5070 | 0.0000 |
| 11 | 0.718748 | 0.722549 | 0.5070 | 0.0000 |
| 12 | 0.718505 | 0.722271 | 0.5053 | 0.0000 |
| 13 | 0.718294 | 0.721987 | 0.5053 | 0.0000 |
| 14 | 0.718079 | 0.721741 | 0.5035 | 0.0000 |
| 15 | 0.717784 | 0.721388 | 0.5053 | 0.0000 |
| 16 | 0.717448 | 0.720936 | 0.5053 | 0.0000 |
| 17 | 0.717178 | 0.720691 | 0.5053 | 0.0000 |
| 18 | 0.716895 | 0.720450 | 0.5070 | 0.0000 |
| 19 | 0.716640 | 0.720347 | 0.5053 | 0.0000 |
| 20 | 0.716379 | 0.720139 | 0.5035 | 0.0000 |

#### joint-id-map-s23

Runtime 3.85 s; weights L2 change 0.354123; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/joint-id-map-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.713986 | 0.722217 | 0.4807 | 0.0000 |
| 1 | 0.713647 | 0.721925 | 0.4825 | 0.0000 |
| 2 | 0.713408 | 0.721708 | 0.4842 | 0.0000 |
| 3 | 0.713226 | 0.721524 | 0.4860 | 0.0000 |
| 4 | 0.712894 | 0.721240 | 0.4842 | 0.0000 |
| 5 | 0.712590 | 0.720931 | 0.4825 | 0.0000 |
| 6 | 0.712397 | 0.720662 | 0.4825 | 0.0000 |
| 7 | 0.712235 | 0.720379 | 0.4772 | 0.0000 |
| 8 | 0.712020 | 0.720026 | 0.4772 | 0.0000 |
| 9 | 0.711837 | 0.719732 | 0.4772 | 0.0000 |
| 10 | 0.711697 | 0.719479 | 0.4789 | 0.0000 |
| 11 | 0.711457 | 0.719181 | 0.4789 | 0.0000 |
| 12 | 0.711316 | 0.719050 | 0.4789 | 0.0000 |
| 13 | 0.711230 | 0.718908 | 0.4807 | 0.0000 |
| 14 | 0.711016 | 0.718604 | 0.4807 | 0.0000 |
| 15 | 0.710740 | 0.718218 | 0.4807 | 0.0000 |
| 16 | 0.710527 | 0.717948 | 0.4807 | 0.0000 |
| 17 | 0.710320 | 0.717650 | 0.4807 | 0.0000 |
| 18 | 0.710158 | 0.717406 | 0.4825 | 0.0000 |
| 19 | 0.710088 | 0.717304 | 0.4825 | 0.0000 |
| 20 | 0.710057 | 0.717212 | 0.4807 | 0.0000 |

#### joint-id-map-s37

Runtime 3.77 s; weights L2 change 0.567952; Qiskit maximum marginal error 6.66e-16. [Saved run](artifacts/pilot/joint-id-map-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.706572 | 0.713597 | 0.5667 | 0.0000 |
| 1 | 0.706384 | 0.713654 | 0.5772 | 0.0000 |
| 2 | 0.706179 | 0.713689 | 0.5772 | 0.0000 |
| 3 | 0.706008 | 0.713617 | 0.5789 | 0.0000 |
| 4 | 0.705813 | 0.713584 | 0.5789 | 0.0000 |
| 5 | 0.705580 | 0.713533 | 0.5772 | 0.0000 |
| 6 | 0.705346 | 0.713610 | 0.5789 | 0.0000 |
| 7 | 0.705129 | 0.713641 | 0.5772 | 0.0000 |
| 8 | 0.704949 | 0.713661 | 0.5719 | 0.0000 |
| 9 | 0.704798 | 0.713650 | 0.5719 | 0.0000 |
| 10 | 0.704634 | 0.713665 | 0.5684 | 0.0000 |
| 11 | 0.704488 | 0.713712 | 0.5684 | 0.0000 |
| 12 | 0.704392 | 0.713824 | 0.5632 | 0.0000 |
| 13 | 0.704331 | 0.713930 | 0.5561 | 0.0000 |
| 14 | 0.704263 | 0.713990 | 0.5561 | 0.0000 |
| 15 | 0.704112 | 0.713963 | 0.5561 | 0.0000 |
| 16 | 0.703932 | 0.713936 | 0.5561 | 0.0000 |
| 17 | 0.703740 | 0.713934 | 0.5526 | 0.0000 |
| 18 | 0.703600 | 0.714012 | 0.5526 | 0.0000 |
| 19 | 0.703429 | 0.714038 | 0.5526 | 0.0000 |
| 20 | 0.703197 | 0.714012 | 0.5526 | 0.0000 |

#### fixed-pairs-s11

Runtime 3.76 s; weights L2 change 0.430605; Qiskit maximum marginal error 4.44e-16. [Saved run](artifacts/pilot/fixed-pairs-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.723247 | 0.713427 | 0.5263 | 0.0000 |
| 1 | 0.722951 | 0.712979 | 0.5246 | 0.0000 |
| 2 | 0.722867 | 0.712635 | 0.5246 | 0.0000 |
| 3 | 0.722794 | 0.712362 | 0.5246 | 0.0000 |
| 4 | 0.722713 | 0.712377 | 0.5246 | 0.0000 |
| 5 | 0.722494 | 0.712134 | 0.5263 | 0.0000 |
| 6 | 0.722272 | 0.711845 | 0.5281 | 0.0000 |
| 7 | 0.722083 | 0.711560 | 0.5263 | 0.0000 |
| 8 | 0.721915 | 0.711235 | 0.5263 | 0.0000 |
| 9 | 0.721816 | 0.711016 | 0.5263 | 0.0000 |
| 10 | 0.721520 | 0.710509 | 0.5246 | 0.0000 |
| 11 | 0.720983 | 0.709878 | 0.5246 | 0.0000 |
| 12 | 0.720612 | 0.709402 | 0.5281 | 0.0000 |
| 13 | 0.720339 | 0.709059 | 0.5298 | 0.0000 |
| 14 | 0.719984 | 0.708791 | 0.5298 | 0.0000 |
| 15 | 0.719538 | 0.708529 | 0.5316 | 0.0000 |
| 16 | 0.719166 | 0.708200 | 0.5316 | 0.0000 |
| 17 | 0.718906 | 0.708017 | 0.5298 | 0.0000 |
| 18 | 0.718727 | 0.707903 | 0.5298 | 0.0000 |
| 19 | 0.718555 | 0.707608 | 0.5298 | 0.0000 |
| 20 | 0.718384 | 0.707370 | 0.5298 | 0.0000 |

#### fixed-pairs-s23

Runtime 3.78 s; weights L2 change 0.411771; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/fixed-pairs-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.725676 | 0.730889 | 0.4825 | 0.0000 |
| 1 | 0.725580 | 0.730507 | 0.4825 | 0.0000 |
| 2 | 0.725239 | 0.730065 | 0.4825 | 0.0000 |
| 3 | 0.724920 | 0.729701 | 0.4860 | 0.0000 |
| 4 | 0.724600 | 0.729334 | 0.4877 | 0.0000 |
| 5 | 0.724306 | 0.728929 | 0.4895 | 0.0000 |
| 6 | 0.724070 | 0.728539 | 0.4895 | 0.0000 |
| 7 | 0.723845 | 0.728193 | 0.4895 | 0.0000 |
| 8 | 0.723611 | 0.727888 | 0.4895 | 0.0000 |
| 9 | 0.723411 | 0.727532 | 0.4912 | 0.0000 |
| 10 | 0.723198 | 0.727120 | 0.4895 | 0.0000 |
| 11 | 0.723061 | 0.726919 | 0.4877 | 0.0000 |
| 12 | 0.722974 | 0.726753 | 0.4877 | 0.0000 |
| 13 | 0.722797 | 0.726575 | 0.4895 | 0.0000 |
| 14 | 0.722482 | 0.726312 | 0.4895 | 0.0000 |
| 15 | 0.722184 | 0.726051 | 0.4895 | 0.0000 |
| 16 | 0.721988 | 0.725760 | 0.4877 | 0.0000 |
| 17 | 0.721824 | 0.725492 | 0.4860 | 0.0000 |
| 18 | 0.721708 | 0.725303 | 0.4860 | 0.0000 |
| 19 | 0.721526 | 0.725081 | 0.4860 | 0.0000 |
| 20 | 0.721374 | 0.724920 | 0.4860 | 0.0000 |

#### fixed-pairs-s37

Runtime 3.78 s; weights L2 change 0.527381; Qiskit maximum marginal error 7.77e-16. [Saved run](artifacts/pilot/fixed-pairs-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.706812 | 0.716842 | 0.4842 | 0.0000 |
| 1 | 0.707071 | 0.716609 | 0.4877 | 0.0000 |
| 2 | 0.707080 | 0.716654 | 0.4877 | 0.0000 |
| 3 | 0.707035 | 0.716746 | 0.4842 | 0.0000 |
| 4 | 0.707035 | 0.716772 | 0.4860 | 0.0000 |
| 5 | 0.707030 | 0.716750 | 0.4877 | 0.0000 |
| 6 | 0.707046 | 0.716701 | 0.4895 | 0.0000 |
| 7 | 0.707016 | 0.716661 | 0.4895 | 0.0000 |
| 8 | 0.706835 | 0.716691 | 0.4895 | 0.0000 |
| 9 | 0.706624 | 0.716755 | 0.4930 | 0.0000 |
| 10 | 0.706535 | 0.716746 | 0.4912 | 0.0000 |
| 11 | 0.706468 | 0.716727 | 0.4912 | 0.0000 |
| 12 | 0.706356 | 0.716718 | 0.4877 | 0.0000 |
| 13 | 0.706263 | 0.716725 | 0.4877 | 0.0000 |
| 14 | 0.706141 | 0.716784 | 0.4877 | 0.0000 |
| 15 | 0.705986 | 0.716839 | 0.4860 | 0.0000 |
| 16 | 0.705855 | 0.716890 | 0.4842 | 0.0000 |
| 17 | 0.705748 | 0.716913 | 0.4842 | 0.0000 |
| 18 | 0.705665 | 0.716966 | 0.4825 | 0.0000 |
| 19 | 0.705454 | 0.717089 | 0.4877 | 0.0000 |
| 20 | 0.705251 | 0.717209 | 0.4825 | 0.0000 |

#### layer-pairs-s11

Runtime 3.80 s; weights L2 change 0.398058; Qiskit maximum marginal error 3.33e-16. [Saved run](artifacts/pilot/layer-pairs-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.721745 | 0.710999 | 0.5439 | 0.0000 |
| 1 | 0.721365 | 0.710752 | 0.5439 | 0.0000 |
| 2 | 0.721351 | 0.710652 | 0.5421 | 0.0000 |
| 3 | 0.721239 | 0.710496 | 0.5421 | 0.0000 |
| 4 | 0.721068 | 0.710402 | 0.5421 | 0.0000 |
| 5 | 0.720889 | 0.710288 | 0.5404 | 0.0000 |
| 6 | 0.720576 | 0.710048 | 0.5404 | 0.0000 |
| 7 | 0.720341 | 0.709822 | 0.5404 | 0.0000 |
| 8 | 0.720158 | 0.709592 | 0.5404 | 0.0000 |
| 9 | 0.720044 | 0.709392 | 0.5404 | 0.0000 |
| 10 | 0.719845 | 0.709088 | 0.5368 | 0.0000 |
| 11 | 0.719325 | 0.708575 | 0.5368 | 0.0000 |
| 12 | 0.718890 | 0.708095 | 0.5386 | 0.0000 |
| 13 | 0.718517 | 0.707763 | 0.5368 | 0.0000 |
| 14 | 0.718077 | 0.707529 | 0.5368 | 0.0000 |
| 15 | 0.717606 | 0.707303 | 0.5368 | 0.0000 |
| 16 | 0.717276 | 0.707084 | 0.5351 | 0.0000 |
| 17 | 0.716975 | 0.706990 | 0.5351 | 0.0000 |
| 18 | 0.716705 | 0.706897 | 0.5368 | 0.0000 |
| 19 | 0.716408 | 0.706651 | 0.5368 | 0.0000 |
| 20 | 0.716177 | 0.706410 | 0.5368 | 0.0000 |

#### layer-pairs-s23

Runtime 3.86 s; weights L2 change 0.419187; Qiskit maximum marginal error 7.77e-16. [Saved run](artifacts/pilot/layer-pairs-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.723708 | 0.730865 | 0.4684 | 0.0000 |
| 1 | 0.723566 | 0.730839 | 0.4684 | 0.0000 |
| 2 | 0.723317 | 0.730588 | 0.4649 | 0.0000 |
| 3 | 0.723057 | 0.730375 | 0.4684 | 0.0000 |
| 4 | 0.722687 | 0.729966 | 0.4684 | 0.0000 |
| 5 | 0.722294 | 0.729589 | 0.4702 | 0.0000 |
| 6 | 0.721976 | 0.729189 | 0.4719 | 0.0000 |
| 7 | 0.721730 | 0.728915 | 0.4719 | 0.0000 |
| 8 | 0.721379 | 0.728543 | 0.4702 | 0.0000 |
| 9 | 0.721169 | 0.728248 | 0.4737 | 0.0000 |
| 10 | 0.720937 | 0.727920 | 0.4737 | 0.0000 |
| 11 | 0.720737 | 0.727724 | 0.4737 | 0.0000 |
| 12 | 0.720623 | 0.727562 | 0.4737 | 0.0000 |
| 13 | 0.720417 | 0.727321 | 0.4754 | 0.0000 |
| 14 | 0.720085 | 0.727000 | 0.4737 | 0.0000 |
| 15 | 0.719782 | 0.726777 | 0.4737 | 0.0000 |
| 16 | 0.719531 | 0.726479 | 0.4772 | 0.0000 |
| 17 | 0.719288 | 0.726172 | 0.4737 | 0.0000 |
| 18 | 0.719118 | 0.725974 | 0.4719 | 0.0000 |
| 19 | 0.718928 | 0.725776 | 0.4719 | 0.0000 |
| 20 | 0.718774 | 0.725718 | 0.4702 | 0.0000 |

#### layer-pairs-s37

Runtime 3.79 s; weights L2 change 0.537008; Qiskit maximum marginal error 4.44e-16. [Saved run](artifacts/pilot/layer-pairs-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.706211 | 0.717460 | 0.5053 | 0.0000 |
| 1 | 0.706067 | 0.717468 | 0.5053 | 0.0000 |
| 2 | 0.706040 | 0.717515 | 0.5035 | 0.0000 |
| 3 | 0.705994 | 0.717596 | 0.5018 | 0.0000 |
| 4 | 0.705984 | 0.717663 | 0.4982 | 0.0000 |
| 5 | 0.705983 | 0.717706 | 0.4965 | 0.0000 |
| 6 | 0.706020 | 0.717675 | 0.4930 | 0.0000 |
| 7 | 0.706036 | 0.717588 | 0.4947 | 0.0000 |
| 8 | 0.705987 | 0.717500 | 0.4965 | 0.0000 |
| 9 | 0.705903 | 0.717447 | 0.5000 | 0.0000 |
| 10 | 0.705886 | 0.717408 | 0.5018 | 0.0000 |
| 11 | 0.705886 | 0.717379 | 0.5035 | 0.0000 |
| 12 | 0.705799 | 0.717344 | 0.5035 | 0.0000 |
| 13 | 0.705758 | 0.717317 | 0.4965 | 0.0000 |
| 14 | 0.705748 | 0.717265 | 0.4947 | 0.0000 |
| 15 | 0.705660 | 0.717168 | 0.4895 | 0.0000 |
| 16 | 0.705594 | 0.717095 | 0.4877 | 0.0000 |
| 17 | 0.705567 | 0.717025 | 0.4912 | 0.0000 |
| 18 | 0.705547 | 0.716992 | 0.4860 | 0.0000 |
| 19 | 0.705411 | 0.716980 | 0.4860 | 0.0000 |
| 20 | 0.705263 | 0.716901 | 0.4842 | 0.0000 |

#### reverse-uploads-s11

Runtime 3.73 s; weights L2 change 0.402914; Qiskit maximum marginal error 9.99e-16. [Saved run](artifacts/pilot/reverse-uploads-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.727148 | 0.719095 | 0.5351 | 0.0000 |
| 1 | 0.726948 | 0.718981 | 0.5333 | 0.0000 |
| 2 | 0.727036 | 0.719058 | 0.5351 | 0.0000 |
| 3 | 0.727006 | 0.718942 | 0.5368 | 0.0000 |
| 4 | 0.726819 | 0.718876 | 0.5386 | 0.0000 |
| 5 | 0.726643 | 0.718834 | 0.5386 | 0.0000 |
| 6 | 0.726553 | 0.718800 | 0.5368 | 0.0000 |
| 7 | 0.726501 | 0.718760 | 0.5368 | 0.0000 |
| 8 | 0.726371 | 0.718652 | 0.5368 | 0.0000 |
| 9 | 0.726253 | 0.718721 | 0.5368 | 0.0000 |
| 10 | 0.725899 | 0.718620 | 0.5368 | 0.0000 |
| 11 | 0.725432 | 0.718188 | 0.5351 | 0.0000 |
| 12 | 0.724984 | 0.717977 | 0.5351 | 0.0000 |
| 13 | 0.724620 | 0.717971 | 0.5368 | 0.0000 |
| 14 | 0.724408 | 0.717942 | 0.5368 | 0.0000 |
| 15 | 0.724213 | 0.717883 | 0.5368 | 0.0000 |
| 16 | 0.723868 | 0.717569 | 0.5351 | 0.0000 |
| 17 | 0.723626 | 0.717209 | 0.5351 | 0.0000 |
| 18 | 0.723209 | 0.716881 | 0.5368 | 0.0000 |
| 19 | 0.722843 | 0.716679 | 0.5368 | 0.0000 |
| 20 | 0.722576 | 0.716480 | 0.5368 | 0.0000 |

#### reverse-uploads-s23

Runtime 3.76 s; weights L2 change 0.390994; Qiskit maximum marginal error 4.44e-16. [Saved run](artifacts/pilot/reverse-uploads-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.714100 | 0.721556 | 0.5088 | 0.0000 |
| 1 | 0.713976 | 0.721336 | 0.5123 | 0.0000 |
| 2 | 0.713842 | 0.721166 | 0.5140 | 0.0000 |
| 3 | 0.713724 | 0.721009 | 0.5140 | 0.0000 |
| 4 | 0.713545 | 0.720619 | 0.5158 | 0.0000 |
| 5 | 0.713390 | 0.720367 | 0.5193 | 0.0000 |
| 6 | 0.713000 | 0.720008 | 0.5193 | 0.0000 |
| 7 | 0.712677 | 0.719715 | 0.5175 | 0.0000 |
| 8 | 0.712420 | 0.719387 | 0.5175 | 0.0000 |
| 9 | 0.712185 | 0.719185 | 0.5158 | 0.0000 |
| 10 | 0.711995 | 0.718935 | 0.5140 | 0.0000 |
| 11 | 0.711730 | 0.718499 | 0.5140 | 0.0000 |
| 12 | 0.711538 | 0.718094 | 0.5158 | 0.0000 |
| 13 | 0.711360 | 0.717821 | 0.5175 | 0.0000 |
| 14 | 0.710925 | 0.717321 | 0.5158 | 0.0000 |
| 15 | 0.710452 | 0.716736 | 0.5158 | 0.0000 |
| 16 | 0.710175 | 0.716300 | 0.5193 | 0.0000 |
| 17 | 0.710009 | 0.716001 | 0.5211 | 0.0000 |
| 18 | 0.709906 | 0.715821 | 0.5211 | 0.0000 |
| 19 | 0.709843 | 0.715624 | 0.5211 | 0.0000 |
| 20 | 0.709619 | 0.715147 | 0.5246 | 0.0000 |

#### reverse-uploads-s37

Runtime 3.75 s; weights L2 change 0.493479; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/reverse-uploads-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.719695 | 0.708278 | 0.5211 | 0.0000 |
| 1 | 0.719682 | 0.708194 | 0.5211 | 0.0000 |
| 2 | 0.719483 | 0.708109 | 0.5246 | 0.0000 |
| 3 | 0.719283 | 0.708013 | 0.5228 | 0.0000 |
| 4 | 0.719134 | 0.707969 | 0.5211 | 0.0000 |
| 5 | 0.718957 | 0.707875 | 0.5211 | 0.0000 |
| 6 | 0.718724 | 0.707753 | 0.5193 | 0.0000 |
| 7 | 0.718532 | 0.707640 | 0.5211 | 0.0000 |
| 8 | 0.718416 | 0.707597 | 0.5246 | 0.0000 |
| 9 | 0.718336 | 0.707560 | 0.5263 | 0.0000 |
| 10 | 0.718271 | 0.707529 | 0.5281 | 0.0000 |
| 11 | 0.718177 | 0.707489 | 0.5281 | 0.0000 |
| 12 | 0.718013 | 0.707417 | 0.5281 | 0.0000 |
| 13 | 0.717790 | 0.707321 | 0.5298 | 0.0000 |
| 14 | 0.717636 | 0.707269 | 0.5281 | 0.0000 |
| 15 | 0.717441 | 0.707142 | 0.5281 | 0.0000 |
| 16 | 0.717328 | 0.707071 | 0.5298 | 0.0000 |
| 17 | 0.717260 | 0.707044 | 0.5246 | 0.0000 |
| 18 | 0.717209 | 0.707041 | 0.5246 | 0.0000 |
| 19 | 0.717124 | 0.706991 | 0.5228 | 0.0000 |
| 20 | 0.716997 | 0.706818 | 0.5211 | 0.0000 |

#### layer-id-maps-s11

Runtime 3.84 s; weights L2 change 0.449325; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/layer-id-maps-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.731952 | 0.705828 | 0.5263 | 0.0000 |
| 1 | 0.731861 | 0.705738 | 0.5263 | 0.0000 |
| 2 | 0.731763 | 0.705696 | 0.5263 | 0.0000 |
| 3 | 0.731472 | 0.705733 | 0.5228 | 0.0000 |
| 4 | 0.731135 | 0.705763 | 0.5246 | 0.0000 |
| 5 | 0.730788 | 0.705629 | 0.5263 | 0.0000 |
| 6 | 0.730405 | 0.705383 | 0.5281 | 0.0000 |
| 7 | 0.730146 | 0.705199 | 0.5281 | 0.0000 |
| 8 | 0.729843 | 0.705044 | 0.5263 | 0.0000 |
| 9 | 0.729667 | 0.704928 | 0.5263 | 0.0000 |
| 10 | 0.729518 | 0.704798 | 0.5263 | 0.0000 |
| 11 | 0.729087 | 0.704598 | 0.5281 | 0.0000 |
| 12 | 0.728660 | 0.704414 | 0.5316 | 0.0000 |
| 13 | 0.728383 | 0.704334 | 0.5316 | 0.0000 |
| 14 | 0.728074 | 0.704288 | 0.5316 | 0.0000 |
| 15 | 0.727582 | 0.704092 | 0.5281 | 0.0000 |
| 16 | 0.727198 | 0.703991 | 0.5281 | 0.0000 |
| 17 | 0.726816 | 0.703925 | 0.5263 | 0.0000 |
| 18 | 0.726572 | 0.703932 | 0.5263 | 0.0000 |
| 19 | 0.726272 | 0.703946 | 0.5246 | 0.0000 |
| 20 | 0.725880 | 0.703799 | 0.5246 | 0.0000 |

#### layer-id-maps-s23

Runtime 3.78 s; weights L2 change 0.391408; Qiskit maximum marginal error 1.11e-15. [Saved run](artifacts/pilot/layer-id-maps-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.704739 | 0.717577 | 0.5333 | 0.0000 |
| 1 | 0.704519 | 0.717505 | 0.5351 | 0.0000 |
| 2 | 0.704260 | 0.717324 | 0.5368 | 0.0000 |
| 3 | 0.704105 | 0.717154 | 0.5368 | 0.0000 |
| 4 | 0.704030 | 0.717100 | 0.5386 | 0.0000 |
| 5 | 0.703855 | 0.716961 | 0.5368 | 0.0000 |
| 6 | 0.703678 | 0.716811 | 0.5386 | 0.0000 |
| 7 | 0.703596 | 0.716788 | 0.5333 | 0.0000 |
| 8 | 0.703527 | 0.716797 | 0.5351 | 0.0000 |
| 9 | 0.703446 | 0.716792 | 0.5333 | 0.0000 |
| 10 | 0.703408 | 0.716878 | 0.5333 | 0.0000 |
| 11 | 0.703359 | 0.716892 | 0.5333 | 0.0000 |
| 12 | 0.703302 | 0.716785 | 0.5351 | 0.0000 |
| 13 | 0.703202 | 0.716648 | 0.5351 | 0.0000 |
| 14 | 0.702954 | 0.716274 | 0.5351 | 0.0000 |
| 15 | 0.702752 | 0.715960 | 0.5386 | 0.0000 |
| 16 | 0.702551 | 0.715755 | 0.5404 | 0.0000 |
| 17 | 0.702403 | 0.715681 | 0.5421 | 0.0000 |
| 18 | 0.702325 | 0.715685 | 0.5439 | 0.0000 |
| 19 | 0.702269 | 0.715700 | 0.5474 | 0.0000 |
| 20 | 0.702166 | 0.715621 | 0.5474 | 0.0000 |

#### layer-id-maps-s37

Runtime 3.84 s; weights L2 change 0.534908; Qiskit maximum marginal error 4.44e-16. [Saved run](artifacts/pilot/layer-id-maps-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.709338 | 0.720604 | 0.5281 | 0.0000 |
| 1 | 0.709356 | 0.720599 | 0.5298 | 0.0000 |
| 2 | 0.709210 | 0.720861 | 0.5298 | 0.0000 |
| 3 | 0.709105 | 0.721036 | 0.5298 | 0.0000 |
| 4 | 0.709006 | 0.721095 | 0.5298 | 0.0000 |
| 5 | 0.708943 | 0.721148 | 0.5281 | 0.0000 |
| 6 | 0.708877 | 0.721221 | 0.5211 | 0.0000 |
| 7 | 0.708802 | 0.721289 | 0.5158 | 0.0000 |
| 8 | 0.708674 | 0.721239 | 0.5158 | 0.0000 |
| 9 | 0.708590 | 0.721281 | 0.5193 | 0.0000 |
| 10 | 0.708543 | 0.721284 | 0.5193 | 0.0000 |
| 11 | 0.708469 | 0.721373 | 0.5193 | 0.0000 |
| 12 | 0.708398 | 0.721425 | 0.5175 | 0.0000 |
| 13 | 0.708332 | 0.721439 | 0.5193 | 0.0000 |
| 14 | 0.708254 | 0.721483 | 0.5158 | 0.0000 |
| 15 | 0.708168 | 0.721511 | 0.5211 | 0.0000 |
| 16 | 0.708049 | 0.721649 | 0.5211 | 0.0000 |
| 17 | 0.707933 | 0.721798 | 0.5211 | 0.0000 |
| 18 | 0.707816 | 0.721791 | 0.5175 | 0.0000 |
| 19 | 0.707730 | 0.721741 | 0.5175 | 0.0000 |
| 20 | 0.707615 | 0.721765 | 0.5175 | 0.0000 |

#### rz-before-rx-s11

Runtime 3.73 s; weights L2 change 0.392351; Qiskit maximum marginal error 9.99e-16. [Saved run](artifacts/pilot/rz-before-rx-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.741560 | 0.771402 | 0.4719 | 0.0000 |
| 1 | 0.739867 | 0.770283 | 0.4754 | 0.0000 |
| 2 | 0.738864 | 0.769741 | 0.4754 | 0.0000 |
| 3 | 0.738138 | 0.769403 | 0.4719 | 0.0000 |
| 4 | 0.737497 | 0.768903 | 0.4719 | 0.0000 |
| 5 | 0.737020 | 0.768525 | 0.4737 | 0.0000 |
| 6 | 0.736593 | 0.768144 | 0.4737 | 0.0000 |
| 7 | 0.736159 | 0.767797 | 0.4719 | 0.0000 |
| 8 | 0.735574 | 0.767283 | 0.4719 | 0.0000 |
| 9 | 0.734756 | 0.766672 | 0.4754 | 0.0000 |
| 10 | 0.734055 | 0.766191 | 0.4737 | 0.0000 |
| 11 | 0.733310 | 0.765656 | 0.4737 | 0.0000 |
| 12 | 0.732641 | 0.765237 | 0.4754 | 0.0000 |
| 13 | 0.732148 | 0.764820 | 0.4737 | 0.0000 |
| 14 | 0.731352 | 0.764270 | 0.4684 | 0.0000 |
| 15 | 0.730386 | 0.763647 | 0.4632 | 0.0000 |
| 16 | 0.729661 | 0.763101 | 0.4684 | 0.0000 |
| 17 | 0.729054 | 0.762611 | 0.4667 | 0.0000 |
| 18 | 0.728644 | 0.762218 | 0.4667 | 0.0000 |
| 19 | 0.728244 | 0.761758 | 0.4667 | 0.0000 |
| 20 | 0.727916 | 0.761370 | 0.4702 | 0.0000 |

#### rz-before-rx-s23

Runtime 3.76 s; weights L2 change 0.403228; Qiskit maximum marginal error 8.88e-16. [Saved run](artifacts/pilot/rz-before-rx-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.740406 | 0.736890 | 0.5175 | 0.0000 |
| 1 | 0.738213 | 0.735091 | 0.5158 | 0.0000 |
| 2 | 0.736213 | 0.733759 | 0.5158 | 0.0000 |
| 3 | 0.734784 | 0.732713 | 0.5175 | 0.0000 |
| 4 | 0.733440 | 0.731708 | 0.5175 | 0.0000 |
| 5 | 0.732247 | 0.730770 | 0.5175 | 0.0000 |
| 6 | 0.731222 | 0.729941 | 0.5175 | 0.0000 |
| 7 | 0.729862 | 0.728863 | 0.5193 | 0.0000 |
| 8 | 0.728238 | 0.727494 | 0.5193 | 0.0000 |
| 9 | 0.726674 | 0.726071 | 0.5193 | 0.0000 |
| 10 | 0.725459 | 0.725067 | 0.5175 | 0.0000 |
| 11 | 0.724271 | 0.724190 | 0.5158 | 0.0000 |
| 12 | 0.723075 | 0.723313 | 0.5158 | 0.0000 |
| 13 | 0.721875 | 0.722324 | 0.5158 | 0.0000 |
| 14 | 0.720881 | 0.721540 | 0.5158 | 0.0000 |
| 15 | 0.719960 | 0.720887 | 0.5158 | 0.0000 |
| 16 | 0.718851 | 0.719962 | 0.5175 | 0.0000 |
| 17 | 0.717853 | 0.719080 | 0.5158 | 0.0000 |
| 18 | 0.716899 | 0.718226 | 0.5158 | 0.0000 |
| 19 | 0.716041 | 0.717479 | 0.5175 | 0.0000 |
| 20 | 0.715285 | 0.716775 | 0.5175 | 0.0000 |

#### rz-before-rx-s37

Runtime 3.79 s; weights L2 change 0.370245; Qiskit maximum marginal error 6.66e-16. [Saved run](artifacts/pilot/rz-before-rx-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.749435 | 0.728888 | 0.5175 | 0.0000 |
| 1 | 0.747812 | 0.727675 | 0.5158 | 0.0000 |
| 2 | 0.746374 | 0.726499 | 0.5140 | 0.0000 |
| 3 | 0.745286 | 0.725488 | 0.5175 | 0.0000 |
| 4 | 0.744290 | 0.724594 | 0.5193 | 0.0000 |
| 5 | 0.743383 | 0.723724 | 0.5211 | 0.0000 |
| 6 | 0.742552 | 0.723059 | 0.5211 | 0.0000 |
| 7 | 0.741946 | 0.722528 | 0.5211 | 0.0000 |
| 8 | 0.741296 | 0.721911 | 0.5228 | 0.0000 |
| 9 | 0.740581 | 0.721231 | 0.5228 | 0.0000 |
| 10 | 0.739710 | 0.720320 | 0.5246 | 0.0000 |
| 11 | 0.738957 | 0.719587 | 0.5246 | 0.0000 |
| 12 | 0.738348 | 0.719023 | 0.5228 | 0.0000 |
| 13 | 0.737956 | 0.718629 | 0.5246 | 0.0000 |
| 14 | 0.737647 | 0.718386 | 0.5228 | 0.0000 |
| 15 | 0.737062 | 0.717877 | 0.5228 | 0.0000 |
| 16 | 0.736070 | 0.716969 | 0.5246 | 0.0000 |
| 17 | 0.735399 | 0.716480 | 0.5228 | 0.0000 |
| 18 | 0.734609 | 0.715858 | 0.5263 | 0.0000 |
| 19 | 0.733718 | 0.715075 | 0.5263 | 0.0000 |
| 20 | 0.732857 | 0.714315 | 0.5246 | 0.0000 |

#### reverse-cnot-s11

Runtime 3.74 s; weights L2 change 0.605350; Qiskit maximum marginal error 1.17e-15. [Saved run](artifacts/pilot/reverse-cnot-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.695199 | 0.699657 | 0.5018 | 0.0000 |
| 1 | 0.695057 | 0.699607 | 0.5035 | 0.0000 |
| 2 | 0.695009 | 0.699651 | 0.5000 | 0.0000 |
| 3 | 0.694932 | 0.699651 | 0.5018 | 0.0000 |
| 4 | 0.694719 | 0.699665 | 0.5018 | 0.0000 |
| 5 | 0.694574 | 0.699686 | 0.5018 | 0.0000 |
| 6 | 0.694474 | 0.699688 | 0.5000 | 0.0000 |
| 7 | 0.694354 | 0.699709 | 0.5000 | 0.0000 |
| 8 | 0.694215 | 0.699737 | 0.5000 | 0.0000 |
| 9 | 0.694105 | 0.699781 | 0.5000 | 0.0000 |
| 10 | 0.694020 | 0.699855 | 0.5000 | 0.0000 |
| 11 | 0.693860 | 0.699861 | 0.5000 | 0.0000 |
| 12 | 0.693688 | 0.699879 | 0.5018 | 0.0000 |
| 13 | 0.693549 | 0.699905 | 0.5018 | 0.0000 |
| 14 | 0.693465 | 0.699931 | 0.5018 | 0.0000 |
| 15 | 0.693396 | 0.699997 | 0.5018 | 0.0000 |
| 16 | 0.693281 | 0.700064 | 0.5018 | 0.0000 |
| 17 | 0.693180 | 0.700129 | 0.5018 | 0.0000 |
| 18 | 0.693070 | 0.700174 | 0.5018 | 0.0000 |
| 19 | 0.692926 | 0.700223 | 0.5000 | 0.0000 |
| 20 | 0.692847 | 0.700307 | 0.5000 | 0.0000 |

#### reverse-cnot-s23

Runtime 3.72 s; weights L2 change 0.482517; Qiskit maximum marginal error 3.89e-16. [Saved run](artifacts/pilot/reverse-cnot-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.696173 | 0.702836 | 0.5088 | 0.0000 |
| 1 | 0.696397 | 0.703040 | 0.5088 | 0.0000 |
| 2 | 0.696227 | 0.702784 | 0.5105 | 0.0000 |
| 3 | 0.695965 | 0.702414 | 0.5123 | 0.0000 |
| 4 | 0.695640 | 0.701953 | 0.5105 | 0.0000 |
| 5 | 0.695417 | 0.701614 | 0.5105 | 0.0000 |
| 6 | 0.695108 | 0.701195 | 0.5088 | 0.0000 |
| 7 | 0.694830 | 0.700857 | 0.5070 | 0.0000 |
| 8 | 0.694580 | 0.700568 | 0.5070 | 0.0000 |
| 9 | 0.694417 | 0.700378 | 0.5053 | 0.0000 |
| 10 | 0.694205 | 0.700123 | 0.5053 | 0.0000 |
| 11 | 0.693998 | 0.699859 | 0.5053 | 0.0000 |
| 12 | 0.693798 | 0.699612 | 0.5053 | 0.0000 |
| 13 | 0.693602 | 0.699332 | 0.5053 | 0.0000 |
| 14 | 0.693360 | 0.698985 | 0.5105 | 0.0000 |
| 15 | 0.693114 | 0.698664 | 0.5105 | 0.0000 |
| 16 | 0.692939 | 0.698463 | 0.5105 | 0.0000 |
| 17 | 0.692811 | 0.698280 | 0.5105 | 0.0000 |
| 18 | 0.692687 | 0.698089 | 0.5105 | 0.0000 |
| 19 | 0.692595 | 0.697963 | 0.5105 | 0.0000 |
| 20 | 0.692466 | 0.697860 | 0.5105 | 0.0000 |

#### reverse-cnot-s37

Runtime 3.75 s; weights L2 change 1.143290; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/reverse-cnot-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.693633 | 0.694559 | 0.4965 | 0.0000 |
| 1 | 0.693595 | 0.694520 | 0.4930 | 0.0000 |
| 2 | 0.693553 | 0.694476 | 0.4912 | 0.0000 |
| 3 | 0.693510 | 0.694439 | 0.4947 | 0.0000 |
| 4 | 0.693458 | 0.694416 | 0.4930 | 0.0000 |
| 5 | 0.693409 | 0.694387 | 0.4930 | 0.0000 |
| 6 | 0.693361 | 0.694352 | 0.4930 | 0.0000 |
| 7 | 0.693318 | 0.694317 | 0.4930 | 0.0000 |
| 8 | 0.693264 | 0.694288 | 0.4947 | 0.0000 |
| 9 | 0.693192 | 0.694240 | 0.4947 | 0.0000 |
| 10 | 0.693135 | 0.694197 | 0.4947 | 0.0000 |
| 11 | 0.693071 | 0.694158 | 0.4912 | 0.0000 |
| 12 | 0.693000 | 0.694106 | 0.4930 | 0.0000 |
| 13 | 0.692922 | 0.694030 | 0.4912 | 0.0000 |
| 14 | 0.692848 | 0.693973 | 0.4912 | 0.0000 |
| 15 | 0.692780 | 0.693923 | 0.4947 | 0.0000 |
| 16 | 0.692707 | 0.693867 | 0.4947 | 0.0000 |
| 17 | 0.692660 | 0.693828 | 0.4965 | 0.0000 |
| 18 | 0.692602 | 0.693795 | 0.4982 | 0.0000 |
| 19 | 0.692535 | 0.693755 | 0.4982 | 0.0000 |
| 20 | 0.692465 | 0.693723 | 0.4982 | 0.0000 |

#### shuffled-labels-s11

Runtime 3.77 s; weights L2 change 0.376787; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/shuffled-labels-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.727205 | 0.736462 | 0.5228 | 0.0000 |
| 1 | 0.726914 | 0.736001 | 0.5228 | 0.0000 |
| 2 | 0.726688 | 0.735791 | 0.5211 | 0.0000 |
| 3 | 0.726519 | 0.735610 | 0.5211 | 0.0000 |
| 4 | 0.726226 | 0.735520 | 0.5193 | 0.0000 |
| 5 | 0.725991 | 0.735189 | 0.5158 | 0.0000 |
| 6 | 0.725759 | 0.735001 | 0.5158 | 0.0000 |
| 7 | 0.725514 | 0.734828 | 0.5158 | 0.0000 |
| 8 | 0.725318 | 0.734683 | 0.5158 | 0.0000 |
| 9 | 0.725254 | 0.734665 | 0.5175 | 0.0000 |
| 10 | 0.725068 | 0.734614 | 0.5175 | 0.0000 |
| 11 | 0.724639 | 0.734252 | 0.5175 | 0.0000 |
| 12 | 0.724299 | 0.733940 | 0.5175 | 0.0000 |
| 13 | 0.724012 | 0.733766 | 0.5175 | 0.0000 |
| 14 | 0.723743 | 0.733578 | 0.5140 | 0.0000 |
| 15 | 0.723361 | 0.733177 | 0.5158 | 0.0000 |
| 16 | 0.722925 | 0.732882 | 0.5123 | 0.0000 |
| 17 | 0.722547 | 0.732666 | 0.5105 | 0.0000 |
| 18 | 0.722247 | 0.732516 | 0.5123 | 0.0000 |
| 19 | 0.721997 | 0.732355 | 0.5105 | 0.0000 |
| 20 | 0.721687 | 0.732050 | 0.5123 | 0.0000 |

#### shuffled-labels-s23

Runtime 3.75 s; weights L2 change 0.427721; Qiskit maximum marginal error 7.77e-16. [Saved run](artifacts/pilot/shuffled-labels-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.728535 | 0.722413 | 0.5175 | 0.0000 |
| 1 | 0.727912 | 0.721873 | 0.5175 | 0.0000 |
| 2 | 0.727332 | 0.721425 | 0.5193 | 0.0000 |
| 3 | 0.726707 | 0.720987 | 0.5211 | 0.0000 |
| 4 | 0.725909 | 0.720465 | 0.5211 | 0.0000 |
| 5 | 0.725149 | 0.719963 | 0.5228 | 0.0000 |
| 6 | 0.724178 | 0.719395 | 0.5211 | 0.0000 |
| 7 | 0.723420 | 0.718930 | 0.5211 | 0.0000 |
| 8 | 0.722805 | 0.718495 | 0.5211 | 0.0000 |
| 9 | 0.722345 | 0.718183 | 0.5211 | 0.0000 |
| 10 | 0.721905 | 0.717873 | 0.5246 | 0.0000 |
| 11 | 0.721428 | 0.717498 | 0.5246 | 0.0000 |
| 12 | 0.720974 | 0.717178 | 0.5211 | 0.0000 |
| 13 | 0.720526 | 0.716975 | 0.5211 | 0.0000 |
| 14 | 0.719832 | 0.716522 | 0.5228 | 0.0000 |
| 15 | 0.719185 | 0.716075 | 0.5228 | 0.0000 |
| 16 | 0.718624 | 0.715781 | 0.5246 | 0.0000 |
| 17 | 0.718186 | 0.715549 | 0.5246 | 0.0000 |
| 18 | 0.717815 | 0.715315 | 0.5246 | 0.0000 |
| 19 | 0.717447 | 0.715093 | 0.5246 | 0.0000 |
| 20 | 0.717042 | 0.714872 | 0.5246 | 0.0000 |

#### shuffled-labels-s37

Runtime 3.73 s; weights L2 change 0.553843; Qiskit maximum marginal error 6.66e-16. [Saved run](artifacts/pilot/shuffled-labels-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.714443 | 0.717881 | 0.5105 | 0.0000 |
| 1 | 0.714112 | 0.717397 | 0.5105 | 0.0000 |
| 2 | 0.713938 | 0.717273 | 0.5105 | 0.0000 |
| 3 | 0.713763 | 0.717153 | 0.5105 | 0.0000 |
| 4 | 0.713461 | 0.716869 | 0.5105 | 0.0000 |
| 5 | 0.713235 | 0.716732 | 0.5088 | 0.0000 |
| 6 | 0.713014 | 0.716605 | 0.5105 | 0.0000 |
| 7 | 0.712758 | 0.716405 | 0.5070 | 0.0000 |
| 8 | 0.712512 | 0.716167 | 0.5105 | 0.0000 |
| 9 | 0.712289 | 0.715964 | 0.5123 | 0.0000 |
| 10 | 0.712157 | 0.715851 | 0.5123 | 0.0000 |
| 11 | 0.712090 | 0.715750 | 0.5123 | 0.0000 |
| 12 | 0.711975 | 0.715655 | 0.5140 | 0.0000 |
| 13 | 0.711830 | 0.715542 | 0.5158 | 0.0000 |
| 14 | 0.711647 | 0.715348 | 0.5158 | 0.0000 |
| 15 | 0.711393 | 0.715106 | 0.5140 | 0.0000 |
| 16 | 0.711136 | 0.714879 | 0.5140 | 0.0000 |
| 17 | 0.710830 | 0.714640 | 0.5140 | 0.0000 |
| 18 | 0.710590 | 0.714455 | 0.5140 | 0.0000 |
| 19 | 0.710375 | 0.714282 | 0.5123 | 0.0000 |
| 20 | 0.710124 | 0.714097 | 0.5123 | 0.0000 |

#### semantic-codes-s11

Runtime 3.75 s; weights L2 change 0.455792; Qiskit maximum marginal error 1.11e-15. [Saved run](artifacts/pilot/semantic-codes-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.745942 | 0.717870 | 0.5211 | 0.0000 |
| 1 | 0.745851 | 0.718129 | 0.5175 | 0.0000 |
| 2 | 0.745603 | 0.717834 | 0.5140 | 0.0000 |
| 3 | 0.745252 | 0.717562 | 0.5158 | 0.0000 |
| 4 | 0.744841 | 0.717336 | 0.5175 | 0.0000 |
| 5 | 0.744398 | 0.717022 | 0.5175 | 0.0000 |
| 6 | 0.744040 | 0.716750 | 0.5175 | 0.0000 |
| 7 | 0.743769 | 0.716663 | 0.5175 | 0.0000 |
| 8 | 0.743479 | 0.716641 | 0.5193 | 0.0000 |
| 9 | 0.743196 | 0.716426 | 0.5175 | 0.0000 |
| 10 | 0.742767 | 0.716238 | 0.5211 | 0.0000 |
| 11 | 0.742335 | 0.716038 | 0.5228 | 0.0000 |
| 12 | 0.742000 | 0.715896 | 0.5211 | 0.0000 |
| 13 | 0.741642 | 0.715650 | 0.5175 | 0.0000 |
| 14 | 0.741308 | 0.715495 | 0.5158 | 0.0000 |
| 15 | 0.740854 | 0.715171 | 0.5175 | 0.0000 |
| 16 | 0.740386 | 0.714793 | 0.5175 | 0.0000 |
| 17 | 0.739929 | 0.714312 | 0.5175 | 0.0000 |
| 18 | 0.739651 | 0.714067 | 0.5123 | 0.0000 |
| 19 | 0.739407 | 0.713955 | 0.5140 | 0.0000 |
| 20 | 0.739039 | 0.713869 | 0.5123 | 0.0000 |

#### semantic-codes-s23

Runtime 3.75 s; weights L2 change 0.392901; Qiskit maximum marginal error 3.33e-16. [Saved run](artifacts/pilot/semantic-codes-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.725678 | 0.749582 | 0.4737 | 0.0000 |
| 1 | 0.725003 | 0.748624 | 0.4719 | 0.0000 |
| 2 | 0.724377 | 0.747630 | 0.4702 | 0.0000 |
| 3 | 0.723952 | 0.746855 | 0.4702 | 0.0000 |
| 4 | 0.723559 | 0.746174 | 0.4719 | 0.0000 |
| 5 | 0.723072 | 0.745263 | 0.4719 | 0.0000 |
| 6 | 0.722767 | 0.744681 | 0.4719 | 0.0000 |
| 7 | 0.722535 | 0.744174 | 0.4719 | 0.0000 |
| 8 | 0.722311 | 0.743707 | 0.4702 | 0.0000 |
| 9 | 0.721888 | 0.742904 | 0.4719 | 0.0000 |
| 10 | 0.721497 | 0.742165 | 0.4719 | 0.0000 |
| 11 | 0.721175 | 0.741622 | 0.4737 | 0.0000 |
| 12 | 0.720954 | 0.741214 | 0.4754 | 0.0000 |
| 13 | 0.720715 | 0.740798 | 0.4754 | 0.0000 |
| 14 | 0.720344 | 0.740182 | 0.4754 | 0.0000 |
| 15 | 0.720059 | 0.739731 | 0.4737 | 0.0000 |
| 16 | 0.719790 | 0.739273 | 0.4737 | 0.0000 |
| 17 | 0.719571 | 0.738876 | 0.4719 | 0.0000 |
| 18 | 0.719370 | 0.738525 | 0.4702 | 0.0000 |
| 19 | 0.719105 | 0.738120 | 0.4719 | 0.0000 |
| 20 | 0.718916 | 0.737874 | 0.4719 | 0.0000 |

#### semantic-codes-s37

Runtime 3.76 s; weights L2 change 0.523824; Qiskit maximum marginal error 6.11e-16. [Saved run](artifacts/pilot/semantic-codes-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.706459 | 0.717997 | 0.4877 | 0.0000 |
| 1 | 0.706446 | 0.718001 | 0.4842 | 0.0000 |
| 2 | 0.706369 | 0.718116 | 0.4825 | 0.0000 |
| 3 | 0.706257 | 0.718159 | 0.4754 | 0.0000 |
| 4 | 0.706211 | 0.718160 | 0.4772 | 0.0000 |
| 5 | 0.706202 | 0.718154 | 0.4789 | 0.0000 |
| 6 | 0.706185 | 0.718131 | 0.4789 | 0.0000 |
| 7 | 0.706162 | 0.718149 | 0.4807 | 0.0000 |
| 8 | 0.706151 | 0.718171 | 0.4789 | 0.0000 |
| 9 | 0.706164 | 0.718165 | 0.4789 | 0.0000 |
| 10 | 0.706173 | 0.718153 | 0.4789 | 0.0000 |
| 11 | 0.706190 | 0.718157 | 0.4789 | 0.0000 |
| 12 | 0.706192 | 0.718193 | 0.4789 | 0.0000 |
| 13 | 0.706189 | 0.718214 | 0.4754 | 0.0000 |
| 14 | 0.706211 | 0.718273 | 0.4807 | 0.0000 |
| 15 | 0.706220 | 0.718340 | 0.4789 | 0.0000 |
| 16 | 0.706222 | 0.718408 | 0.4789 | 0.0000 |
| 17 | 0.706238 | 0.718465 | 0.4789 | 0.0000 |
| 18 | 0.706228 | 0.718541 | 0.4807 | 0.0000 |
| 19 | 0.706147 | 0.718600 | 0.4825 | 0.0000 |
| 20 | 0.706098 | 0.718668 | 0.4807 | 0.0000 |

#### random-codes-s11

Runtime 3.73 s; weights L2 change 0.392883; Qiskit maximum marginal error 6.66e-16. [Saved run](artifacts/pilot/random-codes-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.739654 | 0.755678 | 0.4912 | 0.0000 |
| 1 | 0.739509 | 0.755749 | 0.4860 | 0.0000 |
| 2 | 0.739395 | 0.755771 | 0.4842 | 0.0000 |
| 3 | 0.739203 | 0.755476 | 0.4860 | 0.0000 |
| 4 | 0.739031 | 0.755128 | 0.4860 | 0.0000 |
| 5 | 0.738759 | 0.754590 | 0.4860 | 0.0000 |
| 6 | 0.738364 | 0.754029 | 0.4895 | 0.0000 |
| 7 | 0.738112 | 0.753660 | 0.4895 | 0.0000 |
| 8 | 0.737866 | 0.753357 | 0.4895 | 0.0000 |
| 9 | 0.737669 | 0.753038 | 0.4895 | 0.0000 |
| 10 | 0.737385 | 0.752648 | 0.4877 | 0.0000 |
| 11 | 0.736931 | 0.752080 | 0.4895 | 0.0000 |
| 12 | 0.736579 | 0.751687 | 0.4877 | 0.0000 |
| 13 | 0.736317 | 0.751385 | 0.4895 | 0.0000 |
| 14 | 0.736156 | 0.751135 | 0.4895 | 0.0000 |
| 15 | 0.735920 | 0.750809 | 0.4912 | 0.0000 |
| 16 | 0.735599 | 0.750471 | 0.4930 | 0.0000 |
| 17 | 0.735283 | 0.749988 | 0.4930 | 0.0000 |
| 18 | 0.735064 | 0.749810 | 0.4912 | 0.0000 |
| 19 | 0.734821 | 0.749527 | 0.4930 | 0.0000 |
| 20 | 0.734479 | 0.749119 | 0.4930 | 0.0000 |

#### random-codes-s23

Runtime 3.79 s; weights L2 change 0.403280; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/random-codes-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.712858 | 0.729817 | 0.5035 | 0.0000 |
| 1 | 0.712456 | 0.729649 | 0.5035 | 0.0000 |
| 2 | 0.711911 | 0.729056 | 0.5035 | 0.0000 |
| 3 | 0.711460 | 0.728660 | 0.5035 | 0.0000 |
| 4 | 0.711047 | 0.728249 | 0.5070 | 0.0000 |
| 5 | 0.710443 | 0.727660 | 0.5088 | 0.0000 |
| 6 | 0.709883 | 0.727290 | 0.5088 | 0.0000 |
| 7 | 0.709443 | 0.727003 | 0.5070 | 0.0000 |
| 8 | 0.709081 | 0.726805 | 0.5070 | 0.0000 |
| 9 | 0.708471 | 0.726339 | 0.5070 | 0.0000 |
| 10 | 0.708016 | 0.725927 | 0.5070 | 0.0000 |
| 11 | 0.707731 | 0.725733 | 0.5070 | 0.0000 |
| 12 | 0.707480 | 0.725572 | 0.5070 | 0.0000 |
| 13 | 0.707219 | 0.725381 | 0.5070 | 0.0000 |
| 14 | 0.706906 | 0.725047 | 0.5053 | 0.0000 |
| 15 | 0.706594 | 0.724747 | 0.5053 | 0.0000 |
| 16 | 0.706266 | 0.724524 | 0.5053 | 0.0000 |
| 17 | 0.705931 | 0.724328 | 0.5035 | 0.0000 |
| 18 | 0.705657 | 0.724196 | 0.5035 | 0.0000 |
| 19 | 0.705352 | 0.724024 | 0.5035 | 0.0000 |
| 20 | 0.705110 | 0.723888 | 0.5035 | 0.0000 |

#### random-codes-s37

Runtime 3.76 s; weights L2 change 0.485891; Qiskit maximum marginal error 6.66e-16. [Saved run](artifacts/pilot/random-codes-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.721519 | 0.716972 | 0.5035 | 0.0000 |
| 1 | 0.721268 | 0.717031 | 0.5018 | 0.0000 |
| 2 | 0.721106 | 0.716947 | 0.5035 | 0.0000 |
| 3 | 0.720876 | 0.716589 | 0.5035 | 0.0000 |
| 4 | 0.720597 | 0.716281 | 0.5035 | 0.0000 |
| 5 | 0.720375 | 0.716029 | 0.5053 | 0.0000 |
| 6 | 0.720118 | 0.715637 | 0.5070 | 0.0000 |
| 7 | 0.719906 | 0.715345 | 0.5070 | 0.0000 |
| 8 | 0.719655 | 0.715047 | 0.5088 | 0.0000 |
| 9 | 0.719395 | 0.714798 | 0.5105 | 0.0000 |
| 10 | 0.719224 | 0.714623 | 0.5105 | 0.0000 |
| 11 | 0.719098 | 0.714514 | 0.5105 | 0.0000 |
| 12 | 0.718924 | 0.714376 | 0.5105 | 0.0000 |
| 13 | 0.718811 | 0.714249 | 0.5088 | 0.0000 |
| 14 | 0.718757 | 0.714223 | 0.5123 | 0.0000 |
| 15 | 0.718678 | 0.714171 | 0.5105 | 0.0000 |
| 16 | 0.718601 | 0.714145 | 0.5105 | 0.0000 |
| 17 | 0.718418 | 0.714042 | 0.5123 | 0.0000 |
| 18 | 0.718203 | 0.713929 | 0.5105 | 0.0000 |
| 19 | 0.717901 | 0.713592 | 0.5088 | 0.0000 |
| 20 | 0.717718 | 0.713385 | 0.5105 | 0.0000 |

#### qwen-layout-s11

Runtime 3.77 s; weights L2 change 0.425750; Qiskit maximum marginal error 4.44e-16. [Saved run](artifacts/pilot/qwen-layout-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.724882 | 0.731258 | 0.4930 | 0.0000 |
| 1 | 0.724669 | 0.730804 | 0.4912 | 0.0000 |
| 2 | 0.724531 | 0.730543 | 0.4895 | 0.0000 |
| 3 | 0.724412 | 0.730380 | 0.4912 | 0.0000 |
| 4 | 0.724325 | 0.730325 | 0.4912 | 0.0000 |
| 5 | 0.724189 | 0.730428 | 0.4877 | 0.0000 |
| 6 | 0.724035 | 0.730476 | 0.4877 | 0.0000 |
| 7 | 0.723890 | 0.730514 | 0.4877 | 0.0000 |
| 8 | 0.723663 | 0.730523 | 0.4877 | 0.0000 |
| 9 | 0.723396 | 0.730376 | 0.4877 | 0.0000 |
| 10 | 0.723078 | 0.730112 | 0.4895 | 0.0000 |
| 11 | 0.722732 | 0.729735 | 0.4895 | 0.0000 |
| 12 | 0.722495 | 0.729544 | 0.4860 | 0.0000 |
| 13 | 0.722263 | 0.729353 | 0.4860 | 0.0000 |
| 14 | 0.722054 | 0.729186 | 0.4860 | 0.0000 |
| 15 | 0.721704 | 0.728966 | 0.4860 | 0.0000 |
| 16 | 0.721448 | 0.728785 | 0.4860 | 0.0000 |
| 17 | 0.721262 | 0.728696 | 0.4842 | 0.0000 |
| 18 | 0.721057 | 0.728595 | 0.4825 | 0.0000 |
| 19 | 0.720880 | 0.728507 | 0.4789 | 0.0000 |
| 20 | 0.720720 | 0.728306 | 0.4789 | 0.0000 |

#### qwen-layout-s23

Runtime 3.75 s; weights L2 change 0.406788; Qiskit maximum marginal error 6.66e-16. [Saved run](artifacts/pilot/qwen-layout-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.719605 | 0.719637 | 0.4667 | 0.0000 |
| 1 | 0.719312 | 0.719197 | 0.4649 | 0.0000 |
| 2 | 0.719130 | 0.718905 | 0.4632 | 0.0000 |
| 3 | 0.718926 | 0.718578 | 0.4614 | 0.0000 |
| 4 | 0.718628 | 0.718168 | 0.4614 | 0.0000 |
| 5 | 0.718321 | 0.717804 | 0.4596 | 0.0000 |
| 6 | 0.717858 | 0.717328 | 0.4614 | 0.0000 |
| 7 | 0.717485 | 0.716931 | 0.4614 | 0.0000 |
| 8 | 0.717226 | 0.716657 | 0.4632 | 0.0000 |
| 9 | 0.716955 | 0.716284 | 0.4632 | 0.0000 |
| 10 | 0.716698 | 0.715903 | 0.4632 | 0.0000 |
| 11 | 0.716364 | 0.715431 | 0.4614 | 0.0000 |
| 12 | 0.716155 | 0.715158 | 0.4632 | 0.0000 |
| 13 | 0.715965 | 0.714899 | 0.4632 | 0.0000 |
| 14 | 0.715625 | 0.714437 | 0.4649 | 0.0000 |
| 15 | 0.715242 | 0.714015 | 0.4649 | 0.0000 |
| 16 | 0.714974 | 0.713750 | 0.4649 | 0.0000 |
| 17 | 0.714733 | 0.713495 | 0.4877 | 0.0000 |
| 18 | 0.714605 | 0.713335 | 0.4877 | 0.0000 |
| 19 | 0.714537 | 0.713213 | 0.4877 | 0.0000 |
| 20 | 0.714450 | 0.713162 | 0.4877 | 0.0000 |

#### qwen-layout-s37

Runtime 3.77 s; weights L2 change 0.561595; Qiskit maximum marginal error 7.77e-16. [Saved run](artifacts/pilot/qwen-layout-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.706134 | 0.709473 | 0.5140 | 0.0000 |
| 1 | 0.706013 | 0.709674 | 0.5140 | 0.0000 |
| 2 | 0.705969 | 0.709714 | 0.5158 | 0.0000 |
| 3 | 0.705916 | 0.709764 | 0.5175 | 0.0000 |
| 4 | 0.705863 | 0.709801 | 0.5175 | 0.0000 |
| 5 | 0.705830 | 0.709807 | 0.5175 | 0.0000 |
| 6 | 0.705760 | 0.709710 | 0.5193 | 0.0000 |
| 7 | 0.705675 | 0.709555 | 0.5193 | 0.0000 |
| 8 | 0.705588 | 0.709496 | 0.5193 | 0.0000 |
| 9 | 0.705493 | 0.709562 | 0.5175 | 0.0000 |
| 10 | 0.705425 | 0.709612 | 0.5158 | 0.0000 |
| 11 | 0.705356 | 0.709685 | 0.5175 | 0.0000 |
| 12 | 0.705305 | 0.709785 | 0.5193 | 0.0000 |
| 13 | 0.705248 | 0.709858 | 0.5193 | 0.0000 |
| 14 | 0.705209 | 0.709916 | 0.5211 | 0.0000 |
| 15 | 0.705185 | 0.709881 | 0.5211 | 0.0000 |
| 16 | 0.705152 | 0.709885 | 0.5211 | 0.0000 |
| 17 | 0.705097 | 0.709875 | 0.5228 | 0.0000 |
| 18 | 0.705043 | 0.709957 | 0.5211 | 0.0000 |
| 19 | 0.704993 | 0.709994 | 0.5211 | 0.0000 |
| 20 | 0.704949 | 0.710098 | 0.5211 | 0.0000 |

#### shuffled-qwen-layout-s11

Runtime 3.81 s; weights L2 change 0.472191; Qiskit maximum marginal error 6.66e-16. [Saved run](artifacts/pilot/shuffled-qwen-layout-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.729231 | 0.747075 | 0.4877 | 0.0000 |
| 1 | 0.729041 | 0.746626 | 0.4860 | 0.0000 |
| 2 | 0.728898 | 0.746388 | 0.4860 | 0.0000 |
| 3 | 0.728687 | 0.746065 | 0.4877 | 0.0000 |
| 4 | 0.728485 | 0.745683 | 0.4877 | 0.0000 |
| 5 | 0.728071 | 0.745106 | 0.4860 | 0.0000 |
| 6 | 0.727651 | 0.744554 | 0.4860 | 0.0000 |
| 7 | 0.727328 | 0.744172 | 0.4860 | 0.0000 |
| 8 | 0.727028 | 0.743786 | 0.4860 | 0.0000 |
| 9 | 0.726807 | 0.743439 | 0.4860 | 0.0000 |
| 10 | 0.726481 | 0.742910 | 0.4860 | 0.0000 |
| 11 | 0.725887 | 0.742233 | 0.4860 | 0.0000 |
| 12 | 0.725428 | 0.741650 | 0.4860 | 0.0000 |
| 13 | 0.724907 | 0.741101 | 0.4877 | 0.0000 |
| 14 | 0.724437 | 0.740710 | 0.4877 | 0.0000 |
| 15 | 0.723893 | 0.740305 | 0.4877 | 0.0000 |
| 16 | 0.723362 | 0.740027 | 0.4877 | 0.0000 |
| 17 | 0.722931 | 0.739753 | 0.4877 | 0.0000 |
| 18 | 0.722528 | 0.739532 | 0.4860 | 0.0000 |
| 19 | 0.722200 | 0.739448 | 0.4860 | 0.0000 |
| 20 | 0.721909 | 0.739263 | 0.4860 | 0.0000 |

#### shuffled-qwen-layout-s23

Runtime 3.75 s; weights L2 change 0.366047; Qiskit maximum marginal error 6.66e-16. [Saved run](artifacts/pilot/shuffled-qwen-layout-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.734060 | 0.709320 | 0.4965 | 0.0000 |
| 1 | 0.733929 | 0.709079 | 0.4982 | 0.0000 |
| 2 | 0.733658 | 0.708811 | 0.4965 | 0.0000 |
| 3 | 0.733367 | 0.708533 | 0.5018 | 0.0000 |
| 4 | 0.733056 | 0.708192 | 0.5000 | 0.0000 |
| 5 | 0.732695 | 0.707775 | 0.5000 | 0.0000 |
| 6 | 0.732299 | 0.707317 | 0.5000 | 0.0000 |
| 7 | 0.731893 | 0.706929 | 0.4965 | 0.0000 |
| 8 | 0.731578 | 0.706499 | 0.4965 | 0.0000 |
| 9 | 0.731277 | 0.706139 | 0.5000 | 0.0000 |
| 10 | 0.730994 | 0.705776 | 0.5070 | 0.0000 |
| 11 | 0.730635 | 0.705315 | 0.5105 | 0.0000 |
| 12 | 0.730419 | 0.705042 | 0.5088 | 0.0000 |
| 13 | 0.730230 | 0.704843 | 0.5105 | 0.0000 |
| 14 | 0.729854 | 0.704394 | 0.5088 | 0.0000 |
| 15 | 0.729441 | 0.703882 | 0.5088 | 0.0000 |
| 16 | 0.729102 | 0.703500 | 0.5070 | 0.0000 |
| 17 | 0.728755 | 0.703137 | 0.5088 | 0.0000 |
| 18 | 0.728550 | 0.702919 | 0.5070 | 0.0000 |
| 19 | 0.728449 | 0.702804 | 0.5053 | 0.0000 |
| 20 | 0.728364 | 0.702709 | 0.5053 | 0.0000 |

#### shuffled-qwen-layout-s37

Runtime 3.75 s; weights L2 change 0.461996; Qiskit maximum marginal error 6.66e-16. [Saved run](artifacts/pilot/shuffled-qwen-layout-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.710789 | 0.714500 | 0.5158 | 0.0000 |
| 1 | 0.710555 | 0.714384 | 0.5140 | 0.0000 |
| 2 | 0.710415 | 0.714388 | 0.5140 | 0.0000 |
| 3 | 0.710260 | 0.714308 | 0.5175 | 0.0000 |
| 4 | 0.710080 | 0.714222 | 0.5193 | 0.0000 |
| 5 | 0.709949 | 0.714195 | 0.5193 | 0.0000 |
| 6 | 0.709802 | 0.714148 | 0.5211 | 0.0000 |
| 7 | 0.709608 | 0.714052 | 0.5211 | 0.0000 |
| 8 | 0.709444 | 0.713967 | 0.5211 | 0.0000 |
| 9 | 0.709316 | 0.713940 | 0.5228 | 0.0000 |
| 10 | 0.709194 | 0.713924 | 0.5211 | 0.0000 |
| 11 | 0.709113 | 0.713919 | 0.5211 | 0.0000 |
| 12 | 0.709066 | 0.713968 | 0.5193 | 0.0000 |
| 13 | 0.708965 | 0.714013 | 0.5211 | 0.0000 |
| 14 | 0.708842 | 0.714083 | 0.5228 | 0.0000 |
| 15 | 0.708724 | 0.714079 | 0.5228 | 0.0000 |
| 16 | 0.708604 | 0.714007 | 0.5175 | 0.0000 |
| 17 | 0.708490 | 0.713948 | 0.5211 | 0.0000 |
| 18 | 0.708410 | 0.713890 | 0.5211 | 0.0000 |
| 19 | 0.708302 | 0.713814 | 0.5211 | 0.0000 |
| 20 | 0.708175 | 0.713753 | 0.5140 | 0.0000 |

#### qwen-layout-no-entanglement-s11

Runtime 3.42 s; weights L2 change 0.389348; Qiskit maximum marginal error 7.77e-16. [Saved run](artifacts/pilot/qwen-layout-no-entanglement-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 1.046372 | 1.062472 | 0.5035 | 0.0000 |
| 1 | 1.042264 | 1.060947 | 0.5000 | 0.0000 |
| 2 | 1.038788 | 1.059594 | 0.5000 | 0.0000 |
| 3 | 1.036469 | 1.058703 | 0.5000 | 0.0000 |
| 4 | 1.034817 | 1.057862 | 0.5000 | 0.0000 |
| 5 | 1.032971 | 1.056822 | 0.5000 | 0.0000 |
| 6 | 1.031664 | 1.056066 | 0.5000 | 0.0000 |
| 7 | 1.029496 | 1.054719 | 0.5000 | 0.0000 |
| 8 | 1.026910 | 1.053780 | 0.5018 | 0.0000 |
| 9 | 1.024596 | 1.052547 | 0.5018 | 0.0000 |
| 10 | 1.022044 | 1.051198 | 0.5018 | 0.0000 |
| 11 | 1.018030 | 1.050040 | 0.5018 | 0.0000 |
| 12 | 1.014411 | 1.048601 | 0.5018 | 0.0000 |
| 13 | 1.011133 | 1.047005 | 0.5000 | 0.0000 |
| 14 | 1.009008 | 1.045791 | 0.4965 | 0.0000 |
| 15 | 1.006574 | 1.045162 | 0.4982 | 0.0000 |
| 16 | 1.004205 | 1.044286 | 0.4982 | 0.0000 |
| 17 | 1.000597 | 1.041959 | 0.5000 | 0.0000 |
| 18 | 0.997932 | 1.040722 | 0.4982 | 0.0000 |
| 19 | 0.995564 | 1.039682 | 0.4965 | 0.0000 |
| 20 | 0.993108 | 1.038481 | 0.4947 | 0.0000 |

#### qwen-layout-no-entanglement-s23

Runtime 3.50 s; weights L2 change 0.399840; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/qwen-layout-no-entanglement-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 1.093041 | 1.091595 | 0.5035 | 0.0000 |
| 1 | 1.089425 | 1.083827 | 0.5053 | 0.0000 |
| 2 | 1.084761 | 1.079106 | 0.5035 | 0.0000 |
| 3 | 1.080667 | 1.074629 | 0.5035 | 0.0000 |
| 4 | 1.077651 | 1.071263 | 0.5035 | 0.0000 |
| 5 | 1.074583 | 1.068092 | 0.5018 | 0.0000 |
| 6 | 1.069783 | 1.064400 | 0.5000 | 0.0000 |
| 7 | 1.065488 | 1.060969 | 0.5000 | 0.0000 |
| 8 | 1.062070 | 1.058175 | 0.5000 | 0.0000 |
| 9 | 1.059820 | 1.056582 | 0.5000 | 0.0000 |
| 10 | 1.058241 | 1.055386 | 0.5000 | 0.0000 |
| 11 | 1.056428 | 1.054020 | 0.5000 | 0.0000 |
| 12 | 1.054680 | 1.052440 | 0.4982 | 0.0000 |
| 13 | 1.052929 | 1.050579 | 0.4982 | 0.0000 |
| 14 | 1.051296 | 1.048516 | 0.4982 | 0.0000 |
| 15 | 1.049826 | 1.046535 | 0.4982 | 0.0000 |
| 16 | 1.047197 | 1.043950 | 0.4965 | 0.0000 |
| 17 | 1.042722 | 1.039795 | 0.4965 | 0.0000 |
| 18 | 1.038918 | 1.036047 | 0.4965 | 0.0000 |
| 19 | 1.035223 | 1.032914 | 0.4965 | 0.0000 |
| 20 | 1.032529 | 1.030462 | 0.4982 | 0.0000 |

#### qwen-layout-no-entanglement-s37

Runtime 3.40 s; weights L2 change 0.369684; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/qwen-layout-no-entanglement-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 1.040631 | 0.985426 | 0.4982 | 0.0000 |
| 1 | 1.038553 | 0.984283 | 0.5000 | 0.0000 |
| 2 | 1.037176 | 0.983050 | 0.4965 | 0.0000 |
| 3 | 1.035411 | 0.980047 | 0.4965 | 0.0000 |
| 4 | 1.032616 | 0.977216 | 0.4947 | 0.0000 |
| 5 | 1.030137 | 0.975057 | 0.4930 | 0.0000 |
| 6 | 1.028306 | 0.973423 | 0.4947 | 0.0000 |
| 7 | 1.026764 | 0.971970 | 0.4947 | 0.0000 |
| 8 | 1.025430 | 0.970662 | 0.4947 | 0.0000 |
| 9 | 1.023953 | 0.969010 | 0.4947 | 0.0000 |
| 10 | 1.021047 | 0.966120 | 0.4930 | 0.0000 |
| 11 | 1.017429 | 0.963225 | 0.4912 | 0.0000 |
| 12 | 1.014926 | 0.961334 | 0.4947 | 0.0000 |
| 13 | 1.012903 | 0.959444 | 0.4965 | 0.0000 |
| 14 | 1.010860 | 0.957498 | 0.4947 | 0.0000 |
| 15 | 1.009162 | 0.956365 | 0.4930 | 0.0000 |
| 16 | 1.007783 | 0.955509 | 0.4912 | 0.0000 |
| 17 | 1.006590 | 0.954759 | 0.4930 | 0.0000 |
| 18 | 1.004668 | 0.953554 | 0.4947 | 0.0000 |
| 19 | 1.002964 | 0.952367 | 0.4912 | 0.0000 |
| 20 | 1.001095 | 0.950505 | 0.4912 | 0.0000 |

#### shuffled-layout-no-entanglement-s11

Runtime 3.53 s; weights L2 change 0.366624; Qiskit maximum marginal error 5.55e-16. [Saved run](artifacts/pilot/shuffled-layout-no-entanglement-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 1.030270 | 0.984022 | 0.5316 | 0.0000 |
| 1 | 1.026357 | 0.980782 | 0.5316 | 0.0000 |
| 2 | 1.022725 | 0.978105 | 0.5316 | 0.0000 |
| 3 | 1.019807 | 0.976540 | 0.5316 | 0.0000 |
| 4 | 1.017657 | 0.975420 | 0.5316 | 0.0000 |
| 5 | 1.015631 | 0.973816 | 0.5351 | 0.0000 |
| 6 | 1.014037 | 0.973365 | 0.5333 | 0.0000 |
| 7 | 1.012345 | 0.972352 | 0.5333 | 0.0000 |
| 8 | 1.010597 | 0.971232 | 0.5333 | 0.0000 |
| 9 | 1.009077 | 0.969456 | 0.5316 | 0.0000 |
| 10 | 1.007951 | 0.967835 | 0.5316 | 0.0000 |
| 11 | 1.006879 | 0.967094 | 0.5298 | 0.0000 |
| 12 | 1.005762 | 0.966431 | 0.5298 | 0.0000 |
| 13 | 1.004174 | 0.964962 | 0.5298 | 0.0000 |
| 14 | 1.002776 | 0.963452 | 0.5316 | 0.0000 |
| 15 | 1.001574 | 0.962236 | 0.5298 | 0.0000 |
| 16 | 1.000264 | 0.960381 | 0.5281 | 0.0000 |
| 17 | 0.998090 | 0.956671 | 0.5281 | 0.0000 |
| 18 | 0.996347 | 0.953656 | 0.5263 | 0.0000 |
| 19 | 0.994513 | 0.951482 | 0.5263 | 0.0000 |
| 20 | 0.992370 | 0.949068 | 0.5263 | 0.0000 |

#### shuffled-layout-no-entanglement-s23

Runtime 3.53 s; weights L2 change 0.390693; Qiskit maximum marginal error 8.88e-16. [Saved run](artifacts/pilot/shuffled-layout-no-entanglement-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 1.135981 | 1.011927 | 0.5000 | 0.0000 |
| 1 | 1.134233 | 1.007779 | 0.4982 | 0.0000 |
| 2 | 1.131604 | 1.004653 | 0.4982 | 0.0000 |
| 3 | 1.128679 | 1.002163 | 0.4965 | 0.0000 |
| 4 | 1.125723 | 1.000045 | 0.4965 | 0.0000 |
| 5 | 1.122561 | 0.998216 | 0.4982 | 0.0000 |
| 6 | 1.117570 | 0.995716 | 0.4982 | 0.0000 |
| 7 | 1.113016 | 0.993851 | 0.4982 | 0.0000 |
| 8 | 1.108135 | 0.992014 | 0.4982 | 0.0000 |
| 9 | 1.105095 | 0.990781 | 0.4982 | 0.0000 |
| 10 | 1.102570 | 0.989845 | 0.4982 | 0.0000 |
| 11 | 1.099086 | 0.988689 | 0.4965 | 0.0000 |
| 12 | 1.096166 | 0.987625 | 0.4947 | 0.0000 |
| 13 | 1.093493 | 0.986300 | 0.4982 | 0.0000 |
| 14 | 1.090873 | 0.985665 | 0.4965 | 0.0000 |
| 15 | 1.089257 | 0.985288 | 0.4965 | 0.0000 |
| 16 | 1.087058 | 0.983791 | 0.4947 | 0.0000 |
| 17 | 1.083467 | 0.980620 | 0.4930 | 0.0000 |
| 18 | 1.080368 | 0.978093 | 0.4912 | 0.0000 |
| 19 | 1.077444 | 0.975959 | 0.4912 | 0.0000 |
| 20 | 1.074925 | 0.974114 | 0.4912 | 0.0000 |

#### shuffled-layout-no-entanglement-s37

Runtime 3.47 s; weights L2 change 0.407622; Qiskit maximum marginal error 7.77e-16. [Saved run](artifacts/pilot/shuffled-layout-no-entanglement-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 1.005404 | 1.017034 | 0.4825 | 0.0000 |
| 1 | 1.003343 | 1.015805 | 0.4842 | 0.0000 |
| 2 | 1.002627 | 1.015281 | 0.4860 | 0.0000 |
| 3 | 1.001885 | 1.014177 | 0.4877 | 0.0000 |
| 4 | 1.000283 | 1.012376 | 0.4860 | 0.0000 |
| 5 | 0.998206 | 1.010657 | 0.4877 | 0.0000 |
| 6 | 0.996375 | 1.009201 | 0.4860 | 0.0000 |
| 7 | 0.994321 | 1.007126 | 0.4860 | 0.0000 |
| 8 | 0.992482 | 1.005329 | 0.4877 | 0.0000 |
| 9 | 0.990681 | 1.003389 | 0.4912 | 0.0000 |
| 10 | 0.987711 | 1.000565 | 0.4912 | 0.0000 |
| 11 | 0.983609 | 0.997710 | 0.4895 | 0.0000 |
| 12 | 0.980718 | 0.995973 | 0.4912 | 0.0000 |
| 13 | 0.978186 | 0.994439 | 0.4877 | 0.0000 |
| 14 | 0.976067 | 0.993014 | 0.4842 | 0.0000 |
| 15 | 0.974170 | 0.992108 | 0.4842 | 0.0000 |
| 16 | 0.972681 | 0.991179 | 0.4842 | 0.0000 |
| 17 | 0.970950 | 0.990185 | 0.4860 | 0.0000 |
| 18 | 0.968306 | 0.989571 | 0.4877 | 0.0000 |
| 19 | 0.965854 | 0.989514 | 0.4842 | 0.0000 |
| 20 | 0.963685 | 0.988335 | 0.4877 | 0.0000 |

#### qwen-direct-s11

Runtime 3.62 s; weights L2 change 0.391835; Qiskit maximum marginal error 6.66e-16. [Saved run](artifacts/pilot/qwen-direct-s11/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.749702 | 0.741619 | 0.5281 | 0.0000 |
| 1 | 0.748767 | 0.740666 | 0.5281 | 0.0000 |
| 2 | 0.747724 | 0.739882 | 0.5281 | 0.0000 |
| 3 | 0.746982 | 0.739268 | 0.5281 | 0.0000 |
| 4 | 0.745870 | 0.738326 | 0.5281 | 0.0000 |
| 5 | 0.744793 | 0.737495 | 0.5281 | 0.0000 |
| 6 | 0.743782 | 0.736739 | 0.5281 | 0.0000 |
| 7 | 0.742753 | 0.735904 | 0.5281 | 0.0000 |
| 8 | 0.741627 | 0.735125 | 0.5281 | 0.0000 |
| 9 | 0.740821 | 0.734665 | 0.5281 | 0.0000 |
| 10 | 0.739491 | 0.733495 | 0.5281 | 0.0000 |
| 11 | 0.737969 | 0.732005 | 0.5281 | 0.0000 |
| 12 | 0.736695 | 0.730691 | 0.5281 | 0.0000 |
| 13 | 0.735801 | 0.729903 | 0.5281 | 0.0000 |
| 14 | 0.735155 | 0.729481 | 0.5281 | 0.0000 |
| 15 | 0.734485 | 0.729036 | 0.5281 | 0.0000 |
| 16 | 0.733499 | 0.728218 | 0.5281 | 0.0000 |
| 17 | 0.732678 | 0.727494 | 0.5281 | 0.0000 |
| 18 | 0.731887 | 0.726773 | 0.5281 | 0.0000 |
| 19 | 0.731210 | 0.726121 | 0.5281 | 0.0000 |
| 20 | 0.730550 | 0.725615 | 0.5281 | 0.0000 |

#### qwen-direct-s23

Runtime 3.76 s; weights L2 change 0.390532; Qiskit maximum marginal error 5.00e-16. [Saved run](artifacts/pilot/qwen-direct-s23/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.730334 | 0.721882 | 0.5228 | 0.0000 |
| 1 | 0.729565 | 0.721419 | 0.5211 | 0.0000 |
| 2 | 0.728870 | 0.720861 | 0.5228 | 0.0000 |
| 3 | 0.727951 | 0.720234 | 0.5211 | 0.0000 |
| 4 | 0.726907 | 0.719493 | 0.5246 | 0.0000 |
| 5 | 0.726004 | 0.718724 | 0.5246 | 0.0000 |
| 6 | 0.724802 | 0.717747 | 0.5228 | 0.0000 |
| 7 | 0.723718 | 0.716949 | 0.5263 | 0.0000 |
| 8 | 0.722796 | 0.716280 | 0.5263 | 0.0000 |
| 9 | 0.722167 | 0.715783 | 0.5263 | 0.0000 |
| 10 | 0.721361 | 0.715112 | 0.5263 | 0.0000 |
| 11 | 0.720417 | 0.714242 | 0.5281 | 0.0000 |
| 12 | 0.719556 | 0.713478 | 0.5281 | 0.0000 |
| 13 | 0.718887 | 0.712880 | 0.5298 | 0.0000 |
| 14 | 0.717805 | 0.711757 | 0.5298 | 0.0000 |
| 15 | 0.716643 | 0.710594 | 0.5298 | 0.0000 |
| 16 | 0.715730 | 0.709780 | 0.5281 | 0.0000 |
| 17 | 0.715090 | 0.709228 | 0.5281 | 0.0000 |
| 18 | 0.714634 | 0.708921 | 0.5281 | 0.0000 |
| 19 | 0.714162 | 0.708669 | 0.5281 | 0.0000 |
| 20 | 0.713412 | 0.708099 | 0.5281 | 0.0000 |

#### qwen-direct-s37

Runtime 3.60 s; weights L2 change 0.396721; Qiskit maximum marginal error 6.66e-16. [Saved run](artifacts/pilot/qwen-direct-s37/result.json).

| Epoch | Train BCE | Test BCE | Test bit accuracy | Test exact words |
|---:|---:|---:|---:|---:|
| 0 | 0.742663 | 0.695709 | 0.5351 | 0.0000 |
| 1 | 0.742121 | 0.695069 | 0.5386 | 0.0000 |
| 2 | 0.741273 | 0.694388 | 0.5386 | 0.0000 |
| 3 | 0.740268 | 0.693557 | 0.5386 | 0.0000 |
| 4 | 0.739390 | 0.692808 | 0.5421 | 0.0000 |
| 5 | 0.738875 | 0.692380 | 0.5421 | 0.0000 |
| 6 | 0.738043 | 0.691728 | 0.5439 | 0.0000 |
| 7 | 0.737254 | 0.691129 | 0.5456 | 0.0000 |
| 8 | 0.736345 | 0.690472 | 0.5456 | 0.0000 |
| 9 | 0.735452 | 0.689888 | 0.5491 | 0.0000 |
| 10 | 0.734720 | 0.689384 | 0.5456 | 0.0000 |
| 11 | 0.734021 | 0.688845 | 0.5456 | 0.0000 |
| 12 | 0.733478 | 0.688407 | 0.5456 | 0.0000 |
| 13 | 0.732998 | 0.687990 | 0.5474 | 0.0000 |
| 14 | 0.732644 | 0.687696 | 0.5456 | 0.0000 |
| 15 | 0.731663 | 0.686962 | 0.5456 | 0.0000 |
| 16 | 0.730753 | 0.686273 | 0.5421 | 0.0000 |
| 17 | 0.730035 | 0.685723 | 0.5439 | 0.0000 |
| 18 | 0.729403 | 0.685270 | 0.5439 | 0.0000 |
| 19 | 0.728960 | 0.684951 | 0.5456 | 0.0000 |
| 20 | 0.728555 | 0.684659 | 0.5421 | 0.0000 |

### Completed report index

- [00-summary.md](reports/00-summary.md)
- [01-paper-audit.md](reports/01-paper-audit.md)
- [02-word-index-order.md](reports/02-word-index-order.md)
- [03-between-layers.md](reports/03-between-layers.md)
- [04-gate-order.md](reports/04-gate-order.md)
- [05-semantic-bits.md](reports/05-semantic-bits.md)
- [06-qwen-representations.md](reports/06-qwen-representations.md)
- [07-classical-and-negative-controls.md](reports/07-classical-and-negative-controls.md)

Reports preserve negative findings and separate experimental conventions from the paper.
All work remains on research/qcse-ordering; no push or remote write was performed.
