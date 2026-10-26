---
name: qcse-training
description: Maintain and validate the quantum-semantics QCSE corpus, training, checkpoint, and Slurm pipeline. Use for this repository's experiments and backend debugging.
---

The current focus is Hamiltonian encoders. Read [backlog.md](../../../backlog.md)
for completed work and measured limitations,
and [README.md](../../../README.md) for current commands. Prefer a small fix supported
by a reproducer; leave working simulation and optimization code alone.

Preserve these project invariants:

- Named dataset subsets select training sentences; vocabulary and frequency-based
  IDs use both complete active files, phrases and cleaned sentences, before filtering
  or sampling. Exclude Tatoeba: cleaned sentences are its curated version. `--data`
  instead defines an explicit custom corpus. Known CSV headers are not sentences.
- Keep duplicates and overlapping sentence windows in the same partition. The
  cluster protocol holds out 20% of sentence groups, runs two independent
  shuffle-splits within the remaining 80% (`val_fraction=0.2`), then refits on
  that full 80% and scores test once. Repeated validation sets may overlap;
  omitting `val_fraction` selects exhaustive k-fold CV instead. Reset weights
  and optimizer for each fit; use validation metrics to compare configurations.
- Full-data profiles keep every curated sentence/token example eligible. The
  legacy production sweep draws 5,000 examples per epoch; current robust studies
  explicitly use 100 epochs ×20,000 draws, full final validation and fixed final
  weights (see ../../../docs/protocol.md). In both cases the draw budget is independent
  of batch size; retain the final partial batch. Do not replace the full pool with
  a permanent tiny subset. Source balancing must not select a shared sentence twice.
- Sampled epochs monitor fixed, independently seeded subsets (2,048 per split by
  default), then score each complete validation fold once and the full held-out
  test after refit. Distinguish sampled curves from full final scores. Keep epoch
  saves free of full-corpus inference; sampled checkpoints omit embeddings while
  retaining the sampler RNG. Old runs preserve their saved full-pass schedule.
- Adam batch size changes training; simulation batch size only bounds simultaneous
  contexts. GPU float32 results are compared with tolerances against Qiskit/main.
  The bounded state cache preserves encoding and predictions; it does not bound
  host corpus/embedding arrays or guarantee full-data runtime within a Slurm job.
- New train/prepare invocations create unique output folders. `resume-cv RUN_DIR`
  resumes unfinished fits to the saved epoch target and skips completed ones.
  Single-fit `continue` adds epochs, defaults to the source run, and forks when
  `--output` is explicit. Preserve split/evaluation semantics, vocabulary, RNG,
  optimizer, and the last valid archive if saving fails.
- Validate CUDA inside an actual GPU allocation. CPU/MPS passes cannot establish
  CUDA success. Preserve Slurm's device mask, including MIG UUIDs; use the locked
  checkout environment and the existing backend preflight before changing kernels.
- The historical 45-configuration cluster sweep and completed robust cohorts are
  retired research workflows. Their conclusions/provenance are in
  [findings](../../../docs/findings.md); exact runners/reports remain in Git at
  `fb71a4a`. Do not reintroduce them merely to recover a result. Retained generic
  runners and legacy batch files are supporting code, not the active research plan.
  Current GPU work must stay within four one-10GB MIG instances overall and use
  `sbatch`; see [DGX operations](../../../docs/dgx-a100.md).

Hamiltonian work lives under `experiments/hamiltonian_study/`. Read its REPORT,
PROTOCOL, THEORY and REPRODUCE documents before changing it. Preserve training-only
vocabulary/graph statistics, grouped partitions, the declared UNK policy, selection
locks and distinct marginal/joint-Born readouts. The alternating graph/context walk
and the fixed Hamiltonian encoder plus QCSE ansatz are separate models. Retain
identity, calibrated bigram and classical graph controls. These CPU study results
use different data/targets/budgets from the historical robust benchmark.

The separate `contextual.py` proof of concept uses an eight-dimensional latent
Hamiltonian, learned ordered context and a residual word head, with no rare-role
conditioning. Read CONTEXTUAL_POC.md before extending it. Preserve exact bigram
initialization, paired classical/last-word controls, source/split fingerprints
and validation-only selection. Its checkpoints are inference-only; no optimizer
resume is implemented. Its improved validation does not validate the old fixed
vocabulary graph or establish quantum advantage.

The separate `born_evidence.py` experiment trains only Hermitian matrix coordinates
with fixed encoding and direct Born readout. Read BORN_EVIDENCE.md before extending
it. Preserve paired coherent/retrained-dephased starts, the full sixteen-outcome
digit distribution (including unused labels), recipe/checkpoint locks before test,
and the symbolic versus real-data distinction. `analyze_born.py` audits saved
predictions and derives a post-hoc rank-four dephased NLL bound. Beating this bound
establishes a coherence contribution within the architecture; its stronger classical
MLP and absence of hardware evidence remain material limits. Natural-language
evidence is a separate follow-up, not part of the digit readout bound.

The `language_born.py` follow-up trains all H entries with fixed ordered word
amplitudes and direct joint word measurement. Read LANGUAGE_BORN.md before changes.
Preserve its train-only vocabulary/stats, sentence-group split, full UNK outcome and
known-target metrics without renormalization, paired retrained dephased starts,
matched-input MLP and bigram/count/graph references. Lock recipes before fresh
seeds and all checkpoints before test. The spectral forward/Frechet adjoint is
exact for Hermitian H and tested at degeneracy; only first-order gradients are
supported. It establishes language training/coherence gains relative to the
declared controls, while MLP and subset bigram remain stronger. Dense-H scaling
and convergence are unestablished; H chose the final epoch and lowest tested LR.

The optional report-inspired `scripts/semantic_experiment.py` is separate from
the production QCSE model. Read the historical semantic prototype documentation
in `git show fb71a4a:docs/archive/SEMANTIC_PROTOTYPE.md` before changing it.
Preserve trainable embedding lookup, XYZ readout and tied word
softmax; IDs only address rows. Resume uses a total epoch target and restores the
optimizer, sampling RNG and grouped split. Its local protocol is one 64/16/20
split with explicit final test scoring. Exact-state entropy/Schmidt diagnostics
are not matrix-product-state truncation. CPU results do not establish GPU behavior.

Use the existing tests in [tests/](../../../tests/) for changed behavior and
`scripts/check_backend.py` on the available device. Record actual commands/results
and remaining hardware limitations in docs/validation.md, with brief state in
backlog.md; keep raw generated runs and conversation exports out of Git.
