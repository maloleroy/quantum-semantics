# Supporting robust benchmark protocol

This records the historical robust comparison and retained generic runner.
The active Hamiltonian experiments use their own
[protocol](../experiments/hamiltonian_study/PROTOCOL.md), training-only
vocabulary/graph and smaller saved target subsets. Use that study's settings
for Hamiltonian work. The conclusions below are synthesized in
[findings](findings.md).

The current research budget is **100 epochs ×20,000 replacement draws**,
optimizer batch 64, final-state scoring at 100, full eligible training pool and
full final validation. An epoch is a draw budget, not a corpus pass. The generic
CLI still has historical 50×5,000/early-stopping defaults; experiment manifests
must explicitly specify the intended protocol.

## Two tasks

| | Causal L4 | Bidirectional L2R2 |
| --- | --- | --- |
| Input | Four preceding slots | Two preceding/two following slots |
| Target | Word at t, positive positions only | Same target keys as L4 |
| Information forbidden | Current word and all future words | Current word |
| Native extension | Longer prefix/autoregressive evaluation | Wider masked context/representation evaluation |

Compare models **within each task**. Bidirectional conditional NLL is not an
autoregressive sentence likelihood; its exponent is not GPT perplexity. Native
CBOW in older runs included sentence-initial targets, changing both training
and evaluation populations. Restricting evaluation alone did not fix that.

## Data and exposure

- Active corpus:202,172 exact-deduplicated sentences, 10,864-word shared vocabulary.
  Sources and tokenizer/corpus/vocabulary hashes accompany each study.
- Split sentence groups before creating training/validation/test pools. Keep
  duplicate sentences and their overlapping windows together. Exact grouping
  does not remove every template/paraphrase or make domains independent.
- Reserve 20% outer test groups. Modern five-fold CV partitions development;
  recent controls reuse its fold 0:129,390 training and 32,348 validation groups,
  1,014,085 /252,120 positive-position targets. The outer test has already been
  inspected in earlier project work; it is not a fresh confirmation set.
- Draw from the entire training pool with replacement; retain the final partial
  optimizer batch. At2M draws, expected coverage is about 86.09%, not every
  eligible target. Recorded seed 47/48 sampler replay yields 86.08/86.11%; the
  actual streams/checkpoints were omitted from the compact export.
- Vocabulary/frequency-ranked IDs use complete sources before splitting. Disclose
  that transductive convention; train-only vocabulary/OOV is a separate study.

## Training and selection

Reset weights, optimizer and sampler per independent fit. Log initial state
hashes, not just integer seeds: changing circuit width/depth can consume extra
RNG draws before the word head is initialized. Match actual encoder/head state
where a mechanism comparison requires it.

Distinguish optimizer batch 64 from simulation microbatch 8. Microbatch 64 remains
a candidate execution recipe: one-update checks passed, full trajectories have
not run. Keep the update/draw budget unchanged when testing it. BERT dropout and
floating reductions can change the trajectory even at the same optimizer batch.

Use independently seeded, fixed 2048-target monitors for curves. Compare full
validation endpoints; do not present monitor peaks as full-fold performance.
The fixed 100-epoch protocol disables early stopping. Legacy focused experiments
used validation-selected early stopping and shorter exposure; they remain a
different study, not cheaper replicas of this one.

Tune on validation with declared equal opportunities. Reusing a historical LR
is valid for a fixed-recipe mechanism control; it does not make a new architecture
equally tuned. Freeze choices before held-out refits. Never select the best
test seed or promote a post-hoc secondary metric to the prespecified primary.

## Metrics and uncertainty

Report full-word NLL, top-1 and top-5 together. Paper half-bits similarity,
threshold exact-bit decoding, cosine retrieval and word-softmax ranking are
different endpoints. Record valid output vocabulary and invalid-code handling.

Paired resampling uses complete sentence groups and matching seed labels across
models/tasks. Keep token-weighted and sentence-macro effects and individual
seed differences. Overlapping folds are not independent training replications;
two seeds across two tasks are not four independent initializations.

The latest 10,000-draw intervals are nominal 95%, exploratory and unadjusted for
multiple comparisons/selection. They describe the observed seeds/population,
not uncertainty in the entire tuning process. For new confirmation, prespecify
one primary contrast per task, report a useful effect threshold and adjust the
joint primary family with an appropriate valid test. Prefer a short list with
fresh seeds over five folds ×three seeds for every screening cell.

## Required records

| Record | What must be retained |
| --- | --- |
| Manifest | Task/arm/fold/seed, all hyperparameters, fingerprint, parent manifests |
| Source/data | Executed source snapshots/hashes, dependency lock/runtime, sentence/split/target hashes |
| Training | Initial parameters, optimizer/RNG state, draw/update counts, sampled curves and full final metrics |
| Predictions | IDs/groups/positions/targets, top-five IDs and per-token metrics; full rank/confidence from an explicit inference pass if needed |
| Compute | Per-phase/epoch/fit/wrapper timing, memory, device/MIG and Slurm allocation accounting |
| Export | Included-file hashes, explicit omissions, original archives retained locally |

Frozen DGX studies use source 483bea6. Current source edits are not compatible
with their guards by default. Use copied extensions or an isolated versioned
checkout; do not edit manifests/source hashes to bypass the check. See
[DGX operations](dgx-a100.md) and [validation](validation.md).
