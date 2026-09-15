# Training protocol audit

Updated 2026-09-15 after the local 50-epoch execution audit.

## Bottom line

The training code and split/evaluation protocol are mechanically sound under the
tested CPU conditions, and the active launchers now use 50 epochs. It is not honest
to claim that every 50-epoch, full-corpus, CUDA production training has already
been validated: the cluster archive is historical 10-epoch evidence, and this Mac
has no CUDA device.

## What was actually run at 50 epochs

The capped CPU audit completed:

| Family | Actual fits | Epochs per fit | Local data cap | Result |
| --- | ---: | ---: | ---: | --- |
| Causal QCSE/GPT-like | 2 validation folds + 1 refit | 50 | 16 sentences; 17 draws/epoch | complete |
| Semantic encoding | 10 configurations × 3 seeds | 50 | 16 sentences; 5,000 draws/epoch | complete |
| Attention CV | 5 setups × 2 folds + refit | 50 | 16 sentences; 17 draws/epoch | complete |

Every audited history contains epochs 0 through 50, remains finite, and reaches its
final held-out scoring stage. The causal run also passed an explicit saved-artifact
check for outer and inner sentence-group disjointness. The complete local pytest
suite is now **84 passed, 46 skipped**; the skipped cases require unavailable
CUDA/MPS.

## Why the protocol is legitimate

- Sentence windows are assigned by normalized sentence group, so duplicate sentences
  and all windows from one sentence remain on one side of every split.
- Training sampling uses replacement only within the training IDs. Repeated examples
  or repeated sentences are allowed, but cannot cross into validation or test.
- Production QCSE uses an outer 80/20 development/test split, two independent 20%
  validation shuffle-splits within development, and a fresh development refit. The
  held-out test is scored once after that refit. The five-fold cases in the test suite
  exercise the generic exhaustive-CV path; they are not the production sweep protocol.
- Configurations use the same active sources, cleaning, vocabulary construction,
  sample budget, initialization seed policy and split policy. The intended parameter
  variation is explicit in the manifests.
- The report selectors now use mean validation word/cosine top-1 first. BCE or
  cross-entropy is retained as an optimization/calibration diagnostic and deterministic
  tie-breaker, never as the primary retrieval criterion.
- Final validation and test inference now use deterministic, seeded, no-replacement
  samples of at most 10,000 example IDs per already-separated split. Per-epoch
  monitoring remains smaller (`eval_examples`, 2,048 for QCSE/semantic and 512 for
  attention). Setting `final_eval_examples=0` restores exhaustive final scoring.
  This changes inference cost, not the 80/20 sentence-group split or the training
  sampler. A 98/1/1 split is not substituted because it changes the estimand and
  makes the requested repeated-fold comparison non-comparable.

## Important qualifications

- The 45-run QCSE archive has 44 completed configurations at 10 epochs and one
  incomplete configuration. It cannot establish 50-epoch results.
- The new 50-epoch audit uses a 16-sentence cap. It verifies execution, checkpointing,
  finite metrics, final scoring and split handling, but not full-corpus runtime or
  generalization.
- The causal production sweep has one seed and two validation repeats. Reported spread
  is split-to-split variation, not random-initialization uncertainty.
- Fixed 5,000 draws per epoch give batch sizes 16, 64 and 256 different optimizer
  update counts (313, 79 and 20). Batch/depth conclusions must therefore be treated as
  compute-confounded unless a follow-up equalizes updates or total optimizer work.
- The vocabulary is deliberately built from all active source rows before splitting.
  This fixes IDs across ablations and avoids silently shrinking the output space, but
  is a transductive vocabulary convention rather than a strict train-only benchmark.
- CPU execution and backend parity do not establish CUDA behavior. The cluster must
  still run its real CUDA preflight and record allocation/provenance metadata.
- A full-corpus 5-fold/50-epoch run was started locally but stopped after fold 1,
  epoch 1: the observed rate was about 4.5 CPU minutes per sampled epoch, implying
  roughly 20–25 hours for six fits; final scoring is now capped by default but
  training remains the dominant cost. Its checkpoint is preserved but no partial
  metrics are treated as a result.

## Verdict

The implementation is sane and leakage-controlled for the stated transductive,
sentence-grouped protocol. The experiments are fair for comparing the declared
settings under shared splits and budgets, with the batch-size confound disclosed.
They are not yet fully rigorous as publication evidence until the full 50-epoch QCSE
matrix is run on CUDA, the incomplete pool comparison is completed, and the strongest
configurations are repeated across multiple independent seeds.
