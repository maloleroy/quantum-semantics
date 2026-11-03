# Cosine learning-rate comparison for adaptive spectra

2 November 2026. Implemented and checked; this original CPU/manual protocol was
not run. On 3 November the user authorized DGX research execution; the
[CUDA comparison](dgx-cosine.md) supersedes its launch plan and adds fresh
same-backend constant controls. The original protocol is retained below. The
adaptive-spectrum recipe improves NLL over matched static capacity, but still
trails its MLP and produces fragmented text. Late quantum validation scores improve
slowly with fluctuations under constant .003 updates. Test whether a predeclared
smaller late step size improves the retained model; the curves do not establish
that step size causes the remaining deficit.

## Fixed schedule and controls

Add opt-in `--lr-schedule cosine --min-lr-ratio .1`. For training epoch e = 1,…,30,
use one rate for every batch in that epoch:

\[
\eta_e=.003\left[.1+.9\frac{1+\cos(\pi(e-1)/29)}2\right].
\]

The first epoch uses .003 and the last .0003. There is no warmup, restart,
validation-triggered adaptation or optimizer resume. With one epoch the initial
rate is used. Default `constant` preserves the previous update behavior; the
floor ratio is ignored for constant schedules. Ratios must be in (0,1]. The
schedule and floor are stored in metadata and each applied rate is recorded as
`next_epoch_learning_rate` alongside `next_epoch_online_loss` in history CSV.
The final validation-only row has both fields empty.

All fits start fresh, use seed 11, 30 epochs, batches of 512 and the same 30,720
optimizer updates. NLL loss and NLL checkpoint selection stay fixed. This changes
the rate trajectory and its average, not just its temporal ordering. Without a
separate lower-constant-rate fit, any gain cannot be attributed uniquely to cosine
shape or to reduced late noise.

| Fit | Model | Context retained | Parameters |
|---|---|---|---:|
| adaptive | Adaptive spectrum, rank 128 | 128 tokens | 751,892 |
| short | Adaptive spectrum, rank 128 | Final 16 tokens | 751,892 |
| static | Ordinary product, rank 176 | 128 tokens | 751,988 |
| mlp | Matched adaptive-input MLP | 128 tokens | 750,460 |

All use two stages, auxiliary size 2, recent window 16, phase scale 1 and Born
readout. Amplitude decay .9 remains recorded but unused. No model architecture
changes for this experiment. The quantum path retains local preparation,
context-dependent eigenvalues and direct measurement without a classical word
head, temperature modification or rejection.

Three constant-.003 counterparts from `output/adaptive-spectrum/batch-gzx4eP`
are frozen references: adaptive NLL 2.7007, static 2.7558 and matched MLP 2.4393.
Their source is archived before training-code edits. Their previously reproduced
validation arrays, source, weights and datasets are verified rather than loading
old weights against changed current source. Any adaptive improvement must still be
compared with the strongest available classical reference, including the new MLP.

The short adaptive fit tests whether the longer context helps under this schedule.
Older-context feature parameters remain allocated but receive no input; equal
parameter count does not imply equal effective used capacity. A null or negative
long-context effect would constrain further claims about meaningful context use.
There is no matched short-context classical fit in this screen, so it cannot compare
quantum and classical long-context effects as a difference-in-differences study.

## Data and reporting

Reuse exactly `output/product-coverage/batch-RofZOb/data/full.npz`: 524,288 training
positions, 4,096 validation targets, 512-outcome training-only byte BPE and the
original article partitions. No tokenizer/statistic refit or new training positions.
The manual preparation step creates a paired short dataset by setting every context
column before the final 16 to -1, preserving the 128-column shape, targets, groups,
byte lengths, vocabulary and all-training counts. Validate each partition before
and after serialization. No held-out test is scored.

Checkpoint selection evaluates the same validation targets at epoch zero and after
each of 30 epochs. Weights are locked before report/sample generation. The reporting
script reproduces saved predictions, checks actual learning-rate history, and saves
10 metric rows, 20 space/non-space target buckets and 32 paired interval rows.
Comparisons are full adaptive versus every other model/reference/baseline, plus
static versus constant-static and MLP versus constant-MLP, using NLL, sampling and
argmax. Skip infinite-NLL inference against the training-modal predictor. Intervals
are descriptive article-bootstrap summaries conditional on one seed and reused
validation, excluding selection uncertainty.

Save 15 fixed-prompt, unfiltered 64-token continuations from the three new full-context
fits and bigram/unigram. Use the same three prompts and seeds 20261030–20261032
as before. Raw IDs remain in NPZ, text/tables in CSV. Short-context generation is
omitted because the generic generator uses the full window. No archived-reference
resampling. Retain invalid outcomes and UNKs. Better NLL or sampling alone does not
establish coherent text, useful long-context generation or practical quantum advantage.

## Manual command

```bash
bash output/adaptive-cosine/launch.sh
```

Planning estimate: **2–5 hours**, unmeasured for this schedule comparison. The
preceding adaptive fit took 130.6 minutes. Four fits use two Torch threads each on
the eight-physical-core laptop. The codex-long-jobs controller uses event-only
delivery and no viewer; the agent has not launched research training.

The launcher verifies inputs, prepares/checks the paired short data, trains,
reproduces locked predictions, validates rate histories and reports results.
Source, scripts, dependencies, data and archived references are checksummed before
and after fitting. Generated short data have a separate checksum lock. Completed
fits are preserved on retry:

```bash
bash output/adaptive-cosine/launch.sh output/adaptive-cosine/batch-XXXXXX
```

The final marker is `ADAPTIVE_COSINE_COMPLETE`. All generated tools/artifacts are
under ignored `output/adaptive-cosine/`. All 73 tests, Ruff and strict Pyright pass.
Tests cover schedule endpoints/errors, constant-equivalent updates and weights,
effective changed updates, history/metadata recording and locked validation
selection. Small context-masking checks, generated-tool imports, shell syntax and
archived-reference checks pass. The full short dataset will be prepared during the
manual run; no new predictive result is measured yet.
