# Quantum evolution between context blocks

1 November 2026. Completed and independently audited; the original protocol follows.

The direct-success experiment increased its score while damaging NLL and text.
This screen returns to NLL training and validation selection. It tests whether
feeding later context into an already-evolved quantum state improves prediction.
It is a bounded representation experiment, not evidence of quantum advantage.

## Completed results

All five fits in `output/stream-screen/batch-UCbiR5` completed, and the final
job marker is present. Independent evaluation reproduced every new locked
prediction and metric to 1e-12. Four archived references, 12 metric rows,
32 paired intervals, 24 target buckets and 18 saved sample renderings were checked.
Original/short targets, groups, byte lengths and recent contexts match. No new
samples were drawn and no held-out test was scored.

| Model | NLL | Sampling success | Argmax | Fit minutes |
|---|---:|---:|---:|---:|
| Ordered streaming | 3.2984 | 20.79% | 34.57% | 35.9 |
| Repeated full-context pooling | 2.9734 | 23.49% | 37.13% | 35.9 |
| Short-context streaming | 3.1750 | 21.95% | 34.94% | 35.4 |
| Frozen streaming dynamics | 5.1828 | 2.05% | 30.03% | 28.5 |
| Matched streaming MLP | 3.1300 | 26.08% | 35.82% | 9.8 |
| Earlier rank-128 product | 2.9281 | 23.20% | 37.11% | 20.7 |
| Earlier rank-64 MLP | 2.7966 | 30.84% | 41.06% | 8.6 |
| All-training bigram | 3.0593 | 23.16% | 34.03% | — |

Ordered streaming worsens NLL versus pooling by .32499, interval [.30055, .34804],
and sampling by 2.707 percentage points, interval [2.525, 2.882] in deficit
magnitude. It also loses to the short-context fit by .12345 NLL, interval
[.10506, .13889], and to its matched MLP by .16841, interval [.12543, .20573].
This recipe does not establish useful processing of longer context. The pooling
comparison changes information allocation as well as chronology; it cannot
identify one isolated mechanism causing the deficit.

On non-space targets, ordered sampling is only 2.52% and NLL 4.4722, versus
4.29% / 4.0386 for pooling and 10.85% / 3.8282 for the older rank-64 MLP.
All full-context learned models still generate fragments. Ordered draws contain
four UTF-8 replacement characters across three samples; pooled draws contain
three, and the MLP one. Their samples contain no UNK or other control characters.
Frozen dynamics produce three UNKs, 27 replacement characters and 20 controls.
These are small fixed-prompt diagnostics, not corpus-level language-quality rates.

Ordered selection chooses epoch 29; pooling and short choose 30, MLP 20 and frozen 9.
The longest fit takes 35.9 minutes, within the unmeasured 30–90 minute planning
range. Retain the earlier product model; do not scale this streaming recipe.
Learning Hamiltonians remains useful relative to frozen dynamics here, but neither
quantum superiority nor meaningful generation has been demonstrated.

Evidence: [metrics](stream-validation.csv), [paired intervals](stream-paired.csv),
[target buckets](stream-target-buckets.csv),
[sample diagnostics](stream-sample-diagnostics.csv).
The next bounded question is broader training coverage for the retained product
model, with equal-update repeated-set controls and classical competitors. The
current gradient sample still contains only 1.91% of eligible training transitions.

## Architecture and controls

`stream` splits the chronological 128-token context into four 32-token blocks.
The oldest block prepares a product state with local RY rotations. A separate
rank-32 Hamiltonian evolves the joint register after each block. Later blocks
apply local RZ gates before their Hamiltonian, so their inputs can act on an
entangled state. The four Hamiltonians are independently learned, not shared.
Final Born measurement retains the nine token bits and ignores one auxiliary bit.
There is no classical vocabulary head, temperature adjustment or rejection.

Each block pools its last token, preceding 15 tokens and remaining 16 tokens using
the existing learned token-angle family and recency weights. The first block has
its own table; all three later blocks share another table. Four stage offsets
are learned. Initial RY offsets are pi/2 and later offsets zero. Padding contributes
no token angle; an empty first block therefore initially prepares the plus state.
This initialization differs from the old product model and is part of the recipe.
`--phase-scale 0` removes context dependence. The context width must divide evenly
by the number of stages.

| Fit | Inputs and dynamics | Trainable parameters |
|---|---|---:|
| ordered | Four chronological blocks; learned Hamiltonians | 293,032 |
| pooled | Full 128-token context supplied at every stage | 293,032 |
| short | Only final 16 tokens retained; same four stages | 293,032 |
| frozen | Ordered blocks; fixed initial Hamiltonians | 30,760 |
| mlp | Same initial amplitudes and later control angles; tanh MLP | 292,961 |

The MLP independently learns its controller and sees all initial real amplitudes
plus all later angles (1,054 features, 167 hidden units). It matches the quantum
model's feature family and parameter budget. The pooled control matches parameter
count and stage count, but changes allocation and compression of information;
it does not isolate pure order sensitivity. Short-context padding retains learned
offsets and evolution at earlier stages. Frozen dynamics test the value of learning
Hamiltonians, not the necessity of quantum computation.

The earlier sequential-preparation experiment placed all token processing before
quantum evolution, while preparation remained separable. Here later inputs act
between joint evolutions. There are only three diagonal RZ applications after
preparation, avoiding that experiment's long serial chain of local rotations.
No simulator speed or hardware compilation advantage is assumed.

## Locked protocol

Reuse `output/product-exposure/batch-MDwDbm/data`: train-only byte BPE with 512
outcomes, 131,072 training transitions, 4,096 validation transitions, width 128,
and unchanged official article groups. The short dataset masks positions older
than 16 while preserving width, targets, groups and byte lengths. Gradient training
still covers only about 1.91% of available training transitions. Count baselines
use all training counts. Zero UNK target coverage does not establish text quality.

All five fits use seed 11, four stages, rank 32, auxiliary size 2, recent window 16,
phase scale 1, 30 epochs, batches of 512, Adam learning rate .003, NLL loss and NLL
checkpoint selection. Amplitude decay .9 is recorded but unused by these models.
Each fit uses one Torch thread; the five fits run concurrently. No held-out test
is scored. Weights are locked before validation reporting and sample generation.

Four audited NLL references are retained: product ranks 64/128 and their matched
MLPs, from the expanded-exposure and capacity screens. Archived source, dataset,
weight and prediction files are pinned. Their previously reproduced validation
arrays are reused; old weights are not loaded against changed current source.
The rank-64 MLP's NLL 2.7966 remains the stronger classical reference. Comparisons
to the old product models change stage count, controller allocation, initialization
and rank per stage together, and therefore assess a recipe rather than one cause.

Reporting includes 12 metric rows, 24 modal/non-modal target buckets and 32 paired
interval rows: ordered versus every other fit/reference/count/modal baseline,
for NLL, sampling and argmax, excluding infinite modal-baseline NLL. Intervals
are descriptive article-bootstrap summaries conditional on one seed and reused
validation; they exclude selection uncertainty. Sampling is not argmax accuracy.

Generate three fixed 64-token continuations for each new full-context fit and the
two count baselines (18 total), saving raw token IDs and rendered text. Prompts are
`The history of `, `In 2026, the researchers ` and `Quantum mechanics describes `;
seeds are 20261030 through 20261032. Short-context generation is omitted because
the generic generation routine uses the full window. No draws are rejected.
NLL, non-modal behavior and text quality must support any proposed improvement;
a higher sampling score alone cannot promote a generator.

## Manual command

From the repository root:

```bash
bash output/stream-screen/launch.sh
```

Planning estimate: **30–90 minutes**, unmeasured for this architecture. Earlier
runtime estimates have been inaccurate; the launcher records actual fit times.
It uses the codex-long-jobs controller with event-only delivery and no viewer.
Source, dependencies, scripts, data and archived references are checksummed before
and after fitting. Completed fits are verified and preserved on a retry:

```bash
bash output/stream-screen/launch.sh output/stream-screen/batch-XXXXXX
```

The final marker is `STREAM_SCREEN_COMPLETE`, printed only after verification and
reporting. Generated launchers, tools, snapshots and results live under ignored
`output/stream-screen/`. All 66 tests, Ruff and strict Pyright pass, including
explicit dense circuit probability/gradient comparisons, padding/context behavior,
parameter budgets and frozen-core checks. No performance improvement is measured.
