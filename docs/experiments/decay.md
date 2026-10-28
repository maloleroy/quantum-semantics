# Recency-decay screen with auxiliary word measurement

28 October 2026. Validation-only follow-up to the auxiliary-register experiment.
The previous completed experiment improved prediction with a more flexible
quantum readout; post-fit truncation suggested only a small contribution beyond
16 words. This screen tests whether deliberately retaining older words helps.

## Declared protocol

Keep the exact amplitude preparation rule, including repeated-token interference
and position phase pi/2, but replace population decay .7 with .9 or .97. For
distinct tokens, age-31 population relative to age zero is d^31: about .000016,
.0382 or .389 for the three decays. These ratios describe encoding, not predictive
importance. Repeated token amplitudes add before normalization and can cancel.
This modification has no learned classical weights or corpus-derived features.
The CLI now exposes `--amplitude-decay`, persisted in model/checkpoint settings;
default .7 preserves the earlier recipe. Model config requires 0<d<1, excluding
exact equal-weight cancellations at d=1; the encoder retains its norm guard.

Declared before fits: coherent d=.9 and .97, independently retrained dephased
control d=.9. Compare with existing d=.7 coherent/dephased fits, which used the
same examples, initialization seed, capacity and optimizer settings. Same
512-word vocabulary, 32-word context, official article partitions, training-only
vocabulary/statistics; 8,192 gradient targets and 4,096 validation targets from
`output/interleaved-screen/dataset.npz`. Seed 11, two rank-64 Hamiltonians on
word plus one auxiliary qubit, 262,272 real trainable coordinates, Adam LR .001,
30 epochs, batch 512, clipping 5, NLL training and validation-NLL selection
including epoch zero. Direct word measurement ignores the auxiliary outcome;
no postselection or learned decoder. No test scoring.

All recipes reuse the validation partition for adaptive research. Any gain is
exploratory, not independent confirmation or established long-context benefit.
The model still compresses 32 words into one word-register amplitude vector and
uses periodic position phases; slower decay does not remove that bottleneck.
All-training bigram and the MLP remain classical references. If a decay improves
NLL, evaluate the selected recipe on the expanded training subset with its
retrained control, keeping test untouched. Otherwise do not promote it merely
because it reads more context.

Previous identical-size fits took 80–96 seconds each. This screen changes only
fixed input weights and is estimated under two minutes per fit before contention;
three one-thread workers run concurrently. Generated scripts, locked metadata,
CSV histories and NPZ arrays/weights live under `output/decay-screen/` (ignored).
The scientific check verifies configured amplitude populations and normalization
through the existing suite. All 31 tests, strict Pyright and Ruff pass.

## Adaptive follow-up

Initial results: coherent .9 NLL 3.7864, coherent .97 NLL 3.7821 versus baseline
.7 NLL 3.7892; dephased .9 NLL 3.6905 versus baseline .7 NLL 3.7525. Coherent
NLL gains are small, while argmax rises and sampling changes little. Before
additional fits, add dephased .97 to complete the best coherent encoding's
matched control. Also retrain the existing MLP at .9 and .97, using its earlier
LR .003 and the same 30-epoch budget/subset. These are adaptive exploratory
comparisons, not equal-size exhaustive searches across families. Selection
continues to use validation NLL, not the visually larger argmax change.

## Expanded subset and retrained context-length controls

Dephased .97 completed at NLL 3.6698, better than dephased .9 (3.6905).
The .97 encoding therefore has the lowest screened NLL for both quantum variants.
Before further fits, choose .97 for a matched coherent/dephased comparison on
32,768 gradient targets. Preserve 30 epochs, LR .001, seed 11 and all validation
rows. Independently retrain both variants with only the most recent four words
available, padding earlier positions in the same 32-column arrays. All targets,
groups, vocabulary and training-only counts are identical; padding is applied
uniformly to all partitions but test is never scored. This addresses the prior
post-fit truncation diagnostic's distribution-shift limitation.

The matched window experiment adds no model parameters and uses the same
initialization and minibatch orders. The MLP reference is also fitted on the
expanded subset at .97, LR .003, 30 epochs (32-word context). The earlier .7 MLP
remains the stronger selected classical reference; do not discard it to improve
the quantum comparison. These follow-ups are adaptive validation exploration.
Prior full fits took 257–287 s, and small fits here took about 77–92 s. Four
one-thread quantum workers plus one MLP worker should finish within ten minutes
per fit even with contention, on the eight-physical-core host.

## Results on 8,192 training targets

| Model | Decay | Validation NLL | One-shot success | Known-target success | Argmax |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coherent | .7 | 3.7892 | 13.96% | 2.07% | 33.18% |
| Coherent | .9 | 3.7864 | 13.90% | 1.83% | 38.06% |
| Coherent | .97 | 3.7821 | 13.59% | 1.77% | 39.14% |
| Dephased | .7 | 3.7525 | 13.17% | 2.16% | 31.98% |
| Dephased | .9 | 3.6905 | 13.33% | 2.02% | 38.45% |
| Dephased | .97 | 3.6698 | 13.35% | 2.00% | 40.31% |
| MLP | .7 | 3.4862 | 19.13% | 3.08% | 41.24% |
| MLP | .9 | 3.5440 | 17.01% | 2.18% | 41.77% |
| MLP | .97 | 3.5453 | 19.46% | 1.84% | 41.80% |

These rows share identical validation examples; 41.80% of targets are UNK.
Known-target probabilities are not renormalized. The argmax improvement in the
quantum models coexists with worsening known-target sampling and very low
known-target argmax, so it must not be described as broad language improvement.
The slower-decay MLPs have zero known-target argmax accuracy at the selected
checkpoints. The original MLP remains the stronger reference.

For coherent .97 minus .7, the NLL change is −.00710, article-group bootstrap
95% interval [−.04273, +.02522]. Dephased .97 minus .7 changes NLL by −.08268,
interval [−.11875, −.05130]. These are descriptive conditional validation
intervals after adaptive selection, not fresh confirmation or adjustments for
the recipe/checkpoint search. [Exact scores](decay-validation.csv) retain known
NLL, argmax, selected epoch, training NLL, timings and artifact paths;
[paired differences](decay-paired.csv) retain the complete numerical intervals.

## Results on 32,768 training targets

| Model | Decay | Context | Validation NLL | Known-target NLL | One-shot success | Known-target success | Argmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Coherent reference | .7 | 32 | 3.6379 | 5.2409 | 14.01% | 2.24% | 34.64% |
| Coherent | .97 | 32 | 3.7215 | 5.4276 | 13.97% | 1.86% | 38.77% |
| Coherent, retrained short | .97 | 4 | 3.7730 | 5.3194 | 12.85% | 2.11% | 32.23% |
| Dephased reference | .7 | 32 | 3.5954 | 5.1594 | 13.65% | 2.45% | 35.47% |
| Dephased | .97 | 32 | 3.5643 | 5.1747 | 14.13% | 2.11% | 40.48% |
| Dephased, retrained short | .97 | 4 | 3.7826 | 5.3009 | 12.17% | 2.20% | 32.42% |
| MLP reference | .7 | 32 | 3.3003 | 5.0686 | 21.92% | 4.46% | 42.36% |
| MLP | .97 | 32 | 3.4949 | 5.3677 | 18.98% | 2.17% | 41.77% |

The four quantum fits took 316.0, 317.9, 361.8 and 368.8 seconds; MLP took
75.5 seconds. All finished within ten minutes. Coherent .97 fails to reproduce
its small-subset NLL edge on the expanded subset and worsens NLL by .08362
relative to coherent .7, descriptive group interval [+.03806, +.12364]. Keep
.7 as the stronger coherent reference. The stronger MLP and all-training bigram
(NLL 3.0300, success 22.68%) remain ahead; do not replace them with the weaker
slow-decay MLP to suggest a quantum win.

Dephased .97 improves aggregate NLL by .03114 over .7, interval
[−.06943, +.00543], and sampling by .476 percentage points, interval
[+.149, +.856]. Its known-target NLL and success both worsen slightly. Thus the
aggregate recipe improvement is driven by UNK, and even the aggregate NLL
interval includes zero. This is not a robust, general language-quality gain.

The independently retrained window comparison at .97 is more informative about
context length than post-fit truncation:

- Coherent 32 minus four words: NLL −.05150, interval [−.11850, +.00459];
  sampling +1.115 percentage points, interval [+.586, +1.684]. Known-target NLL
  and success worsen with 32 words. The overall gain is again driven by UNK.
- Dephased 32 minus four words: NLL −.21836, interval [−.28222, −.16360];
  sampling +1.956 percentage points, interval [+1.471, +2.474]. Known-target
  NLL improves 5.3009 → 5.1747, while known-target sampling falls slightly.

Intervals resample validation article groups and condition on one initialization
seed and selected checkpoints. Repeated adaptive validation use and seed
uncertainty are not accounted for. This supports a context-length benefit for
the dephased input-population model under this recipe, not an interference,
entanglement, hardware, or overall classical advantage. The dephased prediction
is classically representable as a stochastic map of fixed input populations.

## Research decision and reproduction

Do not promote slower forgetting as an improvement to the coherent model.
Preserving older words without preserving their predictive relationship to the
latest words is insufficient. Next architectural work should preserve recent
and distant information separately, or address the restricted vocabulary/UNK
bottleneck. Competitive known-word prediction and meaningful long-form quantum
text generation remain unresolved. No held-out test has been scored.

Reproduce the expanded 32-word comparison through the installed CLI:

```bash
uv run qcse train \
  --dataset output/dataset-selection/dataset-20261026T211719Z-c96984fa/dataset.npz \
  --models ancilla ancilla_dephased --ancilla-size 2 --rank 64 --layers 2 \
  --amplitude-decay .97 --seeds 11 --epochs 30 --batch-size 512 --lr .001 \
  --loss nll --selection nll --threads 1 --output output/decay-reproduction
```

This CLI command runs the two fits sequentially; the recorded experiment ran
separate commands concurrently. The generated `output/decay-screen/prepare_short.py`
reproduces the matched four-word dataset (refuses to overwrite an existing file);
use that dataset with the same CLI settings for the retrained short controls.
Generated scripts/arrays/weights require explicit transfer for handoff; compact
CSV results and this protocol are versioned. Checks: 31 tests pass, strict
Pyright reports no errors/warnings, Ruff passes, and Git whitespace checks pass.
