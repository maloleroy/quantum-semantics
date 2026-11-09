# Locked content-model context interventions

Declared 9 November 2026, after seed replication and before this diagnostic.
The full 128-token model beats position baselines, but its independently refitted
16-token counterpart has lower NLL at all three seeds. Diagnose whether the full
model uses older context beneficially at inference before choosing another change.

Use the already selected full-content weights at seeds 11, 22 and 33. Keep the
existing 4,096 validation targets, training-only statistics/tokenizer, article IDs,
and weights fixed. No fitting, checkpoint selection, test scoring or new generation.
This is explicitly post-hoc reuse of validation, not independent confirmation.

Evaluate four input conditions with the last 16 context positions unchanged:

- **Intact:** reproduce the saved validation arrays before accepting results.
- **Mask older:** set the preceding 112 positions to padding.
- **Reverse older:** reverse only valid older token positions, retaining each row's
  older-token multiset and exact padding mask.
- **Other article:** replace valid older positions with tokens from a donor in a
  different validation article having the same number of valid older tokens.
  Choose donors using RNG seed 20261109 without consulting target labels. Leave
  rows with no valid older tokens or no eligible donor unchanged, and report coverage.

Donors come only from validation contexts; this deliberately breaks the causal
context relation and is a corruption diagnostic, never a deployable predictor or
an improved language-model score. All interventions can shift the input
distribution away from training. Loss changes measure sensitivity/co-adaptation;
they do not replace independent refits or identify a causal generalization mechanism.

Report per-seed and three-seed mean NLL, sampling success and argmax, target buckets,
and descriptive article-bootstrap intervals for intervention minus intact metrics.
Seed averages are averages of model scores, not ensemble probabilities. Archive
per-example score arrays and exact intervention contexts/donor indices as NPZ;
use CSV for reports and small JSON for execution metadata. Preserve checkpoint and
source hashes. No beneficial-effect threshold or new-model selection is implied.

Run on the laptop with four CPU threads, batch 512, and unchanged float64/complex128.
Twelve 4,096-row forward-only evaluations are expected to finish within ten minutes.
Generated tools and raw outputs live in `output/content-diagnostics/`; strict
Pyright and Ruff checks must pass before execution. No Slurm job or notification
watcher is required for this bounded local diagnostic.


## Completed results

The final, typed/linted tool completed locally in **16.36 seconds**, with four CPU
threads. All three intact evaluations reproduce the original CUDA validation
arrays within 1e-10; checkpoint hashes remain unchanged. The independent audit
recomputed 12 score rows, 24 target buckets, 36 paired intervals and four means,
and verified the saved context transformations and donors. No test scoring,
training or checkpoint selection occurred. Generated tools/results are hashed in
`output/content-diagnostics/artifacts.sha256`.

| Condition | Mean NLL | Mean sampling S | Mean argmax | NLL change vs intact |
|---|---:|---:|---:|---:|
| Intact full context | 2.575884 | 27.287% | 43.132% | — |
| Older context masked | 2.611466 | 25.551% | 42.944% | +.035582 |
| Valid older positions reversed | 2.587120 | 27.266% | 43.164% | +.011237 |
| Older tokens from another article | 2.602097 | 27.225% | 42.806% | +.026213 |

Mean intervention-minus-intact NLL article intervals are **[+.028380, +.043404]**
for masking, **[+.005338, +.017908]** for reversal and
**[+.016931, +.035511]** for other-article context. Every intervention worsens
NLL at each of the three seeds. These are conditional descriptive intervals on
reused validation, not independent confirmations or corrections for prior selection.

Masking reduces sampling by **1.736 percentage points**, interval
[−1.888, −1.588]. Reversal changes sampling by −.021 points with an interval
crossing zero; the other-article intervention reduces it by .062 points. Argmax
intervals cross zero for all three interventions. Preserve these metric differences
instead of inferring that worse NLL forces worse argmax in every condition.

Masking and reversal change 4,091 of 4,096 validation contexts; five have no valid
older tokens. Donor replacement changes 4,065 contexts, always using a different
article with the same valid older-token count and preserving the recipient padding
mask. The remaining 31 are unchanged: five empty older contexts and 26 without an
eligible donor. All last-16-token contexts and all targets are unchanged. Reversal
preserves the exact older-token multiset; donor replacement intentionally changes it.

Non-space NLL is **3.50566** intact, **3.52461** masked, **3.52182** reversed and
**3.54255** with another article. The diagnostic loss is not confined to space
prediction, though the larger masking-related sampling drop is substantially
associated with space targets. These interventions demonstrate sensitivity to older
content and its arrangement within the trained full model.

This does **not** overturn the independently refitted short model's superior mean
NLL **2.56249** or establish a useful long-context model. Removing information
from a model trained to use it is a different operation from refitting without
that information; corruption also shifts the input distribution. We cannot infer
from this result alone that the remaining gap is caused by overfitting, insufficient
quantum expressivity, or a specific controller defect.

The next controlled question is how to improve full-context generalization while
retaining informative older inputs, rather than assuming the branch is unused.
A training-time older-context regularization comparison would test co-adaptation;
a quantum-capacity/interleaving comparison would test a different expressivity
hypothesis. Either needs fresh declared fits and the short/classical controls.
No new recipe gain, long-text coherence or quantum advantage is established here.

Evidence: [per-seed scores](content-diagnostics-metrics.csv),
[mean scores](content-diagnostics-mean-metrics.csv),
[target buckets](content-diagnostics-target-buckets.csv), and
[paired intervals](content-diagnostics-paired.csv).
