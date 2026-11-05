# Replicate the prior-model gain across training seeds

5 November 2026. Declared before new training. The
[count-initialized screen](prior-screen.md#completed-results) improved quantum
validation NLL from 2.702110 to 2.667615 at seed 11, while sampling fell and the
strongest classical model retained a substantial lead. Replicate the recipes
without selecting a new architecture or changing their hyperparameters.

## Scope and independence

An inventory found 60 official WikiText validation articles and 60 distinct
article groups in the existing 4,096-target selection partition. There are no
unused official validation articles for an independent article-level confirmation.
New positions from those same articles would not resolve article-level adaptive
selection bias. The test set remains untouched. This experiment measures seed
robustness on the existing validation task, not fresh-partition confirmation.
A later broad generalization claim needs a separately declared independent corpus
or an explicit locked final test protocol.

## Locked recipes

Train six fresh fits: each of the following at seeds **22 and 33**. Include the
already audited, locked seed-11 fit for each recipe in reporting; never retrain or
resample that seed as part of this study.

| Recipe | Kind | Schedule | Trainable parameters |
|---|---|---|---:|
| prior | prior | constant .003 | 736,512 |
| adaptive | adaptive | constant .003 | 751,892 |
| mlp | adaptive_mlp | cosine .003 to .0003 | 750,460 |

All use rank 128, two layers, auxiliary size 2, recent window 16, phase scale 1,
auto encoding and direct Born readout where quantum. No core is frozen. The
recorded unused amplitude-decay field stays .7 for prior and .9 for the archived
adaptive/MLP recipes. Preserve all original initializations and precision
(float64/complex128). The stronger classical recipe is the original adaptive MLP,
not the weaker count-prior MLP. This is a recipe-level comparison: input
representations and schedules differ as declared, and prior's fixed count cache
and amplitude preparation still contribute to resource cost.

Use the exact prepared 524,288-training-position dataset, 4,096 validation targets,
8,192 test targets, V512 and W128. SHA256:
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
No tokenizer/statistic refit, new positions, repartitioning or test scoring.
Every fresh fit uses 30 epochs, batch 512, Adam, NLL training and NLL validation
selection, with sampling as the existing tie-breaker. This gives 30,720 updates
and 31 validation checks per fit. No early-stopping or budget change based on
intermediate results. The candidate is fixed before the new seeds are observed.

## Reporting and decision rule

Report all nine learned fits and unchanged bigram/unigram/modal-space baselines:
12 metric rows and 24 modal/non-modal target buckets. Keep NLL, sampling and argmax
separate. For prior minus adaptive, prior minus MLP and prior minus bigram, report
all three metrics for each seed and for the mean of the three per-example scores:
36 paired article interval rows. The seed-mean score is not a mixture predictor or
an ensemble distribution. For each family report mean, sample standard deviation,
minimum and maximum across the three seeds. Article bootstrap intervals conditional
on these three fitted seeds do not measure all training uncertainty.

Call the NLL improvement seed-replicated only if prior beats adaptive separately
at **both new seeds** and its three-seed mean NLL is lower. Show all paired deltas
regardless of the rule outcome. Do not select the best seed and report it as the
family result. A failure leaves the prior result provisional; it is not hidden by
averaging. Even success does not resolve reused-validation bias or establish
quantum superiority, useful older context or coherent long text.

Save three fixed-prompt, unfiltered 64-token continuations for each new fit,
using seeds 20261030–20261032 and the established prompts (18 samples total).
Retain every token and invalid/UNK outcome. Existing seed-11 samples remain in
their audited source studies and are not regenerated. No additional sample-quality
selection criterion is introduced.

## Execution and validation

Use isolated checkout `~/quantum-semantics-prior-seeds-20261105`, one full A100,
four CPU support threads and an eight-hour allocation ceiling. Estimate 1.5–3
hours from completed fit timings plus setup/evaluation. Intensive work must run on
CUDA. Require actual model forward/backward/optimizer checks and the CUDA test
suite before training, then reload selected weights and reproduce validation
arrays on GPU before summary. Keep source, dependencies, tools, fixture, data and
archived reference hashes pinned. CSV histories/tables, NPZ arrays/weights and
small JSON metadata stay under `output/prior-seeds/`.

Submit once and monitor the exact job ID through the long-jobs controller. Network
failure does not authorize resubmission. Pause goal work after verified startup,
as requested, while the experiment and completion delivery continue.
