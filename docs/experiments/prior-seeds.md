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

## Submission record

Slurm job **10895** started on 5 November 2026 under controller
`qcse-prior-seeds-20261105-01`. All **119 tests passed on the DGX** and all six
recipe/seed configurations passed full-size CUDA forward/gradient/optimizer checks.
The first `prior_s22` fit entered training. Startup evidence is retained at
`output/prior-seeds/startup-10895/verification.txt`. Results remain pending; verified
startup does not establish seed replication or a predictive improvement.

## Completed results

5 November 2026. Slurm **10895 completed with exit 0:0 in 01:32:54**. The
controller retrieved all artifacts successfully. All 111 pinned inputs and every
downloaded artifact matched SHA256 manifests. The GPU report reloaded all six new
selected checkpoints and reproduced validation predictions to 1e-10. The independent
local audit checked six locks/selected histories, three archived references,
12 metric rows, 24 buckets, 36 paired comparisons, all three seed summaries and
the decision rule, plus all 18 saved sample renderings. Count baselines were
recomputed directly from training statistics. No held-out test was scored.

| Training seed | Prior NLL | Adaptive NLL | Prior − adaptive | MLP NLL |
|---|---:|---:|---:|---:|
| 11, archived | 2.667615 | 2.702110 | −0.034495 | 2.407749 |
| 22, new | 2.668575 | 2.697155 | −0.028579 | 2.414943 |
| 33, new | 2.667095 | 2.705105 | −0.038010 | 2.413515 |
| Mean | **2.667762** | **2.701456** | **−0.033695** | **2.412069** |

The predeclared rule **passes**: both new seeds improve NLL and the three-seed
mean favors prior. The new-seed paired article intervals are
**[−0.044326, −0.012482]** and **[−0.051416, −0.025881]**. The interval for the
mean per-example difference is **[−0.044926, −0.023641]**, conditional on these
three fits. The NLL gain is seed-replicated on this validation task; neither these
intervals nor three seeds remove adaptive validation reuse or establish a general
population-level training-seed effect.

| Recipe | Mean NLL | NLL sample SD | Mean sampling S | Mean argmax |
|---|---:|---:|---:|---:|
| Prior quantum | 2.667762 | 0.000751 | 25.506% | 40.763% |
| Adaptive quantum | 2.701456 | 0.004015 | 25.737% | 40.291% |
| Cosine adaptive MLP | 2.412069 | 0.003809 | 39.413% | 48.307% |

The prior sampling regression also appears at all three seeds. Its mean deficit
is **0.231 percentage points**, with conditional article interval
[−0.347, −0.112] points for prior minus adaptive. Mean argmax improves by
0.472 points, interval [+0.061, +0.885]. These quantities are averages of individual
models' scores; no mixture distribution, majority vote or multi-shot scheme was
constructed. The MLP remains ahead by **0.255693 NLL**, interval
[+0.192956, +0.307510] for prior minus MLP, and 13.907 sampling points.

On non-space targets, the three-seed mean NLL is **3.6157 prior**, **3.6786
adaptive**, and **3.3322 MLP**; mean sampling is **6.54%, 6.07%, and 19.66%**.
The gain is useful within this quantum family, but the substantial classical
prediction gap remains. The earlier short-context comparison is unchanged: this
study supplies no new evidence for useful older context.

All six new prior continuations remain fragmented. Five have no invalid/control
characters; the first seed-33 sample has one UTF-8 replacement character. The six
new adaptive samples contain five replacements and nine other control characters.
The six MLP samples contain neither, but are not coherent long text. All samples
are retained without filtering. The three clean seed-11 prior samples therefore
did not establish reliable valid-byte generation.

Selections are prior epochs 30/23, adaptive 30/29 and MLP 25/26 for seeds 22/33.
The new prior fits take 13.74–13.78 minutes, adaptive 17.00–17.08, and MLP
14.93–14.99. These are observed simulator fit times, not quantum hardware speedup
or a complete cost comparison; prior still uses its fixed count table and
potentially costly amplitude preparation.

Keep prior as the quantum NLL reference and cosine adaptive MLP as the strong
classical recipe. This closes the immediate seed-sensitivity question for the
observed gain. Meaningful long text, useful long-range context, independent-corpus
confirmation and quantum advantage remain open; no best seed is substituted for
the reported three-seed comparison.

Evidence: [all metrics](prior-seeds-validation.csv),
[seed summaries](prior-seeds-seed-summary.csv), [paired intervals](prior-seeds-paired.csv),
[target buckets](prior-seeds-target-buckets.csv), [unfiltered samples](prior-seeds-samples.csv),
and [sample diagnostics](prior-seeds-sample-diagnostics.csv).
