# Seed replication of learned-position refinement

5 November 2026. Declared before new fitting. The previous single-seed screen
improved NLL and sampling over the prior and matched-capacity pooling, but did not
establish useful older context or close the classical gap. Verify robustness of
that improvement before changing the architecture again.

## Fixed recipes and reused evidence

Run six new fits: `prior_position` with full 128-token context, fixed-pooling
`prior` at rank 131, and independently refitted 16-token `prior_position`, each at
seeds **22 and 33**. Keep all recipes unchanged from the
[completed screen](prior-position.md#completed-results): rank 128 for the two
position models, two layers, one auxiliary qubit, recent-window 16, phase scale 1,
30 epochs, batch 512, Adam .003 constant, clipping 5 and float64/complex128.
Each fit receives 30,720 updates and 31 validation checks including initialization.
Train and select on NLL, with sampling success as the selection tiebreaker.

| Family | New fits | Reused fits | Trainable parameters |
|---|---|---|---:|
| position | Seeds 22, 33 | Seed 11 from job 10898 | 754,038 |
| capacity | Seeds 22, 33 | Seed 11 from job 10898 | 753,414 |
| short | Seeds 22, 33 | Seed 11 from job 10898 | 754,038 |
| prior | None | Seeds 11, 22, 33 from prior screen/replication | 736,512 |
| mlp | None | Strong cosine adaptive MLP at seeds 11, 22, 33 | 750,460 |

All nine reused fits have pinned source/configuration/data/weight locks and aligned
target arrays. Reuse avoids redundant training; the strong MLP keeps its declared
.003→.0003 cosine schedule. The previous fixed-basis and same-feature MLP controls
remain in the original report but are not seed-replicated here. This study does not
newly establish quantum attribution, coherence, entanglement or hardware advantage.
No model-source changes are introduced.

Use exactly the same training-only byte BPE and count statistics, 524,288 gradient
targets, 4,096 validation targets and 8,192 held-out test targets. Full dataset SHA256:
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
The short dataset masks the first 112 positions without changing any targets,
groups, tokenizer or statistics. All article partitions remain disjoint. Do not
score test, refit statistics, reselect archived checkpoints or sample a new partition.
All 60 official validation articles were already exposed: this is seed robustness
on reused validation, not independent article/corpus confirmation.

## Decisions and reporting

Require full-context position NLL to beat **both** prior and matched-capacity
pooling at **each new seed**, and on the three-seed mean against each control.
A better mean cannot hide a failing new seed. Report every seed even if the rule
fails, and preserve sampling/argmax tradeoffs and classical deficits.

For an older-context gain, additionally require full minus short NLL below zero
at both new seeds, the three-seed mean article-bootstrap interval entirely below
zero, and improved three-seed mean non-space NLL. The earlier seed-11 comparison
failed its older-context rule; changed learned lag weights alone cannot satisfy it.

Report all 15 individual fits plus bigram, unigram and train-modal references;
space/non-space buckets; per-family means, sample standard deviations and ranges;
and paired article intervals at each seed and for mean per-example differences.
The latter condition on three trained models and do not include all seed or adaptive
model-selection uncertainty. These are averages of model scores, not ensemble
probabilities or a multi-shot prediction scheme.

For the four new full-context fits, retain three 64-token direct draws each with
the same prompts and seeds 20261030–20261032. Keep all invalid/UNK/control outcomes.
Omit short-model generation through the generic full-window interface. Archived
samples remain archived and are not resampled. Reproduce selected predictions on
GPU before accepting report tables. Save histories/tables as CSV and arrays/weights
as NPZ, with only small JSON metadata.

## Execution

Generated scripts and frozen input/reference/source snapshots are under
`output/position-seeds/`. Use the isolated checkout
`~/quantum-semantics-position-seeds-20261105`, one full A100 and four CPU support
threads. Estimate **2–3 hours**, with a six-hour allocation ceiling; the previous
short fit's unusually high wall time makes this estimate uncertain. Before research
training, require all CUDA tests and full-size forward/gradient/optimizer checks
for all six model/seed configurations. No CPU training fallback.

Local checks cover all input/reference locks and full/short masks, strict generated
tool typing/linting, and reporting rules that distinguish averaged loss from a
probability mixture and reject hiding a failing seed in a better mean. Submit once;
network observer failure does not authorize resubmission. After verified startup,
pause goal work as requested while the experiment and completion delivery continue.
Results are not yet available.

## Submission record

Slurm **10906** started on 5 November 2026 under controller
`qcse-position-seeds-20261105-01`. All **130 DGX tests passed** and all six
model/seed configurations passed full-size CUDA forward/gradient/optimizer and
initial-prior checks. The first `position_s22` fit started, with **90% GPU
utilization** observed across three samples and approximately 1,470 MiB device
memory. Evidence is retained in `output/position-seeds/startup-10906/`.
All 230 source/data/reference/tool inputs are pinned. Results remain pending;
no model inputs or source are changed under the live allocation.

## Completed results

5 November 2026. Slurm **10906 completed with exit 0:0 in 02:09:22**. The local
observer failed after losing its SSH route, while remote training and reporting
continued. Recovery `qcse-position-seeds-recover-10906` retrieved the completed
artifacts without resubmitting any fit. All 230 submitted inputs and all generated
artifact hashes passed. The local `dgx.sbatch` default had subsequently changed to
`prod10`; its original submitted bytes were verified from commit `fff993a` rather
than reverting the new default. The remote submitted manifest remained intact.
This completed allocation used the earlier full-A100 protocol.

The GPU report reloaded every selected checkpoint and reproduced validation
predictions. The independent local audit checked six fit locks/histories, nine
archived references, 18 metric rows, 36 target buckets, 60 paired comparisons,
five seed summaries, both decision rules and all 12 saved sample renderings.
No held-out test was scored.

| Seed | Position NLL | Prior NLL | Capacity NLL | Short NLL |
|---|---:|---:|---:|---:|
| 11, archived | 2.595124 | 2.667615 | 2.668486 | 2.586842 |
| 22, new | 2.609233 | 2.668575 | 2.668876 | 2.599437 |
| 33, new | 2.605401 | 2.667095 | 2.664499 | 2.598301 |

The declared replication rule **passes** against both controls. Position minus
prior NLL is **−0.059343** at seed 22 and **−0.061694** at seed 33; against
matched-capacity pooling the differences are **−0.059643** and **−0.059098**.
The three-seed mean differences are **−0.064509**, article interval
**[−0.075998, −0.052071]**, against prior and **−0.064034**, interval
**[−0.074280, −0.054302]**, against capacity. These intervals condition on the
three fitted models and reused validation articles; they do not establish
independent-corpus generalization or account for all model-selection uncertainty.

| Family | Mean NLL | NLL sample SD | Mean sampling S | Mean argmax |
|---|---:|---:|---:|---:|
| Learned positions, full | **2.603252** | 0.007296 | **26.572%** | **42.326%** |
| Matched-capacity pooling | 2.667287 | 0.002423 | 25.639% | 40.617% |
| Learned positions, short | 2.594860 | 0.006967 | 26.642% | 42.326% |
| Prior | 2.667762 | 0.000751 | 25.506% | 40.763% |
| Strong cosine MLP | 2.412069 | 0.003809 | 39.413% | 48.307% |

Mean sampling improves over prior by **1.066 percentage points**, interval
[+0.912, +1.214], and argmax by 1.5625 points. Sampling also improves over the
capacity control by 0.933 points. These are averages of individual-model scores;
no ensemble probability mixture or multi-shot predictor was constructed. The
full-position model has greater observed seed variability than the prior, although
every tested seed improves NLL over both quantum controls.

The older-context rule **fails**. Full minus short NLL is positive at all three
seeds, with mean **+0.008392**, interval **[−0.000165, +0.017139]**. This interval
does not establish NLL superiority of either length; it provides no evidence for
the claimed benefit of the older 112 tokens. Full-context sampling is slightly
lower by 0.0704 points, interval [−0.1051, −0.0344], and mean argmax is identical.
Mean non-space NLL/sampling are **3.5337 / 7.36%** full and **3.5210 / 7.47%**
short, versus prior **3.6157 / 6.54%** and MLP **3.3322 / 19.66%**.

The MLP remains ahead of full position by **0.191184 NLL**, interval
[+0.132162, +0.239913] for quantum minus MLP, and **12.841 sampling points**.
Seed replication establishes a useful improvement within this quantum recipe
family, not superiority over the strong classical recipe, a newly isolated quantum
mechanism, or useful long-range context.

All six new position continuations contain no invalid UTF-8/control characters or
non-text outcomes, but remain fragmented. The earlier seed-11 sample with a
replacement character remains part of the evidence. Each new capacity seed has
one replacement character in its third continuation. Retain every raw sample;
valid bytes alone do not establish coherent generation.

The new full-position fits take 14.72–14.76 minutes, capacity 13.49–13.50, and
short 35.64–35.65. The slower short fit is repeatable here but is not a controlled
runtime comparison or a quantum speedup claim. Selected epochs are 30/30 full,
30/30 capacity and 27/30 short for seeds 22/33. No optimizer continuation or
additional fitting was performed during recovery.

Retain learned positions as a seed-replicated quantum reference on this validation
task. Keep the prior, capacity, short and strong MLP comparisons visible. Coherent
long text, useful long-range context, fresh confirmation and quantum advantage
remain unresolved; the broader research objective is not complete.

Evidence: [individual metrics](position-seeds-validation.csv),
[seed summaries](position-seeds-seed-summary.csv),
[paired intervals](position-seeds-paired.csv),
[target buckets](position-seeds-target-buckets.csv),
[unfiltered samples](position-seeds-samples.csv), and
[sample diagnostics](position-seeds-sample-diagnostics.csv).
