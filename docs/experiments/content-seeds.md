# Seed replication of content-dependent controls

9 November 2026. Declared before new fitting. The single-seed content screen
improved NLL over position models but failed the older-context criterion and
remained behind classical MLPs. Test whether the small gain survives new seeds
before changing the model again.

## Fixed recipes and archived evidence

Run six fresh fits: full 128-token `prior_content`, position-only `prior_position`
at matched-capacity rank 135, and independently refitted 16-token `prior_content`,
each at seeds **22 and 33**. Retain the exact recipes from the
[audited content screen](content-screen.md#completed-results-audited-9-november-2026):
rank 128 for content models, two layers, one auxiliary qubit, recent window 16,
query/key dimension 16, phase scale 1, float64/complex128, Adam .003 constant,
clipping 5, 30 epochs and batch 512. Each fit gets **30,720 updates** and
31 validation checks including initialization. Select minimum validation NLL,
breaking ties by sampling success. No early stopping or changes after seeing scores.

| Family | New fits | Reused fits | Trainable parameters |
|---|---|---|---:|
| content | Seeds 22, 33 | Seed 11, job 10911 | 794,998 |
| capacity | Seeds 22, 33 | Seed 11, job 10911 | 794,365 |
| short | Seeds 22, 33 | Seed 11, job 10911 | 794,998 |
| position | None | Seeds 11, 22, 33, jobs 10898/10906 | 754,038 |
| mlp | None | Strong archived cosine adaptive MLP, all three seeds | 750,460 |

All nine references retain their own source, weights, data/configuration and
selection locks. Reuse avoids redundant fits. The archived MLP keeps its
.003-to-.0003 cosine schedule. Fixed-basis and same-feature content MLP results
remain visible in the original report but are not replicated in this study.
No model source changes, new attribution claim, quantum advantage or coherent
long-text claim are introduced. The classical content selector remains explicitly
counted; candidate output still comes from quantum evolution and direct Born
measurement without a classical vocabulary head.

Keep the exact training-only byte BPE/count statistics and article partitions:
524,288 training targets, 4,096 validation targets, 8,192 held-out test targets,
vocabulary 512 and window 128. Full dataset SHA256:
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
Short contexts mask the first 112 positions without changing targets, groups,
byte counts, vocabulary or statistics. No test scoring, repartitioning or reselection
of archived checkpoints. All 60 validation articles were previously exposed;
seed robustness on this partition is not independent confirmation.

## Decisions and reporting

Require full-content NLL to beat both the archived position model and the fresh
matched-capacity position control at **each new seed** and on the three-seed mean.
A better average cannot hide a failing new seed. Report every result, including
sampling/argmax tradeoffs and the gap to the strong classical reference.

A useful older-context gain separately requires full minus short NLL below zero
at both new seeds, the three-seed mean paired article interval entirely below zero,
and improved three-seed mean non-space NLL. The previous seed-11 result favored
short context; changes in learned selection weights cannot substitute for this test.

Report 15 individual fits plus bigram/unigram/train-modal baselines, space/non-space
buckets, means/sample standard deviations/ranges, and paired article intervals
per seed and for the mean per-example difference. Average model scores are not
ensemble probabilities or a multi-shot prediction protocol. Intervals condition
on the three trained models and omit adaptive selection and broader seed uncertainty.

Reload selected weights on CUDA and reproduce saved predictions before accepting
reports. Check all 31 history rows and selection locks. For the four new full-context
fits, retain three 64-token direct-draw samples each using the established prompts
and seeds 20261030–20261032. Keep invalid/control outcomes and raw token IDs.
Do not generate short-model samples through the generic full-window interface;
archived samples remain archived. Use CSV for tables/history, NPZ for arrays/weights,
and small JSON for metadata.

## Resources and execution

Use `prod10` with one `nvidia_a100_1g.10gb` MIG slice and four CPU support threads.
Previous full fits measured 26.31 minutes content, 23.26 capacity and 50.73 short.
Two repetitions imply **3h20m** plus setup/reporting and timing variability; use a
six-hour allocation ceiling. All shapes, batches and update budgets are unchanged;
the completed MIG resource probe and full fits establish feasibility without a
full A100 request.

Generated tools, frozen source and reference/input manifests live in
`output/content-seeds/`. Use isolated DGX checkout
`~/quantum-semantics-content-seeds-20261109`. Local checks cover strict typing/linting,
parameter budgets, all reference locks, data/masks and 16 reproduced prediction
rows per reference. Three reporting checks reject probability-mixture substitution,
a failed seed hidden in an average, and unsupported older-context claims.

Before training, rerun all CUDA tests and check actual full-size CUDA forwards,
finite gradients, optimizer changes and count-prior initialization for all six
model/seed configurations. Also require active query/key gradients for content
models. Training starts fresh after these disposable checks. No CPU fallback.
Submit once with codex-long-jobs; observer failure does not authorize resubmission.
After verified startup, pause goal work while the experiment and completion
delivery continue. No new-seed results are available at this declaration.


## Launch record (9 November 2026)

Controller `qcse-content-seeds-20261109-01` started with 230 pinned inputs and
queued completion delivery. The worker was verified alive while rsync transferred
inputs to the isolated checkout. At the final observation, there was no Slurm job
ID yet: CUDA startup and training are **not yet confirmed**. The script requires
all tests and six CUDA learning gates before scientific fitting. Goal work pauses
while the worker continues; inspect its durable state and actual remote artifacts
on completion or failure rather than submitting another copy.


## Completed remote run; local recovery pending

The submission finished and created Slurm **10980**. Its local observer later
exited 255 after SSH reported `No route to host`; training continued remotely.
Reconnection confirms **COMPLETED, exit 0:0, elapsed 03:22:30**. All six model/seed
CUDA learning checks succeeded on the 10 GB MIG slice, and the remote report ends
with `CONTENT_SEEDS_SCREEN_COMPLETE`. Remote input and artifact checksums pass.
No training was restarted.

Recovery worker `qcse-content-seeds-recover-10980` retrieves the original outputs,
verifies local hashes/selection locks and runs an independent audit of every table,
seed summary, paired interval, decision and saved sample rendering. The audit tool
passes Ruff and strict Pyright. Scientific conclusions remain pending that local
audit. Goal work pauses while recovery and queued completion delivery continue.


## Completed results (9 November 2026)

Recovery `qcse-content-seeds-recover-10980` succeeded. Slurm **10980 completed with
exit 0:0 in 03:22:30**, despite the earlier SSH observer failure. All 230 pinned
inputs and all generated-artifact hashes passed both remote and local checks.
All **142 DGX tests** and six full-size CUDA learning gates passed on the prod10
MIG slice. No training was restarted and no held-out test was scored.

Every selected checkpoint reproduced its saved validation predictions on GPU.
The independent local audit reproduced six complete histories/selection locks,
nine archived references, 18 metric rows, 36 target buckets, 60 paired intervals,
five seed summaries, both decision rules and all 12 saved sample renderings.
Recovery's success marker requires that audit to finish, not merely transfer success.

| Family | Mean NLL | NLL sample SD | Mean sampling S | Mean argmax |
|---|---:|---:|---:|---:|
| Content quantum, full 128 | **2.575884** | .003435 | **27.287%** | **43.132%** |
| Matched-capacity position quantum | 2.601242 | .004557 | 26.578% | 42.318% |
| Content quantum, refitted 16 | **2.562495** | .006749 | **27.480%** | **43.294%** |
| Archived position quantum | 2.603252 | .007296 | 26.572% | 42.326% |
| Strong archived cosine MLP | **2.412069** | .003809 | **39.413%** | **48.307%** |

The declared seed-replication rule **passes**. At new seeds 22/33, full-content
NLL is **2.574188 / 2.573627**, beating position by **−.035045 / −.031774** and
matched capacity by **−.028144 / −.031528**. Three-seed mean content minus position
NLL is **−.027369**, conditional article interval **[−.037409, −.017616]**;
versus matched capacity it is **−.025358**, interval **[−.034648, −.015400]**.
Sampling improves over position by **.715 percentage points**, interval
[+.626, +.801], and argmax improves by .806 points. The improvement is not explained
by adding the same parameter count to the position model under this training budget.

The older-context rule **fails again**. Full minus independently refitted short
NLL is positive at all three seeds: +.019357, +.004166 and +.016644. Its mean is
**+.013389**, interval **[+.003598, +.022767]**. Mean non-space NLL is also worse
for full, **3.50566 versus 3.48756**, and mean sampling is .194 percentage points
lower, interval [−.260, −.121]. The results favor short context on this reused
validation set; content selection has not made the older 112 tokens useful.

Mean non-space sampling is **7.674%** content, **7.389%** capacity, **7.886%**
short, **7.360%** position and **19.658%** MLP. The strongest MLP retains a
**.163815 NLL** lead, interval [.107152, .210148] for quantum minus MLP, and
**12.126 sampling percentage points**. Better quantum-family scores do not establish
classical superiority or quantum advantage. The single-seed fixed-basis and
same-feature MLP controls remain in the original report; they are not replicated
by this experiment.

All six new content samples and six new capacity samples decode without invalid
UTF-8 or other control characters. They remain fragmented and incoherent. Preserve
these samples and the earlier capacity sample with a replacement character;
clean character encoding is not coherent generation. No sample was filtered or
resampled. The generic short-generation path remains deliberately unused.

Measured new fit times are approximately **26.34–26.36 minutes** full content,
**23.26** capacity and **50.84–50.89** short, consistent with the declared 3h20m
aggregate projection plus setup/reporting. Default prod10 remains sufficient for
these configurations without changing training budgets.

Retain content-dependent controls as the seed-replicated quantum reference on
this reused validation partition, with the separately refitted short model as a
stronger context-length control. The gain does not inherit independent-partition
confirmation: all 60 validation articles were already exposed, and the reported
article intervals condition on the three fitted models rather than correcting
adaptive selection or all seed uncertainty. Meaningful long context, coherent long
text and quantum advantage remain unresolved.

Before another architectural change, a useful next bounded question is whether
older-context removal or disruption changes locked full-model validation losses.
Such interventions would measure sensitivity/co-adaptation, not replace the
independently refitted short control or prove a causal generalization mechanism.
No intervention result or new experiment is claimed here.

Evidence: [individual metrics](content-seeds-validation.csv),
[seed summaries](content-seeds-seed-summary.csv),
[paired intervals](content-seeds-paired.csv),
[target buckets](content-seeds-target-buckets.csv),
[unfiltered samples](content-seeds-samples.csv), and
[sample diagnostics](content-seeds-sample-diagnostics.csv).
