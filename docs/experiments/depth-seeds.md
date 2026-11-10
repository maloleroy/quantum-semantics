# Seed replication of three-stage quantum refinement

10 November 2026. Declared before new fitting. The seed-11 depth/interleaving
recipe improves NLL over both the old two-stage content model and matched-width
capacity. Replicate that gain while preserving the stronger short-context and
classical controls. No model-source change is introduced.

## Fixed recipes and references

Run six fresh fits at seeds **22 and 33**: three-stage rank-128 full content,
two-stage rank-176 full content, and independently refitted three-stage rank-128
short context. Keep the [completed depth screen](depth-screen.md#completed-results-audited-10-november-2026)
recipes unchanged: vocabulary 512, context 128, auxiliary dimension two, recent
window 16, query/key dimension 16, phase scale one, float64/complex128, Adam .003
constant, clipping 5, 30 epochs and batch 512. Each receives **30,720 updates**.
Select minimum validation NLL across epoch zero and all 30 epochs, with sampling
success as tiebreaker. No early stopping or changes based on emerging results.

| Family | New fits | Archived fits | Trainable parameters |
|---|---|---|---:|
| depth | Seeds 22, 33 | Seed 11, job 11039 | 1,073,900 |
| capacity | Seeds 22, 33 | Seed 11, job 11039 | 1,071,526 |
| short | Seeds 22, 33 | Seed 11, job 11039 | 1,073,900 |
| content | None | Two-stage rank-128, seeds 11/22/33 | 794,998 |
| mlp | None | Strong adaptive cosine MLP, seeds 11/22/33 | 750,460 |

All nine references retain their original data/configuration/source/weight locks
and aligned target arrays. Reuse avoids unnecessary fits. The archived MLP keeps
its .003-to-.0003 cosine schedule. Fixed-basis and same-feature MLP results stay
visible in the first screen; this replication does not newly replicate those
attribution controls. Depth adds mostly quantum Hamiltonian parameters, plus the
explicitly counted extra RZ controller. The query/key selector stays unchanged and
there is no classical vocabulary head in the quantum candidate. This is a
recipe-depth comparison, not a pure quantum-depth isolation or quantum advantage.

Preserve 524,288 training, 4,096 validation and 8,192 test targets, training-only
byte BPE/count statistics, article partitions, target-byte counts and UNK policy.
Full dataset SHA256:
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
The short dataset masks the first 112 positions without changing targets or
statistics. No test scoring, repartitioning, reference reselection or optimizer
continuation. All 60 validation articles were exposed previously: this is seed
robustness on reused validation, not independent confirmation.

## Decisions and reports

Require full three-stage NLL to beat both the archived rank-128 content model and
fresh rank-176 capacity control at **each new seed** and on the three-seed mean.
A good average cannot hide a failing new seed. Report all results and preserve
sampling/argmax tradeoffs, classical deficits and adverse samples.

Useful older context separately requires full minus refitted-short NLL below zero
at both new seeds, a three-seed mean paired article interval entirely below zero,
and improved mean non-space NLL. The first screen favored short context; extra
quantum depth or an intervention sensitivity result does not override this rule.

Report the 15 individual fits, bigram/unigram/train-modal baselines, space/non-space
buckets, means/sample standard deviations/ranges, and paired article intervals
per seed and for mean per-example differences. These are averages of model scores,
not ensemble probabilities. Intervals condition on the fitted models and do not
correct adaptive selection or all seed uncertainty.

Reload all selected checkpoints on GPU and reproduce saved validation arrays.
Check complete 31-row histories and selection locks before accepting reports.
Retain three 64-token direct-draw samples for each of the four new full-context
fits, using the established prompts and seeds 20261030–20261032. Keep invalid and
control outcomes and raw IDs. Do not generate short-model samples through the
unmasked generic interface, and do not resample archived references.

## Execution and validation

Measured prior fits took 32.60 minutes full, 30.53 width and 62.45 short. Two
repetitions project **4h11m** plus setup/reporting and timing variability. Keep
`prod10`, one `nvidia_a100_1g.10gb` slice, four CPU support threads and a six-hour
ceiling. The completed study stayed below 1.18 GiB peak reserved memory during its
resource probe. No larger GPU or batch/update reduction is needed.

Before fitting, rerun all tests and full-size CUDA/resource gates for all six
model/seed pairs: three warmup updates, finite CUDA gradients and optimizer changes,
active query/key gradients, exact initial count prior, then three timed eight-update
blocks with all training rows and a checkpoint snapshot resident. Stop before
training if projected aggregate optimizer time exceeds five hours or peak reserved
memory reaches 80% of the slice. Discard these gate updates and start all fits fresh.
The gate's timing projection omits validation and I/O; it is not a guaranteed runtime.

Local setup verifies parameter counts, input/reference locks, full/short masks and
16 archived prediction rows per reference. Generated tools pass Ruff and strict
Pyright. Three reporting tests guard against probability-mixture substitution,
hiding a failed seed in a mean, and unsupported older-context claims.

Tools and frozen manifests live in `output/depth-seeds/`; isolated DGX checkout
`~/quantum-semantics-depth-seeds-20261110`. Submit once and run **`dgx start` on
the laptop**. Stop model polling while awaiting the run. If the notifier loses its
SSH route, the Slurm job may still continue; inspect that job rather than resubmit.
A queue-empty notification must be followed by accounting, input/output hashes and
the prepared independent report audit. No codex-long-jobs controller is used.


## Verified startup

Slurm **11058** started on 10 November 2026 on prod10 with one 10 GB MIG slice.
All **142 DGX tests passed**, and all six full-size CUDA learning/resource gates
passed, including finite CUDA gradients, changed predictions and active query/key
gradients. The first `depth_s22` scientific fit entered training. Peak reserved
memory stays below 1.18 GiB; update-only projections total approximately 4h07m,
within the declared guard. All 232 pinned inputs still match.

The laptop `dgx start` command succeeded and the systemd notifier is verified
active/running (observed PID 125490). Startup evidence is retained in
`output/depth-seeds/startup/verification.txt`. Goal work pauses while the experiment
runs. No new predictive result is available yet, and no input/source is changed
beneath the running allocation. After notification, inspect accounting and use the
prepared retrieval/audit script; do not treat an SSH failure as a training failure.


## Completed results (10 November 2026)

Slurm **11058 completed with exit 0:0 in 04:11:25**. Retrieval verified all 232
pinned inputs and every generated-artifact hash. All 142 DGX tests and six actual
CUDA learning/resource gates passed on the prod10 MIG slice. No training restart
or held-out test scoring occurred.

GPU reloads reproduced every selected validation array. The independent local
audit reproduced six complete histories/selections, nine archived references,
18 metric rows, 36 target buckets, 60 paired intervals, five seed summaries,
both decision rules and all 12 saved sample renderings. All 18 timing blocks,
six resource measurements and the declared time/memory guards also verify.

| Family | Mean NLL | NLL sample SD | Mean sampling S | Mean argmax |
|---|---:|---:|---:|---:|
| Three-stage full content | **2.535545** | .011040 | **28.016%** | **43.693%** |
| Two-stage rank-176 width | 2.564646 | .008843 | 27.677% | 43.636% |
| Three-stage refitted 16-token | **2.518074** | .008026 | **28.298%** | **44.019%** |
| Original two-stage rank-128 content | 2.575884 | .003435 | 27.287% | 43.132% |
| Strong archived cosine MLP | **2.412069** | .003809 | **39.413%** | **48.307%** |

The declared replication rule **passes**. Three-stage NLL at seeds 22/33 is
**2.522948 / 2.543535**, beating original two-stage content by **−.051240 /
−.030092** and matched width by **−.037881 / −.014816**. The three-seed mean
NLL difference is **−.040339** against original content, conditional article
interval **[−.048820, −.030868]**, and **−.029101** against width, interval
**[−.038285, −.019677]**. Sampling improves by .729 and .339 percentage points,
respectively. Mean argmax only slightly exceeds width; its paired interval crosses
zero. Keep metric distinctions visible.

These results support the extra-depth/interleaving recipe under this training
budget rather than a gain explained solely by total parameter count. The additional
RZ controller is still counted; this does not isolate pure quantum depth or prove
quantum advantage. Three-seed evidence on reused validation is not independent
partition confirmation and does not remove adaptive model-selection uncertainty.

The older-context rule **fails**. Full minus short NLL is positive at all seeds:
+.017568, +.014140 and +.020704. The mean is **+.017471**, interval
**[+.008472, +.026029]**. Mean sampling is .282 percentage points lower for full,
interval [−.360, −.201]. Non-space NLL is also worse, **3.45304 versus 3.43054**.
The refitted 16-token model remains stronger after increasing depth.

Mean non-space sampling is **8.409%** three-stage full, **8.101%** width,
**8.676%** short, **7.674%** original content and **19.658%** strong MLP.
The MLP retains a **.123476 NLL** lead, interval [.070407, .168164] for quantum
minus MLP, and **11.397 sampling percentage points**. The previous single-seed
fixed-basis and same-feature MLP results remain visible in the original screen;
this experiment does not replicate those controls.

Selected-checkpoint training NLL averages **2.35083** full and **2.39099** short,
while validation reverses their ordering. This is consistent with a full-context
generalization issue and motivates a controlled regularization study; it does not
by itself diagnose overfitting, identify its cause or show that masking older
context during training would improve performance. Prior locked-model interventions
remain sensitivity evidence, not a substitute for independently refitted controls.

All six new width samples avoid invalid/control characters. One of the six new
three-stage samples contains a UTF-8 replacement character, and all remain
incoherent. Across this and the original screen, two of nine three-stage samples
contain replacement characters. Every adverse outcome is retained. Short-model
samples are still omitted through the generic unmasked generation path.

New fits took **32.43–32.52 minutes** full, **30.43–30.57** width and
**61.67–61.98** short. Peak reserved memory in the resource gate remains below
1.18 GiB. The measured total is consistent with the declared 4h11m planning
estimate; prod10 remains sufficient without changing training budgets.

Retain three-stage refinement as the seed-replicated quantum reference on this
partition. Improving full-context generalization is the next bounded research
question before treating additional capacity as the whole solution. Useful long
context, coherent long generation, independent confirmation and quantum advantage
remain unproven.

Evidence: [individual metrics](depth-seeds-validation.csv),
[seed summaries](depth-seeds-seed-summary.csv),
[paired intervals](depth-seeds-paired.csv),
[target buckets](depth-seeds-target-buckets.csv),
[unfiltered samples](depth-seeds-samples.csv),
[sample diagnostics](depth-seeds-sample-diagnostics.csv),
[resource measurements](depth-seeds-resources.csv), and
[timing blocks](depth-seeds-timings.csv).
